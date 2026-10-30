# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This trainer supports model-agonistic model initialization with huggingface
"""

import re
import uuid
from pprint import pprint

import os
import math
import numpy as np
import ray
import torch
import gc
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.async_replay.utils import need_critic
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger
from collections import defaultdict, deque
from .replay.buffer import GenerationBatchFuture, Memory
from .replay.selection import ReplaySelectionMixin
from copy import deepcopy
import verl.utils.torch_functional as verl_F

from recipe.async_replay._vendor.qiuputils.logger import get_logger
from recipe.async_replay._vendor.qiuputils.ai_utils import CheckpointManager
from .checkpoints import PeriodicBestCheckpointManager
from .evaluation import average_accuracy

    
    
class OneStepOffRayTrainer(ReplaySelectionMixin, RayPPOTrainer):
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    """
    Main Trainer class implementing One-step Off-policy PPO.
    Inherits from verl's RayPPOTrainer but overrides the execution loop for async logic.
    """
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        self.enable_replay = self.config.replay.enable

        if self.enable_replay:
            self.memory = Memory(k=self.config.replay.k)

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.total_query_size = len(self.train_dataloader) * self.config.data.train_batch_size
        self.qiup_logger = get_logger()
        self.ckpt_manager = CheckpointManager(
            base_dir=self.config.trainer.default_local_dir,
            logger=self.qiup_logger,
            indentation_num=2
        )
        if self.config.trainer.get("checkpoint_policy") == "periodic_and_best":
            self.ckpt_manager = PeriodicBestCheckpointManager(
                self.config.trainer.default_local_dir,
                metric=self.config.trainer.best_checkpoint_metric,
                keep_periodic=self.config.trainer.get("keep_periodic_checkpoints", 5),
            )

    def _validate(self, val_mode=None):
        # Normalize val_mode to a sorted list of integers, e.g. [1, 5]
        val_mode = self._normalize_val_mode(val_mode)
        max_n = max(val_mode) if val_mode else None

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        self.actor_rollout_wg = self.rollout_wg
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch: use max(val_mode) if specified, else val_kwargs.n
            actual_n = max_n if max_n is not None else self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(
                repeat_times=actual_n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        self.actor_rollout_wg = self.actor_wg

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Compute pass@k metrics for each k in val_mode
        if val_mode:
            if self.config.get("project_kwargs", {}).get("val_metric", "pass") == "avg":
                pass_k_metrics = average_accuracy(sample_uids, sample_scores, val_mode, data_sources)
            else:
                pass_k_metrics = self._compute_pass_k_metrics(
                    sample_uids, sample_scores, val_mode, data_sources
                )
            metric_dict.update(pass_k_metrics)

        return metric_dict, pass_k_metrics

    def _normalize_val_mode(self, val_mode):
        """Normalize val_mode to a sorted, deduplicated list of integers.

        Supports:
        - None / [] → []
        - int (e.g. 5) → [5]
        - str (e.g. "5" or "pass@5") → [5]
        - list of int/str (e.g. [1, 5] or ["pass@1", "pass@5"]) → [1, 5]

        Args:
            val_mode: Input from config (None, int, str, or list).

        Returns:
            List[int]: Sorted, deduplicated list of integer k values.
        """
        if val_mode is None:
            return []
        if isinstance(val_mode, (int, float)):
            return [int(val_mode)]
        if isinstance(val_mode, str):
            nums = re.findall(r"\d+", val_mode)
            return [int(nums[0])] if nums else []
        if isinstance(val_mode, (list, tuple, ListConfig)):
            result = []
            for item in val_mode:
                if isinstance(item, (int, float)):
                    result.append(int(item))
                elif isinstance(item, str):
                    nums = re.findall(r"\d+", item)
                    if nums:
                        result.append(int(nums[0]))
            return sorted(set(result))
        return []

    def _compute_pass_k_metrics(
        self, sample_uids, sample_scores, val_mode, data_sources
    ):
        """Compute pass@k metrics for each k in val_mode.

        pass@k = fraction of prompts where any of the first k responses is correct (score > 0).

        Args:
            sample_uids: List of prompt UIDs (repeated n times for n responses per prompt).
            sample_scores: List of scores corresponding to each response.
            val_mode: List of integers, e.g. [1, 5].
            data_sources: Numpy array of data source labels for each response.

        Returns:
            dict: Metrics keyed like "val-core/{data_source}/acc/pass@{k}".
        """
        # Group scores by uid (preserving order — first k responses per prompt)
        uid2scores = defaultdict(list)
        uid2datasource = {}
        for uid, score, ds in zip(sample_uids, sample_scores, data_sources):
            uid2scores[uid].append(score)
            uid2datasource[uid] = ds

        data_source_set = set(uid2datasource.values())
        metrics = {}

        for ds in data_source_set:
            ds_uids = [u for u, d in uid2datasource.items() if d == ds]
            for k in val_mode:
                passed = 0
                total = 0
                for uid in ds_uids:
                    scores = uid2scores[uid]
                    if len(scores) >= k:
                        if any(s > 0 for s in scores[:k]):
                            passed += 1
                        total += 1
                if total > 0:
                    metrics[f"val-core/{ds}/acc/pass@{k}"] = passed / total

        return metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        for role, role_name in [(Role.Actor, "actor"), (Role.Rollout, "rollout")]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=role_name,
            )
            self.resource_pool_to_cls[resource_pool][role_name] = role_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_wg = all_wg["actor"]
        self.rollout_wg = all_wg["rollout"]
        self.actor_wg.init_model()
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)

        self.create_weight_sync_group()
        self.sync_rollout_weights()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async" and self._is_rollout:
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.rollout_wg,
            )

    def create_weight_sync_group(self):
        master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote())
        master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
        world_size = len(self.actor_wg.workers + self.rollout_wg.workers)
        self.actor_wg.create_weight_sync_group(
            master_address,
            master_port,
            0,
            world_size,
        )
        ray.get(
            self.rollout_wg.create_weight_sync_group(
                master_address,
                master_port,
                len(self.actor_wg.workers),
                world_size,
            )
        )

    def sync_rollout_weights(self):
        if not self.hybrid_engine:
            self.actor_wg.sync_rollout_weights()
            ray.get(self.rollout_wg.sync_rollout_weights())

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # sync weights from actor to rollout
        self.sync_rollout_weights()

        # async generation
        gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)

        # Launch individual reward computations as each generation completes
        future_reward = None
        if self.config.reward_model.launch_reward_fn_async:
            # Store the object reference and set up callback
            future_reward = self._launch_individual_rewards.remote(
                gen_batch_output, self.config, self.tokenizer, batch.non_tensor_batch
            )

        # Return the original, now-modified `batch` and the `future_reward`
        return GenerationBatchFuture(epoch, batch, gen_batch_output, future_reward)

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(gen_batch_output, config, tokenizer, original_non_tensor_batch):
        # Get generation results
        gen_batch_result = gen_batch_output.get()

        # Repeat non_tensor_batch to match the number of responses
        n = config.actor_rollout_ref.rollout.n
        repeated_non_tensor_batch = {}
        for key, value in original_non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(value, n, axis=0)

        # Split into individual responses with preserved non_tensor_batch
        responses_split = []
        for i in range(len(gen_batch_result)):
            response_data = gen_batch_result[i : i + 1]  # Get single response
            # Add repeated non_tensor_batch values
            for key in repeated_non_tensor_batch:
                response_data.non_tensor_batch[key] = repeated_non_tensor_batch[key][i : i + 1]
            responses_split.append(response_data)

        # Launch async reward computation
        reward_futures = [
            compute_reward_async.remote(response_data, config, tokenizer) for response_data in responses_split
        ]

        # Wait for results and combine
        results = ray.get(reward_futures)
        rewards_list = [r[0] for r in results]
        extras_list = [r[1] for r in results]

        combined_reward_tensor = torch.cat(rewards_list, dim=0)
        combined_extras_dict = {}
        if extras_list and extras_list[0]:
            for key in extras_list[0].keys():
                combined_extras_dict[key] = [d[key] for d in extras_list if key in d]

        return combined_reward_tensor, combined_extras_dict

    def get_preprocess(self, batch_data_future: GenerationBatchFuture):
        '''
        Precompute reward for sampling        
        '''
        epoch, batch, gen_batch_output, future_reward = batch_data_future.get()
        
        timing_meta_info = gen_batch_output.meta_info.pop("timing", None)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        batch = deepcopy(batch).repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        # .union modifies itself so deepcopy to avoid conflict
        query_response_batch = batch.union(gen_batch_output)

        query_response_batch.batch["response_mask"] = compute_response_mask(query_response_batch)
        
        if self.use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(query_response_batch)
        # Use the pre-launched future reward if available
        if self.config.reward_model.launch_reward_fn_async:
            # future_reward was already started in _async_gen_next_batch
            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(query_response_batch, self.reward_fn)
            
        if isinstance(reward_tensor, DataProto):
            reward_tensor = reward_tensor.batch["reward"]
            
        # Store reward in batch for reuse
        query_response_batch.batch["token_level_scores"] = reward_tensor
        
        # Store reward_extra_infos for later use
        if reward_extra_infos_dict:
            query_response_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
        
        # compute rewards. apply_kl_penalty if available
        if self.config.algorithm.use_kl_in_reward:
            query_response_batch, kl_metrics = apply_kl_penalty(
                query_response_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
        else:
            query_response_batch.batch["token_level_rewards"] = query_response_batch.batch["token_level_scores"]
        
        # MY: add compute ref_log_prob at get_preprocess
        self.qiup_logger.info(f"start ref_policy_wg.compute_ref_log_prob", indentation_num=2)
        if self.use_reference_policy:
            # compute reference log_prob
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(query_response_batch)
            else:
                ref_log_prob = self.actor_wg.compute_ref_log_prob(query_response_batch)
            query_response_batch = query_response_batch.union(ref_log_prob)
        
        return GenerationBatchFuture(
            epoch=epoch, 
            batch=query_response_batch,  # Now contains preprocessed data
            gen_batch_output=gen_batch_output, 
            future_reward=None
        ), timing_meta_info

    def _save_checkpoint(self, val_metrics: dict | None = None):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        separate_retention = isinstance(self.ckpt_manager, PeriodicBestCheckpointManager)
        if separate_retention:
            # Worker FIFO cleanup cannot distinguish periodic checkpoints from best.
            max_actor_ckpt_to_keep = None
            max_critic_ckpt_to_keep = None

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
            
        save_metadata = dict(
            step=self.global_steps, ckpt_path=local_global_step_folder, val_metrics=val_metrics
        )
        if separate_retention:
            save_freq = self.config.trainer.save_freq
            save_metadata["periodic"] = self.global_steps > 0 and save_freq > 0 and (
                self.global_steps % save_freq == 0 or self.global_steps >= self.total_training_steps
            )
        self.ckpt_manager.update_after_save(**save_metadata)
        keep_candidates = [
            x
            for x in [max_actor_ckpt_to_keep, max_critic_ckpt_to_keep]
            if x is not None and x > 0
        ]
        if keep_candidates:
            max_ckpt_to_keep = max(keep_candidates)
            self.ckpt_manager.cleanup_old_checkpoints(
                max_ckpt_to_keep=max_ckpt_to_keep,
            )

    
    
    
    import torch


        


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # MY: params
        self.qiup_logger.info("start fit")
        sample_remain_num = self.config.get("project_kwargs", {}).get("sample_remain_num", None)
        use_loss_for_p_star = self.config.get("project_kwargs", {}).get("use_loss_for_p_star", False)
        select_by = self.config.get("project_kwargs", {}).get("select_by", "cur")   # cur / ref / ref_no_var
        val_only = self.config.trainer.get("val_only", False)
        val_mode = self.config.get("project_kwargs", {}).get("val_mode", None)
        if val_mode != None:
            self.ckpt_manager.set_metrics_startswith("pass")
        self.qiup_logger.log_dict(
            dict_data={
                "sample_remain_num": sample_remain_num,
                "use_loss_for_p_star": use_loss_for_p_star,
                "select_by": select_by,
                "val_only": val_only,
                "val_mode": val_mode,
                "ckpt_metrics_startswith": self.ckpt_manager.metrics_startswith,
                "enable_replay": self.enable_replay,
                "loss_mode": self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla"),
                "total_training_steps": self.total_training_steps,
                "max_response_length": self.config.data.max_response_length,
                "trainer.resume_mode": self.config.trainer.resume_mode
            },
            title="some params"
        )
        save_best_immediately = self.config.trainer.get("save_best_immediately", True)
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        self.qiup_logger.log_dict(
            dict_data={
                "save_best_immediately": save_best_immediately,
                "remove_previous_ckpt_in_save": remove_previous_ckpt_in_save,
                "max_actor_ckpt_to_keep": self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1,
                "max_critic_ckpt_to_keep": self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1,
            },
            title="misc params"
        )

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.sync_rollout_weights()
        # MY!!
        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        val_metrics = None
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            self.qiup_logger.info("start validation before training")
            val_metrics, pass_k_metrics = self._validate(val_mode=val_mode)
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            self.qiup_logger.info(f"validation before training result: {val_metrics}")
            # self.qiup_logger.info(f"val_metrics: {val_metrics}")
            self.qiup_logger.info(f"pass_k_metrics: {pass_k_metrics}")
            if self.config.trainer.get("val_only", False):
                return
            if (
                isinstance(self.ckpt_manager, PeriodicBestCheckpointManager)
                and save_best_immediately
                and self.ckpt_manager.get_improved_metrics(pass_k_metrics)
            ):
                # The initial SFT model remains best if RL never improves it.
                self._save_checkpoint(val_metrics=val_metrics)
        else:
            self.qiup_logger.info("skip validation before training")

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()

        # Start the first asynchronous generation task.
        batch_data_future = self._async_gen_next_batch(continuous_iterator)
        if self.enable_replay:
            self.memory.update(batch_data_future)

        while batch_data_future is not None:
            self.qiup_logger.log_batch_start(self.global_steps, self.total_training_steps)
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            if do_profile:
                self.actor_wg.start_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps
            is_replay = self.global_steps >= self.config.replay.warmup and self.enable_replay
            
            if self.global_steps % len(self.train_dataloader) == 0 and self.enable_replay:
                # clear record per epoch
                self.memory.data_utilization.clear()

            with marked_timer("step", timing_raw):
                self.qiup_logger.info(f"start Data Acquisition", indentation_num=1)
                # Phase 1: Data Acquisition
                # wait for the previous batch
                with marked_timer("wait_prev_gen", timing_raw, color="red"):
                    self.qiup_logger.info(f"start get_preprocess", indentation_num=2)
                    batch, timing = self.get_preprocess(batch_data_future)
                    if self.enable_replay:
                        with marked_timer("sample", timing_raw):
                            self.memory.add(batch)
                            # self.qiup_logger.info(f"memory.buffer:", indentation_num=2)
                            # for i, obj in enumerate(self.memory.buffer):
                            #     _, temp_batch, temp_gen_batch_result, _ = obj.get()
                            #     self.qiup_logger.info(f"{i}: bath - {temp_batch.batch.keys()}", indentation_num=3)
                            #     self.qiup_logger.info(f"{i}: gen_batch_result - {temp_gen_batch_result.batch.keys()}", indentation_num=3)
                            if is_replay:
                                with torch.no_grad():
                                    self.qiup_logger.info(f"start sample_from_buffer", indentation_num=2)
                                    batch_data_future, sample_utils, sample_metrics = self.sample_from_buffer(
                                        remain_num=sample_remain_num,
                                        use_loss_for_p_star=use_loss_for_p_star,
                                        select_by=select_by
                                    )
                                metrics.update(sample_metrics)
                            else:
                                batch_data_future = deepcopy(batch)
                    else:
                        batch_data_future = deepcopy(batch)
                    # MY: change timing setting outside
                    timing_raw.update(timing)
                        # MY: change batch_data_future to batch
                    epoch, batch, gen_batch_output, future_reward = batch_data_future.get()
                    # if not self.enable_replay:
                    #     timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)
                
                self.qiup_logger.info(f"epoch: {epoch}", indentation_num=2)
                self.qiup_logger.info(f"start Async Launch of Next Generation", indentation_num=1)

                # Phase 2: Async Launch of Next Generation
                # asys next generation (with syns weights from actor to rollout)
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    if not is_last_step:
                        batch_data_future = self._async_gen_next_batch(continuous_iterator)
                        if self.enable_replay: 
                            self.memory.update(batch_data_future)

                # MY: no need since change batch_data_future to batch
                # if not is_replay:
                #     batch.non_tensor_batch["uid"] = np.array(
                #         [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                #     )
                #     # repeat to align with repeated responses in rollout
                #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #     batch = batch.union(gen_batch_output)
                #     batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                    
                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm:
                        reward_tensor = self.rm_wg.compute_rm_score(batch)

                    # Use the pre-launched future reward if available
                    if self.config.reward_model.launch_reward_fn_async:
                        # future_reward was already started in _async_gen_next_batch
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        
                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                self.qiup_logger.info(f"start actor_wg.compute_log_prob", indentation_num=2)
                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    if not is_replay:
                        old_log_prob = self.actor_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        
                    else:
                        if select_by == "cur":
                            if sample_remain_num == None or sample_remain_num <= 0:
                                entropys = sample_utils["entropys"]
                        else:
                            old_log_prob = self.actor_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    
                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        rollout_old_log_probs = batch.batch["rollout_log_probs"]
                        actor_old_log_probs = batch.batch["old_log_probs"]
                        attention_mask = batch.batch["attention_mask"]
                        responses = batch.batch["responses"]
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]

                        rollout_probs = torch.exp(rollout_old_log_probs)
                        actor_probs = torch.exp(actor_old_log_probs)
                        rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                        rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                        rollout_probs_diff_max = torch.max(rollout_probs_diff)
                        rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                        rollout_probs_diff_std = torch.std(rollout_probs_diff)
                        metrics.update(
                            {
                                "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                            }
                        )
                        print(rollout_probs_diff_mean)

                # MY: move to get_preprocess
                # self.qiup_logger.info(f"start ref_policy_wg.compute_ref_log_prob", indentation_num=2)
                # if self.use_reference_policy:
                #     # compute reference log_prob
                #     with marked_timer("ref", timing_raw, color="olive"):
                #         if not self.ref_in_actor:
                #             ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                #         else:
                #             ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                #         batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                self.qiup_logger.info(f"start compute_advantage", indentation_num=2)
                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    
                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Compute rollout IS weights and mismatch metrics (inherited from RayPPOTrainer)
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    
                    if select_by == "cur":
                        if self.config.replay.enable and self.config.replay.pmis and is_replay:
                            is_weights = batch.batch["rollout_is_weights"]
                            batch.batch["rollout_is_weights"] = is_weights * batch.batch["pmis_weights"].unsqueeze(-1) 
                        
                    # IS and mismatch metrics already have mismatch/ prefix
                    metrics.update(is_metrics)

                    # compute advantages, executed on the driver process

                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                    # print(batch["token_level_rewards"])
                    # print(batch["advantages"].shape)
                    # raise KeyboardInterrupt

                # compute var_of_j
                # MY: not compute var_of_j
                if not self.enable_replay:
                    tmp_metrics = self.compute_var_of_j(batch)
                    metrics.update(tmp_metrics)
                    
                # Phase 3: Model Updates (Actor & Critic)
                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                self.qiup_logger.info(f"start actor_wg.update_actor", indentation_num=2)
                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                if self.enable_replay:
                    if (sample_remain_num == None or sample_remain_num <= 0) and use_loss_for_p_star:
                        self.memory.record_reward(self.global_steps/self.total_training_steps, 
                                                  actor_output_metrics["actor/pg_loss"])
                    else:
                        # MY: update r by advantages
                        self.memory.record_reward(self.global_steps/self.total_training_steps, 
                                                verl_F.masked_mean(batch.batch["advantages"], batch.batch["response_mask"]))
                    
                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

            # validate & save
            val_metrics = {}
            checkpoint_saved_this_step = False
            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                self.qiup_logger.info(f"start validate", indentation_num=2)
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics, pass_k_metrics = self._validate(val_mode=val_mode)
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
                if save_best_immediately:
                    improved_metrics = self.ckpt_manager.get_improved_metrics(
                        val_metrics=pass_k_metrics,
                    )
                    if improved_metrics:
                        self.qiup_logger.info(
                            f"best metric improved: {improved_metrics}",
                            indentation_num=2,
                        )
                        with marked_timer("save_best_checkpoint", timing_raw, color="green"):
                            try:
                                self.qiup_logger.info(
                                    f"saving best checkpoint immediately",
                                    indentation_num=2,
                                )
                                self._save_checkpoint(
                                    val_metrics=val_metrics,
                                )
                                checkpoint_saved_this_step = True
                            except Exception as e:
                                print(f"Error in save_best_checkpoint: {e}")
                                self.qiup_logger.info(
                                    f"Error in save_best_checkpoint: {e}",
                                    indentation_num=2,
                                )
            # self.qiup_logger.info(f"val_metrics: {val_metrics}", indentation_num=2)
            self.qiup_logger.info(f"pass_k_metrics: {pass_k_metrics}", indentation_num=2)
            # save
            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                if checkpoint_saved_this_step:
                    self.qiup_logger.info(
                        "checkpoint already saved at this step due to best metric improvement, skip regular save",
                        indentation_num=2,
                    )
                else:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        try:
                            self.qiup_logger.info(f"saving checkpoint", indentation_num=2)
                            self._save_checkpoint(
                                val_metrics=val_metrics,
                            )
                        except Exception as e:
                            print(f"Error in _save_checkpoint: {e}")
                            self.qiup_logger.info(
                                f"Error in _save_checkpoint: {e}",
                                indentation_num=2,
                            )

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.actor_wg.stop_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.stop_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.stop_profile()
                if self.use_critic:
                    self.critic_wg.stop_profile()
                if self.use_rm:
                    self.rm_wg.stop_profile()

            gc.collect()

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
