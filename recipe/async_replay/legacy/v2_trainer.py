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

import uuid
from pprint import pprint

import numpy as np
import ray
import torch
import gc
from omegaconf import OmegaConf
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
)
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
from collections import deque
from ..utils import OnlinePolynomialRLS
from copy import deepcopy

from recipe.async_replay._vendor.qiuputils.logger import get_logger

class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output, future_reward=None):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        :param future_reward: Future for reward computation (optional)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.future_reward = future_reward

    def __getitem__(self, item):
        """
        Enhanced indexing for GenerationBatchFuture objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            GenerationBatchFuture: For all indexing types except single integers
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return GenerationBatchFuture(
                epoch=self.epoch,
                batch=self.batch.slice(item.start, item.stop, item.step),
                gen_batch_output=self.gen_batch_output.slice(item.start, item.stop, item.step),
                future_reward=None
                )

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, int | np.integer | list | np.ndarray | torch.Tensor):
            if isinstance(item, list):
                item = np.array(item)
            return GenerationBatchFuture(
                epoch=self.epoch,
                batch=self.batch.select_idxs(item),
                gen_batch_output=self.gen_batch_output.select_idxs(item),
                future_reward=None
                )

        # # Case 3: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    # TODO: support when gen_batch_output is DataProtoFuture
    # TODO: support when future_reward is not None
    def concat(other: list["GenerationBatchFuture"]) -> "GenerationBatchFuture":
        """Aggregates multiple generation futures into a single large batch."""
        return GenerationBatchFuture(
            epoch=other[0].epoch,
            batch=DataProto.concat([obj.batch for obj in other]),
            gen_batch_output=DataProto.concat([obj.gen_batch_output for obj in other]),
            future_reward=None
        )

    def to(self, device) -> "GenerationBatchFuture":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.gen_batch_output is not None:
            self.gen_batch_output = self.gen_batch_output.to(device)
        return self
                
    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (epoch, batch, gen_batch_result, future_reward)
                - epoch: Current epoch
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
                - future_reward: Future for reward computation if available, else None
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
        else:
            gen_batch_result = self.gen_batch_output

        return self.epoch, self.batch, gen_batch_result, self.future_reward
    
class Memory:
    """
    Replay Buffer for Off-policy RL. 
    Stores past generation results and provides reward prediction using RLS.
    """
    def __init__(self, k):
        self.k = k
        self.buffer = deque(maxlen=k)
        self.data_utilization = set()
        self.future_batch = None
        self.new_meta_info = None
        self.r = []
        self.predictor = OnlinePolynomialRLS(degree=3, l2=1e-4)

    def update(self, item: 'GenerationBatchFuture'):
        '''
        Update stored future_batch to delay .get()
        '''
        self.future_batch = item
        
    def add(self, item: 'GenerationBatchFuture'):
        """
        Adds a GenerationBatchFuture item to the buffer.
        """
        if len(self.buffer) == self.k:
            old = self.buffer.pop()
            del old
            gc.collect()
        self.buffer.appendleft(item.to("cpu"))

    def get(self):
        """
        Get from self.future_batch
        """
        if self.future_batch:
            epoch, batch, gen_batch_output, future_reward = self.future_batch.get()
        return GenerationBatchFuture(epoch, batch, gen_batch_output, future_reward)

    def record_reward(self, step, reward):
        """Updates the reward predictor with the latest training feedback."""
        if isinstance(reward, DataProto):
            reward = reward["reward"]
        if isinstance(reward, torch.Tensor):
            reward = reward.sum(-1).mean().detach().item()
        self.r.append(reward)
        print(f"actual: {reward}")
        if reward > 0:
            self.predictor.update(step, reward)
        
    def dummy_predict(self):
        return sum(self.r) / len(self.r)
        
    def predict(self, t):
        return self.predictor.predict(t)

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return f"UnlimitedBuffer with {len(self)} items."
    
class OneStepOffRayTrainer(RayPPOTrainer):
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

    def _validate(self):
        self.actor_rollout_wg = self.rollout_wg
        ret = super()._validate()
        self.actor_rollout_wg = self.actor_wg
        return ret

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
        
        return GenerationBatchFuture(
            epoch=epoch, 
            batch=query_response_batch,  # Now contains preprocessed data
            gen_batch_output=gen_batch_output, 
            future_reward=None
        ), timing_meta_info

    def compute_var_of_j(self, batch: DataProto):
        tmp_metrics = {}
        # epoch, batch, gen_batch_output, future_reward = batch_data.get() # batch是DataProto类型, batch.batch取字典信息

        # prepare for ratio_diff
        actor_log_probs = self.actor_wg.compute_log_prob(batch)
        log_probs = actor_log_probs.batch["old_log_probs"]
        # if "rollout_log_probs" in batch.batch.keys():
        #     rollout_log_probs = batch.batch["rollout_log_probs"]
        # else:
        #     rollout_log_probs = log_probs
        rollout_log_probs = batch.batch.get("rollout_log_probs", log_probs)

        response_mask = batch.batch["response_mask"].float()
        valid_tokens = response_mask.sum().clamp(min=1)  # 防止除零

        norm_adv_by_std_in_grpo = self.config.algorithm.get(
            "norm_adv_by_std_in_grpo", True
        )
        batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)

        # loss_agg_mode=naive 直接返回token-level loss(batch_size, seq_len)
        loss_mat, _ = policy_loss_fn(
            old_log_prob=rollout_log_probs,
            log_prob=log_probs,
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["response_mask"],
            loss_agg_mode="naive",
            config=self.config.actor_rollout_ref.actor,
            rollout_is_weights=None,
        )

        # === 关键：与 sample_from_buffer 完全一致的方差计算 ===
        # 1. 计算 log_prob 差值（log space，不取 exp！）
        batch_is = log_probs - rollout_log_probs  # [B, T]
        batch_is = batch_is * response_mask       # 只考虑 response 部分
        
        # 2. 计算 token-level PG loss
        token_pg_loss = loss_mat * response_mask  # [B, T]
        
        # 3. 计算 E[(token_pg_loss * batch_is)^2] - (E[token_pg_loss * batch_is])^2
        # 注意：sample_from_buffer 使用 predicted_r 作为 E[token_pg_loss * batch_is]
        # 这里我们简化，直接用当前 batch 计算
        
        # 计算 J = token_pg_loss * batch_is
        J = token_pg_loss * batch_is  # [B, T]
        
        # 有效样本数（token 级别）
        valid_count = response_mask.sum().clamp(min=1)
        
        # E[J^2]
        mean_J2 = (torch.square(J) * response_mask).sum() / valid_count
        
        # (E[J])^2
        mean_J = (J * response_mask).sum() / valid_count
        mean_J_sq = torch.square(mean_J)
        
        # 方差
        var_J = mean_J2 - mean_J_sq
        
        # 4. 转换为标量（与 sample_from_buffer 一致）
        var_scalar = var_J.detach().item()

        tmp_metrics["sampling/var_of_j_vanilla"] = var_scalar
    
        # 可选：添加诊断信息
        tmp_metrics["sampling/mean_j_vanilla"] = mean_J.detach().item()
        tmp_metrics["sampling/mean_j2_vanilla"] = mean_J2.detach().item()
        return tmp_metrics
        
    def sample_from_buffer(self):
        """
        Implements Predictive Multi-Importance Sampling (PMIS).
        Selects the best samples from the replay buffer based on ratio difference
        to the current policy, minimizing variance of the gradient estimator.
        """
        sample_metrics = {}
        sample_utils = {}
        eps = 1e-8
        
        # Concatenate all available batches in memory
        full_batch = GenerationBatchFuture.concat([obj for obj in self.memory.buffer])
        batch = full_batch.batch

        # prepare for ratio_diff
        # Importance Sampling: Compute ratio between current Actor and the policy that generated the data
        rollout_log_probs = batch.batch["rollout_log_probs"]
        actor_log_probs = self.actor_wg.compute_log_prob(batch)
        log_probs = actor_log_probs.batch["old_log_probs"]

        norm_adv_by_std_in_grpo = self.config.algorithm.get(
            "norm_adv_by_std_in_grpo", True
        )
        batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        
        # Policy Loss Calculation
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        
        # naive to return the token loss directly
        # loss_mat, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
        #     old_log_prob=rollout_log_probs,
        #     log_prob=log_probs,
        #     advantages=batch.batch["advantages"],
        #     response_mask=batch.batch["response_mask"],
        #     loss_agg_mode="naive",
        #     config=self.config.actor_rollout_ref.actor,
        #     rollout_is_weights=None,
        # )

        loss_mat, _ = policy_loss_fn(
            old_log_prob=rollout_log_probs,
            log_prob=log_probs,
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["response_mask"],
            loss_agg_mode="naive",
            config=self.config.actor_rollout_ref.actor,
            rollout_is_weights=None,
        )
        
        # simply get mean from previous r
        predicted_r = self.memory.dummy_predict()
        sample_metrics["sampling/predicted_r"] = predicted_r
        
        # compute target distribution
        # Target distribution estimation for PMIS
        # p_star corresponds to the distribution that minimizes variance
        p_star_log_probs = log_probs + torch.log(torch.abs(loss_mat / (predicted_r + eps)))

        # token-level min / max
        lmax = torch.maximum(p_star_log_probs, rollout_log_probs)
        lmin = torch.minimum(p_star_log_probs, rollout_log_probs)

        # token-level ratio_diff
        token_ratio_diff = torch.abs(1.0 - torch.exp(lmin - lmax))  # [B, T]
        response_mask = batch.batch["response_mask"].float()  # [B, T]
        token_ratio_diff = token_ratio_diff * response_mask
        
        token_cnt = response_mask.sum(-1).clamp(min=1)
        resp_ratio_diff = token_ratio_diff.sum(-1) / token_cnt   # [B]

        # Select top-k samples with the smallest off-policy degree (most relevant to current policy)
        query_size = self.config.data.train_batch_size
        rollout_n = self.config.actor_rollout_ref.rollout.n
        total_batch_size = rollout_n * query_size
        memory_size = query_size * len(self.memory)
        
        qry_ratio_diff = resp_ratio_diff.view(memory_size, rollout_n).mean(-1)
                
        query_indices = torch.topk(qry_ratio_diff, query_size, largest=False, sorted=False).indices  

        # extend to response level
        offsets = torch.arange(rollout_n).unsqueeze(0)
        indices = (query_indices.unsqueeze(-1) * rollout_n + offsets).flatten()

        # extract batch and pre-union in the case of reordering
        selected_batch = deepcopy(full_batch[indices])
        selected_old_log_probs = actor_log_probs[indices]
        
        entropys = selected_old_log_probs.batch["entropys"]
        sample_utils["entropys"] = entropys
        selected_old_log_probs.batch.pop("entropys")
        selected_batch.batch = selected_batch.batch.union(selected_old_log_probs)
        
        # Log metrics
        self.memory.data_utilization.update(selected_batch.batch.non_tensor_batch["uid"])
        sample_metrics["sampling/selected_log_prob"] = log_probs[indices].mean().detach().item()
        sample_metrics["sampling/selected_rollout_log_prob"] = rollout_log_probs[indices].mean().detach().item()
        sample_metrics["sampling/selected_ratio_diff"] = resp_ratio_diff[indices].mean().detach().item()
        sample_metrics["sampling/data_utilization"] = len(self.memory.data_utilization) / self.total_query_size
        
        # Prepare for PMIS
        batch_is = log_probs - rollout_log_probs
        batch_is = batch_is * response_mask 
        resp_is = batch_is.sum(-1) / token_cnt   # [B]
        token_pg_loss = loss_mat * response_mask
        n = np.zeros((len(self.memory),))
        sigma = np.zeros((len(self.memory),))
        
        for i in range(len(self.memory)):
            cond = (indices >= total_batch_size*i) & (indices < total_batch_size*(i+1))
            selected_local_indices = indices[cond]
            sample_metrics[f"sampling/overlap_t-{i+1}"] = cond.float().mean().detach().item()
            
            # PMIS
            n[i] = cond.sum().detach().item()
            if n[i]:
                sigma[i] = torch.square(token_pg_loss[selected_local_indices] *
                                batch_is[selected_local_indices]).mean().detach().item() - predicted_r ** 2
            
            local_indices = np.arange(query_size*i, query_size*(i+1), 1)
            local_batch_rd = qry_ratio_diff[local_indices]
            local_batch_is = resp_is[local_indices]
            
            sample_metrics[f"sampling/log_prob_t-{i+1}"] = log_probs[local_indices].mean().detach().item()
            sample_metrics[f"sampling/p_star_log_prob_t-{i+1}"] = p_star_log_probs[local_indices].mean().detach().item()
            sample_metrics[f"sampling/rollout_log_prob_t-{i+1}"] = rollout_log_probs[local_indices].mean().detach().item()
            sample_metrics[f"sampling/ratio_diff_t-{i+1}"] = resp_ratio_diff[local_indices].mean().detach().item()
            sample_metrics[f"sampling/ratio_diff_std_t-{i+1}"] = local_batch_rd.std().detach().item()
            sample_metrics[f"sampling/ratio_diff_min_t-{i+1}"] = local_batch_rd.min().detach().item()
            sample_metrics[f"sampling/is_t-{i+1}"] = local_batch_is.mean().detach().item()
            sample_metrics[f"sampling/is_std_t-{i+1}"] = local_batch_is.std().detach().item()
            sample_metrics[f"sampling/is_min_t-{i+1}"] = local_batch_is.min().detach().item()
            sample_metrics[f"sampling/is_max_t-{i+1}"] = local_batch_is.max().detach().item()
            
        inv_sigma = 1.0 / (sigma + eps)
        # sum_j n_j / sigma_j
        global_sum = (n * inv_sigma).sum()

        # S_i = (sigma_i / n_i) * (global_sum - n_i / sigma_i)
        S = (sigma / (n + eps)) * (global_sum - n * inv_sigma)
        w = 1 / (1 + S)
        pmis_weights = w * total_batch_size / (n + eps)
        # sample_utils["pmis_weights"] = torch.tensor(pmis_weights)[indices // total_batch_size]
        selected_batch.batch.batch["pmis_weights"] = torch.tensor(pmis_weights)[indices // total_batch_size]
        
        # Var(J_pmis)
        var = np.sum(w**2 * sigma / n)
        # sample_metrics["sampling/var_of_j_pmis"] = var.detach().item()
        # sample_metrics["sampling/var_of_j_pmis"] = var.item() if hasattr(var, 'item') else float(var)
        sample_metrics["sampling/var_of_j_pmis"] = float(var)

        return selected_batch, sample_utils, sample_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.qiup_logger.info("start fit")

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

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

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
                    if self.enable_replay:
                        with marked_timer("sample", timing_raw):
                            self.qiup_logger.info(f"start get_preprocess", indentation_num=2)
                            batch, timing = self.get_preprocess(self.memory.future_batch)
                            self.memory.add(batch)
                            if is_replay:
                                with torch.no_grad():
                                    self.qiup_logger.info(f"start sample_from_buffer", indentation_num=2)
                                    batch_data_future, sample_utils, sample_metrics = self.sample_from_buffer()
                                metrics.update(sample_metrics)
                        timing_raw.update(timing)
                    epoch, batch, gen_batch_output, future_reward = batch_data_future.get()
                    if not self.enable_replay:
                        timing_raw.update(gen_batch_output.meta_info["timing"])
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

                if not is_replay:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    
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
                        entropys = sample_utils["entropys"]
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

                self.qiup_logger.info(f"start ref_policy_wg.compute_ref_log_prob", indentation_num=2)
                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

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
                    self.memory.record_reward(self.global_steps/self.total_training_steps, 
                                              actor_output_metrics["actor/pg_loss"])
                    
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
            
            self.qiup_logger.info(f"start validate", indentation_num=2)
            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

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
