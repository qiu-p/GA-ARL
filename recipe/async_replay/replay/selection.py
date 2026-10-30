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
"""Replay candidate scoring and diagnostics; original formulas are preserved."""
from copy import deepcopy
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.trainer.ppo.ray_trainer import compute_advantage
from .buffer import GenerationBatchFuture

class ReplaySelectionMixin:
    """Selection helpers used by the asynchronous trainer."""

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
        if loss_mode in ["qr1", "qr2"]:
            loss_mat, _ = policy_loss_fn(
                old_log_prob=rollout_log_probs,
                log_prob=log_probs,
                advantages=batch.batch["advantages"],
                response_mask=batch.batch["response_mask"],
                loss_agg_mode="naive",
                config=self.config.actor_rollout_ref.actor,
                rollout_is_weights=None,
                rollout_log_prob=rollout_log_probs
            )
        else:
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

    def truncate_response(self, batch, remain_num):
        B = batch["responses"].shape[0]
        old_len = batch["response_mask"].sum(dim=1)  # [B]
        new_len = torch.clamp(old_len, max=remain_num)
        old_last = old_len - 1
        new_last = new_len - 1

        # === 1. 截断 response 相关 ===
        batch["responses"][:, remain_num:] = 0
        batch["response_mask"][:, remain_num:] = 0

        batch["rollout_log_probs"][:, remain_num:] = -1
        batch["advantages"][:, remain_num:] = 0
        batch["returns"][:, remain_num:] = 0

        # 取旧 reward
        r = batch["token_level_scores"][torch.arange(B), old_last]
        # 清空
        batch["token_level_scores"].zero_()
        batch["token_level_rewards"].zero_()
        # scatter 到新位置（只对 valid 样本）
        valid = new_len > 0
        batch["token_level_scores"][valid, new_last[valid]] = r[valid]
        batch["token_level_rewards"][valid, new_last[valid]] = r[valid]

        # === 2. 更新 input_ids 和 attention_mask ===
        prompt_len = batch["prompts"].shape[-1]
        # response 在 input_ids 中的位置
        start = prompt_len
        # mask 掉被截断的 response 部分
        batch["input_ids"][:, start + remain_num:] = 0
        batch["attention_mask"][:, start + remain_num:] = 0

        return batch

    def mask_by_entropy(self, batch_dict, remain_num):
        """
        根据 rollout_entropy 保留前 remain_num 大的 token，并更新相关的 mask。
        """
        entropy = batch_dict["rollout_entropy"]
        response_mask = batch_dict["response_mask"]
        
        # 1. 处理非法值：为了防止 topk 选中原本就是 padding 的位置（如示例中的 -1）
        # 我们将原本 response_mask 为 0 的位置熵值设为极小值
        masked_entropy = entropy.clone().float()
        masked_entropy[response_mask == 0] = -float('inf')
        
        # 2. 获取前 remain_num 大的索引
        # 如果某行的有效 token 数少于 remain_num，topk 依然会运行，但我们会通过逻辑与操作剔除
        k = min(remain_num, entropy.size(-1))
        _, topk_indices = torch.topk(masked_entropy, k, dim=-1)
        
        # 3. 生成新的布尔掩码
        # 创建一个全 0 张量，并在 topk 索引处填充 1
        new_entropy_mask = torch.zeros_like(response_mask)
        new_entropy_mask.scatter_(1, topk_indices, 1)
        
        # 确保新掩码不会把原本就是 padding 的地方变成 1（冗余校验）
        new_entropy_mask = new_entropy_mask * response_mask
        
        # 4. 更新 TensorDict 字段
        # 更新 response_mask
        batch_dict["response_mask"] = new_entropy_mask
        
        # 更新 rollout_entropy (可选：根据你的描述“其余置为 0”)
        batch_dict["rollout_entropy"] = entropy * new_entropy_mask
        
        # 5. 更新 attention_mask (仅操作 response 部分)
        # 根据结构，responses 长度为 8，input_ids 长度为 20
        # response 部分在 attention_mask 的索引为 [12:20]
        prompt_len = batch_dict["prompts"].size(-1) # 12
        
        # 仅对 response 部分进行 mask，不影响前面的 prompt 部分
        # 注意：使用 *= 确保原本为 0 的 padding 依然为 0
        batch_dict["attention_mask"][:, prompt_len:] *= new_entropy_mask
        
        return batch_dict

    def calculate_response_metrics(self, token_ratio_diff, response_mask, N):
        """
        Args:
            token_ratio_diff: Tensor [B, T], 包含每个 token 的 ratio 差异
            response_mask: Tensor [B, T], 有效 token 位置为 1, padding 为 0
            N: 计算前 N 个 token 的平均值
        Returns:
            metric1: 全局平均与前 N 个平均的差的绝对值
            metric2: 每个 response 方差的平均值
            metric3: 每个 response 总变差 (TV) 的平均值
        """
        # 1. 准备基础数据
        # 每个 response 的有效 token 数量
        token_cnt = response_mask.sum(dim=-1).clamp(min=1) 
        resp_ratio_diff = (token_ratio_diff * response_mask).sum(dim=-1) / token_cnt
        m0 = torch.abs(resp_ratio_diff).mean()
        
        # --- 指标 (1): 全局平均与前 N 个平均的差的绝对值 ---
        # 1.1 全局平均值
        # 1.2 前 N 个 token 的掩码 (考虑实际长度可能小于 N)
        prefix_mask = response_mask.clone()
        prefix_mask[:, N:] = 0
        prefix_token_cnt = prefix_mask.sum(dim=-1).clamp(min=1)
        # 前 N 个 token 的平均值
        prefix_ratio_diff = (token_ratio_diff * prefix_mask).sum(dim=-1) / prefix_token_cnt
        # 1.3 差的绝对值的 batch 平均
        metric1 = torch.abs(resp_ratio_diff - prefix_ratio_diff).mean()
        
        # --- 指标 (2): 每个 response 方差的平均值 ---
        # 方差公式: E[X^2] - (E[X])^2
        mean_of_sq = ( (token_ratio_diff**2) * response_mask ).sum(dim=-1) / token_cnt
        resp_var = mean_of_sq - (resp_ratio_diff ** 2)
        # 确保由于浮点精度不会出现微小的负数
        resp_var = resp_var.clamp(min=0)
        metric2 = resp_var.mean()
        
        # --- 指标 (3): 每个 response 总变差 (TV) 的平均值 ---
        # 计算相邻差值: [B, T-1]
        diffs = torch.abs(torch.diff(token_ratio_diff, dim=-1))
        # 只有当相邻的两个 token 都是有效时，diff 才有效
        # 掩码形状对应变为 [B, T-1]
        tv_mask = response_mask[:, :-1] * response_mask[:, 1:]
        # 每个 response 的总变差
        resp_tv = (diffs * tv_mask).sum(dim=-1)
        metric3 = resp_tv.mean()

        # --- 指标 (4): 选择的差异 ---
        query_size = self.config.data.train_batch_size
        rollout_n = self.config.actor_rollout_ref.rollout.n
        memory_size = query_size * len(self.memory)
        # Select top-k samples with the smallest off-policy degree (most relevant to current policy)
        qry_ratio_diff = resp_ratio_diff.view(memory_size, rollout_n).mean(-1)
        query_indices = torch.topk(qry_ratio_diff, query_size, largest=False, sorted=False).indices
        prefix_qry_ratio_diff = prefix_ratio_diff.view(memory_size, rollout_n).mean(-1)
        prefix_query_indices = torch.topk(prefix_qry_ratio_diff, query_size, largest=False, sorted=False).indices
        # compute 重合率
        # 将 tensor 转换为 set 进行集合操作
        set_indices = set(query_indices.tolist())
        set_prefix_indices = set(prefix_query_indices.tolist())
        intersection = set_indices.intersection(set_prefix_indices) # 计算交集
        overlap_count = len(intersection)
        metric4 = overlap_count / query_size
        # compute Jaccard 相似度
        union = set_indices.union(set_prefix_indices)
        metric5 = len(intersection) / len(union)
        
        return m0, metric1, metric2, metric3, metric4, metric5

    def calculate_vs_ref_metrics(self, token_ratio_diff, ref_token_ratio_diff, response_mask, label="ref"):
        """
        Args:
            token_ratio_diff: Tensor [B, T], 包含每个 token 相较于 p_star 的离策略度
            ref_token_ratio_diff: Tensor [B, T], 包含每个 token 相较于 ref_p_star 的离策略度
            response_mask: Tensor [B, T], 有效 token 位置为 1, padding 为 0
            N: 计算前 N 个 token 的平均值
        Returns:
            metric1: 全局平均与前 N 个平均的差的绝对值
            metric2: 每个 response 方差的平均值
            metric3: 每个 response 总变差 (TV) 的平均值
        """
        # 1. 准备基础数据
        # 每个 response 的有效 token 数量
        token_cnt = response_mask.sum(dim=-1).clamp(min=1) 
        resp_ratio_diff = (token_ratio_diff * response_mask).sum(dim=-1) / token_cnt
        ref_resp_ratio_diff = (ref_token_ratio_diff * response_mask).sum(dim=-1) / token_cnt
        m0 = torch.abs(resp_ratio_diff).mean()
        ref_m0 = torch.abs(ref_resp_ratio_diff).mean()

        # choose
        query_size = self.config.data.train_batch_size
        rollout_n = self.config.actor_rollout_ref.rollout.n
        memory_size = query_size * len(self.memory)
        # Select top-k samples with the smallest off-policy degree (most relevant to current policy)
        qry_ratio_diff = resp_ratio_diff.view(memory_size, rollout_n).mean(-1)
        query_indices = torch.topk(qry_ratio_diff, query_size, largest=False, sorted=False).indices
        ref_qry_ratio_diff = ref_resp_ratio_diff.view(memory_size, rollout_n).mean(-1)
        ref_query_indices = torch.topk(ref_qry_ratio_diff, query_size, largest=False, sorted=False).indices
        
        # [query_size, query_size]
        same_matrix = query_indices[:, None] == ref_query_indices[None, :]
        intersection_cnt = same_matrix.any(dim=1).sum()

        union_cnt = query_indices.numel() + ref_query_indices.numel() - intersection_cnt

        overlap_rate = intersection_cnt.float() / query_indices.numel()
        jaccard_sim = intersection_cnt.float() / union_cnt.clamp(min=1)

        selected_pstar_degree_by_scheme1 = qry_ratio_diff[query_indices].mean()
        selected_pstar_degree_by_scheme2 = qry_ratio_diff[ref_query_indices].mean()
        selected_ref_pstar_degree_by_scheme1 = ref_qry_ratio_diff[query_indices].mean()
        selected_ref_pstar_degree_by_scheme2 = ref_qry_ratio_diff[ref_query_indices].mean()

        metrics = {
            f"sampling-vs_{label}/overlap_rate": overlap_rate.detach(),
            f"sampling-vs_{label}/jaccard": jaccard_sim.detach(),
            f"sampling-vs_{label}/scheme1_selected_pstar_degree": selected_pstar_degree_by_scheme1.detach(),
            f"sampling-vs_{label}/scheme2_selected_pstar_degree": selected_pstar_degree_by_scheme2.detach(),
            f"sampling-vs_{label}/scheme1_ref_selected_pstar_degree": selected_ref_pstar_degree_by_scheme1.detach(), # *
            f"sampling-vs_{label}/scheme2_ref_selected_pstar_degree": selected_ref_pstar_degree_by_scheme2.detach(),
            f"sampling-vs_{label}/m0": m0.detach(),
            f"sampling-vs_{label}/ref_m0": ref_m0.detach(), # *
        }

        return metrics

    def sample_from_buffer(
            self,
            remain_num=None,
            use_loss_for_p_star=False,
            select_by_entropy=False,
            select_by="cur"
        ):
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
        advantages = batch.batch["advantages"]

        if remain_num == None or remain_num <= 0:
            batch_for_select = deepcopy(batch)
        elif select_by_entropy:
            assert "rollout_entropy" in batch.batch, "select_by_entropy==True require rollout_entropy"
            remain_num = min(256, batch.batch["responses"].shape[-1])
            batch_for_select = deepcopy(batch)
            batch_for_select.batch = self.mask_by_entropy(batch_for_select.batch, remain_num)
        else:
            # MY: truncate response
            remain_num = min(256, batch.batch["responses"].shape[-1])
            batch_for_select = deepcopy(batch)
            batch_for_select.batch = self.truncate_response(batch_for_select.batch, remain_num)

        # prepare for ratio_diff
        # Importance Sampling: Compute ratio between current Actor and the policy that generated the data
        rollout_log_probs = batch_for_select.batch["rollout_log_probs"]
        
        # decide which to use for select
        if select_by == "cur":
            actor_log_probs = self.actor_wg.compute_log_prob(batch_for_select)
            log_probs = actor_log_probs.batch["old_log_probs"]
            if (remain_num == None or remain_num <= 0) and use_loss_for_p_star:
                loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
                policy_loss_fn = get_policy_loss_fn(loss_mode)
                loss_mat, _ = policy_loss_fn(
                    old_log_prob=rollout_log_probs,
                    log_prob=log_probs,
                    advantages=batch.batch["advantages"],
                    response_mask=batch.batch["response_mask"],
                    loss_agg_mode="naive",
                    config=self.config.actor_rollout_ref.actor,
                    rollout_is_weights=None,
                )
                vector_for_p_star_computation = loss_mat
            else:
                vector_for_p_star_computation = advantages

            # simply get mean from previous r
            predicted_r = self.memory.dummy_predict()
            sample_metrics["sampling/predicted_r"] = predicted_r
            
            # compute target distribution
            # Target distribution estimation for PMIS
            # p_star corresponds to the distribution that minimizes variance
            p_star_log_probs = log_probs + torch.log(torch.abs(vector_for_p_star_computation / (predicted_r + eps)))   # MY: change loss_mat to advantages
        elif select_by == "ref_no_var":
            ref_log_probs = batch.batch["ref_log_prob"]
            # predicted_ref_2_r = self.memory.dummy_predict_ref_2_r()
            # sample_metrics["sampling/predicted_ref_2_r"] = predicted_ref_2_r
            # p_star_log_probs = ref_log_probs + advantages - torch.log(torch.as_tensor(predicted_ref_2_r + eps, device=ref_log_probs.device))
            predicted_ref_r = self.memory.dummy_predict_ref_r()
            sample_metrics["sampling/predicted_ref_r"] = predicted_ref_r
            p_star_log_probs = ref_log_probs + advantages - torch.log(torch.as_tensor(predicted_ref_r + eps, device=ref_log_probs.device))
        elif select_by == "ref":
            ref_log_probs = batch.batch["ref_log_prob"]
            predicted_ref_r = self.memory.dummy_predict_ref_r()
            sample_metrics["sampling/predicted_ref_r"] = predicted_ref_r
            p_star_log_probs = ref_log_probs + advantages + torch.log(torch.abs((advantages + eps) / (predicted_ref_r + eps)))
        else:
            raise TypeError(f"select_by {select_by} is not supported")

        # token-level min / max
        lmax = torch.maximum(p_star_log_probs, rollout_log_probs)
        lmin = torch.minimum(p_star_log_probs, rollout_log_probs)

        # token-level ratio_diff
        token_ratio_diff = torch.abs(1.0 - torch.exp(lmin - lmax))  # [B, T]
        response_mask = batch_for_select.batch["response_mask"].float()  # [B, T]
        token_ratio_diff = token_ratio_diff * response_mask

        # MY: add 展示同一 response 前后 ratio_diff 的差距
        # 计算： 方差 总变差 
        if remain_num == None or remain_num <= 0:
            N_remain_num = 256
        else:
            N_remain_num = remain_num
        m0, m1, m2, m3, m4, m5 = self.calculate_response_metrics(token_ratio_diff, response_mask, N=N_remain_num)
        sample_metrics["sampling-ratio_diff/mean"] = m0
        sample_metrics[f"sampling-ratio_diff/all_vs_N={N_remain_num}"] = m1
        sample_metrics["sampling-ratio_diff/variance"] = m2
        sample_metrics["sampling-ratio_diff/total_variation"] = m3
        sample_metrics["sampling-ratio_diff/overlap_ratio"] = m4
        sample_metrics["sampling-ratio_diff/jaccard_sim"] = m5
        if select_by == "cur":
            ref_log_probs = batch.batch["ref_log_prob"]
            predicted_ref_r = self.memory.dummy_predict_ref_r()
            sample_metrics["sampling/predicted_ref_r"] = predicted_ref_r
            ref_p_star_log_probs = ref_log_probs + advantages + torch.log(torch.abs((advantages + eps) / (predicted_ref_r + eps)))
            # token-level min / max
            ref_lmax = torch.maximum(ref_p_star_log_probs, rollout_log_probs)
            ref_lmin = torch.minimum(ref_p_star_log_probs, rollout_log_probs)
            # token-level ratio_diff
            ref_token_ratio_diff = torch.abs(1.0 - torch.exp(ref_lmin - ref_lmax))  # [B, T]
            ref_token_ratio_diff = ref_token_ratio_diff * response_mask
            sample_vs_ref_metrics = self.calculate_vs_ref_metrics(token_ratio_diff, ref_token_ratio_diff, response_mask)
            sample_metrics.update(sample_vs_ref_metrics)
            
            predicted_ref_2_r = self.memory.dummy_predict_ref_2_r()
            sample_metrics["sampling/predicted_ref_2_r"] = predicted_ref_2_r
            ref_2_p_star_log_probs = ref_log_probs + advantages - torch.log(torch.as_tensor(predicted_ref_2_r + eps, device=ref_log_probs.device))
            # token-level min / max
            ref_2_lmax = torch.maximum(ref_2_p_star_log_probs, rollout_log_probs)
            ref_2_lmin = torch.minimum(ref_2_p_star_log_probs, rollout_log_probs)
            # token-level ratio_diff
            ref_2_token_ratio_diff = torch.abs(1.0 - torch.exp(ref_2_lmin - ref_2_lmax))  # [B, T]
            ref_2_token_ratio_diff = ref_2_token_ratio_diff * response_mask
            sample_vs_ref_metrics = self.calculate_vs_ref_metrics(token_ratio_diff, ref_2_token_ratio_diff, response_mask, label="ref_2")
            sample_metrics.update(sample_vs_ref_metrics)
        elif select_by == "ref":
            ref_2_p_star_log_probs = ref_log_probs + advantages - torch.log(
                torch.abs(torch.as_tensor(predicted_ref_r + eps, device=ref_log_probs.device))
            )
            # token-level min / max
            ref_2_lmax = torch.maximum(ref_2_p_star_log_probs, rollout_log_probs)
            ref_2_lmin = torch.minimum(ref_2_p_star_log_probs, rollout_log_probs)
            # token-level ratio_diff
            ref_2_token_ratio_diff = torch.abs(1.0 - torch.exp(ref_2_lmin - ref_2_lmax))  # [B, T]
            ref_2_token_ratio_diff = ref_2_token_ratio_diff * response_mask
            sample_vs_ref_metrics = self.calculate_vs_ref_metrics(token_ratio_diff, ref_2_token_ratio_diff, response_mask, label="ref_2")
            sample_metrics.update(sample_vs_ref_metrics)
        elif select_by == "ref_no_var":
            pass
        else:
            raise TypeError(f"select_by {select_by} is not supported")
        
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
        
        if select_by == "cur":
            selected_old_log_probs = actor_log_probs[indices]
            entropys = selected_old_log_probs.batch["entropys"]
            sample_utils["entropys"] = entropys
            if remain_num == None or remain_num <= 0:
                selected_old_log_probs.batch.pop("entropys")
                selected_batch.batch = selected_batch.batch.union(selected_old_log_probs)
        else:
            # MY: remove union selected_old_log_probs
            pass

        # Log metrics
        self.memory.data_utilization.update(selected_batch.batch.non_tensor_batch["uid"])
        sample_metrics["sampling/selected_rollout_log_prob"] = rollout_log_probs[indices].mean().detach().item()
        sample_metrics["sampling/selected_ratio_diff"] = resp_ratio_diff[indices].mean().detach().item()
        sample_metrics["sampling/data_utilization"] = len(self.memory.data_utilization) / self.total_query_size
        if select_by == "cur":
            sample_metrics["sampling/selected_log_prob"] = log_probs[indices].mean().detach().item()
        
        for i in range(len(self.memory)):
            cond = (indices >= total_batch_size*i) & (indices < total_batch_size*(i+1))
            selected_local_indices = indices[cond]
            sample_metrics[f"sampling/overlap_t-{i+1}"] = cond.float().mean().detach().item()
            
            local_indices = np.arange(query_size*i, query_size*(i+1), 1)
            local_batch_rd = qry_ratio_diff[local_indices]
            
            sample_metrics[f"sampling/p_star_log_prob_t-{i+1}"] = p_star_log_probs[local_indices].mean().detach().item()
            sample_metrics[f"sampling/rollout_log_prob_t-{i+1}"] = rollout_log_probs[local_indices].mean().detach().item()
            sample_metrics[f"sampling/ratio_diff_t-{i+1}"] = resp_ratio_diff[local_indices].mean().detach().item()
            sample_metrics[f"sampling/ratio_diff_std_t-{i+1}"] = local_batch_rd.std().detach().item()
            sample_metrics[f"sampling/ratio_diff_min_t-{i+1}"] = local_batch_rd.min().detach().item()
        if select_by == "cur":
            token_pg_loss = vector_for_p_star_computation * response_mask # MY: change loss_mat to vector_for_p_star_computation
            # Prepare for PMIS
            batch_is = log_probs - rollout_log_probs
            batch_is = batch_is * response_mask 
            resp_is = batch_is.sum(-1) / token_cnt   # [B]
            
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

                local_batch_is = resp_is[local_indices]
                
                sample_metrics[f"sampling/log_prob_t-{i+1}"] = log_probs[local_indices].mean().detach().item()
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
        else:
            # MY: not add pmis_weights
            pass

        del batch_for_select
        return selected_batch, sample_utils, sample_metrics

