# Ray SFT Recipe 设计

## 背景

`verl_-cl-master` 里的旧 SFT 实现是 Ray 架构。它和 PPO 有相似的整体框架：Hydra 入口启动 Ray，`TaskRunner` 创建 role 到 worker 的映射，worker 通过 resource pool 分配，actor worker 负责 SFT 更新，验证阶段使用 rollout 生成，并通过 rule-based reward function 对生成结果打分。

目标实现放在 `verl-0.6.0/recipe/sft` 下，并尽量保持 `recipe/async_replay` 和 `recipe/async_replay_npu` 已有 recipe 的组织方式。`recipe/sft/reward_score` 目录已经存在，默认作为当前 SFT recipe 的本地 reward 函数目录使用。

## 目标

- 新增一个 Ray SFT recipe，使 SFT 的架构和 PPO 风格的 recipe trainer 对齐。
- 保留旧 SFT 的关键行为：Ray trainer、actor-rollout worker、rollout 验证、rule-based reward 验证、验证结果 dump、checkpoint resume、best checkpoint。
- 使用 `verl-0.6.0` recipe 的数据处理模式，尤其参考 `recipe/async_replay/rl_dataset.py` 的路径归一化、多 parquet 加载、元信息保留、max-sample 采样，以及 prompt-only rollout 验证 batch。
- eval/test 结果收集方式向 `verl-0.6.0` 的 PPO 和 async_replay trainer 对齐，包括 validation generation 记录、reward extra info 收集、指标聚合、JSONL dump、Tracking 记录和 qiuputils 日志。
- checkpoint 和日志逻辑向新的 recipe 对齐，使用 `qiuputils`，包括 `CheckpointManager`、`get_logger`、`make_dir`、确定性 seed 设置、tensorboard 目录设置等。
- LoRA 做成可配置项。启用 LoRA 时，是否 merge 成完整模型再保存也做成可配置项。

## 非目标

- 不替换现有 PPO recipe。
- 不把这个实现做成轻量级本地 `fsdp_sft_trainer.py` 风格。
- 除非实现时发现 import 问题，否则不移动、不重写你已经复制到 `recipe/sft/reward_score` 的 reward 代码。
- 第一版不实现 verifier 或 LLM-as-judge reward worker，因为当前 reward 全部是 rule-based。
- 不引入新的数据格式，只兼容已有 verl SFT 和 RLHF parquet 约定。

## 计划新增文件

- `recipe/sft/main_sft.py`
- `recipe/sft/ray_trainer.py`
- `recipe/sft/rl_dataset.py`
- `recipe/sft/utils.py`
- `recipe/sft/config/sft_trainer.yaml`
- `recipe/sft/scripts/sft_peft_2.sh`
- 必要的包标记文件，例如 `recipe/sft/__init__.py`

已有的 `recipe/sft/reward_score` 作为 recipe-local scoring code 保留。默认脚本和配置会把 `custom_reward_function.path` 和 `reward_model.custom_reward_function.path` 指向 `recipe/sft/reward_score/__init__.py`。

## 架构

`main_sft.py` 是 Hydra 和 Ray 入口。它会参考旧的 `verl.trainer.main_sft`，同时采用 `recipe/async_replay/main_ppo.py` 的启动风格。入口负责创建 base output 目录，配置 `qiuputils` 环境变量，设置确定性 seed，记录 accelerator 信息，使用 `get_ppo_ray_runtime_env` 初始化 Ray，并启动 Ray `TaskRunner`。

`TaskRunner` 负责构建 SFT 运行时：

- 根据配置选择 FSDP/FSDP2 或 Megatron 的 actor-rollout worker。
- 注册一个 `ActorRollout` role，用于 SFT 训练和验证阶段 generation。
- 第一版不注册 verifier worker。验证阶段默认通过 `val_reward_fn` 调用 `recipe/sft/reward_score/__init__.py` 中的 rule-based `compute_score`。
- 构建一个 global resource pool，规模来自 `trainer.n_gpus_per_node * trainer.nnodes`。
- 从 `actor_rollout_ref.model.path` 加载 tokenizer 和 processor。
- 通过 `verl.trainer.ppo.reward.load_reward_manager` 加载 `reward_fn` 和 `val_reward_fn`。
- 通过 `recipe.sft.rl_dataset` 创建训练集、验证集和 sampler。
- 启动 `RaySFTTrainer`。

`ray_trainer.py` 管理 Ray SFT 训练循环。主体行为继承旧 SFT trainer，但 checkpoint 和 logging 风格向新的 recipe 靠拢：

- `_create_dataloader` 使用 `StatefulDataLoader` 和 recipe 数据集。
- `init_workers` 创建 `actor_rollout` worker group，以及可选的 reward model worker group。
- `fit` 负责加载 checkpoint、训练前可选验证、用 supervised batch 更新 actor、按频率验证、按频率保存 checkpoint、记录 metrics，并支持 `val_only`。
- `_validate` 把 prompt-only 验证 batch 转成 rollout 请求，生成 response，调用 validation reward manager，聚合按 data source 和 field 的指标，按需 dump validation jsonl，并更新 best-checkpoint 跟踪。
- `_save_checkpoint` 保存 actor checkpoint、dataloader state、latest checkpoint tracker，并更新 qiuputils checkpoint 元信息。
- `_save_best_checkpoint` 按配置的验证指标保存 best actor checkpoint。

`rl_dataset.py` 负责连接 SFT 训练数据和 RL-style 验证数据：

- 训练数据集：构建 prompt+response 序列，以及 actor SFT update 需要的 `loss_mask`。
- 验证数据集：返回 prompt-only RLHF-style row，兼容 rollout generation 和 reward scoring。
- 两类数据集都尽量保留 `data_source`、`ability`、`reward_model`、`extra_info`、`raw_prompt_ids`、`tools_kwargs`、`interaction_kwargs`、`dataframe_index` 等字段。
- 数据路径处理参考 `recipe/async_replay/rl_dataset.py`：目录输入展开为 parquet 文件，多文件 concat，`train_max_samples` 和 `val_max_samples` 支持子采样，shuffle 受 seed 控制。

## Eval/Test 结果收集

eval/test 的结果收集需要和 `verl-0.6.0` 的 PPO 与 `recipe/async_replay` 逻辑相似，而不是只保留旧 SFT 的简单日志。

验证阶段按如下流程收集结果：

- 从 `val_dataloader` 取 prompt-only batch，并按 `actor_rollout_ref.rollout.val_kwargs.n` repeat。
- 调用 `actor_rollout_wg.generate_sequences` 或 async rollout manager 生成 responses。
- 解码并收集 `sample_inputs`、`sample_outputs`、`sample_gts`、`sample_scores`、`data_sources`。
- 调用 `val_reward_fn(test_batch, return_dict=True)`，读取 `reward_tensor` 和可选的 `reward_extra_info`。
- 将 reward 分数写入 `reward_extra_infos_dict["reward"]`，并把 `reward_extra_info` 中的其他字段同步追加到 `reward_extra_infos_dict`。
- 对 `reward_extra_infos_dict` 做长度一致性检查，保证每个 extra metric 与样本数量对齐。
- 使用 `process_validation_metrics(data_sources, sample_inputs 或 sample_uids, reward_extra_infos_dict)` 生成按 data source 聚合的指标。
- 使用 `aggregate_mean1_by_field` 或 recipe-local field mapping 生成按 field 聚合的指标，例如 math、code、logic、science。
- 指标命名沿用 0.6.0 风格：核心指标放入 `val-core/...` 或 `val_aggregate-core/...`，辅助指标放入 `val-aux/...` 或 `val_aggregate-aux/...`。
- 如果有多轮交互信息，收集 `__num_turns__` 并记录 `val-aux/num_turns/min|max|mean`。
- 如果配置了 `trainer.validation_data_dir`，调用 `_dump_generations` 写 JSONL。每行至少包含 input、output、data source、ground truth、score、step，以及 reward extra info 和当前 metric dict。
- 如果配置了 `trainer.log_val_generations` 或 logger backend 支持表格展示，使用 `ValidationGenerationsLogger` 记录抽样生成结果。
- `fit` 中将 validation metrics 合并进当前 step 的 metrics，并通过 `Tracking.log` 和 `qiuputils` logger 同步记录。

best checkpoint 不再只依赖旧 SFT 的手写 `MetricTracker`。优先使用 `qiuputils.ai_utils.CheckpointManager` 的 improved mean metric 判断；如果当前 `qiuputils` 版本不支持某个接口，则用一个兼容层筛选 `val_aggregate-core/*/mean@1` 或 `val-core/*/mean@1` 作为 fallback。

## 数据语义

训练数据可以使用旧脚本中的嵌套字段配置：

- `data.prompt_key=extra_info`
- `data.response_key=extra_info`
- `data.prompt_dict_keys=['question']`
- `data.response_dict_keys=['answer']`

验证数据默认使用 0.6.0 RLHF-style prompt：

- `data.eval_prompt_key=prompt`
- `prompt` 默认应是可被 `tokenizer.apply_chat_template` 处理的 chat-message list。
- `reward_model.ground_truth` 默认作为 reward scoring 的 ground truth 来源。

如果训练 prompt 是普通字符串，SFT train dataset 会把它包装成单轮 user message 后再应用 chat template。如果 prompt 本身已经是 message list，则直接应用 chat template。

## Checkpoint

checkpoint 行为向新的 recipe 对齐：

- 初始化 `CheckpointManager(base_dir=config.trainer.default_local_dir, logger=...)`。
- 普通 checkpoint 保存到 `global_step_{step}/actor`。
- dataloader state 保存到 `global_step_{step}/data.pt`。
- 更新 `latest_checkpointed_iteration.txt`。
- 按 `trainer.max_actor_ckpt_to_keep` 或 `trainer.max_ckpt_to_keep` 清理旧 checkpoint。
- 成功保存后更新 qiuputils checkpoint 元信息。
- 支持 `resume_mode=disable|auto|resume_path` 和 `resume_from_path`。

LoRA checkpoint 行为可配置：

- `actor_rollout_ref.model.lora_rank=0` 表示关闭 LoRA。
- `lora_rank > 0` 表示启用 LoRA。
- `trainer.save_hf_model` 控制普通 actor checkpoint 是否额外保存 Hugging Face 格式，前提是 worker 支持该能力。
- `trainer.save_merged_lora_model` 控制 LoRA checkpoint 是否 merge 成完整 HF 模型。
- `trainer.merged_lora_model_dirname` 控制 merge 后完整模型的目录名。

如果 worker 内置的 `save_checkpoint(..., is_save_hf=True)` 已经能产出需要的完整 artifact，则优先复用 worker 能力。如果它只保存 adapter 权重，则在 recipe 层增加一个 merge helper，逻辑参考现有 FSDP SFT 的 LoRA merge 实现。

## 日志

日志逻辑遵循现有 recipe 风格：

- `base_dir` 默认是 `./{project_name}/{experiment_name}`，也允许配置覆盖。
- 通过 `qiuputils` 设置 `QIUPUTILS_BASE_DIR` 和 `TENSORBOARD_DIR`。
- `get_logger()` 记录启动信息、base dir、describe、accelerator 信息、step 进度、验证指标、checkpoint 保存和异常。
- `Tracking` 负责把 scalar metrics 写入配置的 logger backend。
- 如果配置了 `trainer.validation_data_dir`，验证生成样本会 dump 成 JSONL。
- SFT 训练本身不需要 `trainer.rollout_data_dir`，因为 rollout generation 只用于验证阶段。

## 配置

该 recipe config 应继承 PPO-style component config，而不是轻量 SFT config。配置应包含 actor、rollout、reward model、data、Ray init、profiler、trainer 等部分。

重要默认值：

- `actor_rollout_ref.is_ppo: false`
- `actor_rollout_ref.hybrid_engine: true`
- `actor_rollout_ref.actor.strategy: fsdp`
- `actor_rollout_ref.rollout.name: vllm`
- `actor_rollout_ref.rollout.n: 1`
- `actor_rollout_ref.rollout.val_kwargs.n: 1`
- `actor_rollout_ref.model.lora_rank: 32`
- `actor_rollout_ref.model.lora_alpha: 16`
- `actor_rollout_ref.model.target_modules: all-linear`
- `reward_model.enable: false`
- `reward_model.strategy` 第一版不使用 verifier；如果配置存在，仅保留为未来扩展字段。
- `custom_reward_function.path: recipe/sft/reward_score/__init__.py`
- `custom_reward_function.name: compute_score`
- `trainer.val_before_train: true`
- `trainer.val_only: false`
- `trainer.resume_mode: disable`
- `trainer.save_merged_lora_model: true`

启动脚本可以覆盖这些默认值，使其运行方式接近 `sft_peft_2.sh`。

## 验证

实现完成后的静态验证：

- 使用 `python -m py_compile` 编译新增 Python 模块。
- 通过 config print 或 dry-load 命令确认 Hydra import 能解析。
- 确认脚本调用的是 `python -m recipe.sft.main_sft`。
- 确认 reward function 路径指向 `recipe/sft/reward_score/__init__.py`。
- 确认没有意外修改 `recipe/sft`、spec 或 plan 之外的文件。

完整运行验证在当前 workspace 中不作为硬要求，因为它依赖真实模型路径、数据路径、Ray runtime 和 GPU。若本地有可用的小 parquet 数据，可以补一个 dataset 构建 smoke test。

## 默认假设

- `recipe/sft/reward_score` 是用户提供的输入，视为用户已有内容，不主动重写。
- 第一版优先支持 FSDP/FSDP2；如果 recipe 中已有匹配的 Megatron worker import，也保留 Megatron hook。
- 第一版不复制旧项目的 `verl/workers/verifier.py`。如果后续需要 LLM judge，再单独设计 `recipe/sft/verifier.py` 和对应 `reward_model.strategy=verifier` 分支。
- SFT 训练数据从配置的 `response_key` 和 `response_dict_keys` 取 supervised answer；验证数据则使用生成 response 加 reward scoring。
- best checkpoint 默认优先使用 validation aggregate metrics；如果没有 aggregate metrics，则回退到 reward mean。

## 已确定默认行为

- 如果验证 parquet 同时有 `prompt` 和 `extra_info.question`，默认使用 `prompt` 作为 rollout prompt，除非配置覆盖 `data.eval_prompt_key`。
- 如果 worker checkpoint 的 LoRA merge 行为和本地 FSDP SFT merge 行为不同，优先选择侵入性更小的实现路径，并在实现完成后说明具体 artifact 位置。
