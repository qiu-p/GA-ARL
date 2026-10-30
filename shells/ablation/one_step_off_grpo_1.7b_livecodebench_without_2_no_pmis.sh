set -x

project_name='GRPO'
exp_name='GRPO-Qwen3-4b-livecodebench-without-2-one-step-off-replay-is-tis-4-response-4096'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/async_rl/verl-old/verl-0.6.0/verl_tmp"}
MODEL_PATH=${MODEL_PATH:-"/home/shared/Qwen3-4B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

FILE1="${RAY_DATA_HOME}/data/livecodebench/codegen__leetcode2k_1.3k.parquet"
FILE2="${RAY_DATA_HOME}/data/livecodebench/codegen__primeintellect_7.5k.parquet"
FILE3="${RAY_DATA_HOME}/data/livecodebench/codegen__taco_8.8k.parquet"
FILE4="${RAY_DATA_HOME}/data/livecodebench/humaneval.parquet"
FILE5="${RAY_DATA_HOME}/data/livecodebench/code.parquet"
TRAIN_FILE="[$FILE1, $FILE2, $FILE3]"
TEST_FILE="[$FILE4, $FILE5]"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

n_gpus_rollout=2
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

max_prompt_length=$((512))
max_response_length=$((4096 * 1))

use_dynamic_bsz=False
sp_size=1
fsdp_size=1
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))


python3 -m recipe.async_replay.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.policy_loss.loss_mode="vanilla"\
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_is=False \
    algorithm.rollout_is_threshold=2.0 \
    +replay.enable=True \
    +replay.k=4 \
    +replay.pmis=False \
    +replay.warmup=5 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    ray_kwargs.ray_init.num_gpus="${NGPUS_PER_NODE}" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="${n_gpus_rollout}" 2>&1 | tee logs/qwen3_4b_async_livecodebench_pmis_without_2.log