#!/bin/bash
echo "=== Local Job Info ==="
echo "Job ID : $$"
echo "Assigned Node: $(hostname)"
echo "Time is $(date)"

# # Load necessary modules you use
# . /etc/profile.d/modules.sh
# module load cuda/12.4
# module load cudnn/9.7.1.26/cuda12
# module load anaconda/3/24.06

# Activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate verl
# ======= 旧版环境 =========
conda activate verl-old

unset ROCR_VISIBLE_DEVICES
# 禁用 vLLM 的 HTTP 服务自动启动
export VLLM_DISABLE_LOGGING=1
export VLLM_NO_USAGE_STATS=1

# 设置 NCCL 超时（单位秒）
export NCCL_TIMEOUT=1200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 指定 GPU 型号（避免混合卡型）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际 GPU ID 调整
export TORCH_CUDA_ARCH_LIST="8.0"

# Show ray debug messages
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_EVENT_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1024
ray stop

# --- 分布式通信（关键！）---
export MASTER_ADDR=$(hostname)
# export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))  # 避免端口冲突
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
# export WORLD_SIZE=4
# export RANK=0
# export LOCAL_WORLD_SIZE=4
# export LOCAL_RANK=0

# --- NCCL 优化 ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=INFO

# Isolate ray sessions to avoid confliction
# export RAY_TMPDIR=$HOME/ray_tmp_$$
export RAY_TMPDIR=/tmp/ray_$$
mkdir -p $RAY_TMPDIR

# Go to your workspace
# cd /home/jwangxgroup/ldpan/verl-main/recipe/async_replay
# ========= 新版路径 ===========
# cd /home/jwangxgroup/ldpan/verl-main
# echo "Changing directory to: /home/jwangxgroup/ldpan/verl-main"
# ========= 旧版路径 ===========
cd /home/yqbai/async_rl/verl-old/verl-0.6.0
echo "Changing directory to: /home/yqbai/async_rl/verl-old/verl-0.6.0"

echo "=== GPU Visibility Check ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Increase MAX_THREAD_NUM
ulimit -u 65535

# =============== 启动训练 ===============
echo "=== 开始训练 ==="
# bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_1.7b_livecodebench_async_decoupled.sh
# bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/one_step_off_grpo_1.7b_aime_fsdp2_1_1_replay_4_is_sg.sh
# bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/one_step_off_grpo_4b_aime_fsdp2_1_1_replay_4_is_sg.sh
# bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/one_step_off_grpo_1.7b_livecodebench_fsdp2_1_1_replay_4_is_sg.sh
# =============== 旧版baseline =============
# bash /home/jwangxgroup/ldpan/verl-old/verl-0.6.0/shells/one_step_off_grpo_1.7b_gsm8k_fsdp2_1_1_replay_4_is_tis.sh
# nohup bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/one_step_off_grpo_1.7b_gsm8k_fsdp2_1_1_replay_4_is_sg.sh > ./logs/bash/pmis_sg.log 2>&1 &
# wait
# nohup bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_1.7b_gsm8k_async_truncted.sh > ./logs/bash/tis.log 2>&1 &
# wait
# bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_1.7b_gsm8k_async_decoupled.sh > ./logs/bash/decoupled.log 2>&1 &
# 启动串行实验序列，并放入后台
echo "=== Starting training sequence in background ==="
{
  # echo "[$(date)] Starting experiment 1"
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/one_step_off_grpo_1.7b_gsm8k_fsdp2_1_1_replay_4_is_sg.sh
  
  # echo "[$(date)] Starting experiment 2"
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_1.7b_gsm8k_async_truncted.sh
  
  # echo "[$(date)] Starting experiment 3"
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_1.7b_gsm8k_async_decoupled.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_1.7b_gsm8k_async_cispo.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ppo_qwen3_4b_gsm8k_async_cispo.sh

  bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/o1/one_step_off_grpo_4b_livecodebench_sync.sh
  bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/o1/one_step_off_grpo_4b_livecodebench_pmis_sg.sh
  bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/o1/one_step_off_grpo_1.7b_aime_sync.sh
  bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/o1/one_step_off_grpo_1.7b_aime_pmis_sg.sh

  # ===== ablation ========
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_aime_without_1_k1.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_gsm8k_without_1_k1.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_lighteval_without_1_k1.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_livecodebench_without_1_k1.sh

  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_aime_without_2_no_pmis.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_gsm8k_without_2_no_pmis.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_lighteval_without_2_no_pmis.sh
  # bash /home/yqbai/async_rl/verl-old/verl-0.6.0/shells/ablation/one_step_off_grpo_1.7b_livecodebench_without_2_no_pmis.sh
  
  echo "[$(date)] All experiments finished!"
} > run_all_$(date +%Y%m%d_%H%M%S).log 2>&1
# =============== 5baseline ================
# bash /home/jwangxgroup/ldpan/verl-main/shells/ppo_qwen3_1.7b_gsm8k_async_vanilla.sh
# bash /home/jwangxgroup/ldpan/verl-main/shells/ppo_qwen3_1.7b_gsm8k_async_decoupled.sh
# bash /home/jwangxgroup/ldpan/verl-main/shells/ppo_qwen3_1.7b_gsm8k_async_truncted.sh
# bash /home/jwangxgroup/ldpan/verl-main/shells/ppo_qwen3_1.7b_gsm8k_async_cispo.sh
# bash /home/jwangxgroup/ldpan/verl-main/shells/ppo_qwen3_1.7b_gsm8k_async_topr.sh
# nohup shells/run.sh > run_master.log 2>&1 &
