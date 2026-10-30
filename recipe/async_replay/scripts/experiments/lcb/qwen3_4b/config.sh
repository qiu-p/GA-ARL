GA_ARL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"

set -x

DATASET_NAME=lcb

########## Paths ##########
MODEL_PATH=${MODEL_PATH:-"${GA_ARL_REPO_ROOT}/models/Qwen3-4B"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${GA_ARL_REPO_ROOT}/run_data"}
BASE_DIR_PATH=${BASE_DIR_PATH:-"${RAY_DATA_HOME}/results/${project_name}/${exp_name}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
CONSOLE_OUTPUT_PATH=${CONSOLE_OUTPUT_PATH:-"${RAY_DATA_HOME}/console_output/${project_name}-${exp_name}.log"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/${DATASET_NAME}/train"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/${DATASET_NAME}/online_eval"}
CUSTOM_REWARD_FUNCTION_PATH=${CUSTOM_REWARD_FUNCTION_PATH:-"${GA_ARL_REPO_ROOT}/recipe/async_replay/reward_score/__init__.py"}
CUSTOM_DATASET_CLASS_PATH=${CUSTOM_DATASET_CLASS_PATH:-"${GA_ARL_REPO_ROOT}/recipe/async_replay/rl_dataset.py"}

########## Resource settings ##########
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
N_CPUS=${N_CPUS:-20}
n_gpus_rollout=2
n_gpus_training=$((N_GPUS_PER_NODE - n_gpus_rollout))

########## Training settings ##########
VAL_BEFORE_TRAIN=True
SAVE_FREQ=-1
TEST_FREQ=5

########## Data hyperparameters ##########
TOTAL_EPOCHS=4
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=64
MICRO_BATCH_SIZE_PER_GPU=4
TENSOR_MODEL_PARALLEL_SIZE=1

########## Training hyperparameters ##########
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=16384
MAX_NUM_BATCHED_TOKENS=20000
GRPO_ROLLOUT_N=4
LR=1e-6
SEED=42
