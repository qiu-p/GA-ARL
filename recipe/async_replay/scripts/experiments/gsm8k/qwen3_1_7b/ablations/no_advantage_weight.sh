set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

project_name="gsm8k_qwen3_1_7b"
exp_name="no_advantage_weight"
DESCRIBE="${project_name}-${exp_name}"

source "$SCRIPT_DIR/../config.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../../lib/launch_ppo.sh"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../../methods/no_advantage_weight.sh"

ga_arl_launch_ppo \
    "${GA_ARL_METHOD_ARGS[@]}" \
    actor_rollout_ref.rollout.name=vllm
