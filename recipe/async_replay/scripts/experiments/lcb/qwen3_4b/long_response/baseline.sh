set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


project_name="lcb_qwen3_4b_long_response"
exp_name="baseline"
DESCRIBE="${project_name}-${exp_name}"

source "$SCRIPT_DIR/config.sh"

VAL_ONLY=False

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../../lib/launch_ppo.sh"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../../methods/baseline_ref_settings.sh"

ga_arl_launch_ppo \
    "${GA_ARL_METHOD_ARGS[@]}" \
    actor_rollout_ref.rollout.name=vllm \
    trainer.val_only=${VAL_ONLY} \
    trainer.resume_mode=${RESUME_MODE}
