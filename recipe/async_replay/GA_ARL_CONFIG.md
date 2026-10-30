# Experiment configuration

Run commands from the repository root after completing installation in the [README](../../README.md).

## Model, data and output paths

Prepare a local Qwen3-1.7B or Qwen3-4B checkpoint matching the selected script. Use task-specific Parquet files with these fields:

- `prompt`: chat messages containing `role` and `content`.
- `data_source`: reward-routing identifier, such as `math__gsm8k`, `codegen__humaneval`, `codegen__livecodebench`, `logic__ordering_puzzle_dataset`, or `logic__zebra_puzzle_dataset`.
- `reward_model`: a mapping containing `ground_truth` in the format expected by the task scorer.
- `ability` and `extra_info`: task metadata; preserve dataset-specific reward settings.

GSM8K uses its training data; code and logic experiments use the corresponding Guru-RL-92K domain subsets. Evaluation uses GSM8K, HumanEval, LiveCodeBench, Ordering Puzzle and Zebra Puzzle as appropriate. Exact subsets and model checkpoints must be supplied separately. See `reward_score/__init__.py` and the task scorer for the ground-truth schema; changing reward metadata can change scoring behavior.

Set absolute paths before launching, for example:

```bash
export MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_FILE=/path/to/train.parquet
export TEST_FILE=/path/to/eval.parquet
export RAY_DATA_HOME="$PWD/run_data"
export BASE_DIR_PATH="$RAY_DATA_HOME/results/gsm8k-ga-arl"
export CKPTS_DIR="$RAY_DATA_HOME/checkpoints/gsm8k-ga-arl"
export CONSOLE_OUTPUT_PATH="$RAY_DATA_HOME/logs/gsm8k-ga-arl.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "$BASE_DIR_PATH" "$CKPTS_DIR" "$(dirname "$CONSOLE_OUTPUT_PATH")"
bash recipe/async_replay/scripts/experiments/gsm8k/qwen3_4b/ga_arl.sh
```

Use a separate output directory and log for each method and seed. The reward and dataset Python hooks resolve within this checkout automatically.

## Training parameters

Edit the selected experiment directory's `config.sh` for batch size, sequence lengths, learning rate, seed, epochs and GPU allocation. These variables are often assigned directly in the script, so exporting them alone does not necessarily override them. Model and data paths above support environment overrides in the documented main entries.

Common main-entry settings are a prompt batch of 256, four responses per prompt, PPO mini-batch size 64, per-GPU micro-batch size 4, learning rate `1e-6`, and seed 42. Sequence lengths and epochs vary by experiment; use its `config.sh` as the source of truth. The default allocation uses four GPUs, two for rollout and two for training. Set `N_GPUS_PER_NODE`, `n_gpus_rollout` and tensor parallelism consistently with visible devices. Adjust memory-related settings for your hardware.

Validation frequency is `TEST_FREQ`; `VAL_BEFORE_TRAIN` controls initial validation. `SAVE_FREQ=-1` disables periodic checkpoint saving in the main configurations. Set a positive saving frequency if checkpoints are required. The launcher uses `nohup`; inspect `CONSOLE_OUTPUT_PATH` for errors and progress.

## Method settings

Method arguments are defined in `scripts/methods/` and loaded by the experiment entries.

| Setting | GA-ARL main configuration |
| --- | --- |
| `replay.enable` | `True` |
| `project_kwargs.select_by` | `ref` |
| `replay.k` | `4` recent rollout batches |
| `replay.warmup` | `5` |
| `project_kwargs.sample_remain_num` | `-1`, score the full response |
| `project_kwargs.use_loss_for_p_star` | `False` |
| `actor_rollout_ref.rollout.calculate_log_probs` | `True` |
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.rollout_is` | `True` |
| `algorithm.rollout_is_threshold` | `2.0` |

The shared launcher sets actor KL loss coefficient `0.001`. The implemented reference selection target uses `ref_logprob + A + log(abs((A + eps)/(predicted_ref_r + eps)))`; the actor KL coefficient is not an additional temperature in this expression. Historical statistics approximate normalization. `replay.pmis` has no weighting effect in the `ref` branch.

For the no-replay asynchronous baseline, use `baseline.sh`. For the current-policy loss-based replay comparison, use `d_arl_loss.sh`. The documented ablations change the reference score to `ref_no_var` or reduce the buffer to `K=2`. See the [script index](scripts/README.md) for available entries.

The paper's additional eight-seed comparisons use seeds 42–49 and experiment-specific training budgets. The default scripts are not a dedicated runner for these comparisons or reward-corruption experiments; do not infer those settings from the main entries.
