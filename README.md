# GA-ARL

Gradient-Aware Asynchronous Reinforcement Learning for large language models, built on verl.
GA-ARL selects prompt groups from recent asynchronous rollouts using cached reference-policy probabilities, behavior-policy probabilities, and advantages. The implementation is in `recipe/async_replay`.

## Installation

Use Linux with NVIDIA GPUs and a CUDA-compatible PyTorch environment. The reward implementation uses Linux signals. Python 3.10 or later is required by the package metadata.

From this repository root, install the local package and dependencies:

```bash
python -m pip install -e ".[vllm,math]"
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

Install PyTorch for your CUDA environment before these commands. The repository declares a vLLM version range rather than a fully pinned experiment environment; choose compatible PyTorch, vLLM and FlashAttention builds. See [installation resources](docker/README.md) for the inherited verl containers. These commands do not download experiment models or datasets.

## Reproduce experiments

1. Prepare the Qwen3 model checkpoint and task-specific Parquet data.
2. Follow the [configuration guide](recipe/async_replay/GA_ARL_CONFIG.md) to set model, data, output paths and GPU resources.
3. Choose a training entry from the [experiment index](recipe/async_replay/scripts/README.md).

For example, after configuring paths:

```bash
bash recipe/async_replay/scripts/experiments/gsm8k/qwen3_4b/ga_arl.sh
```

The scripts launch training in the background and write console output to `CONSOLE_OUTPUT_PATH`. Validation runs during training. Check that log to confirm startup and follow progress.

The documented entries cover the available GSM8K, code-generation and logical-reasoning configurations, together with replay ablations and long-response LiveCodeBench. Model weights and exact paper data splits are not bundled. Dedicated reproduction entries for the additional matched-seed, reward-corruption and synchronous-GRPO comparisons are not provided in this index; existing defaults should not be treated as those experiment settings.

## Code structure

| Path | Purpose |
| --- | --- |
| `recipe/async_replay/main_ppo.py` | Hydra entry and Ray worker setup |
| `recipe/async_replay/trainer.py` | Asynchronous training loop |
| `recipe/async_replay/replay/` | Replay buffer and group selection |
| `recipe/async_replay/scripts/` | Experiment settings and launch scripts |
| `recipe/async_replay/reward_score/` | Task-specific rewards |
| `verl/` | Underlying training framework |

## Acknowledgments and license

This repository builds on [verl](https://github.com/volcengine/verl). Upstream copyright notices and the [Apache 2.0 license](LICENSE) are retained. Bundled components retain their respective notices.
