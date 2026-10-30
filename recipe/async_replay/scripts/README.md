# Paper experiment scripts

Complete the [configuration guide](../GA_ARL_CONFIG.md) first. All commands run from the repository root. Adjust resource settings in each experiment configuration for your environment.

## Main entries

| Task | Model | Training scripts |
| --- | --- | --- |
| GSM8K | qwen3_1_7b | [ga_arl](experiments/gsm8k/qwen3_1_7b/ga_arl.sh) · [baseline](experiments/gsm8k/qwen3_1_7b/baseline.sh) · [d_arl_loss](experiments/gsm8k/qwen3_1_7b/d_arl_loss.sh) |
| GSM8K | qwen3_4b | [ga_arl](experiments/gsm8k/qwen3_4b/ga_arl.sh) · [baseline](experiments/gsm8k/qwen3_4b/baseline.sh) · [d_arl_loss](experiments/gsm8k/qwen3_4b/d_arl_loss.sh) |
| Code generation / HumanEval | qwen3_1_7b | [ga_arl](experiments/code/qwen3_1_7b/ga_arl.sh) · [baseline](experiments/code/qwen3_1_7b/baseline.sh) · [d_arl_loss](experiments/code/qwen3_1_7b/d_arl_loss.sh) |
| Code generation / HumanEval | qwen3_4b | [ga_arl](experiments/code/qwen3_4b/ga_arl.sh) · [baseline](experiments/code/qwen3_4b/baseline.sh) · [d_arl_loss](experiments/code/qwen3_4b/d_arl_loss.sh) |
| Ordering Puzzle / Zebra Puzzle | qwen3_1_7b | [ga_arl](experiments/logic/qwen3_1_7b/ga_arl.sh) · [baseline](experiments/logic/qwen3_1_7b/baseline.sh) · [d_arl_loss](experiments/logic/qwen3_1_7b/d_arl_loss.sh) |
| Ordering Puzzle / Zebra Puzzle | qwen3_4b | [ga_arl](experiments/logic/qwen3_4b/ga_arl.sh) · [baseline](experiments/logic/qwen3_4b/baseline.sh) · [d_arl_loss](experiments/logic/qwen3_4b/d_arl_loss.sh) |

Code training uses the code-domain data; select the appropriate benchmark through `TEST_FILE` and preserve its reward metadata. Logic evaluation similarly depends on the supplied evaluation files.

## Long-response LiveCodeBench

- [GA-ARL](experiments/lcb/qwen3_4b/long_response/ga_arl.sh)
- [Asynchronous baseline](experiments/lcb/qwen3_4b/long_response/baseline.sh)

These configurations use a maximum response length of 12,288 tokens. Set model, train/evaluation paths and outputs as described in the configuration guide.

## Ablations

- [experiments/code/qwen3_1_7b/ablations/buffer_k2.sh](experiments/code/qwen3_1_7b/ablations/buffer_k2.sh)
- [experiments/code/qwen3_1_7b/ablations/no_advantage_weight.sh](experiments/code/qwen3_1_7b/ablations/no_advantage_weight.sh)
- [experiments/gsm8k/qwen3_1_7b/ablations/buffer_k2.sh](experiments/gsm8k/qwen3_1_7b/ablations/buffer_k2.sh)
- [experiments/gsm8k/qwen3_1_7b/ablations/no_advantage_weight.sh](experiments/gsm8k/qwen3_1_7b/ablations/no_advantage_weight.sh)
- [experiments/logic/qwen3_1_7b/ablations/no_advantage_weight.sh](experiments/logic/qwen3_1_7b/ablations/no_advantage_weight.sh)

`no_advantage_weight.sh` uses `ref_no_var`; `buffer_k2.sh` uses a two-batch replay buffer. Keep the dataset, model and optimization settings matched to the corresponding main run.

## Launch and evaluation

```bash
bash recipe/async_replay/scripts/experiments/gsm8k/qwen3_4b/ga_arl.sh
```

Replace the path with an entry above to select another experiment. Training includes periodic validation; its frequency is configured by `TEST_FREQ`. Outputs are written to the paths you configure. Method arguments live in `methods/`; `lib/launch_ppo.sh` contains shared launch arguments. 

## Additional response-length setting

The [4,096-token code configuration](experiments/code/qwen3_4b/response_4096/config.sh) provides GA-ARL, baseline and D-ARL entries in the same directory. The standard code configuration uses its own response-length limit.

The [standard LiveCodeBench configuration](experiments/lcb/qwen3_4b/config.sh) has a baseline entry only; the `long_response` directory contains the available GA-ARL and baseline comparison.
