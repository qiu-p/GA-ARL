# Async Replay CUDA SFT Design

## Context

`recipe/async_replay_npu` already has an FSDP SFT trainer, SFT config, and an AIME Qwen3-4B LoRA script. `recipe/async_replay` is the CUDA-oriented recipe tree and already contains CUDA RL scripts under `scripts/o1`, but it does not currently contain an SFT trainer/config implementation or populated scripts under `scripts/o1/o1_SFT_H20_aime_qwen3_4b`.

The requested implementation should mirror the NPU SFT workflow while remaining CUDA-focused for H20-style GPU execution.

## Approach

Use the upstream CUDA trainer `verl/trainer/fsdp_sft_trainer.py` as the device baseline, then fold in the useful recipe-level behavior from `recipe/async_replay_npu/fsdp_sft_trainer.py`.

This avoids carrying NPU-specific bootstrap code into `recipe/async_replay`, while preserving the local additions that matter for this project:

- `qiuputils` base directory and tensorboard setup.
- deterministic seed setup and accelerator logging.
- dataset path normalization with `normalize_dataset_paths`.
- LoRA config reuse and optional merged LoRA model export.
- checkpoint metric formatting and checkpoint retention cleanup.
- guards for disabled validation/save frequencies.

## Files

- Add `recipe/async_replay/fsdp_sft_trainer.py`.
- Add `recipe/async_replay/config/sft_trainer.yaml`.
- Add `recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/common.sh`.
- Add `recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/sft_lora_v1.sh`.

## Trainer Behavior

The CUDA SFT trainer will:

- Set `TOKENIZERS_PARALLELISM=true` and `NCCL_DEBUG=WARN`.
- Use `trainer.device=cuda` as the default.
- Build FSDP/FSDP2 training with the same SFT dataset, validation, checkpoint, resume, LoRA, Ulysses sequence-parallel, and remove-padding behavior as the NPU recipe.
- Prefer `flash_attention_2` on CUDA when `model.attn_implementation` is not explicitly configured.
- Avoid `torch_npu`, `HCCL_CONNECT_TIMEOUT`, and automatic device rewrites for NPU.

The trainer will not introduce new data formats or alter async replay RL behavior.

## Config Behavior

The CUDA SFT config will inherit the base verl `sft_trainer` config and override recipe defaults:

- `trainer.device: cuda`
- console and tensorboard logging
- disabled resume by default
- bounded checkpoint retention
- LoRA enabled by default for Qwen3-4B style SFT
- bf16 model dtype
- `use_remove_padding: true`

These defaults match the NPU SFT workflow shape while remaining CUDA-oriented.

## Script Behavior

The H20 AIME SFT script directory will include:

- `common.sh` for model, data, output, GPU, batch, LoRA, and SFT hyperparameter defaults.
- `sft_lora_v1.sh` to launch SFT with `torchrun`.

The launch script will run in the background with `nohup`, matching the existing CUDA async_replay RL scripts:

```bash
nohup torchrun --standalone ... \
    -m recipe.async_replay.fsdp_sft_trainer \
    ... > "${CONSOLE_OUTPUT_PATH}" 2>&1 &
```

It will use `CUDA_VISIBLE_DEVICES` and will not set Ascend/NPU environment variables.

## Validation

After implementation, run static verification rather than a real training job:

- Python compile check for the new trainer.
- Shell syntax check for the new scripts if `bash` is available.
- Targeted content checks to ensure the CUDA script points at `recipe.async_replay.fsdp_sft_trainer` and not the NPU recipe.

Full runtime validation is out of scope in this workspace because it requires CUDA/H20 hardware plus the configured model and dataset paths.
