# Async Replay CUDA SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA/H20 FSDP SFT implementation and background launch scripts under `recipe/async_replay`.

**Architecture:** Create a CUDA-focused recipe trainer by adapting the upstream CUDA FSDP SFT trainer and preserving the project-specific enhancements from the NPU recipe. Keep config and launch scripts colocated with the existing async replay recipe tree, and keep runtime execution asynchronous via `nohup`.

**Tech Stack:** Python, PyTorch FSDP/FSDP2, Hydra/OmegaConf, verl SFT datasets, PEFT LoRA, torchrun, bash.

---

### Task 1: Add CUDA SFT Trainer

**Files:**
- Create: `recipe/async_replay/fsdp_sft_trainer.py`
- Modify: `recipe/async_replay/utils.py`
- Read: `verl/trainer/fsdp_sft_trainer.py`
- Read: `recipe/async_replay_npu/fsdp_sft_trainer.py`

- [x] **Step 1: Copy the NPU recipe trainer as the starting implementation**

Run:

```powershell
Copy-Item -LiteralPath 'recipe/async_replay_npu/fsdp_sft_trainer.py' -Destination 'recipe/async_replay/fsdp_sft_trainer.py'
```

Expected: `recipe/async_replay/fsdp_sft_trainer.py` exists.

- [x] **Step 2: Remove NPU bootstrap and set CUDA environment defaults**

Replace the top environment block with:

```python
import os

os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
```

Expected: no `torch_npu` import and no `HCCL_CONNECT_TIMEOUT` in this file.

- [x] **Step 3: Replace local utility import**

Change:

```python
from .utils import auto_set_device, log_accelerator_infos, set_deterministic
```

to:

```python
from .utils import log_accelerator_infos, set_deterministic
```

Expected: the trainer no longer imports `auto_set_device`.

- [x] **Step 4: Remove automatic device rewrites**

Delete the `auto_set_device(config)` calls from `run_sft` and `main`.

Expected: `trainer.device` stays controlled by Hydra config and script overrides.

- [x] **Step 5: Keep CUDA-safe behavior from the NPU recipe**

Ensure these functions and behaviors remain present:

```python
def _build_lora_config(self):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=self.config.model.lora_rank,
        lora_alpha=self.config.model.lora_alpha,
        target_modules=convert_to_regular_types(self.config.model.target_modules),
        bias="none",
    )
```

Expected: LoRA merge, checkpoint cleanup, `normalize_dataset_paths`, `base_dir`, qiuputils logging, and save/test frequency guards remain intact.

- [x] **Step 5a: Add CUDA utility functions used by the SFT trainer**

Add `set_deterministic` and `log_accelerator_infos` to `recipe/async_replay/utils.py` so `recipe.async_replay.fsdp_sft_trainer` can import them without relying on the NPU recipe utility module.

Expected: `recipe/async_replay/utils.py` contains CUDA-only accelerator logging and deterministic seeding helpers.

- [x] **Step 6: Compile the trainer**

Run:

```powershell
python -m py_compile recipe/async_replay/fsdp_sft_trainer.py
```

Expected: command exits 0.

### Task 2: Add CUDA SFT Hydra Config

**Files:**
- Create: `recipe/async_replay/config/sft_trainer.yaml`
- Read: `recipe/async_replay_npu/config/sft_trainer.yaml`
- Read: `verl/trainer/config/sft_trainer.yaml`

- [x] **Step 1: Create the config**

Write:

```yaml
hydra:
  searchpath:
    - file://${oc.env:HOME}/works/verl-0.6.0/verl/trainer/config

defaults:
  - sft_trainer
  - _self_

model:
  strategy: fsdp
  trust_remote_code: true
  attn_implementation: null
  enable_gradient_checkpointing: true
  lora_rank: 32
  lora_alpha: 16
  target_modules: all-linear
  fsdp_config:
    model_dtype: bf16
    cpu_offload: false
    offload_params: false
    use_torch_compile: false

trainer:
  device: cuda
  logger: ["console", "tensorboard"]
  resume_mode: disable
  max_ckpt_to_keep: 5
  remove_previous_ckpt_in_save: false
  save_merged_lora_model: true
  merged_lora_model_dirname: merged_hf_model

ulysses_sequence_parallel_size: 1
use_remove_padding: true
```

Expected: config mirrors the NPU SFT config with `trainer.device: cuda`.

- [x] **Step 2: Check config target**

Run:

```powershell
rg -n "device: cuda|device: npu|HCCL|torch_npu" recipe/async_replay/config/sft_trainer.yaml
```

Expected: only `device: cuda` is found among device-specific values.

### Task 3: Add H20 Background Launch Scripts

**Files:**
- Create: `recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/common.sh`
- Create: `recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/sft_lora_v1.sh`
- Read: `recipe/async_replay_npu/scripts/o1_SFT_aime_qwen3_4b/common.sh`
- Read: `recipe/async_replay_npu/scripts/o1_SFT_aime_qwen3_4b/sft_lora_v1.sh`
- Read: `recipe/async_replay/scripts/o1/o1_ARL_A800_lcb_qwen3_4b/GA_ARL_v1.sh`

- [x] **Step 1: Create `common.sh`**

Write a CUDA version of the NPU SFT common file with these key settings:

```bash
project_name='o1_SFT_H20_aime_qwen3_4b'
DATASET_NAME=aime
MODEL_PATH=${MODEL_PATH:-"/home/shared/Qwen3-4B"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/works/ex_datas"}
TRAINER_DEVICE=${TRAINER_DEVICE:-cuda}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
SFT_NPROC_PER_NODE=${SFT_NPROC_PER_NODE:-${N_GPUS_PER_NODE}}
```

Expected: the file exports `CUDA_VISIBLE_DEVICES` and does not reference Ascend variables.

- [x] **Step 2: Create `sft_lora_v1.sh`**

Use `nohup torchrun --standalone` and target the CUDA recipe module:

```bash
nohup torchrun --standalone \
    --nnodes="${NNODES}" \
    --nproc_per_node="${SFT_NPROC_PER_NODE}" \
    -m recipe.async_replay.fsdp_sft_trainer \
    ... \
    > "${CONSOLE_OUTPUT_PATH}" 2>&1 &
```

Expected: the script starts in the background and writes logs to `${CONSOLE_OUTPUT_PATH}`.

- [x] **Step 3: Check shell syntax**

Run:

```powershell
bash -n recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/common.sh
bash -n recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/sft_lora_v1.sh
```

Expected: both commands exit 0 when bash is available.

Result: attempted in the current Windows workspace, but `bash` resolves to a WSL stub with no installed Linux distribution. Syntax was reviewed textually instead.

### Task 4: Static Verification and Diff Review

**Files:**
- Verify: `recipe/async_replay/fsdp_sft_trainer.py`
- Verify: `recipe/async_replay/config/sft_trainer.yaml`
- Verify: `recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/common.sh`
- Verify: `recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b/sft_lora_v1.sh`

- [x] **Step 1: Verify no NPU-specific runtime code entered CUDA trainer/scripts**

Run:

```powershell
rg -n "torch_npu|HCCL_CONNECT_TIMEOUT|ASCEND_RT_VISIBLE_DEVICES|trainer.device: npu|recipe.async_replay_npu" recipe/async_replay/fsdp_sft_trainer.py recipe/async_replay/config/sft_trainer.yaml recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b
```

Expected: no matches.

- [x] **Step 2: Verify CUDA module and background launch**

Run:

```powershell
rg -n "recipe.async_replay.fsdp_sft_trainer|nohup torchrun|CUDA_VISIBLE_DEVICES|TRAINER_DEVICE=\\$\\{TRAINER_DEVICE:-cuda\\}" recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b
```

Expected: matches show CUDA trainer module, background torchrun, CUDA device visibility, and CUDA trainer default.

- [x] **Step 3: Review git diff**

Run:

```powershell
git diff -- recipe/async_replay/fsdp_sft_trainer.py recipe/async_replay/config/sft_trainer.yaml recipe/async_replay/scripts/o1/o1_SFT_H20_aime_qwen3_4b
```

Expected: diff only contains the requested CUDA SFT implementation and scripts.
