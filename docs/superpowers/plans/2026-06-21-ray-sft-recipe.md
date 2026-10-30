# Ray SFT Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `verl-0.6.0/recipe/sft` 中实现一个 Ray 架构的 SFT recipe，使其训练框架接近旧 SFT 和 PPO recipe，并使用 0.6.0 风格的数据、eval/test 结果收集、checkpoint、日志和 LoRA 配置。

**Architecture:** 第一版实现 rule-based reward，不实现 verifier。`main_sft.py` 负责 Hydra/Ray 启动和 worker/dataset 组装，`ray_trainer.py` 负责 SFT 训练循环、rollout 验证、metrics 聚合和 checkpoint，`rl_dataset.py` 负责 SFT train dataset 与 prompt-only eval dataset。checkpoint/logging 使用 `qiuputils`，eval/test 指标收集向 `verl.trainer.ppo.ray_trainer` 与 `recipe.async_replay.ray_trainer_wip` 对齐。

**Tech Stack:** Python, Ray, Hydra/OmegaConf, PyTorch, verl `DataProto`, `StatefulDataLoader`, PEFT LoRA, qiuputils, pytest.

---

## File Structure

- Create `recipe/sft/__init__.py`: package marker so `python -m recipe.sft.main_sft` imports cleanly.
- Create `recipe/sft/utils.py`: deterministic setup, CUDA logging helper, optional metric key helpers.
- Create `recipe/sft/rl_dataset.py`: SFT train dataset, rollout eval dataset, data path normalization wrappers, sampler creation.
- Create `recipe/sft/ray_trainer.py`: Ray SFT trainer and resource pool manager, adapted from old SFT trainer and 0.6.0 PPO validation style.
- Create `recipe/sft/main_sft.py`: Hydra entrypoint and Ray `TaskRunner`.
- Create `recipe/sft/config/sft_trainer.yaml`: PPO-style config overrides for Ray SFT.
- Create `recipe/sft/scripts/sft_peft_2.sh`: launch script close to old `sft_peft_2.sh`.
- Create `tests/recipe/sft/test_rl_dataset_on_cpu.py`: dataset and sampler unit tests with local mock tokenizer.
- Create `tests/recipe/sft/test_validation_metrics_on_cpu.py`: validation metrics helper tests without GPU/Ray.
- Do not modify `recipe/sft/reward_score` unless an import failure is discovered during verification.
- Do not add `recipe/sft/verifier.py` in this implementation.

Because the user wants to manage git manually, this plan uses "人工检查点" instead of `git commit` commands.

---

### Task 1: Package Scaffolding and Utilities

**Files:**
- Create: `verl-0.6.0/recipe/sft/__init__.py`
- Create: `verl-0.6.0/recipe/sft/utils.py`
- Test: `verl-0.6.0/tests/recipe/sft/test_validation_metrics_on_cpu.py`

- [ ] **Step 1: Create package marker**

Create `recipe/sft/__init__.py` with:

```python
"""Ray SFT recipe package."""
```

- [ ] **Step 2: Write failing utility test**

Create `tests/recipe/sft/test_validation_metrics_on_cpu.py`:

```python
import math

from recipe.sft.utils import select_best_metric_candidates


def test_select_best_metric_candidates_prefers_aggregate_core_mean():
    metrics = {
        "val-core/math/reward/mean@1": 0.5,
        "val_aggregate-core/math/reward/mean@1": 0.7,
        "val-aux/math/reward/std@1": 0.1,
    }

    selected = select_best_metric_candidates(metrics)

    assert selected == {"val_aggregate-core/math/reward/mean@1": 0.7}


def test_select_best_metric_candidates_falls_back_to_val_core_mean():
    metrics = {
        "val-core/math/reward/mean@1": 0.5,
        "val-aux/math/reward/std@1": 0.1,
    }

    selected = select_best_metric_candidates(metrics)

    assert selected == {"val-core/math/reward/mean@1": 0.5}


def test_select_best_metric_candidates_filters_non_finite_values():
    metrics = {
        "val_aggregate-core/math/reward/mean@1": float("nan"),
        "val-core/math/reward/mean@1": 0.4,
    }

    selected = select_best_metric_candidates(metrics)

    assert selected == {"val-core/math/reward/mean@1": 0.4}
    assert all(math.isfinite(value) for value in selected.values())
```

- [ ] **Step 3: Run utility test to verify it fails**

Run:

```powershell
python -m pytest tests/recipe/sft/test_validation_metrics_on_cpu.py -q
```

Expected: fails with `ModuleNotFoundError` or `ImportError` because `recipe.sft.utils` or `select_best_metric_candidates` does not exist yet.

- [ ] **Step 4: Implement `recipe/sft/utils.py`**

Create `recipe/sft/utils.py` with:

```python
import logging
import math
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic(seed=42, set_base=True, set_numpy=True, set_accelerator=True) -> None:
    if set_base:
        random.seed(seed)

    if set_numpy:
        np.random.seed(seed)

    torch.manual_seed(seed)

    if set_accelerator and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        logger.warning("Failed to enable deterministic algorithms: %s", exc)


def log_accelerator_infos(log) -> None:
    if not torch.cuda.is_available():
        log.info("No CUDA accelerator is available.")
        return

    device_info = (
        f"torch.cuda: current_device: {torch.cuda.current_device()} "
        f"device_count: {torch.cuda.device_count()} "
        f"get_device_name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    rank_info = (
        f"os.environ: RANK: {os.getenv('RANK', None)} "
        f"LOCAL_RANK: {os.getenv('LOCAL_RANK', None)} "
        f"WORLD_SIZE: {os.getenv('WORLD_SIZE', None)} "
        f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', None)}"
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        distributed_info = (
            f"torch.distributed: get_rank: {torch.distributed.get_rank()} "
            f"get_world_size: {torch.distributed.get_world_size()}"
        )
        log.info(f"CUDA is available.\n{device_info}\n{rank_info}\n{distributed_info}")
    else:
        log.info(f"CUDA is available.\n{device_info}\n{rank_info}")


def _to_finite_float(value):
    try:
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def select_best_metric_candidates(val_metrics: dict) -> dict[str, float]:
    if not val_metrics:
        return {}

    aggregate_candidates = {}
    core_candidates = {}
    for key, value in val_metrics.items():
        if not key.endswith("/mean@1") and "/mean@" not in key:
            continue
        value = _to_finite_float(value)
        if value is None:
            continue
        if key.startswith("val_aggregate-core/"):
            aggregate_candidates[key] = value
        elif key.startswith("val-core/"):
            core_candidates[key] = value

    return aggregate_candidates or core_candidates
```

- [ ] **Step 5: Run utility test to verify it passes**

Run:

```powershell
python -m pytest tests/recipe/sft/test_validation_metrics_on_cpu.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Manual checkpoint**

Record these files for the user's manual commit:

```text
verl-0.6.0/recipe/sft/__init__.py
verl-0.6.0/recipe/sft/utils.py
verl-0.6.0/tests/recipe/sft/test_validation_metrics_on_cpu.py
```

---

### Task 2: SFT Recipe Datasets and Sampler

**Files:**
- Create: `verl-0.6.0/recipe/sft/rl_dataset.py`
- Test: `verl-0.6.0/tests/recipe/sft/test_rl_dataset_on_cpu.py`
- Read: `verl-0.6.0/recipe/async_replay/rl_dataset.py`
- Read: `verl-0.6.0/verl/utils/dataset/sft_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/recipe/sft/test_rl_dataset_on_cpu.py`:

```python
import os

import pandas as pd
import pytest
from omegaconf import OmegaConf

from recipe.sft.rl_dataset import (
    SFTRolloutEvalDataset,
    SFTTrainDataset,
    create_sft_dataset,
    create_sft_sampler,
    normalize_data_paths,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert tokenize is False
        if isinstance(messages, str):
            text = messages
        else:
            text = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        return text + ("\nassistant:" if add_generation_prompt else "")

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        import torch

        ids = [max(1, (ord(ch) % 97) + 1) for ch in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        return [max(1, (ord(ch) % 97) + 1) for ch in text]


def write_parquet(path):
    dataframe = pd.DataFrame(
        [
            {
                "data_source": "math__gsm8k",
                "prompt": [{"role": "user", "content": "What is 1+1?"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "2"},
                "extra_info": {"question": "What is 1+1?", "answer": "2", "index": 0},
            },
            {
                "data_source": "logic__zebra_puzzle_dataset",
                "prompt": [{"role": "user", "content": "A before B?"}],
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": "A"},
                "extra_info": {"question": "A before B?", "answer": "A", "index": 1},
            },
        ]
    )
    dataframe.to_parquet(path)


def test_normalize_data_paths_expands_directory(tmp_path):
    file_path = tmp_path / "part.parquet"
    write_parquet(file_path)

    paths = normalize_data_paths(str(tmp_path))

    assert paths == [str(file_path)]


def test_train_dataset_builds_sft_tensors_from_nested_fields(tmp_path):
    file_path = tmp_path / "train.parquet"
    write_parquet(file_path)
    config = OmegaConf.create(
        {
            "prompt_key": "extra_info",
            "response_key": "extra_info",
            "prompt_dict_keys": ["question"],
            "response_dict_keys": ["answer"],
            "max_length": 128,
            "truncation": "error",
            "use_shm": False,
        }
    )

    dataset = SFTTrainDataset([str(file_path)], tokenizer=DummyTokenizer(), config=config)
    item = dataset[0]

    assert set(["input_ids", "attention_mask", "position_ids", "loss_mask"]).issubset(item)
    assert item["input_ids"].shape[0] == 128
    assert item["loss_mask"].sum().item() > 0
    assert item["data_source"] == "math__gsm8k"
    assert item["reward_model"]["ground_truth"] == "2"


def test_eval_dataset_returns_prompt_only_rollout_fields(tmp_path):
    file_path = tmp_path / "val.parquet"
    write_parquet(file_path)
    config = OmegaConf.create(
        {
            "eval_prompt_key": "prompt",
            "max_prompt_length": 96,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "use_shm": False,
        }
    )

    dataset = SFTRolloutEvalDataset([str(file_path)], tokenizer=DummyTokenizer(), config=config)
    item = dataset[0]

    assert set(["input_ids", "attention_mask", "position_ids", "raw_prompt_ids"]).issubset(item)
    assert item["input_ids"].shape[0] == 96
    assert item["data_source"] == "math__gsm8k"
    assert item["field"] == "math"
    assert item["dataframe_index"] == 0


def test_create_sft_dataset_selects_train_or_eval_dataset(tmp_path):
    file_path = tmp_path / "data.parquet"
    write_parquet(file_path)
    config = OmegaConf.create(
        {
            "prompt_key": "extra_info",
            "response_key": "extra_info",
            "prompt_dict_keys": ["question"],
            "response_dict_keys": ["answer"],
            "eval_prompt_key": "prompt",
            "max_length": 128,
            "max_prompt_length": 96,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "shuffle": False,
            "sampler": None,
            "use_shm": False,
        }
    )

    train_dataset = create_sft_dataset([str(file_path)], config, DummyTokenizer(), is_train=True)
    val_dataset = create_sft_dataset([str(file_path)], config, DummyTokenizer(), is_train=False)

    assert isinstance(train_dataset, SFTTrainDataset)
    assert isinstance(val_dataset, SFTRolloutEvalDataset)
    assert create_sft_sampler(config, train_dataset).__class__.__name__ == "SequentialSampler"
```

- [ ] **Step 2: Run dataset tests to verify they fail**

Run:

```powershell
python -m pytest tests/recipe/sft/test_rl_dataset_on_cpu.py -q
```

Expected: fails because `recipe.sft.rl_dataset` does not exist.

- [ ] **Step 3: Implement dataset module**

Create `recipe/sft/rl_dataset.py` by adapting these exact responsibilities:

```python
import copy
import glob
import os

import datasets
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

import verl.utils.torch_functional as verl_F
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_type
from verl.utils.model import compute_position_id_with_mask

from recipe.async_replay.rl_dataset import default_get_field_for_source


def normalize_data_paths(data_paths):
    if isinstance(data_paths, list | ListConfig):
        return list(data_paths)
    if not isinstance(data_paths, str):
        raise ValueError(f"Invalid data_paths type: {type(data_paths)}, expected str or list")
    expanded_path = os.path.expanduser(data_paths)
    if os.path.isfile(expanded_path):
        return [expanded_path]
    if os.path.isdir(expanded_path):
        parquet_files = sorted(glob.glob(os.path.join(expanded_path, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in folder: {data_paths}")
        return parquet_files
    return [data_paths]


def _series_to_item(value):
    while isinstance(value, pd.Series | np.ndarray) and len(value) == 1:
        value = value[0]
    return value


def _extract_nested(row, key, dict_keys):
    value = row[key]
    value = _series_to_item(value)
    for dict_key in dict_keys or []:
        value = _series_to_item(value)[dict_key]
    return value


def _ensure_messages(prompt):
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


class SFTTrainDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config: DictConfig, max_samples: int = -1):
        self.parquet_files = normalize_data_paths(parquet_files)
        self.tokenizer = hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.config = config
        self.max_samples = max_samples
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "response")
        self.prompt_dict_keys = config.get("prompt_dict_keys", None) or []
        self.response_dict_keys = config.get("response_dict_keys", None) or []
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.use_shm = config.get("use_shm", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]
        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files(self):
        dataframes = [pd.read_parquet(path) for path in self.parquet_files]
        dataframe = pd.concat(dataframes, ignore_index=True)
        total = len(dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.config.get("shuffle", False):
                seed = self.config.get("seed", None)
                rng = np.random.default_rng(seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            dataframe = dataframe.iloc[indices.tolist()]
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item].to_dict()
        prompt = _extract_nested(row, self.prompt_key, self.prompt_dict_keys)
        response = _extract_nested(row, self.response_key, self.response_dict_keys)
        prompt_chat_str = self.tokenizer.apply_chat_template(
            _ensure_messages(prompt),
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        response_chat_str = str(response) + self.tokenizer.eos_token
        prompt_output = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        response_output = self.tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_output["input_ids"][0]
        prompt_attention_mask = prompt_output["attention_mask"][0]
        response_ids = response_output["input_ids"][0]
        response_attention_mask = response_output["attention_mask"][0]
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            input_ids = torch.cat(
                (input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype))
            )
            attention_mask = torch.cat((attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise RuntimeError(f"sequence_length={sequence_length} is larger than max_length={self.max_length}")
        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        end_idx = min(prompt_length + response_length, loss_mask.size(0)) - 1
        if end_idx >= 0:
            loss_mask[end_idx] = 0
        row.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "dataframe_index": item,
            }
        )
        return row


class SFTRolloutEvalDataset(Dataset):
    def __init__(self, data_files, tokenizer, config: DictConfig, processor=None, max_samples: int = -1):
        self.data_files = normalize_data_paths(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.prompt_key = config.get("eval_prompt_key", config.get("prompt_key", "prompt"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.use_shm = config.get("use_shm", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files(self):
        frames = []
        for parquet_file in self.data_files:
            frame = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            frames.append(frame)
        rows = []
        for frame in frames:
            rows.extend(frame)
        if self.filter_overlong_prompts:
            rows = [row for row in rows if len(self._raw_prompt_ids(row)) <= self.max_prompt_length]
        total = len(rows)
        if self.max_samples > 0 and self.max_samples < total:
            if self.config.get("shuffle", False):
                seed = self.config.get("seed", None)
                rng = np.random.default_rng(seed)
                indices = rng.choice(total, size=self.max_samples, replace=False).tolist()
            else:
                indices = list(range(self.max_samples))
            rows = [rows[i] for i in indices]
        self.rows = rows

    def _messages(self, row):
        return row[self.prompt_key]

    def _raw_prompt(self, row):
        return self.tokenizer.apply_chat_template(
            self._messages(row),
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

    def _raw_prompt_ids(self, row):
        return self.tokenizer.encode(self._raw_prompt(row), add_special_tokens=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = dict(self.rows[item])
        raw_prompt = self._raw_prompt(row)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            else:
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}.")
        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]
        row["raw_prompt_ids"] = raw_prompt_ids
        row["index"] = row.get("extra_info", {}).get("index", item)
        row["tools_kwargs"] = row.get("extra_info", {}).get("tools_kwargs", {})
        row["interaction_kwargs"] = row.get("extra_info", {}).get("interaction_kwargs", {})
        row["field"] = default_get_field_for_source(row.get("data_source", "unknown"))
        row["dataframe_index"] = item
        if self.return_raw_chat:
            row["raw_prompt"] = self._messages(row)
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt
        return row


def create_sft_dataset(data_paths, data_config, tokenizer, processor=None, is_train=True, max_samples=-1):
    if is_train and data_config.custom_cls.get("path", None):
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        return dataset_cls(data_paths, tokenizer=tokenizer, config=data_config)
    if is_train:
        return SFTTrainDataset(data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return SFTRolloutEvalDataset(
        data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def create_sft_sampler(data_config, dataset):
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        sampler_cls = load_extern_type(data_config.sampler.class_path, data_config.sampler.class_name)
        sampler = sampler_cls(dataset, data_config=data_config)
        assert isinstance(sampler, torch.utils.data.Sampler)
        assert data_config.get("dataloader_num_workers", 8) == 0
        return sampler
    if data_config.get("shuffle", False):
        generator = torch.Generator()
        generator.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=generator)
    return SequentialSampler(data_source=dataset)
```

- [ ] **Step 4: Run dataset tests**

Run:

```powershell
python -m pytest tests/recipe/sft/test_rl_dataset_on_cpu.py -q
```

Expected: all tests pass. If `pyarrow` or `fastparquet` is missing, rerun the test after installing the missing parquet engine in the project environment.

- [ ] **Step 5: Manual checkpoint**

Record these files for the user's manual commit:

```text
verl-0.6.0/recipe/sft/rl_dataset.py
verl-0.6.0/tests/recipe/sft/test_rl_dataset_on_cpu.py
```

---

### Task 3: Ray SFT Config and Launch Script

**Files:**
- Create: `verl-0.6.0/recipe/sft/config/sft_trainer.yaml`
- Create: `verl-0.6.0/recipe/sft/scripts/sft_peft_2.sh`
- Read: `verl_-cl-master/sft_peft_2.sh`
- Read: `verl-0.6.0/recipe/async_replay/config/one_step_off_ppo_trainer.yaml`

- [ ] **Step 1: Create config directory and SFT config**

Create `recipe/sft/config/sft_trainer.yaml`:

```yaml
hydra:
  searchpath:
    - file://${oc.env:HOME}/works/verl-0.6.0/verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

actor_rollout_ref:
  is_ppo: false
  hybrid_engine: true
  model:
    trust_remote_code: true
    enable_gradient_checkpointing: true
    use_remove_padding: true
    lora_rank: 32
    lora_alpha: 16
    target_modules: all-linear
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2
    optim:
      lr: 1.0e-5
  rollout:
    name: vllm
    mode: sync
    n: 1
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.6
    max_num_batched_tokens: 20000
    log_prob_micro_batch_size_per_gpu: 2
    val_kwargs:
      n: 1
      do_sample: false
  ref:
    log_prob_micro_batch_size_per_gpu: 2

data:
  train_files: null
  val_files: null
  train_batch_size: 64
  val_batch_size: 128
  gen_batch_size: null
  train_max_samples: -1
  val_max_samples: -1
  prompt_key: extra_info
  response_key: extra_info
  prompt_dict_keys: ["question"]
  response_dict_keys: ["answer"]
  eval_prompt_key: prompt
  max_length: 10002
  max_prompt_length: 1024
  max_response_length: 16384
  filter_overlong_prompts: true
  filter_overlong_prompts_workers: 8
  dataloader_num_workers: 0
  shuffle: true
  validation_shuffle: true
  reward_fn_key: data_source
  trust_remote_code: true
  sampler: null
  custom_cls:
    path: null
    name: null

custom_reward_function:
  path: recipe/sft/reward_score/__init__.py
  name: compute_score

reward_model:
  enable: false
  strategy: rule
  custom_reward_function:
    path: recipe/sft/reward_score/__init__.py
    name: compute_score
  reward_kwargs: {}

trainer:
  project_name: ray-sft
  experiment_name: default
  logger: ["console", "tensorboard"]
  nnodes: 1
  n_gpus_per_node: 4
  total_epochs: 2
  total_training_steps: null
  val_before_train: true
  val_only: false
  save_freq: 10
  test_freq: 5
  resume_mode: disable
  resume_from_path: null
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  default_hdfs_dir: null
  validation_data_dir: null
  log_val_generations: 0
  max_actor_ckpt_to_keep: 5
  max_ckpt_to_keep: 5
  remove_previous_ckpt_in_save: false
  save_hf_model: true
  save_merged_lora_model: true
  merged_lora_model_dirname: merged_model
  save_best_immediately: true
  device: cuda
  critic_warmup: 0
  profile_steps: null
  profile_continuous_steps: false
  ray_wait_register_center_timeout: 300
  del_local_ckpt_after_load: false

ray_kwargs:
  ray_init:
    address: local
    num_cpus: null
    include_dashboard: false
    runtime_env: {}
  timeline_json_file: null
```

- [ ] **Step 2: Create launch script**

Create `recipe/sft/scripts/sft_peft_2.sh`:

```bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME=${DATASET_NAME:-mof}
BASE_PATH=${BASE_PATH:-"${HOME}/works/verl-0.6.0"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${BASE_PATH}/data/dataset/${DATASET_NAME}/train/"}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-"${BASE_PATH}/data/dataset/${DATASET_NAME}/online_eval_2/"}
MODEL_PATH=${MODEL_PATH:-"${BASE_PATH}/models/Qwen3-4B-Thinking"}

TAG=${TAG:-sft4math_eval2}
MODEL_NAME=${MODEL_NAME:-Qwen3-4B-Thinking}
PROJECT_NAME=${PROJECT_NAME:-sft_${MODEL_NAME}_${TAG}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATASET_NAME}}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${BASE_PATH}/sft/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
BASE_DIR_PATH=${BASE_DIR_PATH:-"${BASE_PATH}/my_output/${PROJECT_NAME}_${EXPERIMENT_NAME}"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${BASE_DIR_PATH}/validation_data"}

N_GPUS=${N_GPUS:-4}
N_CPUS=${N_CPUS:-16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_LENGTH=${MAX_LENGTH:-10002}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
LR=${LR:-1e-5}
SEED=${SEED:-3}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
SAVE_MERGED_LORA_MODEL=${SAVE_MERGED_LORA_MODEL:-true}

python -m recipe.sft.main_sft \
    +base_dir="${BASE_DIR_PATH}" \
    +describe="'Ray SFT recipe, ${PROJECT_NAME}/${EXPERIMENT_NAME}'" \
    +seed="${SEED}" \
    +actor_rollout_ref.rollout.seed="${SEED}" \
    data.train_files="${TRAIN_DATASET_PATH}" \
    data.val_files="${VAL_DATASET_PATH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.eval_prompt_key=prompt \
    data.max_length="${MAX_LENGTH}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=true \
    custom_reward_function.path="${RECIPE_DIR}/reward_score/__init__.py" \
    custom_reward_function.name=compute_score \
    reward_model.enable=false \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    ray_kwargs.ray_init.num_cpus="${N_CPUS}" \
    trainer.resume_mode=disable \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_merged_lora_model="${SAVE_MERGED_LORA_MODEL}" \
    "$@" 2>&1 | tee "output_${TAG}_${MODEL_NAME}.log"
```

- [ ] **Step 3: Static config and script checks**

Run:

```powershell
rg -n "verifier|recipe/sft/reward_score|python -m recipe.sft.main_sft|save_merged_lora_model" recipe/sft/config/sft_trainer.yaml recipe/sft/scripts/sft_peft_2.sh
```

Expected: reward path and launch module are present; `verifier` only appears in the explanatory config field if retained, and no script path points to old `verl_-cl-master`.

- [ ] **Step 4: Shell syntax check if bash is available**

Run:

```powershell
bash -n recipe/sft/scripts/sft_peft_2.sh
```

Expected: exits 0. If Windows reports no WSL distribution, record that shell syntax was not executable in this workspace and continue with text review.

- [ ] **Step 5: Manual checkpoint**

Record these files for the user's manual commit:

```text
verl-0.6.0/recipe/sft/config/sft_trainer.yaml
verl-0.6.0/recipe/sft/scripts/sft_peft_2.sh
```

---

### Task 4: Main Ray Entrypoint

**Files:**
- Create: `verl-0.6.0/recipe/sft/main_sft.py`
- Read: `verl_-cl-master/verl/trainer/main_sft.py`
- Read: `verl-0.6.0/recipe/async_replay/main_ppo.py`

- [ ] **Step 1: Create `main_sft.py` by adapting existing entrypoints**

Create `recipe/sft/main_sft.py` with these exact structural elements:

```python
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.fs import copy_to_local

from qiuputils.ai_utils import normalize_dataset_paths
from qiuputils.logger import get_logger
from qiuputils.misc import make_dir, set_qiuputils_env_var

from .rl_dataset import create_sft_dataset, create_sft_sampler
from .ray_trainer import RaySFTTrainer, ResourcePoolManager, Role
from .utils import log_accelerator_infos, set_deterministic


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    base_dir = config.get("base_dir", f"./{config.trainer.project_name}/{config.trainer.experiment_name}")
    make_dir(base_dir, mode="m")
    set_qiuputils_env_var(
        QIUPUTILS_BASE_DIR=base_dir,
        TENSORBOARD_DIR=os.path.join(base_dir, "tensorboard"),
    )
    set_deterministic(config.get("seed", config.trainer.get("seed", 1)))
    logger = get_logger()
    logger.log_start()
    logger.info(f"base_dir: {base_dir}")
    logger.info(f"describe: {config.get('describe', 'No description provided')}")
    log_accelerator_infos(logger)
    run_sft(config)


def run_sft(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}")

        role_worker_mapping = {Role.ActorRollout: ray.remote(actor_rollout_cls)}
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id}

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        train_dataset = create_sft_dataset(
            normalize_dataset_paths(config.data.train_files),
            config.data,
            tokenizer,
            processor=processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_sft_dataset(
            normalize_dataset_paths(config.data.val_files),
            config.data,
            tokenizer,
            processor=processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_sft_sampler(config.data, train_dataset)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RaySFTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile entrypoint to catch import syntax errors**

Run:

```powershell
python -m py_compile recipe/sft/main_sft.py
```

Expected: fails until `recipe/sft/ray_trainer.py` exists, or passes if Task 5 has already run.

- [ ] **Step 3: Manual checkpoint**

Record this file for the user's manual commit:

```text
verl-0.6.0/recipe/sft/main_sft.py
```

---

### Task 5: Ray SFT Trainer Core

**Files:**
- Create: `verl-0.6.0/recipe/sft/ray_trainer.py`
- Read: `verl_-cl-master/verl/trainer/sft/ray_trainer.py`
- Read: `verl-0.6.0/verl/trainer/ppo/ray_trainer.py`
- Read: `verl-0.6.0/recipe/async_replay/ray_trainer_wip.py`

- [ ] **Step 1: Copy old SFT trainer as the starting point**

Run:

```powershell
Copy-Item -LiteralPath ..\verl_-cl-master\verl\trainer\sft\ray_trainer.py -Destination recipe\sft\ray_trainer.py
```

Expected: `recipe/sft/ray_trainer.py` exists.

- [ ] **Step 2: Replace imports and remove old project utilities**

In `recipe/sft/ray_trainer.py`, replace old imports:

```python
from verl.utils.dataset import SOURCE_TO_FIELD
from verl.utils.my_utils import get_logger, make_dir
```

with:

```python
from recipe.async_replay.rl_dataset import default_get_field_for_source
from recipe.sft.utils import select_best_metric_candidates
from qiuputils.ai_utils import CheckpointManager
from qiuputils.logger import get_logger
```

Also remove unused imports created by old best-checkpoint folder deletion if the implementation no longer creates `best_*` folders.

- [ ] **Step 3: Replace dataloader construction imports**

In `_create_dataloader`, replace:

```python
from verl.trainer.main_sft import create_sft_dataset, create_sft_sampler
```

with:

```python
from recipe.sft.rl_dataset import create_sft_dataset, create_sft_sampler
```

When creating datasets internally, call:

```python
train_dataset = create_sft_dataset(
    self.config.data.train_files,
    self.config.data,
    self.tokenizer,
    processor=self.processor,
    is_train=True,
    max_samples=self.config.data.get("train_max_samples", -1),
)
val_dataset = create_sft_dataset(
    self.config.data.val_files,
    self.config.data,
    self.tokenizer,
    processor=self.processor,
    is_train=False,
    max_samples=self.config.data.get("val_max_samples", -1),
)
```

- [ ] **Step 4: Initialize qiuputils checkpoint manager**

In `RaySFTTrainer.__init__`, after `_create_dataloader`, initialize:

```python
self.qiup_logger = get_logger()
self.ckpt_manager = CheckpointManager(
    base_dir=self.config.trainer.default_local_dir,
    logger=self.qiup_logger,
    indentation_num=2,
)
```

Remove `my_log_path`, file-local `metrics.json` writing, and old `MetricTracker` unless still needed as a fallback. Use `select_best_metric_candidates` only when `CheckpointManager.get_improved_mean_metrics` cannot be called.

- [ ] **Step 5: Keep single ActorRollout worker group**

In `init_workers`, keep only role setup for `Role.ActorRollout` and optional async rollout manager. Do not create verifier worker. The worker group class should be created with:

```python
actor_rollout_cls = RayClassWithInitArgs(
    cls=self.role_worker_mapping[Role.ActorRollout],
    config=self.config.actor_rollout_ref,
    role="actor_rollout",
)
self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
```

If the old code passes `profile_option=self.config.trainer.npu_profile.options`, remove that argument because this CUDA recipe config does not define `trainer.npu_profile`.

- [ ] **Step 6: Align validation metrics collection**

Update `_validate` to keep the old rollout generation flow but ensure the result collection matches the spec:

```python
reward_extra_infos_dict["reward"].extend(scores)
if "reward_extra_info" in result:
    for key, values in result["reward_extra_info"].items():
        reward_extra_infos_dict[key].extend(values)

for key_info, values in reward_extra_infos_dict.items():
    assert len(values) == 0 or len(values) == len(sample_scores), (
        f"{key_info}: len(values)={len(values)}, len(sample_scores)={len(sample_scores)}"
    )
```

Use:

```python
data_src2var2metric2val = process_validation_metrics(
    data_sources,
    sample_inputs,
    reward_extra_infos_dict,
)
```

Then aggregate fields with:

```python
source_to_field = {
    data_source: default_get_field_for_source(data_source)
    for data_source in set(data_sources.tolist())
}
field2var2metric2val = aggregate_mean1_by_field(data_src2var2metric2val, source_to_field)
```

Metric prefixes:

```python
metric_sec = "val-core" if var_name == core_var and metric_name == "mean@1" else "val-aux"
pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"

metric_sec = "val_aggregate-core" if var_name == core_var and metric_name == "mean@1" else "val_aggregate-aux"
pfx = f"{metric_sec}/{field}/{var_name}/{metric_name}"
```

- [ ] **Step 7: Align validation dump**

Ensure `_dump_generations` writes JSONL where each row contains:

```python
item = {
    "input": inputs[i],
    "output": outputs[i],
    "datasource": datasources[i],
    "ground_truth": gts[i],
    "score": scores[i],
    "step": self.global_steps,
    "metrics": metric_dict,
}
for key, values in reward_extra_infos_dict.items():
    if len(values) == len(inputs):
        item[key] = values[i]
```

Write to:

```python
filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
```

- [ ] **Step 8: Align checkpoint save**

Update `_save_checkpoint` signature to:

```python
def _save_checkpoint(self, val_metrics=None):
```

Inside it, compute:

```python
max_actor_ckpt_to_keep = self.config.trainer.get(
    "max_actor_ckpt_to_keep",
    self.config.trainer.get("max_ckpt_to_keep", None),
)
if self.config.trainer.get("remove_previous_ckpt_in_save", False):
    max_actor_ckpt_to_keep = 1
```

Call the 0.6.0 worker checkpoint interface without the old SFT `is_save_hf` argument:

```python
self.actor_rollout_wg.save_checkpoint(
    actor_local_path,
    actor_remote_path,
    self.global_steps,
    max_ckpt_to_keep=max_actor_ckpt_to_keep,
)
```

After worker checkpoint and dataloader state save:

```python
self.ckpt_manager.update_after_save(
    step=self.global_steps,
    ckpt_path=local_global_step_folder,
    val_metrics=val_metrics,
)
self.ckpt_manager.cleanup_old_checkpoints(max_ckpt_to_keep=max_actor_ckpt_to_keep)
```

- [ ] **Step 9: Save merged LoRA model when configured**

Add `_maybe_save_merged_lora_model(self, actor_local_path)` that:

```python
if self.config.actor_rollout_ref.model.get("lora_rank", 0) <= 0:
    return
if not self.config.trainer.get("save_merged_lora_model", True):
    return
lora_adapter_path = os.path.join(actor_local_path, "lora_adapter")
if not os.path.exists(lora_adapter_path):
    self.qiup_logger.info(f"LoRA adapter not found for merge: {lora_adapter_path}", indentation_num=2)
    return
```

Then call a small helper that loads base model and adapter on rank/controller process:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model_path = self.config.actor_rollout_ref.model.path
merged_dirname = self.config.trainer.get("merged_lora_model_dirname", "merged_model")
merged_path = os.path.join(actor_local_path, merged_dirname)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=self.config.actor_rollout_ref.model.get("trust_remote_code", False),
)
model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload()
model.save_pretrained(merged_path, safe_serialization=True)
self.tokenizer.save_pretrained(merged_path)
```

Wrap this in `try/except Exception as exc` and log the error without failing checkpoint save, because worker checkpoint is the primary artifact.

- [ ] **Step 10: Align fit loop best checkpoint logic**

In `fit`, after validation:

```python
if save_best_immediately:
    try:
        improved_metrics = self.ckpt_manager.get_improved_mean_metrics(val_metrics)
    except Exception:
        improved_metrics = select_best_metric_candidates(val_metrics)
    if improved_metrics:
        self.qiup_logger.info(f"best metric improved: {improved_metrics}", indentation_num=2)
        self._save_checkpoint(val_metrics=val_metrics)
        checkpoint_saved_this_step = True
```

Avoid writing a separate `metrics.json`; all metrics should go through `Tracking.log` and qiuputils logger.

- [ ] **Step 11: Compile trainer**

Run:

```powershell
python -m py_compile recipe/sft/ray_trainer.py recipe/sft/main_sft.py
```

Expected: exits 0.

- [ ] **Step 12: Manual checkpoint**

Record this file for the user's manual commit:

```text
verl-0.6.0/recipe/sft/ray_trainer.py
```

---

### Task 6: Import, Config, and Dataset Smoke Verification

**Files:**
- Verify: `verl-0.6.0/recipe/sft/*.py`
- Verify: `verl-0.6.0/recipe/sft/config/sft_trainer.yaml`
- Verify: `verl-0.6.0/tests/recipe/sft/*.py`

- [ ] **Step 1: Run unit tests**

Run:

```powershell
python -m pytest tests/recipe/sft/test_rl_dataset_on_cpu.py tests/recipe/sft/test_validation_metrics_on_cpu.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile all new Python modules**

Run:

```powershell
python -m py_compile recipe/sft/__init__.py recipe/sft/utils.py recipe/sft/rl_dataset.py recipe/sft/ray_trainer.py recipe/sft/main_sft.py
```

Expected: exits 0.

- [ ] **Step 3: Verify no verifier implementation entered first version**

Run:

```powershell
rg -n "verifier|RewardModelWorker|LLM-as-judge" recipe/sft --glob "!reward_score/**"
```

Expected: no matches, or only comments/config text saying verifier is not implemented in first version.

- [ ] **Step 4: Verify reward function paths are recipe-local**

Run:

```powershell
rg -n "reward_score/__init__.py|verl_-cl-master" recipe/sft/config/sft_trainer.yaml recipe/sft/scripts/sft_peft_2.sh
```

Expected: `recipe/sft/reward_score/__init__.py` appears; `verl_-cl-master` does not appear.

- [ ] **Step 5: Try Hydra config import without starting training**

Run:

```powershell
python -m recipe.sft.main_sft --cfg job trainer.val_only=true data.train_files=dummy data.val_files=dummy actor_rollout_ref.model.path=dummy
```

Expected: Hydra prints the composed config and exits 0. If it attempts to initialize Ray, stop and instead run:

```powershell
python -c "from hydra import initialize, compose; import os; os.chdir('D:/from_git/works/verl-0.6.0'); initialize(config_path='recipe/sft/config', version_base=None); cfg=compose(config_name='sft_trainer'); print(cfg.trainer.project_name)"
```

Expected: prints `ray-sft`.

- [ ] **Step 6: Manual checkpoint**

Record verification results in the implementation notes for the user's manual commit.

---

### Task 7: Final Review

**Files:**
- Review: `verl-0.6.0/recipe/sft`
- Review: `verl-0.6.0/tests/recipe/sft`
- Review: `verl-0.6.0/docs/superpowers/specs/2026-06-21-ray-sft-recipe-design.md`

- [ ] **Step 1: Check worktree scope**

Run:

```powershell
git status --short -- verl-0.6.0/recipe/sft verl-0.6.0/tests/recipe/sft verl-0.6.0/docs/superpowers
```

Expected: changes are limited to the SFT recipe, tests, and docs. `recipe/sft/reward_score` remains user-provided and should not be modified by implementation unless an import fix was explicitly required.

- [ ] **Step 2: Search for forbidden first-version verifier code**

Run:

```powershell
rg -n "vllm.LLM|SamplingParams|Final Decision|recipe.sft.verifier|reward_model.strategy=verifier" recipe/sft --glob "!reward_score/**"
```

Expected: no matches.

- [ ] **Step 3: Search for unresolved implementation markers**

Run:

```powershell
$pattern = ("TB" + "D") + "|" + ("FIX" + "ME") + "|" + ("not imple" + "mented") + "|pass$"
rg -n $pattern recipe/sft tests/recipe/sft docs/superpowers/plans/2026-06-21-ray-sft-recipe.md
```

Expected: no matches that indicate unfinished implementation.

- [ ] **Step 4: Run final static checks**

Run:

```powershell
python -m pytest tests/recipe/sft -q
python -m py_compile recipe/sft/__init__.py recipe/sft/utils.py recipe/sft/rl_dataset.py recipe/sft/ray_trainer.py recipe/sft/main_sft.py
git diff --check -- verl-0.6.0/recipe/sft verl-0.6.0/tests/recipe/sft verl-0.6.0/docs/superpowers
```

Expected: tests pass, compilation passes, diff check exits 0. Git may warn about CRLF on Windows; that warning is acceptable if no whitespace errors are reported.

- [ ] **Step 5: Prepare user-facing implementation summary**

Summarize:

```text
Implemented Ray SFT recipe under recipe/sft.
Rule-based reward only; verifier is not implemented in first version.
Eval/test result collection follows 0.6.0 validation metrics and JSONL dump style.
Checkpoint/logging uses qiuputils CheckpointManager and get_logger.
LoRA can be disabled with lora_rank=0 and merged-model saving is controlled by trainer.save_merged_lora_model.
```

Because the user will commit manually, do not run `git add` or `git commit`.

---

## Self-Review Notes

- Spec coverage: tasks cover Ray entrypoint, trainer, datasets, config, script, rule-based reward path, eval/test metrics collection, qiuputils checkpoint/logging, LoRA config and optional merge, and static verification.
- Marker scan: plan avoids unresolved markers and names concrete files, functions, and commands.
- Type consistency: dataset constructors, `create_sft_dataset`, `create_sft_sampler`, `RaySFTTrainer`, `ResourcePoolManager`, and `Role.ActorRollout` names are consistent across tasks.
