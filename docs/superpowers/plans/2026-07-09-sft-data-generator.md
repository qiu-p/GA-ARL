# SFT Data Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `recipe/sft_data_generator`, a vLLM-backed offline generator that turns guided correct model completions into a new SFT-style parquet dataset.

**Architecture:** Keep GPU/model work behind a small vLLM adapter and keep dataset transformation, prompt building, reward filtering, and info JSON generation as testable pure Python modules. The CLI reads parquet, builds guided prompts using `extra_info.answer`, generates `n` completions per source row, filters with bundled `reward_score.compute_score`, and writes output parquet plus stats.

**Tech Stack:** Python, pandas, pyarrow, transformers tokenizer chat templates, vLLM, pytest.

---

## File Structure

- Create `recipe/sft_data_generator/__init__.py`: package marker and public exports.
- Use existing `recipe/sft_data_generator/reward_score/`: bundled reward functions copied by the user.
- Create `recipe/sft_data_generator/data.py`: input validation, row normalization, output item construction, parquet writing, and info JSON construction.
- Create `recipe/sft_data_generator/prompting.py`: guided chat message construction and optional tokenizer chat-template rendering.
- Create `recipe/sft_data_generator/reward.py`: wrapper around bundled `compute_score`, numeric conversion, thresholding, and error accounting.
- Create `recipe/sft_data_generator/vllm_backend.py`: lazy vLLM imports and batched generation adapter.
- Create `recipe/sft_data_generator/generate.py`: argument parsing and orchestration.
- Create `tests/recipe/sft_data_generator/test_data_on_cpu.py`: pure data transformation tests.
- Create `tests/recipe/sft_data_generator/test_prompting_on_cpu.py`: prompt construction tests.
- Create `tests/recipe/sft_data_generator/test_reward_on_cpu.py`: reward wrapper tests with fake scorer.
- Create `tests/recipe/sft_data_generator/test_generate_on_cpu.py`: CLI orchestration tests with fake tokenizer, generator, and scorer.

---

### Task 1: Data Model And Output Row Construction

**Files:**
- Create: `recipe/sft_data_generator/__init__.py`
- Create: `recipe/sft_data_generator/data.py`
- Test: `tests/recipe/sft_data_generator/test_data_on_cpu.py`

- [ ] **Step 1: Write failing data tests**

Create `tests/recipe/sft_data_generator/test_data_on_cpu.py`:

```python
import pandas as pd
import pyarrow as pa

from recipe.sft_data_generator.data import (
    GenerationRecord,
    GenerationStats,
    build_info_json,
    build_output_rows,
    dataframe_from_rows,
    table_from_dataframe,
    validate_input_dataframe,
)


def _source_row():
    return {
        "data_source": "math__aime2025",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": 70},
        "extra_info": {
            "answer": "original answer",
            "index": 42,
            "question": "question",
            "split": "train",
            "is_correct": True,
        },
    }


def test_validate_input_dataframe_accepts_required_columns():
    dataframe = pd.DataFrame([_source_row()])
    validate_input_dataframe(dataframe)


def test_validate_input_dataframe_rejects_missing_answer():
    row = _source_row()
    row["extra_info"] = {"index": 42}
    dataframe = pd.DataFrame([row])

    try:
        validate_input_dataframe(dataframe)
    except ValueError as exc:
        assert "extra_info.answer" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_input_dataframe_allows_missing_ground_truth_for_later_skip():
    row = _source_row()
    row["reward_model"] = {"style": "rule"}
    dataframe = pd.DataFrame([row])

    validate_input_dataframe(dataframe)


def test_build_output_rows_splits_multiple_correct_generations():
    row = _source_row()
    records = [
        GenerationRecord(source_position=0, source_index=42, generation_index=0, text="first", score=1.0),
        GenerationRecord(source_position=0, source_index=42, generation_index=2, text="third", score=1.0),
    ]

    output_rows = build_output_rows([row], records)

    assert len(output_rows) == 2
    assert output_rows[0]["prompt"] == row["prompt"]
    assert output_rows[0]["extra_info"]["answer"] == "first"
    assert output_rows[0]["extra_info"]["origin_answer"] == "original answer"
    assert output_rows[0]["extra_info"]["source_index"] == 42
    assert output_rows[0]["extra_info"]["index"] == 0
    assert output_rows[0]["extra_info"]["generation_index"] == 0
    assert output_rows[0]["extra_info"]["score"] == 1.0
    assert output_rows[1]["extra_info"]["answer"] == "third"
    assert output_rows[1]["extra_info"]["source_index"] == 42
    assert output_rows[1]["extra_info"]["index"] == 1
    assert output_rows[1]["extra_info"]["generation_index"] == 2


def test_build_info_json_includes_generation_stats():
    output_rows = [build_output_rows([_source_row()], [
        GenerationRecord(source_position=0, source_index=42, generation_index=0, text="ok", score=1.0)
    ])[0]]
    dataframe = pd.DataFrame(output_rows)
    stats = GenerationStats(
        input_rows=2,
        generations_per_problem=3,
        total_generations=6,
        correct_generations=1,
        kept_rows=1,
        problems_with_correct_generation=1,
        reward_errors=0,
        generation_errors=0,
    )

    info = build_info_json(dataframe, stats, examples=1)

    assert info["Rows"] == 1
    assert info["Columns"] == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    assert info["GenerationStats"]["generation_accuracy"] == 1 / 6
    assert info["GenerationStats"]["problem_pass_rate"] == 1 / 2
    assert info["Examples"][0]["extra_info"]["answer"] == "ok"


def test_empty_output_table_keeps_nested_schema_from_source():
    source_dataframe = pd.DataFrame([_source_row()])
    empty_output = dataframe_from_rows([])

    table = table_from_dataframe(empty_output, source_dataframe=source_dataframe)

    assert table.num_rows == 0
    assert table.schema.names == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    assert pa.types.is_list(table.schema.field("prompt").type)
    assert pa.types.is_struct(table.schema.field("reward_model").type)
    assert pa.types.is_struct(table.schema.field("extra_info").type)
    extra_fields = {field.name for field in table.schema.field("extra_info").type}
    assert {"answer", "origin_answer", "source_index", "generation_index", "score"}.issubset(extra_fields)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/recipe/sft_data_generator/test_data_on_cpu.py -q`

Expected: FAIL with import errors for `recipe.sft_data_generator.data`.

- [ ] **Step 3: Implement data module**

Create `recipe/sft_data_generator/__init__.py`:

```python
"""Utilities for guided SFT data generation."""
```

Create `recipe/sft_data_generator/data.py`:

```python
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


@dataclass(frozen=True)
class GenerationRecord:
    source_position: int
    source_index: Any
    generation_index: int
    text: str
    score: float


@dataclass(frozen=True)
class GenerationStats:
    input_rows: int
    generations_per_problem: int
    total_generations: int
    correct_generations: int
    kept_rows: int
    problems_with_correct_generation: int
    reward_errors: int
    generation_errors: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation_accuracy"] = (
            self.correct_generations / self.total_generations if self.total_generations else 0.0
        )
        data["problem_pass_rate"] = (
            self.problems_with_correct_generation / self.input_rows if self.input_rows else 0.0
        )
        return data


def _as_dict(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{field_name} must be a dict, got {type(value).__name__}")


def validate_input_dataframe(dataframe: pd.DataFrame) -> None:
    missing = [column for column in OUTPUT_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"input parquet missing required columns: {missing}")

    for position, row in enumerate(dataframe.to_dict("records")):
        extra_info = _as_dict(row.get("extra_info"), f"row {position} extra_info")
        _as_dict(row.get("reward_model"), f"row {position} reward_model")
        if not extra_info.get("answer"):
            raise ValueError(f"row {position} missing required extra_info.answer")


def source_index_for_row(row: dict[str, Any], source_position: int) -> Any:
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return extra_info.get("index", source_position)


def build_output_rows(source_rows: list[dict[str, Any]], records: list[GenerationRecord]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for output_index, record in enumerate(records):
        source = source_rows[record.source_position]
        source_extra = copy.deepcopy(source.get("extra_info") or {})
        origin_answer = source_extra.get("answer", "")
        source_extra.update(
            {
                "answer": record.text,
                "origin_answer": origin_answer,
                "source_index": record.source_index,
                "index": output_index,
                "is_correct": True,
                "generation_index": record.generation_index,
                "score": float(record.score),
            }
        )
        output_rows.append(
            {
                "data_source": source.get("data_source"),
                "prompt": copy.deepcopy(source.get("prompt")),
                "ability": source.get("ability"),
                "reward_model": copy.deepcopy(source.get("reward_model")),
                "extra_info": source_extra,
            }
        )
    return output_rows


def dataframe_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _replace_struct_field(fields: list[pa.Field], field: pa.Field) -> list[pa.Field]:
    remaining = [existing for existing in fields if existing.name != field.name]
    remaining.append(field)
    return remaining


def output_schema_from_source(source_dataframe: pd.DataFrame) -> pa.Schema:
    source_table = pa.Table.from_pandas(source_dataframe[OUTPUT_COLUMNS].head(1), preserve_index=False)
    fields = {field.name: field for field in source_table.schema}
    extra_type = fields["extra_info"].type
    extra_fields = list(extra_type) if pa.types.is_struct(extra_type) else []
    extra_fields = _replace_struct_field(extra_fields, pa.field("answer", pa.string()))
    extra_fields = _replace_struct_field(extra_fields, pa.field("origin_answer", pa.string()))
    extra_fields = _replace_struct_field(extra_fields, pa.field("source_index", pa.int64()))
    extra_fields = _replace_struct_field(extra_fields, pa.field("index", pa.int64()))
    extra_fields = _replace_struct_field(extra_fields, pa.field("is_correct", pa.bool_()))
    extra_fields = _replace_struct_field(extra_fields, pa.field("generation_index", pa.int64()))
    extra_fields = _replace_struct_field(extra_fields, pa.field("score", pa.float64()))
    return pa.schema(
        [
            fields["data_source"],
            fields["prompt"],
            fields["ability"],
            fields["reward_model"],
            pa.field("extra_info", pa.struct(extra_fields)),
        ]
    )


def table_from_dataframe(dataframe: pd.DataFrame, source_dataframe: pd.DataFrame | None = None) -> pa.Table:
    if len(dataframe) == 0 and source_dataframe is not None and len(source_dataframe) > 0:
        schema = output_schema_from_source(source_dataframe)
        arrays = [pa.array([], type=schema.field(column).type) for column in OUTPUT_COLUMNS]
        return pa.Table.from_arrays(arrays, schema=schema)
    return pa.Table.from_pandas(dataframe, preserve_index=False)


def _schema_to_struct(schema: pa.Schema) -> dict[str, Any]:
    def field_to_shape(field_type: pa.DataType) -> Any:
        if pa.types.is_struct(field_type):
            return {field.name: field_to_shape(field.type) for field in field_type}
        if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
            return [field_to_shape(field_type.value_type)]
        return str(field_type)

    return {field.name: field_to_shape(field.type) for field in schema}


def build_info_json(
    dataframe: pd.DataFrame,
    stats: GenerationStats,
    examples: int = 3,
    source_dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    table = table_from_dataframe(dataframe, source_dataframe=source_dataframe)
    schema = table.schema
    return {
        "Rows": len(dataframe),
        "Columns": list(dataframe.columns),
        "Schema": {field.name: str(field.type) for field in schema},
        "Struct": _schema_to_struct(schema),
        "Examples": dataframe.head(examples).to_dict("records"),
        "GenerationStats": stats.to_dict(),
    }


def write_outputs(
    dataframe: pd.DataFrame,
    output_parquet: str | Path,
    info: dict[str, Any],
    info_json: str | Path,
    source_dataframe: pd.DataFrame | None = None,
) -> None:
    output_path = Path(output_parquet)
    info_path = Path(info_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table_from_dataframe(dataframe, source_dataframe=source_dataframe), output_path)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/recipe/sft_data_generator/test_data_on_cpu.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add recipe/sft_data_generator/__init__.py recipe/sft_data_generator/data.py tests/recipe/sft_data_generator/test_data_on_cpu.py
git commit -m "feat: add sft generator data utilities"
```

Expected: commit succeeds.

---

### Task 2: Guided Prompt Construction

**Files:**
- Create: `recipe/sft_data_generator/prompting.py`
- Test: `tests/recipe/sft_data_generator/test_prompting_on_cpu.py`

- [ ] **Step 1: Write failing prompt tests**

Create `tests/recipe/sft_data_generator/test_prompting_on_cpu.py`:

```python
from recipe.sft_data_generator.prompting import build_guided_messages


def test_build_guided_messages_appends_reference_to_last_user_without_mutating_source():
    source_prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]

    guided = build_guided_messages(source_prompt, "original answer")

    assert source_prompt[1]["content"] == "question"
    assert guided[0] == {"role": "system", "content": "system"}
    assert guided[1]["role"] == "user"
    assert guided[1]["content"].startswith("question")
    assert "Reference solution for guidance:" in guided[1]["content"]
    assert "original answer" in guided[1]["content"]
    assert "put the final answer in \\\\boxed{}" in guided[1]["content"]


def test_build_guided_messages_normalizes_string_prompt():
    guided = build_guided_messages("question", "original answer")

    assert guided[0]["role"] == "user"
    assert guided[0]["content"].startswith("question")
    assert "original answer" in guided[0]["content"]


def test_render_prompt_falls_back_when_tokenizer_has_no_chat_template():
    from recipe.sft_data_generator.prompting import render_prompt

    class PlainTokenizer:
        chat_template = None

    rendered = render_prompt(PlainTokenizer(), [{"role": "user", "content": "question"}])

    assert rendered == "User: question\nAssistant:"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/recipe/sft_data_generator/test_prompting_on_cpu.py -q`

Expected: FAIL with import errors for `recipe.sft_data_generator.prompting`.

- [ ] **Step 3: Implement prompt module**

Create `recipe/sft_data_generator/prompting.py`:

```python
from __future__ import annotations

import copy
from typing import Any

GUIDANCE_TEMPLATE = """\

Reference solution for guidance:
{origin_answer}

Please solve the original problem again in your own words. Use the reference only as guidance, do not copy it verbatim, and put the final answer in \\boxed{{}}."""


def normalize_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        messages = copy.deepcopy(prompt)
    else:
        messages = [{"role": "user", "content": str(prompt)}]
    return [{"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))} for msg in messages]


def build_guided_messages(prompt: Any, origin_answer: str) -> list[dict[str, str]]:
    messages = normalize_messages(prompt)
    guidance = GUIDANCE_TEMPLATE.format(origin_answer=origin_answer)

    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            messages[index]["content"] = messages[index]["content"].rstrip() + guidance
            return messages

    messages.append({"role": "user", "content": guidance.lstrip()})
    return messages


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    rendered = "\n".join(f"{message['role'].title()}: {message['content']}" for message in messages)
    return rendered.rstrip() + "\nAssistant:"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/recipe/sft_data_generator/test_prompting_on_cpu.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add recipe/sft_data_generator/prompting.py tests/recipe/sft_data_generator/test_prompting_on_cpu.py
git commit -m "feat: add guided prompt builder"
```

Expected: commit succeeds.

---

### Task 3: Reward Wrapper

**Files:**
- Create: `recipe/sft_data_generator/reward.py`
- Test: `tests/recipe/sft_data_generator/test_reward_on_cpu.py`

- [ ] **Step 1: Write failing reward tests**

Create `tests/recipe/sft_data_generator/test_reward_on_cpu.py`:

```python
from recipe.sft_data_generator.reward import RewardResult, score_generation


def test_score_generation_marks_numeric_score_correct():
    def scorer(**kwargs):
        assert kwargs["data_source"] == "math__aime2025"
        assert kwargs["solution_str"] == "answer"
        assert kwargs["ground_truth"] == 70
        assert kwargs["extra_info"] == {"answer": "origin"}
        assert kwargs["exec_timeout"] == 5
        return 1.0

    result = score_generation(
        scorer=scorer,
        data_source="math__aime2025",
        generated_text="answer",
        ground_truth=70,
        extra_info={"answer": "origin"},
        reward_timeout=5,
        min_score=1.0,
    )

    assert result == RewardResult(score=1.0, is_correct=True, error=None)


def test_score_generation_treats_timeout_string_as_error():
    result = score_generation(
        scorer=lambda **kwargs: "timeout",
        data_source="math__aime2025",
        generated_text="answer",
        ground_truth=70,
        extra_info={},
        reward_timeout=5,
        min_score=1.0,
    )

    assert result.score == 0.0
    assert result.is_correct is False
    assert result.error == "non_numeric_score"


def test_score_generation_catches_exception():
    def scorer(**kwargs):
        raise RuntimeError("boom")

    result = score_generation(
        scorer=scorer,
        data_source="math__aime2025",
        generated_text="answer",
        ground_truth=70,
        extra_info={},
        reward_timeout=5,
        min_score=1.0,
    )

    assert result.score == 0.0
    assert result.is_correct is False
    assert result.error == "boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/recipe/sft_data_generator/test_reward_on_cpu.py -q`

Expected: FAIL with import errors for `recipe.sft_data_generator.reward`.

- [ ] **Step 3: Implement reward module**

Create `recipe/sft_data_generator/reward.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from recipe.sft_data_generator.reward_score import compute_score


@dataclass(frozen=True)
class RewardResult:
    score: float
    is_correct: bool
    error: str | None = None


RewardScorer = Callable[..., Any]


def score_generation(
    *,
    data_source: str,
    generated_text: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    reward_timeout: int,
    min_score: float,
    scorer: RewardScorer | None = None,
) -> RewardResult:
    active_scorer = scorer or compute_score
    try:
        raw_score = active_scorer(
            data_source=data_source,
            solution_str=generated_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
            exec_timeout=reward_timeout,
        )
        score = float(raw_score)
    except (TypeError, ValueError):
        return RewardResult(score=0.0, is_correct=False, error="non_numeric_score")
    except Exception as exc:
        return RewardResult(score=0.0, is_correct=False, error=str(exc))

    return RewardResult(score=score, is_correct=score >= min_score, error=None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/recipe/sft_data_generator/test_reward_on_cpu.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add recipe/sft_data_generator/reward.py tests/recipe/sft_data_generator/test_reward_on_cpu.py
git commit -m "feat: add sft generator reward wrapper"
```

Expected: commit succeeds.

---

### Task 4: vLLM Backend

**Files:**
- Create: `recipe/sft_data_generator/vllm_backend.py`

- [ ] **Step 1: Implement lazy vLLM adapter**

Create `recipe/sft_data_generator/vllm_backend.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VLLMConfig:
    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    seed: int | None = None


@dataclass(frozen=True)
class SamplingConfig:
    n: int = 1
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1


class VLLMGenerator:
    def __init__(self, config: VLLMConfig):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vLLM is required for generation. Install vllm in the runtime environment.") from exc

        kwargs = {
            "model": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
        }
        if config.seed is not None:
            kwargs["seed"] = config.seed
        self.llm = LLM(**kwargs)

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[list[str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=sampling.n,
            max_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
        )
        outputs = self.llm.generate(prompts, params)
        return [[completion.text for completion in request_output.outputs] for request_output in outputs]


def batched(items: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]
```

- [ ] **Step 2: Run import check**

Run: `python -c "from recipe.sft_data_generator.vllm_backend import SamplingConfig, VLLMConfig, batched; print(SamplingConfig(n=2)); print(list(batched(['a','b','c'], 2)))"`

Expected: prints a `SamplingConfig` and `[(0, ['a', 'b']), (2, ['c'])]` without importing vLLM.

- [ ] **Step 3: Commit**

Run:

```bash
git add recipe/sft_data_generator/vllm_backend.py
git commit -m "feat: add vllm backend adapter"
```

Expected: commit succeeds.

---

### Task 5: CLI Orchestration

**Files:**
- Create: `recipe/sft_data_generator/generate.py`
- Test: `tests/recipe/sft_data_generator/test_generate_on_cpu.py`

- [ ] **Step 1: Write failing orchestration test**

Create `tests/recipe/sft_data_generator/test_generate_on_cpu.py`:

```python
from argparse import Namespace

import pandas as pd

from recipe.sft_data_generator.generate import run_generation


def _row(index: int, has_ground_truth: bool = True):
    reward_model = {"style": "rule"}
    if has_ground_truth:
        reward_model["ground_truth"] = 70
    return {
        "data_source": "math__aime2025",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"question {index}"},
        ],
        "ability": "math",
        "reward_model": reward_model,
        "extra_info": {
            "answer": f"origin answer {index}",
            "index": index,
            "question": f"question {index}",
            "split": "train",
        },
    }


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return "\n".join(message["content"] for message in messages) + "\nAssistant:"


class FakeGenerator:
    prompts_seen = []

    def __init__(self, config):
        self.config = config

    def generate(self, prompts, sampling):
        type(self).prompts_seen.extend(prompts)
        assert sampling.n == 2
        return [["correct answer"]]


def test_run_generation_skips_missing_ground_truth_counts_errors_and_keeps_prompt_clean(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    info_path = tmp_path / "output_info.json"
    pd.DataFrame([_row(0, has_ground_truth=True), _row(1, has_ground_truth=False)]).to_parquet(input_path)

    args = Namespace(
        model_path="fake-model",
        input_parquet=str(input_path),
        output_parquet=str(output_path),
        output_info_json=str(info_path),
        n=2,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        top_k=-1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        batch_size=8,
        reward_timeout=5,
        min_score=1.0,
        seed=None,
        max_samples=-1,
        info_examples=3,
    )

    info = run_generation(
        args,
        generator_cls=FakeGenerator,
        tokenizer_loader=lambda *unused_args, **unused_kwargs: FakeTokenizer(),
        scorer=lambda **kwargs: 1.0 if kwargs["solution_str"] == "correct answer" else 0.0,
    )

    output = pd.read_parquet(output_path)
    assert len(output) == 1
    assert output.iloc[0]["extra_info"]["answer"] == "correct answer"
    assert output.iloc[0]["extra_info"]["origin_answer"] == "origin answer 0"
    assert output.iloc[0]["extra_info"]["source_index"] == 0
    assert "Reference solution for guidance:" not in str(output.iloc[0]["prompt"])
    assert "origin answer 0" not in str(output.iloc[0]["prompt"])
    assert "Reference solution for guidance:" in FakeGenerator.prompts_seen[0]
    assert "origin answer 0" in FakeGenerator.prompts_seen[0]
    assert info["GenerationStats"]["reward_errors"] == 2
    assert info["GenerationStats"]["generation_errors"] == 1
    assert info["GenerationStats"]["total_generations"] == 4
    assert info["GenerationStats"]["correct_generations"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/recipe/sft_data_generator/test_generate_on_cpu.py -q`

Expected: FAIL with import errors for `recipe.sft_data_generator.generate`.

- [ ] **Step 3: Implement CLI orchestration**

Create `recipe/sft_data_generator/generate.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from recipe.sft_data_generator.data import (
    GenerationRecord,
    GenerationStats,
    build_info_json,
    build_output_rows,
    dataframe_from_rows,
    source_index_for_row,
    validate_input_dataframe,
    write_outputs,
)
from recipe.sft_data_generator.prompting import build_guided_messages, render_prompt
from recipe.sft_data_generator.reward import score_generation
from recipe.sft_data_generator.vllm_backend import SamplingConfig, VLLMConfig, VLLMGenerator, batched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrected SFT data from guided model completions.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--output_info_json", default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--reward_timeout", type=int, default=60)
    parser.add_argument("--min_score", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--info_examples", type=int, default=3)
    return parser.parse_args()


def default_info_path(output_parquet: str) -> str:
    path = Path(output_parquet)
    return str(path.with_name(path.stem + "_info.json"))


def _has_ground_truth(row: dict) -> bool:
    reward_model = row.get("reward_model")
    return isinstance(reward_model, dict) and "ground_truth" in reward_model


def run_generation(
    args: argparse.Namespace,
    *,
    generator_cls=VLLMGenerator,
    tokenizer_loader=AutoTokenizer.from_pretrained,
    scorer=None,
) -> dict:
    dataframe = pd.read_parquet(args.input_parquet)
    if args.max_samples > 0:
        dataframe = dataframe.head(args.max_samples).reset_index(drop=True)
    validate_input_dataframe(dataframe)
    source_rows = dataframe.to_dict("records")

    tokenizer = tokenizer_loader(args.model_path, trust_remote_code=True)
    generation_jobs: list[tuple[int, str]] = []
    reward_errors = 0
    for source_position, row in enumerate(source_rows):
        if not _has_ground_truth(row):
            reward_errors += args.n
            print(f"skipping row {source_position}: missing reward_model.ground_truth")
            continue
        generation_jobs.append(
            (
                source_position,
                render_prompt(
                    tokenizer,
                    build_guided_messages(row["prompt"], row["extra_info"]["answer"]),
                ),
            )
        )

    generator = generator_cls(
        VLLMConfig(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            seed=args.seed,
        )
    )
    sampling = SamplingConfig(
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    correct_records: list[GenerationRecord] = []
    generation_errors = 0
    problems_with_correct: set[int] = set()
    prompt_positions = [source_position for source_position, _prompt in generation_jobs]
    guided_prompts = [prompt for _source_position, prompt in generation_jobs]

    for batch_start, prompt_batch in batched(guided_prompts, args.batch_size):
        try:
            batch_outputs = generator.generate(prompt_batch, sampling)
        except Exception as exc:
            generation_errors += len(prompt_batch) * args.n
            print(f"generation failed for batch starting at {batch_start}: {exc}")
            continue

        if len(batch_outputs) < len(prompt_batch):
            generation_errors += (len(prompt_batch) - len(batch_outputs)) * args.n

        for batch_offset, completions in enumerate(batch_outputs):
            source_position = prompt_positions[batch_start + batch_offset]
            row = source_rows[source_position]
            source_index = source_index_for_row(row, source_position)
            if len(completions) < args.n:
                generation_errors += args.n - len(completions)
            for generation_index, generated_text in enumerate(completions):
                result = score_generation(
                    data_source=row["data_source"],
                    generated_text=generated_text,
                    ground_truth=row["reward_model"]["ground_truth"],
                    extra_info=row["extra_info"],
                    reward_timeout=args.reward_timeout,
                    min_score=args.min_score,
                    scorer=scorer,
                )
                if result.error is not None:
                    reward_errors += 1
                if result.is_correct:
                    problems_with_correct.add(source_position)
                    correct_records.append(
                        GenerationRecord(
                            source_position=source_position,
                            source_index=source_index,
                            generation_index=generation_index,
                            text=generated_text,
                            score=result.score,
                        )
                    )

    output_rows = build_output_rows(source_rows, correct_records)
    output_dataframe = dataframe_from_rows(output_rows)
    stats = GenerationStats(
        input_rows=len(source_rows),
        generations_per_problem=args.n,
        total_generations=len(source_rows) * args.n,
        correct_generations=len(correct_records),
        kept_rows=len(output_rows),
        problems_with_correct_generation=len(problems_with_correct),
        reward_errors=reward_errors,
        generation_errors=generation_errors,
    )
    info = build_info_json(output_dataframe, stats, examples=args.info_examples, source_dataframe=dataframe)
    info_path = args.output_info_json or default_info_path(args.output_parquet)
    write_outputs(output_dataframe, args.output_parquet, info, info_path, source_dataframe=dataframe)
    print(f"wrote {len(output_dataframe)} rows to {args.output_parquet}")
    print(f"wrote info json to {info_path}")
    print(f"generation accuracy: {info['GenerationStats']['generation_accuracy']:.6f}")
    return info


def main() -> None:
    run_generation(parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI checks**

Run:

```bash
pytest tests/recipe/sft_data_generator/test_generate_on_cpu.py -q
python -m recipe.sft_data_generator.generate --help
```

Expected: pytest passes; help command exits 0 and shows `Generate corrected SFT data from guided model completions.`

- [ ] **Step 5: Commit**

Run:

```bash
git add recipe/sft_data_generator/generate.py tests/recipe/sft_data_generator/test_generate_on_cpu.py
git commit -m "feat: add sft data generator cli"
```

Expected: commit succeeds.

---

### Task 6: Final Verification And Reward Package Inclusion

**Files:**
- Include: `recipe/sft_data_generator/reward_score/**`
- Verify: all files from previous tasks.

- [ ] **Step 1: Run focused CPU tests**

Run:

```bash
pytest tests/recipe/sft_data_generator/test_data_on_cpu.py tests/recipe/sft_data_generator/test_prompting_on_cpu.py tests/recipe/sft_data_generator/test_reward_on_cpu.py tests/recipe/sft_data_generator/test_generate_on_cpu.py -q
```

Expected: PASS.

- [ ] **Step 2: Run import and help checks**

Run:

```bash
python -c "from recipe.sft_data_generator.reward_score import compute_score; print(callable(compute_score))"
python -m recipe.sft_data_generator.generate --help
```

Expected: first command prints `True`; second command exits 0.

- [ ] **Step 3: Inspect git status**

Run: `git status --short`

Expected: shows only intended new/modified `recipe/sft_data_generator`, `tests/recipe/sft_data_generator`, and this plan file, plus pre-existing unrelated user changes outside this task.

- [ ] **Step 4: Commit implementation and copied reward package**

Run:

```bash
git add recipe/sft_data_generator tests/recipe/sft_data_generator docs/superpowers/plans/2026-07-09-sft-data-generator.md
git commit -m "feat: add guided sft data generator"
```

Expected: commit succeeds. If previous task commits already included some files, this commit includes the copied `reward_score` directory, the plan file, and any remaining implementation changes.

---

## Self-Review

Spec coverage:

- Reads only parquet: Task 5 reads `--input_parquet`; no info input is used.
- Uses `extra_info.answer` as guidance only: Task 2 builds guided messages; Task 5 does not write guided prompt.
- Missing `reward_model.ground_truth` rows are skipped and counted as reward errors: Task 5 adds `reward_errors += n` before generation for those rows.
- Empty output parquet keeps nested schema instead of `null` columns: Task 1 builds a pyarrow table from source schema and tests empty output.
- vLLM generation with `n`, length, sampling, and runtime controls: Tasks 4 and 5.
- vLLM short returns are counted: Task 5 increments `generation_errors` for missing request outputs and missing completions.
- Bundled reward score: Task 3 imports `recipe.sft_data_generator.reward_score`; Task 6 includes the copied package.
- Correct outputs split into independent items: Task 1 tests and implements one output row per `GenerationRecord`.
- Output `extra_info.answer/origin_answer/source_index/index/generation_index/score`: Task 1 tests and implements all fields.
- Guided prompt does not leak to output prompt: Task 5 tests that the fake generator sees guidance but output parquet keeps the original prompt.
- Tokenizers without chat templates have a plain-text fallback: Task 2 tests and implements fallback rendering.
- Info JSON and accuracy stats: Task 1 implements info JSON and stats; Task 5 writes it.
- CPU tests without GPU: Tasks 1, 2, 3, 5, and 6 use pure tests and lazy vLLM import.

Completeness scan: No unresolved blanks are present in executable steps.

Type consistency: `GenerationRecord`, `GenerationStats`, `build_guided_messages`, `score_generation`, `SamplingConfig`, and `VLLMConfig` are introduced before use and referenced consistently.
