# SFT Data Generator Design

## Goal

Add a new `recipe/sft_data_generator` recipe that reads an existing SFT-style parquet dataset, uses each sample's original SFT answer as guidance during generation, keeps only model generations that pass the bundled rule reward check, and writes a new SFT-style parquet dataset plus an info JSON with accuracy statistics.

The generated dataset should train on the model's own correct answers. The original SFT answer is only a generation-time hint and must not be included in the output prompt.

## Inputs

The generator reads only the input parquet file. It does not read or require the existing `_info.json`.

Required input columns follow the current dataset shape:

- `data_source`
- `prompt`
- `ability`
- `reward_model`
- `extra_info`

Required fields inside the row:

- `extra_info.answer`: original SFT answer used as the generation guidance.
- `reward_model.ground_truth`: answer used by the reward function.
- `extra_info.index`: source row identifier when present. If missing, the dataframe row number is used as the source index.

## CLI

The main command will be:

```bash
python -m recipe.sft_data_generator.generate \
  --model_path /path/to/model \
  --input_parquet /path/to/input.parquet \
  --output_parquet /path/to/output.parquet \
  --n 4 \
  --max_new_tokens 4096
```

Important options:

- `--model_path`: model path or identifier passed to vLLM.
- `--input_parquet`: source parquet path.
- `--output_parquet`: destination parquet path.
- `--n`: number of guided generations per input sample.
- `--max_new_tokens`: maximum generation length.
- `--temperature`, `--top_p`, `--top_k`: sampling controls.
- `--tensor_parallel_size`, `--gpu_memory_utilization`, `--dtype`: vLLM runtime controls.
- `--batch_size`: number of prompts sent to vLLM per batch.
- `--reward_timeout`: timeout forwarded to `compute_score`.
- `--seed`: optional seed for generation.
- `--max_samples`: optional input row limit for debugging.
- `--output_info_json`: optional destination for the generated info JSON.

The info JSON path defaults to the output parquet stem plus `_info.json`.

## Prompt Construction

For generation only, the script builds a temporary prompt from the original `prompt` messages.

The output dataset keeps the original `prompt` unchanged. The guided prompt is never written as the training prompt.

Guidance text is appended to the last user message:

```text

Reference solution for guidance:
{extra_info.answer}

Please solve the original problem again in your own words. Use the reference only as guidance, do not copy it verbatim, and put the final answer in \boxed{}.
```

If a row has a non-list prompt, it is normalized to a single user message for generation while the original prompt value is preserved for output.

## Generation

vLLM is the primary generation backend.

For each input row, the script creates one guided prompt and requests `n` completions. Each completion is tracked with:

- source row position
- source index
- generation index in `[0, n)`
- generated text

Generation failures for a row or batch are recorded in runtime counters. Failed completions do not produce output items.

## Reward Filtering

Each completion is scored with the reward function bundled under `recipe.sft_data_generator.reward_score`:

```python
recipe.sft_data_generator.reward_score.compute_score(
    data_source=row["data_source"],
    solution_str=generated_text,
    ground_truth=row["reward_model"]["ground_truth"],
    extra_info=row["extra_info"],
    exec_timeout=reward_timeout,
)
```

A completion is considered correct if its numeric score is greater than or equal to `--min_score`, default `1.0`.

Timeouts, exceptions, and non-numeric reward outputs are treated as incorrect and counted for reporting.

## Output Dataset

Every correct completion becomes one independent output data item.

If one source problem has three correct generations, the output parquet contains three rows. These rows have the same `extra_info.source_index`, but different output `extra_info.index` values and usually different `extra_info.generation_index` values.

Output columns:

- `data_source`: copied from source row.
- `prompt`: copied from source row unchanged.
- `ability`: copied from source row.
- `reward_model`: copied from source row.
- `extra_info`: copied and updated as described below.

Output `extra_info` rules:

- `answer`: model generated answer that passed reward checking.
- `origin_answer`: original source `extra_info.answer`.
- `source_index`: original `extra_info.index`, falling back to the source dataframe row number.
- `index`: new sequential output row index.
- `is_correct`: `true`.
- `generation_index`: generation attempt index for this source problem.
- `score`: reward score for this generation.
- Other existing source `extra_info` fields, such as `question` and `split`, are preserved unless overwritten by the fields above.

No `response` column is added. The SFT answer remains in `extra_info.answer` to match the current dataset convention.

## Info JSON

The generator writes a new `_info.json` next to the output parquet unless an explicit info path is provided.

The file contains:

- `Rows`: output row count.
- `Columns`: output columns.
- `Schema`: pyarrow schema summary.
- `Struct`: nested schema shape.
- `Examples`: first few output examples.
- `GenerationStats`:
  - `input_rows`
  - `generations_per_problem`
  - `total_generations`
  - `correct_generations`
  - `generation_accuracy`
  - `kept_rows`
  - `problems_with_correct_generation`
  - `problem_pass_rate`
  - `reward_errors`
  - `generation_errors`

`generation_accuracy` is `correct_generations / total_generations`.
`problem_pass_rate` is the number of source problems with at least one correct generation divided by `input_rows`.

## Module Layout

`recipe/sft_data_generator` will contain:

- `__init__.py`: package marker.
- `generate.py`: CLI entry point and orchestration.
- `data.py`: parquet loading, row normalization, output row construction, info JSON generation.
- `prompting.py`: guided prompt construction and chat template preparation.
- `reward.py`: reward wrapper and correctness decision.
- `vllm_backend.py`: vLLM imports and batch generation adapter.
- `reward_score/`: copied reward implementations used by this recipe.

Tests will live under `tests/recipe/sft_data_generator`.

## Error Handling

Input validation fails fast for missing required columns or missing `extra_info.answer`.

Rows with missing `reward_model.ground_truth` are skipped and counted as reward errors.

If no generations pass reward filtering, the script still writes an empty parquet with the expected columns and an info JSON showing zero kept rows.

## Testing

Unit tests should cover:

- guided prompt construction without mutating the output prompt.
- output item construction, including `answer`, `origin_answer`, `source_index`, `index`, `generation_index`, and `score`.
- one source row producing multiple output rows when multiple generations are correct.
- reward filtering with a fake scorer.
- info JSON statistics.

The vLLM path is integration-tested only when vLLM and a local model are available. Unit tests do not require GPU or model downloads.
