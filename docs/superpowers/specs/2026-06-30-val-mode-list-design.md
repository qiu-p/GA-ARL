# val_mode List Support Design

**Date**: 2026-06-30
**Author**: shyang
**Status**: Approved

## Overview

Modify `_validate()` in `verl/trainer/ppo/ray_trainer.py` to support `val_mode` as a list of integers (e.g., `[1, 5]`), enabling simultaneous collection of `pass@1`, `pass@5`, etc. from a single inference run.

## Background

Currently `_validate(val_mode=None)` accepts but does not use the `val_mode` parameter. The user needs to compute multiple `pass@k` metrics at once — e.g., `pass@1` and `pass@5` — without running inference multiple times.

## Requirements

1. `val_mode` accepts: `None`, `int` (e.g., `5`), `str` (e.g., `"5"` or `"pass@5"` for backward compat), `List[int]` (e.g., `[1, 5]`)
2. When `val_mode` is not a list, normalize it to a single-element list
3. Inference runs **once** with `n = max(val_mode)`, overriding `val_kwargs.n`
4. `pass@k` = for each prompt, take first k responses; if **any** is correct → pass. Final metric = fraction of prompts that pass
5. Report metrics under `val-core/{data_source}/acc/pass@{k}`

## Implementation Plan

### Files to Modify

1. **`verl/trainer/ppo/ray_trainer.py`** — primary changes
2. **`verl/trainer/ppo/metric_utils.py`** — optional, if helper fits better there
3. **Shell scripts** — update `val_mode` format from `[pass@1,pass@5]` to `[1,5]`

### New Functions

#### `_normalize_val_mode(val_mode) -> List[int]`

Normalize input to a sorted, deduplicated list of integers:

| Input | Output |
|-------|--------|
| `None` | `[]` |
| `5` (int) | `[5]` |
| `"pass@5"` (str) | `[5]` |
| `[1, 5]` (list) | `[1, 5]` |
| `["pass@1", "pass@5"]` (list of str, compat) | `[1, 5]` |

#### `_compute_pass_k_metrics(sample_uids, sample_scores, val_mode, data_sources) -> dict`

For each k in val_mode, group scores by uid, check if any of the first k scores > 0 (correct), and compute the pass rate per data_source.

Returns metrics like:
```
val-core/aime/acc/pass@1 → 0.42
val-core/aime/acc/pass@5 → 0.68
```

### Modified Logic in `_validate()`

```
_validate(val_mode=None):
    1. val_mode = self._normalize_val_mode(val_mode)
    2. max_n = max(val_mode) if val_mode else None
    3. For each test_batch:
       a. actual_n = max_n if max_n is not None else val_kwargs.n
       b. repeat test_batch actual_n times
       c. Generate, score (unchanged)
    4. Call existing process_validation_metrics() (unchanged)
    5. If val_mode is non-empty:
       a. Compute pass@k metrics via _compute_pass_k_metrics()
       b. Merge into metric_dict
    6. Return metric_dict
```

### Error Handling & Edge Cases

- **Empty val_mode**: behave exactly as before (no pass@k metrics)
- **val_mode contains k > actual n generated**: skip that k for prompts with fewer responses, log a warning
- **val_mode=[1] when val_kwargs.n=1**: zero overhead, just adds `pass@1` metric
- **Compatibility**: all existing callers passing `val_mode=None` or `val_mode="pass@5"` continue to work

## Design Decisions

- **Deterministic first-k, not bootstrap**: `pass@k` uses the first k responses deterministically. The existing `best@N` bootstrap metrics remain separate and unchanged. This avoids mixing two different statistical approaches.
- **max(val_mode) overrides val_kwargs.n**: The user explicitly stated inference should use the largest k. If val_kwargs.n is larger, we could consider using that instead, but max(val_mode) is the simpler and more predictable behavior.
- **Sorted, deduplicated val_mode**: `[5, 1, 5]` → `[1, 5]` to avoid redundant computation.
