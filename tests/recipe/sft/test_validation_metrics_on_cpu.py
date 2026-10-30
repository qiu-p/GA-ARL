import math

from recipe.sft.utils import build_sft_target_loss_mask, select_best_metric_candidates


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


def test_build_sft_target_loss_mask_marks_response_targets_only():
    mask = build_sft_target_loss_mask(
        prompt_length=4,
        response_length=3,
        sequence_length=9,
        attention_mask=[1, 1, 1, 1, 1, 1, 1, 0, 0],
    )

    assert mask == [0, 0, 0, 0, 1, 1, 1, 0, 0]
