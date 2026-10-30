# GA-ARL training recipe

Start with the [configuration guide](GA_ARL_CONFIG.md), then select a script from the [experiment index](scripts/README.md). Installation instructions are in the [repository README](../../README.md).

`main_ppo.py` creates the workers and invokes `OneStepOffRayTrainer` in `trainer.py`. `replay/buffer.py` stores recent rollout batches and historical statistics. `replay/selection.py` computes group scores and selects data for actor updates. Evaluation and checkpoint helpers are in `evaluation.py` and `checkpoints.py`.

The main method enables replay and uses `select_by=ref`. It scores cached responses without a current-actor forward over the entire replay candidate pool. Selected samples still undergo actor log-probability evaluation and optimization. The `cur` branch provides the current-policy selection comparison; `ref_no_var` removes advantage-magnitude weighting.

The vendored utilities in `_vendor/` are imported locally and require no separate qiuputils installation. `legacy/` and compatibility import modules support earlier implementations; the documented experiments use `trainer.py`.
