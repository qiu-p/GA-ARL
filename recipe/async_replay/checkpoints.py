"""Separate retention for periodic checkpoints and the highest test score."""
import json
import math
import os
from pathlib import Path
import shutil


class PeriodicBestCheckpointManager:
    def __init__(self, base_dir, metric, keep_periodic=5):
        if keep_periodic < 1:
            raise ValueError("keep_periodic must be positive")
        self.root = Path(base_dir).resolve()
        self.metric = metric
        self.metrics_startswith = "pass"
        self.keep_periodic = keep_periodic
        self.manifest = self.root / "checkpoint_manifest.json"

    def set_metrics_startswith(self, prefix):
        # Compatibility with trainer diagnostics; selection uses the exact metric.
        self.metrics_startswith = prefix

    def _load(self):
        if not self.manifest.exists():
            return {"metric": self.metric, "saved_steps": [], "periodic_steps": [], "best": None}
        state = json.loads(self.manifest.read_text(encoding="utf-8"))
        if state["metric"] != self.metric:
            raise ValueError("Checkpoint directory already uses a different best metric")
        return state

    def _score(self, metrics):
        if not metrics:
            return None
        if self.metric not in metrics:
            raise ValueError(f"Missing best-checkpoint metric: {self.metric}")
        score = float(metrics[self.metric])
        return score if math.isfinite(score) else None

    def get_improved_metrics(self, val_metrics):
        score = self._score(val_metrics)
        best = self._load()["best"]
        if score is not None and (best is None or score > best["score"]):
            return {self.metric: {"old_best": None if best is None else best["score"], "new_best": score}}
        return {}

    def _write(self, path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def update_after_save(self, step, ckpt_path, val_metrics, *, periodic=False):
        """Call only after model, optimizer and dataloader saving succeeds."""
        step = int(step)
        path = self.root / f"global_step_{step}"
        if Path(ckpt_path).resolve() != path or path.is_symlink() or not path.is_dir():
            raise ValueError(f"Unexpected checkpoint directory: {ckpt_path}")
        state = self._load()
        saved = set(state["saved_steps"]) | {step}
        periods = set(state["periodic_steps"])
        if periodic:
            periods.add(step)
        periods = set(sorted(periods)[-self.keep_periodic:])
        score = self._score(val_metrics)
        best = state["best"]
        if score is not None and (best is None or score > best["score"]):
            best = {"metric": self.metric, "score": score, "step": step, "path": str(path)}
        keep = periods | ({best["step"]} if best is not None else set())
        state.update(saved_steps=sorted(saved), periodic_steps=sorted(periods), best=best)
        # Commit the new best pointer before cleaning any former best directory.
        self._write(self.manifest, state)
        if best is not None:
            self._write(self.root / "best_checkpoint.json", best)
        for old_step in sorted(saved - keep):
            candidate = self.root / f"global_step_{old_step}"
            if candidate.is_symlink() or candidate.resolve().parent != self.root:
                raise ValueError(f"Refusing to remove checkpoint outside root: {candidate}")
            if candidate.exists():
                shutil.rmtree(candidate)
            saved.remove(old_step)
        state["saved_steps"] = sorted(saved)
        self._write(self.manifest, state)
