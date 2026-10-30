"""Filesystem retention checks; no model or GPU dependencies."""
import json
from pathlib import Path
import tempfile
import unittest

from recipe.async_replay.checkpoints import PeriodicBestCheckpointManager


METRIC = "val-core/codegen__livecodebench/acc/pass@1"


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.manager = PeriodicBestCheckpointManager(self.root, METRIC, 5)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, step, score=None, periodic=False):
        path = self.root / f"global_step_{step}"
        path.mkdir(exist_ok=True)
        (path / "weights").write_text("fixture")
        self.manager.update_after_save(
            step, path, {} if score is None else {METRIC: score}, periodic=periodic
        )

    def steps(self):
        return {int(p.name.removeprefix("global_step_")) for p in self.root.glob("global_step_*")}

    def test_five_periodic_plus_initial_best_then_replace_best(self):
        self.save(0, .5)
        for step in range(20, 141, 20):
            self.save(step, .4, periodic=True)
        self.assertEqual(self.steps(), {0, 60, 80, 100, 120, 140})
        self.save(145, .6)
        self.assertEqual(self.steps(), {60, 80, 100, 120, 140, 145})
        best = json.loads((self.root / "best_checkpoint.json").read_text())
        self.assertEqual(best["step"], 145)
        self.assertEqual(best["score"], .6)

    def test_overlap_and_final_step_count_as_periodic(self):
        for step in [20, 40, 60, 80]:
            self.save(step, step / 100, periodic=True)
        self.save(82, .9, periodic=True)
        self.assertEqual(len(self.steps()), 5)
        self.save(100, .8, periodic=True)
        self.assertEqual(self.steps(), {40, 60, 80, 82, 100})

    def test_reload_equal_score_and_missing_metric(self):
        self.save(5, .7)
        reloaded = PeriodicBestCheckpointManager(self.root, METRIC, 5)
        reloaded.set_metrics_startswith("pass")
        self.assertEqual(reloaded.metrics_startswith, "pass")
        self.assertFalse(reloaded.get_improved_metrics({METRIC: .7}))
        self.assertFalse(reloaded.get_improved_metrics({METRIC: float("nan")}))
        self.assertTrue(reloaded.get_improved_metrics({METRIC: .8}))
        with self.assertRaises(ValueError):
            reloaded.get_improved_metrics({"wrong_metric": .9})

    def test_unknown_directories_are_not_deleted(self):
        (self.root / "global_step_999").mkdir()
        self.save(0, .5)
        for step in range(20, 141, 20):
            self.save(step, .4, periodic=True)
        self.assertTrue((self.root / "global_step_999").exists())

    def test_former_best_still_retained_when_it_is_periodic(self):
        self.save(20, .5, periodic=True)
        self.save(25, .6)
        self.assertEqual(self.steps(), {20, 25})
        self.save(30, .7)
        self.assertEqual(self.steps(), {20, 30})

    def test_reject_checkpoint_outside_root(self):
        with self.assertRaises(ValueError):
            self.manager.update_after_save(1, self.root.parent, {METRIC: 1.0})


if __name__ == "__main__":
    unittest.main()
