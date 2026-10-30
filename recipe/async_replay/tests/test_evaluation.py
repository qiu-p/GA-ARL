import unittest
from recipe.async_replay.evaluation import average_accuracy


class AverageAccuracyTests(unittest.TestCase):
    def test_average_is_not_any_sample_pass_rate(self):
        metrics = average_accuracy(["a"] * 8 + ["b"] * 8,
                                   [1] + [0] * 7 + [1] * 8, [8], ["code"] * 16)
        self.assertEqual(metrics["val-core/code/acc/avg@8"], 9 / 16)

    def test_sources_and_counts_are_separate(self):
        metrics = average_accuracy(["a"] * 8 + ["a"] * 8,
                                   [0] * 8 + [1] * 8, [1, 8], ["x"] * 8 + ["y"] * 8)
        self.assertEqual(metrics["val-core/x/acc/avg@8"], 0)
        self.assertEqual(metrics["val-core/y/acc/avg@8"], 1)

    def test_incomplete_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            average_accuracy(["a"], [1], [8], ["x"])


if __name__ == "__main__":
    unittest.main()
