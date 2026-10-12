"""Validate published full-dataset aggregate evidence without restricted data."""
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"train": 14041, "valid": 3250, "test": 3453}
EXPECTED_EPOCHS = {"bilstm": 10, "distilbert": 3}


class RecordedResultsTests(unittest.TestCase):
    def test_full_dataset_metrics_and_history(self):
        for name, epochs in EXPECTED_EPOCHS.items():
            with self.subTest(model=name):
                path = ROOT / "results" / f"{name}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(result["model"], name)
                self.assertEqual(result["split_counts"], EXPECTED_COUNTS)
                self.assertEqual(result["epochs"], epochs)
                self.assertEqual(len(result["validation_history"]), epochs)
                self.assertEqual(
                    result["sample_limits"],
                    {"train": None, "valid": None, "test": None},
                )
                for metric in ("token_accuracy",):
                    value = result["test"][metric]
                    self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
                scores = result["test_entities"]
                for metric in ("precision", "recall", "f1"):
                    value = scores[metric]
                    self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
                precision, recall = scores["precision"], scores["recall"]
                expected_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                self.assertAlmostEqual(scores["f1"], expected_f1, places=9)
                for i, epoch in enumerate(result["validation_history"], 1):
                    self.assertEqual(epoch["epoch"], i)
                    self.assertTrue(math.isfinite(epoch["train_loss"]))
                    self.assertTrue(math.isfinite(epoch["validation"]["loss"]))

    def test_environment_manifest_contains_no_dataset(self):
        path = ROOT / "results" / "reproduction_environment.json"
        environment = json.loads(path.read_text(encoding="utf-8"))
        for field in ("python", "torch", "transformers", "scikit_learn", "seqeval", "dataset"):
            self.assertIn(field, environment)
        self.assertIn("not included", environment["dataset"])


if __name__ == "__main__":
    unittest.main()
