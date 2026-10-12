import unittest
from compare_results import REFERENCE, compare_results


def full_result():
    return {
        "model": "bilstm",
        "split_counts": {"train": 14041, "valid": 3250, "test": 3453},
        "sample_limits": {"train": None, "valid": None, "test": None},
        "test": {"token_accuracy": 0.9355},
        "test_entities": {"precision": 0.7328, "recall": 0.7512, "f1": 0.7419},
    }


class ComparisonTests(unittest.TestCase):
    def test_equal_results_have_zero_delta(self):
        rows = compare_results(full_result())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["difference_percentage_points"] == 0 for row in rows))

    def test_rejects_smoke_results(self):
        result = full_result()
        result["sample_limits"]["train"] = 2
        with self.assertRaisesRegex(ValueError, "Partial"):
            compare_results(result)

    def test_rejects_wrong_dataset(self):
        result = full_result()
        result["split_counts"]["test"] = 2
        with self.assertRaisesRegex(ValueError, "Split counts"):
            compare_results(result)

    def test_rejects_missing_metric(self):
        result = full_result()
        del result["test_entities"]["f1"]
        with self.assertRaisesRegex(ValueError, "f1"):
            compare_results(result)


if __name__ == "__main__":
    unittest.main()
