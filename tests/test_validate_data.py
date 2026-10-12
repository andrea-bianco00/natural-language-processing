"""Fast tests; no dataset download, GPU, or third-party dependencies."""
import unittest
from src.validate_data import validate_dataset


def example_data():
    example = {"tokens": ["Ada", "works", "here"], "tags": ["B-PER", "O", "O"]}
    return {split: [dict(example)] for split in ("train", "valid", "test")}


class DatasetValidationTests(unittest.TestCase):
    def test_valid_sample(self):
        self.assertEqual(validate_dataset(example_data()), {"train": 1, "valid": 1, "test": 1})

    def test_mismatched_lengths(self):
        data = example_data()
        data["train"][0]["tags"] = ["B-PER"]
        with self.assertRaisesRegex(ValueError, "equal length"):
            validate_dataset(data)

    def test_invalid_tag(self):
        data = example_data()
        data["test"][0]["tags"][0] = "B-UNKNOWN"
        with self.assertRaisesRegex(ValueError, "invalid IOB tag"):
            validate_dataset(data)

    def test_missing_split(self):
        data = example_data()
        del data["valid"]
        with self.assertRaisesRegex(ValueError, "exactly these splits"):
            validate_dataset(data)

    def test_expected_counts(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_dataset(example_data(), {"train": 2, "valid": 1, "test": 1})


if __name__ == "__main__":
    unittest.main()
