"""Validate the JSON structure used by the original ENSAI NER project.

Standard library only. Does not download or redistribute CoNLL-2003 data.
"""
import argparse
import gzip
import json
from pathlib import Path

TAGS = {"O"} | {f"{prefix}-{kind}" for prefix in ("B", "I") for kind in ("LOC", "ORG", "PER", "MISC")}
SPLITS = ("train", "valid", "test")


def validate_dataset(data, expected_counts=None):
    """Return split counts, or raise ValueError for malformed input."""
    if not isinstance(data, dict) or set(data) != set(SPLITS):
        raise ValueError(f"Expected exactly these splits: {', '.join(SPLITS)}")
    counts = {}
    for split in SPLITS:
        samples = data[split]
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{split}: expected a non-empty list")
        for index, sample in enumerate(samples):
            prefix = f"{split}[{index}]"
            if not isinstance(sample, dict) or not {"tokens", "tags"} <= set(sample):
                raise ValueError(f"{prefix}: missing tokens or tags")
            tokens, tags = sample["tokens"], sample["tags"]
            if not isinstance(tokens, list) or not isinstance(tags, list) or not tokens or len(tokens) != len(tags):
                raise ValueError(f"{prefix}: tokens and tags must be non-empty lists of equal length")
            if any(not isinstance(token, str) or not token for token in tokens):
                raise ValueError(f"{prefix}: invalid token")
            if any(not isinstance(tag, str) or tag not in TAGS for tag in tags):
                raise ValueError(f"{prefix}: invalid IOB tag")
        counts[split] = len(samples)
        if expected_counts and counts[split] != expected_counts[split]:
            raise ValueError(f"{split}: found {counts[split]} sentences; expected {expected_counts[split]}")
    return counts


def load_dataset(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Authorized local JSON or JSON.gz dataset")
    parser.add_argument("--original-split-counts", action="store_true", help="Check original ENSAI split sizes")
    args = parser.parse_args()
    expected = {"train": 14041, "valid": 3250, "test": 3453} if args.original_split_counts else None
    try:
        counts = validate_dataset(load_dataset(args.path), expected)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    print("Dataset structure valid: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
