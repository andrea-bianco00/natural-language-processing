"""Compare newly measured NER results with saved original-notebook metrics.

This script never labels the reference numbers as independently reproduced.
"""
import argparse
import json
from pathlib import Path

REFERENCE = {
    "bilstm": {"token_accuracy": 0.9355, "precision": 0.7328, "recall": 0.7512, "f1": 0.7419},
    "distilbert": {"token_accuracy": 0.9642, "precision": 0.8430, "recall": 0.8810, "f1": 0.8616},
}
METRICS = ("token_accuracy", "precision", "recall", "f1")


def compare_results(results):
    """Return comparison rows; raise on malformed or synthetic-only results."""
    model = results.get("model")
    if model not in REFERENCE:
        raise ValueError("Unrecognized model name")
    if any(value is not None for value in results.get("sample_limits", {}).values()):
        raise ValueError("Partial/smoke results must not be compared with full-dataset reference metrics")
    counts = results.get("split_counts", {})
    expected = {"train": 14041, "valid": 3250, "test": 3453}
    if counts != expected:
        raise ValueError("Split counts differ from original dataset; comparison rejected")
    test = results.get("test", {})
    entities = results.get("test_entities", {})
    actual = {"token_accuracy": test.get("token_accuracy"),
              **{key: entities.get(key) for key in ("precision", "recall", "f1")}}
    rows = []
    for metric in METRICS:
        value = actual[metric]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Missing or invalid metric: {metric}")
        ref = REFERENCE[model][metric]
        rows.append({"metric": metric, "notebook_reference": ref,
                     "new_run": value, "difference_percentage_points": round(100 * (value - ref), 2)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="JSON produced by run_experiment.py")
    args = parser.parse_args()
    try:
        data = json.loads(args.result.read_text(encoding="utf-8"))
        rows = compare_results(data)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot compare: {exc}\n")
    print(f"Model: {data['model']} | Original notebook vs new full-data run")
    print(f"{'Metric':<18} {'Notebook':>10} {'New run':>10} {'Delta (pp)':>12}")
    for row in rows:
        print(f"{row['metric']:<18} {100*row['notebook_reference']:>9.2f}% "
              f"{100*row['new_run']:>9.2f}% {row['difference_percentage_points']:>+11.2f}")


if __name__ == "__main__":
    main()
