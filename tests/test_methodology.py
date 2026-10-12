"""Regression tests for BIO subword alignment and entity evaluation."""
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.preprocessing import TAG_TO_ID, align_labels, collate_batch
from src.training import entity_metrics, evaluate


class FixedModel(torch.nn.Module):
    num_labels = 9

    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions

    def forward(self, input_ids, attention_mask):
        batch, width = input_ids.shape
        logits = torch.full((batch, width, self.num_labels), -10.0, device=input_ids.device)
        for i, row in enumerate(self.predictions):
            for j, label in enumerate(row[:width]):
                logits[i, j, label] = 10.0
        return logits


class MethodologyTests(unittest.TestCase):
    def test_b_to_i_on_continuation(self):
        self.assertEqual(
            align_labels([None, 0, 0, 1, None], ["B-PER", "O"]),
            [-100, TAG_TO_ID["B-PER"], TAG_TO_ID["I-PER"], 0, -100],
        )

    def test_i_and_o_continuations(self):
        self.assertEqual(
            align_labels([None, 0, 0, 1, 1, None], ["I-ORG", "O"]),
            [-100, 4, 4, 0, 0, -100],
        )

    def test_padding_and_special_tokens(self):
        batch = collate_batch([
            {"input_ids": [101, 12, 102], "attention_mask": [1, 1, 1], "labels": [-100, 5, -100]},
            {"input_ids": [101, 13, 14, 102], "attention_mask": [1, 1, 1, 1], "labels": [-100, 0, 0, -100]},
        ])
        self.assertEqual(batch["labels"][0].tolist(), [-100, 5, -100, -100])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0])

    def test_token_accuracy_ignores_masked_positions(self):
        sample = {"input_ids": [101, 12, 13, 102], "attention_mask": [1, 1, 1, 1],
                  "labels": [-100, 5, 6, -100]}
        loader = DataLoader([sample], batch_size=1, collate_fn=collate_batch)
        result = evaluate(FixedModel([[0, 5, 0, 0]]), loader, torch.device("cpu"))
        self.assertEqual(result["token_accuracy"], 0.5)

    def test_entity_metrics_drop_ignored_positions(self):
        sample = {"input_ids": [101, 12, 13, 102], "attention_mask": [1, 1, 1, 1],
                  "labels": [-100, 5, 6, -100]}
        loader = DataLoader([sample], batch_size=1, collate_fn=collate_batch)
        captured = []
        def scorer(true, predicted):
            captured.append((true, predicted))
            return 1.0
        fake = types.ModuleType("seqeval")
        metrics = types.ModuleType("seqeval.metrics")
        metrics.precision_score = scorer
        metrics.recall_score = scorer
        metrics.f1_score = scorer
        with patch.dict(sys.modules, {"seqeval": fake, "seqeval.metrics": metrics}):
            result = entity_metrics(FixedModel([[0, 5, 6, 0]]), loader,
                                    torch.device("cpu"), list(TAG_TO_ID))
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(captured[0], ([["B-PER", "I-PER"]], [["B-PER", "I-PER"]]))


if __name__ == "__main__":
    unittest.main()
