"""Small offline tests: no pretrained weights or CoNLL dataset required."""
import unittest
import torch
from torch.utils.data import DataLoader
from src.preprocessing import align_labels, collate_batch, NERDataset
from src.models import LSTMNER
from src.training import evaluate, train_epoch, seed_everything


class PipelineTests(unittest.TestCase):
    def test_subword_label_alignment(self):
        self.assertEqual(
            align_labels([None, 0, 0, 1, None], ["B-PER", "O"]),
            [-100, 5, 6, 0, -100],
        )

    def test_padding(self):
        examples = [
            {"input_ids": [2, 4], "attention_mask": [1, 1], "labels": [5, 0]},
            {"input_ids": [2], "attention_mask": [1], "labels": [0]},
        ]
        batch = collate_batch(examples)
        self.assertEqual(batch["input_ids"].shape, (2, 2))
        self.assertEqual(batch["labels"][1].tolist(), [0, -100])

    def test_lstm_forward_train_and_evaluate(self):
        seed_everything(42)
        examples = [
            {"input_ids": [2, 4], "attention_mask": [1, 1], "labels": [5, 0]},
            {"input_ids": [3], "attention_mask": [1], "labels": [0]},
        ]
        loader = DataLoader(NERDataset(examples), batch_size=2, collate_fn=collate_batch)
        model = LSTMNER(vocab_size=10, embed_dim=8, hidden_dim=6,
                        num_labels=9, n_layers=1, dropout=0)
        batch = next(iter(loader))
        self.assertEqual(model(batch["input_ids"], batch["attention_mask"]).shape, (2, 2, 9))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.assertGreaterEqual(train_epoch(model, loader, optimizer, torch.device("cpu")), 0)
        result = evaluate(model, loader, torch.device("cpu"))
        self.assertTrue(0 <= result["token_accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
