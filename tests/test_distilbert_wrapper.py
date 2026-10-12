"""Offline DistilBERT smoke test using a tiny encoder stub.

Tests the model wrapper and gradients, NOT pretrained model quality or downloads.
"""
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from src.models import BERTNER


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DistilBERTWrapperTests(unittest.TestCase):
    def test_forward_backward_with_mock_encoder(self):
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModel = types.SimpleNamespace(
            from_pretrained=lambda name: TinyEncoder()
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = BERTNER("distilbert-base-cased", num_labels=9, dropout=0)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        logits = model(tokens, mask)
        self.assertEqual(tuple(logits.shape), (2, 3, 9))
        labels = torch.tensor([[5, 0, 0], [0, 3, -100]])
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.reshape(-1, 9), labels.reshape(-1)
        )
        loss.backward()
        self.assertIsNotNone(model.encoder.embedding.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
