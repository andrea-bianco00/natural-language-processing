"""The two NER architectures from the ENSAI project, separated from the notebook."""
import torch
from torch import nn


class LSTMNER(nn.Module):
    """Two-layer bidirectional LSTM with randomly initialized token embeddings."""

    def __init__(self, vocab_size, embed_dim=200, hidden_dim=200,
                 num_labels=9, n_layers=2, dropout=0.3, pad_token_id=0):
        super().__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        embedded = self.dropout(self.embedding(input_ids))
        lengths = attention_mask.sum(dim=1).cpu()
        if (lengths == 0).any():
            raise ValueError("Empty sequences are not supported")
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=input_ids.size(1)
        )
        return self.classifier(output)


class BERTNER(nn.Module):
    """Pretrained DistilBERT encoder with dropout and token-classification head."""

    def __init__(self, encoder_checkpoint="distilbert-base-cased",
                 num_labels=9, dropout=0.3):
        super().__init__()
        # Lazy import keeps LSTM-only tests independent of transformers.
        from transformers import AutoModel
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(encoder_checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden))
