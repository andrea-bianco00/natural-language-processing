"""Tokenization and label alignment adapted from the original ENSAI notebook.

The original experiment labels continuation subwords with I-ENTITY (rather than
ignoring them). This behavior is intentionally preserved.
"""
from torch.utils.data import Dataset
import torch

TAG_TO_ID = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-ORG": 3, "I-ORG": 4,
             "B-PER": 5, "I-PER": 6, "B-MISC": 7, "I-MISC": 8}
ID_TO_TAG = list(TAG_TO_ID)
IGNORE_INDEX = -100


def align_labels(word_ids, word_tags):
    """Align IOB labels to tokenizer word IDs, ignoring special tokens."""
    aligned = []
    previous_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)
            continue
        tag = TAG_TO_ID[word_tags[word_id]]
        if word_id == previous_word and tag in (1, 3, 5, 7):
            tag += 1
        aligned.append(tag)
        previous_word = word_id
    return aligned


def encode_split(samples, tokenizer):
    """Return unpadded, subword-aligned examples from a fast tokenizer."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for word_ids() alignment")
    encoded = tokenizer([sample["tokens"] for sample in samples],
                        is_split_into_words=True, padding=False, truncation=True)
    return [
        {"input_ids": encoded["input_ids"][i],
         "attention_mask": encoded["attention_mask"][i],
         "labels": align_labels(encoded.word_ids(batch_index=i), sample["tags"])}
        for i, sample in enumerate(samples)
    ]


class NERDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id=0):
    """Pad batch to the longest example; do not train on padded positions."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(example["input_ids"]) for example in batch)
    return {
        key: torch.tensor([
            example[key] + [pad_value] * (length - len(example[key]))
            for example in batch
        ], dtype=torch.long)
        for key, pad_value in (
            ("input_ids", pad_token_id), ("attention_mask", 0),
            ("labels", IGNORE_INDEX)
        )
    }
