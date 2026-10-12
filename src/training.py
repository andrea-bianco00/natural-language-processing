"""Device-safe training and evaluation, adapted from the original notebook."""
import random
import numpy as np
import torch
from torch import nn


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeds improve repeatability but do not guarantee identical GPU results.


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits.reshape(-1, model.num_labels),
                         batch["labels"].reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if not losses:
        raise ValueError("Training loader is empty")
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    losses, correct, total = [], 0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        losses.append(criterion(logits.reshape(-1, model.num_labels),
                                labels.reshape(-1)).item())
        valid = labels != -100
        correct += ((logits.argmax(-1) == labels) & valid).sum().item()
        total += valid.sum().item()
    if not losses or total == 0:
        raise ValueError("Evaluation loader has no labeled tokens")
    return {"loss": sum(losses) / len(losses), "token_accuracy": correct / total}


@torch.no_grad()
def entity_metrics(model, loader, device, id_to_tag):
    """Entity-level precision, recall and F1 via seqeval."""
    from seqeval.metrics import precision_score, recall_score, f1_score
    model.eval()
    true_sequences, predicted_sequences = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        predicted = logits.argmax(-1)
        for true_ids, pred_ids in zip(batch["labels"].tolist(), predicted.tolist()):
            valid_pairs = [(a, b) for a, b in zip(true_ids, pred_ids) if a != -100]
            true_sequences.append([id_to_tag[a] for a, _ in valid_pairs])
            predicted_sequences.append([id_to_tag[b] for _, b in valid_pairs])
    return {
        "precision": precision_score(true_sequences, predicted_sequences),
        "recall": recall_score(true_sequences, predicted_sequences),
        "f1": f1_score(true_sequences, predicted_sequences),
    }
