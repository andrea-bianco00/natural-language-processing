"""Run the ENSAI NER experiment on an authorized local dataset."""
import argparse
import json
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bilstm", "distilbert"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.output is None:
        args.output = Path("outputs") / f"{args.model}.json"

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from src.validate_data import load_dataset, validate_dataset
    from src.preprocessing import ID_TO_TAG, NERDataset, collate_batch, encode_split
    from src.models import LSTMNER, BERTNER
    from src.training import (seed_everything, resolve_device, train_epoch,
                              evaluate, entity_metrics)

    seed_everything(args.seed)
    device = resolve_device()
    data = load_dataset(args.data)
    counts = validate_dataset(data)
    checkpoint = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    collate = partial(collate_batch, pad_token_id=tokenizer.pad_token_id)
    loaders = {}
    for split in ("train", "valid", "test"):
        examples = encode_split(data[split], tokenizer)
        loaders[split] = DataLoader(
            NERDataset(examples), batch_size=args.batch_size,
            shuffle=split == "train", collate_fn=collate,
        )

    if args.model == "bilstm":
        model = LSTMNER(len(tokenizer), embed_dim=200, hidden_dim=200,
                        num_labels=len(ID_TO_TAG), n_layers=2, dropout=0.3,
                        pad_token_id=tokenizer.pad_token_id)
        epochs, lr = 10, 5e-4
    else:
        model = BERTNER(checkpoint, num_labels=len(ID_TO_TAG), dropout=0.3)
        epochs, lr = 3, 1e-5
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        valid = evaluate(model, loaders["valid"], device)
        history.append({"epoch": epoch, "train_loss": loss, "validation": valid})
        print(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, validation={valid}")

    # Final-epoch evaluation only: no tuning on the test split.
    test = evaluate(model, loaders["test"], device)
    entities = entity_metrics(model, loaders["test"], device, ID_TO_TAG)
    results = {
        "model": args.model, "checkpoint": checkpoint, "seed": args.seed,
        "epochs": epochs, "learning_rate": lr, "batch_size": args.batch_size,
        "device": str(device), "split_counts": counts,
        "validation_history": history, "test": test, "test_entities": entities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; test entity F1={entities['f1']:.4f}")


if __name__ == "__main__":
    main()
