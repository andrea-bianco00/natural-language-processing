"""Generate synthetic NER examples; never uses or copies CoNLL-2003."""
import json
from pathlib import Path

examples = [
    {"tokens": ["Alice", "works", "in", "Paris", "."], "tags": ["B-PER", "O", "O", "B-LOC", "O"]},
    {"tokens": ["Bob", "joined", "Acme", "in", "London", "."], "tags": ["B-PER", "O", "B-ORG", "O", "B-LOC", "O"]},
    {"tokens": ["Carla", "visited", "Rome", "."], "tags": ["B-PER", "O", "B-LOC", "O"]},
    {"tokens": ["Delta", "opened", "an", "office", "."], "tags": ["B-ORG", "O", "O", "O", "O"]},
]
path = Path("outputs/synthetic_smoke_data.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"train": examples, "valid": examples[:2], "test": examples[2:]}), encoding="utf-8")
print(path)
