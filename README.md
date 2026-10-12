# Natural Language Processing
## Named Entity Recognition: BiLSTM vs DistilBERT

**Can a pretrained Transformer recognize named entities more accurately than a bidirectional recurrent neural network?**

This project investigates **Named Entity Recognition (NER)**: identifying mentions of people, organizations, locations, and miscellaneous entities in text. It compares a **bidirectional LSTM (BiLSTM)** sequence tagger with **DistilBERT** fine-tuned for token classification, using the **CoNLL-2003** benchmark.

Developed as an individual academic project for *Machine Learning for Natural Language Processing* at **ENSAI**.

## Research question

Both models predict an IOB tag for each word, but they represent language differently:

- **BiLSTM** learns contextual representations through a recurrent architecture.
- **DistilBERT** starts from pretrained Transformer representations and is adapted to NER.

The central question is whether pretrained contextual representations improve **entity-level recognition**, rather than merely increasing token-level accuracy.

## Approach

1. **Data:** use the CoNLL-2003 train, validation, and test splits.
2. **Preprocessing:** map IOB tags to IDs; build token sequences and batches. For DistilBERT, align word-level labels with subword tokens.
3. **Modeling:** train a BiLSTM tagger and fine-tune DistilBERT for token classification.
4. **Evaluation:** compare token accuracy and entity-level precision, recall, and F1. Entity-level metrics are especially important because a predicted entity must have the correct boundaries and type.
5. **Analysis:** examine differences across entity categories and prediction errors.

## Results from the original notebook

| Test-set metric | BiLSTM | DistilBERT |
| --- | ---: | ---: |
| Token accuracy | 93.55% | 96.42% |
| Entity precision | 73.28% | 84.30% |
| Entity recall | 75.12% | 88.10% |
| **Entity F1** | **74.19%** | **86.16%** |

**Observed difference:** DistilBERT achieves **+11.97 percentage points** in entity F1 over BiLSTM in the original saved notebook outputs.

> **Reproducibility status:** These figures were extracted from the original notebook's saved outputs. They have **not yet been independently reproduced by rerunning the complete training pipeline**. This repository is being prepared for a clean, documented rerun; please do not interpret the table as a verified reproduction.

## Dataset and access

The experiments use **CoNLL-2003**, with 14,041 training sentences, 3,250 validation sentences, and 3,453 test sentences in the original project archive.

**The dataset files are not included in this repository.** CoNLL-2003 contains Reuters-derived text subject to redistribution restrictions. Users must obtain the data through an authorized source and follow the applicable terms. Dataset placement and validation instructions will be added with the reproducible pipeline.

## Code and current verification status

The original notebook has been split into focused Python modules:

- [`src/preprocessing.py`](src/preprocessing.py): subword IOB alignment, datasets and dynamic padding.
- [`src/models.py`](src/models.py): the BiLSTM and pretrained DistilBERT architectures.
- [`src/training.py`](src/training.py): device-safe training, token accuracy and entity-level metrics.
- [`tests/test_pipeline.py`](tests/test_pipeline.py): synthetic examples for alignment, padding and a small BiLSTM training step.
- [`src/validate_data.py`](src/validate_data.py): dataset structure validation.

**Local smoke test (requires PyTorch and NumPy):**

```bash
python -m unittest discover -s tests -v
```

The source modules and tests have been added, but the entire test suite has **not yet been executed and verified in a fresh environment**. The DistilBERT training pipeline and a complete command-line experiment runner are still pending. Do not use the saved results as evidence of a successful reproduction.

## Reproducing the experiment

**In progress.** The original notebook has been audited and needs several corrections before it can be reliably run in a fresh environment, including explicit device configuration, deterministic seeds, consistent CPU/GPU handling, dependency versions, and an undefined plotting reference.

I am separating the project-specific code from course exercises, preserving the original experimental logic, and adding tests and setup instructions. **There is not yet a verified one-command reproduction workflow.**

## Limitations

- This is a comparison on **one NER benchmark**, not evidence of general superiority across NLP tasks.
- The models differ both in architecture and in the use of **pretraining**; the comparison does not isolate architecture alone.
- Saved results may vary on rerun because the original experiment did not fully control randomness or environment versions.
- Full training requires compute resources and an authorized copy of the dataset.

## Tools

Python · PyTorch · Hugging Face Transformers · sequence labeling · entity-level evaluation

## Project provenance

**Author:** Andrea Bianco. This work originated in an ENSAI NLP course. The original notebook notes the use of AI assistance and course reference materials. This repository will contain a cleaned presentation of the project, with reused course material separated or attributed as appropriate.

---

**Current status:** Documentation and reproducibility improvements in progress. The source code and tests will be added after validation.
