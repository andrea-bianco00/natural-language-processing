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

> **Provenance:** These are historical values from the original notebook, not the results of the new Colab execution documented below.

## New full-dataset Colab execution (October 2026)

These are **newly recorded test-set results**, not the original notebook's saved figures. Both runs used the complete local dataset (14,041 train / 3,250 validation / 3,453 test sentences), seed 42, and a Tesla T4 GPU.

| Test-set metric | BiLSTM | DistilBERT |
| --- | ---: | ---: |
| Token accuracy | 92.70% | 96.36% |
| Entity precision | 67.41% | 83.22% |
| Entity recall | 72.66% | 88.51% |
| **Entity F1** | **69.94%** | **85.78%** |
| Epochs | 10 | 3 |

DistilBERT exceeded BiLSTM by **15.84 percentage points** in entity F1 in this run. The original notebook reports 74.19% (BiLSTM) and 86.16% (DistilBERT); those historical results are **not identical** to the new runs. In particular, the BiLSTM difference needs methodological scrutiny before claiming faithful reproduction.

**Execution evidence:** [BiLSTM JSON](results/bilstm.json) · [DistilBERT JSON](results/distilbert.json) · [Environment manifest](results/reproduction_environment.json). These files contain aggregate metrics and training history, not the restricted dataset or model weights. They are evidence of the reported execution, not proof that every methodological choice is correct.

**Methodology audit:** Source-level checks of BIO alignment, ignored labels and seqeval evaluation are documented in [METHODOLOGY_AUDIT.md](METHODOLOGY_AUDIT.md). The original data contain identical token sequences across splits, and reported entity F1 is measured on subword-expanded tags. A clean independent end-to-end rerun and word-level evaluation remain open.

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

**Local automated tests (requires PyTorch and NumPy; seqeval for metric tests):**

```bash
python -m unittest discover -s tests -v
```

The earlier Colab session reported 13 passing tests. Additional methodology tests have since been added, but the expanded suite has **not yet been independently verified in a fresh environment**. The complete command-line runner is present and has produced the recorded Colab artifacts; the notebook revision still needs its own end-to-end rerun.

## Running the extracted experiment

The command-line runner is available, and full-dataset Colab artifacts are recorded; a separate clean-environment rerun has not yet been verified. It requires an authorized copy of CoNLL-2003 and downloads pretrained tokenizer/encoder files from Hugging Face.

From the repository root, with Python 3.10 or 3.11:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.validate_data data/raw/conll03-iob-pos.json.gz --original-split-counts
python run_experiment.py --model bilstm --data data/raw/conll03-iob-pos.json.gz
python run_experiment.py --model distilbert --data data/raw/conll03-iob-pos.json.gz
```

The two commands write JSON results to `outputs/`, which is excluded from version control. The default training configuration mirrors the original notebook (BiLSTM: 10 epochs, AdamW 5e-4; DistilBERT: 3 epochs, AdamW 1e-5; batch size 32). The added seed is a reproducibility improvement, so new results may differ from the notebook's saved results.

**Known limitations:** dependency versions are provisional; pretrained-model downloads and full training have not been independently retested in a separate clean environment. This runner does not yet implement resume/checkpoint support or fully deterministic GPU execution.

## Google Colab notebook

The cleaned [Colab reproduction notebook](notebooks/ENSAI_NER_Colab_Reproduction.ipynb) runs the repository's existing Python modules instead of copying model and evaluation logic into notebook cells. It has no embedded training outputs or restricted data. A current repository ZIP and an authorized original `Project.zip` are required for a rerun. The notebook structure was checked, but its newly committed revision has **not** been independently executed end-to-end; the recorded full-dataset runs are documented separately in `results/`.

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

**Current status:** Full-dataset Colab result artifacts recorded. Methodological audit and notebook publication are still pending.


For a full-dataset reproduction protocol and comparison against the saved notebook metrics, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
