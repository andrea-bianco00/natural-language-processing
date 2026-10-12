# Results and interpretation

These are aggregate metrics from the **full-dataset October 2026 Colab execution**, not estimates or synthetic examples. The restricted CoNLL-2003 corpus and model checkpoints are **not** included.

## Test-set performance

| Metric | BiLSTM | DistilBERT | Difference (DistilBERT − BiLSTM) |
| --- | ---: | ---: | ---: |
| Token accuracy | 92.70% | 96.36% | +3.65 percentage points |
| Entity precision | 67.41% | 83.22% | +15.81 percentage points |
| Entity recall | 72.66% | 88.51% | +15.85 percentage points |
| **Entity F1** | **69.94%** | **85.78%** | **+15.84 percentage points** |

The strongest improvement is in **entity-level recognition**, where correctly identifying the span and entity type is more demanding than classifying individual tokens. DistilBERT outperformed the BiLSTM on all four recorded test metrics.

### Entity F1 — new Colab run

```text
BiLSTM       69.94%  ████████████████████████████
DistilBERT   85.78%  ██████████████████████████████████
```

The bars are schematic; use the numerical table for precise comparison.

## Original notebook versus cleaned pipeline

| Entity F1 | Original notebook (saved output) | New Colab execution | Difference |
| --- | ---: | ---: | ---: |
| BiLSTM | 74.19% | 69.94% | −4.25 pp |
| DistilBERT | 86.16% | 85.78% | −0.38 pp |

**Do not describe the new results as an exact replication.** In particular, the BiLSTM discrepancy has not been causally explained. The new pipeline sets a seed and uses a documented environment; other implementation or environment differences could contribute.

## Training protocol

| Setting | BiLSTM | DistilBERT |
| --- | --- | --- |
| Architecture | 2-layer bidirectional LSTM | `distilbert-base-cased` encoder + classification head |
| Epochs | 10 | 3 |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-4 | 1e-5 |
| Seed | 42 | 42 |
| Device | Tesla T4 (CUDA) | Tesla T4 (CUDA) |

The dataset splits contained **14,041 train**, **3,250 validation**, and **3,453 test** sentences.

## Limitations that matter

- **Subword-expanded scoring:** The evaluation assigns labels to continuation subwords and calculates `seqeval` entity metrics on those expanded sequences. It is **not** the canonical word-level CoNLL-2003 evaluation.
- **Dataset overlaps:** The locally supplied splits contain some identical token sequences across different splits. See [methodology audit](../METHODOLOGY_AUDIT.md).
- **Comparison scope:** The Transformer is pretrained while the BiLSTM embedding layer is randomly initialized; this comparison does not isolate architecture from pretraining.
- **Single run:** One full-dataset execution per model is recorded; there are no repeated-seed uncertainty estimates.
- **Model selection:** The reported test results are from the final training epoch, not from an independently selected checkpoint.

## Evidence and reproducibility

- [BiLSTM metrics and validation history](bilstm.json)
- [DistilBERT metrics and validation history](distilbert.json)
- [Recorded environment](reproduction_environment.json)
- [Reproduction instructions](../REPRODUCIBILITY.md)
- [Methodology audit](../METHODOLOGY_AUDIT.md)

All claims above are scoped to the recorded runs. The current GitHub Actions workflow validates code and notebook structure; it does not retrain on the restricted dataset.
