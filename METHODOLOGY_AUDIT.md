# Methodological audit — CoNLL-2003 NER

Audit date: 2026-10-12. Scope: original ENSAI notebook `project-2025-Bianco-Andrea.ipynb`, extracted `src/preprocessing.py`, `src/models.py`, `src/training.py`, `run_experiment.py`, and the locally supplied CoNLL-2003 JSON.

## Verified by source inspection

| Aspect | Original notebook | Extracted pipeline | Assessment |
| --- | --- | --- | --- |
| Tag IDs | O=0; B/I LOC=1/2, ORG=3/4, PER=5/6, MISC=7/8 | Same mapping | Matched |
| Subword alignment | First subword retains word tag; subsequent B-X becomes I-X | Same rule | Matched |
| Special tokens | Labels -100 for tokens without word ID | Same | Matched |
| Batch padding | Input padding 0, attention 0, labels -100 | Same; tokenizer pad ID configurable | Matched for DistilBERT |
| Token accuracy | Mask labels == -100 | Same | Matched |
| Entity metrics | seqeval precision/recall/F1 on non--100 positions | Same | Matched |
| BiLSTM | 2-layer bidirectional LSTM, 200-dimensional embeddings and hidden size, dropout 0.3 | Same architecture | Matched by inspection |
| DistilBERT | distilbert-base-cased encoder and dropout classification head | Same model family | Matched by inspection |
| Test evaluation | After training | After training, final epoch | No obvious train/test use in optimization |

**Important metric interpretation:** The original and extracted pipelines both **label all subword pieces**. Entity F1 is therefore computed on subword-expanded tag sequences, not exclusively on original word-level CoNLL tags. This preserves comparability with the original notebook but is not the canonical word-level NER evaluation. Do not label the reported F1 scores as canonical CoNLL-2003 benchmark scores without qualification.

## Original dataset inspection

The locally supplied JSON has 14,041 train, 3,250 validation, and 3,453 test records. Exact **token-sequence** overlap (distinct sequences, ignoring labels) was measured as:

- train vs validation: **129**
- train vs test: **78**
- validation vs test: **25**

These are already present in the supplied splits. This is **not evidence of a pipeline bug**, but the benchmark split is not strictly disjoint by sentence text. The script uses the provided train/valid/test splits separately and does not concatenate them for training. A further assessment of the effect of duplicate text on metrics would require prediction-level evidence.

## Evidence and limitations

New full-dataset Colab outputs are stored under `results/`. They report entity F1 of 0.6993865031 for BiLSTM and 0.8578292578 for DistilBERT; original notebook saved outputs report approximately 0.7419 and 0.8616. **These are different runs, not exact numerical replications.**

This audit is **static code inspection plus local data inspection**, not an independent rerun of both full trainings. Automated unit tests and an end-to-end clean-environment rerun remain separate verification tasks. The original notebook did not fully control random seeds, and the new implementation explicitly seeds the RNG; this alone does not establish the cause of the BiLSTM difference.

## Recommended publication language

“Reproduced the experimental workflow on the complete locally supplied CoNLL-2003 splits using a cleaned PyTorch implementation. Reported entity metrics follow the original notebook's subword-expanded BIO evaluation protocol. Full-run results differ from the saved original notebook metrics; see the comparison and methodological limitations.”

Do not distribute the locally supplied Reuters-derived dataset. Do not claim that test/train token-sequence overlap is zero.
