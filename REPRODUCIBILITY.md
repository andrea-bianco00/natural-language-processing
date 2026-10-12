# Full-dataset reproduction protocol

## Scope and provenance

This repository is a refactoring of the author's ENSAI Named Entity Recognition notebook. The reference values in `compare_results.py` were transcribed from **saved notebook outputs**, not independently reproduced. The new implementation sets a random seed (42) absent from the original notebook; exact agreement is not expected or guaranteed. The evaluation uses **subword-level aligned BIO tags** for seqeval, matching the original notebook's procedure; this is not necessarily equivalent to standard word-level CoNLL-2003 evaluation.

## Prepare an authorized local dataset

Obtain an authorized copy of the author's original `conll03-iob-pos.json.gz` and put it at `data/raw/conll03-iob-pos.json.gz`. Do not commit the file. It should contain `train` (14,041 sentences), `valid` (3,250), and `test` (3,453), with `tokens` and `tags` arrays.

## Run

Use a Python 3.11 environment with the packages in `requirements.txt`. A CUDA-capable GPU is recommended for DistilBERT.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.validate_data data/raw/conll03-iob-pos.json.gz --original-split-counts
python run_experiment.py --model bilstm --data data/raw/conll03-iob-pos.json.gz
python run_experiment.py --model distilbert --data data/raw/conll03-iob-pos.json.gz
python compare_results.py outputs/bilstm.json
python compare_results.py outputs/distilbert.json
```

The run writes JSON files under the ignored `outputs/` directory. No model weights or private training data are uploaded by these commands.

**Do not** compare a smoke run or a subset of the dataset against the full-dataset notebook numbers: `compare_results.py` rejects incomplete split counts and explicit sample limits.

## Interpreting results

A successful end-to-end run is evidence that the training pipeline executes; comparable entity F1 values provide stronger evidence of experimental reproducibility. Differences may arise from the original unseeded initialization, dependency versions, hardware, numerical nondeterminism, or differences in tokenization/implementation. Report these honestly; do not silently tune on the test split. A rigorous exact reproduction would additionally pin package/model revisions, hardware and preprocessing checksums.

## Status

The synthetic real-pretrained-DistilBERT GitHub Actions smoke test passed on 2026-10-12, as verified in the GitHub Actions UI. **Full CoNLL-2003 training has not yet been run and verified in this refactored repository.**
