# Dataset setup

This project uses the **CoNLL-2003 English Named Entity Recognition** benchmark.

**Dataset files are not redistributed here.** The benchmark includes Reuters-derived text with licensing and redistribution restrictions. Obtain a copy from an authorized source and comply with its terms.

The original ENSAI project used a JSON (or gzip-compressed JSON) file with three top-level keys: `train`, `valid`, and `test`. Each split is a list of sentence records with `tokens` and `tags` fields. The original archive contained 14,041 training sentences, 3,250 validation sentences, and 3,453 test sentences.

The cleaned data loader and an input validation command will be added in the next development stage. Until then, **do not assume that an arbitrary CoNLL-2003 download has the exact JSON format expected by the original notebook**.

For local work, place authorized source data under `data/raw/`; this directory is excluded by `.gitignore`.

Never commit the original `conll03-iob-pos.json` or `conll03-iob-pos.json.gz` files.
