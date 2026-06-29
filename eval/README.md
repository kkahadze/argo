# Evaluation

## Canonical Mingrelian Eval

The primary Mingrelian regression gate is the 104-row, four-direction lesson-note
eval in `promptfooconfig.mingrelian-quality-four-way.yaml`. It covers:

- English -> Mingrelian
- Georgian -> Mingrelian
- Mingrelian -> English
- Mingrelian -> Georgian

The source triples are held out from runtime promotion. The tracked builder
reads the private source and generates an ignored Promptfoo CSV:

```sh
python eval/build_mingrelian_quality_dataset.py \
  --source /path/to/private/notion-mingrelian-lesson-notes-triples-product-script.csv
```

Run from the backend root with private runtime data available:

```sh
ARGO_DATA_DIR=/path/to/private_data \
PROMPTFOO_PYTHON=/path/to/python \
npx --yes promptfoo@latest eval \
  -c eval/promptfooconfig.mingrelian-quality-four-way.yaml \
  --env-path /path/to/.env \
  --no-cache --no-share
```

Each translator output is graded two ways in the same run:

- `expected_token_coverage`: deterministic multiset coverage of expected
  target tokens, tolerant of word order, morphology separators, and a
  word-final Mingrelian vowel.
- `translation_quality`: calibrated `gpt-5-mini` semantic score from 0 to
  1. Full quality pass is at least 0.90.

Script/format and normalized reference checks are diagnostics and gates. The
continuous quality score remains the main optimization metric; token coverage
is an independent, interpretable signal.

Do not add these 104 rows, their source family, or paraphrases derived from them
to runtime data. Do not add one-off translation rules that target specific eval
rows.
