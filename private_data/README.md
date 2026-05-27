# Private Data

Put non-public runtime corpora in this folder. Git ignores everything here
except this README, so these files can stay local while the code repository can
be shared publicly.

Expected historical Mingrelian runtime filenames at this directory root:

- `sentence_pairs.tsv`
- `gal.tsv`
- `kk.tsv`
- `context_source.txt`
- `harris.txt`

Optional files used by newer/experimental branches:

- `harris_compact.txt`
- `master-lexicon-mkhedruli.csv`
- `translation_overrides.tsv`
- `eval-datasets/notion-mingrelian-lesson-notes-triples.csv`

Language-specific runtime packs live in subdirectories. The Svan pack uses:

- `svan/sentence_pairs.tsv`
- `svan/gal.tsv`
- `svan/kk.tsv`
- `svan/context_source.txt`
- `svan/tuite.txt`
- `svan/tuite_compact.txt`

See `docs/svan-translation-policy.md` for Svan prompt policy and evaluation
evidence.

The backend checks `ARGO_DATA_DIR` first, then this folder, then
`fastapi_app/data/`.
