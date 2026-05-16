# Private Data

Put non-public runtime corpora in this folder. Git ignores everything here
except this README, so these files can stay local while the code repository can
be shared publicly.

Expected runtime filenames for the current backend:

- `sentence_pairs.tsv`
- `gal.tsv`
- `kk.tsv`
- `context_source.txt`
- `harris.txt`
- `harris_compact.txt`

Optional files used by newer/experimental branches:

- `master-lexicon-mkhedruli.csv`
- `translation_overrides.tsv`
- `eval-datasets/notion-mingrelian-lesson-notes-triples.csv`

The backend checks `ARGO_DATA_DIR` first, then this folder, then
`fastapi_app/data/`.
