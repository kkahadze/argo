# Runtime Data

The public repository does not include the full runtime corpora. Keep those
files in the ignored root-level `private_data/` folder, or set `ARGO_DATA_DIR`
to another folder that contains the same filenames.

Expected runtime filenames for this branch:

- `sentence_pairs.tsv`
- `gal.tsv`
- `kk.tsv`
- `context_source.txt`
- `harris.txt`

Optional runtime filenames:

- `harris_compact.txt`
- `master-lexicon-mkhedruli.csv`
- `translation_overrides.tsv`

This directory can hold small public sample fixtures such as
`translation_overrides.tsv`, but do not commit research-derived or otherwise
non-redistributable corpora here.
