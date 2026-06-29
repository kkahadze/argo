# Eval Instructions

- Treat `promptfooconfig.mingrelian-quality-four-way.yaml` as the canonical
  Mingrelian regression eval.
- Keep its private source triples held out from runtime data and prompt
  retrieval.
- Run all four directions after every general translator change.
- Compare both continuous quality and deterministic expected-token coverage.
- Never optimize by adding row-specific overrides, prompt examples, or rules.
- Keep generated datasets and result JSON out of git.
