# Svan Translation Policy

## Current Decision

As of 2026-05-27, the live Svan pipeline intentionally uses different prompt
strategies by direction:

| Direction | Retrieval policy | Grammar policy by default |
| --- | --- | --- |
| Svan to Georgian | Search Svan-side fields only; prioritize exact token candidates; cap combined retrieved context at 8,000 characters | Do not include Tuite grammar |
| Georgian to Svan | Preserve the existing high-resource-to-Svan retrieval path | Include the full Tuite grammar |

The policy is directional because the grammar is useful when the model must
produce Svan, but adds a large payload when the model is reading Svan and
producing Georgian.

An explicit caller-provided `grammar_policy` remains authoritative. The
directional default applies only when the caller does not specify a policy.

## Runtime Assets

The Svan language pack selects these files:

| Asset | Purpose |
| --- | --- |
| `private_data/svan/tuite.txt` | Full Svan grammar used by default for Georgian to Svan |
| `private_data/svan/tuite_compact.txt` | Compact-policy asset, when selected explicitly |
| `private_data/svan/kk.tsv` | Svan, Russian, and Georgian lexical entries |
| `private_data/svan/gal.tsv` | Svan and Russian lexical entries |
| `private_data/svan/sentence_pairs.tsv` | Svan and English aligned material |
| `private_data/svan/context_source.txt` | Larger fallback context collection |

At evaluation time, `tuite.txt` contained `334,149` characters. This size is
the main reason grammar selection has a material prompt-cost impact.

## Implemented Behavior

For Svan to Georgian:

1. Dictionary searches match only the Svan field of `sentence_pairs.tsv`,
   `gal.tsv`, and `kk.tsv`. Georgian gloss matches are not treated as Svan
   source evidence.
2. `context_source.txt` matching is limited to `Svan:` lines, while retaining
   the complete matching entry block as context.
3. Exact token-level candidates are placed before fallback snippets.
4. The complete retrieved dictionary section is bounded by
   `ARGO_MAX_RETRIEVAL_CONTEXT_CHARS`, defaulting to `8000`.
5. The default prompt omits the Tuite grammar block.

For Georgian to Svan, none of those source-direction restrictions are applied.
The existing retrieval builder and full-grammar default remain in place.
Svan-to-English and English-to-Svan behavior is also unchanged by this
decision; it was not part of the evaluated comparison.

## Evaluation Evidence

The promotion decision used the clean DoReCo held-out dataset at
`eval/datasets/svan-doreco-heldout-georgian.csv` and the strict assertion
configuration at `eval/assertions/svan-doreco-georgian-strict.yaml`.
Both compared flows loaded the current Tuite grammar assets.

| Direction | Baseline with Tuite | Directional candidate with Tuite | Decision |
| --- | ---: | ---: | --- |
| Svan to Georgian | 7 / 32 | 9 / 32 | Promote directional behavior |
| Georgian to Svan, run 1 | 16 / 32 | 16 / 32 | No demonstrated gain |
| Georgian to Svan, run 2 | 15 / 32 | 13 / 32 | Retain existing behavior |
| Georgian to Svan, aggregate | 31 / 64 | 29 / 64 | Do not promote candidate behavior |

Prompt measurements from the matched Svan-to-Georgian run:

| Flow | Average prompt characters | Average retrieved dictionary characters | Grammar included rows |
| --- | ---: | ---: | ---: |
| Baseline with Tuite | 351,318.5 | 15,874.0 | 32 / 32 |
| Directional candidate with Tuite | 3,634.3 | 2,555.2 | 0 / 32 |

Prompt measurements from Georgian-to-Svan run 1:

| Flow | Average prompt characters | Average retrieved dictionary characters | Grammar included rows |
| --- | ---: | ---: | ---: |
| Baseline with Tuite | 347,570.4 | 12,132.6 | 32 / 32 |
| Directional candidate with Tuite | 340,584.4 | 5,146.6 | 32 / 32 |

The Georgian-to-Svan prompt remains large because it deliberately includes the
full Tuite grammar; its directional retrieval experiment did not establish a
quality improvement.

## Result Artifacts

Historical baseline and candidate artifacts used for the decision:

- `eval/results.svan-doreco-heldout.svan-to-georgian.baseline-tuite.strict-v3.json`
- `../experiments/argo-svan-p0/eval/results.svan-doreco-heldout.svan-to-georgian.p0-directional-tuite.strict-v3.json`
- `eval/results.svan-doreco-heldout.georgian-to-svan.baseline-tuite-temp0-rerun.json`
- `eval/results.svan-doreco-heldout.georgian-to-svan.baseline-tuite-temp0-rerun2.json`
- `../experiments/argo-svan-p0/eval/results.svan-doreco-heldout.georgian-to-svan.p0-directional-tuite-temp0.json`
- `../experiments/argo-svan-p0/eval/results.svan-doreco-heldout.georgian-to-svan.p0-directional-tuite-temp0-rerun2.json`

The `baseline` Svan-to-Georgian artifact is historical. Rerunning that old
configuration against current live code will execute the promoted behavior,
not reconstruct the pre-promotion implementation.

## Rerunning The Current Flow

Validate and run the promoted Svan-to-Georgian evaluation:

```bash
promptfoo validate config -c eval/promptfooconfig.svan-doreco-heldout.svan-to-georgian.directional-v1.strict-v3.yaml
promptfoo eval -c eval/promptfooconfig.svan-doreco-heldout.svan-to-georgian.directional-v1.strict-v3.yaml --no-cache --no-share -o eval/results.svan-doreco-heldout.svan-to-georgian.directional-v1.strict-v3.json
```

Run the retained Georgian-to-Svan behavior:

```bash
promptfoo validate config -c eval/promptfooconfig.svan-doreco-heldout.georgian-to-svan.baseline-temp0.yaml
promptfoo eval -c eval/promptfooconfig.svan-doreco-heldout.georgian-to-svan.baseline-temp0.yaml --no-cache --no-share -o eval/results.svan-doreco-heldout.georgian-to-svan.current-tuite-temp0.json
```

The provider requires `OPENAI_API_KEY` in the environment for these runs.

## Regression Coverage

`tests/test_svan_directional_prompting.py` protects the accepted boundary:

- Svan input does not retrieve entries solely because Georgian gloss text matches.
- Context-source fallback matches Svan source lines rather than Georgian target lines.
- Svan-to-Georgian omits grammar by default and bounds retrieved context.
- The translation pipeline applies the Svan-source grammar default.
- Georgian-to-Svan keeps its existing full-grammar and retrieval behavior.
- Svan-to-English keeps its existing full-grammar default.

Run it with:

```bash
PYTHONPATH="$PWD" python3 -m unittest discover -s tests -p 'test_svan_directional_prompting.py' -v
```
