# Argo Backend

FastAPI backend for the Mingrelian translator. It serves `POST /chat` as an SSE stream and tries deterministic translation paths before using an LLM.

## Working Rules

- Check `git status --short --branch` before editing. This repo often has active worktrees and local edits; do not revert user changes.
- Never read, print, commit, or move secrets from `.env`.
- Keep private corpora out of git. Full runtime data belongs in `ARGO_DATA_DIR` or `private_data/`; `fastapi_app/data/` is for public docs/sample fixtures.
- Treat Georgian, Mingrelian, and Mkhedruli text as UTF-8 data. Preserve exact spellings in fixtures and overrides.
- Keep backend/frontend API contract changes synchronized with `mkhedruli-megruli`.

## Key Files

- `fastapi_app/api.py`: FastAPI app, CORS, `/chat`, SSE formatting, request validation, lazy LLM credentials, `visitor_id` normalization.
- `src/single_call_translator.py`: Backward-compatible facade for legacy imports and tests.
- `src/translator/`: Translation implementation split by responsibility:
  - `data.py`: data-file resolution and cached loaders
  - `lookup.py`: exact matches, dictionary searches, Google Translate bridge
  - `prompts.py`: prompt construction and grammar-policy handling
  - `pipeline.py`: end-to-end translation path selection
  - `extraction.py`: final translation extraction
- `src/dictionary_store.py`: indexed dictionary/override loading.
- `src/provider_config.py`: provider defaults, valid languages, model aliases, reasoning defaults, server-key allowlist.
- `src/llm_client.py`: OpenAI, Anthropic, and Gemini API wrapper. Keep provider calls centralized here.
- `src/translation_analytics.py`: Supabase analytics payloads, including `translation_path`, `used_llm`, `used_evidence_bundle`, dictionary/grammar flags, and timing/count fields.
- `supabase/translation_events.sql`: Analytics table/RLS setup.
- `render.yaml`: Render deployment config.
- `eval/`: Promptfoo configs and evaluation scripts.
- `tests/`: Unit tests with synthetic fixtures.

## Runtime Data

Data lookup order is:

1. `ARGO_DATA_DIR`
2. `private_data/`
3. `fastapi_app/data/`

Recognized source files include:

- `sentence_pairs.tsv`: Mingrelian-English sentence/phrase pairs
- `gal.tsv`: Russian-Mingrelian dictionary
- `kk.tsv`: Mingrelian, IPA, Russian definition, Georgian definition
- `context_source.txt`: large fallback context corpus
- `harris.txt`: full grammar reference
- `harris_compact.txt`: compact grammar reference
- `master-lexicon-mkhedruli.csv`: exact Mingrelian-English candidates
- `translation_overrides.tsv`: pair-specific exact overrides

When adding data behavior, update loaders/search in `src/translator/data.py`, `src/translator/lookup.py`, and `src/dictionary_store.py`; update evals/tests if ranking or exact-match behavior changes.

## API Contract

`POST /chat` accepts:

- `prompt`
- optional `api_key`
- `source_language`: `mingrelian`, `georgian`, or `english`
- `target_language`: `mingrelian`, `georgian`, or `english`
- `provider`: `openai`, `anthropic`, or `gemini`
- `model`
- optional `reasoning_effort` for GPT-5 family models
- optional analytics-only `visitor_id`

The response is `text/event-stream`. The frontend expects the final SSE event to contain `data: {"result": ...}` with legacy fields: `target_text`, `mingrelian_latinized`, `mingrelian_mkhedruli`, `georgian`, `english`, and `full_response`.

## Provider Defaults

Provider/model defaults live in `src/provider_config.py`.

- Default provider: `openai`
- Default OpenAI model: `gpt-5.5`
- Default reasoning for `gpt-5.5`: `none`
- Default Anthropic model: `claude-sonnet-4-5-20250929`
- Default Gemini model: `gemini-3.1-flash-lite`
- Server-side keys are limited by `SERVER_KEY_MODELS`; user-provided keys may request other provider-supported models.

If model names, aliases, reasoning defaults, or server-key allowlists change, update the frontend model list and docs too.

## Local Commands

Start backend:

```bash
bash run_local.sh
```

Manual run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r fastapi_app/requirements.txt
uvicorn fastapi_app.api:app --reload --host 127.0.0.1 --port 8000
```

Verify:

```bash
python3 -m py_compile fastapi_app/api.py src/*.py src/translator/*.py eval/*.py tests/*.py
python3 -m unittest discover -s tests
```

Focused useful tests:

```bash
python3 -m unittest tests.test_translation_analytics tests.test_provider_config tests.test_api_lazy_credentials
python3 -m unittest tests.test_dictionary_store tests.test_dictionary_loaders tests.test_grammar_policy
```

## Analytics

Supabase logging is controlled by:

- `SUPABASE_LOGGING_ENABLED`
- `SUPABASE_URL`
- `SUPABASE_API_KEY` or `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_LOGGING_TABLE`
- `SUPABASE_LOGGING_TIMEOUT_SECONDS`

Keep RLS enabled on `translation_events` with insert-only policies for public clients. If path tracking changes, update both `src/translation_analytics.py` and `supabase/translation_events.sql`, then apply/verify the live schema.

## Logging And Evals

- Console logging is always enabled.
- File logs are opt-in via `LOG_TO_FILE=true`.
- Full prompt/response traces require `LOG_LEVEL=DEBUG` and file logging; treat logs as sensitive local diagnostics.
- `ARGO_GRAMMAR_POLICY` accepts `full`, `compact`, or `none`.
- Compare grammar policies with `python3 eval/run_grammar_policy_eval.py --measure-only --policies full,compact,none`.
- Promptfoo provider behavior lives in `eval/provider.py`.
