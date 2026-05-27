# Argo - Shared Translation Backend

FastAPI backend for the unified Margo translator. It provides translation
services for Mingrelian, Tsova-Tush / Bats, and Svan, bridged through Georgian
and English, using LLM-augmented dictionary lookups.

## Features

- **Multi-language Translation**: Translate Mingrelian, Tsova-Tush / Bats, or Svan through Georgian and English
- **Multiple LLM Providers**: Support for OpenAI (GPT-5.4 family, GPT-5.2), Anthropic (Claude), and Google (Gemini)
- **Smart Dictionary Lookups**: Standalone word matching with short-circuit optimization
- **Google Translate Bridge**: Instant translations via high-resource language bridging
- **Configurable Logging**: Console logging by default, with opt-in file logs; full prompt/response traces require DEBUG file logging
- **Single API Call**: Optimized to use only one LLM call per translation

## Quick Start

### 1. Install Dependencies

```bash
bash run_local.sh
```

The helper script creates a virtualenv, installs dependencies, and starts the server.

To install manually without starting the server:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r fastapi_app/requirements.txt
```

### Private Runtime Data

The public repository does not include the full dictionary/reference corpora.
For local full-quality translation, put the private files in `private_data/`
at the repo root, or set `ARGO_DATA_DIR` to another folder with the same
language-pack layout:

```text
private_data/
├── sentence_pairs.tsv
├── gal.tsv
├── kk.tsv
├── context_source.txt
├── harris.txt
├── tsova_tush/
│   ├── sentence_pairs.tsv
│   ├── gal.tsv
│   ├── kk.tsv
│   ├── context_source.txt
│   └── translation_overrides.tsv
└── svan/
    ├── sentence_pairs.tsv
    ├── gal.tsv
    ├── kk.tsv
    ├── context_source.txt
    ├── tuite.txt
    └── tuite_compact.txt
```

Optional private files include `harris_compact.txt`,
`master-lexicon-mkhedruli.csv`, and local `translation_overrides.tsv`.

`private_data/` is ignored by git, so the code can be public while the corpora
stay local/private. The root files are the historical Mingrelian pack; the
Tsova-Tush / Bats and Svan packs live in their language-specific subfolders.

### Share A Complete Private Bundle

To send someone the code plus your private data in one zip:

```bash
bash scripts/build_share_bundle.sh
```

Send `share/argo-share.zip`. The recipient only needs:

```bash
unzip argo-share.zip
cd argo-share/argo
bash run_local.sh
```

If your private files live somewhere else, run:

```bash
ARGO_PRIVATE_DATA_DIR=/path/to/private/data bash scripts/build_share_bundle.sh
```

### 2. Set Up Environment Variables

Create a `.env` file in the argo root directory:

```bash
# OpenAI (optional, also used for server-side fallback if the client omits api_key)
OPENAI_API_KEY=your_openai_key_here

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google Gemini (optional)
GEMINI_API_KEY=your_gemini_key_here

# Default provider (optional, defaults to openai)
LLM_PROVIDER=openai  # or "anthropic" or "gemini"

# Default model (optional)
LLM_MODEL=gpt-5.5  # or gpt-5.4-nano, gpt-5.4-mini, gpt-5.4, claude-sonnet-4-5-20250929, gemini-3.1-flash-lite-preview, etc.

# Logging (optional, defaults to INFO console logs only)
LOG_LEVEL=INFO
LOG_TO_FILE=false  # set true with LOG_LEVEL=DEBUG for full prompt/response file logs
```

### 3. Run the Server

```bash
# Using the run script
bash run_local.sh

# Or manually
source venv/bin/activate
uvicorn fastapi_app.api:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Translation Endpoint

**POST** `/chat`

```json
{
  "prompt": "მა",
  "api_key": "your_api_key",
  "source_language": "mingrelian",
  "target_language": "english",
  "provider": "openai",
  "model": "gpt-5.5",
  "reasoning_effort": "none"
}
```

**Parameters:**
- `prompt` (string, required): Text to translate
- `api_key` (string, optional): API key for the LLM provider (if omitted, the backend uses the configured server-side key for the selected provider)
- `source_language` (string, optional): Source language - "mingrelian", "tsova_tush", "svan", "georgian", or "english" (default: "mingrelian" from `src/provider_config.py`)
- `target_language` (string, optional): Target language - "mingrelian", "tsova_tush", "svan", "georgian", or "english" (default: "english" from `src/provider_config.py`)
- `provider` (string, optional): LLM provider - "openai", "anthropic", or "gemini" (reads from env if not specified)
- `model` (string, optional): Model name (reads from env if not specified, then uses provider default)
- `reasoning_effort` (string, optional): OpenAI reasoning effort for GPT-5 family models, such as `"none"` or `"low"`

**Response (SSE Stream):**

The endpoint streams server-sent events. The final event payload looks like:

```json
{
  "result": {
    "source_text": "მა",
    "target_text": "I",
    "translated_text": "I",
    "romanized_text": "",
    "source_language": "mingrelian",
    "target_language": "english",
    "mingrelian_latinized": "",
    "mingrelian_mkhedruli": "მა",
    "georgian": "",
    "english": "I",
    "full_response": "Exact lexicon match:\nI"
  }
}
```

## Supported Models

Provider names, language names/defaults, provider model defaults, provider reasoning defaults, provider API key environment variables, and the server-side-key model allowlist are defined in `src/provider_config.py`. User-provided API keys may still request other provider-supported model names.

### OpenAI
- `gpt-5.5` (default)
- `gpt-5.4-nano`
- `gpt-5.4-mini`
- `gpt-5.4`
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-5.2`

### Anthropic
- `claude-sonnet-4-5-20250929` (default)
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`

### Google
- `gemini-3.1-flash-lite-preview` (default)
- `gemini-2.0-flash-exp`

## Architecture

### Translation Pipeline

The system uses an optimized single-call translation approach with multiple fallback strategies:

```
User Input
    ↓
┌─────────────────────────────────────────┐
│ 1. Exact Dictionary Match Check         │
│    - Check sentence_pairs.tsv           │
│    - Check gal.tsv (Russian)            │
│    - Check kk.tsv (Russian/Georgian)    │
│    → If found: INSTANT RETURN (no LLM)  │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Google Translate Bridge              │
│    TO low-resource language:             │
│      - Translate input → Russian/etc     │
│      - Search dicts for Mingrelian       │
│    FROM low-resource language:           │
│      - Search for any high-resource lang │
│      - Google Translate → target lang    │
│    → If found: INSTANT RETURN (no LLM)  │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Direct Google Translate              │
│    Georgian ↔ English only              │
│    → If applicable: INSTANT RETURN      │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. LLM-Based Translation                │
│    - Build context from dictionaries    │
│    - Standalone word matching (priority)│
│    - Construct prompt with examples     │
│    - Single LLM API call                │
│    - Extract translation from response  │
└─────────────┬───────────────────────────┘
                   ↓
           Final Translation
```

### Dictionary Data Sources

Private runtime corpora are loaded from `ARGO_DATA_DIR` when set, then
`private_data/`, then `fastapi_app/data/` for any public/sample fixtures:

1. **sentence_pairs.tsv** - English-low-resource parallel sentences
2. **gal.tsv** - Russian-low-resource dictionary
3. **kk.tsv** - Low-resource-Russian-Georgian dictionary (4 columns: word, IPA, Russian, Georgian)
4. **context_source.txt** - Large fallback reference used for LLM context, not extractive lookups
5. **harris.txt** - Full grammar reference
6. **harris_compact.txt** - Compact grammar reference for prompt-size experiments
7. **master-lexicon-mkhedruli.csv** - Optional master lexicon for exact Mingrelian-English candidates
8. **translation_overrides.tsv** - Small pair-specific exact overrides; public sample overrides may live in `fastapi_app/data/`

The Svan pack uses `tuite.txt` and `tuite_compact.txt` rather than the
historical Mingrelian `harris*.txt` assets. Its directional prompt policy and
evaluation evidence are documented in `docs/svan-translation-policy.md`.

### Optimization Strategies

1. **Standalone Word Matching**: Prioritizes exact word matches (surrounded by spaces) over substring matches to reduce irrelevant context

2. **Short-Circuit for Extractive Dictionaries**: If a standalone match is found in sentence_pairs.tsv, gal.tsv, or kk.tsv, the system skips searching context_source.txt

3. **Instant Lookup**: If an exact match for the full input is found in extractive dictionaries, translation is returned instantly without any LLM call

4. **Google Translate Bridge**: Leverages Google Translate for high-resource languages to find Mingrelian translations or intermediate translations without LLM calls

## Logging

Console logs are enabled by default. File logging is opt-in so tests and eval runs do not create local log files unless you ask for them.

Set `LOG_TO_FILE=true` to save logs to the `logs/` directory:

- `translator_YYYYMMDD.log` - Logs at the configured `LOG_LEVEL` and above
- `errors_YYYYMMDD.log` - Error logs only

**At `LOG_LEVEL=INFO`, log files include:**
- Translation requests with language pairs
- Truncated prompt/response previews
- Extracted translations
- Instant lookup results
- Error details with context

Full prompts sent to LLMs and full LLM responses are DEBUG entries. To write them to `translator_YYYYMMDD.log`, set both `LOG_TO_FILE=true` and `LOG_LEVEL=DEBUG`.

**Environment variable:**
```bash
LOG_LEVEL=DEBUG   # Include DEBUG details such as full prompts/responses
LOG_TO_FILE=true  # Write logs/translator_YYYYMMDD.log and logs/errors_YYYYMMDD.log
```

## Translation Analytics

The backend can optionally write each translation request to Supabase for later analysis. These writes are scheduled in the background so they do not block the streamed response to the user.

Suggested fields captured:
- source text and translated output
- language pair
- provider and model
- duration in milliseconds
- whether the request used a user-supplied API key
- optional anonymous `visitor_id` generated by the browser
- queryable translation path fields such as `translation_path`, `used_llm`, `used_evidence_bundle`, `used_dictionary_entries`, and `used_grammar`
- prompt metrics when the request reached prompt construction
- error details when a request fails

### Setup

1. Create the analytics table by running the SQL in `supabase/translation_events.sql` in your Supabase SQL editor.
2. Add the following environment variables:

```bash
SUPABASE_LOGGING_ENABLED=true
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_API_KEY=your_supabase_runtime_key
SUPABASE_LOGGING_TABLE=translation_events
SUPABASE_LOGGING_TIMEOUT_SECONDS=2.5
```

For production, prefer a service-role key via `SUPABASE_SERVICE_ROLE_KEY`. If you want to start quickly with a publishable key, keep the insert-only RLS policy from the provided SQL so clients cannot read the table through that key.

## Project Structure

```
argo/
├── fastapi_app/
│   ├── api.py              # FastAPI application & /chat endpoint
│   ├── requirements.txt    # Python dependencies
│   └── data/               # Public data docs/sample fixtures only
├── src/
│   ├── single_call_translator.py  # Backward-compatible translator facade
│   ├── translator/                # Translation data, lookup, prompts, extraction, pipeline
│   ├── dictionary_store.py        # Dictionary loading and lookup indexes
│   ├── provider_config.py         # Provider, model, language defaults and allowlists
│   ├── llm_client.py              # LLM provider abstraction
│   └── logger.py                  # Logging configuration
├── eval/                   # Promptfoo configs and evaluation helpers
├── private_data/           # Ignored private corpora for local/full-quality runs
├── scripts/
│   └── build_share_bundle.sh # Build code+private-data zip for sharing
├── supabase/
│   └── translation_events.sql
├── logs/                   # Optional log files when LOG_TO_FILE=true (gitignored)
├── tests/                  # Unit tests with synthetic fixtures
├── venv/                   # Virtual environment (gitignored)
├── .env                    # Environment variables (gitignored)
├── env.example             # Example environment file
├── render.yaml             # Render Blueprint deployment config
├── run_local.sh            # Local development script
└── README.md               # This file
```

## Development

Promptfoo evaluations use `eval/provider.py`. If an eval config specifies a provider but omits `model`, the provider uses that provider's default model from `src/provider_config.py`; if the provider is also omitted, the eval-specific default provider remains Gemini.

Grammar prompt policy can be compared without LLM calls:

```bash
python3 eval/run_grammar_policy_eval.py --measure-only --policies full,compact,none
```

To run the lesson-note promptfoo evals head-to-head, omit `--measure-only`:

```bash
python3 eval/run_grammar_policy_eval.py --policies full,compact,none --repeat 1
```

### Running Tests

Run the checked-in unit tests:

```bash
python3 -m unittest discover -s tests
```

For a quick verification pass:

```bash
python3 -m py_compile fastapi_app/api.py src/*.py src/translator/*.py eval/*.py tests/*.py
python3 -m unittest tests.test_provider_config
```

### Adding New Dictionary Data

1. Add private TSV/TXT/CSV files to `private_data/` or an `ARGO_DATA_DIR` folder
2. Update loaders/search functions in `src/translator/data.py`, `src/translator/lookup.py`, and `src/dictionary_store.py`
3. Add to prompt construction as needed
4. Commit only docs, code, and small public sample fixtures unless the data has clear redistribution rights

### Debugging

- Set `LOG_LEVEL=DEBUG` and `LOG_TO_FILE=true` in `.env` to write full prompt/response logs
- Check `logs/translator_YYYYMMDD.log` for full request/response traces when `LOG_TO_FILE=true` and `LOG_LEVEL=DEBUG`
- Check `logs/errors_YYYYMMDD.log` for error details

## Deployment

Render is the documented deployment path for this backend. The repository includes
`render.yaml` so the service can be created or synced from a Render Blueprint.

**Render configuration:**
- Runtime: Python
- Build command: `pip install -r fastapi_app/requirements.txt`
- Start command: `uvicorn fastapi_app.api:app --host 0.0.0.0 --port $PORT`
- Service name: `argo-translator`

### Deploying on Render

1. In Render, create a Blueprint or Web Service from this repository.
2. Let Render read `render.yaml`, or manually use the build and start commands above.
3. Set secrets in the Render dashboard. Do not commit secrets to the repository.

The default Render config uses `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-5.5`,
and prompts for `OPENAI_API_KEY` as a secret value. If you switch providers,
set the matching provider and model variables plus the matching secret key
(`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`) in Render.

Optional Supabase analytics remain disabled by default. To enable them, set
`SUPABASE_LOGGING_ENABLED=true` and provide `SUPABASE_URL` plus a Supabase API
key in Render.

Render builds and deploys from the linked repository or Blueprint.

## Contributing

When contributing:

1. Keep all LLM calls centralized in `src/llm_client.py`
2. Keep provider, model, language defaults, API key environment variable names, and server-key allowlists centralized in `src/provider_config.py`
3. Add comprehensive logging for debugging
4. Keep private dictionary/reference data out of git; use `private_data/` or `ARGO_DATA_DIR`
5. Test with multiple LLM providers
6. Update this README with any architectural changes

## License

MIT License
