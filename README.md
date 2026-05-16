# Argo - Mingrelian Translation Backend

FastAPI backend for the Mingrelian translation application. Provides translation services between Mingrelian, Georgian, and English using LLM-augmented dictionary lookups.

## Features

- **Multi-directional Translation**: Translate between any pair of Mingrelian, Georgian, and English
- **Multiple LLM Providers**: Support for OpenAI (GPT-5.4 family, GPT-5.2), Anthropic (Claude), and Google (Gemini)
- **Smart Dictionary Lookups**: Standalone word matching with short-circuit optimization
- **Google Translate Bridge**: Instant translations via high-resource language bridging
- **Configurable Logging**: Console logging by default, with opt-in file logs for debugging prompts, responses, and errors
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
LLM_MODEL=gpt-5.4-nano  # or gpt-5.4-mini, gpt-5.4, claude-sonnet-4-5-20250929, gemini-3.1-flash-lite-preview, etc.

# Logging level (optional, defaults to INFO)
LOG_LEVEL=INFO  # or DEBUG for more verbose logs
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
  "model": "gpt-5.4-nano"
}
```

**Parameters:**
- `prompt` (string, required): Text to translate
- `api_key` (string, optional): API key for the LLM provider (if omitted, the backend uses the configured server-side key for the selected provider)
- `source_language` (string, optional): Source language - "mingrelian", "georgian", or "english" (default: "mingrelian")
- `target_language` (string, optional): Target language - "mingrelian", "georgian", or "english" (default: "english")
- `provider` (string, optional): LLM provider - "openai", "anthropic", or "gemini" (reads from env if not specified)
- `model` (string, optional): Model name (reads from env if not specified, then uses provider default)

**Response (SSE Stream):**

The endpoint streams server-sent events. The final event payload looks like:

```json
{
  "result": {
    "source_text": "მა",
    "target_text": "I",
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

### OpenAI
- `gpt-5.4-nano` (default)
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
│    TO Mingrelian:                        │
│      - Translate input → Russian/etc     │
│      - Search dicts for Mingrelian       │
│    FROM Mingrelian:                      │
│      - Search for any high-resource lang │
│      - Google Translate → target lang    │
│    → If found: INSTANT RETURN (no LLM)  │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Direct Google Translate              │
│    Georgian ↔ English (no Mingrelian)   │
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

Located in `fastapi_app/data/`:

1. **sentence_pairs.tsv** - English-Mingrelian parallel sentences
2. **gal.tsv** - Russian-Mingrelian dictionary
3. **kk.tsv** - Mingrelian-Russian-Georgian dictionary (4 columns: word, IPA, Russian, Georgian)
4. **context_source.txt** - Large fallback reference used for LLM context, not extractive lookups
5. **harris.txt** - Grammar reference

### Optimization Strategies

1. **Standalone Word Matching**: Prioritizes exact word matches (surrounded by spaces) over substring matches to reduce irrelevant context

2. **Short-Circuit for Extractive Dictionaries**: If a standalone match is found in sentence_pairs.tsv, gal.tsv, or kk.tsv, the system skips searching context_source.txt

3. **Instant Lookup**: If an exact match for the full input is found in extractive dictionaries, translation is returned instantly without any LLM call

4. **Google Translate Bridge**: Leverages Google Translate for high-resource languages to find Mingrelian translations or intermediate translations without LLM calls

## Logging

Console logs are enabled by default. File logging is opt-in so tests and eval runs do not create local log files unless you ask for them.

Set `LOG_TO_FILE=true` to save logs to the `logs/` directory:

- `translator_YYYYMMDD.log` - All logs (DEBUG level and above)
- `errors_YYYYMMDD.log` - Error logs only

**Log files include:**
- Translation requests with language pairs
- Full prompts sent to LLMs
- Full LLM responses
- Extracted translations
- Instant lookup results
- Error details with context

**Environment variable:**
```bash
LOG_LEVEL=DEBUG   # Capture DEBUG details such as full prompts/responses
LOG_TO_FILE=true  # Enable logs/translator_YYYYMMDD.log and logs/errors_YYYYMMDD.log
```

## Translation Analytics

The backend can optionally write each translation request to Supabase for later analysis. These writes are scheduled in the background so they do not block the streamed response to the user.

Suggested fields captured:
- source text and translated output
- language pair
- provider and model
- duration in milliseconds
- whether the request used a user-supplied API key
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
│   └── data/               # Dictionary and reference data
│       ├── sentence_pairs.tsv
│       ├── gal.tsv
│       ├── kk.tsv
│       ├── context_source.txt
│       └── harris.txt
├── src/
│   ├── single_call_translator.py  # Core translation logic
│   ├── llm_client.py              # LLM provider abstraction
│   └── logger.py                  # Logging configuration
├── eval/                   # Promptfoo configs and evaluation helpers
├── supabase/
│   └── translation_events.sql
├── logs/                   # Optional log files when LOG_TO_FILE=true (gitignored)
├── venv/                   # Virtual environment (gitignored)
├── .env                    # Environment variables (gitignored)
├── env.example             # Example environment file
├── run_local.sh            # Local development script
└── README.md               # This file
```

## Development

### Running Tests

There is no automated unit test suite checked into this repo yet.

For a quick verification pass:

```bash
python3 -m py_compile fastapi_app/api.py src/*.py eval/provider.py
```

### Adding New Dictionary Data

1. Add TSV/TXT files to `fastapi_app/data/`
2. Update search functions in `src/single_call_translator.py`
3. Add to prompt construction as needed

### Debugging

- Set `LOG_LEVEL=DEBUG` and `LOG_TO_FILE=true` in `.env` to write full prompt/response logs
- Check `logs/translator_YYYYMMDD.log` for full request/response traces
- Check `logs/errors_YYYYMMDD.log` for error details

## Deployment

This backend is designed to be deployed on platforms like Render, Heroku, or Railway.

**Key files for deployment:**
- `fastapi_app/requirements.txt` - Dependencies
- `fastapi_app/api.py` - Entry point
- Environment variables must be set in the platform's dashboard

**Render example:**
- Build Command: `pip install -r fastapi_app/requirements.txt`
- Start Command: `uvicorn fastapi_app.api:app --host 0.0.0.0 --port $PORT`

## Contributing

When contributing:

1. Keep all LLM calls centralized in `src/llm_client.py`
2. Add comprehensive logging for debugging
3. Update dictionary data in `fastapi_app/data/` as needed
4. Test with multiple LLM providers
5. Update this README with any architectural changes

## License

MIT License
