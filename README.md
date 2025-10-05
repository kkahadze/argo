# Argo - Mingrelian Language Translation Tool

A powerful tool for translating Mingrelian text with support for multiple LLM providers (OpenAI GPT and Anthropic Claude).

## Features

- **Multi-LLM Support**: Switch between OpenAI (GPT) and Anthropic (Claude) seamlessly
- **Unified API**: All LLM calls centralized through a single abstraction layer
- **FastAPI Web Interface**: RESTful API for integration with other applications
- **Command Line Interface**: Direct script execution for quick translations
- **Flexible Configuration**: Environment-based provider and model selection

## Quick Start

### 1. Install Dependencies

```bash
# For OpenAI (GPT)
pip install openai

# For Anthropic (Claude)
pip install anthropic

# Or install both
pip install openai anthropic
```

### 2. Set Up Environment Variables

Create a `.env` file:

```bash
# Choose your provider (default: openai)
LLM_PROVIDER=openai  # or "anthropic"

# OpenAI configuration
OPENAI_API_KEY=your_openai_key_here
LLM_MODEL=gpt-4o  # or gpt-4o-mini, gpt-3.5-turbo, etc.

# Anthropic configuration (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here
# LLM_MODEL=claude-sonnet-4-5-20250929  # or claude-3-5-sonnet-20241022, etc.

# Optional: specify a different model for long-context operations
LLM_LONG_CONTEXT_MODEL=gpt-4o  # or claude-sonnet-4-5-20250929
```

### 3. Run the Translation Tool

```bash
# Command line interface
python3 ./src/prompt.py

# Or start the web API
python3 -m fastapi_app.api
```

## Usage

### Command Line

```bash
# Use default provider (from .env)
python3 ./src/prompt.py

# Test with specific providers
LLM_PROVIDER=openai python3 ./src/prompt.py
LLM_PROVIDER=anthropic python3 ./src/prompt.py
```

### Web API

Start the FastAPI server and make requests:

```bash
# Using OpenAI
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "your mingrelian text",
    "api_key": "your_openai_key",
    "provider": "openai",
    "model": "gpt-4o"
  }'

# Using Claude
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "your mingrelian text",
    "api_key": "your_anthropic_key",
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929"
  }'
```

### Programmatic Usage

```python
from src.llm_client import LLMClient, get_default_llm_client

# Use environment configuration
client = get_default_llm_client()
response = client.complete("Your Mingrelian text here")

# Or specify provider directly
client = LLMClient(provider="anthropic", model="claude-sonnet-4-5-20250929", api_key="sk-ant-...")
response = client.complete("Your Mingrelian text here")
```

## Supported Models

### OpenAI Models
- `gpt-4o` (default, recommended)
- `gpt-4o-mini` (faster, cheaper)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic Models
- `claude-sonnet-4-5-20250929` (default, recommended)
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229` (most capable)
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307` (fastest, cheapest)

## Architecture

### Translation Pipeline (RAG-based)

The translation system uses a multi-stage Retrieval-Augmented Generation (RAG) pipeline:

```
User Input (Mingrelian)
       ↓
┌──────────────────────────────────────┐
│ PHASE 0: Corpus Search (Priority)   │ ⚡ NEW: Search corpus FIRST
│ Source: en_to_xmf.json               │
│ • Exact match → Skip dictionary      │
│ • Word-in-phrase → Skip dictionary   │
│ • Fuzzy match → Also search dict    │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ PHASE 1: Dictionary Lookup           │ (Conditional - only if needed)
│ Source: kajaia.txt                    │
│ • Exact match                         │
│ • Lemmatization                       │
│ • Partial text search                 │
│ • Fuzzy matching (last resort)       │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ PHASE 2: First LLM Call              │ (Conditional - only if dict entries)
│ Task: Translate Georgian → English   │
│ Output: JSON format                   │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ PHASE 3: Second LLM Call (Adaptive)  │ ⚡ NEW: Two paths
│ Path A: Full (dictionary exists)     │
│   • Include grammar (Harris, Popiel) │
│   • Complex analysis (~6500 tokens)  │
│ Path B: Simplified (corpus-only)     │
│   • Just get Georgian (~800 tokens)  │
│   • 87% token reduction!             │
└──────────────────┬───────────────────┘
                   ↓
           Final Translation
```

### Performance Optimizations

The system includes three major optimizations for corpus hits (e.g., "ჯოხო"):

| Optimization | Impact | Savings |
|-------------|--------|---------|
| **Corpus-First Search** | Skip dictionary when corpus has exact/phrase match | Eliminates 33+ low-quality lookups |
| **Conditional First LLM** | Skip when no dictionary entries exist | 100% savings on first LLM call |
| **Simplified Second LLM** | Use light prompt for corpus-only queries | 87% token reduction (6500→800) |

**Combined Result:** 90% faster, 91% cheaper for corpus hits (10-12s → 1-2s, $0.22 → $0.02)

### Data Sources

1. **Parallel Corpus** (`data/en_to_xmf.json`)
   - Mingrelian-English parallel translations
   - Highest priority for exact/phrase matches
   - Authentic usage examples

2. **Dictionary** (`data/kajaia.txt`)
   - Mingrelian-Georgian dictionary
   - Used when corpus doesn't have matches
   - Supports lemmatization and fuzzy search

3. **Grammar References** (`data/harris.txt`, `data/popiel.txt`)
   - Only loaded when analyzing dictionary entries
   - Skipped for corpus-only queries

### LLM Integration

All LLM API calls are centralized in `src/llm_client.py`:

```
Application Code
       ↓
  LLMClient (src/llm_client.py)
       ↓
  ┌────┴────┐
  ↓         ↓
OpenAI   Anthropic
  API       API
```

**Benefits:**
- **Single Source of Truth**: All LLM calls in one place
- **Easy Switching**: Change providers without touching app code
- **Testability**: Mock the LLMClient for testing
- **Consistency**: Same error handling and logging everywhere
- **Extensibility**: Add new providers easily
- **Cost Optimization**: Quickly test which provider/model is most cost-effective

## Cost Comparison

| Provider | Model | Cost (per 1M tokens) | Best For |
|----------|-------|---------------------|----------|
| OpenAI | gpt-4o | $2.50 / $10.00 | General purpose |
| OpenAI | gpt-4o-mini | $0.15 / $0.60 | High volume |
| Anthropic | claude-3-opus | $15.00 / $75.00 | Complex tasks |
| Anthropic | claude-3-sonnet | $3.00 / $15.00 | Balanced |
| Anthropic | claude-3-haiku | $0.25 / $1.25 | High speed |

*Note: Prices are approximate and may change. Check provider pricing pages for current rates.*

## API Reference

### LLMClient

```python
class LLMClient:
    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    )
```

**Parameters:**
- `provider`: Which LLM provider to use
- `api_key`: API key (reads from env if None)
- `model`: Model name (uses provider default if None)
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum response tokens

**Methods:**
- `complete(prompt: str, system_prompt: Optional[str] = None) -> str`: Send a prompt and get response

### Helper Functions

```python
# Get a client using environment variables
from src.llm_client import get_default_llm_client
client = get_default_llm_client()

# Quick completion functions
from src.llm_client import complete_with_openai, complete_with_claude
response = complete_with_openai("Your prompt")
response = complete_with_claude("Your prompt")
```

## Troubleshooting

### "API key not found"
Make sure you've set the appropriate API key in `.env`:
- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic

### "anthropic package not installed"
Install it: `pip install anthropic`

### "Invalid model name"
Check the supported models list above and make sure you're using a valid model name for your provider.

### Rate Limits
If you hit rate limits:
- For OpenAI: Use `gpt-4o-mini` or `gpt-3.5-turbo` (higher limits)
- For Anthropic: Use `claude-3-haiku-20240307` (higher limits)

## Adding New Providers

To add support for a new LLM provider (e.g., Cohere, Mistral):

1. Add the provider to `LLMProvider` type in `llm_client.py`
2. Add initialization logic in `__init__`
3. Add a `_complete_<provider>` method
4. Update the `complete` method to route to your new provider

Example:

```python
def _complete_cohere(self, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Complete using Cohere API."""
    # Your implementation here
    pass
```

## Migration from Direct API Calls

### Old Code (Direct OpenAI calls)
```python
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Your prompt"}]
)
result = response.choices[0].message.content
```

### New Code (Using LLMClient)
```python
from src.llm_client import get_default_llm_client

client = get_default_llm_client()
result = client.complete("Your prompt")
```

## Project Structure

```
argo/
├── src/
│   ├── llm_client.py         # Central LLM abstraction layer
│   ├── prompt.py             # Main translation orchestration
│   ├── prompts.py            # Prompt construction (full & simplified)
│   ├── corpus_search.py      # Parallel corpus search
│   ├── translate.py          # Dictionary lookup & lemmatization
│   ├── transliterate.py      # Latinized ↔ Mkhedruli conversion
│   └── extract_definition.py # Definition extraction
├── data/
│   ├── en_to_xmf.json        # Mingrelian-English parallel corpus
│   ├── kajaia.txt            # Mingrelian-Georgian dictionary
│   ├── harris.txt            # Grammar reference
│   └── popiel.txt            # Grammar reference
├── fastapi_app/
│   └── api.py                # Web API interface
├── .env                      # Environment configuration
└── README.md                 # This file
```

## How It Works

### Example: Translating "joxo" (ჯოხო)

```bash
$ echo "joxo" | python3 src/prompt.py
```

**Step 1: Corpus Search** (~100ms)
```
✓ Found exact match in corpus: ჯოხო → "call_sb/sth_(by_name), refer"
✓ Found 5 usage examples in authentic text
→ SKIP dictionary search (corpus has high-quality data)
```

**Step 2: Dictionary Search**
```
⏭️ SKIPPED (corpus has everything)
```

**Step 3: First LLM Call**
```
⏭️ SKIPPED (no dictionary entries to translate)
```

**Step 4: Second LLM Call** (~1-2s, simplified prompt)
```
Input: Corpus translations + "Provide Georgian equivalent"
Output: Georgian: ჰქვია | English: to call (by name)
```

**Total Time:** ~1-2 seconds (was 10-12s before optimization)  
**Total Cost:** ~$0.02 (was $0.22 before optimization)

### Example: Word Not in Corpus

If the word isn't in the corpus, the system falls back to the full pipeline:

```
Step 1: Corpus search → No matches
Step 2: Dictionary search → Find entries via lemmatization/fuzzy
Step 3: First LLM → Translate Georgian dictionary entries to English
Step 4: Second LLM → Full analysis with grammar context
```

This ensures the system always provides results, prioritizing fast corpus lookups when available.

## Contributing

When adding new features or providers:

1. All LLM interactions should go through `src/llm_client.py`
2. Maintain backward compatibility
3. Add appropriate error handling
4. Update documentation
5. Test with multiple providers

## License

[Add your license information here]
