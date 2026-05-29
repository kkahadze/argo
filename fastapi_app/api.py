from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path so we can import modules from it
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_client import LLMClient
from src.provider_config import (
    DEFAULT_PROVIDER,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    SUPPORTED_PROVIDERS,
    VALID_LANGUAGES,
    get_api_key_env_var,
    get_default_model_for_provider,
    get_default_reasoning_effort_for_model,
    is_server_key_model_allowed,
)
from src.language_packs import (
    get_language_pack,
    get_low_resource_pack_for_pair,
    is_supported_translation_pair,
)
from src.single_call_translator import translate as single_call_translate
from src.logger import (
    setup_logger,
    log_translation_request,
    log_error
)
from src.translation_analytics import (
    build_translation_event,
    infer_response_source,
    schedule_translation_event,
)
import re

VISITOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Request model
class PromptIn(BaseModel):
    prompt: str
    api_key: Optional[str] = None  # Optional: if None, server will use the configured key for the selected provider
    source_language: str = DEFAULT_SOURCE_LANGUAGE  # Source language: "mingrelian", "georgian", or "english"
    target_language: str = DEFAULT_TARGET_LANGUAGE  # Target language: "mingrelian", "georgian", or "english"
    provider: Optional[str] = None  # "openai", "anthropic", or "gemini" (if None, reads from env)
    model: Optional[str] = None  # Optional: specify model name (if None, reads from env)
    reasoning_effort: Optional[str] = None  # Optional: OpenAI reasoning effort for GPT-5 family models
    visitor_id: Optional[str] = None  # Anonymous browser identifier for analytics only

# Initialize FastAPI app
app = FastAPI(title="Argo Translator API")

# Setup logger
logger = setup_logger('api')

# Configure CORS to allow browser clients from any origin. We do not rely on
# cookies or browser credentials, so keep credentials disabled with a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def _sse_event(payload: dict) -> str:
    """Format a server-sent event payload."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def is_mkhedruli(text: str) -> bool:
    """Check if text contains Georgian Mkhedruli script characters."""
    return bool(re.search('[\u10D0-\u10FF]', text))


def normalize_visitor_id(visitor_id: Optional[str]) -> Optional[str]:
    """Keep only short, opaque browser-generated visitor IDs for analytics."""
    if not visitor_id:
        return None

    cleaned = visitor_id.strip()
    if not VISITOR_ID_PATTERN.fullmatch(cleaned):
        return None
    return cleaned


def get_server_api_key(provider: str) -> Optional[str]:
    """Resolve the configured server-side API key for the requested provider."""
    env_var = get_api_key_env_var(provider)
    return os.getenv(env_var) if env_var else None


class LLMConfigurationError(Exception):
    """Raised when an LLM call is needed but credentials are not usable."""


class LLMInitializationError(Exception):
    """Raised when the provider client cannot be initialized."""


def resolve_api_key_for_llm(provider: str, model: str, api_key: Optional[str]) -> str:
    """Resolve credentials only when the request actually reaches the LLM path."""
    if api_key:
        return api_key

    if not is_server_key_model_allowed(provider, model):
        raise LLMConfigurationError(f"Model '{model}' requires a user-provided API key")

    server_api_key = get_server_api_key(provider)
    if not server_api_key:
        raise LLMConfigurationError(f"Server-side API key not configured for provider '{provider}'")

    return server_api_key


class LazyLLMClient:
    """Delay provider credentials and SDK client setup until the first LLM call."""

    def __init__(self, provider: str, model: str, api_key: Optional[str], reasoning_effort: Optional[str] = None):
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._reasoning_effort = reasoning_effort
        self._client = None

    def _get_client(self):
        if self._client is None:
            resolved_api_key = resolve_api_key_for_llm(self.provider, self.model, self._api_key)
            try:
                self._client = LLMClient(
                    provider=self.provider,
                    model=self.model,
                    api_key=resolved_api_key,
                    reasoning_effort=self._reasoning_effort,
                )
            except Exception as exc:
                raise LLMInitializationError(f"Failed to initialize LLM client: {str(exc)}") from exc

            self.model = self._client.model

        return self._client

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._get_client().complete(prompt, system_prompt=system_prompt)


def format_output_for_legacy(result, source_lang, target_lang, source_text):
    """
    Format the new single-call output to match legacy format for backward compatibility.
    
    Args:
        result: Result dict from single_call_translate
        source_lang: Source language
        target_lang: Target language
        source_text: Original input text
        
    Returns:
        dict: Formatted result with legacy fields populated
    """
    target_pack = get_low_resource_pack_for_pair(source_lang, target_lang)
    translation = result['translation']
    if target_pack and target_lang == target_pack.code:
        translation = target_pack.normalize_output(translation)
    full_response = result['full_response']
    
    # Initialize all fields
    output = {
        'source_text': source_text,
        'target_text': translation,
        'translated_text': translation,
        'romanized_text': '',
        'source_language': source_lang,
        'target_language': target_lang,
        'mingrelian_latinized': '',
        'mingrelian_mkhedruli': '',
        'tsova_tush_latinized': '',
        'tsova_tush_mkhedruli': '',
        'svan': '',
        'georgian': '',
        'english': '',
        'full_response': full_response
    }
    
    # Populate legacy fields based on language directions
    if source_lang == 'mingrelian':
        # Detect if source is mkhedruli or latinized
        if is_mkhedruli(source_text):
            output['mingrelian_mkhedruli'] = source_text
        else:
            output['mingrelian_latinized'] = source_text
        
        if target_lang == 'english':
            output['english'] = translation
        elif target_lang == 'georgian':
            output['georgian'] = translation
    
    elif target_lang == 'mingrelian':
        # Target is mingrelian
        if is_mkhedruli(translation):
            output['mingrelian_mkhedruli'] = translation
        else:
            output['mingrelian_latinized'] = translation
        
        if source_lang == 'english':
            output['english'] = source_text
        elif source_lang == 'georgian':
            output['georgian'] = source_text
        output['romanized_text'] = (
            output['mingrelian_latinized']
            if output['mingrelian_latinized']
            else ''
        )

    elif source_lang == 'tsova_tush':
        if is_mkhedruli(source_text):
            output['tsova_tush_mkhedruli'] = source_text
        else:
            output['tsova_tush_latinized'] = source_text

        if target_lang == 'english':
            output['english'] = translation
        elif target_lang == 'georgian':
            output['georgian'] = translation

    elif target_lang == 'tsova_tush':
        if is_mkhedruli(translation):
            output['tsova_tush_mkhedruli'] = translation
        else:
            output['tsova_tush_latinized'] = translation

        if source_lang == 'english':
            output['english'] = source_text
        elif source_lang == 'georgian':
            output['georgian'] = source_text
        output['romanized_text'] = (
            output['tsova_tush_latinized']
            if output['tsova_tush_latinized']
            else ''
        )

    elif source_lang == 'svan':
        output['svan'] = source_text

        if target_lang == 'english':
            output['english'] = translation
        elif target_lang == 'georgian':
            output['georgian'] = translation

    elif target_lang == 'svan':
        output['svan'] = translation

        if source_lang == 'english':
            output['english'] = source_text
        elif source_lang == 'georgian':
            output['georgian'] = source_text
    
    else:
        # Neither source nor target is mingrelian (georgian <-> english)
        if source_lang == 'english':
            output['english'] = source_text
            output['georgian'] = translation
        elif source_lang == 'georgian':
            output['georgian'] = source_text
            output['english'] = translation

    if target_lang in {'mingrelian', 'tsova_tush', 'svan'} and not output['romanized_text']:
        if not is_mkhedruli(translation):
            output['romanized_text'] = translation

    return output


async def stream_translation(
    prompt_text,
    api_key,
    source_language=DEFAULT_SOURCE_LANGUAGE,
    target_language=DEFAULT_TARGET_LANGUAGE,
    provider=None,
    model=None,
    reasoning_effort=None,
    *,
    used_user_api_key=False,
    request_meta=None,
):
    """
    Stream translation with a single LLM API call.
    Yields JSON events for progress updates.
    
    Args:
        prompt_text: Text to translate
        api_key: API key for the LLM provider (optional if the server has a provider key configured)
        source_language: Source language ("mingrelian", "tsova_tush", "georgian", or "english")
        target_language: Target language ("mingrelian", "tsova_tush", "georgian", or "english")
        provider: "openai", "anthropic", or "gemini" (if None, reads from LLM_PROVIDER env var, defaults to "openai")
        model: Optional model name (if None, reads from LLM_MODEL env var, then uses provider default)
        reasoning_effort: Optional OpenAI reasoning effort for GPT-5 family models
    """
    request_meta = request_meta or {}
    request_started_at = time.time()

    # Use environment variables if provider/model not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    if model is None:
        model = os.getenv("LLM_MODEL") or get_default_model_for_provider(provider)
    if reasoning_effort is None:
        reasoning_effort = get_default_reasoning_effort_for_model(model)
    
    # Log the translation request
    log_translation_request(
        logger, 
        prompt_text, 
        source_language, 
        target_language, 
        provider,
        model or 'default'
    )
    
    llm_client = LazyLLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )
    
    try:
        # Call the single-call translator
        result = single_call_translate(
            input_text=prompt_text,
            source_lang=source_language,
            target_lang=target_language,
            llm_client=llm_client
        )
        # Format output for legacy compatibility
        formatted_result = format_output_for_legacy(
            result=result,
            source_lang=source_language,
            target_lang=target_language,
            source_text=prompt_text
        )

        schedule_translation_event(
            build_translation_event(
                source_text=prompt_text,
                target_text=formatted_result["target_text"],
                source_language=source_language,
                target_language=target_language,
                provider=provider,
                model=llm_client.model,
                duration_ms=int((time.time() - request_started_at) * 1000),
                response_source=result.get("response_source") or infer_response_source(result),
                used_user_api_key=used_user_api_key,
                visitor_id=request_meta.get("visitor_id"),
                prompt_metrics=result.get("prompt_metrics"),
                app_origin=request_meta.get("origin"),
                referer=request_meta.get("referer"),
                user_agent=request_meta.get("user_agent"),
            )
        )
        
        # Send final result
        yield _sse_event({"result": formatted_result})
        return
        
    except LLMConfigurationError as e:
        log_error(logger, e, {'provider': provider, 'model': model})
        schedule_translation_event(
            build_translation_event(
                source_text=prompt_text,
                target_text=None,
                source_language=source_language,
                target_language=target_language,
                provider=provider,
                model=model,
                duration_ms=int((time.time() - request_started_at) * 1000),
                response_source="credential_error",
                used_user_api_key=used_user_api_key,
                visitor_id=request_meta.get("visitor_id"),
                status="error",
                error_message=str(e),
                app_origin=request_meta.get("origin"),
                referer=request_meta.get("referer"),
                user_agent=request_meta.get("user_agent"),
            )
        )
        yield _sse_event({"error": str(e)})
        return

    except LLMInitializationError as e:
        log_error(logger, e, {'provider': provider, 'model': model})
        schedule_translation_event(
            build_translation_event(
                source_text=prompt_text,
                target_text=None,
                source_language=source_language,
                target_language=target_language,
                provider=provider,
                model=model,
                duration_ms=int((time.time() - request_started_at) * 1000),
                response_source="init_error",
                used_user_api_key=used_user_api_key,
                visitor_id=request_meta.get("visitor_id"),
                status="error",
                error_message=str(e),
                app_origin=request_meta.get("origin"),
                referer=request_meta.get("referer"),
                user_agent=request_meta.get("user_agent"),
            )
        )
        yield _sse_event({"error": str(e)})
        return

    except Exception as e:
        log_error(logger, e, {
            'input_text': prompt_text,
            'source_language': source_language,
            'target_language': target_language,
            'provider': provider,
            'model': model
        })
        schedule_translation_event(
            build_translation_event(
                source_text=prompt_text,
                target_text=None,
                source_language=source_language,
                target_language=target_language,
                provider=provider,
                model=getattr(llm_client, "model", model),
                duration_ms=int((time.time() - request_started_at) * 1000),
                response_source="translation_error",
                used_user_api_key=used_user_api_key,
                visitor_id=request_meta.get("visitor_id"),
                status="error",
                error_message=str(e),
                app_origin=request_meta.get("origin"),
                referer=request_meta.get("referer"),
                user_agent=request_meta.get("user_agent"),
            )
        )
        yield _sse_event({"error": f"Translation error: {str(e)}"})
        return


@app.post("/chat")
async def chat(data: PromptIn, request: Request):
    """
    Process text translation between Mingrelian, Bats, Georgian, and English.
    
    Parameters:
    - prompt: Text to translate
    - api_key: API key for the LLM provider (optional if the server has a provider key configured)
    - source_language: Source language ("mingrelian", "tsova_tush", "georgian", or "english")
    - target_language: Target language ("mingrelian", "tsova_tush", "georgian", or "english")
    - provider: LLM provider to use ("openai", "anthropic", or "gemini")
    - model: Optional model name (uses provider default if None)
    - reasoning_effort: Optional OpenAI reasoning effort for GPT-5 family models
    """
    if not data.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt text is required")
    
    provider = data.provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider must be one of: {', '.join(SUPPORTED_PROVIDERS)}",
        )

    env_provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    env_model = os.getenv("LLM_MODEL") if provider == env_provider else None
    model = data.model or env_model or get_default_model_for_provider(provider)
    reasoning_effort = data.reasoning_effort or get_default_reasoning_effort_for_model(model)
    
    # Validate language parameters
    if data.source_language not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Source language must be one of: {', '.join(VALID_LANGUAGES)}")
    if data.target_language not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Target language must be one of: {', '.join(VALID_LANGUAGES)}")
    if data.source_language == data.target_language:
        raise HTTPException(status_code=400, detail="Source and target languages must be different")
    if not is_supported_translation_pair(data.source_language, data.target_language):
        raise HTTPException(
            status_code=400,
            detail=f"Translation pair not supported yet: {data.source_language} -> {data.target_language}",
        )

    visitor_id = normalize_visitor_id(data.visitor_id)
    
    # Return streaming response
    return StreamingResponse(
        stream_translation(
            data.prompt,
            data.api_key,
            data.source_language,
            data.target_language,
            provider,
            model,
            reasoning_effort,
            used_user_api_key=bool(data.api_key),
            request_meta={
                "visitor_id": visitor_id,
                "origin": request.headers.get("origin"),
                "referer": request.headers.get("referer"),
                "user_agent": request.headers.get("user-agent"),
            },
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    ) 
