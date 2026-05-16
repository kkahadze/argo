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
    is_server_key_model_allowed,
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

# Request model
class PromptIn(BaseModel):
    prompt: str
    api_key: Optional[str] = None  # Optional: if None, server will use the configured key for the selected provider
    source_language: str = DEFAULT_SOURCE_LANGUAGE  # Source language: "mingrelian", "georgian", or "english"
    target_language: str = DEFAULT_TARGET_LANGUAGE  # Target language: "mingrelian", "georgian", or "english"
    provider: Optional[str] = None  # "openai", "anthropic", or "gemini" (if None, reads from env)
    model: Optional[str] = None  # Optional: specify model name (if None, reads from env)

# Initialize FastAPI app
app = FastAPI(title="Mingrelian Translator API")

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


def get_server_api_key(provider: str) -> Optional[str]:
    """Resolve the configured server-side API key for the requested provider."""
    env_var = get_api_key_env_var(provider)
    return os.getenv(env_var) if env_var else None


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
    translation = result['translation']
    full_response = result['full_response']
    
    # Initialize all fields
    output = {
        'source_text': source_text,
        'target_text': translation,
        'source_language': source_lang,
        'target_language': target_lang,
        'mingrelian_latinized': '',
        'mingrelian_mkhedruli': '',
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
    
    else:
        # Neither source nor target is mingrelian (georgian <-> english)
        if source_lang == 'english':
            output['english'] = source_text
            output['georgian'] = translation
        elif source_lang == 'georgian':
            output['georgian'] = source_text
            output['english'] = translation
    
    return output


async def stream_translation(
    prompt_text,
    api_key,
    source_language=DEFAULT_SOURCE_LANGUAGE,
    target_language=DEFAULT_TARGET_LANGUAGE,
    provider=None,
    model=None,
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
        source_language: Source language ("mingrelian", "georgian", or "english")
        target_language: Target language ("mingrelian", "georgian", or "english")
        provider: "openai" or "anthropic" (if None, reads from LLM_PROVIDER env var, defaults to "openai")
        model: Optional model name (if None, reads from LLM_MODEL env var, then uses provider default)
    """
    request_meta = request_meta or {}
    request_started_at = time.time()

    # Use environment variables if provider/model not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    if model is None:
        model = os.getenv("LLM_MODEL")
    
    # Log the translation request
    log_translation_request(
        logger, 
        prompt_text, 
        source_language, 
        target_language, 
        provider,
        model or 'default'
    )
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
    except Exception as e:
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
                status="error",
                error_message=f"Failed to initialize LLM client: {str(e)}",
                app_origin=request_meta.get("origin"),
                referer=request_meta.get("referer"),
                user_agent=request_meta.get("user_agent"),
            )
        )

        yield _sse_event({"error": f"Failed to initialize LLM client: {str(e)}"})
        return
    
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
                prompt_metrics=result.get("prompt_metrics"),
                app_origin=request_meta.get("origin"),
                referer=request_meta.get("referer"),
                user_agent=request_meta.get("user_agent"),
            )
        )
        
        # Send final result
        yield _sse_event({"result": formatted_result})
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
    Process text translation between Mingrelian, Georgian, and English.
    
    Parameters:
    - prompt: Text to translate
    - api_key: API key for the LLM provider (optional if the server has a provider key configured)
    - source_language: Source language ("mingrelian", "georgian", or "english")
    - target_language: Target language ("mingrelian", "georgian", or "english")
    - provider: LLM provider to use ("openai", "anthropic", or "gemini")
    - model: Optional model name (uses provider default if None)
    """
    if not data.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt text is required")
    
    api_key = data.api_key
    provider = data.provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider must be one of: {', '.join(SUPPORTED_PROVIDERS)}",
        )

    env_provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    env_model = os.getenv("LLM_MODEL") if provider == env_provider else None
    model = data.model or env_model or get_default_model_for_provider(provider)

    if not api_key:
        if not is_server_key_model_allowed(provider, model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' requires a user-provided API key"
            )
        api_key = get_server_api_key(provider)
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail=f"Server-side API key not configured for provider '{provider}'"
            )
    
    # Validate language parameters
    if data.source_language not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Source language must be one of: {', '.join(VALID_LANGUAGES)}")
    if data.target_language not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Target language must be one of: {', '.join(VALID_LANGUAGES)}")
    if data.source_language == data.target_language:
        raise HTTPException(status_code=400, detail="Source and target languages must be different")
    
    # Return streaming response
    return StreamingResponse(
        stream_translation(
            data.prompt,
            api_key,
            data.source_language,
            data.target_language,
            provider,
            model,
            used_user_api_key=bool(data.api_key),
            request_meta={
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
