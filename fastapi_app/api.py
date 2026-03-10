from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path so we can import modules from it
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_client import LLMClient
from src.single_call_translator import translate as single_call_translate
from src.logger import (
    setup_logger,
    log_translation_request,
    log_translation_result,
    log_error
)
import re

# Request model
class PromptIn(BaseModel):
    prompt: str
    api_key: str = None  # Optional: if None, server will use default Gemini key
    source_language: str = "mingrelian"  # Source language: "mingrelian", "georgian", or "english"
    target_language: str = "english"  # Target language: "mingrelian", "georgian", or "english"
    provider: str = None  # "openai", "anthropic", or "gemini" (if None, reads from env)
    model: str = None  # Optional: specify model name (if None, reads from env)

# Response model
class ResponseOut(BaseModel):
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    # Legacy fields for backward compatibility
    mingrelian_latinized: str = ""
    mingrelian_mkhedruli: str = ""
    georgian: str = ""
    english: str = ""
    full_response: str = None

# Initialize FastAPI app
app = FastAPI(title="Mingrelian Translator API")

# Setup logger
logger = setup_logger('api')

# Setup logger
logger = setup_logger('api')

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def is_mkhedruli(text):
    """Check if text contains Georgian Mkhedruli script characters."""
    return bool(re.search('[\u10D0-\u10FF]', text))


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


async def stream_translation(prompt_text, api_key, source_language="mingrelian", target_language="english", provider=None, model=None):
    """
    Stream translation with a single LLM API call.
    Yields JSON events for progress updates.
    
    Args:
        prompt_text: Text to translate
        api_key: API key for the LLM provider
        source_language: Source language ("mingrelian", "georgian", or "english")
        target_language: Target language ("mingrelian", "georgian", or "english")
        provider: "openai" or "anthropic" (if None, reads from LLM_PROVIDER env var, defaults to "openai")
        model: Optional model name (if None, reads from LLM_MODEL env var, then uses provider default)
    """
    # Use environment variables if provider/model not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
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

        yield f"data: {json.dumps({'error': f'Failed to initialize LLM client: {str(e)}'})}\n\n"
        return
    
    try:
        # Call the single-call translator
        print(f"Starting translation: {source_language} → {target_language}")
        log_translation_request(
            logger, 
            prompt_text, 
            source_language, 
            target_language, 
            provider,
            model or 'default'
        )
        result = single_call_translate(
            input_text=prompt_text,
            source_lang=source_language,
            target_lang=target_language,
            llm_client=llm_client
        )
        print(f"Translation complete")
        
        # Format output for legacy compatibility
        formatted_result = format_output_for_legacy(
            result=result,
            source_lang=source_language,
            target_lang=target_language,
            source_text=prompt_text
        )
        
        # Send final result
        final_event = f"data: {json.dumps({'result': formatted_result})}\n\n"
        print(f"Sending final result")
        yield final_event
        print("Stream completed successfully")
        return
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in stream_translation: {str(e)}")
        print(error_details)
        log_error(logger, e, {
            'input_text': prompt_text,
            'source_language': source_language,
            'target_language': target_language,
            'provider': provider,
            'model': model
        })
        yield f"data: {json.dumps({'error': f'Translation error: {str(e)}'})}\n\n"
        return


@app.post("/chat")
async def chat(data: PromptIn):
    """
    Process text translation between Mingrelian, Georgian, and English.
    
    Parameters:
    - prompt: Text to translate
    - api_key: API key for the LLM provider
    - source_language: Source language ("mingrelian", "georgian", or "english")
    - target_language: Target language ("mingrelian", "georgian", or "english")
    - provider: LLM provider to use ("openai", "anthropic", or "gemini")
    - model: Optional model name (uses provider default if None)
    """
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt text is required")
    
    # Default to Gemini if no API key provided (free tier)
    api_key = data.api_key
    provider = data.provider
    model = data.model
    
    if not api_key:
        # Use server-side Gemini key for free public access
        provider = "gemini"
        model = model or "gemini-3.1-flash-lite-preview"
        api_key = os.getenv("GEMINI_API_KEY")  # Server-side key
        if not api_key:
            raise HTTPException(status_code=500, detail="Server API key not configured")
    
    if provider is not None and provider not in ["openai", "anthropic", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider must be 'openai', 'anthropic', or 'gemini'")
    
    # Validate language parameters
    valid_languages = ["mingrelian", "georgian", "english"]
    if data.source_language not in valid_languages:
        raise HTTPException(status_code=400, detail=f"Source language must be one of: {', '.join(valid_languages)}")
    if data.target_language not in valid_languages:
        raise HTTPException(status_code=400, detail=f"Target language must be one of: {', '.join(valid_languages)}")
    if data.source_language == data.target_language:
        raise HTTPException(status_code=400, detail="Source and target languages must be different")
    
    # Return streaming response
    return StreamingResponse(
        stream_translation(data.prompt, api_key, data.source_language, data.target_language, provider, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    ) 
