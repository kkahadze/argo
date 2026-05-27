#!/usr/bin/env python3
"""
Promptfoo custom provider for Mingrelian translation evaluation.
This wraps the entire translation pipeline including dynamic dictionary lookups.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import from src/
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(parent_dir, '.env'), override=True)

from src.single_call_translator import translate
from src.llm_client import LLMClient
from src.provider_config import (
    DEFAULT_MODEL_BY_PROVIDER,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    get_api_key_env_var,
)


EVAL_DEFAULT_PROVIDER = 'gemini'


def call_api(prompt, options, context):
    """
    Promptfoo provider interface.
    
    This function performs the full translation pipeline:
    1. Takes the input text (prompt)
    2. Performs dictionary lookups dynamically
    3. Constructs the full prompt with RAG context
    4. Calls the LLM
    5. Extracts and returns the translation
    
    Args:
        prompt (str): The text to translate
        options (dict): Configuration from promptfooconfig.yaml
            - provider: LLM provider (openai, anthropic, gemini)
            - model: Model name (defaults to the selected provider's configured default)
            - api_key: Optional API key
            - source_language: Source language (default: mingrelian)
            - target_language: Target language (default: english)
            - temperature: Temperature setting (default: 1.0 except omitted for
              higher-effort OpenAI GPT-5 reasoning runs)
            - max_tokens: Max tokens (optional)
            - reasoning_effort: OpenAI reasoning effort for GPT-5 family models (optional)
        context (dict): Additional context from promptfoo
    
    Returns:
        dict: Response with 'output' and optional 'error'
    """
    try:
        runtime_options = options.get('config', options)

        # Extract configuration
        provider = runtime_options.get('provider', EVAL_DEFAULT_PROVIDER)
        model = runtime_options.get('model', DEFAULT_MODEL_BY_PROVIDER.get(provider))
        api_key_env_var = get_api_key_env_var(provider)
        env_api_key = os.getenv(api_key_env_var) if api_key_env_var else None
        api_key = runtime_options.get('api_key') or env_api_key
        source_language = runtime_options.get('source_language', DEFAULT_SOURCE_LANGUAGE)
        target_language = runtime_options.get('target_language', DEFAULT_TARGET_LANGUAGE)
        max_tokens = runtime_options.get('max_tokens')
        reasoning_effort = runtime_options.get('reasoning_effort')
        if 'temperature' in runtime_options:
            temperature = runtime_options['temperature']
        elif (
            provider == 'openai'
            and model
            and model.lower().startswith('gpt-5')
            and reasoning_effort
            and reasoning_effort != 'none'
        ):
            # OpenAI reasoning models do not accept temperature at higher efforts.
            temperature = None
        else:
            temperature = 1.0
        grammar_policy = runtime_options.get('grammar_policy') or os.getenv('ARGO_GRAMMAR_POLICY')
        
        # Initialize LLM client
        llm_client = LLMClient(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        
        # Perform translation (includes dynamic dictionary lookup and prompt construction)
        result = translate(
            input_text=prompt,
            source_lang=source_language,
            target_lang=target_language,
            llm_client=llm_client,
            grammar_policy=grammar_policy,
        )
        
        # Return the translation
        return {
            'output': result['translation'],
            'metadata': {
                'full_response': result['full_response'],
                'provider': provider,
                'model': llm_client.model,
                'reasoning_effort': reasoning_effort,
                'source_language': source_language,
                'target_language': target_language,
                'response_source': result.get('response_source'),
                'prompt_metrics': result.get('prompt_metrics'),
            }
        }
        
    except Exception as e:
        # Return error for promptfoo to handle
        return {
            'error': str(e),
            'output': f"ERROR: {str(e)}"
        }


# Optional: For testing the provider directly
if __name__ == '__main__':
    import json
    
    # Test configuration
    test_prompt = "მა"
    test_options = {
        'provider': EVAL_DEFAULT_PROVIDER,
        'model': DEFAULT_MODEL_BY_PROVIDER[EVAL_DEFAULT_PROVIDER],
        'source_language': DEFAULT_SOURCE_LANGUAGE,
        'target_language': DEFAULT_TARGET_LANGUAGE,
        'temperature': 0.7
    }
    
    print("Testing provider with prompt:", test_prompt)
    result = call_api(test_prompt, test_options, {})
    print("\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
