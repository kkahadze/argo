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
from src.prompt import extract_translations
from src.transliterate import latinized_to_mkhedruli, mkhedruli_to_latinized
from src.prompts import (
    get_initial_translation_prompt,
    get_follow_up_phrase,
    get_grammar_text,
    get_after_phrase,
    get_follow_up_prompt,
    log_to_file
)
from src.llm_client import LLMClient
import re

# Import dictionary processing functions
from src.translate import translate_lemma, search_containing_word, lemmatize_mingrelian, find_close_lemma_matches

# Request model
class PromptIn(BaseModel):
    prompt: str
    api_key: str
    target_language: str = "english"  # Default to English if not specified
    provider: str = None  # "openai" or "anthropic" (if None, reads from env)
    model: str = None  # Optional: specify model name (if None, reads from env)

# Response model
class ResponseOut(BaseModel):
    mingrelian_latinized: str
    mingrelian_mkhedruli: str
    georgian: str
    english: str
    full_response: str = None

# Initialize FastAPI app
app = FastAPI(title="Mingrelian Translator API")

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

def process_prompt(prompt_text, api_key, provider=None, model=None):
    """
    Process the prompt text using the specified LLM provider.
    
    Args:
        prompt_text: Text to translate
        api_key: API key for the LLM provider
        provider: "openai" or "anthropic" (if None, reads from LLM_PROVIDER env var, defaults to "openai")
        model: Optional model name (if None, reads from LLM_MODEL env var, then uses provider default)
    """
    # Use environment variables if provider/model not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
    if model is None:
        model = os.getenv("LLM_MODEL")
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize LLM client: {str(e)}")
    
    # Determine if input is mkhedruli or latinized
    if is_mkhedruli(prompt_text):
        mkhedruli = prompt_text
        latinized = mkhedruli_to_latinized(prompt_text)
    else:
        mkhedruli = latinized_to_mkhedruli(prompt_text)
        latinized = prompt_text
    
    # Load grammar for translation
    grammar_path = Path(__file__).parent.parent / 'data' / 'harris.txt'
    try:
        with open(grammar_path, 'r') as file:
            grammar = file.read()
    except FileNotFoundError:
        # If harris.txt doesn't exist, use an empty string
        grammar = ""
    
    # Default dictionary file path
    dict_file = Path(__file__).parent.parent / 'data' / 'kajaia.txt'
    
    # Create a variable to collect all output similar to prompt.py
    dict_entries = []
    all_entries = []
    
    # Break the string into words
    words = latinized.split()
    
    # Process each word similar to prompt.py
    for word in words:
        # Check if the word is in Mkhedruli script and convert if needed
        original_word = word
        if is_mkhedruli(word):
            latinized_word = mkhedruli_to_latinized(word)
            dict_entries.append(f"Input in Mkhedruli script: {word}")
            dict_entries.append(f"Converted to latinized form: {latinized_word}")
            word = latinized_word
        else:
            dict_entries.append(f"Latinized Mingrelian word: {word}")
        
        # Get translation results for this word - use debug_output=False to avoid console printing
        results = translate_lemma(dict_file, word, debug_output=False)
        
        # Show Mkhedruli form - only if input was originally latinized
        if not is_mkhedruli(original_word):
            dict_entries.append(f"Mkhedruli Mingrelian word: {latinized_to_mkhedruli(word)}")
        
        if len(results) == 1 and results[0][1] is None:
            # No direct translation found, try with lemmatized form
            lemmatized_word = lemmatize_mingrelian(word)
            if lemmatized_word != word:
                dict_entries.append(f"Trying lemmatized form: '{lemmatized_word}'")
                lemma_results = translate_lemma(dict_file, lemmatized_word, debug_output=False)
                
                # If we found results with the lemmatized form
                if not (len(lemma_results) == 1 and lemma_results[0][1] is None):
                    dict_entries.append(f"Found translation for lemmatized form '{lemmatized_word}'")
                    for curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word in lemma_results:
                        if entry_text:
                            # Store the complete entry text
                            all_entries.append(f"Match for lemmatized form '{lemmatized_word}' of '{word}':\n{entry_text}")
                        else:
                            # If no entry found
                            all_entries.append(f"No translation found for lemmatized form '{lemmatized_word}' of '{word}'")
                    continue  # Skip to next word since we found a match with the lemmatized form
                
                # Look for similar lemmas to the lemmatized form
                dict_entries.append(f"Looking for similar lemmas (edit distance of 1) to lemmatized form '{lemmatized_word}'...")
                lemma_close_matches = find_close_lemma_matches(dict_file, lemmatized_word, max_distance=1)
                
                if lemma_close_matches:
                    # Process similar lemmas
                    dict_entries.append(f"Found {len(lemma_close_matches)} similar lemmas with small differences to lemmatized form")
                    all_entries.append(f"Similar lemmas found for '{lemmatized_word}'")
            
            # Try to find direct similar matches
            dict_entries.append(f"Looking for similar lemmas (edit distance of 1) to '{word}'...")
            close_matches = find_close_lemma_matches(dict_file, word, max_distance=1)
            
            if close_matches:
                dict_entries.append(f"Found {len(close_matches)} similar lemmas with small differences")
                all_entries.append(f"Similar direct lemmas found for '{word}'")
            else:
                # Last resort, search for occurrences
                dict_entries.append(f"\nNo direct or similar match in dictionary found for '{word}'. Searching for occurrences...")
                
                # Transliterate to Georgian
                georgian_word = latinized_to_mkhedruli(word)
                dict_entries.append(f"'{word}' transliterates to: '{georgian_word}'")
                
                # Search for occurrences of this word in the dictionary
                containing_results = search_containing_word(dict_file, word)
                
                if containing_results:
                    dict_entries.append(f"Found {len(containing_results)} entries containing '{georgian_word}'")
                    all_entries.append(f"Partial matches found for '{word}'")
                else:
                    all_entries.append(f"No translation or references found for '{word}' ({georgian_word})")
        else:
            for curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word in results:
                if entry_text:
                    # Store the complete entry text
                    all_entries.append(entry_text)
                else:
                    # If no entry found
                    all_entries.append(f"No translation found for '{word}'")
    
    # Add the complete entries section to the dict_entries
    dict_entries.append("\n" + "="*40)
    dict_entries.append("COMPLETE TRANSLATION ENTRIES:")
    dict_entries.append("="*40 + "\n")
    
    for entry in all_entries:
        dict_entries.append(entry)
        dict_entries.append("\n" + "-"*40 + "\n")
    
    # Get initial prompt
    initial_prompt = get_initial_translation_prompt(dict_entries, logging_mode=True)
    
    try:
        # Initial API call using LLM client
        initial_response_text = llm_client.complete(initial_prompt)
        
        # Log the initial response to a file
        log_to_file(initial_response_text, 'initial_response_log.txt', True)
        
        # Prepare follow-up prompt
        follow_up_phrase = get_follow_up_phrase(latinized, mkhedruli)
        grammar_text = get_grammar_text(grammar)
        
        # Create follow-up prompt
        follow_up_prompt = get_follow_up_prompt(
            follow_up_phrase, 
            grammar_text, 
            initial_response_text, 
            dict_entries,
            logging_mode=True
        )
        
        # Make follow-up API call using LLM client
        follow_up_response_text = llm_client.complete(follow_up_prompt)
        
        # Log the follow-up response to a file
        log_to_file(follow_up_response_text, 'followup_response_log.txt', True)
        
        # Extract Georgian and English translations
        georgian_translation, english_translation = extract_translations(follow_up_response_text)
        
        # Return translations
        return {
            'mingrelian_latinized': latinized,
            'mingrelian_mkhedruli': mkhedruli,
            'georgian': georgian_translation,
            'english': english_translation,
            'full_response': follow_up_response_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")

async def stream_translation(prompt_text, api_key, provider=None, model=None):
    """
    Stream translation progress with updates after first API call.
    Yields JSON events for progress updates.
    """
    # Use environment variables if provider/model not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
    if model is None:
        model = os.getenv("LLM_MODEL")
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Failed to initialize LLM client: {str(e)}'})}\n\n"
        return
    
    # Determine if input is mkhedruli or latinized
    if is_mkhedruli(prompt_text):
        mkhedruli = prompt_text
        latinized = mkhedruli_to_latinized(prompt_text)
    else:
        mkhedruli = latinized_to_mkhedruli(prompt_text)
        latinized = prompt_text
    
    # Load grammar for translation
    grammar_path = Path(__file__).parent.parent / 'data' / 'harris.txt'
    try:
        with open(grammar_path, 'r') as file:
            grammar = file.read()
    except FileNotFoundError:
        grammar = ""
    
    # Default dictionary file path
    dict_file = Path(__file__).parent.parent / 'data' / 'kajaia.txt'
    
    # Create a variable to collect all output similar to prompt.py
    dict_entries = []
    all_entries = []
    
    # Break the string into words
    words = latinized.split()
    
    # Lookup words in dictionary
    for word in words:
        entries = lookup_word(word, dict_file)
        all_entries.append({'word': word, 'entries': entries})
        if entries:
            dict_entries.extend(entries)
            print(f"Dict 1:  Latinized Mingrelian word: {word}")
    
    # Prepare initial prompt
    initial_prompt = get_initial_translation_prompt(dict_entries, logging_mode=True)
    
    try:
        # Initial API call using LLM client
        initial_response_text = llm_client.complete(initial_prompt)
        
        # ✨ FIRST API CALL COMPLETE - Send progress update
        yield f"data: {json.dumps({'progress': 50, 'message': 'First translation complete'})}\n\n"
        
        # Log the initial response to a file
        log_to_file(initial_response_text, 'initial_response_log.txt', True)
        
        # Prepare follow-up prompt
        follow_up_phrase = get_follow_up_phrase(latinized, mkhedruli)
        grammar_text = get_grammar_text(grammar)
        
        # Create follow-up prompt
        follow_up_prompt = get_follow_up_prompt(
            follow_up_phrase, 
            grammar_text, 
            initial_response_text, 
            dict_entries,
            logging_mode=True
        )
        
        # Make follow-up API call using LLM client
        follow_up_response_text = llm_client.complete(follow_up_prompt)
        
        # Log the follow-up response to a file
        log_to_file(follow_up_response_text, 'followup_response_log.txt', True)
        
        # Extract Georgian and English translations
        georgian_translation, english_translation = extract_translations(follow_up_response_text)
        
        # Send final result
        result = {
            'mingrelian_latinized': latinized,
            'mingrelian_mkhedruli': mkhedruli,
            'georgian': georgian_translation,
            'english': english_translation,
            'full_response': follow_up_response_text
        }
        yield f"data: {json.dumps({'result': result})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': f'API error: {str(e)}'})}\n\n"

@app.post("/chat")
async def chat(data: PromptIn):
    """
    Process a Mingrelian text and return translations with streaming progress.
    
    Parameters:
    - prompt: Text in Mingrelian (either latinized or mkhedruli script)
    - api_key: API key for the LLM provider
    - target_language: Language for translation (english or georgian)
    - provider: LLM provider to use ("openai" or "anthropic")
    - model: Optional model name (uses provider default if None)
    """
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt text is required")
    
    if not data.api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    
    if data.provider is not None and data.provider not in ["openai", "anthropic"]:
        raise HTTPException(status_code=400, detail="Provider must be 'openai' or 'anthropic'")
    
    # Return streaming response
    return StreamingResponse(
        stream_translation(data.prompt, data.api_key, data.provider, data.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    ) 