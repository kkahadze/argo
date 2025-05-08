#!/usr/bin/env python3
import sys
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Import our refactored modules
import config
from logger import get_logger
from search_engine import SearchEngine
from formatter import Formatter
from transliterate import latinized_to_mkhedruli, mkhedruli_to_latinized

# Load environment variables from .env file
load_dotenv()

# Get a logger for this module
logger = get_logger(__name__)

def is_mkhedruli(text: str) -> bool:
    """
    Check if text contains Georgian Mkhedruli script characters.
    
    Args:
        text: Text to check
    
    Returns:
        True if text contains Mkhedruli characters, False otherwise
    """
    # Georgian Mkhedruli Unicode range is approximately U+10D0 to U+10FF
    return bool(re.search('[\u10D0-\u10FF]', text))

def main() -> None:
    # Initialize our components
    formatter = Formatter()
    
    # Parse command line arguments for dictionary path
    dict_file = config.DEFAULT_DICTIONARY_PATH
    if len(sys.argv) > 1:
        dict_file = sys.argv[1]
        logger.info(f"Using custom dictionary file: {dict_file}")
    else:
        logger.info(f"Using default dictionary file: {dict_file}")
    
    # Initialize search engine with the dictionary path
    logger.debug("Initializing search engine")
    search_engine = SearchEngine(dict_file)
    
    # Take in a string from the user
    user_input = input("Enter a phrase in Mingrelian (latinized or Mkhedruli): ")
    logger.info(f"Received user input: {user_input}")
    
    # Break the string into words
    words = user_input.split()
    logger.debug(f"Split into {len(words)} words: {words}")
    
    # Store all complete entries for the final section
    all_entries: List[str] = []
    
    # Process each word
    for word in words:
        logger.info(f"Processing word: {word}")
        # Check if the word is in Mkhedruli script and convert if needed
        original_word = word
        if is_mkhedruli(word):
            logger.debug(f"Detected Mkhedruli script: {word}")
            latinized_word = mkhedruli_to_latinized(word)
            formatter.add_line(f"Input in Mkhedruli script: {word}")
            formatter.add_line(f"Converted to latinized form: {latinized_word}")
            word = latinized_word
        else:
            logger.debug(f"Word is in Latinized form: {word}")
            formatter.add_line(f"Latinized Mingrelian word: {word}")
            formatter.add_line(f"Mkhedruli Mingrelian word: {latinized_to_mkhedruli(word)}")
        
        # Search for the word using all strategies
        logger.debug(f"Starting search for word: {word}")
        best_result, all_results = search_engine.search(word)
        
        # If we found matches, format the result and add to entries
        if best_result.has_matches():
            logger.info(f"Found matches for '{word}' using strategy: {best_result.strategy_name}")
            formatter.format_search_result(best_result)
            
            # Add the matches to all_entries
            for match in best_result.matches:
                if 'entry_text' in match:
                    all_entries.append(match['entry_text'])
                else:
                    # Fallback for any match types that don't have entry_text
                    all_entries.append(f"Match for '{word}' found but no entry text available")
        else:
            # No matches found with any strategy
            logger.warning(f"No matches found for '{word}' with any strategy")
            formatter.add_line(f"No translation or references found for '{word}'")
            all_entries.append(f"No translation or references found for '{word}'")
    
    # Format the complete entries section
    logger.debug("Formatting complete entries section")
    formatter.format_all_entries_section(all_entries)
    
    # Get the formatted output
    output = formatter.get_output()
    
    # Print all collected output
    print(output)
    
    # Prepare prompt for the language model
    command = "Can you translate these entries from a Mingrelian-Georgian dictionary to me in English. don't translate the Mingrelian into English, just the Georgian.\n"
    prompt = command + "\n" + output
    
    # Process with OpenAI
    logger.info("Preparing to call OpenAI API")
    call_openai_api(prompt, user_input, output)

def call_openai_api(prompt: str, user_input: str, output: str) -> None:
    """
    Call the OpenAI API with the prepared prompt and handle responses
    
    Args:
        prompt: The initial prompt to send to OpenAI
        user_input: The original user input
        output: The formatted search output
    """
    # Import OpenAI here to avoid potential variable conflicts
    import openai
    
    # Set OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found in environment variables")
        print("ERROR: No OpenAI API key found. Make sure it's set in your .env file as OPENAI_API_KEY=your_key_here")
        return
    
    openai.api_key = api_key
    
    # Store LM response in a variable instead of printing it
    initial_response_text = ""
    
    try:
        # Try the newer OpenAI API format (1.0.0+)
        logger.debug(f"Attempting to call OpenAI API with model: {config.DEFAULT_MODEL}")
        try:
            response = openai.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            initial_response_text = response.choices[0].message.content
            logger.info("Successfully called newer OpenAI API")
        except AttributeError:
            # Fall back to older OpenAI API format (0.x)
            # Use 'text-davinci-003' which works with the Completion API
            logger.warning(f"Falling back to older OpenAI API with model: {config.FALLBACK_MODEL}")
            print("Using older OpenAI API with text-davinci-003 model...")
            response = openai.Completion.create(
                model=config.FALLBACK_MODEL,  # Use model that works with Completion API
                prompt=prompt,
                max_tokens=config.MAX_TOKENS
            )
            initial_response_text = response.choices[0].text.strip()
            logger.info("Successfully called older OpenAI API")
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
        print(f"Error calling OpenAI API: {e}")
        print("\nTIP: Consider updating your OpenAI package with: pip install --upgrade openai")
        return
    
    # Format and print the initial response
    formatted_response = formatter.format_api_response(initial_response_text)
    print(formatted_response)
    logger.debug("Displayed initial API response")
    
    # Send follow-up request
    logger.info("Preparing follow-up request to OpenAI API")
    send_followup_request(prompt, initial_response_text, user_input, formatter)

def send_followup_request(
    initial_prompt: str, 
    initial_response_text: str, 
    user_input: str, 
    formatter: Formatter
) -> None:
    """
    Send a follow-up request to the OpenAI API
    
    Args:
        initial_prompt: The initial prompt sent to OpenAI
        initial_response_text: The response from the initial prompt
        user_input: The original user input
        formatter: The formatter object for formatting responses
    """
    import openai
    
    # Create a follow-up prompt with all context
    follow_up_phrase = f"""
                        Now translate the following phrase into Georgian and then English
                        Phrase in Mingrelian (latinized): {user_input}
                        Phrase in Mingrelian (mkhedruli): {latinized_to_mkhedruli(user_input)}
                        """
    
    # Construct the follow-up message with all previous context
    follow_up_prompt = follow_up_phrase + "\n\nOriginal dictionary entries:\n" + initial_prompt + "\n\nInitial translation:\n" + initial_response_text
    
    # Send the follow-up prompt to the LM
    follow_up_response_text = ""
    try:
        # Try the newer OpenAI API format (1.0.0+)
        logger.debug("Attempting to send follow-up request using newer API")
        try:
            follow_up_response = openai.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[
                    {"role": "user", "content": initial_prompt},
                    {"role": "assistant", "content": initial_response_text},
                    {"role": "user", "content": follow_up_phrase}
                ]
            )
            follow_up_response_text = follow_up_response.choices[0].message.content
            logger.info("Successfully received follow-up response from newer API")
        except AttributeError:
            # Fall back to older OpenAI API format (0.x)
            logger.warning("Falling back to older API for follow-up request")
            follow_up_response = openai.Completion.create(
                model=config.FALLBACK_MODEL,
                prompt=follow_up_prompt,
                max_tokens=config.MAX_TOKENS
            )
            follow_up_response_text = follow_up_response.choices[0].text.strip()
            logger.info("Successfully received follow-up response from older API")
    except Exception as e:
        logger.error(f"Error calling OpenAI API for follow-up: {str(e)}", exc_info=True)
        print(f"Error calling OpenAI API for follow-up: {e}")
        return
    
    # Format and print the follow-up response
    print(formatter.format_api_response(follow_up_response_text, is_followup=True))
    logger.debug("Displayed follow-up API response")

if __name__ == "__main__":
    logger.info("Starting application")
    main()
    logger.info("Application completed") 