#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Check if we're running the script directly or importing it
if __name__ == "__main__":
    # When running directly from src, add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now both import styles will work
from src.translate import translate_lemma, search_containing_word, lemmatize_mingrelian, find_close_lemma_matches
from src.transliterate import latinized_to_mkhedruli, mkhedruli_to_latinized
from src.extract_definition import extract_definition
from src.prompts import (
    get_initial_translation_prompt,
    get_follow_up_phrase,
    get_grammar_text,
    get_after_phrase,
    get_follow_up_prompt,
    get_corpus_only_prompt,
    log_to_file,
    parse_json_response,
    format_translations_for_llm
)
from src.corpus_search import search_word_in_corpus
from src.llm_client import LLMClient, get_default_llm_client
import sys
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configuration - can be overridden by environment variables
# Set LLM_PROVIDER in .env to "openai" or "anthropic"
# Set LLM_MODEL in .env to specify which model to use
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # or "claude-sonnet-4-5-20250929" for Claude
LONG_CONTEXT_MODEL = os.getenv("LLM_LONG_CONTEXT_MODEL", DEFAULT_MODEL)

# ANSI color codes for terminal output
COLORS = {
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'MAGENTA': '\033[95m',
    'CYAN': '\033[96m',
    'WHITE': '\033[97m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'END': '\033[0m'
}

def colorize(text, color):
    """Apply color formatting to text for terminal display"""
    if color in COLORS:
        return f"{COLORS[color]}{text}{COLORS['END']}"
    return text

def is_mkhedruli(text):
    """
    Check if text contains Georgian Mkhedruli script characters.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if text contains Mkhedruli characters, False otherwise
    """
    # Georgian Mkhedruli Unicode range is approximately U+10D0 to U+10FF
    return bool(re.search('[\u10D0-\u10FF]', text))

def extract_translations(response_text):
    """
    Extract Georgian and English translations from the model's response.
    
    Args:
        response_text (str): The model's response text
        
    Returns:
        tuple: (georgian_translation, english_translation)
    """
    # Extract Georgian translation using regex
    georgian_match = re.search(r'Georgian:\s*(.*?)(?:\n|$)', response_text)
    georgian_translation = georgian_match.group(1).strip() if georgian_match else ""
    
    # Extract English translation using regex
    english_match = re.search(r'English:\s*(.*?)(?:\n|$)', response_text)
    english_translation = english_match.group(1).strip() if english_match else ""
    
    return georgian_translation, english_translation

def main(terminal_logging=True, dict_file=None):
    # Default dictionary file path if not provided
    if dict_file is None:
        dict_file = str(Path(__file__).parent.parent / 'data' / 'kajaia.txt')
    
    # Create a variable to collect all output
    dict_entries = []
    
    # Take in a string from the user
    user_input = input("Enter a phrase in Mingrelian (latinized or Mkhedruli): ")
    
    # Break the string into words
    words = user_input.split()

    # check if first word is mkhedruli or latinized
    if is_mkhedruli(words[0]):
        mkhedruli = user_input
        latinized = mkhedruli_to_latinized(user_input)
    else:
        mkhedruli = latinized_to_mkhedruli(user_input)
        latinized = user_input

    
    # Store all complete entries
    all_entries = []
    
    # CORPUS SEARCH FIRST: Search corpus before dictionary
    corpus_file = str(Path(__file__).parent.parent / 'data' / 'en_to_xmf.json')
    all_corpus_results = []
    words_needing_dict_search = []  # Track which words need dictionary lookup
    
    if terminal_logging:
        print("\n" + "="*60)
        print("STEP 1: SEARCHING PARALLEL CORPUS (en_to_xmf.json)")
        print("="*60)
    
    for word in words:
        # Convert to Mkhedruli for corpus search
        original_word = word
        if is_mkhedruli(word):
            search_word = word
            word_for_display = word
        else:
            search_word = latinized_to_mkhedruli(word)
            word_for_display = word
        
        # Search corpus
        corpus_results = search_word_in_corpus(corpus_file, search_word)
        
        # Check if we have good corpus matches (exact or word_in_phrase)
        good_matches = [r for r in corpus_results if r['match_type'] in ['exact_match', 'word_in_phrase']]
        
        if good_matches:
            # We have good corpus data - no need for dictionary
            all_corpus_results.extend(good_matches)
            if terminal_logging:
                print(f"\n✓ Word: '{word_for_display}' ({search_word})")
                print(f"  Found {len(good_matches)} high-quality corpus match(es) - SKIPPING dictionary")
                for result in good_matches:
                    print(f"    - {result['match_type']}: {result['mingrelian'][:50]}... → {result['english'][:50]}...")
        else:
            # No good corpus matches - will need dictionary
            words_needing_dict_search.append(word)
            if corpus_results:
                # Only fuzzy matches found - add them but still search dictionary
                all_corpus_results.extend(corpus_results)
                if terminal_logging:
                    print(f"\n⚠ Word: '{word_for_display}' ({search_word})")
                    print(f"  Only fuzzy corpus matches - WILL search dictionary")
            else:
                if terminal_logging:
                    print(f"\n✗ Word: '{word_for_display}' ({search_word})")
                    print(f"  No corpus matches - WILL search dictionary")
    
    if terminal_logging:
        print("="*60)
        print(f"Corpus: {len(all_corpus_results)} matches found")
        print(f"Dictionary: {len(words_needing_dict_search)} words need lookup")
        print("="*60 + "\n")
    
    # DICTIONARY SEARCH: Only for words without good corpus matches
    if terminal_logging and words_needing_dict_search:
        print("STEP 2: SEARCHING DICTIONARY (kajaia.txt)")
        print("="*60)
    
    # Process each word that needs dictionary lookup
    for word in words_needing_dict_search:
        # Check if the word is in Mkhedruli script and convert if needed
        original_word = word
        if is_mkhedruli(word):
            latinized_word = mkhedruli_to_latinized(word)
            dict_entries.append(f"Input in Mkhedruli script: {word}")
            dict_entries.append(f"Converted to latinized form: {latinized_word}")
            word = latinized_word
        else:
            dict_entries.append(f"Latinized Mingrelian word: {word}")
        
        # Get translation results for this word
        results = translate_lemma(dict_file, word)
        # print("RESULTS: ", results)
        # dict_entries.append(str(results))
        
        # Show Mkhedruli form - only if input was originally latinized
        if not is_mkhedruli(original_word):
            dict_entries.append(f"Mkhedruli Mingrelian word: {latinized_to_mkhedruli(word)}")
        
        if len(results) == 1 and results[0][1] is None:
            # No direct translation found, try hyphenated form with "-i" first (higher priority)
            hyphenated_tried = False
            if (not word.startswith('-') and not '-' in word and 
                word[-1] not in ['a', 'i', 'e', 'o', 'u', 'h', 's']):
                hyphenated_word = word + '-i'
                dict_entries.append(f"Trying hyphenated form: '{hyphenated_word}'")
                hyphenated_results = translate_lemma(dict_file, hyphenated_word)
                hyphenated_tried = True
                
                # If we found results with the hyphenated form
                if not (len(hyphenated_results) == 1 and hyphenated_results[0][1] is None):
                    dict_entries.append(f"Found translation for hyphenated form '{hyphenated_word}'")
                    for curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word in hyphenated_results:
                        if entry_text:
                            # Store the complete entry text
                            all_entries.append(f"Match for hyphenated form '{hyphenated_word}' of '{word}':\n{entry_text}")
                        else:
                            # If no entry found
                            all_entries.append(f"No translation found for hyphenated form '{hyphenated_word}' of '{word}'")
                    continue  # Skip to next word since we found a match with the hyphenated form
            
            # If still no results, try with lemmatized form (morphological rules)
            lemmatized_word = lemmatize_mingrelian(word)
            if lemmatized_word != word:
                dict_entries.append(f"Trying lemmatized form (morphological): '{lemmatized_word}'")
                lemma_results = translate_lemma(dict_file, lemmatized_word)
                
                # If we found results with the lemmatized form
                if not (len(lemma_results) == 1 and lemma_results[0][1] is None):
                    dict_entries.append(f"Found translation for lemmatized form '{lemmatized_word}' (e.g., '{word}' with suffix removed/modified)")
                    for curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word in lemma_results:
                        if entry_text:
                            # Store the complete entry text
                            all_entries.append(f"Match for lemmatized form '{lemmatized_word}' of '{word}':\n{entry_text}")
                        else:
                            # If no entry found
                            all_entries.append(f"No translation found for lemmatized form '{lemmatized_word}' of '{word}'")
                    continue  # Skip to next word since we found a match with the lemmatized form
            
            # NEW PRIORITY: Search for actual occurrences in the text BEFORE fuzzy matching
            # Transliterate to Georgian
            georgian_word = latinized_to_mkhedruli(word)
            dict_entries.append(f"\nNo exact/morphological match found for '{word}'. Searching for occurrences in text...")
            dict_entries.append(f"'{word}' transliterates to: '{georgian_word}'")
            
            # Search for occurrences of this word in the dictionary
            containing_results = search_containing_word(dict_file, word)
            
            if containing_results:
                # Limit to at most 5 matches
                max_matches = 5
                total_matches = len(containing_results)
                limited_results = containing_results[:max_matches]
                
                dict_entries.append(f"Found {total_matches} entries containing '{georgian_word}' (showing up to {max_matches})")
                for entry_lemma, context, entry_text in limited_results:
                    all_entries.append(f"Partial match for '{word}' ({georgian_word}):\nEntry: {entry_lemma}\nContext: {context}\n\n{entry_text}")
                
                # If there were more than max_matches, add a note
                if total_matches > max_matches:
                    all_entries.append(f"Note: {total_matches - max_matches} additional partial matches for '{word}' were found but not displayed.")
                
                continue  # Skip to next word since we found partial matches
            
            # LAST RESORT: Fuzzy matching (edit distance) - only if no occurrences found
            dict_entries.append(f"No occurrences found in text. Trying fuzzy matching (edit distance of 1) as last resort...")
            close_matches = find_close_lemma_matches(dict_file, word, max_distance=1)
            
            if close_matches:
                dict_entries.append(f"Found {len(close_matches)} similar lemmas with small differences")
                
                # Process up to 3 close matches
                max_similar_matches = 3
                limited_matches = close_matches[:max_similar_matches]
                
                for lemma, entry_text in limited_matches:
                    # Extract definition from the entry text
                    curr_lemma, definition, curr_entry_text = extract_definition(entry_text)
                    
                    # Show similarity information
                    dict_entries.append(f"Similar lemma: '{lemma}' (possibly a spelling variant or related form)")
                    
                    # Add the entry to results
                    all_entries.append(f"Close match for '{word}':\nLemma: {lemma}\n\n{entry_text}")
                
                # Add note if we limited the number of matches
                if len(close_matches) > max_similar_matches:
                    all_entries.append(f"Note: {len(close_matches) - max_similar_matches} additional similar matches for '{word}' were found but not displayed.")
                
                continue  # Skip to next word since we found close matches
            
            # VERY LAST RESORT: For longer words (>7 letters), try with edit distance of 2
            if len(word) > 7:
                dict_entries.append(f"No matches with edit distance 1. Since '{word}' is longer than 7 letters, trying with edit distance 2...")
                close_matches_dist2 = find_close_lemma_matches(dict_file, word, max_distance=2)
                
                if close_matches_dist2:
                    dict_entries.append(f"Found {len(close_matches_dist2)} similar lemmas with edit distance of 2")
                    
                    # Process up to 3 close matches
                    max_similar_matches = 3
                    limited_matches = close_matches_dist2[:max_similar_matches]
                    
                    for lemma, entry_text in limited_matches:
                        # Extract definition from the entry text
                        curr_lemma, definition, curr_entry_text = extract_definition(entry_text)
                        
                        # Show similarity information
                        dict_entries.append(f"Similar lemma (edit distance 2): '{lemma}' (possibly a spelling variant or related form)")
                        
                        # Add the entry to results
                        all_entries.append(f"Close match (edit distance 2) for '{word}':\nLemma: {lemma}\n\n{entry_text}")
                    
                    # Add note if we limited the number of matches
                    if len(close_matches_dist2) > max_similar_matches:
                        all_entries.append(f"Note: {len(close_matches_dist2) - max_similar_matches} additional similar matches with edit distance 2 for '{word}' were found but not displayed.")
                    
                    continue  # Skip to next word since we found close matches with distance 2
            
            # If we get here, absolutely nothing was found
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
    dict_entries.append("DICTIONARY ENTRIES:")
    
    for entry in all_entries:
        dict_entries.append(entry)
        dict_entries.append("\n" + "-"*40 + "\n")

    # Print all collected dict_entries if terminal logging is enabled
    if terminal_logging:
        print("\n".join(dict_entries))

    # Check if we need to call first LLM at all
    has_dictionary_entries = len(all_entries) > 0
    translations_json = None
    initial_response_text = ""  # Initialize to empty string
    
    if has_dictionary_entries:
        # We have dictionary entries that need translation
        # Get initial prompt using the new function
        initial_prompt = get_initial_translation_prompt(dict_entries, logging_mode=True)
        
        # File logging is always enabled but terminal logging can be disabled
        if terminal_logging:
            print(f"\nSTEP 3: FIRST LLM CALL (Translate dictionary entries)")
            print("="*60)
            print(f"Initial prompt logged to initial_prompt_log.txt")
        
        # Initialize LLM client
        try:
            llm_client = get_default_llm_client(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)
        except Exception as e:
            print(f"ERROR: Failed to initialize LLM client: {e}")
            print("\nTIP: Make sure you have set the appropriate API key in your .env file:")
            print("  - For OpenAI: OPENAI_API_KEY=your_key_here")
            print("  - For Claude: ANTHROPIC_API_KEY=your_key_here")
            print("\nYou can also set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in .env")
            return
        
        # Store LM response in a variable instead of printing it
        initial_response_text = ""
        
        try:
            # Call the LLM with the initial prompt
            initial_response_text = llm_client.complete(initial_prompt)
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            print(f"\nTIP: Check your {DEFAULT_PROVIDER.upper()} API key and internet connection")
            return
        
        # Log the raw JSON response to a file
        log_to_file(initial_response_text, 'initial_response_log.txt', True)
        
        # Parse JSON response
        translations_json = parse_json_response(initial_response_text)
    else:
        # All words had good corpus matches - skip first LLM call!
        if terminal_logging:
            print(f"\n✓✓ All words found in corpus with high-quality matches!")
            print(f"   SKIPPING first LLM call (no dictionary entries to translate)")
            print("="*60 + "\n")
    
    # Log corpus results (already collected earlier)
    if all_corpus_results:
        corpus_log = "\n".join([
            f"Word: {r['search_word']}\n"
            f"Match Type: {r['match_type']}\n"
            f"Mingrelian: {r['mingrelian']}\n"
            f"English: {r['english']}\n"
            + "-"*60
            for r in all_corpus_results
        ])
        log_to_file(corpus_log, 'corpus_matches.txt', True)
    
    if translations_json is None and has_dictionary_entries:
        # Only warn if we called LLM but parsing failed
        print("WARNING: Failed to parse JSON from LLM response. Using raw text as fallback.")
        if terminal_logging:
            print("Initial Translation (raw): \n" + initial_response_text)
        formatted_translations = initial_response_text  # Fallback to raw text
    elif translations_json is not None:
        # Convert JSON to readable format for second LLM, including corpus results
        formatted_translations = format_translations_for_llm(translations_json, all_corpus_results)
        
        # Also log the parsed JSON for debugging
        log_to_file(str(translations_json), 'parsed_translations.json', True)
        
        if terminal_logging:
            print("Initial Translation (parsed and formatted):\n" + formatted_translations)
    else:
        # No dictionary entries - use only corpus results
        formatted_translations = format_translations_for_llm({}, all_corpus_results)
        if terminal_logging:
            print("Formatted corpus translations:\n" + formatted_translations)
    
    # Get follow-up phrase using the new function
    follow_up_phrase = get_follow_up_phrase(latinized, mkhedruli)

    # Choose prompt type based on whether we have dictionary entries
    if has_dictionary_entries:
        # Full prompt with grammar (we have dictionary data to analyze)
        if terminal_logging:
            print("\nSTEP 4: SECOND LLM CALL (Full synthesis with grammar)")
            print("="*60)
        
        # import harris.txt using absolute path
        harris_path = Path(__file__).parent.parent / 'data' / 'harris.txt'
        with open(harris_path, 'r') as file:
            grammar = file.read()

        # Get grammar text using the new function
        grammar_text = get_grammar_text(grammar)

        # Get follow-up prompt using the formatted translations (not raw JSON)
        follow_up_prompt = get_follow_up_prompt(
            follow_up_phrase, 
            grammar_text, 
            formatted_translations,  # Use formatted text instead of raw JSON
            dict_entries,
            logging_mode=True
        )
    else:
        # Simplified prompt (corpus-only, just need Georgian equivalent)
        if terminal_logging:
            print("\nSTEP 4: SECOND LLM CALL (Simplified - corpus only, getting Georgian)")
            print("="*60)
        
        follow_up_prompt = get_corpus_only_prompt(
            follow_up_phrase,
            formatted_translations,
            logging_mode=True
        )
    
    # Terminal logging message is handled inside the get_follow_up_prompt function
    # but we'll respect the terminal_logging setting
    if not terminal_logging:
        print(f"Working on translation...")
    
    # Initialize LLM client for follow-up (may use different model for long context)
    try:
        follow_up_client = get_default_llm_client(provider=DEFAULT_PROVIDER, model=LONG_CONTEXT_MODEL)
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client for follow-up: {e}")
        return
        
    # Send the follow-up prompt to the LM
    follow_up_response_text = ""
    try:
        follow_up_response_text = follow_up_client.complete(follow_up_prompt)
    except Exception as e:
        print(f"Error calling LLM API for follow-up: {e}")
        return
    
    # Log the follow-up response to a file (always)
    log_to_file(follow_up_response_text, 'followup_response_log.txt', True)
    
    if terminal_logging:
        print(f"Follow-up response logged to followup_response_log.txt")
    
    # Extract Georgian and English translations
    georgian_translation, english_translation = extract_translations(follow_up_response_text)

    # Store translations in a dictionary for easy access (useful for web apps)
    translations = {
        'mingrelian_latinized': latinized,
        'mingrelian_mkhedruli': mkhedruli,
        'georgian': georgian_translation,
        'english': english_translation,
        'full_response': follow_up_response_text
    }
    
    # Print the follow-up response with color formatting (always)
    print(f"\n{colorize('Follow-up Response:', 'RED')}")
    
    # Process the follow-up response line by line to apply formatting (always)
    for line in follow_up_response_text.split('\n'):
        # Apply special formatting to key elements
        if line.startswith('Georgian:') or line.startswith('English:'):
            # Highlight Georgian and English translations in red
            print(colorize(line, 'RED'))
        elif "Phrase in Mingrelian" in line or "Translation to" in line:
            # Bold headings
            print(colorize(line, 'CYAN'))
        elif line.startswith('**') and line.endswith('**'):
            # Bold text that's already marked with Markdown
            print(colorize(line, 'CYAN'))
        elif "Note:" in line:
            # Highlight notes
            print(colorize(line, 'YELLOW'))
        else:
            # Regular text
            print(colorize(line, 'CYAN'))
    
    # Return the translations dictionary for use in a web app
    return translations

if __name__ == "__main__":
    # Parse command line arguments
    dict_file = None
    quiet_mode = False
    
    # Process command line arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ['-q', '--quiet', '-quiet']:
            quiet_mode = True
        elif not arg.startswith('-'):
            # If it doesn't start with a dash, treat it as a dictionary file
            dict_file = arg
        i += 1
    
    # Run main function with parsed arguments
    translations = main(terminal_logging=not quiet_mode, dict_file=dict_file)
    
    # Example of how to access translations in another script or web app
    if translations:
        # Print the extracted translations to demonstrate usage
        if not quiet_mode:
            print("\nExtracted translations for web app usage:")
            print(f"Mingrelian (latinized): {translations['mingrelian_latinized']}")
            print(f"Mingrelian (mkhedruli): {translations['mingrelian_mkhedruli']}")
            print(f"Georgian: {translations['georgian']}")
            print(f"English: {translations['english']}") 