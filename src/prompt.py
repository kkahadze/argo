#!/usr/bin/env python3
from translate import translate_lemma, search_containing_word, lemmatize_mingrelian, find_close_lemma_matches
from transliterate import latinized_to_mkhedruli, mkhedruli_to_latinized
from extract_definition import extract_definition
from prompts import (
    get_initial_translation_prompt,
    get_follow_up_phrase,
    get_grammar_text,
    get_after_phrase,
    get_follow_up_prompt,
    log_to_file
)
import sys
import os
import re
from dotenv import load_dotenv
import openai

MODEL = 'o4-mini'
# GPT 4.1 for context
LONG_CONTEXT_MODEL = 'o4-mini'
# Load environment variables from .env file
load_dotenv()

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

def main(terminal_logging=True, dict_file=None):
    # Default dictionary file path if not provided
    if dict_file is None:
        dict_file = '/Users/konstantinekahadze/Desktop/argo/kajaia.txt'
    
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
    
    # Process each word
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
        
        # Get translation results for this word
        results = translate_lemma(dict_file, word)
        # print("RESULTS: ", results)
        # dict_entries.append(str(results))
        
        # Show Mkhedruli form - only if input was originally latinized
        if not is_mkhedruli(original_word):
            dict_entries.append(f"Mkhedruli Mingrelian word: {latinized_to_mkhedruli(word)}")
        
        if len(results) == 1 and results[0][1] is None:
            # No direct translation found, try with lemmatized form
            lemmatized_word = lemmatize_mingrelian(word)
            if lemmatized_word != word:
                dict_entries.append(f"Trying lemmatized form: '{lemmatized_word}'")
                lemma_results = translate_lemma(dict_file, lemmatized_word)
                
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
                
                # NEW: If the lemmatized form has no exact matches, try similar lemmas to the lemmatized form
                dict_entries.append(f"Looking for similar lemmas (edit distance of 1) to lemmatized form '{lemmatized_word}'...")
                lemma_close_matches = find_close_lemma_matches(dict_file, lemmatized_word, max_distance=1)
                
                if lemma_close_matches:
                    dict_entries.append(f"Found {len(lemma_close_matches)} similar lemmas with small differences to lemmatized form")
                    
                    # Process up to 3 close matches
                    max_similar_matches = 3
                    limited_matches = lemma_close_matches[:max_similar_matches]
                    
                    for lemma, entry_text in limited_matches:
                        # Extract definition from the entry text
                        curr_lemma, definition, curr_entry_text = extract_definition(entry_text)
                        
                        # Show similarity information
                        dict_entries.append(f"Similar lemma to lemmatized form: '{lemma}' (possibly a spelling variant or related form)")
                        
                        # Add the entry to results
                        all_entries.append(f"Close match for lemmatized form '{lemmatized_word}' of '{word}':\nLemma: {lemma}\n\n{entry_text}")
                    
                    # Add note if we limited the number of matches
                    if len(lemma_close_matches) > max_similar_matches:
                        all_entries.append(f"Note: {len(lemma_close_matches) - max_similar_matches} additional similar matches for lemmatized form '{lemmatized_word}' were found but not displayed.")
                    
                    continue  # Skip to next word since we found close matches to the lemmatized form
                
                # NEW: For longer lemmatized words (>5 letters), try with edit distance of 2 if no matches found with distance 1
                if len(lemmatized_word) > 7:
                    dict_entries.append(f"No matches with edit distance 1. Since lemmatized form '{lemmatized_word}' is longer than 5 letters, trying with edit distance 2...")
                    lemma_matches_dist2 = find_close_lemma_matches(dict_file, lemmatized_word, max_distance=2)
                    
                    if lemma_matches_dist2:
                        dict_entries.append(f"Found {len(lemma_matches_dist2)} similar lemmas with edit distance of 2 to lemmatized form")
                        
                        # Process up to 3 close matches
                        max_similar_matches = 3
                        limited_matches = lemma_matches_dist2[:max_similar_matches]
                        
                        for lemma, entry_text in limited_matches:
                            # Extract definition from the entry text
                            curr_lemma, definition, curr_entry_text = extract_definition(entry_text)
                            
                            # Show similarity information
                            dict_entries.append(f"Similar lemma (edit distance 2) to lemmatized form: '{lemma}' (possibly a spelling variant or related form)")
                            
                            # Add the entry to results
                            all_entries.append(f"Close match (edit distance 2) for lemmatized form '{lemmatized_word}' of '{word}':\nLemma: {lemma}\n\n{entry_text}")
                        
                        # Add note if we limited the number of matches
                        if len(lemma_matches_dist2) > max_similar_matches:
                            all_entries.append(f"Note: {len(lemma_matches_dist2) - max_similar_matches} additional similar matches with edit distance 2 for lemmatized form '{lemmatized_word}' were found but not displayed.")
                        
                        continue  # Skip to next word since we found close matches with distance 2 to the lemmatized form
            
            # NEW STEP: Try to find lemmas with an edit distance of 1 to the original word
            dict_entries.append(f"Looking for similar lemmas (edit distance of 1) to '{word}'...")
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
            
            # NEW: For longer words (>5 letters), try with edit distance of 2 if no matches found with distance 1
            if len(word) > 7:
                dict_entries.append(f"No matches with edit distance 1. Since '{word}' is longer than 5 letters, trying with edit distance 2...")
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
            
            # If we get here, no exact, lemmatized, or similar matches were found
            # Now try search_containing_word as a last resort
            dict_entries.append(f"\nNo direct or similar match in dictionary found for '{word}'. Searching for occurrences...")
            
            # Transliterate to Georgian
            georgian_word = latinized_to_mkhedruli(word)
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
    
    # Print all collected dict_entries if terminal logging is enabled
    if terminal_logging:
        print("\n".join(dict_entries))

    # Get initial prompt using the new function
    initial_prompt = get_initial_translation_prompt(dict_entries, logging_mode=True)
    
    # File logging is always enabled but terminal logging can be disabled
    if terminal_logging:
        print(f"Initial prompt logged to initial_prompt_log.txt")
    
    # Set OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OpenAI API key found. Make sure it's set in your .env file as OPENAI_API_KEY=your_key_here")
        return
    
    openai.api_key = api_key
    
    # Store LM response in a variable instead of printing it
    initial_response_text = ""
    
    try:
        # Try the newer OpenAI API format (1.0.0+)
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": initial_prompt}]
            )
            initial_response_text = response.choices[0].message.content
        except AttributeError:
            # Fall back to older OpenAI API format (0.x)
            # Use 'text-davinci-003' which works with the Completion API
            if terminal_logging:
                print("Using older OpenAI API with text-davinci-003 model...")
            response = openai.Completion.create(
                model="text-davinci-003",  # Use model that works with Completion API
                prompt=initial_prompt,
                max_tokens=2000
            )
            initial_response_text = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        print("\nTIP: Consider updating your OpenAI package with: pip install --upgrade openai")
        return
    
    # Log the initial response to a file
    log_to_file(initial_response_text, 'initial_response_log.txt', True)
    
    # Print the initial response if terminal logging is enabled
    if terminal_logging:
        print("Initial Translation: \n" + initial_response_text)
    
    # Get follow-up phrase using the new function
    follow_up_phrase = get_follow_up_phrase(latinized, mkhedruli)
    
    # import harris.txt 
    with open('/Users/konstantinekahadze/Desktop/argo/harris.txt', 'r') as file:
        grammar = file.read()

    # Get grammar text using the new function
    grammar_text = get_grammar_text(grammar)

    # Get after phrase using the new function
    after_phrase = get_after_phrase(latinized, mkhedruli)

    # Get follow-up prompt using the new function (log to file but respect terminal_logging)
    follow_up_prompt = get_follow_up_prompt(
        follow_up_phrase, 
        grammar_text, 
        initial_response_text, 
        dict_entries,
        logging_mode=True
    )
    
    # Terminal logging message is handled inside the get_follow_up_prompt function
    # but we'll respect the terminal_logging setting
    if not terminal_logging:
        print(f"Working on translation...")
        
    # Send the follow-up prompt to the LM
    follow_up_response_text = ""
    try:
        # Try the newer OpenAI API format (1.0.0+)
        try:
            follow_up_response = openai.chat.completions.create(
                model=LONG_CONTEXT_MODEL,
                messages=[
                    {"role": "user", "content": follow_up_prompt},
                ]
            )
            follow_up_response_text = follow_up_response.choices[0].message.content
        except AttributeError:
            # Fall back to older OpenAI API format (0.x)
            if terminal_logging:
                print("Using older OpenAI API with text-davinci-003 model...")
            follow_up_response = openai.Completion.create(
                model="text-davinci-003",
                prompt=follow_up_prompt,
                max_tokens=2000
            )
            follow_up_response_text = follow_up_response.choices[0].text.strip()
    except Exception as e:
        print(f"Error calling OpenAI API for follow-up: {e}")
        return
    
    # Log the follow-up response to a file (always)
    log_to_file(follow_up_response_text, 'followup_response_log.txt', True)
    
    if terminal_logging:
        print(f"Follow-up response logged to followup_response_log.txt")
    
    # Print the follow-up response with color formatting (always)
    print(f"\n{colorize('Follow-up Response:', 'CYAN')}")
    
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
    main(terminal_logging=not quiet_mode, dict_file=dict_file) 