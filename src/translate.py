#!/usr/bin/env python3
import sys
import argparse
import re
import os

# Check if we're running the script directly or importing it
if __name__ == "__main__":
    # When running directly from src, add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract_definition import extract_definition
from src.transliterate import latinized_to_mkhedruli

def normalize_for_comparison(text):
    """
    Normalize text by removing special characters except hyphens for better comparison.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text with special characters removed but hyphens preserved
    """
    # Remove special characters but preserve hyphens
    return re.sub(r'[^\w\s\-]', '', text)

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance (edit distance) between two strings.
    Ignores special characters like curly braces.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        int: Edit distance between s1 and s2
    """
    # Normalize strings by removing special characters
    s1_normalized = normalize_for_comparison(s1)
    s2_normalized = normalize_for_comparison(s2)
    
    if len(s1_normalized) < len(s2_normalized):
        return levenshtein_distance(s2_normalized, s1_normalized)
    
    if len(s2_normalized) == 0:
        return len(s1_normalized)
    
    previous_row = range(len(s2_normalized) + 1)
    for i, c1 in enumerate(s1_normalized):
        current_row = [i + 1]
        for j, c2 in enumerate(s2_normalized):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_close_lemma_matches(file_path, search_term, max_distance=1):
    """
    Find lemmas in the dictionary that are close to the search term (edit distance <= max_distance).
    
    Args:
        file_path (str): Path to the dictionary file
        search_term (str): Word to search for
        max_distance (int): Maximum edit distance allowed
        
    Returns:
        list: List of lemma entry tuples with (lemma, entry_text)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lemmas = []
            current_lemma = None
            current_entry = []
            
            for line in file:
                # Start of a new lemma
                if line.strip().startswith('Lemma:'):
                    # Save the previous entry if there was one
                    if current_lemma and current_entry:
                        lemmas.append((current_lemma, '\n'.join(current_entry)))
                    
                    # Extract the new lemma
                    current_lemma = line.replace('Lemma:', '').strip()
                    current_entry = [line.strip()]
                # Continue recording the current entry
                elif current_entry:  # Only append if we've started a lemma
                    current_entry.append(line.strip())
            
            # Add the last entry
            if current_lemma and current_entry:
                lemmas.append((current_lemma, '\n'.join(current_entry)))
            
            # Now find lemmas with edit distance <= max_distance
            close_matches = []
            search_term_lower = search_term.lower()
            
            for lemma, entry_text in lemmas:
                # Keep hyphens when comparing to preserve distinction
                lemma_lower = lemma.lower()
                
                # Compute edit distance between the search term and the lemma
                distance = levenshtein_distance(search_term_lower, lemma_lower)
                
                # Check if the lemma is within the specified edit distance
                if distance <= max_distance:
                    close_matches.append((lemma, entry_text))
                    
                # Check if lemma without numbers matches (e.g., "mu" vs "mu1")
                base_lemma = re.sub(r'[0-9]+$', '', lemma_lower)
                if (base_lemma != lemma_lower and 
                    levenshtein_distance(search_term_lower, base_lemma) <= max_distance):
                    close_matches.append((lemma, entry_text))
            
            return close_matches
                
    except FileNotFoundError:
        error_msg = f"Error: Dictionary file '{file_path}' not found."
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    return []

def lemmatize_mingrelian(word):
    """
    Lemmatize a Mingrelian word to its base form.
    Handles various inflection patterns in Mingrelian including hyphenated forms.
    
    Args:
        word (str): Word to lemmatize
        
    Returns:
        str: Lemmatized form of the word
    """
    # If it's an affix (starts with a dash), keep it as is
    if word.startswith('-'):
        return word
    
    # Check if the word already has a word-internal hyphen
    if '-' in word:
        # If it's already in a form like "zhir-i", keep it as is
        return word
    
    # Standard lemmatization rules
    if word.endswith('s'):
        return word[:-1]
    elif word.endswith('ep'):
        return word[:-3] + 'i'
    elif word.endswith('sh'):
        return word[:-1]
    elif word.endswith('sha'):
        return word[:-2]
    # For words ending in consonants, generate both forms
    elif word[-1] not in ['a', 'i', 'e', 'o', 'u', 'h', 's']:
        # This will be matched by the modified find_lemma_entry
        return word + 'i'
    
    # Return the word unchanged for other cases
    return word

def find_lemma_entry(file_path, search_term):
    """
    Find lemma entries in the dictionary and return them.
    
    Args:
        file_path (str): Path to the dictionary file
        search_term (str): Lemma to search for
    
    Returns:
        list: List of entry texts if found, empty list otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            entries = []
            current_entry = []
            recording = False
            
            for line in file:
                # Start of a new lemma
                if line.strip().startswith('Lemma:'):
                    # If we were recording an entry, save it as we've reached the next lemma
                    if recording and current_entry:
                        entries.append('\n'.join(current_entry))
                    
                    # Check if this is the lemma we're looking for
                    current_lemma = line.replace('Lemma:', '').strip()
                    
                    # Skip comparison if current_lemma starts with a hyphen and search_term doesn't
                    if current_lemma.startswith('-') and not search_term.startswith('-'):
                        recording = False
                        current_entry = []
                        continue
                    
                    # Do exact matches with hyphen preserved
                    is_exact_match = search_term.lower() == current_lemma.lower()
                    
                    # Check if the lemma starts with our search term followed by a number
                    is_base_lemma_match = re.match(rf'^{re.escape(search_term.lower())}[0-9]+$', current_lemma.lower())
                    
                    # Check for variant forms (e.g., "chkim-i" matches "chkim-i//chke'm-i")
                    is_variant_match = False
                    if '//' in current_lemma:
                        # Split by // and check if any variant matches
                        variants = current_lemma.lower().split('//')
                        is_variant_match = any(search_term.lower() == variant.strip() for variant in variants)
                    
                    # Check for hyphenated forms (e.g., "zhiri" matches "zhir-i")
                    is_hyphenated_match = False
                    # If search_term ends with 'i'
                    if search_term.lower().endswith('i') and len(search_term) > 2:
                        base = search_term.lower()[:-1]  # Remove the 'i'
                        hyphenated = f"{base}-i"
                        is_hyphenated_match = hyphenated == current_lemma.lower()
                        # Also check against variants if present
                        if '//' in current_lemma and not is_hyphenated_match:
                            variants = current_lemma.lower().split('//')
                            is_hyphenated_match = any(hyphenated == variant.strip() for variant in variants)
                    
                    # Only check dehyphenated match if current_lemma doesn't start with hyphen
                    is_dehyphenated_match = False
                    if not current_lemma.startswith('-'):
                        is_dehyphenated_match = current_lemma.lower().replace('-', '') == search_term.lower()
                    
                    if is_exact_match or is_base_lemma_match or is_variant_match or is_hyphenated_match or is_dehyphenated_match:
                        recording = True
                        current_entry = [line.strip()]
                    else:
                        recording = False
                        current_entry = []
                
                # Continue recording the current entry
                elif recording:
                    current_entry.append(line.strip())
            
            # Add the last entry if we were recording
            if recording and current_entry:
                entries.append('\n'.join(current_entry))
            
            return entries
            
    except FileNotFoundError:
        error_msg = f"Error: Dictionary file '{file_path}' not found."
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    return []

def find_entries_containing_word(file_path, georgian_word):
    """
    Find entries in the dictionary that contain a specific Georgian word.
    
    Args:
        file_path (str): Path to the dictionary file
        georgian_word (str): Georgian word to search for
    
    Returns:
        list: List of entry texts containing the word
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            entries = []
            current_entry = []
            in_lemma = False
            
            for line in file:
                # Start of a new lemma
                if line.strip().startswith('Lemma:'):
                    # If we were recording an entry, check if it contains our word
                    if current_entry:
                        entry_text = '\n'.join(current_entry)
                        if georgian_word in entry_text:
                            entries.append(entry_text)
                    
                    # Start recording the new entry
                    current_entry = [line.strip()]
                    in_lemma = True
                
                # Continue recording the current entry
                elif in_lemma:
                    current_entry.append(line.strip())
            
            # Check the last entry in the file
            if current_entry:
                entry_text = '\n'.join(current_entry)
                if georgian_word in entry_text:
                    entries.append(entry_text)
                
            return entries
                
    except FileNotFoundError:
        error_msg = f"Error: Dictionary file '{file_path}' not found."
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    return []

def translate_lemma(file_path, lemma, debug_output=False):
    """
    Translate a Mingrelian lemma to Georgian.
    
    Args:
        file_path (str): Path to the dictionary file
        lemma (str): Lemma to translate
        debug_output (bool): Whether to print debug messages
    
    Returns:
        list: List of tuples with (lemma, georgian_definition, mingrelian_word, definition_line, entry_text, georgian_script_word)
    """
    # Find the entries for the lemma
    entry_texts = find_lemma_entry(file_path, lemma)
    
    # If no entries found, try hyphenated form with "-i" first (higher priority)
    if not entry_texts:
        # For words ending in consonants, try the hyphenated "-i" form first
        if (not lemma.startswith('-') and not '-' in lemma and 
            lemma[-1] not in ['a', 'i', 'e', 'o', 'u', 'h', 's']):
            hyphenated_form = lemma + '-i'
            if debug_output:
                print(f"Trying hyphenated form: {hyphenated_form}")
            entry_texts = find_lemma_entry(file_path, hyphenated_form)
            if entry_texts and debug_output:
                print(f"Found entries using hyphenated form: {hyphenated_form}")
        
        # If still no entries found, try to lemmatize the word and search again
        if not entry_texts:
            lemmatized_form = lemmatize_mingrelian(lemma)
            # Only search again if the lemmatized form is different from the original
            if lemmatized_form != lemma:
                if debug_output:
                    print("PRINTING LEMMATIZED FORM: ", lemmatized_form)
                entry_texts = find_lemma_entry(file_path, lemmatized_form)
                # If entries found with the lemmatized form, log this
                if entry_texts and debug_output:
                    print(f"Found entries using lemmatized form: {lemmatized_form}")
    
    if not entry_texts:
        return [(lemma, None, None, None, None, None)]
    
    results = []
    for entry_text in entry_texts:
        # Extract the lemma and definition from the entry
        curr_lemma, definition, curr_entry_text = extract_definition(entry_text)
        
        # Extract the main definition line for display purposes
        definition_line = None
        first_content_line = None
        
        if curr_entry_text:
            lines = curr_entry_text.strip().split('\n')
            for i, line in enumerate(lines):
                if i > 1 and line.strip() and not first_content_line:  # Skip Lemma and Number lines
                    first_content_line = line.strip()
                if line.startswith('Number:') and i+1 < len(lines) and lines[i+1].strip():
                    definition_line = lines[i+1].strip()
                    break
        
        # Try to extract the Georgian script word from the first content line
        georgian_script_word = None
        if first_content_line:
            # Look for Georgian word at beginning of first content line
            georgian_match = re.search(r'^([ა-ჰ][ა-ჰ\-]+)', first_content_line)
            if georgian_match:
                georgian_script_word = georgian_match.group(1).strip()
        
        # If not found in the first line, try the definition line
        if not georgian_script_word and definition_line:
            georgian_match = re.search(r'^([ა-ჰ][ა-ჰ\-]+)', definition_line)
            if georgian_match:
                georgian_script_word = georgian_match.group(1).strip()
        
        # Try to extract the Mingrelian word from the entry
        mingrelian_word = None
        if definition_line:
            paren_match = re.search(r'^([ა-ჰ\-]+)\s*\(([^)]+)\)', definition_line)
            if paren_match:
                mingrelian_word = paren_match.group(1).strip()
        
        results.append((curr_lemma, definition, mingrelian_word, definition_line, curr_entry_text, georgian_script_word))
    
    return results

def search_containing_word(file_path, lemma):
    """
    Search for entries containing the Georgian version of the input word.
    
    Args:
        file_path (str): Path to the dictionary file
        lemma (str): Word to search for (in latinized form)
    
    Returns:
        list: List of tuples with (lemma_line, context_line, entry_text)
    """
    # Transliterate the lemma to Georgian script
    georgian_word = latinized_to_mkhedruli(lemma)
    
    # Find entries containing this Georgian word
    entry_texts = find_entries_containing_word(file_path, georgian_word)
    
    if not entry_texts:
        return []
    
    results = []
    for entry_text in entry_texts:
        lines = entry_text.strip().split('\n')
        
        # Get the lemma line
        lemma_line = lines[0] if lines else ""
        if lemma_line.startswith("Lemma:"):
            lemma_line = lemma_line.replace("Lemma:", "").strip()
            
            # Skip affix entries if we're not searching for an affix
            if lemma_line.startswith('-') and not lemma.startswith('-'):
                continue
        
        # Find a context line that contains our Georgian word
        context_line = ""
        for line in lines[2:]:  # Skip Lemma and Number lines
            if georgian_word in line:
                context_line = line.strip()
                break
        
        results.append((lemma_line, context_line, entry_text))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Translate Mingrelian lemmas to Georgian.')
    parser.add_argument('lemma', nargs='?', help='Lemma to translate')
    parser.add_argument('--file', default='../kajaia.txt', help='Path to dictionary file')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("Mingrelian-Georgian Translator")
        print("Enter a lemma to translate, or 'q' to quit")
        
        while True:
            try:
                lemma = input("\nLemma> ").strip()
                if lemma.lower() in ('q', 'quit', 'exit'):
                    break
                
                if not lemma:
                    continue
                
                # Display the lemma in Georgian script
                georgian_lemma = latinized_to_mkhedruli(lemma)
                print(f"Latinized: {lemma} → Georgian script: {georgian_lemma}")
                
                results = translate_lemma(args.file, lemma, debug_output=True)
                
                if len(results) == 1 and results[0][1] is None:
                    print(f"No direct translation found for '{lemma}' ({georgian_lemma})")
                    
                    # Try to find entries containing this word
                    print(f"Searching for entries containing '{georgian_lemma}'...")
                    containing_results = search_containing_word(args.file, lemma)
                    
                    if containing_results:
                        print(f"\nFound {len(containing_results)} entries containing '{georgian_lemma}':")
                        for i, (entry_lemma, context, _) in enumerate(containing_results):
                            if i > 0:
                                print("-" * 40)
                            print(f"Entry: {entry_lemma}")
                            if context:
                                print(f"Context: {context}")
                    else:
                        print(f"No entries found containing '{georgian_lemma}'")
                        print("\nTry using other forms of the lemma with or without hyphens.")
                else:
                    for i, (curr_lemma, definition, mingrelian, definition_line, _, georgian_word) in enumerate(results):
                        if i > 0:
                            print("\n" + "-" * 40 + "\n")  # Separator between multiple entries
                        
                        # Extract the actual lemma name from the entry
                        lemma_line = curr_lemma
                        if "Lemma:" in curr_lemma:
                            lemma_line = curr_lemma.replace("Lemma:", "").strip()
                        
                        print(f"Entry {i+1}: {lemma_line}")
                        
                        if definition:
                            # Display both latinized and Georgian script versions
                            display_word = georgian_word if georgian_word else georgian_lemma
                            print(f"Latinized: {lemma} → Georgian script: {display_word}")
                            print(f"Definition: {definition}")
                            if mingrelian and mingrelian != georgian_word:
                                print(f"Mingrelian form: {mingrelian}")
                            if definition_line:
                                print("\nDefinition:")
                                print(definition_line)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.lemma:
        # Display the lemma in Georgian script
        georgian_lemma = latinized_to_mkhedruli(args.lemma)
        print(f"Latinized: {args.lemma} → Georgian script: {georgian_lemma}")
        
        results = translate_lemma(args.file, args.lemma, debug_output=True)
        
        if len(results) == 1 and results[0][1] is None:
            print(f"No direct translation found for '{args.lemma}' ({georgian_lemma})")
            
            # Try to find entries containing this word
            print(f"Searching for entries containing '{georgian_lemma}'...")
            containing_results = search_containing_word(args.file, args.lemma)
            
            if containing_results:
                print(f"\nFound {len(containing_results)} entries containing '{georgian_lemma}':")
                for i, (entry_lemma, context, _) in enumerate(containing_results):
                    if i > 0:
                        print("-" * 40)
                    print(f"Entry: {entry_lemma}")
                    if context:
                        print(f"Context: {context}")
            else:
                print(f"No entries found containing '{georgian_lemma}'")
                print("\nTry using other forms of the lemma with or without hyphens.")
        else:
            for i, (curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word) in enumerate(results):
                if i > 0:
                    print("\n" + "-" * 40 + "\n")  # Separator between multiple entries
                
                # Extract the actual lemma name from the entry
                lemma_line = curr_lemma
                if "Lemma:" in curr_lemma:
                    lemma_line = curr_lemma.replace("Lemma:", "").strip()
                
                print(f"Entry {i+1}: {lemma_line}")
                
                if definition:
                    # Display both latinized and Georgian script versions
                    display_word = georgian_word if georgian_word else georgian_lemma
                    print(f"Latinized: {args.lemma} → Georgian script: {display_word}")
                    print(f"Definition: {definition}")
                    
                    if mingrelian and mingrelian != georgian_word:
                        print(f"Mingrelian form: {mingrelian}")
                    
                    # Print the entry for context
                    if entry_text:
                        print("\nEntry:")
                        lines = entry_text.strip().split('\n')
                        # Skip the lemma and number lines
                        for j, line in enumerate(lines):
                            if j >= 2 and line.strip():  # Skip Lemma and Number lines
                                print(line)
    
    else:
        print("Mingrelian-Georgian Translator")
        print("\nUsage examples:")
        print("  ./translate.py q'ilo           # Translate a specific lemma")
        print("  ./translate.py --interactive   # Run in interactive mode\n")
        parser.print_help()

if __name__ == "__main__":
    main() 