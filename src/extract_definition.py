#!/usr/bin/env python3
import sys
import re
import argparse

def extract_definition(entry_text):
    """
    Extract the Georgian definition from a dictionary entry.
    
    Args:
        entry_text (str): The text of the dictionary entry
    
    Returns:
        tuple: (lemma, definition, full_entry)
    """
    lines = entry_text.strip().split('\n')
    
    # Extract lemma
    lemma = None
    for line in lines:
        if line.startswith('Lemma:'):
            lemma = line.replace('Lemma:', '').strip()
            break
    
    if not lemma:
        return None, None, entry_text
    
    # Skip empty lines and find the first non-empty line after "Number:" line
    definition_line = None
    for i, line in enumerate(lines):
        if line.startswith('Number:'):
            # Look for the next non-empty line
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    definition_line = lines[j].strip()
                    break
            break
    
    if not definition_line:
        return lemma, None, entry_text
    
    # Clean up definition line first - remove any line breaks in the middle of text
    definition_line = ' '.join(definition_line.split())
    
    # Extract the Georgian definition
    definition = None
    
    # Pattern 1: "იხ." reference followed by actual translation
    if "იხ." in definition_line and "--" in definition_line:
        # Example: "ჸვალ-ი (ჸვალ/რს) იხ. ყვალი, -- ყველი."
        reference_pattern = re.search(r'იხ\.[^,]*,[^-]*--\s*([ა-ჰ][ა-ჰ\-]+)', definition_line)
        if reference_pattern:
            definition = reference_pattern.group(1).strip()
    
    # Pattern 2: "იგივეა, რაც X, -- Y" pattern
    # This pattern indicates Y is the actual definition
    if "იგივეა, რაც" in definition_line and "--" in definition_line:
        # Use a more specific regex to find the definition after "--" followed by an optional number
        igiveac_pattern = re.search(r'იგივეა,\s*რაც\s*[^,]*,\s*--\s*(?:\d+\.\s*)?([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)', definition_line)
        if igiveac_pattern:
            definition = igiveac_pattern.group(1).strip()
        # If that fails, try the split approach
        else:
            parts = definition_line.split("--")
            if len(parts) > 1:
                after_dash = parts[1].strip()
                # Handle numbered definitions like "1. შვილი"
                numbered_def_match = re.search(r'(?:\d+\.\s*)?([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)', after_dash)
                if numbered_def_match:
                    definition = numbered_def_match.group(1).strip()
    
    # Pattern 2: Line with grammatical annotations followed by actual definition
    if not definition and "მიმღ." in definition_line and "ზმნისა" in definition_line and "--" in definition_line:
        # Extract content after double dash
        parts = definition_line.split("ზმნისა --", 1)
        if len(parts) > 1:
            after_dash = parts[1].strip()
            # Handle cases with {ga} or other prefixes in curly braces
            if "{" in after_dash and "}" in after_dash:
                # Remove prefixes in curly braces
                cleaned = re.sub(r'\{[^}]+\}', '', after_dash)
                # Extract the Georgian word
                match = re.search(r'([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)', cleaned)
                if match:
                    definition = match.group(1).strip()
            else:
                # Regular case
                match = re.search(r'([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)', after_dash)
                if match:
                    definition = match.group(1).strip()
    
    # Pattern 3: For entries with parenthesis followed by Georgian definition
    if not definition:
        # First try with more constrained pattern that matches just the first Georgian word
        direct_def = re.search(r'^\s*[^(]+ \([^)]+\)\s+([ა-ჰ][ა-ჰ\-]+)(?=\s|[,.]|$|\()', definition_line)
        if direct_def:
            definition = direct_def.group(1).strip()
        else:
            # Try for parentheses pattern with multi-word definition
            direct_multi = re.search(r'^\s*[^(]+ \([^)]+\)\s+([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)(?=\s*[,.]|$|\()', definition_line)
            if direct_multi:
                definition = direct_multi.group(1).strip()
            # Broader pattern anywhere in the line
            else:
                broader_def = re.search(r'\([^)]+\)\s+([ა-ჰ][ა-ჰ\s\-]+?)(?=\s*[,;\-]|\.|\s*$)', definition_line)
                if broader_def:
                    definition = broader_def.group(1).strip()

    # If we still don't have a definition, try an alternative pattern for multi-word phrases
    if not definition or len(definition.split()) < 2:  # If only one word was captured
        # Look for multi-word definitions before punctuation
        multi_word = re.search(r'^\s*[^(]+ \([^)]+\)\s+([ა-ჰ][ა-ჰ\s\-]+?მქონე)(?=\s*[,;\-]|\.|\s*$)', definition_line)
        if multi_word:
            definition = multi_word.group(1).strip()
    
    # Pattern 4: After explicit double dash
    if not definition:
        dash_def = re.search(r'--\s*([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)(?=\s*[.;:]|\s*$)', definition_line)
        if dash_def:
            definition = dash_def.group(1).strip()
    
    # Pattern 5: After comma and double dash (common in many entries)
    if not definition:
        comma_dash = re.search(r'[,،]\s*--\s*([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)(?=\s*[.;:]|\s*$)', definition_line)
        if comma_dash:
            definition = comma_dash.group(1).strip()
    
    # Pattern 6: After single dash (less reliable but catch remaining cases)
    if not definition:
        single_dash = re.search(r'[^-][-—–]\s*([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)(?=\s*[.;:]|\s*$)', definition_line)
        if single_dash:
            definition = single_dash.group(1).strip()
    
    # If we still have no definition but found Georgian text, extract the first Georgian phrase
    if not definition:
        georgian_text = re.search(r'([ა-ჰ][ა-ჰ\-]+(?:\s+[ა-ჰ][ა-ჰ\-]+)*)', definition_line)
        if georgian_text:
            definition = georgian_text.group(1).strip()
    
    # Clean up definition - remove trailing punctuation and common noise words
    if definition:
        definition = re.sub(r'[.,;:]\s*$', '', definition)
        
        # Remove common noise words at the beginning
        noise_words = ["იხ", "იხ.", "შდრ", "შდრ.", "ე.ი.", "ე.წ.", "ანუ"]
        for word in noise_words:
            if definition.startswith(word + " "):
                definition = definition[len(word)+1:].strip()
    
    return lemma, definition, entry_text

def parse_entry_from_file(file_path, lemma=None):
    """
    Extract a dictionary entry for a specific lemma from a file.
    
    Args:
        file_path (str): Path to the dictionary file
        lemma (str, optional): Specific lemma to search for
    
    Returns:
        list: List of (lemma, definition, entry_text) tuples
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            current_entry = []
            recording = False if lemma else True
            
            for line in file:
                # Start of a new lemma
                if line.strip().startswith('Lemma:'):
                    # Process the previous entry if we were recording
                    if recording and current_entry:
                        entry_text = '\n'.join(current_entry)
                        results.append(extract_definition(entry_text))
                        
                    # Reset for new entry
                    current_entry = [line.strip()]
                    
                    # If we're looking for a specific lemma, check if this is it
                    if lemma:
                        current_lemma = line.replace('Lemma:', '').strip()
                        recording = lemma.lower() == current_lemma.lower() or lemma.lower() == current_lemma.lower().replace('-', '')
                    
                # Continue recording the current entry
                elif recording:
                    current_entry.append(line.strip())
                    
            # Don't forget the last entry
            if recording and current_entry:
                entry_text = '\n'.join(current_entry)
                results.append(extract_definition(entry_text))
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return results

def search_by_definition(file_path, search_term):
    """
    Search for entries whose Georgian definition contains the search term.
    
    Args:
        file_path (str): Path to the dictionary file
        search_term (str): Georgian word to search for in definitions
    
    Returns:
        list: List of (lemma, definition, entry_text) tuples
    """
    results = []
    all_entries = parse_entry_from_file(file_path)
    
    for lemma, definition, entry_text in all_entries:
        if definition and search_term.lower() in definition.lower():
            results.append((lemma, definition, entry_text))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract definitions from dictionary entries.')
    parser.add_argument('--lemma', help='Specific lemma to search for')
    parser.add_argument('--entry', help='Text of a dictionary entry')
    parser.add_argument('--file', default='../kajaia.txt', help='Path to dictionary file')
    parser.add_argument('--definition', help='Search for entries with this Georgian word in their definition')
    
    args = parser.parse_args()
    
    if args.entry:
        # Process a single entry from command line
        lemma, definition, _ = extract_definition(args.entry)
        if definition:
            print(f"Lemma: {lemma}")
            print(f"Definition: {definition}")
        else:
            print(f"Could not extract definition for lemma: {lemma}")
    elif args.definition:
        # Search by definition
        results = search_by_definition(args.file, args.definition)
        if results:
            print(f"Found {len(results)} entries with '{args.definition}' in their definition:")
            for lemma, definition, entry in results:
                print(f"Lemma: {lemma}")
                print(f"Definition: {definition}")
                print("-" * 40)
            
            # Display the first match in detail
            print("\nFirst match full entry:")
            print("=" * 40)
            print(results[0][2])
            print("=" * 40)
        else:
            print(f"No entries found with '{args.definition}' in their definition.")
    elif args.lemma:
        # Search for a specific lemma in the file
        results = parse_entry_from_file(args.file, args.lemma)
        if results:
            for lemma, definition, entry in results:
                if definition:
                    print(f"Lemma: {lemma}")
                    print(f"Definition: {definition}")
                    print("-" * 40)
                    print(entry)
                    print("=" * 40)
                else:
                    print(f"Could not extract definition for lemma: {lemma}")
        else:
            print(f"Lemma '{args.lemma}' not found.")
    else:
        # Process all entries in the file
        results = parse_entry_from_file(args.file)
        
        total = len(results)
        extracted = sum(1 for _, definition, _ in results if definition)
        
        print(f"Processed {total} entries, extracted {extracted} definitions")
        print(f"Success rate: {extracted/total*100:.2f}%")
        
        # Show a few examples
        print("\nSample results:")
        for i, (lemma, definition, _) in enumerate(results[:5]):
            if definition:
                print(f"{i+1}. {lemma}: {definition}")

if __name__ == "__main__":
    main() 