#!/usr/bin/env python3
import sys
import re

def search_lemma(file_path, search_term):
    """
    Search for lemmas matching the search term in the specified file.
    
    Args:
        file_path (str): Path to the dictionary file
        search_term (str): Term to search for
    
    Returns:
        tuple: (list of result lines, count of actual lemma matches)
    """
    matches = []
    match_count = 0
    
    # Normalize search term - replace hyphens with optional hyphens in the regex
    normalized_term = re.escape(search_term).replace('-', '\\-?')
    
    # If user didn't include a hyphen, also try to match versions with hyphens
    if '-' not in search_term:
        # For each position in the search term, create a pattern that allows an optional hyphen there
        chars = list(re.escape(search_term))
        for i in range(1, len(chars)):
            chars_with_hyphen = chars.copy()
            chars_with_hyphen.insert(i, '\\-?')
            normalized_term = f"({normalized_term}|{''.join(chars_with_hyphen)})"
    
    lemma_pattern = re.compile(r'Lemma:\s+(' + normalized_term + r')(?:\s+|$)')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_num = 0
            for line in file:
                line_num += 1
                if lemma_pattern.search(line):
                    match_count += 1
                    # Add the matching lemma line
                    matches.append(f"{line_num}: {line.strip()}")
                    
                    # Also include a few lines after the match for context
                    context_lines = []
                    for _ in range(10):  # Increase to grab more context
                        try:
                            context_line = next(file)
                            line_num += 1
                            context_lines.append(f"{line_num}: {context_line.strip()}")
                        except StopIteration:
                            break
                    
                    if context_lines:
                        matches.extend(context_lines)
                    matches.append("")  # Empty line for readability
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
        
    return matches, match_count

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python lemma_search.py <search_term> [dictionary_file]")
        print("  <search_term>: Term to search for in the dictionary")
        print("  [dictionary_file]: Optional path to dictionary file (default: ../kajaia.txt)")
        print()
        print("Note: The search matches lemmas with or without hyphens")
        sys.exit(1)
    
    search_term = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else "../kajaia.txt"
    
    # Perform the search
    results, match_count = search_lemma(file_path, search_term)
    
    # Display results
    if results:
        print(f"Found {match_count} lemma match{'es' if match_count != 1 else ''} for '{search_term}':")
        for result in results:
            print(result)
    else:
        print(f"No lemmas matching '{search_term}' found.")
        
        # Suggestion for alternative searches
        if '-' in search_term:
            # If user included a hyphen, suggest searching without it
            suggestion = search_term.replace('-', '')
            print(f"Try searching for '{suggestion}' without the hyphen")
        else:
            # If there was no hyphen, suggest possible hyphenated versions
            for i in range(1, len(search_term)):
                suggestion = search_term[:i] + '-' + search_term[i:]
                print(f"Try searching for '{suggestion}' with a hyphen")

if __name__ == "__main__":
    main() 