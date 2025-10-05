import json
import sys
from pathlib import Path
from collections import defaultdict
import os

def aggregate_word_dictionaries(root_dir: str, extension: str, output_filename: str):
    """
    Recursively finds JSON files, aggregates ALL 'original'-'translation' pairs 
    from any top-level list (e.g., 'words' or 'phrases'), and saves the result.
    
    Identical words across different files will have their translations merged 
    and comma-separated.

    Args:
        root_dir (str): The starting directory to search from.
        extension (str): The file extension to look for (e.g., 'json').
        output_filename (str): The name of the resulting aggregated JSON file.
    """
    # Use a defaultdict where the value is a set, ensuring we only collect 
    # unique translations for each word before merging.
    word_translations = defaultdict(set) 
    
    clean_extension = extension.lstrip('.')
    search_pattern = f'*.{clean_extension}'
    root_path = Path(root_dir)

    if not root_path.is_dir():
        print(f"Error: Directory not found at '{root_dir}'")
        sys.exit(1)

    print(f"\n--- Starting aggregation for *.{clean_extension} files in '{root_dir}' ---")

    file_count = 0
    
    # Use rglob to search recursively within the root directory
    for file_path in root_path.rglob(search_pattern):
        if not file_path.is_file():
            continue

        file_count += 1
        print(f"  Processing file: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Iterate through all top-level elements (e.g., 'words', 'phrases')
                for _, list_of_items in data.items():
                    # Only process values that are lists
                    if isinstance(list_of_items, list):
                        for item in list_of_items:
                            original = item.get('original')
                            translation = item.get('translation')
                            
                            if original and translation:
                                # Add the translation to the set for this word/phrase
                                word_translations[original].add(translation)
            
        except json.JSONDecodeError:
            print(f"  Warning: Failed to parse JSON from '{file_path}' (Skipping)")
        except Exception as e:
            print(f"  Warning: An unexpected error occurred processing '{file_path}': {e}")
            
    if file_count == 0:
        print("No files found matching the criteria. Exiting.")
        return

    # Convert the dictionary of sets into the final dictionary of strings
    # Translations for the same word are joined by ", "
    final_dictionary = {
        # Sort translations alphabetically for consistent output
        word: ", ".join(sorted(list(translations))) 
        for word, translations in word_translations.items()
    }
    
    # Write the final aggregated dictionary to the output file
    output_path = Path(output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False to preserve Mingrelian characters properly
            json.dump(final_dictionary, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Aggregation complete!")
        print(f"   Processed {file_count} files.")
        print(f"   Saved {len(final_dictionary)} unique entries to '{output_filename}'")
    except Exception as e:
        print(f"\nError: Failed to write output file '{output_filename}': {e}")


# --- Execution Block ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_aggregator.py <target_directory_path> [output_file_name]")
        print("\nExample 1: python json_aggregator.py ./my_repo")
        print("Example 2: python json_aggregator.py /data/vocab my_master_list.json")
        sys.exit(1)

    # Get the target directory from the first command-line argument
    target_dir = sys.argv[1]
    
    # Get the output filename (use 'combined_dictionary.json' if not provided)
    output_name = sys.argv[2] if len(sys.argv) > 2 else "combined_dictionary.json"
    
    # The extension to search for (assumed to be json based on your data)
    file_extension = "json" 
    
    aggregate_word_dictionaries(target_dir, file_extension, output_name)
