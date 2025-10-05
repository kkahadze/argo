import xml.etree.ElementTree as ET
import json
import os
import sys

def eaf_to_json(file_path):
    """
    Parses an ELAN Annotation Format (.eaf) file to extract translations
    and returns them as a JSON formatted string.

    Args:
        file_path (str): The path to the .eaf file.

    Returns:
        str: A JSON string containing the extracted translations, organized
             by 'phrases' and 'words'. Returns a JSON string with an error
             message if parsing fails.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return json.dumps({"error": f"Error parsing XML file: {e}"})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path}"})

    # Namespace handling
    namespace = ''
    if '}' in root.tag:
        namespace = root.tag.split('}')[0][1:]
    ns = {'': namespace} if namespace else {}

    # Data storage
    source_phrases = {}
    source_words = {}
    phrase_translations = []
    word_translations = []

    # Process Tiers
    for tier in root.findall('TIER', ns):
        tier_id = tier.get('TIER_ID')

        # 1. Collect source phrases (sentences)
        if tier_id == 'A_phrase-txt-xmf':
            for annotation in tier.findall('ANNOTATION/ALIGNABLE_ANNOTATION', ns):
                ann_id = annotation.get('ANNOTATION_ID')
                text = annotation.find('ANNOTATION_VALUE', ns).text
                if ann_id and text:
                    source_phrases[ann_id] = text.strip()

        # 2. Collect source words
        if tier_id == 'A_word-txt-xmf':
            for annotation in tier.findall('ANNOTATION/REF_ANNOTATION', ns):
                ann_id = annotation.get('ANNOTATION_ID')
                text = annotation.find('ANNOTATION_VALUE', ns).text
                if ann_id and text:
                    source_words[ann_id] = text.strip()

    # Match Translations
    for tier in root.findall('TIER', ns):
        tier_id = tier.get('TIER_ID')

        # 3. Match phrase translations
        if tier_id == 'A_phrase-gls-en':
            for annotation in tier.findall('ANNOTATION/REF_ANNOTATION', ns):
                ref_id = annotation.get('ANNOTATION_REF')
                translation = annotation.find('ANNOTATION_VALUE', ns).text
                if ref_id in source_phrases and translation:
                    phrase_translations.append({
                        "original": source_phrases[ref_id],
                        "translation": translation.strip()
                    })
        
        # 4. Match word translations
        if tier_id == 'A_word-gls-en':
            for annotation in tier.findall('ANNOTATION/REF_ANNOTATION', ns):
                ref_id = annotation.get('ANNOTATION_REF')
                translation = annotation.find('ANNOTATION_VALUE', ns).text
                if ref_id in source_words and translation:
                    if not any(d['original'] == source_words[ref_id] for d in word_translations):
                        word_translations.append({
                            "original": source_words[ref_id],
                            "translation": translation.strip()
                        })

    # Final JSON Output
    output_data = {
        "phrases": phrase_translations,
        "words": word_translations
    }
    return json.dumps(output_data, indent=2, ensure_ascii=False)

# --- Main execution block for Command-Line Interface ---
if __name__ == '__main__':
    # 1. Check if a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python eaf_to_json_cli.py <path_to_eaf_file>")
        sys.exit(1)

    # 2. Get the input file path from the arguments
    input_file_path = sys.argv[1]

    # 3. Check if the file exists
    if not os.path.exists(input_file_path):
        print(f"Error: File not found at '{input_file_path}'")
        sys.exit(1)

    print(f"Processing file: '{input_file_path}'...")

    # 4. Call the function to get the JSON data
    json_output_string = eaf_to_json(input_file_path)

    # 5. Check if the function returned an error before proceeding
    temp_data = json.loads(json_output_string)
    if "error" in temp_data:
        print(f"An error occurred during processing: {temp_data['error']}")
        sys.exit(1)

    # 6. Determine the output file path
    # os.path.splitext splits the path into a (root, ext) tuple.
    # We take the root and add '.json'.
    base_path_without_ext = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_path_without_ext}.json"

    # 7. Write the JSON string to the new file
    try:
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_output_string)
        print(f"Successfully created JSON file at: '{output_file_path}'")
    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)
