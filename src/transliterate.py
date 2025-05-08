#!/usr/bin/env python3

def latinized_to_mkhedruli(text):
    """
    Convert latinized Georgian text to Mkhedruli script.
    
    Args:
        text (str): Text in latinized Georgian
        
    Returns:
        str: Text in Georgian Mkhedruli script
    """
    # Mapping of latinized Georgian to Mkhedruli alphabet
    latin_to_georgian = {
        'a': 'ა',
        'b': 'ბ',
        'g': 'გ',
        'd': 'დ',
        'e': 'ე',
        'v': 'ვ',
        'z': 'ზ',
        't': 'თ',
        'i': 'ი',
        'y': 'ჲ',
        'k\'': 'კ',
        'l': 'ლ',
        'm': 'მ',
        'n': 'ნ',
        'o': 'ო',
        'p\'': 'პ',
        'zh': 'ჟ',
        'r': 'რ',
        's': 'ს',
        't\'': 'ტ',
        'u': 'უ',
        'p': 'ფ',
        'k': 'ქ',  # 'k-prime'
        'gh': 'ღ',
        'q\'': 'ყ',
        'sh': 'შ',
        'ch': 'ჩ',
        'c': 'ც',
        'dz': 'ძ',
        'ts\'': 'წ', # 'ts-prime'
        'ch\'': 'ჭ', # 'ch-prime'
        'x': 'ხ',
        'j': 'ჯ',
        'h': 'ჰ',
        'e\'': 'ჷ',
        '՚': 'ჸ',  # For apostrophes that may appear in latinized text
    }
    
    # Process multi-character sequences first, then single characters
    # This ensures proper handling of digraphs and trigraphs
    
    # Step 1: Convert the text to lowercase for consistent processing
    processed_text = text.lower()
    
    # Step 2: Replace apostrophes with a standard form
    processed_text = processed_text.replace("'", "\'")
    
    # Step 3: Create a list of mappings sorted by length (longest first)
    # This ensures that longer sequences (like 'ts\'' or 'ch\'') are processed before shorter ones
    sorted_mappings = sorted(
        [(k, v) for k, v in latin_to_georgian.items()],
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    # Step 4: Replace each multi-character sequence first, then single characters
    result = processed_text
    for latin, georgian in sorted_mappings:
        # Create a temporary result where we can safely replace characters
        temp_result = ""
        i = 0
        while i < len(result):
            if i <= len(result) - len(latin) and result[i:i+len(latin)] == latin:
                temp_result += georgian
                i += len(latin)
            else:
                temp_result += result[i]
                i += 1
        result = temp_result
    
    # Step 5: Preserve original case structure (though Georgian doesn't have case)
    final_result = ""
    for i, char in enumerate(text):
        if char.isupper() and i < len(result):
            # Georgian doesn't have uppercase, but we maintain the spacing/structure
            final_result += result[i]
        else:
            if i < len(result):
                final_result += result[i]
    
    return final_result

def mkhedruli_to_latinized(text):
    """
    Convert Georgian Mkhedruli script to latinized Georgian.
    This is the reverse of latinized_to_mkhedruli.
    
    Args:
        text (str): Text in Georgian Mkhedruli script
        
    Returns:
        str: Text in latinized Georgian
    """
    # Mapping of Mkhedruli to latinized Georgian
    # This is the inverse of the latin_to_georgian dictionary
    georgian_to_latin = {
        'ა': 'a',
        'ბ': 'b',
        'გ': 'g',
        'დ': 'd',
        'ე': 'e',
        'ვ': 'v',
        'ზ': 'z',
        'თ': 't',
        'ი': 'i',
        'კ': 'k',
        'ლ': 'l',
        'მ': 'm',
        'ნ': 'n',
        'ო': 'o',
        'პ': 'p',
        'ჟ': 'zh',
        'რ': 'r',
        'ს': 's',
        'ტ': 't\'',
        'უ': 'u',
        'ფ': 'p\'',
        'ქ': 'k',
        'ღ': 'gh',
        'ყ': 'q',
        'შ': 'sh',
        'ჩ': 'ch',
        'ც': 'c',
        'ძ': 'dz',
        'წ': 'ts\'',
        'ჭ': 'ch\'',
        'ხ': 'x',
        'ჯ': 'j',
        'ჰ': 'h',
        'ჸ': '՚',  # For apostrophes in Georgian text
        'ჷ': 'e\'',
        'ʼ': '\''  # For apostrophes in Georgian text
    }
    
    result = ''
    for char in text:
        if char in georgian_to_latin:
            result += georgian_to_latin[char]
        else:
            # Keep characters that don't have a mapping
            result += char
    
    return result

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        if any(char in 'აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ' for char in text):
            print(f"Converting from Georgian to Latin: {text}")
            result = mkhedruli_to_latinized(text)
        else:
            print(f"Converting from Latin to Georgian: {text}")
            result = latinized_to_mkhedruli(text)
        
        print(f"Result: {result}")
    else:
        print("Usage: python transliterate.py <text>")
        print("Example: python transliterate.py skua")
        print("         python transliterate.py შვილი") 