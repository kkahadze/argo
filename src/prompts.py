#!/usr/bin/env python3
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import re

# Example prompt showing how to translate dictionary entries into JSON format
FIRST_SHOT_EXAMPLE = """
    Input:    

    Latinized Mingrelian word: vemxi 
    Mkhedruli Mingrelian word: ვემხი

    ========================================
    DICTIONARY ENTRIES:

    Lemma: vemx-i
    Number: 8022

    ვემხ-ი, ვემხვ-ი (ვემხ{ვ}ის) ზოოლ. ვეფხვი. ჩქჷნიანეფქ ვემხიცალო უზოგალო ქათირსეს: კ. სამუშ., ქართ.ზეპ., გვ. 101 -ჩვენიანები ვეფხვივით უზოგველად დაეჯახნენ (ეცნენ).
    
    Output (JSON format):
    {
      "entries": [
        {
          "lemma": "vemx-i",
          "number": "8022",
          "mingrelian": "ვემხი",
          "georgian": "ვეფხვი",
          "english": "tiger",
          "definition": "zoological term",
          "match_type": "exact_match",
          "examples": [
            {
              "georgian": "ჩქჷნიანეფქ ვემხიცალო უზოგალო ქათირსეს",
              "english": "Swiftly, like a tiger, they swooped down upon the flock.",
              "source": "K. Samush, Georgian Oral Traditions, p. 101"
            }
          ],
          "notes": "vemx-i, vemxv-i (vemxv'is)"
        }
      ]
    }
"""

def log_to_file(content: str, filename: str, logging_mode: bool = True) -> None:
    """
    Helper function to handle file logging based on logging mode.
    
    Args:
        content (str): Content to write to the file
        filename (str): Path to the log file
        logging_mode (bool): Whether to log the content to file (always True for file logging)
    """
    if logging_mode:
        # Get the src directory path
        import os
        
        # Use the src directory for log files
        src_dir = Path(__file__).parent
        log_file_path = src_dir / filename
        
        # Write content to the log file
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(content)

# Initial translation prompt
def get_initial_translation_prompt(dict_entries: List[str], logging_mode: bool = True) -> str:
    """
    Generate the initial translation prompt with the example and dictionary entries.
    Requests JSON output for structured parsing.
    
    Args:
        dict_entries (List[str]): List of dictionary entries to translate
        logging_mode (bool): Whether to log the prompt to file
        
    Returns:
        str: The complete initial translation prompt
    """
    command = f"""
Can you translate these entries from a Mingrelian-Georgian dictionary into English. 
Don't translate the Mingrelian into English, just the Georgian which you should be able to understand.

IMPORTANT: Return your response as valid JSON following the structure in the example below.

Here is an example of what I expect you to do: {FIRST_SHOT_EXAMPLE}

Now translate the following entries into English and return them as a JSON object with an "entries" array.
Each entry should have: lemma, number (if available), mingrelian, georgian, english, definition, match_type, examples (if available), and notes (if available).

IMPORTANT: The "match_type" field should preserve information about how this entry was found:
- "exact_match" - if it's a direct lemma match
- "hyphenated_form" - if found via adding "-i" 
- "lemmatized_form" - if found via morphological rules (e.g., suffix removal)
- "partial_match" - if the word appears in the entry text but isn't the main lemma
- "fuzzy_match_1" - if found via edit distance of 1 (spelling variant)
- "fuzzy_match_2" - if found via edit distance of 2 (possible typo)

Look for phrases like "Match for hyphenated form", "Match for lemmatized form", "Partial match", "Close match" in the input to determine the match_type.

"""
    print("Dict 1: ", dict_entries[0])
    prompt = command + "\n" + "\n".join(dict_entries) + "\n\nReturn your response as valid JSON starting with { and ending with }."
    
    # Log the initial prompt if in logging mode
    log_to_file(prompt, 'initial_prompt_log.txt', logging_mode)
    
    return prompt

# Follow-up translation prompts
def get_follow_up_phrase(latinized: str, mkhedruli: str) -> str:
    """
    Generate the follow-up phrase prompt.
    
    Args:
        latinized (str): Latinized Mingrelian phrase
        mkhedruli (str): Mkhedruli Mingrelian phrase
        
    Returns:
        str: The formatted follow-up phrase prompt
    """
    return f"""
                        Now you will translate the following phrase into Georgian and then English
                        Phrase in Mingrelian (latinized script): {latinized}
                        Same phrase in Mingrelian (mkhedruli script): {mkhedruli}
                        """

def get_grammar_text(grammar: str) -> str:
    """
    Generate the grammar text section of the prompt.
    
    Args:
        grammar (str): Grammar description text
        
    Returns:
        str: The formatted grammar section
    """
    
    # import popiel.txt using absolute path
    popiel_path = Path(__file__).parent.parent / 'data' / 'popiel.txt'
    with open(popiel_path, 'r') as file:
        popiel_grammar = file.read()

    return f"\nYou will also be given two grammars written in English to help you translate the phrase. Beware of any differences in transcription that may occur between the grammer which is written in English by a linguist and uses IPA, and the dictionarty entries which use latinized Mingrelian at times and Mkhedruli (Georgian script) at other times.\n\n GRAMMAR START\n{grammar}\nGRAMMAR END\n\n GRAMMAR START\n{popiel_grammar}\nGRAMMAR END\n"




def get_after_phrase(latinized: str, mkhedruli: str) -> str:
    """
    Generate the after phrase prompt with output format instructions.
    
    Args:
        latinized (str): Latinized Mingrelian phrase
        mkhedruli (str): Mkhedruli Mingrelian phrase
        
    Returns:
        str: The formatted after phrase prompt
    """
    return f"""
Now you will translate the phrase into Georgian and then English
Phrase in Mingrelian (latinized script): {latinized}
Phrase in Mingrelian (mkhedruli script): {mkhedruli}

VERY IMPORTANT: After analyzing the phrase and thinking through the translation step by step, 
please output your answer IN EXACTLY THIS FORMAT:

Translation:
Georgian: [Georgian translation]  
English: [English translation]

You MUST include both lines with exactly those labels ("Georgian:" and "English:") 
as they will be automatically parsed by the system.

You can add additional notes, etymology, or explanations AFTER those two lines.
                        """

def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON response from LLM with error handling.
    
    Args:
        response_text (str): Raw response text from LLM
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dict or None if parsing fails
    """
    try:
        # Try direct JSON parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    
    return None


def format_corpus_entry_for_llm(entry: Dict[str, Any]) -> str:
    """
    Format a single corpus entry for LLM consumption.
    
    Args:
        entry: Corpus entry dictionary
        
    Returns:
        Formatted string
    """
    parts = []
    
    # Header
    search_word = entry.get('search_word', 'Unknown')
    header = f"Corpus Entry (searched: {search_word})"
    parts.append(header)
    parts.append("-" * 60)
    
    # Match type with clear labeling
    match_type = entry.get('match_type', 'unknown')
    match_type_display = {
        'exact_match': '✓✓ CORPUS EXACT MATCH (verified parallel translation)',
        'word_in_phrase': '✓ CORPUS WORD IN PHRASE (word found in authentic usage)',
        'fuzzy_match_1': '⚠ Corpus fuzzy match (edit distance 1)',
        'fuzzy_match_2': '⚠⚠ Corpus fuzzy match (edit distance 2)',
    }
    parts.append(f"Match Quality: {match_type_display.get(match_type, match_type)}")
    parts.append(f"Source: Parallel Corpus (authentic usage)")
    parts.append("")  # Blank line
    
    # Core translations
    if entry.get('mingrelian'):
        parts.append(f"Mingrelian: {entry['mingrelian']}")
    if entry.get('english'):
        parts.append(f"English: {entry['english']}")
    
    return "\n".join(parts)


def format_translations_for_llm(translations_json: Dict[str, Any], corpus_entries: List[Dict[str, Any]] = None) -> str:
    """
    Convert JSON translations and corpus entries into readable text format for the second LLM.
    
    Args:
        translations_json (Dict[str, Any]): Parsed JSON with dictionary translations
        corpus_entries (List[Dict[str, Any]]): List of corpus match entries
        
    Returns:
        str: Formatted readable text
    """
    formatted_parts = []
    
    # Format dictionary entries
    if translations_json and 'entries' in translations_json:
        for entry in translations_json['entries']:
            parts = []
            
            # Header
            lemma = entry.get('lemma', 'Unknown')
            number = entry.get('number', '')
            header = f"Dictionary Entry: {lemma}"
            if number:
                header += f" (#{number})"
            parts.append(header)
            parts.append("-" * 60)
            
            # Match type (IMPORTANT for second LLM to know reliability)
            match_type = entry.get('match_type', 'unknown')
            match_type_display = {
                'exact_match': '✓ EXACT MATCH',
                'hyphenated_form': '✓ Hyphenated form (morphological)',
                'lemmatized_form': '✓ Lemmatized form (morphological)',
                'partial_match': '⚠ Partial match (found in entry text)',
                'fuzzy_match_1': '⚠ FUZZY MATCH (edit distance 1 - possible spelling variant)',
                'fuzzy_match_2': '⚠⚠ FUZZY MATCH (edit distance 2 - possible typo or error)',
                'unknown': 'Match type: unknown'
            }
            parts.append(f"Match Quality: {match_type_display.get(match_type, match_type)}")
            parts.append(f"Source: Kajaia Dictionary")
            parts.append("")  # Blank line for separation
            
            # Core translations
            if entry.get('mingrelian'):
                parts.append(f"Mingrelian: {entry['mingrelian']}")
            if entry.get('georgian'):
                parts.append(f"Georgian: {entry['georgian']}")
            if entry.get('english'):
                parts.append(f"English: {entry['english']}")
            
            # Definition
            if entry.get('definition'):
                parts.append(f"Definition: {entry['definition']}")
            
            # Notes
            if entry.get('notes'):
                parts.append(f"Notes: {entry['notes']}")
            
            # Examples
            if entry.get('examples'):
                parts.append("\nExamples:")
                for i, example in enumerate(entry['examples'], 1):
                    if example.get('georgian'):
                        parts.append(f"  {i}. Georgian: {example['georgian']}")
                    if example.get('english'):
                        parts.append(f"     English: {example['english']}")
                    if example.get('source'):
                        parts.append(f"     Source: {example['source']}")
            
            formatted_parts.append("\n".join(parts))
    
    # Format corpus entries
    if corpus_entries:
        for corpus_entry in corpus_entries:
            formatted_parts.append(format_corpus_entry_for_llm(corpus_entry))
    
    if not formatted_parts:
        return "No translations available."
    
    return "\n\n".join(formatted_parts)


def get_corpus_only_prompt(
    follow_up_phrase: str,
    corpus_translations: str,
    logging_mode: bool = True
) -> str:
    """
    Generate a simplified prompt when we only have corpus data (no dictionary entries).
    This is much lighter than the full prompt - just asks for Georgian equivalent.
    
    Args:
        follow_up_phrase (str): The follow-up phrase prompt with latinized/mkhedruli
        corpus_translations (str): Formatted corpus translations
        logging_mode (bool): Whether to log the prompt to file
        
    Returns:
        str: The simplified follow-up prompt
    """
    # Extract the latinized and mkhedruli script
    latinized = ""
    mkhedruli = ""
    
    lines = follow_up_phrase.strip().split('\n')
    for line in lines:
        if "latinized script" in line:
            latinized = line.split(":")[-1].strip()
        elif "mkhedruli script" in line:
            mkhedruli = line.split(":")[-1].strip()
    
    prompt = f"""You are translating Mingrelian to Georgian and English.

The word/phrase in question:
- Latinized: {latinized}
- Mkhedruli: {mkhedruli}

Below are verified parallel translations from authentic Mingrelian corpus:

{corpus_translations}

Task: Provide a concise translation with Georgian equivalent.

Output format:
Translation:
Georgian: [Georgian translation]
English: [concise English translation based on corpus]

Notes:
[Brief usage note if helpful, referencing corpus examples]
"""
    
    # Log the prompt
    log_to_file(prompt, 'followup_prompt_log.txt', logging_mode)
    
    return prompt


def get_follow_up_prompt(
    follow_up_phrase: str, 
    grammar_text: str, 
    initial_response_text: str, 
    dict_entries: List[str],
    logging_mode: bool = True
) -> str:
    """
    Generate the complete follow-up prompt combining all components.
    Now expects initial_response_text to be formatted readable text (not raw JSON).
    
    Args:
        follow_up_phrase (str): The follow-up phrase prompt
        grammar_text (str): The grammar section
        initial_response_text (str): Formatted translations (from format_translations_for_llm)
        dict_entries (List[str]): The dictionary entries
        logging_mode (bool): Whether to log the prompt to file
        
    Returns:
        str: The complete follow-up prompt
    """
    # Extract the latinized and mkhedruli script from the follow_up_phrase
    latinized = ""
    mkhedruli = ""
    
    lines = follow_up_phrase.strip().split('\n')
    for line in lines:
        if "latinized script" in line:
            latinized = line.split(":")[-1].strip()
        elif "mkhedruli script" in line:
            mkhedruli = line.split(":")[-1].strip()
    
    # Get the after phrase with format instructions
    after_phrase = get_after_phrase(latinized, mkhedruli)
    
    # Combine all components with formatted translations
    prompt = follow_up_phrase + grammar_text + "\n\nDictionary entries translated into English:\n\n" + initial_response_text + "\n\n" + after_phrase
    
    # Log the follow-up prompt if in logging mode
    log_to_file(prompt, 'followup_prompt_log.txt', logging_mode)
    
    return prompt 