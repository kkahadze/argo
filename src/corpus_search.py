#!/usr/bin/env python3
"""
Corpus search functionality for finding Mingrelian translations in parallel corpora.
"""
import json
import re
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance between strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_for_search(text: str) -> str:
    """
    Normalize text for searching by removing extra whitespace and lowercasing.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    return ' '.join(text.lower().split())


def load_corpus(corpus_file: str) -> Dict[str, str]:
    """
    Load parallel corpus from JSON file.
    
    Args:
        corpus_file: Path to corpus JSON file
        
    Returns:
        Dictionary mapping Mingrelian to English
    """
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Corpus file {corpus_file} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing corpus file {corpus_file}: {e}")
        return {}


def search_corpus(
    corpus: Dict[str, str],
    word: str,
    max_fuzzy_distance: int = 1
) -> List[Tuple[str, str, str]]:
    """
    Search corpus for a Mingrelian word.
    
    Args:
        corpus: Dictionary of Mingrelian->English translations
        word: Word to search for (in Mkhedruli script)
        max_fuzzy_distance: Maximum edit distance for fuzzy matching
        
    Returns:
        List of tuples: (match_type, mingrelian_text, english_text)
        match_type can be: 'exact_match', 'word_in_phrase', 'fuzzy_match_1', 'fuzzy_match_2'
    """
    results = []
    word_normalized = normalize_for_search(word)
    
    # First pass: exact matches and word-in-phrase matches
    for mingrelian_text, english_text in corpus.items():
        mingrelian_normalized = normalize_for_search(mingrelian_text)
        
        # Exact match
        if mingrelian_normalized == word_normalized:
            results.append(('exact_match', mingrelian_text, english_text))
            continue
        
        # Word appears in phrase (exact word boundary match)
        # Split by whitespace and punctuation
        words_in_text = re.findall(r'\S+', mingrelian_normalized)
        if word_normalized in words_in_text:
            results.append(('word_in_phrase', mingrelian_text, english_text))
            continue
    
    # If we found exact or word-in-phrase matches, return those
    if results:
        return results
    
    # Second pass: fuzzy matching on single words only (not phrases)
    for mingrelian_text, english_text in corpus.items():
        mingrelian_normalized = normalize_for_search(mingrelian_text)
        
        # Skip phrases for fuzzy matching (only match single words)
        if ' ' in mingrelian_normalized:
            continue
        
        # Calculate edit distance
        distance = levenshtein_distance(word_normalized, mingrelian_normalized)
        
        if distance == 1:
            results.append(('fuzzy_match_1', mingrelian_text, english_text))
        elif distance == 2 and len(word_normalized) > 5:
            results.append(('fuzzy_match_2', mingrelian_text, english_text))
    
    # Limit fuzzy results
    if results:
        # Prioritize distance 1 over distance 2
        distance_1 = [r for r in results if r[0] == 'fuzzy_match_1']
        distance_2 = [r for r in results if r[0] == 'fuzzy_match_2']
        
        limited = distance_1[:3]
        if len(limited) < 3:
            limited.extend(distance_2[:3 - len(limited)])
        
        return limited
    
    return []


def format_corpus_results(
    word: str,
    results: List[Tuple[str, str, str]]
) -> List[Dict[str, Any]]:
    """
    Format corpus search results as structured entries for LLM consumption.
    
    Args:
        word: The original search word
        results: List of (match_type, mingrelian_text, english_text) tuples
        
    Returns:
        List of formatted entry dictionaries
    """
    formatted_entries = []
    
    for match_type, mingrelian_text, english_text in results:
        entry = {
            'source': 'corpus',
            'match_type': match_type,
            'search_word': word,
            'mingrelian': mingrelian_text,
            'english': english_text
        }
        formatted_entries.append(entry)
    
    return formatted_entries


def search_word_in_corpus(
    corpus_file: str,
    word: str,
    max_fuzzy_distance: int = 1
) -> List[Dict[str, Any]]:
    """
    Main function to search for a word in the corpus and return formatted results.
    
    Args:
        corpus_file: Path to corpus JSON file
        word: Mingrelian word to search for (in Mkhedruli script)
        max_fuzzy_distance: Maximum edit distance for fuzzy matching
        
    Returns:
        List of formatted entry dictionaries
    """
    corpus = load_corpus(corpus_file)
    
    if not corpus:
        return []
    
    results = search_corpus(corpus, word, max_fuzzy_distance)
    
    return format_corpus_results(word, results)


def main():
    """Test corpus search functionality."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python corpus_search.py <word>")
        print("Example: python corpus_search.py ჯოხო")
        sys.exit(1)
    
    word = sys.argv[1]
    corpus_file = str(Path(__file__).parent.parent / 'data' / 'en_to_xmf.json')
    
    results = search_word_in_corpus(corpus_file, word)
    
    if results:
        print(f"\nFound {len(results)} corpus matches for '{word}':\n")
        for entry in results:
            print(f"Match Type: {entry['match_type']}")
            print(f"Mingrelian: {entry['mingrelian']}")
            print(f"English: {entry['english']}")
            print("-" * 60)
    else:
        print(f"No corpus matches found for '{word}'")


if __name__ == "__main__":
    main()
