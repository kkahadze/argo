#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union

import config
from logger import get_logger
from dictionary_handler import DictionaryHandler
from translate import translate_lemma, search_containing_word, lemmatize_mingrelian, find_close_lemma_matches
from transliterate import latinized_to_mkhedruli
from extract_definition import extract_definition

class SearchResult:
    """Standardized container for search results"""
    
    def __init__(self, 
                 word: str, 
                 strategy_name: str, 
                 matches: List[Dict[str, Any]], 
                 total_matches: int = 0,
                 description: Optional[str] = None):
        self.word: str = word
        self.strategy_name: str = strategy_name
        self.matches: List[Dict[str, Any]] = matches  # List of dictionary entries
        self.total_matches: int = total_matches or len(matches)
        self.description: Optional[str] = description
        self.limited: bool = total_matches > len(matches)
    
    def has_matches(self) -> bool:
        """Check if this result contains any matches"""
        return len(self.matches) > 0

class SearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.dict_handler = DictionaryHandler()
    
    @abstractmethod
    def search(self, word: str, dictionary_path: str) -> SearchResult:
        """Search for a word using this strategy"""
        pass

class ExactMatchStrategy(SearchStrategy):
    """Strategy for finding exact matches for a word"""
    
    def search(self, word: str, dictionary_path: str) -> SearchResult:
        self.logger.debug(f"Searching for exact matches for '{word}'")
        results = translate_lemma(dictionary_path, word)
        
        # Check if we found any real matches
        if len(results) == 1 and results[0][1] is None:
            self.logger.debug(f"No exact matches found for '{word}'")
            return SearchResult(word, "exact_match", [], description="No exact matches found")
        
        # Process matches
        matches = []
        for curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word in results:
            if entry_text:
                matches.append({
                    "lemma": curr_lemma,
                    "definition": definition,
                    "mingrelian": mingrelian,
                    "definition_line": definition_line,
                    "entry_text": entry_text,
                    "georgian_word": georgian_word
                })
        
        self.logger.info(f"Found {len(matches)} exact matches for '{word}'")
        return SearchResult(
            word, 
            "exact_match", 
            matches[:config.MAX_EXACT_MATCHES], 
            total_matches=len(matches),
            description=f"Found {len(matches)} exact matches"
        )

class LemmatizedMatchStrategy(SearchStrategy):
    """Strategy for finding matches for the lemmatized form of a word"""
    
    def search(self, word: str, dictionary_path: str) -> SearchResult:
        lemmatized_word = lemmatize_mingrelian(word)
        
        # Don't bother if lemmatization didn't change anything
        if lemmatized_word == word:
            self.logger.debug(f"Lemmatization had no effect on '{word}'")
            return SearchResult(word, "lemmatized_match", [], description="Lemmatization had no effect")
        
        self.logger.debug(f"Searching for lemmatized form '{lemmatized_word}' of '{word}'")
        results = translate_lemma(dictionary_path, lemmatized_word)
        
        # Check if we found any real matches
        if len(results) == 1 and results[0][1] is None:
            self.logger.debug(f"No matches found for lemmatized form '{lemmatized_word}'")
            return SearchResult(word, "lemmatized_match", [], 
                              description=f"No matches found for lemmatized form '{lemmatized_word}'")
        
        # Process matches
        matches = []
        for curr_lemma, definition, mingrelian, definition_line, entry_text, georgian_word in results:
            if entry_text:
                matches.append({
                    "lemma": curr_lemma,
                    "definition": definition,
                    "mingrelian": mingrelian,
                    "definition_line": definition_line,
                    "entry_text": entry_text,
                    "georgian_word": georgian_word,
                    "lemmatized_word": lemmatized_word
                })
        
        self.logger.info(f"Found {len(matches)} matches for lemmatized form '{lemmatized_word}' of '{word}'")
        return SearchResult(
            word, 
            "lemmatized_match", 
            matches[:config.MAX_SIMILAR_MATCHES], 
            total_matches=len(matches),
            description=f"Found {len(matches)} matches for lemmatized form '{lemmatized_word}'"
        )

class EditDistanceStrategy(SearchStrategy):
    """Strategy for finding words with a small edit distance"""
    
    def __init__(self, use_lemmatized: bool = False, edit_distance: Optional[int] = None) -> None:
        """
        Initialize the strategy
        
        Args:
            use_lemmatized: Whether to use the lemmatized form of the word
            edit_distance: Override the default edit distance
        """
        super().__init__()
        self.use_lemmatized: bool = use_lemmatized
        self.edit_distance: Optional[int] = edit_distance
    
    def search(self, word: str, dictionary_path: str) -> SearchResult:
        # Determine which word form to use and what edit distance to apply
        if self.use_lemmatized:
            search_word = lemmatize_mingrelian(word)
            if search_word == word:
                self.logger.debug(f"Lemmatization had no effect on '{word}', skipping edit distance on lemmatized form")
                return SearchResult(word, "edit_distance", [], 
                                  description="Lemmatization had no effect, skipping edit distance on lemmatized form")
            strategy_name = "lemmatized_edit_distance"
            description_prefix = f"lemmatized form '{search_word}'"
        else:
            search_word = word
            strategy_name = "edit_distance"
            description_prefix = f"'{word}'"
        
        # Determine edit distance based on word length
        if self.edit_distance is not None:
            edit_distance = self.edit_distance
        else:
            is_long_word = len(search_word) > config.LONG_WORD_MIN_LENGTH
            edit_distance = config.LONG_WORD_EDIT_DISTANCE if is_long_word else config.DEFAULT_EDIT_DISTANCE
        
        # Perform the search
        self.logger.debug(f"Searching with edit distance {edit_distance} for {description_prefix}")
        close_matches = find_close_lemma_matches(dictionary_path, search_word, max_distance=edit_distance)
        
        if not close_matches:
            self.logger.debug(f"No matches with edit distance {edit_distance} for {description_prefix}")
            return SearchResult(word, strategy_name, [], 
                              description=f"No matches with edit distance {edit_distance} for {description_prefix}")
        
        # Process matches
        matches = []
        for lemma, entry_text in close_matches:
            curr_lemma, definition, curr_entry_text = extract_definition(entry_text)
            matches.append({
                "lemma": lemma,
                "definition": definition,
                "entry_text": entry_text,
                "edit_distance": edit_distance,
                "search_word": search_word
            })
        
        self.logger.info(f"Found {len(matches)} matches with edit distance {edit_distance} for {description_prefix}")
        return SearchResult(
            word, 
            strategy_name, 
            matches[:config.MAX_SIMILAR_MATCHES], 
            total_matches=len(matches),
            description=f"Found {len(matches)} matches with edit distance {edit_distance} for {description_prefix}"
        )

class ContainingWordStrategy(SearchStrategy):
    """Strategy for finding entries that contain the word anywhere in the text"""
    
    def search(self, word: str, dictionary_path: str) -> SearchResult:
        georgian_word = latinized_to_mkhedruli(word)
        self.logger.debug(f"Searching for entries containing '{word}' ({georgian_word})")
        containing_results = search_containing_word(dictionary_path, word)
        
        if not containing_results:
            self.logger.debug(f"No entries contain '{word}' ({georgian_word})")
            return SearchResult(word, "containing_word", [], 
                              description=f"No entries contain '{word}' ({georgian_word})")
        
        # Process matches
        matches = []
        for entry_lemma, context, entry_text in containing_results:
            matches.append({
                "lemma": entry_lemma,
                "context": context,
                "entry_text": entry_text,
                "georgian_word": georgian_word
            })
        
        self.logger.info(f"Found {len(matches)} entries containing '{word}' ({georgian_word})")
        return SearchResult(
            word, 
            "containing_word", 
            matches[:config.MAX_PARTIAL_MATCHES], 
            total_matches=len(matches),
            description=f"Found {len(matches)} entries containing '{georgian_word}'"
        )

class SearchEngine:
    """Main search engine that applies multiple search strategies in sequence"""
    
    def __init__(self, dictionary_path: str = config.DEFAULT_DICTIONARY_PATH) -> None:
        self.dictionary_path: str = dictionary_path
        self.logger = get_logger("SearchEngine")
        self.dict_handler = DictionaryHandler()
        
        # Preload the dictionary
        self.dict_handler.get_dictionary(dictionary_path)
        
        # Define the sequence of search strategies
        self.strategies: List[SearchStrategy] = [
            ExactMatchStrategy(),
            LemmatizedMatchStrategy(),
            # Edit distance 1 for lemmatized form
            EditDistanceStrategy(use_lemmatized=True, edit_distance=1),
            # Edit distance 1 for original word
            EditDistanceStrategy(use_lemmatized=False, edit_distance=1),
            # Try containing word search before edit distance 2
            ContainingWordStrategy(),
            # Edit distance 2 searches last (for longer words)
            EditDistanceStrategy(use_lemmatized=True, edit_distance=2),
            EditDistanceStrategy(use_lemmatized=False, edit_distance=2)
        ]
    
    def search(self, word: str) -> Tuple[SearchResult, List[SearchResult]]:
        """
        Search for a word using all available strategies
        
        Args:
            word: The word to search for
            
        Returns:
            Tuple containing:
            - The first successful search result
            - List of all search results (successful or not)
        """
        self.logger.info(f"Starting search for '{word}'")
        all_results: List[SearchResult] = []
        
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            self.logger.debug(f"Trying strategy: {strategy_name}")
            
            result = strategy.search(word, self.dictionary_path)
            all_results.append(result)
            
            # If we found matches, return this result
            if result.has_matches():
                self.logger.info(f"Found matches using strategy: {strategy_name}")
                return result, all_results
        
        # If we reach here, no strategy found matches
        self.logger.warning(f"No matches found for '{word}' using any strategy")
        return all_results[-1], all_results  # Return the last result as the "best" one 