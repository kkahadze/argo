#!/usr/bin/env python3
from typing import List, Dict, Any, Optional, Union
import config
from search_engine import SearchResult
from transliterate import latinized_to_mkhedruli
from logger import get_logger

class Formatter:
    """Formats search results for display"""
    
    def __init__(self) -> None:
        self.output_lines: List[str] = []
        self.logger = get_logger(__class__.__name__)
    
    def colorize(self, text: str, color: str) -> str:
        """
        Apply color formatting to text for terminal display
        
        Args:
            text: The text to colorize
            color: The color to apply
            
        Returns:
            Colorized text
        """
        if color in config.COLORS:
            return f"{config.COLORS[color]}{text}{config.COLORS['END']}"
        return text
    
    def add_line(self, line: str, color: Optional[str] = None) -> None:
        """
        Add a line to the output
        
        Args:
            line: The line to add
            color: Optional color to apply
        """
        if color:
            line = self.colorize(line, color)
        self.output_lines.append(line)
        self.logger.debug(f"Added line: {line[:50]}{'...' if len(line) > 50 else ''}")
    
    def add_divider(self) -> None:
        """Add a divider line to the output"""
        divider = "\n" + config.DIVIDER_CHAR * config.DIVIDER_LENGTH + "\n"
        self.output_lines.append(divider)
    
    def format_word_info(self, word: str, is_mkhedruli: bool = False) -> None:
        """
        Format and add word information to output
        
        Args:
            word: The word to format
            is_mkhedruli: Whether the word is in Mkhedruli script
        """
        self.logger.debug(f"Formatting word info: {word}, is_mkhedruli={is_mkhedruli}")
        if is_mkhedruli:
            self.add_line(f"Input in Mkhedruli script: {word}")
            latinized = None  # This should be provided by the caller
            self.add_line(f"Converted to latinized form: {latinized}")
        else:
            self.add_line(f"Latinized Mingrelian word: {word}")
            self.add_line(f"Mkhedruli Mingrelian word: {latinized_to_mkhedruli(word)}")
    
    def format_exact_match(self, result: SearchResult) -> None:
        """
        Format exact match results
        
        Args:
            result: The search result to format
        """
        if not result.has_matches():
            return
        
        self.logger.debug(f"Formatting {len(result.matches)} exact matches")
        self.add_line(result.description)
        for match in result.matches:
            self.add_line(f"Entry: {match['lemma']}")
            if 'definition' in match and match['definition']:
                self.add_line(f"Definition: {match['definition']}")
            self.add_line(match['entry_text'])
            self.add_divider()
    
    def format_lemmatized_match(self, result: SearchResult) -> None:
        """
        Format lemmatized match results
        
        Args:
            result: The search result to format
        """
        if not result.has_matches():
            return
        
        self.logger.debug(f"Formatting {len(result.matches)} lemmatized matches")
        self.add_line(result.description)
        for match in result.matches:
            lemmatized_word = match.get('lemmatized_word', 'unknown')
            self.add_line(f"Match for lemmatized form '{lemmatized_word}' of '{result.word}':")
            self.add_line(f"Entry: {match['lemma']}")
            if 'definition' in match and match['definition']:
                self.add_line(f"Definition: {match['definition']}")
            self.add_line(match['entry_text'])
            self.add_divider()
        
        if result.limited:
            self.add_line(f"Note: {result.total_matches - len(result.matches)} additional matches for lemmatized form were found but not displayed.")
    
    def format_edit_distance_match(self, result: SearchResult) -> None:
        """
        Format edit distance match results
        
        Args:
            result: The search result to format
        """
        if not result.has_matches():
            return
        
        self.logger.debug(f"Formatting {len(result.matches)} edit distance matches with strategy {result.strategy_name}")
        self.add_line(result.description)
        for match in result.matches:
            edit_distance = match.get('edit_distance', 'unknown')
            search_word = match.get('search_word', result.word)
            
            # Different format depending on whether this is for lemmatized form or original word
            if 'lemmatized' in result.strategy_name:
                self.add_line(f"Similar lemma (edit distance {edit_distance}) to lemmatized form: '{match['lemma']}'")
                self.add_line(f"Close match (edit distance {edit_distance}) for lemmatized form '{search_word}' of '{result.word}':")
            else:
                self.add_line(f"Similar lemma (edit distance {edit_distance}): '{match['lemma']}'")
                self.add_line(f"Close match for '{result.word}':")
            
            self.add_line(f"Lemma: {match['lemma']}")
            self.add_line(match['entry_text'])
            self.add_divider()
        
        if result.limited:
            self.add_line(f"Note: {result.total_matches - len(result.matches)} additional similar matches were found but not displayed.")
    
    def format_containing_match(self, result: SearchResult) -> None:
        """
        Format containing match results
        
        Args:
            result: The search result to format
        """
        if not result.has_matches():
            return
        
        self.logger.debug(f"Formatting {len(result.matches)} containing matches")
        self.add_line(result.description)
        for match in result.matches:
            georgian_word = match.get('georgian_word', latinized_to_mkhedruli(result.word))
            self.add_line(f"Partial match for '{result.word}' ({georgian_word}):")
            self.add_line(f"Entry: {match['lemma']}")
            if 'context' in match and match['context']:
                self.add_line(f"Context: {match['context']}")
            self.add_line(match['entry_text'])
            self.add_divider()
        
        if result.limited:
            self.add_line(f"Note: {result.total_matches - len(result.matches)} additional partial matches were found but not displayed.")
    
    def format_search_result(self, result: SearchResult) -> None:
        """
        Format any search result based on its strategy type
        
        Args:
            result: The search result to format
        """
        self.logger.debug(f"Formatting search result for strategy: {result.strategy_name}")
        if not result.has_matches():
            if result.description:
                self.add_line(result.description)
            return
        
        if result.strategy_name == "exact_match":
            self.format_exact_match(result)
        elif result.strategy_name == "lemmatized_match":
            self.format_lemmatized_match(result)
        elif "edit_distance" in result.strategy_name:
            self.format_edit_distance_match(result)
        elif result.strategy_name == "containing_word":
            self.format_containing_match(result)
        else:
            # Generic fallback formatter
            self.logger.warning(f"Unknown strategy type: {result.strategy_name}")
            self.add_line(result.description)
            for i, match in enumerate(result.matches):
                self.add_line(f"Match {i+1}:")
                for key, value in match.items():
                    if key == 'entry_text':
                        self.add_line(value)
                    else:
                        self.add_line(f"{key}: {value}")
                self.add_divider()
    
    def format_all_entries_section(self, entries: List[str]) -> None:
        """
        Format the complete entries section
        
        Args:
            entries: List of entry texts to format
        """
        self.logger.debug(f"Formatting all entries section with {len(entries)} entries")
        self.add_line("\n" + "=" * config.DIVIDER_LENGTH)
        self.add_line("COMPLETE TRANSLATION ENTRIES:")
        self.add_line("=" * config.DIVIDER_LENGTH + "\n")
        
        for entry in entries:
            self.add_line(entry)
            self.add_divider()
    
    def format_api_response(self, response_text: str, is_followup: bool = False) -> str:
        """
        Format the API response for display
        
        Args:
            response_text: The response text from the API
            is_followup: Whether this is a follow-up response
            
        Returns:
            Formatted response text
        """
        self.logger.debug(f"Formatting API response, is_followup={is_followup}")
        header = "Follow-up Response:" if is_followup else "Initial Translation:"
        output = f"\n{self.colorize(header, 'CYAN')}\n"
        
        # Process the response line by line to apply formatting
        for line in response_text.split('\n'):
            # Apply special formatting to key elements
            if "Phrase in Mingrelian" in line or "Translation to" in line:
                # Bold headings
                output += self.colorize(line, 'CYAN') + "\n"
            elif line.startswith('**') and line.endswith('**'):
                # Bold text that's already marked with Markdown
                output += self.colorize(line, 'CYAN') + "\n"
            elif "Note:" in line:
                # Highlight notes
                output += self.colorize(line, 'YELLOW') + "\n"
            else:
                # Regular text
                output += self.colorize(line, 'CYAN') + "\n"
        
        return output
    
    def get_output(self) -> str:
        """
        Get the formatted output as a string
        
        Returns:
            All formatted output as a single string
        """
        result = "\n".join(self.output_lines)
        self.logger.debug(f"Retrieved output: {len(result)} characters")
        return result
    
    def clear(self) -> None:
        """Clear the formatter's output buffer"""
        self.logger.debug("Clearing formatter output buffer")
        self.output_lines = [] 