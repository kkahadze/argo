#!/usr/bin/env python3
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import os
import logging
import config

class DictionaryHandler:
    """
    Handles dictionary loading and caching for better performance.
    Implements a singleton pattern so the dictionary is only loaded once.
    """
    _instance = None
    _dictionaries: Dict[str, List[str]] = {}
    _logger = logging.getLogger('DictionaryHandler')
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DictionaryHandler, cls).__new__(cls)
            cls._initialize_logging()
        return cls._instance
    
    @classmethod
    def _initialize_logging(cls):
        """Initialize logging for this class"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(config.LOG_FORMAT)
        handler.setFormatter(formatter)
        cls._logger.addHandler(handler)
        cls._logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        if config.LOG_TO_FILE:
            file_handler = logging.FileHandler(config.LOG_FILE)
            file_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)
    
    def load_dictionary(self, file_path: str) -> List[str]:
        """
        Load a dictionary from a file or return cached version if available
        
        Args:
            file_path: Path to the dictionary file
            
        Returns:
            List of lines in the dictionary file
        """
        # Use absolute path as the cache key
        abs_path = os.path.abspath(file_path)
        
        # Check if this dictionary has already been loaded
        if abs_path in self._dictionaries:
            self._logger.debug(f"Using cached dictionary: {abs_path}")
            return self._dictionaries[abs_path]
        
        # Load the dictionary
        try:
            self._logger.info(f"Loading dictionary: {abs_path}")
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Store in cache
            self._dictionaries[abs_path] = lines
            self._logger.info(f"Loaded dictionary with {len(lines)} entries")
            return lines
            
        except Exception as e:
            self._logger.error(f"Error loading dictionary {abs_path}: {str(e)}")
            raise
    
    def get_dictionary(self, file_path: str) -> List[str]:
        """
        Get a dictionary, loading it if necessary
        
        Args:
            file_path: Path to the dictionary file
            
        Returns:
            List of lines in the dictionary file
        """
        return self.load_dictionary(file_path)
    
    def clear_cache(self) -> None:
        """Clear the dictionary cache"""
        self._dictionaries.clear()
        self._logger.info("Dictionary cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'num_dictionaries': len(self._dictionaries),
            'paths': list(self._dictionaries.keys()),
            'total_entries': sum(len(d) for d in self._dictionaries.values())
        } 