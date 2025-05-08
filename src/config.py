#!/usr/bin/env python3

# Dictionary configuration
DEFAULT_DICTIONARY_PATH = '../kajaia.txt'

# Search configuration
MAX_EXACT_MATCHES = 5  # Maximum number of exact matches to display
MAX_SIMILAR_MATCHES = 3  # Maximum number of similar matches to display
MAX_PARTIAL_MATCHES = 5  # Maximum number of partial matches to display

# Edit distance configuration
DEFAULT_EDIT_DISTANCE = 1  # Default edit distance for similarity search
LONG_WORD_EDIT_DISTANCE = 2  # Edit distance for longer words
LONG_WORD_MIN_LENGTH = 5  # Minimum length for a word to be considered "long"

# API configuration
DEFAULT_MODEL = "gpt-4o"  # Default OpenAI model
FALLBACK_MODEL = "text-davinci-003"  # Fallback model for older API versions
MAX_TOKENS = 2000  # Maximum tokens for API responses

# Output formatting
DIVIDER_LENGTH = 40  # Length of divider lines in output
DIVIDER_CHAR = '-'  # Character to use for dividers

# Terminal colors
COLORS = {
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'MAGENTA': '\033[95m',
    'CYAN': '\033[96m',
    'WHITE': '\033[97m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'END': '\033[0m'
}

# Debug configuration
DEBUG = False  # Set to True to enable debug output

# Logging configuration
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_TO_FILE = False
LOG_FILE = 'argo.log' 