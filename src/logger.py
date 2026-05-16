#!/usr/bin/env python3
"""
Logging configuration for the Mingrelian translator.
Provides structured logging for debugging prompts, responses, and errors.
"""
import logging
import os
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps


# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging level from environment variable (default: INFO)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()


def setup_logger(name: str = 'mingrelian_translator') -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    logger.propagate = False
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler - main log (all levels)
    log_file = LOGS_DIR / f'translator_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # File handler - errors only
    error_log_file = LOGS_DIR / f'errors_{datetime.now().strftime("%Y%m%d")}.log'
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger


def log_translation_request(
    logger: logging.Logger,
    input_text: str,
    source_lang: str,
    target_lang: str,
    provider: str,
    model: str
) -> None:
    """Log a translation request."""
    logger.info(
        f"Translation request: '{input_text[:50]}...' "
        f"({source_lang} → {target_lang}) using {provider}/{model}"
    )


def log_prompt(
    logger: logging.Logger,
    prompt: str,
    source_lang: str,
    target_lang: str,
    truncate: bool = True
) -> None:
    """
    Log the constructed prompt for debugging.
    
    Args:
        logger: Logger instance
        prompt: The full prompt sent to LLM
        source_lang: Source language
        target_lang: Target language
        truncate: Whether to truncate long prompts in console (still logs full to file)
    """
    # Log full prompt to file at DEBUG level
    logger.debug(f"Full prompt ({source_lang} → {target_lang}):\n{prompt}")
    
    # Log truncated version to console at INFO level
    if truncate and len(prompt) > 500:
        logger.info(f"Prompt preview: {prompt[:500]}... [truncated, see log file for full prompt]")
    else:
        logger.info(f"Prompt: {prompt}")


def log_llm_response(
    logger: logging.Logger,
    response: str,
    source_lang: str,
    target_lang: str,
    truncate: bool = True
) -> None:
    """
    Log the LLM response for debugging.
    
    Args:
        logger: Logger instance
        response: The full LLM response
        source_lang: Source language
        target_lang: Target language
        truncate: Whether to truncate long responses in console
    """
    # Log full response to file at DEBUG level
    logger.debug(f"Full LLM response ({source_lang} → {target_lang}):\n{response}")
    
    # Log truncated version to console
    if truncate and len(response) > 300:
        logger.info(f"Response preview: {response[:300]}... [truncated, see log file for full response]")
    else:
        logger.info(f"Response: {response}")


def log_translation_result(
    logger: logging.Logger,
    translation: str,
    source_lang: str,
    target_lang: str
) -> None:
    """Log the extracted translation."""
    logger.info(f"Translation result ({source_lang} → {target_lang}): '{translation}'")


def log_instant_lookup(
    logger: logging.Logger,
    input_text: str,
    translation: str,
    method: str
) -> None:
    """Log instant dictionary/Google Translate lookups."""
    logger.info(f"Instant lookup ({method}): '{input_text}' → '{translation}'")


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        error: The exception
        context: Additional context (input text, languages, etc.)
    """
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg += f"\nContext: {json.dumps(context, indent=2, ensure_ascii=False)}"
    
    logger.error(error_msg, exc_info=True)


@contextmanager
def log_timing(logger: logging.Logger, operation: str):
    """
    Context manager for timing operations.
    
    Usage:
        with log_timing(logger, "Dictionary lookup"):
            # ... code to time ...
    """
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    logger.info(f"⏱️  {operation}: {elapsed:.3f}s")


def timed_function(logger: logging.Logger):
    """
    Decorator for timing functions.
    
    Usage:
        @timed_function(logger)
        def my_function():
            # ... code ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"⏱️  {func.__name__}: {elapsed:.3f}s")
            return result
        return wrapper
    return decorator


def log_stage_timing(
    logger: logging.Logger,
    stage: str,
    elapsed_time: float,
    details: Optional[str] = None
) -> None:
    """
    Log timing for a specific stage with optional details.
    
    Args:
        logger: Logger instance
        stage: Name of the stage (e.g., "LLM Call", "Dictionary Search")
        elapsed_time: Time in seconds
        details: Optional additional details
    """
    msg = f"⏱️  {stage}: {elapsed_time:.3f}s"
    if details:
        msg += f" ({details})"
    logger.info(msg)
