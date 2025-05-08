#!/usr/bin/env python3
import logging
from typing import Optional
import config

class LoggerSetup:
    """
    Centralized logging setup for the application.
    Ensures consistent logging configuration across all modules.
    """
    
    _initialized = False
    _root_logger = None
    
    @classmethod
    def setup(cls) -> None:
        """Set up application-wide logging"""
        if cls._initialized:
            return
        
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Set log level based on config
        log_level = getattr(logging, config.LOG_LEVEL)
        root_logger.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(config.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        root_logger.addHandler(console_handler)
        
        # Add file handler if configured
        if config.LOG_TO_FILE:
            file_handler = logging.FileHandler(config.LOG_FILE)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
        cls._root_logger = root_logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the given name
        
        Args:
            name: Logger name, typically the module name
            
        Returns:
            Configured logger
        """
        if not cls._initialized:
            cls.setup()
        return logging.getLogger(name)

# Define convenience functions to get loggers
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name
    
    Args:
        name: Logger name, typically the module name
        
    Returns:
        Configured logger
    """
    return LoggerSetup.get_logger(name)

# Setup logging when this module is imported
LoggerSetup.setup() 