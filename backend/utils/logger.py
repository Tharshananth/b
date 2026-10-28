"""Logging configuration"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import get_logging_config

def setup_logger(name: str = "pingus") -> logging.Logger:
    """
    Setup application logger
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    config = get_logging_config()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if config.console.enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level))
        formatter = logging.Formatter(config.format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.file.enabled:
        log_path = Path(config.file.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.file.max_bytes,
            backupCount=config.file.backup_count
        )
        file_handler.setLevel(getattr(logging, config.level))
        formatter = logging.Formatter(config.format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
