"""
Logging configuration module for the crypto prediction system.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(config):
    """
    Set up logging configuration based on the provided config dictionary.
    
    Args:
        config (dict): Configuration dictionary containing logging settings
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logging configuration from config
    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', 'logs')
    log_level = log_config.get('level', 'INFO')
    max_bytes = log_config.get('max_bytes', 10 * 1024 * 1024)  # 10MB default
    backup_count = log_config.get('backup_count', 5)
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('crypto_prediction')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    log_file = os.path.join(
        log_dir,
        f'crypto_prediction_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger 