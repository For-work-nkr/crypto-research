"""
Crypto Prediction Package
A package for cryptocurrency price prediction and analysis.
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            If None, looks for 'config.yaml' or 'config.json' in the current directory.
            
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If no configuration file is found
        ValueError: If the configuration file format is not supported
    """
    if config_path is None:
        # Look for config files in the current directory
        if os.path.exists('config.yaml'):
            config_path = 'config.yaml'
        elif os.path.exists('config.json'):
            config_path = 'config.json'
        else:
            raise FileNotFoundError(
                "No configuration file found. Please provide a config.yaml or config.json file."
            )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration based on file extension
    file_ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r') as f:
        if file_ext == '.yaml' or file_ext == '.yml':
            config = yaml.safe_load(f)
        elif file_ext == '.json':
            config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {file_ext}. "
                "Please use YAML (.yaml, .yml) or JSON (.json) format."
            )
    
    return config

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
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

# Initialize the project
def init_project():
    """Initialize the project environment."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Cryptocurrency Price Prediction System initialized")
    
    return config, logger

# If this file is run directly, initialize the project
if __name__ == "__main__":
    config, logger = init_project()
    logger.info("Project initialization complete")

__version__ = '0.1.0'
