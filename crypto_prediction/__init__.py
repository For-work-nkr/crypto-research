"""
Main initialization file for the cryptocurrency prediction project.
This file sets up the project environment and imports.
"""

import os
import sys
import logging
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Setup logging
def setup_logging(config=None):
    """Set up logging configuration based on config or defaults."""
    if config is None:
        log_level = "INFO"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_file = "logs/crypto_prediction.log"
    else:
        log_level = config.get("logging", {}).get("level", "INFO")
        log_format = config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = config.get("logging", {}).get("file", "logs/crypto_prediction.log")
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Load configuration
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    config_file = os.path.join(PROJECT_ROOT, config_path)
    
    if not os.path.exists(config_file):
        example_config = os.path.join(PROJECT_ROOT, "config.example.yaml")
        if os.path.exists(example_config):
            logging.warning(f"Config file {config_file} not found. Please copy from {example_config} and update with your settings.")
        else:
            logging.error(f"Neither config file {config_file} nor example config found.")
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

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
