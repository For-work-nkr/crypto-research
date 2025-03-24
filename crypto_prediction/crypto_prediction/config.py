"""
Configuration loading module for the crypto prediction system.
"""

import os
import yaml
import json
from typing import Dict, Any

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