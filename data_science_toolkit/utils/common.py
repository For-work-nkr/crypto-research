"""
Utility functions shared across all modules.
"""

import os
import json
import logging
import datetime
import hashlib
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def load_json(file_path: str) -> Union[Dict, List, None]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data or None if error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def save_json(data: Union[Dict, List], file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for JSON formatting
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def generate_file_hash(file_path: str) -> Optional[str]:
    """
    Generate a hash for a file to track changes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hash string or None if error
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return None

def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        Timestamp string
    """
    return datetime.datetime.now().isoformat()

def create_filename_with_timestamp(base_name: str, extension: str) -> str:
    """
    Create a filename with a timestamp.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension.lstrip('.')}"

def detect_file_type(file_path: str) -> str:
    """
    Detect file type from extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type string (e.g., 'csv', 'json')
    """
    return os.path.splitext(file_path)[1].lower().lstrip('.')

def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
        return result
    return wrapper
