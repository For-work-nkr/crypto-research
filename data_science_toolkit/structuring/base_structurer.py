"""
Base structuring module providing common functionality for all data structuring operations.
"""

import os
import json
import csv
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseStructurer(ABC):
    """
    Abstract base class for all data structuring operations.
    
    This class provides common functionality for data structuring,
    including loading data, transforming it into structured formats,
    and saving the results.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the structurer with optional configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        self.data = None
        self.structured_data = None
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def load_data(self, input_path: str) -> Union[pd.DataFrame, List[Dict], None]:
        """
        Load data from a file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Loaded data as DataFrame or list of dictionaries
        """
        try:
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.csv':
                self.data = pd.read_csv(input_path)
                logger.info(f"Loaded CSV data from {input_path}, shape: {self.data.shape}")
            elif file_ext == '.json':
                with open(input_path, 'r') as f:
                    self.data = json.load(f)
                
                # Convert to DataFrame if it's a list of dictionaries
                if isinstance(self.data, list):
                    logger.info(f"Loaded JSON data from {input_path}, records: {len(self.data)}")
                    if self.data and isinstance(self.data[0], dict):
                        self.data = pd.DataFrame(self.data)
                        logger.info(f"Converted JSON to DataFrame, shape: {self.data.shape}")
            elif file_ext in ['.xlsx', '.xls']:
                self.data = pd.read_excel(input_path)
                logger.info(f"Loaded Excel data from {input_path}, shape: {self.data.shape}")
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    @abstractmethod
    def structure(self) -> Union[pd.DataFrame, List[Dict], Any]:
        """
        Structure the loaded data.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Structured data in the appropriate format
        """
        pass
    
    def save_data(self, output_path: str, format: Optional[str] = None) -> bool:
        """
        Save structured data to a file.
        
        Args:
            output_path: Path to save the data
            format: Format to save the data (inferred from output_path if None)
            
        Returns:
            True if successful, False otherwise
        """
        if self.structured_data is None:
            logger.warning("No structured data to save")
            return False
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Infer format from output path if not specified
            if format is None:
                format = os.path.splitext(output_path)[1].lower().lstrip('.')
            
            # Handle different data types and formats
            if isinstance(self.structured_data, pd.DataFrame):
                if format == 'csv':
                    self.structured_data.to_csv(output_path, index=False)
                elif format == 'json':
                    self.structured_data.to_json(output_path, orient='records', indent=2)
                elif format in ['xlsx', 'xls']:
                    self.structured_data.to_excel(output_path, index=False)
                elif format == 'parquet':
                    self.structured_data.to_parquet(output_path, index=False)
                elif format == 'feather':
                    self.structured_data.to_feather(output_path)
                else:
                    logger.error(f"Unsupported format for DataFrame: {format}")
                    return False
            elif isinstance(self.structured_data, dict) or (isinstance(self.structured_data, list) and self.structured_data and isinstance(self.structured_data[0], dict)):
                if format == 'json':
                    with open(output_path, 'w') as f:
                        json.dump(self.structured_data, f, indent=2)
                elif format == 'csv' and isinstance(self.structured_data, list):
                    # Convert list of dicts to CSV
                    df = pd.DataFrame(self.structured_data)
                    df.to_csv(output_path, index=False)
                else:
                    logger.error(f"Unsupported format for dict/list: {format}")
                    return False
            else:
                logger.error(f"Unsupported data type for saving: {type(self.structured_data)}")
                return False
            
            logger.info(f"Saved structured data to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def add_structuring_metadata(self) -> None:
        """
        Add metadata about the structuring process.
        """
        if self.structured_data is None:
            return
        
        metadata = {
            "structurer": self.__class__.__name__,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if isinstance(self.structured_data, pd.DataFrame):
            # Add metadata as new columns
            for key, value in metadata.items():
                self.structured_data[f"_structuring_{key}"] = value
        elif isinstance(self.structured_data, list) and self.structured_data and isinstance(self.structured_data[0], dict):
            # Add metadata to each record
            for record in self.structured_data:
                record["_structuring_metadata"] = metadata
        elif isinstance(self.structured_data, dict):
            # Add metadata to the dictionary
            self.structured_data["_structuring_metadata"] = metadata
    
    def run(self, input_path: str, output_path: Optional[str] = None, format: Optional[str] = None, add_metadata: bool = True) -> Any:
        """
        Run the structurer and optionally save the results.
        
        Args:
            input_path: Path to input file
            output_path: Path to save the structured data (optional)
            format: Format to save the data (inferred from output_path if None)
            add_metadata: Whether to add metadata about the structuring process
            
        Returns:
            Structured data
        """
        try:
            logger.info(f"Starting {self.__class__.__name__}")
            
            # Load data
            if self.load_data(input_path) is None:
                return None
            
            # Structure data
            self.structured_data = self.structure()
            
            if self.structured_data is None:
                logger.error("Structuring process returned None")
                return None
            
            if add_metadata:
                self.add_structuring_metadata()
            
            if output_path:
                self.save_data(output_path, format)
            
            logger.info(f"Completed {self.__class__.__name__}")
            return self.structured_data
        except Exception as e:
            logger.error(f"Error running structurer: {e}")
            return None


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Base structurer module')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--format', type=str, help='Output format (inferred from output path if not specified)')
    parser.add_argument('--no-metadata', action='store_true', help='Do not add metadata to structured data')
    
    return parser.parse_args()


if __name__ == "__main__":
    # This script is not meant to be run directly
    logger.error("This is an abstract base class and cannot be run directly")
