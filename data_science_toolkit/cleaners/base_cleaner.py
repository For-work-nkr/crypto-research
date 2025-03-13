"""
Base cleaner module providing common functionality for all data cleaners.
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

class BaseCleaner(ABC):
    """
    Abstract base class for all data cleaners.
    
    This class provides common functionality for data cleaning,
    including loading data, saving cleaned data, and logging.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the cleaner with optional configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        self.data = None
        self.cleaned_data = None
        
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
    def clean(self) -> Union[pd.DataFrame, List[Dict]]:
        """
        Clean the loaded data.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Cleaned data as DataFrame or list of dictionaries
        """
        pass
    
    def save_data(self, output_path: str, format: str = 'csv') -> bool:
        """
        Save cleaned data to a file.
        
        Args:
            output_path: Path to save the data
            format: Format to save the data (csv, json)
            
        Returns:
            True if successful, False otherwise
        """
        if self.cleaned_data is None:
            logger.warning("No cleaned data to save")
            return False
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(self.cleaned_data, list) and self.cleaned_data and isinstance(self.cleaned_data[0], dict):
                df = pd.DataFrame(self.cleaned_data)
            elif isinstance(self.cleaned_data, pd.DataFrame):
                df = self.cleaned_data
            else:
                logger.error(f"Unsupported data type for saving: {type(self.cleaned_data)}")
                return False
            
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                if isinstance(self.cleaned_data, pd.DataFrame):
                    records = df.to_dict(orient='records')
                    with open(output_path, 'w') as f:
                        json.dump(records, f, indent=2)
                else:
                    with open(output_path, 'w') as f:
                        json.dump(self.cleaned_data, f, indent=2)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Saved cleaned data to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def add_cleaning_metadata(self) -> None:
        """
        Add metadata about the cleaning process.
        """
        if self.cleaned_data is None:
            return
        
        metadata = {
            "cleaner": self.__class__.__name__,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if isinstance(self.cleaned_data, pd.DataFrame):
            # Add metadata as new columns
            for key, value in metadata.items():
                self.cleaned_data[f"_cleaning_{key}"] = value
        elif isinstance(self.cleaned_data, list) and self.cleaned_data and isinstance(self.cleaned_data[0], dict):
            # Add metadata to each record
            for record in self.cleaned_data:
                record["_cleaning_metadata"] = metadata
    
    def run(self, input_path: str, output_path: Optional[str] = None, format: str = 'csv', add_metadata: bool = True) -> Union[pd.DataFrame, List[Dict], None]:
        """
        Run the cleaner and optionally save the results.
        
        Args:
            input_path: Path to input file
            output_path: Path to save the cleaned data (optional)
            format: Format to save the data (csv, json)
            add_metadata: Whether to add metadata about the cleaning process
            
        Returns:
            Cleaned data as DataFrame or list of dictionaries
        """
        try:
            logger.info(f"Starting {self.__class__.__name__}")
            
            # Load data
            if self.load_data(input_path) is None:
                return None
            
            # Clean data
            self.cleaned_data = self.clean()
            
            if self.cleaned_data is None:
                logger.error("Cleaning process returned None")
                return None
            
            if add_metadata:
                self.add_cleaning_metadata()
            
            if output_path:
                self.save_data(output_path, format)
            
            logger.info(f"Completed {self.__class__.__name__}")
            return self.cleaned_data
        except Exception as e:
            logger.error(f"Error running cleaner: {e}")
            return None


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Base cleaner module')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv',
                        help='Output format (csv, json)')
    parser.add_argument('--no-metadata', action='store_true', help='Do not add metadata to cleaned data')
    
    return parser.parse_args()


if __name__ == "__main__":
    # This script is not meant to be run directly
    logger.error("This is an abstract base class and cannot be run directly")
