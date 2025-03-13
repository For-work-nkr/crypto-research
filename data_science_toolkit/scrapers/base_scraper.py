"""
Base scraper module providing common functionality for all scrapers.
"""

import os
import json
import csv
import logging
import argparse
import datetime
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.
    
    This class provides common functionality for data scraping,
    including configuration loading, data saving, and logging.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the scraper with optional configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        self.session = requests.Session()
        self.data = []
        
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
    
    @abstractmethod
    def scrape(self, *args, **kwargs) -> List[Dict]:
        """
        Scrape data from the source.
        
        This method must be implemented by all subclasses.
        
        Returns:
            List of dictionaries containing scraped data
        """
        pass
    
    def save_data(self, output_path: str, format: str = 'json') -> bool:
        """
        Save scraped data to a file.
        
        Args:
            output_path: Path to save the data
            format: Format to save the data (json, csv)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.data:
            logger.warning("No data to save")
            return False
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.data, f, indent=2)
            elif format.lower() == 'csv':
                if not self.data:
                    logger.warning("No data to save as CSV")
                    return False
                
                # Get fieldnames from first item
                fieldnames = self.data[0].keys()
                
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.data)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Saved {len(self.data)} records to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def add_metadata(self) -> None:
        """
        Add metadata to the scraped data.
        """
        if not self.data:
            return
        
        metadata = {
            "scraper": self.__class__.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "record_count": len(self.data)
        }
        
        # Add metadata to each record
        for record in self.data:
            record["_metadata"] = metadata
    
    def run(self, output_path: Optional[str] = None, format: str = 'json', add_metadata: bool = True) -> List[Dict]:
        """
        Run the scraper and optionally save the results.
        
        Args:
            output_path: Path to save the data (optional)
            format: Format to save the data (json, csv)
            add_metadata: Whether to add metadata to the scraped data
            
        Returns:
            List of dictionaries containing scraped data
        """
        try:
            logger.info(f"Starting {self.__class__.__name__}")
            self.data = self.scrape()
            
            if add_metadata:
                self.add_metadata()
            
            if output_path:
                self.save_data(output_path, format)
            
            logger.info(f"Completed {self.__class__.__name__}, scraped {len(self.data)} records")
            return self.data
        except Exception as e:
            logger.error(f"Error running scraper: {e}")
            return []


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Base scraper module')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                        help='Output format (json, csv)')
    parser.add_argument('--no-metadata', action='store_true', help='Do not add metadata to scraped data')
    
    return parser.parse_args()


if __name__ == "__main__":
    # This script is not meant to be run directly
    logger.error("This is an abstract base class and cannot be run directly")
