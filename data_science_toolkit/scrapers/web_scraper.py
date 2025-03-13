"""
Web scraper module for extracting data from websites.
"""

import os
import json
import logging
import argparse
import time
import random
from typing import Dict, List, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Import base scraper
from scrapers.base_scraper import BaseScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebScraper(BaseScraper):
    """
    Scraper for extracting data from websites using BeautifulSoup.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the web scraper.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        super().__init__(config_path)
        
        # Set default headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Update headers from config if provided
        if 'headers' in self.config:
            self.headers.update(self.config.get('headers', {}))
        
        # Set default request parameters
        self.timeout = self.config.get('timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
        self.retry_delay = self.config.get('retry_delay', 2)
        self.random_delay = self.config.get('random_delay', False)
        self.delay_range = self.config.get('delay_range', [1, 3])
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a web page and parse it with BeautifulSoup.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if error
        """
        for attempt in range(self.retry_count):
            try:
                # Add random delay if configured
                if self.random_delay and attempt > 0:
                    delay = random.uniform(self.delay_range[0], self.delay_range[1])
                    logger.info(f"Random delay before retry: {delay:.2f} seconds")
                    time.sleep(delay)
                
                # Make the request
                response = self.session.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                logger.info(f"Successfully fetched {url}")
                return soup
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching {url} (attempt {attempt+1}/{self.retry_count}): {e}")
                
                if attempt < self.retry_count - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch {url} after {self.retry_count} attempts")
                    return None
    
    def extract_data(self, soup: BeautifulSoup, selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract data from a BeautifulSoup object using CSS selectors.
        
        Args:
            soup: BeautifulSoup object
            selectors: Dictionary mapping field names to CSS selectors
            
        Returns:
            Dictionary with extracted data
        """
        result = {}
        
        for field, selector in selectors.items():
            try:
                element = soup.select_one(selector)
                if element:
                    result[field] = element.get_text(strip=True)
                else:
                    result[field] = None
                    logger.warning(f"Selector '{selector}' for field '{field}' returned no results")
            except Exception as e:
                logger.error(f"Error extracting field '{field}' with selector '{selector}': {e}")
                result[field] = None
        
        return result
    
    def extract_list(self, soup: BeautifulSoup, list_selector: str, item_selectors: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Extract a list of items from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            list_selector: CSS selector for the list container
            item_selectors: Dictionary mapping field names to CSS selectors relative to each list item
            
        Returns:
            List of dictionaries with extracted data
        """
        results = []
        
        try:
            items = soup.select(list_selector)
            logger.info(f"Found {len(items)} items with selector '{list_selector}'")
            
            for item in items:
                item_data = {}
                
                for field, selector in item_selectors.items():
                    try:
                        element = item.select_one(selector)
                        if element:
                            item_data[field] = element.get_text(strip=True)
                        else:
                            item_data[field] = None
                            logger.warning(f"Selector '{selector}' for field '{field}' returned no results in list item")
                    except Exception as e:
                        logger.error(f"Error extracting field '{field}' with selector '{selector}' in list item: {e}")
                        item_data[field] = None
                
                results.append(item_data)
        
        except Exception as e:
            logger.error(f"Error extracting list with selector '{list_selector}': {e}")
        
        return results
    
    def scrape(self, url: Optional[str] = None, selectors: Optional[Dict[str, str]] = None, list_selector: Optional[str] = None, item_selectors: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Scrape data from a website.
        
        Args:
            url: URL to scrape (overrides config)
            selectors: Dictionary mapping field names to CSS selectors (overrides config)
            list_selector: CSS selector for list container (overrides config)
            item_selectors: Dictionary mapping field names to CSS selectors for list items (overrides config)
            
        Returns:
            List of dictionaries containing scraped data
        """
        # Use parameters or fall back to config
        url = url or self.config.get('url')
        selectors = selectors or self.config.get('selectors', {})
        list_selector = list_selector or self.config.get('list_selector')
        item_selectors = item_selectors or self.config.get('item_selectors', {})
        
        if not url:
            logger.error("No URL provided")
            return []
        
        # Fetch the page
        soup = self.fetch_page(url)
        if not soup:
            return []
        
        results = []
        
        # Extract page-level data if selectors are provided
        if selectors:
            page_data = self.extract_data(soup, selectors)
            
            # If we're not extracting a list, return the page data
            if not list_selector:
                results.append(page_data)
        
        # Extract list data if list_selector and item_selectors are provided
        if list_selector and item_selectors:
            list_data = self.extract_list(soup, list_selector, item_selectors)
            
            # If we have page-level data, add it to each list item
            if selectors and list_data:
                for item in list_data:
                    item.update({f"page_{k}": v for k, v in page_data.items()})
            
            results.extend(list_data)
        
        # Add URL to each result
        for item in results:
            item['source_url'] = url
        
        return results


def main():
    """
    Main function to run the scraper from command line.
    """
    parser = argparse.ArgumentParser(description='Web scraper')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--url', type=str, help='URL to scrape')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                        help='Output format (json, csv)')
    parser.add_argument('--no-metadata', action='store_true', help='Do not add metadata to scraped data')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = WebScraper(args.config)
    
    # Run scraper
    scraper.run(
        output_path=args.output,
        format=args.format,
        add_metadata=not args.no_metadata
    )


if __name__ == "__main__":
    main()
