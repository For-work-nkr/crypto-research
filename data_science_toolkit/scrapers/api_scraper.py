"""
API scraper module for extracting data from REST APIs.
"""

import os
import json
import logging
import argparse
import time
import random
from typing import Dict, List, Any, Optional, Union
import requests
import urllib.parse

# Import base scraper
from scrapers.base_scraper import BaseScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIScraper(BaseScraper):
    """
    Scraper for extracting data from REST APIs.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API scraper.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        super().__init__(config_path)
        
        # Set default headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        # Update headers from config if provided
        if 'headers' in self.config:
            self.headers.update(self.config.get('headers', {}))
        
        # Set default request parameters
        self.base_url = self.config.get('base_url', '')
        self.endpoint = self.config.get('endpoint', '')
        self.method = self.config.get('method', 'GET')
        self.params = self.config.get('params', {})
        self.body = self.config.get('body', {})
        self.auth = self.config.get('auth', {})
        self.timeout = self.config.get('timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
        self.retry_delay = self.config.get('retry_delay', 2)
        self.random_delay = self.config.get('random_delay', False)
        self.delay_range = self.config.get('delay_range', [1, 3])
        self.pagination = self.config.get('pagination', {})
        self.rate_limit = self.config.get('rate_limit', {})
        
        # Set up authentication if provided
        if self.auth:
            auth_type = self.auth.get('type', '').lower()
            
            if auth_type == 'basic':
                username = self.auth.get('username', '')
                password = self.auth.get('password', '')
                self.session.auth = (username, password)
            elif auth_type == 'bearer':
                token = self.auth.get('token', '')
                self.headers['Authorization'] = f"Bearer {token}"
            elif auth_type == 'api_key':
                key_name = self.auth.get('key_name', 'api_key')
                key_value = self.auth.get('key_value', '')
                key_location = self.auth.get('key_location', 'query')
                
                if key_location == 'query':
                    self.params[key_name] = key_value
                elif key_location == 'header':
                    self.headers[key_name] = key_value
    
    def make_request(self, url: str, method: str = 'GET', params: Optional[Dict] = None, 
                    data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make an HTTP request to the API.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, PUT, DELETE)
            params: Query parameters
            data: Request body for POST/PUT
            headers: HTTP headers
            
        Returns:
            Response data as dictionary or None if error
        """
        # Use provided parameters or fall back to instance defaults
        method = method or self.method
        params = params or self.params
        data = data or self.body
        headers = headers or self.headers
        
        for attempt in range(self.retry_count):
            try:
                # Add random delay if configured
                if self.random_delay and attempt > 0:
                    delay = random.uniform(self.delay_range[0], self.delay_range[1])
                    logger.info(f"Random delay before retry: {delay:.2f} seconds")
                    time.sleep(delay)
                
                # Make the request
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data if method in ['POST', 'PUT', 'PATCH'] else None,
                    headers=headers,
                    timeout=self.timeout
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse the response
                if response.text:
                    try:
                        result = response.json()
                        logger.info(f"Successfully fetched data from {url}")
                        return result
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON from {url}")
                        return None
                else:
                    logger.warning(f"Empty response from {url}")
                    return {}
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error requesting {url} (attempt {attempt+1}/{self.retry_count}): {e}")
                
                if attempt < self.retry_count - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to request {url} after {self.retry_count} attempts")
                    return None
    
    def extract_data(self, response_data: Dict, data_path: Optional[str] = None) -> List[Dict]:
        """
        Extract relevant data from the API response.
        
        Args:
            response_data: API response data
            data_path: JSON path to the data array (e.g., 'results' or 'data.items')
            
        Returns:
            List of dictionaries containing extracted data
        """
        if not response_data:
            return []
        
        # If no data path is provided, return the response as is
        if not data_path:
            if isinstance(response_data, list):
                return response_data
            else:
                return [response_data]
        
        # Navigate to the specified data path
        current_data = response_data
        for key in data_path.split('.'):
            if isinstance(current_data, dict) and key in current_data:
                current_data = current_data[key]
            else:
                logger.error(f"Key '{key}' not found in response data at path '{data_path}'")
                return []
        
        # Ensure the result is a list
        if isinstance(current_data, list):
            return current_data
        else:
            logger.warning(f"Data at path '{data_path}' is not a list, wrapping in list")
            return [current_data]
    
    def handle_pagination(self, initial_response: Dict) -> List[Dict]:
        """
        Handle paginated API responses.
        
        Args:
            initial_response: Initial API response
            
        Returns:
            List of all items across all pages
        """
        if not self.pagination or not self.pagination.get('enabled', False):
            return self.extract_data(initial_response, self.pagination.get('data_path'))
        
        all_items = []
        current_response = initial_response
        page = 1
        max_pages = self.pagination.get('max_pages', 10)
        
        # Extract pagination type and parameters
        pagination_type = self.pagination.get('type', 'offset')
        next_page_key = self.pagination.get('next_page_key')
        page_param = self.pagination.get('page_param', 'page')
        size_param = self.pagination.get('size_param', 'size')
        size_value = self.pagination.get('size_value', 10)
        has_more_key = self.pagination.get('has_more_key')
        data_path = self.pagination.get('data_path')
        
        while True:
            # Extract items from current response
            items = self.extract_data(current_response, data_path)
            all_items.extend(items)
            
            logger.info(f"Retrieved {len(items)} items from page {page}")
            
            # Check if we've reached the maximum number of pages
            if page >= max_pages:
                logger.info(f"Reached maximum number of pages ({max_pages})")
                break
            
            # Check if there are more pages
            if pagination_type == 'offset':
                # Offset-based pagination
                if not items or len(items) < size_value:
                    logger.info("No more items to retrieve")
                    break
                
                # Prepare parameters for next page
                next_params = self.params.copy()
                next_params[page_param] = page + 1
                next_params[size_param] = size_value
                
            elif pagination_type == 'cursor':
                # Cursor-based pagination
                if has_more_key and not self._get_nested_value(current_response, has_more_key):
                    logger.info("No more items to retrieve")
                    break
                
                # Get next page cursor
                next_cursor = self._get_nested_value(current_response, next_page_key)
                if not next_cursor:
                    logger.info("No next page cursor found")
                    break
                
                # Prepare parameters for next page
                next_params = self.params.copy()
                next_params[page_param] = next_cursor
                
            else:
                logger.error(f"Unsupported pagination type: {pagination_type}")
                break
            
            # Apply rate limiting if configured
            if self.rate_limit and self.rate_limit.get('enabled', False):
                delay = self.rate_limit.get('delay_seconds', 1)
                logger.info(f"Rate limiting: waiting {delay} seconds before next request")
                time.sleep(delay)
            
            # Request next page
            url = self._build_url()
            current_response = self.make_request(url, params=next_params)
            
            if not current_response:
                logger.error("Failed to retrieve next page")
                break
            
            page += 1
        
        return all_items
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """
        Get a value from a nested dictionary using a dot-separated path.
        
        Args:
            data: Dictionary to extract value from
            path: Dot-separated path to the value
            
        Returns:
            Extracted value or None if not found
        """
        current = data
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _build_url(self) -> str:
        """
        Build the full URL for the API request.
        
        Returns:
            Complete URL
        """
        if self.base_url and self.endpoint:
            return urllib.parse.urljoin(self.base_url, self.endpoint)
        elif self.base_url:
            return self.base_url
        else:
            return self.endpoint
    
    def scrape(self, base_url: Optional[str] = None, endpoint: Optional[str] = None, 
              method: Optional[str] = None, params: Optional[Dict] = None, 
              body: Optional[Dict] = None) -> List[Dict]:
        """
        Scrape data from an API.
        
        Args:
            base_url: Base URL for the API (overrides config)
            endpoint: API endpoint (overrides config)
            method: HTTP method (overrides config)
            params: Query parameters (overrides config)
            body: Request body for POST/PUT (overrides config)
            
        Returns:
            List of dictionaries containing scraped data
        """
        # Use parameters or fall back to instance attributes
        base_url = base_url or self.base_url
        endpoint = endpoint or self.endpoint
        method = method or self.method
        params = params or self.params
        body = body or self.body
        
        # Build the URL
        url = urllib.parse.urljoin(base_url, endpoint) if base_url and endpoint else base_url or endpoint
        
        if not url:
            logger.error("No URL provided")
            return []
        
        # Make the initial request
        initial_response = self.make_request(url, method, params, body)
        if not initial_response:
            return []
        
        # Handle pagination if enabled
        if self.pagination and self.pagination.get('enabled', False):
            return self.handle_pagination(initial_response)
        else:
            # Extract data from the response
            data_path = self.config.get('data_path')
            return self.extract_data(initial_response, data_path)


def main():
    """
    Main function to run the scraper from command line.
    """
    parser = argparse.ArgumentParser(description='API scraper')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                        help='Output format (json, csv)')
    parser.add_argument('--no-metadata', action='store_true', help='Do not add metadata to scraped data')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = APIScraper(args.config)
    
    # Run scraper
    scraper.run(
        output_path=args.output,
        format=args.format,
        add_metadata=not args.no_metadata
    )


if __name__ == "__main__":
    main()
