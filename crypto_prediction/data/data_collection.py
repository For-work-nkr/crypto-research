"""
Data collection module for cryptocurrency price prediction system.
This module handles fetching data from various cryptocurrency APIs.
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class BaseDataCollector:
    """Base class for cryptocurrency data collectors."""
    
    def __init__(self, config: Dict):
        """Initialize the data collector with configuration."""
        self.config = config
        self.base_url = None
        self.api_key = None
        self.rate_limit = None
        self.last_request_time = 0
        
    def _respect_rate_limit(self):
        """Ensure rate limits are respected by adding delay if needed."""
        if self.rate_limit:
            min_interval = 60 / self.rate_limit  # seconds per request
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make an API request with rate limiting."""
        self._respect_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['X-API-Key'] = self.api_key
            
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
            
    def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get historical price data for a cryptocurrency."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a cryptocurrency."""
        raise NotImplementedError("Subclasses must implement this method")


class CoinGeckoCollector(BaseDataCollector):
    """Data collector for CoinGecko API."""
    
    def __init__(self, config: Dict):
        """Initialize the CoinGecko data collector."""
        super().__init__(config)
        self.base_url = config.get('data_sources', {}).get('coingecko', {}).get('base_url', 'https://api.coingecko.com/api/v3')
        self.api_key = config.get('data_sources', {}).get('coingecko', {}).get('api_key')
        self.rate_limit = config.get('data_sources', {}).get('coingecko', {}).get('rate_limit', 50)
        
        # Map of cryptocurrency symbols to CoinGecko IDs
        self.symbol_to_id = {}
        self._initialize_coin_list()
        
    def _initialize_coin_list(self):
        """Initialize the list of coins and their IDs."""
        coin_list = self._make_request('coins/list')
        if coin_list:
            for coin in coin_list:
                self.symbol_to_id[coin['symbol'].upper()] = coin['id']
                
    def _get_coin_id(self, symbol: str) -> str:
        """Get CoinGecko coin ID from symbol."""
        if not self.symbol_to_id:
            self._initialize_coin_list()
            
        coin_id = self.symbol_to_id.get(symbol.upper())
        if not coin_id:
            logger.error(f"Symbol {symbol} not found in CoinGecko")
            return None
            
        return coin_id
        
    def get_historical_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Get historical price data from CoinGecko."""
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            return pd.DataFrame()
            
        # Convert timeframe to CoinGecko format
        interval = 'daily'
        if timeframe == '1h':
            interval = 'hourly'
            
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': interval
        }
        
        market_data = self._make_request(f'coins/{coin_id}/market_chart', params)
        
        if not market_data or 'prices' not in market_data:
            logger.error(f"Failed to get historical data for {symbol}")
            return pd.DataFrame()
            
        # Create DataFrame from price data
        prices = market_data['prices']
        volumes = market_data['total_volumes']
        market_caps = market_data['market_caps']
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add volume and market cap
        df['volume'] = [v[1] for v in volumes]
        df['market_cap'] = [m[1] for m in market_caps]
        
        # Rename columns to standard format
        df.rename(columns={'price': 'close'}, inplace=True)
        
        return df
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price from CoinGecko."""
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            return None
            
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd'
        }
        
        price_data = self._make_request('simple/price', params)
        
        if not price_data or coin_id not in price_data:
            logger.error(f"Failed to get current price for {symbol}")
            return None
            
        return price_data[coin_id]['usd']


class BinanceCollector(BaseDataCollector):
    """Data collector for Binance API."""
    
    def __init__(self, config: Dict):
        """Initialize the Binance data collector."""
        super().__init__(config)
        self.base_url = config.get('data_sources', {}).get('binance', {}).get('base_url', 'https://api.binance.com/api/v3')
        self.api_key = config.get('data_sources', {}).get('binance', {}).get('api_key')
        self.api_secret = config.get('data_sources', {}).get('binance', {}).get('api_secret')
        self.rate_limit = config.get('data_sources', {}).get('binance', {}).get('rate_limit', 1200)
        
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Binance API."""
        return f"{symbol}USDT"
        
    def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get historical price data from Binance."""
        formatted_symbol = self._format_symbol(symbol)
        
        # Convert timeframe to Binance format
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        
        interval = interval_map.get(timeframe, '1d')
        
        params = {
            'symbol': formatted_symbol,
            'interval': interval,
            'limit': limit
        }
        
        klines = self._make_request('klines', params)
        
        if not klines:
            logger.error(f"Failed to get historical data for {symbol}")
            return pd.DataFrame()
            
        # Create DataFrame from klines data
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                   
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        df.set_index('timestamp', inplace=True)
        
        # Select only needed columns
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Binance."""
        formatted_symbol = self._format_symbol(symbol)
        
        params = {
            'symbol': formatted_symbol
        }
        
        ticker = self._make_request('ticker/price', params)
        
        if not ticker or 'price' not in ticker:
            logger.error(f"Failed to get current price for {symbol}")
            return None
            
        return float(ticker['price'])


class CoinMarketCapCollector(BaseDataCollector):
    """Data collector for CoinMarketCap API."""
    
    def __init__(self, config: Dict):
        """Initialize the CoinMarketCap data collector."""
        super().__init__(config)
        self.base_url = config.get('data_sources', {}).get('coinmarketcap', {}).get('base_url', 'https://pro-api.coinmarketcap.com/v1')
        self.api_key = config.get('data_sources', {}).get('coinmarketcap', {}).get('api_key')
        self.rate_limit = config.get('data_sources', {}).get('coinmarketcap', {}).get('rate_limit', 333)
        
        # Map of cryptocurrency symbols to CoinMarketCap IDs
        self.symbol_to_id = {}
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make an API request with rate limiting."""
        self._respect_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'X-CMC_PRO_API_KEY': self.api_key
        }
            
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
            
    def _get_coin_id(self, symbol: str) -> int:
        """Get CoinMarketCap coin ID from symbol."""
        if not self.symbol_to_id:
            self._initialize_coin_list()
            
        coin_id = self.symbol_to_id.get(symbol.upper())
        if not coin_id:
            logger.error(f"Symbol {symbol} not found in CoinMarketCap")
            return None
            
        return coin_id
        
    def _initialize_coin_list(self):
        """Initialize the list of coins and their IDs."""
        params = {
            'limit': 5000,
            'sort': 'cmc_rank'
        }
        
        coin_list = self._make_request('cryptocurrency/map', params)
        
        if coin_list and 'data' in coin_list:
            for coin in coin_list['data']:
                self.symbol_to_id[coin['symbol'].upper()] = coin['id']
                
    def get_historical_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Get historical price data from CoinMarketCap."""
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            return pd.DataFrame()
            
        # Calculate time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Convert to ISO format
        start_str = start_date.strftime('%Y-%m-%dT00:00:00.000Z')
        end_str = end_date.strftime('%Y-%m-%dT00:00:00.000Z')
        
        # Convert timeframe to CoinMarketCap format
        interval_map = {
            '1d': 'daily',
            '1h': 'hourly',
            '5m': '5m',
            '1m': '1m'
        }
        
        interval = interval_map.get(timeframe, 'daily')
        
        params = {
            'id': coin_id,
            'time_start': start_str,
            'time_end': end_str,
            'interval': interval,
            'convert': 'USD'
        }
        
        quotes = self._make_request('cryptocurrency/quotes/historical', params)
        
        if not quotes or 'data' not in quotes:
            logger.error(f"Failed to get historical data for {symbol}")
            return pd.DataFrame()
            
        # Create DataFrame from quotes data
        data = []
        for quote in quotes['data']['quotes']:
            data.append({
                'timestamp': quote['timestamp'],
                'open': quote['quote']['USD']['open'],
                'high': quote['quote']['USD']['high'],
                'low': quote['quote']['USD']['low'],
                'close': quote['quote']['USD']['close'],
                'volume': quote['quote']['USD']['volume'],
                'market_cap': quote['quote']['USD']['market_cap']
            })
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price from CoinMarketCap."""
        params = {
            'symbol': symbol,
            'convert': 'USD'
        }
        
        quotes = self._make_request('cryptocurrency/quotes/latest', params)
        
        if not quotes or 'data' not in quotes or symbol not in quotes['data']:
            logger.error(f"Failed to get current price for {symbol}")
            return None
            
        return quotes['data'][symbol]['quote']['USD']['price']


class DataCollector:
    """Main data collector that orchestrates multiple data sources."""
    
    def __init__(self, config: Dict):
        """Initialize the data collector with configuration."""
        self.config = config
        self.collectors = {}
        
        # Initialize data collectors
        if 'coingecko' in config.get('data_sources', {}):
            self.collectors['coingecko'] = CoinGeckoCollector(config)
            
        if 'binance' in config.get('data_sources', {}):
            self.collectors['binance'] = BinanceCollector(config)
            
        if 'coinmarketcap' in config.get('data_sources', {}):
            self.collectors['coinmarketcap'] = CoinMarketCapCollector(config)
            
        # Set primary data source
        self.primary_source = config.get('data_collection', {}).get('primary_source', 'coingecko')
        
        # Ensure primary source is available
        if self.primary_source not in self.collectors:
            available_sources = list(self.collectors.keys())
            if available_sources:
                self.primary_source = available_sources[0]
                logger.warning(f"Primary source {self.primary_source} not available. Using {available_sources[0]} instead.")
            else:
                logger.error("No data collectors available.")
                
    def get_historical_data(self, symbol: str, timeframe: str = '1d', days: int = 365, source: str = None) -> pd.DataFrame:
        """Get historical price data for a cryptocurrency."""
        # Use specified source or primary source
        source = source or self.primary_source
        
        if source not in self.collectors:
            logger.error(f"Data source {source} not available")
            return pd.DataFrame()
            
        return self.collectors[source].get_historical_data(symbol, timeframe, days)
        
    def get_current_price(self, symbol: str, source: str = None) -> float:
        """Get current price for a cryptocurrency."""
        # Use specified source or primary source
        source = source or self.primary_source
        
        if source not in self.collectors:
            logger.error(f"Data source {source} not available")
            return None
            
        return self.collectors[source].get_current_price(symbol)
        
    def collect_data_for_all_cryptocurrencies(self, timeframe: str = '1d', days: int = 365) -> Dict[str, pd.DataFrame]:
        """Collect historical data for all configured cryptocurrencies."""
        results = {}
        
        cryptocurrencies = self.config.get('cryptocurrencies', [])
        for crypto in cryptocurrencies:
            symbol = crypto['symbol']
            logger.info(f"Collecting data for {symbol}")
            
            df = self.get_historical_data(symbol, timeframe, days)
            
            if not df.empty:
                results[symbol] = df
                logger.info(f"Collected {len(df)} records for {symbol}")
            else:
                logger.warning(f"No data collected for {symbol}")
                
        return results
        
    def save_data(self, data: Dict[str, pd.DataFrame], directory: str = 'data/raw'):
        """Save collected data to CSV files."""
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        for symbol, df in data.items():
            filename = os.path.join(directory, f"{symbol.lower()}_data.csv")
            df.to_csv(filename)
            logger.info(f"Saved data for {symbol} to {filename}")


# Main function to run data collection
def collect_data(config_path: str = "config.yaml"):
    """Run the data collection process."""
    # Import here to avoid circular imports
    from crypto_prediction import load_config, setup_logging
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting data collection process")
    
    # Initialize data collector
    collector = DataCollector(config)
    
    # Collect data based on configuration
    historical_config = config.get('data_collection', {}).get('historical', {})
    timeframe = historical_config.get('timeframe', '1d')
    lookback_period = historical_config.get('lookback_period', 730)
    
    # Collect and save data
    data = collector.collect_data_for_all_cryptocurrencies(timeframe, lookback_period)
    collector.save_data(data)
    
    logger.info("Data collection process completed")
    
    return data


if __name__ == "__main__":
    # If run directly, execute data collection
    collect_data()
