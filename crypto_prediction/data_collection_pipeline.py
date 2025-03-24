"""
Main script to run the data collection pipeline for cryptocurrency price prediction system.
This script orchestrates the data collection process from multiple sources.
"""

import os
import logging
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from crypto_prediction import load_config, setup_logging
from crypto_prediction.data.data_collection import DataCollector

def setup_data_directories():
    """Create necessary data directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'data/historical',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")

def collect_historical_data(config, collector, days=None):
    """Collect historical data for all configured cryptocurrencies."""
    # Get configuration
    historical_config = config.get('data_collection', {}).get('historical', {})
    timeframe = historical_config.get('timeframe', '1d')
    lookback_period = days or historical_config.get('lookback_period', 730)
    
    logging.info(f"Collecting historical data for past {lookback_period} days with timeframe {timeframe}")
    
    # Collect data
    data = collector.collect_data_for_all_cryptocurrencies(timeframe, lookback_period)
    
    # Save data
    collector.save_data(data, 'data/raw')
    
    # Also save a timestamped copy for historical reference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    historical_dir = f'data/historical/{timestamp}'
    collector.save_data(data, historical_dir)
    
    return data

def collect_recent_data(config, collector):
    """Collect recent data for all configured cryptocurrencies."""
    # Get configuration
    recent_config = config.get('data_collection', {}).get('recent', {})
    timeframe = recent_config.get('timeframe', '1h')
    lookback_period = recent_config.get('lookback_period', 30)
    
    logging.info(f"Collecting recent data for past {lookback_period} days with timeframe {timeframe}")
    
    # Collect data
    data = collector.collect_data_for_all_cryptocurrencies(timeframe, lookback_period)
    
    # Save data
    collector.save_data(data, 'data/raw/recent')
    
    return data

def collect_realtime_data(config, collector):
    """Collect real-time data for all configured cryptocurrencies."""
    # Get configuration
    realtime_config = config.get('data_collection', {}).get('realtime', {})
    timeframe = realtime_config.get('timeframe', '1m')
    
    logging.info(f"Collecting real-time data with timeframe {timeframe}")
    
    # Get list of cryptocurrencies
    cryptocurrencies = config.get('cryptocurrencies', [])
    
    # Collect current prices
    results = {}
    for crypto in cryptocurrencies:
        symbol = crypto['symbol']
        price = collector.get_current_price(symbol)
        
        if price is not None:
            logging.info(f"Current price for {symbol}: {price}")
            
            # Create a simple DataFrame with current price
            df = pd.DataFrame({
                'timestamp': [datetime.now()],
                'price': [price]
            })
            df.set_index('timestamp', inplace=True)
            
            results[symbol] = df
        else:
            logging.warning(f"Failed to get current price for {symbol}")
    
    # Save data
    if results:
        realtime_dir = 'data/raw/realtime'
        os.makedirs(realtime_dir, exist_ok=True)
        
        for symbol, df in results.items():
            filename = os.path.join(realtime_dir, f"{symbol.lower()}_realtime.csv")
            
            # Append to existing file if it exists
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename, index_col=0, parse_dates=True)
                df = pd.concat([existing_df, df])
                
                # Keep only the last 1440 entries (24 hours of minute data)
                if len(df) > 1440:
                    df = df.iloc[-1440:]
            
            df.to_csv(filename)
            logging.info(f"Saved real-time data for {symbol} to {filename}")
    
    return results

def run_data_collection_pipeline(config_path=None, mode='all', days=None):
    """Run the complete data collection pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting data collection pipeline")
    
    # Setup directories
    setup_data_directories()
    
    # Initialize data collector
    collector = DataCollector(config)
    
    # Collect data based on mode
    if mode in ['all', 'historical']:
        collect_historical_data(config, collector, days)
    
    if mode in ['all', 'recent']:
        collect_recent_data(config, collector)
    
    if mode in ['all', 'realtime']:
        collect_realtime_data(config, collector)
    
    logger.info("Data collection pipeline completed successfully")

def run_scheduled_collection(config_path=None):
    """Run continuous scheduled data collection."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting scheduled data collection")
    
    # Setup directories
    setup_data_directories()
    
    # Initialize data collector
    collector = DataCollector(config)
    
    # Get update frequencies
    historical_freq = config.get('data_collection', {}).get('historical', {}).get('update_frequency', 86400)  # Daily
    recent_freq = config.get('data_collection', {}).get('recent', {}).get('update_frequency', 3600)  # Hourly
    realtime_freq = config.get('data_collection', {}).get('realtime', {}).get('update_frequency', 60)  # Every minute
    
    # Track last update times
    last_historical = 0
    last_recent = 0
    last_realtime = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to update historical data
            if current_time - last_historical >= historical_freq:
                logger.info("Running scheduled historical data collection")
                collect_historical_data(config, collector)
                last_historical = current_time
            
            # Check if it's time to update recent data
            if current_time - last_recent >= recent_freq:
                logger.info("Running scheduled recent data collection")
                collect_recent_data(config, collector)
                last_recent = current_time
            
            # Check if it's time to update real-time data
            if current_time - last_realtime >= realtime_freq:
                logger.info("Running scheduled real-time data collection")
                collect_realtime_data(config, collector)
                last_realtime = current_time
            
            # Sleep for a short time to avoid high CPU usage
            time.sleep(min(realtime_freq, 10))  # Sleep for at most 10 seconds
            
    except KeyboardInterrupt:
        logger.info("Scheduled data collection stopped by user")
    except Exception as e:
        logger.error(f"Error in scheduled data collection: {e}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cryptocurrency data collection pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['all', 'historical', 'recent', 'realtime', 'scheduled'], 
                        default='all', help='Collection mode')
    parser.add_argument('--days', type=int, help='Number of days of historical data to collect')
    
    args = parser.parse_args()
    
    if args.mode == 'scheduled':
        run_scheduled_collection(args.config)
    else:
        run_data_collection_pipeline(args.config, args.mode, args.days)
