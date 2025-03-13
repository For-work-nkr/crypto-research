"""
Main script to run the data preprocessing pipeline for cryptocurrency price prediction system.
This script orchestrates the data preprocessing process for model training.
"""

import os
import logging
import argparse
import pandas as pd
from pathlib import Path

# Import project modules
from crypto_prediction import load_config, setup_logging
from crypto_prediction.preprocessing.data_preprocessing import DataPreprocessor

def preprocess_all_cryptocurrencies(config):
    """Preprocess data for all configured cryptocurrencies."""
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process all cryptocurrencies
    results = preprocessor.process_all_cryptocurrencies()
    
    logging.info(f"Preprocessed data for {len(results)} cryptocurrencies")
    
    return results

def preprocess_specific_cryptocurrency(config, symbol):
    """Preprocess data for a specific cryptocurrency."""
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Get sequence length from model configuration
    sequence_length = config.get('models', {}).get('bilstm', {}).get('sequence_length', 60)
    
    # Process data
    logging.info(f"Preprocessing data for {symbol}")
    data_splits = preprocessor.process_data(symbol, sequence_length)
    
    if data_splits[0] is not None:
        logging.info(f"Successfully preprocessed data for {symbol}")
        return {symbol: data_splits}
    else:
        logging.warning(f"Failed to preprocess data for {symbol}")
        return {}

def run_preprocessing_pipeline(config_path=None, symbol=None):
    """Run the data preprocessing pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting data preprocessing pipeline")
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Preprocess data
    if symbol:
        results = preprocess_specific_cryptocurrency(config, symbol)
    else:
        results = preprocess_all_cryptocurrencies(config)
    
    # Log results
    if results:
        for symbol, data_splits in results.items():
            X_train, y_train, X_val, y_val, X_test, y_test = data_splits
            logger.info(f"Preprocessed data for {symbol}:")
            logger.info(f"  Training set: {X_train.shape[0]} samples")
            logger.info(f"  Validation set: {X_val.shape[0]} samples")
            logger.info(f"  Test set: {X_test.shape[0]} samples")
    else:
        logger.warning("No data was successfully preprocessed")
    
    logger.info("Data preprocessing pipeline completed")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cryptocurrency data preprocessing pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Specific cryptocurrency symbol to preprocess')
    
    args = parser.parse_args()
    
    run_preprocessing_pipeline(args.config, args.symbol)
