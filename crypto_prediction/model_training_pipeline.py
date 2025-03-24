"""
Main script to run the model training pipeline for cryptocurrency price prediction system.
This script orchestrates the model training process.
"""

import os
import logging
import argparse
import json
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from crypto_prediction import load_config, setup_logging
from crypto_prediction.models.model_training import (
    BiLSTMModel, ProphetModel, XGBoostModel, ARIMAModel, EnsembleModel, ModelTrainer
)
from crypto_prediction.preprocessing.data_preprocessing import DataPreprocessor

def train_models_for_all_cryptocurrencies(config):
    """Train models for all preprocessed cryptocurrencies."""
    # Initialize data preprocessor to get preprocessed data
    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.process_all_cryptocurrencies()
    
    if not data_dict:
        logging.error("No preprocessed data available for training")
        return None
    
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Train models
    results = trainer.train_models(data_dict)
    
    logging.info(f"Model training completed for {len(results)} cryptocurrencies")
    
    # Save results
    save_training_results(results)
    
    return results

def train_models_for_specific_cryptocurrency(config, symbol):
    """Train models for a specific cryptocurrency."""
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Get sequence length from model configuration
    sequence_length = config.get('models', {}).get('bilstm', {}).get('sequence_length', 60)
    
    # Process data
    logging.info(f"Preprocessing data for {symbol}")
    data_splits = preprocessor.process_data(symbol, sequence_length)
    
    if data_splits[0] is None:
        logging.error(f"Failed to preprocess data for {symbol}")
        return None
    
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Train models
    data_dict = {symbol: data_splits}
    results = trainer.train_models(data_dict)
    
    logging.info(f"Model training completed for {symbol}")
    
    # Save results
    save_training_results(results)
    
    return results

def save_training_results(results):
    """Save training results to file."""
    if not results:
        return
    
    # Create directory if it doesn't exist
    os.makedirs('models/metrics', exist_ok=True)
    
    # Save results for each cryptocurrency
    for symbol, metrics in results.items():
        filename = f'models/metrics/{symbol.lower()}_metrics.json'
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logging.info(f"Saved training metrics for {symbol} to {filename}")

def run_model_training_pipeline(config_path=None, symbol=None, model_type=None):
    """Run the model training pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting model training pipeline")
    
    # Create necessary directories
    os.makedirs('models/saved', exist_ok=True)
    
    # Train models
    if symbol:
        results = train_models_for_specific_cryptocurrency(config, symbol)
    else:
        results = train_models_for_all_cryptocurrencies(config)
    
    logger.info("Model training pipeline completed")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cryptocurrency model training pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Specific cryptocurrency symbol to train models for')
    parser.add_argument('--model', type=str, choices=['bilstm', 'prophet', 'xgboost', 'arima', 'ensemble'],
                        help='Specific model type to train')
    
    args = parser.parse_args()
    
    run_model_training_pipeline(args.config, args.symbol, args.model)
