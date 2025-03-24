"""
Test script for cryptocurrency price prediction system.
This script tests all components of the system to ensure they work correctly.
"""

import os
import sys
import logging
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import time

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from crypto_prediction import load_config, setup_logging
from data.data_collection import DataCollector
from preprocessing.data_preprocessing import DataPreprocessor
from models.model_training import (
    BiLSTMModel, ProphetModel, XGBoostModel, ARIMAModel, EnsembleModel
)

# Setup logging
config = load_config()
logger = setup_logging(config)

class TestDataCollection(unittest.TestCase):
    """Test cases for data collection module."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = load_config()
        self.collector = DataCollector(self.config)
        
    def test_api_connection(self):
        """Test connection to cryptocurrency APIs."""
        # Test CoinGecko API
        result = self.collector.test_api_connection('coingecko')
        self.assertTrue(result, "Failed to connect to CoinGecko API")
        
        # Test Binance API if configured
        if 'binance' in self.collector.api_keys:
            result = self.collector.test_api_connection('binance')
            self.assertTrue(result, "Failed to connect to Binance API")
        
        # Test CoinMarketCap API if configured
        if 'coinmarketcap' in self.collector.api_keys:
            result = self.collector.test_api_connection('coinmarketcap')
            self.assertTrue(result, "Failed to connect to CoinMarketCap API")
    
    def test_get_historical_data(self):
        """Test retrieving historical data."""
        # Get first cryptocurrency from config
        crypto = self.config.get('cryptocurrencies', [{'symbol': 'BTC'}])[0]['symbol']
        
        # Get historical data
        df = self.collector.get_historical_data(crypto, timeframe='1d', days=30)
        
        # Check if data is not empty
        self.assertFalse(df.empty, f"Failed to retrieve historical data for {crypto}")
        
        # Check if data has expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing column {col} in historical data")
        
        # Check if data has expected length
        self.assertGreaterEqual(len(df), 20, f"Not enough historical data points for {crypto}")
    
    def test_get_current_price(self):
        """Test retrieving current price."""
        # Get first cryptocurrency from config
        crypto = self.config.get('cryptocurrencies', [{'symbol': 'BTC'}])[0]['symbol']
        
        # Get current price
        price = self.collector.get_current_price(crypto)
        
        # Check if price is valid
        self.assertIsNotNone(price, f"Failed to retrieve current price for {crypto}")
        self.assertGreater(price, 0, f"Invalid price value for {crypto}")
    
    def test_save_and_load_data(self):
        """Test saving and loading data."""
        # Get first cryptocurrency from config
        crypto = self.config.get('cryptocurrencies', [{'symbol': 'BTC'}])[0]['symbol']
        
        # Get historical data
        df = self.collector.get_historical_data(crypto, timeframe='1d', days=10)
        
        # Create test directory
        test_dir = 'data/test'
        os.makedirs(test_dir, exist_ok=True)
        
        # Save data
        data = {crypto: df}
        self.collector.save_data(data, test_dir)
        
        # Check if file exists
        filename = os.path.join(test_dir, f"{crypto.lower()}_1d.csv")
        self.assertTrue(os.path.exists(filename), f"Data file {filename} not created")
        
        # Load data
        loaded_df = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        # Check if loaded data matches original
        self.assertEqual(len(df), len(loaded_df), "Loaded data length doesn't match original")
        
        # Clean up
        os.remove(filename)


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing module."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = load_config()
        self.collector = DataCollector(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        
        # Get first cryptocurrency from config
        self.crypto = self.config.get('cryptocurrencies', [{'symbol': 'BTC'}])[0]['symbol']
        
        # Get historical data
        self.df = self.collector.get_historical_data(self.crypto, timeframe='1d', days=60)
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Clean data
        cleaned_df = self.preprocessor.clean_data(self.df)
        
        # Check if data is not empty
        self.assertFalse(cleaned_df.empty, "Cleaned data is empty")
        
        # Check if data has expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, cleaned_df.columns, f"Missing column {col} in cleaned data")
        
        # Check if there are no NaN values
        self.assertFalse(cleaned_df.isnull().any().any(), "Cleaned data contains NaN values")
    
    def test_add_technical_indicators(self):
        """Test adding technical indicators."""
        # Clean data
        cleaned_df = self.preprocessor.clean_data(self.df)
        
        # Add technical indicators
        indicators_df = self.preprocessor.add_technical_indicators(cleaned_df)
        
        # Check if data is not empty
        self.assertFalse(indicators_df.empty, "Data with indicators is empty")
        
        # Check if data has expected indicators
        expected_indicators = ['sma_7', 'sma_14', 'rsi_14', 'macd', 'macd_signal']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators_df.columns, f"Missing indicator {indicator}")
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Clean data and add features
        cleaned_df = self.preprocessor.clean_data(self.df)
        features_df = self.preprocessor.add_technical_indicators(cleaned_df)
        features_df = self.preprocessor.add_temporal_features(features_df)
        
        # Normalize features
        normalized_df = self.preprocessor.normalize_features(features_df)
        
        # Check if data is not empty
        self.assertFalse(normalized_df.empty, "Normalized data is empty")
        
        # Check if normalized values are in range [0, 1]
        for col in normalized_df.columns:
            if col not in ['day_of_week', 'month', 'is_weekend']:  # Skip categorical features
                self.assertGreaterEqual(normalized_df[col].min(), 0, f"Minimum value for {col} is less than 0")
                self.assertLessEqual(normalized_df[col].max(), 1, f"Maximum value for {col} is greater than 1")
    
    def test_prepare_sequences(self):
        """Test sequence preparation."""
        # Clean data and add features
        cleaned_df = self.preprocessor.clean_data(self.df)
        features_df = self.preprocessor.add_technical_indicators(cleaned_df)
        features_df = self.preprocessor.add_temporal_features(features_df)
        normalized_df = self.preprocessor.normalize_features(features_df)
        
        # Prepare sequences
        sequence_length = 10
        X, y = self.preprocessor.prepare_sequences(normalized_df, sequence_length)
        
        # Check if sequences are not empty
        self.assertIsNotNone(X, "X sequences are None")
        self.assertIsNotNone(y, "y sequences are None")
        
        # Check if sequences have expected shape
        self.assertEqual(X.shape[1], sequence_length, f"X sequence length is not {sequence_length}")
        self.assertEqual(X.shape[2], len(normalized_df.columns), "X feature count doesn't match")
        self.assertEqual(len(y), len(X), "y length doesn't match X length")


class TestModelTraining(unittest.TestCase):
    """Test cases for model training module."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = load_config()
        self.collector = DataCollector(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        
        # Get first cryptocurrency from config
        self.crypto = self.config.get('cryptocurrencies', [{'symbol': 'BTC'}])[0]['symbol']
        
        # Get historical data
        self.df = self.collector.get_historical_data(self.crypto, timeframe='1d', days=60)
        
        # Preprocess data
        cleaned_df = self.preprocessor.clean_data(self.df)
        features_df = self.preprocessor.add_technical_indicators(cleaned_df)
        features_df = self.preprocessor.add_temporal_features(features_df)
        normalized_df = self.preprocessor.normalize_features(features_df)
        
        # Prepare sequences
        sequence_length = 10
        X, y = self.preprocessor.prepare_sequences(normalized_df, sequence_length)
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        
        self.X_val = X[train_size:train_size+val_size]
        self.y_val = y[train_size:train_size+val_size]
        
        self.X_test = X[train_size+val_size:]
        self.y_test = y[train_size+val_size:]
    
    def test_bilstm_model(self):
        """Test Bi-LSTM model."""
        # Initialize model
        model = BiLSTMModel(self.config, self.crypto)
        
        # Train model with small number of epochs for testing
        model.epochs = 2
        history = model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Check if model is trained
        self.assertIsNotNone(model.model, "Bi-LSTM model is None after training")
        
        # Make predictions
        predictions = model.predict(self.X_test)
        
        # Check if predictions have expected length
        self.assertEqual(len(predictions), len(self.y_test), "Predictions length doesn't match test data length")
        
        # Evaluate model
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # Check if metrics are calculated
        self.assertIn('rmse', metrics, "RMSE metric not calculated")
        self.assertIn('mape', metrics, "MAPE metric not calculated")
        
        # Save and load model
        test_dir = 'models/test'
        os.makedirs(test_dir, exist_ok=True)
        
        model.save(test_dir)
        
        # Check if model file exists
        model_path = os.path.join(test_dir, f'bilstm_{self.crypto.lower()}.h5')
        self.assertTrue(os.path.exists(model_path), f"Model file {model_path} not created")
        
        # Load model
        new_model = BiLSTMModel(self.config, self.crypto)
        result = new_model.load(test_dir)
        
        # Check if model is loaded
        self.assertTrue(result, "Failed to load Bi-LSTM model")
        self.assertIsNotNone(new_model.model, "Loaded Bi-LSTM model is None")
        
        # Clean up
        os.remove(model_path)
        config_path = os.path.join(test_dir, f'bilstm_{self.crypto.lower()}_config.json')
        if os.path.exists(config_path):
            os.remove(config_path)
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        # Initialize model
        model = XGBoostModel(self.config, self.crypto)
        
        # Train model with small number of estimators for testing
        model.n_estimators = 10
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Check if model is trained
        self.assertIsNotNone(model.model, "XGBoost model is None after training")
        
        # Make predictions
        predictions = model.predict(self.X_test)
        
        # Check if predictions have expected length
        self.assertEqual(len(predictions), len(self.y_test), "Predictions length doesn't match test data length")
        
        # Evaluate model
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # Check if metrics are calculated
        self.assertIn('rmse', metrics, "RMSE metric not calculated")
        self.assertIn('mape', metrics, "MAPE metric not calculated")
        
        # Save and load model
        test_dir = 'models/test'
        os.makedirs(test_dir, exist_ok=True)
        
        model.save(test_dir)
        
        # Check if model file exists
        model_path = os.path.join(test_dir, f'xgboost_{self.crypto.lower()}.json')
        self.assertTrue(os.path.exists(model_path), f"Model file {model_path} not created")
        
        # Load model
        new_model = XGBoostModel(self.config, self.crypto)
        result = new_model.load(test_dir)
        
        # Check if model is loaded
        self.assertTrue(result, "Failed to load XGBoost model")
        self.assertIsNotNone(new_model.model, "Loaded XGBoost model is None")
        
        # Clean up
        os.remove(model_path)
        config_path = os.path.join(test_dir, f'xgboost_{self.crypto.lower()}_config.json')
        if os.path.exists(config_path):
            os.remove(config_path)


class TestPredictionAPI(unittest.TestCase):
    """Test cases for prediction API."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = load_config()
        
        # Get API configuration
        api_config = self.config.get('api', {})
        self.api_host = api_config.get('host', '0.0.0.0')
        self.api_port = api_config.get('port', 8000)
        self.api_url = f"http://{self.api_host}:{self.api_port}"
        
        # Check if API is running
        try:
            response = requests.get(f"{self.api_url}/health", timeout=1)
            self.api_running = response.status_code == 200
        except:
            self.api_running = False
    
    def test_api_health(self):
        """Test API health endpoint."""
        if not self.api_running:
            self.skipTest("API is not running")
        
        response = requests.get(f"{self.api_url}/health")
        
        # Check if response is successful
        self.assertEqual(response.status_code, 200, "API health check failed")
        
        # Check if response has expected fields
        data = response.json()
        self.assertIn('status', data, "API health response missing 'status' field")
        self.assertEqual(data['status'], 'healthy', "API status is not 'healthy'")
    
    def test_list_cryptocurrencies(self):
        """Test listing cryptocurrencies endpoint."""
        if not self.api_running:
            self.skipTest("API is not running")
        
        response = requests.get(f"{self.api_url}/cryptocurrencies")
        
        # Check if response is successful
        self.assertEqual(response.status_code, 200, "API cryptocurrencies endpoint failed")
        
        # Check if response has expected format
        data = response.json()
        self.assertIsInstance(data, list, "API cryptocurrencies response is not a list")
        
        if data:
            crypto = data[0]
            self.assertIn('symbol', crypto, "Cryptocurrency missing 'symbol' field")
            self.assertIn('name', crypto, "Cryptocurrency missing 'name' field")
    
    def test_list_models(self):
        """Test listing models endpoint."""
        if not self.api_running:
            self.skipTest("API is not running")
        
        response = requests.get(f"{self.api_url}/models")
        
        # Check if response is successful
        self.assertEqual(response.status_code, 200, "API models endpoint failed")
        
        # Check if response has expected format
        data = response.json()
        self.assertIsInstance(data, list, "API models response is not a list")
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint."""
        if not self.api_running:
            self.skipTest("API is not running")
        
        # Get first cryptocurrency from config
        crypto = self.config.get('cryptocurrencies', [{'symbol': 'BTC'}])[0]['symbol']
        
        # Make prediction request
        response = requests.get(
            f"{self.api_url}/predict/{crypto}",
            params={"horizon": 1, "model_type": "ensemble"}
        )
        
        # If models are not trained, this might fail
        if response.status_code == 404:
            self.skipTest("Models not trained for prediction")
        
        # Check if response is successful
        self.assertEqual(response.status_code, 200, "API prediction endpoint failed")
        
        # Check if response has expected fields
        data = response.json()
        self.assertIn('symbol', data, "Prediction missing 'symbol' field")
        self.assertIn('predictions', data, "Prediction missing 'predictions' field")
        self.assertIn('timestamps', data, "Prediction missing 'timestamps' field")
        self.assertIn('model_type', data, "Prediction missing 'model_type' field")
        
        # Check if predictions have expected length
        self.assertEqual(len(data['predictions']), 1, "Prediction length doesn't match horizon")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataCollection))
    test_suite.addTest(unittest.makeSuite(TestDataPreprocessing))
    test_suite.addTest(unittest.makeSuite(TestModelTraining))
    test_suite.addTest(unittest.makeSuite(TestPredictionAPI))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    # Run tests
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())
