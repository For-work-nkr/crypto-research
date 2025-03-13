"""
Data preprocessing module for cryptocurrency price prediction system.
This module handles data cleaning, feature engineering, and transformation.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import ta  # Technical analysis library

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing cryptocurrency data."""
    
    def __init__(self, config: Dict):
        """Initialize the data preprocessor with configuration."""
        self.config = config
        self.features_config = config.get('features', {})
        
    def load_data(self, symbol: str, directory: str = 'data/raw') -> pd.DataFrame:
        """Load raw data for a cryptocurrency."""
        filename = os.path.join(directory, f"{symbol.lower()}_data.csv")
        
        if not os.path.exists(filename):
            logger.error(f"Data file {filename} not found")
            return pd.DataFrame()
            
        df = pd.DataFrame()
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} records for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            
        return df
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle missing values
        logger.info(f"Missing values before cleaning: {df.isna().sum().sum()}")
        
        # Forward fill missing values (use previous day's value)
        df.fillna(method='ffill', inplace=True)
        
        # If still missing values at the beginning, backward fill
        df.fillna(method='bfill', inplace=True)
        
        logger.info(f"Missing values after cleaning: {df.isna().sum().sum()}")
        
        # Handle outliers using IQR method for price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Identify outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in {col}")
                    
                    # Replace outliers with median of nearby points
                    outlier_indices = df.index[outliers]
                    for idx in outlier_indices:
                        # Get 5 points before and after (if available)
                        window_start = max(0, df.index.get_loc(idx) - 5)
                        window_end = min(len(df), df.index.get_loc(idx) + 6)
                        window = df.iloc[window_start:window_end]
                        
                        # Replace with median of window (excluding the outlier itself)
                        window_without_outlier = window[window.index != idx]
                        df.loc[idx, col] = window_without_outlier[col].median()
        
        return df
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Get technical indicators configuration
        tech_indicators = self.features_config.get('technical_indicators', [])
        
        # Add technical indicators based on configuration
        for indicator in tech_indicators:
            indicator_name = indicator.get('name')
            parameters = indicator.get('parameters', [])
            
            if indicator_name == 'SMA':
                # Simple Moving Average
                for window in parameters:
                    df[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
                    
            elif indicator_name == 'EMA':
                # Exponential Moving Average
                for window in parameters:
                    df[f'EMA_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
                    
            elif indicator_name == 'RSI':
                # Relative Strength Index
                for window in parameters:
                    df[f'RSI_{window}'] = ta.momentum.rsi(df['close'], window=window)
                    
            elif indicator_name == 'MACD':
                # Moving Average Convergence Divergence
                if len(parameters) >= 3:
                    fast, slow, signal = parameters[0], parameters[1], parameters[2]
                    macd = ta.trend.MACD(df['close'], fast, slow, signal)
                    df['MACD'] = macd.macd()
                    df['MACD_signal'] = macd.macd_signal()
                    df['MACD_diff'] = macd.macd_diff()
                    
            elif indicator_name == 'Bollinger':
                # Bollinger Bands
                if len(parameters) >= 2:
                    window, std = parameters[0], parameters[1]
                    bollinger = ta.volatility.BollingerBands(df['close'], window=window, window_dev=std)
                    df['Bollinger_mavg'] = bollinger.bollinger_mavg()
                    df['Bollinger_hband'] = bollinger.bollinger_hband()
                    df['Bollinger_lband'] = bollinger.bollinger_lband()
                    df['Bollinger_width'] = bollinger.bollinger_wband()
                    
            elif indicator_name == 'ATR':
                # Average True Range
                for window in parameters:
                    df[f'ATR_{window}'] = ta.volatility.average_true_range(
                        df['high'], df['low'], df['close'], window=window
                    )
                    
            elif indicator_name == 'OBV':
                # On-Balance Volume
                if 'volume' in df.columns:
                    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        logger.info(f"Added {len(df.columns) - 5} technical indicators")  # 5 = open, high, low, close, volume
        
        return df
        
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataframe."""
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Get temporal features configuration
        temporal_features = self.features_config.get('temporal_features', [])
        
        # Add temporal features based on configuration
        for feature in temporal_features:
            if feature == 'day_of_week':
                df['day_of_week'] = df.index.dayofweek
                
            elif feature == 'hour_of_day':
                df['hour_of_day'] = df.index.hour
                
            elif feature == 'month':
                df['month'] = df.index.month
                
            elif feature == 'is_weekend':
                df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
                
            elif feature == 'quarter':
                df['quarter'] = df.index.quarter
                
            elif feature == 'day_of_month':
                df['day_of_month'] = df.index.day
                
            elif feature == 'week_of_year':
                df['week_of_year'] = df.index.isocalendar().week
                
            elif feature == 'year':
                df['year'] = df.index.year
                
        # Add cyclical encoding for temporal features
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
        if 'hour_of_day' in df.columns:
            df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            
        logger.info(f"Added temporal features")
        
        return df
        
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features to the dataframe."""
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_7d'] = df['close'].pct_change(7)
        df['return_30d'] = df['close'].pct_change(30)
        
        # Calculate volatility
        df['volatility_7d'] = df['return_1d'].rolling(window=7).std()
        df['volatility_30d'] = df['return_1d'].rolling(window=30).std()
        
        # Calculate price momentum
        df['momentum_7d'] = df['close'] / df['close'].shift(7) - 1
        df['momentum_30d'] = df['close'] / df['close'].shift(30) - 1
        
        # Calculate price acceleration
        df['acceleration_7d'] = df['momentum_7d'] - df['momentum_7d'].shift(7)
        
        # Calculate high-low range
        df['range'] = (df['high'] - df['low']) / df['close']
        df['range_7d_avg'] = df['range'].rolling(window=7).mean()
        
        logger.info(f"Added price-based features")
        
        return df
        
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using min-max scaling."""
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Identify columns to normalize (exclude temporal categorical features)
        exclude_cols = ['day_of_week', 'hour_of_day', 'month', 'is_weekend', 'quarter', 
                        'day_of_month', 'week_of_year', 'year']
        
        normalize_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Apply min-max normalization
        for col in normalize_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Avoid division by zero
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0  # If all values are the same
                
        logger.info(f"Normalized {len(normalize_cols)} features")
        
        return df
        
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int, target_column: str = 'close') -> tuple:
        """Prepare sequences for time series prediction."""
        if df.empty or len(df) <= sequence_length:
            logger.error(f"Not enough data to create sequences (need > {sequence_length} points)")
            return None, None
            
        # Drop rows with NaN values
        df = df.dropna()
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(df) - sequence_length):
            # Get sequence of features
            sequence = df.iloc[i:i+sequence_length].values
            X.append(sequence)
            
            # Get target value (next day's close price)
            target = df.iloc[i+sequence_length][target_column]
            y.append(target)
            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        
        return X, y
        
    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
        """Split data into training, validation, and test sets."""
        if X is None or y is None:
            return None, None, None, None, None, None
            
        # Calculate split indices
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split data
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Split data into train ({len(X_train)}), validation ({len(X_val)}), and test ({len(X_test)}) sets")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    def process_data(self, symbol: str, sequence_length: int = 60) -> tuple:
        """Process data for a cryptocurrency from raw data to model-ready sequences."""
        # Load raw data
        df = self.load_data(symbol)
        
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None, None, None, None, None, None
            
        # Clean data
        df = self.clean_data(df)
        
        # Add features
        df = self.add_technical_indicators(df)
        df = self.add_temporal_features(df)
        df = self.add_price_features(df)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        processed_file = os.path.join('data/processed', f"{symbol.lower()}_processed.csv")
        df.to_csv(processed_file)
        logger.info(f"Saved processed data to {processed_file}")
        
        # Normalize features
        df_norm = self.normalize_features(df)
        
        # Prepare sequences
        X, y = self.prepare_sequences(df_norm, sequence_length)
        
        # Split data
        return self.split_data(X, y)
        
    def process_all_cryptocurrencies(self) -> Dict[str, tuple]:
        """Process data for all configured cryptocurrencies."""
        results = {}
        
        # Get sequence length from model configuration
        sequence_length = self.config.get('models', {}).get('bilstm', {}).get('sequence_length', 60)
        
        cryptocurrencies = self.config.get('cryptocurrencies', [])
        for crypto in cryptocurrencies:
            symbol = crypto['symbol']
            logger.info(f"Processing data for {symbol}")
            
            data_splits = self.process_data(symbol, sequence_length)
            
            if data_splits[0] is not None:
                results[symbol] = data_splits
                logger.info(f"Successfully processed data for {symbol}")
            else:
                logger.warning(f"Failed to process data for {symbol}")
                
        return results


# Main function to run data preprocessing
def preprocess_data(config_path: str = "config.yaml"):
    """Run the data preprocessing process."""
    # Import here to avoid circular imports
    from crypto_prediction import load_config, setup_logging
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting data preprocessing process")
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process all cryptocurrencies
    results = preprocessor.process_all_cryptocurrencies()
    
    logger.info(f"Processed data for {len(results)} cryptocurrencies")
    
    return results


if __name__ == "__main__":
    # If run directly, execute data preprocessing
    preprocess_data()
