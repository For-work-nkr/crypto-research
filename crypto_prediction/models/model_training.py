"""
Model implementation module for cryptocurrency price prediction system.
This module implements various prediction models including Bi-LSTM, Prophet, XGBoost, and ARIMA.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
from typing import Dict, List, Optional, Union, Tuple
import json
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for cryptocurrency prediction models."""
    
    def __init__(self, config: Dict, symbol: str):
        """Initialize the model with configuration."""
        self.config = config
        self.symbol = symbol
        self.model = None
        self.model_path = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def predict(self, X):
        """Make predictions using the trained model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return {}
            
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Calculate directional accuracy
        direction_actual = np.diff(y_test, prepend=y_test[0]) > 0
        direction_pred = np.diff(y_pred, prepend=y_pred[0]) > 0
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"Model evaluation for {self.symbol}: RMSE={rmse:.4f}, MAPE={mape:.2f}%, Directional Accuracy={directional_accuracy:.2f}%")
        
        return metrics
        
    def save(self, directory: str = 'models/saved'):
        """Save the trained model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def load(self, directory: str = 'models/saved'):
        """Load a trained model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def plot_predictions(self, X_test, y_test, output_dir: str = 'visualization/results'):
        """Plot actual vs predicted values."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return
            
        y_pred = self.predict(X_test)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'{self.symbol} Price Prediction - {self.__class__.__name__}')
        plt.xlabel('Time')
        plt.ylabel('Price (normalized)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{self.symbol.lower()}_{self.__class__.__name__.lower()}_prediction.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {plot_path}")


class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model for cryptocurrency prediction."""
    
    def __init__(self, config: Dict, symbol: str):
        """Initialize the Bi-LSTM model."""
        super().__init__(config, symbol)
        
        # Get model configuration
        self.model_config = config.get('models', {}).get('bilstm', {})
        self.layers = self.model_config.get('layers', 2)
        self.units = self.model_config.get('units', 128)
        self.dropout = self.model_config.get('dropout', 0.2)
        self.batch_size = self.model_config.get('batch_size', 64)
        self.epochs = self.model_config.get('epochs', 100)
        self.early_stopping_patience = self.model_config.get('early_stopping_patience', 10)
        
        # Set model path
        self.model_path = f'models/saved/bilstm_{symbol.lower()}.h5'
        
    def build_model(self, input_shape):
        """Build the Bi-LSTM model architecture."""
        model = Sequential()
        
        # First Bi-LSTM layer
        model.add(Bidirectional(LSTM(self.units, return_sequences=(self.layers > 1)), 
                               input_shape=input_shape))
        model.add(Dropout(self.dropout))
        
        # Additional Bi-LSTM layers if specified
        for i in range(1, self.layers):
            return_sequences = i < self.layers - 1
            model.add(Bidirectional(LSTM(self.units, return_sequences=return_sequences)))
            model.add(Dropout(self.dropout))
            
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Bi-LSTM model."""
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Create callbacks
        callbacks = []
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint to save best model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Train model
        logger.info(f"Training Bi-LSTM model for {self.symbol}")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        self.model = load_model(self.model_path)
        
        # Return training history
        return history
        
    def predict(self, X):
        """Make predictions using the trained Bi-LSTM model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
            
        return self.model.predict(X).flatten()
        
    def save(self, directory: str = 'models/saved'):
        """Save the trained Bi-LSTM model."""
        if self.model is None:
            logger.error("No model to save")
            return
            
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f'bilstm_{self.symbol.lower()}.h5')
        self.model.save(model_path)
        logger.info(f"Bi-LSTM model saved to {model_path}")
        
        # Save model configuration
        config_path = os.path.join(directory, f'bilstm_{self.symbol.lower()}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f)
            
        self.model_path = model_path
        
    def load(self, directory: str = 'models/saved'):
        """Load a trained Bi-LSTM model."""
        model_path = os.path.join(directory, f'bilstm_{self.symbol.lower()}.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False
            
        try:
            self.model = load_model(model_path)
            logger.info(f"Bi-LSTM model loaded from {model_path}")
            self.model_path = model_path
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class ProphetModel(BaseModel):
    """Facebook Prophet model for cryptocurrency prediction."""
    
    def __init__(self, config: Dict, symbol: str):
        """Initialize the Prophet model."""
        super().__init__(config, symbol)
        
        # Get model configuration
        self.model_config = config.get('models', {}).get('prophet', {})
        self.changepoint_prior_scale = self.model_config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = self.model_config.get('seasonality_prior_scale', 10)
        self.seasonality_mode = self.model_config.get('seasonality_mode', 'multiplicative')
        
        # Set model path
        self.model_path = f'models/saved/prophet_{symbol.lower()}.pkl'
        
    def prepare_data(self, df):
        """Prepare data for Prophet model."""
        # Prophet requires 'ds' (date) and 'y' (target) columns
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['close']
        })
        
        return prophet_df
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Prophet model.
        
        Note: Prophet doesn't use the same data format as other models.
        Instead, we'll reconstruct a DataFrame from the sequence data.
        """
        # For Prophet, we need to reconstruct the time series
        # This is a simplified approach - in practice, you'd use the original time series
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(y_train), freq='D')
        train_df = pd.DataFrame({
            'ds': dates,
            'y': y_train
        })
        
        # Initialize and train Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        logger.info(f"Training Prophet model for {self.symbol}")
        self.model.fit(train_df)
        
        return self.model
        
    def predict(self, X):
        """Make predictions using the trained Prophet model.
        
        Note: Prophet doesn't use X directly. Instead, we create a future DataFrame
        with the appropriate number of periods.
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
            
        # Create future dataframe for prediction
        future = self.model.make_future_dataframe(periods=len(X))
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return only the predictions for the test period
        return forecast['yhat'].tail(len(X)).values
        
    def save(self, directory: str = 'models/saved'):
        """Save the trained Prophet model."""
        if self.model is None:
            logger.error("No model to save")
            return
            
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f'prophet_{self.symbol.lower()}.pkl')
        
        with open(model_path, 'wb') as f:
            joblib.dump(self.model, f)
            
        logger.info(f"Prophet model saved to {model_path}")
        self.model_path = model_path
        
        # Save model configuration
        config_path = os.path.join(directory, f'prophet_{self.symbol.lower()}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f)
            
    def load(self, directory: str = 'models/saved'):
        """Load a trained Prophet model."""
        model_path = os.path.join(directory, f'prophet_{self.symbol.lower()}.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False
            
        try:
            with open(model_path, 'rb') as f:
                self.model = joblib.load(f)
                
            logger.info(f"Prophet model loaded from {model_path}")
            self.model_path = model_path
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class XGBoostModel(BaseModel):
    """XGBoost model for cryptocurrency prediction."""
    
    def __init__(self, config: Dict, symbol: str):
        """Initialize the XGBoost model."""
        super().__init__(config, symbol)
        
        # Get model configuration
        self.model_config = config.get('models', {}).get('xgboost', {})
        self.max_depth = self.model_config.get('max_depth', 6)
        self.learning_rate = self.model_config.get('learning_rate', 0.1)
        self.n_estimators = self.model_config.get('n_estimators', 100)
        self.objective = self.model_config.get('objective', 'reg:squarederror')
        
        # Set model path
        self.model_path = f'models/saved/xgboost_{symbol.lower()}.json'
        
    def prepare_data(self, X):
        """Prepare data for XGBoost model."""
        # XGBoost requires 2D input, so reshape if necessary
        if len(X.shape) == 3:
            # Flatten the sequence dimension
            X_reshaped = X.reshape(X.shape[0], -1)
            return X_reshaped
        return X
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model."""
        # Prepare data
        X_train_2d = self.prepare_data(X_train)
        X_val_2d = self.prepare_data(X_val) if X_val is not None else None
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective=self.objective,
            n_jobs=-1
        )
        
        # Train model
        logger.info(f"Training XGBoost model for {self.symbol}")
        
        eval_set = [(X_train_2d, y_train)]
        if X_val_2d is not None and y_val is not None:
            eval_set.append((X_val_2d, y_val))
            
        self.model.fit(
            X_train_2d, y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        logger.info(f"XGBoost feature importance: {feature_importance}")
        
        return self.model
        
    def predict(self, X):
        """Make predictions using the trained XGBoost model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
            
        # Prepare data
        X_2d = self.prepare_data(X)
        
        # Make predictions
        return self.model.predict(X_2d)
        
    def save(self, directory: str = 'models/saved'):
        """Save the trained XGBoost model."""
        if self.model is None:
            logger.error("No model to save")
            return
            
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f'xgboost_{self.symbol.lower()}.json')
        
        self.model.save_model(model_path)
        logger.info(f"XGBoost model saved to {model_path}")
        self.model_path = model_path
        
        # Save model configuration
        config_path = os.path.join(directory, f'xgboost_{self.symbol.lower()}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f)
            
    def load(self, directory: str = 'models/saved'):
        """Load a trained XGBoost model."""
        model_path = os.path.join(directory, f'xgboost_{self.symbol.lower()}.json')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False
            
        try:
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            logger.info(f"XGBoost model loaded from {model_path}")
            self.model_path = model_path
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class ARIMAModel(BaseModel):
    """ARIMA model for cryptocurrency prediction."""
    
    def __init__(self, config: Dict, symbol: str):
        """Initialize the ARIMA model."""
        super().__init__(config, symbol)
        
        # Get model configuration
        self.model_config = config.get('models', {}).get('arima', {})
        self.p = self.model_config.get('p', 5)
        self.d = self.model_config.get('d', 1)
        self.q = self.model_config.get('q', 0)
        
        # Set model path
        self.model_path = f'models/saved/arima_{symbol.lower()}.pkl'
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ARIMA model.
        
        Note: ARIMA doesn't use X directly. It uses the time series y.
        """
        # Initialize and train ARIMA model
        logger.info(f"Training ARIMA({self.p},{self.d},{self.q}) model for {self.symbol}")
        
        try:
            self.model = ARIMA(y_train, order=(self.p, self.d, self.q))
            self.model = self.model.fit()
            logger.info(f"ARIMA model summary: {self.model.summary()}")
            return self.model
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return None
        
    def predict(self, X):
        """Make predictions using the trained ARIMA model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
            
        # For ARIMA, we predict n_steps ahead
        n_steps = len(X)
        
        try:
            # Make forecast
            forecast = self.model.forecast(steps=n_steps)
            return forecast
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            return None
        
    def save(self, directory: str = 'models/saved'):
        """Save the trained ARIMA model."""
        if self.model is None:
            logger.error("No model to save")
            return
            
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f'arima_{self.symbol.lower()}.pkl')
        
        with open(model_path, 'wb') as f:
            joblib.dump(self.model, f)
            
        logger.info(f"ARIMA model saved to {model_path}")
        self.model_path = model_path
        
        # Save model configuration
        config_path = os.path.join(directory, f'arima_{self.symbol.lower()}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f)
            
    def load(self, directory: str = 'models/saved'):
        """Load a trained ARIMA model."""
        model_path = os.path.join(directory, f'arima_{self.symbol.lower()}.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False
            
        try:
            with open(model_path, 'rb') as f:
                self.model = joblib.load(f)
                
            logger.info(f"ARIMA model loaded from {model_path}")
            self.model_path = model_path
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple prediction models."""
    
    def __init__(self, config: Dict, symbol: str):
        """Initialize the Ensemble model."""
        super().__init__(config, symbol)
        
        # Get model configuration
        self.model_config = config.get('models', {}).get('ensemble', {})
        self.weights = self.model_config.get('weights', {
            'bilstm': 0.5,
            'prophet': 0.2,
            'xgboost': 0.2,
            'arima': 0.1
        })
        
        # Initialize component models
        self.models = {
            'bilstm': BiLSTMModel(config, symbol),
            'prophet': ProphetModel(config, symbol),
            'xgboost': XGBoostModel(config, symbol),
            'arima': ARIMAModel(config, symbol)
        }
        
        # Set model path
        self.model_path = f'models/saved/ensemble_{symbol.lower()}.json'
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train all component models."""
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model for ensemble")
            result = model.train(X_train, y_train, X_val, y_val)
            results[name] = result
            
        return results
        
    def predict(self, X):
        """Make predictions using the weighted ensemble of models."""
        if any(model.model is None for model in self.models.values()):
            logger.error("One or more component models not trained or loaded")
            return None
            
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            pred = model.predict(X)
            if pred is not None:
                predictions[name] = pred
                
        if not predictions:
            logger.error("No valid predictions from component models")
            return None
            
        # Combine predictions using weights
        weighted_sum = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                weighted_sum += weight * pred
                total_weight += weight
                
        # Normalize by total weight
        if total_weight > 0:
            weighted_sum /= total_weight
            
        return weighted_sum
        
    def evaluate(self, X_test, y_test):
        """Evaluate the ensemble model and its components."""
        ensemble_metrics = super().evaluate(X_test, y_test)
        
        # Evaluate each component model
        component_metrics = {}
        for name, model in self.models.items():
            metrics = model.evaluate(X_test, y_test)
            component_metrics[name] = metrics
            
        # Log comparison
        logger.info(f"Ensemble model performance for {self.symbol}:")
        logger.info(f"Ensemble: RMSE={ensemble_metrics['rmse']:.4f}, MAPE={ensemble_metrics['mape']:.2f}%")
        
        for name, metrics in component_metrics.items():
            logger.info(f"{name}: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")
            
        # Return both ensemble and component metrics
        return {
            'ensemble': ensemble_metrics,
            'components': component_metrics
        }
        
    def save(self, directory: str = 'models/saved'):
        """Save all component models and ensemble configuration."""
        os.makedirs(directory, exist_ok=True)
        
        # Save each component model
        for name, model in self.models.items():
            model.save(directory)
            
        # Save ensemble configuration
        config_path = os.path.join(directory, f'ensemble_{self.symbol.lower()}_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'weights': self.weights
            }, f)
            
        logger.info(f"Ensemble model configuration saved to {config_path}")
        self.model_path = config_path
        
    def load(self, directory: str = 'models/saved'):
        """Load all component models and ensemble configuration."""
        # Load ensemble configuration
        config_path = os.path.join(directory, f'ensemble_{self.symbol.lower()}_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.weights = config.get('weights', self.weights)
                
        # Load each component model
        success = True
        for name, model in self.models.items():
            if not model.load(directory):
                logger.warning(f"Failed to load {name} model for ensemble")
                success = False
                
        if success:
            logger.info(f"Ensemble model loaded successfully")
            self.model_path = config_path
            
        return success
        
    def plot_predictions(self, X_test, y_test, output_dir: str = 'visualization/results'):
        """Plot actual vs predicted values for ensemble and components."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions from ensemble and components
        ensemble_pred = self.predict(X_test)
        component_preds = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            if pred is not None:
                component_preds[name] = pred
                
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Plot actual values
        plt.plot(y_test, 'k-', label='Actual', linewidth=2)
        
        # Plot ensemble prediction
        plt.plot(ensemble_pred, 'r-', label='Ensemble', linewidth=2)
        
        # Plot component predictions
        colors = ['b-', 'g-', 'c-', 'm-']
        for i, (name, pred) in enumerate(component_preds.items()):
            plt.plot(pred, colors[i % len(colors)], label=name, alpha=0.7)
            
        plt.title(f'{self.symbol} Price Prediction - Ensemble Model Comparison')
        plt.xlabel('Time')
        plt.ylabel('Price (normalized)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{self.symbol.lower()}_ensemble_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Ensemble comparison plot saved to {plot_path}")


class ModelTrainer:
    """Class for training and managing cryptocurrency prediction models."""
    
    def __init__(self, config: Dict):
        """Initialize the model trainer with configuration."""
        self.config = config
        
    def train_models(self, data_dict: Dict[str, Tuple]):
        """Train models for all cryptocurrencies in the data dictionary."""
        results = {}
        
        for symbol, data_splits in data_dict.items():
            logger.info(f"Training models for {symbol}")
            
            X_train, y_train, X_val, y_val, X_test, y_test = data_splits
            
            # Train ensemble model (which trains all component models)
            ensemble = EnsembleModel(self.config, symbol)
            ensemble.train(X_train, y_train, X_val, y_val)
            
            # Evaluate models
            metrics = ensemble.evaluate(X_test, y_test)
            
            # Plot predictions
            ensemble.plot_predictions(X_test, y_test)
            
            # Save models
            ensemble.save()
            
            results[symbol] = metrics
            
        return results


# Main function to train models
def train_models(config_path: str = "config.yaml"):
    """Run the model training process."""
    # Import here to avoid circular imports
    from crypto_prediction import load_config, setup_logging
    from crypto_prediction.preprocessing.data_preprocessing import preprocess_data
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting model training process")
    
    # Preprocess data
    data_dict = preprocess_data(config_path)
    
    if not data_dict:
        logger.error("No preprocessed data available for training")
        return None
        
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Train models
    results = trainer.train_models(data_dict)
    
    logger.info(f"Model training completed for {len(results)} cryptocurrencies")
    
    return results


if __name__ == "__main__":
    # If run directly, execute model training
    train_models()
