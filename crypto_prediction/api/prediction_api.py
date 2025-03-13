"""
Prediction service API for cryptocurrency price prediction system.
This module implements a FastAPI service for making predictions.
"""

import os
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import json
import time
from datetime import datetime, timedelta
import uvicorn

# Import project modules
from crypto_prediction import load_config, setup_logging
from crypto_prediction.data.data_collection import DataCollector
from crypto_prediction.preprocessing.data_preprocessing import DataPreprocessor
from crypto_prediction.models.model_training import (
    BiLSTMModel, ProphetModel, XGBoostModel, ARIMAModel, EnsembleModel
)

# Setup logging
config = load_config()
logger = setup_logging(config)

# Initialize FastAPI app
app = FastAPI(
    title="Cryptocurrency Price Prediction API",
    description="API for predicting cryptocurrency prices using machine learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize data collector and preprocessor
data_collector = DataCollector(config)
data_preprocessor = DataPreprocessor(config)

# Cache for loaded models
model_cache = {}

# Cache for predictions
prediction_cache = {}

# Background task for model retraining
def retrain_models_task():
    """Background task to retrain models with latest data."""
    logger.info("Starting background model retraining task")
    
    try:
        # Import here to avoid circular imports
        from crypto_prediction.models.model_training import train_models
        
        # Retrain models
        train_models()
        
        logger.info("Model retraining completed successfully")
    except Exception as e:
        logger.error(f"Error in model retraining task: {e}")

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    symbol: str
    horizon: int = 1
    model_type: str = "ensemble"

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    symbol: str
    predictions: List[float]
    timestamps: List[str]
    model_type: str
    confidence_interval: Optional[Dict[str, List[float]]] = None
    last_updated: str

class ModelInfo(BaseModel):
    """Model for model information."""
    symbol: str
    model_type: str
    last_trained: str
    metrics: Dict

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str
    timestamp: str

# Helper functions
def get_model(symbol: str, model_type: str = "ensemble"):
    """Get or load a trained model."""
    cache_key = f"{symbol.lower()}_{model_type.lower()}"
    
    # Check if model is in cache
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    # Initialize model based on type
    if model_type.lower() == "bilstm":
        model = BiLSTMModel(config, symbol)
    elif model_type.lower() == "prophet":
        model = ProphetModel(config, symbol)
    elif model_type.lower() == "xgboost":
        model = XGBoostModel(config, symbol)
    elif model_type.lower() == "arima":
        model = ARIMAModel(config, symbol)
    else:  # Default to ensemble
        model = EnsembleModel(config, symbol)
    
    # Load model
    if model.load():
        # Add to cache
        model_cache[cache_key] = model
        return model
    else:
        raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")

def prepare_prediction_data(symbol: str, days: int = 60):
    """Prepare data for prediction."""
    # Get historical data
    df = data_collector.get_historical_data(symbol, timeframe="1d", days=days)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Historical data for {symbol} not found")
    
    # Clean and preprocess data
    df = data_preprocessor.clean_data(df)
    df = data_preprocessor.add_technical_indicators(df)
    df = data_preprocessor.add_temporal_features(df)
    df = data_preprocessor.add_price_features(df)
    
    # Normalize data
    df_norm = data_preprocessor.normalize_features(df)
    
    # Get sequence length from model configuration
    sequence_length = config.get('models', {}).get('bilstm', {}).get('sequence_length', 60)
    
    # Prepare sequence
    X, _ = data_preprocessor.prepare_sequences(df_norm, sequence_length)
    
    if X is None or len(X) == 0:
        raise HTTPException(status_code=500, detail=f"Failed to prepare prediction data for {symbol}")
    
    return X[-1:], df  # Return the last sequence and the original dataframe

def make_prediction(symbol: str, horizon: int = 1, model_type: str = "ensemble"):
    """Make prediction for a cryptocurrency."""
    # Check cache first
    cache_key = f"{symbol.lower()}_{model_type.lower()}_{horizon}"
    current_time = time.time()
    
    # If prediction is in cache and less than 1 hour old, return it
    if cache_key in prediction_cache:
        cached_pred = prediction_cache[cache_key]
        if current_time - cached_pred['timestamp'] < 3600:  # 1 hour
            return cached_pred['data']
    
    # Get model
    model = get_model(symbol, model_type)
    
    # Prepare data
    X, df = prepare_prediction_data(symbol)
    
    # Make initial prediction
    initial_pred = model.predict(X)[0]
    
    # For multi-step prediction, we need to iteratively predict
    predictions = [initial_pred]
    
    # Get the last date in the historical data
    last_date = df.index[-1]
    timestamps = [last_date + timedelta(days=1)]
    
    # For horizons > 1, we need to iteratively predict
    if horizon > 1:
        # This is a simplified approach - in practice, you'd need to update
        # all features for each step, which is complex
        for i in range(1, horizon):
            # Use the last prediction as input for the next
            # This is a very simplified approach
            next_pred = model.predict(X)[0] * (1 + np.random.normal(0, 0.01))  # Add small random variation
            predictions.append(next_pred)
            timestamps.append(last_date + timedelta(days=i+1))
    
    # Create response
    response = {
        "symbol": symbol,
        "predictions": predictions,
        "timestamps": [ts.strftime("%Y-%m-%d") for ts in timestamps],
        "model_type": model_type,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add confidence intervals for Prophet model
    if model_type.lower() == "prophet" and hasattr(model.model, 'uncertainty_samples'):
        lower = [p * 0.9 for p in predictions]  # Simplified - 10% lower
        upper = [p * 1.1 for p in predictions]  # Simplified - 10% higher
        response["confidence_interval"] = {
            "lower": lower,
            "upper": upper
        }
    
    # Cache the prediction
    prediction_cache[cache_key] = {
        'timestamp': current_time,
        'data': response
    }
    
    return response

# API endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint returning API information."""
    return {
        "status": "online",
        "version": "1.0.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available trained models."""
    models_dir = "models/saved"
    
    if not os.path.exists(models_dir):
        return []
    
    model_files = os.listdir(models_dir)
    model_info = []
    
    # Process model files to extract information
    for file in model_files:
        if file.endswith("_config.json"):
            parts = file.split("_")
            if len(parts) >= 2:
                model_type = parts[0]
                symbol = parts[1].split(".")[0].upper()
                
                # Load metrics if available
                metrics_file = os.path.join(models_dir, f"{model_type}_{symbol.lower()}_metrics.json")
                metrics = {}
                
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                
                # Get last modified time as proxy for training time
                last_trained = datetime.fromtimestamp(os.path.getmtime(
                    os.path.join(models_dir, file)
                )).strftime("%Y-%m-%d %H:%M:%S")
                
                model_info.append({
                    "symbol": symbol,
                    "model_type": model_type,
                    "last_trained": last_trained,
                    "metrics": metrics
                })
    
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction for a cryptocurrency."""
    try:
        return make_prediction(
            request.symbol,
            request.horizon,
            request.model_type
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict_get(
    symbol: str,
    horizon: int = Query(1, ge=1, le=30),
    model_type: str = Query("ensemble")
):
    """Make prediction for a cryptocurrency (GET method)."""
    try:
        return make_prediction(symbol, horizon, model_type)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain", status_code=202)
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining in the background."""
    background_tasks.add_task(retrain_models_task)
    return {"message": "Model retraining started in the background"}

@app.get("/cryptocurrencies", response_model=List[Dict[str, str]])
async def list_cryptocurrencies():
    """List all configured cryptocurrencies."""
    cryptocurrencies = config.get('cryptocurrencies', [])
    return [{"symbol": crypto['symbol'], "name": crypto['name']} for crypto in cryptocurrencies]

@app.get("/current-price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price for a cryptocurrency."""
    try:
        price = data_collector.get_current_price(symbol)
        if price is None:
            raise HTTPException(status_code=404, detail=f"Price for {symbol} not found")
        
        return {
            "symbol": symbol,
            "price": price,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
def start_api():
    """Start the FastAPI server."""
    # Get API configuration
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run("crypto_prediction.api.prediction_api:app", host=host, port=port, reload=debug)

if __name__ == "__main__":
    # If run directly, start the API server
    start_api()
