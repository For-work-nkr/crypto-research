# Cryptocurrency Price Prediction System

This repository contains a comprehensive cryptocurrency price prediction system that enables financial institutions and mobile money platforms to forecast cryptocurrency price movements accurately and efficiently.

## Overview

The system uses real-time market data from APIs such as CoinGecko, Binance, and CoinMarketCap, combined with machine learning algorithms, to deliver reliable short-term price forecasts for multiple cryptocurrencies. The system integrates technical indicators, historical data, and sentiment analysis to improve prediction accuracy.

## Features

- **Multi-source Data Collection**: Collects data from multiple cryptocurrency APIs with different timeframes (historical, recent, real-time)
- **Advanced Preprocessing**: Implements data cleaning, feature engineering, normalization, and sequence preparation
- **Multiple Prediction Models**: Includes Bi-LSTM, Prophet, XGBoost, ARIMA, and ensemble models
- **REST API**: Provides a FastAPI-based prediction service for making forecasts
- **Interactive Dashboard**: Offers a Dash-based visualization dashboard for monitoring predictions and model performance
- **Comprehensive Testing**: Includes a test suite to ensure all components work correctly
- **Flexible Deployment**: Supports deployment as systemd services or Docker containers

## System Architecture

The system consists of the following components:

1. **Data Collection Pipeline**: Collects cryptocurrency data from various sources
2. **Data Preprocessing Module**: Transforms raw data into a format suitable for model training
3. **Model Training Framework**: Trains and evaluates prediction models
4. **Prediction Service**: Provides REST API for making price predictions
5. **Visualization Dashboard**: Web interface for visualizing predictions and model performance

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker and Docker Compose for containerized deployment

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crypto-prediction.git
cd crypto-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a configuration file:

```bash
cp config.example.yaml config.yaml
```

4. Edit the configuration file to set up your API keys and preferences.

## Usage

### Data Collection

To collect historical cryptocurrency data:

```bash
python data_collection_pipeline.py --mode historical
```

To run continuous data collection:

```bash
python data_collection_pipeline.py --mode scheduled
```

### Data Preprocessing

To preprocess data for all configured cryptocurrencies:

```bash
python data_preprocessing_pipeline.py
```

### Model Training

To train models for all cryptocurrencies:

```bash
python model_training_pipeline.py
```

### Prediction Service

To start the prediction API:

```bash
python prediction_service.py
```

The API will be available at http://localhost:8000.

### Visualization Dashboard

To start the visualization dashboard:

```bash
python visualization_dashboard.py
```

The dashboard will be available at http://localhost:8050.

### Testing

To run the test suite:

```bash
python run_tests.py
```

### Deployment

To deploy the system:

```bash
python deploy_system.py --type systemd  # or --type docker
```

See the deployment documentation in `deployment/docs/deployment_guide.md` for detailed instructions.

## API Documentation

The prediction API provides the following endpoints:

- `GET /`: Root endpoint returning API information
- `GET /health`: Health check endpoint
- `GET /models`: List all available trained models
- `POST /predict`: Make prediction for a cryptocurrency
- `GET /predict/{symbol}`: Make prediction for a cryptocurrency (GET method)
- `POST /retrain`: Trigger model retraining
- `GET /cryptocurrencies`: List all configured cryptocurrencies
- `GET /current-price/{symbol}`: Get current price for a cryptocurrency

## Dashboard Features

The visualization dashboard provides:

- Cryptocurrency selection and model type selection
- Price prediction visualization with confidence intervals
- Historical price charts with moving averages
- Model performance metrics display
- Model comparison charts

## Configuration

The system is configured using a YAML file with the following sections:

- `cryptocurrencies`: List of cryptocurrencies to track
- `data_collection`: Settings for data collection (APIs, timeframes, update frequencies)
- `preprocessing`: Settings for data preprocessing
- `models`: Settings for prediction models
- `api`: Settings for the prediction API
- `visualization`: Settings for the visualization dashboard
- `logging`: Logging configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
