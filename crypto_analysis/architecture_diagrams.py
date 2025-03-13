#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL, MongoDB, InfluxDB
from diagrams.onprem.queue import Kafka
from diagrams.onprem.analytics import Spark
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.aws.storage import S3
from diagrams.aws.compute import Lambda
from diagrams.aws.ml import SageMaker
from diagrams.generic.compute import Rack
from diagrams.generic.network import Firewall
from diagrams.generic.place import Datacenter
from diagrams.generic.storage import Storage
from diagrams.custom import Custom

# Create the output directory if it doesn't exist
import os
os.makedirs("/home/ubuntu/crypto_analysis/diagrams", exist_ok=True)

# System Architecture Diagram
with Diagram("Cryptocurrency Price Prediction System Architecture", 
             filename="/home/ubuntu/crypto_analysis/diagrams/system_architecture", 
             show=False):
    
    # Users and external interfaces
    users = Users("Financial Institutions\nMobile Money Platforms")
    
    with Cluster("Data Collection Layer"):
        data_collectors = [
            Python("CoinGecko Collector"),
            Python("Binance Collector"),
            Python("CoinMarketCap Collector")
        ]
    
    with Cluster("Data Storage Layer"):
        raw_storage = S3("Raw Data Storage")
        time_series_db = InfluxDB("Time Series DB")
        feature_store = PostgreSQL("Feature Store")
    
    with Cluster("Data Processing Layer"):
        with Cluster("ETL Pipeline"):
            data_cleaning = Python("Data Cleaning")
            feature_engineering = Python("Feature Engineering")
            data_transformation = Python("Data Transformation")
        
        data_validation = Python("Data Validation")
    
    with Cluster("Model Layer"):
        with Cluster("Model Training"):
            bilstm = Custom("Bi-LSTM Model", "./crypto_analysis/diagrams/neural.png")
            prophet = Custom("Prophet Model", "./crypto_analysis/diagrams/prophet.png")
            xgboost = Custom("XGBoost Model", "./crypto_analysis/diagrams/xgboost.png")
            arima = Custom("ARIMA Model", "./crypto_analysis/diagrams/arima.png")
        
        model_ensemble = Python("Model Ensemble")
        model_evaluation = Python("Model Evaluation")
    
    with Cluster("Prediction Service"):
        prediction_api = Python("Prediction API")
        cache = MongoDB("Prediction Cache")
    
    with Cluster("Visualization Layer"):
        dashboard = Python("Interactive Dashboard")
        reports = Python("Automated Reports")
    
    # Connect components
    users >> dashboard
    users >> prediction_api
    
    for collector in data_collectors:
        collector >> raw_storage
    
    raw_storage >> data_cleaning >> feature_engineering >> data_transformation
    data_transformation >> feature_store
    data_transformation >> time_series_db
    
    feature_store >> data_validation
    time_series_db >> data_validation
    
    data_validation >> bilstm
    data_validation >> prophet
    data_validation >> xgboost
    data_validation >> arima
    
    bilstm >> model_ensemble
    prophet >> model_ensemble
    xgboost >> model_ensemble
    arima >> model_ensemble
    
    model_ensemble >> model_evaluation
    model_evaluation >> prediction_api
    
    prediction_api >> cache
    cache >> dashboard
    model_evaluation >> reports
    reports >> users

# Data Pipeline Architecture
with Diagram("Cryptocurrency Price Prediction Data Pipeline", 
             filename="/home/ubuntu/crypto_analysis/diagrams/data_pipeline", 
             show=False):
    
    with Cluster("Data Sources"):
        coingecko = Custom("CoinGecko API", "./crypto_analysis/diagrams/api.png")
        binance = Custom("Binance API", "./crypto_analysis/diagrams/api.png")
        coinmarketcap = Custom("CoinMarketCap API", "./crypto_analysis/diagrams/api.png")
    
    with Cluster("Data Collection"):
        scheduler = Python("Scheduler")
        collectors = [
            Python("Historical Data Collector"),
            Python("Real-time Data Collector"),
            Python("Market Metrics Collector")
        ]
    
    with Cluster("Data Storage"):
        raw_data = S3("Raw Data Lake")
        time_series = InfluxDB("Time Series DB")
        relational_db = PostgreSQL("Relational DB")
    
    with Cluster("Data Processing"):
        with Cluster("Data Cleaning"):
            outlier_detection = Python("Outlier Detection")
            missing_values = Python("Missing Values Handler")
            normalization = Python("Data Normalization")
        
        with Cluster("Feature Engineering"):
            technical_indicators = Python("Technical Indicators")
            time_features = Python("Time-based Features")
            market_features = Python("Market Features")
        
        feature_store = PostgreSQL("Feature Store")
    
    with Cluster("Data Validation"):
        quality_checks = Python("Data Quality Checks")
        schema_validation = Python("Schema Validation")
        consistency_checks = Python("Consistency Checks")
    
    # Connect components
    coingecko >> scheduler
    binance >> scheduler
    coinmarketcap >> scheduler
    
    scheduler >> collectors[0]
    scheduler >> collectors[1]
    scheduler >> collectors[2]
    
    collectors[0] >> raw_data
    collectors[1] >> time_series
    collectors[2] >> relational_db
    
    raw_data >> outlier_detection
    time_series >> outlier_detection
    relational_db >> outlier_detection
    
    outlier_detection >> missing_values >> normalization
    
    normalization >> technical_indicators
    normalization >> time_features
    normalization >> market_features
    
    technical_indicators >> feature_store
    time_features >> feature_store
    market_features >> feature_store
    
    feature_store >> quality_checks >> schema_validation >> consistency_checks

# Model Architecture
with Diagram("Cryptocurrency Price Prediction Model Architecture", 
             filename="/home/ubuntu/crypto_analysis/diagrams/model_architecture", 
             show=False):
    
    with Cluster("Feature Store"):
        feature_store = PostgreSQL("Feature Store")
        feature_registry = Python("Feature Registry")
    
    with Cluster("Model Training Pipeline"):
        data_split = Python("Train/Val/Test Split")
        
        with Cluster("Model Training"):
            bilstm_train = Custom("Bi-LSTM Training", "./crypto_analysis/diagrams/neural.png")
            prophet_train = Custom("Prophet Training", "./crypto_analysis/diagrams/prophet.png")
            xgboost_train = Custom("XGBoost Training", "./crypto_analysis/diagrams/xgboost.png")
            arima_train = Custom("ARIMA Training", "./crypto_analysis/diagrams/arima.png")
        
        with Cluster("Hyperparameter Tuning"):
            bilstm_tuning = Python("Bi-LSTM Tuning")
            prophet_tuning = Python("Prophet Tuning")
            xgboost_tuning = Python("XGBoost Tuning")
            arima_tuning = Python("ARIMA Tuning")
    
    with Cluster("Model Registry"):
        model_registry = Storage("Model Registry")
        model_versioning = Python("Model Versioning")
    
    with Cluster("Model Ensemble"):
        ensemble_training = Python("Ensemble Training")
        weight_optimization = Python("Weight Optimization")
    
    with Cluster("Model Evaluation"):
        metrics_calculation = Python("Metrics Calculation")
        backtesting = Python("Backtesting")
        performance_analysis = Python("Performance Analysis")
    
    with Cluster("Model Deployment"):
        model_serving = Python("Model Serving")
        prediction_api = Python("Prediction API")
        monitoring = Python("Model Monitoring")
    
    # Connect components
    feature_store >> feature_registry >> data_split
    
    data_split >> bilstm_train >> bilstm_tuning
    data_split >> prophet_train >> prophet_tuning
    data_split >> xgboost_train >> xgboost_tuning
    data_split >> arima_train >> arima_tuning
    
    bilstm_tuning >> model_registry
    prophet_tuning >> model_registry
    xgboost_tuning >> model_registry
    arima_tuning >> model_registry
    
    model_registry >> model_versioning
    model_versioning >> ensemble_training
    
    ensemble_training >> weight_optimization
    
    weight_optimization >> metrics_calculation
    metrics_calculation >> backtesting
    backtesting >> performance_analysis
    
    performance_analysis >> model_serving
    model_serving >> prediction_api
    prediction_api >> monitoring
    monitoring >> ensemble_training

# Deployment Architecture
with Diagram("Cryptocurrency Price Prediction Deployment Architecture", 
             filename="/home/ubuntu/crypto_analysis/diagrams/deployment_architecture", 
             show=False):
    
    with Cluster("Development Environment"):
        dev_git = Custom("Git Repository", "./crypto_analysis/diagrams/git.png")
        dev_docker = Custom("Docker Containers", "./crypto_analysis/diagrams/docker.png")
        dev_ci = Custom("CI/CD Pipeline", "./crypto_analysis/diagrams/cicd.png")
    
    with Cluster("Testing Environment"):
        test_docker = Custom("Docker Compose", "./crypto_analysis/diagrams/docker.png")
        test_automation = Python("Test Automation")
        test_monitoring = Python("Performance Monitoring")
    
    with Cluster("Production Environment"):
        with Cluster("Data Layer"):
            prod_storage = S3("Data Storage")
            prod_db = PostgreSQL("Database Cluster")
            prod_cache = MongoDB("Cache Layer")
        
        with Cluster("Application Layer"):
            api_service = Custom("API Service", "./crypto_analysis/diagrams/api.png")
            prediction_service = Python("Prediction Service")
            training_service = Python("Training Service")
        
        with Cluster("Presentation Layer"):
            web_dashboard = Python("Web Dashboard")
            mobile_api = Python("Mobile API")
        
        with Cluster("Monitoring & Operations"):
            logging = Custom("Logging", "./crypto_analysis/diagrams/logging.png")
            metrics = Custom("Metrics", "./crypto_analysis/diagrams/metrics.png")
            alerts = Custom("Alerts", "./crypto_analysis/diagrams/alerts.png")
    
    # Connect components
    dev_git >> dev_docker >> dev_ci
    
    dev_ci >> test_docker
    test_docker >> test_automation >> test_monitoring
    
    test_monitoring >> prod_storage
    test_monitoring >> prod_db
    test_monitoring >> prod_cache
    
    prod_storage >> api_service
    prod_db >> api_service
    prod_cache >> api_service
    
    api_service >> prediction_service
    api_service >> training_service
    
    prediction_service >> web_dashboard
    prediction_service >> mobile_api
    
    web_dashboard >> logging
    mobile_api >> logging
    prediction_service >> metrics
    training_service >> metrics
    metrics >> alerts

print("Architecture diagrams generated in /home/ubuntu/crypto_analysis/diagrams/")
