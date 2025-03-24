"""
Main script to run the prediction service for cryptocurrency price prediction system.
This script starts the FastAPI server for making predictions.
"""

import os
import logging
import argparse
import uvicorn
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from crypto_prediction import load_config, setup_logging

def run_prediction_service(config_path=None, host=None, port=None, debug=None):
    """Run the prediction service API."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Get API configuration
    api_config = config.get('api', {})
    host = host or api_config.get('host', '0.0.0.0')
    port = port or api_config.get('port', 8000)
    debug = debug if debug is not None else api_config.get('debug', False)
    
    logger.info(f"Starting prediction service API on {host}:{port}")
    
    # Start the API server
    uvicorn.run("crypto_prediction.api.prediction_api:app", 
                host=host, 
                port=port, 
                reload=debug,
                log_level="info")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cryptocurrency prediction service')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--host', type=str, help='Host to bind the server to')
    parser.add_argument('--port', type=int, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_prediction_service(args.config, args.host, args.port, args.debug)
