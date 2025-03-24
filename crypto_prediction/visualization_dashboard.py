"""
Main script to run the visualization dashboard for cryptocurrency price prediction system.
This script starts the Dash web application for visualizing predictions and model performance.
"""

import os
import logging
import argparse

# Import project modules
from crypto_prediction import load_config, setup_logging
from visualization.dashboard import run_dashboard

def run_visualization_dashboard(config_path=None, host=None, port=None, debug=None):
    """Run the visualization dashboard."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Get dashboard configuration
    dashboard_config = config.get('visualization', {}).get('dashboard', {})
    host = host or dashboard_config.get('host', '0.0.0.0')
    port = port or dashboard_config.get('port', 8050)
    debug = debug if debug is not None else dashboard_config.get('debug', False)
    
    logger.info(f"Starting visualization dashboard on {host}:{port}")
    
    # Use the logger
    logger.info("Application started")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Run dashboard
    run_dashboard(host, port, debug)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cryptocurrency visualization dashboard')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--host', type=str, help='Host to bind the server to')
    parser.add_argument('--port', type=int, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_visualization_dashboard(args.config, args.host, args.port, args.debug)
