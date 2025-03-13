"""
Main script to run the test suite for cryptocurrency price prediction system.
This script runs all tests to ensure the system works correctly.
"""

import os
import sys
import logging
import argparse

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from crypto_prediction import load_config, setup_logging
from tests.test_system import run_tests

def run_test_suite(config_path=None):
    """Run the complete test suite."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting test suite for cryptocurrency price prediction system")
    
    # Run tests
    result = run_tests()
    
    # Log results
    if result.wasSuccessful():
        logger.info("All tests passed successfully")
    else:
        logger.error(f"Tests failed: {result.failures} failures, {result.errors} errors")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cryptocurrency prediction system test suite')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run test suite
    success = run_test_suite(args.config)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
