"""
Main script to deploy the cryptocurrency price prediction system.
This script orchestrates the deployment of all system components.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import yaml
import json
from pathlib import Path

# Import project modules
from crypto_prediction import load_config, setup_logging

def create_deployment_directories():
    """Create necessary directories for deployment."""
    directories = [
        'deployment/logs',
        'deployment/data',
        'deployment/models',
        'deployment/configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created deployment directory: {directory}")

def copy_configuration(config_path=None):
    """Copy configuration file to deployment directory."""
    # Load configuration
    config = load_config(config_path)
    
    # Create deployment config
    deployment_config = config.copy()
    
    # Update paths for deployment
    deployment_config['data_collection']['data_dir'] = 'deployment/data'
    deployment_config['models']['save_dir'] = 'deployment/models/saved'
    deployment_config['logging']['log_dir'] = 'deployment/logs'
    
    # Save deployment config
    deployment_config_path = 'deployment/configs/config.yaml'
    
    with open(deployment_config_path, 'w') as f:
        yaml.dump(deployment_config, f, default_flow_style=False)
    
    logging.info(f"Created deployment configuration at {deployment_config_path}")
    
    return deployment_config_path

def create_systemd_services():
    """Create systemd service files for system components."""
    # Define service templates
    service_templates = {
        'crypto-data-collection': {
            'Description': 'Cryptocurrency Data Collection Service',
            'ExecStart': '/usr/bin/python3 /home/ubuntu/crypto_prediction/data_collection_pipeline.py --mode scheduled --config /home/ubuntu/crypto_prediction/deployment/configs/config.yaml'
        },
        'crypto-prediction-api': {
            'Description': 'Cryptocurrency Prediction API Service',
            'ExecStart': '/usr/bin/python3 /home/ubuntu/crypto_prediction/prediction_service.py --config /home/ubuntu/crypto_prediction/deployment/configs/config.yaml'
        },
        'crypto-dashboard': {
            'Description': 'Cryptocurrency Prediction Dashboard Service',
            'ExecStart': '/usr/bin/python3 /home/ubuntu/crypto_prediction/visualization_dashboard.py --config /home/ubuntu/crypto_prediction/deployment/configs/config.yaml'
        }
    }
    
    # Create service files
    service_dir = 'deployment/systemd'
    os.makedirs(service_dir, exist_ok=True)
    
    for service_name, service_config in service_templates.items():
        service_content = f"""[Unit]
Description={service_config['Description']}
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto_prediction
ExecStart={service_config['ExecStart']}
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier={service_name}

[Install]
WantedBy=multi-user.target
"""
        
        service_path = os.path.join(service_dir, f"{service_name}.service")
        
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        logging.info(f"Created systemd service file: {service_path}")
    
    return service_dir

def create_docker_compose():
    """Create Docker Compose file for containerized deployment."""
    docker_compose = {
        'version': '3',
        'services': {
            'data-collection': {
                'build': {
                    'context': '.',
                    'dockerfile': 'deployment/docker/Dockerfile'
                },
                'command': 'python data_collection_pipeline.py --mode scheduled --config /app/deployment/configs/config.yaml',
                'volumes': [
                    './deployment:/app/deployment'
                ],
                'restart': 'always'
            },
            'prediction-api': {
                'build': {
                    'context': '.',
                    'dockerfile': 'deployment/docker/Dockerfile'
                },
                'command': 'python prediction_service.py --config /app/deployment/configs/config.yaml',
                'volumes': [
                    './deployment:/app/deployment'
                ],
                'ports': [
                    '8000:8000'
                ],
                'depends_on': [
                    'data-collection'
                ],
                'restart': 'always'
            },
            'dashboard': {
                'build': {
                    'context': '.',
                    'dockerfile': 'deployment/docker/Dockerfile'
                },
                'command': 'python visualization_dashboard.py --config /app/deployment/configs/config.yaml',
                'volumes': [
                    './deployment:/app/deployment'
                ],
                'ports': [
                    '8050:8050'
                ],
                'depends_on': [
                    'prediction-api'
                ],
                'restart': 'always'
            }
        }
    }
    
    # Create Docker directory
    docker_dir = 'deployment/docker'
    os.makedirs(docker_dir, exist_ok=True)
    
    # Create Docker Compose file
    docker_compose_path = os.path.join(docker_dir, 'docker-compose.yml')
    
    with open(docker_compose_path, 'w') as f:
        yaml.dump(docker_compose, f, default_flow_style=False)
    
    logging.info(f"Created Docker Compose file: {docker_compose_path}")
    
    # Create Dockerfile
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "prediction_service.py"]
"""
    
    dockerfile_path = os.path.join(docker_dir, 'Dockerfile')
    
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    logging.info(f"Created Dockerfile: {dockerfile_path}")
    
    return docker_dir

def create_deployment_documentation():
    """Create deployment documentation."""
    documentation = """# Cryptocurrency Price Prediction System Deployment Guide

This document provides instructions for deploying the cryptocurrency price prediction system.

## System Components

The system consists of the following components:

1. **Data Collection Service**: Collects cryptocurrency data from various sources.
2. **Prediction API**: Provides REST API for making price predictions.
3. **Visualization Dashboard**: Web interface for visualizing predictions and model performance.

## Deployment Options

There are two deployment options:

1. **Systemd Services**: Deploy as systemd services on a Linux server.
2. **Docker Containers**: Deploy as Docker containers using Docker Compose.

## Systemd Deployment

To deploy as systemd services:

1. Copy the service files from `deployment/systemd/` to `/etc/systemd/system/`:

```bash
sudo cp deployment/systemd/*.service /etc/systemd/system/
```

2. Reload systemd:

```bash
sudo systemctl daemon-reload
```

3. Enable and start the services:

```bash
sudo systemctl enable crypto-data-collection.service
sudo systemctl enable crypto-prediction-api.service
sudo systemctl enable crypto-dashboard.service

sudo systemctl start crypto-data-collection.service
sudo systemctl start crypto-prediction-api.service
sudo systemctl start crypto-dashboard.service
```

4. Check service status:

```bash
sudo systemctl status crypto-data-collection.service
sudo systemctl status crypto-prediction-api.service
sudo systemctl status crypto-dashboard.service
```

## Docker Deployment

To deploy using Docker:

1. Make sure Docker and Docker Compose are installed.

2. Build and start the containers:

```bash
cd deployment/docker
docker-compose up -d
```

3. Check container status:

```bash
docker-compose ps
```

## Accessing the System

- **Prediction API**: Available at `http://localhost:8000`
- **Visualization Dashboard**: Available at `http://localhost:8050`

## Configuration

The deployment configuration is stored in `deployment/configs/config.yaml`. You can modify this file to change system behavior.

## Logs

- **Systemd Deployment**: Logs are available via journalctl:

```bash
journalctl -u crypto-data-collection.service
journalctl -u crypto-prediction-api.service
journalctl -u crypto-dashboard.service
```

- **Docker Deployment**: Logs are available via docker-compose:

```bash
docker-compose logs data-collection
docker-compose logs prediction-api
docker-compose logs dashboard
```

## Troubleshooting

If you encounter issues:

1. Check the logs for error messages.
2. Verify that all required dependencies are installed.
3. Ensure the configuration file is correctly set up.
4. Check network connectivity for data collection services.

## Maintenance

- **Model Retraining**: Models can be retrained by calling the `/retrain` endpoint of the Prediction API.
- **Data Backup**: The system stores data in `deployment/data/`. Consider setting up regular backups of this directory.
"""
    
    # Create documentation file
    docs_dir = 'deployment/docs'
    os.makedirs(docs_dir, exist_ok=True)
    
    docs_path = os.path.join(docs_dir, 'deployment_guide.md')
    
    with open(docs_path, 'w') as f:
        f.write(documentation)
    
    logging.info(f"Created deployment documentation: {docs_path}")
    
    return docs_path

def deploy_system(config_path=None, deployment_type='systemd'):
    """Deploy the cryptocurrency price prediction system."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info(f"Starting deployment of cryptocurrency price prediction system using {deployment_type}")
    
    # Create deployment directories
    create_deployment_directories()
    
    # Copy configuration
    deployment_config_path = copy_configuration(config_path)
    
    # Create deployment files based on type
    if deployment_type == 'systemd':
        service_dir = create_systemd_services()
        logger.info(f"Created systemd service files in {service_dir}")
    elif deployment_type == 'docker':
        docker_dir = create_docker_compose()
        logger.info(f"Created Docker deployment files in {docker_dir}")
    else:
        logger.error(f"Unsupported deployment type: {deployment_type}")
        return False
    
    # Create deployment documentation
    docs_path = create_deployment_documentation()
    logger.info(f"Created deployment documentation at {docs_path}")
    
    logger.info("Deployment preparation completed successfully")
    
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deploy cryptocurrency price prediction system')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--type', type=str, choices=['systemd', 'docker'], default='systemd',
                        help='Deployment type (systemd or docker)')
    
    args = parser.parse_args()
    
    # Deploy system
    success = deploy_system(args.config, args.type)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
