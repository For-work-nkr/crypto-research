"""
Visualization dashboard for cryptocurrency price prediction system.
This module implements a Dash web application for visualizing predictions and model performance.
"""

import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests
import json
from datetime import datetime, timedelta

# Import project modules
from crypto_prediction import load_config, setup_logging

# Load configuration
config = load_config()
logger = setup_logging(config)

# Get API configuration
api_config = config.get('api', {})
api_host = api_config.get('host', '0.0.0.0')
api_port = api_config.get('port', 8000)
API_BASE_URL = f"http://{api_host}:{api_port}"

# Get visualization configuration
viz_config = config.get('visualization', {}).get('dashboard', {})
refresh_interval = viz_config.get('refresh_interval', 300) * 1000  # Convert to milliseconds

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Cryptocurrency Price Prediction Dashboard"

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Cryptocurrency Price Prediction Dashboard", className="text-center my-4"),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Settings"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Cryptocurrency"),
                            dcc.Dropdown(
                                id="crypto-dropdown",
                                options=[],  # Will be populated from API
                                value=None,
                                placeholder="Select a cryptocurrency"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Model Type"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[
                                    {"label": "Ensemble", "value": "ensemble"},
                                    {"label": "Bi-LSTM", "value": "bilstm"},
                                    {"label": "Prophet", "value": "prophet"},
                                    {"label": "XGBoost", "value": "xgboost"},
                                    {"label": "ARIMA", "value": "arima"}
                                ],
                                value="ensemble",
                                placeholder="Select a model"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Prediction Horizon (Days)"),
                            dcc.Slider(
                                id="horizon-slider",
                                min=1,
                                max=30,
                                step=1,
                                value=7,
                                marks={1: "1", 7: "7", 14: "14", 30: "30"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Generate Prediction", id="predict-button", color="primary", className="mt-3")
                        ], width=12, className="text-center")
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Prediction"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-prediction",
                        type="circle",
                        children=[
                            dcc.Graph(id="prediction-graph", style={"height": "400px"})
                        ]
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Historical Performance"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-historical",
                        type="circle",
                        children=[
                            dcc.Graph(id="historical-graph", style={"height": "400px"})
                        ]
                    )
                ])
            ], className="mb-4")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Performance Metrics"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-metrics",
                        type="circle",
                        children=[
                            html.Div(id="metrics-container")
                        ]
                    )
                ])
            ], className="mb-4")
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Comparison"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-comparison",
                        type="circle",
                        children=[
                            dcc.Graph(id="model-comparison-graph", style={"height": "400px"})
                        ]
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Hidden div for storing data
    html.Div(id="prediction-data-store", style={"display": "none"}),
    html.Div(id="historical-data-store", style={"display": "none"}),
    html.Div(id="models-data-store", style={"display": "none"}),
    
    # Interval for refreshing data
    dcc.Interval(
        id="refresh-interval",
        interval=refresh_interval,
        n_intervals=0
    )
], fluid=True)

# Callback to populate cryptocurrency dropdown
@app.callback(
    Output("crypto-dropdown", "options"),
    Input("refresh-interval", "n_intervals")
)
def update_crypto_dropdown(n_intervals):
    try:
        response = requests.get(f"{API_BASE_URL}/cryptocurrencies")
        if response.status_code == 200:
            cryptos = response.json()
            options = [{"label": f"{crypto['name']} ({crypto['symbol']})", "value": crypto['symbol']} for crypto in cryptos]
            
            # Set default value if available
            if options and not app.layout["crypto-dropdown"].value:
                app.layout["crypto-dropdown"].value = options[0]["value"]
                
            return options
        else:
            logger.error(f"Failed to fetch cryptocurrencies: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching cryptocurrencies: {e}")
        return []

# Callback to fetch and store models data
@app.callback(
    Output("models-data-store", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_models_data(n_intervals):
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()
            return json.dumps(models)
        else:
            logger.error(f"Failed to fetch models: {response.status_code}")
            return json.dumps([])
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return json.dumps([])

# Callback to generate prediction
@app.callback(
    Output("prediction-data-store", "children"),
    Input("predict-button", "n_clicks"),
    State("crypto-dropdown", "value"),
    State("model-dropdown", "value"),
    State("horizon-slider", "value"),
    prevent_initial_call=True
)
def generate_prediction(n_clicks, symbol, model_type, horizon):
    if not symbol:
        return json.dumps({})
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/predict/{symbol}",
            params={"horizon": horizon, "model_type": model_type}
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            return json.dumps(prediction_data)
        else:
            logger.error(f"Failed to generate prediction: {response.status_code}")
            return json.dumps({})
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return json.dumps({})

# Callback to fetch historical data
@app.callback(
    Output("historical-data-store", "children"),
    Input("crypto-dropdown", "value"),
    prevent_initial_call=True
)
def fetch_historical_data(symbol):
    if not symbol:
        return json.dumps({})
    
    try:
        # This is a simplified approach - in a real implementation,
        # you would have an API endpoint to fetch historical data
        # For now, we'll create some dummy data
        
        # Get current price
        response = requests.get(f"{API_BASE_URL}/current-price/{symbol}")
        
        if response.status_code == 200:
            current_price = response.json()["price"]
            
            # Generate dummy historical data
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
            
            # Create some random price movements around the current price
            np.random.seed(42)  # For reproducibility
            price_factor = np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
            prices = [current_price / price_factor[-1] * factor for factor in price_factor]
            
            historical_data = {
                "symbol": symbol,
                "dates": dates,
                "prices": prices
            }
            
            return json.dumps(historical_data)
        else:
            logger.error(f"Failed to fetch current price: {response.status_code}")
            return json.dumps({})
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return json.dumps({})

# Callback to update prediction graph
@app.callback(
    Output("prediction-graph", "figure"),
    Input("prediction-data-store", "children"),
    Input("historical-data-store", "children")
)
def update_prediction_graph(prediction_json, historical_json):
    try:
        prediction_data = json.loads(prediction_json) if prediction_json else {}
        historical_data = json.loads(historical_json) if historical_json else {}
        
        if not prediction_data or not historical_data:
            return {
                "data": [],
                "layout": {
                    "title": "No prediction data available",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price (USD)"},
                    "template": "plotly_dark"
                }
            }
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data["dates"],
            y=historical_data["prices"],
            mode="lines",
            name="Historical",
            line={"color": "blue"}
        ))
        
        # Add prediction data
        fig.add_trace(go.Scatter(
            x=prediction_data["timestamps"],
            y=prediction_data["predictions"],
            mode="lines+markers",
            name="Prediction",
            line={"color": "green", "dash": "dash"},
            marker={"size": 8}
        ))
        
        # Add confidence interval if available
        if "confidence_interval" in prediction_data:
            fig.add_trace(go.Scatter(
                x=prediction_data["timestamps"] + prediction_data["timestamps"][::-1],
                y=prediction_data["confidence_interval"]["upper"] + prediction_data["confidence_interval"]["lower"][::-1],
                fill="toself",
                fillcolor="rgba(0,176,0,0.2)",
                line={"color": "rgba(255,255,255,0)"},
                name="Confidence Interval"
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{prediction_data['symbol']} Price Prediction ({prediction_data['model_type']} model)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.1},
            template="plotly_dark"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error updating prediction graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": "Error generating prediction graph",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price (USD)"},
                "template": "plotly_dark"
            }
        }

# Callback to update historical graph
@app.callback(
    Output("historical-graph", "figure"),
    Input("historical-data-store", "children")
)
def update_historical_graph(historical_json):
    try:
        historical_data = json.loads(historical_json) if historical_json else {}
        
        if not historical_data:
            return {
                "data": [],
                "layout": {
                    "title": "No historical data available",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price (USD)"},
                    "template": "plotly_dark"
                }
            }
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data["dates"],
            y=historical_data["prices"],
            mode="lines",
            name="Historical Price",
            line={"color": "blue"}
        ))
        
        # Calculate moving averages
        prices = historical_data["prices"]
        dates = historical_data["dates"]
        
        # 7-day moving average
        ma7 = [sum(prices[max(0, i-6):i+1]) / min(i+1, 7) for i in range(len(prices))]
        
        # 14-day moving average
        ma14 = [sum(prices[max(0, i-13):i+1]) / min(i+1, 14) for i in range(len(prices))]
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma7,
            mode="lines",
            name="7-Day MA",
            line={"color": "orange"}
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma14,
            mode="lines",
            name="14-Day MA",
            line={"color": "red"}
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{historical_data['symbol']} Historical Price and Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.1},
            template="plotly_dark"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error updating historical graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": "Error generating historical graph",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price (USD)"},
                "template": "plotly_dark"
            }
        }

# Callback to update metrics container
@app.callback(
    Output("metrics-container", "children"),
    Input("models-data-store", "children"),
    Input("crypto-dropdown", "value"),
    Input("model-dropdown", "value")
)
def update_metrics_container(models_json, symbol, model_type):
    try:
        models_data = json.loads(models_json) if models_json else []
        
        if not models_data or not symbol:
            return html.Div("No model metrics available")
        
        # Filter models by symbol and model_type
        filtered_models = [
            model for model in models_data 
            if model["symbol"] == symbol and (not model_type or model["model_type"] == model_type)
        ]
        
        if not filtered_models:
            return html.Div("No metrics available for selected model")
        
        # Create metrics table
        metrics_components = []
        
        for model in filtered_models:
            metrics = model.get("metrics", {})
            
            if not metrics:
                continue
            
            # Create metrics card
            card = dbc.Card([
                dbc.CardHeader(f"{model['model_type'].upper()} Model Metrics"),
                dbc.CardBody([
                    html.Table([
                        html.Tr([html.Th("Metric"), html.Th("Value")], className="table-header"),
                        html.Tr([html.Td("RMSE"), html.Td(f"{metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else "N/A")]),
                        html.Tr([html.Td("MAPE"), html.Td(f"{metrics.get('mape', 'N/A'):.2f}%" if isinstance(metrics.get('mape'), (int, float)) else "N/A")]),
                        html.Tr([html.Td("MAE"), html.Td(f"{metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else "N/A")]),
                        html.Tr([html.Td("Directional Accuracy"), html.Td(f"{metrics.get('directional_accuracy', 'N/A'):.2f}%" if isinstance(metrics.get('directional_accuracy'), (int, float)) else "N/A")]),
                    ], className="metrics-table"),
                    html.Div(f"Last Trained: {model.get('last_trained', 'Unknown')}", className="text-muted mt-2")
                ])
            ], className="mb-3")
            
            metrics_components.append(card)
        
        if not metrics_components:
            return html.Div("No metrics available for selected model")
        
        return html.Div(metrics_components)
    except Exception as e:
        logger.error(f"Error updating metrics container: {e}")
        return html.Div("Error loading model metrics")

# Callback to update model comparison graph
@app.callback(
    Output("model-comparison-graph", "figure"),
    Input("models-data-store", "children"),
    Input("crypto-dropdown", "value")
)
def update_model_comparison_graph(models_json, symbol):
    try:
        models_data = json.loads(models_json) if models_json else []
        
        if not models_data or not symbol:
            return {
                "data": [],
                "layout": {
                    "title": "No model comparison data available",
                    "xaxis": {"title": "Model"},
                    "yaxis": {"title": "Metric Value"},
                    "template": "plotly_dark"
                }
            }
        
        # Filter models by symbol
        filtered_models = [model for model in models_data if model["symbol"] == symbol]
        
        if not filtered_models:
            return {
                "data": [],
                "layout": {
                    "title": f"No models available for {symbol}",
                    "xaxis": {"title": "Model"},
                    "yaxis": {"title": "Metric Value"},
                    "template": "plotly_dark"
                }
            }
        
        # Extract metrics
        model_names = []
        rmse_values = []
        mape_values = []
        dir_acc_values = []
        
        for model in filtered_models:
            metrics = model.get("metrics", {})
            
            if not metrics:
                continue
            
            model_names.append(model["model_type"].upper())
            rmse_values.append(metrics.get("rmse", None))
            mape_values.append(metrics.get("mape", None))
            dir_acc_values.append(metrics.get("directional_accuracy", None))
        
        if not model_names:
            return {
                "data": [],
                "layout": {
                    "title": f"No metrics available for {symbol} models",
                    "xaxis": {"title": "Model"},
                    "yaxis": {"title": "Metric Value"},
                    "template": "plotly_dark"
                }
            }
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("RMSE (lower is better)", "MAPE % (lower is better)", "Directional Accuracy % (higher is better)")
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_values, name="RMSE", marker_color="blue"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=mape_values, name="MAPE", marker_color="orange"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=dir_acc_values, name="Directional Accuracy", marker_color="green"),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Model Performance Comparison",
            showlegend=False,
            height=400,
            template="plotly_dark"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error updating model comparison graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": "Error generating model comparison graph",
                "xaxis": {"title": "Model"},
                "yaxis": {"title": "Metric Value"},
                "template": "plotly_dark"
            }
        }

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
            }
            .metrics-table th, .metrics-table td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .metrics-table .table-header {
                background-color: #333;
            }
            .dash-spinner {
                margin: 50px auto;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def run_dashboard(host='0.0.0.0', port=8050, debug=False):
    """Run the Dash dashboard."""
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Get dashboard configuration
    dashboard_config = config.get('visualization', {}).get('dashboard', {})
    host = dashboard_config.get('host', '0.0.0.0')
    port = dashboard_config.get('port', 8050)
    debug = dashboard_config.get('debug', False)
    
    # Run dashboard
    run_dashboard(host, port, debug)
