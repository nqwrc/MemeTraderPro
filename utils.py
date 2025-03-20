import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, create it if it doesn't
    
    Parameters:
    directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json_data(data, filename, directory='data'):
    """
    Save data to a JSON file
    
    Parameters:
    data (dict): Data to save
    filename (str): Filename
    directory (str): Directory to save to
    
    Returns:
    str: Full path to the saved file
    """
    ensure_directory_exists(directory)
    
    # Make sure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Full path
    full_path = os.path.join(directory, filename)
    
    # Save data
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return full_path

def load_json_data(filename, directory='data'):
    """
    Load data from a JSON file
    
    Parameters:
    filename (str): Filename
    directory (str): Directory to load from
    
    Returns:
    dict: Loaded data or None if file doesn't exist
    """
    # Make sure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Full path
    full_path = os.path.join(directory, filename)
    
    # Check if file exists
    if not os.path.exists(full_path):
        return None
    
    # Load data
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    return data

def save_dataframe(df, filename, directory='data'):
    """
    Save a DataFrame to a CSV file
    
    Parameters:
    df (pd.DataFrame): DataFrame to save
    filename (str): Filename
    directory (str): Directory to save to
    
    Returns:
    str: Full path to the saved file
    """
    ensure_directory_exists(directory)
    
    # Make sure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Full path
    full_path = os.path.join(directory, filename)
    
    # Save data
    df.to_csv(full_path, index=True)
    
    return full_path

def load_dataframe(filename, directory='data'):
    """
    Load a DataFrame from a CSV file
    
    Parameters:
    filename (str): Filename
    directory (str): Directory to load from
    
    Returns:
    pd.DataFrame: Loaded DataFrame or None if file doesn't exist
    """
    # Make sure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Full path
    full_path = os.path.join(directory, filename)
    
    # Check if file exists
    if not os.path.exists(full_path):
        return None
    
    # Load data
    df = pd.read_csv(full_path, index_col=0)
    
    return df

def calculate_performance_metrics(trade_history):
    """
    Calculate performance metrics from trade history
    
    Parameters:
    trade_history (pd.DataFrame): Trade history DataFrame
    
    Returns:
    dict: Performance metrics
    """
    # Initialize metrics
    metrics = {
        'total_trades': 0,
        'profitable_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'total_profit_loss': 0.0,
        'avg_profit_per_trade': 0.0,
        'avg_profit_per_winning_trade': 0.0,
        'avg_loss_per_losing_trade': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0
    }
    
    # Check if trade history is empty
    if trade_history.empty:
        return metrics
    
    # Make sure we have profit_loss column
    if 'profit_loss' not in trade_history.columns:
        return metrics
    
    # Calculate metrics
    metrics['total_trades'] = len(trade_history)
    
    profitable_trades = trade_history[trade_history['profit_loss'] > 0]
    losing_trades = trade_history[trade_history['profit_loss'] < 0]
    
    metrics['profitable_trades'] = len(profitable_trades)
    metrics['losing_trades'] = len(losing_trades)
    
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades'] * 100
    
    metrics['total_profit_loss'] = trade_history['profit_loss'].sum()
    
    if metrics['total_trades'] > 0:
        metrics['avg_profit_per_trade'] = metrics['total_profit_loss'] / metrics['total_trades']
    
    if metrics['profitable_trades'] > 0:
        metrics['avg_profit_per_winning_trade'] = profitable_trades['profit_loss'].sum() / metrics['profitable_trades']
    
    if metrics['losing_trades'] > 0:
        metrics['avg_loss_per_losing_trade'] = losing_trades['profit_loss'].sum() / metrics['losing_trades']
    
    # Calculate profit factor (gross profit / gross loss)
    gross_profit = profitable_trades['profit_loss'].sum()
    gross_loss = abs(losing_trades['profit_loss'].sum())
    
    if gross_loss > 0:
        metrics['profit_factor'] = gross_profit / gross_loss
    
    # Calculate maximum drawdown
    if not trade_history.empty:
        trade_history_sorted = trade_history.sort_values('timestamp')
        cumulative_pl = trade_history_sorted['profit_loss'].cumsum()
        
        # Calculate running maximum
        running_max = cumulative_pl.cummax()
        
        # Calculate drawdown
        drawdown = running_max - cumulative_pl
        
        # Maximum drawdown
        metrics['max_drawdown'] = drawdown.max()
    
    return metrics

def create_performance_charts(trade_history):
    """
    Create performance charts from trade history
    
    Parameters:
    trade_history (pd.DataFrame): Trade history DataFrame
    
    Returns:
    dict: Dictionary containing Plotly figure objects
    """
    charts = {}
    
    # Check if trade history is empty
    if trade_history.empty:
        return charts
    
    # Make sure we have profit_loss column
    if 'profit_loss' not in trade_history.columns:
        return charts
    
    # Sort trade history by timestamp
    trade_history_sorted = trade_history.copy()
    trade_history_sorted['timestamp'] = pd.to_datetime(trade_history_sorted['timestamp'])
    trade_history_sorted = trade_history_sorted.sort_values('timestamp')
    
    # Cumulative P/L chart
    cumulative_pl = trade_history_sorted['profit_loss'].cumsum()
    
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=trade_history_sorted['timestamp'],
        y=cumulative_pl,
        mode='lines',
        name='Cumulative P/L'
    ))
    
    fig_cumulative.update_layout(
        title='Cumulative Profit/Loss Over Time',
        xaxis_title='Date',
        yaxis_title='Profit/Loss',
        hovermode='x unified'
    )
    
    charts['cumulative_pl'] = fig_cumulative
    
    # P/L by symbol
    symbol_pl = trade_history.groupby('symbol')['profit_loss'].sum().reset_index()
    symbol_pl = symbol_pl.sort_values('profit_loss', ascending=False)
    
    fig_symbol = go.Figure()
    colors = ['green' if x > 0 else 'red' for x in symbol_pl['profit_loss']]
    
    fig_symbol.add_trace(go.Bar(
        x=symbol_pl['symbol'],
        y=symbol_pl['profit_loss'],
        marker_color=colors,
        name='P/L by Symbol'
    ))
    
    fig_symbol.update_layout(
        title='Profit/Loss by Symbol',
        xaxis_title='Symbol',
        yaxis_title='Profit/Loss',
        hovermode='x unified'
    )
    
    charts['symbol_pl'] = fig_symbol
    
    return charts

def format_number(number, precision=2):
    """
    Format a number with commas and specified precision
    
    Parameters:
    number (float): Number to format
    precision (int): Decimal precision
    
    Returns:
    str: Formatted number
    """
    return f"{number:,.{precision}f}"

def format_percentage(number):
    """
    Format a number as percentage
    
    Parameters:
    number (float): Number to format
    
    Returns:
    str: Formatted percentage
    """
    return f"{number:.2f}%"

def get_timestamp_str():
    """
    Get current timestamp as a string
    
    Returns:
    str: Current timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def estimate_transaction_fees(price, amount, fee_rate=0.001):
    """
    Estimate transaction fees
    
    Parameters:
    price (float): Price
    amount (float): Amount
    fee_rate (float): Fee rate (default: 0.1%)
    
    Returns:
    float: Estimated transaction fee
    """
    return price * amount * fee_rate

def validate_api_credentials(api_key, api_secret, api_passphrase):
    """
    Validate API credentials format
    
    Parameters:
    api_key (str): API key
    api_secret (str): API secret
    api_passphrase (str): API passphrase
    
    Returns:
    tuple: (is_valid, message)
    """
    if not api_key or len(api_key) < 10:
        return False, "API Key is missing or too short"
    
    if not api_secret or len(api_secret) < 10:
        return False, "API Secret is missing or too short"
    
    if not api_passphrase:
        return False, "API Passphrase is missing"
    
    return True, "Credentials format is valid"

if __name__ == "__main__":
    # Example usage
    print("Timestamp:", get_timestamp_str())
    print("Formatted number:", format_number(1234567.89))
    print("Formatted percentage:", format_percentage(12.345))
    
    # Test validate_api_credentials
    valid, message = validate_api_credentials("abcdefghijk", "1234567890abcdefg", "pass123")
    print(f"API credentials valid: {valid}, Message: {message}")
    
    # Test directory creation
    ensure_directory_exists("test_dir")
    print("Directory created:", os.path.exists("test_dir"))
    
    # Clean up
    if os.path.exists("test_dir"):
        os.rmdir("test_dir")
