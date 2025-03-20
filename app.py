import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
from datetime import datetime, timedelta
import joblib
import numpy as np
from threading import Thread
import traceback

import kucoin_api as kapi
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from trading_logic import TradingLogic
import utils

# Set page config and title
st.set_page_config(
    page_title="MemeCoin Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=['timestamp', 'symbol', 'type', 'price', 'amount', 'cost', 'profit_loss'])
if 'bot_thread' not in st.session_state:
    st.session_state.bot_thread = None
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = {'total_trades': 0, 'profitable_trades': 0, 'total_profit_loss': 0.0, 'win_rate': 0.0}
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'current_positions' not in st.session_state:
    st.session_state.current_positions = {}

# Function to log messages
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(f"[{level}] {timestamp} - {message}")
    # Keep only last 100 messages
    st.session_state.log_messages = st.session_state.log_messages[-100:]

# Trading bot function that runs in a separate thread
def run_trading_bot(symbols, api_key, api_secret, api_passphrase, timeframe, initial_balance, 
                   risk_per_trade, stop_loss_pct, take_profit_pct, dry_run=True):
    try:
        log_message("Initializing trading bot...")
        
        # Initialize components
        kucoin = kapi.KuCoinAPI(api_key, api_secret, api_passphrase, dry_run=dry_run)
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        trading_logic = TradingLogic(risk_per_trade, stop_loss_pct, take_profit_pct)
        
        if dry_run:
            log_message("Running in DRY RUN mode - no real trades will be executed")
        
        # Check if API connection works
        if not kucoin.test_connection():
            log_message("Failed to connect to KuCoin API. Please check your credentials.", "ERROR")
            st.session_state.bot_running = False
            return
        
        log_message(f"Successfully connected to KuCoin API")
        
        # Trading loop
        while st.session_state.bot_running:
            try:
                for symbol in symbols:
                    # Fetch latest price data
                    ohlcv_data = kucoin.fetch_ohlcv(symbol, timeframe)
                    if ohlcv_data is None or len(ohlcv_data) == 0:
                        log_message(f"No data received for {symbol}. Skipping.", "WARNING")
                        continue
                    
                    # Process data
                    df = data_processor.process_ohlcv_data(ohlcv_data)
                    
                    # Check if we have enough data
                    if len(df) < 100:  # Need enough data for features and training
                        log_message(f"Not enough historical data for {symbol}. Skipping.", "WARNING")
                        continue
                    
                    # Prepare features
                    features_df = data_processor.create_features(df)
                    
                    # Check if model exists, if not create one
                    model_path = f"models/{symbol.replace('/', '_')}_model.joblib"
                    if not os.path.exists(model_path):
                        log_message(f"Training new model for {symbol}...")
                        model, metrics = model_trainer.train_model(features_df)
                        joblib.dump(model, model_path)
                        st.session_state.model_metrics[symbol] = metrics
                        log_message(f"Model trained for {symbol} with accuracy: {metrics['accuracy']:.4f}")
                    else:
                        # Periodically retrain the model (e.g., every day)
                        model_stats = os.stat(model_path)
                        last_modified = datetime.fromtimestamp(model_stats.st_mtime)
                        if datetime.now() - last_modified > timedelta(days=1):
                            log_message(f"Retraining model for {symbol}...")
                            model, metrics = model_trainer.train_model(features_df)
                            joblib.dump(model, model_path)
                            st.session_state.model_metrics[symbol] = metrics
                            log_message(f"Model retrained for {symbol} with accuracy: {metrics['accuracy']:.4f}")
                        else:
                            model = joblib.load(model_path)
                    
                    # Get prediction for the latest data point
                    latest_features = features_df.iloc[-1:].drop(['target'], axis=1, errors='ignore')
                    prediction = model.predict(latest_features)[0]
                    prediction_proba = model.predict_proba(latest_features)[0]
                    confidence = max(prediction_proba)
                    
                    # Current market price
                    current_price = df.iloc[-1]['close']
                    
                    # Check if we have an open position for this symbol
                    position = st.session_state.current_positions.get(symbol, None)
                    
                    # Execute trading logic
                    action, amount, reason = trading_logic.decide_action(
                        symbol, prediction, confidence, current_price, position, kucoin.get_balance()
                    )
                    
                    # If action is to do something
                    if action != 'hold':
                        log_message(f"Signal for {symbol}: {action.upper()} at {current_price} ({reason})")
                        
                        # Execute trade
                        trade_result = kucoin.execute_trade(symbol, action, amount, current_price)
                        
                        if trade_result:
                            # Update trade history
                            trade_record = {
                                'timestamp': datetime.now(),
                                'symbol': symbol,
                                'type': action,
                                'price': current_price,
                                'amount': amount,
                                'cost': amount * current_price
                            }
                            
                            # Calculate profit/loss if it's a sell
                            if action == 'sell' and position:
                                profit_loss = (current_price - position['entry_price']) * amount
                                trade_record['profit_loss'] = profit_loss
                                
                                # Update performance metrics
                                st.session_state.performance_data['total_trades'] += 1
                                st.session_state.performance_data['total_profit_loss'] += profit_loss
                                if profit_loss > 0:
                                    st.session_state.performance_data['profitable_trades'] += 1
                                
                                # Calculate win rate
                                if st.session_state.performance_data['total_trades'] > 0:
                                    st.session_state.performance_data['win_rate'] = (
                                        st.session_state.performance_data['profitable_trades'] / 
                                        st.session_state.performance_data['total_trades'] * 100
                                    )
                                
                                # Remove position
                                st.session_state.current_positions.pop(symbol, None)
                            
                            # If it's a buy, record the position
                            elif action == 'buy':
                                st.session_state.current_positions[symbol] = {
                                    'entry_price': current_price,
                                    'amount': amount,
                                    'timestamp': datetime.now()
                                }
                                trade_record['profit_loss'] = 0
                            
                            # Add to trade history
                            new_trade_df = pd.DataFrame([trade_record])
                            st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade_df], ignore_index=True)
                            
                            log_message(f"Executed {action} for {symbol}: {amount} at {current_price}")
                        else:
                            log_message(f"Failed to execute {action} for {symbol}", "ERROR")
                
                # Sleep to avoid hitting API rate limits
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                log_message(f"Error in trading loop: {str(e)}", "ERROR")
                log_message(traceback.format_exc(), "ERROR")
                time.sleep(60)  # Wait a bit before retrying
    
    except Exception as e:
        log_message(f"Critical error in trading bot: {str(e)}", "ERROR")
        log_message(traceback.format_exc(), "ERROR")
        st.session_state.bot_running = False

# Start/Stop the trading bot
def toggle_bot():
    if st.session_state.bot_running:
        st.session_state.bot_running = False
        log_message("Stopping trading bot...")
        if st.session_state.bot_thread and st.session_state.bot_thread.is_alive():
            st.session_state.bot_thread.join(timeout=1)
        log_message("Trading bot stopped")
    else:
        st.session_state.bot_running = True
        log_message("Starting trading bot...")
        # Get parameters from session state
        symbols = st.session_state.selected_symbols
        api_key = st.session_state.api_key
        api_secret = st.session_state.api_secret
        api_passphrase = st.session_state.api_passphrase
        timeframe = st.session_state.timeframe
        initial_balance = st.session_state.initial_balance
        risk_per_trade = st.session_state.risk_per_trade
        stop_loss_pct = st.session_state.stop_loss_pct
        take_profit_pct = st.session_state.take_profit_pct
        
        # Get dry run setting
        dry_run = st.session_state.dry_run
        
        # Start the bot in a new thread
        st.session_state.bot_thread = Thread(
            target=run_trading_bot,
            args=(symbols, api_key, api_secret, api_passphrase, timeframe, 
                  initial_balance, risk_per_trade, stop_loss_pct, take_profit_pct, dry_run)
        )
        st.session_state.bot_thread.daemon = True  # Set as daemon so it terminates when main thread ends
        st.session_state.bot_thread.start()

# Main app layout
st.title("MemeCoin Trading Bot")

tabs = st.tabs(["Dashboard", "Settings", "Logs", "Performance", "Models"])

# Dashboard Tab
with tabs[0]:
    st.header("Trading Dashboard")
    
    # Status and controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        status = "🟢 Running" if st.session_state.bot_running else "🔴 Stopped"
        st.metric("Bot Status", status)
    
    with col2:
        button_label = "Stop Bot" if st.session_state.bot_running else "Start Bot"
        button_color = "primary" if st.session_state.bot_running else "primary"
        st.button(button_label, on_click=toggle_bot, use_container_width=True, type=button_color)
    
    with col3:
        if st.session_state.performance_data['total_trades'] > 0:
            pl_value = f"${st.session_state.performance_data['total_profit_loss']:.2f}"
            st.metric("Total P/L", pl_value)
        else:
            st.metric("Total P/L", "$0.00")
    
    # Current positions
    st.subheader("Current Positions")
    if st.session_state.current_positions:
        positions_data = []
        for symbol, pos in st.session_state.current_positions.items():
            # Try to get current price
            try:
                kucoin = kapi.KuCoinAPI(
                    st.session_state.get('api_key', ''),
                    st.session_state.get('api_secret', ''),
                    st.session_state.get('api_passphrase', ''),
                    dry_run=st.session_state.get('dry_run', True)
                )
                current_price = kucoin.get_current_price(symbol)
                if current_price:
                    unrealized_pl = (current_price - pos['entry_price']) * pos['amount']
                    pl_pct = (current_price / pos['entry_price'] - 1) * 100
                else:
                    unrealized_pl = 0
                    pl_pct = 0
            except:
                current_price = None
                unrealized_pl = 0
                pl_pct = 0
            
            positions_data.append({
                'Symbol': symbol,
                'Entry Price': f"${pos['entry_price']:.4f}",
                'Amount': f"{pos['amount']:.6f}",
                'Current Price': f"${current_price:.4f}" if current_price else "Unknown",
                'Unrealized P/L': f"${unrealized_pl:.2f}",
                'P/L %': f"{pl_pct:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    else:
        st.info("No open positions")
    
    # Recent trades
    st.subheader("Recent Trades")
    if not st.session_state.trade_history.empty:
        recent_trades = st.session_state.trade_history.tail(10)
        # Format for display
        display_trades = recent_trades.copy()
        display_trades['timestamp'] = pd.to_datetime(display_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_trades['price'] = display_trades['price'].apply(lambda x: f"${x:.4f}")
        display_trades['amount'] = display_trades['amount'].apply(lambda x: f"{x:.6f}")
        display_trades['cost'] = display_trades['cost'].apply(lambda x: f"${x:.2f}")
        display_trades['profit_loss'] = display_trades['profit_loss'].apply(lambda x: f"${x:.2f}")
        # Highlight based on type (buy/sell)
        display_trades['type'] = display_trades['type'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "buy" else f"🔴 {x.upper()}"
        )
        
        st.dataframe(display_trades, use_container_width=True)
    else:
        st.info("No trade history yet")
    
    # Price charts for selected symbols
    st.subheader("Price Charts")
    
    if 'selected_symbols' in st.session_state and st.session_state.selected_symbols:
        for symbol in st.session_state.selected_symbols:
            try:
                # Fetch data for chart
                kucoin = kapi.KuCoinAPI(
                    st.session_state.get('api_key', ''),
                    st.session_state.get('api_secret', ''),
                    st.session_state.get('api_passphrase', ''),
                    dry_run=st.session_state.get('dry_run', True)
                )
                ohlcv_data = kucoin.fetch_ohlcv(symbol, '1h', limit=100)
                if ohlcv_data and len(ohlcv_data) > 0:
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Create candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=symbol
                    )])
                    
                    # Add volume as bar chart
                    fig.add_trace(go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name='Volume',
                        marker_color='rgba(128,128,128,0.5)',
                        yaxis='y2'
                    ))
                    
                    # Check if we have trades for this symbol and add markers
                    symbol_trades = st.session_state.trade_history[st.session_state.trade_history['symbol'] == symbol]
                    if not symbol_trades.empty:
                        for idx, trade in symbol_trades.iterrows():
                            color = 'green' if trade['type'] == 'buy' else 'red'
                            fig.add_trace(go.Scatter(
                                x=[pd.to_datetime(trade['timestamp'])],
                                y=[trade['price']],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=color,
                                    symbol='triangle-up' if trade['type'] == 'buy' else 'triangle-down'
                                ),
                                name=f"{trade['type'].upper()} {trade['amount']:.4f}"
                            ))
                    
                    # Layout
                    fig.update_layout(
                        title=f"{symbol} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        yaxis2=dict(
                            title="Volume",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        ),
                        height=500,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {symbol}")
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
    else:
        st.info("No symbols selected. Please configure symbols in Settings tab.")

# Settings Tab
with tabs[1]:
    st.header("Bot Configuration")
    
    # API Configuration
    st.subheader("KuCoin API Settings")
    
    # Store API credentials in session state
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv("KUCOIN_API_KEY", "")
    if 'api_secret' not in st.session_state:
        st.session_state.api_secret = os.getenv("KUCOIN_API_SECRET", "")
    if 'api_passphrase' not in st.session_state:
        st.session_state.api_passphrase = os.getenv("KUCOIN_API_PASSPHRASE", "")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
        st.session_state.api_key = api_key
        
        api_secret = st.text_input("API Secret", value=st.session_state.api_secret, type="password")
        st.session_state.api_secret = api_secret
    
    with col2:
        api_passphrase = st.text_input("API Passphrase", value=st.session_state.api_passphrase, type="password")
        st.session_state.api_passphrase = api_passphrase
        
        # Test API connection
        if st.button("Test API Connection"):
            try:
                # Get current dry run setting
                dry_run = st.session_state.get('dry_run', True)
                
                kucoin = kapi.KuCoinAPI(api_key, api_secret, api_passphrase, dry_run=dry_run)
                if kucoin.test_connection():
                    st.success("Connection successful!")
                    # Get and display available balance
                    balance = kucoin.get_balance()
                    if balance:
                        st.info(f"Available USDT balance: ${balance:.2f}")
                else:
                    st.error("Connection failed. Check your credentials.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Trading Settings
    st.subheader("Trading Settings")
    
    # Initialize session state for trading settings
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT"]
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = "1h"
    if 'initial_balance' not in st.session_state:
        st.session_state.initial_balance = 1000.0
    if 'risk_per_trade' not in st.session_state:
        st.session_state.risk_per_trade = 0.02
    if 'stop_loss_pct' not in st.session_state:
        st.session_state.stop_loss_pct = 0.05
    if 'take_profit_pct' not in st.session_state:
        st.session_state.take_profit_pct = 0.15
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Default memecoins
        default_memecoins = ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", 
                            "BONK/USDT", "WIF/USDT", "MEME/USDT", "SPONGE/USDT"]
        
        selected_symbols = st.multiselect(
            "Select Memecoins", 
            options=default_memecoins, 
            default=st.session_state.selected_symbols,
            help="Select the memecoin pairs you want to trade"
        )
        st.session_state.selected_symbols = selected_symbols
        
        timeframe = st.selectbox(
            "Timeframe",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=4,  # Default to 1h
            help="The timeframe for candlestick data"
        )
        st.session_state.timeframe = timeframe
    
    with col2:
        initial_balance = st.number_input(
            "Initial Balance (USDT)",
            min_value=10.0,
            max_value=10000.0,
            value=st.session_state.initial_balance,
            step=10.0,
            help="Your initial trading balance"
        )
        st.session_state.initial_balance = initial_balance
        
        risk_per_trade = st.slider(
            "Risk Per Trade (%)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.risk_per_trade * 100),
            step=0.5,
            help="Percentage of balance to risk on each trade"
        ) / 100
        st.session_state.risk_per_trade = risk_per_trade
    
    # Risk Management
    st.subheader("Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stop_loss_pct = st.slider(
            "Stop Loss (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(st.session_state.stop_loss_pct * 100),
            step=0.5,
            help="Percentage below entry price to set stop loss"
        ) / 100
        st.session_state.stop_loss_pct = stop_loss_pct
    
    with col2:
        take_profit_pct = st.slider(
            "Take Profit (%)",
            min_value=2.0,
            max_value=50.0,
            value=float(st.session_state.take_profit_pct * 100),
            step=1.0,
            help="Percentage above entry price to take profit"
        ) / 100
        st.session_state.take_profit_pct = take_profit_pct
    
    # Dry Run Mode
    st.subheader("Operation Mode")
    
    # Initialize dry run mode in session state if not exists
    if 'dry_run' not in st.session_state:
        st.session_state.dry_run = True  # Default to dry run mode enabled
    
    dry_run = st.checkbox("Dry Run Mode (Simulated trading without real money)", 
                         value=st.session_state.dry_run,
                         help="When enabled, the bot will simulate trades without using real funds")
    st.session_state.dry_run = dry_run
    
    if dry_run:
        st.info("📢 Dry run mode is enabled. The bot will simulate trades without using real funds. Great for testing strategies!")
    else:
        st.warning("⚠️ Dry run mode is disabled. The bot will use REAL FUNDS for trading. Make sure your API credentials are correct.")
    
    # Save settings
    if st.button("Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")
        log_message("Bot settings updated")

# Logs Tab
with tabs[2]:
    st.header("Bot Logs")
    
    # Filter options
    log_level = st.selectbox(
        "Filter by Level",
        options=["ALL", "INFO", "WARNING", "ERROR"],
        index=0
    )
    
    # Display logs with filter
    st.subheader("Log Messages")
    
    log_container = st.container()
    with log_container:
        filtered_logs = []
        for log in st.session_state.log_messages:
            if log_level == "ALL" or log_level in log:
                filtered_logs.append(log)
        
        if filtered_logs:
            for log in reversed(filtered_logs):  # Show newest first
                if "ERROR" in log:
                    st.error(log)
                elif "WARNING" in log:
                    st.warning(log)
                else:
                    st.info(log)
        else:
            st.info("No logs to display")
    
    # Clear logs button
    if st.button("Clear Logs"):
        st.session_state.log_messages = []
        st.rerun()

# Performance Tab
with tabs[3]:
    st.header("Trading Performance")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", st.session_state.performance_data['total_trades'])
    
    with col2:
        st.metric("Win Rate", f"{st.session_state.performance_data['win_rate']:.2f}%")
    
    with col3:
        st.metric("Profitable Trades", st.session_state.performance_data['profitable_trades'])
    
    with col4:
        pl_value = f"${st.session_state.performance_data['total_profit_loss']:.2f}"
        st.metric("Total P/L", pl_value)
    
    # Trade history table
    st.subheader("Trade History")
    if not st.session_state.trade_history.empty:
        # Format for display
        display_trades = st.session_state.trade_history.copy()
        display_trades['timestamp'] = pd.to_datetime(display_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_trades['price'] = display_trades['price'].apply(lambda x: f"${x:.4f}")
        display_trades['amount'] = display_trades['amount'].apply(lambda x: f"{x:.6f}")
        display_trades['cost'] = display_trades['cost'].apply(lambda x: f"${x:.2f}")
        display_trades['profit_loss'] = display_trades['profit_loss'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_trades, use_container_width=True)
        
        # Allow downloading trade history as CSV
        if st.button("Download Trade History as CSV"):
            csv_data = st.session_state.trade_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Performance Charts
        st.subheader("Performance Charts")
        
        # Create cumulative P/L chart
        if 'profit_loss' in st.session_state.trade_history.columns and not st.session_state.trade_history.empty:
            profit_history = st.session_state.trade_history.copy()
            profit_history['timestamp'] = pd.to_datetime(profit_history['timestamp'])
            profit_history = profit_history.sort_values('timestamp')
            profit_history['cumulative_pl'] = profit_history['profit_loss'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=profit_history['timestamp'],
                y=profit_history['cumulative_pl'],
                mode='lines+markers',
                name='Cumulative P/L',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Profit/Loss Over Time",
                xaxis_title="Date",
                yaxis_title="Profit/Loss (USDT)",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # P/L by symbol
            symbol_pl = st.session_state.trade_history.groupby('symbol')['profit_loss'].sum().reset_index()
            symbol_pl = symbol_pl.sort_values('profit_loss', ascending=False)
            
            fig = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in symbol_pl['profit_loss']]
            
            fig.add_trace(go.Bar(
                x=symbol_pl['symbol'],
                y=symbol_pl['profit_loss'],
                marker_color=colors,
                name='P/L by Symbol'
            ))
            
            fig.update_layout(
                title="Profit/Loss by Symbol",
                xaxis_title="Symbol",
                yaxis_title="Profit/Loss (USDT)",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trade history available yet")

# Models Tab
with tabs[4]:
    st.header("Predictive Models")
    
    # Model metrics display
    if st.session_state.model_metrics:
        st.subheader("Model Metrics")
        
        # Convert metrics to dataframe for display
        metrics_data = []
        for symbol, metrics in st.session_state.model_metrics.items():
            metrics_data.append({
                'Symbol': symbol,
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision', 0):.4f}",
                'Recall': f"{metrics.get('recall', 0):.4f}",
                'F1 Score': f"{metrics.get('f1', 0):.4f}",
                'Training Date': metrics.get('training_date', 'Unknown')
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Feature importance if available
        st.subheader("Feature Importance")
        
        # Check for saved models and display feature importance
        if 'selected_symbols' in st.session_state and st.session_state.selected_symbols:
            for symbol in st.session_state.selected_symbols:
                model_path = f"models/{symbol.replace('/', '_')}_model.joblib"
                if os.path.exists(model_path):
                    try:
                        model = joblib.load(model_path)
                        # Check if model has feature_importances_ attribute (like RandomForest)
                        if hasattr(model, 'feature_importances_'):
                            # Get features from model
                            data_processor = DataProcessor()
                            feature_names = data_processor.get_feature_names()
                            
                            importances = pd.DataFrame({
                                'Feature': feature_names[:len(model.feature_importances_)],
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=importances['Feature'],
                                y=importances['Importance'],
                                marker_color='blue',
                                name='Feature Importance'
                            ))
                            
                            fig.update_layout(
                                title=f"Feature Importance for {symbol} Model",
                                xaxis_title="Feature",
                                yaxis_title="Importance",
                                height=400,
                                margin=dict(l=50, r=50, t=50, b=50)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"Feature importance not available for {symbol} model type")
                    except Exception as e:
                        st.error(f"Error loading model for {symbol}: {str(e)}")
                else:
                    st.info(f"No trained model found for {symbol}")
    else:
        st.info("No model metrics available yet. Models will be trained when the bot starts running.")
    
    # Model Management
    st.subheader("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Retrain All Models", use_container_width=True):
            if 'selected_symbols' in st.session_state and st.session_state.selected_symbols:
                try:
                    for symbol in st.session_state.selected_symbols:
                        log_message(f"Manually triggering retraining for {symbol}...")
                        
                        # Initialize components
                        kucoin = kapi.KuCoinAPI(
                            st.session_state.get('api_key', ''),
                            st.session_state.get('api_secret', ''),
                            st.session_state.get('api_passphrase', ''),
                            dry_run=st.session_state.get('dry_run', True)
                        )
                        data_processor = DataProcessor()
                        model_trainer = ModelTrainer()
                        
                        # Fetch latest price data
                        ohlcv_data = kucoin.fetch_ohlcv(symbol, st.session_state.timeframe, limit=500)
                        if ohlcv_data and len(ohlcv_data) > 0:
                            # Process data
                            df = data_processor.process_ohlcv_data(ohlcv_data)
                            
                            # Check if we have enough data
                            if len(df) >= 100:  # Need enough data for features and training
                                # Prepare features
                                features_df = data_processor.create_features(df)
                                
                                # Train model
                                model, metrics = model_trainer.train_model(features_df)
                                
                                # Save model
                                model_path = f"models/{symbol.replace('/', '_')}_model.joblib"
                                joblib.dump(model, model_path)
                                
                                # Update metrics
                                st.session_state.model_metrics[symbol] = metrics
                                
                                log_message(f"Model retrained for {symbol} with accuracy: {metrics['accuracy']:.4f}")
                            else:
                                log_message(f"Not enough data to train model for {symbol}", "WARNING")
                        else:
                            log_message(f"Could not fetch data for {symbol}", "ERROR")
                    
                    st.success("All models retrained successfully!")
                    st.rerun()
                except Exception as e:
                    log_message(f"Error retraining models: {str(e)}", "ERROR")
                    st.error(f"Error retraining models: {str(e)}")
            else:
                st.warning("No symbols selected. Please configure symbols in Settings tab.")
    
    with col2:
        if st.button("Delete All Models", use_container_width=True):
            try:
                # Check if models directory exists
                if os.path.exists("models"):
                    # Get all model files
                    model_files = [f for f in os.listdir("models") if f.endswith("_model.joblib")]
                    
                    if model_files:
                        for model_file in model_files:
                            os.remove(os.path.join("models", model_file))
                        
                        # Clear model metrics
                        st.session_state.model_metrics = {}
                        
                        log_message(f"Deleted {len(model_files)} models")
                        st.success(f"Successfully deleted {len(model_files)} models")
                        st.rerun()
                    else:
                        st.info("No models found to delete")
                else:
                    st.info("No models directory found")
            except Exception as e:
                log_message(f"Error deleting models: {str(e)}", "ERROR")
                st.error(f"Error deleting models: {str(e)}")

# Add a footer
st.markdown("---")
st.markdown("MemeCoin Trading Bot | Built with Streamlit | Not financial advice")

# Initialize with a welcome message if first run
if len(st.session_state.log_messages) == 0:
    log_message("Welcome to MemeCoin Trading Bot! Configure your settings and start the bot to begin trading.")
