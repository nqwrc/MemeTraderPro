import ccxt
import pandas as pd
import time
import os
from datetime import datetime

class KuCoinAPI:
    """
    Class to handle interactions with the KuCoin exchange API using ccxt library
    """
    def __init__(self, api_key=None, api_secret=None, api_passphrase=None, dry_run=False):
        """
        Initialize the KuCoin API connection

        Parameters:
        api_key (str): KuCoin API key
        api_secret (str): KuCoin API secret
        api_passphrase (str): KuCoin API passphrase
        dry_run (bool): Whether to run in dry-run mode (no real trades)
        """
        # Use provided credentials or try to get from environment variables
        self.api_key = api_key or os.getenv("KUCOIN_API_KEY", "")
        self.api_secret = api_secret or os.getenv("KUCOIN_API_SECRET", "")
        self.api_passphrase = api_passphrase or os.getenv("KUCOIN_API_PASSPHRASE", "")
        
        # Dry run mode settings
        self.dry_run = dry_run
        self.dry_run_balance = 1000.0  # Default initial balance for dry run
        self.dry_run_positions = {}    # Track simulated positions {symbol: {amount, entry_price, timestamp}}
        
        # Initialize exchange connection
        self.exchange = None
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """
        Initialize the connection to KuCoin exchange
        """
        try:
            self.exchange = ccxt.kucoin({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_passphrase,
                'enableRateLimit': True,  # Important to avoid getting banned
                'options': {
                    'adjustForTimeDifference': True,  # Handle server time differences
                    'recvWindow': 60000,  # Set a longer receive window
                }
            })
            # Load markets to have symbol information available
            self.exchange.load_markets()
        except Exception as e:
            print(f"Error initializing KuCoin exchange: {str(e)}")
            self.exchange = None
    
    def test_connection(self):
        """
        Test the connection to KuCoin API
        
        Returns:
        bool: True if connection is successful, False otherwise
        """
        # In dry-run mode, always return success
        if self.dry_run:
            return True
            
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Try to fetch account balance as a test
            self.exchange.fetch_balance()
            return True
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False
    
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol
        
        Parameters:
        symbol (str): Trading pair symbol (e.g., 'DOGE/USDT')
        timeframe (str): Timeframe for candlestick data (e.g., '1m', '1h', '1d')
        limit (int): Number of candles to fetch
        
        Returns:
        list: List of OHLCV data or None if error
        """
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Handle rate limiting
            self.exchange.enableRateLimit = True
            
            # Fetch the OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to list format [timestamp, open, high, low, close, volume]
            return ohlcv
        except Exception as e:
            print(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            return None
    
    def get_current_price(self, symbol):
        """
        Get the current price for a symbol
        
        Parameters:
        symbol (str): Trading pair symbol (e.g., 'DOGE/USDT')
        
        Returns:
        float: Current price or None if error
        """
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Fetch ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Return last price
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    def get_balance(self, currency='USDT'):
        """
        Get available balance for a specific currency
        
        Parameters:
        currency (str): Currency to check balance for (default: 'USDT')
        
        Returns:
        float: Available balance or None if error
        """
        # In dry-run mode, return simulated balance
        if self.dry_run:
            # Calculate balance - subtract any allocated to positions
            allocated_funds = 0.0
            for symbol, position in self.dry_run_positions.items():
                if currency in symbol:  # Check if this position involves this currency
                    allocated_funds += position.get('cost', 0.0)
            
            return self.dry_run_balance - allocated_funds
            
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Fetch balance
            balance = self.exchange.fetch_balance()
            
            # Return available balance for the currency
            if currency in balance:
                return balance[currency]['free']
            return 0.0
        except Exception as e:
            print(f"Error fetching balance for {currency}: {str(e)}")
            return None
    
    def execute_trade(self, symbol, side, amount, price=None):
        """
        Execute a trade on KuCoin
        
        Parameters:
        symbol (str): Trading pair symbol (e.g., 'DOGE/USDT')
        side (str): Trade side ('buy' or 'sell')
        amount (float): Amount to trade
        price (float, optional): Price for limit order, None for market order
        
        Returns:
        dict: Order information or None if error
        """
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Make sure side is lowercase
            side = side.lower()
            
            # Check if side is valid
            if side not in ['buy', 'sell']:
                print(f"Invalid side: {side}. Must be 'buy' or 'sell'")
                return None
            
            # Determine order type
            order_type = 'limit' if price else 'market'
            
            # Create order parameters
            params = {}
            
            # Execute the order
            if order_type == 'limit':
                order = self.exchange.create_order(symbol, order_type, side, amount, price, params)
            else:
                order = self.exchange.create_order(symbol, order_type, side, amount, params=params)
            
            return order
        except Exception as e:
            print(f"Error executing {side} trade for {symbol}: {str(e)}")
            return None
    
    def get_memecoin_symbols(self):
        """
        Get a list of available memecoin trading pairs on KuCoin
        This is a simplified approach - in a real application, you might want to use a more dynamic approach
        
        Returns:
        list: List of memecoin symbols
        """
        # Common memecoins - this list can be expanded
        memecoins = [
            "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", 
            "BONK/USDT", "WIF/USDT", "MEME/USDT", "SPONGE/USDT"
        ]
        
        available_symbols = []
        
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Get all available symbols
            markets = self.exchange.load_markets()
            
            # Filter for available memecoins
            for symbol in memecoins:
                if symbol in markets:
                    available_symbols.append(symbol)
            
            return available_symbols
        except Exception as e:
            print(f"Error fetching memecoin symbols: {str(e)}")
            return memecoins  # Return default list if there's an error
    
    def check_order_status(self, order_id):
        """
        Check the status of an order
        
        Parameters:
        order_id (str): ID of the order to check
        
        Returns:
        dict: Order information or None if error
        """
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Fetch order status
            order = self.exchange.fetch_order(order_id)
            
            return order
        except Exception as e:
            print(f"Error checking order status for {order_id}: {str(e)}")
            return None
    
    def cancel_order(self, order_id, symbol=None):
        """
        Cancel an open order
        
        Parameters:
        order_id (str): ID of the order to cancel
        symbol (str, optional): Trading pair symbol (required for some exchanges)
        
        Returns:
        dict: Cancellation result or None if error
        """
        try:
            if not self.exchange:
                self.initialize_exchange()
            
            # Cancel order
            result = self.exchange.cancel_order(order_id, symbol)
            
            return result
        except Exception as e:
            print(f"Error canceling order {order_id}: {str(e)}")
            return None

if __name__ == "__main__":
    # Simple test code to verify functionality
    api_key = os.getenv("KUCOIN_API_KEY", "")
    api_secret = os.getenv("KUCOIN_API_SECRET", "")
    api_passphrase = os.getenv("KUCOIN_API_PASSPHRASE", "")
    
    kucoin = KuCoinAPI(api_key, api_secret, api_passphrase)
    
    if kucoin.test_connection():
        print("Connection successful!")
        
        # Fetch DOGE/USDT OHLCV data
        ohlcv = kucoin.fetch_ohlcv("DOGE/USDT", "1h", 10)
        if ohlcv:
            print("OHLCV data:")
            for candle in ohlcv:
                timestamp = datetime.fromtimestamp(candle[0]/1000).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{timestamp}: Open: {candle[1]}, High: {candle[2]}, Low: {candle[3]}, Close: {candle[4]}, Volume: {candle[5]}")
        
        # Get current price for DOGE/USDT
        price = kucoin.get_current_price("DOGE/USDT")
        if price:
            print(f"Current DOGE/USDT price: {price}")
        
        # Get USDT balance
        balance = kucoin.get_balance()
        if balance is not None:
            print(f"USDT balance: {balance}")
    else:
        print("Connection failed!")
