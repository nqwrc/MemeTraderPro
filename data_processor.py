import pandas as pd
import numpy as np
from datetime import datetime
import talib

class DataProcessor:
    """
    Class to handle data processing for the trading bot
    """
    def __init__(self):
        """
        Initialize data processor
        """
        # Define column names for OHLCV data
        self.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Define parameters for technical indicators
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.ema_short = 9
        self.ema_medium = 21
        self.ema_long = 50
        
        # Define target parameters
        self.prediction_horizon = 12  # Predict price movement 12 periods ahead
        self.prediction_threshold = 0.015  # 1.5% price movement threshold
    
    def process_ohlcv_data(self, ohlcv_data):
        """
        Process raw OHLCV data into a pandas DataFrame
        
        Parameters:
        ohlcv_data (list): Raw OHLCV data from exchange API
        
        Returns:
        pd.DataFrame: Processed DataFrame with OHLCV data
        """
        # Convert list of lists to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=self.columns)
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def create_features(self, df):
        """
        Create technical features from OHLCV data
        
        Parameters:
        df (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
        pd.DataFrame: DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original DataFrame
        df_features = df.copy()
        
        # Price and volume features
        df_features['price_change'] = df_features['close'].pct_change()
        df_features['volume_change'] = df_features['volume'].pct_change()
        
        # Calculate returns
        df_features['return_1'] = df_features['close'].pct_change(1)
        df_features['return_3'] = df_features['close'].pct_change(3)
        df_features['return_6'] = df_features['close'].pct_change(6)
        df_features['return_12'] = df_features['close'].pct_change(12)
        df_features['return_24'] = df_features['close'].pct_change(24)
        
        # Calculate volatility
        df_features['volatility_3'] = df_features['return_1'].rolling(window=3).std()
        df_features['volatility_6'] = df_features['return_1'].rolling(window=6).std()
        df_features['volatility_12'] = df_features['return_1'].rolling(window=12).std()
        df_features['volatility_24'] = df_features['return_1'].rolling(window=24).std()
        
        # Technical indicators using TALib
        try:
            # RSI
            df_features['rsi'] = talib.RSI(df_features['close'], timeperiod=self.rsi_period)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df_features['close'], 
                fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            df_features['macd'] = macd
            df_features['macd_signal'] = macd_signal
            df_features['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df_features['close'], 
                timeperiod=self.bollinger_period, 
                nbdevup=self.bollinger_std, 
                nbdevdn=self.bollinger_std
            )
            df_features['bollinger_upper'] = upper
            df_features['bollinger_middle'] = middle
            df_features['bollinger_lower'] = lower
            df_features['bollinger_width'] = (upper - lower) / middle
            df_features['bollinger_pct'] = (df_features['close'] - lower) / (upper - lower)
            
            # Moving Averages
            df_features['ema_short'] = talib.EMA(df_features['close'], timeperiod=self.ema_short)
            df_features['ema_medium'] = talib.EMA(df_features['close'], timeperiod=self.ema_medium)
            df_features['ema_long'] = talib.EMA(df_features['close'], timeperiod=self.ema_long)
            
            # ATR (Average True Range) - volatility indicator
            df_features['atr'] = talib.ATR(
                df_features['high'], 
                df_features['low'], 
                df_features['close'], 
                timeperiod=14
            )
            
            # OBV (On-Balance Volume) - volume indicator
            df_features['obv'] = talib.OBV(df_features['close'], df_features['volume'])
            df_features['obv_change'] = df_features['obv'].pct_change()
            
            # ADX (Average Directional Index) - trend strength indicator
            df_features['adx'] = talib.ADX(
                df_features['high'], 
                df_features['low'], 
                df_features['close'], 
                timeperiod=14
            )
            
            # Create some cross-features
            df_features['ema_short_over_medium'] = df_features['ema_short'] / df_features['ema_medium'] - 1
            df_features['ema_short_over_long'] = df_features['ema_short'] / df_features['ema_long'] - 1
            df_features['ema_medium_over_long'] = df_features['ema_medium'] / df_features['ema_long'] - 1
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            # If TALib fails, create basic features
            df_features['rsi'] = np.nan
            df_features['macd'] = np.nan
            df_features['macd_signal'] = np.nan
            df_features['macd_hist'] = np.nan
        
        # Create target variable - future return
        future_return = df_features['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        df_features['future_return'] = future_return
        
        # Create binary target
        df_features['target'] = 0
        df_features.loc[future_return > self.prediction_threshold, 'target'] = 1
        df_features.loc[future_return < -self.prediction_threshold, 'target'] = -1
        
        # Convert to binary classification
        df_features['target'] = df_features['target'].apply(lambda x: 1 if x > 0 else 0)
        
        # Drop rows with NaN values
        df_features.dropna(inplace=True)
        
        return df_features
    
    def get_feature_names(self):
        """
        Get list of feature names for the model
        
        Returns:
        list: List of feature names
        """
        return [
            'price_change', 'volume_change', 
            'return_1', 'return_3', 'return_6', 'return_12', 'return_24',
            'volatility_3', 'volatility_6', 'volatility_12', 'volatility_24',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'bollinger_width', 'bollinger_pct',
            'ema_short', 'ema_medium', 'ema_long',
            'atr', 'obv', 'obv_change', 'adx',
            'ema_short_over_medium', 'ema_short_over_long', 'ema_medium_over_long'
        ]
    
    def prepare_prediction_data(self, df_features):
        """
        Prepare data for prediction (latest data point)
        
        Parameters:
        df_features (pd.DataFrame): DataFrame with features
        
        Returns:
        pd.DataFrame: DataFrame with features for prediction
        """
        # Make a copy of the latest data point
        prediction_data = df_features.iloc[-1:].copy()
        
        # Remove the target column if it exists
        if 'target' in prediction_data.columns:
            prediction_data = prediction_data.drop('target', axis=1)
        if 'future_return' in prediction_data.columns:
            prediction_data = prediction_data.drop('future_return', axis=1)
        
        return prediction_data

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create some sample OHLCV data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='1h')
    ohlcv_data = []
    
    for i, date in enumerate(dates):
        timestamp = int(date.timestamp() * 1000)  # Convert to milliseconds
        close = 100 + np.sin(i/10) * 10 + np.random.normal(0, 1)  # Create a sine wave with noise
        open_price = close - np.random.normal(0, 1)
        high = max(open_price, close) + np.random.normal(0, 0.5)
        low = min(open_price, close) - np.random.normal(0, 0.5)
        volume = 1000 + np.random.normal(0, 100)
        
        ohlcv_data.append([timestamp, open_price, high, low, close, volume])
    
    # Process data
    processor = DataProcessor()
    df = processor.process_ohlcv_data(ohlcv_data)
    df_features = processor.create_features(df)
    
    # Display results
    print("Original data shape:", df.shape)
    print("Features data shape:", df_features.shape)
    print("\nFeature columns:")
    print(df_features.columns.tolist())
    print("\nSample data:")
    print(df_features.head())
    print("\nTarget distribution:")
    if 'target' in df_features.columns:
        print(df_features['target'].value_counts())
