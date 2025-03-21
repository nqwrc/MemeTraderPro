import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ModelTrainer:
    """
    Class to handle model training and evaluation for the trading bot
    """
    def __init__(self):
        """
        Initialize model trainer
        """
        # Model parameters
        self.test_size = 0.2
        self.random_state = 42
        self.n_estimators = 100
        self.max_depth = 10
        self.n_splits = 5  # for time series cross-validation
    
    def train_model(self, features_df):
        """
        Train a predictive model using the provided features
        
        Parameters:
        features_df (pd.DataFrame): DataFrame with features and target
        
        Returns:
        tuple: (trained_model, metrics_dict)
        """
        # Make sure we have a target column
        if 'target' not in features_df.columns:
            raise ValueError("Feature DataFrame must contain a 'target' column")
        
        # Split data into features (X) and target (y)
        X = features_df.drop(['target', 'future_return'] if 'future_return' in features_df.columns else ['target'], axis=1)
        y = features_df['target']
        
        # Drop any non-numeric columns
        numeric_cols = X.select_dtypes(include=np.number).columns
        X = X[numeric_cols]
        
        # Feature scaling - important for many ML algorithms
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Use time-series split for chronological data
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Initialize RandomForest classifier
        model = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            random_state=self.random_state
        )
        
        # Perform time series cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        
        # Train on full dataset
        model.fit(X_scaled, y)
        
        # Get predictions on the training set to evaluate performance
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        # Return the trained model and metrics
        return model, metrics
    
    def evaluate_model(self, model, features_df):
        """
        Evaluate a model on new data
        
        Parameters:
        model: Trained model object
        features_df (pd.DataFrame): DataFrame with features and target
        
        Returns:
        dict: Evaluation metrics
        """
        # Make sure we have a target column
        if 'target' not in features_df.columns:
            raise ValueError("Feature DataFrame must contain a 'target' column")
        
        # Split data into features (X) and target (y)
        X = features_df.drop(['target', 'future_return'] if 'future_return' in features_df.columns else ['target'], axis=1)
        y = features_df['target']
        
        # Drop any non-numeric columns
        numeric_cols = X.select_dtypes(include=np.number).columns
        X = X[numeric_cols]
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        return metrics
    
    def save_model(self, model, symbol, directory='models'):
        """
        Save a trained model to disk
        
        Parameters:
        model: Trained model object
        symbol (str): Symbol the model was trained on
        directory (str): Directory to save the model
        
        Returns:
        str: Path to the saved model
        """
        # Make sure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Create a safe filename from the symbol
        safe_symbol = symbol.replace('/', '_')
        
        # Save the model
        model_path = os.path.join(directory, f"{safe_symbol}_model.joblib")
        joblib.dump(model, model_path)
        
        return model_path
    
    def load_model(self, symbol, directory='models'):
        """
        Load a trained model from disk
        
        Parameters:
        symbol (str): Symbol the model was trained on
        directory (str): Directory where the model is saved
        
        Returns:
        object: Loaded model or None if model file doesn't exist
        """
        # Create a safe filename from the symbol
        safe_symbol = symbol.replace('/', '_')
        
        # Path to the model file
        model_path = os.path.join(directory, f"{safe_symbol}_model.joblib")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return None
        
        # Load and return the model
        return joblib.load(model_path)

if __name__ == "__main__":
    # Example usage
    from data_processor import DataProcessor
    import pandas as pd
    import numpy as np
    
    # Create some sample OHLCV data
    dates = pd.date_range(start='2022-01-01', periods=500, freq='1h')
    ohlcv_data = []
    
    for i, date in enumerate(dates):
        timestamp = int(date.timestamp() * 1000)  # Convert to milliseconds
        close = 100 + np.sin(i/30) * 20 + np.random.normal(0, 2)  # Create a sine wave with noise
        open_price = close - np.random.normal(0, 1.5)
        high = max(open_price, close) + np.random.normal(0, 1)
        low = min(open_price, close) - np.random.normal(0, 1)
        volume = 1000 + np.random.normal(0, 200)
        
        ohlcv_data.append([timestamp, open_price, high, low, close, volume])
    
    # Process data
    processor = DataProcessor()
    df = processor.process_ohlcv_data(ohlcv_data)
    df_features = processor.create_features(df)
    
    # Train model
    trainer = ModelTrainer()
    model, metrics = trainer.train_model(df_features)
    
    # Print metrics
    print("Model Training Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix' and key != 'cv_scores':
            print(f"{key}: {value}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    
    # Save model
    model_path = trainer.save_model(model, "EXAMPLE/USDT")
    print(f"\nModel saved to: {model_path}")
    
    # Load model
    loaded_model = trainer.load_model("EXAMPLE/USDT")
    if loaded_model:
        print("Model loaded successfully!")
