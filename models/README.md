# Models Directory

This directory stores trained machine learning models for the memecoin trading bot.

## Model Naming Convention

Models are saved with the following naming pattern:
`{symbol}_model.joblib`

Where:
- `{symbol}` is the trading pair symbol with "/" replaced by "_" (e.g., "DOGE_USDT")

## Model Information

Each model is a trained classifier that predicts price movements for a specific memecoin trading pair.

### Features Used

The models are trained on various technical indicators and price features, including:
- Price changes and returns
- Volume indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (EMA)
- Volatility metrics
- And more...

### Target

The models predict whether the price will increase by a certain percentage threshold within a specific time horizon.

### Algorithm

The default model type is RandomForestClassifier, which was chosen for its:
- Robustness to noise
- Ability to capture non-linear relationships
- Feature importance information
- Resistance to overfitting

## Model Maintenance

Models are automatically retrained in the following scenarios:
1. When the trading bot starts and no model exists for a symbol
2. Periodically (e.g., once per day) to adapt to changing market conditions
3. Manually via the "Retrain All Models" button in the UI

## Performance Metrics

Model performance is tracked using the following metrics:
- Accuracy: Overall prediction accuracy
- Precision: Proportion of positive identifications that were actually correct
- Recall: Proportion of actual positives that were identified correctly
- F1 Score: Harmonic mean of precision and recall
- Cross-validation scores: Performance across different time periods

Models are saved in the joblib format, which efficiently serializes Python objects to disk.
