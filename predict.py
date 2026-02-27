"""
Example usage script for making predictions with the trained model
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BTCDataFetcher
from utils.indicators import TechnicalIndicators
from data.preprocessor import DataPreprocessor
from models.ann_model import BTCTradingModel


def make_prediction():
    """Make a prediction using the trained model"""
    print("=" * 60)
    print("BTC Trading Model - Prediction")
    print("=" * 60)
    
    # Configuration
    SEQUENCE_LENGTH = 20
    PREDICTION_THRESHOLD = 0.65
    MODEL_PATH = 'models/btc_ann_model.h5'
    
    # Load model
    print("\n[1/4] Loading trained model...")
    model = BTCTradingModel(sequence_length=SEQUENCE_LENGTH)
    try:
        model.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running: python train.py")
        return
    
    # Fetch recent data
    print("\n[2/4] Fetching recent BTC data...")
    fetcher = BTCDataFetcher()
    
    try:
        # Load existing data
        df = fetcher.load_data('btc_5m_data.csv')
        print(f"Loaded data: {len(df)} candles")
    except FileNotFoundError:
        print("No data found. Please run train.py first to fetch data.")
        return
    
    # Add indicators
    print("\n[3/4] Calculating indicators...")
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Prepare data
    print("\n[4/4] Making predictions...")
    preprocessor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    X, y = preprocessor.create_sequences(df)
    
    # Scale the data
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    preprocessor.scaler.fit(X_reshaped)
    X_scaled = preprocessor.scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Get predictions with threshold
    predictions, probabilities, confident_mask = model.predict_with_threshold(
        X_scaled[-10:], threshold=PREDICTION_THRESHOLD
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Recent Predictions (Last 10 periods)")
    print("=" * 60)
    
    recent_dates = df.index[-10:]
    recent_prices = df['close'].iloc[-10:].values
    
    for i in range(len(predictions)):
        date = recent_dates[i]
        price = recent_prices[i]
        pred = predictions[i]
        prob = probabilities[i]
        confident = confident_mask[i]
        
        if pred == 1:
            signal = "BUY (UP)"
            color = "🟢"
        elif pred == 0:
            signal = "SELL (DOWN)"
            color = "🔴"
        else:
            signal = "HOLD (NO ACTION)"
            color = "⚪"
        
        conf_str = "CONFIDENT" if confident else "LOW CONFIDENCE"
        
        print(f"{color} {date} | Price: ${price:,.2f} | {signal} | "
              f"Prob: {prob:.2%} | {conf_str}")
    
    print("\n" + "=" * 60)
    print("Prediction Guidelines:")
    print(f"  - Only act on predictions with >{PREDICTION_THRESHOLD*100}% confidence")
    print(f"  - BUY when probability > {PREDICTION_THRESHOLD*100}%")
    print(f"  - SELL when probability < {(1-PREDICTION_THRESHOLD)*100}%")
    print(f"  - HOLD otherwise")
    print("=" * 60)


if __name__ == "__main__":
    make_prediction()
