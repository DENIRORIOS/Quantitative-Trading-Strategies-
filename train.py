"""
Main training script for BTC trading model
Weekly retraining capability for adapting to volatility
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BTCDataFetcher
from utils.indicators import TechnicalIndicators
from data.preprocessor import DataPreprocessor
from models.ann_model import BTCTradingModel
from utils.strategy import TradingStrategy


class WeeklyTrainer:
    """Weekly retraining manager for the trading model"""
    
    def __init__(self, sequence_length=20, prediction_threshold=0.65):
        """
        Initialize weekly trainer
        
        Args:
            sequence_length: Sequence length for model (default: 20)
            prediction_threshold: Prediction threshold (default: 0.65)
        """
        self.sequence_length = sequence_length
        self.prediction_threshold = prediction_threshold
        self.model = BTCTradingModel(sequence_length=sequence_length)
        self.preprocessor = DataPreprocessor(sequence_length=sequence_length)
        
    def train_model(self, df, epochs=50, validation_split=0.2):
        """
        Train the model on provided data
        
        Args:
            df: DataFrame with features and indicators
            epochs: Number of training epochs
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        print("Preparing data for training...")
        train_X, train_y, test_X, test_y = self.preprocessor.prepare_data(df)
        
        print(f"Training data shape: {train_X.shape}")
        print(f"Training on {len(train_X)} samples, testing on {len(test_X)} samples")
        
        print("\nBuilding and training model...")
        self.model.build_model()
        history = self.model.train(train_X, train_y, 
                                  validation_split=validation_split, 
                                  epochs=epochs)
        
        print("\nEvaluating model...")
        metrics = self.model.evaluate(test_X, test_y)
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test AUC: {metrics['auc']:.4f}")
        print(f"Confident Predictions: {metrics['confident_predictions_ratio']*100:.2f}%")
        print(f"Confident Accuracy: {metrics['confident_accuracy']:.4f}")
        
        return history, metrics
    
    def weekly_retrain(self, df, weeks_back=8):
        """
        Perform weekly retraining to adapt to volatility
        
        Args:
            df: Full DataFrame with historical data
            weeks_back: Number of weeks to use for training (default: 8)
            
        Returns:
            Updated model
        """
        # Use last weeks_back weeks for training
        cutoff_date = df.index[-1] - timedelta(weeks=weeks_back)
        recent_df = df[df.index >= cutoff_date]
        
        print(f"Retraining on last {weeks_back} weeks of data...")
        print(f"Data range: {recent_df.index[0]} to {recent_df.index[-1]}")
        
        history, metrics = self.train_model(recent_df, epochs=30, validation_split=0.2)
        
        return self.model
    
    def save_model(self, filepath='models/btc_ann_model.h5'):
        """Save the trained model"""
        self.model.save_model(filepath)


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("BTC Trading Model - Training Pipeline")
    print("=" * 60)
    
    # Configuration
    SEQUENCE_LENGTH = 20
    PREDICTION_THRESHOLD = 0.65
    POSITION_SIZE_PCT = 0.01  # 1% of portfolio per trade
    
    # Step 1: Fetch data
    print("\n[1/6] Fetching BTC data...")
    fetcher = BTCDataFetcher()
    
    # Try to load existing data first
    try:
        df = fetcher.load_data('btc_5m_data.csv')
    except FileNotFoundError:
        print("No existing data found. Fetching from exchange...")
        df = fetcher.fetch_historical_data(days=730)  # ~2 years
        fetcher.save_data(df, 'btc_5m_data.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Calculate technical indicators
    print("\n[2/6] Calculating technical indicators...")
    df = TechnicalIndicators.add_all_indicators(df)
    print(f"Added indicators: RSI(14), MACD(12,26,9), Volume Delta")
    print(f"Data shape after indicators: {df.shape}")
    
    # Step 3: Initialize trainer
    print("\n[3/6] Initializing model trainer...")
    trainer = WeeklyTrainer(
        sequence_length=SEQUENCE_LENGTH,
        prediction_threshold=PREDICTION_THRESHOLD
    )
    
    # Step 4: Train model
    print("\n[4/6] Training model...")
    history, metrics = trainer.train_model(df, epochs=50)
    
    # Step 5: Save model
    print("\n[5/6] Saving model...")
    trainer.save_model('models/btc_ann_model.h5')
    
    # Step 6: Backtest strategy
    print("\n[6/6] Backtesting trading strategy...")
    preprocessor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    train_X, train_y, test_X, test_y = preprocessor.prepare_data(df)
    
    # Get predictions for test set
    predictions, probabilities, confident_mask = trainer.model.predict_with_threshold(
        test_X, threshold=PREDICTION_THRESHOLD
    )
    
    # Get corresponding prices
    test_start_idx = len(df) - len(test_X)
    test_prices = df['close'].iloc[test_start_idx:].values
    test_dates = df.index[test_start_idx:]
    
    # Run backtest
    strategy = TradingStrategy(
        initial_capital=10000,
        position_size_pct=POSITION_SIZE_PCT,
        prediction_threshold=PREDICTION_THRESHOLD
    )
    
    results = strategy.backtest(predictions, probabilities, test_prices, test_dates)
    print(strategy.get_summary(results))
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nModel Configuration:")
    print(f"  - Architecture: 3-layer ANN (512-256-128 neurons)")
    print(f"  - Activation: ReLU")
    print(f"  - Optimizer: Adam")
    print(f"  - Sequence Length: {SEQUENCE_LENGTH} periods")
    print(f"  - Prediction Threshold: {PREDICTION_THRESHOLD*100}%")
    print(f"  - Position Size: {POSITION_SIZE_PCT*100}% per trade")
    print("\nWeekly Retraining:")
    print("  - Retrain weekly using the trainer.weekly_retrain() method")
    print("  - This adapts the model to recent volatility changes")


if __name__ == "__main__":
    main()
