"""
Demo script with synthetic data to show end-to-end workflow
This demonstrates the complete pipeline without requiring live data
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.indicators import TechnicalIndicators
from data.preprocessor import DataPreprocessor
from models.ann_model import BTCTradingModel
from utils.strategy import TradingStrategy


def generate_synthetic_btc_data(periods=10000, start_price=50000):
    """
    Generate synthetic BTC price data with realistic patterns
    
    Args:
        periods: Number of 5-minute periods (default: 10000 = ~35 days)
        start_price: Starting price (default: $50,000)
        
    Returns:
        DataFrame with OHLCV data
    """
    print("Generating synthetic BTC data...")
    
    # Generate random walk with trend
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.002, periods)
    
    # Add some trend and cycles
    trend = np.linspace(0, 0.1, periods)
    cycle = 0.05 * np.sin(np.linspace(0, 10*np.pi, periods))
    
    prices = start_price * np.exp(np.cumsum(returns + trend/periods + cycle/periods))
    
    # Generate OHLCV
    dates = pd.date_range(start=datetime.now() - timedelta(minutes=5*periods), 
                         periods=periods, freq='5min')
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': prices * (1 + np.random.normal(0, 0.0005, periods)),
        'volume': np.random.uniform(100, 1000, periods)
    }, index=dates)
    
    print(f"Generated {len(df)} periods of data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def demo_full_pipeline():
    """Demonstrate the complete training and trading pipeline"""
    print("=" * 70)
    print("BTC TRADING MODEL - FULL PIPELINE DEMO")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/7] Generating synthetic BTC data...")
    df = generate_synthetic_btc_data(periods=10000)
    
    # Step 2: Calculate technical indicators
    print("\n[Step 2/7] Calculating technical indicators...")
    df = TechnicalIndicators.add_all_indicators(df)
    print(f"Added indicators: RSI(14), MACD(12,26,9), Volume Delta")
    print(f"Final data shape: {df.shape}")
    
    # Step 3: Prepare data
    print("\n[Step 3/7] Preparing data with rolling windows...")
    preprocessor = DataPreprocessor(sequence_length=20)
    train_X, train_y, test_X, test_y = preprocessor.prepare_data(df, train_split=0.8)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Testing samples: {len(test_X)}")
    print(f"Feature shape: {train_X.shape}")
    
    # Step 4: Build and train model
    print("\n[Step 4/7] Building 3-layer ANN model...")
    model = BTCTradingModel(sequence_length=20, n_features=6)
    model.build_model()
    
    print("\nModel Summary:")
    print(f"  - Layer 1: 512 neurons (ReLU)")
    print(f"  - Layer 2: 256 neurons (ReLU)")
    print(f"  - Layer 3: 128 neurons (ReLU)")
    print(f"  - Output: 1 neuron (Sigmoid)")
    print(f"  - Optimizer: Adam")
    
    print("\n[Step 5/7] Training model...")
    history = model.train(train_X, train_y, validation_split=0.2, epochs=20, batch_size=64)
    
    print(f"\nTraining completed in {len(history.history['loss'])} epochs")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Step 5: Evaluate model
    print("\n[Step 6/7] Evaluating model on test set...")
    metrics = model.evaluate(test_X, test_y)
    
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Confident Predictions (>65%): {metrics['confident_predictions_ratio']*100:.2f}%")
    print(f"Accuracy on Confident Predictions: {metrics['confident_accuracy']:.4f}")
    
    # Step 6: Backtest strategy
    print("\n[Step 7/7] Backtesting trading strategy...")
    
    # Get predictions with threshold
    predictions, probabilities, confident_mask = model.predict_with_threshold(
        test_X, threshold=0.65
    )
    
    # Get corresponding prices
    test_start_idx = len(df) - len(test_X)
    test_prices = df['close'].iloc[test_start_idx:].values
    test_dates = df.index[test_start_idx:]
    
    # Run backtest
    strategy = TradingStrategy(
        initial_capital=10000,
        position_size_pct=0.01,  # 1% per trade
        prediction_threshold=0.65
    )
    
    results = strategy.backtest(predictions, probabilities, test_prices, test_dates)
    
    # Display results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(strategy.get_summary(results))
    
    # Show some example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (Last 10 periods)")
    print("=" * 70)
    
    last_predictions = predictions[-10:]
    last_probs = probabilities[-10:]
    last_prices = test_prices[-10:]
    last_dates = test_dates[-10:]
    
    for i in range(len(last_predictions)):
        pred = last_predictions[i]
        prob = last_probs[i]
        price = last_prices[i]
        date = last_dates[i]
        
        if pred == 1:
            signal = "BUY (UP)"
            emoji = "🟢"
        elif pred == 0:
            signal = "SELL (DOWN)"
            emoji = "🔴"
        else:
            signal = "HOLD"
            emoji = "⚪"
        
        print(f"{emoji} {date.strftime('%Y-%m-%d %H:%M')} | "
              f"Price: ${price:,.2f} | {signal:15} | Confidence: {prob:.2%}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ 3-layer ANN (512-256-128 neurons)")
    print("  ✓ ReLU activation, Adam optimizer")
    print("  ✓ Technical indicators: RSI(14), MACD(12,26,9), Volume Delta")
    print("  ✓ 20-period price sequences")
    print("  ✓ 65% prediction threshold")
    print("  ✓ 1% position sizing per trade")
    print("  ✓ Rolling window validation")
    print("  ✓ Complete backtesting")
    print("\nNext Steps:")
    print("  - Use real data: Run `python train.py` to fetch actual BTC data")
    print("  - Weekly retraining: Use WeeklyTrainer.weekly_retrain() method")
    print("  - Live predictions: Run `python predict.py` after training")
    print("=" * 70)


if __name__ == "__main__":
    demo_full_pipeline()
