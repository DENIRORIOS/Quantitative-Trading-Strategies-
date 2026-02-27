"""
Simple tests to verify the model implementation
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ann_model import BTCTradingModel
from utils.indicators import TechnicalIndicators
from data.preprocessor import DataPreprocessor
from utils.strategy import TradingStrategy


def test_model_architecture():
    """Test that the model builds with correct architecture"""
    print("Testing model architecture...")
    
    model = BTCTradingModel(sequence_length=20, n_features=6)
    model.build_model()
    
    # Check model layers
    assert len(model.model.layers) == 8, "Expected 8 layers (Flatten + 3 Dense + 3 Dropout + Output)"
    
    # Check layer sizes
    assert model.model.layers[1].units == 512, "First hidden layer should have 512 neurons"
    assert model.model.layers[3].units == 256, "Second hidden layer should have 256 neurons"
    assert model.model.layers[5].units == 128, "Third hidden layer should have 128 neurons"
    assert model.model.layers[7].units == 1, "Output layer should have 1 neuron"
    
    print("✓ Model architecture is correct")


def test_prediction_threshold():
    """Test prediction with threshold filtering"""
    print("\nTesting prediction threshold...")
    
    model = BTCTradingModel(sequence_length=20, n_features=6)
    model.build_model()
    
    # Create dummy data
    X = np.random.randn(100, 20, 6)
    
    # Make predictions
    predictions, probabilities, confident_mask = model.predict_with_threshold(X, threshold=0.65)
    
    # Verify predictions are in correct range
    assert np.all(predictions >= -1) and np.all(predictions <= 1), "Predictions should be -1, 0, or 1"
    
    # Verify threshold logic
    assert np.all(predictions[probabilities > 0.65] == 1), "High probability should predict UP (1)"
    assert np.all(predictions[probabilities < 0.35] == 0), "Low probability should predict DOWN (0)"
    assert np.all(predictions[(probabilities >= 0.35) & (probabilities <= 0.65)] == -1), "Medium probability should be NO ACTION (-1)"
    
    print("✓ Prediction threshold works correctly")


def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\nTesting technical indicators...")
    
    # Create dummy price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.randn(100).cumsum() + 1000
    }, index=dates)
    
    # Add indicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Check that indicators were added
    assert 'rsi' in df.columns, "RSI should be added"
    assert 'macd' in df.columns, "MACD should be added"
    assert 'macd_signal' in df.columns, "MACD signal should be added"
    assert 'volume_delta' in df.columns, "Volume delta should be added"
    assert 'target' in df.columns, "Target should be added"
    
    print("✓ Technical indicators calculated correctly")


def test_data_preprocessing():
    """Test data preprocessing and sequence creation"""
    print("\nTesting data preprocessing...")
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.randn(100).cumsum() + 1000
    }, index=dates)
    
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Create sequences
    preprocessor = DataPreprocessor(sequence_length=20)
    X, y = preprocessor.create_sequences(df)
    
    # Check shapes
    assert X.shape[1] == 20, "Sequence length should be 20"
    assert X.shape[2] == 6, "Should have 6 features"
    assert len(X) == len(y), "X and y should have same length"
    
    print("✓ Data preprocessing works correctly")


def test_position_sizing():
    """Test position sizing logic"""
    print("\nTesting position sizing...")
    
    strategy = TradingStrategy(
        initial_capital=10000,
        position_size_pct=0.01,
        prediction_threshold=0.65
    )
    
    # Test 1% position size
    price = 50000
    position_size = strategy.calculate_position_size(price)
    position_value = position_size * price
    
    expected_value = 10000 * 0.01  # 1% of $10,000 = $100
    assert abs(position_value - expected_value) < 0.01, f"Position value should be ${expected_value}"
    
    print("✓ Position sizing works correctly")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Tests for BTC Trading Model")
    print("=" * 60)
    
    try:
        test_model_architecture()
        test_prediction_threshold()
        test_technical_indicators()
        test_data_preprocessing()
        test_position_sizing()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
