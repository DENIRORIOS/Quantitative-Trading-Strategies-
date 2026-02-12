"""
Configuration file for BTC trading model
Adjust these parameters to customize the model behavior
"""

# Model Architecture
SEQUENCE_LENGTH = 20  # Number of periods to look back (20 periods of 5-minute data)
HIDDEN_LAYERS = [512, 256, 128]  # Neurons in each hidden layer
ACTIVATION = 'relu'  # Activation function
OPTIMIZER = 'adam'  # Optimizer

# Training Parameters
EPOCHS = 50  # Maximum number of training epochs
BATCH_SIZE = 64  # Batch size for training
VALIDATION_SPLIT = 0.2  # Validation split ratio
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

# Data Parameters
LOOKBACK_DAYS = 730  # Days of historical data (2 years)
TIMEFRAME = '5m'  # Candlestick timeframe
TRAIN_TEST_SPLIT = 0.8  # Train/test split ratio

# Technical Indicators
RSI_PERIOD = 14  # RSI period
MACD_FAST = 12  # MACD fast period
MACD_SLOW = 26  # MACD slow period
MACD_SIGNAL = 9  # MACD signal period

# Trading Strategy
PREDICTION_THRESHOLD = 0.65  # Minimum prediction confidence (65%)
POSITION_SIZE = 0.01  # Position size as % of portfolio per trade (1% = 0.01)
# Note: Position size can be configured between 0.5-1% (0.005-0.01) for different risk levels
INITIAL_CAPITAL = 10000  # Initial portfolio value

# Retraining
RETRAIN_WEEKS = 8  # Number of weeks of data for retraining
RETRAIN_FREQUENCY = 'weekly'  # Retraining frequency

# Data Source
EXCHANGE = 'binance'  # Exchange name
SYMBOL = 'BTC/USDT'  # Trading pair

# File Paths
DATA_FILE = 'data/btc_5m_data.csv'
MODEL_FILE = 'models/btc_ann_model.h5'
