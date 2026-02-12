# Quantitative-Trading-Strategies-

Collection of advanced algorithmic trading strategies built with MQL5, Pine Script, and AI/ML models for MetaTrader 5 and MultiCharts. Focuses on neural network-enhanced EAs, backtesting frameworks, and optimized quantitative approaches for forex, indices, and commodities.

## BTC Trading Model with 3-Layer ANN

This repository includes a sophisticated Bitcoin trading model using a 3-layer Artificial Neural Network (ANN) for predicting price direction.

### Model Specifications

**Architecture:**
- 3-layer ANN with 512-256-128 neurons
- ReLU activation functions
- Adam optimizer
- Dropout layers for regularization

**Training Data:**
- 5-minute BTC/USDT candlestick data
- 2 years of historical data
- Rolling windows for out-of-sample validation

**Features (Inputs):**
- 20-period price sequences
- RSI(14) - Relative Strength Index
- MACD(12,26,9) - Moving Average Convergence Divergence
- Volume delta (volume change)

**Output:**
- Binary classification: Next-bar direction (up/down)
- Probability scores for confidence filtering

**Risk Management:**
- Prediction threshold: Only act on predictions >65% probability
- Position size: 0.5-1% of portfolio per trade
- Weekly retraining to adapt to market volatility

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DENIRORIOS/Quantitative-Trading-Strategies-.git
cd Quantitative-Trading-Strategies-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Model

Run the training script to fetch data, calculate indicators, and train the model:

```bash
python train.py
```

This will:
1. Fetch 2 years of 5-minute BTC data from Binance
2. Calculate technical indicators (RSI, MACD, volume delta)
3. Train the 3-layer ANN model
4. Save the trained model to `models/btc_ann_model.h5`
5. Perform backtesting and display results

#### Making Predictions

Use the trained model to make predictions:

```bash
python predict.py
```

This will:
1. Load the trained model
2. Fetch recent data
3. Generate predictions with confidence scores
4. Display trading signals (BUY/SELL/HOLD)

#### Weekly Retraining

To adapt to market volatility, retrain the model weekly:

```python
from train import WeeklyTrainer
from data.fetcher import BTCDataFetcher
from utils.indicators import TechnicalIndicators

# Fetch latest data
fetcher = BTCDataFetcher()
df = fetcher.fetch_historical_data(days=730)
df = TechnicalIndicators.add_all_indicators(df)

# Retrain on recent data
trainer = WeeklyTrainer()
trainer.weekly_retrain(df, weeks_back=8)
trainer.save_model()
```

### Project Structure

```
.
├── data/
│   ├── __init__.py
│   ├── fetcher.py          # Data fetching from exchanges
│   └── preprocessor.py     # Data preprocessing and rolling windows
├── models/
│   ├── __init__.py
│   └── ann_model.py        # 3-layer ANN implementation
├── utils/
│   ├── __init__.py
│   ├── indicators.py       # Technical indicators calculation
│   └── strategy.py         # Trading strategy and position sizing
├── train.py                # Main training script
├── predict.py              # Prediction script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Key Features

1. **Adaptive Learning**: Weekly retraining keeps the model current with market conditions
2. **Risk Management**: Only acts on high-confidence predictions (>65% probability)
3. **Position Sizing**: Conservative 0.5-1% position sizes for low drawdown
4. **Out-of-Sample Validation**: Rolling windows ensure realistic performance estimates
5. **Technical Analysis**: Combines price action with RSI, MACD, and volume indicators

### Model Performance Metrics

The model provides several performance metrics:
- **Accuracy**: Overall prediction accuracy
- **AUC**: Area Under the ROC Curve
- **Confident Predictions Ratio**: Percentage of predictions above threshold
- **Confident Accuracy**: Accuracy on high-confidence predictions only
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Portfolio return over backtest period

### Trading Strategy

The strategy follows these rules:
1. **Entry**: Only enter positions when prediction confidence >65%
2. **Position Size**: 0.5-1% of portfolio per trade
3. **Direction**: 
   - BUY when prediction probability >65% (up)
   - SELL/SHORT when prediction probability <35% (down)
   - HOLD when probability between 35-65% (uncertain)
4. **Retraining**: Weekly retraining to adapt to volatility changes

### Requirements

- Python 3.8+
- TensorFlow 2.10+
- pandas
- numpy
- scikit-learn
- ccxt (for exchange data)
- ta (for technical indicators)

### Disclaimer

This is a research and educational project. Cryptocurrency trading involves significant risk. Always perform your own due diligence and never invest more than you can afford to lose. Past performance does not guarantee future results.

### License

MIT License - See LICENSE file for details
