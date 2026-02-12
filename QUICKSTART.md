# Quick Start Guide

## Bitcoin Trading Model - 3-Layer ANN

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/DENIRORIOS/Quantitative-Trading-Strategies-.git
cd Quantitative-Trading-Strategies-
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Quick Start

#### Option 1: Run Demo (Recommended for first time)

The demo uses synthetic data and shows the complete workflow without requiring API access:

```bash
python demo.py
```

This will:
- Generate synthetic BTC price data
- Calculate technical indicators
- Train the 3-layer ANN model
- Run backtesting
- Show sample predictions

**Expected output:** You'll see the complete training process and results.

#### Option 2: Train with Real Data

To train the model with real Bitcoin data from Binance:

```bash
python train.py
```

This will:
1. Fetch ~2 years of 5-minute BTC/USDT data from Binance
2. Calculate RSI(14), MACD(12,26,9), and volume delta
3. Train the 3-layer ANN (512-256-128 neurons)
4. Save the model to `models/btc_ann_model.h5`
5. Display backtesting results

**Note:** First run will take time to fetch data. Data is cached in `data/btc_5m_data.csv` for subsequent runs.

#### Option 3: Make Predictions

After training the model, make predictions on recent data:

```bash
python predict.py
```

This will:
- Load the trained model
- Fetch recent BTC data
- Generate predictions with confidence scores
- Display trading signals (BUY/SELL/HOLD)

### Understanding the Output

#### Prediction Signals

- 🟢 **BUY (UP)**: Prediction confidence >65% for price increase
- 🔴 **SELL (DOWN)**: Prediction confidence >65% for price decrease  
- ⚪ **HOLD**: Prediction confidence between 35-65% (uncertain)

#### Key Metrics

- **Accuracy**: Overall prediction accuracy
- **AUC**: Model's ability to distinguish between up/down movements
- **Confident Predictions**: % of predictions with >65% confidence
- **Confident Accuracy**: Accuracy on high-confidence predictions only
- **Win Rate**: % of profitable trades
- **Total Return**: Portfolio return over backtest period

### Model Configuration

Edit `config.py` to customize:

```python
# Model Architecture
SEQUENCE_LENGTH = 20  # Look-back periods
HIDDEN_LAYERS = [512, 256, 128]  # Neurons per layer

# Trading Strategy  
PREDICTION_THRESHOLD = 0.65  # Confidence threshold (65%)
POSITION_SIZE = 0.01  # Position size (1% of portfolio)

# Training
EPOCHS = 50  # Training epochs
BATCH_SIZE = 64  # Batch size
```

### Weekly Retraining

To keep the model adapted to market changes, retrain weekly:

```python
from train import WeeklyTrainer
from data.fetcher import BTCDataFetcher
from utils.indicators import TechnicalIndicators

# Fetch latest data
fetcher = BTCDataFetcher()
df = fetcher.fetch_historical_data(days=730)
df = TechnicalIndicators.add_all_indicators(df)

# Retrain on recent 8 weeks
trainer = WeeklyTrainer()
trainer.weekly_retrain(df, weeks_back=8)
trainer.save_model()
```

### Testing

Run the test suite to verify everything works:

```bash
python test_model.py
```

Expected output: "All tests passed! ✓"

### Project Structure

```
.
├── data/                  # Data fetching and preprocessing
│   ├── fetcher.py        # Fetch BTC data from exchanges
│   └── preprocessor.py   # Create sequences and rolling windows
├── models/               # Neural network models
│   └── ann_model.py      # 3-layer ANN implementation
├── utils/                # Utility functions
│   ├── indicators.py     # Technical indicators (RSI, MACD)
│   └── strategy.py       # Trading strategy and position sizing
├── train.py              # Main training script
├── predict.py            # Prediction script
├── demo.py               # Demo with synthetic data
├── test_model.py         # Test suite
├── config.py             # Configuration
└── requirements.txt      # Dependencies
```

### Common Issues

#### Issue: No module named 'xxx'
**Solution:** Install dependencies: `pip install -r requirements.txt`

#### Issue: Connection error when fetching data
**Solution:** Check internet connection or run demo: `python demo.py`

#### Issue: Model not found
**Solution:** Train the model first: `python train.py`

#### Issue: Out of memory during training
**Solution:** Reduce batch size in config.py or use fewer epochs

### Next Steps

1. **Experiment with parameters**: Adjust thresholds, position sizes in `config.py`
2. **Add more features**: Extend `indicators.py` with additional technical indicators
3. **Implement live trading**: Integrate with exchange API for automated trading
4. **Set up monitoring**: Add logging and alerts for production use
5. **Optimize performance**: Try different network architectures

### Support

For issues or questions:
- Check `IMPLEMENTATION_SUMMARY.md` for detailed information
- Review the code comments for implementation details
- Run `python test_model.py` to verify installation

### Warning

⚠️ **Trading Disclaimer**: Cryptocurrency trading involves significant risk. This is a research and educational project. Never invest more than you can afford to lose. Past performance does not guarantee future results. Always do your own research before trading.

### License

MIT License - See repository for details
