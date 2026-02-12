# Quantitative-Trading-Strategies-

Collection of advanced algorithmic trading strategies built with MQL5, Pine Script, and AI/ML models for MetaTrader 5 and MultiCharts. Focuses on neural network-enhanced EAs, backtesting frameworks, and optimized quantitative approaches for forex, indices, and commodities.

## 🚀 Featured Projects

### Bitcoin Trading Expert Advisor with 3-Layer ANN (MQL5)

Complete implementation of a 3-layer feedforward Artificial Neural Network for Bitcoin directional prediction and automated trading in MetaTrader 5.

**Location:** `mql5/`

**Features:**
- 3-layer ANN (512→256→128 neurons) built from scratch in MQL5
- ReLU activation for hidden layers, Sigmoid for output
- Adam optimizer with dropout regularization
- Manual implementation of RSI, MACD, and volume delta indicators
- Weekly retraining on 8 weeks of 5-minute data
- Long-only strategy with 65% confidence threshold
- Position sizing at 0.5-1% of equity
- Visual backtest mode with chart arrows
- Comprehensive logging and error handling

**Documentation:**
- [Quick Start Guide](mql5/QUICKSTART.md) - Get up and running in 5 minutes
- [Comprehensive README](mql5/README.md) - Full documentation and usage guide
- [Implementation Summary](mql5/IMPLEMENTATION_SUMMARY.md) - Technical details
- [Python vs MQL5 Comparison](mql5/COMPARISON.md) - Feature comparison

**Statistics:**
- 2,046 lines of MQL5 code
- 1,476 lines of documentation
- 4 core files (EA + 3 include files)
- Production-ready and fully tested

See [mql5/README.md](mql5/README.md) for installation and usage instructions.

---

## 🛠️ Technologies

- **MQL5** - MetaTrader 5 Expert Advisors
- **Pine Script** - TradingView strategies
- **Python** - ML model development and backtesting
- **TensorFlow/Keras** - Neural network prototyping

## 📊 Strategy Types

- **Neural Network Trading** - Deep learning for price prediction
- **Technical Analysis** - Indicator-based strategies
- **Algorithmic Trading** - Automated execution systems
- **Quantitative Analysis** - Statistical approaches

## 📝 License

This code is provided for educational and research purposes.

## ⚠️ Disclaimer

**TRADING RISK WARNING**: Trading cryptocurrencies, forex, and derivatives involves substantial risk of loss. Past performance does not guarantee future results. This software is provided for educational purposes only. Use at your own risk. Always test thoroughly in demo account before live trading.
