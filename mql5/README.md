# Bitcoin Trading Expert Advisor with 3-Layer ANN

## Overview

This Expert Advisor (EA) implements a complete 3-layer feedforward Artificial Neural Network (ANN) from scratch in MQL5 for Bitcoin directional prediction and automated trading.

## Architecture

### Neural Network
- **Input Layer**: 120 neurons (20 periods × 6 features)
- **Hidden Layer 1**: 512 neurons (ReLU activation, 30% dropout)
- **Hidden Layer 2**: 256 neurons (ReLU activation, 30% dropout)
- **Hidden Layer 3**: 128 neurons (ReLU activation, 20% dropout)
- **Output Layer**: 1 neuron (Sigmoid activation - probability of price going up)

### Optimizer
- **Adam** optimizer with:
  - Learning rate: 0.001 (configurable)
  - Beta1: 0.9
  - Beta2: 0.999
  - Epsilon: 1e-8

### Weight Initialization
- **He initialization** for ReLU layers: weights ~ N(0, sqrt(2/n_in))

## Features

### Input Features (6 features per period)
1. **Close Price** - Raw closing price
2. **RSI(14)** - Relative Strength Index
3. **MACD Line** - Fast EMA - Slow EMA
4. **MACD Signal** - EMA of MACD line
5. **MACD Histogram** - MACD Line - Signal Line
6. **Volume Delta** - Approximation of buy/sell pressure

### Technical Indicators (Calculated Manually)
All indicators are calculated from scratch without using built-in MQL5 indicators:

- **RSI(14)**: Wilder's smoothing method
- **MACD(12, 26, 9)**: Exponential moving averages
- **Volume Delta**: Tick volume with directional weighting

### Data Pipeline
- Fetches 5-minute BTCUSD/BTCUSDT candle data
- Normalizes inputs using online StandardScaler (running mean/std)
- Creates sliding window sequences of 20 periods
- Handles 70/30 train/test split for retraining

### Trading Strategy
- **Long-only strategy**
- **Entry**: Open BUY when P(up) > 65% (confidence threshold)
- **Exit**: Close position when:
  - P(down) > 65% (bearish signal), or
  - P(up) < 65% (confidence too low)
- **Position Sizing**: 0.5-1% of account equity per trade (default: 1%)

### Weekly Retraining
- Automatically retrains every 7 days (configurable)
- Uses last 8 weeks of 5-minute bars for training
- Implements early stopping with patience=10 on validation loss
- Saves trained weights to binary file: `BTC_ANN_weights.bin`
- Weights persist across EA restarts

## Installation

### 1. Copy Files to MetaTrader 5 Directory

Copy the files to your MT5 data folder (File → Open Data Folder in MT5):

```
MQL5/
├── Experts/
│   └── BTC_ANN_EA.mq5
└── Include/
    └── ANN/
        ├── NeuralNetwork.mqh
        ├── DataPipeline.mqh
        └── TradeManager.mqh
```

### 2. Compile in MetaEditor

1. Open MetaEditor (F4 in MT5)
2. Open `Experts/BTC_ANN_EA.mq5`
3. Click Compile (F7) or press the Compile button
4. Verify no errors appear in the Toolbox window

### 3. Attach to Chart

1. In MT5, open a BTCUSD or BTCUSDT chart (5-minute timeframe recommended)
2. Drag `BTC_ANN_EA` from Navigator → Expert Advisors onto the chart
3. Configure parameters (see below)
4. Enable AutoTrading (click the AutoTrading button in toolbar)

## Configuration Parameters

### Neural Network Parameters
```
SequenceLength      = 20        // Length of input sequences (20 bars)
LearningRate        = 0.001     // Adam optimizer learning rate
DropoutRate1        = 0.3       // Dropout rate for layer 1 (30%)
DropoutRate2        = 0.3       // Dropout rate for layer 2 (30%)
DropoutRate3        = 0.2       // Dropout rate for layer 3 (20%)
```

### Trading Parameters
```
PredictionThreshold = 0.65      // Confidence threshold (65%)
PositionSizePct     = 0.01      // Position size (1% of equity)
MagicNumber         = 12345     // Magic number for orders
```

### Technical Indicators
```
RSI_Period          = 14        // RSI calculation period
MACD_Fast           = 12        // MACD fast EMA period
MACD_Slow           = 26        // MACD slow EMA period
MACD_Signal         = 9         // MACD signal line period
```

### Training Parameters
```
TrainingEpochs      = 50        // Maximum training epochs
EarlyStopPatience   = 10        // Early stopping patience
RetrainWeeks        = 8         // Weeks of data for retraining
TradingTimeframe    = PERIOD_M5 // Trading timeframe (5-minute)
```

### Visual & Optimization
```
ShowVisuals         = true      // Show buy/sell arrows on chart
EnableWeeklyRetrain = true      // Enable automatic retraining
```

## How It Works

### Initialization (OnInit)
1. Creates neural network with specified architecture
2. Attempts to load previously trained weights from file
3. Initializes data pipeline for indicator calculation
4. Initializes trade manager for position management
5. Sets up timer for weekly retraining

### On Each Tick (OnTick)
1. Waits for new bar formation (trades only on new bars)
2. Fetches current 20-period sequence with indicators
3. Normalizes the sequence using fitted scaler
4. Runs forward propagation through neural network
5. Gets probability prediction P(up)
6. Executes trading logic based on confidence threshold
7. Optionally displays visual arrow on chart

### Weekly Retraining (OnTimer)
1. Checks if 7 days have passed since last training
2. Fetches 8 weeks of historical 5-minute data
3. Calculates all technical indicators
4. Normalizes features using StandardScaler
5. Creates sequences and targets
6. Splits into 70% training, 30% validation
7. Trains network using:
   - Forward propagation
   - Backpropagation with gradient calculation
   - Adam optimizer weight updates
   - Dropout regularization
8. Implements early stopping based on validation loss
9. Saves best weights to file

### Deinitialization (OnDeinit)
1. Saves current weights to file
2. Prints trade statistics (win rate, total trades)
3. Cleans up memory and timer

## Neural Network Mathematics

### Forward Propagation
```
Layer 1: z1 = W1^T * input + b1
         a1 = ReLU(z1) * dropout_mask1

Layer 2: z2 = W2^T * a1 + b2
         a2 = ReLU(z2) * dropout_mask2

Layer 3: z3 = W3^T * a2 + b3
         a3 = ReLU(z3) * dropout_mask3

Output:  z4 = W4^T * a3 + b4
         output = Sigmoid(z4)
```

### Backpropagation
```
Binary Cross-Entropy Loss: L = -[y*log(p) + (1-y)*log(1-p)]

Output gradient:  dL/dz4 = output - target

Layer 4 gradients:
  dW4 = a3 * dz4
  db4 = dz4

Layer 3 gradients:
  dz3 = (W4 * dz4) * ReLU'(z3) * dropout_mask3
  dW3 = a2 * dz3
  db3 = dz3

Layer 2 gradients:
  dz2 = (W3 * dz3) * ReLU'(z2) * dropout_mask2
  dW2 = a1 * dz2
  db2 = dz2

Layer 1 gradients:
  dz1 = (W2 * dz2) * ReLU'(z1) * dropout_mask1
  dW1 = input * dz1
  db1 = dz1
```

### Adam Optimizer Update
```
For each weight W and gradient dW:

  m_t = beta1 * m_{t-1} + (1 - beta1) * dW        // First moment
  v_t = beta2 * v_{t-1} + (1 - beta2) * dW^2      // Second moment
  
  m_hat = m_t / (1 - beta1^t)                     // Bias correction
  v_hat = v_t / (1 - beta2^t)
  
  W = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

## File Format

### Weight File (BTC_ANN_weights.bin)
Binary file storing:
- Layer sizes (5 integers)
- All weight matrices (W1, W2, W3, W4)
- All bias vectors (b1, b2, b3, b4)

The file is saved in the MT5 data folder under `MQL5/Files/`

## Backtesting

### Strategy Tester
1. Open Strategy Tester (Ctrl+R)
2. Select `BTC_ANN_EA`
3. Choose symbol (BTCUSD or BTCUSDT)
4. Set timeframe to M5 (5-minute)
5. Configure date range (at least 2-3 months recommended)
6. Select "Every tick" or "1 minute OHLC" mode
7. Click Start

**Note**: First backtest run will train the model, which may take several minutes depending on the data range.

### Optimization
The EA includes `OnTester()` function for genetic algorithm optimization:
- Fitness = (Final Balance / Initial Balance) × Win Rate
- Optimize parameters like:
  - PredictionThreshold (0.60 - 0.75)
  - PositionSizePct (0.005 - 0.02)
  - LearningRate (0.0001 - 0.01)

## Visual Indicators

When `ShowVisuals = true`:
- **Green Up Arrow**: Buy signal (confidence > threshold)
- Arrow tooltip shows confidence percentage

## Logging

The EA provides comprehensive logging:
- Initialization details
- Training progress (every 5 epochs)
- Validation accuracy
- Trade signals with confidence
- Position opening/closing with reasons
- Weekly retraining triggers
- Trade statistics on exit

Check the Experts tab in MT5 Terminal for logs.

## Performance Considerations

### Training Time
- Training on 8 weeks (80,000+ bars) takes approximately 2-10 minutes
- Depends on CPU speed and epoch count
- Uses early stopping to avoid unnecessary epochs

### Memory Usage
- Neural network weights: ~1.5 MB
- Training data: ~50 MB for 8 weeks
- MQL5 arrays are memory-efficient

### Tick Processing
- Only processes on new bar formation (not every tick)
- Typical: 288 bars per day (5-minute timeframe)
- Prediction takes <1ms per bar

## Edge Cases & Error Handling

The EA handles:
- ✅ Insufficient historical data (prints warning, skips training)
- ✅ Market closed (skips trading)
- ✅ Symbol not available (initialization fails gracefully)
- ✅ File I/O errors (uses initialized weights)
- ✅ Order execution failures (logs errors with reasons)
- ✅ Invalid parameters (validates on initialization)

## Troubleshooting

### EA Not Trading
1. Check AutoTrading is enabled (green button in toolbar)
2. Verify symbol name matches (BTCUSD or BTCUSDT)
3. Check if enough historical data is available (2+ months)
4. Review Experts log for error messages
5. Ensure market is open (crypto markets are 24/7)

### Compilation Errors
- Ensure all `.mqh` files are in correct directories
- Check MQL5 version (requires build 2361 or higher)
- Try closing/reopening MetaEditor

### Training Fails
- Check internet connection for data download
- Increase `RetrainWeeks` parameter if data is insufficient
- Review log for specific error messages

### Poor Performance
- Try adjusting `PredictionThreshold` (0.60-0.75)
- Reduce `PositionSizePct` for lower risk
- Increase `RetrainWeeks` for more training data
- Optimize using Strategy Tester genetic algorithm

## Customization

### Changing Architecture
Edit `NeuralNetwork.mqh` constructor:
```cpp
CNeuralNetwork(int input_size=120, 
               int h1=512,    // Change layer sizes here
               int h2=256, 
               int h3=128,
               ...)
```

### Adding Features
Edit `DataPipeline.mqh` → `FetchData()`:
1. Add indicator calculation
2. Resize features array to include new feature
3. Update input_size in EA (120 → new_size)

### Different Strategy
Edit `TradeManager.mqh` → `ExecuteTrade()`:
- Implement short positions
- Add stop-loss/take-profit
- Modify entry/exit logic

## Credits

Based on the Python implementation in `copilot/train-3-layer-ann-model` branch.

## License

This code is provided for educational and research purposes.

## Disclaimer

**TRADING RISK WARNING**: Trading cryptocurrencies and forex involves substantial risk of loss. Past performance does not guarantee future results. This EA is provided for educational purposes only. Use at your own risk. Always test thoroughly in demo account before live trading.
