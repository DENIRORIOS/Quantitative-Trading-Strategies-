# Quick Start Guide - BTC ANN Expert Advisor

## 🚀 Installation in 5 Minutes

### Step 1: Locate MT5 Data Folder
1. Open MetaTrader 5
2. Click **File** → **Open Data Folder**
3. Navigate to the `MQL5` folder

### Step 2: Copy Files
Copy the entire `mql5/` folder contents from this repository to your MT5 data folder:

```
Your MT5 Data Folder/
└── MQL5/
    ├── Experts/
    │   └── BTC_ANN_EA.mq5          ← Copy this
    └── Include/
        └── ANN/                     ← Copy this entire folder
            ├── NeuralNetwork.mqh
            ├── DataPipeline.mqh
            └── TradeManager.mqh
```

**Quick Copy Command (if using command line):**
```bash
# From repository root
cp -r mql5/Experts/* <MT5_DATA_FOLDER>/MQL5/Experts/
cp -r mql5/Include/* <MT5_DATA_FOLDER>/MQL5/Include/
```

### Step 3: Compile the EA
1. Open **MetaEditor** (press F4 in MT5)
2. In Navigator, find `Experts → BTC_ANN_EA.mq5`
3. Double-click to open it
4. Press **F7** or click the **Compile** button
5. Check **Toolbox** window at bottom - should say "0 error(s), 0 warning(s)"

### Step 4: Attach to Chart
1. In MT5, open a **BTCUSD** or **BTCUSDT** chart
2. Right-click chart → **Timeframe** → **M5** (5 minutes)
3. In **Navigator** panel (Ctrl+N), expand **Expert Advisors**
4. Drag **BTC_ANN_EA** onto the chart
5. A settings dialog appears - use defaults for first run
6. Click **OK**

### Step 5: Enable AutoTrading
1. Look for the **AutoTrading** button in the toolbar (red/green icon)
2. Click it - should turn GREEN
3. Check **Experts** tab in Terminal window (Ctrl+T)
4. You should see initialization messages

## ✅ Verify It's Working

### Check the Log
In Terminal window → **Experts** tab, you should see:
```
Initializing BTC ANN Expert Advisor v1.00
Trading Symbol: BTCUSD
Initializing neural network (512-256-128)...
Neural network weights initialized with He initialization
Initialization completed successfully
```

### See Visual Arrows
If `ShowVisuals = true`, you'll see:
- **Green arrows** on chart when buy signals occur
- Hover over arrow to see confidence percentage

### Monitor Trading
In Terminal window → **Trade** tab, you'll see:
- Open positions when EA takes trades
- P&L in real-time

## 📊 First Run Behavior

**Important**: On first run, the EA has **untrained weights**. You have two options:

### Option A: Wait for Weekly Retrain (Automatic)
- The EA will automatically retrain after 7 days
- Uses last 8 weeks of historical data
- Training takes 2-10 minutes
- Weights are saved and reused

### Option B: Force Immediate Training (Manual)
To train immediately, modify the EA temporarily:

1. Open `BTC_ANN_EA.mq5` in MetaEditor
2. In `OnInit()`, after line that loads weights, add:
   ```cpp
   // Force training on first run
   TrainModel();
   ```
3. Recompile (F7) and restart EA
4. Check Experts log - training will start immediately
5. Remove this line after first successful training

## ⚙️ Recommended Settings for First Run

### Conservative (Low Risk)
```
PredictionThreshold = 0.70        // Higher threshold = fewer trades
PositionSizePct     = 0.005       // 0.5% of equity
```

### Balanced (Medium Risk) - **DEFAULT**
```
PredictionThreshold = 0.65        // 65% confidence
PositionSizePct     = 0.01        // 1% of equity
```

### Aggressive (Higher Risk)
```
PredictionThreshold = 0.60        // More trades
PositionSizePct     = 0.02        // 2% of equity
```

## 🧪 Test Before Live Trading

### Demo Account Testing
**ALWAYS test on demo first!**

1. Open demo account in MT5 (File → Open Account → Demo)
2. Run EA on demo for at least 1-2 weeks
3. Monitor performance and tune parameters
4. Only move to live after successful demo testing

### Strategy Tester Backtest
1. Press **Ctrl+R** to open Strategy Tester
2. Select `BTC_ANN_EA` expert
3. Symbol: **BTCUSD** or **BTCUSDT**
4. Period: Last **3 months**
5. Timeframe: **M5** (5 minutes)
6. Model: **Every tick** (most accurate)
7. Click **Start**

**Note**: First backtest will train the model (2-10 min), then trade based on trained weights.

## 📈 What to Expect

### Training Phase
When weekly retraining triggers, you'll see:
```
Starting model training...
Fetching 8 weeks of historical data
Training samples: 11000
Test samples: 4700
Epoch 1/50 - Train Loss: 0.6931, Val Loss: 0.6925, Val Accuracy: 52.00%
Epoch 5/50 - Train Loss: 0.6523, Val Loss: 0.6480, Val Accuracy: 58.50%
...
Early stopping triggered at epoch 35 (patience=10)
Training completed!
Best validation loss: 0.4823
```

### Trading Phase
On each 5-minute bar close:
```
Prediction: 72.50% (Up: 72.50%, Down: 27.50%)
Opening LONG position: Lot=0.01, Price=45234.50, Confidence=72.50%
Long position opened successfully, ticket: 123456
```

When closing:
```
Closing position: Ticket=123456, Profit=45.23, Reason=Bearish signal: 68.00%
Position closed successfully
```

## 🔍 Troubleshooting

### Problem: EA not trading
**Solutions:**
- ✅ Enable AutoTrading (green button)
- ✅ Check symbol name matches (BTCUSD or BTCUSDT)
- ✅ Ensure 5-minute timeframe (M5)
- ✅ Verify enough historical data (right-click chart → Refresh)
- ✅ Check Experts log for errors

### Problem: Compilation errors
**Solutions:**
- ✅ Ensure all 4 files copied to correct locations
- ✅ Check `Include/ANN/` folder exists
- ✅ Restart MetaEditor
- ✅ Update MT5 to latest version (build 2361+)

### Problem: "Not enough data for training"
**Solutions:**
- ✅ Let MT5 download more history (right-click chart → Refresh)
- ✅ Reduce `RetrainWeeks` from 8 to 4
- ✅ Wait a few minutes for data to download
- ✅ Try different broker with more history

### Problem: Slow training
**Solutions:**
- ✅ Reduce `TrainingEpochs` from 50 to 30
- ✅ Reduce `RetrainWeeks` from 8 to 4
- ✅ Normal: training takes 2-10 minutes depending on CPU

## 📞 Support

### Log Files Location
```
<MT5_DATA_FOLDER>/MQL5/Logs/
```

### Weight Files Location
```
<MT5_DATA_FOLDER>/MQL5/Files/BTC_ANN_weights.bin
```

### Check System Resources
- **Memory**: EA uses ~50-100 MB during training
- **CPU**: Training is CPU-intensive (2-10 minutes)
- **Disk**: Weight file is ~1.5 MB

## 🎯 Next Steps

1. ✅ Run on demo account for 1-2 weeks
2. ✅ Monitor win rate and profitability
3. ✅ Tune `PredictionThreshold` if needed (0.60-0.75)
4. ✅ Adjust `PositionSizePct` based on risk tolerance
5. ✅ Use Strategy Tester optimization for best parameters
6. ✅ Only go live after thorough testing

## 🎓 Learning Resources

### Understanding the Neural Network
- See `mql5/README.md` for detailed architecture explanation
- Check code comments in `NeuralNetwork.mqh` for math details
- Review `DataPipeline.mqh` for indicator calculations

### Optimizing Parameters
Use Strategy Tester's genetic algorithm to optimize:
- `PredictionThreshold`: 0.60 to 0.75 (step 0.05)
- `PositionSizePct`: 0.005 to 0.02 (step 0.005)
- `LearningRate`: 0.0001 to 0.01 (step 0.001)

## ⚠️ Important Reminders

- 🚨 **Always test on DEMO first**
- 🚨 **Never risk more than you can afford to lose**
- 🚨 **Past performance ≠ future results**
- 🚨 **Use proper position sizing**
- 🚨 **Monitor the EA regularly**
- 🚨 **Keep MT5 updated**

## 🔄 Weekly Maintenance

The EA is mostly hands-off, but recommended weekly checks:
1. Review trade performance in Journal
2. Check if retraining occurred successfully
3. Monitor win rate and adjust threshold if needed
4. Verify account equity is sufficient for trading
5. Check for any error messages in log

---

**Ready to Go!** Your EA should now be running. Watch the Experts log and Trade tab for activity. Remember: test thoroughly before live trading! 🎉
