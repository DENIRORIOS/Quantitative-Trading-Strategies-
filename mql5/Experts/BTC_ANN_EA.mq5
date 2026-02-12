//+------------------------------------------------------------------+
//|                                                   BTC_ANN_EA.mq5 |
//|                  Bitcoin Trading Expert Advisor with 3-Layer ANN |
//|                                                                  |
//| Expert Advisor implementing a 3-layer feedforward neural network |
//| for Bitcoin directional prediction with weekly retraining.       |
//|                                                                  |
//| Features:                                                         |
//| - 3-layer ANN (512-256-128 neurons) built from scratch          |
//| - ReLU activation for hidden layers, Sigmoid for output         |
//| - Adam optimizer with learning rate scheduling                  |
//| - Dropout regularization during training                        |
//| - Technical indicators: RSI(14), MACD(12,26,9), Volume Delta    |
//| - 20-period sequence inputs (6 features × 20 = 120 inputs)      |
//| - 65% confidence threshold for trades                           |
//| - Weekly retraining on 8 weeks of 5-minute data                 |
//| - Long-only strategy with position sizing (0.5-1% of equity)    |
//+------------------------------------------------------------------+

#property copyright "BTC Trading ANN"
#property link      ""
#property version   "1.00"
#property strict

// Include our custom classes
#include <ANN/NeuralNetwork.mqh>
#include <ANN/DataPipeline.mqh>
#include <ANN/TradeManager.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Neural Network Parameters ==="
input int       SequenceLength = 20;              // Length of input sequences
input double    LearningRate = 0.001;             // Learning rate for Adam optimizer
input double    DropoutRate1 = 0.3;               // Dropout rate for layer 1
input double    DropoutRate2 = 0.3;               // Dropout rate for layer 2
input double    DropoutRate3 = 0.2;               // Dropout rate for layer 3

input group "=== Trading Parameters ==="
input double    PredictionThreshold = 0.65;       // Confidence threshold (0.65 = 65%)
input double    PositionSizePct = 0.01;           // Position size (% of equity, 0.01 = 1%)
input int       MagicNumber = 12345;              // Magic number for orders

input group "=== Technical Indicators ==="
input int       RSI_Period = 14;                  // RSI period
input int       MACD_Fast = 12;                   // MACD fast EMA period
input int       MACD_Slow = 26;                   // MACD slow EMA period
input int       MACD_Signal = 9;                  // MACD signal period

input group "=== Training Parameters ==="
input int       TrainingEpochs = 50;              // Maximum training epochs
input int       EarlyStopPatience = 10;           // Early stopping patience
input int       RetrainWeeks = 8;                 // Weeks of history for retraining
input ENUM_TIMEFRAMES TradingTimeframe = PERIOD_M5; // Trading timeframe

input group "=== Visual & Optimization ==="
input bool      ShowVisuals = true;               // Show buy/sell arrows on chart
input bool      EnableWeeklyRetrain = true;       // Enable automatic weekly retraining

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CNeuralNetwork* g_network = NULL;
CDataPipeline* g_pipeline = NULL;
CTradeManager* g_trader = NULL;

datetime g_last_retrain_time = 0;
datetime g_last_bar_time = 0;
bool g_is_initialized = false;

const string WEIGHTS_FILE = "BTC_ANN_weights.bin";

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=======================================================");
    Print("Initializing BTC ANN Expert Advisor v1.00");
    Print("=======================================================");
    
    // Validate inputs
    if(PredictionThreshold < 0.5 || PredictionThreshold > 1.0)
    {
        Print("ERROR: PredictionThreshold must be between 0.5 and 1.0");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    if(PositionSizePct < 0.001 || PositionSizePct > 0.1)
    {
        Print("ERROR: PositionSizePct must be between 0.001 (0.1%) and 0.1 (10%)");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    // Get symbol
    string symbol = Symbol();
    Print("Trading Symbol: ", symbol);
    Print("Timeframe: ", EnumToString(TradingTimeframe));
    
    // Initialize neural network
    Print("Initializing neural network (512-256-128)...");
    g_network = new CNeuralNetwork(120, 512, 256, 128, LearningRate, 
                                   DropoutRate1, DropoutRate2, DropoutRate3);
    
    // Try to load existing weights
    if(g_network.LoadWeights(WEIGHTS_FILE))
    {
        Print("Loaded trained weights from: ", WEIGHTS_FILE);
    }
    else
    {
        Print("No existing weights found, using initialized weights");
        Print("The EA will train the model on first run or weekly retrain");
    }
    
    // Initialize data pipeline
    Print("Initializing data pipeline...");
    g_pipeline = new CDataPipeline(symbol, TradingTimeframe, SequenceLength,
                                   RSI_Period, MACD_Fast, MACD_Slow, MACD_Signal);
    
    // Initialize trade manager
    Print("Initializing trade manager...");
    g_trader = new CTradeManager(symbol, PositionSizePct, PredictionThreshold, MagicNumber);
    
    // Set up timer for retraining check (check every minute)
    if(EnableWeeklyRetrain)
    {
        EventSetTimer(60);  // 60 seconds = 1 minute
        Print("Weekly retraining enabled (checking every minute)");
    }
    
    g_is_initialized = true;
    g_last_retrain_time = TimeCurrent();
    
    Print("=======================================================");
    Print("Initialization completed successfully");
    Print("Neural Network: 120 inputs -> 512 -> 256 -> 128 -> 1 output");
    Print("Confidence Threshold: ", PredictionThreshold * 100, "%");
    Print("Position Size: ", PositionSizePct * 100, "%");
    Print("=======================================================");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("=======================================================");
    Print("Deinitializing BTC ANN EA...");
    Print("Reason: ", reason);
    
    // Save current weights
    if(g_network != NULL)
    {
        g_network.SaveWeights(WEIGHTS_FILE);
        delete g_network;
        g_network = NULL;
    }
    
    // Print final statistics
    if(g_trader != NULL)
    {
        g_trader.PrintStatistics();
        delete g_trader;
        g_trader = NULL;
    }
    
    if(g_pipeline != NULL)
    {
        delete g_pipeline;
        g_pipeline = NULL;
    }
    
    EventKillTimer();
    
    Print("Deinitialization completed");
    Print("=======================================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(!g_is_initialized)
        return;
    
    // Check if this is a new bar (only trade on new bars)
    datetime current_bar_time = iTime(Symbol(), TradingTimeframe, 0);
    if(current_bar_time == g_last_bar_time)
        return;  // Same bar, skip
    
    g_last_bar_time = current_bar_time;
    
    // Check if market is open
    if(!g_trader.IsMarketOpen())
    {
        Print("Market is closed, skipping tick");
        return;
    }
    
    // Get current market sequence
    double input_sequence[];
    if(!g_pipeline.GetCurrentSequence(input_sequence))
    {
        Print("Failed to get current market sequence");
        return;
    }
    
    // Make prediction
    double prediction = g_network.Predict(input_sequence);
    
    // Log prediction
    static datetime last_log_time = 0;
    if(TimeCurrent() - last_log_time > 300)  // Log every 5 minutes
    {
        PrintFormat("Prediction: %.2f%% (Up: %.2f%%, Down: %.2f%%)", 
                    prediction * 100, prediction * 100, (1.0 - prediction) * 100);
        last_log_time = TimeCurrent();
    }
    
    // Execute trading logic
    g_trader.ExecuteTrade(prediction);
    
    // Show visual arrow if enabled
    if(ShowVisuals && prediction > PredictionThreshold)
    {
        CreateSignalArrow(prediction, current_bar_time);
    }
}

//+------------------------------------------------------------------+
//| Timer function - check for weekly retraining                     |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(!EnableWeeklyRetrain)
        return;
    
    // Check if a week has passed since last retraining
    datetime current_time = TimeCurrent();
    int seconds_since_retrain = (int)(current_time - g_last_retrain_time);
    int seconds_per_week = 7 * 24 * 60 * 60;  // 7 days × 24 hours × 60 minutes × 60 seconds
    
    if(seconds_since_retrain >= seconds_per_week)
    {
        Print("=======================================================");
        Print("Weekly retraining triggered");
        Print("=======================================================");
        
        bool success = TrainModel();
        
        if(success)
        {
            g_last_retrain_time = current_time;
            Print("Retraining completed successfully");
            Print("Next retraining scheduled in 7 days");
        }
        else
        {
            Print("Retraining failed, will retry next week");
        }
        
        Print("=======================================================");
    }
}

//+------------------------------------------------------------------+
//| Train the neural network                                         |
//+------------------------------------------------------------------+
bool TrainModel()
{
    Print("Starting model training...");
    Print("Fetching ", RetrainWeeks, " weeks of historical data");
    
    // Prepare training data with 70/30 split
    double train_X[][][], train_y[], test_X[][][], test_y[];
    
    if(!g_pipeline.PrepareTrainingData(RetrainWeeks, 0.7, train_X, train_y, test_X, test_y))
    {
        Print("ERROR: Failed to prepare training data");
        return false;
    }
    
    int n_train = ArraySize(train_y);
    int n_test = ArraySize(test_y);
    
    Print("Training samples: ", n_train);
    Print("Test samples: ", n_test);
    
    // Training loop with early stopping
    double best_val_loss = 1000000.0;
    int patience_counter = 0;
    
    g_network.SetTrainingMode(true);
    
    for(int epoch = 0; epoch < TrainingEpochs; epoch++)
    {
        double epoch_loss = 0.0;
        int n_batches = 0;
        
        // Training phase
        for(int i = 0; i < n_train; i++)
        {
            // Flatten sequence
            double input[];
            g_pipeline.FlattenSequence(train_X[i], input);
            
            // Forward pass
            double prediction = g_network.Forward(input);
            
            // Calculate loss
            double loss = g_network.CalculateLoss(prediction, train_y[i]);
            epoch_loss += loss;
            
            // Backward pass - calculate gradients
            double dW1[][], db1[], dW2[][], db2[], dW3[][], db3[], dW4[][], db4[];
            g_network.Backward(input, train_y[i], dW1, db1, dW2, db2, dW3, db3, dW4, db4);
            
            // Update weights using Adam
            g_network.UpdateWeightsAdam(dW1, db1, dW2, db2, dW3, db3, dW4, db4);
            
            n_batches++;
        }
        
        double avg_train_loss = epoch_loss / n_batches;
        
        // Validation phase
        g_network.SetTrainingMode(false);
        double val_loss = 0.0;
        int val_correct = 0;
        
        for(int i = 0; i < n_test; i++)
        {
            double input[];
            g_pipeline.FlattenSequence(test_X[i], input);
            
            double prediction = g_network.Predict(input);
            val_loss += g_network.CalculateLoss(prediction, test_y[i]);
            
            // Calculate accuracy
            double predicted_class = (prediction > 0.5) ? 1.0 : 0.0;
            if(predicted_class == test_y[i])
                val_correct++;
        }
        
        double avg_val_loss = val_loss / n_test;
        double val_accuracy = (double)val_correct / n_test * 100.0;
        
        // Log progress every 5 epochs
        if(epoch % 5 == 0 || epoch == TrainingEpochs - 1)
        {
            PrintFormat("Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, Val Accuracy: %.2f%%",
                        epoch + 1, TrainingEpochs, avg_train_loss, avg_val_loss, val_accuracy);
        }
        
        // Early stopping
        if(avg_val_loss < best_val_loss)
        {
            best_val_loss = avg_val_loss;
            patience_counter = 0;
            // Save best weights
            g_network.SaveWeights(WEIGHTS_FILE);
        }
        else
        {
            patience_counter++;
            if(patience_counter >= EarlyStopPatience)
            {
                PrintFormat("Early stopping triggered at epoch %d (patience=%d)", 
                            epoch + 1, EarlyStopPatience);
                break;
            }
        }
        
        // Re-enable training mode for next epoch
        g_network.SetTrainingMode(true);
    }
    
    // Load best weights
    g_network.LoadWeights(WEIGHTS_FILE);
    g_network.SetTrainingMode(false);
    
    Print("Training completed!");
    Print("Best validation loss: ", best_val_loss);
    
    return true;
}

//+------------------------------------------------------------------+
//| Create visual signal arrow on chart                              |
//+------------------------------------------------------------------+
void CreateSignalArrow(double prediction, datetime time)
{
    static datetime last_arrow_time = 0;
    if(time == last_arrow_time)
        return;  // Already created arrow for this bar
    
    string arrow_name = StringFormat("ANN_Arrow_%I64d", time);
    double price = iClose(Symbol(), TradingTimeframe, 0);
    
    // Delete old arrow if exists
    ObjectDelete(0, arrow_name);
    
    // Create new arrow
    if(prediction > PredictionThreshold)
    {
        // Buy signal
        ObjectCreate(0, arrow_name, OBJ_ARROW_UP, 0, time, price);
        ObjectSetInteger(0, arrow_name, OBJPROP_COLOR, clrLime);
        ObjectSetInteger(0, arrow_name, OBJPROP_WIDTH, 2);
        ObjectSetString(0, arrow_name, OBJPROP_TEXT, 
                       StringFormat("BUY %.1f%%", prediction * 100));
    }
    
    last_arrow_time = time;
}

//+------------------------------------------------------------------+
//| OnTester function for Strategy Tester optimization               |
//+------------------------------------------------------------------+
double OnTester()
{
    // Calculate custom metric for optimization
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Get trade statistics
    double trades = g_trader.GetTotalTrades();
    double win_rate = g_trader.GetWinRate();
    
    // Custom fitness: balance * win_rate / max(1, drawdown)
    double profit_factor = (balance > 10000) ? (balance / 10000.0) : 0.0;
    double fitness = profit_factor * (win_rate / 100.0);
    
    return fitness;
}

//+------------------------------------------------------------------+
//| OnChartEvent function for manual training trigger                |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    // You can add manual training trigger here if needed
    // For example, pressing a button on the chart to retrain the model
}
