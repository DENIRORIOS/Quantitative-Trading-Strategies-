//+------------------------------------------------------------------+
//|                                                 DataPipeline.mqh  |
//|                      Data Processing and Technical Indicators     |
//|                                                                  |
//| Handles:                                                          |
//| - Data fetching from MetaTrader (CopyRates, CopyClose, etc.)    |
//| - Technical indicator calculations (RSI, MACD, Volume Delta)     |
//| - Data normalization using online StandardScaler                 |
//| - Sequence creation for time series input (20 periods)           |
//+------------------------------------------------------------------+

#property copyright "BTC Trading ANN"
#property strict

//+------------------------------------------------------------------+
//| Online Standard Scaler for normalization                         |
//| Maintains running mean and standard deviation                    |
//+------------------------------------------------------------------+
class CStandardScaler
{
private:
    double m_mean[];
    double m_std[];
    double m_sum[];
    double m_sum_sq[];
    int m_count;
    int m_n_features;
    bool m_fitted;
    
public:
    CStandardScaler()
    {
        m_count = 0;
        m_fitted = false;
    }
    
    //+------------------------------------------------------------------+
    //| Fit the scaler with data                                        |
    //+------------------------------------------------------------------+
    void Fit(double &data[][], int n_samples, int n_features)
    {
        m_n_features = n_features;
        ArrayResize(m_mean, n_features);
        ArrayResize(m_std, n_features);
        ArrayResize(m_sum, n_features);
        ArrayResize(m_sum_sq, n_features);
        ArrayInitialize(m_sum, 0.0);
        ArrayInitialize(m_sum_sq, 0.0);
        
        // Calculate sum and sum of squares
        for(int i = 0; i < n_samples; i++)
        {
            for(int j = 0; j < n_features; j++)
            {
                m_sum[j] += data[i][j];
                m_sum_sq[j] += data[i][j] * data[i][j];
            }
        }
        
        m_count = n_samples;
        
        // Calculate mean and std
        for(int j = 0; j < n_features; j++)
        {
            m_mean[j] = m_sum[j] / m_count;
            double variance = (m_sum_sq[j] / m_count) - (m_mean[j] * m_mean[j]);
            m_std[j] = MathSqrt(MathMax(variance, 1e-8));  // Avoid division by zero
            if(m_std[j] < 1e-8) m_std[j] = 1.0;  // Avoid division by near-zero
        }
        
        m_fitted = true;
    }
    
    //+------------------------------------------------------------------+
    //| Transform data using fitted parameters                          |
    //+------------------------------------------------------------------+
    void Transform(double &data[][], int n_samples, int n_features)
    {
        if(!m_fitted)
        {
            Print("Error: Scaler not fitted yet!");
            return;
        }
        
        for(int i = 0; i < n_samples; i++)
            for(int j = 0; j < n_features; j++)
                data[i][j] = (data[i][j] - m_mean[j]) / m_std[j];
    }
    
    //+------------------------------------------------------------------+
    //| Transform single sample                                         |
    //+------------------------------------------------------------------+
    void TransformSample(double &sample[])
    {
        if(!m_fitted)
        {
            Print("Error: Scaler not fitted yet!");
            return;
        }
        
        for(int j = 0; j < m_n_features; j++)
            sample[j] = (sample[j] - m_mean[j]) / m_std[j];
    }
    
    //+------------------------------------------------------------------+
    //| Fit and transform in one step                                   |
    //+------------------------------------------------------------------+
    void FitTransform(double &data[][], int n_samples, int n_features)
    {
        Fit(data, n_samples, n_features);
        Transform(data, n_samples, n_features);
    }
    
    bool IsFitted() { return m_fitted; }
};

//+------------------------------------------------------------------+
//| Data Pipeline Class                                              |
//| Manages all data operations for the neural network               |
//+------------------------------------------------------------------+
class CDataPipeline
{
private:
    string m_symbol;
    ENUM_TIMEFRAMES m_timeframe;
    int m_sequence_length;
    int m_rsi_period;
    int m_macd_fast;
    int m_macd_slow;
    int m_macd_signal;
    
    CStandardScaler m_scaler;
    
public:
    CDataPipeline(string symbol, ENUM_TIMEFRAMES timeframe, int sequence_length=20,
                  int rsi_period=14, int macd_fast=12, int macd_slow=26, int macd_signal=9)
    {
        m_symbol = symbol;
        m_timeframe = timeframe;
        m_sequence_length = sequence_length;
        m_rsi_period = rsi_period;
        m_macd_fast = macd_fast;
        m_macd_slow = macd_slow;
        m_macd_signal = macd_signal;
    }
    
    //+------------------------------------------------------------------+
    //| Calculate RSI manually                                          |
    //| RSI = 100 - (100 / (1 + RS))                                    |
    //| RS = Average Gain / Average Loss over period                    |
    //+------------------------------------------------------------------+
    void CalculateRSI(double &close[], int size, double &rsi[])
    {
        ArrayResize(rsi, size);
        ArrayInitialize(rsi, 50.0);  // Default neutral value
        
        if(size < m_rsi_period + 1)
            return;
        
        // Calculate price changes
        double gains = 0.0;
        double losses = 0.0;
        
        // Initial average gain and loss
        for(int i = 1; i <= m_rsi_period; i++)
        {
            double change = close[i] - close[i-1];
            if(change > 0)
                gains += change;
            else
                losses += MathAbs(change);
        }
        
        double avg_gain = gains / m_rsi_period;
        double avg_loss = losses / m_rsi_period;
        
        // Calculate RSI for first period
        double rs = (avg_loss > 0) ? (avg_gain / avg_loss) : 0.0;
        rsi[m_rsi_period] = 100.0 - (100.0 / (1.0 + rs));
        
        // Calculate RSI for remaining periods using smoothed averages
        for(int i = m_rsi_period + 1; i < size; i++)
        {
            double change = close[i] - close[i-1];
            double gain = (change > 0) ? change : 0.0;
            double loss = (change < 0) ? MathAbs(change) : 0.0;
            
            // Smoothed moving average (Wilder's smoothing)
            avg_gain = (avg_gain * (m_rsi_period - 1) + gain) / m_rsi_period;
            avg_loss = (avg_loss * (m_rsi_period - 1) + loss) / m_rsi_period;
            
            rs = (avg_loss > 0) ? (avg_gain / avg_loss) : 0.0;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    //+------------------------------------------------------------------+
    //| Calculate EMA (Exponential Moving Average)                      |
    //+------------------------------------------------------------------+
    void CalculateEMA(double &price[], int size, int period, double &ema[])
    {
        ArrayResize(ema, size);
        
        if(size < period)
            return;
        
        double multiplier = 2.0 / (period + 1);
        
        // Calculate initial SMA for first EMA value
        double sum = 0.0;
        for(int i = 0; i < period; i++)
            sum += price[i];
        
        ema[period - 1] = sum / period;
        
        // Fill earlier values with first calculated EMA
        for(int i = 0; i < period - 1; i++)
            ema[i] = ema[period - 1];
        
        // Calculate EMA for remaining values
        for(int i = period; i < size; i++)
            ema[i] = (price[i] - ema[i-1]) * multiplier + ema[i-1];
    }
    
    //+------------------------------------------------------------------+
    //| Calculate MACD manually                                         |
    //| MACD = EMA(fast) - EMA(slow)                                    |
    //| Signal = EMA(MACD, signal_period)                               |
    //| Histogram = MACD - Signal                                       |
    //+------------------------------------------------------------------+
    void CalculateMACD(double &close[], int size, double &macd[], double &signal[], double &histogram[])
    {
        ArrayResize(macd, size);
        ArrayResize(signal, size);
        ArrayResize(histogram, size);
        
        // Calculate fast and slow EMAs
        double ema_fast[], ema_slow[];
        CalculateEMA(close, size, m_macd_fast, ema_fast);
        CalculateEMA(close, size, m_macd_slow, ema_slow);
        
        // Calculate MACD line
        for(int i = 0; i < size; i++)
            macd[i] = ema_fast[i] - ema_slow[i];
        
        // Calculate signal line (EMA of MACD)
        CalculateEMA(macd, size, m_macd_signal, signal);
        
        // Calculate histogram
        for(int i = 0; i < size; i++)
            histogram[i] = macd[i] - signal[i];
    }
    
    //+------------------------------------------------------------------+
    //| Calculate Volume Delta                                          |
    //| Approximation: positive delta on up bars, negative on down bars |
    //+------------------------------------------------------------------+
    void CalculateVolumeDelta(double &close[], long &volume[], int size, double &volume_delta[])
    {
        ArrayResize(volume_delta, size);
        volume_delta[0] = 0.0;
        
        for(int i = 1; i < size; i++)
        {
            // Approximate buy/sell volume based on price direction
            if(close[i] > close[i-1])
                volume_delta[i] = (double)volume[i];  // Buying pressure
            else if(close[i] < close[i-1])
                volume_delta[i] = -(double)volume[i]; // Selling pressure
            else
                volume_delta[i] = 0.0;  // Neutral
        }
    }
    
    //+------------------------------------------------------------------+
    //| Fetch historical data and calculate all features                |
    //| Returns: number of bars fetched, -1 on error                    |
    //+------------------------------------------------------------------+
    int FetchData(int bars_needed, double &features[][], double &close_prices[])
    {
        // Request bars
        int bars = Bars(m_symbol, m_timeframe);
        if(bars < bars_needed)
        {
            Print("Not enough bars available. Need: ", bars_needed, ", Available: ", bars);
            return -1;
        }
        
        // Fetch OHLCV data
        MqlRates rates[];
        int copied = CopyRates(m_symbol, m_timeframe, 0, bars_needed, rates);
        if(copied <= 0)
        {
            Print("Failed to copy rates, error: ", GetLastError());
            return -1;
        }
        
        // Extract close prices and volume
        ArrayResize(close_prices, copied);
        long volume[];
        ArrayResize(volume, copied);
        
        for(int i = 0; i < copied; i++)
        {
            close_prices[i] = rates[i].close;
            volume[i] = rates[i].tick_volume;
        }
        
        // Calculate technical indicators
        double rsi[], macd[], macd_signal[], macd_histogram[], volume_delta[];
        CalculateRSI(close_prices, copied, rsi);
        CalculateMACD(close_prices, copied, macd, macd_signal, macd_histogram);
        CalculateVolumeDelta(close_prices, volume, copied, volume_delta);
        
        // Prepare features array: [close, rsi, macd, macd_signal, macd_histogram, volume_delta]
        // 6 features per bar
        ArrayResize(features, copied);
        for(int i = 0; i < copied; i++)
        {
            ArrayResize(features[i], 6);
            features[i][0] = close_prices[i];
            features[i][1] = rsi[i];
            features[i][2] = macd[i];
            features[i][3] = macd_signal[i];
            features[i][4] = macd_histogram[i];
            features[i][5] = volume_delta[i];
        }
        
        return copied;
    }
    
    //+------------------------------------------------------------------+
    //| Create sequences for training                                   |
    //| Each sequence contains sequence_length bars of 6 features       |
    //| Target: 1 if next close > current close, 0 otherwise            |
    //+------------------------------------------------------------------+
    int CreateSequences(double &features[][], int n_bars, double &X[][][], double &y[])
    {
        int n_sequences = n_bars - m_sequence_length;
        if(n_sequences <= 0)
        {
            Print("Not enough bars to create sequences");
            return 0;
        }
        
        ArrayResize(X, n_sequences);
        ArrayResize(y, n_sequences);
        
        for(int i = 0; i < n_sequences; i++)
        {
            // Create sequence of past sequence_length bars
            ArrayResize(X[i], m_sequence_length);
            for(int j = 0; j < m_sequence_length; j++)
            {
                ArrayResize(X[i][j], 6);
                for(int k = 0; k < 6; k++)
                    X[i][j][k] = features[i + j][k];
            }
            
            // Target: 1 if price goes up, 0 if price goes down or stays same
            int current_idx = i + m_sequence_length - 1;
            int next_idx = i + m_sequence_length;
            y[i] = (features[next_idx][0] > features[current_idx][0]) ? 1.0 : 0.0;
        }
        
        return n_sequences;
    }
    
    //+------------------------------------------------------------------+
    //| Flatten sequences into 1D input for neural network              |
    //| Shape: [sequence_length, n_features] -> [sequence_length * n_features] |
    //+------------------------------------------------------------------+
    void FlattenSequence(double &sequence[][], double &flattened[])
    {
        int seq_len = ArraySize(sequence);
        ArrayResize(flattened, seq_len * 6);
        
        int idx = 0;
        for(int i = 0; i < seq_len; i++)
            for(int j = 0; j < 6; j++)
                flattened[idx++] = sequence[i][j];
    }
    
    //+------------------------------------------------------------------+
    //| Normalize features using the scaler                             |
    //+------------------------------------------------------------------+
    void NormalizeFeatures(double &features[][], int n_samples)
    {
        if(!m_scaler.IsFitted())
            m_scaler.FitTransform(features, n_samples, 6);
        else
            m_scaler.Transform(features, n_samples, 6);
    }
    
    //+------------------------------------------------------------------+
    //| Normalize single sequence                                       |
    //+------------------------------------------------------------------+
    void NormalizeSequence(double &sequence[][])
    {
        int seq_len = ArraySize(sequence);
        if(!m_scaler.IsFitted())
        {
            Print("Warning: Scaler not fitted, sequence may not be properly normalized");
            return;
        }
        
        for(int i = 0; i < seq_len; i++)
            m_scaler.TransformSample(sequence[i]);
    }
    
    //+------------------------------------------------------------------+
    //| Get current market sequence for prediction                      |
    //+------------------------------------------------------------------+
    bool GetCurrentSequence(double &flattened_sequence[])
    {
        double features[][], close_prices[];
        int bars_fetched = FetchData(m_sequence_length + 50, features, close_prices);
        
        if(bars_fetched < m_sequence_length)
        {
            Print("Not enough data for current sequence");
            return false;
        }
        
        // Get the most recent sequence
        double sequence[][];
        ArrayResize(sequence, m_sequence_length);
        for(int i = 0; i < m_sequence_length; i++)
        {
            int idx = bars_fetched - m_sequence_length + i;
            ArrayResize(sequence[i], 6);
            for(int j = 0; j < 6; j++)
                sequence[i][j] = features[idx][j];
        }
        
        // Normalize the sequence
        NormalizeSequence(sequence);
        
        // Flatten for network input
        FlattenSequence(sequence, flattened_sequence);
        
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Prepare training data with train/test split                     |
    //+------------------------------------------------------------------+
    bool PrepareTrainingData(int weeks, double train_ratio, 
                             double &train_X[][][], double &train_y[],
                             double &test_X[][][], double &test_y[])
    {
        // Calculate bars needed (5-minute bars for 24/7 crypto markets: 288 per day, 2016 per week)
        int bars_per_week = 288 * 7;  // 5-minute bars in a week (24/7 trading)
        int bars_needed = bars_per_week * weeks + m_sequence_length + 50;
        
        Print("Fetching ", bars_needed, " bars for ", weeks, " weeks of training data");
        
        // Fetch historical data
        double features[][], close_prices[];
        int n_bars = FetchData(bars_needed, features, close_prices);
        
        if(n_bars < bars_needed / 2)
        {
            Print("Warning: Could only fetch ", n_bars, " bars, need ", bars_needed);
        }
        
        if(n_bars < m_sequence_length + 100)
        {
            Print("Not enough data for training");
            return false;
        }
        
        // Normalize features
        NormalizeFeatures(features, n_bars);
        
        // Create sequences
        double X[][][], y[];
        int n_sequences = CreateSequences(features, n_bars, X, y);
        
        if(n_sequences < 100)
        {
            Print("Not enough sequences for training: ", n_sequences);
            return false;
        }
        
        // Split into train and test
        int train_size = (int)(n_sequences * train_ratio);
        int test_size = n_sequences - train_size;
        
        ArrayResize(train_X, train_size);
        ArrayResize(train_y, train_size);
        ArrayResize(test_X, test_size);
        ArrayResize(test_y, test_size);
        
        // Copy training data
        for(int i = 0; i < train_size; i++)
        {
            train_X[i] = X[i];
            train_y[i] = y[i];
        }
        
        // Copy test data
        for(int i = 0; i < test_size; i++)
        {
            test_X[i] = X[train_size + i];
            test_y[i] = y[train_size + i];
        }
        
        Print("Training data prepared: ", train_size, " train samples, ", test_size, " test samples");
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol and timeframe                                        |
    //+------------------------------------------------------------------+
    string GetSymbol() { return m_symbol; }
    ENUM_TIMEFRAMES GetTimeframe() { return m_timeframe; }
};
