"""
Data preprocessing and feature engineering for ML models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Preprocess data for neural network training"""
    
    def __init__(self, sequence_length=20):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Number of time periods to look back (default: 20)
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def create_sequences(self, df):
        """
        Create sequences for time series prediction
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        feature_columns = ['close', 'rsi', 'macd', 'macd_signal', 
                          'macd_histogram', 'volume_delta']
        
        # Extract features
        features = df[feature_columns].values
        targets = df['target'].values
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            # Get sequence of past sequence_length periods
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
            
        return np.array(X), np.array(y)
    
    def create_rolling_windows(self, df, train_size=0.7, step_size=30):
        """
        Create rolling windows for out-of-sample validation
        
        Args:
            df: DataFrame with features and target
            train_size: Proportion of data for training (default: 0.7)
            step_size: Number of periods to roll forward (default: 30 days of 5m data)
            
        Returns:
            List of (train_X, train_y, test_X, test_y) tuples
        """
        X, y = self.create_sequences(df)
        
        windows = []
        total_samples = len(X)
        train_samples = int(total_samples * train_size)
        
        start_idx = 0
        
        while start_idx + train_samples < total_samples:
            end_train_idx = start_idx + train_samples
            end_test_idx = min(end_train_idx + step_size, total_samples)
            
            # Training data
            train_X = X[start_idx:end_train_idx]
            train_y = y[start_idx:end_train_idx]
            
            # Test data
            test_X = X[end_train_idx:end_test_idx]
            test_y = y[end_train_idx:end_test_idx]
            
            # Fit scaler on training data
            scaler = StandardScaler()
            
            # Reshape for scaling
            n_samples, n_timesteps, n_features = train_X.shape
            train_X_reshaped = train_X.reshape(-1, n_features)
            train_X_scaled = scaler.fit_transform(train_X_reshaped)
            train_X_scaled = train_X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Scale test data using same scaler
            test_X_reshaped = test_X.reshape(-1, n_features)
            test_X_scaled = scaler.transform(test_X_reshaped)
            test_X_scaled = test_X_scaled.reshape(test_X.shape[0], n_timesteps, n_features)
            
            windows.append((train_X_scaled, train_y, test_X_scaled, test_y))
            
            # Move window forward
            start_idx += step_size
            
        return windows
    
    def prepare_data(self, df, train_split=0.8):
        """
        Prepare data for training with simple train/test split
        
        Args:
            df: DataFrame with features and target
            train_split: Proportion for training (default: 0.8)
            
        Returns:
            Tuple of (train_X, train_y, test_X, test_y)
        """
        X, y = self.create_sequences(df)
        
        # Split into train and test
        split_idx = int(len(X) * train_split)
        train_X, test_X = X[:split_idx], X[split_idx:]
        train_y, test_y = y[:split_idx], y[split_idx:]
        
        # Scale features
        n_samples, n_timesteps, n_features = train_X.shape
        train_X_reshaped = train_X.reshape(-1, n_features)
        self.scaler.fit(train_X_reshaped)
        
        train_X_scaled = self.scaler.transform(train_X_reshaped)
        train_X_scaled = train_X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        test_X_reshaped = test_X.reshape(-1, n_features)
        test_X_scaled = self.scaler.transform(test_X_reshaped)
        test_X_scaled = test_X_scaled.reshape(test_X.shape[0], n_timesteps, n_features)
        
        return train_X_scaled, train_y, test_X_scaled, test_y
