"""
Technical indicators calculation for trading features
"""
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for trading strategy"""
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            data: Price series (close prices)
            period: RSI period (default: 14)
            
        Returns:
            RSI values as pandas Series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series (close prices)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_volume_delta(df):
        """
        Calculate volume delta (volume change)
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            Volume delta as pandas Series
        """
        return df['volume'].diff()
    
    @staticmethod
    def add_all_indicators(df):
        """
        Add all technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Calculate RSI(14)
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], period=14)
        
        # Calculate MACD(12, 26, 9)
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(
            df['close'], fast=12, slow=26, signal=9
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Calculate volume delta
        df['volume_delta'] = TechnicalIndicators.calculate_volume_delta(df)
        
        # Calculate price returns for target variable
        df['returns'] = df['close'].pct_change()
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)  # 1 for up, 0 for down
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
