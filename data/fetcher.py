"""
Data fetcher for cryptocurrency data using CCXT
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time


class BTCDataFetcher:
    """Fetches 5-minute BTC/USDT data from exchanges"""
    
    def __init__(self, exchange_name='binance', symbol='BTC/USDT', timeframe='5m'):
        """
        Initialize the data fetcher
        
        Args:
            exchange_name: Name of the exchange (default: binance)
            symbol: Trading pair symbol (default: BTC/USDT)
            timeframe: Candle timeframe (default: 5m)
        """
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbol = symbol
        self.timeframe = timeframe
        
    def fetch_historical_data(self, days=730):
        """
        Fetch historical OHLCV data
        
        Args:
            days: Number of days of historical data (default: 730 for ~2 years)
            
        Returns:
            DataFrame with OHLCV data
        """
        since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_candles = []
        
        print(f"Fetching {days} days of {self.timeframe} {self.symbol} data...")
        
        while since < self.exchange.milliseconds():
            try:
                candles = self.exchange.fetch_ohlcv(
                    self.symbol, 
                    self.timeframe, 
                    since=since,
                    limit=1000
                )
                
                if len(candles) == 0:
                    break
                    
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
                print(f"Fetched {len(all_candles)} candles so far...")
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Total candles fetched: {len(df)}")
        return df
    
    def save_data(self, df, filename='btc_5m_data.csv'):
        """Save data to CSV file"""
        filepath = f'data/{filename}'
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        
    def load_data(self, filename='btc_5m_data.csv'):
        """Load data from CSV file"""
        filepath = f'data/{filename}'
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        print(f"Data loaded from {filepath}")
        return df
