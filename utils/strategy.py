"""
Trading strategy with position sizing and risk management
"""
import numpy as np
import pandas as pd


class TradingStrategy:
    """Trading strategy implementation with position sizing"""
    
    def __init__(self, initial_capital=10000, position_size_pct=0.01, 
                 prediction_threshold=0.65):
        """
        Initialize trading strategy
        
        Args:
            initial_capital: Initial portfolio value (default: $10,000)
            position_size_pct: Position size as % of portfolio (range: 0.5-1%, default: 1% = 0.01)
            prediction_threshold: Minimum prediction probability (default: 0.65)
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.prediction_threshold = prediction_threshold
        self.portfolio_value = initial_capital
        self.positions = []
        self.trade_history = []
        
    def calculate_position_size(self, price, portfolio_value=None):
        """
        Calculate position size based on portfolio value
        Position size: 0.5-1% of portfolio per trade
        
        Args:
            price: Current asset price
            portfolio_value: Current portfolio value (uses self.portfolio_value if None)
            
        Returns:
            Position size in units of the asset
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
            
        # Calculate dollar amount to invest
        position_value = portfolio_value * self.position_size_pct
        
        # Calculate number of units
        position_size = position_value / price
        
        return position_size
    
    def execute_signal(self, signal, price, probability, timestamp=None):
        """
        Execute trading signal
        
        Args:
            signal: Trading signal (-1=no action, 0=sell/short, 1=buy/long)
            price: Current price
            probability: Prediction probability
            timestamp: Time of signal (optional)
            
        Returns:
            Trade execution details or None
        """
        # Only act if signal is not -1 (no action)
        if signal == -1:
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(price)
        
        trade = {
            'timestamp': timestamp,
            'signal': signal,
            'price': price,
            'probability': probability,
            'position_size': position_size,
            'position_value': position_size * price
        }
        
        if signal == 1:  # Buy signal
            self.positions.append(trade)
            self.portfolio_value -= trade['position_value']
            trade['action'] = 'BUY'
            
        elif signal == 0 and len(self.positions) > 0:  # Sell signal with open positions
            # Close oldest position (Note: Short selling is not implemented in this strategy)
            opened_position = self.positions.pop(0)
            profit = (price - opened_position['price']) * opened_position['position_size']
            self.portfolio_value += opened_position['position_value'] + profit
            
            trade['action'] = 'SELL'
            trade['opened_price'] = opened_position['price']
            trade['profit'] = profit
            trade['profit_pct'] = (profit / opened_position['position_value']) * 100
        else:
            return None
        
        self.trade_history.append(trade)
        return trade
    
    def backtest(self, predictions, probabilities, prices, dates=None):
        """
        Backtest the trading strategy
        
        Args:
            predictions: Array of predictions (-1, 0, 1)
            probabilities: Array of prediction probabilities
            prices: Array of prices
            dates: Array of dates (optional)
            
        Returns:
            Dictionary with backtest results
        """
        self.portfolio_value = self.initial_capital
        self.positions = []
        self.trade_history = []
        
        for i in range(len(predictions)):
            timestamp = dates[i] if dates is not None else i
            self.execute_signal(
                predictions[i], 
                prices[i], 
                probabilities[i],
                timestamp
            )
        
        # Close any remaining positions at final price
        final_price = prices[-1]
        while len(self.positions) > 0:
            position = self.positions.pop(0)
            profit = (final_price - position['price']) * position['position_size']
            self.portfolio_value += position['position_value'] + profit
        
        # Calculate metrics
        total_return = self.portfolio_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate trade statistics
        profitable_trades = [t for t in self.trade_history 
                           if t.get('action') == 'SELL' and t.get('profit', 0) > 0]
        
        total_trades = sum(1 for t in self.trade_history if t.get('action') == 'SELL')
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'profitable_trades': len(profitable_trades),
            'win_rate': win_rate,
            'trade_history': self.trade_history
        }
        
        return results
    
    def get_summary(self, results):
        """
        Get formatted summary of backtest results
        
        Args:
            results: Results dictionary from backtest
            
        Returns:
            Formatted string summary
        """
        summary = f"""
        ===== Trading Strategy Results =====
        Initial Capital: ${results['initial_capital']:,.2f}
        Final Portfolio Value: ${results['final_portfolio_value']:,.2f}
        Total Return: ${results['total_return']:,.2f} ({results['total_return_pct']:.2f}%)
        
        Total Trades: {results['total_trades']}
        Profitable Trades: {results['profitable_trades']}
        Win Rate: {results['win_rate']*100:.2f}%
        
        Position Size: {self.position_size_pct*100:.1f}% of portfolio per trade
        Prediction Threshold: {self.prediction_threshold*100:.0f}%
        ===================================
        """
        return summary
