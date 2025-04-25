"""
Backtester
---------
Simulates trading strategies on historical data.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

from data.indicators import TechnicalIndicators
from risk.risk_manager import RiskManager
from execution.portfolio_manager import PortfolioManager
from strategies.base_strategy import SignalType

logger = logging.getLogger(__name__)

class Backtester:
    """Simulates trading strategies on historical data"""
    
    def __init__(self, config, strategy, data_collector):
        """
        Initialize backtester.
        
        Args:
            config (dict): Bot configuration
            strategy: Strategy instance to test
            data_collector: Data collector instance
        """
        self.config = config
        self.strategy = strategy
        self.data_collector = data_collector
        
        # Initialize components
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
    
    def run_backtest(self, start_date=None, end_date=None, plot_results=True):
        """
        Run backtest over a period of time.
        
        Args:
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            plot_results (bool): Whether to plot results
            
        Returns:
            dict: Backtest results
        """
        logger.info("Starting backtest...")
        
        # Default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()
            
        logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")
        
        # Reset portfolio to initial state
        self.portfolio_manager.reset_portfolio()
        
        # Load historical data for the backtest period
        logger.info("Loading historical data...")
        historical_data = self.data_collector.load_dataset()
        
        # Get all trading dates
        all_dates = self._extract_trading_dates(historical_data, start_date, end_date)
        
        if not all_dates:
            logger.error("No trading dates available in the selected period")
            return None
            
        logger.info(f"Found {len(all_dates)} trading dates in the backtest period")
        
        # Run day-by-day simulation
        daily_results = []
        signals_by_date = {}
        
        for i, current_date in enumerate(all_dates):
            logger.debug(f"Processing date: {current_date}")
            
            # Filter data up to current date
            current_data = self._filter_data_up_to_date(historical_data, current_date)
            
            # Calculate indicators
            indicators_data = self._calculate_indicators(current_data)
            
            # Generate signals
            signals = self.strategy.generate_signals(indicators_data)
            signals_by_date[current_date] = signals
            
            # Execute trades based on signals
            self._execute_backtest_trades(signals, current_data, current_date)
            
            # Calculate portfolio value
            portfolio_value = self.portfolio_manager.calculate_portfolio_value(current_data)
            
            # Update risk metrics
            self.risk_manager.update_drawdown(portfolio_value)
            
            # Record daily result
            daily_result = {
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio_manager.cash,
                'drawdown': self.risk_manager.current_drawdown
            }
            
            daily_results.append(daily_result)
            
            # Log progress occasionally
            if i % 20 == 0 or i == len(all_dates) - 1:
                logger.info(f"Backtest progress: {i+1}/{len(all_dates)} days processed")
                logger.info(f"Current date: {current_date}, Portfolio value: ${portfolio_value:.2f}")
        
        # Calculate backtest metrics
        results = self._calculate_backtest_metrics(daily_results)
        
        # Plot results if requested
        if plot_results:
            self._plot_backtest_results(daily_results, signals_by_date)
        
        return results
    
    def _extract_trading_dates(self, data, start_date, end_date):
        """Extract all unique trading dates from the data"""
        all_dates = set()
        
        for pair in data:
            for timeframe in data[pair]:
                df = data[pair][timeframe]
                if not df.empty:
                    # Filter dates in range
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    dates = df.loc[mask].index.date
                    all_dates.update(dates)
        
        return sorted(all_dates)
    
    def _filter_data_up_to_date(self, data, current_date):
        """Filter data up to a specific date"""
        filtered_data = {}
        
        for pair in data:
            filtered_data[pair] = {}
            
            for timeframe in data[pair]:
                df = data[pair][timeframe]
                if not df.empty:
                    # Filter data up to current date
                    current_dt = datetime.combine(current_date, datetime.min.time())
                    filtered = df[df.index <= current_dt].copy()
                    
                    if not filtered.empty:
                        filtered_data[pair][timeframe] = filtered
        
        return filtered_data
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators for all pairs and timeframes"""
        indicators = {}
        
        for pair in data:
            indicators[pair] = {}
            
            for timeframe in data[pair]:
                df = data[pair][timeframe]
                if not df.empty:
                    indicators[pair][timeframe] = TechnicalIndicators.add_indicators(df)
        
        return indicators
    
    def _execute_backtest_trades(self, signals, data, current_date):
        """Execute trades based on signals"""
        for pair in signals:
            for timeframe in signals[pair]:
                signal = signals[pair][timeframe]
                
                # Skip if no clear signal
                if signal['signal'] == SignalType.NEUTRAL:
                    continue
                
                # Get current price
                current_price = signal['current_price']
                
                # Check if we have a position already
                position = self.portfolio_manager.get_position(pair)
                
                if signal['signal'] == SignalType.BUY:
                    # Only buy if we don't have a position already
                    if position is None:
                        # Calculate position size
                        position_usd, quantity = self.risk_manager.calculate_position_size(
                            self.portfolio_manager.cash,
                            current_price,
                            signal['confidence'],
                            signal.get('volatility', None)
                        )
                        
                        # Ensure we have enough cash
                        if position_usd <= self.portfolio_manager.cash and quantity > 0:
                            # Execute trade
                            trade = {
                                'symbol': pair,
                                'side': 'buy',
                                'quantity': quantity,
                                'price': current_price,
                                'timestamp': datetime.combine(current_date, datetime.min.time())
                            }
                            
                            self.portfolio_manager.update_portfolio(trade)
                
                elif signal['signal'] == SignalType.SELL:
                    # Only sell if we have a position
                    if position is not None:
                        # Sell entire position
                        quantity = position['quantity']
                        
                        # Execute trade
                        trade = {
                            'symbol': pair,
                            'side': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'timestamp': datetime.combine(current_date, datetime.min.time())
                        }
                        
                        self.portfolio_manager.update_portfolio(trade)
    
    def _calculate_backtest_metrics(self, daily_results):
        """Calculate performance metrics from backtest results"""
        if not daily_results:
            return None
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(daily_results)
        
        # Calculate basic metrics
        initial_value = df['portfolio_value'].iloc[0]
        final_value = df['portfolio_value'].iloc[-1]
        
        total_return = (final_value - initial_value) / initial_value
        max_drawdown = self.risk_manager.max_drawdown_reached
        
        # Calculate daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Remove NaN
        daily_returns = df['daily_return'].dropna().values
        
        # Calculate annualized metrics
        trading_days = len(daily_results)
        years = trading_days / 252  # Assuming 252 trading days per year
        
        if years > 0:
            annual_return = ((1 + total_return) ** (1 / years)) - 1
        else:
            annual_return = 0
            
        # Calculate volatility
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0
            
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        if volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0
            
        # Calculate win rate from trades
        trades = self.portfolio_manager.get_trade_history()
        buy_trades = [t for t in trades if t['side'].lower() == 'buy']
        sell_trades = [t for t in trades if t['side'].lower() == 'sell']
        
        winning_trades = 0
        
        # Simple approximation of win rate
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Match buys with sells
            for sell in sell_trades:
                matching_buys = [b for b in buy_trades if b['symbol'] == sell['symbol']]
                if matching_buys:
                    # Use the first matching buy (this is a simplification)
                    buy = matching_buys[0]
                    if sell['price'] > buy['price']:
                        winning_trades += 1
        
        win_rate = winning_trades / max(len(sell_trades), 1)
        
        # Return all metrics
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'trading_days': trading_days,
            'duration_years': years
        }
    
    def _plot_backtest_results(self, daily_results, signals_by_date):
        """Plot backtest results"""
        if not daily_results:
            return
            
        # Create plots directory if it doesn't exist
        if not os.path.exists("plots"):
            os.makedirs("plots")
            
        # Convert to DataFrame
        df = pd.DataFrame(daily_results)
        
        # 1. Portfolio value chart
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['date'], df['portfolio_value'], 'b-', label='Portfolio Value')
        plt.title('Backtest Results: Portfolio Value')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        # 2. Drawdown chart
        plt.subplot(2, 1, 2)
        plt.fill_between(df['date'], 0, df['drawdown'] * 100, color='r', alpha=0.3)
        plt.plot(df['date'], df['drawdown'] * 100, 'r-', label='Drawdown %')
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/backtest_portfolio_performance.png')
        
        # 3. Equity curve with buy/sell markers
        plt.figure(figsize=(12, 6))
        
        plt.plot(df['date'], df['portfolio_value'], 'b-', label='Portfolio Value')
        
        # Add markers for buy/sell signals (just a sample)
        trades = self.portfolio_manager.get_trade_history()
        
        # Extract buy trades
        buy_dates = [t['timestamp'].date() for t in trades if t['side'].lower() == 'buy']
        buy_values = [df.loc[df['date'] == date, 'portfolio_value'].iloc[0] if len(df.loc[df['date'] == date]) > 0 else None for date in buy_dates]
        buy_values = [v for v in buy_values if v is not None]
        
        # Extract sell trades
        sell_dates = [t['timestamp'].date() for t in trades if t['side'].lower() == 'sell']
        sell_values = [df.loc[df['date'] == date, 'portfolio_value'].iloc[0] if len(df.loc[df['date'] == date]) > 0 else None for date in sell_dates]
        sell_values = [v for v in sell_values if v is not None]
        
        # Plot markers
        if buy_dates and buy_values:
            plt.scatter(buy_dates[:len(buy_values)], buy_values, color='g', marker='^', s=100, label='Buy')
        
        if sell_dates and sell_values:
            plt.scatter(sell_dates[:len(sell_values)], sell_values, color='r', marker='v', s=100, label='Sell')
            
        plt.title('Equity Curve with Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/backtest_equity_curve.png')
        
        # 4. Daily returns distribution
        plt.figure(figsize=(12, 6))
        
        df['daily_return'] = df['portfolio_value'].pct_change()
        daily_returns = df['daily_return'].dropna() * 100  # Convert to percentage
        
        plt.subplot(1, 2, 1)
        plt.hist(daily_returns, bins=50, alpha=0.75, color='b')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(daily_returns)
        plt.title('Daily Returns Box Plot')
        plt.ylabel('Daily Return (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/backtest_return_distribution.png')
        
        logger.info("Backtest plots saved to 'plots' directory")