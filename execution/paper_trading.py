"""
Paper Trading Order Manager
--------------------------
Simulates order execution without real money.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

class PaperTradingOrderManager:
    """Simulates order execution for paper trading"""
    
    def __init__(self, config):
        """Initialize paper trading manager"""
        self.config = config
        self.initial_capital = config.get("initial_capital", 100)
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.performance_history = []
        
        # Create directory for paper trading logs
        os.makedirs("data/paper_trading", exist_ok=True)
        
        # Try to load previous paper trading state if available
        self._load_state()
        
        logger.info(f"Paper trading initialized with ${self.cash} cash")
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.performance_history = []
        self._save_state()
        logger.info(f"Paper trading portfolio reset to initial capital of ${self.initial_capital}")
    
    def get_portfolio_summary(self):
        """
        Get portfolio summary.
        
        Returns:
            dict: Portfolio summary
        """
        return {
            'cash': self.cash,
            'positions': self.positions,
            'trade_count': len(self.trades),
            'initial_capital': self.initial_capital
        }
    
    def update_portfolio(self, trade):
        """
        Update portfolio with a new trade (paper).
        
        Args:
            trade (dict): Trade information
        """
        symbol = trade['symbol']
        side = trade['side']
        quantity = trade['quantity']
        price = trade['price']
        timestamp = trade.get('timestamp', datetime.now())
        
        # Calculate trade value
        trade_value = quantity * price
        
        # Apply trading fee (simulate exchange fee)
        fee_rate = 0.001  # 0.1% (typical for Binance)
        fee = trade_value * fee_rate
        
        # Update positions and cash
        if side.lower() == 'buy':
            # Deduct cash (including fee)
            total_cost = trade_value + fee
            self.cash -= total_cost
            
            # Update position
            if symbol in self.positions:
                # Update existing position with weighted average price
                current_qty = self.positions[symbol]['quantity']
                current_price = self.positions[symbol]['average_price']
                
                # Calculate new average price
                new_total_qty = current_qty + quantity
                new_avg_price = (current_qty * current_price + quantity * price) / new_total_qty
                
                self.positions[symbol] = {
                    'quantity': new_total_qty,
                    'average_price': new_avg_price,
                    'last_update': timestamp
                }
            else:
                # Create new position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'average_price': price,
                    'last_update': timestamp
                }
                
            logger.info(f"PAPER TRADE: BUY {quantity:.6f} {symbol} @ ${price:.2f} = ${trade_value:.2f} (Fee: ${fee:.2f})")
            
        elif side.lower() == 'sell':
            # Calculate fee
            fee = trade_value * fee_rate
            
            # Add to cash (minus fee)
            self.cash += (trade_value - fee)
            
            # Update position
            if symbol in self.positions:
                current_qty = self.positions[symbol]['quantity']
                
                # Calculate remaining quantity
                remaining_qty = current_qty - quantity
                
                if remaining_qty <= 0:
                    # Position closed
                    del self.positions[symbol]
                else:
                    # Position partially closed
                    self.positions[symbol]['quantity'] = remaining_qty
                    self.positions[symbol]['last_update'] = timestamp
            
            logger.info(f"PAPER TRADE: SELL {quantity:.6f} {symbol} @ ${price:.2f} = ${trade_value:.2f} (Fee: ${fee:.2f})")
        
        # Record trade with fees
        trade_record = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'fee': fee,
            'timestamp': timestamp,
            'remaining_cash': self.cash
        }
        
        self.trades.append(trade_record)
        
        # Save updated state
        self._save_state()
        
        # Add to performance history
        self.performance_history.append({
            'timestamp': timestamp,
            'portfolio_value': self.calculate_portfolio_value({'dummy': {'dummy': pd.DataFrame({'close': [price]})}}),
            'cash': self.cash,
            'trade_id': len(self.trades) - 1
        })
    
    def calculate_portfolio_value(self, market_data):
        """
        Calculate total portfolio value based on current prices.
        
        Args:
            market_data (dict): Market data with current prices
            
        Returns:
            float: Total portfolio value
        """
        total_value = self.cash
        
        # Add value of all positions
        for symbol, position in self.positions.items():
            quantity = position['quantity']
            
            # Find current price
            current_price = self._get_current_price(symbol, market_data)
            
            if current_price:
                position_value = quantity * current_price
                total_value += position_value
                logger.debug(f"Paper position {symbol}: {quantity:.6f} units @ ${current_price:.2f} = ${position_value:.2f}")
                
                # Update position with current price (for reporting)
                self.positions[symbol]['current_price'] = current_price
                self.positions[symbol]['current_value'] = position_value
        
        # Add performance record
        self.performance_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': total_value,
            'cash': self.cash
        })
        
        # Save updated state periodically
        if len(self.performance_history) % 10 == 0:
            self._save_state()
        
        logger.debug(f"Paper portfolio value: ${total_value:.2f} (Cash: ${self.cash:.2f})")
        
        return total_value
    
    def _get_current_price(self, symbol, market_data):
        """
        Get current price for a symbol from market data.
        
        Args:
            symbol (str): Trading pair symbol
            market_data (dict): Market data
            
        Returns:
            float: Current price or None if not found
        """
        # Try to find price in market data
        if symbol in market_data:
            # Use the first available timeframe
            for timeframe in market_data[symbol]:
                df = market_data[symbol][timeframe]
                if not df.empty:
                    return df['close'].iloc[-1]
        
        # If not found, use last known price from position
        if symbol in self.positions and 'current_price' in self.positions[symbol]:
            return self.positions[symbol]['current_price']
        
        # As a fallback, use average price
        if symbol in self.positions:
            return self.positions[symbol]['average_price']
        
        # If all else fails, log warning
        logger.warning(f"Could not find current price for {symbol} in paper trading")
        return None
    
    def get_position(self, symbol):
        """
        Get position details for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Position details or None if not found
        """
        return self.positions.get(symbol)
    
    def get_trade_history(self):
        """
        Get trade history.
        
        Returns:
            list: Trade history
        """
        return self.trades
    
    def get_performance_history(self):
        """
        Get portfolio performance history.
        
        Returns:
            list: Performance history
        """
        return self.performance_history
    
    def _save_state(self):
        """Save paper trading state to file"""
        try:
            state = {
                'cash': self.cash,
                'positions': self.positions,
                'trades': self.trades,
                'initial_capital': self.initial_capital,
                'last_updated': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            state_json = json.dumps(state, default=self._datetime_serializer, indent=2)
            
            with open('data/paper_trading/state.json', 'w') as f:
                f.write(state_json)
                
            # Also save performance history to CSV
            if self.performance_history:
                performance_df = pd.DataFrame(self.performance_history)
                performance_df.to_csv('data/paper_trading/performance_log.csv', index=False)
                
        except Exception as e:
            logger.error(f"Error saving paper trading state: {str(e)}")
    
    def _load_state(self):
        """Load paper trading state from file"""
        try:
            if os.path.exists('data/paper_trading/state.json'):
                with open('data/paper_trading/state.json', 'r') as f:
                    state = json.load(f)
                
                self.cash = state.get('cash', self.initial_capital)
                self.positions = state.get('positions', {})
                self.trades = state.get('trades', [])
                self.initial_capital = state.get('initial_capital', self.initial_capital)
                
                logger.info(f"Loaded paper trading state: ${self.cash} cash, {len(self.positions)} positions, {len(self.trades)} trades")
                
                # Also load performance history if available
                if os.path.exists('data/paper_trading/performance_log.csv'):
                    performance_df = pd.read_csv('data/paper_trading/performance_log.csv')
                    self.performance_history = performance_df.to_dict('records')
                    
        except Exception as e:
            logger.error(f"Error loading paper trading state: {str(e)}")
    
    @staticmethod
    def _datetime_serializer(obj):
        """Helper for JSON serialization of datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")