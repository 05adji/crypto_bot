"""
Portfolio Manager
---------------
Manages trading portfolio and positions.
"""
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages trading portfolio and positions"""
    
    def __init__(self, config):
        """
        Initialize portfolio manager.
        
        Args:
            config (dict): Bot configuration
        """
        self.config = config
        self.initial_capital = config.get("initial_capital", 100)
        self.cash = self.initial_capital
        self.positions = {}  # Current positions
        self.trades = []  # Trade history
        self.performance_history = []  # Portfolio value history
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.performance_history = []
        logger.info(f"Portfolio reset to initial capital of ${self.initial_capital}")
    
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
        Update portfolio with a new trade.
        
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
        
        # Update positions and cash
        if side.lower() == 'buy':
            # Deduct cash
            self.cash -= trade_value
            
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
                
            logger.info(f"BUY {quantity:.6f} {symbol} @ ${price:.2f} = ${trade_value:.2f}")
            
        elif side.lower() == 'sell':
            # Add to cash
            self.cash += trade_value
            
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
            
            logger.info(f"SELL {quantity:.6f} {symbol} @ ${price:.2f} = ${trade_value:.2f}")
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'timestamp': timestamp,
            'remaining_cash': self.cash
        }
        
        self.trades.append(trade_record)
    
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
                logger.debug(f"Position {symbol}: {quantity:.6f} units @ ${current_price:.2f} = ${position_value:.2f}")
        
        # Record portfolio value
        self.performance_history.append({
            'timestamp': datetime.now(),
            'value': total_value,
            'cash': self.cash
        })
        
        logger.debug(f"Portfolio value: ${total_value:.2f} (Cash: ${self.cash:.2f})")
        
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
        
        # If not found, log warning
        logger.warning(f"Could not find current price for {symbol}")
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