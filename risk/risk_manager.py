"""
Risk Manager
-----------
Handles position sizing and risk management.
"""
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk for trading positions"""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Trading configuration
        """
        self.config = config
        self.initial_capital = config.get("initial_capital", 100)
        self.max_position_size = config.get("max_position_size", 10) / 100  # Convert from percentage to decimal
        self.max_drawdown = config.get("max_drawdown", 80) / 100  # Convert from percentage to decimal
        self.risk_per_trade = config.get("risk_per_trade", 1.5) / 100  # Convert from percentage to decimal
        
        # Risk parameters
        self.stop_loss_pct = config.get("strategies", {}).get("risk", {}).get("stop_loss_pct", 15) / 100
        self.take_profit_pct = config.get("strategies", {}).get("risk", {}).get("take_profit_pct", 25) / 100
        
        # Keep track of metrics
        self.peak_value = self.initial_capital
        self.current_drawdown = 0
        self.max_drawdown_reached = 0
    
    def calculate_position_size(self, capital, price, signal_confidence, volatility=None):
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            capital (float): Available capital
            price (float): Current asset price
            signal_confidence (float): Confidence level of signal (0-1)
            volatility (float, optional): Asset volatility
            
        Returns:
            tuple: (position_size_in_usd, quantity)
        """
        # Base position size based on max position size
        max_position_usd = capital * self.max_position_size
        
        # Apply Kelly position sizing with confidence as win rate proxy
        win_rate = max(0.5, signal_confidence)  # Minimum 50% win rate assumption
        profit_loss_ratio = self.take_profit_pct / self.stop_loss_pct
        
        # Kelly fraction calculation
        kelly_fraction = (win_rate * profit_loss_ratio - (1 - win_rate)) / profit_loss_ratio
        
        # Restrict Kelly to avoid over-betting (half-Kelly is common in practice)
        kelly_fraction = max(0, min(kelly_fraction * 0.5, self.max_position_size))
        
        # Apply volatility adjustment if provided
        if volatility is not None:
            # Lower position size for high volatility, increase for low volatility
            baseline_volatility = 0.3  # Annualized volatility baseline (30%)
            volatility_ratio = volatility / baseline_volatility
            
            if volatility_ratio > 1.5:
                # High volatility - reduce position size
                kelly_fraction = kelly_fraction / volatility_ratio
            elif volatility_ratio < 0.5:
                # Low volatility - increase position size (with limit)
                kelly_fraction = min(kelly_fraction * 1.5, kelly_fraction / volatility_ratio)
        
        # Calculate final position size
        position_usd = capital * kelly_fraction
        
        # Apply minimum and maximum constraints
        position_usd = min(position_usd, max_position_usd)
        position_usd = max(position_usd, 10)  # Minimum $10 position
        
        # Calculate quantity
        quantity = position_usd / price
        
        logger.debug(f"Position size: ${position_usd:.2f} ({quantity:.6f} units), Kelly: {kelly_fraction:.4f}")
        
        return position_usd, quantity
    
    def update_drawdown(self, portfolio_value):
        """
        Update drawdown statistics.
        
        Args:
            portfolio_value (float): Current portfolio value
            
        Returns:
            bool: True if max drawdown not exceeded, False otherwise
        """
        # Update peak value if we have a new high
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown_reached = max(self.max_drawdown_reached, self.current_drawdown)
            
            logger.debug(f"Current drawdown: {self.current_drawdown:.2%}, Max drawdown: {self.max_drawdown_reached:.2%}")
            
            # Check if max drawdown exceeded
            if self.current_drawdown > self.max_drawdown:
                logger.warning(f"Maximum drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}")
                return False
        
        return True
    
    # Lanjutan risk/risk_manager.py
    def calculate_stop_loss(self, entry_price, position_type):
        """
        Calculate stop loss price.
        
        Args:
            entry_price (float): Entry price
            position_type (str): 'long' or 'short'
            
        Returns:
            float: Stop loss price
        """
        if position_type.lower() == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price, position_type):
        """
        Calculate take profit price.
        
        Args:
            entry_price (float): Entry price
            position_type (str): 'long' or 'short'
            
        Returns:
            float: Take profit price
        """
        if position_type.lower() == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:  # short
            return entry_price * (1 - self.take_profit_pct)
    
    def get_risk_metrics(self):
        """
        Get current risk metrics.
        
        Returns:
            dict: Risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown_reached': self.max_drawdown_reached,
            'max_drawdown_limit': self.max_drawdown,
            'peak_value': self.peak_value
        }