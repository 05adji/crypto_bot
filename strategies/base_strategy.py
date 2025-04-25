"""
Base Strategy
------------
Define base strategy interface for all trading strategies.
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = 1
    SELL = 2
    NEUTRAL = 3

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Strategy configuration
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on market data.
        Must be implemented by all strategy subclasses.
        
        Args:
            data (dict): Market data with indicators
            
        Returns:
            dict: Trading signals for each pair/timeframe
        """
        pass
    
    def get_strategy_name(self):
        """Returns the strategy name"""
        return self.name
    
    def get_strategy_config(self):
        """Returns the strategy configuration"""
        return self.config
    
    def combine_signals(self, signals, weights=None):
        """
        Combine multiple signals with optional weights.
        
        Args:
            signals (list): List of SignalType
            weights (list): Optional weights for each signal
            
        Returns:
            tuple: (SignalType, confidence)
        """
        if not signals:
            return SignalType.NEUTRAL, 0.0
        
        if weights is None:
            weights = [1.0] * len(signals)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Count signal types
        buy_score = sum(weights[i] for i, s in enumerate(signals) if s == SignalType.BUY)
        sell_score = sum(weights[i] for i, s in enumerate(signals) if s == SignalType.SELL)
        
        # Calculate normalized score (-1 to 1)
        normalized_score = buy_score - sell_score
        
        # Convert to signal with confidence
        if normalized_score > 0.2:
            return SignalType.BUY, min(normalized_score * 1.5, 1.0)
        elif normalized_score < -0.2:
            return SignalType.SELL, min(abs(normalized_score) * 1.5, 1.0)
        else:
            return SignalType.NEUTRAL, 1 - abs(normalized_score) * 2