"""
Moving Average Strategy
---------------------
Implements a simple moving average crossover strategy.
"""
import logging
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, SignalType

logger = logging.getLogger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy.
    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Strategy configuration
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.fast_ma = config.get("strategies", {}).get("moving_average", {}).get("fast_ma", "ema12")
        self.slow_ma = config.get("strategies", {}).get("moving_average", {}).get("slow_ma", "ema26")
        self.lookback = config.get("strategies", {}).get("moving_average", {}).get("lookback", 3)
        self.rsi_oversold = config.get("strategies", {}).get("moving_average", {}).get("rsi_oversold", 30)
        self.rsi_overbought = config.get("strategies", {}).get("moving_average", {}).get("rsi_overbought", 70)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (dict): Market data with indicators
            
        Returns:
            dict: Trading signals for each pair/timeframe
        """
        signals = {}
        
        for pair in data:
            signals[pair] = {}
            
            for timeframe in data[pair]:
                df = data[pair][timeframe]
                
                if df.empty or len(df) < self.lookback + 1:
                    signals[pair][timeframe] = {
                        'signal': SignalType.NEUTRAL,
                        'confidence': 0.0,
                        'reason': 'Not enough data'
                    }
                    continue
                
                # Check if required indicators exist
                if self.fast_ma not in df.columns or self.slow_ma not in df.columns or 'rsi' not in df.columns:
                    logger.warning(f"Required indicators missing for {pair} {timeframe}")
                    signals[pair][timeframe] = {
                        'signal': SignalType.NEUTRAL,
                        'confidence': 0.0,
                        'reason': 'Missing indicators'
                    }
                    continue
                
                # Get the current and previous indicators
                current = df.iloc[-1]
                previous = df.iloc[-2]
                
                # Initialize signal components
                signal_components = []
                
                # 1. Moving Average Crossover
                if current[self.fast_ma] > current[self.slow_ma] and previous[self.fast_ma] <= previous[self.slow_ma]:
                    # Bullish crossover
                    signal_components.append(SignalType.BUY)
                elif current[self.fast_ma] < current[self.slow_ma] and previous[self.fast_ma] >= previous[self.slow_ma]:
                    # Bearish crossover
                    signal_components.append(SignalType.SELL)
                else:
                    # Check trend direction
                    if current[self.fast_ma] > current[self.slow_ma]:
                        signal_components.append(SignalType.BUY)
                    elif current[self.fast_ma] < current[self.slow_ma]:
                        signal_components.append(SignalType.SELL)
                    else:
                        signal_components.append(SignalType.NEUTRAL)
                
                # 2. RSI Conditions
                if current['rsi'] < self.rsi_oversold:
                    # Oversold condition
                    signal_components.append(SignalType.BUY)
                elif current['rsi'] > self.rsi_overbought:
                    # Overbought condition
                    signal_components.append(SignalType.SELL)
                else:
                    signal_components.append(SignalType.NEUTRAL)
                
                # 3. Price relative to moving averages
                if 'price_sma20_ratio' in df.columns:
                    if current['price_sma20_ratio'] > 1.05:
                        # Price well above short-term MA (potential overextension)
                        signal_components.append(SignalType.SELL)
                    elif current['price_sma20_ratio'] < 0.95:
                        # Price well below short-term MA (potential buying opportunity)
                        signal_components.append(SignalType.BUY)
                    else:
                        signal_components.append(SignalType.NEUTRAL)
                
                # Combine signal components
                weights = [0.5, 0.3, 0.2]  # Crossover has highest weight
                if len(signal_components) < 3:
                    weights = weights[:len(signal_components)]
                    
                final_signal, confidence = self.combine_signals(signal_components, weights)
                
                # Generate detailed signal
                signal_detail = {
                    'signal': final_signal,
                    'confidence': confidence,
                    'current_price': df['close'].iloc[-1],
                    'fast_ma': current[self.fast_ma],
                    'slow_ma': current[self.slow_ma],
                    'rsi': current['rsi'],
                    'components': [s.name for s in signal_components],
                    'weights': weights,
                    'reason': self._get_signal_reason(final_signal, current, previous)
                }
                
                signals[pair][timeframe] = signal_detail
                
                logger.debug(f"Generated {final_signal.name} signal for {pair} {timeframe} with {confidence:.2f} confidence")
                
        return signals
    
    def _get_signal_reason(self, signal, current, previous):
        """Generate a human-readable reason for the signal"""
        if signal == SignalType.BUY:
            if current[self.fast_ma] > current[self.slow_ma] and previous[self.fast_ma] <= previous[self.slow_ma]:
                return f"Bullish crossover: {self.fast_ma} crossed above {self.slow_ma}"
            elif current['rsi'] < self.rsi_oversold:
                return f"Oversold condition: RSI at {current['rsi']:.2f}"
            else:
                return f"Bullish trend: {self.fast_ma} above {self.slow_ma}"
        elif signal == SignalType.SELL:
            if current[self.fast_ma] < current[self.slow_ma] and previous[self.fast_ma] >= previous[self.slow_ma]:
                return f"Bearish crossover: {self.fast_ma} crossed below {self.slow_ma}"
            elif current['rsi'] > self.rsi_overbought:
                return f"Overbought condition: RSI at {current['rsi']:.2f}"
            else:
                return f"Bearish trend: {self.fast_ma} below {self.slow_ma}"
        else:
            return "No clear signal"