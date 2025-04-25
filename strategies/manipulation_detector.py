"""
Market Manipulation Detector
--------------------------
Detects potential market manipulation patterns.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ManipulationDetector:
    """Detects potential market manipulation patterns"""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Bot configuration
        """
        self.config = config
        
        # Detection thresholds
        manipulation_config = config.get("strategies", {}).get("manipulation", {})
        self.volume_spike_threshold = manipulation_config.get("volume_spike_threshold", 3.0)
        self.price_spike_threshold = manipulation_config.get("price_spike_threshold", 0.1)
        self.bid_ask_imbalance_threshold = manipulation_config.get("bid_ask_imbalance_threshold", 3.0)
        self.wash_trade_threshold = manipulation_config.get("wash_trade_threshold", 0.8)
    
    def detect_manipulation(self, market_data):
        """
        Analyze market data for manipulation patterns.
        
        Args:
            market_data (dict): Market data with indicators
            
        Returns:
            dict: Manipulation detection results by asset
        """
        results = {}
        
        for pair in market_data:
            # Use hourly data if available, otherwise use the first available timeframe
            timeframes = list(market_data[pair].keys())
            
            if not timeframes:
                continue
                
            if '1h' in timeframes:
                timeframe = '1h'
            else:
                timeframe = timeframes[0]
                
            df = market_data[pair][timeframe]
            
            if df.empty:
                continue
            
            # Run detectors
            pump_dump = self.detect_pump_dump(df)
            wash_trading = self.detect_wash_trading(df)
            unusual_volatility = self.detect_unusual_volatility(df)
            
            # Combine results
            results[pair] = {
                'pump_dump_detected': pump_dump[0],
                'pump_dump_confidence': pump_dump[1],
                'pump_dump_details': pump_dump[2],
                'wash_trading_detected': wash_trading[0],
                'wash_trading_confidence': wash_trading[1],
                'unusual_volatility_detected': unusual_volatility[0],
                'volatility_z_score': unusual_volatility[1],
                'timestamp': datetime.now()
            }
            
            # Log any detected manipulation
            if pump_dump[0] or wash_trading[0] or unusual_volatility[0]:
                logger.warning(f"Potential market manipulation detected for {pair}:")
                if pump_dump[0]:
                    logger.warning(f"  - Pump and dump pattern (confidence: {pump_dump[1]:.2f})")
                if wash_trading[0]:
                    logger.warning(f"  - Wash trading pattern (confidence: {wash_trading[1]:.2f})")
                if unusual_volatility[0]:
                    logger.warning(f"  - Unusual volatility (z-score: {unusual_volatility[1]:.2f})")
        
        return results
    
    def detect_pump_dump(self, df):
        """
        Detect pump and dump patterns.
        
        Args:
            df (DataFrame): OHLCV data with indicators
            
        Returns:
            tuple: (is_detected, confidence, details)
        """
        if len(df) < 24:  # Need at least 24 data points
            return False, 0, {}
        
        # Calculate percentage changes
        price_pct_change = df['close'].pct_change()
        volume_pct_change = df['volume'].pct_change()
        
        # Define criteria
        price_spikes = price_pct_change > self.price_spike_threshold
        volume_spikes = volume_pct_change > self.volume_spike_threshold
        
        # Look for price spike followed by volume spike (classic pump pattern)
        pump_signals = price_spikes & volume_spikes.shift(-1)
        
        # Look for price drops after pump
        post_pump_drops = []
        
        for idx in pump_signals[pump_signals].index:
            idx_pos = df.index.get_loc(idx)
            if idx_pos + 12 < len(df):  # Look at next 12 periods
                subsequent_data = df.iloc[idx_pos:idx_pos+12]
                max_drop = (subsequent_data['close'].min() - df.loc[idx, 'close']) / df.loc[idx, 'close']
                if max_drop < -0.1:  # Drop of more than 10%
                    post_pump_drops.append((idx, max_drop))
        
        # Calculate confidence
        confidence = min(len(post_pump_drops) / max(1, len(pump_signals[pump_signals])), 1.0)
        
        details = {
            'pump_periods': list(pump_signals[pump_signals].index),
            'dump_events': post_pump_drops,
            'avg_pump_pct': price_pct_change[price_spikes].mean() if any(price_spikes) else 0,
            'avg_volume_increase': volume_pct_change[volume_spikes].mean() if any(volume_spikes) else 0
        }
        
        return len(post_pump_drops) > 0, confidence, details
    
    def detect_wash_trading(self, df):
        """
        Detect potential wash trading patterns.
        
        Args:
            df (DataFrame): OHLCV data with indicators
            
        Returns:
            tuple: (is_detected, confidence)
        """
        if len(df) < 48:  # Need at least 48 data points (2 days of hourly data)
            return False, 0
        
        # Wash trading often has high volume but little price movement
        # Calculate the ratio of volume to price movement
        price_movement = df['high'] - df['low']
        volume_to_movement_ratio = df['volume'] / price_movement.replace(0, 1e-10)
        
        # Normalize and calculate z-score
        mean_ratio = volume_to_movement_ratio.mean()
        std_ratio = volume_to_movement_ratio.std()
        ratio_z_scores = (volume_to_movement_ratio - mean_ratio) / max(std_ratio, 1e-10)
        
        # Check for sustained periods of abnormally high volume/movement ratio
        suspicious_periods = ratio_z_scores > 2.0  # More than 2 std devs above mean
        
        # Calculate percentage of suspicious periods
        suspicious_pct = suspicious_periods.mean()
        
        # Determine if wash trading is detected
        is_detected = suspicious_pct > self.wash_trade_threshold / 100
        confidence = min(suspicious_pct * 2, 1.0)
        
        return is_detected, confidence
    
    def detect_unusual_volatility(self, df):
        """
        Detect unusual volatility patterns that may indicate manipulation.
        
        Args:
            df (DataFrame): OHLCV data with indicators
            
        Returns:
            tuple: (is_detected, volatility_z_score)
        """
        if len(df) < 30 or 'volatility' not in df.columns:
            return False, 0
        
        # Get current volatility
        current_volatility = df['volatility'].iloc[-1]
        
        # Calculate volatility z-score
        mean_volatility = df['volatility'].mean()
        std_volatility = df['volatility'].std()
        
        volatility_z_score = (current_volatility - mean_volatility) / max(std_volatility, 1e-10)
        
        # Unusual volatility is detected if z-score is > 3 (3 std devs above mean)
        is_detected = volatility_z_score > 3.0
        
        return is_detected, volatility_z_score