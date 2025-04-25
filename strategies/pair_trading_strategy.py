"""
Pair Trading Strategy
-------------------
Implements a statistical arbitrage strategy using cointegrated pairs.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from strategies.base_strategy import BaseStrategy, SignalType

logger = logging.getLogger(__name__)

class PairTradingStrategy(BaseStrategy):
    """
    Pair trading strategy based on cointegration and mean reversion.
    Trades pairs of assets that show statistical relationship.
    """
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Strategy configuration
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.z_score_threshold = config.get("strategies", {}).get("pair_trading", {}).get("z_score_threshold", 2.0)
        self.lookback_periods = config.get("strategies", {}).get("pair_trading", {}).get("lookback_periods", 90)
        self.half_life = config.get("strategies", {}).get("pair_trading", {}).get("half_life", 30)
        self.correlation_threshold = config.get("strategies", {}).get("pair_trading", {}).get("correlation_threshold", 0.7)
        
        # Store discovered pairs
        self.cointegrated_pairs = []
    
    def find_cointegrated_pairs(self, data):
        """
        Find pairs of assets that are cointegrated.
        
        Args:
            data (dict): Market data with indicators
            
        Returns:
            list: List of cointegrated pairs (asset1, asset2, p-value, hedge_ratio)
        """
        logger.info("Searching for cointegrated pairs...")
        
        cointegrated_pairs = []
        assets = list(data.keys())
        
        if len(assets) < 2:
            logger.warning("Not enough assets to find pairs")
            return []
        
        # Extract price data for each asset (using first timeframe)
        price_data = {}
        for asset in assets:
            timeframes = list(data[asset].keys())
            if not timeframes:
                continue
                
            timeframe = timeframes[0]  # Use first available timeframe
            df = data[asset][timeframe]
            
            if not df.empty:
                price_data[asset] = df['close']
        
        # Test each pair for cointegration
        tested_pairs = 0
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                if asset1 not in price_data or asset2 not in price_data:
                    continue
                
                tested_pairs += 1
                
                # Extract price series
                series1 = price_data[asset1]
                series2 = price_data[asset2]
                
                # Check if we have enough data
                min_length = min(len(series1), len(series2))
                if min_length < self.lookback_periods:
                    logger.debug(f"Not enough data for {asset1}-{asset2} cointegration test")
                    continue
                
                # Align the series
                common_index = series1.index.intersection(series2.index)
                if len(common_index) < self.lookback_periods:
                    continue
                    
                aligned_series1 = series1.loc[common_index]
                aligned_series2 = series2.loc[common_index]
                
                # Check correlation
                correlation = aligned_series1.corr(aligned_series2)
                if abs(correlation) < self.correlation_threshold:
                    continue
                
                # Test for cointegration
                try:
                    score, p_value, _ = coint(aligned_series1, aligned_series2)
                    
                    if p_value < 0.05:  # Statistically significant
                        # Calculate hedge ratio
                        model = sm.OLS(aligned_series1, sm.add_constant(aligned_series2)).fit()
                        hedge_ratio = model.params[1]
                        
                        # Calculate current z-score
                        spread = aligned_series1 - hedge_ratio * aligned_series2
                        z_score = (spread - spread.mean()) / spread.std()
                        current_z = z_score.iloc[-1]
                        
                        pair_info = (asset1, asset2, p_value, hedge_ratio, current_z, correlation)
                        cointegrated_pairs.append(pair_info)
                        
                        logger.info(f"Found cointegrated pair: {asset1}-{asset2}, p-value: {p_value:.5f}, " 
                                    f"hedge ratio: {hedge_ratio:.4f}, z-score: {current_z:.2f}")
                except Exception as e:
                    logger.warning(f"Error testing cointegration for {asset1}-{asset2}: {str(e)}")
        
        logger.info(f"Tested {tested_pairs} pairs, found {len(cointegrated_pairs)} cointegrated pairs")
        
        # Sort by p-value (most significant first)
        cointegrated_pairs.sort(key=lambda x: x[2])
        
        self.cointegrated_pairs = cointegrated_pairs
        return cointegrated_pairs
    
    def generate_signals(self, data):
        """
        Generate trading signals based on pair relationships.
        
        Args:
            data (dict): Market data with indicators
            
        Returns:
            dict: Trading signals for each pair
        """
        # First, find or update cointegrated pairs
        if not self.cointegrated_pairs:
            self.find_cointegrated_pairs(data)
        
        signals = {}
        
        # No pairs found, no signals
        if not self.cointegrated_pairs:
            logger.warning("No cointegrated pairs found, no signals generated")
            return signals
        
        # Process each pair
        for pair_info in self.cointegrated_pairs:
            asset1, asset2, _, hedge_ratio, _, correlation = pair_info
            
            # Get the current prices
            price1 = self._get_current_price(asset1, data)
            price2 = self._get_current_price(asset2, data)
            
            if price1 is None or price2 is None:
                continue
            
            # Calculate current spread and z-score
            spread = price1 - hedge_ratio * price2
            
            # Extract recent price history to calculate spread statistics
            series1 = self._get_price_series(asset1, data)
            series2 = self._get_price_series(asset2, data)
            
            if series1 is None or series2 is None:
                continue
            
            # Align series and calculate historical spread
            common_index = series1.index.intersection(series2.index)
            if len(common_index) < 30:  # Need at least 30 data points
                continue
                
            aligned_series1 = series1.loc[common_index]
            aligned_series2 = series2.loc[common_index]
            
            historical_spread = aligned_series1 - hedge_ratio * aligned_series2
            mean_spread = historical_spread.mean()
            std_spread = historical_spread.std()
            
            # Calculate z-score
            z_score = (spread - mean_spread) / std_spread
            
            # Generate signals based on z-score
            if z_score > self.z_score_threshold:
                # Spread is too wide, expect convergence
                # SELL asset1 and BUY asset2
                signal_type = SignalType.SELL
                confidence = min(abs(z_score) / 4, 0.9)  # Cap at 0.9
                reason = f"Pair spread z-score ({z_score:.2f}) above threshold, expecting convergence"
            elif z_score < -self.z_score_threshold:
                # Spread is too narrow, expect divergence
                # BUY asset1 and SELL asset2
                signal_type = SignalType.BUY
                confidence = min(abs(z_score) / 4, 0.9)  # Cap at 0.9
                reason = f"Pair spread z-score ({z_score:.2f}) below threshold, expecting divergence"
            else:
                # No clear signal
                signal_type = SignalType.NEUTRAL
                confidence = 0.5
                reason = f"Pair spread z-score ({z_score:.2f}) within normal range"
            
            # Create pair trading signal
            pair_key = f"{asset1}_{asset2}_PAIR"
            signals[pair_key] = {
                'signal': signal_type,
                'confidence': confidence,
                'asset1': asset1,
                'asset2': asset2,
                'price1': price1,
                'price2': price2,
                'hedge_ratio': hedge_ratio,
                'z_score': z_score,
                'correlation': correlation,
                'reason': reason
            }
            
            logger.debug(f"Generated {signal_type.name} signal for pair {asset1}-{asset2} with z-score {z_score:.2f}")
        
        return signals
    
    def _get_current_price(self, asset, data):
        """Get the current price for an asset"""
        if asset in data:
            timeframes = list(data[asset].keys())
            if timeframes:
                timeframe = timeframes[0]  # Use first available timeframe
                df = data[asset][timeframe]
                if not df.empty:
                    return df['close'].iloc[-1]
        return None
    
    def _get_price_series(self, asset, data):
        """Get the price series for an asset"""
        if asset in data:
            timeframes = list(data[asset].keys())
            if timeframes:
                timeframe = timeframes[0]  # Use first available timeframe
                df = data[asset][timeframe]
                if not df.empty:
                    return df['close']
        return None