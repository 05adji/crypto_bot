"""
Technical Indicators
-------------------
Calculates technical indicators for market analysis.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculates technical indicators from OHLCV data"""
    
    @staticmethod
    def add_indicators(df):
        """
        Add technical indicators to a DataFrame of OHLCV data.
        
        Args:
            df (DataFrame): OHLCV price data
            
        Returns:
            DataFrame: Original dataframe with indicators added
        """
        if df.empty:
            return df
            
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Moving Averages
            result['sma20'] = TechnicalIndicators.sma(result['close'], 20)
            result['sma50'] = TechnicalIndicators.sma(result['close'], 50)
            result['sma200'] = TechnicalIndicators.sma(result['close'], 200)
            result['ema12'] = TechnicalIndicators.ema(result['close'], 12)
            result['ema26'] = TechnicalIndicators.ema(result['close'], 26)
            
            # MACD
            macd_line, signal_line, histogram = TechnicalIndicators.macd(result['close'])
            result['macd'] = macd_line
            result['macd_signal'] = signal_line
            result['macd_hist'] = histogram
            
            # RSI
            result['rsi'] = TechnicalIndicators.rsi(result['close'], 14)
            
            # Bollinger Bands
            upper, middle, lower = TechnicalIndicators.bollinger_bands(result['close'])
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower
            result['bb_width'] = (upper - lower) / middle
            
            # Average True Range
            result['atr'] = TechnicalIndicators.atr(result['high'], result['low'], result['close'], 14)
            
            # Percentage change and returns
            result['pct_change'] = result['close'].pct_change()
            result['log_return'] = np.log(result['close'] / result['close'].shift(1))
            
            # Volatility (20-day rolling standard deviation of returns)
            result['volatility'] = result['log_return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # Price relative to moving averages (trend strength indicators)
            result['price_sma20_ratio'] = result['close'] / result['sma20']
            result['price_sma50_ratio'] = result['close'] / result['sma50']
            
            # Custom Trend Strength Indicator
            result['trend_strength'] = abs(result['sma20'] - result['sma50']) / result['sma50']
            
            # Remove NaN values
            result.dropna(inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    @staticmethod
    def sma(series, window):
        """Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series, window):
        """Exponential Moving Average"""
        return series.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        """
        Moving Average Convergence Divergence
        
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(series, window=14):
        """Relative Strength Index"""
        delta = series.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        down = down.abs()
        
        avg_gain = up.rolling(window=window).mean()
        avg_loss = down.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(series, window=20, num_std=2):
        """
        Bollinger Bands
        
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        middle_band = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr