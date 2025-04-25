"""
Market Data Collector
--------------------
Fetches and processes market data from exchanges.
"""
import logging
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Collects market data from exchanges"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.trading_pairs = config["trading_pairs"]
        self.timeframes = config["timeframes"]
        
        # Initialize exchange connection if API keys provided
        self.exchange = None
        if "api_keys" in config and "binance" in config["api_keys"]:
            self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Set up connection to the exchange"""
        try:
            # Connect to Binance using ccxt
            self.exchange = ccxt.binance({
                'apiKey': self.config["api_keys"]["binance"]["api_key"],
                'secret': self.config["api_keys"]["binance"]["api_secret"],
                'enableRateLimit': True,
                'timeout': 30000  # 30 seconds timeout
            })
            
            # Test connection with minimal operation
            self.exchange.load_markets()
            logger.info("Exchange connection established successfully")
        except Exception as e:
            logger.error(f"Exchange connection error: {str(e)}")
            # Set to None, will use public methods
            self.exchange = None
    
    def fetch_historical_data(self, symbol, timeframe, limit=500, max_retries=3):
        """
        Fetch OHLCV data from exchange with retry logic.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe (1h, 4h, 1d)
            limit (int): Number of candles to fetch
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            DataFrame: OHLCV data
        """
        for attempt in range(max_retries):
            try:
                # Try using public API endpoint
                exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'timeout': 30000,  # Longer timeout
                    'options': {'defaultType': 'spot'}  # Explicitly use spot market
                })
                
                logger.info(f"Attempt {attempt+1}: Fetching data for {symbol} ({timeframe})")
                
                # Fetch data 
                candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # Progressive backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {symbol} ({timeframe}): {str(e)}")
                    # Try fallback method as last resort
                    return self._fetch_fallback_data(symbol, timeframe)
        
        return pd.DataFrame()  # Return empty DataFrame if all attempts fail
    
    def _fetch_fallback_data(self, symbol, timeframe):
        """
        Fallback method to generate sample data for testing purposes.
        In a real scenario, this could try an alternative data source.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            DataFrame: Synthetic OHLCV data for testing
        """
        logger.warning(f"Using synthetic data for {symbol} ({timeframe})")
        
        # Create date range
        end_date = datetime.now()
        if timeframe == '1h':
            hours = 500
            start_date = end_date - timedelta(hours=hours)
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        elif timeframe == '4h':
            hours = 500 * 4
            start_date = end_date - timedelta(hours=hours)
            dates = pd.date_range(start=start_date, end=end_date, freq='4H')
        else:  # Default to daily
            days = 500
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        # Generate synthetic price data
        base_price = 30000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
        np.random.seed(42)  # For reproducibility
        
        # Create random walk
        random_walk = np.random.normal(0, 0.02, size=len(dates))
        cumulative_returns = np.exp(np.cumsum(random_walk)) 
        prices = base_price * cumulative_returns
        
        # Create OHLCV data
        volatility = base_price * 0.02
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, size=len(df)))
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, volatility, size=len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, volatility, size=len(df))
        df['volume'] = np.random.uniform(base_price*5, base_price*20, size=len(df))
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    def load_dataset(self):
        """
        Load historical data for all trading pairs and timeframes.
        
        Returns:
            dict: Historical data organized by pair and timeframe
        """
        dataset = {}
        
        for pair in self.trading_pairs:
            dataset[pair] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Loading data for {pair} ({timeframe})")
                
                # Fetch historical data
                data = self.fetch_historical_data(
                    pair, 
                    timeframe, 
                    limit=self.config.get("backtesting", {}).get("lookback_periods", 500)
                )
                
                if not data.empty:
                    dataset[pair][timeframe] = data
                    logger.info(f"Loaded {len(data)} candles for {pair} ({timeframe})")
                else:
                    logger.warning(f"No data retrieved for {pair} ({timeframe})")
        
        return dataset