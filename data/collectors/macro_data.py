"""
Macroeconomic Data Collector
--------------------------
Collects macroeconomic indicators for market analysis.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os

logger = logging.getLogger(__name__)

class MacroDataCollector:
    """Collects macroeconomic data from various sources"""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Bot configuration
        """
        self.config = config
        self.api_keys = config.get("api_keys", {})
        
        # Load cached data if available
        self.cache_file = os.path.join("data", "macro_data_cache.csv")
        self.macro_data = self._load_cached_data()
        
        # Set up indicators to track
        self.indicators = config.get("strategies", {}).get("macro", {}).get("indicators", [
            "DFF", "UNRATE", "CPIAUCSL", "T10Y2Y", "VIXCLS"
        ])
    
    def _load_cached_data(self):
        """Load cached macroeconomic data"""
        try:
            if os.path.exists(self.cache_file):
                df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached macro data with {len(df)} entries")
                return df
        except Exception as e:
            logger.error(f"Error loading cached macro data: {str(e)}")
            
        return pd.DataFrame()
    
    def _save_cached_data(self, df):
        """Save macroeconomic data to cache"""
        try:
            os.makedirs("data", exist_ok=True)
            df.to_csv(self.cache_file)
            logger.info(f"Saved macro data to cache ({len(df)} entries)")
        except Exception as e:
            logger.error(f"Error saving macro data to cache: {str(e)}")
    
    def collect_macro_data(self):
        """
        Collect macroeconomic indicators.
        
        Returns:
            DataFrame: Macroeconomic indicators
        """
        # Check if we need to update the data
        if not self.macro_data.empty:
            last_update = self.macro_data.index.max()
            days_since_update = (datetime.now() - last_update).days
            
            # If updated recently (within last week), use cached data
            if days_since_update < 7:
                logger.info(f"Using cached macro data (last updated {days_since_update} days ago)")
                return self.macro_data
        
        logger.info("Collecting macroeconomic data...")
        
        try:
            # Try using FRED API if available
            fred_api_key = self.api_keys.get("fred", {}).get("api_key", "")
            
            if fred_api_key:
                data = self._collect_fred_data(fred_api_key)
            else:
                logger.warning("No FRED API key provided, using synthetic data")
                data = self._generate_synthetic_macro_data()
                
            # Cache the data
            if not data.empty:
                self._save_cached_data(data)
                self.macro_data = data
                
            return data
            
        except Exception as e:
            logger.error(f"Error collecting macro data: {str(e)}")
            
            # Return cached data if available, otherwise generate synthetic data
            if not self.macro_data.empty:
                return self.macro_data
            else:
                return self._generate_synthetic_macro_data()
    
    def _collect_fred_data(self, api_key):
        """
        Collect data from FRED API.
        
        Args:
            api_key (str): FRED API key
            
        Returns:
            DataFrame: Macroeconomic indicators
        """
        macro_data = {}
        
        for indicator_id in self.indicators:
            logger.info(f"Fetching {indicator_id} from FRED")
            
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={indicator_id}&api_key={api_key}&file_type=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                observations = response.json().get('observations', [])
                
                dates = []
                values = []
                
                for obs in observations:
                    if obs['value'] != '.':  # Skip missing values
                        dates.append(obs['date'])
                        values.append(float(obs['value']))
                
                macro_data[indicator_id] = pd.Series(values, index=pd.to_datetime(dates))
                
                logger.info(f"Fetched {len(values)} observations for {indicator_id}")
        
        # Convert to DataFrame
        if macro_data:
            df = pd.DataFrame(macro_data)
            
            # Fill missing values
            df = df.resample('D').asfreq()  # Ensure daily frequency
            df = df.interpolate(method='linear')  # Interpolate missing values
            
            return df
        else:
            return pd.DataFrame()
    
    def _generate_synthetic_macro_data(self):
        """
        Generate synthetic macroeconomic data for testing.
        
        Returns:
            DataFrame: Synthetic macroeconomic data
        """
        logger.warning("Generating synthetic macroeconomic data")
        
        # Create date range for last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic data with reasonable values
        df = pd.DataFrame(index=dates)
        
        # Federal Funds Rate (DFF): Typically 0-5%
        base_rate = 2.0
        df['DFF'] = base_rate + np.cumsum(np.random.normal(0, 0.01, size=len(dates)))
        df['DFF'] = np.clip(df['DFF'], 0, 5)
        
        # Unemployment Rate (UNRATE): Typically 3-10%
        base_unemployment = 4.0
        df['UNRATE'] = base_unemployment + np.cumsum(np.random.normal(0, 0.005, size=len(dates)))
        df['UNRATE'] = np.clip(df['UNRATE'], 3, 10)
        
        # Consumer Price Index (CPIAUCSL): Always increasing
        base_cpi = 280
        df['CPIAUCSL'] = base_cpi + np.cumsum(np.random.uniform(0, 0.1, size=len(dates)))
        
        # Treasury Yield Spread (T10Y2Y): Typically -2 to 2
        base_spread = 0.5
        df['T10Y2Y'] = base_spread + np.cumsum(np.random.normal(0, 0.01, size=len(dates)))
        df['T10Y2Y'] = np.clip(df['T10Y2Y'], -2, 2)
        
        # VIX Volatility Index (VIXCLS): Typically 10-40
        base_vix = 20
        df['VIXCLS'] = base_vix + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 2, size=len(dates))
        df['VIXCLS'] = np.clip(df['VIXCLS'], 10, 40)
        
        return df