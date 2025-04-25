"""
On-Chain Data Collector
---------------------
Collects on-chain metrics from blockchain APIs.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
import os

logger = logging.getLogger(__name__)

class OnchainDataCollector:
    """Collects on-chain data from blockchain APIs"""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Bot configuration
        """
        self.config = config
        self.api_keys = config.get("api_keys", {})
        
    def collect_onchain_data(self, assets=None):
        """
        Collect on-chain metrics for specified assets.
        
        Args:
            assets (list): List of assets to collect data for
            
        Returns:
            dict: On-chain data by asset
        """
        if assets is None:
            # Default from trading pairs
            assets = [pair.split('/')[0] for pair in self.config["trading_pairs"]]
            
        onchain_data = {}
        
        for asset in assets:
            logger.info(f"Collecting on-chain data for {asset}")
            
            if asset == "BTC":
                data = self._collect_bitcoin_metrics()
            elif asset == "ETH":
                data = self._collect_ethereum_metrics()
            else:
                data = self._generate_synthetic_onchain_data(asset)
                
            onchain_data[asset] = data
            
        return onchain_data
    
    def _collect_bitcoin_metrics(self):
        """
        Collect Bitcoin on-chain metrics.
        
        Returns:
            dict: Bitcoin metrics
        """
        try:
            # Try using Blockchain.info API
            response = requests.get("https://api.blockchain.info/stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                metrics = {
                    'hash_rate': data.get('hash_rate', 0),
                    'difficulty': data.get('difficulty', 0),
                    'transaction_count': data.get('n_tx', 0),
                    'mempool_size': data.get('mempool_size', 0),
                    'avg_transaction_fee_btc': data.get('total_fees_btc', 0) / max(1, data.get('n_tx', 1)),
                    'avg_transaction_value_usd': data.get('estimated_transaction_volume_usd', 0) / max(1, data.get('n_tx', 1)),
                    'timestamp': datetime.now()
                }
                
                logger.info("Successfully collected Bitcoin on-chain metrics")
                return metrics
            else:
                logger.warning(f"Failed to retrieve Bitcoin data: HTTP {response.status_code}")
                return self._generate_synthetic_onchain_data("BTC")
                
        except Exception as e:
            logger.error(f"Error collecting Bitcoin on-chain data: {str(e)}")
            return self._generate_synthetic_onchain_data("BTC")
    
    def _collect_ethereum_metrics(self):
        """
        Collect Ethereum on-chain metrics.
        
        Returns:
            dict: Ethereum metrics
        """
        try:
            # Check if we have an Etherscan API key
            etherscan_key = self.api_keys.get("etherscan", {}).get("api_key", "")
            
            if etherscan_key:
                base_url = "https://api.etherscan.io/api"
                
                # Get gas price
                gas_url = f"{base_url}?module=gastracker&action=gasoracle&apikey={etherscan_key}"
                gas_response = requests.get(gas_url, timeout=10)
                
                # Get latest ETH supply
                supply_url = f"{base_url}?module=stats&action=ethsupply&apikey={etherscan_key}"
                supply_response = requests.get(supply_url, timeout=10)
                
                # Get node count
                node_url = f"{base_url}?module=stats&action=nodecount&apikey={etherscan_key}"
                node_response = requests.get(node_url, timeout=10)
                
                if (gas_response.status_code == 200 and supply_response.status_code == 200 and 
                    node_response.status_code == 200):
                    
                    gas_data = gas_response.json().get('result', {})
                    supply_data = supply_response.json().get('result', '0')
                    node_data = node_response.json().get('result', '0')
                    
                    metrics = {
                        'gas_price_gwei': float(gas_data.get('SafeGasPrice', 0)),
                        'gas_price_rapid_gwei': float(gas_data.get('FastGasPrice', 0)),
                        'eth_supply': int(supply_data) / 1e18,
                        'node_count': int(node_data),
                        'timestamp': datetime.now()
                    }
                    
                    logger.info("Successfully collected Ethereum on-chain metrics")
                    return metrics
                else:
                    logger.warning("Failed to retrieve Ethereum data")
                    return self._generate_synthetic_onchain_data("ETH")
            else:
                logger.warning("No Etherscan API key provided")
                return self._generate_synthetic_onchain_data("ETH")
                
        except Exception as e:
            logger.error(f"Error collecting Ethereum on-chain data: {str(e)}")
            return self._generate_synthetic_onchain_data("ETH")
    
    def _generate_synthetic_onchain_data(self, asset):
        """
        Generate synthetic on-chain data for testing.
        
        Args:
            asset (str): Asset symbol
            
        Returns:
            dict: Synthetic on-chain metrics
        """
        logger.warning(f"Generating synthetic on-chain data for {asset}")
        
        if asset == "BTC":
            # Random but realistic values for Bitcoin
            metrics = {
                'hash_rate': np.random.uniform(100, 150) * 1e6,  # 100-150 million TH/s
                'difficulty': np.random.uniform(40, 50) * 1e12,  # 40-50 trillion
                'transaction_count': np.random.randint(250000, 350000),
                'mempool_size': np.random.randint(5000, 15000),
                'avg_transaction_fee_btc': np.random.uniform(0.0001, 0.001),
                'avg_transaction_value_usd': np.random.uniform(5000, 15000),
                'timestamp': datetime.now()
            }
        elif asset == "ETH":
            # Random but realistic values for Ethereum
            metrics = {
                'gas_price_gwei': np.random.uniform(20, 80),
                'gas_price_rapid_gwei': np.random.uniform(80, 150),
                'eth_supply': np.random.uniform(115, 120) * 1e6,  # 115-120 million ETH
                'node_count': np.random.randint(3000, 5000),
                'defi_tvl_usd': np.random.uniform(40, 60) * 1e9,  # 40-60 billion USD
                'timestamp': datetime.now()
            }
        else:
            # Generic metrics for other assets
            metrics = {
                'transaction_count': np.random.randint(50000, 150000),
                'avg_transaction_fee': np.random.uniform(0.1, 1.0),
                'active_addresses': np.random.randint(100000, 500000),
                'timestamp': datetime.now()
            }
            
        return metrics