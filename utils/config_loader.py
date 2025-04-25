"""
Configuration utilities for the trading bot.
"""
import json
import os
import logging

logger = logging.getLogger(__name__)

def load_configuration(config_path):
    """
    Load bot configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        logger.info(f"Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise