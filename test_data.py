"""
Test script to verify data collection and technical indicators.
"""
from config.logging_config import setup_logging
from utils.config_loader import load_configuration
from data.collectors.market_data import MarketDataCollector
from data.indicators import TechnicalIndicators
import matplotlib.pyplot as plt
import os
import pandas as pd

# Setup logging
logger = setup_logging()

# Create directory for plots
if not os.path.exists("plots"):
    os.makedirs("plots")

def plot_with_indicators(df, pair, timeframe):
    """Create plots with technical indicators"""
    # Basic price plot
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 subplot grid
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['close'], label='Close')
    plt.plot(df.index, df['sma20'], label='SMA 20')
    plt.plot(df.index, df['sma50'], label='SMA 50')
    plt.plot(df.index, df['bb_upper'], 'r--', label='Upper BB')
    plt.plot(df.index, df['bb_lower'], 'r--', label='Lower BB')
    plt.title(f"{pair} - {timeframe} - Price and Moving Averages")
    plt.legend()
    plt.grid(True)
    
    # MACD plot
    plt.subplot(2, 2, 2)
    plt.plot(df.index, df['macd'], label='MACD')
    plt.plot(df.index, df['macd_signal'], label='Signal')
    plt.bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3)
    plt.title('MACD')
    plt.legend()
    plt.grid(True)
    
    # RSI plot
    plt.subplot(2, 2, 3)
    plt.plot(df.index, df['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI')
    plt.legend()
    plt.grid(True)
    
    # Volatility plot
    plt.subplot(2, 2, 4)
    plt.plot(df.index, df['volatility'], label='Volatility')
    plt.title('Volatility')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/{pair.replace('/', '_')}_{timeframe}_indicators.png")
    plt.close()
    
    logger.info(f"Created indicator plots for {pair} {timeframe}")

if __name__ == "__main__":
    # Load configuration
    logger.info("Loading configuration...")
    config = load_configuration("config/config.json")
    
    # Initialize data collector
    logger.info("Initializing data collector...")
    collector = MarketDataCollector(config)
    
    # Fetch data
    logger.info("Fetching historical data...")
    data = collector.load_dataset()
    
    # Process data with indicators
    logger.info("Calculating technical indicators...")
    indicator_data = {}
    
    # Check if we got data
    data_found = False
    
    for pair in config["trading_pairs"]:
        indicator_data[pair] = {}
        
        for timeframe in config["timeframes"]:
            if pair in data and timeframe in data[pair] and not data[pair][timeframe].empty:
                df = data[pair][timeframe]
                data_found = True
                logger.info(f"{pair} {timeframe}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                
                # Add technical indicators
                df_with_indicators = TechnicalIndicators.add_indicators(df)
                indicator_data[pair][timeframe] = df_with_indicators
                
                # Plot basic price chart
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['close'], label="Close Price")
                plt.title(f"{pair} - {timeframe}")
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"plots/{pair.replace('/', '_')}_{timeframe}_price.png")
                plt.close()
                
                # Plot with indicators
                plot_with_indicators(df_with_indicators, pair, timeframe)
                
                # Plot candlesticks if mplfinance is available
                try:
                    import mplfinance as mpf
                    df_copy = df.copy()
                    mpf.plot(df_copy, type='candle', style='yahoo',
                            title=f"{pair} - {timeframe}",
                            ylabel='Price',
                            volume=True,
                            savefig=f"plots/{pair.replace('/', '_')}_{timeframe}_candle.png")
                    logger.info(f"Candlestick plot created for {pair} {timeframe}")
                except ImportError:
                    logger.warning("mplfinance not installed, skipping candlestick plot")
            else:
                logger.warning(f"No valid data found for {pair} {timeframe}")
    
    if not data_found:
        logger.error("No data was found for any trading pair. Please check your configuration and network connection.")
    else:
        logger.info("Successfully processed data and created plots. Check the 'plots' directory.")
        
        # Basic statistics
        logger.info("\nBasic Statistics:")
        for pair in indicator_data:
            for timeframe in indicator_data[pair]:
                df = indicator_data[pair][timeframe]
                if not df.empty:
                    logger.info(f"\n{pair} {timeframe} Statistics:")
                    logger.info(f"Current Price: {df['close'].iloc[-1]:.2f}")
                    logger.info(f"50-day SMA: {df['sma50'].iloc[-1]:.2f}")
                    logger.info(f"Current RSI: {df['rsi'].iloc[-1]:.2f}")
                    logger.info(f"Current MACD: {df['macd'].iloc[-1]:.4f}")
                    logger.info(f"Current Volatility: {df['volatility'].iloc[-1]:.4f}")