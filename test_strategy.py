"""
Test script for strategy implementation.
"""
from config.logging_config import setup_logging
from utils.config_loader import load_configuration
from data.collectors.market_data import MarketDataCollector
from data.indicators import TechnicalIndicators
from strategies.moving_average_strategy import MovingAverageStrategy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Setup logging
logger = setup_logging()

# Create directory for plots
if not os.path.exists("plots"):
    os.makedirs("plots")

def plot_signals(df, pair, timeframe, signals):
    """Create a plot showing entry/exit signals"""
    plt.figure(figsize=(12, 8))
    
    # Price plot with MAs
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Close')
    plt.plot(df.index, df['ema12'], label='EMA 12')
    plt.plot(df.index, df['ema26'], label='EMA 26')
    
    # Mark buy and sell signals
    buy_points = []
    sell_points = []
    
    # Generate some historical signals for visualization
    for i in range(10, len(df)):
        # Simple detection of crossovers for visualization
        if df['ema12'].iloc[i] > df['ema26'].iloc[i] and df['ema12'].iloc[i-1] <= df['ema26'].iloc[i-1]:
            buy_points.append((df.index[i], df['close'].iloc[i]))
        elif df['ema12'].iloc[i] < df['ema26'].iloc[i] and df['ema12'].iloc[i-1] >= df['ema26'].iloc[i-1]:
            sell_points.append((df.index[i], df['close'].iloc[i]))
    
    # Plot the points
    if buy_points:
        buy_x, buy_y = zip(*buy_points)
        plt.scatter(buy_x, buy_y, color='green', s=100, marker='^', label='Buy Signal')
    
    if sell_points:
        sell_x, sell_y = zip(*sell_points)
        plt.scatter(sell_x, sell_y, color='red', s=100, marker='v', label='Sell Signal')
    
    # Mark current signal if it exists
    current_signal = signals[pair][timeframe]['signal']
    if current_signal.name == 'BUY':
        plt.scatter([df.index[-1]], [df['close'].iloc[-1]], color='green', s=200, marker='*', label='Current: BUY')
    elif current_signal.name == 'SELL':
        plt.scatter([df.index[-1]], [df['close'].iloc[-1]], color='red', s=200, marker='*', label='Current: SELL')
    
    plt.title(f"{pair} - {timeframe} - Moving Average Strategy")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    # RSI plot
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI')
    plt.ylabel("RSI Value")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/{pair.replace('/', '_')}_{timeframe}_strategy_signals.png")
    plt.close()
    
    logger.info(f"Created strategy signal plot for {pair} {timeframe}")

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
    
    for pair in config["trading_pairs"]:
        indicator_data[pair] = {}
        
        for timeframe in config["timeframes"]:
            if pair in data and timeframe in data[pair] and not data[pair][timeframe].empty:
                df = data[pair][timeframe]
                logger.info(f"{pair} {timeframe}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                
                # Add technical indicators
                indicator_data[pair][timeframe] = TechnicalIndicators.add_indicators(df)
    
    # Initialize strategy
    logger.info("Initializing trading strategy...")
    strategy = MovingAverageStrategy(config)
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = strategy.generate_signals(indicator_data)
    
    # Display and visualize results
    for pair in signals:
        for timeframe in signals[pair]:
            signal_data = signals[pair][timeframe]
            
            logger.info(f"\n{pair} {timeframe} Signal:")
            logger.info(f"Signal: {signal_data['signal'].name}")
            logger.info(f"Confidence: {signal_data['confidence']:.2f}")
            logger.info(f"Reason: {signal_data['reason']}")
            logger.info(f"Current Price: {signal_data['current_price']:.2f}")
            logger.info(f"RSI: {signal_data['rsi']:.2f}")
            
            # Create signal visualization
            if pair in indicator_data and timeframe in indicator_data[pair]:
                plot_signals(indicator_data[pair][timeframe], pair, timeframe, signals)
    
    logger.info("\nStrategy test completed successfully.")