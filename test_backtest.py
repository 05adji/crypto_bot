"""
Test script for backtesting.
"""
from config.logging_config import setup_logging
from utils.config_loader import load_configuration
from data.collectors.market_data import MarketDataCollector
from strategies.moving_average_strategy import MovingAverageStrategy
from backtesting.backtester import Backtester
from datetime import datetime, timedelta

# Setup logging
logger = setup_logging()

if __name__ == "__main__":
    # Load configuration
    logger.info("Loading configuration...")
    config = load_configuration("config/config.json")
    
    # Initialize data collector
    logger.info("Initializing data collector...")
    collector = MarketDataCollector(config)
    
    # Initialize strategy
    logger.info("Initializing trading strategy...")
    strategy = MovingAverageStrategy(config)
    
    # Initialize backtester
    logger.info("Initializing backtester...")
    backtester = Backtester(config, strategy, collector)
    
    # Define backtest period (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Run backtest
    logger.info("Running backtest...")
    results = backtester.run_backtest(start_date, end_date, plot_results=True)
    
    # Display results
    if results:
        logger.info("\nBacktest Results:")
        logger.info(f"Initial Portfolio: ${results['initial_value']:.2f}")
        logger.info(f"Final Portfolio: ${results['final_value']:.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Annual Return: {results['annual_return_pct']:.2f}%")
        logger.info(f"Volatility: {results['volatility_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
        logger.info(f"Win Rate: {results['win_rate_pct']:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Buy Trades: {results['buy_trades']}")
        logger.info(f"Sell Trades: {results['sell_trades']}")
        logger.info(f"Trading Days: {results['trading_days']}")
        logger.info(f"Duration: {results['duration_years']:.2f} years")
        logger.info("\nBacktest completed successfully. Check the 'plots' directory for visualizations.")
    else:
        logger.error("Backtest failed to produce results.")