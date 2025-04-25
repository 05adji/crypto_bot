"""
Main entry point for the trading bot.
"""
import argparse
import logging
from datetime import datetime, timedelta
from bot import CryptoTradingBot
from config.logging_config import setup_logging

logger = setup_logging()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    
    parser.add_argument('--mode', type=str, default='backtest',
                      choices=['live', 'backtest'],
                      help='Trading mode: live or backtest')
    
    parser.add_argument('--config', type=str, default='config/config.json',
                      help='Path to configuration file')
    
    parser.add_argument('--days', type=int, default=90,
                      help='Number of days for backtest')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Time between trading cycles in seconds (for live mode)')
    
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable plotting in backtest mode')
    
    parser.add_argument('--paper-trading', action='store_true',
                      help='Run in paper trading mode (no real money)')
    
    parser.add_argument('--discord-notify', action='store_true',
                      help='Enable Discord notifications')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Initialize bot
        bot = CryptoTradingBot(args.config, paper_trading=args.paper_trading, 
                             discord_notify=args.discord_notify)
        
        if args.mode == 'live':
            logger.info(f"Starting bot in LIVE mode (Paper Trading: {args.paper_trading})")
            bot.run_live(cycle_interval=args.interval)
        else:
            logger.info(f"Starting bot in BACKTEST mode for {args.days} days")
            bot.run_backtest(days=args.days, plot_results=not args.no_plot)
            
    except Exception as e:
        logger.exception(f"Error running bot: {str(e)}")

if __name__ == "__main__":
    main()