"""
Crypto Trading Bot
-----------------
Main bot controller that ties all components together.
"""
import logging
import time
import os
from datetime import datetime, timedelta

from config.logging_config import setup_logging
from utils.config_loader import load_configuration
from data.collectors.market_data import MarketDataCollector
from data.collectors.sentiment_data import SentimentDataCollector
from data.collectors.onchain_data import OnchainDataCollector
from data.collectors.macro_data import MacroDataCollector
from data.indicators import TechnicalIndicators
from strategies.moving_average_strategy import MovingAverageStrategy
from strategies.pair_trading_strategy import PairTradingStrategy
from strategies.manipulation_detector import ManipulationDetector
from risk.risk_manager import RiskManager
from execution.portfolio_manager import PortfolioManager
from backtesting.backtester import Backtester
from strategies.base_strategy import SignalType

logger = setup_logging()

class CryptoTradingBot:
    """Main trading bot controller"""
    
    def __init__(self, config_path="config/config.json"):
        """
        Initialize trading bot.
        
        Args:
            config_path (str): Path to configuration file
        """
        logger.info(f"Initializing trading bot with config: {config_path}")
        
        # Load configuration
        self.config = load_configuration(config_path)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Trading bot initialized successfully")
    
    def _initialize_components(self):
        """Initialize all components"""
        # Data collectors
        self.market_collector = MarketDataCollector(self.config)
        self.sentiment_collector = SentimentDataCollector(self.config)
        self.onchain_collector = OnchainDataCollector(self.config)
        self.macro_collector = MacroDataCollector(self.config)
        
        # Strategies
        self.ma_strategy = MovingAverageStrategy(self.config)
        self.pair_strategy = PairTradingStrategy(self.config)
        
        # Risk management
        self.risk_manager = RiskManager(self.config)
        
        # Manipulation detection
        self.manipulation_detector = ManipulationDetector(self.config)
        
        # Portfolio management
        self.portfolio_manager = PortfolioManager(self.config)
        
        # Initialize backtester
        self.backtester = Backtester(self.config, self.ma_strategy, self.market_collector)
        
        # Create directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Load initial data
        self._load_initial_data()
        
        logger.info("All components initialized successfully")
    
    def _load_initial_data(self):
        """Load initial data for all sources"""
        logger.info("Loading initial data...")
        
        # Market data
        self.market_data = self.market_collector.load_dataset()
        
        # Calculate technical indicators
        self.indicators_data = {}
        for pair in self.market_data:
            self.indicators_data[pair] = {}
            for timeframe in self.market_data[pair]:
                if not self.market_data[pair][timeframe].empty:
                    self.indicators_data[pair][timeframe] = TechnicalIndicators.add_indicators(
                        self.market_data[pair][timeframe]
                    )
        
        # Sentiment data (for main assets)
        self.sentiment_data = {}
        for pair in self.config["trading_pairs"]:
            asset = pair.split('/')[0]
            self.sentiment_data[asset] = self.sentiment_collector.collect_sentiment_data(asset)
        
        # On-chain data
        self.onchain_data = self.onchain_collector.collect_onchain_data()
        
        # Macroeconomic data
        self.macro_data = self.macro_collector.collect_macro_data()
        
        # Detect market manipulation
        self.manipulation_data = self.manipulation_detector.detect_manipulation(self.market_data)
        
        # Find cointegrated pairs
        self.pair_strategy.find_cointegrated_pairs(self.market_data)
        
        logger.info("Initial data loaded successfully")
    
    def run_live(self, cycle_interval=60):
        """
        Run bot in live trading mode.
        
        Args:
            cycle_interval (int): Time between trading cycles in seconds
        """
        logger.info("Starting bot in LIVE mode")
        
        try:
            # Main trading loop
            while True:
                cycle_start = datetime.now()
                
                # 1. Update all data
                logger.info("Updating market data...")
                self.market_data = self.market_collector.load_dataset()
                
                logger.info("Calculating technical indicators...")
                self.indicators_data = {}
                for pair in self.market_data:
                    self.indicators_data[pair] = {}
                    for timeframe in self.market_data[pair]:
                        if not self.market_data[pair][timeframe].empty:
                            self.indicators_data[pair][timeframe] = TechnicalIndicators.add_indicators(
                                self.market_data[pair][timeframe]
                            )
                
                # Update other data sources periodically
                current_hour = datetime.now().hour
                if current_hour % 6 == 0:  # Every 6 hours
                    logger.info("Updating sentiment data...")
                    for pair in self.config["trading_pairs"]:
                        asset = pair.split('/')[0]
                        self.sentiment_data[asset] = self.sentiment_collector.collect_sentiment_data(asset)
                        
                    logger.info("Updating on-chain data...")
                    self.onchain_data = self.onchain_collector.collect_onchain_data()
                
                if current_hour == 0:  # Once a day
                    logger.info("Updating macroeconomic data...")
                    self.macro_data = self.macro_collector.collect_macro_data()
                    
                    logger.info("Updating cointegrated pairs...")
                    self.pair_strategy.find_cointegrated_pairs(self.market_data)
                
                # 2. Detect market manipulation
                logger.info("Checking for market manipulation...")
                self.manipulation_data = self.manipulation_detector.detect_manipulation(self.market_data)
                
                # 3. Generate trading signals from multiple strategies
                logger.info("Generating trading signals...")
                ma_signals = self.ma_strategy.generate_signals(self.indicators_data)
                pair_signals = self.pair_strategy.generate_signals(self.market_data)
                
                # 4. Combine signals
                combined_signals = self._combine_strategy_signals(ma_signals, pair_signals)
                
                # 5. Apply manipulation filter
                filtered_signals = self._filter_manipulation_signals(combined_signals, self.manipulation_data)
                
                # 6. Apply macro analysis filter
                macro_filtered_signals = self._apply_macro_filter(filtered_signals)
                
                # 7. Apply sentiment analysis
                final_signals = self._apply_sentiment_analysis(macro_filtered_signals)
                
                # 8. Execute trades based on signals
                self._execute_live_trades(final_signals, self.market_data)
                
                # 9. Update portfolio value
                portfolio_value = self.portfolio_manager.calculate_portfolio_value(self.market_data)
                
                # 10. Update risk metrics
                within_limits = self.risk_manager.update_drawdown(portfolio_value)
                
                if not within_limits:
                    logger.warning("Maximum drawdown limit exceeded. Bot is in risk-off mode.")
                
                # 11. Manage existing positions (stop loss, take profit)
                self._manage_existing_positions()
                
                # 12. Log current status
                self._log_status(portfolio_value)
                
                # 13. Calculate sleep time
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(cycle_interval - cycle_duration, 0)
                
                logger.info(f"Trading cycle completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
        except Exception as e:
            logger.exception(f"Error in live trading: {str(e)}")
    
    def _combine_strategy_signals(self, ma_signals, pair_signals):
        """
        Combine signals from multiple strategies.
        
        Args:
            ma_signals (dict): Signals from moving average strategy
            pair_signals (dict): Signals from pair trading strategy
            
        Returns:
            dict: Combined signals
        """
        combined = ma_signals.copy()
        
        # Add pair trading signals
        for pair_key, signal in pair_signals.items():
            combined[pair_key] = signal
        
        return combined
    
    def _filter_manipulation_signals(self, signals, manipulation_data):
        """
        Filter out signals for assets with detected manipulation.
        
        Args:
            signals (dict): Trading signals
            manipulation_data (dict): Manipulation detection results
            
        Returns:
            dict: Filtered signals
        """
        filtered = signals.copy()
        
        for pair, manipulation in manipulation_data.items():
            # If manipulation detected with high confidence, reduce signal confidence or remove
            pump_dump_detected = manipulation.get('pump_dump_detected', False)
            wash_trading_detected = manipulation.get('wash_trading_detected', False)
            
            if pump_dump_detected or wash_trading_detected:
                confidence = max(
                    manipulation.get('pump_dump_confidence', 0),
                    manipulation.get('wash_trading_confidence', 0)
                )
                
                if confidence > 0.7:
                    logger.warning(f"Filtering signals for {pair} due to detected manipulation (confidence: {confidence:.2f})")
                    
                    # If pair in signals, reduce confidence or set to NEUTRAL
                    if pair in filtered:
                        # For high confidence manipulation, set to NEUTRAL
                        if confidence > 0.9:
                            for timeframe in filtered[pair]:
                                filtered[pair][timeframe]['signal'] = SignalType.NEUTRAL
                                filtered[pair][timeframe]['confidence'] = 0.1
                                filtered[pair][timeframe]['reason'] = "Signal filtered due to detected market manipulation"
                        else:
                            # Otherwise just reduce confidence
                            for timeframe in filtered[pair]:
                                filtered[pair][timeframe]['confidence'] *= (1 - confidence)
                                filtered[pair][timeframe]['reason'] += " (reduced confidence due to potential manipulation)"
        
        return filtered
    
    def _apply_macro_filter(self, signals):
        """
        Apply macroeconomic analysis to filter signals.
        
        Args:
            signals (dict): Trading signals
            
        Returns:
            dict: Filtered signals
        """
        filtered = signals.copy()
        
        # Skip if macro data is empty
        if self.macro_data is None or self.macro_data.empty:
            return filtered
        
        # Get latest macro indicators
        latest_macro = self.macro_data.iloc[-1]
        
        # Determine market regime based on macro indicators
        vix = latest_macro.get('VIXCLS', 20)  # VIX Volatility Index
        yield_curve = latest_macro.get('T10Y2Y', 0.5)  # 10Y-2Y Treasury Spread
        
        # High fear (VIX > 30) or inverted yield curve (< 0) -> risk-off
        if vix > 30 or yield_curve < 0:
            market_regime = 'RISK_OFF'
            logger.info(f"Macro Analysis: Risk-off regime detected (VIX: {vix:.2f}, Yield Curve: {yield_curve:.2f})")
            
            # In risk-off mode, reduce buy signal confidence
            for pair in filtered:
                if pair.endswith('_PAIR'):
                    continue  # Skip pair signals
                    
                for timeframe in filtered[pair]:
                    signal = filtered[pair][timeframe]
                    if signal['signal'] == SignalType.BUY:
                        # Reduce confidence by 30%
                        filtered[pair][timeframe]['confidence'] *= 0.7
                        filtered[pair][timeframe]['reason'] += " (confidence reduced due to risk-off macro environment)"
                        
                        # If confidence too low, switch to NEUTRAL
                        if filtered[pair][timeframe]['confidence'] < 0.4:
                            filtered[pair][timeframe]['signal'] = SignalType.NEUTRAL
                            filtered[pair][timeframe]['reason'] = "Insufficient confidence in risk-off environment"
        
        # Low fear (VIX < 15) and steep yield curve (> 1) -> risk-on
        elif vix < 15 and yield_curve > 1:
            market_regime = 'RISK_ON'
            logger.info(f"Macro Analysis: Risk-on regime detected (VIX: {vix:.2f}, Yield Curve: {yield_curve:.2f})")
            
            # In risk-on mode, slightly boost buy signals
            for pair in filtered:
                if pair.endswith('_PAIR'):
                    continue  # Skip pair signals
                    
                for timeframe in filtered[pair]:
                    signal = filtered[pair][timeframe]
                    if signal['signal'] == SignalType.BUY:
                        # Increase confidence by 20%, capped at 0.95
                        filtered[pair][timeframe]['confidence'] = min(signal['confidence'] * 1.2, 0.95)
                        filtered[pair][timeframe]['reason'] += " (confidence boosted due to risk-on macro environment)"
        
        else:
            market_regime = 'NEUTRAL'
            logger.info(f"Macro Analysis: Neutral market regime (VIX: {vix:.2f}, Yield Curve: {yield_curve:.2f})")
            # No adjustments in neutral regime
        
        return filtered
    
    def _apply_sentiment_analysis(self, signals):
        """
        Apply sentiment analysis to adjust signal confidence.
        
        Args:
            signals (dict): Trading signals
            
        Returns:
            dict: Adjusted signals
        """
        adjusted = signals.copy()
        
        # Skip if no sentiment data
        if not self.sentiment_data:
            return adjusted
        
        for pair in adjusted:
            if pair.endswith('_PAIR'):
                continue  # Skip pair signals
                
            # Get the base asset
            base_asset = pair.split('/')[0]
            
            if base_asset in self.sentiment_data and not self.sentiment_data[base_asset].empty:
                # Get latest sentiment
                latest_sentiment = self.sentiment_data[base_asset].iloc[-1]
                sentiment_score = latest_sentiment.get('weighted_sentiment', 0)
                
                # Only adjust if sentiment is strong enough
                if abs(sentiment_score) > 0.2:
                    logger.info(f"Sentiment for {base_asset}: {sentiment_score:.2f}")
                    
                    for timeframe in adjusted[pair]:
                        signal = adjusted[pair][timeframe]
                        
                        # Positive sentiment boosts buy signals and reduces sell signals
                        if sentiment_score > 0.2:
                            if signal['signal'] == SignalType.BUY:
                                # Boost confidence by up to 25%
                                boost = min(sentiment_score, 0.5) / 2  # Max 25% boost
                                adjusted[pair][timeframe]['confidence'] = min(signal['confidence'] * (1 + boost), 0.95)
                                adjusted[pair][timeframe]['reason'] += f" (confidence boosted by {boost:.0%} due to positive sentiment)"
                            elif signal['signal'] == SignalType.SELL:
                                # Reduce confidence
                                reduction = min(sentiment_score, 0.5) / 2  # Max 25% reduction
                                adjusted[pair][timeframe]['confidence'] *= (1 - reduction)
                                adjusted[pair][timeframe]['reason'] += f" (confidence reduced by {reduction:.0%} due to positive sentiment)"
                        
                        # Negative sentiment boosts sell signals and reduces buy signals
                        elif sentiment_score < -0.2:
                            if signal['signal'] == SignalType.SELL:
                                # Boost confidence by up to 25%
                                boost = min(abs(sentiment_score), 0.5) / 2  # Max 25% boost
                                adjusted[pair][timeframe]['confidence'] = min(signal['confidence'] * (1 + boost), 0.95)
                                adjusted[pair][timeframe]['reason'] += f" (confidence boosted by {boost:.0%} due to negative sentiment)"
                            elif signal['signal'] == SignalType.BUY:
                                # Reduce confidence
                                reduction = min(abs(sentiment_score), 0.5) / 2  # Max 25% reduction
                                adjusted[pair][timeframe]['confidence'] *= (1 - reduction)
                                adjusted[pair][timeframe]['reason'] += f" (confidence reduced by {reduction:.0%} due to negative sentiment)"
        
        return adjusted
    
    def _execute_live_trades(self, signals, market_data):
        """
        Execute trades based on signals in live mode.
        
        Args:
            signals (dict): Trading signals
            market_data (dict): Market data
        """
        # Track which pairs have been traded this cycle
        traded_pairs = set()
        
        # Process regular trading signals first
        for pair in signals:
            # Skip pair trading signals (processed separately)
            if '_PAIR' in pair:
                continue
                
            for timeframe in signals[pair]:
                signal = signals[pair][timeframe]
                signal_type = signal['signal']
                
                # Skip if no clear signal or already traded this pair
                if signal_type == SignalType.NEUTRAL or pair in traded_pairs:
                    continue
                
                # Get current price
                current_price = signal['current_price']
                
                # Check if we have a position already
                position = self.portfolio_manager.get_position(pair)
                
                if signal_type == SignalType.BUY:
                    # Only buy if we don't have a position already
                    if position is None:
                        # Check if we have enough confidence
                        if signal['confidence'] < 0.5:
                            logger.info(f"Skipping {pair} BUY signal: Confidence too low ({signal['confidence']:.2f})")
                            continue
                        
                        # Calculate position size
                        position_usd, quantity = self.risk_manager.calculate_position_size(
                            self.portfolio_manager.cash,
                            current_price,
                            signal['confidence'],
                            signal.get('volatility', None)
                        )
                        
                        # Ensure we have enough cash and quantity is reasonable
                        if position_usd <= self.portfolio_manager.cash and quantity > 0:
                            # Execute trade
                            trade = {
                                'symbol': pair,
                                'side': 'buy',
                                'quantity': quantity,
                                'price': current_price,
                                'timestamp': datetime.now()
                            }
                            
                            logger.info(f"EXECUTING: BUY {quantity:.6f} {pair} @ ${current_price:.2f}")
                            self.portfolio_manager.update_portfolio(trade)
                            
                            # Mark this pair as traded
                            traded_pairs.add(pair)
                
                elif signal_type == SignalType.SELL:
                    # Only sell if we have a position
                    if position is not None:
                        # Sell entire position
                        quantity = position['quantity']
                        
                        # Execute trade
                        trade = {
                            'symbol': pair,
                            'side': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'timestamp': datetime.now()
                        }
                        
                        logger.info(f"EXECUTING: SELL {quantity:.6f} {pair} @ ${current_price:.2f}")
                        self.portfolio_manager.update_portfolio(trade)
                        
                        # Mark this pair as traded
                        traded_pairs.add(pair)
        
        # Process pair trading signals
        for pair_key in [p for p in signals if '_PAIR' in p]:
            signal = signals[pair_key]
            
            if signal['signal'] == SignalType.NEUTRAL:
                continue
                
            # Extract assets from pair
            asset1 = signal['asset1']
            asset2 = signal['asset2']
            
            # Find the corresponding trading pairs
            asset1_pair = next((p for p in self.config["trading_pairs"] if p.startswith(f"{asset1}/")), None)
            asset2_pair = next((p for p in self.config["trading_pairs"] if p.startswith(f"{asset2}/")), None)
            
            if not asset1_pair or not asset2_pair:
                logger.warning(f"Cannot execute pair trade: Missing trading pairs for {asset1} or {asset2}")
                continue
                
            # Skip if either pair was already traded this cycle
            if asset1_pair in traded_pairs or asset2_pair in traded_pairs:
                continue
                
            # Get current prices
            price1 = signal['price1']
            price2 = signal['price2']
            
            # Get hedge ratio for balanced exposure
            hedge_ratio = signal['hedge_ratio']
            
            # Calculate position sizes for balanced pair trade (10% of portfolio)
            portfolio_value = self.portfolio_manager.calculate_portfolio_value(market_data)
            pair_allocation = portfolio_value * 0.1 * signal['confidence']
            
            # Ensure minimum allocation
            if pair_allocation < 20:
                logger.info(f"Skipping pair trade {asset1}-{asset2}: Allocation too small (${pair_allocation:.2f})")
                continue
                
            # Split allocation between the two assets
            asset1_allocation = pair_allocation / 2
            asset2_allocation = pair_allocation / 2
            
            # Calculate quantities
            asset1_quantity = asset1_allocation / price1
            asset2_quantity = (asset2_allocation / price2) * hedge_ratio  # Apply hedge ratio
            
            # Execute trades based on signal
            if signal['signal'] == SignalType.BUY:
                # BUY asset1, SELL asset2
                logger.info(f"EXECUTING PAIR TRADE: BUY {asset1}, SELL {asset2}")
                
                # Check if we can execute both sides
                position2 = self.portfolio_manager.get_position(asset2_pair)
                if position2 is None or position2['quantity'] < asset2_quantity:
                    logger.warning(f"Cannot execute pair trade: Insufficient {asset2} position")
                    continue
                
                # Execute trades
                if asset1_quantity > 0 and asset1_allocation <= self.portfolio_manager.cash:
                    # Buy asset1
                    trade1 = {
                        'symbol': asset1_pair,
                        'side': 'buy',
                        'quantity': asset1_quantity,
                        'price': price1,
                        'timestamp': datetime.now(),
                        'pair_trade': True
                    }
                    self.portfolio_manager.update_portfolio(trade1)
                    
                    # Sell asset2
                    trade2 = {
                        'symbol': asset2_pair,
                        'side': 'sell',
                        'quantity': asset2_quantity,
                        'price': price2,
                        'timestamp': datetime.now(),
                        'pair_trade': True
                    }
                    self.portfolio_manager.update_portfolio(trade2)
                    
                    # Mark pairs as traded
                    traded_pairs.add(asset1_pair)
                    traded_pairs.add(asset2_pair)
                
            elif signal['signal'] == SignalType.SELL:
                # SELL asset1, BUY asset2
                logger.info(f"EXECUTING PAIR TRADE: SELL {asset1}, BUY {asset2}")
                
                # Check if we can execute both sides
                position1 = self.portfolio_manager.get_position(asset1_pair)
                if position1 is None or position1['quantity'] < asset1_quantity:
                    logger.warning(f"Cannot execute pair trade: Insufficient {asset1} position")
                    continue
                
                # Execute trades
                if asset2_quantity > 0 and asset2_allocation <= self.portfolio_manager.cash:
                    # Sell asset1
                    trade1 = {
                        'symbol': asset1_pair,
                        'side': 'sell',
                        'quantity': asset1_quantity,
                        'price': price1,
                        'timestamp': datetime.now(),
                        'pair_trade': True
                    }
                    self.portfolio_manager.update_portfolio(trade1)
                    
                    # Buy asset2
                    trade2 = {
                        'symbol': asset2_pair,
                        'side': 'buy',
                        'quantity': asset2_quantity,
                        'price': price2,
                        'timestamp': datetime.now(),
                        'pair_trade': True
                    }
                    self.portfolio_manager.update_portfolio(trade2)
                    
                    # Mark pairs as traded
                    traded_pairs.add(asset1_pair)
                    traded_pairs.add(asset2_pair)
    
    def _manage_existing_positions(self):
        """Manage existing positions (stop-loss, take-profit)"""
        positions = self.portfolio_manager.positions.copy()
        
        for pair, position in positions.items():
            if pair not in self.market_data:
                continue
                
            # Get current price
            current_price = None
            for timeframe in self.market_data[pair]:
                if not self.market_data[pair][timeframe].empty:
                    current_price = self.market_data[pair][timeframe]['close'].iloc[-1]
                    break
            
            if current_price is None:
                continue
                
            # Calculate profit/loss
            entry_price = position['average_price']
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Check stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(entry_price, 'long')
            take_profit = self.risk_manager.calculate_take_profit(entry_price, 'long')
            
            if current_price <= stop_loss:
                # Stop loss triggered
                logger.info(f"STOP LOSS triggered for {pair}: Entry ${entry_price:.2f}, Current ${current_price:.2f}, Loss {pnl_pct:.2%}")
                
                # Sell entire position
                quantity = position['quantity']
                trade = {
                    'symbol': pair,
                    'side': 'sell',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'reason': 'stop_loss'
                }
                self.portfolio_manager.update_portfolio(trade)
                
            elif current_price >= take_profit:
                # Take profit triggered
                logger.info(f"TAKE PROFIT triggered for {pair}: Entry ${entry_price:.2f}, Current ${current_price:.2f}, Profit {pnl_pct:.2%}")
                
                # Sell half position or full position
                quantity = position['quantity'] / 2  # Sell half
                trade = {
                    'symbol': pair,
                    'side': 'sell',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'reason': 'take_profit'
                }
                self.portfolio_manager.update_portfolio(trade)
    
    def _log_status(self, portfolio_value):
        """Log current bot status"""
        logger.info("\nCurrent Status:")
        logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        logger.info(f"Cash: ${self.portfolio_manager.cash:.2f}")
        
        # Log positions
        positions = self.portfolio_manager.positions
        if positions:
            logger.info("\nCurrent Positions:")
            for symbol, pos in positions.items():
                logger.info(f"{symbol}: {pos['quantity']} @ ${pos['average_price']:.2f}")
        else:
            logger.info("No current positions")
        
        # Log risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        logger.info(f"\nCurrent Drawdown: {risk_metrics['current_drawdown']:.2%}")
        logger.info(f"Maximum Drawdown: {risk_metrics['max_drawdown_reached']:.2%}")
    
    def run_backtest(self, days=90, plot_results=True):
        """
        Run backtesting mode.
        
        Args:
            days (int): Number of days to backtest
            plot_results (bool): Whether to plot results
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Starting backtesting mode for last {days} days")
        
        # Define backtest period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Run backtest
        results = self.backtester.run_backtest(start_date, end_date, plot_results)
        
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
            
        return results

def __init__(self, config_path="config/config.json", paper_trading=False, discord_notify=False):
    """
    Initialize trading bot.
    
    Args:
        config_path (str): Path to configuration file
        paper_trading (bool): Whether to run in paper trading mode
        discord_notify (bool): Whether to enable Discord notifications
    """
    logger.info(f"Initializing trading bot with config: {config_path}")
    
    # Load configuration
    self.config = load_configuration(config_path)
    
    # Operation mode flags
    self.paper_trading = paper_trading
    self.discord_notify = discord_notify
    
    # Initialize components
    self._initialize_components()
    
    logger.info("Trading bot initialized successfully")

def _initialize_components(self):
    """Initialize all components"""
    # Data collectors
    self.market_collector = MarketDataCollector(self.config)
    self.sentiment_collector = SentimentDataCollector(self.config)
    self.onchain_collector = OnchainDataCollector(self.config)
    self.macro_collector = MacroDataCollector(self.config)
    
    # Strategies
    self.ma_strategy = MovingAverageStrategy(self.config)
    self.pair_strategy = PairTradingStrategy(self.config)
    
    # Risk management
    self.risk_manager = RiskManager(self.config)
    
    # Manipulation detection
    self.manipulation_detector = ManipulationDetector(self.config)
    
    # Portfolio management - use regular or paper trading version
    if self.paper_trading:
        from execution.paper_trading import PaperTradingOrderManager
        self.portfolio_manager = PaperTradingOrderManager(self.config)
        logger.info("PAPER TRADING MODE ENABLED - No real money will be used")
    else:
        self.portfolio_manager = PortfolioManager(self.config)
    
    # Initialize Discord notifications if enabled
    if self.discord_notify:
        from utils.discord_notifier import DiscordNotifier
        self.discord = DiscordNotifier(self.config)
        logger.info("Discord notifications enabled")
    
    # Initialize backtester
    self.backtester = Backtester(self.config, self.ma_strategy, self.market_collector)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Load initial data
    self._load_initial_data()
    
    # Send startup notification if Discord is enabled
    if self.discord_notify and hasattr(self, 'discord'):
        self.discord.notify_startup(
            "live" if hasattr(self, 'run_live') else "backtest", 
            paper_trading=self.paper_trading
        )
    
    logger.info("All components initialized successfully")