# utils/discord_notifier.py
"""
Discord Notification System
-------------------------
Sends notifications about bot status and trades to Discord.
"""
import logging
import requests
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DiscordNotifier:
    """Handles sending notifications to Discord using webhooks"""
    
    def __init__(self, config):
        """
        Initialize Discord notification system.
        
        Args:
            config (dict): Bot configuration
        """
        self.config = config
        self.discord_enabled = False
        
        # Set up Discord notifications if configured
        if "notifications" in config and "discord" in config["notifications"]:
            discord_config = config["notifications"]["discord"]
            self.webhook_url = discord_config.get("webhook_url")
            
            if self.webhook_url:
                self.discord_enabled = True
                self.bot_name = discord_config.get("bot_name", "Crypto Trading Bot")
                self.avatar_url = discord_config.get("avatar_url", "")
                logger.info("Discord notifications enabled")
    
    def send_message(self, message, embed=None, level="info"):
        """
        Send message to Discord webhook.
        
        Args:
            message (str): Message to send
            embed (dict, optional): Discord embed object
            level (str): Notification level (info, warning, error)
        
        Returns:
            bool: Success status
        """
        if not self.discord_enabled:
            logger.info(f"Discord notification skipped (not enabled): {message}")
            return False
        
        try:
            # Choose color based on level
            colors = {
                "info": 3447003,      # Blue
                "success": 5763719,   # Green
                "warning": 16776960,  # Yellow
                "error": 15158332     # Red
            }
            
            # Format timestamp
            timestamp = datetime.now().isoformat()
            
            # Create payload
            payload = {
                "username": self.bot_name,
                "content": message
            }
            
            # Add avatar if provided
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            # Add embed if provided or create one based on level
            if embed:
                payload["embeds"] = [embed]
            elif level != "info":
                # Create embed with appropriate color
                payload["embeds"] = [{
                    "title": level.upper(),
                    "description": message,
                    "color": colors.get(level, colors["info"]),
                    "timestamp": timestamp
                }]
                # Clear content if we're using an embed
                payload["content"] = ""
            
            # Send to Discord
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 204]:
                logger.debug(f"Discord notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send Discord notification: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {str(e)}")
            return False
    
    def notify_trade(self, trade):
        """
        Send trade notification to Discord.
        
        Args:
            trade (dict): Trade details
        """
        if not self.discord_enabled:
            return
        
        symbol = trade['symbol']
        side = trade['side'].upper()
        quantity = trade['quantity']
        price = trade['price']
        value = quantity * price
        timestamp = trade.get('timestamp', datetime.now()).isoformat()
        
        # Choose color based on trade side
        color = 5763719 if side == "BUY" else 15158332  # Green for buy, red for sell
        
        embed = {
            "title": f"{side} {symbol}",
            "description": f"Trade executed successfully",
            "color": color,
            "fields": [
                {"name": "Quantity", "value": f"{quantity:.8f}", "inline": True},
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Total Value", "value": f"${value:.2f}", "inline": True}
            ],
            "footer": {"text": "Crypto Trading Bot"},
            "timestamp": timestamp
        }
        
        self.send_message("", embed=embed, level="success")
    
    def notify_portfolio_update(self, portfolio_value, cash, positions=None, previous_value=None):
        """
        Send portfolio update notification to Discord.
        
        Args:
            portfolio_value (float): Current portfolio value
            cash (float): Available cash
            positions (dict, optional): Current positions
            previous_value (float, optional): Previous portfolio value
        """
        if not self.discord_enabled:
            return
        
        # Calculate change if previous value is provided
        if previous_value:
            change = portfolio_value - previous_value
            change_pct = (change / previous_value) * 100
            change_text = f"${change:.2f} ({change_pct:.2f}%)"
            
            # Determine color based on change
            if change > 0:
                color = 5763719  # Green
            elif change < 0:
                color = 15158332  # Red
            else:
                color = 3447003  # Blue
        else:
            change_text = "N/A"
            color = 3447003  # Blue
        
        # Create embed fields
        fields = [
            {"name": "Total Value", "value": f"${portfolio_value:.2f}", "inline": True},
            {"name": "Cash", "value": f"${cash:.2f}", "inline": True},
            {"name": "Change", "value": change_text, "inline": True}
        ]
        
        # Add positions if provided
        if positions:
            positions_text = ""
            for symbol, pos in positions.items():
                positions_text += f"{symbol}: {pos['quantity']:.8f} @ ${pos['average_price']:.2f}\n"
            
            if positions_text:
                fields.append({"name": "Current Positions", "value": positions_text, "inline": False})
        
        embed = {
            "title": "Portfolio Update",
            "color": color,
            "fields": fields,
            "footer": {"text": "Crypto Trading Bot"},
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_message("", embed=embed, level="info")
    
    def notify_error(self, error_message):
        """
        Send error notification to Discord.
        
        Args:
            error_message (str): Error message
        """
        self.send_message(error_message, level="error")
    
    def notify_warning(self, warning_message):
        """
        Send warning notification to Discord.
        
        Args:
            warning_message (str): Warning message
        """
        self.send_message(warning_message, level="warning")
    
    def notify_startup(self, mode, paper_trading=False):
        """
        Send bot startup notification to Discord.
        
        Args:
            mode (str): Bot mode (live or backtest)
            paper_trading (bool): Whether running in paper trading mode
        """
        if mode == "live":
            if paper_trading:
                title = "Bot Started - PAPER TRADING Mode"
                description = "Bot is running in paper trading mode (no real money)"
                color = 16776960  # Yellow
            else:
                title = "Bot Started - LIVE TRADING Mode"
                description = "Bot is running in live trading mode with real money"
                color = 15158332  # Red
        else:
            title = "Bot Started - BACKTEST Mode"
            description = "Bot is running in backtest mode"
            color = 3447003  # Blue
        
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "footer": {"text": "Crypto Trading Bot"},
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_message("", embed=embed, level="info")