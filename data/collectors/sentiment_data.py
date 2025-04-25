"""
Sentiment Data Collector
----------------------
Collects sentiment data from social media and news sources.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import re
import json
import os

# Untuk lingkungan produksi, gunakan API nyata
try:
    import tweepy
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk_available = True
except ImportError:
    nltk_available = False

logger = logging.getLogger(__name__)

class SentimentDataCollector:
    """Collects sentiment data from various sources"""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config (dict): Bot configuration
        """
        self.config = config
        self.api_keys = config.get("api_keys", {})
        self.initialize_sentiment_analyzer()
        
    def initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis tools"""
        if nltk_available:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                
                # Initialize Twitter API if keys are provided
                if "twitter" in self.api_keys:
                    auth = tweepy.OAuthHandler(
                        self.api_keys["twitter"]["consumer_key"],
                        self.api_keys["twitter"]["consumer_secret"]
                    )
                    auth.set_access_token(
                        self.api_keys["twitter"]["access_token"],
                        self.api_keys["twitter"]["access_token_secret"]
                    )
                    self.twitter_api = tweepy.API(auth)
                    logger.info("Twitter API initialized successfully")
                else:
                    self.twitter_api = None
                    logger.warning("Twitter API keys not found in config")
                
                # Initialize Reddit API if keys are provided
                if "reddit" in self.api_keys:
                    try:
                        import praw
                        self.reddit_api = praw.Reddit(
                            client_id=self.api_keys["reddit"]["client_id"],
                            client_secret=self.api_keys["reddit"]["client_secret"],
                            user_agent=self.api_keys["reddit"]["user_agent"]
                        )
                        logger.info("Reddit API initialized successfully")
                    except ImportError:
                        self.reddit_api = None
                        logger.warning("PRAW package not installed for Reddit API")
                else:
                    self.reddit_api = None
                    logger.warning("Reddit API keys not found in config")
                    
            except Exception as e:
                logger.error(f"Error initializing sentiment analyzer: {str(e)}")
                self.sentiment_analyzer = None
                self.twitter_api = None
                self.reddit_api = None
        else:
            logger.warning("NLTK not available. Using synthetic sentiment data.")
            self.sentiment_analyzer = None
            self.twitter_api = None
            self.reddit_api = None
    
    def collect_sentiment_data(self, asset, days=7):
        """
        Collect sentiment data for a specific asset.
        
        Args:
            asset (str): Asset symbol (BTC, ETH, etc.)
            days (int): Number of days to look back
            
        Returns:
            DataFrame: Sentiment data
        """
        logger.info(f"Collecting sentiment data for {asset}")
        
        # Use real APIs if available
        if self.sentiment_analyzer and (self.twitter_api or self.reddit_api):
            sentiment_data = self._collect_real_sentiment(asset, days)
        else:
            sentiment_data = self._generate_synthetic_sentiment(asset, days)
            
        return sentiment_data
    
    def _collect_real_sentiment(self, asset, days):
        """
        Collect real sentiment data from Twitter and Reddit.
        
        Args:
            asset (str): Asset symbol
            days (int): Number of days to look back
            
        Returns:
            DataFrame: Sentiment data
        """
        sentiment_records = []
        since_date = datetime.now() - timedelta(days=days)
        
        # Collect Twitter data
        if self.twitter_api:
            try:
                search_query = f"#{asset} OR ${asset} crypto"
                tweets = self.twitter_api.search_tweets(
                    q=search_query,
                    count=100,
                    result_type="mixed",
                    lang="en"
                )
                
                for tweet in tweets:
                    if tweet.created_at >= since_date:
                        # Clean text
                        text = re.sub(r'http\S+', '', tweet.text)
                        text = re.sub(r'@\w+', '', text)
                        
                        # Calculate sentiment
                        sentiment = self.sentiment_analyzer.polarity_scores(text)
                        
                        sentiment_records.append({
                            'date': tweet.created_at,
                            'source': 'twitter',
                            'text': text[:200],  # Truncate long texts
                            'sentiment_score': sentiment['compound'],
                            'likes': tweet.favorite_count,
                            'retweets': tweet.retweet_count,
                            'followers': tweet.user.followers_count
                        })
                
                logger.info(f"Collected {len(tweets)} tweets for {asset}")
                
            except Exception as e:
                logger.error(f"Error collecting Twitter data for {asset}: {str(e)}")
        
        # Collect Reddit data
        if self.reddit_api:
            try:
                subreddits = ["cryptocurrency", "Bitcoin", "ethtrader"]
                
                for subreddit_name in subreddits:
                    subreddit = self.reddit_api.subreddit(subreddit_name)
                    
                    for post in subreddit.search(asset, limit=50):
                        if datetime.fromtimestamp(post.created_utc) >= since_date:
                            # Clean text
                            text = re.sub(r'http\S+', '', post.title + " " + post.selftext)
                            
                            # Calculate sentiment
                            sentiment = self.sentiment_analyzer.polarity_scores(text)
                            
                            sentiment_records.append({
                                'date': datetime.fromtimestamp(post.created_utc),
                                'source': 'reddit',
                                'text': text[:200],  # Truncate long texts
                                'sentiment_score': sentiment['compound'],
                                'likes': post.score,
                                'comments': post.num_comments,
                                'followers': 0  # No direct equivalent on Reddit
                            })
                
                logger.info(f"Collected Reddit data for {asset} from {len(subreddits)} subreddits")
                
            except Exception as e:
                logger.error(f"Error collecting Reddit data for {asset}: {str(e)}")
        
        # Create DataFrame
        if sentiment_records:
            df = pd.DataFrame(sentiment_records)
            
            # Calculate weighted sentiment
            if 'followers' in df.columns and df['followers'].sum() > 0:
                df['weight'] = df['followers'] / df['followers'].max()
            else:
                df['weight'] = 1
                
            df['weighted_sentiment'] = df['sentiment_score'] * df['weight']
            
            # Aggregate by date
            df['date'] = pd.to_datetime(df['date']).dt.date
            daily_sentiment = df.groupby('date').agg({
                'sentiment_score': 'mean',
                'weighted_sentiment': 'mean',
                'text': 'count'
            }).reset_index()
            
            daily_sentiment.rename(columns={'text': 'mention_count'}, inplace=True)
            
            return daily_sentiment
        else:
            return pd.DataFrame()
    
    def _generate_synthetic_sentiment(self, asset, days):
        """
        Generate synthetic sentiment data for testing.
        
        Args:
            asset (str): Asset symbol
            days (int): Number of days to look back
            
        Returns:
            DataFrame: Synthetic sentiment data
        """
        logger.warning(f"Generating synthetic sentiment data for {asset}")
        
        # Create date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Set random seed for reproducibility
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        # Generate synthetic sentiment with some randomness but a trend
        if asset in ["BTC", "ETH"]:
            # More popular assets have higher mention counts
            mention_counts = np.random.randint(50, 200, size=len(date_range))
            base_sentiment = 0.1  # Slightly positive
        else:
            mention_counts = np.random.randint(10, 100, size=len(date_range))
            base_sentiment = 0  # Neutral
        
        # Create some randomness in sentiment
        sentiment_scores = np.random.normal(base_sentiment, 0.3, size=len(date_range))
        sentiment_scores = np.clip(sentiment_scores, -1, 1)  # Restrict to [-1, 1]
        
        # Add a trend: sentiment starts neutral and goes up or down
        trend = np.linspace(-0.2, 0.2, len(date_range))
        sentiment_scores += trend
        
        # Create dataframe
        df = pd.DataFrame({
            'date': date_range.date,
            'sentiment_score': sentiment_scores,
            'weighted_sentiment': sentiment_scores * 1.1,  # Slightly higher weighted
            'mention_count': mention_counts
        })
        
        return df