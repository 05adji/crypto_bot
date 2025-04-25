"""
Trading Bot Dashboard
-------------------
Simple monitoring dashboard for the bot.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Trading Bot Dashboard", layout="wide")

st.title("Crypto Trading Bot Monitor")

# Sidebar
st.sidebar.header("Dashboard Options")
data_mode = st.sidebar.selectbox("Data Source", ["Paper Trading", "Backtest Results"])

if data_mode == "Paper Trading":
    # Load paper trading performance data
    paper_trading_file = "data/paper_trading/performance_log.csv"
    
    if os.path.exists(paper_trading_file):
        paper_data = pd.read_csv(paper_trading_file)
        paper_data['timestamp'] = pd.to_datetime(paper_data['timestamp'])
        
        # Main metrics
        st.header("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        last_record = paper_data.iloc[-1]
        start_record = paper_data.iloc[0]
        
        total_return = (last_record['portfolio_value'] - start_record['portfolio_value']) / start_record['portfolio_value'] * 100
        
        col1.metric("Current Value", f"${last_record['portfolio_value']:.2f}", 
                    f"{total_return:.2f}%")
        
        col2.metric("Cash", f"${last_record['cash']:.2f}")
        col3.metric("Positions Value", f"${last_record['positions_value']:.2f}")
        col4.metric("Return", f"{last_record['pct_return']:.2f}%")
        
        # Performance chart
        st.subheader("Portfolio Performance")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(paper_data['timestamp'], paper_data['portfolio_value'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True)
        st.pyplot(fig)
        
        # Show raw data
        st.subheader("Raw Performance Data")
        st.dataframe(paper_data)
    else:
        st.warning("No paper trading data available. Run the bot in paper trading mode first.")

elif data_mode == "Backtest Results":
    # Load backtest summary if available
    backtest_summary_file = "test_results/backtest_summary.csv"
    
    if os.path.exists(backtest_summary_file):
        backtest_data = pd.read_csv(backtest_summary_file)
        
        # Display comparative results
        st.subheader("Backtest Results Comparison")
        
        # Metrics by configuration
        st.write("Performance by Configuration")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        configs = backtest_data['config'].unique()
        for config in configs:
            config_data = backtest_data[backtest_data['config'] == config]
            ax.plot(config_data['period'], config_data['total_return'], 
                    marker='o', label=config)
        
        ax.set_xlabel("Backtest Period (days)")
        ax.set_ylabel("Total Return (%)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Show best configurations
        st.subheader("Top Configurations")
        best_sharpe = backtest_data.sort_values('sharpe_ratio', ascending=False).head(5)
        st.write("Best by Sharpe Ratio")
        st.dataframe(best_sharpe)
        
        best_return = backtest_data.sort_values('total_return', ascending=False).head(5)
        st.write("Best by Total Return")
        st.dataframe(best_return)
        
        # Show all results
        st.subheader("All Backtest Results")
        st.dataframe(backtest_data)
    else:
        st.warning("No backtest summary found. Run comprehensive backtests first.")

# Refresh button
if st.button("Refresh Data"):
    st.experimental_rerun()