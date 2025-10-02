"""
Main Streamlit Dashboard for Algorithmic Trading System

This is the primary UI for the algorithmic trading system, providing:
- Real-time option chain data with OI, IV, Greeks, LTP
- Price charting with Supertrend indicators
- Trade log and P&L tracking
- Risk management dashboard
- Live monitoring and alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import sys
import os
from typing import Dict, List, Any, Optional
import json

# Add project root to path
sys.path.append('.')

# Import our trading system components
from data.candle_builder import CandleBuilder, TickData, Candle
from strategies.supertrend import SupertrendStrategy, SupertrendIndicator
from strategies.strategy_runner import StrategyRunner
from backtest.backtest_engine import BacktestEngine
from live.paper_trader import PaperTrader, PaperOrder, PaperPosition, OrderStatus
from live.live_strategy_runner import LiveStrategyRunner
from risk.risk_engine import RiskEngine, RiskConfig
from risk.risk_monitor import RiskMonitor, MonitorConfig
from risk.risk_alerts import RiskAlertManager, AlertConfig
from risk.risk_dashboard import RiskDashboard, DashboardConfig
from broker.base_broker import BrokerOrder, BrokerPosition, OrderSide, OrderType, ProductType
from utils.logging_utils import get_logger
from utils.config import load_config

# Setup logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .option-chain-table {
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Main trading dashboard class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.config = load_config()
        self.initialize_session_state()
        self.setup_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.current_page = "Overview"
            st.session_state.live_trading = False
            st.session_state.risk_engine = None
            st.session_state.paper_trader = None
            st.session_state.alert_manager = None
            st.session_state.risk_dashboard = None
            st.session_state.option_data = None
            st.session_state.trade_history = []
            st.session_state.performance_data = {}
            st.session_state.alerts = []
    
    def setup_components(self):
        """Setup trading system components."""
        try:
            # Initialize risk management
            risk_config = RiskConfig()
            st.session_state.risk_engine = RiskEngine(risk_config)
            
            # Initialize paper trader
            st.session_state.paper_trader = PaperTrader(
                initial_capital=1000000.0,
                commission_rate=0.0003
            )
            
            # Initialize alert manager
            alert_config = AlertConfig()
            st.session_state.alert_manager = RiskAlertManager(alert_config)
            
            # Initialize risk dashboard
            dashboard_config = DashboardConfig()
            st.session_state.risk_dashboard = RiskDashboard(dashboard_config)
            
            logger.info("Dashboard components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard components: {e}")
            st.error(f"Error initializing dashboard: {e}")
    
    def generate_mock_option_data(self) -> pd.DataFrame:
        """Generate mock option chain data for demonstration."""
        try:
            # Base price around 18000 (NIFTY)
            base_price = 18000
            strikes = range(base_price - 1000, base_price + 1000, 50)
            
            option_data = []
            for strike in strikes:
                # Call options
                call_price = max(0.05, (strike - base_price) * 0.01 + np.random.normal(50, 20))
                call_iv = np.random.uniform(0.15, 0.35)
                call_delta = min(0.99, max(0.01, 0.5 + (base_price - strike) / base_price * 0.3))
                call_gamma = np.random.uniform(0.0001, 0.001)
                call_theta = -np.random.uniform(0.5, 2.0)
                call_vega = np.random.uniform(5, 20)
                
                option_data.append({
                    'Strike': strike,
                    'Type': 'CE',
                    'LTP': round(call_price, 2),
                    'OI': np.random.randint(1000, 50000),
                    'Change': round(np.random.normal(0, 5), 2),
                    'IV': round(call_iv, 3),
                    'Delta': round(call_delta, 3),
                    'Gamma': round(call_gamma, 4),
                    'Theta': round(call_theta, 2),
                    'Vega': round(call_vega, 2),
                    'Bid': round(call_price - 0.5, 2),
                    'Ask': round(call_price + 0.5, 2),
                    'Volume': np.random.randint(100, 5000)
                })
                
                # Put options
                put_price = max(0.05, (base_price - strike) * 0.01 + np.random.normal(50, 20))
                put_iv = np.random.uniform(0.15, 0.35)
                put_delta = min(-0.01, max(-0.99, -0.5 + (strike - base_price) / base_price * 0.3))
                put_gamma = np.random.uniform(0.0001, 0.001)
                put_theta = -np.random.uniform(0.5, 2.0)
                put_vega = np.random.uniform(5, 20)
                
                option_data.append({
                    'Strike': strike,
                    'Type': 'PE',
                    'LTP': round(put_price, 2),
                    'OI': np.random.randint(1000, 50000),
                    'Change': round(np.random.normal(0, 5), 2),
                    'IV': round(put_iv, 3),
                    'Delta': round(put_delta, 3),
                    'Gamma': round(put_gamma, 4),
                    'Theta': round(put_theta, 2),
                    'Vega': round(put_vega, 2),
                    'Bid': round(put_price - 0.5, 2),
                    'Ask': round(put_price + 0.5, 2),
                    'Volume': np.random.randint(100, 5000)
                })
            
            return pd.DataFrame(option_data)
            
        except Exception as e:
            logger.error(f"Error generating mock option data: {e}")
            return pd.DataFrame()
    
    def render_header(self):
        """Render the dashboard header."""
        st.markdown('<div class="main-header">📈 Algorithmic Trading System</div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", "₹18,250", "125.50 (0.69%)")
        
        with col2:
            st.metric("Portfolio Value", "₹1,250,000", "50,000 (4.17%)")
        
        with col3:
            st.metric("Daily P&L", "₹25,000", "2,500 (11.11%)")
        
        with col4:
            risk_level = "LOW"
            risk_color = "🟢"
            st.metric("Risk Level", f"{risk_color} {risk_level}", "Stable")
    
    def render_option_chain(self):
        """Render the option chain table."""
        from ui_components import OptionChainComponent, DataGenerator
        
        option_data = DataGenerator.generate_mock_option_data()
        OptionChainComponent.render(option_data)
    
    def render_charts(self):
        """Render price charts with indicators."""
        from ui_components import ChartComponent, DataGenerator
        
        price_data = DataGenerator.generate_mock_price_data()
        ChartComponent.render(price_data)
    
    def render_trade_log(self):
        """Render trade log and P&L dashboard."""
        from ui_components import TradeLogComponent, DataGenerator
        
        trades_data = DataGenerator.generate_mock_trade_data()
        TradeLogComponent.render(trades_data)
    
    def render_risk_dashboard(self):
        """Render risk management dashboard."""
        from ui_components import RiskDashboardComponent
        
        RiskDashboardComponent.render()
    
    def render_live_monitoring(self):
        """Render live monitoring dashboard."""
        from ui_components import LiveMonitoringComponent
        
        LiveMonitoringComponent.render()
    
    def render_sidebar(self):
        """Render the sidebar with settings."""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Trading parameters
            st.subheader("Trading Parameters")
            
            initial_capital = st.number_input(
                "Initial Capital", 
                min_value=100000, 
                max_value=10000000, 
                value=1000000,
                step=100000
            )
            
            commission_rate = st.slider(
                "Commission Rate", 
                min_value=0.0, 
                max_value=0.01, 
                value=0.0003,
                format="%.4f"
            )
            
            # Risk parameters
            st.subheader("Risk Parameters")
            
            max_daily_loss = st.slider(
                "Max Daily Loss %", 
                min_value=1.0, 
                max_value=10.0, 
                value=5.0
            )
            
            max_position_size = st.slider(
                "Max Position Size %", 
                min_value=5.0, 
                max_value=25.0, 
                value=10.0
            )
            
            # Strategy parameters
            st.subheader("Strategy Parameters")
            
            supertrend_period = st.slider(
                "Supertrend Period", 
                min_value=5, 
                max_value=20, 
                value=10
            )
            
            supertrend_multiplier = st.slider(
                "Supertrend Multiplier", 
                min_value=1.0, 
                max_value=3.0, 
                value=2.0
            )
            
            # System info
            st.subheader("System Info")
            st.info(f"Dashboard initialized: {st.session_state.dashboard_initialized}")
            st.info(f"Live trading: {'🟢 Active' if st.session_state.live_trading else '🔴 Inactive'}")
            
            # Refresh button
            if st.button("🔄 Refresh Data"):
                st.rerun()
    
    def run(self):
        """Run the main dashboard."""
        try:
            # Render header
            self.render_header()
            
            # Main navigation
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Option Chain", 
                "📈 Charts", 
                "📋 Trade Log", 
                "🛡️ Risk Management", 
                "🔴 Live Monitoring"
            ])
            
            with tab1:
                self.render_option_chain()
            
            with tab2:
                self.render_charts()
            
            with tab3:
                self.render_trade_log()
            
            with tab4:
                self.render_risk_dashboard()
            
            with tab5:
                self.render_live_monitoring()
            
            # Render sidebar
            self.render_sidebar()
            
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            st.error(f"Dashboard error: {e}")


def main():
    """Main function to run the dashboard."""
    try:
        dashboard = TradingDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Error in main dashboard: {e}")
        st.error(f"Critical error: {e}")


if __name__ == "__main__":
    main()