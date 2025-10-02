"""
UI Components for the Algorithmic Trading Dashboard

This module contains reusable UI components for the trading dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class OptionChainComponent:
    """Component for displaying option chain data."""
    
    @staticmethod
    def render(option_data: pd.DataFrame):
        """Render the option chain table."""
        st.header("📊 Option Chain")
        
        if not option_data.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_type = st.selectbox("Option Type", ["All", "CE", "PE"])
            
            with col2:
                min_strike = int(option_data['Strike'].min())
                max_strike = int(option_data['Strike'].max())
                strike_range = st.slider("Strike Range", min_strike, max_strike, (min_strike, max_strike))
            
            with col3:
                min_oi = int(option_data['OI'].min())
                max_oi = int(option_data['OI'].max())
                oi_filter = st.slider("Min OI", min_oi, max_oi, min_oi)
            
            # Filter data
            filtered_data = option_data.copy()
            
            if selected_type != "All":
                filtered_data = filtered_data[filtered_data['Type'] == selected_type]
            
            filtered_data = filtered_data[
                (filtered_data['Strike'] >= strike_range[0]) & 
                (filtered_data['Strike'] <= strike_range[1]) &
                (filtered_data['OI'] >= oi_filter)
            ]
            
            # Display option chain
            st.markdown(f"**Showing {len(filtered_data)} options**")
            
            # Style the dataframe
            styled_df = filtered_data.style.format({
                'LTP': '₹{:.2f}',
                'Change': '₹{:.2f}',
                'IV': '{:.1%}',
                'Delta': '{:.3f}',
                'Gamma': '{:.4f}',
                'Theta': '₹{:.2f}',
                'Vega': '₹{:.2f}',
                'Bid': '₹{:.2f}',
                'Ask': '₹{:.2f}',
                'OI': '{:,}',
                'Volume': '{:,}'
            }).applymap(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', 
                       subset=['Change'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Option chain summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_call_oi = filtered_data[filtered_data['Type'] == 'CE']['OI'].sum()
                st.metric("Total Call OI", f"{total_call_oi:,}")
            
            with col2:
                total_put_oi = filtered_data[filtered_data['Type'] == 'PE']['OI'].sum()
                st.metric("Total Put OI", f"{total_put_oi:,}")
            
            with col3:
                pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                st.metric("PCR", f"{pcr:.2f}")
            
            with col4:
                avg_iv = filtered_data['IV'].mean()
                st.metric("Avg IV", f"{avg_iv:.1%}")


class ChartComponent:
    """Component for displaying price charts with indicators."""
    
    @staticmethod
    def render(price_data: pd.DataFrame):
        """Render price charts with indicators."""
        st.header("📈 Price Charts")
        
        if not price_data.empty:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price & Supertrend', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Price candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data['timestamp'],
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name="Price",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )
            
            # Supertrend indicator
            if 'supertrend' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data['timestamp'],
                        y=price_data['supertrend'],
                        mode='lines',
                        name='Supertrend',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=price_data['timestamp'],
                    y=price_data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title="NIFTY Price Chart with Supertrend Indicator",
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = price_data['close'].iloc[-1]
                prev_price = price_data['close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                st.metric("Current Price", f"₹{current_price:,.2f}", 
                         f"₹{change:+.2f} ({change_pct:+.2f}%)")
            
            with col2:
                # Mock VIX
                vix = np.random.uniform(15, 25)
                st.metric("VIX", f"{vix:.2f}")
            
            with col3:
                # Mock PCR
                pcr = np.random.uniform(0.8, 1.2)
                st.metric("PCR", f"{pcr:.2f}")
            
            with col4:
                # Market sentiment
                sentiment = "Bullish" if change > 0 else "Bearish"
                sentiment_color = "🟢" if change > 0 else "🔴"
                st.metric("Sentiment", f"{sentiment_color} {sentiment}")


class TradeLogComponent:
    """Component for displaying trade log and P&L."""
    
    @staticmethod
    def render(trades_data: pd.DataFrame):
        """Render trade log and P&L dashboard."""
        st.header("📋 Trade Log & P&L")
        
        if not trades_data.empty:
            # Trade log table
            st.subheader("Recent Trades")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox("Status", ["All", "OPEN", "CLOSED"])
            
            with col2:
                side_filter = st.selectbox("Side", ["All", "BUY", "SELL"])
            
            with col3:
                pnl_threshold = st.number_input("Min P&L", value=0.0)
            
            # Filter trades
            filtered_trades = trades_data.copy()
            
            if status_filter != "All":
                filtered_trades = filtered_trades[filtered_trades['Status'] == status_filter]
            
            if side_filter != "All":
                filtered_trades = filtered_trades[filtered_trades['Side'] == side_filter]
            
            filtered_trades = filtered_trades[filtered_trades['P&L'] >= pnl_threshold]
            
            # Display trades
            if not filtered_trades.empty:
                # Style the dataframe
                styled_trades = filtered_trades.style.format({
                    'Entry Price': '₹{:.2f}',
                    'Exit Price': '₹{:.2f}',
                    'P&L': '₹{:.2f}'
                }).applymap(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', 
                           subset=['P&L'])
                
                st.dataframe(styled_trades, use_container_width=True, height=300)
            
            # P&L Summary
            st.subheader("P&L Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = trades_data['P&L'].sum()
                st.metric("Total P&L", f"₹{total_pnl:,.2f}")
            
            with col2:
                winning_trades = len(trades_data[trades_data['P&L'] > 0])
                total_trades = len(trades_data)
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                avg_win = trades_data[trades_data['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
                st.metric("Avg Win", f"₹{avg_win:,.2f}")
            
            with col4:
                avg_loss = trades_data[trades_data['P&L'] < 0]['P&L'].mean() if len(trades_data[trades_data['P&L'] < 0]) > 0 else 0
                st.metric("Avg Loss", f"₹{avg_loss:,.2f}")
            
            # P&L Chart
            trades_data['Cumulative P&L'] = trades_data['P&L'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_data['Timestamp'],
                y=trades_data['Cumulative P&L'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Time",
                yaxis_title="P&L (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)


class RiskDashboardComponent:
    """Component for displaying risk management dashboard."""
    
    @staticmethod
    def render():
        """Render risk management dashboard."""
        st.header("🛡️ Risk Management")
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_value = 1250000.0
            st.metric("Portfolio Value", f"₹{portfolio_value:,.2f}")
        
        with col2:
            daily_pnl = 25000.0
            daily_pnl_pct = (daily_pnl / portfolio_value) * 100
            st.metric("Daily P&L", f"₹{daily_pnl:,.2f}", f"{daily_pnl_pct:+.2f}%")
        
        with col3:
            max_drawdown = 5.2
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        
        with col4:
            risk_score = 45.0
            risk_level = "LOW" if risk_score < 50 else "MEDIUM" if risk_score < 75 else "HIGH"
            risk_color = "🟢" if risk_score < 50 else "🟡" if risk_score < 75 else "🔴"
            st.metric("Risk Score", f"{risk_color} {risk_level}")
        
        # Risk limits
        st.subheader("Risk Limits")
        
        limits_data = {
            'Limit': ['Max Daily Loss', 'Max Position Size', 'Max Open Positions', 'Max Margin Usage'],
            'Current': ['2.0%', '8.5%', '3', '25.0%'],
            'Limit': ['5.0%', '10.0%', '10', '80.0%'],
            'Status': ['✅ Safe', '✅ Safe', '✅ Safe', '✅ Safe']
        }
        
        limits_df = pd.DataFrame(limits_data)
        st.dataframe(limits_df, use_container_width=True)
        
        # Risk alerts
        st.subheader("Recent Alerts")
        
        mock_alerts = [
            {
                'Time': datetime.now() - timedelta(minutes=30),
                'Type': 'INFO',
                'Message': 'Position size within limits',
                'Priority': 'LOW'
            },
            {
                'Time': datetime.now() - timedelta(hours=2),
                'Type': 'WARNING',
                'Message': 'Margin utilization approaching limit',
                'Priority': 'MEDIUM'
            },
            {
                'Time': datetime.now() - timedelta(hours=6),
                'Type': 'SUCCESS',
                'Message': 'Trade executed successfully',
                'Priority': 'LOW'
            }
        ]
        
        alerts_df = pd.DataFrame(mock_alerts)
        
        # Color code alerts
        def color_priority(val):
            if val == 'HIGH':
                return 'background-color: #ffebee'
            elif val == 'MEDIUM':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e8'
        
        styled_alerts = alerts_df.style.applymap(color_priority, subset=['Priority'])
        st.dataframe(styled_alerts, use_container_width=True)


class LiveMonitoringComponent:
    """Component for live monitoring dashboard."""
    
    @staticmethod
    def render():
        """Render live monitoring dashboard."""
        st.header("🔴 Live Monitoring")
        
        # Trading status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Live Trading", type="primary"):
                st.session_state.live_trading = True
                st.success("Live trading started!")
        
        with col2:
            if st.button("⏸️ Pause Trading"):
                st.session_state.live_trading = False
                st.warning("Trading paused!")
        
        with col3:
            if st.button("🛑 Emergency Stop", type="secondary"):
                st.session_state.live_trading = False
                st.error("Emergency stop activated!")
        
        # Live metrics
        st.subheader("Live Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Live Price", "₹18,250", "↗️ +125.50")
        
        with col2:
            st.metric("Open Positions", "3", "2")
        
        with col3:
            st.metric("Pending Orders", "1", "0")
        
        with col4:
            st.metric("System Status", "🟢 Online", "Stable")
        
        # Live positions
        st.subheader("Live Positions")
        
        positions_data = {
            'Symbol': ['NIFTY CE 18200', 'NIFTY PE 17800', 'BANKNIFTY CE 42000'],
            'Side': ['BUY', 'SELL', 'BUY'],
            'Quantity': [50, 25, 30],
            'Entry Price': [150.50, 120.25, 200.75],
            'Current Price': [165.25, 115.50, 195.25],
            'P&L': [737.50, -118.75, -165.00],
            'P&L %': [4.90, -0.99, -0.82]
        }
        
        positions_df = pd.DataFrame(positions_data)
        
        # Style positions
        styled_positions = positions_df.style.format({
            'Entry Price': '₹{:.2f}',
            'Current Price': '₹{:.2f}',
            'P&L': '₹{:.2f}',
            'P&L %': '{:.2f}%'
        }).applymap(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', 
                   subset=['P&L', 'P&L %'])
        
        st.dataframe(styled_positions, use_container_width=True)
        
        # Auto-refresh
        if st.session_state.get('live_trading', False):
            st.info("🔄 Live data updates every 5 seconds")
            # In a real implementation, this would trigger actual data updates


class DataGenerator:
    """Utility class for generating mock data."""
    
    @staticmethod
    def generate_mock_option_data() -> pd.DataFrame:
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
    
    @staticmethod
    def generate_mock_price_data() -> pd.DataFrame:
        """Generate mock price data with Supertrend indicator."""
        try:
            # Generate 100 data points
            dates = pd.date_range(start=datetime.now() - timedelta(hours=100), 
                                end=datetime.now(), freq='1H')
            
            # Generate price data with some trend
            base_price = 18000
            prices = []
            current_price = base_price
            
            for i in range(len(dates)):
                # Add some trend and volatility
                change = np.random.normal(0, 50)
                if i > 50:  # Add upward trend after 50 hours
                    change += 10
                current_price += change
                prices.append(current_price)
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                high = price + np.random.uniform(0, 30)
                low = price - np.random.uniform(0, 30)
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(100000, 500000)
                
                data.append({
                    'timestamp': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            
            # Calculate Supertrend indicator
            try:
                from strategies.supertrend import SupertrendIndicator
                indicator = SupertrendIndicator(period=10, multiplier=2.0)
                df['supertrend'] = indicator.calculate(df['close'].values)
            except ImportError:
                # Fallback if SupertrendIndicator is not available
                df['supertrend'] = df['close'] * 0.98  # Simple fallback
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock price data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def generate_mock_trade_data() -> pd.DataFrame:
        """Generate mock trade data."""
        try:
            mock_trades = []
            base_time = datetime.now() - timedelta(hours=24)
            
            for i in range(20):
                trade_time = base_time + timedelta(hours=i*1.2)
                symbol = f"NIFTY{np.random.choice(['CE', 'PE'])}{np.random.randint(17500, 18500, 1)[0]}"
                side = np.random.choice(['BUY', 'SELL'])
                quantity = np.random.randint(25, 100)
                entry_price = np.random.uniform(50, 200)
                exit_price = entry_price + np.random.normal(0, 20)
                pnl = (exit_price - entry_price) * quantity * (1 if side == 'SELL' else -1)
                
                mock_trades.append({
                    'Timestamp': trade_time,
                    'Symbol': symbol,
                    'Side': side,
                    'Quantity': quantity,
                    'Entry Price': round(entry_price, 2),
                    'Exit Price': round(exit_price, 2),
                    'P&L': round(pnl, 2),
                    'Status': 'CLOSED' if i < 15 else 'OPEN'
                })
            
            return pd.DataFrame(mock_trades)
            
        except Exception as e:
            logger.error(f"Error generating mock trade data: {e}")
            return pd.DataFrame()
