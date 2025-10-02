"""
Portfolio Component for Streamlit Dashboard.

This module provides comprehensive portfolio management and analytics
including position tracking, P&L analysis, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys

# Add project root to path
sys.path.append('.')


class PortfolioComponent:
    """Portfolio management and analytics component."""
    
    def __init__(self):
        self.initial_capital = 1000000  # ₹10 lakh
        self.positions = []
        self.trades = []
        self.performance_data = []
    
    def render_portfolio_dashboard(self) -> None:
        """Render the complete portfolio dashboard."""
        
        # Generate portfolio data
        portfolio_data = self._generate_portfolio_data()
        
        # Portfolio Summary
        self._render_portfolio_summary(portfolio_data)
        
        # Positions Overview
        self._render_positions_overview(portfolio_data)
        
        # Performance Charts
        self._render_performance_charts(portfolio_data)
        
        # Risk Analytics
        self._render_risk_analytics(portfolio_data)
        
        # Trade Analysis
        self._render_trade_analysis(portfolio_data)
    
    def _generate_portfolio_data(self) -> Dict[str, Any]:
        """Generate comprehensive portfolio data."""
        
        # Generate positions
        positions = [
            {
                'symbol': 'NIFTY',
                'instrument_type': 'FUT',
                'quantity': 100,
                'avg_price': 18000,
                'ltp': 18150,
                'unrealized_pnl': 15000,
                'realized_pnl': 25000,
                'margin_used': 180000,
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'symbol': 'BANKNIFTY',
                'instrument_type': 'OPT',
                'quantity': 50,
                'avg_price': 42000,
                'ltp': 41800,
                'unrealized_pnl': -10000,
                'realized_pnl': 15000,
                'margin_used': 210000,
                'timestamp': datetime.now() - timedelta(hours=1)
            },
            {
                'symbol': 'NIFTY',
                'instrument_type': 'OPT',
                'quantity': 200,
                'avg_price': 150,
                'ltp': 175,
                'unrealized_pnl': 5000,
                'realized_pnl': 8000,
                'margin_used': 30000,
                'timestamp': datetime.now() - timedelta(minutes=30)
            }
        ]
        
        # Calculate portfolio metrics
        total_invested = sum(pos['quantity'] * pos['avg_price'] for pos in positions)
        total_current_value = sum(pos['quantity'] * pos['ltp'] for pos in positions)
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
        total_realized_pnl = sum(pos['realized_pnl'] for pos in positions)
        total_margin_used = sum(pos['margin_used'] for pos in positions)
        
        # Generate historical performance
        performance_history = self._generate_performance_history()
        
        # Generate trade history
        trade_history = self._generate_trade_history()
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(performance_history)
        
        return {
            'positions': positions,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_margin_used': total_margin_used,
            'available_cash': self.initial_capital - total_margin_used,
            'performance_history': performance_history,
            'trade_history': trade_history,
            'risk_metrics': risk_metrics
        }
    
    def _generate_performance_history(self) -> pd.DataFrame:
        """Generate historical performance data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), 
                            end=datetime.now(), freq='D')
        
        # Generate realistic portfolio performance
        base_value = self.initial_capital
        returns = np.random.normal(0.0008, 0.02, len(dates))  # Slightly positive bias
        
        # Add some trend and volatility
        trend = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.005
        returns = returns + trend
        
        portfolio_values = [base_value]
        for ret in returns[1:]:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # Calculate daily P&L
        daily_pnl = [0]
        for i in range(1, len(portfolio_values)):
            daily_pnl.append(portfolio_values[i] - portfolio_values[i-1])
        
        # Calculate cumulative P&L
        cumulative_pnl = np.cumsum(daily_pnl)
        
        return pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'daily_pnl': daily_pnl,
            'cumulative_pnl': cumulative_pnl,
            'daily_return': returns
        })
    
    def _generate_trade_history(self) -> pd.DataFrame:
        """Generate trade history data."""
        n_trades = 150
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), 
                            end=datetime.now(), freq='1H')
        
        trades = []
        symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
        sides = ['BUY', 'SELL']
        
        for i in range(n_trades):
            trade_date = np.random.choice(dates)
            symbol = np.random.choice(symbols)
            side = np.random.choice(sides)
            quantity = np.random.randint(10, 200)
            price = np.random.uniform(15000, 45000)
            pnl = np.random.uniform(-5000, 5000)
            
            trades.append({
                'timestamp': trade_date,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'status': np.random.choice(['FILLED', 'PARTIAL'], p=[0.9, 0.1])
            })
        
        return pd.DataFrame(trades)
    
    def _calculate_risk_metrics(self, performance_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics from performance data."""
        
        returns = performance_data['daily_return'].dropna()
        
        # Basic metrics
        total_return = (performance_data['portfolio_value'].iloc[-1] / 
                       performance_data['portfolio_value'].iloc[0] - 1) * 100
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 6% risk-free rate)
        risk_free_rate = 0.06
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1] * 100
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = (winning_days / total_days) * 100
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # VaR (95% and 99%)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'win_rate': win_rate,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'var_99': var_99
        }
    
    def _render_portfolio_summary(self, data: Dict[str, Any]) -> None:
        """Render portfolio summary metrics."""
        
        st.subheader("Portfolio Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = data['total_current_value']
            initial_value = self.initial_capital
            total_return_pct = ((total_value - initial_value) / initial_value) * 100
            
            st.metric(
                label="Portfolio Value",
                value=f"₹{total_value:,.0f}",
                delta=f"₹{total_value - initial_value:,.0f} ({total_return_pct:.2f}%)",
                delta_color="normal" if total_return_pct >= 0 else "inverse"
            )
        
        with col2:
            available_cash = data['available_cash']
            st.metric(
                label="Available Cash",
                value=f"₹{available_cash:,.0f}",
                delta="₹0"
            )
        
        with col3:
            total_pnl = data['total_pnl']
            st.metric(
                label="Total P&L",
                value=f"₹{total_pnl:,.0f}",
                delta=f"₹{data['total_unrealized_pnl']:,.0f} (Unrealized)",
                delta_color="normal" if total_pnl >= 0 else "inverse"
            )
        
        with col4:
            margin_utilization = (data['total_margin_used'] / self.initial_capital) * 100
            st.metric(
                label="Margin Used",
                value=f"₹{data['total_margin_used']:,.0f}",
                delta=f"{margin_utilization:.1f}% of Capital"
            )
    
    def _render_positions_overview(self, data: Dict[str, Any]) -> None:
        """Render positions overview table."""
        
        st.subheader("Current Positions")
        
        positions_df = pd.DataFrame(data['positions'])
        
        if not positions_df.empty:
            # Calculate additional metrics
            positions_df['investment'] = positions_df['quantity'] * positions_df['avg_price']
            positions_df['current_value'] = positions_df['quantity'] * positions_df['ltp']
            positions_df['pnl_percent'] = (positions_df['unrealized_pnl'] / positions_df['investment']) * 100
            
            # Format the display
            display_df = positions_df[[
                'symbol', 'instrument_type', 'quantity', 'avg_price', 'ltp', 
                'unrealized_pnl', 'pnl_percent', 'margin_used'
            ]].copy()
            
            # Round numeric columns
            numeric_cols = ['avg_price', 'ltp', 'unrealized_pnl', 'pnl_percent', 'margin_used']
            for col in numeric_cols:
                display_df[col] = display_df[col].round(2)
            
            # Rename columns
            display_df.columns = [
                'Symbol', 'Type', 'Qty', 'Avg Price', 'LTP', 
                'P&L', 'P&L %', 'Margin Used'
            ]
            
            # Color code P&L
            def color_pnl(val):
                if val >= 0:
                    return 'color: green; font-weight: bold'
                else:
                    return 'color: red; font-weight: bold'
            
            styled_df = display_df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Position summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Positions", len(positions_df))
                st.metric("Profitable Positions", len(positions_df[positions_df['unrealized_pnl'] > 0]))
            
            with col2:
                st.metric("Largest Position", f"₹{positions_df['investment'].max():,.0f}")
                st.metric("Average P&L %", f"{positions_df['pnl_percent'].mean():.2f}%")
        
        else:
            st.info("No open positions")
    
    def _render_performance_charts(self, data: Dict[str, Any]) -> None:
        """Render performance charts."""
        
        st.subheader("Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Portfolio Value Over Time**")
            
            fig_value = go.Figure()
            fig_value.add_trace(go.Scatter(
                x=data['performance_history']['date'],
                y=data['performance_history']['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add initial capital line
            fig_value.add_hline(
                y=self.initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
            
            fig_value.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                showlegend=False
            )
            
            st.plotly_chart(fig_value, use_container_width=True)
        
        with col2:
            st.markdown("**Daily P&L**")
            
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=data['performance_history']['date'],
                y=data['performance_history']['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='green' if data['performance_history']['cumulative_pnl'].iloc[-1] >= 0 else 'red', width=2)
            ))
            
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_pnl.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Cumulative P&L (₹)",
                showlegend=False
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Asset allocation
        st.markdown("**Asset Allocation**")
        
        allocation_data = {
            'Cash': data['available_cash'],
            'Equity Futures': data['total_margin_used'] * 0.6,
            'Options': data['total_margin_used'] * 0.4
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(allocation_data.keys()),
            values=list(allocation_data.values()),
            hole=0.3
        )])
        
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    def _render_risk_analytics(self, data: Dict[str, Any]) -> None:
        """Render risk analytics."""
        
        st.subheader("Risk Analytics")
        
        risk_metrics = data['risk_metrics']
        
        # Risk metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2f}%")
            st.metric("Current Drawdown", f"{risk_metrics['current_drawdown']:.2f}%")
        
        with col3:
            st.metric("Volatility", f"{risk_metrics['volatility']:.2f}%")
            st.metric("Win Rate", f"{risk_metrics['win_rate']:.1f}%")
        
        with col4:
            st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2f}%")
            st.metric("VaR (99%)", f"{risk_metrics['var_99']:.2f}%")
        
        # Drawdown chart
        st.markdown("**Drawdown Analysis**")
        
        performance_data = data['performance_history']
        cumulative = (1 + performance_data['daily_return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=performance_data['date'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig_drawdown.update_layout(
            height=300,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            showlegend=False
        )
        
        st.plotly_chart(fig_drawdown, use_container_width=True)
    
    def _render_trade_analysis(self, data: Dict[str, Any]) -> None:
        """Render trade analysis."""
        
        st.subheader("Trade Analysis")
        
        trade_data = data['trade_history']
        
        # Trade summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = len(trade_data)
            st.metric("Total Trades", total_trades)
        
        with col2:
            winning_trades = len(trade_data[trade_data['pnl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            total_pnl = trade_data['pnl'].sum()
            st.metric("Total P&L", f"₹{total_pnl:,.0f}")
        
        with col4:
            avg_pnl = trade_data['pnl'].mean()
            st.metric("Avg P&L per Trade", f"₹{avg_pnl:,.0f}")
        
        # Recent trades
        st.markdown("**Recent Trades**")
        
        recent_trades = trade_data.tail(10).copy()
        recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        recent_trades['price'] = recent_trades['price'].round(2)
        recent_trades['pnl'] = recent_trades['pnl'].round(2)
        
        # Rename columns
        recent_trades.columns = ['Time', 'Symbol', 'Side', 'Qty', 'Price', 'P&L', 'Status']
        
        st.dataframe(recent_trades, use_container_width=True, hide_index=True)
        
        # P&L distribution
        st.markdown("**P&L Distribution**")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trade_data['pnl'],
            nbinsx=20,
            name='P&L Distribution',
            marker_color='lightblue'
        ))
        
        fig_hist.update_layout(
            height=300,
            xaxis_title="P&L (₹)",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)


def render_portfolio_page():
    """Render the complete portfolio page."""
    st.markdown('<div class="main-header">💼 Portfolio Analytics</div>', 
                unsafe_allow_html=True)
    
    # Initialize component
    portfolio = PortfolioComponent()
    
    # Render portfolio dashboard
    portfolio.render_portfolio_dashboard()


if __name__ == "__main__":
    render_portfolio_page()
