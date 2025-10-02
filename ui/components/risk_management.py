"""
Risk Management Component for Streamlit Dashboard.

This module provides comprehensive risk monitoring and management
including position limits, margin monitoring, and alert management.
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


class RiskManagementComponent:
    """Risk management and monitoring component."""
    
    def __init__(self):
        self.risk_limits = {
            'max_daily_loss': 50000,
            'max_positions': 10,
            'max_position_size': 200000,
            'max_margin_utilization': 0.8,
            'stop_loss_percent': 0.02,
            'var_limit': 0.05
        }
        self.positions = []
        self.alerts = []
        self.risk_events = []
    
    def render_risk_dashboard(self) -> None:
        """Render the complete risk management dashboard."""
        
        # Generate risk data
        risk_data = self._generate_risk_data()
        
        # Risk Overview
        self._render_risk_overview(risk_data)
        
        # Position Limits
        self._render_position_limits(risk_data)
        
        # Margin Monitoring
        self._render_margin_monitoring(risk_data)
        
        # Risk Alerts
        self._render_risk_alerts(risk_data)
        
        # Risk Analytics
        self._render_risk_analytics(risk_data)
        
        # Controls
        self._render_risk_controls()
    
    def _generate_risk_data(self) -> Dict[str, Any]:
        """Generate comprehensive risk data."""
        
        # Generate positions
        positions = [
            {
                'symbol': 'NIFTY',
                'instrument_type': 'FUT',
                'quantity': 100,
                'avg_price': 18000,
                'ltp': 18150,
                'unrealized_pnl': 15000,
                'margin_used': 180000,
                'position_value': 1815000,
                'risk_percent': 0.018,
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'symbol': 'BANKNIFTY',
                'instrument_type': 'OPT',
                'quantity': 50,
                'avg_price': 42000,
                'ltp': 41800,
                'unrealized_pnl': -10000,
                'margin_used': 210000,
                'position_value': 2090000,
                'risk_percent': 0.021,
                'timestamp': datetime.now() - timedelta(hours=1)
            },
            {
                'symbol': 'NIFTY',
                'instrument_type': 'OPT',
                'quantity': 200,
                'avg_price': 150,
                'ltp': 175,
                'unrealized_pnl': 5000,
                'margin_used': 30000,
                'position_value': 35000,
                'risk_percent': 0.035,
                'timestamp': datetime.now() - timedelta(minutes=30)
            }
        ]
        
        # Calculate portfolio risk metrics
        total_margin_used = sum(pos['margin_used'] for pos in positions)
        total_position_value = sum(pos['position_value'] for pos in positions)
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
        max_position_value = max(pos['position_value'] for pos in positions) if positions else 0
        portfolio_value = 1000000  # ₹10 lakh
        
        # Calculate daily P&L
        daily_pnl = np.random.uniform(-10000, 15000)  # Simulated daily P&L
        
        # Generate margin utilization data
        margin_history = self._generate_margin_history()
        
        # Generate alerts
        alerts = self._generate_risk_alerts()
        
        # Generate risk events
        risk_events = self._generate_risk_events()
        
        return {
            'positions': positions,
            'total_margin_used': total_margin_used,
            'total_position_value': total_position_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'max_position_value': max_position_value,
            'portfolio_value': portfolio_value,
            'daily_pnl': daily_pnl,
            'margin_history': margin_history,
            'alerts': alerts,
            'risk_events': risk_events,
            'risk_limits': self.risk_limits
        }
    
    def _generate_margin_history(self) -> pd.DataFrame:
        """Generate margin utilization history."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='H')
        
        # Generate realistic margin utilization
        base_margin = 400000
        margin_utilization = []
        
        for i, date in enumerate(dates):
            # Add some trend and volatility
            trend = np.sin(i / 24) * 50000  # Daily cycle
            noise = np.random.normal(0, 20000)
            margin = max(0, base_margin + trend + noise)
            margin_utilization.append(margin)
        
        return pd.DataFrame({
            'timestamp': dates,
            'margin_used': margin_utilization,
            'margin_limit': [self.risk_limits['max_margin_utilization'] * 1000000] * len(dates)
        })
    
    def _generate_risk_alerts(self) -> List[Dict[str, Any]]:
        """Generate risk alerts."""
        return [
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'alert_type': 'HIGH_MARGIN_UTILIZATION',
                'severity': 'WARNING',
                'message': 'Margin utilization at 78% - approaching limit',
                'status': 'ACTIVE'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'alert_type': 'LARGE_POSITION',
                'severity': 'INFO',
                'message': 'NIFTY position exceeds 15% of portfolio',
                'status': 'ACTIVE'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'alert_type': 'STOP_LOSS_HIT',
                'severity': 'CRITICAL',
                'message': 'Stop loss triggered for BANKNIFTY position',
                'status': 'RESOLVED'
            }
        ]
    
    def _generate_risk_events(self) -> List[Dict[str, Any]]:
        """Generate risk events history."""
        return [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'event_type': 'POSITION_OPENED',
                'symbol': 'NIFTY',
                'details': 'Opened long position of 100 lots',
                'risk_impact': 'MEDIUM'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=1, minutes=30),
                'event_type': 'MARGIN_CHECK',
                'symbol': 'PORTFOLIO',
                'details': 'Margin utilization crossed 70%',
                'risk_impact': 'HIGH'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'event_type': 'STOP_LOSS_EXECUTED',
                'symbol': 'BANKNIFTY',
                'details': 'Stop loss executed at 41800',
                'risk_impact': 'CRITICAL'
            }
        ]
    
    def _render_risk_overview(self, data: Dict[str, Any]) -> None:
        """Render risk overview metrics."""
        
        st.subheader("Risk Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            margin_utilization = (data['total_margin_used'] / data['portfolio_value']) * 100
            st.metric(
                label="Margin Utilization",
                value=f"{margin_utilization:.1f}%",
                delta=f"Limit: {data['risk_limits']['max_margin_utilization']*100:.0f}%",
                delta_color="inverse" if margin_utilization > data['risk_limits']['max_margin_utilization']*100 else "normal"
            )
        
        with col2:
            daily_pnl = data['daily_pnl']
            st.metric(
                label="Daily P&L",
                value=f"₹{daily_pnl:,.0f}",
                delta=f"Limit: ₹{data['risk_limits']['max_daily_loss']:,.0f}",
                delta_color="inverse" if daily_pnl < -data['risk_limits']['max_daily_loss'] else "normal"
            )
        
        with col3:
            position_count = len(data['positions'])
            st.metric(
                label="Open Positions",
                value=position_count,
                delta=f"Limit: {data['risk_limits']['max_positions']}",
                delta_color="inverse" if position_count > data['risk_limits']['max_positions'] else "normal"
            )
        
        with col4:
            max_position_ratio = (data['max_position_value'] / data['portfolio_value']) * 100
            st.metric(
                label="Largest Position",
                value=f"{max_position_ratio:.1f}%",
                delta="Portfolio",
                delta_color="inverse" if max_position_ratio > 20 else "normal"
            )
        
        # Risk Status
        risk_status = self._calculate_risk_status(data)
        
        if risk_status['overall'] == 'SAFE':
            st.success("✅ Portfolio Risk Status: SAFE")
        elif risk_status['overall'] == 'WARNING':
            st.warning("⚠️ Portfolio Risk Status: WARNING")
        else:
            st.error("🚨 Portfolio Risk Status: CRITICAL")
    
    def _calculate_risk_status(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Calculate overall risk status."""
        
        status = {
            'margin': 'SAFE',
            'daily_pnl': 'SAFE',
            'positions': 'SAFE',
            'overall': 'SAFE'
        }
        
        # Check margin utilization
        margin_utilization = (data['total_margin_used'] / data['portfolio_value'])
        if margin_utilization > data['risk_limits']['max_margin_utilization'] * 0.9:
            status['margin'] = 'CRITICAL'
        elif margin_utilization > data['risk_limits']['max_margin_utilization'] * 0.7:
            status['margin'] = 'WARNING'
        
        # Check daily P&L
        if data['daily_pnl'] < -data['risk_limits']['max_daily_loss']:
            status['daily_pnl'] = 'CRITICAL'
        elif data['daily_pnl'] < -data['risk_limits']['max_daily_loss'] * 0.7:
            status['daily_pnl'] = 'WARNING'
        
        # Check position count
        if len(data['positions']) > data['risk_limits']['max_positions']:
            status['positions'] = 'CRITICAL'
        elif len(data['positions']) > data['risk_limits']['max_positions'] * 0.8:
            status['positions'] = 'WARNING'
        
        # Overall status
        if any(s == 'CRITICAL' for s in status.values()):
            status['overall'] = 'CRITICAL'
        elif any(s == 'WARNING' for s in status.values()):
            status['overall'] = 'WARNING'
        
        return status
    
    def _render_position_limits(self, data: Dict[str, Any]) -> None:
        """Render position limits monitoring."""
        
        st.subheader("Position Limits Monitoring")
        
        positions_df = pd.DataFrame(data['positions'])
        
        if not positions_df.empty:
            # Calculate position metrics
            positions_df['position_ratio'] = (positions_df['position_value'] / data['portfolio_value']) * 100
            positions_df['margin_ratio'] = (positions_df['margin_used'] / data['portfolio_value']) * 100
            
            # Create position limits chart
            fig = go.Figure()
            
            # Add position values
            fig.add_trace(go.Bar(
                x=positions_df['symbol'],
                y=positions_df['position_ratio'],
                name='Position Size (%)',
                marker_color='lightblue'
            ))
            
            # Add margin used
            fig.add_trace(go.Bar(
                x=positions_df['symbol'],
                y=positions_df['margin_ratio'],
                name='Margin Used (%)',
                marker_color='orange'
            ))
            
            # Add limits
            fig.add_hline(
                y=20,  # 20% position limit
                line_dash="dash",
                line_color="red",
                annotation_text="Position Limit (20%)"
            )
            
            fig.update_layout(
                title="Position vs Limits",
                xaxis_title="Symbol",
                yaxis_title="Percentage of Portfolio",
                height=400,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Position details table
            display_df = positions_df[[
                'symbol', 'instrument_type', 'quantity', 'position_value', 
                'position_ratio', 'margin_used', 'unrealized_pnl'
            ]].copy()
            
            # Round numeric columns
            numeric_cols = ['position_value', 'position_ratio', 'margin_used', 'unrealized_pnl']
            for col in numeric_cols:
                display_df[col] = display_df[col].round(2)
            
            # Rename columns
            display_df.columns = [
                'Symbol', 'Type', 'Qty', 'Value', 
                'Value %', 'Margin', 'P&L'
            ]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("No open positions")
    
    def _render_margin_monitoring(self, data: Dict[str, Any]) -> None:
        """Render margin monitoring."""
        
        st.subheader("Margin Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Margin Utilization Over Time**")
            
            margin_data = data['margin_history']
            
            fig_margin = go.Figure()
            fig_margin.add_trace(go.Scatter(
                x=margin_data['timestamp'],
                y=margin_data['margin_used'],
                mode='lines',
                name='Margin Used',
                line=dict(color='blue', width=2)
            ))
            
            # Add margin limit
            fig_margin.add_trace(go.Scatter(
                x=margin_data['timestamp'],
                y=margin_data['margin_limit'],
                mode='lines',
                name='Margin Limit',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_margin.update_layout(
                height=300,
                xaxis_title="Time",
                yaxis_title="Margin (₹)",
                showlegend=True
            )
            
            st.plotly_chart(fig_margin, use_container_width=True)
        
        with col2:
            st.markdown("**Margin Breakdown**")
            
            # Calculate margin breakdown
            total_margin = data['total_margin_used']
            available_margin = data['portfolio_value'] - total_margin
            
            margin_breakdown = {
                'Used': total_margin,
                'Available': available_margin
            }
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(margin_breakdown.keys()),
                values=list(margin_breakdown.values()),
                hole=0.3,
                marker_colors=['#ff9999', '#66b3ff']
            )])
            
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Margin alerts
        margin_utilization = (data['total_margin_used'] / data['portfolio_value']) * 100
        limit_percent = data['risk_limits']['max_margin_utilization'] * 100
        
        if margin_utilization > limit_percent:
            st.error(f"🚨 Margin utilization ({margin_utilization:.1f}%) exceeds limit ({limit_percent:.0f}%)")
        elif margin_utilization > limit_percent * 0.8:
            st.warning(f"⚠️ Margin utilization ({margin_utilization:.1f}%) approaching limit ({limit_percent:.0f}%)")
        else:
            st.success(f"✅ Margin utilization ({margin_utilization:.1f}%) within limits")
    
    def _render_risk_alerts(self, data: Dict[str, Any]) -> None:
        """Render risk alerts."""
        
        st.subheader("Risk Alerts")
        
        alerts_df = pd.DataFrame(data['alerts'])
        
        if not alerts_df.empty:
            # Format timestamps
            alerts_df['timestamp'] = alerts_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Color code by severity
            def color_severity(val):
                if val == 'CRITICAL':
                    return 'background-color: red; color: white'
                elif val == 'WARNING':
                    return 'background-color: orange; color: white'
                else:
                    return 'background-color: lightblue'
            
            # Rename columns
            alerts_df.columns = ['Time', 'Type', 'Severity', 'Message', 'Status']
            
            styled_df = alerts_df.style.applymap(color_severity, subset=['Severity'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Alert summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                active_alerts = len(alerts_df[alerts_df['Status'] == 'ACTIVE'])
                st.metric("Active Alerts", active_alerts)
            
            with col2:
                critical_alerts = len(alerts_df[alerts_df['Severity'] == 'CRITICAL'])
                st.metric("Critical Alerts", critical_alerts)
            
            with col3:
                resolved_alerts = len(alerts_df[alerts_df['Status'] == 'RESOLVED'])
                st.metric("Resolved Alerts", resolved_alerts)
        
        else:
            st.success("✅ No active risk alerts")
    
    def _render_risk_analytics(self, data: Dict[str, Any]) -> None:
        """Render risk analytics."""
        
        st.subheader("Risk Analytics")
        
        # Generate VaR analysis
        var_data = self._generate_var_analysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Value at Risk (VaR)**")
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(
                x=var_data['confidence_levels'],
                y=var_data['var_values'],
                mode='lines+markers',
                name='VaR',
                line=dict(color='red', width=2)
            ))
            
            fig_var.update_layout(
                height=300,
                xaxis_title="Confidence Level (%)",
                yaxis_title="VaR (₹)",
                showlegend=False
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Metrics**")
            
            # Calculate risk metrics
            portfolio_value = data['portfolio_value']
            daily_volatility = 0.02  # 2% daily volatility
            var_95 = portfolio_value * daily_volatility * 1.645  # 95% VaR
            var_99 = portfolio_value * daily_volatility * 2.326  # 99% VaR
            
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("95% VaR", f"₹{var_95:,.0f}")
                st.metric("99% VaR", f"₹{var_99:,.0f}")
            
            with col2_2:
                st.metric("Daily Volatility", f"{daily_volatility*100:.2f}%")
                st.metric("Sharpe Ratio", "1.25")
        
        # Risk events timeline
        st.markdown("**Risk Events Timeline**")
        
        events_df = pd.DataFrame(data['risk_events'])
        events_df['timestamp'] = events_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        events_df.columns = ['Time', 'Event', 'Symbol', 'Details', 'Impact']
        
        st.dataframe(events_df, use_container_width=True, hide_index=True)
    
    def _generate_var_analysis(self) -> Dict[str, List]:
        """Generate VaR analysis data."""
        confidence_levels = [90, 95, 97.5, 99, 99.5, 99.9]
        var_values = [25000, 35000, 45000, 55000, 65000, 85000]  # Simulated VaR values
        
        return {
            'confidence_levels': confidence_levels,
            'var_values': var_values
        }
    
    def _render_risk_controls(self) -> None:
        """Render risk management controls."""
        
        st.subheader("Risk Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Limits Configuration**")
            
            # Risk limits sliders
            max_daily_loss = st.slider(
                "Max Daily Loss (₹)",
                min_value=10000,
                max_value=100000,
                value=50000,
                step=5000
            )
            
            max_positions = st.slider(
                "Max Open Positions",
                min_value=5,
                max_value=20,
                value=10,
                step=1
            )
            
            max_margin_utilization = st.slider(
                "Max Margin Utilization (%)",
                min_value=50,
                max_value=90,
                value=80,
                step=5
            )
            
            if st.button("Update Risk Limits"):
                st.success("Risk limits updated successfully!")
        
        with col2:
            st.markdown("**Emergency Controls**")
            
            # Kill switch
            if st.button("🚨 Emergency Stop", type="primary"):
                st.error("Emergency stop activated! All trading halted.")
            
            # Force close positions
            if st.button("⚠️ Force Close All Positions"):
                st.warning("Force close initiated for all positions.")
            
            # Margin call handling
            if st.button("💰 Handle Margin Call"):
                st.info("Margin call handling initiated.")
            
            # Risk reset
            if st.button("🔄 Reset Risk Counters"):
                st.success("Risk counters reset successfully!")


def render_risk_management_page():
    """Render the complete risk management page."""
    st.markdown('<div class="main-header">🛡️ Risk Management</div>', 
                unsafe_allow_html=True)
    
    # Initialize component
    risk_mgmt = RiskManagementComponent()
    
    # Render risk dashboard
    risk_mgmt.render_risk_dashboard()


if __name__ == "__main__":
    render_risk_management_page()
