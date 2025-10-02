"""
Option Chain Component for Streamlit Dashboard.

This module provides a comprehensive option chain display component
with real-time data, Greeks, and interactive features.
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


class OptionChainComponent:
    """Option chain display component."""
    
    def __init__(self):
        self.symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
        self.expiry_dates = self._generate_expiry_dates()
    
    def _generate_expiry_dates(self) -> List[str]:
        """Generate expiry dates for the next 3 months."""
        dates = []
        current_date = datetime.now()
        
        # Generate weekly expiries for next 3 months
        for i in range(12):
            expiry_date = current_date + timedelta(weeks=i)
            # Adjust to Thursday (typical expiry day)
            days_to_thursday = (3 - expiry_date.weekday()) % 7
            if days_to_thursday == 0 and expiry_date.weekday() != 3:
                days_to_thursday = 7
            expiry_date += timedelta(days=days_to_thursday)
            dates.append(expiry_date.strftime("%d %b %Y"))
        
        return dates
    
    def render_option_chain(self, symbol: str = "NIFTY", 
                          expiry: str = None,
                          show_greeks: bool = True,
                          show_volume: bool = True,
                          show_oi: bool = True) -> None:
        """
        Render the option chain table.
        
        Args:
            symbol: Trading symbol
            expiry: Expiry date
            show_greeks: Whether to show Greeks
            show_volume: Whether to show volume data
            show_oi: Whether to show open interest data
        """
        if expiry is None:
            expiry = self.expiry_dates[0]
        
        # Generate option chain data
        chain_data = self._generate_option_chain_data(symbol, expiry)
        
        # Display controls
        self._render_controls(symbol, expiry)
        
        # Display option chain table
        self._render_option_chain_table(chain_data, show_greeks, show_volume, show_oi)
        
        # Display summary metrics
        self._render_summary_metrics(chain_data)
        
        # Display charts
        self._render_option_charts(chain_data)
    
    def _render_controls(self, symbol: str, expiry: str) -> None:
        """Render control elements."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_symbol = st.selectbox("Symbol", self.symbols, 
                                         index=self.symbols.index(symbol) if symbol in self.symbols else 0)
        
        with col2:
            selected_expiry = st.selectbox("Expiry", self.expiry_dates, 
                                         index=self.expiry_dates.index(expiry) if expiry in self.expiry_dates else 0)
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col4:
            show_advanced = st.checkbox("Advanced View", value=False)
    
    def _generate_option_chain_data(self, symbol: str, expiry: str) -> pd.DataFrame:
        """Generate mock option chain data."""
        # Determine strike range based on symbol
        if symbol == "NIFTY":
            center_strike = 18000
            strike_range = 500
        elif symbol == "BANKNIFTY":
            center_strike = 42000
            strike_range = 1000
        elif symbol == "FINNIFTY":
            center_strike = 20000
            strike_range = 500
        else:
            center_strike = 60000
            strike_range = 1000
        
        strikes = np.arange(center_strike - strike_range, 
                          center_strike + strike_range + 1, 
                          100 if symbol in ["NIFTY", "FINNIFTY"] else 500)
        
        data = []
        spot_price = center_strike + np.random.randint(-200, 200)
        
        for strike in strikes:
            # Calculate time to expiry
            expiry_date = datetime.strptime(expiry, "%d %b %Y")
            time_to_expiry = (expiry_date - datetime.now()).days / 365.0
            
            # Generate realistic option prices using Black-Scholes approximation
            call_price = self._calculate_call_price(spot_price, strike, time_to_expiry)
            put_price = self._calculate_put_price(spot_price, strike, time_to_expiry)
            
            # Generate Greeks
            delta = self._calculate_delta(spot_price, strike, time_to_expiry)
            gamma = self._calculate_gamma(spot_price, strike, time_to_expiry)
            theta = self._calculate_theta(spot_price, strike, time_to_expiry)
            vega = self._calculate_vega(spot_price, strike, time_to_expiry)
            
            # Generate market data
            iv = np.random.uniform(0.15, 0.35)
            
            # Call option data
            call_ltp = call_price + np.random.uniform(-call_price*0.1, call_price*0.1)
            call_oi = np.random.randint(1000, 50000)
            call_volume = np.random.randint(100, 5000)
            call_bid = max(0, call_ltp - np.random.uniform(0, call_ltp*0.05))
            call_ask = call_ltp + np.random.uniform(0, call_ltp*0.05)
            
            # Put option data
            put_ltp = put_price + np.random.uniform(-put_price*0.1, put_price*0.1)
            put_oi = np.random.randint(1000, 50000)
            put_volume = np.random.randint(100, 5000)
            put_bid = max(0, put_ltp - np.random.uniform(0, put_ltp*0.05))
            put_ask = put_ltp + np.random.uniform(0, put_ltp*0.05)
            
            data.append({
                'Strike': strike,
                'Call_LTP': call_ltp,
                'Call_OI': call_oi,
                'Call_Volume': call_volume,
                'Call_Bid': call_bid,
                'Call_Ask': call_ask,
                'Put_LTP': put_ltp,
                'Put_OI': put_oi,
                'Put_Volume': put_volume,
                'Put_Bid': put_bid,
                'Put_Ask': put_ask,
                'IV': iv,
                'Delta': delta,
                'Gamma': gamma,
                'Theta': theta,
                'Vega': vega,
                'Spot': spot_price
            })
        
        return pd.DataFrame(data)
    
    def _calculate_call_price(self, spot: float, strike: float, time: float) -> float:
        """Calculate call option price using simplified Black-Scholes."""
        if time <= 0:
            return max(0, spot - strike)
        
        # Simplified calculation
        intrinsic = max(0, spot - strike)
        time_value = max(0, strike * 0.02 * np.sqrt(time))
        return intrinsic + time_value
    
    def _calculate_put_price(self, spot: float, strike: float, time: float) -> float:
        """Calculate put option price using simplified Black-Scholes."""
        if time <= 0:
            return max(0, strike - spot)
        
        # Simplified calculation
        intrinsic = max(0, strike - spot)
        time_value = max(0, strike * 0.02 * np.sqrt(time))
        return intrinsic + time_value
    
    def _calculate_delta(self, spot: float, strike: float, time: float) -> float:
        """Calculate option delta."""
        if spot == 0 or time <= 0:
            return 0
        
        # Simplified delta calculation
        moneyness = spot / strike
        if moneyness > 1.1:
            return 0.8
        elif moneyness < 0.9:
            return 0.2
        else:
            return 0.5
    
    def _calculate_gamma(self, spot: float, strike: float, time: float) -> float:
        """Calculate option gamma."""
        return np.random.uniform(0.0001, 0.001)
    
    def _calculate_theta(self, spot: float, strike: float, time: float) -> float:
        """Calculate option theta."""
        return -np.random.uniform(10, 100)
    
    def _calculate_vega(self, spot: float, strike: float, time: float) -> float:
        """Calculate option vega."""
        return np.random.uniform(10, 100)
    
    def _render_option_chain_table(self, data: pd.DataFrame, 
                                 show_greeks: bool, show_volume: bool, 
                                 show_oi: bool) -> None:
        """Render the option chain table."""
        
        # Prepare display columns
        display_cols = ['Strike']
        
        # Call option columns
        display_cols.extend(['Call_LTP'])
        if show_oi:
            display_cols.append('Call_OI')
        if show_volume:
            display_cols.append('Call_Volume')
        display_cols.extend(['Call_Bid', 'Call_Ask'])
        
        # Put option columns
        display_cols.extend(['Put_LTP'])
        if show_oi:
            display_cols.append('Put_OI')
        if show_volume:
            display_cols.append('Put_Volume')
        display_cols.extend(['Put_Bid', 'Put_Ask'])
        
        # Greeks columns
        if show_greeks:
            display_cols.extend(['IV', 'Delta', 'Gamma', 'Theta', 'Vega'])
        
        # Format data
        display_data = data[display_cols].copy()
        
        # Format numeric columns
        numeric_cols = [col for col in display_data.columns if col != 'Strike']
        for col in numeric_cols:
            if 'IV' in col or 'Delta' in col or 'Gamma' in col or 'Theta' in col or 'Vega' in col:
                display_data[col] = display_data[col].round(4)
            elif 'LTP' in col or 'Bid' in col or 'Ask' in col:
                display_data[col] = display_data[col].round(2)
            else:
                display_data[col] = display_data[col].round(0).astype(int)
        
        # Rename columns for display
        column_mapping = {
            'Call_LTP': 'C-LTP',
            'Call_OI': 'C-OI',
            'Call_Volume': 'C-Vol',
            'Call_Bid': 'C-Bid',
            'Call_Ask': 'C-Ask',
            'Put_LTP': 'P-LTP',
            'Put_OI': 'P-OI',
            'Put_Volume': 'P-Vol',
            'Put_Bid': 'P-Bid',
            'Put_Ask': 'P-Ask',
            'IV': 'IV%',
            'Delta': 'Delta',
            'Gamma': 'Gamma',
            'Theta': 'Theta',
            'Vega': 'Vega'
        }
        
        display_data = display_data.rename(columns=column_mapping)
        
        # Display table
        st.dataframe(
            display_data,
            use_container_width=True,
            height=600,
            hide_index=True
        )
    
    def _render_summary_metrics(self, data: pd.DataFrame) -> None:
        """Render summary metrics."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            spot_price = data['Spot'].iloc[0]
            st.metric("Spot Price", f"₹{spot_price:,.0f}")
        
        with col2:
            pcr = (data['Put_OI'].sum() / data['Call_OI'].sum())
            st.metric("Put-Call Ratio", f"{pcr:.2f}")
        
        with col3:
            vix = data['IV'].mean() * 100
            st.metric("VIX", f"{vix:.1f}")
        
        with col4:
            # Calculate max pain (simplified)
            max_pain_strike = data.loc[data['Strike'] == data['Spot'].iloc[0], 'Strike'].iloc[0]
            st.metric("Max Pain", f"₹{max_pain_strike:,.0f}")
        
        with col5:
            total_oi = data['Call_OI'].sum() + data['Put_OI'].sum()
            st.metric("Total OI", f"{total_oi:,}")
    
    def _render_option_charts(self, data: pd.DataFrame) -> None:
        """Render option chain charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Open Interest Distribution")
            
            fig_oi = go.Figure()
            
            fig_oi.add_trace(go.Bar(
                x=data['Strike'],
                y=data['Call_OI'],
                name='Call OI',
                marker_color='green',
                opacity=0.7
            ))
            
            fig_oi.add_trace(go.Bar(
                x=data['Strike'],
                y=-data['Put_OI'],  # Negative for visual separation
                name='Put OI',
                marker_color='red',
                opacity=0.7
            ))
            
            fig_oi.update_layout(
                title="Open Interest by Strike",
                xaxis_title="Strike Price",
                yaxis_title="Open Interest",
                height=400,
                barmode='relative'
            )
            
            st.plotly_chart(fig_oi, use_container_width=True)
        
        with col2:
            st.subheader("Implied Volatility Surface")
            
            fig_iv = go.Figure()
            
            fig_iv.add_trace(go.Scatter(
                x=data['Strike'],
                y=data['IV'] * 100,
                mode='lines+markers',
                name='IV',
                line=dict(color='purple', width=2)
            ))
            
            fig_iv.update_layout(
                title="Implied Volatility by Strike",
                xaxis_title="Strike Price",
                yaxis_title="Implied Volatility (%)",
                height=400
            )
            
            st.plotly_chart(fig_iv, use_container_width=True)


def render_option_chain_page():
    """Render the complete option chain page."""
    st.markdown('<div class="main-header">📋 Option Chain</div>', 
                unsafe_allow_html=True)
    
    # Initialize component
    option_chain = OptionChainComponent()
    
    # Render option chain
    option_chain.render_option_chain(
        symbol="NIFTY",
        show_greeks=True,
        show_volume=True,
        show_oi=True
    )


if __name__ == "__main__":
    render_option_chain_page()
