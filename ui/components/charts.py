"""
Charts Component for Streamlit Dashboard.

This module provides comprehensive charting capabilities including
candlestick charts, technical indicators, and real-time data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add project root to path
sys.path.append('.')

# Import our technical indicators
from strategies.supertrend import SupertrendIndicator


class ChartsComponent:
    """Charts and technical analysis component."""
    
    def __init__(self):
        self.symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.indicators = {
            "Supertrend": SupertrendIndicator,
            "RSI": self._calculate_rsi,
            "MACD": self._calculate_macd,
            "Bollinger Bands": self._calculate_bollinger_bands,
            "SMA": self._calculate_sma,
            "EMA": self._calculate_ema,
            "Volume": self._calculate_volume_indicators
        }
    
    def render_price_chart(self, symbol: str = "NIFTY", 
                          timeframe: str = "1h",
                          show_indicators: List[str] = None,
                          days: int = 30) -> None:
        """
        Render comprehensive price chart with technical indicators.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            show_indicators: List of indicators to show
            days: Number of days to display
        """
        if show_indicators is None:
            show_indicators = ["Supertrend", "RSI", "Volume"]
        
        # Generate price data
        price_data = self._generate_price_data(symbol, timeframe, days)
        
        # Render controls
        self._render_chart_controls(symbol, timeframe, show_indicators)
        
        # Render main price chart
        self._render_main_chart(price_data, symbol, show_indicators)
        
        # Render additional charts
        self._render_additional_charts(price_data, show_indicators)
    
    def _render_chart_controls(self, symbol: str, timeframe: str, 
                              show_indicators: List[str]) -> None:
        """Render chart control elements."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_symbol = st.selectbox("Symbol", self.symbols, 
                                         index=self.symbols.index(symbol) if symbol in self.symbols else 0,
                                         key="chart_symbol")
        
        with col2:
            selected_timeframe = st.selectbox("Timeframe", self.timeframes, 
                                            index=self.timeframes.index(timeframe) if timeframe in self.timeframes else 3,
                                            key="chart_timeframe")
        
        with col3:
            days = st.selectbox("Period", [7, 15, 30, 60, 90, 180, 365], 
                              index=2, key="chart_period")
        
        with col4:
            if st.button("🔄 Refresh Chart", use_container_width=True, key="refresh_chart"):
                st.rerun()
    
    def _generate_price_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate realistic price data with technical indicators."""
        
        # Determine frequency based on timeframe
        freq_map = {
            "1m": "1min",
            "5m": "5min", 
            "15m": "15min",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D"
        }
        
        freq = freq_map.get(timeframe, "1H")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Determine base price based on symbol
        base_prices = {
            "NIFTY": 18000,
            "BANKNIFTY": 42000,
            "FINNIFTY": 20000,
            "SENSEX": 60000
        }
        base_price = base_prices.get(symbol, 18000)
        
        # Generate realistic price movements
        n_points = len(dates)
        returns = np.random.normal(0, 0.01, n_points)
        
        # Add some trend and volatility clustering
        trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.005
        volatility = 1 + 0.5 * np.sin(np.linspace(0, 8*np.pi, n_points))
        
        returns = returns * volatility + trend
        
        # Generate OHLC data
        prices = [base_price]
        opens = [base_price]
        highs = [base_price]
        lows = [base_price]
        volumes = []
        
        for i in range(1, n_points):
            # Calculate new price
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
            
            # Generate OHLC for this period
            open_price = prices[-2] * (1 + np.random.normal(0, 0.002))
            close_price = new_price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            
            # Generate volume
            base_volume = 1000000
            volume_multiplier = 1 + 0.5 * abs(returns[i]) * 10  # Higher volume on big moves
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
            volumes.append(volume)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        # Calculate technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        
        # Supertrend
        supertrend_indicator = SupertrendIndicator(period=10, multiplier=2.0)
        supertrend_data = supertrend_indicator.calculate(
            df['close'].values,
            df['high'].values,
            df['low'].values
        )
        df['supertrend'] = supertrend_data['supertrend']
        df['supertrend_direction'] = supertrend_data['direction']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        
        # Moving Averages
        df['sma_20'] = self._calculate_sma(df['close'], 20)
        df['sma_50'] = self._calculate_sma(df['close'], 50)
        df['ema_20'] = self._calculate_ema(df['close'], 20)
        df['ema_50'] = self._calculate_ema(df['close'], 50)
        
        # Volume indicators
        df['volume_sma'] = self._calculate_sma(df['volume'], 20)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd, signal)
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # Volume SMA
        df['volume_sma'] = self._calculate_sma(df['volume'], 20)
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _render_main_chart(self, data: pd.DataFrame, symbol: str, 
                          show_indicators: List[str]) -> None:
        """Render the main price chart with candlesticks and indicators."""
        
        # Create subplots
        if "Volume" in show_indicators:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Price Chart', 'Volume'),
                row_width=[0.7, 0.3]
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=(f'{symbol} Price Chart',)
            )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add technical indicators
        if "Supertrend" in show_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['supertrend'],
                    mode='lines',
                    name='Supertrend',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        if "SMA" in show_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if "EMA" in show_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['ema_20'],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color='purple', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        if "Bollinger Bands" in show_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['bb_lower'],
                    mode='lines',
                    name='Bollinger Bands',
                    line=dict(color='gray', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )
        
        # Add volume chart
        if "Volume" in show_indicators:
            fig.add_trace(
                go.Bar(
                    x=data['timestamp'],
                    y=data['volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Chart with Technical Indicators',
            xaxis_rangeslider_visible=False,
            height=600 if "Volume" in show_indicators else 500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if "Volume" in show_indicators:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_additional_charts(self, data: pd.DataFrame, 
                                show_indicators: List[str]) -> None:
        """Render additional indicator charts."""
        
        if "RSI" in show_indicators:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RSI (Relative Strength Index)")
                
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    )
                )
                
                # Add overbought/oversold lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                 annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                 annotation_text="Oversold (30)")
                
                fig_rsi.update_layout(
                    height=300,
                    yaxis=dict(range=[0, 100]),
                    showlegend=False
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                st.subheader("MACD (Moving Average Convergence Divergence)")
                
                fig_macd = go.Figure()
                fig_macd.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    )
                )
                fig_macd.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['macd_signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    )
                )
                fig_macd.add_trace(
                    go.Bar(
                        x=data['timestamp'],
                        y=data['macd_histogram'],
                        name='Histogram',
                        marker_color='gray',
                        opacity=0.7
                    )
                )
                
                fig_macd.update_layout(
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)


def render_charts_page():
    """Render the complete charts page."""
    st.markdown('<div class="main-header">📈 Charts & Technical Analysis</div>', 
                unsafe_allow_html=True)
    
    # Initialize component
    charts = ChartsComponent()
    
    # Render price chart with indicators
    charts.render_price_chart(
        symbol="NIFTY",
        timeframe="1h",
        show_indicators=["Supertrend", "RSI", "MACD", "Volume"],
        days=30
    )


if __name__ == "__main__":
    render_charts_page()
