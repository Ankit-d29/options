"""
Supertrend indicator implementation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy, TradingSignal, SignalType
from utils.logging_utils import strategy_logger


class SupertrendIndicator:
    """Supertrend indicator implementation based on ATR."""
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        strategy_logger.info(f"Initialized Supertrend with period={period}, multiplier={multiplier}")
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=self.period, adjust=False).mean()
        
        return atr
    
    def calculate_supertrend(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Supertrend indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (supertrend_values, trend_direction)
            trend_direction: 1 for uptrend, -1 for downtrend
        """
        if len(df) < self.period:
            strategy_logger.warning("Insufficient data for Supertrend calculation")
            return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)
        
        # Initialize arrays
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        # Calculate final upper and lower bands
        final_upper_band = pd.Series(index=df.index, dtype=float)
        final_lower_band = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i == 0:
                final_upper_band.iloc[i] = upper_band.iloc[i]
                final_lower_band.iloc[i] = lower_band.iloc[i]
            else:
                # Final Upper Band
                if (upper_band.iloc[i] < final_upper_band.iloc[i-1] or 
                    close.iloc[i-1] > final_upper_band.iloc[i-1]):
                    final_upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
                
                # Final Lower Band
                if (lower_band.iloc[i] > final_lower_band.iloc[i-1] or 
                    close.iloc[i-1] < final_lower_band.iloc[i-1]):
                    final_lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        
        # Calculate Supertrend and Trend
        for i in range(len(df)):
            if i == 0:
                trend.iloc[i] = 1  # Start with uptrend assumption
                supertrend.iloc[i] = final_lower_band.iloc[i]
            else:
                # Trend calculation
                if (trend.iloc[i-1] == 1 and close.iloc[i] <= final_lower_band.iloc[i]):
                    trend.iloc[i] = -1
                elif (trend.iloc[i-1] == -1 and close.iloc[i] >= final_upper_band.iloc[i]):
                    trend.iloc[i] = 1
                else:
                    trend.iloc[i] = trend.iloc[i-1]
                
                # Supertrend calculation
                if trend.iloc[i] == 1:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
        
        return supertrend, trend
    
    def add_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Supertrend indicator to DataFrame."""
        df_copy = df.copy()
        
        supertrend, trend = self.calculate_supertrend(df_copy)
        
        df_copy['supertrend'] = supertrend
        df_copy['supertrend_trend'] = trend
        df_copy['supertrend_atr'] = self.calculate_atr(df_copy)
        
        # Add trend change signals
        df_copy['supertrend_signal'] = 0
        trend_change = trend.diff()
        df_copy.loc[trend_change == 2, 'supertrend_signal'] = 1  # Uptrend
        df_copy.loc[trend_change == -2, 'supertrend_signal'] = -1  # Downtrend
        
        strategy_logger.info(f"Added Supertrend indicator to DataFrame with {len(df_copy)} rows")
        
        return df_copy
    
    def get_signals(self, df: pd.DataFrame) -> list[TradingSignal]:
        """Extract trading signals from Supertrend indicator."""
        signals = []
        
        if 'supertrend_signal' not in df.columns:
            df = self.add_to_dataframe(df)
        
        signal_points = df[df['supertrend_signal'] != 0]
        
        for idx, row in signal_points.iterrows():
            if row['supertrend_signal'] == 1:  # Uptrend signal
                signal_type = SignalType.BUY
                confidence = min(0.9, 0.5 + abs(row['close'] - row['supertrend']) / row['supertrend'] * 10)
            else:  # Downtrend signal
                signal_type = SignalType.SELL
                confidence = min(0.9, 0.5 + abs(row['close'] - row['supertrend']) / row['supertrend'] * 10)
            
            signal = TradingSignal(
                timestamp=row['timestamp'],
                symbol=row.get('symbol', 'UNKNOWN'),
                signal_type=signal_type,
                price=row['close'],
                confidence=confidence,
                metadata={
                    'supertrend_value': row['supertrend'],
                    'atr_value': row['supertrend_atr'],
                    'trend_direction': row['supertrend_trend']
                }
            )
            
            signals.append(signal)
        
        strategy_logger.info(f"Generated {len(signals)} Supertrend signals")
        return signals


class SupertrendStrategy(BaseStrategy):
    """Strategy implementation using Supertrend indicator."""
    
    def __init__(self, period: int = 10, multiplier: float = 3.0, config: Dict[str, Any] = None):
        super().__init__("SupertrendStrategy", config)
        self.supertrend = SupertrendIndicator(period, multiplier)
        self.period = period
        self.multiplier = multiplier
        
        strategy_logger.info(f"Initialized SupertrendStrategy with period={period}, multiplier={multiplier}")
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals using Supertrend indicator."""
        if len(df) < self.period:
            strategy_logger.warning("Insufficient data for Supertrend calculation")
            return df
        
        # Add Supertrend indicator to DataFrame
        df_with_supertrend = self.supertrend.add_to_dataframe(df)
        
        # Generate signals
        signals = self.supertrend.get_signals(df_with_supertrend)
        
        # Add signals to strategy
        for signal in signals:
            self.signals.append(signal)
        
        strategy_logger.info(f"Generated {len(signals)} signals for {self.name}")
        
        return df_with_supertrend
    
    def get_trend_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get current trend status from Supertrend indicator."""
        if len(df) < self.period:
            return {'trend': 'unknown', 'supertrend_value': None, 'confidence': 0.0}
        
        df_with_supertrend = self.supertrend.add_to_dataframe(df)
        latest = df_with_supertrend.iloc[-1]
        
        trend_direction = 'uptrend' if latest['supertrend_trend'] == 1 else 'downtrend'
        distance_from_supertrend = abs(latest['close'] - latest['supertrend']) / latest['close']
        confidence = min(0.9, 0.3 + distance_from_supertrend * 50)
        
        return {
            'trend': trend_direction,
            'supertrend_value': latest['supertrend'],
            'current_price': latest['close'],
            'distance_percent': distance_from_supertrend * 100,
            'confidence': confidence,
            'atr_value': latest['supertrend_atr']
        }
    
    def should_enter_long(self, df: pd.DataFrame) -> bool:
        """Check if conditions are met for long entry."""
        if len(df) < self.period:
            return False
        
        df_with_supertrend = self.supertrend.add_to_dataframe(df)
        latest = df_with_supertrend.iloc[-1]
        
        # Enter long when price crosses above Supertrend
        return (latest['supertrend_trend'] == 1 and 
                latest['close'] > latest['supertrend'] and
                df_with_supertrend.iloc[-2]['supertrend_trend'] == -1)
    
    def should_enter_short(self, df: pd.DataFrame) -> bool:
        """Check if conditions are met for short entry."""
        if len(df) < self.period:
            return False
        
        df_with_supertrend = self.supertrend.add_to_dataframe(df)
        latest = df_with_supertrend.iloc[-1]
        
        # Enter short when price crosses below Supertrend
        return (latest['supertrend_trend'] == -1 and 
                latest['close'] < latest['supertrend'] and
                df_with_supertrend.iloc[-2]['supertrend_trend'] == 1)
