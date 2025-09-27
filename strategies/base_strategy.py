"""
Base classes for trading strategies and signals.
"""
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from utils.logging_utils import strategy_logger


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }
    
    def __repr__(self):
        return (f"TradingSignal({self.timestamp}, {self.symbol}, "
                f"{self.signal_type.value}, {self.price}, conf:{self.confidence})")


class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.signals = []  # Store generated signals
        strategy_logger.info(f"Initialized strategy: {self.name}")
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals added
        """
        raise NotImplementedError("Subclasses must implement calculate_signals")
    
    def generate_signal(self, timestamp: datetime, symbol: str, price: float,
                       signal_type: SignalType, confidence: float = 0.0,
                       metadata: Dict[str, Any] = None) -> TradingSignal:
        """Generate a trading signal."""
        signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            metadata=metadata
        )
        
        self.signals.append(signal)
        strategy_logger.debug(f"Generated signal: {signal}")
        
        return signal
    
    def get_signals(self) -> list[TradingSignal]:
        """Get all generated signals."""
        return self.signals.copy()
    
    def clear_signals(self):
        """Clear all generated signals."""
        self.signals.clear()
        strategy_logger.debug(f"Cleared signals for strategy: {self.name}")
    
    def get_last_signal(self, symbol: str = None) -> Optional[TradingSignal]:
        """Get the last signal, optionally filtered by symbol."""
        if not self.signals:
            return None
        
        if symbol:
            symbol_signals = [s for s in self.signals if s.symbol == symbol]
            return symbol_signals[-1] if symbol_signals else None
        
        return self.signals[-1]
    
    def save_signals(self, filepath: str):
        """Save signals to CSV file."""
        if not self.signals:
            strategy_logger.warning(f"No signals to save for strategy: {self.name}")
            return
        
        df = pd.DataFrame([signal.to_dict() for signal in self.signals])
        df.to_csv(filepath, index=False)
        strategy_logger.info(f"Saved {len(self.signals)} signals to {filepath}")
    
    def load_signals(self, filepath: str):
        """Load signals from CSV file."""
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.signals.clear()
            for _, row in df.iterrows():
                signal = TradingSignal(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    signal_type=SignalType(row['signal_type']),
                    price=float(row['price']),
                    confidence=float(row['confidence']),
                    metadata=eval(row['metadata']) if pd.notna(row['metadata']) else {}
                )
                self.signals.append(signal)
            
            strategy_logger.info(f"Loaded {len(self.signals)} signals from {filepath}")
            
        except Exception as e:
            strategy_logger.error(f"Failed to load signals from {filepath}: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get basic performance metrics for the strategy."""
        if not self.signals:
            return {}
        
        total_signals = len(self.signals)
        buy_signals = len([s for s in self.signals if s.signal_type == SignalType.BUY])
        sell_signals = len([s for s in self.signals if s.signal_type == SignalType.SELL])
        
        avg_confidence = sum(s.confidence for s in self.signals) / total_signals
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'strategy_name': self.name
        }
