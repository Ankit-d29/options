"""
Trade execution and position management for backtesting.
"""
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd

from utils.logging_utils import get_logger
from utils.config import config
from strategies.base_strategy import TradingSignal, SignalType

# Set up logging
trade_logger = get_logger('trade_executor')


class TradeStatus(Enum):
    """Trade execution status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    position_type: str  # 'LONG' or 'SHORT'
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_unrealized_pnl(self, current_price: float, commission: float = 0.0):
        """Update unrealized P&L based on current price."""
        if self.position_type == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity - commission
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity - commission
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_price,
            'position_type': self.position_type,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.get_total_pnl(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    position_type: str  # 'LONG' or 'SHORT'
    pnl: float
    commission: float
    slippage: float
    duration_minutes: int
    signal_entry: Optional[TradingSignal] = None
    signal_exit: Optional[TradingSignal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_return_percent(self) -> float:
        """Calculate return percentage."""
        if self.position_type == 'LONG':
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:  # SHORT
            return (self.entry_price - self.exit_price) / self.entry_price * 100
    
    def is_winning_trade(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'position_type': self.position_type,
            'pnl': self.pnl,
            'commission': self.commission,
            'slippage': self.slippage,
            'duration_minutes': self.duration_minutes,
            'return_percent': self.get_return_percent(),
            'is_winning': self.is_winning_trade(),
            'signal_entry_type': self.signal_entry.signal_type.value if self.signal_entry else None,
            'signal_exit_type': self.signal_exit.signal_type.value if self.signal_exit else None,
            'metadata': self.metadata
        }


class TradeExecutor:
    """Handles trade execution and position management for backtesting."""
    
    def __init__(self, initial_capital: float = None, commission_rate: float = None, 
                 slippage_rate: float = None, max_positions: int = None):
        self.initial_capital = initial_capital or config.get('backtesting.initial_capital', 100000)
        self.commission_rate = commission_rate or config.get('backtesting.commission', 0.001)
        self.slippage_rate = slippage_rate or config.get('backtesting.slippage', 0.0005)
        self.max_positions = max_positions or config.get('trading.max_positions', 10)
        
        # State tracking
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.current_time: Optional[datetime] = None
        
        trade_logger.info(f"Initialized TradeExecutor with capital: ${self.initial_capital:,.2f}")
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                position.update_unrealized_pnl(current_prices[symbol], self.commission_rate)
            portfolio_value += position.get_total_pnl()
        
        return portfolio_value
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check if position already exists for this symbol
        if symbol in self.positions:
            return False
        
        return True
    
    def calculate_position_size(self, signal: TradingSignal, current_price: float, 
                              risk_per_trade: float = 0.02) -> int:
        """Calculate position size based on risk management."""
        # Simple position sizing based on risk per trade
        risk_amount = self.get_portfolio_value() * risk_per_trade
        
        # Calculate position size based on price volatility (simplified)
        # In practice, this would use ATR or other volatility measures
        volatility_factor = 0.01  # 1% price volatility assumption
        
        max_loss_per_share = current_price * volatility_factor
        position_size = int(risk_amount / max_loss_per_share)
        
        # Ensure we have enough cash
        required_cash = position_size * current_price * (1 + self.commission_rate)
        if required_cash > self.cash:
            position_size = int(self.cash / (current_price * (1 + self.commission_rate)))
        
        return max(1, position_size)  # At least 1 share
    
    def execute_entry(self, signal: TradingSignal, current_price: float, 
                     current_time: datetime, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """Execute a trade entry."""
        if not self.can_open_position(signal.symbol):
            trade_logger.warning(f"Cannot open position for {signal.symbol}: limits exceeded")
            return False
        
        # Determine position type based on signal
        if signal.signal_type == SignalType.BUY:
            position_type = 'LONG'
        elif signal.signal_type == SignalType.SELL:
            position_type = 'SHORT'
        else:
            trade_logger.warning(f"Invalid signal type for entry: {signal.signal_type}")
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(signal, current_price)
        if quantity <= 0:
            trade_logger.warning(f"Invalid position size calculated: {quantity}")
            return False
        
        # Apply slippage
        if position_type == 'LONG':
            execution_price = current_price * (1 + self.slippage_rate)
        else:  # SHORT
            execution_price = current_price * (1 - self.slippage_rate)
        
        # Calculate costs
        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate
        
        # Check if we have enough cash
        if position_type == 'LONG' and trade_value + commission > self.cash:
            trade_logger.warning(f"Insufficient cash for {signal.symbol}: ${self.cash:.2f} < ${trade_value + commission:.2f}")
            return False
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            quantity=quantity,
            entry_price=execution_price,
            entry_time=current_time,
            position_type=position_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={'entry_signal': signal.to_dict()}
        )
        
        # Update cash
        if position_type == 'LONG':
            self.cash -= (trade_value + commission)
        else:  # SHORT - we receive cash but need margin (simplified)
            self.cash += (trade_value - commission)
        
        # Store position
        self.positions[signal.symbol] = position
        
        trade_logger.info(f"Opened {position_type} position: {signal.symbol} x{quantity} @ ${execution_price:.2f}")
        
        return True
    
    def execute_exit(self, signal: TradingSignal, current_price: float, 
                    current_time: datetime) -> Optional[Trade]:
        """Execute a trade exit."""
        if signal.symbol not in self.positions:
            trade_logger.warning(f"No position found for {signal.symbol}")
            return None
        
        position = self.positions[signal.symbol]
        
        # Apply slippage
        if position.position_type == 'LONG':
            execution_price = current_price * (1 - self.slippage_rate)
        else:  # SHORT
            execution_price = current_price * (1 + self.slippage_rate)
        
        # Calculate P&L
        trade_value = position.quantity * execution_price
        commission = trade_value * self.commission_rate
        
        if position.position_type == 'LONG':
            pnl = (execution_price - position.entry_price) * position.quantity - commission
            self.cash += trade_value - commission
        else:  # SHORT
            pnl = (position.entry_price - execution_price) * position.quantity - commission
            self.cash += (position.quantity * position.entry_price) - commission
        
        # Calculate trade duration
        duration_minutes = int((current_time - position.entry_time).total_seconds() / 60)
        
        # Create trade record
        trade = Trade(
            symbol=signal.symbol,
            entry_time=position.entry_time,
            exit_time=current_time,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            position_type=position.position_type,
            pnl=pnl,
            commission=commission,
            slippage=abs(execution_price - current_price),
            duration_minutes=duration_minutes,
            signal_entry=position.metadata.get('entry_signal'),
            signal_exit=signal,
            metadata={'exit_signal': signal.to_dict()}
        )
        
        # Update position realized P&L
        position.realized_pnl = pnl
        
        # Remove position
        del self.positions[signal.symbol]
        
        # Store trade
        self.trades.append(trade)
        
        trade_logger.info(f"Closed {position.position_type} position: {signal.symbol} @ ${execution_price:.2f}, P&L: ${pnl:.2f}")
        
        return trade
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float], 
                                   current_time: datetime) -> List[Trade]:
        """Check and execute stop loss/take profit orders."""
        closed_trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if position.stop_loss is not None:
                if position.position_type == 'LONG' and current_price <= position.stop_loss:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif position.position_type == 'SHORT' and current_price >= position.stop_loss:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
            
            # Check take profit
            if position.take_profit is not None and not should_exit:
                if position.position_type == 'LONG' and current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                elif position.position_type == 'SHORT' and current_price <= position.take_profit:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
            
            if should_exit:
                # Create exit signal
                exit_signal = TradingSignal(
                    timestamp=current_time,
                    symbol=symbol,
                    signal_type=SignalType.SELL if position.position_type == 'LONG' else SignalType.BUY,
                    price=current_price,
                    confidence=1.0,
                    metadata={'exit_reason': exit_reason}
                )
                
                trade = self.execute_exit(exit_signal, current_price, current_time)
                if trade:
                    closed_trades.append(trade)
        
        return closed_trades
    
    def update_equity_curve(self, current_time: datetime, current_prices: Dict[str, float] = None):
        """Update equity curve with current portfolio value."""
        portfolio_value = self.get_portfolio_value(current_prices)
        
        self.equity_curve.append({
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_count': len(self.positions),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values())
        })
        
        self.current_time = current_time
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current open positions."""
        return self.positions.copy()
    
    def get_trades(self) -> List[Trade]:
        """Get all completed trades."""
        return self.trades.copy()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def reset(self):
        """Reset executor to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.current_time = None
        trade_logger.info("TradeExecutor reset to initial state")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'max_pnl': 0.0,
                'min_pnl': 0.0
            }
        
        winning_trades = [t for t in self.trades if t.is_winning_trade()]
        losing_trades = [t for t in self.trades if not t.is_winning_trade()]
        
        total_pnl = sum(t.pnl for t in self.trades)
        pnls = [t.pnl for t in self.trades]
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades * 100,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_trades,
            'max_pnl': max(pnls),
            'min_pnl': min(pnls),
            'avg_win': sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        }
