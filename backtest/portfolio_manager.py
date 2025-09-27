"""
Portfolio management for backtesting.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from utils.logging_utils import get_logger
from .trade_executor import TradeExecutor, Position, Trade
from strategies.base_strategy import TradingSignal, SignalType

# Set up logging
portfolio_logger = get_logger('portfolio_manager')


@dataclass
class PortfolioConfig:
    """Portfolio configuration parameters."""
    initial_capital: float = 100000
    max_positions: int = 10
    max_position_size_percent: float = 0.1  # 10% of portfolio per position
    risk_per_trade_percent: float = 0.02  # 2% risk per trade
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    stop_loss_percent: float = 0.05  # 5% stop loss
    take_profit_percent: float = 0.10  # 10% take profit
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly


class PortfolioManager:
    """Manages portfolio allocation and risk for backtesting."""
    
    def __init__(self, config: PortfolioConfig = None):
        """
        Initialize portfolio manager.
        
        Args:
            config: Portfolio configuration parameters
        """
        self.config = config or PortfolioConfig()
        self.trade_executor = TradeExecutor(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            max_positions=self.config.max_positions
        )
        
        self.allocation_history: List[Dict[str, Any]] = []
        self.rebalance_dates: List[datetime] = []
        self.risk_free_rate = 0.02  # Default 2% risk-free rate
        
        portfolio_logger.info(f"Initialized PortfolioManager with capital: ${self.config.initial_capital:,.2f}")
    
    def should_rebalance(self, current_time: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if not self.rebalance_dates:
            return True
        
        last_rebalance = self.rebalance_dates[-1]
        
        if self.config.rebalance_frequency == 'daily':
            return current_time.date() > last_rebalance.date()
        elif self.config.rebalance_frequency == 'weekly':
            return (current_time - last_rebalance).days >= 7
        elif self.config.rebalance_frequency == 'monthly':
            return current_time.month != last_rebalance.month or current_time.year != last_rebalance.year
        
        return False
    
    def calculate_position_sizes(self, signals: List[TradingSignal], 
                               current_prices: Dict[str, float]) -> Dict[str, int]:
        """Calculate optimal position sizes based on risk management."""
        if not signals:
            return {}
        
        portfolio_value = self.trade_executor.get_portfolio_value(current_prices)
        available_capital = portfolio_value * 0.95  # Keep 5% cash buffer
        
        position_sizes = {}
        
        # Equal weight allocation among signals
        num_signals = len(signals)
        capital_per_signal = available_capital / num_signals
        
        for signal in signals:
            if signal.symbol in current_prices:
                current_price = current_prices[signal.symbol]
                
                # Calculate position size based on risk per trade
                risk_amount = capital_per_signal * self.config.risk_per_trade_percent
                
                # Simple volatility assumption (in practice, use ATR or historical volatility)
                volatility = current_price * 0.02  # 2% volatility assumption
                position_size = int(risk_amount / volatility)
                
                # Apply maximum position size limit
                max_position_value = portfolio_value * self.config.max_position_size_percent
                max_position_size = int(max_position_value / current_price)
                
                position_size = min(position_size, max_position_size)
                position_sizes[signal.symbol] = max(1, position_size)
        
        return position_sizes
    
    def execute_signals(self, signals: List[TradingSignal], current_prices: Dict[str, float], 
                       current_time: datetime) -> List[Trade]:
        """Execute trading signals with portfolio management."""
        executed_trades = []
        
        # Filter signals for symbols we can trade
        valid_signals = []
        for signal in signals:
            if signal.symbol in current_prices:
                # Check if we can open new position or need to close existing one
                if signal.symbol in self.trade_executor.positions:
                    # Close existing position if signal is opposite
                    existing_position = self.trade_executor.positions[signal.symbol]
                    should_close = False
                    
                    if existing_position.position_type == 'LONG' and signal.signal_type == SignalType.SELL:
                        should_close = True
                    elif existing_position.position_type == 'SHORT' and signal.signal_type == SignalType.BUY:
                        should_close = True
                    
                    if should_close:
                        trade = self.trade_executor.execute_exit(
                            signal, current_prices[signal.symbol], current_time
                        )
                        if trade:
                            executed_trades.append(trade)
                else:
                    # New position signal
                    valid_signals.append(signal)
        
        # Execute new positions
        for signal in valid_signals:
            if signal.symbol in current_prices:
                current_price = current_prices[signal.symbol]
                
                # Calculate stop loss and take profit
                stop_loss = None
                take_profit = None
                
                if signal.signal_type == SignalType.BUY:
                    stop_loss = current_price * (1 - self.config.stop_loss_percent)
                    take_profit = current_price * (1 + self.config.take_profit_percent)
                elif signal.signal_type == SignalType.SELL:
                    stop_loss = current_price * (1 + self.config.stop_loss_percent)
                    take_profit = current_price * (1 - self.config.take_profit_percent)
                
                success = self.trade_executor.execute_entry(
                    signal, current_price, current_time, stop_loss, take_profit
                )
                
                if success:
                    portfolio_logger.info(f"Executed {signal.signal_type.value} signal for {signal.symbol}")
        
        return executed_trades
    
    def update_portfolio(self, current_time: datetime, current_prices: Dict[str, float]):
        """Update portfolio state and check for stop losses/take profits."""
        # Check stop losses and take profits
        closed_trades = self.trade_executor.check_stop_loss_take_profit(current_prices, current_time)
        
        # Update equity curve
        self.trade_executor.update_equity_curve(current_time, current_prices)
        
        # Record allocation
        self._record_allocation(current_time, current_prices)
        
        return closed_trades
    
    def _record_allocation(self, current_time: datetime, current_prices: Dict[str, float]):
        """Record current portfolio allocation."""
        portfolio_value = self.trade_executor.get_portfolio_value(current_prices)
        cash_percent = (self.trade_executor.cash / portfolio_value) * 100
        
        position_allocation = {}
        for symbol, position in self.trade_executor.positions.items():
            if symbol in current_prices:
                position_value = position.quantity * current_prices[symbol]
                position_percent = (position_value / portfolio_value) * 100
                position_allocation[symbol] = {
                    'quantity': position.quantity,
                    'value': position_value,
                    'percent': position_percent,
                    'pnl': position.get_total_pnl(),
                    'position_type': position.position_type
                }
        
        allocation_record = {
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'cash': self.trade_executor.cash,
            'cash_percent': cash_percent,
            'positions_count': len(self.trade_executor.positions),
            'position_allocation': position_allocation
        }
        
        self.allocation_history.append(allocation_record)
    
    def rebalance_portfolio(self, current_time: datetime, current_prices: Dict[str, float]):
        """Rebalance portfolio based on strategy."""
        if not self.should_rebalance(current_time):
            return []
        
        portfolio_logger.info(f"Rebalancing portfolio at {current_time}")
        
        # For now, implement simple rebalancing
        # In practice, this could implement more sophisticated allocation strategies
        
        rebalance_trades = []
        
        # Close all positions for rebalancing
        for symbol in list(self.trade_executor.positions.keys()):
            position = self.trade_executor.positions[symbol]
            
            # Create exit signal
            exit_signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                signal_type=SignalType.SELL if position.position_type == 'LONG' else SignalType.BUY,
                price=current_prices.get(symbol, position.entry_price),
                confidence=1.0,
                metadata={'rebalance': True}
            )
            
            trade = self.trade_executor.execute_exit(
                exit_signal, current_prices.get(symbol, position.entry_price), current_time
            )
            
            if trade:
                rebalance_trades.append(trade)
        
        self.rebalance_dates.append(current_time)
        
        return rebalance_trades
    
    def get_portfolio_summary(self, current_time: datetime = None) -> Dict[str, Any]:
        """Get current portfolio summary."""
        if current_time is None:
            current_time = datetime.now()
        
        portfolio_value = self.trade_executor.get_portfolio_value()
        
        summary = {
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'cash': self.trade_executor.cash,
            'cash_percent': (self.trade_executor.cash / portfolio_value) * 100,
            'positions_count': len(self.trade_executor.positions),
            'total_pnl': portfolio_value - self.config.initial_capital,
            'return_percent': ((portfolio_value - self.config.initial_capital) / self.config.initial_capital) * 100,
            'positions': {}
        }
        
        for symbol, position in self.trade_executor.positions.items():
            summary['positions'][symbol] = {
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'position_type': position.position_type,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.get_total_pnl()
            }
        
        return summary
    
    def get_allocation_history(self) -> pd.DataFrame:
        """Get portfolio allocation history as DataFrame."""
        if not self.allocation_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.allocation_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.trade_executor.reset()
        self.allocation_history.clear()
        self.rebalance_dates.clear()
        portfolio_logger.info("PortfolioManager reset to initial state")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        trades = self.trade_executor.get_trades()
        equity_curve = self.trade_executor.get_equity_curve()
        
        if not trades:
            return {
                'total_trades': 0,
                'portfolio_value': self.config.initial_capital,
                'total_pnl': 0,
                'return_percent': 0
            }
        
        total_pnl = sum(trade.pnl for trade in trades)
        winning_trades = [t for t in trades if t.is_winning_trade()]
        win_rate = len(winning_trades) / len(trades) * 100
        
        current_portfolio_value = self.trade_executor.get_portfolio_value()
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(trades) - len(winning_trades),
            'win_rate': win_rate,
            'portfolio_value': current_portfolio_value,
            'total_pnl': total_pnl,
            'return_percent': ((current_portfolio_value - self.config.initial_capital) / self.config.initial_capital) * 100,
            'avg_trade_duration': sum(t.duration_minutes for t in trades) / len(trades),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self._calculate_sharpe_ratio(equity_curve)
        }
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        
        portfolio_values = equity_curve['portfolio_value']
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = equity_curve['portfolio_value'].pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Assuming daily returns
        return np.sqrt(252) * excess_returns.mean() / returns.std()
