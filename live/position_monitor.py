"""
Position monitoring and risk management for live trading.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from utils.logging_utils import get_logger
from utils.config import config
from .paper_trader import PaperTrader, PaperPosition, PaperOrder
from .alerts import AlertManager, AlertType

# Set up logging
position_monitor_logger = get_logger('position_monitor')


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_portfolio_loss_percent: float = 0.05  # 5% max portfolio loss
    max_position_size_percent: float = 0.10   # 10% max position size
    max_daily_trades: int = 50
    max_drawdown_percent: float = 0.10        # 10% max drawdown
    stop_loss_percent: float = 0.05           # 5% stop loss
    take_profit_percent: float = 0.10         # 10% take profit
    max_open_positions: int = 5


@dataclass
class PositionAlert:
    """Position alert information."""
    alert_id: str
    position_symbol: str
    alert_type: str
    message: str
    timestamp: datetime
    severity: str = "INFO"  # INFO, WARNING, CRITICAL
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionMonitor:
    """Monitors positions and enforces risk management rules."""
    
    def __init__(self, paper_trader: PaperTrader, alert_manager: AlertManager = None):
        self.paper_trader = paper_trader
        self.alert_manager = alert_manager or AlertManager()
        
        # Risk limits
        self.risk_limits = RiskLimits()
        
        # Monitoring state
        self.initial_portfolio_value = paper_trader.initial_capital
        self.peak_portfolio_value = paper_trader.initial_capital
        self.daily_trades_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Alert history
        self.position_alerts: List[PositionAlert] = []
        
        # Callbacks
        self.alert_callbacks: List[Callable[[PositionAlert], None]] = []
        
        position_monitor_logger.info("Initialized PositionMonitor")
    
    def update_risk_limits(self, limits: RiskLimits):
        """Update risk management limits."""
        self.risk_limits = limits
        position_monitor_logger.info("Updated risk limits")
    
    def check_portfolio_risk(self) -> List[PositionAlert]:
        """Check overall portfolio risk."""
        alerts = []
        
        current_value = self.paper_trader.get_portfolio_value()
        portfolio_loss = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        # Check max portfolio loss
        if portfolio_loss <= -self.risk_limits.max_portfolio_loss_percent:
            alert = PositionAlert(
                alert_id=f"portfolio_loss_{datetime.now().timestamp()}",
                position_symbol="PORTFOLIO",
                alert_type="MAX_PORTFOLIO_LOSS",
                message=f"Portfolio loss exceeded limit: {portfolio_loss:.2%} > -{self.risk_limits.max_portfolio_loss_percent:.2%}",
                timestamp=datetime.now(),
                severity="CRITICAL",
                metadata={
                    'current_value': current_value,
                    'initial_value': self.initial_portfolio_value,
                    'loss_percent': portfolio_loss,
                    'limit': -self.risk_limits.max_portfolio_loss_percent
                }
            )
            alerts.append(alert)
        
        # Check max drawdown
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        if drawdown >= self.risk_limits.max_drawdown_percent:
            alert = PositionAlert(
                alert_id=f"max_drawdown_{datetime.now().timestamp()}",
                position_symbol="PORTFOLIO",
                alert_type="MAX_DRAWDOWN",
                message=f"Maximum drawdown exceeded: {drawdown:.2%} > {self.risk_limits.max_drawdown_percent:.2%}",
                timestamp=datetime.now(),
                severity="WARNING",
                metadata={
                    'current_value': current_value,
                    'peak_value': self.peak_portfolio_value,
                    'drawdown_percent': drawdown,
                    'limit': self.risk_limits.max_drawdown_percent
                }
            )
            alerts.append(alert)
        
        # Check daily trade limit
        self._reset_daily_counters_if_needed()
        if self.daily_trades_count >= self.risk_limits.max_daily_trades:
            alert = PositionAlert(
                alert_id=f"daily_trades_{datetime.now().timestamp()}",
                position_symbol="PORTFOLIO",
                alert_type="MAX_DAILY_TRADES",
                message=f"Daily trade limit reached: {self.daily_trades_count} >= {self.risk_limits.max_daily_trades}",
                timestamp=datetime.now(),
                severity="WARNING",
                metadata={
                    'daily_trades': self.daily_trades_count,
                    'limit': self.risk_limits.max_daily_trades
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def check_position_risk(self, position: PaperPosition) -> List[PositionAlert]:
        """Check risk for a specific position."""
        alerts = []
        
        # Check position size
        portfolio_value = self.paper_trader.get_portfolio_value()
        position_value = position.get_market_value()
        position_size_percent = position_value / portfolio_value
        
        if position_size_percent > self.risk_limits.max_position_size_percent:
            alert = PositionAlert(
                alert_id=f"position_size_{position.symbol}_{datetime.now().timestamp()}",
                position_symbol=position.symbol,
                alert_type="MAX_POSITION_SIZE",
                message=f"Position size exceeded limit: {position_size_percent:.2%} > {self.risk_limits.max_position_size_percent:.2%}",
                timestamp=datetime.now(),
                severity="WARNING",
                metadata={
                    'position_value': position_value,
                    'portfolio_value': portfolio_value,
                    'position_size_percent': position_size_percent,
                    'limit': self.risk_limits.max_position_size_percent
                }
            )
            alerts.append(alert)
        
        # Check stop loss
        if position.quantity > 0:  # Long position
            stop_loss_price = position.average_price * (1 - self.risk_limits.stop_loss_percent)
            if position.current_price <= stop_loss_price:
                alert = PositionAlert(
                    alert_id=f"stop_loss_{position.symbol}_{datetime.now().timestamp()}",
                    position_symbol=position.symbol,
                    alert_type="STOP_LOSS",
                    message=f"Stop loss triggered: {position.current_price:.2f} <= {stop_loss_price:.2f}",
                    timestamp=datetime.now(),
                    severity="WARNING",
                    metadata={
                        'current_price': position.current_price,
                        'stop_loss_price': stop_loss_price,
                        'average_price': position.average_price,
                        'unrealized_pnl': position.unrealized_pnl
                    }
                )
                alerts.append(alert)
        
        elif position.quantity < 0:  # Short position
            stop_loss_price = position.average_price * (1 + self.risk_limits.stop_loss_percent)
            if position.current_price >= stop_loss_price:
                alert = PositionAlert(
                    alert_id=f"stop_loss_{position.symbol}_{datetime.now().timestamp()}",
                    position_symbol=position.symbol,
                    alert_type="STOP_LOSS",
                    message=f"Stop loss triggered: {position.current_price:.2f} >= {stop_loss_price:.2f}",
                    timestamp=datetime.now(),
                    severity="WARNING",
                    metadata={
                        'current_price': position.current_price,
                        'stop_loss_price': stop_loss_price,
                        'average_price': position.average_price,
                        'unrealized_pnl': position.unrealized_pnl
                    }
                )
                alerts.append(alert)
        
        # Check take profit
        if position.quantity > 0:  # Long position
            take_profit_price = position.average_price * (1 + self.risk_limits.take_profit_percent)
            if position.current_price >= take_profit_price:
                alert = PositionAlert(
                    alert_id=f"take_profit_{position.symbol}_{datetime.now().timestamp()}",
                    position_symbol=position.symbol,
                    alert_type="TAKE_PROFIT",
                    message=f"Take profit target reached: {position.current_price:.2f} >= {take_profit_price:.2f}",
                    timestamp=datetime.now(),
                    severity="INFO",
                    metadata={
                        'current_price': position.current_price,
                        'take_profit_price': take_profit_price,
                        'average_price': position.average_price,
                        'unrealized_pnl': position.unrealized_pnl
                    }
                )
                alerts.append(alert)
        
        elif position.quantity < 0:  # Short position
            take_profit_price = position.average_price * (1 - self.risk_limits.take_profit_percent)
            if position.current_price <= take_profit_price:
                alert = PositionAlert(
                    alert_id=f"take_profit_{position.symbol}_{datetime.now().timestamp()}",
                    position_symbol=position.symbol,
                    alert_type="TAKE_PROFIT",
                    message=f"Take profit target reached: {position.current_price:.2f} <= {take_profit_price:.2f}",
                    timestamp=datetime.now(),
                    severity="INFO",
                    metadata={
                        'current_price': position.current_price,
                        'take_profit_price': take_profit_price,
                        'average_price': position.average_price,
                        'unrealized_pnl': position.unrealized_pnl
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def check_all_risks(self) -> List[PositionAlert]:
        """Check all risk parameters."""
        alerts = []
        
        # Check portfolio risk
        portfolio_alerts = self.check_portfolio_risk()
        alerts.extend(portfolio_alerts)
        
        # Check individual position risks
        for position in self.paper_trader.positions.values():
            position_alerts = self.check_position_risk(position)
            alerts.extend(position_alerts)
        
        # Store alerts
        self.position_alerts.extend(alerts)
        
        # Notify callbacks
        for alert in alerts:
            self._notify_alert_callbacks(alert)
            self.alert_manager.send_alert(alert)
        
        return alerts
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        current_value = self.paper_trader.get_portfolio_value()
        portfolio_loss = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        # Count positions by risk level
        high_risk_positions = []
        for position in self.paper_trader.positions.values():
            portfolio_value = self.paper_trader.get_portfolio_value()
            position_size_percent = position.get_market_value() / portfolio_value
            
            if position_size_percent > self.risk_limits.max_position_size_percent * 0.8:
                high_risk_positions.append(position.symbol)
        
        return {
            'portfolio_value': current_value,
            'initial_value': self.initial_portfolio_value,
            'portfolio_loss_percent': portfolio_loss,
            'peak_value': self.peak_portfolio_value,
            'drawdown_percent': drawdown,
            'daily_trades': self.daily_trades_count,
            'total_positions': len(self.paper_trader.positions),
            'high_risk_positions': high_risk_positions,
            'risk_limits': {
                'max_portfolio_loss': self.risk_limits.max_portfolio_loss_percent,
                'max_position_size': self.risk_limits.max_position_size_percent,
                'max_daily_trades': self.risk_limits.max_daily_trades,
                'max_drawdown': self.risk_limits.max_drawdown_percent,
                'stop_loss': self.risk_limits.stop_loss_percent,
                'take_profit': self.risk_limits.take_profit_percent,
                'max_positions': self.risk_limits.max_open_positions
            },
            'alerts_count': len(self.position_alerts),
            'recent_alerts': [alert for alert in self.position_alerts[-10:]]
        }
    
    def on_trade_executed(self, order: PaperOrder):
        """Handle trade execution event."""
        if order.is_filled():
            self.daily_trades_count += 1
            position_monitor_logger.debug(f"Trade executed, daily count: {self.daily_trades_count}")
    
    def _reset_daily_counters_if_needed(self):
        """Reset daily counters if it's a new day."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades_count = 0
            self.last_reset_date = current_date
            position_monitor_logger.info("Reset daily trade counter")
    
    def add_alert_callback(self, callback: Callable[[PositionAlert], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def _notify_alert_callbacks(self, alert: PositionAlert):
        """Notify alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                position_monitor_logger.error(f"Error in alert callback: {e}")
    
    def get_recent_alerts(self, limit: int = 10) -> List[PositionAlert]:
        """Get recent alerts."""
        return self.position_alerts[-limit:] if self.position_alerts else []
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.position_alerts.clear()
        position_monitor_logger.info("Cleared all alerts")
