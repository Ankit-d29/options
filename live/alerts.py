"""
Alert system for live trading notifications.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from utils.logging_utils import get_logger

# Set up logging
alerts_logger = get_logger('alerts')


class AlertType(Enum):
    """Alert type enumeration."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRADE = "TRADE"
    SIGNAL = "SIGNAL"
    RISK = "RISK"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: AlertType
    title: str
    message: str
    timestamp: datetime
    severity: str = "INFO"
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'symbol': self.symbol,
            'metadata': self.metadata
        }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.console_enabled = True
        self.file_logging_enabled = True
        
        alerts_logger.info("Initialized AlertManager")
    
    def send_alert(self, alert: Alert):
        """Send an alert."""
        self.alerts.append(alert)
        
        # Console output
        if self.console_enabled:
            self._print_alert(alert)
        
        # File logging
        if self.file_logging_enabled:
            self._log_alert(alert)
        
        # Notify callbacks
        self._notify_callbacks(alert)
        
        alerts_logger.debug(f"Alert sent: {alert.title}")
    
    def create_alert(self, alert_type: AlertType, title: str, message: str,
                    severity: str = "INFO", symbol: str = None,
                    metadata: Dict[str, Any] = None) -> Alert:
        """Create and send an alert."""
        alert_id = f"{alert_type.value}_{datetime.now().timestamp()}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            severity=severity,
            symbol=symbol,
            metadata=metadata or {}
        )
        
        self.send_alert(alert)
        return alert
    
    def send_trade_alert(self, symbol: str, side: str, quantity: int, price: float):
        """Send trade execution alert."""
        title = f"Trade Executed: {side} {quantity} {symbol}"
        message = f"Executed {side} order for {quantity} shares of {symbol} at ${price:.2f}"
        
        self.create_alert(
            AlertType.TRADE,
            title,
            message,
            severity="INFO",
            symbol=symbol,
            metadata={
                'side': side,
                'quantity': quantity,
                'price': price
            }
        )
    
    def send_signal_alert(self, symbol: str, signal_type: str, price: float, confidence: float):
        """Send trading signal alert."""
        title = f"Trading Signal: {signal_type} {symbol}"
        message = f"Generated {signal_type} signal for {symbol} at ${price:.2f} (confidence: {confidence:.2f})"
        
        self.create_alert(
            AlertType.SIGNAL,
            title,
            message,
            severity="INFO",
            symbol=symbol,
            metadata={
                'signal_type': signal_type,
                'price': price,
                'confidence': confidence
            }
        )
    
    def send_risk_alert(self, risk_type: str, message: str, severity: str = "WARNING"):
        """Send risk management alert."""
        title = f"Risk Alert: {risk_type}"
        
        self.create_alert(
            AlertType.RISK,
            title,
            message,
            severity=severity,
            metadata={'risk_type': risk_type}
        )
    
    def send_portfolio_alert(self, message: str, portfolio_value: float, pnl: float):
        """Send portfolio update alert."""
        title = "Portfolio Update"
        full_message = f"{message} - Portfolio Value: ${portfolio_value:,.2f}, P&L: ${pnl:,.2f}"
        
        self.create_alert(
            AlertType.INFO,
            title,
            full_message,
            severity="INFO",
            metadata={
                'portfolio_value': portfolio_value,
                'pnl': pnl
            }
        )
    
    def send_error_alert(self, error_type: str, message: str, exception: Exception = None):
        """Send error alert."""
        title = f"Error: {error_type}"
        full_message = message
        if exception:
            full_message += f" - Exception: {str(exception)}"
        
        self.create_alert(
            AlertType.ERROR,
            title,
            full_message,
            severity="ERROR",
            metadata={
                'error_type': error_type,
                'exception': str(exception) if exception else None
            }
        )
    
    def _print_alert(self, alert: Alert):
        """Print alert to console with formatting."""
        # Choose color based on severity
        colors = {
            "INFO": "\033[94m",      # Blue
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
            "CRITICAL": "\033[95m",  # Magenta
            "TRADE": "\033[92m",     # Green
            "SIGNAL": "\033[96m",    # Cyan
            "RISK": "\033[93m"       # Yellow
        }
        reset_color = "\033[0m"
        
        color = colors.get(alert.severity, colors["INFO"])
        
        print(f"\n{color}=== {alert.title} ==={reset_color}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if alert.symbol:
            print(f"Symbol: {alert.symbol}")
        print(f"Severity: {alert.severity}")
        print(f"Message: {alert.message}")
        
        if alert.metadata:
            print("Metadata:")
            for key, value in alert.metadata.items():
                print(f"  {key}: {value}")
        
        print(f"{color}{'=' * (len(alert.title) + 8)}{reset_color}\n")
    
    def _log_alert(self, alert: Alert):
        """Log alert to file."""
        # This would typically write to a log file
        # For now, we'll use the logger
        log_level = {
            "INFO": alerts_logger.info,
            "WARNING": alerts_logger.warning,
            "ERROR": alerts_logger.error,
            "CRITICAL": alerts_logger.critical
        }.get(alert.severity, alerts_logger.info)
        
        log_message = f"{alert.title} - {alert.message}"
        if alert.symbol:
            log_message += f" [{alert.symbol}]"
        
        log_level(log_message)
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
        alerts_logger.info(f"Added alert callback. Total callbacks: {len(self.alert_callbacks)}")
    
    def remove_callback(self, callback: Callable[[Alert], None]):
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            alerts_logger.info(f"Removed alert callback. Total callbacks: {len(self.alert_callbacks)}")
    
    def _notify_callbacks(self, alert: Alert):
        """Notify all callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                alerts_logger.error(f"Error in alert callback: {e}")
    
    def get_recent_alerts(self, limit: int = 10, alert_type: AlertType = None) -> List[Alert]:
        """Get recent alerts, optionally filtered by type."""
        alerts = self.alerts
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts[-limit:] if alerts else []
    
    def get_alerts_by_symbol(self, symbol: str, limit: int = 10) -> List[Alert]:
        """Get alerts for a specific symbol."""
        symbol_alerts = [a for a in self.alerts if a.symbol == symbol]
        return symbol_alerts[-limit:] if symbol_alerts else []
    
    def get_alerts_by_severity(self, severity: str, limit: int = 10) -> List[Alert]:
        """Get alerts by severity level."""
        severity_alerts = [a for a in self.alerts if a.severity == severity]
        return severity_alerts[-limit:] if severity_alerts else []
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
        alerts_logger.info("Cleared all alerts")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        if not self.alerts:
            return {
                'total_alerts': 0,
                'alerts_by_type': {},
                'alerts_by_severity': {},
                'recent_activity': []
            }
        
        # Count by type
        alerts_by_type = {}
        for alert in self.alerts:
            alert_type = alert.alert_type.value
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # Count by severity
        alerts_by_severity = {}
        for alert in self.alerts:
            severity = alert.severity
            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
        
        # Recent activity (last 5 alerts)
        recent_activity = [alert.to_dict() for alert in self.alerts[-5:]]
        
        return {
            'total_alerts': len(self.alerts),
            'alerts_by_type': alerts_by_type,
            'alerts_by_severity': alerts_by_severity,
            'recent_activity': recent_activity
        }
    
    def enable_console_output(self, enabled: bool = True):
        """Enable/disable console output."""
        self.console_enabled = enabled
        alerts_logger.info(f"Console output {'enabled' if enabled else 'disabled'}")
    
    def enable_file_logging(self, enabled: bool = True):
        """Enable/disable file logging."""
        self.file_logging_enabled = enabled
        alerts_logger.info(f"File logging {'enabled' if enabled else 'disabled'}")
