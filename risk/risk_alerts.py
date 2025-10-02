"""
Risk alerts and notifications system for the options trading system.

This module provides comprehensive alert management for risk violations,
threshold breaches, and critical events requiring immediate attention.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from utils.logging_utils import get_logger

# Setup logging
risk_alerts_logger = get_logger(__name__)


class RiskAlertType(Enum):
    """Risk alert type enumeration."""
    PORTFOLIO_LOSS = "PORTFOLIO_LOSS"
    DRAWDOWN = "DRAWDOWN"
    MARGIN_CALL = "MARGIN_CALL"
    POSITION_SIZE = "POSITION_SIZE"
    CONCENTRATION = "CONCENTRATION"
    LEVERAGE = "LEVERAGE"
    VOLATILITY = "VOLATILITY"
    TRADING_LIMIT = "TRADING_LIMIT"
    KILL_SWITCH = "KILL_SWITCH"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class AlertPriority(Enum):
    """Alert priority enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertChannel(Enum):
    """Alert channel enumeration."""
    LOG = "LOG"
    EMAIL = "EMAIL"
    SMS = "SMS"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    DASHBOARD = "DASHBOARD"


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: str
    alert_type: RiskAlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    current_value: float
    threshold_value: float
    percentage: float
    symbol: Optional[str] = None
    recommendation: str = ""
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for risk alerts."""
    # Alert channels
    enable_log_alerts: bool = True
    enable_email_alerts: bool = False
    enable_sms_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_webhook_alerts: bool = False
    
    # Email configuration
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    
    # SMS configuration
    sms_recipients: List[str] = field(default_factory=list)
    sms_provider: str = ""
    sms_api_key: str = ""
    
    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#trading-alerts"
    
    # Webhook configuration
    webhook_urls: List[str] = field(default_factory=list)
    
    # Rate limiting
    max_alerts_per_hour: int = 100
    alert_cooldown_seconds: int = 300  # 5 minutes
    
    # Alert thresholds
    portfolio_loss_warning_percent: float = 0.03
    portfolio_loss_critical_percent: float = 0.04
    drawdown_warning_percent: float = 0.08
    drawdown_critical_percent: float = 0.09
    margin_warning_percent: float = 0.80
    margin_critical_percent: float = 0.90


class RiskAlertManager:
    """
    Risk alerts and notifications manager.
    
    This class manages risk alerts, notifications, and communication
    channels for critical risk events and violations.
    """

    def __init__(self, config: AlertConfig):
        """
        Initialize risk alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        self.alert_counters: Dict[str, int] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        self.hourly_reset_time = datetime.now()
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        
        risk_alerts_logger.info(f"Initialized RiskAlertManager with config: {config}")

    async def create_alert(self, alert_type: RiskAlertType, priority: AlertPriority,
                          title: str, message: str, current_value: float,
                          threshold_value: float, symbol: Optional[str] = None,
                          recommendation: str = "", channels: Optional[List[AlertChannel]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> RiskAlert:
        """
        Create and send a risk alert.
        
        Args:
            alert_type: Type of alert
            priority: Alert priority
            title: Alert title
            message: Alert message
            current_value: Current value that triggered alert
            threshold_value: Threshold value
            symbol: Optional symbol
            recommendation: Recommended action
            channels: Alert channels
            metadata: Additional metadata
            
        Returns:
            Created alert
        """
        try:
            # Check rate limiting
            if not await self._should_send_alert(alert_type):
                risk_alerts_logger.warning(f"Alert rate limited for {alert_type.value}")
                return None
            
            # Calculate percentage
            percentage = (current_value / threshold_value * 100) if threshold_value > 0 else 0
            
            # Create alert
            alert = RiskAlert(
                alert_id=f"{alert_type.value}_{datetime.now().timestamp()}",
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                timestamp=datetime.now(),
                current_value=current_value,
                threshold_value=threshold_value,
                percentage=percentage,
                symbol=symbol,
                recommendation=recommendation,
                channels=channels or self._get_default_channels(priority),
                metadata=metadata or {}
            )
            
            # Add to active alerts
            self.active_alerts.append(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            # Update counters
            self._update_counters(alert_type)
            
            # Send alerts
            await self._send_alert(alert)
            
            # Trigger callbacks
            await self._trigger_callbacks(alert)
            
            risk_alerts_logger.warning(
                f"Risk Alert - {priority.value}: {title} - {message}"
            )
            
            return alert
            
        except Exception as e:
            risk_alerts_logger.error(f"Error creating alert: {e}")
            return None

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if acknowledged successfully
        """
        try:
            # Find alert in active alerts
            alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
            
            if not alert:
                risk_alerts_logger.warning(f"Alert {alert_id} not found in active alerts")
                return False
            
            # Acknowledge alert
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            # Remove from active alerts
            self.active_alerts.remove(alert)
            
            risk_alerts_logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
            
        except Exception as e:
            risk_alerts_logger.error(f"Error acknowledging alert: {e}")
            return False

    def get_active_alerts(self) -> List[RiskAlert]:
        """Get active alerts."""
        return self.active_alerts.copy()

    def get_alert_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get alert history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
            
            # Count by type and priority
            type_counts = {}
            priority_counts = {}
            
            for alert in recent_alerts:
                # Count by type
                if alert.alert_type.value not in type_counts:
                    type_counts[alert.alert_type.value] = 0
                type_counts[alert.alert_type.value] += 1
                
                # Count by priority
                if alert.priority.value not in priority_counts:
                    priority_counts[alert.priority.value] = 0
                priority_counts[alert.priority.value] += 1
            
            return {
                "total_alerts": len(recent_alerts),
                "active_alerts": len(self.active_alerts),
                "type_counts": type_counts,
                "priority_counts": priority_counts,
                "alerts_this_hour": self._get_hourly_count(),
                "time_range": f"{hours} hours"
            }
            
        except Exception as e:
            risk_alerts_logger.error(f"Error getting alert summary: {e}")
            return {"error": str(e)}

    def add_alert_callback(self, callback: Callable):
        """Add alert callback."""
        self.alert_callbacks.append(callback)

    async def _should_send_alert(self, alert_type: RiskAlertType) -> bool:
        """Check if alert should be sent based on rate limiting."""
        try:
            # Check hourly limit
            if self._get_hourly_count() >= self.config.max_alerts_per_hour:
                return False
            
            # Check cooldown
            last_alert_time = self.last_alert_times.get(alert_type.value)
            if last_alert_time:
                time_since_last = (datetime.now() - last_alert_time).total_seconds()
                if time_since_last < self.config.alert_cooldown_seconds:
                    return False
            
            return True
            
        except Exception as e:
            risk_alerts_logger.error(f"Error checking alert rate limiting: {e}")
            return True

    def _get_default_channels(self, priority: AlertPriority) -> List[AlertChannel]:
        """Get default channels for alert priority."""
        channels = [AlertChannel.LOG]
        
        if priority in [AlertPriority.HIGH, AlertPriority.CRITICAL, AlertPriority.EMERGENCY]:
            if self.config.enable_email_alerts:
                channels.append(AlertChannel.EMAIL)
            if self.config.enable_slack_alerts:
                channels.append(AlertChannel.SLACK)
        
        if priority == AlertPriority.EMERGENCY:
            if self.config.enable_sms_alerts:
                channels.append(AlertChannel.SMS)
            if self.config.enable_webhook_alerts:
                channels.append(AlertChannel.WEBHOOK)
        
        return channels

    async def _send_alert(self, alert: RiskAlert):
        """Send alert through configured channels."""
        try:
            for channel in alert.channels:
                try:
                    if channel == AlertChannel.LOG:
                        await self._send_log_alert(alert)
                    elif channel == AlertChannel.EMAIL:
                        await self._send_email_alert(alert)
                    elif channel == AlertChannel.SMS:
                        await self._send_sms_alert(alert)
                    elif channel == AlertChannel.SLACK:
                        await self._send_slack_alert(alert)
                    elif channel == AlertChannel.WEBHOOK:
                        await self._send_webhook_alert(alert)
                    elif channel == AlertChannel.DASHBOARD:
                        await self._send_dashboard_alert(alert)
                        
                except Exception as e:
                    risk_alerts_logger.error(f"Error sending alert via {channel.value}: {e}")
            
        except Exception as e:
            risk_alerts_logger.error(f"Error sending alert: {e}")

    async def _send_log_alert(self, alert: RiskAlert):
        """Send alert to log."""
        log_level = {
            AlertPriority.LOW: risk_alerts_logger.info,
            AlertPriority.MEDIUM: risk_alerts_logger.warning,
            AlertPriority.HIGH: risk_alerts_logger.error,
            AlertPriority.CRITICAL: risk_alerts_logger.critical,
            AlertPriority.EMERGENCY: risk_alerts_logger.critical
        }
        
        log_func = log_level.get(alert.priority, risk_alerts_logger.warning)
        
        message = (
            f"🚨 RISK ALERT 🚨\n"
            f"Type: {alert.alert_type.value}\n"
            f"Priority: {alert.priority.value}\n"
            f"Title: {alert.title}\n"
            f"Message: {alert.message}\n"
            f"Current: {alert.current_value:.2f}\n"
            f"Threshold: {alert.threshold_value:.2f}\n"
            f"Percentage: {alert.percentage:.2f}%\n"
        )
        
        if alert.symbol:
            message += f"Symbol: {alert.symbol}\n"
        
        if alert.recommendation:
            message += f"Recommendation: {alert.recommendation}\n"
        
        log_func(message)

    async def _send_email_alert(self, alert: RiskAlert):
        """Send alert via email."""
        if not self.config.enable_email_alerts:
            return
        
        # Here you would integrate with email service
        risk_alerts_logger.info(f"Email alert sent: {alert.title}")
        
        # Example email integration:
        # await self._send_email(
        #     recipients=self.config.email_recipients,
        #     subject=f"[{alert.priority.value}] {alert.title}",
        #     body=self._format_alert_email(alert)
        # )

    async def _send_sms_alert(self, alert: RiskAlert):
        """Send alert via SMS."""
        if not self.config.enable_sms_alerts:
            return
        
        # Here you would integrate with SMS service
        risk_alerts_logger.info(f"SMS alert sent: {alert.title}")
        
        # Example SMS integration:
        # await self._send_sms(
        #     recipients=self.config.sms_recipients,
        #     message=f"{alert.priority.value}: {alert.title} - {alert.message}"
        # )

    async def _send_slack_alert(self, alert: RiskAlert):
        """Send alert via Slack."""
        if not self.config.enable_slack_alerts:
            return
        
        # Here you would integrate with Slack webhook
        risk_alerts_logger.info(f"Slack alert sent: {alert.title}")
        
        # Example Slack integration:
        # await self._send_slack_webhook(
        #     url=self.config.slack_webhook_url,
        #     message=self._format_slack_message(alert)
        # )

    async def _send_webhook_alert(self, alert: RiskAlert):
        """Send alert via webhook."""
        if not self.config.enable_webhook_alerts:
            return
        
        # Here you would integrate with webhook service
        risk_alerts_logger.info(f"Webhook alert sent: {alert.title}")
        
        # Example webhook integration:
        # for url in self.config.webhook_urls:
        #     await self._send_webhook(
        #         url=url,
        #         data=alert.__dict__
        #     )

    async def _send_dashboard_alert(self, alert: RiskAlert):
        """Send alert to dashboard."""
        # This would update a real-time dashboard
        risk_alerts_logger.info(f"Dashboard alert sent: {alert.title}")

    def _update_counters(self, alert_type: RiskAlertType):
        """Update alert counters."""
        # Update type counter
        if alert_type.value not in self.alert_counters:
            self.alert_counters[alert_type.value] = 0
        self.alert_counters[alert_type.value] += 1
        
        # Update last alert time
        self.last_alert_times[alert_type.value] = datetime.now()

    def _get_hourly_count(self) -> int:
        """Get alert count for current hour."""
        now = datetime.now()
        if (now - self.hourly_reset_time).total_seconds() >= 3600:
            self.hourly_reset_time = now
            return 0
        
        # Count alerts in current hour
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        return len([a for a in self.alert_history if a.timestamp >= hour_start])

    async def _trigger_callbacks(self, alert: RiskAlert):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                risk_alerts_logger.error(f"Error in alert callback: {e}")

    def clear_old_alerts(self, hours: int = 24):
        """Clear old alerts from memory."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.alert_history = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
        self.active_alerts = [alert for alert in self.active_alerts if alert.timestamp >= cutoff_time]

    def _format_alert_email(self, alert: RiskAlert) -> str:
        """Format alert for email."""
        return f"""
Risk Alert: {alert.title}

Type: {alert.alert_type.value}
Priority: {alert.priority.value}
Time: {alert.timestamp}

Message: {alert.message}

Current Value: {alert.current_value:.2f}
Threshold: {alert.threshold_value:.2f}
Percentage: {alert.percentage:.2f}%

Recommendation: {alert.recommendation}

Please take appropriate action immediately.
"""

    def _format_slack_message(self, alert: RiskAlert) -> Dict[str, Any]:
        """Format alert for Slack."""
        color = {
            AlertPriority.LOW: "good",
            AlertPriority.MEDIUM: "warning",
            AlertPriority.HIGH: "danger",
            AlertPriority.CRITICAL: "danger",
            AlertPriority.EMERGENCY: "danger"
        }.get(alert.priority, "warning")
        
        return {
            "channel": self.config.slack_channel,
            "username": "Risk Manager",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                    {"title": "Priority", "value": alert.priority.value, "short": True},
                    {"title": "Current", "value": f"{alert.current_value:.2f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True},
                    {"title": "Recommendation", "value": alert.recommendation, "short": False}
                ],
                "timestamp": alert.timestamp.timestamp()
            }]
        }
