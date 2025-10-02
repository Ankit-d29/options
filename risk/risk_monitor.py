"""
Real-time risk monitoring system for the options trading system.

This module provides continuous monitoring of risk metrics and real-time
alerts for risk violations and threshold breaches.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .risk_engine import RiskMetrics, RiskViolation, RiskLevel, RiskViolationType
from utils.logging_utils import get_logger

# Setup logging
risk_monitor_logger = get_logger(__name__)


class MonitorStatus(Enum):
    """Monitor status enumeration."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: str
    timestamp: datetime
    alert_level: AlertLevel
    alert_type: str
    title: str
    message: str
    current_value: float
    threshold_value: float
    percentage: float
    recommendation: str
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorConfig:
    """Configuration for risk monitoring."""
    update_interval: float = 1.0  # seconds
    alert_cooldown: float = 300.0  # 5 minutes
    max_alerts_per_hour: int = 100
    enable_real_time_alerts: bool = True
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    alert_recipients: List[str] = field(default_factory=list)
    
    # Thresholds for alerts
    portfolio_loss_warning_percent: float = 0.03  # 3%
    portfolio_loss_critical_percent: float = 0.04  # 4%
    drawdown_warning_percent: float = 0.08  # 8%
    drawdown_critical_percent: float = 0.09  # 9%
    margin_warning_percent: float = 0.80  # 80%
    margin_critical_percent: float = 0.90  # 90%
    position_size_warning_percent: float = 0.08  # 8%
    position_size_critical_percent: float = 0.09  # 9%


class RiskMonitor:
    """
    Real-time risk monitoring system.
    
    This class provides continuous monitoring of risk metrics and generates
    real-time alerts for risk violations and threshold breaches.
    """

    def __init__(self, config: MonitorConfig):
        """
        Initialize risk monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.status = MonitorStatus.STOPPED
        self.monitoring_task: Optional[asyncio.Task] = None
        self.current_metrics: Optional[RiskMetrics] = None
        
        # Alert tracking
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.alerts_this_hour = 0
        self.hour_start = datetime.now()
        
        # Callbacks
        self.metrics_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        risk_monitor_logger.info(f"Initialized RiskMonitor with config: {config}")

    async def start(self) -> bool:
        """
        Start risk monitoring.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.status == MonitorStatus.ACTIVE:
                risk_monitor_logger.warning("Monitor already active")
                return True
            
            self.status = MonitorStatus.ACTIVE
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            risk_monitor_logger.info("Risk monitoring started")
            return True
            
        except Exception as e:
            risk_monitor_logger.error(f"Failed to start risk monitoring: {e}")
            self.status = MonitorStatus.ERROR
            return False

    async def stop(self) -> bool:
        """
        Stop risk monitoring.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            self.status = MonitorStatus.STOPPED
            
            # Cancel monitoring task
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            risk_monitor_logger.info("Risk monitoring stopped")
            return True
            
        except Exception as e:
            risk_monitor_logger.error(f"Failed to stop risk monitoring: {e}")
            return False

    async def pause(self) -> bool:
        """
        Pause risk monitoring.
        
        Returns:
            True if paused successfully, False otherwise
        """
        try:
            if self.status != MonitorStatus.ACTIVE:
                risk_monitor_logger.warning("Monitor not active, cannot pause")
                return False
            
            self.status = MonitorStatus.PAUSED
            risk_monitor_logger.info("Risk monitoring paused")
            return True
            
        except Exception as e:
            risk_monitor_logger.error(f"Failed to pause risk monitoring: {e}")
            return False

    async def resume(self) -> bool:
        """
        Resume risk monitoring.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        try:
            if self.status != MonitorStatus.PAUSED:
                risk_monitor_logger.warning("Monitor not paused, cannot resume")
                return False
            
            self.status = MonitorStatus.ACTIVE
            risk_monitor_logger.info("Risk monitoring resumed")
            return True
            
        except Exception as e:
            risk_monitor_logger.error(f"Failed to resume risk monitoring: {e}")
            return False

    async def update_metrics(self, metrics: RiskMetrics):
        """
        Update risk metrics.
        
        Args:
            metrics: Current risk metrics
        """
        try:
            self.current_metrics = metrics
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            # Trigger callbacks
            await self._trigger_metrics_callbacks(metrics)
            
        except Exception as e:
            risk_monitor_logger.error(f"Error updating metrics: {e}")

    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics."""
        return self.current_metrics

    def get_active_alerts(self) -> List[RiskAlert]:
        """Get active alerts."""
        return self.active_alerts

    def get_alert_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get alert history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

    def get_monitor_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        return {
            "status": self.status.value,
            "active": self.status == MonitorStatus.ACTIVE,
            "current_metrics": self.current_metrics.__dict__ if self.current_metrics else None,
            "active_alerts_count": len(self.active_alerts),
            "alerts_this_hour": self.alerts_this_hour,
            "total_alerts": len(self.alert_history),
            "uptime_seconds": (datetime.now() - self.hour_start).total_seconds()
        }

    def add_metrics_callback(self, callback: Callable):
        """Add metrics update callback."""
        self.metrics_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable):
        """Add alert callback."""
        self.alert_callbacks.append(callback)

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.status == MonitorStatus.ACTIVE:
                # Check alert rate limiting
                await self._check_alert_rate_limits()
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval)
                
        except asyncio.CancelledError:
            risk_monitor_logger.info("Monitoring loop cancelled")
        except Exception as e:
            risk_monitor_logger.error(f"Error in monitoring loop: {e}")
            self.status = MonitorStatus.ERROR

    async def _check_alerts(self, metrics: RiskMetrics):
        """Check for risk alerts."""
        try:
            alerts = []
            
            # Check portfolio loss alerts
            alerts.extend(await self._check_portfolio_loss_alerts(metrics))
            
            # Check drawdown alerts
            alerts.extend(await self._check_drawdown_alerts(metrics))
            
            # Check margin alerts
            alerts.extend(await self._check_margin_alerts(metrics))
            
            # Check position size alerts
            alerts.extend(await self._check_position_size_alerts(metrics))
            
            # Check risk score alerts
            alerts.extend(await self._check_risk_score_alerts(metrics))
            
            # Process alerts
            for alert in alerts:
                await self._process_alert(alert)
            
        except Exception as e:
            risk_monitor_logger.error(f"Error checking alerts: {e}")

    async def _check_portfolio_loss_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """Check portfolio loss alerts."""
        alerts = []
        
        if metrics.portfolio_loss_percent >= self.config.portfolio_loss_critical_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"portfolio_loss_critical_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.CRITICAL,
                alert_type="PORTFOLIO_LOSS",
                title="Critical Portfolio Loss",
                message=f"Portfolio loss {metrics.portfolio_loss_percent:.2f}% exceeds critical threshold",
                current_value=metrics.portfolio_loss_percent,
                threshold_value=self.config.portfolio_loss_critical_percent * 100,
                percentage=(metrics.portfolio_loss_percent / (self.config.portfolio_loss_critical_percent * 100)) * 100,
                recommendation="Consider closing positions or reducing exposure"
            ))
        elif metrics.portfolio_loss_percent >= self.config.portfolio_loss_warning_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"portfolio_loss_warning_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.WARNING,
                alert_type="PORTFOLIO_LOSS",
                title="Portfolio Loss Warning",
                message=f"Portfolio loss {metrics.portfolio_loss_percent:.2f}% exceeds warning threshold",
                current_value=metrics.portfolio_loss_percent,
                threshold_value=self.config.portfolio_loss_warning_percent * 100,
                percentage=(metrics.portfolio_loss_percent / (self.config.portfolio_loss_warning_percent * 100)) * 100,
                recommendation="Monitor positions closely"
            ))
        
        return alerts

    async def _check_drawdown_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """Check drawdown alerts."""
        alerts = []
        
        if metrics.current_drawdown_percent >= self.config.drawdown_critical_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"drawdown_critical_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.CRITICAL,
                alert_type="DRAWDOWN",
                title="Critical Drawdown",
                message=f"Current drawdown {metrics.current_drawdown_percent:.2f}% exceeds critical threshold",
                current_value=metrics.current_drawdown_percent,
                threshold_value=self.config.drawdown_critical_percent * 100,
                percentage=(metrics.current_drawdown_percent / (self.config.drawdown_critical_percent * 100)) * 100,
                recommendation="Consider reducing positions or stopping trading"
            ))
        elif metrics.current_drawdown_percent >= self.config.drawdown_warning_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"drawdown_warning_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.WARNING,
                alert_type="DRAWDOWN",
                title="Drawdown Warning",
                message=f"Current drawdown {metrics.current_drawdown_percent:.2f}% exceeds warning threshold",
                current_value=metrics.current_drawdown_percent,
                threshold_value=self.config.drawdown_warning_percent * 100,
                percentage=(metrics.current_drawdown_percent / (self.config.drawdown_warning_percent * 100)) * 100,
                recommendation="Monitor portfolio performance"
            ))
        
        return alerts

    async def _check_margin_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """Check margin alerts."""
        alerts = []
        
        if metrics.margin_utilization_percent >= self.config.margin_critical_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"margin_critical_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.CRITICAL,
                alert_type="MARGIN",
                title="Critical Margin Usage",
                message=f"Margin utilization {metrics.margin_utilization_percent:.2f}% exceeds critical threshold",
                current_value=metrics.margin_utilization_percent,
                threshold_value=self.config.margin_critical_percent * 100,
                percentage=(metrics.margin_utilization_percent / (self.config.margin_critical_percent * 100)) * 100,
                recommendation="Reduce position sizes or add capital"
            ))
        elif metrics.margin_utilization_percent >= self.config.margin_warning_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"margin_warning_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.WARNING,
                alert_type="MARGIN",
                title="Margin Usage Warning",
                message=f"Margin utilization {metrics.margin_utilization_percent:.2f}% exceeds warning threshold",
                current_value=metrics.margin_utilization_percent,
                threshold_value=self.config.margin_warning_percent * 100,
                percentage=(metrics.margin_utilization_percent / (self.config.margin_warning_percent * 100)) * 100,
                recommendation="Monitor margin usage"
            ))
        
        return alerts

    async def _check_position_size_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """Check position size alerts."""
        alerts = []
        
        if metrics.largest_position_percent >= self.config.position_size_critical_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"position_size_critical_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.WARNING,
                alert_type="POSITION_SIZE",
                title="Large Position Warning",
                message=f"Largest position {metrics.largest_position_percent:.2f}% exceeds critical threshold",
                current_value=metrics.largest_position_percent,
                threshold_value=self.config.position_size_critical_percent * 100,
                percentage=(metrics.largest_position_percent / (self.config.position_size_critical_percent * 100)) * 100,
                recommendation="Consider reducing position size"
            ))
        elif metrics.largest_position_percent >= self.config.position_size_warning_percent * 100:
            alerts.append(RiskAlert(
                alert_id=f"position_size_warning_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.INFO,
                alert_type="POSITION_SIZE",
                title="Position Size Notice",
                message=f"Largest position {metrics.largest_position_percent:.2f}% exceeds warning threshold",
                current_value=metrics.largest_position_percent,
                threshold_value=self.config.position_size_warning_percent * 100,
                percentage=(metrics.largest_position_percent / (self.config.position_size_warning_percent * 100)) * 100,
                recommendation="Monitor position size"
            ))
        
        return alerts

    async def _check_risk_score_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """Check risk score alerts."""
        alerts = []
        
        if metrics.risk_score >= 90:
            alerts.append(RiskAlert(
                alert_id=f"risk_score_critical_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.CRITICAL,
                alert_type="RISK_SCORE",
                title="Critical Risk Score",
                message=f"Overall risk score {metrics.risk_score:.1f} indicates very high risk",
                current_value=metrics.risk_score,
                threshold_value=90.0,
                percentage=metrics.risk_score,
                recommendation="Immediately review and reduce risk exposure"
            ))
        elif metrics.risk_score >= 75:
            alerts.append(RiskAlert(
                alert_id=f"risk_score_warning_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                alert_level=AlertLevel.WARNING,
                alert_type="RISK_SCORE",
                title="High Risk Score",
                message=f"Overall risk score {metrics.risk_score:.1f} indicates high risk",
                current_value=metrics.risk_score,
                threshold_value=75.0,
                percentage=metrics.risk_score,
                recommendation="Review risk exposure and consider reducing positions"
            ))
        
        return alerts

    async def _process_alert(self, alert: RiskAlert):
        """Process a risk alert."""
        try:
            # Check cooldown
            if not await self._should_send_alert(alert):
                return
            
            # Add to active alerts
            self.active_alerts.append(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            # Update rate limiting
            self.alerts_this_hour += 1
            self.last_alert_times[alert.alert_type] = datetime.now()
            
            # Log alert
            risk_monitor_logger.warning(
                f"Risk Alert - {alert.alert_level.value}: {alert.title} - {alert.message}"
            )
            
            # Trigger callbacks
            await self._trigger_alert_callbacks(alert)
            
        except Exception as e:
            risk_monitor_logger.error(f"Error processing alert: {e}")

    async def _should_send_alert(self, alert: RiskAlert) -> bool:
        """Check if alert should be sent (cooldown and rate limiting)."""
        # Check rate limiting
        if self.alerts_this_hour >= self.config.max_alerts_per_hour:
            return False
        
        # Check cooldown
        last_alert_time = self.last_alert_times.get(alert.alert_type)
        if last_alert_time:
            time_since_last = (datetime.now() - last_alert_time).total_seconds()
            if time_since_last < self.config.alert_cooldown:
                return False
        
        return True

    async def _check_alert_rate_limits(self):
        """Check and reset alert rate limits."""
        now = datetime.now()
        if (now - self.hour_start).total_seconds() >= 3600:  # 1 hour
            self.hour_start = now
            self.alerts_this_hour = 0

    async def _trigger_metrics_callbacks(self, metrics: RiskMetrics):
        """Trigger metrics callbacks."""
        for callback in self.metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                risk_monitor_logger.error(f"Error in metrics callback: {e}")

    async def _trigger_alert_callbacks(self, alert: RiskAlert):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                risk_monitor_logger.error(f"Error in alert callback: {e}")

    def clear_old_alerts(self, hours: int = 24):
        """Clear old alerts from memory."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.alert_history = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
        self.active_alerts = [alert for alert in self.active_alerts if alert.timestamp >= cutoff_time]
