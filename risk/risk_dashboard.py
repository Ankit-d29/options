"""
Risk monitoring dashboard for the options trading system.

This module provides a simple dashboard interface for monitoring
risk metrics, alerts, and system status.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .risk_engine import RiskMetrics, RiskViolation
from .risk_monitor import RiskAlert
from .margin_manager import MarginStatus
from utils.logging_utils import get_logger

# Setup logging
dashboard_logger = get_logger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for risk dashboard."""
    update_interval: float = 1.0  # seconds
    max_history_hours: int = 24
    enable_auto_refresh: bool = True
    show_alerts: bool = True
    show_metrics: bool = True
    show_positions: bool = True
    show_margin: bool = True


@dataclass
class DashboardData:
    """Dashboard data structure."""
    timestamp: datetime
    risk_metrics: Optional[RiskMetrics] = None
    margin_status: Optional[MarginStatus] = None
    active_alerts: List[RiskAlert] = field(default_factory=list)
    recent_violations: List[RiskViolation] = field(default_factory=list)
    system_status: Dict[str, Any] = field(default_factory=dict)


class RiskDashboard:
    """
    Risk monitoring dashboard.
    
    This class provides a simple dashboard interface for monitoring
    risk metrics, alerts, and system status.
    """

    def __init__(self, config: DashboardConfig):
        """
        Initialize risk dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.dashboard_data: List[DashboardData] = []
        self.current_data: Optional[DashboardData] = None
        
        dashboard_logger.info(f"Initialized RiskDashboard with config: {config}")

    def update_dashboard(self, risk_metrics: Optional[RiskMetrics] = None,
                        margin_status: Optional[MarginStatus] = None,
                        active_alerts: Optional[List[RiskAlert]] = None,
                        recent_violations: Optional[List[RiskViolation]] = None,
                        system_status: Optional[Dict[str, Any]] = None) -> DashboardData:
        """
        Update dashboard data.
        
        Args:
            risk_metrics: Current risk metrics
            margin_status: Current margin status
            active_alerts: Active alerts
            recent_violations: Recent violations
            system_status: System status
            
        Returns:
            Updated dashboard data
        """
        try:
            # Create new dashboard data
            dashboard_data = DashboardData(
                timestamp=datetime.now(),
                risk_metrics=risk_metrics,
                margin_status=margin_status,
                active_alerts=active_alerts or [],
                recent_violations=recent_violations or [],
                system_status=system_status or {}
            )
            
            # Update current data
            self.current_data = dashboard_data
            
            # Add to history
            self.dashboard_data.append(dashboard_data)
            
            # Trim history
            self._trim_history()
            
            return dashboard_data
            
        except Exception as e:
            dashboard_logger.error(f"Error updating dashboard: {e}")
            raise

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary."""
        try:
            if not self.current_data:
                return {"error": "No dashboard data available"}
            
            data = self.current_data
            
            # Risk metrics summary
            risk_summary = {}
            if data.risk_metrics:
                risk_summary = {
                    "portfolio_value": data.risk_metrics.portfolio_value,
                    "daily_pnl": data.risk_metrics.daily_pnl,
                    "daily_pnl_percent": data.risk_metrics.daily_pnl_percent,
                    "current_drawdown_percent": data.risk_metrics.current_drawdown_percent,
                    "risk_score": data.risk_metrics.risk_score,
                    "position_count": data.risk_metrics.position_count,
                    "largest_position_percent": data.risk_metrics.largest_position_percent,
                    "violations_count": len(data.risk_metrics.violations)
                }
            
            # Margin summary
            margin_summary = {}
            if data.margin_status:
                margin_summary = {
                    "margin_utilization_percent": data.margin_status.margin_utilization_percent,
                    "margin_status": data.margin_status.margin_status.value,
                    "excess_margin": data.margin_status.excess_margin,
                    "deficit_margin": data.margin_status.deficit_margin,
                    "leverage_ratio": data.margin_status.leverage_ratio
                }
            
            # Alerts summary
            alerts_summary = {
                "active_alerts_count": len(data.active_alerts),
                "critical_alerts": len([a for a in data.active_alerts if a.priority.value == "CRITICAL"]),
                "high_alerts": len([a for a in data.active_alerts if a.priority.value == "HIGH"]),
                "recent_violations_count": len(data.recent_violations)
            }
            
            # System status
            system_summary = {
                "timestamp": data.timestamp.isoformat(),
                "status": "ACTIVE",  # Simplified
                "uptime": "N/A"  # Would calculate from system start time
            }
            
            return {
                "timestamp": data.timestamp.isoformat(),
                "risk_summary": risk_summary,
                "margin_summary": margin_summary,
                "alerts_summary": alerts_summary,
                "system_summary": system_summary
            }
            
        except Exception as e:
            dashboard_logger.error(f"Error getting dashboard summary: {e}")
            return {"error": str(e)}

    def get_risk_metrics_chart_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get risk metrics chart data."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [d for d in self.dashboard_data if d.timestamp >= cutoff_time]
            
            if not recent_data:
                return {"error": "No data available"}
            
            # Extract time series data
            timestamps = [d.timestamp.isoformat() for d in recent_data]
            
            # Risk metrics
            portfolio_values = [d.risk_metrics.portfolio_value for d in recent_data if d.risk_metrics]
            daily_pnl = [d.risk_metrics.daily_pnl for d in recent_data if d.risk_metrics]
            risk_scores = [d.risk_metrics.risk_score for d in recent_data if d.risk_metrics]
            drawdowns = [d.risk_metrics.current_drawdown_percent for d in recent_data if d.risk_metrics]
            
            # Margin metrics
            margin_utilization = [d.margin_status.margin_utilization_percent for d in recent_data if d.margin_status]
            leverage_ratios = [d.margin_status.leverage_ratio for d in recent_data if d.margin_status]
            
            return {
                "timestamps": timestamps,
                "portfolio_values": portfolio_values,
                "daily_pnl": daily_pnl,
                "risk_scores": risk_scores,
                "drawdowns": drawdowns,
                "margin_utilization": margin_utilization,
                "leverage_ratios": leverage_ratios,
                "data_points": len(recent_data)
            }
            
        except Exception as e:
            dashboard_logger.error(f"Error getting chart data: {e}")
            return {"error": str(e)}

    def get_alerts_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts history for dashboard."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [d for d in self.dashboard_data if d.timestamp >= cutoff_time]
            
            alerts_history = []
            for data in recent_data:
                for alert in data.active_alerts:
                    alerts_history.append({
                        "timestamp": alert.timestamp.isoformat(),
                        "type": alert.alert_type.value,
                        "priority": alert.priority.value,
                        "title": alert.title,
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "percentage": alert.percentage,
                        "symbol": alert.symbol,
                        "recommendation": alert.recommendation
                    })
            
            # Sort by timestamp (newest first)
            alerts_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return alerts_history
            
        except Exception as e:
            dashboard_logger.error(f"Error getting alerts history: {e}")
            return []

    def get_violations_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get violations history for dashboard."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [d for d in self.dashboard_data if d.timestamp >= cutoff_time]
            
            violations_history = []
            for data in recent_data:
                for violation in data.recent_violations:
                    violations_history.append({
                        "timestamp": violation.timestamp.isoformat(),
                        "violation_type": violation.violation_type.value,
                        "risk_level": violation.risk_level.value,
                        "message": violation.message,
                        "current_value": violation.current_value,
                        "limit_value": violation.limit_value,
                        "violation_percent": violation.violation_percent,
                        "recommended_action": violation.recommended_action,
                        "symbol": violation.symbol
                    })
            
            # Sort by timestamp (newest first)
            violations_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return violations_history
            
        except Exception as e:
            dashboard_logger.error(f"Error getting violations history: {e}")
            return []

    def generate_dashboard_report(self, hours: int = 24) -> str:
        """Generate a text-based dashboard report."""
        try:
            summary = self.get_dashboard_summary()
            if "error" in summary:
                return f"Dashboard Error: {summary['error']}"
            
            report = []
            report.append("=" * 60)
            report.append("RISK MANAGEMENT DASHBOARD")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Risk Summary
            if "risk_summary" in summary and summary["risk_summary"]:
                risk = summary["risk_summary"]
                report.append("📊 RISK SUMMARY")
                report.append("-" * 20)
                report.append(f"Portfolio Value: ${risk.get('portfolio_value', 0):,.2f}")
                report.append(f"Daily P&L: ${risk.get('daily_pnl', 0):,.2f} ({risk.get('daily_pnl_percent', 0):.2f}%)")
                report.append(f"Current Drawdown: {risk.get('current_drawdown_percent', 0):.2f}%")
                report.append(f"Risk Score: {risk.get('risk_score', 0):.1f}/100")
                report.append(f"Positions: {risk.get('position_count', 0)}")
                report.append(f"Largest Position: {risk.get('largest_position_percent', 0):.2f}%")
                report.append(f"Violations: {risk.get('violations_count', 0)}")
                report.append("")
            
            # Margin Summary
            if "margin_summary" in summary and summary["margin_summary"]:
                margin = summary["margin_summary"]
                report.append("💰 MARGIN SUMMARY")
                report.append("-" * 20)
                report.append(f"Margin Utilization: {margin.get('margin_utilization_percent', 0):.2f}%")
                report.append(f"Margin Status: {margin.get('margin_status', 'UNKNOWN')}")
                report.append(f"Excess Margin: ${margin.get('excess_margin', 0):,.2f}")
                report.append(f"Deficit Margin: ${margin.get('deficit_margin', 0):,.2f}")
                report.append(f"Leverage Ratio: {margin.get('leverage_ratio', 0):.2f}x")
                report.append("")
            
            # Alerts Summary
            if "alerts_summary" in summary:
                alerts = summary["alerts_summary"]
                report.append("🚨 ALERTS SUMMARY")
                report.append("-" * 20)
                report.append(f"Active Alerts: {alerts.get('active_alerts_count', 0)}")
                report.append(f"Critical Alerts: {alerts.get('critical_alerts', 0)}")
                report.append(f"High Alerts: {alerts.get('high_alerts', 0)}")
                report.append(f"Recent Violations: {alerts.get('recent_violations_count', 0)}")
                report.append("")
            
            # System Status
            if "system_summary" in summary:
                system = summary["system_summary"]
                report.append("⚙️ SYSTEM STATUS")
                report.append("-" * 20)
                report.append(f"Status: {system.get('status', 'UNKNOWN')}")
                report.append(f"Last Update: {system.get('timestamp', 'UNKNOWN')}")
                report.append("")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            dashboard_logger.error(f"Error generating dashboard report: {e}")
            return f"Error generating report: {e}"

    def _trim_history(self):
        """Trim dashboard history to configured limit."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.max_history_hours)
            self.dashboard_data = [d for d in self.dashboard_data if d.timestamp >= cutoff_time]
            
        except Exception as e:
            dashboard_logger.error(f"Error trimming history: {e}")

    def get_current_data(self) -> Optional[DashboardData]:
        """Get current dashboard data."""
        return self.current_data

    def get_data_history(self, hours: int = 24) -> List[DashboardData]:
        """Get dashboard data history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [d for d in self.dashboard_data if d.timestamp >= cutoff_time]

    def clear_history(self):
        """Clear dashboard history."""
        self.dashboard_data.clear()
        self.current_data = None
        dashboard_logger.info("Dashboard history cleared")
