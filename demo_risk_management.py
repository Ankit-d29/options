"""
Demo script for Phase 6 - Risk Management & Monitoring.

This script demonstrates the risk management functionality including:
- Risk engine and risk metrics
- Kill switch functionality
- Risk monitoring and alerts
- Position limits and controls
- Margin management
- Risk dashboard
"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.append('.')

# Import risk management components
from risk.risk_engine import RiskEngine, RiskConfig
from risk.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchReason
from risk.risk_monitor import RiskMonitor, MonitorConfig
from risk.position_limits import PositionLimits, PositionLimitConfig
from risk.margin_manager import MarginManager, MarginConfig
from risk.risk_alerts import RiskAlertManager, AlertConfig, RiskAlertType, AlertPriority
from risk.risk_dashboard import RiskDashboard, DashboardConfig

# Import broker and strategy components
from broker.base_broker import BrokerOrder, BrokerPosition, BrokerQuote, OrderSide, OrderType, ProductType, Variety
from strategies.base_strategy import TradingSignal, SignalType
from utils.logging_utils import get_logger

# Setup logging
logger = get_logger(__name__)


def create_sample_positions() -> List[BrokerPosition]:
    """Create sample broker positions for testing."""
    positions = []
    
    # NIFTY position
    positions.append(BrokerPosition(
        symbol="NIFTY",
        quantity=100,
        average_price=Decimal('18000'),
        last_price=Decimal('18100'),
        day_change=Decimal('100'),
        day_change_percent=Decimal('0.56'),
        unrealized_pnl=Decimal('10000'),
        realized_pnl=Decimal('5000')
    ))
    
    # BANKNIFTY position
    positions.append(BrokerPosition(
        symbol="BANKNIFTY",
        quantity=50,
        average_price=Decimal('42000'),
        last_price=Decimal('41800'),
        day_change=Decimal('-200'),
        day_change_percent=Decimal('-0.48'),
        unrealized_pnl=Decimal('-10000'),
        realized_pnl=Decimal('2000')
    ))
    
    return positions


def create_sample_signals() -> List[TradingSignal]:
    """Create sample trading signals for testing."""
    signals = []
    
    # Buy signal
    signals.append(TradingSignal(
        timestamp=datetime.now(),
        symbol="NIFTY",
        signal_type=SignalType.BUY,
        price=18100.0,
        confidence=0.85
    ))
    
    # Sell signal
    signals.append(TradingSignal(
        timestamp=datetime.now(),
        symbol="BANKNIFTY",
        signal_type=SignalType.SELL,
        price=41800.0,
        confidence=0.75
    ))
    
    return signals


async def demo_risk_engine():
    """Demonstrate risk engine functionality."""
    print("\n" + "="*60)
    print("RISK ENGINE DEMONSTRATION")
    print("="*60)
    
    # Initialize risk engine
    config = RiskConfig(
        max_portfolio_loss_percent=0.05,
        max_daily_loss_percent=0.02,
        max_drawdown_percent=0.10,
        max_position_size_percent=0.10,
        max_concentration_percent=0.25,
        max_open_positions=10,
        max_daily_trades=50,
        max_trades_per_hour=10
    )
    
    risk_engine = RiskEngine(config)
    
    # Start risk engine
    await risk_engine.start()
    print("✓ Risk engine started successfully")
    
    # Create sample data
    positions = create_sample_positions()
    signals = create_sample_signals()
    
    # Test signal risk checking
    print("\n--- Signal Risk Checking ---")
    for signal in signals:
        result = await risk_engine.check_signal_risk(
            signal=signal,
            current_portfolio_value=1000000.0,
            positions=positions
        )
        
        print(f"Signal: {signal.signal_type.value} {signal.symbol} @ {signal.price}")
        print(f"  Allowed: {result['allowed']}")
        print(f"  Reason: {result.get('reason', 'N/A')}")
        print(f"  Risk Score: {result.get('risk_score', 'N/A')}")
        
    # Test portfolio metrics update
    print("\n--- Portfolio Metrics Update ---")
    metrics = await risk_engine.update_portfolio_metrics(
        portfolio_value=1000000.0,
        available_margin=800000.0,
        used_margin=200000.0,
        positions=positions,
        daily_trades=15
    )
    
    print(f"Portfolio Value: ₹{metrics.portfolio_value:,.2f}")
    print(f"Available Margin: ₹{metrics.available_margin:,.2f}")
    print(f"Used Margin: ₹{metrics.used_margin:,.2f}")
    print(f"Margin Utilization: {metrics.margin_utilization_percent:.2f}%")
    print(f"Position Count: {metrics.position_count}")
    print(f"Daily Trades: {metrics.daily_trades}")
    print(f"Risk Score: {metrics.risk_score:.2f}")
    
    # Test kill switch trigger
    print("\n--- Kill Switch Test ---")
    kill_result = await risk_engine.trigger_kill_switch("Demo kill switch test")
    print(f"Kill switch triggered: {kill_result}")
    print(f"Kill switch active: {risk_engine.kill_switch_triggered}")
    
    # Stop risk engine
    await risk_engine.stop()
    print("✓ Risk engine stopped successfully")


async def demo_kill_switch():
    """Demonstrate kill switch functionality."""
    print("\n" + "="*60)
    print("KILL SWITCH DEMONSTRATION")
    print("="*60)
    
    # Mock broker for demo
    class MockBroker:
        async def get_orders(self):
            return []
        
        async def get_positions(self):
            return []
        
        async def cancel_order(self, order_id):
            return True
        
        async def close_position(self, symbol):
            return True
    
    mock_broker = MockBroker()
    
    # Initialize kill switch
    config = KillSwitchConfig(
        enable_auto_close=True,
        close_all_positions=True,
        cancel_pending_orders=True,
        send_notifications=True
    )
    
    kill_switch = KillSwitch(config, mock_broker)
    
    print("✓ Kill switch initialized")
    
    # Test kill switch triggering
    print("\n--- Triggering Kill Switch ---")
    trigger_result = await kill_switch.trigger(
        reason=KillSwitchReason.MANUAL,
        triggered_by="demo_user",
        message="Demo kill switch activation",
        portfolio_value=1000000.0
    )
    
    print(f"Kill switch triggered: {trigger_result}")
    print(f"Kill switch active: {kill_switch.is_triggered()}")
    print(f"Current event: {kill_switch.current_event.reason.value if kill_switch.current_event else 'None'}")
    
    # Test kill switch reset
    print("\n--- Resetting Kill Switch ---")
    reset_result = await kill_switch.reset("demo_user")
    print(f"Kill switch reset: {reset_result}")
    print(f"Kill switch active: {kill_switch.is_triggered()}")


async def demo_risk_monitor():
    """Demonstrate risk monitor functionality."""
    print("\n" + "="*60)
    print("RISK MONITOR DEMONSTRATION")
    print("="*60)
    
    # Initialize risk monitor
    config = MonitorConfig(
        update_interval=1.0,
        alert_cooldown=300.0,
        max_alerts_per_hour=100
    )
    
    monitor = RiskMonitor(config)
    
    # Start monitor
    await monitor.start()
    print("✓ Risk monitor started")
    
    # Create sample metrics
    from risk.risk_engine import RiskMetrics
    
    metrics = RiskMetrics(
        timestamp=datetime.now(),
        portfolio_value=1000000.0,
        available_margin=800000.0,
        used_margin=200000.0,
        margin_utilization_percent=25.0,
        portfolio_loss_percent=2.0,
        daily_pnl=-20000.0,
        daily_pnl_percent=-2.0,
        max_drawdown_percent=10.0,
        current_drawdown_percent=3.0,
        position_count=2,
        largest_position_percent=8.0,
        concentration_risk_percent=20.0,
        daily_trades=15,
        hourly_trades=3,
        volatility_percent=2.5,
        risk_score=45.0,
        violations=[]
    )
    
    # Update metrics
    await monitor.update_metrics(metrics)
    print("✓ Metrics updated")
    
    # Test monitor controls
    print("\n--- Monitor Controls ---")
    
    # Pause monitor
    pause_result = await monitor.pause()
    print(f"Monitor paused: {pause_result}")
    print(f"Monitor status: {monitor.status.value}")
    
    # Resume monitor
    resume_result = await monitor.resume()
    print(f"Monitor resumed: {resume_result}")
    print(f"Monitor status: {monitor.status.value}")
    
    # Stop monitor
    await monitor.stop()
    print("✓ Risk monitor stopped")


async def demo_position_limits():
    """Demonstrate position limits functionality."""
    print("\n" + "="*60)
    print("POSITION LIMITS DEMONSTRATION")
    print("="*60)
    
    # Initialize position limits
    config = PositionLimitConfig(
        max_position_size_percent=0.10,
        max_position_value=50000.0,
        max_position_quantity=1000,
        max_concentration_percent=0.25,
        max_positions_per_symbol=3,
        max_total_positions=10
    )
    
    position_limits = PositionLimits(config)
    print("✓ Position limits initialized")
    
    # Create sample signal
    signal = TradingSignal(
        timestamp=datetime.now(),
        symbol="NIFTY",
        signal_type=SignalType.BUY,
        price=18000.0,
        confidence=0.8
    )
    
    positions = create_sample_positions()
    
    # Test position size calculation
    print("\n--- Position Size Calculation ---")
    size_result = position_limits.calculate_position_size(
        signal=signal,
        portfolio_value=1000000.0,
        available_cash=500000.0,
        positions=positions,
        volatility=0.02
    )
    
    print(f"Signal: {signal.signal_type.value} {signal.symbol} @ {signal.price}")
    print(f"Recommended Quantity: {size_result.recommended_quantity}")
    print(f"Recommended Value: ₹{size_result.recommended_value:,.2f}")
    print(f"Sizing Method: {size_result.sizing_method}")
    print(f"Risk Amount: ₹{size_result.risk_amount:,.2f}")
    
    # Test position limits checking
    print("\n--- Position Limits Checking ---")
    limits = position_limits.check_position_limits(
        signal=signal,
        portfolio_value=1000000.0,
        positions=positions
    )
    
    for limit in limits:
        print(f"Limit: {limit.limit_type.value}")
        print(f"  Current: {limit.current_value}")
        print(f"  Limit: {limit.limit_value}")
        print(f"  Violated: {limit.is_violated}")
        print(f"  Recommendation: {limit.recommendation}")


async def demo_margin_manager():
    """Demonstrate margin manager functionality."""
    print("\n" + "="*60)
    print("MARGIN MANAGER DEMONSTRATION")
    print("="*60)
    
    # Initialize margin manager
    config = MarginConfig(
        initial_margin_percent=0.20,
        maintenance_margin_percent=0.15,
        max_margin_utilization_percent=0.80,
        warning_margin_percent=0.70
    )
    
    margin_manager = MarginManager(config)
    print("✓ Margin manager initialized")
    
    positions = create_sample_positions()
    
    # Test margin requirements calculation
    print("\n--- Margin Requirements Calculation ---")
    margin_status = margin_manager.calculate_margin_requirements(
        positions=positions,
        available_capital=1000000.0,
        portfolio_value=1000000.0
    )
    
    print(f"Total Margin Required: ₹{margin_status.total_margin_required:,.2f}")
    print(f"Total Margin Available: ₹{margin_status.total_margin_available:,.2f}")
    print(f"Margin Utilization: {margin_status.margin_utilization_percent:.2f}%")
    print(f"Status: {margin_status.margin_status.value}")
    
    # Test order margin checking
    print("\n--- Order Margin Checking ---")
    order = BrokerOrder(
        order_id="test_order_001",
        symbol="NIFTY",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        price=Decimal('18000'),
        product=ProductType.MIS,
        variety=Variety.REGULAR
    )
    
    order_result = margin_manager.check_order_margin(
        order=order,
        available_capital=1000000.0,
        current_positions=positions
    )
    
    print(f"Order: {order.side.value} {order.quantity} {order.symbol}")
    print(f"Allowed: {order_result['allowed']}")
    print(f"Order Margin: ₹{order_result['order_margin']:,.2f}")
    print(f"Current Margin: ₹{order_result['current_margin']:,.2f}")
    print(f"Excess Margin: ₹{order_result['excess_margin']:,.2f}")


async def demo_risk_alerts():
    """Demonstrate risk alert manager functionality."""
    print("\n" + "="*60)
    print("RISK ALERTS DEMONSTRATION")
    print("="*60)
    
    # Initialize alert manager
    config = AlertConfig(
        enable_log_alerts=True,
        max_alerts_per_hour=100,
        alert_cooldown_seconds=300
    )
    
    alert_manager = RiskAlertManager(config)
    print("✓ Risk alert manager initialized")
    
    # Create various alerts
    print("\n--- Creating Risk Alerts ---")
    
    alerts = []
    
    # Portfolio loss alert
    alert1 = await alert_manager.create_alert(
        alert_type=RiskAlertType.PORTFOLIO_LOSS,
        priority=AlertPriority.HIGH,
        title="Portfolio Loss Alert",
        message="Portfolio loss exceeds 3% threshold",
        current_value=5.0,
        threshold_value=3.0,
        symbol="PORTFOLIO",
        recommendation="Reduce positions or exit trades"
    )
    alerts.append(alert1)
    print(f"✓ Created alert: {alert1.title}")
    
    # Margin utilization alert
    alert2 = await alert_manager.create_alert(
        alert_type=RiskAlertType.MARGIN_CALL,
        priority=AlertPriority.MEDIUM,
        title="Margin Utilization Warning",
        message="Margin utilization approaching limit",
        current_value=75.0,
        threshold_value=70.0,
        symbol="MARGIN",
        recommendation="Reduce position sizes"
    )
    alerts.append(alert2)
    print(f"✓ Created alert: {alert2.title}")
    
    # Position concentration alert
    alert3 = await alert_manager.create_alert(
        alert_type=RiskAlertType.CONCENTRATION,
        priority=AlertPriority.MEDIUM,
        title="Position Concentration Alert",
        message="Single position exceeds 20% of portfolio",
        current_value=25.0,
        threshold_value=20.0,
        symbol="NIFTY",
        recommendation="Diversify positions"
    )
    alerts.append(alert3)
    print(f"✓ Created alert: {alert3.title}")
    
    # Test alert acknowledgment
    print("\n--- Alert Acknowledgment ---")
    ack_result = await alert_manager.acknowledge_alert(
        alert_id=alert1.alert_id,
        acknowledged_by="demo_user"
    )
    print(f"Alert acknowledged: {ack_result}")
    
    # Get alert summary
    print("\n--- Alert Summary ---")
    summary = alert_manager.get_alert_summary()
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Active Alerts: {summary['active_alerts']}")
    print(f"High Priority: {summary['high_priority_alerts']}")
    print(f"Medium Priority: {summary['medium_priority_alerts']}")
    print(f"Low Priority: {summary['low_priority_alerts']}")


async def demo_risk_dashboard():
    """Demonstrate risk dashboard functionality."""
    print("\n" + "="*60)
    print("RISK DASHBOARD DEMONSTRATION")
    print("="*60)
    
    # Initialize dashboard
    config = DashboardConfig(
        update_interval=1.0,
        max_history_hours=24
    )
    
    dashboard = RiskDashboard(config)
    print("✓ Risk dashboard initialized")
    
    # Create sample metrics
    from risk.risk_engine import RiskMetrics
    
    metrics = RiskMetrics(
        timestamp=datetime.now(),
        portfolio_value=1000000.0,
        available_margin=800000.0,
        used_margin=200000.0,
        margin_utilization_percent=25.0,
        portfolio_loss_percent=2.0,
        daily_pnl=-20000.0,
        daily_pnl_percent=-2.0,
        max_drawdown_percent=10.0,
        current_drawdown_percent=3.0,
        position_count=2,
        largest_position_percent=8.0,
        concentration_risk_percent=20.0,
        daily_trades=15,
        hourly_trades=3,
        volatility_percent=2.5,
        risk_score=45.0,
        violations=[]
    )
    
    # Update dashboard
    print("\n--- Updating Dashboard ---")
    dashboard_data = dashboard.update_dashboard(
        risk_metrics=metrics,
        active_alerts=[],
        recent_violations=[]
    )
    print("✓ Dashboard updated")
    
    # Get dashboard summary
    print("\n--- Dashboard Summary ---")
    summary = dashboard.get_dashboard_summary()
    if "error" not in summary:
        risk_summary = summary["risk_summary"]
        print(f"Overall Risk Score: {risk_summary['overall_risk_score']}")
        print(f"Risk Level: {risk_summary['risk_level']}")
        print(f"Portfolio Value: ₹{risk_summary['portfolio_value']:,.2f}")
        print(f"Daily P&L: ₹{risk_summary['daily_pnl']:,.2f}")
        print(f"Margin Utilization: {risk_summary['margin_utilization']:.2f}%")
        print(f"Position Count: {risk_summary['position_count']}")
        print(f"Active Alerts: {risk_summary['active_alerts']}")
    else:
        print(f"Error: {summary['error']}")
    
    # Generate dashboard report
    print("\n--- Dashboard Report ---")
    report = dashboard.generate_dashboard_report()
    print("✓ Dashboard report generated")
    print(f"Report length: {len(report)} characters")
    
    # Show first few lines of report
    report_lines = report.split('\n')[:10]
    print("\nReport preview:")
    for line in report_lines:
        print(f"  {line}")


async def demo_integration():
    """Demonstrate integration between risk components."""
    print("\n" + "="*60)
    print("RISK MANAGEMENT INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Initialize all components
    risk_config = RiskConfig()
    risk_engine = RiskEngine(risk_config)
    
    monitor_config = MonitorConfig()
    monitor = RiskMonitor(monitor_config)
    
    alert_config = AlertConfig()
    alert_manager = RiskAlertManager(alert_config)
    
    dashboard_config = DashboardConfig()
    dashboard = RiskDashboard(dashboard_config)
    
    print("✓ All components initialized")
    
    # Start components
    await risk_engine.start()
    await monitor.start()
    print("✓ Components started")
    
    # Create sample data
    positions = create_sample_positions()
    signals = create_sample_signals()
    
    # Simulate trading workflow
    print("\n--- Simulating Trading Workflow ---")
    
    for signal in signals:
        # Check signal risk
        risk_check = await risk_engine.check_signal_risk(
            signal=signal,
            current_portfolio_value=1000000.0,
            positions=positions
        )
        
        print(f"Signal: {signal.signal_type.value} {signal.symbol}")
        print(f"  Risk Check: {'PASSED' if risk_check['allowed'] else 'FAILED'}")
        
        if not risk_check['allowed']:
            # Create alert for rejected signal
            await alert_manager.create_alert(
                alert_type=RiskAlertType.TRADING_LIMIT,
                priority=AlertPriority.MEDIUM,
                title="Signal Rejected",
                message=f"Signal for {signal.symbol} rejected due to risk limits",
                current_value=risk_check.get('risk_score', 0),
                threshold_value=100,
                symbol=signal.symbol,
                recommendation="Review position sizing or risk parameters"
            )
            print(f"  Alert created for rejected signal")
    
    # Update portfolio metrics
    metrics = await risk_engine.update_portfolio_metrics(
        portfolio_value=1000000.0,
        available_margin=800000.0,
        used_margin=200000.0,
        positions=positions,
        daily_trades=len(signals)
    )
    
    # Update monitor and dashboard
    await monitor.update_metrics(metrics)
    dashboard.update_dashboard(
        risk_metrics=metrics,
        active_alerts=[],
        recent_violations=[]
    )
    
    print(f"✓ Portfolio metrics updated (Risk Score: {metrics.risk_score:.2f})")
    
    # Get final summary
    print("\n--- Final Summary ---")
    dashboard_summary = dashboard.get_dashboard_summary()
    alert_summary = alert_manager.get_alert_summary()
    
    if "error" not in dashboard_summary:
        risk_summary = dashboard_summary["risk_summary"]
        print(f"Overall Risk Score: {risk_summary['overall_risk_score']}")
        print(f"Risk Level: {risk_summary['risk_level']}")
        print(f"Active Alerts: {risk_summary['active_alerts']}")
    
    print(f"Total Alerts: {alert_summary['total_alerts']}")
    print(f"Active Alerts: {alert_summary['active_alerts']}")
    
    # Stop components
    await monitor.stop()
    await risk_engine.stop()
    print("✓ Components stopped")


async def main():
    """Main demo function."""
    print("RISK MANAGEMENT & MONITORING DEMO")
    print("=" * 60)
    print("This demo showcases Phase 6 components:")
    print("- Risk Engine & Metrics")
    print("- Kill Switch")
    print("- Risk Monitor")
    print("- Position Limits")
    print("- Margin Manager")
    print("- Risk Alerts")
    print("- Risk Dashboard")
    print("- Integration")
    
    try:
        # Run individual demos
        await demo_risk_engine()
        await demo_kill_switch()
        await demo_risk_monitor()
        await demo_position_limits()
        await demo_margin_manager()
        await demo_risk_alerts()
        await demo_risk_dashboard()
        
        # Run integration demo
        await demo_integration()
        
        print("\n" + "="*60)
        print("RISK MANAGEMENT DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
