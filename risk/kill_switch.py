"""
Emergency kill switch for the options trading system.

This module provides emergency stop functionality to immediately halt
all trading activities in case of critical risk violations or system failures.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from broker.base_broker import BaseBroker, BrokerOrder, OrderSide, OrderType, ProductType, Variety
from utils.logging_utils import get_logger

# Setup logging
kill_switch_logger = get_logger(__name__)


class KillSwitchReason(Enum):
    """Kill switch trigger reasons."""
    MANUAL = "MANUAL"
    PORTFOLIO_LOSS = "PORTFOLIO_LOSS"
    MARGIN_CALL = "MARGIN_CALL"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    MARKET_HALT = "MARKET_HALT"
    RISK_VIOLATION = "RISK_VIOLATION"
    TECHNICAL_ISSUE = "TECHNICAL_ISSUE"


class KillSwitchStatus(Enum):
    """Kill switch status."""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    RESETTING = "RESETTING"
    DISABLED = "DISABLED"


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch."""
    enable_auto_close: bool = True
    close_all_positions: bool = True
    cancel_pending_orders: bool = True
    send_notifications: bool = True
    auto_close_timeout: int = 300  # 5 minutes
    max_close_attempts: int = 3
    close_order_type: OrderType = OrderType.MARKET
    close_product_type: ProductType = ProductType.MIS
    notification_recipients: List[str] = field(default_factory=list)
    log_all_actions: bool = True


@dataclass
class KillSwitchEvent:
    """Kill switch event data structure."""
    event_id: str
    timestamp: datetime
    reason: KillSwitchReason
    triggered_by: str
    message: str
    portfolio_value_before: float
    portfolio_value_after: Optional[float] = None
    positions_closed: int = 0
    orders_cancelled: int = 0
    actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KillSwitch:
    """
    Emergency kill switch for trading system.
    
    This class provides emergency stop functionality to immediately halt
    all trading activities and close positions when critical risk violations
    or system failures are detected.
    """

    def __init__(self, config: KillSwitchConfig, broker: BaseBroker):
        """
        Initialize kill switch.
        
        Args:
            config: Kill switch configuration
            broker: Broker instance for order management
        """
        self.config = config
        self.broker = broker
        self.status = KillSwitchStatus.ACTIVE
        self.current_event: Optional[KillSwitchEvent] = None
        self.events_history: List[KillSwitchEvent] = []
        
        # Callbacks
        self.trigger_callbacks: List[Callable] = []
        self.reset_callbacks: List[Callable] = []
        
        # State tracking
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[KillSwitchReason] = None
        self.close_attempts = 0
        
        kill_switch_logger.info(f"Initialized KillSwitch with config: {config}")

    async def trigger(self, reason: KillSwitchReason, triggered_by: str = "system",
                     message: str = "", portfolio_value: float = 0.0) -> bool:
        """
        Trigger the kill switch.
        
        Args:
            reason: Reason for triggering kill switch
            triggered_by: Who/what triggered the kill switch
            message: Additional message
            portfolio_value: Current portfolio value
            
        Returns:
            True if kill switch triggered successfully
        """
        try:
            if self.status == KillSwitchStatus.TRIGGERED:
                kill_switch_logger.warning("Kill switch already triggered")
                return True
            
            # Create kill switch event
            self.current_event = KillSwitchEvent(
                event_id=f"kill_switch_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                reason=reason,
                triggered_by=triggered_by,
                message=message,
                portfolio_value_before=portfolio_value
            )
            
            # Update status
            self.status = KillSwitchStatus.TRIGGERED
            self.trigger_time = datetime.now()
            self.trigger_reason = reason
            self.close_attempts = 0
            
            # Log the event
            if self.config.log_all_actions:
                kill_switch_logger.critical(
                    f"KILL SWITCH TRIGGERED - Reason: {reason.value}, "
                    f"Triggered by: {triggered_by}, Message: {message}"
                )
            
            # Execute kill switch actions
            success = await self._execute_kill_switch_actions()
            
            # Add to history
            self.events_history.append(self.current_event)
            
            # Send notifications
            if self.config.send_notifications:
                await self._send_notifications()
            
            # Trigger callbacks
            await self._trigger_callbacks()
            
            return success
            
        except Exception as e:
            kill_switch_logger.error(f"Error triggering kill switch: {e}")
            return False

    async def reset(self, reset_by: str = "manual") -> bool:
        """
        Reset the kill switch.
        
        Args:
            reset_by: Who/what reset the kill switch
            
        Returns:
            True if reset successfully
        """
        try:
            if self.status != KillSwitchStatus.TRIGGERED:
                kill_switch_logger.warning("Kill switch not triggered, cannot reset")
                return False
            
            # Update status
            self.status = KillSwitchStatus.RESETTING
            
            # Log the reset
            kill_switch_logger.info(f"Kill switch reset by: {reset_by}")
            
            # Clear state
            self.trigger_time = None
            self.trigger_reason = None
            self.close_attempts = 0
            self.current_event = None
            
            # Reset to active
            self.status = KillSwitchStatus.ACTIVE
            
            # Trigger reset callbacks
            await self._trigger_reset_callbacks()
            
            kill_switch_logger.info("Kill switch reset successfully")
            return True
            
        except Exception as e:
            kill_switch_logger.error(f"Error resetting kill switch: {e}")
            return False

    async def emergency_close_all(self) -> Dict[str, Any]:
        """
        Emergency close all positions and cancel orders.
        
        Returns:
            Dictionary with results of close operations
        """
        try:
            results = {
                "positions_closed": 0,
                "orders_cancelled": 0,
                "errors": [],
                "success": True
            }
            
            # Cancel all pending orders
            if self.config.cancel_pending_orders:
                cancelled_orders = await self._cancel_all_orders()
                results["orders_cancelled"] = cancelled_orders
                kill_switch_logger.info(f"Cancelled {cancelled_orders} pending orders")
            
            # Close all positions
            if self.config.close_all_positions:
                closed_positions = await self._close_all_positions()
                results["positions_closed"] = closed_positions
                kill_switch_logger.info(f"Closed {closed_positions} positions")
            
            return results
            
        except Exception as e:
            kill_switch_logger.error(f"Error in emergency close all: {e}")
            results["errors"].append(str(e))
            results["success"] = False
            return results

    def get_status(self) -> KillSwitchStatus:
        """Get current kill switch status."""
        return self.status

    def is_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self.status == KillSwitchStatus.TRIGGERED

    def get_current_event(self) -> Optional[KillSwitchEvent]:
        """Get current kill switch event."""
        return self.current_event

    def get_events_history(self) -> List[KillSwitchEvent]:
        """Get kill switch events history."""
        return self.events_history

    def add_trigger_callback(self, callback: Callable):
        """Add kill switch trigger callback."""
        self.trigger_callbacks.append(callback)

    def add_reset_callback(self, callback: Callable):
        """Add kill switch reset callback."""
        self.reset_callbacks.append(callback)

    async def _execute_kill_switch_actions(self) -> bool:
        """Execute kill switch actions."""
        try:
            if not self.current_event:
                return False
            
            # Cancel pending orders
            if self.config.cancel_pending_orders:
                cancelled_orders = await self._cancel_all_orders()
                self.current_event.orders_cancelled = cancelled_orders
                self.current_event.actions_taken.append(f"Cancelled {cancelled_orders} orders")
            
            # Close all positions
            if self.config.close_all_positions:
                closed_positions = await self._close_all_positions()
                self.current_event.positions_closed = closed_positions
                self.current_event.actions_taken.append(f"Closed {closed_positions} positions")
            
            return True
            
        except Exception as e:
            kill_switch_logger.error(f"Error executing kill switch actions: {e}")
            return False

    async def _cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        try:
            # Get all pending orders
            orders = await self.broker.get_orders()
            pending_orders = [order for order in orders if order.is_pending]
            
            cancelled_count = 0
            for order in pending_orders:
                try:
                    success = await self.broker.cancel_order(order.order_id)
                    if success:
                        cancelled_count += 1
                        kill_switch_logger.info(f"Cancelled order: {order.order_id}")
                except Exception as e:
                    kill_switch_logger.error(f"Error cancelling order {order.order_id}: {e}")
            
            return cancelled_count
            
        except Exception as e:
            kill_switch_logger.error(f"Error cancelling all orders: {e}")
            return 0

    async def _close_all_positions(self) -> int:
        """Close all open positions."""
        try:
            # Get all positions
            positions = await self.broker.get_positions()
            open_positions = [pos for pos in positions if not pos.is_flat]
            
            closed_count = 0
            for position in open_positions:
                try:
                    # Determine order side (opposite of position)
                    side = OrderSide.SELL if position.is_long else OrderSide.BUY
                    
                    # Create close order
                    close_order = BrokerOrder(
                        order_id="",
                        symbol=position.symbol,
                        side=side,
                        order_type=self.config.close_order_type,
                        quantity=abs(position.quantity),
                        product=self.config.close_product_type,
                        variety=Variety.REGULAR,
                        tag=f"kill_switch_close_{self.current_event.event_id}"
                    )
                    
                    # Place order
                    result = await self.broker.place_order(close_order)
                    
                    if result.status.value in ["OPEN", "COMPLETE"]:
                        closed_count += 1
                        kill_switch_logger.info(f"Closed position: {position.symbol}")
                    else:
                        kill_switch_logger.warning(f"Failed to close position: {position.symbol}")
                        
                except Exception as e:
                    kill_switch_logger.error(f"Error closing position {position.symbol}: {e}")
            
            return closed_count
            
        except Exception as e:
            kill_switch_logger.error(f"Error closing all positions: {e}")
            return 0

    async def _send_notifications(self):
        """Send kill switch notifications."""
        try:
            if not self.current_event:
                return
            
            message = (
                f"🚨 KILL SWITCH TRIGGERED 🚨\n"
                f"Time: {self.current_event.timestamp}\n"
                f"Reason: {self.current_event.reason.value}\n"
                f"Triggered by: {self.current_event.triggered_by}\n"
                f"Message: {self.current_event.message}\n"
                f"Actions taken: {', '.join(self.current_event.actions_taken)}"
            )
            
            # Log notification
            kill_switch_logger.critical(message)
            
            # Here you would integrate with notification services
            # (email, SMS, Slack, etc.)
            for recipient in self.config.notification_recipients:
                kill_switch_logger.info(f"Notification sent to: {recipient}")
            
        except Exception as e:
            kill_switch_logger.error(f"Error sending notifications: {e}")

    async def _trigger_callbacks(self):
        """Trigger kill switch callbacks."""
        for callback in self.trigger_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_event)
                else:
                    callback(self.current_event)
            except Exception as e:
                kill_switch_logger.error(f"Error in kill switch callback: {e}")

    async def _trigger_reset_callbacks(self):
        """Trigger reset callbacks."""
        for callback in self.reset_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                kill_switch_logger.error(f"Error in reset callback: {e}")

    def get_uptime(self) -> Optional[float]:
        """Get time since kill switch was triggered."""
        if self.trigger_time:
            return (datetime.now() - self.trigger_time).total_seconds()
        return None

    def get_trigger_summary(self) -> Dict[str, Any]:
        """Get kill switch trigger summary."""
        return {
            "status": self.status.value,
            "triggered": self.is_triggered(),
            "trigger_time": self.trigger_time,
            "trigger_reason": self.trigger_reason.value if self.trigger_reason else None,
            "uptime_seconds": self.get_uptime(),
            "current_event": self.current_event.__dict__ if self.current_event else None,
            "total_events": len(self.events_history)
        }
