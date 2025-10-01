"""
Live broker trader integration for the options trading system.

This module provides integration between the paper trading system and real brokers,
allowing seamless transition from paper trading to live trading.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from broker.base_broker import (
    BaseBroker, BrokerOrder, BrokerPosition, BrokerQuote,
    OrderType, OrderSide, OrderStatus, ProductType, Variety
)
from live.paper_trader import PaperTrader, PaperOrder, OrderType as PaperOrderType
from live.alerts import AlertManager, AlertType
from strategies.base_strategy import TradingSignal, SignalType
from utils.logging_utils import get_logger

# Setup logging
live_broker_logger = get_logger(__name__)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading."""
    broker: BaseBroker
    initial_capital: Decimal
    max_position_size: Decimal
    max_daily_loss: Decimal
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    enable_paper_mode: bool = False  # For testing with paper trading
    risk_management: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.risk_management is None:
            self.risk_management = {
                "max_portfolio_loss_percent": 0.05,
                "max_position_size_percent": 0.10,
                "max_daily_trades": 50,
                "stop_loss_percent": 0.05,
                "take_profit_percent": 0.10
            }


class LiveBrokerTrader:
    """
    Live broker trader that integrates real broker with trading system.
    
    This class provides a bridge between the paper trading system and real brokers,
    allowing for seamless transition from paper trading to live trading while
    maintaining the same interface and risk management capabilities.
    """

    def __init__(self, config: LiveTradingConfig):
        """
        Initialize live broker trader.
        
        Args:
            config: Live trading configuration
        """
        self.config = config
        self.broker = config.broker
        self.alert_manager = AlertManager()
        self.is_connected = False
        self.is_trading = False
        
        # Initialize paper trader for fallback/testing
        if config.enable_paper_mode:
            self.paper_trader = PaperTrader(
                initial_capital=float(config.initial_capital),
                commission_rate=config.commission_rate
            )
        else:
            self.paper_trader = None
        
        # Trading state
        self.active_orders: Dict[str, BrokerOrder] = {}
        self.positions: Dict[str, BrokerPosition] = {}
        self.daily_pnl: Decimal = Decimal('0')
        self.daily_trades: int = 0
        
        live_broker_logger.info(f"Initialized LiveBrokerTrader with capital: ${config.initial_capital}")
        live_broker_logger.info(f"Paper mode: {config.enable_paper_mode}")

    async def start(self) -> bool:
        """
        Start live trading system.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Connect to broker
            if not await self.broker.connect():
                live_broker_logger.error("Failed to connect to broker")
                return False
            
            self.is_connected = True
            self.is_trading = True
            
            # Get initial positions
            await self._sync_positions()
            
            # Get active orders
            await self._sync_orders()
            
            live_broker_logger.info("Live trading system started successfully")
            
            # Send startup alert
            self.alert_manager.create_alert(
                alert_type=AlertType.INFO,
                title="System Status",
                message="Live trading system started",
                metadata={"status": "started", "timestamp": datetime.now()}
            )
            
            return True
            
        except Exception as e:
            live_broker_logger.error(f"Failed to start live trading system: {e}")
            return False

    async def stop(self) -> bool:
        """
        Stop live trading system.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            self.is_trading = False
            
            # Cancel all pending orders
            await self._cancel_all_orders()
            
            # Disconnect from broker
            await self.broker.disconnect()
            self.is_connected = False
            
            live_broker_logger.info("Live trading system stopped")
            
            # Send shutdown alert
            self.alert_manager.create_alert(
                alert_type=AlertType.INFO,
                title="System Status",
                message="Live trading system stopped",
                metadata={"status": "stopped", "timestamp": datetime.now()}
            )
            
            return True
            
        except Exception as e:
            live_broker_logger.error(f"Failed to stop live trading system: {e}")
            return False

    async def execute_signal(self, signal: TradingSignal) -> Optional[BrokerOrder]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Created order or None if execution failed
        """
        try:
            # Check if trading is enabled
            if not self.is_trading:
                live_broker_logger.warning("Trading is disabled, ignoring signal")
                return None
            
            # Check daily limits
            if not await self._check_daily_limits():
                live_broker_logger.warning("Daily limits exceeded, ignoring signal")
                return None
            
            # Check risk management
            if not await self._check_risk_limits(signal):
                live_broker_logger.warning("Risk limits exceeded, ignoring signal")
                return None
            
            # Create order
            order = await self._create_order_from_signal(signal)
            if not order:
                return None
            
            # Execute order
            if self.config.enable_paper_mode and self.paper_trader:
                # Use paper trader for testing
                result = await self._execute_paper_order(order)
            else:
                # Use real broker
                result = await self.broker.place_order(order)
            
            # Update state
            if result.status in [OrderStatus.OPEN, OrderStatus.COMPLETE]:
                self.active_orders[result.order_id] = result
                self.daily_trades += 1
                
                # Send trade alert
                self.alert_manager.create_alert(
                    alert_type=AlertType.TRADE,
                    title="Trade Executed",
                    message=f"Order executed: {result.side.value} {result.quantity} {result.symbol}",
                    metadata={
                        "order_id": result.order_id,
                        "symbol": result.symbol,
                        "side": result.side.value,
                        "quantity": result.quantity,
                        "price": float(result.price) if result.price else None
                    }
                )
            
            return result
            
        except Exception as e:
            live_broker_logger.error(f"Failed to execute signal: {e}")
            
            # Send error alert
            self.alert_manager.create_alert(
                alert_type=AlertType.ERROR,
                title="Signal Execution Error",
                message=f"Failed to execute signal: {e}",
                metadata={"error": str(e), "signal": signal.to_dict()}
            )
            
            return None

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary.
        
        Returns:
            Dictionary containing portfolio information
        """
        try:
            # Get positions
            positions = await self.broker.get_positions()
            
            # Calculate portfolio metrics
            total_value = Decimal('0')
            unrealized_pnl = Decimal('0')
            realized_pnl = Decimal('0')
            
            for position in positions:
                total_value += position.market_value
                unrealized_pnl += position.unrealized_pnl
                realized_pnl += position.realized_pnl
            
            # Get margins
            margins = await self.broker.get_margins()
            available_cash = Decimal(str(margins.get('available', {}).get('cash', 0)))
            
            return {
                "total_value": float(total_value),
                "available_cash": float(available_cash),
                "unrealized_pnl": float(unrealized_pnl),
                "realized_pnl": float(realized_pnl),
                "total_pnl": float(unrealized_pnl + realized_pnl),
                "positions_count": len(positions),
                "active_orders": len(self.active_orders),
                "daily_trades": self.daily_trades,
                "daily_pnl": float(self.daily_pnl),
                "is_connected": self.is_connected,
                "is_trading": self.is_trading
            }
            
        except Exception as e:
            live_broker_logger.error(f"Failed to get portfolio summary: {e}")
            return {}

    async def _sync_positions(self):
        """Sync positions with broker."""
        try:
            positions = await self.broker.get_positions()
            self.positions = {pos.symbol: pos for pos in positions}
            live_broker_logger.info(f"Synced {len(positions)} positions")
        except Exception as e:
            live_broker_logger.error(f"Failed to sync positions: {e}")

    async def _sync_orders(self):
        """Sync orders with broker."""
        try:
            orders = await self.broker.get_orders()
            self.active_orders = {order.order_id: order for order in orders if order.is_pending}
            live_broker_logger.info(f"Synced {len(self.active_orders)} active orders")
        except Exception as e:
            live_broker_logger.error(f"Failed to sync orders: {e}")

    async def _cancel_all_orders(self):
        """Cancel all pending orders."""
        try:
            for order_id in list(self.active_orders.keys()):
                await self.broker.cancel_order(order_id)
                live_broker_logger.info(f"Cancelled order: {order_id}")
        except Exception as e:
            live_broker_logger.error(f"Failed to cancel orders: {e}")

    async def _check_daily_limits(self) -> bool:
        """Check daily trading limits."""
        max_trades = self.config.risk_management.get("max_daily_trades", 50)
        return self.daily_trades < max_trades

    async def _check_risk_limits(self, signal: TradingSignal) -> bool:
        """Check risk management limits."""
        try:
            # Get current portfolio summary
            summary = await self.get_portfolio_summary()
            
            # Check portfolio loss limit
            max_loss_percent = self.config.risk_management.get("max_portfolio_loss_percent", 0.05)
            total_pnl = summary.get("total_pnl", 0)
            total_value = summary.get("total_value", 0)
            
            if total_value > 0:
                loss_percent = abs(total_pnl) / total_value
                if loss_percent > max_loss_percent:
                    live_broker_logger.warning(f"Portfolio loss limit exceeded: {loss_percent:.2%}")
                    return False
            
            # Check position size limit
            max_position_percent = self.config.risk_management.get("max_position_size_percent", 0.10)
            signal_value = float(signal.price * 100)  # Assuming 100 quantity for options
            
            if total_value > 0:
                position_percent = signal_value / total_value
                if position_percent > max_position_percent:
                    live_broker_logger.warning(f"Position size limit exceeded: {position_percent:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            live_broker_logger.error(f"Failed to check risk limits: {e}")
            return False

    async def _create_order_from_signal(self, signal: TradingSignal) -> Optional[BrokerOrder]:
        """
        Create broker order from trading signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Broker order or None if creation failed
        """
        try:
            # Determine order side
            side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            
            # Determine order type (default to market for now)
            order_type = OrderType.MARKET
            
            # Create order
            order = BrokerOrder(
                order_id="",  # Will be set by broker
                symbol=signal.symbol,
                side=side,
                order_type=order_type,
                quantity=100,  # Default quantity for options
                price=Decimal(str(signal.price)) if signal.price else None,
                product=ProductType.MIS,  # Intraday
                variety=Variety.REGULAR,
                tag=f"signal_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
            )
            
            return order
            
        except Exception as e:
            live_broker_logger.error(f"Failed to create order from signal: {e}")
            return None

    async def _execute_paper_order(self, order: BrokerOrder) -> BrokerOrder:
        """
        Execute order using paper trader (for testing).
        
        Args:
            order: Order to execute
            
        Returns:
            Updated order
        """
        try:
            # Convert to paper order
            paper_order = PaperOrder(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                order_type=PaperOrderType.MARKET,
                quantity=order.quantity,
                price=float(order.price) if order.price else None
            )
            
            # Execute with paper trader
            result = self.paper_trader.submit_order(
                symbol=paper_order.symbol,
                side=paper_order.side,
                quantity=paper_order.quantity,
                order_type=paper_order.order_type,
                price=paper_order.price
            )
            
            # Convert back to broker order
            order.order_id = result.order_id
            if result.status.value == "FILLED":
                order.status = OrderStatus.COMPLETE
            elif result.status.value == "REJECTED":
                order.status = OrderStatus.REJECTED
            else:
                order.status = OrderStatus.OPEN
            order.filled_quantity = result.filled_quantity
            order.filled_price = Decimal(str(result.filled_price)) if result.filled_price else None
            order.average_price = order.filled_price
            
            return order
            
        except Exception as e:
            live_broker_logger.error(f"Failed to execute paper order: {e}")
            order.status = OrderStatus.REJECTED
            order.status_message = str(e)
            return order

    async def subscribe_market_data(self, symbols: List[str], callback: Callable) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call when data is received
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            return await self.broker.subscribe_market_data(symbols, callback)
        except Exception as e:
            live_broker_logger.error(f"Failed to subscribe to market data: {e}")
            return False

    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        try:
            return await self.broker.unsubscribe_market_data(symbols)
        except Exception as e:
            live_broker_logger.error(f"Failed to unsubscribe from market data: {e}")
            return False
