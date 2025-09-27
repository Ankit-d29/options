"""
Paper trading system for live simulation without real money.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from utils.logging_utils import get_logger
from utils.config import config
from .websocket_feed import MarketData

# Set up logging
paper_trader_logger = get_logger('paper_trader')


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class PaperOrder:
    """Paper trading order representation."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: OrderType
    quantity: int
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    filled_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == 'BUY'
    
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == 'SELL'
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    def get_remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    def get_total_value(self) -> float:
        """Get total order value."""
        return self.filled_quantity * self.filled_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
            'filled_timestamp': self.filled_timestamp.isoformat() if self.filled_timestamp else None,
            'metadata': self.metadata
        }


@dataclass
class PaperPosition:
    """Paper trading position representation."""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    open_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, new_price: float):
        """Update position with new price."""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.average_price) * self.quantity
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def get_market_value(self) -> float:
        """Get current market value of position."""
        return self.quantity * self.current_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'current_price': self.current_price,
            'market_value': self.get_market_value(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.get_total_pnl(),
            'total_commission': self.total_commission,
            'open_time': self.open_time.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class PaperTrade:
    """Paper trading trade representation."""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    order_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'metadata': self.metadata
        }


class PaperTrader:
    """Paper trading system for live simulation."""
    
    def __init__(self, initial_capital: float = None, commission_rate: float = None,
                 slippage_rate: float = None):
        self.initial_capital = initial_capital or config.get('paper_trading.initial_capital', 100000)
        self.commission_rate = commission_rate or config.get('paper_trading.commission_rate', 0.001)
        self.slippage_rate = slippage_rate or config.get('paper_trading.slippage_rate', 0.0005)
        
        # Portfolio state
        self.cash = self.initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trades: List[PaperTrade] = []
        
        # Callbacks
        self.order_callbacks: List[Callable[[PaperOrder], None]] = []
        self.trade_callbacks: List[Callable[[PaperTrade], None]] = []
        self.position_callbacks: List[Callable[[PaperPosition], None]] = []
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        
        paper_trader_logger.info(f"Initialized PaperTrader with capital: ${self.initial_capital:,.2f}")
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        portfolio_value = self.cash
        
        for position in self.positions.values():
            portfolio_value += position.get_market_value()
        
        return portfolio_value
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        total_pnl = sum(pos.get_total_pnl() for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value(),
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'return_percent': (total_pnl / self.initial_capital) * 100,
            'positions_count': len(self.positions),
            'open_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]]),
            'total_trades': len(self.trades)
        }
    
    def submit_order(self, symbol: str, side: str, quantity: int, 
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    metadata: Dict[str, Any] = None) -> PaperOrder:
        """Submit a new order."""
        order_id = str(uuid.uuid4())
        
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            paper_trader_logger.warning(f"Order rejected: {order}")
            return order
        
        # Add to orders
        self.orders[order_id] = order
        
        # Try to fill immediately for market orders
        if order_type == OrderType.MARKET:
            self._try_fill_market_order(order)
        else:
            order.status = OrderStatus.SUBMITTED
        
        # Notify callbacks
        self._notify_order_callbacks(order)
        
        paper_trader_logger.info(f"Order submitted: {order}")
        return order
    
    def _validate_order(self, order: PaperOrder) -> bool:
        """Validate order before submission."""
        # Check if we have enough cash for buy orders
        if order.is_buy():
            # For market orders, use current price; for limit orders, use order price
            price = order.price if order.price else self.current_prices.get(order.symbol, 0)
            if price == 0:
                paper_trader_logger.warning(f"No current price available for {order.symbol}")
                return False
            
            required_cash = order.quantity * price
            required_cash += required_cash * self.commission_rate
            
            if required_cash > self.cash:
                paper_trader_logger.warning(f"Insufficient cash for order: {required_cash:.2f} > {self.cash:.2f}")
                return False
        
        # Check if we have enough position for sell orders
        if order.is_sell():
            current_position = self.positions.get(order.symbol)
            if not current_position or current_position.quantity < order.quantity:
                paper_trader_logger.warning(f"Insufficient position for sell order: {order.quantity}")
                return False
        
        return True
    
    def _try_fill_market_order(self, order: PaperOrder):
        """Try to fill a market order immediately."""
        current_price = self.current_prices.get(order.symbol)
        if not current_price:
            paper_trader_logger.warning(f"No current price available for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return
        
        # Apply slippage
        if order.is_buy():
            fill_price = current_price * (1 + self.slippage_rate)
        else:
            fill_price = current_price * (1 - self.slippage_rate)
        
        # Fill the order
        self._fill_order(order, fill_price, order.quantity)
    
    def _fill_order(self, order: PaperOrder, fill_price: float, fill_quantity: int):
        """Fill an order."""
        commission = fill_quantity * fill_price * self.commission_rate
        
        # Update order
        order.filled_quantity += fill_quantity
        order.filled_price = (order.filled_price * (order.filled_quantity - fill_quantity) + 
                            fill_price * fill_quantity) / order.filled_quantity
        order.commission += commission
        order.filled_timestamp = datetime.now()
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Create trade record
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now(),
            order_id=order.order_id
        )
        
        self.trades.append(trade)
        
        # Update cash and positions
        if order.is_buy():
            self.cash -= (fill_quantity * fill_price + commission)
            self._update_position(order.symbol, fill_quantity, fill_price, commission)
        else:
            self.cash += (fill_quantity * fill_price - commission)
            self._update_position(order.symbol, -fill_quantity, fill_price, commission)
        
        # Notify callbacks
        self._notify_trade_callbacks(trade)
        self._notify_order_callbacks(order)
        
        paper_trader_logger.info(f"Order filled: {order.side} {fill_quantity} {order.symbol} @ ${fill_price:.2f}")
    
    def _update_position(self, symbol: str, quantity_change: int, price: float, commission: float):
        """Update position after trade."""
        if symbol not in self.positions:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=0,
                average_price=0.0,
                current_price=price
            )
        
        position = self.positions[symbol]
        
        if quantity_change > 0:  # Buy
            if position.quantity >= 0:  # Adding to long position or closing short
                total_cost = position.quantity * position.average_price + quantity_change * price
                total_quantity = position.quantity + quantity_change
                position.average_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
            else:  # Closing short position
                position.realized_pnl += (position.average_price - price) * min(abs(quantity_change), abs(position.quantity))
                position.quantity += quantity_change
        
        else:  # Sell
            if position.quantity <= 0:  # Adding to short position or closing long
                total_cost = abs(position.quantity) * position.average_price + abs(quantity_change) * price
                total_quantity = abs(position.quantity) + abs(quantity_change)
                position.average_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = -total_quantity
            else:  # Closing long position
                position.realized_pnl += (price - position.average_price) * min(abs(quantity_change), position.quantity)
                position.quantity += quantity_change
        
        position.total_commission += commission
        position.current_price = price
        position.update_price(price)
        
        # Remove position if quantity is zero
        if position.quantity == 0:
            del self.positions[symbol]
        
        # Notify position callbacks
        self._notify_position_callbacks(position)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            paper_trader_logger.warning(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            paper_trader_logger.warning(f"Cannot cancel order in status: {order.status}")
            return False
        
        order.status = OrderStatus.CANCELLED
        self._notify_order_callbacks(order)
        
        paper_trader_logger.info(f"Order cancelled: {order_id}")
        return True
    
    def update_market_data(self, market_data: MarketData):
        """Update market data and check for order fills."""
        self.current_prices[market_data.symbol] = market_data.price
        
        # Update position prices
        if market_data.symbol in self.positions:
            position = self.positions[market_data.symbol]
            position.update_price(market_data.price)
            self._notify_position_callbacks(position)
        
        # Check for order fills
        self._check_order_fills(market_data)
    
    def _check_order_fills(self, market_data: MarketData):
        """Check if any pending orders can be filled."""
        for order in self.orders.values():
            if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                continue
            
            if order.symbol != market_data.symbol:
                continue
            
            current_price = market_data.price
            
            # Check limit orders
            if order.order_type == OrderType.LIMIT and order.price:
                if order.is_buy() and current_price <= order.price:
                    self._fill_order(order, order.price, order.get_remaining_quantity())
                elif order.is_sell() and current_price >= order.price:
                    self._fill_order(order, order.price, order.get_remaining_quantity())
            
            # Check stop orders
            elif order.order_type == OrderType.STOP and order.stop_price:
                if order.is_buy() and current_price >= order.stop_price:
                    self._fill_order(order, current_price, order.get_remaining_quantity())
                elif order.is_sell() and current_price <= order.stop_price:
                    self._fill_order(order, current_price, order.get_remaining_quantity())
    
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_open_orders(self) -> List[PaperOrder]:
        """Get all open orders."""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]
    
    def get_recent_trades(self, limit: int = 10) -> List[PaperTrade]:
        """Get recent trades."""
        return self.trades[-limit:] if self.trades else []
    
    def add_order_callback(self, callback: Callable[[PaperOrder], None]):
        """Add order event callback."""
        self.order_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[PaperTrade], None]):
        """Add trade event callback."""
        self.trade_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable[[PaperPosition], None]):
        """Add position event callback."""
        self.position_callbacks.append(callback)
    
    def _notify_order_callbacks(self, order: PaperOrder):
        """Notify order callbacks."""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                paper_trader_logger.error(f"Error in order callback: {e}")
    
    def _notify_trade_callbacks(self, trade: PaperTrade):
        """Notify trade callbacks."""
        for callback in self.trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                paper_trader_logger.error(f"Error in trade callback: {e}")
    
    def _notify_position_callbacks(self, position: PaperPosition):
        """Notify position callbacks."""
        for callback in self.position_callbacks:
            try:
                callback(position)
            except Exception as e:
                paper_trader_logger.error(f"Error in position callback: {e}")
    
    def reset(self):
        """Reset paper trader to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.current_prices.clear()
        paper_trader_logger.info("PaperTrader reset to initial state")
