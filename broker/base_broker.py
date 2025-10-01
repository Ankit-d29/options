"""
Base broker interface for the options trading system.

This module defines the abstract base class that all broker integrations
must implement, ensuring a consistent interface across different brokers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal


class OrderType(Enum):
    """Order types supported by brokers."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class OrderSide(Enum):
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status values."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER_PENDING"


class ProductType(Enum):
    """Product types for orders."""
    MIS = "MIS"  # Intraday
    CNC = "CNC"  # Cash and Carry
    NRML = "NRML"  # Normal


class Variety(Enum):
    """Order varieties."""
    REGULAR = "regular"
    BO = "bo"  # Bracket Order
    CO = "co"  # Cover Order
    AMO = "amo"  # After Market Order


@dataclass
class BrokerOrder:
    """Represents an order placed through a broker."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    product: ProductType = ProductType.MIS
    variety: Variety = Variety.REGULAR
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    pending_quantity: int = 0
    cancelled_quantity: int = 0
    order_timestamp: Optional[datetime] = None
    exchange_timestamp: Optional[datetime] = None
    status_message: Optional[str] = None
    tag: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status == OrderStatus.COMPLETE

    @property
    def is_pending(self) -> bool:
        """Check if order is pending."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.TRIGGER_PENDING]

    @property
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status == OrderStatus.CANCELLED

    @property
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.status == OrderStatus.REJECTED


@dataclass
class BrokerPosition:
    """Represents a position held through a broker."""
    symbol: str
    quantity: int
    average_price: Decimal
    last_price: Decimal
    day_change: Decimal
    day_change_percent: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    product: ProductType = ProductType.MIS
    instrument_token: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no quantity)."""
        return self.quantity == 0

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position."""
        return Decimal(abs(self.quantity)) * self.last_price


@dataclass
class BrokerQuote:
    """Represents a market quote from a broker."""
    symbol: str
    last_price: Decimal
    last_quantity: int
    average_price: Decimal
    volume: int
    buy_quantity: int
    sell_quantity: int
    ohlc: Dict[str, Decimal]  # {open, high, low, close}
    net_change: Decimal
    oi: Optional[int] = None  # Open Interest
    oi_day_high: Optional[int] = None
    oi_day_low: Optional[int] = None
    timestamp: Optional[datetime] = None
    depth: Optional[Dict[str, List[Dict[str, Any]]]] = None  # Market depth
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseBroker(ABC):
    """
    Abstract base class for broker integrations.
    
    All broker implementations must inherit from this class and implement
    all abstract methods to ensure consistent functionality across brokers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the broker with configuration.
        
        Args:
            config: Broker-specific configuration dictionary
        """
        self.config = config
        self.is_connected = False
        self.connection_status = "disconnected"

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the broker.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile information.
        
        Returns:
            Dictionary containing user profile data
        """
        pass

    @abstractmethod
    async def get_margins(self) -> Dict[str, Any]:
        """
        Get account margins information.
        
        Returns:
            Dictionary containing margin data
        """
        pass

    @abstractmethod
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """
        Place an order through the broker.
        
        Args:
            order: Order to place
            
        Returns:
            Updated order with broker-assigned order_id
        """
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, **kwargs) -> BrokerOrder:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            **kwargs: Parameters to modify (price, quantity, etc.)
            
        Returns:
            Updated order object
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_orders(self, order_ids: Optional[List[str]] = None) -> List[BrokerOrder]:
        """
        Get order information.
        
        Args:
            order_ids: Optional list of specific order IDs to retrieve
            
        Returns:
            List of order objects
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[BrokerPosition]:
        """
        Get current positions.
        
        Returns:
            List of position objects
        """
        pass

    @abstractmethod
    async def get_quote(self, symbols: List[str]) -> Dict[str, BrokerQuote]:
        """
        Get market quotes for symbols.
        
        Args:
            symbols: List of symbol names
            
        Returns:
            Dictionary mapping symbol to quote object
        """
        pass

    @abstractmethod
    async def get_historical_data(self, symbol: str, from_date: datetime, 
                                to_date: datetime, interval: str) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Symbol name
            from_date: Start date
            to_date: End date
            interval: Data interval (minute, day, etc.)
            
        Returns:
            List of historical data points
        """
        pass

    @abstractmethod
    async def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of tradable instruments.
        
        Args:
            exchange: Optional exchange filter
            
        Returns:
            List of instrument dictionaries
        """
        pass

    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str], 
                                  callback: callable) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call when data is received
            
        Returns:
            True if subscription successful, False otherwise
        """
        pass

    @abstractmethod
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        pass

    def get_connection_status(self) -> str:
        """
        Get current connection status.
        
        Returns:
            Connection status string
        """
        return self.connection_status

    def is_connected(self) -> bool:
        """
        Check if connected to broker.
        
        Returns:
            True if connected, False otherwise
        """
        return self.is_connected
