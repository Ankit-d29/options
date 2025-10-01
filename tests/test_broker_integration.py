"""
Test suite for broker integration components.

This module tests the broker integration functionality including:
- Base broker interface
- Kite Connect broker implementation
- Live broker trader
- Order management
- Position tracking
"""

import unittest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Import broker components
from broker.base_broker import (
    BaseBroker, BrokerOrder, BrokerPosition, BrokerQuote,
    OrderType, OrderSide, OrderStatus, ProductType, Variety
)
from broker.kite_connect import KiteConnectBroker, KiteConnectConfig
from live.live_broker_trader import LiveBrokerTrader, LiveTradingConfig
from strategies.base_strategy import TradingSignal, SignalType
from utils.logging_utils import get_logger

# Setup logging
logger = get_logger(__name__)


class MockBroker(BaseBroker):
    """Mock broker for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.orders = {}
        self.positions = {}
        self.quotes = {}
        self.profile = {"user_name": "Test User"}
        self.margins = {"available": {"cash": 100000}}
        
    async def connect(self) -> bool:
        self.is_connected = True
        self.connection_status = "connected"
        return True
        
    async def disconnect(self) -> bool:
        self.is_connected = False
        self.connection_status = "disconnected"
        return True
        
    async def get_profile(self) -> Dict[str, Any]:
        return self.profile
        
    async def get_margins(self) -> Dict[str, Any]:
        return self.margins
        
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        order.order_id = f"mock_order_{len(self.orders) + 1}"
        order.status = OrderStatus.OPEN
        order.order_timestamp = datetime.now()
        self.orders[order.order_id] = order
        return order
        
    async def modify_order(self, order_id: str, **kwargs) -> BrokerOrder:
        if order_id in self.orders:
            order = self.orders[order_id]
            for key, value in kwargs.items():
                setattr(order, key, value)
            return order
        raise ValueError(f"Order {order_id} not found")
        
    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
        
    async def get_orders(self, order_ids: List[str] = None) -> List[BrokerOrder]:
        if order_ids:
            return [order for order_id, order in self.orders.items() if order_id in order_ids]
        return list(self.orders.values())
        
    async def get_positions(self) -> List[BrokerPosition]:
        return list(self.positions.values())
        
    async def get_quote(self, symbols: List[str]) -> Dict[str, BrokerQuote]:
        return {symbol: quote for symbol, quote in self.quotes.items() if symbol in symbols}
        
    async def get_historical_data(self, symbol: str, from_date: datetime, 
                                to_date: datetime, interval: str) -> List[Dict[str, Any]]:
        return []
        
    async def get_instruments(self, exchange: str = None) -> List[Dict[str, Any]]:
        return []
        
    async def subscribe_market_data(self, symbols: List[str], callback: callable) -> bool:
        return True
        
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        return True


class TestBaseBroker(unittest.TestCase):
    """Test base broker interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"api_key": "test_key"}
        self.broker = MockBroker(self.config)
        
    def test_broker_initialization(self):
        """Test broker initialization."""
        assert self.broker.config == self.config
        assert not self.broker.is_connected
        assert self.broker.connection_status == "disconnected"
        
    async def test_connect_disconnect(self):
        """Test broker connection and disconnection."""
        # Test connection
        result = await self.broker.connect()
        assert result is True
        assert self.broker.is_connected
        assert self.broker.connection_status == "connected"
        
        # Test disconnection
        result = await self.broker.disconnect()
        assert result is True
        assert not self.broker.is_connected
        assert self.broker.connection_status == "disconnected"


class TestBrokerOrder(unittest.TestCase):
    """Test broker order functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.order = BrokerOrder(
            order_id="test_order_123",
            symbol="NIFTY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=Decimal('18000'),
            product=ProductType.MIS,
            variety=Variety.REGULAR
        )
        
    def test_order_creation(self):
        """Test order creation."""
        assert self.order.order_id == "test_order_123"
        assert self.order.symbol == "NIFTY"
        assert self.order.side == OrderSide.BUY
        assert self.order.order_type == OrderType.MARKET
        assert self.order.quantity == 100
        assert self.order.price == Decimal('18000')
        
    def test_order_properties(self):
        """Test order properties."""
        assert self.order.is_buy is True
        assert self.order.is_sell is False
        assert self.order.is_pending is True
        assert self.order.is_complete is False
        assert self.order.is_cancelled is False
        assert self.order.is_rejected is False
        
    def test_order_status_changes(self):
        """Test order status changes."""
        # Test complete status
        self.order.status = OrderStatus.COMPLETE
        assert self.order.is_complete is True
        assert self.order.is_pending is False
        
        # Test cancelled status
        self.order.status = OrderStatus.CANCELLED
        assert self.order.is_cancelled is True
        assert self.order.is_pending is False


class TestBrokerPosition(unittest.TestCase):
    """Test broker position functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position = BrokerPosition(
            symbol="NIFTY",
            quantity=100,
            average_price=Decimal('18000'),
            last_price=Decimal('18100'),
            day_change=Decimal('100'),
            day_change_percent=Decimal('0.56'),
            unrealized_pnl=Decimal('10000'),
            realized_pnl=Decimal('5000')
        )
        
    def test_position_creation(self):
        """Test position creation."""
        assert self.position.symbol == "NIFTY"
        assert self.position.quantity == 100
        assert self.position.average_price == Decimal('18000')
        assert self.position.last_price == Decimal('18100')
        
    def test_position_properties(self):
        """Test position properties."""
        assert self.position.is_long is True
        assert self.position.is_short is False
        assert self.position.is_flat is False
        assert self.position.market_value == Decimal('1810000')  # 100 * 18100
        
    def test_short_position(self):
        """Test short position."""
        self.position.quantity = -100
        assert self.position.is_long is False
        assert self.position.is_short is True
        assert self.position.is_flat is False


class TestBrokerQuote(unittest.TestCase):
    """Test broker quote functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quote = BrokerQuote(
            symbol="NIFTY",
            last_price=Decimal('18100'),
            last_quantity=100,
            average_price=Decimal('18050'),
            volume=1000000,
            buy_quantity=50000,
            sell_quantity=45000,
            ohlc={
                'open': Decimal('18000'),
                'high': Decimal('18150'),
                'low': Decimal('17950'),
                'close': Decimal('18100')
            },
            net_change=Decimal('100')
        )
        
    def test_quote_creation(self):
        """Test quote creation."""
        assert self.quote.symbol == "NIFTY"
        assert self.quote.last_price == Decimal('18100')
        assert self.quote.volume == 1000000
        assert self.quote.ohlc['open'] == Decimal('18000')


class TestKiteConnectBroker(unittest.TestCase):
    """Test Kite Connect broker implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = KiteConnectConfig(
            api_key="test_api_key",
            api_secret="test_api_secret",
            access_token="test_access_token"
        )
        
    @patch('broker.kite_connect.KiteConnect')
    def test_kite_connect_initialization(self, mock_kite_connect):
        """Test Kite Connect broker initialization."""
        broker = KiteConnectBroker(self.config)
        
        assert broker.config == self.config
        assert broker.is_connected is True  # Because access_token is provided
        assert broker.connection_status == "connected"
        
    @patch('broker.kite_connect.KiteConnect')
    async def test_kite_connect_connection(self, mock_kite_connect):
        """Test Kite Connect connection."""
        # Mock successful connection
        mock_kite = Mock()
        mock_kite.profile.return_value = {"user_name": "Test User"}
        mock_kite_connect.return_value = mock_kite
        
        broker = KiteConnectBroker(self.config)
        result = await broker.connect()
        
        assert result is True
        assert broker.is_connected is True


class TestLiveBrokerTrader(unittest.TestCase):
    """Test live broker trader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_broker = MockBroker({"api_key": "test"})
        self.config = LiveTradingConfig(
            broker=self.mock_broker,
            initial_capital=Decimal('100000'),
            max_position_size=Decimal('10000'),
            max_daily_loss=Decimal('5000'),
            enable_paper_mode=True
        )
        self.trader = LiveBrokerTrader(self.config)
        
    async def test_trader_initialization(self):
        """Test trader initialization."""
        assert self.trader.config == self.config
        assert self.trader.broker == self.mock_broker
        assert not self.trader.is_connected
        assert not self.trader.is_trading
        assert self.trader.paper_trader is not None  # Because paper mode is enabled
        
    async def test_start_stop_trader(self):
        """Test starting and stopping trader."""
        # Test start
        result = await self.trader.start()
        assert result is True
        assert self.trader.is_connected
        assert self.trader.is_trading
        
        # Test stop
        result = await self.trader.stop()
        assert result is True
        assert not self.trader.is_trading
        assert not self.trader.is_connected
        
    async def test_execute_signal(self):
        """Test signal execution."""
        # Start trader first
        await self.trader.start()
        
        # Create test signal
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0,
            confidence=0.8
        )
        
        # Execute signal
        result = await self.trader.execute_signal(signal)
        
        assert result is not None
        assert result.symbol == "NIFTY"
        assert result.side == OrderSide.BUY
        assert result.quantity == 100  # Default quantity for options
        
    async def test_portfolio_summary(self):
        """Test portfolio summary."""
        # Start trader first
        await self.trader.start()
        
        # Get portfolio summary
        summary = await self.trader.get_portfolio_summary()
        
        assert "total_value" in summary
        assert "available_cash" in summary
        assert "unrealized_pnl" in summary
        assert "realized_pnl" in summary
        assert "positions_count" in summary
        assert "active_orders" in summary
        assert "is_connected" in summary
        assert "is_trading" in summary
        
    async def test_risk_limits(self):
        """Test risk limit checking."""
        # Start trader first
        await self.trader.start()
        
        # Test with normal signal
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0,
            confidence=0.8
        )
        
        # This should pass risk checks
        result = await self.trader.execute_signal(signal)
        assert result is not None
        
        # Test with large position size (should fail)
        large_signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=500000.0,  # Very large price
            confidence=0.8
        )
        
        # This should fail risk checks or be rejected by paper trader
        result = await self.trader.execute_signal(large_signal)
        # Either risk checks should reject it (result is None) or paper trader should reject it (status is REJECTED)
        assert result is None or result.status == OrderStatus.REJECTED


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    async def test_full_trading_flow(self):
        """Test complete trading flow."""
        # Create mock broker
        mock_broker = MockBroker({"api_key": "test"})
        
        # Create trader config
        config = LiveTradingConfig(
            broker=mock_broker,
            initial_capital=Decimal('100000'),
            max_position_size=Decimal('10000'),
            max_daily_loss=Decimal('5000'),
            enable_paper_mode=True
        )
        
        # Create trader
        trader = LiveBrokerTrader(config)
        
        # Start trader
        await trader.start()
        
        # Create and execute signal
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0,
            confidence=0.8
        )
        
        order = await trader.execute_signal(signal)
        assert order is not None
        
        # Get portfolio summary
        summary = await trader.get_portfolio_summary()
        assert summary["is_trading"] is True
        
        # Stop trader
        await trader.stop()
        assert not trader.is_trading


def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Convert async tests to sync tests
class TestAsyncIntegration(unittest.TestCase):
    """Convert async tests to sync tests."""
    
    def test_base_broker_connect_disconnect(self):
        """Test broker connection and disconnection."""
        # Create test instance with proper setup
        test_instance = TestBaseBroker()
        test_instance.setUp()
        run_async_test(test_instance.test_connect_disconnect())
        
    def test_kite_connect_connection(self):
        """Test Kite Connect connection."""
        # Create test instance with proper setup
        test_instance = TestKiteConnectBroker()
        test_instance.setUp()
        run_async_test(test_instance.test_kite_connect_connection())
        
    def test_trader_initialization(self):
        """Test trader initialization."""
        # Create test instance with proper setup
        test_instance = TestLiveBrokerTrader()
        test_instance.setUp()
        run_async_test(test_instance.test_trader_initialization())
        
    def test_start_stop_trader(self):
        """Test starting and stopping trader."""
        # Create test instance with proper setup
        test_instance = TestLiveBrokerTrader()
        test_instance.setUp()
        run_async_test(test_instance.test_start_stop_trader())
        
    def test_execute_signal(self):
        """Test signal execution."""
        # Create test instance with proper setup
        test_instance = TestLiveBrokerTrader()
        test_instance.setUp()
        run_async_test(test_instance.test_execute_signal())
        
    def test_portfolio_summary(self):
        """Test portfolio summary."""
        # Create test instance with proper setup
        test_instance = TestLiveBrokerTrader()
        test_instance.setUp()
        run_async_test(test_instance.test_portfolio_summary())
        
    def test_risk_limits(self):
        """Test risk limit checking."""
        # Create test instance with proper setup
        test_instance = TestLiveBrokerTrader()
        test_instance.setUp()
        run_async_test(test_instance.test_risk_limits())
        
    def test_full_trading_flow(self):
        """Test complete trading flow."""
        # Create test instance with proper setup
        test_instance = TestIntegration()
        run_async_test(test_instance.test_full_trading_flow())


if __name__ == "__main__":
    unittest.main()
