"""
Unit tests for live trading and paper trading components.
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from live.websocket_feed import MockWebSocketFeed, MarketData, MarketDataBuffer, MarketDataAnalyzer
from live.paper_trader import PaperTrader, PaperOrder, OrderStatus, OrderType, PaperPosition, PaperTrade
from live.live_strategy_runner import LiveStrategyRunner, LiveStrategyConfig
from live.position_monitor import PositionMonitor, RiskLimits, PositionAlert
from live.alerts import AlertManager, AlertType, Alert
from strategies.supertrend import SupertrendStrategy
from strategies.base_strategy import TradingSignal, SignalType


class TestMarketData:
    """Test cases for MarketData class."""
    
    def test_market_data_creation(self):
        """Test creating MarketData object."""
        timestamp = datetime.now()
        data = MarketData(
            symbol="NIFTY",
            timestamp=timestamp,
            price=18000.0,
            volume=1000,
            bid=17999.0,
            ask=18001.0,
            bid_size=500,
            ask_size=500
        )
        
        assert data.symbol == "NIFTY"
        assert data.price == 18000.0
        assert data.volume == 1000
        assert data.bid == 17999.0
        assert data.ask == 18001.0
    
    def test_market_data_to_dict(self):
        """Test converting MarketData to dictionary."""
        data = MarketData(
            symbol="NIFTY",
            timestamp=datetime.now(),
            price=18000.0,
            volume=1000,
            bid=17999.0,
            ask=18001.0,
            bid_size=500,
            ask_size=500
        )
        
        data_dict = data.to_dict()
        
        assert 'symbol' in data_dict
        assert 'price' in data_dict
        assert 'volume' in data_dict
        assert data_dict['symbol'] == "NIFTY"
        assert data_dict['price'] == 18000.0


class TestMockWebSocketFeed:
    """Test cases for MockWebSocketFeed class."""
    
    def test_feed_initialization(self):
        """Test MockWebSocketFeed initialization."""
        symbols = ["NIFTY", "BANKNIFTY"]
        feed = MockWebSocketFeed(symbols)
        
        assert feed.symbols == symbols
        assert not feed.is_running
        assert feed.connection_status == "disconnected"
    
    def test_subscribe_unsubscribe(self):
        """Test subscription management."""
        feed = MockWebSocketFeed(["NIFTY"])
        
        def callback(data):
            pass
        
        feed.subscribe(callback)
        assert len(feed.subscribers) == 1
        
        feed.unsubscribe(callback)
        assert len(feed.subscribers) == 0
    
    def test_set_volatility_and_trend(self):
        """Test setting volatility and trend."""
        feed = MockWebSocketFeed(["NIFTY"])
        
        feed.set_volatility("NIFTY", 0.002)
        feed.set_trend("NIFTY", 0.0001)
        
        assert feed.volatility["NIFTY"] == 0.002
        assert feed.trend["NIFTY"] == 0.0001
    
    def test_get_current_price(self):
        """Test getting current price."""
        feed = MockWebSocketFeed(["NIFTY"])
        
        price = feed.get_current_price("NIFTY")
        assert price is not None
        assert price > 0


class TestMarketDataBuffer:
    """Test cases for MarketDataBuffer class."""
    
    def test_buffer_initialization(self):
        """Test MarketDataBuffer initialization."""
        buffer = MarketDataBuffer(max_size=100)
        
        assert buffer.max_size == 100
        assert len(buffer.data) == 0
    
    def test_add_and_get_data(self):
        """Test adding and retrieving data."""
        buffer = MarketDataBuffer()
        
        data = MarketData(
            symbol="NIFTY",
            timestamp=datetime.now(),
            price=18000.0,
            volume=1000,
            bid=17999.0,
            ask=18001.0,
            bid_size=500,
            ask_size=500
        )
        
        buffer.add_data(data)
        
        latest_data = buffer.get_latest_data("NIFTY", 1)
        assert len(latest_data) == 1
        assert latest_data[0].symbol == "NIFTY"
    
    def test_get_data_range(self):
        """Test getting data within time range."""
        buffer = MarketDataBuffer()
        
        now = datetime.now()
        
        # Add data at different times
        for i in range(5):
            data = MarketData(
                symbol="NIFTY",
                timestamp=now + timedelta(minutes=i),
                price=18000.0 + i,
                volume=1000,
                bid=17999.0 + i,
                ask=18001.0 + i,
                bid_size=500,
                ask_size=500
            )
            buffer.add_data(data)
        
        # Get data in range
        start_time = now + timedelta(minutes=1)
        end_time = now + timedelta(minutes=3)
        
        range_data = buffer.get_data_range("NIFTY", start_time, end_time)
        assert len(range_data) == 3


class TestMarketDataAnalyzer:
    """Test cases for MarketDataAnalyzer class."""
    
    def test_price_statistics(self):
        """Test price statistics calculation."""
        buffer = MarketDataBuffer()
        analyzer = MarketDataAnalyzer(buffer)
        
        # Add sample data with timestamps in the last 5 minutes
        now = datetime.now()
        prices = [18000, 18050, 18100, 18075, 18125]
        
        for i, price in enumerate(prices):
            data = MarketData(
                symbol="NIFTY",
                timestamp=now - timedelta(minutes=4-i),  # Data from 4 minutes ago to now
                price=price,
                volume=1000,
                bid=price - 1,
                ask=price + 1,
                bid_size=500,
                ask_size=500
            )
            buffer.add_data(data)
        
        stats = analyzer.get_price_statistics("NIFTY", period_minutes=5)
        
        assert stats['symbol'] == "NIFTY"
        assert stats['current_price'] == 18125
        assert stats['min_price'] == 18000
        assert stats['max_price'] == 18125


class TestPaperOrder:
    """Test cases for PaperOrder class."""
    
    def test_order_creation(self):
        """Test creating PaperOrder object."""
        order = PaperOrder(
            order_id="test_123",
            symbol="NIFTY",
            side="BUY",
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.order_id == "test_123"
        assert order.symbol == "NIFTY"
        assert order.side == "BUY"
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
    
    def test_order_methods(self):
        """Test PaperOrder utility methods."""
        buy_order = PaperOrder(
            order_id="buy_123",
            symbol="NIFTY",
            side="BUY",
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        sell_order = PaperOrder(
            order_id="sell_123",
            symbol="NIFTY",
            side="SELL",
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert buy_order.is_buy()
        assert not buy_order.is_sell()
        assert sell_order.is_sell()
        assert not sell_order.is_buy()
    
    def test_filled_order(self):
        """Test filled order status."""
        order = PaperOrder(
            order_id="test_123",
            symbol="NIFTY",
            side="BUY",
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = 100
        
        assert order.is_filled()
        assert order.get_remaining_quantity() == 0


class TestPaperPosition:
    """Test cases for PaperPosition class."""
    
    def test_position_creation(self):
        """Test creating PaperPosition object."""
        position = PaperPosition(
            symbol="NIFTY",
            quantity=100,
            average_price=18000.0,
            current_price=18100.0
        )
        
        assert position.symbol == "NIFTY"
        assert position.quantity == 100
        assert position.average_price == 18000.0
        assert position.current_price == 18100.0
    
    def test_position_pnl_calculation(self):
        """Test position P&L calculation."""
        # Long position
        long_position = PaperPosition(
            symbol="NIFTY",
            quantity=100,
            average_price=18000.0,
            current_price=18100.0
        )
        
        long_position.update_price(18100.0)
        assert long_position.unrealized_pnl == 10000.0  # (18100 - 18000) * 100
        
        # Short position
        short_position = PaperPosition(
            symbol="NIFTY",
            quantity=-100,
            average_price=18000.0,
            current_price=17900.0
        )
        
        short_position.update_price(17900.0)
        assert short_position.unrealized_pnl == 10000.0  # (18000 - 17900) * 100


class TestPaperTrader:
    """Test cases for PaperTrader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trader = PaperTrader(initial_capital=100000, commission_rate=0.001)
    
    def test_trader_initialization(self):
        """Test PaperTrader initialization."""
        assert self.trader.initial_capital == 100000
        assert self.trader.cash == 100000
        assert self.trader.commission_rate == 0.001
        assert len(self.trader.positions) == 0
        assert len(self.trader.orders) == 0
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        portfolio_value = self.trader.get_portfolio_value()
        assert portfolio_value == 100000  # Should equal cash initially
    
    def test_submit_market_order(self):
        """Test submitting market order."""
        # Set current price
        market_data = MarketData(
            symbol="NIFTY",
            timestamp=datetime.now(),
            price=18000.0,
            volume=1000,
            bid=17999.0,
            ask=18001.0,
            bid_size=500,
            ask_size=500
        )
        self.trader.update_market_data(market_data)
        
        order = self.trader.submit_order(
            symbol="NIFTY",
            side="BUY",
            quantity=5,  # Reduced quantity to fit within cash limit
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "NIFTY"
        assert order.side == "BUY"
        assert order.quantity == 5
        assert order.status == OrderStatus.FILLED
    
    def test_submit_limit_order(self):
        """Test submitting limit order."""
        market_data = MarketData(
            symbol="NIFTY",
            timestamp=datetime.now(),
            price=18000.0,
            volume=1000,
            bid=17999.0,
            ask=18001.0,
            bid_size=500,
            ask_size=500
        )
        self.trader.update_market_data(market_data)
        
        order = self.trader.submit_order(
            symbol="NIFTY",
            side="BUY",
            quantity=5,
            order_type=OrderType.LIMIT,
            price=17900.0
        )
        
        assert order.status == OrderStatus.SUBMITTED
        assert order.price == 17900.0
    
    def test_cancel_order(self):
        """Test cancelling order."""
        order = self.trader.submit_order(
            symbol="NIFTY",
            side="BUY",
            quantity=5,
            order_type=OrderType.LIMIT,
            price=17900.0
        )
        
        success = self.trader.cancel_order(order.order_id)
        assert success
        assert order.status == OrderStatus.CANCELLED
    
    def test_get_portfolio_summary(self):
        """Test portfolio summary."""
        summary = self.trader.get_portfolio_summary()
        
        assert 'initial_capital' in summary
        assert 'cash' in summary
        assert 'portfolio_value' in summary
        assert 'total_pnl' in summary
        assert 'positions_count' in summary
    
    def test_reset(self):
        """Test trader reset."""
        # Make some changes
        order = self.trader.submit_order("NIFTY", "BUY", 100, OrderType.MARKET)
        
        # Reset
        self.trader.reset()
        
        assert self.trader.cash == self.trader.initial_capital
        assert len(self.trader.positions) == 0
        assert len(self.trader.orders) == 0


class TestAlertManager:
    """Test cases for AlertManager class."""
    
    def test_alert_manager_initialization(self):
        """Test AlertManager initialization."""
        manager = AlertManager()
        
        assert len(manager.alerts) == 0
        assert len(manager.alert_callbacks) == 0
        assert manager.console_enabled
        assert manager.file_logging_enabled
    
    def test_create_alert(self):
        """Test creating alert."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            AlertType.INFO,
            "Test Alert",
            "This is a test message"
        )
        
        assert alert.alert_type == AlertType.INFO
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test message"
        assert len(manager.alerts) == 1
    
    def test_send_trade_alert(self):
        """Test sending trade alert."""
        manager = AlertManager()
        
        manager.send_trade_alert("NIFTY", "BUY", 100, 18000.0)
        
        assert len(manager.alerts) == 1
        alert = manager.alerts[0]
        assert alert.alert_type == AlertType.TRADE
        assert "NIFTY" in alert.title
        assert alert.symbol == "NIFTY"
    
    def test_send_signal_alert(self):
        """Test sending signal alert."""
        manager = AlertManager()
        
        manager.send_signal_alert("NIFTY", "BUY", 18000.0, 0.8)
        
        assert len(manager.alerts) == 1
        alert = manager.alerts[0]
        assert alert.alert_type == AlertType.SIGNAL
        assert "BUY" in alert.title
    
    def test_get_alert_summary(self):
        """Test getting alert summary."""
        manager = AlertManager()
        
        # Add some alerts
        manager.send_trade_alert("NIFTY", "BUY", 100, 18000.0)
        manager.send_signal_alert("NIFTY", "SELL", 18100.0, 0.7)
        
        summary = manager.get_alert_summary()
        
        assert summary['total_alerts'] == 2
        assert 'alerts_by_type' in summary
        assert 'alerts_by_severity' in summary


class TestPositionMonitor:
    """Test cases for PositionMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trader = PaperTrader(initial_capital=100000)
        self.monitor = PositionMonitor(self.trader)
    
    def test_monitor_initialization(self):
        """Test PositionMonitor initialization."""
        assert self.monitor.paper_trader == self.trader
        assert self.monitor.initial_portfolio_value == 100000
        assert len(self.monitor.position_alerts) == 0
    
    def test_check_portfolio_risk(self):
        """Test portfolio risk checking."""
        # Simulate a large loss
        self.trader.cash = 90000  # 10% loss
        
        alerts = self.monitor.check_portfolio_risk()
        
        # Should generate an alert for exceeding max portfolio loss
        assert len(alerts) > 0
        assert any(alert.alert_type == "MAX_PORTFOLIO_LOSS" for alert in alerts)
    
    def test_check_position_risk(self):
        """Test position risk checking."""
        # Create a large position
        position = PaperPosition(
            symbol="NIFTY",
            quantity=1000,
            average_price=18000.0,
            current_price=18000.0
        )
        self.trader.positions["NIFTY"] = position
        
        alerts = self.monitor.check_position_risk(position)
        
        # Should generate an alert for exceeding max position size
        assert len(alerts) > 0
        assert any(alert.alert_type == "MAX_POSITION_SIZE" for alert in alerts)
    
    def test_get_risk_summary(self):
        """Test getting risk summary."""
        summary = self.monitor.get_risk_summary()
        
        assert 'portfolio_value' in summary
        assert 'portfolio_loss_percent' in summary
        assert 'risk_limits' in summary
        assert 'alerts_count' in summary


class TestLiveStrategyConfig:
    """Test cases for LiveStrategyConfig class."""
    
    def test_config_creation(self):
        """Test creating LiveStrategyConfig."""
        strategy = SupertrendStrategy()
        
        config = LiveStrategyConfig(
            symbol="NIFTY",
            strategy=strategy,
            timeframe="1m",
            max_positions=2,
            position_size=100
        )
        
        assert config.symbol == "NIFTY"
        assert config.strategy == strategy
        assert config.timeframe == "1m"
        assert config.max_positions == 2
        assert config.position_size == 100
        assert config.enabled


class TestLiveStrategyRunner:
    """Test cases for LiveStrategyRunner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trader = PaperTrader(initial_capital=100000)
        self.feed = MockWebSocketFeed(["NIFTY"])
        self.runner = LiveStrategyRunner(self.trader, self.feed)
    
    def test_runner_initialization(self):
        """Test LiveStrategyRunner initialization."""
        assert self.runner.paper_trader == self.trader
        assert self.runner.websocket_feed == self.feed
        assert len(self.runner.strategies) == 0
        assert not self.runner.is_running
    
    def test_add_strategy(self):
        """Test adding strategy."""
        strategy = SupertrendStrategy()
        config = LiveStrategyConfig(
            symbol="NIFTY",
            strategy=strategy,
            timeframe="1m"
        )
        
        self.runner.add_strategy(config)
        
        strategy_key = "NIFTY_SupertrendStrategy_1m"
        assert strategy_key in self.runner.strategies
        assert strategy_key in self.runner.results
    
    def test_remove_strategy(self):
        """Test removing strategy."""
        strategy = SupertrendStrategy()
        config = LiveStrategyConfig(
            symbol="NIFTY",
            strategy=strategy,
            timeframe="1m"
        )
        
        self.runner.add_strategy(config)
        strategy_key = "NIFTY_SupertrendStrategy_1m"
        
        self.runner.remove_strategy(strategy_key)
        
        assert strategy_key not in self.runner.strategies
        assert self.runner.results[strategy_key].status == "STOPPED"


class TestRiskLimits:
    """Test cases for RiskLimits class."""
    
    def test_risk_limits_creation(self):
        """Test creating RiskLimits."""
        limits = RiskLimits(
            max_portfolio_loss_percent=0.05,
            max_position_size_percent=0.10,
            max_daily_trades=50
        )
        
        assert limits.max_portfolio_loss_percent == 0.05
        assert limits.max_position_size_percent == 0.10
        assert limits.max_daily_trades == 50


class TestPositionAlert:
    """Test cases for PositionAlert class."""
    
    def test_position_alert_creation(self):
        """Test creating PositionAlert."""
        alert = PositionAlert(
            alert_id="test_123",
            position_symbol="NIFTY",
            alert_type="STOP_LOSS",
            message="Stop loss triggered",
            timestamp=datetime.now(),
            severity="WARNING"
        )
        
        assert alert.alert_id == "test_123"
        assert alert.position_symbol == "NIFTY"
        assert alert.alert_type == "STOP_LOSS"
        assert alert.message == "Stop loss triggered"
        assert alert.severity == "WARNING"


if __name__ == "__main__":
    pytest.main([__file__])
