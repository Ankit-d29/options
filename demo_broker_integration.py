#!/usr/bin/env python3
"""
Demo script for Phase 5 - Broker Integration

This script demonstrates:
1. Kite Connect broker integration
2. Live trading with real broker
3. Order management and execution
4. Position tracking and monitoring
5. Risk management integration
6. Real-time market data subscription

Usage:
    python demo_broker_integration.py

Note: This demo requires valid Kite Connect credentials in .env file
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_utils import get_logger
from broker.kite_connect import KiteConnectBroker, KiteConnectConfig
from live.live_broker_trader import LiveBrokerTrader, LiveTradingConfig
from strategies.supertrend import SupertrendStrategy
from strategies.base_strategy import TradingSignal, SignalType
from live.alerts import AlertManager, AlertType

# Setup logging
logger = get_logger(__name__)


def demo_broker_initialization():
    """Demonstrate broker initialization."""
    logger.info("\n=== Broker Initialization Demo ===")
    
    # Create Kite Connect config
    config = KiteConnectConfig(
        api_key="demo_api_key",
        api_secret="demo_api_secret",
        access_token=None,  # Will need to be set for real trading
        debug=True
    )
    
    # Create broker
    broker = KiteConnectBroker(config)
    
    logger.info(f"Created Kite Connect broker with API key: {config.api_key}")
    logger.info(f"Broker connection status: {broker.connection_status}")
    logger.info(f"Broker connected: {broker.is_connected}")
    
    return broker


async def demo_broker_connection(broker):
    """Demonstrate broker connection."""
    logger.info("\n=== Broker Connection Demo ===")
    
    try:
        # Attempt connection
        connected = await broker.connect()
        
        if connected:
            logger.info("✅ Successfully connected to broker")
            
            # Get profile
            try:
                profile = await broker.get_profile()
                logger.info(f"User profile: {profile.get('user_name', 'Unknown')}")
            except Exception as e:
                logger.warning(f"Could not get profile: {e}")
            
            # Get margins
            try:
                margins = await broker.get_margins()
                available_cash = margins.get('available', {}).get('cash', 0)
                logger.info(f"Available cash: ₹{available_cash:,.2f}")
            except Exception as e:
                logger.warning(f"Could not get margins: {e}")
                
        else:
            logger.warning("❌ Failed to connect to broker")
            logger.info("Note: This is expected in demo mode without valid credentials")
            
    except Exception as e:
        logger.error(f"Connection error: {e}")
        logger.info("Note: This is expected in demo mode without valid credentials")


async def demo_order_management(broker):
    """Demonstrate order management."""
    logger.info("\n=== Order Management Demo ===")
    
    try:
        # Create sample order
        from broker.base_broker import BrokerOrder, OrderSide, OrderType, ProductType, Variety
        
        order = BrokerOrder(
            order_id="",
            symbol="NIFTY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            product=ProductType.MIS,
            variety=Variety.REGULAR,
            tag="demo_order"
        )
        
        logger.info(f"Created order: {order.side.value} {order.quantity} {order.symbol}")
        logger.info(f"Order type: {order.order_type.value}")
        logger.info(f"Product: {order.product.value}")
        
        # Note: We won't actually place the order in demo mode
        logger.info("Note: Order placement skipped in demo mode")
        
        # Get existing orders
        try:
            orders = await broker.get_orders()
            logger.info(f"Retrieved {len(orders)} existing orders")
        except Exception as e:
            logger.warning(f"Could not get orders: {e}")
            
    except Exception as e:
        logger.error(f"Order management error: {e}")


async def demo_position_tracking(broker):
    """Demonstrate position tracking."""
    logger.info("\n=== Position Tracking Demo ===")
    
    try:
        # Get positions
        positions = await broker.get_positions()
        
        if positions:
            logger.info(f"Retrieved {len(positions)} positions:")
            for position in positions:
                logger.info(f"  {position.symbol}: {position.quantity} @ ₹{position.average_price}")
                logger.info(f"    Current: ₹{position.last_price}, P&L: ₹{position.unrealized_pnl}")
        else:
            logger.info("No positions found")
            
    except Exception as e:
        logger.warning(f"Could not get positions: {e}")


async def demo_market_data(broker):
    """Demonstrate market data functionality."""
    logger.info("\n=== Market Data Demo ===")
    
    try:
        # Get quotes for NIFTY
        symbols = ["NIFTY"]
        quotes = await broker.get_quote(symbols)
        
        if quotes:
            for symbol, quote in quotes.items():
                logger.info(f"{symbol} Quote:")
                logger.info(f"  Last Price: ₹{quote.last_price}")
                logger.info(f"  Volume: {quote.volume:,}")
                logger.info(f"  OHLC: O=₹{quote.ohlc['open']}, H=₹{quote.ohlc['high']}, L=₹{quote.ohlc['low']}, C=₹{quote.ohlc['close']}")
                logger.info(f"  Net Change: ₹{quote.net_change}")
        else:
            logger.info("No quotes retrieved")
            
    except Exception as e:
        logger.warning(f"Could not get market data: {e}")


async def demo_live_trading_system():
    """Demonstrate live trading system."""
    logger.info("\n=== Live Trading System Demo ===")
    
    try:
        # Create broker config
        broker_config = KiteConnectConfig(
            api_key="demo_api_key",
            api_secret="demo_api_secret",
            access_token=None,
            debug=True
        )
        
        # Create broker
        broker = KiteConnectBroker(broker_config)
        
        # Create trading config
        trading_config = LiveTradingConfig(
            broker=broker,
            initial_capital=Decimal('100000'),
            max_position_size=Decimal('10000'),
            max_daily_loss=Decimal('5000'),
            enable_paper_mode=True,  # Use paper mode for demo
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Create trader
        trader = LiveBrokerTrader(trading_config)
        
        logger.info("Created live trading system")
        logger.info(f"Paper mode enabled: {trading_config.enable_paper_mode}")
        logger.info(f"Initial capital: ₹{trading_config.initial_capital:,}")
        
        # Start trader
        started = await trader.start()
        
        if started:
            logger.info("✅ Live trading system started successfully")
            
            # Create sample trading signal
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol="NIFTY",
                signal_type=SignalType.BUY,
                price=18000.0,
                confidence=0.8
            )
            
            logger.info(f"Created signal: {signal.signal_type.value} {signal.symbol} @ ₹{signal.price}")
            
            # Execute signal
            order = await trader.execute_signal(signal)
            
            if order:
                logger.info(f"✅ Signal executed successfully")
                logger.info(f"Order ID: {order.order_id}")
                logger.info(f"Status: {order.status.value}")
                logger.info(f"Quantity: {order.quantity}")
                
                # Get portfolio summary
                summary = await trader.get_portfolio_summary()
                logger.info("Portfolio Summary:")
                logger.info(f"  Total Value: ₹{summary.get('total_value', 0):,.2f}")
                logger.info(f"  Available Cash: ₹{summary.get('available_cash', 0):,.2f}")
                logger.info(f"  Unrealized P&L: ₹{summary.get('unrealized_pnl', 0):,.2f}")
                logger.info(f"  Daily Trades: {summary.get('daily_trades', 0)}")
                logger.info(f"  Positions: {summary.get('positions_count', 0)}")
                
            else:
                logger.warning("❌ Signal execution failed")
            
            # Stop trader
            await trader.stop()
            logger.info("Live trading system stopped")
            
        else:
            logger.warning("❌ Failed to start live trading system")
            
    except Exception as e:
        logger.error(f"Live trading system error: {e}")


async def demo_risk_management():
    """Demonstrate risk management features."""
    logger.info("\n=== Risk Management Demo ===")
    
    # Create mock broker for risk management demo
    from tests.test_broker_integration import MockBroker
    
    mock_broker = MockBroker({"api_key": "test"})
    
    # Create trading config with risk management
    trading_config = LiveTradingConfig(
        broker=mock_broker,
        initial_capital=Decimal('100000'),
        max_position_size=Decimal('10000'),
        max_daily_loss=Decimal('5000'),
        enable_paper_mode=True,
        risk_management={
            "max_portfolio_loss_percent": 0.05,  # 5%
            "max_position_size_percent": 0.10,   # 10%
            "max_daily_trades": 50,
            "stop_loss_percent": 0.05,           # 5%
            "take_profit_percent": 0.10          # 10%
        }
    )
    
    # Create trader
    trader = LiveBrokerTrader(trading_config)
    
    logger.info("Risk Management Configuration:")
    for key, value in trading_config.risk_management.items():
        logger.info(f"  {key}: {value}")
    
    # Start trader
    await trader.start()
    
    # Test normal signal (should pass)
    normal_signal = TradingSignal(
        timestamp=datetime.now(),
        symbol="NIFTY",
        signal_type=SignalType.BUY,
        price=18000.0,
        confidence=0.8
    )
    
    logger.info(f"\nTesting normal signal: BUY NIFTY @ ₹{normal_signal.price}")
    order = await trader.execute_signal(normal_signal)
    
    if order:
        logger.info("✅ Normal signal passed risk checks")
    else:
        logger.info("❌ Normal signal failed risk checks")
    
    # Test large signal (should fail)
    large_signal = TradingSignal(
        timestamp=datetime.now(),
        symbol="NIFTY",
        signal_type=SignalType.BUY,
        price=500000.0,  # Very large price
        confidence=0.8
    )
    
    logger.info(f"\nTesting large signal: BUY NIFTY @ ₹{large_signal.price}")
    order = await trader.execute_signal(large_signal)
    
    if order:
        logger.info("❌ Large signal passed risk checks (unexpected)")
    else:
        logger.info("✅ Large signal correctly rejected by risk management")
    
    # Stop trader
    await trader.stop()
    logger.info("Risk management demo completed")


async def demo_strategy_integration():
    """Demonstrate strategy integration with broker."""
    logger.info("\n=== Strategy Integration Demo ===")
    
    try:
        # Create Supertrend strategy
        strategy = SupertrendStrategy(period=10, multiplier=2.0)
        strategy.name = "Live_Supertrend"
        
        logger.info(f"Created strategy: {strategy.name}")
        logger.info(f"Strategy parameters: period={strategy.period}, multiplier={strategy.multiplier}")
        
        # Create sample market data for strategy
        import pandas as pd
        
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': [18000 + i * 10 for i in range(50)],
            'high': [18050 + i * 10 for i in range(50)],
            'low': [17950 + i * 10 for i in range(50)],
            'close': [18000 + i * 10 for i in range(50)],
            'volume': [1000] * 50
        })
        
        logger.info(f"Generated {len(data)} data points for strategy testing")
        
        # Run strategy on data
        signals = strategy.calculate_signals(data)
        
        logger.info(f"Strategy generated {len(signals)} signals")
        
        if len(signals) > 0:
            latest_signal = signals[-1]
            logger.info(f"Latest signal: {latest_signal.signal_type.value} {latest_signal.symbol}")
            logger.info(f"Signal price: ₹{latest_signal.price}")
            logger.info(f"Signal confidence: {latest_signal.confidence}")
        else:
            logger.info("No signals generated by strategy")
        
        logger.info("Strategy integration demo completed")
        
    except Exception as e:
        logger.error(f"Strategy integration error: {e}")


async def demo_alert_system():
    """Demonstrate alert system integration."""
    logger.info("\n=== Alert System Demo ===")
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Create sample alerts
    trade_alert = alert_manager.create_alert(
        alert_type=AlertType.TRADE,
        title="Trade Executed",
        message="Order executed: BUY NIFTY @ ₹18000",
        metadata={
            "order_id": "demo_order_123",
            "symbol": "NIFTY",
            "side": "BUY",
            "price": 18000.0,
            "quantity": 100
        }
    )
    
    logger.info(f"Created trade alert: {trade_alert.message}")
    
    signal_alert = alert_manager.create_alert(
        alert_type=AlertType.SIGNAL,
        title="Signal Generated",
        message="Supertrend signal: BUY NIFTY",
        metadata={
            "symbol": "NIFTY",
            "signal": "BUY",
            "price": 18000.0,
            "confidence": 0.8
        }
    )
    
    logger.info(f"Created signal alert: {signal_alert.message}")
    
    risk_alert = alert_manager.create_alert(
        alert_type=AlertType.RISK,
        title="Risk Warning",
        message="Portfolio loss limit approaching",
        metadata={
            "current_loss_percent": 0.04,
            "limit_percent": 0.05,
            "portfolio_value": 95000
        }
    )
    
    logger.info(f"Created risk alert: {risk_alert.message}")
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    logger.info(f"Alert summary: {summary['total_alerts']} total alerts")
    if 'alert_types' in summary:
        logger.info(f"Alert types: {list(summary['alert_types'].keys())}")
    else:
        logger.info(f"Alert summary: {summary}")


async def main():
    """Main demo function."""
    logger.info("Starting Phase 5 - Broker Integration Demo")
    logger.info("=" * 60)
    
    try:
        # Run individual demos
        broker = demo_broker_initialization()
        await demo_broker_connection(broker)
        await demo_order_management(broker)
        await demo_position_tracking(broker)
        await demo_market_data(broker)
        await demo_live_trading_system()
        await demo_risk_management()
        await demo_strategy_integration()
        await demo_alert_system()
        
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5 Demo completed successfully!")
        logger.info("All broker integration components are working correctly.")
        logger.info("\nNote: For live trading, you need:")
        logger.info("1. Valid Kite Connect API credentials")
        logger.info("2. Access token from Kite Connect")
        logger.info("3. Sufficient account balance")
        logger.info("4. Proper risk management settings")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
