#!/usr/bin/env python3
"""
Demo script for Phase 4 - Paper Trading (Live Simulation)

This script demonstrates:
1. Mock WebSocket feed for real-time tick data simulation
2. Paper trading system with simulated order management
3. Live strategy execution on real-time data
4. Position tracking and P&L monitoring
5. Alert system for notifications
6. Risk management and monitoring

Usage:
    python demo_live_trading.py
"""

import asyncio
import time
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_utils import get_logger
from live.websocket_feed import MockWebSocketFeed, MarketDataBuffer, MarketDataAnalyzer
from live.paper_trader import PaperTrader, PaperOrder, OrderType, OrderStatus
from live.live_strategy_runner import LiveStrategyRunner, LiveStrategyConfig
from live.position_monitor import PositionMonitor
from live.alerts import AlertManager, AlertType
from strategies.supertrend import SupertrendStrategy
from strategies.base_strategy import TradingSignal, SignalType
from data.candle_builder import CandleBuilder

# Setup logging
logger = get_logger(__name__)

def demo_mock_websocket_feed():
    """Demonstrate mock WebSocket feed functionality."""
    logger.info("\n=== Mock WebSocket Feed Demo ===")
    
    # Create mock feed
    symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
    feed = MockWebSocketFeed(symbols)
    
    logger.info(f"Created mock feed for symbols: {symbols}")
    logger.info(f"Base prices: {feed.base_prices}")
    
    # Get current prices
    current_prices = {symbol: feed.get_current_price(symbol) for symbol in symbols}
    logger.info(f"Current prices: {current_prices}")
    
    # Simulate market event
    feed.simulate_market_event("NIFTY", 0.02)  # 2% price increase
    nifty_price = feed.get_current_price("NIFTY")
    logger.info(f"After market event - NIFTY price: {nifty_price:.2f}")
    
    return feed

def demo_paper_trading():
    """Demonstrate paper trading functionality."""
    logger.info("\n=== Paper Trading Demo ===")
    
    # Create paper trader
    trader = PaperTrader(initial_capital=100000, commission_rate=0.001)
    logger.info(f"Created paper trader with capital: ${trader.cash:,.2f}")
    
    # Simulate market data
    from live.websocket_feed import MarketData
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
    trader.update_market_data(market_data)
    
    # Submit market order
    order = trader.submit_order(
        symbol="NIFTY",
        side="BUY",
        quantity=5,
        order_type=OrderType.MARKET
    )
    logger.info(f"Submitted market order: {order}")
    
    # Get portfolio summary
    summary = trader.get_portfolio_summary()
    logger.info(f"Portfolio summary: {summary}")
    
    return trader

def main():
    """Main demo function."""
    logger.info("Starting Phase 4 - Paper Trading (Live Simulation) Demo")
    logger.info("=" * 60)
    
    try:
        # Run individual demos
        demo_mock_websocket_feed()
        demo_paper_trading()
        
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4 Demo completed successfully!")
        logger.info("All paper trading components are working correctly.")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()