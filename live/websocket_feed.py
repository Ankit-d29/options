"""
WebSocket feed for real-time market data simulation.
"""
import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
import threading

from utils.logging_utils import get_logger
from data.tick_data import TickData

# Set up logging
websocket_logger = get_logger('websocket_feed')


@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    
    def to_tick_data(self) -> TickData:
        """Convert to TickData format."""
        return TickData(
            symbol=self.symbol,
            timestamp=self.timestamp,
            price=self.price,
            volume=self.volume
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size
        }


class WebSocketFeed:
    """Base class for WebSocket market data feeds."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.subscribers: List[Callable[[MarketData], None]] = []
        self.is_running = False
        self.connection_status = "disconnected"
        
        websocket_logger.info(f"Initialized WebSocketFeed for symbols: {symbols}")
    
    def subscribe(self, callback: Callable[[MarketData], None]):
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
        websocket_logger.info(f"Added subscriber. Total subscribers: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable[[MarketData], None]):
        """Unsubscribe from market data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            websocket_logger.info(f"Removed subscriber. Total subscribers: {len(self.subscribers)}")
    
    def _notify_subscribers(self, data: MarketData):
        """Notify all subscribers of new market data."""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                websocket_logger.error(f"Error in subscriber callback: {e}")
    
    async def start(self):
        """Start the WebSocket feed."""
        raise NotImplementedError("Subclasses must implement start method")
    
    async def stop(self):
        """Stop the WebSocket feed."""
        self.is_running = False
        self.connection_status = "disconnected"
        websocket_logger.info("WebSocket feed stopped")
    
    def get_connection_status(self) -> str:
        """Get current connection status."""
        return self.connection_status


class MockWebSocketFeed(WebSocketFeed):
    """Mock WebSocket feed that simulates real-time market data."""
    
    def __init__(self, symbols: List[str], update_interval: float = 0.1, 
                 base_prices: Dict[str, float] = None):
        super().__init__(symbols)
        self.update_interval = update_interval
        self.base_prices = base_prices or {symbol: 18000.0 for symbol in symbols}
        self.current_prices = self.base_prices.copy()
        self.volatility = {symbol: 0.001 for symbol in symbols}  # 0.1% volatility
        self.trend = {symbol: 0.0001 for symbol in symbols}  # Slight upward trend
        self.tick_counter = {symbol: 0 for symbol in symbols}
        
        websocket_logger.info(f"Initialized MockWebSocketFeed with {len(symbols)} symbols")
    
    async def start(self):
        """Start the mock WebSocket feed."""
        self.is_running = True
        self.connection_status = "connected"
        websocket_logger.info("MockWebSocketFeed started")
        
        # Start data generation task
        asyncio.create_task(self._generate_data())
    
    async def _generate_data(self):
        """Generate mock market data."""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    await self._generate_symbol_data(symbol)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                websocket_logger.error(f"Error generating market data: {e}")
                await asyncio.sleep(1)
    
    async def _generate_symbol_data(self, symbol: str):
        """Generate market data for a specific symbol."""
        # Update price with random walk
        current_price = self.current_prices[symbol]
        
        # Add trend and volatility
        trend_change = self.trend[symbol] * current_price
        volatility_change = random.gauss(0, self.volatility[symbol] * current_price)
        
        # Add some momentum (price tends to continue in current direction)
        momentum = 0.1 * random.uniform(-1, 1) * abs(volatility_change)
        
        new_price = current_price + trend_change + volatility_change + momentum
        
        # Ensure price doesn't go negative
        new_price = max(new_price, current_price * 0.5)
        
        self.current_prices[symbol] = new_price
        self.tick_counter[symbol] += 1
        
        # Generate bid/ask spread
        spread = new_price * 0.0005  # 0.05% spread
        bid = new_price - spread / 2
        ask = new_price + spread / 2
        
        # Generate volume (higher volume during volatile periods)
        base_volume = 100
        volume_multiplier = 1 + abs(volatility_change) / current_price * 10
        volume = int(base_volume * volume_multiplier * random.uniform(0.5, 2.0))
        
        # Create market data
        market_data = MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=new_price,
            volume=volume,
            bid=bid,
            ask=ask,
            bid_size=random.randint(50, 500),
            ask_size=random.randint(50, 500)
        )
        
        # Notify subscribers
        self._notify_subscribers(market_data)
    
    def set_volatility(self, symbol: str, volatility: float):
        """Set volatility for a specific symbol."""
        if symbol in self.symbols:
            self.volatility[symbol] = volatility
            websocket_logger.info(f"Set volatility for {symbol}: {volatility:.4f}")
    
    def set_trend(self, symbol: str, trend: float):
        """Set trend for a specific symbol."""
        if symbol in self.symbols:
            self.trend[symbol] = trend
            websocket_logger.info(f"Set trend for {symbol}: {trend:.6f}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return self.current_prices.get(symbol)
    
    def get_tick_count(self, symbol: str) -> int:
        """Get tick count for a symbol."""
        return self.tick_counter.get(symbol, 0)
    
    def reset_prices(self):
        """Reset prices to base prices."""
        self.current_prices = self.base_prices.copy()
        websocket_logger.info("Reset prices to base prices")
    
    def simulate_market_event(self, symbol: str, price_change_percent: float):
        """Simulate a market event (e.g., news, earnings)."""
        if symbol in self.current_prices:
            current_price = self.current_prices[symbol]
            new_price = current_price * (1 + price_change_percent)
            self.current_prices[symbol] = new_price
            
            websocket_logger.info(f"Simulated market event for {symbol}: "
                                f"{price_change_percent:+.2%} price change")


class LiveWebSocketFeed(WebSocketFeed):
    """Real WebSocket feed for live market data (placeholder for future implementation)."""
    
    def __init__(self, symbols: List[str], api_key: str = None, api_secret: str = None):
        super().__init__(symbols)
        self.api_key = api_key
        self.api_secret = api_secret
        
        websocket_logger.info("Initialized LiveWebSocketFeed (placeholder)")
    
    async def start(self):
        """Start the live WebSocket feed."""
        self.is_running = True
        self.connection_status = "connecting"
        
        try:
            # TODO: Implement real WebSocket connection
            # This would connect to a real broker's WebSocket API
            websocket_logger.info("Connecting to live WebSocket feed...")
            
            # Simulate connection
            await asyncio.sleep(1)
            self.connection_status = "connected"
            websocket_logger.info("Connected to live WebSocket feed")
            
            # TODO: Start listening for real market data
            # asyncio.create_task(self._listen_for_data())
            
        except Exception as e:
            self.connection_status = "error"
            websocket_logger.error(f"Failed to connect to live WebSocket feed: {e}")
    
    async def _listen_for_data(self):
        """Listen for real market data from WebSocket."""
        # TODO: Implement real WebSocket data listening
        pass


class MarketDataBuffer:
    """Buffer for storing recent market data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data: List[MarketData] = []
        self.lock = threading.Lock()
        
    def add_data(self, data: MarketData):
        """Add new market data."""
        with self.lock:
            self.data.append(data)
            if len(self.data) > self.max_size:
                self.data.pop(0)
    
    def get_latest_data(self, symbol: str = None, count: int = 1) -> List[MarketData]:
        """Get latest market data."""
        with self.lock:
            if symbol:
                symbol_data = [d for d in self.data if d.symbol == symbol]
                return symbol_data[-count:] if symbol_data else []
            else:
                return self.data[-count:] if self.data else []
    
    def get_data_range(self, symbol: str, start_time: datetime, 
                      end_time: datetime) -> List[MarketData]:
        """Get market data within time range."""
        with self.lock:
            return [d for d in self.data 
                   if d.symbol == symbol and start_time <= d.timestamp <= end_time]
    
    def clear(self):
        """Clear all data."""
        with self.lock:
            self.data.clear()
    
    def get_count(self) -> int:
        """Get total data count."""
        with self.lock:
            return len(self.data)


class MarketDataAnalyzer:
    """Analyze market data for patterns and statistics."""
    
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
    
    def get_price_statistics(self, symbol: str, period_minutes: int = 5) -> Dict[str, Any]:
        """Get price statistics for a symbol over a period."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=period_minutes)
        
        data = self.buffer.get_data_range(symbol, start_time, end_time)
        
        if not data:
            return {}
        
        prices = [d.price for d in data]
        volumes = [d.volume for d in data]
        
        return {
            'symbol': symbol,
            'period_minutes': period_minutes,
            'data_points': len(data),
            'current_price': prices[-1] if prices else 0,
            'min_price': min(prices) if prices else 0,
            'max_price': max(prices) if prices else 0,
            'avg_price': sum(prices) / len(prices) if prices else 0,
            'price_change': prices[-1] - prices[0] if len(prices) >= 2 else 0,
            'price_change_percent': (prices[-1] - prices[0]) / prices[0] * 100 if len(prices) >= 2 and prices[0] != 0 else 0,
            'volatility': self._calculate_volatility(prices),
            'total_volume': sum(volumes),
            'avg_volume': sum(volumes) / len(volumes) if volumes else 0
        }
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return (sum(r**2 for r in returns) / len(returns))**0.5 * 100
    
    def detect_volume_spikes(self, symbol: str, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect volume spikes."""
        recent_data = self.buffer.get_latest_data(symbol, 100)
        
        if len(recent_data) < 10:
            return []
        
        volumes = [d.volume for d in recent_data]
        avg_volume = sum(volumes) / len(volumes)
        
        spikes = []
        for data in recent_data:
            if data.volume > avg_volume * threshold:
                spikes.append({
                    'timestamp': data.timestamp,
                    'price': data.price,
                    'volume': data.volume,
                    'avg_volume': avg_volume,
                    'spike_ratio': data.volume / avg_volume
                })
        
        return spikes
