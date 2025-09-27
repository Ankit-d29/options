"""
Live strategy runner for real-time trading simulation.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from utils.logging_utils import get_logger
from utils.config import config
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from .websocket_feed import WebSocketFeed, MarketData
from .paper_trader import PaperTrader, PaperOrder, OrderType
from data.candle_builder import CandleBuilder

# Set up logging
live_strategy_logger = get_logger('live_strategy_runner')


@dataclass
class LiveStrategyConfig:
    """Configuration for live strategy execution."""
    symbol: str
    strategy: BaseStrategy
    timeframe: str = '1m'
    max_positions: int = 1
    position_size: int = 100
    stop_loss_percent: float = 0.05
    take_profit_percent: float = 0.10
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveStrategyResult:
    """Result of live strategy execution."""
    config: LiveStrategyConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    total_signals: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    status: str = "RUNNING"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiveStrategyRunner:
    """Runs trading strategies in real-time with live market data."""
    
    def __init__(self, paper_trader: PaperTrader, websocket_feed: WebSocketFeed):
        self.paper_trader = paper_trader
        self.websocket_feed = websocket_feed
        self.strategies: Dict[str, LiveStrategyConfig] = {}
        self.results: Dict[str, LiveStrategyResult] = {}
        self.is_running = False
        
        # Candle builders for each symbol/timeframe combination
        self.candle_builders: Dict[str, CandleBuilder] = {}
        
        # Event callbacks
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []
        self.trade_callbacks: List[Callable[[PaperOrder], None]] = []
        
        live_strategy_logger.info("Initialized LiveStrategyRunner")
    
    def add_strategy(self, config: LiveStrategyConfig):
        """Add a strategy to the runner."""
        strategy_key = f"{config.symbol}_{config.strategy.name}_{config.timeframe}"
        self.strategies[strategy_key] = config
        
        # Initialize candle builder
        candle_key = f"{config.symbol}_{config.timeframe}"
        if candle_key not in self.candle_builders:
            self.candle_builders[candle_key] = CandleBuilder([config.timeframe])
        
        # Initialize result tracking
        self.results[strategy_key] = LiveStrategyResult(
            config=config,
            start_time=datetime.now()
        )
        
        live_strategy_logger.info(f"Added strategy: {strategy_key}")
    
    def remove_strategy(self, strategy_key: str):
        """Remove a strategy from the runner."""
        if strategy_key in self.strategies:
            del self.strategies[strategy_key]
            if strategy_key in self.results:
                self.results[strategy_key].end_time = datetime.now()
                self.results[strategy_key].status = "STOPPED"
            live_strategy_logger.info(f"Removed strategy: {strategy_key}")
    
    async def start(self):
        """Start the live strategy runner."""
        if self.is_running:
            live_strategy_logger.warning("LiveStrategyRunner already running")
            return
        
        self.is_running = True
        live_strategy_logger.info("Starting LiveStrategyRunner")
        
        # Subscribe to market data
        self.websocket_feed.subscribe(self._handle_market_data)
        
        # Start strategy execution task
        asyncio.create_task(self._run_strategies())
        
        live_strategy_logger.info("LiveStrategyRunner started")
    
    async def stop(self):
        """Stop the live strategy runner."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Unsubscribe from market data
        self.websocket_feed.unsubscribe(self._handle_market_data)
        
        # Stop all strategies
        for strategy_key, result in self.results.items():
            result.end_time = datetime.now()
            result.status = "STOPPED"
        
        live_strategy_logger.info("LiveStrategyRunner stopped")
    
    async def _run_strategies(self):
        """Main strategy execution loop."""
        while self.is_running:
            try:
                await self._process_strategies()
                await asyncio.sleep(0.1)  # 100ms loop
            except Exception as e:
                live_strategy_logger.error(f"Error in strategy execution loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_strategies(self):
        """Process all active strategies."""
        for strategy_key, config in self.strategies.items():
            if not config.enabled:
                continue
            
            try:
                await self._process_strategy(strategy_key, config)
            except Exception as e:
                live_strategy_logger.error(f"Error processing strategy {strategy_key}: {e}")
    
    async def _process_strategy(self, strategy_key: str, config: LiveStrategyConfig):
        """Process a single strategy."""
        # Get recent candles for the strategy
        candle_key = f"{config.symbol}_{config.timeframe}"
        candle_builder = self.candle_builders.get(candle_key)
        
        if not candle_builder:
            return
        
        # Get recent candles (last 100 for strategy calculation)
        recent_candles = candle_builder.get_recent_candles(config.symbol, config.timeframe, 100)
        
        if len(recent_candles) < 10:  # Need minimum data for strategy
            return
        
        # Convert to DataFrame for strategy
        import pandas as pd
        candles_df = pd.DataFrame([candle.to_dict() for candle in recent_candles])
        
        if candles_df.empty:
            return
        
        # Run strategy
        try:
            result_df = config.strategy.calculate_signals(candles_df)
            
            # Get latest signals
            latest_signals = config.strategy.get_signals()
            if not latest_signals:
                return
            
            # Process latest signal
            latest_signal = latest_signals[-1]
            
            # Check if signal is recent (within last minute)
            if (datetime.now() - latest_signal.timestamp).total_seconds() > 60:
                return
            
            # Process the signal
            await self._process_signal(strategy_key, config, latest_signal)
            
        except Exception as e:
            live_strategy_logger.error(f"Error running strategy {config.strategy.name}: {e}")
    
    async def _process_signal(self, strategy_key: str, config: LiveStrategyConfig, signal: TradingSignal):
        """Process a trading signal."""
        result = self.results[strategy_key]
        result.total_signals += 1
        
        # Check if we can trade (max positions limit)
        current_positions = len([p for p in self.paper_trader.positions.values() 
                               if p.symbol == config.symbol])
        
        if current_positions >= config.max_positions:
            live_strategy_logger.debug(f"Max positions reached for {config.symbol}: {current_positions}")
            return
        
        # Determine order side
        if signal.signal_type == SignalType.BUY:
            order_side = 'BUY'
        elif signal.signal_type == SignalType.SELL:
            order_side = 'SELL'
        else:
            return  # Skip HOLD signals
        
        # Submit order
        order = self.paper_trader.submit_order(
            symbol=config.symbol,
            side=order_side,
            quantity=config.position_size,
            order_type=OrderType.MARKET,
            metadata={
                'strategy': config.strategy.name,
                'signal_confidence': signal.confidence,
                'signal_timestamp': signal.timestamp.isoformat()
            }
        )
        
        if order.status.value not in ['REJECTED']:
            result.total_trades += 1
            live_strategy_logger.info(f"Order submitted: {order_side} {config.position_size} {config.symbol} "
                                    f"(Strategy: {config.strategy.name}, Confidence: {signal.confidence:.2f})")
        
        # Notify signal callbacks
        self._notify_signal_callbacks(signal)
    
    def _handle_market_data(self, market_data: MarketData):
        """Handle incoming market data."""
        # Update paper trader
        self.paper_trader.update_market_data(market_data)
        
        # Update candle builders
        candle_key = f"{market_data.symbol}_1m"  # Default to 1m
        if candle_key in self.candle_builders:
            tick_data = market_data.to_tick_data()
            self.candle_builders[candle_key].process_tick(tick_data)
        
        # Update other timeframes if needed
        for strategy_key, config in self.strategies.items():
            if config.symbol == market_data.symbol:
                candle_key = f"{config.symbol}_{config.timeframe}"
                if candle_key in self.candle_builders:
                    tick_data = market_data.to_tick_data()
                    self.candle_builders[candle_key].process_tick(tick_data)
    
    def get_strategy_results(self, strategy_key: str = None) -> Dict[str, LiveStrategyResult]:
        """Get strategy results."""
        if strategy_key:
            return {strategy_key: self.results.get(strategy_key)} if strategy_key in self.results else {}
        return self.results.copy()
    
    def get_strategy_summary(self, strategy_key: str) -> Dict[str, Any]:
        """Get summary for a specific strategy."""
        if strategy_key not in self.results:
            return {}
        
        result = self.results[strategy_key]
        config = result.config
        
        # Get current position
        position = self.paper_trader.get_position(config.symbol)
        
        # Get recent trades for this symbol
        recent_trades = [t for t in self.paper_trader.get_recent_trades(20) 
                        if t.symbol == config.symbol]
        
        return {
            'strategy_key': strategy_key,
            'symbol': config.symbol,
            'strategy_name': config.strategy.name,
            'timeframe': config.timeframe,
            'status': result.status,
            'start_time': result.start_time,
            'end_time': result.end_time,
            'total_signals': result.total_signals,
            'total_trades': result.total_trades,
            'current_position': position.to_dict() if position else None,
            'recent_trades_count': len(recent_trades),
            'portfolio_value': self.paper_trader.get_portfolio_value(),
            'enabled': config.enabled
        }
    
    def enable_strategy(self, strategy_key: str):
        """Enable a strategy."""
        if strategy_key in self.strategies:
            self.strategies[strategy_key].enabled = True
            live_strategy_logger.info(f"Enabled strategy: {strategy_key}")
    
    def disable_strategy(self, strategy_key: str):
        """Disable a strategy."""
        if strategy_key in self.strategies:
            self.strategies[strategy_key].enabled = False
            live_strategy_logger.info(f"Disabled strategy: {strategy_key}")
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """Add signal event callback."""
        self.signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[PaperOrder], None]):
        """Add trade event callback."""
        self.trade_callbacks.append(callback)
    
    def _notify_signal_callbacks(self, signal: TradingSignal):
        """Notify signal callbacks."""
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                live_strategy_logger.error(f"Error in signal callback: {e}")
    
    def _notify_trade_callbacks(self, order: PaperOrder):
        """Notify trade callbacks."""
        for callback in self.trade_callbacks:
            try:
                callback(order)
            except Exception as e:
                live_strategy_logger.error(f"Error in trade callback: {e}")
    
    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all strategies."""
        summaries = []
        for strategy_key in self.strategies.keys():
            summary = self.get_strategy_summary(strategy_key)
            if summary:
                summaries.append(summary)
        return summaries
