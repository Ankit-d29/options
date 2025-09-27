"""
Candle builder for aggregating tick data into OHLCV candles.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils.logging_utils import candle_logger
from utils.config import config


class TickData:
    """Represents a single tick of market data."""
    
    def __init__(self, timestamp: datetime, symbol: str, price: float, 
                 volume: int, bid: Optional[float] = None, ask: Optional[float] = None):
        self.timestamp = timestamp
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.bid = bid
        self.ask = ask
    
    def to_dict(self) -> Dict:
        """Convert tick to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }


class Candle:
    """Represents an OHLCV candle."""
    
    def __init__(self, timestamp: datetime, symbol: str, timeframe: str,
                 open_price: float, high_price: float, low_price: float, 
                 close_price: float, volume: int):
        self.timestamp = timestamp
        self.symbol = symbol
        self.timeframe = timeframe
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume
    
    def to_dict(self) -> Dict:
        """Convert candle to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    def __repr__(self):
        return (f"Candle({self.timestamp}, {self.symbol}, {self.timeframe}, "
                f"O:{self.open}, H:{self.high}, L:{self.low}, C:{self.close}, V:{self.volume})")


class CandleBuilder:
    """Builds OHLCV candles from tick data with configurable timeframes."""
    
    def __init__(self, timeframes: List[str] = None):
        if timeframes is None:
            self.timeframes = config.get('candle_builder.timeframes', ['1m', '5m'])
        else:
            self.timeframes = timeframes
        self.timeframe_seconds = self._parse_timeframes()
        self.active_candles = {}  # {symbol: {timeframe: Candle}}
        self.executor = ThreadPoolExecutor(max_workers=4)
        candle_logger.info(f"Initialized CandleBuilder with timeframes: {self.timeframes}")
    
    def _parse_timeframes(self) -> Dict[str, int]:
        """Parse timeframe strings to seconds."""
        timeframe_map = {
            '1s': 1,
            '5s': 5,
            '10s': 10,
            '30s': 30,
            '1m': 60,
            '2m': 120,
            '3m': 180,
            '5m': 300,
            '10m': 600,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '1d': 86400
        }
        
        parsed = {}
        for tf in self.timeframes:
            if tf in timeframe_map:
                parsed[tf] = timeframe_map[tf]
            else:
                candle_logger.warning(f"Unknown timeframe: {tf}, skipping")
        
        return parsed
    
    def _get_candle_timestamp(self, tick_timestamp: datetime, timeframe: str) -> datetime:
        """Get the start timestamp for a candle containing the given tick."""
        seconds = self.timeframe_seconds[timeframe]
        
        # Round down to the nearest timeframe boundary
        timestamp_seconds = int(tick_timestamp.timestamp())
        candle_start_seconds = (timestamp_seconds // seconds) * seconds
        
        return datetime.fromtimestamp(candle_start_seconds)
    
    def _initialize_candle(self, tick: TickData, timeframe: str) -> Candle:
        """Initialize a new candle with the given tick."""
        candle_timestamp = self._get_candle_timestamp(tick.timestamp, timeframe)
        
        return Candle(
            timestamp=candle_timestamp,
            symbol=tick.symbol,
            timeframe=timeframe,
            open_price=tick.price,
            high_price=tick.price,
            low_price=tick.price,
            close_price=tick.price,
            volume=tick.volume
        )
    
    def _update_candle(self, candle: Candle, tick: TickData):
        """Update an existing candle with new tick data."""
        candle.high = max(candle.high, tick.price)
        candle.low = min(candle.low, tick.price)
        candle.close = tick.price
        candle.volume += tick.volume
    
    def process_tick(self, tick: TickData) -> List[Candle]:
        """Process a single tick and return completed candles."""
        completed_candles = []
        
        if tick.symbol not in self.active_candles:
            self.active_candles[tick.symbol] = {}
        
        symbol_candles = self.active_candles[tick.symbol]
        
        for timeframe in self.timeframes:
            candle_timestamp = self._get_candle_timestamp(tick.timestamp, timeframe)
            
            # Check if we need a new candle
            if (timeframe not in symbol_candles or 
                symbol_candles[timeframe].timestamp != candle_timestamp):
                
                # Complete the previous candle if it exists
                if timeframe in symbol_candles:
                    completed_candles.append(symbol_candles[timeframe])
                
                # Start a new candle
                symbol_candles[timeframe] = self._initialize_candle(tick, timeframe)
            else:
                # Update existing candle
                self._update_candle(symbol_candles[timeframe], tick)
        
        return completed_candles
    
    def process_ticks_batch(self, ticks: List[TickData]) -> List[Candle]:
        """Process a batch of ticks and return all completed candles."""
        all_completed_candles = []
        
        for tick in ticks:
            completed = self.process_tick(tick)
            all_completed_candles.extend(completed)
        
        return all_completed_candles
    
    def get_current_candles(self, symbol: str = None) -> Dict:
        """Get current active candles."""
        if symbol:
            return self.active_candles.get(symbol, {})
        return self.active_candles
    
    def finalize_candles(self) -> List[Candle]:
        """Finalize all active candles and return them."""
        all_candles = []
        
        for symbol_candles in self.active_candles.values():
            all_candles.extend(symbol_candles.values())
        
        # Clear active candles
        self.active_candles.clear()
        
        return all_candles
    
    async def process_tick_async(self, tick: TickData) -> List[Candle]:
        """Asynchronously process a single tick."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_tick, tick)
    
    async def process_ticks_batch_async(self, ticks: List[TickData]) -> List[Candle]:
        """Asynchronously process a batch of ticks."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_ticks_batch, ticks)
    
    def save_candles(self, candles: List[Candle], output_dir: str = None):
        """Save candles to CSV files."""
        if not candles:
            return
        
        output_path = Path(output_dir) if output_dir else Path(config.get('data.candles_path', 'data/candles'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group candles by symbol and timeframe
        grouped_candles = {}
        for candle in candles:
            key = f"{candle.symbol}_{candle.timeframe}"
            if key not in grouped_candles:
                grouped_candles[key] = []
            grouped_candles[key].append(candle)
        
        # Save each group to separate CSV file
        for key, candle_list in grouped_candles.items():
            df = pd.DataFrame([candle.to_dict() for candle in candle_list])
            df = df.sort_values('timestamp')
            
            filename = f"{key}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = output_path / filename
            
            # Append to existing file or create new one
            if filepath.exists():
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
            
            candle_logger.info(f"Saved {len(candle_list)} candles to {filepath}")
    
    def load_candles(self, symbol: str, timeframe: str, 
                    start_date: datetime = None, end_date: datetime = None, 
                    data_path: str = None) -> pd.DataFrame:
        """Load candles from CSV files."""
        if data_path:
            candles_path = Path(data_path)
        else:
            candles_path = Path(config.get('data.candles_path', 'data/candles'))
        
        # Find matching files
        pattern = f"{symbol}_{timeframe}_*.csv"
        files = list(candles_path.glob(pattern))
        
        if not files:
            candle_logger.warning(f"No candle files found for {symbol}_{timeframe}")
            return pd.DataFrame()
        
        # Load and combine all files
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        # Filter by date range if specified
        if start_date:
            combined_df = combined_df[combined_df['timestamp'] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df['timestamp'] <= end_date]
        
        candle_logger.info(f"Loaded {len(combined_df)} candles for {symbol}_{timeframe}")
        return combined_df
    
    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
