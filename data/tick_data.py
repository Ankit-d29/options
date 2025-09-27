"""
Tick data utilities for loading and processing market data.
"""
import pandas as pd
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import asyncio

from utils.logging_utils import tick_logger
from utils.config import config
from .candle_builder import TickData


class TickDataLoader:
    """Loads and processes tick data from various sources."""
    
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else Path(config.get('data.tick_data_path', 'data/tick_data'))
        self.data_path.mkdir(parents=True, exist_ok=True)
        tick_logger.info(f"Initialized TickDataLoader with path: {self.data_path}")
    
    def save_ticks_csv(self, ticks: List[TickData], filename: str):
        """Save ticks to CSV file."""
        filepath = self.data_path / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'price', 'volume', 'bid', 'ask']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for tick in ticks:
                writer.writerow(tick.to_dict())
        
        tick_logger.info(f"Saved {len(ticks)} ticks to {filepath}")
    
    def load_ticks_csv(self, filename: str) -> List[TickData]:
        """Load ticks from CSV file."""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            tick_logger.warning(f"File not found: {filepath}")
            return []
        
        ticks = []
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tick = TickData(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    symbol=row['symbol'],
                    price=float(row['price']),
                    volume=int(row['volume']),
                    bid=float(row['bid']) if row['bid'] else None,
                    ask=float(row['ask']) if row['ask'] else None
                )
                ticks.append(tick)
        
        tick_logger.info(f"Loaded {len(ticks)} ticks from {filepath}")
        return ticks
    
    def save_ticks_json(self, ticks: List[TickData], filename: str):
        """Save ticks to JSON file."""
        filepath = self.data_path / filename
        
        data = []
        for tick in ticks:
            tick_dict = tick.to_dict()
            tick_dict['timestamp'] = tick_dict['timestamp'].isoformat()
            data.append(tick_dict)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)
        
        tick_logger.info(f"Saved {len(ticks)} ticks to {filepath}")
    
    def load_ticks_json(self, filename: str) -> List[TickData]:
        """Load ticks from JSON file."""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            tick_logger.warning(f"File not found: {filepath}")
            return []
        
        with open(filepath, 'r') as jsonfile:
            data = json.load(jsonfile)
        
        ticks = []
        for item in data:
            tick = TickData(
                timestamp=datetime.fromisoformat(item['timestamp']),
                symbol=item['symbol'],
                price=float(item['price']),
                volume=int(item['volume']),
                bid=float(item['bid']) if item['bid'] else None,
                ask=float(item['ask']) if item['ask'] else None
            )
            ticks.append(tick)
        
        tick_logger.info(f"Loaded {len(ticks)} ticks from {filepath}")
        return ticks
    
    def load_ticks_dataframe(self, filename: str) -> pd.DataFrame:
        """Load ticks as pandas DataFrame."""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            tick_logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        tick_logger.info(f"Loaded DataFrame with {len(df)} ticks from {filepath}")
        return df
    
    def get_available_files(self, pattern: str = "*") -> List[str]:
        """Get list of available tick data files."""
        files = list(self.data_path.glob(pattern))
        return [f.name for f in files if f.is_file()]
    
    def stream_ticks_csv(self, filename: str) -> Iterator[TickData]:
        """Stream ticks from CSV file one by one."""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            tick_logger.warning(f"File not found: {filepath}")
            return
        
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tick = TickData(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    symbol=row['symbol'],
                    price=float(row['price']),
                    volume=int(row['volume']),
                    bid=float(row['bid']) if row['bid'] else None,
                    ask=float(row['ask']) if row['ask'] else None
                )
                yield tick


class MockTickGenerator:
    """Generates mock tick data for testing and simulation."""
    
    def __init__(self, symbol: str = "NIFTY", base_price: float = 18000.0):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        self.tick_count = 0
        tick_logger.info(f"Initialized MockTickGenerator for {symbol} at base price {base_price}")
    
    def generate_tick(self, timestamp: datetime = None) -> TickData:
        """Generate a single mock tick."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Simple random walk with some volatility
        import random
        change_percent = random.uniform(-0.001, 0.001)  # ±0.1%
        self.current_price *= (1 + change_percent)
        
        # Add some bid-ask spread
        spread = self.current_price * 0.0001  # 0.01% spread
        bid = self.current_price - spread/2
        ask = self.current_price + spread/2
        
        # Random volume
        volume = random.randint(1, 1000)
        
        self.tick_count += 1
        
        return TickData(
            timestamp=timestamp,
            symbol=self.symbol,
            price=self.current_price,
            volume=volume,
            bid=bid,
            ask=ask
        )
    
    def generate_ticks(self, count: int, start_time: datetime = None, 
                      interval_seconds: float = 1.0) -> List[TickData]:
        """Generate multiple mock ticks."""
        if start_time is None:
            start_time = datetime.now()
        
        ticks = []
        for i in range(count):
            timestamp = start_time + pd.Timedelta(seconds=i * interval_seconds)
            tick = self.generate_tick(timestamp)
            ticks.append(tick)
        
        tick_logger.info(f"Generated {count} mock ticks for {self.symbol}")
        return ticks
    
    async def generate_tick_stream(self, duration_seconds: int = 60, 
                                  interval_seconds: float = 1.0) -> Iterator[TickData]:
        """Generate a stream of mock ticks."""
        end_time = datetime.now() + pd.Timedelta(seconds=duration_seconds)
        
        while datetime.now() < end_time:
            tick = self.generate_tick()
            yield tick
            await asyncio.sleep(interval_seconds)


# Utility functions
def create_sample_ticks(symbol: str = "NIFTY", count: int = 1000, 
                       base_price: float = 18000.0) -> List[TickData]:
    """Create sample tick data for testing."""
    generator = MockTickGenerator(symbol, base_price)
    return generator.generate_ticks(count)


def ticks_to_dataframe(ticks: List[TickData]) -> pd.DataFrame:
    """Convert list of ticks to pandas DataFrame."""
    data = [tick.to_dict() for tick in ticks]
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def dataframe_to_ticks(df: pd.DataFrame) -> List[TickData]:
    """Convert pandas DataFrame to list of ticks."""
    ticks = []
    for _, row in df.iterrows():
        tick = TickData(
            timestamp=row['timestamp'],
            symbol=row['symbol'],
            price=float(row['price']),
            volume=int(row['volume']),
            bid=float(row['bid']) if pd.notna(row['bid']) else None,
            ask=float(row['ask']) if pd.notna(row['ask']) else None
        )
        ticks.append(tick)
    
    return ticks
