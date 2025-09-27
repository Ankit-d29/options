"""
Unit tests for the candle builder functionality.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from data.candle_builder import CandleBuilder, TickData, Candle
from data.tick_data import MockTickGenerator


class TestTickData:
    """Test cases for TickData class."""
    
    def test_tick_data_creation(self):
        """Test creating a TickData object."""
        timestamp = datetime.now()
        tick = TickData(
            timestamp=timestamp,
            symbol="NIFTY",
            price=18000.0,
            volume=100,
            bid=17999.5,
            ask=18000.5
        )
        
        assert tick.timestamp == timestamp
        assert tick.symbol == "NIFTY"
        assert tick.price == 18000.0
        assert tick.volume == 100
        assert tick.bid == 17999.5
        assert tick.ask == 18000.5
    
    def test_tick_data_to_dict(self):
        """Test converting TickData to dictionary."""
        timestamp = datetime.now()
        tick = TickData(
            timestamp=timestamp,
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        tick_dict = tick.to_dict()
        expected = {
            'timestamp': timestamp,
            'symbol': 'NIFTY',
            'price': 18000.0,
            'volume': 100,
            'bid': None,
            'ask': None
        }
        
        assert tick_dict == expected


class TestCandle:
    """Test cases for Candle class."""
    
    def test_candle_creation(self):
        """Test creating a Candle object."""
        timestamp = datetime.now()
        candle = Candle(
            timestamp=timestamp,
            symbol="NIFTY",
            timeframe="1m",
            open_price=18000.0,
            high_price=18050.0,
            low_price=17950.0,
            close_price=18025.0,
            volume=1000
        )
        
        assert candle.timestamp == timestamp
        assert candle.symbol == "NIFTY"
        assert candle.timeframe == "1m"
        assert candle.open == 18000.0
        assert candle.high == 18050.0
        assert candle.low == 17950.0
        assert candle.close == 18025.0
        assert candle.volume == 1000
    
    def test_candle_to_dict(self):
        """Test converting Candle to dictionary."""
        timestamp = datetime.now()
        candle = Candle(
            timestamp=timestamp,
            symbol="NIFTY",
            timeframe="1m",
            open_price=18000.0,
            high_price=18050.0,
            low_price=17950.0,
            close_price=18025.0,
            volume=1000
        )
        
        candle_dict = candle.to_dict()
        expected = {
            'timestamp': timestamp,
            'symbol': 'NIFTY',
            'timeframe': '1m',
            'open': 18000.0,
            'high': 18050.0,
            'low': 17950.0,
            'close': 18025.0,
            'volume': 1000
        }
        
        assert candle_dict == expected


class TestCandleBuilder:
    """Test cases for CandleBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeframes = ['1m', '5m']
        self.builder = CandleBuilder(timeframes=self.timeframes)
    
    def test_candle_builder_initialization(self):
        """Test CandleBuilder initialization."""
        assert self.builder.timeframes == self.timeframes
        assert '1m' in self.builder.timeframe_seconds
        assert '5m' in self.builder.timeframe_seconds
        assert self.builder.timeframe_seconds['1m'] == 60
        assert self.builder.timeframe_seconds['5m'] == 300
        assert isinstance(self.builder.active_candles, dict)
    
    def test_get_candle_timestamp(self):
        """Test getting candle timestamp for a given tick."""
        # Test with 1-minute timeframe
        tick_time = datetime(2024, 1, 1, 10, 30, 45)  # 10:30:45
        candle_time = self.builder._get_candle_timestamp(tick_time, '1m')
        expected_time = datetime(2024, 1, 1, 10, 30, 0)  # 10:30:00
        
        assert candle_time == expected_time
        
        # Test with 5-minute timeframe
        candle_time_5m = self.builder._get_candle_timestamp(tick_time, '5m')
        expected_time_5m = datetime(2024, 1, 1, 10, 30, 0)  # 10:30:00
        
        assert candle_time_5m == expected_time_5m
        
        # Test edge case - tick at exact minute boundary
        tick_time_boundary = datetime(2024, 1, 1, 10, 30, 0)
        candle_time_boundary = self.builder._get_candle_timestamp(tick_time_boundary, '1m')
        
        assert candle_time_boundary == tick_time_boundary
    
    def test_initialize_candle(self):
        """Test initializing a new candle."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        candle = self.builder._initialize_candle(tick, '1m')
        
        assert candle.timestamp == datetime(2024, 1, 1, 10, 30, 0)
        assert candle.symbol == "NIFTY"
        assert candle.timeframe == "1m"
        assert candle.open == 18000.0
        assert candle.high == 18000.0
        assert candle.low == 18000.0
        assert candle.close == 18000.0
        assert candle.volume == 100
    
    def test_update_candle(self):
        """Test updating an existing candle."""
        # Create initial candle
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            timeframe="1m",
            open_price=18000.0,
            high_price=18000.0,
            low_price=18000.0,
            close_price=18000.0,
            volume=100
        )
        
        # Update with higher price
        tick_high = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 30),
            symbol="NIFTY",
            price=18050.0,
            volume=200
        )
        
        self.builder._update_candle(candle, tick_high)
        
        assert candle.high == 18050.0
        assert candle.close == 18050.0
        assert candle.volume == 300
        
        # Update with lower price
        tick_low = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 45),
            symbol="NIFTY",
            price=17950.0,
            volume=150
        )
        
        self.builder._update_candle(candle, tick_low)
        
        assert candle.high == 18050.0  # Should remain high
        assert candle.low == 17950.0   # Should be updated
        assert candle.close == 17950.0  # Should be latest price
        assert candle.volume == 450     # Should accumulate
    
    def test_process_single_tick(self):
        """Test processing a single tick."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        completed_candles = self.builder.process_tick(tick)
        
        # Should not complete any candles on first tick
        assert len(completed_candles) == 0
        
        # Check that active candles were created
        assert "NIFTY" in self.builder.active_candles
        assert "1m" in self.builder.active_candles["NIFTY"]
        assert "5m" in self.builder.active_candles["NIFTY"]
        
        candle_1m = self.builder.active_candles["NIFTY"]["1m"]
        assert candle_1m.open == 18000.0
        assert candle_1m.close == 18000.0
    
    def test_process_tick_completes_candle(self):
        """Test processing tick that completes a candle."""
        # First tick - starts candle
        tick1 = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        completed1 = self.builder.process_tick(tick1)
        assert len(completed1) == 0
        
        # Second tick - in new candle timeframe
        tick2 = TickData(
            timestamp=datetime(2024, 1, 1, 10, 31, 0),
            symbol="NIFTY",
            price=18050.0,
            volume=200
        )
        
        completed2 = self.builder.process_tick(tick2)
        
        # Should complete the 1m candle
        assert len(completed2) == 1
        completed_candle = completed2[0]
        assert completed_candle.timeframe == "1m"
        assert completed_candle.open == 18000.0
        assert completed_candle.close == 18000.0  # Only one tick in this candle
    
    def test_process_ticks_batch(self):
        """Test processing a batch of ticks."""
        # Create ticks spanning multiple candles
        ticks = []
        base_time = datetime(2024, 1, 1, 10, 30, 0)
        
        for i in range(5):
            tick = TickData(
                timestamp=base_time + timedelta(minutes=i),
                symbol="NIFTY",
                price=18000.0 + (i * 10),
                volume=100 + (i * 10)
            )
            ticks.append(tick)
        
        completed_candles = self.builder.process_ticks_batch(ticks)
        
        # Should complete 4 candles (one for each minute transition)
        assert len(completed_candles) == 4
        
        # Check first completed candle
        first_candle = completed_candles[0]
        assert first_candle.timeframe == "1m"
        assert first_candle.timestamp == datetime(2024, 1, 1, 10, 30, 0)
    
    def test_multiple_symbols(self):
        """Test processing ticks for multiple symbols."""
        tick_nifty = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        tick_banknifty = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="BANKNIFTY",
            price=45000.0,
            volume=200
        )
        
        self.builder.process_tick(tick_nifty)
        self.builder.process_tick(tick_banknifty)
        
        # Check both symbols have active candles
        assert "NIFTY" in self.builder.active_candles
        assert "BANKNIFTY" in self.builder.active_candles
        
        nifty_candle = self.builder.active_candles["NIFTY"]["1m"]
        banknifty_candle = self.builder.active_candles["BANKNIFTY"]["1m"]
        
        assert nifty_candle.open == 18000.0
        assert banknifty_candle.open == 45000.0
    
    def test_finalize_candles(self):
        """Test finalizing all active candles."""
        # Add some active candles
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        self.builder.process_tick(tick)
        
        # Finalize candles
        finalized_candles = self.builder.finalize_candles()
        
        # Should return all active candles
        assert len(finalized_candles) == 2  # 1m and 5m for NIFTY
        
        # Active candles should be cleared
        assert len(self.builder.active_candles) == 0
    
    def test_save_and_load_candles(self):
        """Test saving and loading candles."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test candles
            candles = []
            base_time = datetime(2024, 1, 1, 10, 30, 0)
            
            for i in range(3):
                candle = Candle(
                    timestamp=base_time + timedelta(minutes=i),
                    symbol="NIFTY",
                    timeframe="1m",
                    open_price=18000.0 + (i * 10),
                    high_price=18050.0 + (i * 10),
                    low_price=17950.0 + (i * 10),
                    close_price=18025.0 + (i * 10),
                    volume=1000 + (i * 100)
                )
                candles.append(candle)
            
            # Save candles
            self.builder.save_candles(candles, temp_dir)
            
            # Load candles from the temp directory
            builder_temp = CandleBuilder(['1m'])
            loaded_df = builder_temp.load_candles("NIFTY", "1m", data_path=temp_dir)
            
            assert len(loaded_df) == 3
            assert list(loaded_df['symbol'].unique()) == ["NIFTY"]
            assert list(loaded_df['timeframe'].unique()) == ["1m"]
    
    def test_mock_data_integration(self):
        """Test integration with mock tick generator."""
        generator = MockTickGenerator("NIFTY", 18000.0)
        ticks = generator.generate_ticks(100)  # Generate 100 ticks
        
        builder = CandleBuilder(['1m'])
        completed_candles = builder.process_ticks_batch(ticks)
        
        # Should have some completed candles
        assert len(completed_candles) > 0
        
        # All candles should be valid
        for candle in completed_candles:
            assert candle.symbol == "NIFTY"
            assert candle.timeframe == "1m"
            assert candle.high >= candle.low
            assert candle.volume > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_volume_tick(self):
        """Test handling ticks with zero volume."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=0
        )
        
        builder = CandleBuilder(['1m'])
        completed_candles = builder.process_tick(tick)
        
        # Should still create candle
        assert "NIFTY" in builder.active_candles
        candle = builder.active_candles["NIFTY"]["1m"]
        assert candle.volume == 0
    
    def test_negative_price_tick(self):
        """Test handling ticks with negative price (should not happen in real data)."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=-100.0,  # Invalid price
            volume=100
        )
        
        builder = CandleBuilder(['1m'])
        completed_candles = builder.process_tick(tick)
        
        # Should still process (no validation in current implementation)
        assert "NIFTY" in builder.active_candles
        candle = builder.active_candles["NIFTY"]["1m"]
        assert candle.open == -100.0
    
    def test_empty_timeframes_list(self):
        """Test candle builder with empty timeframes list."""
        builder = CandleBuilder([])
        assert builder.timeframes == []
        assert builder.timeframe_seconds == {}
        
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),
            symbol="NIFTY",
            price=18000.0,
            volume=100
        )
        
        completed_candles = builder.process_tick(tick)
        assert len(completed_candles) == 0


if __name__ == "__main__":
    pytest.main([__file__])
