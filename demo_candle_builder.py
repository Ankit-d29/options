#!/usr/bin/env python3
"""
Demo script for the candle builder functionality.
This script demonstrates how to use the candle builder to process tick data.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.candle_builder import CandleBuilder, TickData
from data.tick_data import MockTickGenerator, TickDataLoader
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger('demo')


def demo_basic_candle_building():
    """Demonstrate basic candle building functionality."""
    logger.info("=== Basic Candle Building Demo ===")
    
    # Create candle builder with multiple timeframes
    timeframes = ['1m', '5m', '15m']
    builder = CandleBuilder(timeframes)
    
    # Create mock tick generator
    generator = MockTickGenerator("NIFTY", 18000.0)
    
    # Generate some sample ticks
    logger.info("Generating sample tick data...")
    ticks = generator.generate_ticks(120, interval_seconds=30)  # 2 hours of data, 30s intervals
    
    logger.info(f"Generated {len(ticks)} ticks")
    logger.info(f"First tick: {ticks[0].timestamp}, Price: {ticks[0].price}")
    logger.info(f"Last tick: {ticks[-1].timestamp}, Price: {ticks[-1].price}")
    
    # Process ticks and build candles
    logger.info("Processing ticks to build candles...")
    completed_candles = builder.process_ticks_batch(ticks)
    
    logger.info(f"Completed {len(completed_candles)} candles")
    
    # Group candles by timeframe
    candles_by_timeframe = {}
    for candle in completed_candles:
        if candle.timeframe not in candles_by_timeframe:
            candles_by_timeframe[candle.timeframe] = []
        candles_by_timeframe[candle.timeframe].append(candle)
    
    # Display results
    for timeframe, candles in candles_by_timeframe.items():
        logger.info(f"{timeframe} candles: {len(candles)}")
        if candles:
            first_candle = candles[0]
            last_candle = candles[-1]
            logger.info(f"  First: {first_candle.timestamp} O:{first_candle.open} H:{first_candle.high} L:{first_candle.low} C:{first_candle.close}")
            logger.info(f"  Last:  {last_candle.timestamp} O:{last_candle.open} H:{last_candle.high} L:{last_candle.low} C:{last_candle.close}")
    
    # Finalize remaining active candles
    remaining_candles = builder.finalize_candles()
    logger.info(f"Finalized {len(remaining_candles)} remaining candles")
    
    return completed_candles + remaining_candles


def demo_candle_saving_and_loading():
    """Demonstrate saving and loading candles."""
    logger.info("\n=== Candle Saving and Loading Demo ===")
    
    # Create some test candles
    builder = CandleBuilder(['1m', '5m'])
    generator = MockTickGenerator("BANKNIFTY", 45000.0)
    
    # Generate ticks for different time periods
    ticks1 = generator.generate_ticks(60, interval_seconds=60)  # 1 hour of 1-minute data
    
    # Reset generator for new data
    generator2 = MockTickGenerator("NIFTY", 18000.0)
    ticks2 = generator2.generate_ticks(60, interval_seconds=60)
    
    all_ticks = ticks1 + ticks2
    completed_candles = builder.process_ticks_batch(all_ticks)
    
    # Save candles
    output_dir = "demo_candles"
    logger.info(f"Saving candles to {output_dir}...")
    builder.save_candles(completed_candles, output_dir)
    
    # Load candles back
    logger.info("Loading candles back...")
    
    # Load NIFTY 1m candles
    nifty_1m_df = builder.load_candles("NIFTY", "1m")
    logger.info(f"Loaded {len(nifty_1m_df)} NIFTY 1m candles")
    
    # Load BANKNIFTY 5m candles
    banknifty_5m_df = builder.load_candles("BANKNIFTY", "5m")
    logger.info(f"Loaded {len(banknifty_5m_df)} BANKNIFTY 5m candles")
    
    if len(nifty_1m_df) > 0:
        logger.info(f"NIFTY 1m sample data:")
        logger.info(f"  Columns: {list(nifty_1m_df.columns)}")
        logger.info(f"  First row: {nifty_1m_df.iloc[0].to_dict()}")
    
    return nifty_1m_df, banknifty_5m_df


def demo_tick_data_persistence():
    """Demonstrate tick data saving and loading."""
    logger.info("\n=== Tick Data Persistence Demo ===")
    
    # Create tick data loader
    loader = TickDataLoader("demo_ticks")
    
    # Generate sample ticks
    generator = MockTickGenerator("NIFTY", 18000.0)
    ticks = generator.generate_ticks(100, interval_seconds=10)
    
    logger.info(f"Generated {len(ticks)} sample ticks")
    
    # Save to CSV
    csv_filename = "nifty_ticks_demo.csv"
    loader.save_ticks_csv(ticks, csv_filename)
    
    # Save to JSON
    json_filename = "nifty_ticks_demo.json"
    loader.save_ticks_json(ticks, json_filename)
    
    # Load back from CSV
    loaded_ticks_csv = loader.load_ticks_csv(csv_filename)
    logger.info(f"Loaded {len(loaded_ticks_csv)} ticks from CSV")
    
    # Load back from JSON
    loaded_ticks_json = loader.load_ticks_json(json_filename)
    logger.info(f"Loaded {len(loaded_ticks_json)} ticks from JSON")
    
    # Verify data integrity
    assert len(loaded_ticks_csv) == len(ticks)
    assert len(loaded_ticks_json) == len(ticks)
    
    logger.info("Tick data persistence test passed!")


def demo_real_time_simulation():
    """Demonstrate real-time candle building simulation."""
    logger.info("\n=== Real-time Candle Building Simulation ===")
    
    import asyncio
    
    async def simulate_realtime():
        builder = CandleBuilder(['1m', '5m'])
        generator = MockTickGenerator("NIFTY", 18000.0)
        
        logger.info("Starting real-time simulation (10 seconds)...")
        
        start_time = datetime.now()
        tick_count = 0
        
        while (datetime.now() - start_time).seconds < 10:
            # Generate a tick
            tick = generator.generate_tick()
            tick_count += 1
            
            # Process tick asynchronously
            completed_candles = await builder.process_tick_async(tick)
            
            if completed_candles:
                for candle in completed_candles:
                    logger.info(f"Completed {candle.timeframe} candle: "
                              f"{candle.timestamp} O:{candle.open} H:{candle.high} "
                              f"L:{candle.low} C:{candle.close} V:{candle.volume}")
            
            # Wait a bit before next tick
            await asyncio.sleep(0.5)
        
        # Finalize remaining candles
        remaining_candles = builder.finalize_candles()
        logger.info(f"Simulation complete! Processed {tick_count} ticks")
        logger.info(f"Finalized {len(remaining_candles)} remaining candles")
    
    # Run the simulation
    asyncio.run(simulate_realtime())


def main():
    """Run all demo functions."""
    logger.info("Starting Options Trading System - Candle Builder Demo")
    logger.info("=" * 60)
    
    try:
        # Run demos
        demo_basic_candle_building()
        demo_candle_saving_and_loading()
        demo_tick_data_persistence()
        demo_real_time_simulation()
        
        logger.info("\n" + "=" * 60)
        logger.info("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
