#!/usr/bin/env python3
"""
Demo script for Phase 2 - Indicators & Strategy functionality.
This script demonstrates Supertrend indicator and strategy execution.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.supertrend import SupertrendIndicator, SupertrendStrategy
from strategies.strategy_runner import StrategyRunner, create_supertrend_strategy, run_supertrend_on_data
from strategies.base_strategy import SignalType
from data.tick_data import MockTickGenerator
from data.candle_builder import CandleBuilder
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger('demo_strategies')


def create_realistic_market_data(symbol: str = "NIFTY", length: int = 200) -> pd.DataFrame:
    """Create realistic market data with trends and volatility."""
    dates = pd.date_range(start='2024-01-01 09:15:00', periods=length, freq='1min')
    
    # Create realistic price movement with clear trend changes
    base_price = 18000 if symbol == "NIFTY" else 45000
    
    # Create a more volatile price series with clear trend changes
    prices = []
    current_price = base_price
    
    for i in range(length):
        # Create trend changes at specific points
        if i < length // 3:
            # Uptrend
            change = np.random.normal(0.001, 0.002)
        elif i < 2 * length // 3:
            # Downtrend
            change = np.random.normal(-0.001, 0.002)
        else:
            # Uptrend again
            change = np.random.normal(0.001, 0.002)
        
        current_price *= (1 + change)
        prices.append(current_price)
    
    prices = np.array(prices)
    
    # Create OHLC data with realistic spreads
    highs = prices * (1 + np.random.uniform(0, 0.005, length))
    lows = prices * (1 - np.random.uniform(0, 0.005, length))
    
    # Ensure OHLC consistency
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # Add some gap openings
    for i in range(5, length, 50):
        opens[i] = prices[i-1] * (1 + np.random.normal(0, 0.01))
    
    volumes = np.random.randint(100, 2000, length)
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': symbol,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })


def demo_supertrend_indicator():
    """Demonstrate Supertrend indicator functionality."""
    logger.info("=== Supertrend Indicator Demo ===")
    
    # Create sample data
    df = create_realistic_market_data("NIFTY", 100)
    logger.info(f"Created sample data with {len(df)} candles")
    logger.info(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Create Supertrend indicator
    indicator = SupertrendIndicator(period=10, multiplier=3.0)
    logger.info(f"Created Supertrend indicator with period=10, multiplier=3.0")
    
    # Add indicator to DataFrame
    df_with_supertrend = indicator.add_to_dataframe(df)
    logger.info("Added Supertrend indicator to DataFrame")
    
    # Show some statistics
    signal_points = df_with_supertrend[df_with_supertrend['supertrend_signal'] != 0]
    logger.info(f"Found {len(signal_points)} signal points")
    
    if not signal_points.empty:
        logger.info("Signal details:")
        for idx, row in signal_points.head(5).iterrows():
            signal_type = "BUY" if row['supertrend_signal'] == 1 else "SELL"
            logger.info(f"  {row['timestamp']}: {signal_type} at {row['close']:.2f} "
                       f"(Supertrend: {row['supertrend']:.2f})")
    
    # Generate signals
    signals = indicator.get_signals(df_with_supertrend)
    logger.info(f"Generated {len(signals)} trading signals")
    
    # Show signal details
    for i, signal in enumerate(signals[:3]):
        logger.info(f"Signal {i+1}: {signal}")
    
    return df_with_supertrend, signals


def demo_supertrend_strategy():
    """Demonstrate Supertrend strategy functionality."""
    logger.info("\n=== Supertrend Strategy Demo ===")
    
    # Create sample data
    df = create_realistic_market_data("NIFTY", 150)
    
    # Create strategy with different parameters
    strategy = SupertrendStrategy(period=14, multiplier=2.5)
    logger.info(f"Created SupertrendStrategy with period=14, multiplier=2.5")
    
    # Run strategy
    result_df = strategy.calculate_signals(df)
    signals = strategy.get_signals()
    
    logger.info(f"Strategy generated {len(signals)} signals")
    
    # Get trend status
    trend_status = strategy.get_trend_status(df)
    logger.info(f"Current trend status: {trend_status}")
    
    # Check entry conditions
    should_long = strategy.should_enter_long(df)
    should_short = strategy.should_enter_short(df)
    
    logger.info(f"Should enter long: {should_long}")
    logger.info(f"Should enter short: {should_short}")
    
    # Get performance metrics
    metrics = strategy.get_performance_metrics()
    logger.info(f"Strategy metrics: {metrics}")
    
    return strategy, signals


def demo_strategy_runner():
    """Demonstrate strategy runner functionality."""
    logger.info("\n=== Strategy Runner Demo ===")
    
    # Create sample data for multiple symbols
    nifty_data = create_realistic_market_data("NIFTY", 100)
    banknifty_data = create_realistic_market_data("BANKNIFTY", 100)
    
    # Create strategy runner
    runner = StrategyRunner()
    logger.info("Created StrategyRunner")
    
    # Add multiple strategies
    strategy1 = create_supertrend_strategy(period=10, multiplier=3.0)
    strategy2 = create_supertrend_strategy(period=20, multiplier=2.0)
    
    runner.add_strategy("supertrend_10_3", strategy1)
    runner.add_strategy("supertrend_20_2", strategy2)
    
    logger.info("Added multiple Supertrend strategies with different parameters")
    
    # Run single strategy
    result = runner.run_strategy("supertrend_10_3", nifty_data, "NIFTY")
    logger.info(f"Ran single strategy: {result['metrics']['signals_generated']} signals generated")
    
    # Run multiple strategies
    results = runner.run_multiple_strategies(
        ["supertrend_10_3", "supertrend_20_2"], 
        banknifty_data, 
        "BANKNIFTY"
    )
    
    logger.info("Ran multiple strategies on BANKNIFTY data:")
    for strategy_name, result in results.items():
        if 'error' not in result:
            signals_count = result['metrics'].get('signals_generated', 0)
            logger.info(f"  {strategy_name}: {signals_count} signals")
        else:
            logger.info(f"  {strategy_name}: Error - {result['error']}")
    
    # Compare strategies
    comparison_df = runner.compare_strategies(
        ["supertrend_10_3", "supertrend_20_2"], 
        nifty_data, 
        "NIFTY"
    )
    
    logger.info("Strategy comparison:")
    logger.info(f"{comparison_df.to_string(index=False)}")
    
    return runner, results


def demo_signal_analysis():
    """Demonstrate signal analysis functionality."""
    logger.info("\n=== Signal Analysis Demo ===")
    
    # Create data with clear trend
    df = create_realistic_market_data("NIFTY", 200)
    
    # Run Supertrend strategy
    result = run_supertrend_on_data(df, "NIFTY", period=14, multiplier=2.5)
    
    signals = result['signals']
    logger.info(f"Generated {len(signals)} signals for analysis")
    
    if signals:
        # Analyze signal types
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        logger.info(f"Signal breakdown:")
        logger.info(f"  Buy signals: {len(buy_signals)}")
        logger.info(f"  Sell signals: {len(sell_signals)}")
        
        # Analyze confidence levels
        confidences = [s.confidence for s in signals]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        min_confidence = min(confidences)
        
        logger.info(f"Confidence analysis:")
        logger.info(f"  Average: {avg_confidence:.3f}")
        logger.info(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Show high-confidence signals
        high_conf_signals = [s for s in signals if s.confidence > 0.7]
        logger.info(f"High confidence signals (>0.7): {len(high_conf_signals)}")
        
        for signal in high_conf_signals[:3]:
            logger.info(f"  {signal.timestamp}: {signal.signal_type.value} at {signal.price:.2f} "
                       f"(conf: {signal.confidence:.3f})")
    
    return signals


def demo_integration_with_candle_builder():
    """Demonstrate integration with candle builder."""
    logger.info("\n=== Integration with Candle Builder Demo ===")
    
    # Generate tick data using mock generator
    generator = MockTickGenerator("NIFTY", 18000.0)
    ticks = generator.generate_ticks(500, interval_seconds=30)  # ~4 hours of data
    
    logger.info(f"Generated {len(ticks)} ticks")
    
    # Build candles using candle builder
    builder = CandleBuilder(['1m', '5m'])
    completed_candles = builder.process_ticks_batch(ticks)
    
    logger.info(f"Built {len(completed_candles)} candles")
    
    # Convert to DataFrame
    candles_data = [candle.to_dict() for candle in completed_candles]
    df = pd.DataFrame(candles_data)
    
    if not df.empty:
        logger.info(f"Created DataFrame with {len(df)} candles")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Run Supertrend strategy on the generated candles
        result = run_supertrend_on_data(df, "NIFTY")
        
        signals = result['signals']
        logger.info(f"Generated {len(signals)} signals from candle builder data")
        
        # Show first few signals
        for i, signal in enumerate(signals[:3]):
            logger.info(f"Signal {i+1}: {signal}")
    
    return df, result.get('signals', [])


def demo_signal_persistence():
    """Demonstrate signal saving and loading."""
    logger.info("\n=== Signal Persistence Demo ===")
    
    # Create strategy and generate signals
    df = create_realistic_market_data("NIFTY", 100)
    strategy = create_supertrend_strategy()
    strategy.calculate_signals(df)
    
    signals = strategy.get_signals()
    logger.info(f"Generated {len(signals)} signals")
    
    # Save signals
    signal_file = "demo_signals.csv"
    strategy.save_signals(signal_file)
    logger.info(f"Saved signals to {signal_file}")
    
    # Load signals into new strategy
    new_strategy = create_supertrend_strategy()
    new_strategy.load_signals(signal_file)
    
    loaded_signals = new_strategy.get_signals()
    logger.info(f"Loaded {len(loaded_signals)} signals")
    
    # Verify signals match
    if signals and loaded_signals:
        assert len(signals) == len(loaded_signals)
        assert signals[0].symbol == loaded_signals[0].symbol
        assert signals[0].signal_type == loaded_signals[0].signal_type
        logger.info("Signal persistence verification passed!")
    
    # Clean up
    Path(signal_file).unlink(missing_ok=True)
    logger.info("Cleaned up temporary files")


def main():
    """Run all Phase 2 demos."""
    logger.info("Starting Options Trading System - Phase 2 Demo (Indicators & Strategy)")
    logger.info("=" * 80)
    
    try:
        # Run all demos
        demo_supertrend_indicator()
        demo_supertrend_strategy()
        demo_strategy_runner()
        demo_signal_analysis()
        demo_integration_with_candle_builder()
        demo_signal_persistence()
        
        logger.info("\n" + "=" * 80)
        logger.info("All Phase 2 demos completed successfully!")
        logger.info("\nPhase 2 Features Demonstrated:")
        logger.info("✅ Supertrend Indicator (ATR-based trend following)")
        logger.info("✅ Strategy Runner (multiple strategy execution)")
        logger.info("✅ Signal Generation (BUY/SELL with confidence)")
        logger.info("✅ Signal Persistence (CSV save/load)")
        logger.info("✅ Integration with Candle Builder")
        logger.info("✅ Performance Metrics and Analysis")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
