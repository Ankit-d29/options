#!/usr/bin/env python3
"""
Demo script for Phase 3 - Backtesting Engine functionality.
This script demonstrates historical trade simulation, P&L calculation, and performance analysis.
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

from backtest.backtest_engine import BacktestEngine, BacktestConfig
from backtest.trade_executor import TradeExecutor, Trade
from backtest.performance_analyzer import PerformanceAnalyzer
from backtest.portfolio_manager import PortfolioManager, PortfolioConfig
from strategies.supertrend import SupertrendStrategy
from strategies.base_strategy import TradingSignal, SignalType
from data.tick_data import MockTickGenerator
from data.candle_builder import CandleBuilder
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger('demo_backtest')


def create_realistic_backtest_data(symbol: str = "NIFTY", length: int = 500) -> pd.DataFrame:
    """Create realistic market data for backtesting with clear trends."""
    dates = pd.date_range(start='2024-01-01 09:15:00', periods=length, freq='1min')
    
    # Create realistic price movement with clear trend changes for better signals
    base_price = 18000 if symbol == "NIFTY" else 45000
    
    prices = []
    current_price = base_price
    
    for i in range(length):
        # Create different market regimes
        if i < length // 4:
            # Strong uptrend
            change = np.random.normal(0.002, 0.001)
        elif i < length // 2:
            # Sideways with volatility
            change = np.random.normal(0.000, 0.003)
        elif i < 3 * length // 4:
            # Downtrend
            change = np.random.normal(-0.001, 0.002)
        else:
            # Recovery uptrend
            change = np.random.normal(0.001, 0.0015)
        
        current_price *= (1 + change)
        prices.append(current_price)
    
    prices = np.array(prices)
    
    # Create OHLC data with realistic spreads
    highs = prices * (1 + np.random.uniform(0, 0.003, length))
    lows = prices * (1 - np.random.uniform(0, 0.003, length))
    
    # Ensure OHLC consistency
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    volumes = np.random.randint(500, 3000, length)
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': symbol,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })


def demo_trade_executor():
    """Demonstrate trade execution functionality."""
    logger.info("=== Trade Executor Demo ===")
    
    # Create trade executor
    executor = TradeExecutor(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_positions=3
    )
    
    logger.info(f"Initialized TradeExecutor with ${executor.initial_capital:,}")
    
    # Create some trading signals
    signals = [
        TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0,
            confidence=0.8
        ),
        TradingSignal(
            timestamp=datetime.now() + timedelta(minutes=30),
            symbol="BANKNIFTY",
            signal_type=SignalType.BUY,
            price=45000.0,
            confidence=0.7
        )
    ]
    
    # Execute entries
    for i, signal in enumerate(signals):
        current_time = signal.timestamp
        current_price = signal.price
        
        success = executor.execute_entry(signal, current_price, current_time)
        
        if success:
            logger.info(f"Executed {signal.signal_type.value} entry for {signal.symbol} @ ${current_price}")
        else:
            logger.info(f"Failed to execute entry for {signal.symbol}")
    
    # Update portfolio value
    current_prices = {"NIFTY": 18100.0, "BANKNIFTY": 45100.0}
    executor.update_equity_curve(datetime.now() + timedelta(hours=1), current_prices)
    
    portfolio_value = executor.get_portfolio_value(current_prices)
    logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
    
    # Show positions
    positions = executor.get_current_positions()
    logger.info(f"Open positions: {len(positions)}")
    for symbol, position in positions.items():
        logger.info(f"  {symbol}: {position.position_type} x{position.quantity} @ ${position.entry_price:.2f}")
    
    # Execute exits
    exit_signals = [
        TradingSignal(
            timestamp=datetime.now() + timedelta(hours=2),
            symbol="NIFTY",
            signal_type=SignalType.SELL,
            price=18200.0,
            confidence=0.9
        ),
        TradingSignal(
            timestamp=datetime.now() + timedelta(hours=2),
            symbol="BANKNIFTY",
            signal_type=SignalType.SELL,
            price=44800.0,
            confidence=0.8
        )
    ]
    
    for signal in exit_signals:
        trade = executor.execute_exit(signal, signal.price, signal.timestamp)
        if trade:
            logger.info(f"Closed position: {trade.symbol} P&L: ${trade.pnl:.2f}")
    
    # Show summary
    trades = executor.get_trades()
    summary = executor.get_summary_stats()
    
    logger.info(f"Trade Summary:")
    logger.info(f"  Total trades: {summary['total_trades']}")
    logger.info(f"  Winning trades: {summary['winning_trades']}")
    logger.info(f"  Win rate: {summary['win_rate']:.1f}%")
    logger.info(f"  Total P&L: ${summary['total_pnl']:.2f}")
    
    return executor


def demo_performance_analyzer():
    """Demonstrate performance analysis functionality."""
    logger.info("\n=== Performance Analyzer Demo ===")
    
    # Create sample equity curve
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    
    # Simulate realistic equity curve with some drawdowns
    base_value = 100000
    equity_values = []
    current_value = base_value
    
    for i in range(100):
        # Add some random walk with occasional drawdowns
        if i % 20 == 0 and i > 0:
            # Simulate drawdown
            current_value *= 0.95
        else:
            # Normal growth
            change = np.random.normal(0.001, 0.01)
            current_value *= (1 + change)
        
        equity_values.append(current_value)
    
    equity_curve = pd.DataFrame({
        'timestamp': dates,
        'portfolio_value': equity_values
    })
    
    # Create sample trades
    trades = []
    for i in range(10):
        is_winning = np.random.random() > 0.4  # 60% win rate
        pnl = np.random.uniform(100, 500) if is_winning else np.random.uniform(-300, -50)
        
        trade = Trade(
            symbol="NIFTY",
            entry_time=dates[i * 10],
            exit_time=dates[i * 10] + timedelta(hours=1),
            entry_price=18000.0,
            exit_price=18000.0 + (pnl / 100),
            quantity=100,
            position_type="LONG",
            pnl=pnl,
            commission=10.0,
            slippage=5.0,
            duration_minutes=60
        )
        trades.append(trade)
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    
    # Calculate various metrics
    returns = analyzer.calculate_returns(equity_curve)
    drawdown, peak = analyzer.calculate_drawdown(equity_curve)
    max_dd = analyzer.calculate_max_drawdown(equity_curve)
    sharpe_ratio = analyzer.calculate_sharpe_ratio(returns)
    trade_metrics = analyzer.calculate_trade_metrics(trades)
    
    logger.info("Performance Analysis Results:")
    logger.info(f"  Total return: {((equity_values[-1] - base_value) / base_value * 100):.2f}%")
    logger.info(f"  Annualized return: {((1 + returns.mean()) ** (252 * 24) - 1) * 100:.2f}%")  # Hourly data
    logger.info(f"  Sharpe ratio: {sharpe_ratio:.3f}")
    logger.info(f"  Max drawdown: {max_dd['max_drawdown'] * 100:.2f}%")
    logger.info(f"  Max drawdown duration: {max_dd['max_drawdown_duration']} periods")
    
    logger.info("Trade Analysis:")
    logger.info(f"  Total trades: {trade_metrics['total_trades']}")
    logger.info(f"  Win rate: {trade_metrics['win_rate']:.1f}%")
    logger.info(f"  Profit factor: {trade_metrics['profit_factor']:.2f}")
    logger.info(f"  Average win: ${trade_metrics['avg_win']:.2f}")
    logger.info(f"  Average loss: ${trade_metrics['avg_loss']:.2f}")
    
    # Generate comprehensive report
    report = analyzer.generate_performance_report(equity_curve, trades, base_value)
    
    logger.info("Comprehensive Report Highlights:")
    logger.info(f"  Sortino ratio: {report.get('sortino_ratio', 0):.3f}")
    logger.info(f"  Calmar ratio: {report.get('calmar_ratio', 0):.3f}")
    logger.info(f"  Annual volatility: {report.get('annual_volatility', 0) * 100:.2f}%")
    
    return analyzer, report


def demo_portfolio_manager():
    """Demonstrate portfolio management functionality."""
    logger.info("\n=== Portfolio Manager Demo ===")
    
    # Create portfolio configuration
    config = PortfolioConfig(
        initial_capital=100000,
        max_positions=5,
        risk_per_trade_percent=0.02,
        stop_loss_percent=0.05,
        take_profit_percent=0.10
    )
    
    manager = PortfolioManager(config)
    logger.info(f"Initialized PortfolioManager with ${config.initial_capital:,}")
    
    # Create trading signals
    signals = [
        TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0,
            confidence=0.8
        ),
        TradingSignal(
            timestamp=datetime.now() + timedelta(minutes=30),
            symbol="BANKNIFTY",
            signal_type=SignalType.BUY,
            price=45000.0,
            confidence=0.7
        ),
        TradingSignal(
            timestamp=datetime.now() + timedelta(hours=1),
            symbol="RELIANCE",
            signal_type=SignalType.BUY,
            price=2500.0,
            confidence=0.6
        )
    ]
    
    current_prices = {"NIFTY": 18000.0, "BANKNIFTY": 45000.0, "RELIANCE": 2500.0}
    
    # Execute signals
    for signal in signals:
        trades = manager.execute_signals([signal], current_prices, signal.timestamp)
        logger.info(f"Executed {signal.signal_type.value} for {signal.symbol}")
        
        # Update portfolio
        manager.update_portfolio(signal.timestamp, current_prices)
    
    # Show portfolio summary
    summary = manager.get_portfolio_summary()
    logger.info(f"Portfolio Summary:")
    logger.info(f"  Portfolio value: ${summary['portfolio_value']:,.2f}")
    logger.info(f"  Cash: ${summary['cash']:,.2f} ({summary['cash_percent']:.1f}%)")
    logger.info(f"  Positions: {summary['positions_count']}")
    logger.info(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    
    # Show individual positions
    for symbol, position_info in summary['positions'].items():
        logger.info(f"  {symbol}: {position_info['quantity']} shares @ ${position_info['entry_price']:.2f}")
        logger.info(f"    P&L: ${position_info['total_pnl']:.2f}")
    
    # Simulate price movements and check stop losses
    logger.info("\nSimulating price movements...")
    
    # Price movements that trigger stop losses
    new_prices = {
        "NIFTY": 17100.0,  # 5% drop - triggers stop loss
        "BANKNIFTY": 42750.0,  # 5% drop - triggers stop loss
        "RELIANCE": 2375.0  # 5% drop - triggers stop loss
    }
    
    # Update portfolio and check for stop losses
    closed_trades = manager.update_portfolio(datetime.now() + timedelta(hours=2), new_prices)
    
    if closed_trades:
        logger.info(f"Stop losses triggered - closed {len(closed_trades)} positions")
        for trade in closed_trades:
            logger.info(f"  {trade.symbol}: P&L ${trade.pnl:.2f}")
    
    # Final performance summary
    perf_summary = manager.get_performance_summary()
    logger.info(f"\nFinal Performance:")
    logger.info(f"  Total trades: {perf_summary['total_trades']}")
    logger.info(f"  Win rate: {perf_summary['win_rate']:.1f}%")
    logger.info(f"  Portfolio value: ${perf_summary['portfolio_value']:,.2f}")
    logger.info(f"  Return: {perf_summary['return_percent']:.2f}%")
    
    return manager


def demo_backtest_engine():
    """Demonstrate the main backtesting engine."""
    logger.info("\n=== Backtesting Engine Demo ===")
    
    # Create realistic market data
    data = create_realistic_backtest_data("NIFTY", 300)
    logger.info(f"Created backtest data: {len(data)} candles")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_positions=3,
        risk_per_trade=0.02,
        stop_loss_percent=0.05,
        take_profit_percent=0.10
    )
    
    engine = BacktestEngine(config)
    logger.info(f"Initialized BacktestEngine with ${config.initial_capital:,}")
    
    # Create strategy
    strategy = SupertrendStrategy(period=14, multiplier=2.5)
    logger.info(f"Created strategy: {strategy.name}")
    
    # Run backtest
    logger.info("Running backtest...")
    result = engine.run_backtest(strategy, data, "NIFTY")
    
    # Display results
    logger.info(f"Backtest completed in {result.execution_time:.2f} seconds")
    
    summary = result.get_summary()
    logger.info(f"\nBacktest Results:")
    logger.info(f"  Total trades: {summary['total_trades']}")
    logger.info(f"  Winning trades: {summary['winning_trades']}")
    logger.info(f"  Win rate: {summary['win_rate']:.1f}%")
    logger.info(f"  Total P&L: ${summary['total_pnl']:.2f}")
    logger.info(f"  Average P&L per trade: ${summary['avg_pnl_per_trade']:.2f}")
    logger.info(f"  Max drawdown: {summary['max_drawdown'] * 100:.2f}%")
    logger.info(f"  Sharpe ratio: {summary['sharpe_ratio']:.3f}")
    logger.info(f"  Final portfolio value: ${summary['final_value']:,.2f}")
    
    # Show some detailed performance metrics
    if result.performance_metrics:
        metrics = result.performance_metrics
        logger.info(f"\nDetailed Performance Metrics:")
        logger.info(f"  Annual return: {metrics.get('annual_return', 0) * 100:.2f}%")
        logger.info(f"  Annual volatility: {metrics.get('annual_volatility', 0) * 100:.2f}%")
        logger.info(f"  Sortino ratio: {metrics.get('sortino_ratio', 0):.3f}")
        logger.info(f"  Calmar ratio: {metrics.get('calmar_ratio', 0):.3f}")
        logger.info(f"  VaR (95%): {metrics.get('var_95', 0) * 100:.2f}%")
        logger.info(f"  CVaR (95%): {metrics.get('cvar_95', 0) * 100:.2f}%")
    
    # Show equity curve info
    if not result.equity_curve.empty:
        logger.info(f"\nEquity Curve:")
        logger.info(f"  Data points: {len(result.equity_curve)}")
        logger.info(f"  Start value: ${result.equity_curve['portfolio_value'].iloc[0]:,.2f}")
        logger.info(f"  End value: ${result.equity_curve['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"  Peak value: ${result.equity_curve['portfolio_value'].max():,.2f}")
        logger.info(f"  Min value: ${result.equity_curve['portfolio_value'].min():,.2f}")
    
    return result


def demo_strategy_comparison():
    """Demonstrate strategy comparison functionality."""
    logger.info("\n=== Strategy Comparison Demo ===")
    
    # Create market data
    data = create_realistic_backtest_data("NIFTY", 200)
    
    # Create multiple strategies with different parameters
    strategies = [
        SupertrendStrategy(period=10, multiplier=2.0),
        SupertrendStrategy(period=14, multiplier=2.5),
        SupertrendStrategy(period=20, multiplier=3.0)
    ]
    
    # Create backtest engine
    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    
    # Run strategy comparison
    logger.info("Running strategy comparison...")
    results = engine.run_strategy_comparison(strategies, data, "NIFTY")
    
    # Display comparison results
    logger.info("\nStrategy Comparison Results:")
    logger.info("=" * 80)
    
    comparison_data = []
    for strategy_name, result in results.items():
        summary = result.get_summary()
        comparison_data.append({
            'Strategy': strategy_name,
            'Trades': summary['total_trades'],
            'Win Rate': f"{summary['win_rate']:.1f}%",
            'Total P&L': f"${summary['total_pnl']:.2f}",
            'Max DD': f"{summary['max_drawdown'] * 100:.2f}%",
            'Sharpe': f"{summary['sharpe_ratio']:.3f}",
            'Final Value': f"${summary['final_value']:,.2f}"
        })
    
    # Create comparison DataFrame for nice formatting
    comparison_df = pd.DataFrame(comparison_data)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # Find best performing strategy
    best_strategy = max(results.keys(), key=lambda x: results[x].get_summary()['total_pnl'])
    best_summary = results[best_strategy].get_summary()
    
    logger.info(f"\nBest performing strategy: {best_strategy}")
    logger.info(f"  Total P&L: ${best_summary['total_pnl']:.2f}")
    logger.info(f"  Win rate: {best_summary['win_rate']:.1f}%")
    logger.info(f"  Sharpe ratio: {best_summary['sharpe_ratio']:.3f}")
    
    return results


def demo_integration_with_candle_builder():
    """Demonstrate integration with Phase 1 candle builder."""
    logger.info("\n=== Integration with Candle Builder Demo ===")
    
    # Generate tick data using mock generator
    generator = MockTickGenerator("NIFTY", 18000.0)
    ticks = generator.generate_ticks(1000, interval_seconds=30)  # ~8 hours of data
    
    logger.info(f"Generated {len(ticks)} ticks")
    
    # Build candles using candle builder
    builder = CandleBuilder(['1m', '5m'])
    completed_candles = builder.process_ticks_batch(ticks)
    
    logger.info(f"Built {len(completed_candles)} candles")
    
    # Convert to DataFrame for backtesting
    candles_data = [candle.to_dict() for candle in completed_candles]
    df = pd.DataFrame(candles_data)
    
    if not df.empty:
        logger.info(f"Created DataFrame with {len(df)} candles")
        
        # Run backtest on the generated candles
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        strategy = SupertrendStrategy(period=10, multiplier=2.0)
        
        result = engine.run_backtest(strategy, df, "NIFTY")
        
        summary = result.get_summary()
        logger.info(f"Backtest Results on Generated Data:")
        logger.info(f"  Total trades: {summary['total_trades']}")
        logger.info(f"  Win rate: {summary['win_rate']:.1f}%")
        logger.info(f"  Total P&L: ${summary['total_pnl']:.2f}")
        logger.info(f"  Final value: ${summary['final_value']:,.2f}")
        
        # Show some trades
        if result.trades:
            logger.info(f"\nSample Trades:")
            for i, trade in enumerate(result.trades[:3]):
                logger.info(f"  Trade {i+1}: {trade.symbol} {trade.position_type} "
                          f"{trade.entry_price:.2f} -> {trade.exit_price:.2f} "
                          f"P&L: ${trade.pnl:.2f}")
    
    return result


def main():
    """Run all Phase 3 demos."""
    logger.info("Starting Options Trading System - Phase 3 Demo (Backtesting Engine)")
    logger.info("=" * 80)
    
    try:
        # Run all demos
        demo_trade_executor()
        demo_performance_analyzer()
        demo_portfolio_manager()
        demo_backtest_engine()
        demo_strategy_comparison()
        demo_integration_with_candle_builder()
        
        logger.info("\n" + "=" * 80)
        logger.info("All Phase 3 demos completed successfully!")
        logger.info("\nPhase 3 Features Demonstrated:")
        logger.info("✅ Trade Execution Engine (entry/exit with commission & slippage)")
        logger.info("✅ Performance Analysis (Sharpe, Sortino, Calmar ratios)")
        logger.info("✅ Portfolio Management (position sizing, risk management)")
        logger.info("✅ Backtesting Engine (historical simulation)")
        logger.info("✅ Strategy Comparison (multi-strategy evaluation)")
        logger.info("✅ Integration with Phase 1 & 2 (tick → candle → signal → trade)")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
