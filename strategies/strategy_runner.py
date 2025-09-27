"""
Strategy runner for executing trading strategies on historical data.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio

from .base_strategy import BaseStrategy, TradingSignal, SignalType
from .supertrend import SupertrendStrategy
from utils.logging_utils import strategy_logger, get_logger
from utils.config import config
from data.candle_builder import CandleBuilder


class StrategyRunner:
    """Runs trading strategies on historical candle data."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("strategy_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.strategies = {}
        self.results = {}
        strategy_logger.info(f"Initialized StrategyRunner with output directory: {self.output_dir}")
    
    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a strategy to the runner."""
        self.strategies[name] = strategy
        strategy_logger.info(f"Added strategy: {name}")
    
    def run_strategy(self, strategy_name: str, df: pd.DataFrame, 
                    symbol: str = None) -> Dict[str, Any]:
        """
        Run a strategy on historical data.
        
        Args:
            strategy_name: Name of the strategy to run
            df: DataFrame with OHLCV data
            symbol: Symbol name (if not in DataFrame)
            
        Returns:
            Dictionary with results and signals
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        strategy.clear_signals()
        
        # Add symbol to DataFrame if not present
        if symbol and 'symbol' not in df.columns:
            df = df.copy()
            df['symbol'] = symbol
        
        strategy_logger.info(f"Running strategy '{strategy_name}' on {len(df)} candles")
        
        # Run strategy
        start_time = datetime.now()
        result_df = strategy.calculate_signals(df)
        end_time = datetime.now()
        
        # Get generated signals
        signals = strategy.get_signals()
        
        # Calculate performance metrics
        metrics = strategy.get_performance_metrics()
        metrics.update({
            'execution_time': (end_time - start_time).total_seconds(),
            'data_points': len(df),
            'signals_generated': len(signals)
        })
        
        # Store results
        result_key = f"{strategy_name}_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results[result_key] = {
            'strategy_name': strategy_name,
            'symbol': symbol,
            'data': result_df,
            'signals': signals,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        
        # Save signals to CSV
        if signals:
            signal_filename = f"{result_key}_signals.csv"
            signal_filepath = self.output_dir / signal_filename
            strategy.save_signals(str(signal_filepath))
            metrics['signals_file'] = str(signal_filepath)
        
        strategy_logger.info(f"Strategy '{strategy_name}' completed: {len(signals)} signals generated")
        
        return self.results[result_key]
    
    def run_multiple_strategies(self, strategies: List[str], df: pd.DataFrame, 
                              symbol: str = None) -> Dict[str, Any]:
        """Run multiple strategies on the same data."""
        results = {}
        
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                try:
                    result = self.run_strategy(strategy_name, df, symbol)
                    results[strategy_name] = result
                except Exception as e:
                    strategy_logger.error(f"Error running strategy '{strategy_name}': {e}")
                    results[strategy_name] = {'error': str(e)}
            else:
                strategy_logger.warning(f"Strategy '{strategy_name}' not found")
        
        return results
    
    def compare_strategies(self, strategies: List[str], df: pd.DataFrame, 
                          symbol: str = None) -> pd.DataFrame:
        """Compare performance of multiple strategies."""
        results = self.run_multiple_strategies(strategies, df, symbol)
        
        comparison_data = []
        for strategy_name, result in results.items():
            if 'error' not in result:
                metrics = result['metrics']
                comparison_data.append({
                    'strategy': strategy_name,
                    'total_signals': metrics.get('total_signals', 0),
                    'buy_signals': metrics.get('buy_signals', 0),
                    'sell_signals': metrics.get('sell_signals', 0),
                    'avg_confidence': metrics.get('avg_confidence', 0.0),
                    'execution_time': metrics.get('execution_time', 0.0),
                    'signals_generated': metrics.get('signals_generated', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('signals_generated', ascending=False)
        
        # Save comparison
        comparison_filename = f"strategy_comparison_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_filepath = self.output_dir / comparison_filename
        comparison_df.to_csv(comparison_filepath, index=False)
        
        strategy_logger.info(f"Strategy comparison saved to {comparison_filepath}")
        
        return comparison_df
    
    def get_strategy_results(self, strategy_name: str = None) -> Dict[str, Any]:
        """Get results for a specific strategy or all strategies."""
        if strategy_name:
            return {k: v for k, v in self.results.items() if v['strategy_name'] == strategy_name}
        return self.results
    
    def save_all_results(self, filename: str = None):
        """Save all strategy results to files."""
        if not filename:
            filename = f"strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        
        # Combine all signals from all results
        all_signals = []
        for result_key, result in self.results.items():
            for signal in result['signals']:
                signal_dict = signal.to_dict()
                signal_dict['result_key'] = result_key
                all_signals.append(signal_dict)
        
        if all_signals:
            df = pd.DataFrame(all_signals)
            df.to_csv(filepath, index=False)
            strategy_logger.info(f"Saved all strategy results to {filepath}")
        else:
            strategy_logger.warning("No signals to save")
    
    def load_candles_and_run(self, symbol: str, timeframe: str, 
                           strategy_name: str, start_date: datetime = None,
                           end_date: datetime = None) -> Dict[str, Any]:
        """Load candles from storage and run strategy."""
        # Load candles using CandleBuilder
        builder = CandleBuilder([timeframe])
        df = builder.load_candles(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            strategy_logger.warning(f"No candle data found for {symbol}_{timeframe}")
            return {}
        
        # Run strategy
        return self.run_strategy(strategy_name, df, symbol)
    
    async def run_strategy_async(self, strategy_name: str, df: pd.DataFrame, 
                               symbol: str = None) -> Dict[str, Any]:
        """Asynchronously run a strategy."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_strategy, strategy_name, df, symbol)


# Convenience functions for quick strategy execution
def create_supertrend_strategy(period: int = 10, multiplier: float = 3.0) -> SupertrendStrategy:
    """Create a Supertrend strategy with default parameters."""
    return SupertrendStrategy(period, multiplier)


def run_supertrend_on_data(df: pd.DataFrame, symbol: str = None, 
                          period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
    """Quick function to run Supertrend strategy on data."""
    runner = StrategyRunner()
    strategy = create_supertrend_strategy(period, multiplier)
    runner.add_strategy("supertrend", strategy)
    
    return runner.run_strategy("supertrend", df, symbol)


def analyze_supertrend_signals(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Analyze Supertrend signals and return detailed DataFrame."""
    strategy = create_supertrend_strategy()
    df_with_signals = strategy.calculate_signals(df)
    
    # Add signal analysis columns
    if 'supertrend_signal' in df_with_signals.columns:
        signal_points = df_with_signals[df_with_signals['supertrend_signal'] != 0].copy()
        
        if not signal_points.empty:
            signal_points['signal_type'] = signal_points['supertrend_signal'].map({1: 'BUY', -1: 'SELL'})
            signal_points['price_change'] = signal_points['close'].pct_change()
            signal_points['volatility'] = signal_points['supertrend_atr'] / signal_points['close']
            
            return signal_points
    
    return pd.DataFrame()
