"""
Main backtesting engine for historical trade simulation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from utils.logging_utils import get_logger
from utils.config import config
from strategies.base_strategy import BaseStrategy, TradingSignal
from strategies.strategy_runner import StrategyRunner
from .trade_executor import TradeExecutor, Trade
from .portfolio_manager import PortfolioManager, PortfolioConfig
from .performance_analyzer import PerformanceAnalyzer

# Set up logging
backtest_logger = get_logger('backtest_engine')


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_positions: int = 10
    risk_per_trade: float = 0.02
    stop_loss_percent: float = 0.05
    take_profit_percent: float = 0.10
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    rebalance_frequency: str = 'daily'


@dataclass
class BacktestResult:
    """Backtesting results container."""
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    allocation_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    execution_time: float = 0.0
    strategy_name: str = ""
    symbol: str = ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of backtest results."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        winning_trades = [t for t in self.trades if t.is_winning_trade()]
        total_pnl = sum(t.pnl for t in self.trades)
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(self.trades),
            'max_drawdown': self.performance_metrics.get('max_drawdown', 0),
            'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
            'final_value': self.equity_curve['portfolio_value'].iloc[-1] if len(self.equity_curve) > 0 else self.config.initial_capital
        }


class BacktestEngine:
    """Main backtesting engine for historical trade simulation."""
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.portfolio_config = PortfolioConfig(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
            risk_per_trade_percent=self.config.risk_per_trade,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            stop_loss_percent=self.config.stop_loss_percent,
            take_profit_percent=self.config.take_profit_percent,
            rebalance_frequency=self.config.rebalance_frequency
        )
        
        self.portfolio_manager = PortfolioManager(self.portfolio_config)
        self.performance_analyzer = PerformanceAnalyzer()
        
        backtest_logger.info(f"Initialized BacktestEngine with capital: ${self.config.initial_capital:,.2f}")
    
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                    symbol: str = None) -> BacktestResult:
        """
        Run backtest with a strategy on historical data.
        
        Args:
            strategy: Trading strategy to backtest
            data: Historical OHLCV data
            symbol: Symbol name
            
        Returns:
            BacktestResult with complete backtest results
        """
        start_time = datetime.now()
        
        backtest_logger.info(f"Starting backtest for {strategy.name} on {len(data)} data points")
        
        # Reset portfolio manager
        self.portfolio_manager.reset()
        
        # Filter data by date range if specified
        if self.config.start_date or self.config.end_date:
            data = self._filter_data_by_date(data)
        
        if len(data) == 0:
            backtest_logger.warning("No data available after date filtering")
            return BacktestResult(self.config, strategy_name=strategy.name, symbol=symbol)
        
        # Generate signals
        strategy.clear_signals()
        data_with_signals = strategy.calculate_signals(data)
        signals = strategy.get_signals()
        
        backtest_logger.info(f"Generated {len(signals)} signals")
        
        # Process data chronologically
        all_trades = []
        
        for idx, row in data_with_signals.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            # Get signals for this timestamp
            current_signals = [s for s in signals if s.timestamp == current_time]
            
            # Execute signals
            if current_signals:
                current_prices = {symbol or 'UNKNOWN': current_price}
                executed_trades = self.portfolio_manager.execute_signals(
                    current_signals, current_prices, current_time
                )
                all_trades.extend(executed_trades)
            
            # Update portfolio
            current_prices = {symbol or 'UNKNOWN': current_price}
            self.portfolio_manager.update_portfolio(current_time, current_prices)
            
            # Check for rebalancing
            if self.portfolio_manager.should_rebalance(current_time):
                rebalance_trades = self.portfolio_manager.rebalance_portfolio(current_time, current_prices)
                all_trades.extend(rebalance_trades)
        
        # Finalize any remaining positions
        final_trades = self._finalize_positions(data_with_signals.iloc[-1], symbol)
        all_trades.extend(final_trades)
        
        # Calculate performance metrics
        equity_curve = self.portfolio_manager.trade_executor.get_equity_curve()
        performance_metrics = self.performance_analyzer.generate_performance_report(
            equity_curve, all_trades, self.config.initial_capital
        )
        
        # Get allocation history
        allocation_history = self.portfolio_manager.get_allocation_history()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = BacktestResult(
            config=self.config,
            trades=all_trades,
            equity_curve=equity_curve,
            performance_metrics=performance_metrics,
            allocation_history=allocation_history,
            execution_time=execution_time,
            strategy_name=strategy.name,
            symbol=symbol
        )
        
        backtest_logger.info(f"Backtest completed in {execution_time:.2f}s: {len(all_trades)} trades executed")
        
        return result
    
    def run_strategy_comparison(self, strategies: List[BaseStrategy], data: pd.DataFrame,
                              symbol: str = None) -> Dict[str, BacktestResult]:
        """Run backtest comparison for multiple strategies."""
        results = {}
        
        backtest_logger.info(f"Running strategy comparison with {len(strategies)} strategies")
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, symbol)
                results[strategy.name] = result
                
                summary = result.get_summary()
                backtest_logger.info(f"{strategy.name}: {summary['total_trades']} trades, "
                                   f"P&L: ${summary['total_pnl']:.2f}, "
                                   f"Win Rate: {summary['win_rate']:.1f}%")
                
            except Exception as e:
                backtest_logger.error(f"Error running backtest for {strategy.name}: {e}")
                results[strategy.name] = BacktestResult(
                    self.config, 
                    strategy_name=strategy.name, 
                    symbol=symbol
                )
        
        return results
    
    def run_parameter_optimization(self, strategy_class, data: pd.DataFrame,
                                 param_ranges: Dict[str, List], symbol: str = None) -> pd.DataFrame:
        """Run parameter optimization for a strategy."""
        results = []
        
        backtest_logger.info(f"Starting parameter optimization with {len(param_ranges)} parameters")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(param_ranges)
        
        for i, params in enumerate(param_combinations):
            try:
                # Create strategy with current parameters
                strategy = strategy_class(**params)
                
                # Run backtest
                result = self.run_backtest(strategy, data, symbol)
                summary = result.get_summary()
                
                # Store results
                result_row = {
                    'combination_id': i,
                    'total_trades': summary['total_trades'],
                    'total_pnl': summary['total_pnl'],
                    'win_rate': summary['win_rate'],
                    'max_drawdown': summary['max_drawdown'],
                    'sharpe_ratio': summary['sharpe_ratio'],
                    'execution_time': result.execution_time
                }
                
                # Add parameter values
                for param_name, param_value in params.items():
                    result_row[f'param_{param_name}'] = param_value
                
                results.append(result_row)
                
                backtest_logger.info(f"Optimization {i+1}/{len(param_combinations)}: "
                                   f"P&L: ${summary['total_pnl']:.2f}, "
                                   f"Win Rate: {summary['win_rate']:.1f}%")
                
            except Exception as e:
                backtest_logger.error(f"Error in optimization combination {i}: {e}")
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Sort by total P&L
            results_df = results_df.sort_values('total_pnl', ascending=False)
            backtest_logger.info(f"Optimization completed. Best P&L: ${results_df.iloc[0]['total_pnl']:.2f}")
        
        return results_df
    
    def _filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data by start and end dates."""
        filtered_data = data.copy()
        
        if self.config.start_date:
            filtered_data = filtered_data[filtered_data['timestamp'] >= self.config.start_date]
        
        if self.config.end_date:
            filtered_data = filtered_data[filtered_data['timestamp'] <= self.config.end_date]
        
        return filtered_data
    
    def _finalize_positions(self, last_row: pd.Series, symbol: str) -> List[Trade]:
        """Finalize any remaining open positions."""
        final_trades = []
        
        if not self.portfolio_manager.trade_executor.positions:
            return final_trades
        
        final_price = last_row['close']
        final_time = last_row['timestamp']
        
        # Close all remaining positions
        for symbol_pos, position in list(self.portfolio_manager.trade_executor.positions.items()):
            # Create exit signal
            exit_signal = TradingSignal(
                timestamp=final_time,
                symbol=symbol_pos,
                signal_type=TradingSignal.SignalType.SELL if position.position_type == 'LONG' else TradingSignal.SignalType.BUY,
                price=final_price,
                confidence=1.0,
                metadata={'final_exit': True}
            )
            
            trade = self.portfolio_manager.trade_executor.execute_exit(
                exit_signal, final_price, final_time
            )
            
            if trade:
                final_trades.append(trade)
        
        return final_trades
    
    def _generate_parameter_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization."""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def save_backtest_results(self, result: BacktestResult, output_dir: str = "backtest_results"):
        """Save backtest results to files."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{result.strategy_name}_{result.symbol}_{timestamp}"
        
        # Save trades
        if result.trades:
            trades_df = pd.DataFrame([trade.to_dict() for trade in result.trades])
            trades_file = output_path / f"{prefix}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            backtest_logger.info(f"Saved {len(result.trades)} trades to {trades_file}")
        
        # Save equity curve
        if not result.equity_curve.empty:
            equity_file = output_path / f"{prefix}_equity_curve.csv"
            result.equity_curve.to_csv(equity_file, index=False)
            backtest_logger.info(f"Saved equity curve to {equity_file}")
        
        # Save allocation history
        if not result.allocation_history.empty:
            allocation_file = output_path / f"{prefix}_allocation.csv"
            result.allocation_history.to_csv(allocation_file, index=False)
            backtest_logger.info(f"Saved allocation history to {allocation_file}")
        
        # Save performance metrics
        if result.performance_metrics:
            import json
            metrics_file = output_path / f"{prefix}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(result.performance_metrics, f, indent=2, default=str)
            backtest_logger.info(f"Saved performance metrics to {metrics_file}")
    
    def load_backtest_results(self, file_prefix: str, output_dir: str = "backtest_results") -> BacktestResult:
        """Load backtest results from files."""
        from pathlib import Path
        
        output_path = Path(output_dir)
        
        # Load trades
        trades_file = output_path / f"{file_prefix}_trades.csv"
        trades = []
        if trades_file.exists():
            trades_df = pd.read_csv(trades_file)
            # Convert back to Trade objects (simplified)
            trades = []  # Would need to implement proper deserialization
        
        # Load equity curve
        equity_file = output_path / f"{file_prefix}_equity_curve.csv"
        equity_curve = pd.DataFrame()
        if equity_file.exists():
            equity_curve = pd.read_csv(equity_file)
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        
        # Load allocation history
        allocation_file = output_path / f"{file_prefix}_allocation.csv"
        allocation_history = pd.DataFrame()
        if allocation_file.exists():
            allocation_history = pd.read_csv(allocation_file)
            allocation_history['timestamp'] = pd.to_datetime(allocation_history['timestamp'])
        
        # Load performance metrics
        metrics_file = output_path / f"{file_prefix}_metrics.json"
        performance_metrics = {}
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                performance_metrics = json.load(f)
        
        return BacktestResult(
            config=self.config,
            trades=trades,
            equity_curve=equity_curve,
            performance_metrics=performance_metrics,
            allocation_history=allocation_history
        )
