"""
Unit tests for backtesting engine components.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from backtest.trade_executor import TradeExecutor, Trade, Position, TradeStatus, OrderType
from backtest.performance_analyzer import PerformanceAnalyzer
from backtest.portfolio_manager import PortfolioManager, PortfolioConfig
from backtest.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from strategies.supertrend import SupertrendStrategy
from strategies.base_strategy import TradingSignal, SignalType


class TestTrade:
    """Test cases for Trade class."""
    
    def test_trade_creation(self):
        """Test creating a Trade object."""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=1)
        
        trade = Trade(
            symbol="NIFTY",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=18000.0,
            exit_price=18100.0,
            quantity=100,
            position_type="LONG",
            pnl=100.0,
            commission=10.0,
            slippage=5.0,
            duration_minutes=60
        )
        
        assert trade.symbol == "NIFTY"
        assert trade.entry_price == 18000.0
        assert trade.exit_price == 18100.0
        assert trade.quantity == 100
        assert trade.position_type == "LONG"
        assert trade.pnl == 100.0
        assert trade.is_winning_trade() == True
    
    def test_trade_return_percent(self):
        """Test return percentage calculation."""
        trade = Trade(
            symbol="NIFTY",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=18000.0,
            exit_price=18100.0,
            quantity=100,
            position_type="LONG",
            pnl=100.0,
            commission=10.0,
            slippage=5.0,
            duration_minutes=60
        )
        
        # Long position: (18100 - 18000) / 18000 * 100 = 0.556%
        assert abs(trade.get_return_percent() - 0.556) < 0.001
        
        # Short position
        trade_short = Trade(
            symbol="NIFTY",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=18000.0,
            exit_price=17900.0,
            quantity=100,
            position_type="SHORT",
            pnl=100.0,
            commission=10.0,
            slippage=5.0,
            duration_minutes=60
        )
        
        # Short position: (18000 - 17900) / 18000 * 100 = 0.556%
        assert abs(trade_short.get_return_percent() - 0.556) < 0.001
    
    def test_trade_to_dict(self):
        """Test converting trade to dictionary."""
        trade = Trade(
            symbol="NIFTY",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=18000.0,
            exit_price=18100.0,
            quantity=100,
            position_type="LONG",
            pnl=100.0,
            commission=10.0,
            slippage=5.0,
            duration_minutes=60
        )
        
        trade_dict = trade.to_dict()
        
        assert 'symbol' in trade_dict
        assert 'entry_price' in trade_dict
        assert 'exit_price' in trade_dict
        assert 'pnl' in trade_dict
        assert 'return_percent' in trade_dict
        assert 'is_winning' in trade_dict


class TestPosition:
    """Test cases for Position class."""
    
    def test_position_creation(self):
        """Test creating a Position object."""
        position = Position(
            symbol="NIFTY",
            quantity=100,
            entry_price=18000.0,
            entry_time=datetime.now(),
            position_type="LONG"
        )
        
        assert position.symbol == "NIFTY"
        assert position.quantity == 100
        assert position.entry_price == 18000.0
        assert position.position_type == "LONG"
        assert position.unrealized_pnl == 0.0
    
    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions."""
        position = Position(
            symbol="NIFTY",
            quantity=100,
            entry_price=18000.0,
            entry_time=datetime.now(),
            position_type="LONG"
        )
        
        # Test long position P&L
        position.update_unrealized_pnl(18100.0, 10.0)
        expected_pnl = (18100.0 - 18000.0) * 100 - 10.0  # 9990.0
        assert abs(position.unrealized_pnl - expected_pnl) < 0.01
        
        # Test short position P&L
        position_short = Position(
            symbol="NIFTY",
            quantity=100,
            entry_price=18000.0,
            entry_time=datetime.now(),
            position_type="SHORT"
        )
        
        position_short.update_unrealized_pnl(17900.0, 10.0)
        expected_pnl_short = (18000.0 - 17900.0) * 100 - 10.0  # 9990.0
        assert abs(position_short.unrealized_pnl - expected_pnl_short) < 0.01
    
    def test_position_total_pnl(self):
        """Test total P&L calculation."""
        position = Position(
            symbol="NIFTY",
            quantity=100,
            entry_price=18000.0,
            entry_time=datetime.now(),
            position_type="LONG",
            realized_pnl=500.0,
            unrealized_pnl=200.0
        )
        
        assert position.get_total_pnl() == 700.0


class TestTradeExecutor:
    """Test cases for TradeExecutor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = TradeExecutor(initial_capital=100000, commission_rate=0.001)
    
    def test_executor_initialization(self):
        """Test TradeExecutor initialization."""
        assert self.executor.initial_capital == 100000
        assert self.executor.cash == 100000
        assert self.executor.commission_rate == 0.001
        assert len(self.executor.positions) == 0
        assert len(self.executor.trades) == 0
    
    def test_can_open_position(self):
        """Test position opening constraints."""
        # Should be able to open position initially
        assert self.executor.can_open_position("NIFTY") == True
        
        # Create a position
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0
        )
        
        success = self.executor.execute_entry(signal, 18000.0, datetime.now())
        assert success == True
        
        # Should not be able to open another position for same symbol
        assert self.executor.can_open_position("NIFTY") == False
        
        # Should be able to open position for different symbol
        assert self.executor.can_open_position("BANKNIFTY") == True
    
    def test_execute_entry_long(self):
        """Test executing a long position entry."""
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0
        )
        
        initial_cash = self.executor.cash
        success = self.executor.execute_entry(signal, 18000.0, datetime.now())
        
        assert success == True
        assert "NIFTY" in self.executor.positions
        assert self.executor.cash < initial_cash
        
        position = self.executor.positions["NIFTY"]
        assert position.position_type == "LONG"
        assert position.quantity > 0
    
    def test_execute_entry_short(self):
        """Test executing a short position entry."""
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.SELL,
            price=18000.0
        )
        
        initial_cash = self.executor.cash
        success = self.executor.execute_entry(signal, 18000.0, datetime.now())
        
        assert success == True
        assert "NIFTY" in self.executor.positions
        
        position = self.executor.positions["NIFTY"]
        assert position.position_type == "SHORT"
        assert position.quantity > 0
    
    def test_execute_exit(self):
        """Test executing a position exit."""
        # First create a position
        entry_signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0
        )
        
        self.executor.execute_entry(entry_signal, 18000.0, datetime.now())
        
        # Now exit the position
        exit_signal = TradingSignal(
            timestamp=datetime.now() + timedelta(hours=1),
            symbol="NIFTY",
            signal_type=SignalType.SELL,
            price=18100.0
        )
        
        trade = self.executor.execute_exit(exit_signal, 18100.0, datetime.now() + timedelta(hours=1))
        
        assert trade is not None
        assert trade.symbol == "NIFTY"
        assert trade.position_type == "LONG"
        assert "NIFTY" not in self.executor.positions
        assert len(self.executor.trades) == 1
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        # Create a position
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0
        )
        
        self.executor.execute_entry(signal, 18000.0, datetime.now())
        
        # Calculate portfolio value with current prices
        current_prices = {"NIFTY": 18100.0}
        portfolio_value = self.executor.get_portfolio_value(current_prices)
        
        assert portfolio_value > 0  # Should have positive portfolio value
    
    def test_equity_curve_update(self):
        """Test equity curve updating."""
        current_time = datetime.now()
        current_prices = {"NIFTY": 18000.0}
        
        self.executor.update_equity_curve(current_time, current_prices)
        
        assert len(self.executor.equity_curve) == 1
        assert self.executor.equity_curve[0]['timestamp'] == current_time
        assert self.executor.equity_curve[0]['portfolio_value'] == self.executor.initial_capital
    
    def test_reset(self):
        """Test executor reset."""
        # Create some state
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0
        )
        
        self.executor.execute_entry(signal, 18000.0, datetime.now())
        self.executor.update_equity_curve(datetime.now(), {})
        
        # Reset
        self.executor.reset()
        
        assert self.executor.cash == self.executor.initial_capital
        assert len(self.executor.positions) == 0
        assert len(self.executor.trades) == 0
        assert len(self.executor.equity_curve) == 0


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        assert self.analyzer.risk_free_rate == 0.02
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        # Create sample equity curve
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': [100000, 101000, 102000, 101500, 102500, 103000, 102800, 103500, 104000, 103500]
        })
        
        returns = self.analyzer.calculate_returns(equity_curve)
        
        assert len(returns) == 9  # One less than original due to pct_change
        assert abs(returns.iloc[0] - 0.01) < 0.0001  # 1% return from 100000 to 101000
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        # Create equity curve with drawdown
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': [100000, 110000, 120000, 105000, 115000]  # Peak at 120000, drawdown to 105000
        })
        
        drawdown, peak = self.analyzer.calculate_drawdown(equity_curve)
        
        # Maximum drawdown should be (105000 - 120000) / 120000 = -0.125
        assert abs(drawdown.min() - (-0.125)) < 0.001
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create returns with known characteristics
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        
        sharpe = self.analyzer.calculate_sharpe_ratio(returns)
        
        # Should be positive for positive mean returns
        assert sharpe > 0
    
    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        # Create sample trades
        trades = [
            Trade(
                symbol="NIFTY",
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=18000.0,
                exit_price=18100.0,
                quantity=100,
                position_type="LONG",
                pnl=100.0,
                commission=10.0,
                slippage=5.0,
                duration_minutes=60
            ),
            Trade(
                symbol="NIFTY",
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=18000.0,
                exit_price=17900.0,
                quantity=100,
                position_type="LONG",
                pnl=-100.0,
                commission=10.0,
                slippage=5.0,
                duration_minutes=60
            )
        ]
        
        metrics = self.analyzer.calculate_trade_metrics(trades)
        
        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == 50.0
        assert metrics['profit_factor'] == 1.0  # Equal profits and losses
    
    def test_generate_performance_report(self):
        """Test comprehensive performance report generation."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': [100000 + i * 1000 for i in range(10)]
        })
        
        trades = [
            Trade(
                symbol="NIFTY",
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=18000.0,
                exit_price=18100.0,
                quantity=100,
                position_type="LONG",
                pnl=100.0,
                commission=10.0,
                slippage=5.0,
                duration_minutes=60
            )
        ]
        
        report = self.analyzer.generate_performance_report(equity_curve, trades, 100000)
        
        assert 'total_return' in report
        assert 'annual_return' in report
        assert 'sharpe_ratio' in report
        assert 'max_drawdown' in report
        assert 'total_trades' in report


class TestPortfolioManager:
    """Test cases for PortfolioManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PortfolioConfig(
            initial_capital=100000,
            max_positions=5,
            risk_per_trade_percent=0.02
        )
        self.manager = PortfolioManager(self.config)
    
    def test_manager_initialization(self):
        """Test PortfolioManager initialization."""
        assert self.manager.config.initial_capital == 100000
        assert self.manager.config.max_positions == 5
        assert len(self.manager.allocation_history) == 0
    
    def test_execute_signals(self):
        """Test signal execution."""
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                symbol="NIFTY",
                signal_type=SignalType.BUY,
                price=18000.0
            )
        ]
        
        current_prices = {"NIFTY": 18000.0}
        current_time = datetime.now()
        
        trades = self.manager.execute_signals(signals, current_prices, current_time)
        
        # Should have executed the signal
        assert len(trades) == 0  # No exit trades, only entry
        assert "NIFTY" in self.manager.trade_executor.positions
    
    def test_update_portfolio(self):
        """Test portfolio update."""
        current_time = datetime.now()
        current_prices = {"NIFTY": 18000.0}
        
        # First create a position
        signals = [
            TradingSignal(
                timestamp=current_time,
                symbol="NIFTY",
                signal_type=SignalType.BUY,
                price=18000.0
            )
        ]
        
        self.manager.execute_signals(signals, current_prices, current_time)
        
        # Update portfolio
        self.manager.update_portfolio(current_time, current_prices)
        
        assert len(self.manager.allocation_history) == 1
        assert self.manager.allocation_history[0]['positions_count'] == 1
    
    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        summary = self.manager.get_portfolio_summary()
        
        assert 'portfolio_value' in summary
        assert 'cash' in summary
        assert 'positions_count' in summary
        assert summary['portfolio_value'] == self.config.initial_capital
    
    def test_reset(self):
        """Test portfolio manager reset."""
        # Create some state
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                symbol="NIFTY",
                signal_type=SignalType.BUY,
                price=18000.0
            )
        ]
        
        self.manager.execute_signals(signals, {"NIFTY": 18000.0}, datetime.now())
        self.manager.update_portfolio(datetime.now(), {"NIFTY": 18000.0})
        
        # Reset
        self.manager.reset()
        
        assert len(self.manager.trade_executor.positions) == 0
        assert len(self.manager.allocation_history) == 0


class TestBacktestEngine:
    """Test cases for BacktestEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            initial_capital=100000,
            commission_rate=0.001,
            max_positions=5
        )
        self.engine = BacktestEngine(self.config)
    
    def test_engine_initialization(self):
        """Test BacktestEngine initialization."""
        assert self.engine.config.initial_capital == 100000
        assert self.engine.config.commission_rate == 0.001
    
    def test_run_backtest(self):
        """Test running a backtest."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': [18000 + i * 10 for i in range(50)],
            'high': [18050 + i * 10 for i in range(50)],
            'low': [17950 + i * 10 for i in range(50)],
            'close': [18000 + i * 10 for i in range(50)],
            'volume': [1000] * 50
        })
        
        # Create strategy
        strategy = SupertrendStrategy(period=10, multiplier=2.0)
        
        # Run backtest
        result = self.engine.run_backtest(strategy, data, "NIFTY")
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "SupertrendStrategy"
        assert result.symbol == "NIFTY"
        assert result.execution_time > 0
    
    def test_run_strategy_comparison(self):
        """Test running strategy comparison."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': [18000 + i * 10 for i in range(30)],
            'high': [18050 + i * 10 for i in range(30)],
            'low': [17950 + i * 10 for i in range(30)],
            'close': [18000 + i * 10 for i in range(30)],
            'volume': [1000] * 30
        })
        
        # Create strategies with different names
        strategy1 = SupertrendStrategy(period=10, multiplier=2.0)
        strategy1.name = "Supertrend_10_2"
        strategy2 = SupertrendStrategy(period=20, multiplier=3.0)
        strategy2.name = "Supertrend_20_3"
        strategies = [strategy1, strategy2]
        
        # Run comparison
        results = self.engine.run_strategy_comparison(strategies, data, "NIFTY")
        
        assert len(results) == 2
        assert "Supertrend_10_2" in results
        assert "Supertrend_20_3" in results


class TestBacktestResult:
    """Test cases for BacktestResult class."""
    
    def test_result_creation(self):
        """Test BacktestResult creation."""
        config = BacktestConfig()
        result = BacktestResult(config)
        
        assert result.config == config
        assert len(result.trades) == 0
        assert result.execution_time == 0.0
    
    def test_result_summary(self):
        """Test BacktestResult summary."""
        config = BacktestConfig()
        trades = [
            Trade(
                symbol="NIFTY",
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=18000.0,
                exit_price=18100.0,
                quantity=100,
                position_type="LONG",
                pnl=100.0,
                commission=10.0,
                slippage=5.0,
                duration_minutes=60
            )
        ]
        
        result = BacktestResult(config, trades=trades)
        summary = result.get_summary()
        
        assert summary['total_trades'] == 1
        assert summary['total_pnl'] == 100.0
        assert summary['win_rate'] == 100.0


if __name__ == "__main__":
    pytest.main([__file__])
