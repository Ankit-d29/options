"""
Unit tests for trading strategies and indicators.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from strategies.supertrend import SupertrendIndicator, SupertrendStrategy
from strategies.strategy_runner import StrategyRunner, create_supertrend_strategy, run_supertrend_on_data


class TestTradingSignal:
    """Test cases for TradingSignal class."""
    
    def test_trading_signal_creation(self):
        """Test creating a TradingSignal object."""
        timestamp = datetime.now()
        signal = TradingSignal(
            timestamp=timestamp,
            symbol="NIFTY",
            signal_type=SignalType.BUY,
            price=18000.0,
            confidence=0.8,
            metadata={"test": "value"}
        )
        
        assert signal.timestamp == timestamp
        assert signal.symbol == "NIFTY"
        assert signal.signal_type == SignalType.BUY
        assert signal.price == 18000.0
        assert signal.confidence == 0.8
        assert signal.metadata == {"test": "value"}
    
    def test_trading_signal_to_dict(self):
        """Test converting TradingSignal to dictionary."""
        timestamp = datetime.now()
        signal = TradingSignal(
            timestamp=timestamp,
            symbol="NIFTY",
            signal_type=SignalType.SELL,
            price=18000.0,
            confidence=0.7
        )
        
        signal_dict = signal.to_dict()
        expected = {
            'timestamp': timestamp,
            'symbol': 'NIFTY',
            'signal_type': 'SELL',
            'price': 18000.0,
            'confidence': 0.7,
            'metadata': {}
        }
        
        assert signal_dict == expected


class TestBaseStrategy:
    """Test cases for BaseStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = BaseStrategy("TestStrategy", {"param1": "value1"})
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "TestStrategy"
        assert self.strategy.config == {"param1": "value1"}
        assert self.strategy.signals == []
    
    def test_generate_signal(self):
        """Test signal generation."""
        timestamp = datetime.now()
        signal = self.strategy.generate_signal(
            timestamp=timestamp,
            symbol="NIFTY",
            price=18000.0,
            signal_type=SignalType.BUY,
            confidence=0.8
        )
        
        assert len(self.strategy.signals) == 1
        assert signal.symbol == "NIFTY"
        assert signal.signal_type == SignalType.BUY
        assert signal.price == 18000.0
        assert signal.confidence == 0.8
    
    def test_get_signals(self):
        """Test getting signals."""
        # Add some signals
        self.strategy.generate_signal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            price=18000.0,
            signal_type=SignalType.BUY,
            confidence=0.8
        )
        
        signals = self.strategy.get_signals()
        assert len(signals) == 1
        assert signals[0].symbol == "NIFTY"
    
    def test_clear_signals(self):
        """Test clearing signals."""
        # Add some signals
        self.strategy.generate_signal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            price=18000.0,
            signal_type=SignalType.BUY,
            confidence=0.8
        )
        
        assert len(self.strategy.signals) == 1
        
        self.strategy.clear_signals()
        assert len(self.strategy.signals) == 0
    
    def test_get_last_signal(self):
        """Test getting last signal."""
        # Add multiple signals
        self.strategy.generate_signal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            price=18000.0,
            signal_type=SignalType.BUY,
            confidence=0.8
        )
        
        self.strategy.generate_signal(
            timestamp=datetime.now(),
            symbol="BANKNIFTY",
            price=45000.0,
            signal_type=SignalType.SELL,
            confidence=0.7
        )
        
        last_signal = self.strategy.get_last_signal()
        assert last_signal.symbol == "BANKNIFTY"
        
        nifty_last = self.strategy.get_last_signal("NIFTY")
        assert nifty_last.symbol == "NIFTY"
        assert nifty_last.signal_type == SignalType.BUY
    
    def test_save_and_load_signals(self):
        """Test saving and loading signals."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_signals.csv"
            
            # Add some signals
            self.strategy.generate_signal(
                timestamp=datetime.now(),
                symbol="NIFTY",
                price=18000.0,
                signal_type=SignalType.BUY,
                confidence=0.8,
                metadata={"test": "value"}
            )
            
            # Save signals
            self.strategy.save_signals(str(filepath))
            assert filepath.exists()
            
            # Create new strategy and load signals
            new_strategy = BaseStrategy("NewStrategy")
            new_strategy.load_signals(str(filepath))
            
            assert len(new_strategy.signals) == 1
            assert new_strategy.signals[0].symbol == "NIFTY"
            assert new_strategy.signals[0].signal_type == SignalType.BUY
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Add some signals
        self.strategy.generate_signal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            price=18000.0,
            signal_type=SignalType.BUY,
            confidence=0.8
        )
        
        self.strategy.generate_signal(
            timestamp=datetime.now(),
            symbol="NIFTY",
            price=18100.0,
            signal_type=SignalType.SELL,
            confidence=0.7
        )
        
        metrics = self.strategy.get_performance_metrics()
        
        assert metrics['total_signals'] == 2
        assert metrics['buy_signals'] == 1
        assert metrics['sell_signals'] == 1
        assert metrics['avg_confidence'] == 0.75
        assert metrics['strategy_name'] == "TestStrategy"


class TestSupertrendIndicator:
    """Test cases for SupertrendIndicator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.indicator = SupertrendIndicator(period=10, multiplier=3.0)
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self, length: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1min')
        
        # Create realistic price data with trend
        base_price = 18000
        price_changes = np.random.normal(0, 0.001, length).cumsum()
        prices = base_price * (1 + price_changes)
        
        # Create OHLC data
        highs = prices * (1 + np.random.uniform(0, 0.005, length))
        lows = prices * (1 - np.random.uniform(0, 0.005, length))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        volumes = np.random.randint(100, 1000, length)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    def test_indicator_initialization(self):
        """Test indicator initialization."""
        assert self.indicator.period == 10
        assert self.indicator.multiplier == 3.0
    
    def test_calculate_atr(self):
        """Test ATR calculation."""
        atr = self.indicator.calculate_atr(self.sample_data)
        
        assert len(atr) == len(self.sample_data)
        assert not atr.isna().any()
        assert (atr > 0).all()
    
    def test_calculate_supertrend(self):
        """Test Supertrend calculation."""
        supertrend, trend = self.indicator.calculate_supertrend(self.sample_data)
        
        assert len(supertrend) == len(self.sample_data)
        assert len(trend) == len(self.sample_data)
        assert not supertrend.isna().any()
        assert not trend.isna().any()
        assert (trend.isin([1, -1])).all()
    
    def test_add_to_dataframe(self):
        """Test adding Supertrend to DataFrame."""
        result_df = self.indicator.add_to_dataframe(self.sample_data)
        
        required_columns = ['supertrend', 'supertrend_trend', 'supertrend_atr', 'supertrend_signal']
        for col in required_columns:
            assert col in result_df.columns
        
        assert not result_df['supertrend'].isna().any()
        assert not result_df['supertrend_trend'].isna().any()
        assert not result_df['supertrend_atr'].isna().any()
    
    def test_get_signals(self):
        """Test signal generation."""
        signals = self.indicator.get_signals(self.sample_data)
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert signal.confidence > 0


class TestSupertrendStrategy:
    """Test cases for SupertrendStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = SupertrendStrategy(period=10, multiplier=3.0)
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self, length: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1min')
        
        # Create realistic price data with trend
        base_price = 18000
        price_changes = np.random.normal(0, 0.001, length).cumsum()
        prices = base_price * (1 + price_changes)
        
        # Create OHLC data
        highs = prices * (1 + np.random.uniform(0, 0.005, length))
        lows = prices * (1 - np.random.uniform(0, 0.005, length))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        volumes = np.random.randint(100, 1000, length)
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "SupertrendStrategy"
        assert self.strategy.period == 10
        assert self.strategy.multiplier == 3.0
        assert isinstance(self.strategy.supertrend, SupertrendIndicator)
    
    def test_calculate_signals(self):
        """Test signal calculation."""
        result_df = self.strategy.calculate_signals(self.sample_data)
        
        # Check that Supertrend columns were added
        assert 'supertrend' in result_df.columns
        assert 'supertrend_trend' in result_df.columns
        assert 'supertrend_signal' in result_df.columns
        
        # Check that signals were generated
        signals = self.strategy.get_signals()
        assert isinstance(signals, list)
    
    def test_get_trend_status(self):
        """Test getting trend status."""
        status = self.strategy.get_trend_status(self.sample_data)
        
        required_keys = ['trend', 'supertrend_value', 'current_price', 'confidence']
        for key in required_keys:
            assert key in status
        
        assert status['trend'] in ['uptrend', 'downtrend']
        assert status['confidence'] > 0
    
    def test_should_enter_long(self):
        """Test long entry conditions."""
        # This is a basic test - in practice, we'd need specific data patterns
        result = self.strategy.should_enter_long(self.sample_data)
        assert isinstance(result, (bool, np.bool_))
    
    def test_should_enter_short(self):
        """Test short entry conditions."""
        # This is a basic test - in practice, we'd need specific data patterns
        result = self.strategy.should_enter_short(self.sample_data)
        assert isinstance(result, (bool, np.bool_))


class TestStrategyRunner:
    """Test cases for StrategyRunner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = StrategyRunner()
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self, length: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1min')
        
        base_price = 18000
        price_changes = np.random.normal(0, 0.001, length).cumsum()
        prices = base_price * (1 + price_changes)
        
        highs = prices * (1 + np.random.uniform(0, 0.005, length))
        lows = prices * (1 - np.random.uniform(0, 0.005, length))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        volumes = np.random.randint(100, 1000, length)
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        assert isinstance(self.runner.strategies, dict)
        assert isinstance(self.runner.results, dict)
        assert self.runner.output_dir.exists()
    
    def test_add_strategy(self):
        """Test adding strategy."""
        strategy = create_supertrend_strategy()
        self.runner.add_strategy("test_supertrend", strategy)
        
        assert "test_supertrend" in self.runner.strategies
        assert self.runner.strategies["test_supertrend"] == strategy
    
    def test_run_strategy(self):
        """Test running a strategy."""
        strategy = create_supertrend_strategy()
        self.runner.add_strategy("test_supertrend", strategy)
        
        result = self.runner.run_strategy("test_supertrend", self.sample_data, "NIFTY")
        
        assert 'strategy_name' in result
        assert 'symbol' in result
        assert 'data' in result
        assert 'signals' in result
        assert 'metrics' in result
        
        assert result['strategy_name'] == "test_supertrend"
        assert result['symbol'] == "NIFTY"
        assert isinstance(result['signals'], list)
    
    def test_run_strategy_with_nonexistent_strategy(self):
        """Test running a non-existent strategy."""
        with pytest.raises(ValueError):
            self.runner.run_strategy("nonexistent", self.sample_data)
    
    def test_run_multiple_strategies(self):
        """Test running multiple strategies."""
        strategy1 = create_supertrend_strategy(period=10)
        strategy2 = create_supertrend_strategy(period=20)
        
        self.runner.add_strategy("supertrend_10", strategy1)
        self.runner.add_strategy("supertrend_20", strategy2)
        
        results = self.runner.run_multiple_strategies(
            ["supertrend_10", "supertrend_20"], 
            self.sample_data, 
            "NIFTY"
        )
        
        assert len(results) == 2
        assert "supertrend_10" in results
        assert "supertrend_20" in results
    
    def test_compare_strategies(self):
        """Test strategy comparison."""
        strategy1 = create_supertrend_strategy(period=10)
        strategy2 = create_supertrend_strategy(period=20)
        
        self.runner.add_strategy("supertrend_10", strategy1)
        self.runner.add_strategy("supertrend_20", strategy2)
        
        comparison_df = self.runner.compare_strategies(
            ["supertrend_10", "supertrend_20"], 
            self.sample_data, 
            "NIFTY"
        )
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'strategy' in comparison_df.columns
        assert 'total_signals' in comparison_df.columns
        assert len(comparison_df) == 2
    
    def test_get_strategy_results(self):
        """Test getting strategy results."""
        strategy = create_supertrend_strategy()
        self.runner.add_strategy("test_supertrend", strategy)
        
        # Run strategy first
        self.runner.run_strategy("test_supertrend", self.sample_data, "NIFTY")
        
        # Get results
        results = self.runner.get_strategy_results("test_supertrend")
        assert len(results) > 0
        
        all_results = self.runner.get_strategy_results()
        assert len(all_results) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_supertrend_strategy(self):
        """Test creating Supertrend strategy."""
        strategy = create_supertrend_strategy(period=15, multiplier=2.5)
        
        assert isinstance(strategy, SupertrendStrategy)
        assert strategy.period == 15
        assert strategy.multiplier == 2.5
    
    def test_run_supertrend_on_data(self):
        """Test running Supertrend on data."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='1min')
        prices = 18000 + np.random.normal(0, 100, 30)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(100, 1000, 30)
        })
        
        result = run_supertrend_on_data(df, "NIFTY", period=10, multiplier=3.0)
        
        assert 'strategy_name' in result
        assert 'signals' in result
        assert 'metrics' in result


if __name__ == "__main__":
    pytest.main([__file__])
