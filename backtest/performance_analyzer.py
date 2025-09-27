"""
Performance analysis and metrics calculation for backtesting results.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from utils.logging_utils import get_logger
from .trade_executor import Trade, Position

# Set up logging
perf_logger = get_logger('performance_analyzer')


class PerformanceAnalyzer:
    """Analyzes trading performance and calculates various metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        perf_logger.info(f"Initialized PerformanceAnalyzer with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """Calculate returns from equity curve."""
        if len(equity_curve) < 2:
            return pd.Series(dtype=float)
        
        portfolio_values = equity_curve['portfolio_value']
        returns = portfolio_values.pct_change().dropna()
        
        return returns
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod() - 1
    
    def calculate_drawdown(self, equity_curve: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate drawdown series.
        
        Returns:
            Tuple of (drawdown_series, peak_series)
        """
        if len(equity_curve) < 2:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        portfolio_values = equity_curve['portfolio_value']
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        
        return drawdown, peak
    
    def calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate maximum drawdown metrics."""
        drawdown, peak = self.calculate_drawdown(equity_curve)
        
        if drawdown.empty:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
        
        max_dd = drawdown.min()
        
        # Calculate maximum drawdown duration
        dd_start = None
        max_duration = 0
        current_duration = 0
        
        for i, (timestamp, dd_value) in enumerate(drawdown.items()):
            if dd_value < 0:
                if dd_start is None:
                    dd_start = timestamp
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if dd_start is not None:
                    max_duration = max(max_duration, current_duration)
                    dd_start = None
                    current_duration = 0
        
        # Check if we're still in drawdown at the end
        if dd_start is not None:
            max_duration = max(max_duration, current_duration)
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_duration
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0.0
    
    def calculate_calmar_ratio(self, equity_curve: pd.DataFrame, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = self.calculate_returns(equity_curve)
        annual_return = (1 + returns.mean()) ** periods_per_year - 1
        max_dd = self.calculate_max_drawdown(equity_curve)['max_drawdown']
        
        return annual_return / abs(max_dd) if max_dd != 0 else 0.0
    
    def calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate comprehensive trade metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_trade_duration': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        winning_trades = [t for t in trades if t.is_winning_trade()]
        losing_trades = [t for t in trades if not t.is_winning_trade()]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats(trades)
        
        # Average trade duration
        avg_duration = sum(t.duration_minutes for t in trades) / total_trades
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'total_pnl': sum(t.pnl for t in trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_consecutive_stats(self, trades: List[Trade]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not trades:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for trade in trades:
            if trade.is_winning_trade():
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            else:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def calculate_volatility_metrics(self, returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
        """Calculate volatility-related metrics."""
        if len(returns) < 2:
            return {'annual_volatility': 0.0, 'downside_volatility': 0.0}
        
        annual_volatility = returns.std() * np.sqrt(periods_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0.0
        
        return {
            'annual_volatility': annual_volatility,
            'downside_volatility': downside_volatility
        }
    
    def calculate_return_metrics(self, equity_curve: pd.DataFrame, periods_per_year: int = 252) -> Dict[str, float]:
        """Calculate return-related metrics."""
        if len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'monthly_return': 0.0,
                'daily_return': 0.0
            }
        
        returns = self.calculate_returns(equity_curve)
        initial_value = equity_curve['portfolio_value'].iloc[0]
        final_value = equity_curve['portfolio_value'].iloc[-1]
        
        total_return = (final_value - initial_value) / initial_value
        annual_return = (1 + returns.mean()) ** periods_per_year - 1
        monthly_return = (1 + returns.mean()) ** 21 - 1  # ~21 trading days per month
        daily_return = returns.mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': monthly_return,
            'daily_return': daily_return
        }
    
    def calculate_risk_metrics(self, equity_curve: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        if len(equity_curve) < 2 or returns.empty:
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0
            }
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def generate_performance_report(self, equity_curve: pd.DataFrame, trades: List[Trade], 
                                  initial_capital: float) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        returns = self.calculate_returns(equity_curve)
        
        # Basic metrics
        return_metrics = self.calculate_return_metrics(equity_curve)
        volatility_metrics = self.calculate_volatility_metrics(returns)
        drawdown_metrics = self.calculate_max_drawdown(equity_curve)
        trade_metrics = self.calculate_trade_metrics(trades)
        risk_metrics = self.calculate_risk_metrics(equity_curve, returns)
        
        # Risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(equity_curve)
        
        # Portfolio metrics
        final_value = equity_curve['portfolio_value'].iloc[-1] if len(equity_curve) > 0 else initial_capital
        
        performance_report = {
            # Portfolio Overview
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_pnl': final_value - initial_capital,
            
            # Return Metrics
            **return_metrics,
            
            # Risk Metrics
            **volatility_metrics,
            **drawdown_metrics,
            **risk_metrics,
            
            # Risk-Adjusted Returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Trade Metrics
            **trade_metrics,
            
            # Additional Metrics
            'trading_days': len(equity_curve),
            'avg_daily_return': returns.mean(),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }
        
        perf_logger.info(f"Generated performance report with {len(trades)} trades")
        
        return performance_report
    
    def create_performance_charts_data(self, equity_curve: pd.DataFrame, trades: List[Trade]) -> Dict[str, Any]:
        """Create data for performance visualization charts."""
        returns = self.calculate_returns(equity_curve)
        drawdown, peak = self.calculate_drawdown(equity_curve)
        cumulative_returns = self.calculate_cumulative_returns(returns)
        
        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Trade analysis
        trade_data = []
        for trade in trades:
            trade_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': trade.pnl,
                'return_percent': trade.get_return_percent(),
                'duration_minutes': trade.duration_minutes,
                'symbol': trade.symbol
            })
        
        return {
            'equity_curve': equity_curve.to_dict('records'),
            'returns': returns.to_dict(),
            'drawdown': drawdown.to_dict(),
            'cumulative_returns': cumulative_returns.to_dict(),
            'monthly_returns': monthly_returns.to_dict(),
            'trades': trade_data
        }
