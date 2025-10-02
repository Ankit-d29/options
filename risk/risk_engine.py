"""
Core risk management engine for the options trading system.

This module provides the central risk management engine that coordinates
all risk-related activities including position limits, portfolio risk,
margin management, and emergency controls.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from broker.base_broker import BrokerOrder, BrokerPosition, BrokerQuote
from strategies.base_strategy import TradingSignal
from utils.logging_utils import get_logger

# Setup logging
risk_logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskViolationType(Enum):
    """Risk violation type enumeration."""
    POSITION_SIZE = "POSITION_SIZE"
    PORTFOLIO_LOSS = "PORTFOLIO_LOSS"
    DAILY_LOSS = "DAILY_LOSS"
    MARGIN_CALL = "MARGIN_CALL"
    DRAWDOWN = "DRAWDOWN"
    CONCENTRATION = "CONCENTRATION"
    LEVERAGE = "LEVERAGE"
    VOLATILITY = "VOLATILITY"


@dataclass
class RiskViolation:
    """Risk violation data structure."""
    violation_id: str
    violation_type: RiskViolationType
    risk_level: RiskLevel
    message: str
    timestamp: datetime
    current_value: float
    limit_value: float
    violation_percent: float
    recommended_action: str
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Portfolio limits
    max_portfolio_loss_percent: float = 0.05  # 5%
    max_daily_loss_percent: float = 0.02      # 2%
    max_drawdown_percent: float = 0.10        # 10%
    
    # Position limits
    max_position_size_percent: float = 0.10   # 10%
    max_concentration_percent: float = 0.25   # 25%
    max_open_positions: int = 10
    
    # Trading limits
    max_daily_trades: int = 50
    max_trades_per_hour: int = 10
    
    # Risk controls
    enable_kill_switch: bool = True
    enable_position_limits: bool = True
    enable_portfolio_limits: bool = True
    enable_margin_checks: bool = True
    
    # Alert thresholds
    warning_threshold_percent: float = 0.75   # 75% of limit
    critical_threshold_percent: float = 0.90  # 90% of limit
    
    # Margin settings
    margin_requirement_percent: float = 0.20  # 20%
    maintenance_margin_percent: float = 0.15  # 15%
    
    # Volatility controls
    max_volatility_percent: float = 0.05      # 5% daily volatility
    enable_volatility_stops: bool = True


@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    timestamp: datetime
    portfolio_value: float
    available_margin: float
    used_margin: float
    margin_utilization_percent: float
    portfolio_loss_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    max_drawdown_percent: float
    current_drawdown_percent: float
    position_count: int
    largest_position_percent: float
    concentration_risk_percent: float
    daily_trades: int
    hourly_trades: int
    volatility_percent: float
    risk_score: float
    violations: List[RiskViolation] = field(default_factory=list)


class RiskEngine:
    """
    Central risk management engine.
    
    This class coordinates all risk management activities including
    position limits, portfolio risk assessment, margin management,
    and emergency controls.
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize risk engine.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.is_active = False
        self.start_time = None
        
        # Risk state
        self.current_metrics: Optional[RiskMetrics] = None
        self.violations: List[RiskViolation] = []
        self.kill_switch_triggered = False
        
        # Callbacks
        self.risk_callbacks: List[Callable] = []
        self.violation_callbacks: List[Callable] = []
        
        # Risk monitoring
        self.daily_pnl_start = 0.0
        self.portfolio_high_water_mark = 0.0
        self.hourly_trade_count = 0
        self.last_hour_reset = datetime.now()
        
        risk_logger.info(f"Initialized RiskEngine with config: {config}")

    async def start(self) -> bool:
        """
        Start the risk management engine.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            self.is_active = True
            self.start_time = datetime.now()
            self.kill_switch_triggered = False
            
            # Reset daily metrics
            self.daily_pnl_start = 0.0
            self.portfolio_high_water_mark = 0.0
            self.hourly_trade_count = 0
            self.last_hour_reset = datetime.now()
            
            risk_logger.info("Risk management engine started")
            return True
            
        except Exception as e:
            risk_logger.error(f"Failed to start risk engine: {e}")
            return False

    async def stop(self) -> bool:
        """
        Stop the risk management engine.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            self.is_active = False
            
            # Trigger kill switch if needed
            if self.kill_switch_triggered:
                await self._trigger_kill_switch("Risk engine stopping")
            
            risk_logger.info("Risk management engine stopped")
            return True
            
        except Exception as e:
            risk_logger.error(f"Failed to stop risk engine: {e}")
            return False

    async def check_signal_risk(self, signal: TradingSignal, 
                              current_portfolio_value: float,
                              positions: List[BrokerPosition]) -> Dict[str, Any]:
        """
        Check risk for a trading signal before execution.
        
        Args:
            signal: Trading signal to check
            current_portfolio_value: Current portfolio value
            positions: Current positions
            
        Returns:
            Risk assessment result
        """
        try:
            if not self.is_active:
                return {"allowed": True, "reason": "Risk engine not active"}
            
            if self.kill_switch_triggered:
                return {"allowed": False, "reason": "Kill switch triggered"}
            
            # Check position size limits
            position_risk = await self._check_position_size_risk(
                signal, current_portfolio_value
            )
            if not position_risk["allowed"]:
                return position_risk
            
            # Check portfolio limits
            portfolio_risk = await self._check_portfolio_risk(
                signal, current_portfolio_value, positions
            )
            if not portfolio_risk["allowed"]:
                return portfolio_risk
            
            # Check trading limits
            trading_risk = await self._check_trading_limits(signal)
            if not trading_risk["allowed"]:
                return trading_risk
            
            # Check concentration risk
            concentration_risk = await self._check_concentration_risk(
                signal, positions
            )
            if not concentration_risk["allowed"]:
                return concentration_risk
            
            return {"allowed": True, "reason": "All risk checks passed"}
            
        except Exception as e:
            risk_logger.error(f"Error checking signal risk: {e}")
            return {"allowed": False, "reason": f"Risk check error: {e}"}

    async def update_portfolio_metrics(self, portfolio_value: float,
                                     available_margin: float,
                                     used_margin: float,
                                     positions: List[BrokerPosition],
                                     daily_trades: int) -> RiskMetrics:
        """
        Update portfolio risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            available_margin: Available margin
            used_margin: Used margin
            positions: Current positions
            daily_trades: Number of trades today
            
        Returns:
            Updated risk metrics
        """
        try:
            # Calculate metrics
            margin_utilization = (used_margin / available_margin * 100) if available_margin > 0 else 0
            
            # Calculate daily P&L
            daily_pnl = portfolio_value - self.daily_pnl_start
            daily_pnl_percent = (daily_pnl / self.daily_pnl_start * 100) if self.daily_pnl_start > 0 else 0
            
            # Update high water mark
            if portfolio_value > self.portfolio_high_water_mark:
                self.portfolio_high_water_mark = portfolio_value
            
            # Calculate drawdown
            current_drawdown = (self.portfolio_high_water_mark - portfolio_value) / self.portfolio_high_water_mark * 100 if self.portfolio_high_water_mark > 0 else 0
            
            # Calculate position metrics
            largest_position_percent = 0.0
            total_position_value = 0.0
            
            for position in positions:
                position_value = float(position.market_value)
                total_position_value += position_value
                position_percent = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
                if position_percent > largest_position_percent:
                    largest_position_percent = position_percent
            
            # Calculate concentration risk (top 3 positions)
            position_values = [float(pos.market_value) for pos in positions]
            position_values.sort(reverse=True)
            top_3_value = sum(position_values[:3])
            concentration_risk = (top_3_value / portfolio_value * 100) if portfolio_value > 0 else 0
            
            # Calculate volatility (simplified)
            volatility = self._calculate_volatility(positions)
            
            # Calculate risk score (0-100, higher = riskier)
            risk_score = self._calculate_risk_score(
                margin_utilization, daily_pnl_percent, current_drawdown,
                largest_position_percent, concentration_risk, volatility
            )
            
            # Check for violations
            violations = await self._check_violations(
                portfolio_value, daily_pnl_percent, current_drawdown,
                largest_position_percent, concentration_risk, daily_trades
            )
            
            # Create metrics
            self.current_metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                available_margin=available_margin,
                used_margin=used_margin,
                margin_utilization_percent=margin_utilization,
                portfolio_loss_percent=abs(daily_pnl_percent) if daily_pnl_percent < 0 else 0,
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_percent,
                max_drawdown_percent=self.config.max_drawdown_percent * 100,
                current_drawdown_percent=current_drawdown,
                position_count=len(positions),
                largest_position_percent=largest_position_percent,
                concentration_risk_percent=concentration_risk,
                daily_trades=daily_trades,
                hourly_trades=self.hourly_trade_count,
                volatility_percent=volatility,
                risk_score=risk_score,
                violations=violations
            )
            
            # Trigger callbacks
            await self._trigger_risk_callbacks(self.current_metrics)
            
            return self.current_metrics
            
        except Exception as e:
            risk_logger.error(f"Error updating portfolio metrics: {e}")
            raise

    async def trigger_kill_switch(self, reason: str) -> bool:
        """
        Trigger the emergency kill switch.
        
        Args:
            reason: Reason for triggering kill switch
            
        Returns:
            True if kill switch triggered successfully
        """
        try:
            if self.kill_switch_triggered:
                risk_logger.warning("Kill switch already triggered")
                return True
            
            self.kill_switch_triggered = True
            
            # Create violation
            violation = RiskViolation(
                violation_id=f"kill_switch_{datetime.now().timestamp()}",
                violation_type=RiskViolationType.PORTFOLIO_LOSS,
                risk_level=RiskLevel.CRITICAL,
                message=f"Kill switch triggered: {reason}",
                timestamp=datetime.now(),
                current_value=0.0,
                limit_value=0.0,
                violation_percent=100.0,
                recommended_action="Close all positions immediately"
            )
            
            self.violations.append(violation)
            
            # Trigger callbacks
            await self._trigger_violation_callbacks(violation)
            
            risk_logger.critical(f"Kill switch triggered: {reason}")
            return True
            
        except Exception as e:
            risk_logger.error(f"Error triggering kill switch: {e}")
            return False

    def add_risk_callback(self, callback: Callable):
        """Add risk metrics callback."""
        self.risk_callbacks.append(callback)

    def add_violation_callback(self, callback: Callable):
        """Add risk violation callback."""
        self.violation_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics."""
        return self.current_metrics

    def get_violations(self) -> List[RiskViolation]:
        """Get all risk violations."""
        return self.violations

    def is_kill_switch_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self.kill_switch_triggered

    async def _check_position_size_risk(self, signal: TradingSignal, 
                                      portfolio_value: float) -> Dict[str, Any]:
        """Check position size risk."""
        if not self.config.enable_position_limits:
            return {"allowed": True}
        
        # Calculate position value
        position_value = float(signal.price * 100)  # Assuming 100 quantity for options
        position_percent = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        if position_percent > self.config.max_position_size_percent * 100:
            return {
                "allowed": False,
                "reason": f"Position size {position_percent:.2f}% exceeds limit {self.config.max_position_size_percent * 100:.2f}%"
            }
        
        return {"allowed": True}

    async def _check_portfolio_risk(self, signal: TradingSignal,
                                  portfolio_value: float,
                                  positions: List[BrokerPosition]) -> Dict[str, Any]:
        """Check portfolio-level risk."""
        if not self.config.enable_portfolio_limits:
            return {"allowed": True}
        
        # Check daily loss limit
        if self.current_metrics and self.current_metrics.daily_pnl_percent < -self.config.max_daily_loss_percent * 100:
            return {
                "allowed": False,
                "reason": f"Daily loss {abs(self.current_metrics.daily_pnl_percent):.2f}% exceeds limit {self.config.max_daily_loss_percent * 100:.2f}%"
            }
        
        # Check drawdown limit
        if self.current_metrics and self.current_metrics.current_drawdown_percent > self.config.max_drawdown_percent * 100:
            return {
                "allowed": False,
                "reason": f"Drawdown {self.current_metrics.current_drawdown_percent:.2f}% exceeds limit {self.config.max_drawdown_percent * 100:.2f}%"
            }
        
        return {"allowed": True}

    async def _check_trading_limits(self, signal: TradingSignal) -> Dict[str, Any]:
        """Check trading limits."""
        # Check daily trade limit
        if self.current_metrics and self.current_metrics.daily_trades >= self.config.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"Daily trade limit {self.config.max_daily_trades} exceeded"
            }
        
        # Check hourly trade limit
        if self.hourly_trade_count >= self.config.max_trades_per_hour:
            return {
                "allowed": False,
                "reason": f"Hourly trade limit {self.config.max_trades_per_hour} exceeded"
            }
        
        return {"allowed": True}

    async def _check_concentration_risk(self, signal: TradingSignal,
                                      positions: List[BrokerPosition]) -> Dict[str, Any]:
        """Check concentration risk."""
        # Count positions in same symbol
        symbol_count = sum(1 for pos in positions if pos.symbol == signal.symbol)
        
        if symbol_count >= 3:  # Max 3 positions in same symbol
            return {
                "allowed": False,
                "reason": f"Too many positions in {signal.symbol} (max 3)"
            }
        
        return {"allowed": True}

    async def _check_violations(self, portfolio_value: float, daily_pnl_percent: float,
                              current_drawdown: float, largest_position_percent: float,
                              concentration_risk: float, daily_trades: int) -> List[RiskViolation]:
        """Check for risk violations."""
        violations = []
        
        # Check daily loss limit
        if abs(daily_pnl_percent) > self.config.max_daily_loss_percent * 100:
            violations.append(RiskViolation(
                violation_id=f"daily_loss_{datetime.now().timestamp()}",
                violation_type=RiskViolationType.DAILY_LOSS,
                risk_level=RiskLevel.HIGH,
                message=f"Daily loss {abs(daily_pnl_percent):.2f}% exceeds limit",
                timestamp=datetime.now(),
                current_value=abs(daily_pnl_percent),
                limit_value=self.config.max_daily_loss_percent * 100,
                violation_percent=(abs(daily_pnl_percent) / (self.config.max_daily_loss_percent * 100)) * 100,
                recommended_action="Reduce position sizes or stop trading"
            ))
        
        # Check drawdown limit
        if current_drawdown > self.config.max_drawdown_percent * 100:
            violations.append(RiskViolation(
                violation_id=f"drawdown_{datetime.now().timestamp()}",
                violation_type=RiskViolationType.DRAWDOWN,
                risk_level=RiskLevel.HIGH,
                message=f"Drawdown {current_drawdown:.2f}% exceeds limit",
                timestamp=datetime.now(),
                current_value=current_drawdown,
                limit_value=self.config.max_drawdown_percent * 100,
                violation_percent=(current_drawdown / (self.config.max_drawdown_percent * 100)) * 100,
                recommended_action="Consider reducing positions"
            ))
        
        # Check position size limit
        if largest_position_percent > self.config.max_position_size_percent * 100:
            violations.append(RiskViolation(
                violation_id=f"position_size_{datetime.now().timestamp()}",
                violation_type=RiskViolationType.POSITION_SIZE,
                risk_level=RiskLevel.MEDIUM,
                message=f"Largest position {largest_position_percent:.2f}% exceeds limit",
                timestamp=datetime.now(),
                current_value=largest_position_percent,
                limit_value=self.config.max_position_size_percent * 100,
                violation_percent=(largest_position_percent / (self.config.max_position_size_percent * 100)) * 100,
                recommended_action="Reduce position size"
            ))
        
        # Check concentration limit
        if concentration_risk > self.config.max_concentration_percent * 100:
            violations.append(RiskViolation(
                violation_id=f"concentration_{datetime.now().timestamp()}",
                violation_type=RiskViolationType.CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"Concentration risk {concentration_risk:.2f}% exceeds limit",
                timestamp=datetime.now(),
                current_value=concentration_risk,
                limit_value=self.config.max_concentration_percent * 100,
                violation_percent=(concentration_risk / (self.config.max_concentration_percent * 100)) * 100,
                recommended_action="Diversify positions"
            ))
        
        return violations

    def _calculate_volatility(self, positions: List[BrokerPosition]) -> float:
        """Calculate portfolio volatility (simplified)."""
        if not positions:
            return 0.0
        
        # Simple volatility calculation based on position sizes
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        # Calculate weighted volatility (simplified)
        weighted_volatility = 0.0
        for position in positions:
            position_weight = float(position.market_value) / total_value
            # Assume 2% volatility per position (simplified)
            weighted_volatility += position_weight * 2.0
        
        return weighted_volatility

    def _calculate_risk_score(self, margin_utilization: float, daily_pnl_percent: float,
                            current_drawdown: float, largest_position_percent: float,
                            concentration_risk: float, volatility: float) -> float:
        """Calculate overall risk score (0-100)."""
        risk_factors = [
            margin_utilization / 100,  # Normalize to 0-1
            abs(daily_pnl_percent) / 10,  # Normalize to 0-1
            current_drawdown / 20,  # Normalize to 0-1
            largest_position_percent / 20,  # Normalize to 0-1
            concentration_risk / 50,  # Normalize to 0-1
            volatility / 10  # Normalize to 0-1
        ]
        
        # Weighted average
        weights = [0.2, 0.25, 0.2, 0.15, 0.1, 0.1]
        weighted_score = sum(w * f for w, f in zip(weights, risk_factors))
        
        # Convert to 0-100 scale
        return min(100.0, max(0.0, weighted_score * 100))

    async def _trigger_risk_callbacks(self, metrics: RiskMetrics):
        """Trigger risk metrics callbacks."""
        for callback in self.risk_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                risk_logger.error(f"Error in risk callback: {e}")

    async def _trigger_violation_callbacks(self, violation: RiskViolation):
        """Trigger violation callbacks."""
        for callback in self.violation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(violation)
                else:
                    callback(violation)
            except Exception as e:
                risk_logger.error(f"Error in violation callback: {e}")

    async def _trigger_kill_switch(self, reason: str):
        """Internal kill switch trigger."""
        await self.trigger_kill_switch(reason)
