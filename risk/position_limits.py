"""
Position limits and controls for the options trading system.

This module provides dynamic position sizing, limits, and controls
to manage individual position risk and portfolio concentration.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from broker.base_broker import BrokerPosition
from strategies.base_strategy import TradingSignal
from utils.logging_utils import get_logger

# Setup logging
position_limits_logger = get_logger(__name__)


class LimitType(Enum):
    """Limit type enumeration."""
    POSITION_SIZE = "POSITION_SIZE"
    CONCENTRATION = "CONCENTRATION"
    SYMBOL_COUNT = "SYMBOL_COUNT"
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"
    VOLATILITY = "VOLATILITY"


class LimitViolation(Enum):
    """Limit violation enumeration."""
    WARNING = "WARNING"
    BLOCK = "BLOCK"
    REDUCE = "REDUCE"


@dataclass
class PositionLimitConfig:
    """Configuration for position limits."""
    # Position size limits
    max_position_size_percent: float = 0.10   # 10% of portfolio
    max_position_value: float = 50000.0       # $50k max position value
    max_position_quantity: int = 1000         # Max quantity per position
    
    # Concentration limits
    max_concentration_percent: float = 0.25   # 25% in top position
    max_top_3_concentration_percent: float = 0.50  # 50% in top 3 positions
    max_sector_concentration_percent: float = 0.40  # 40% per sector
    
    # Symbol limits
    max_positions_per_symbol: int = 3         # Max 3 positions per symbol
    max_total_positions: int = 10             # Max 10 total positions
    max_new_positions_per_day: int = 5        # Max 5 new positions per day
    
    # Risk-based limits
    max_position_risk_percent: float = 0.05   # 5% max risk per position
    max_correlation_exposure: float = 0.30    # 30% max correlated exposure
    enable_volatility_adjustment: bool = True
    volatility_multiplier: float = 1.5        # Adjust size based on volatility
    
    # Dynamic sizing
    enable_kelly_criterion: bool = False      # Use Kelly criterion for sizing
    kelly_fraction: float = 0.25             # Fraction of Kelly
    min_position_size: float = 1000.0        # Minimum position size
    max_position_size: float = 50000.0       # Maximum position size


@dataclass
class PositionLimit:
    """Position limit data structure."""
    limit_type: LimitType
    current_value: float
    limit_value: float
    utilization_percent: float
    is_violated: bool
    violation_level: Optional[LimitViolation] = None
    recommendation: str = ""
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSizingResult:
    """Position sizing calculation result."""
    recommended_quantity: int
    recommended_value: float
    sizing_method: str
    risk_amount: float
    confidence: float
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionLimits:
    """
    Position limits and controls manager.
    
    This class manages position sizing, limits, and controls to ensure
    proper risk management at the position level.
    """

    def __init__(self, config: PositionLimitConfig):
        """
        Initialize position limits manager.
        
        Args:
            config: Position limits configuration
        """
        self.config = config
        self.daily_new_positions = 0
        self.daily_reset_time = datetime.now().date()
        
        # Position tracking
        self.position_history: List[BrokerPosition] = []
        self.symbol_counts: Dict[str, int] = {}
        self.sector_exposures: Dict[str, float] = {}
        
        position_limits_logger.info(f"Initialized PositionLimits with config: {config}")

    def calculate_position_size(self, signal: TradingSignal, 
                              portfolio_value: float,
                              available_cash: float,
                              positions: List[BrokerPosition],
                              volatility: float = 0.02) -> PositionSizingResult:
        """
        Calculate optimal position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            available_cash: Available cash
            positions: Current positions
            volatility: Asset volatility
            
        Returns:
            Position sizing result
        """
        try:
            # Reset daily counters if needed
            self._reset_daily_counters()
            
            # Check basic limits first
            basic_checks = self._check_basic_limits(signal, portfolio_value, positions)
            if not basic_checks["allowed"]:
                return PositionSizingResult(
                    recommended_quantity=0,
                    recommended_value=0.0,
                    sizing_method="limit_blocked",
                    risk_amount=0.0,
                    confidence=0.0,
                    warnings=[basic_checks["reason"]]
                )
            
            # Calculate position size using different methods
            sizing_methods = []
            
            # Fixed percentage method
            fixed_percent_size = self._calculate_fixed_percentage_size(
                signal, portfolio_value
            )
            sizing_methods.append(fixed_percent_size)
            
            # Risk-based sizing
            risk_based_size = self._calculate_risk_based_size(
                signal, portfolio_value, volatility
            )
            sizing_methods.append(risk_based_size)
            
            # Kelly criterion (if enabled)
            if self.config.enable_kelly_criterion:
                kelly_size = self._calculate_kelly_size(signal, portfolio_value)
                sizing_methods.append(kelly_size)
            
            # Choose the most conservative size
            recommended_size = min(sizing_methods, key=lambda x: x["value"])
            
            # Apply volatility adjustment
            if self.config.enable_volatility_adjustment:
                volatility_adjustment = min(1.0, 1.0 / (volatility * self.config.volatility_multiplier))
                recommended_size["value"] *= volatility_adjustment
                recommended_size["quantity"] = int(recommended_size["value"] / signal.price)
            
            # Apply final limits
            recommended_size = self._apply_final_limits(
                recommended_size, signal, available_cash
            )
            
            return PositionSizingResult(
                recommended_quantity=recommended_size["quantity"],
                recommended_value=recommended_size["value"],
                sizing_method=recommended_size["method"],
                risk_amount=recommended_size["risk_amount"],
                confidence=recommended_size["confidence"],
                warnings=recommended_size["warnings"]
            )
            
        except Exception as e:
            position_limits_logger.error(f"Error calculating position size: {e}")
            return PositionSizingResult(
                recommended_quantity=0,
                recommended_value=0.0,
                sizing_method="error",
                risk_amount=0.0,
                confidence=0.0,
                warnings=[f"Error: {e}"]
            )

    def check_position_limits(self, signal: TradingSignal,
                            portfolio_value: float,
                            positions: List[BrokerPosition]) -> List[PositionLimit]:
        """
        Check all position limits for a signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            positions: Current positions
            
        Returns:
            List of position limits
        """
        try:
            limits = []
            
            # Check position size limit
            position_size_limit = self._check_position_size_limit(
                signal, portfolio_value
            )
            limits.append(position_size_limit)
            
            # Check concentration limit
            concentration_limit = self._check_concentration_limit(
                signal, portfolio_value, positions
            )
            limits.append(concentration_limit)
            
            # Check symbol count limit
            symbol_count_limit = self._check_symbol_count_limit(signal, positions)
            limits.append(symbol_count_limit)
            
            # Check total positions limit
            total_positions_limit = self._check_total_positions_limit(positions)
            limits.append(total_positions_limit)
            
            # Check daily new positions limit
            daily_positions_limit = self._check_daily_positions_limit()
            limits.append(daily_positions_limit)
            
            return limits
            
        except Exception as e:
            position_limits_logger.error(f"Error checking position limits: {e}")
            return []

    def update_position_tracking(self, positions: List[BrokerPosition]):
        """Update position tracking data."""
        try:
            self.position_history = positions.copy()
            
            # Update symbol counts
            self.symbol_counts = {}
            for position in positions:
                if position.symbol not in self.symbol_counts:
                    self.symbol_counts[position.symbol] = 0
                if not position.is_flat:
                    self.symbol_counts[position.symbol] += 1
            
            # Update sector exposures (simplified)
            self.sector_exposures = {}
            for position in positions:
                # Simplified sector mapping
                sector = self._get_sector(position.symbol)
                if sector not in self.sector_exposures:
                    self.sector_exposures[sector] = 0.0
                if not position.is_flat:
                    self.sector_exposures[sector] += float(position.market_value)
            
        except Exception as e:
            position_limits_logger.error(f"Error updating position tracking: {e}")

    def increment_daily_positions(self):
        """Increment daily position counter."""
        self._reset_daily_counters()
        self.daily_new_positions += 1

    def get_limits_summary(self) -> Dict[str, Any]:
        """Get position limits summary."""
        return {
            "daily_new_positions": self.daily_new_positions,
            "symbol_counts": self.symbol_counts,
            "sector_exposures": self.sector_exposures,
            "total_positions": len([p for p in self.position_history if not p.is_flat]),
            "config": self.config.__dict__
        }

    def _reset_daily_counters(self):
        """Reset daily counters if needed."""
        today = datetime.now().date()
        if today > self.daily_reset_time:
            self.daily_new_positions = 0
            self.daily_reset_time = today

    def _check_basic_limits(self, signal: TradingSignal, 
                          portfolio_value: float,
                          positions: List[BrokerPosition]) -> Dict[str, Any]:
        """Check basic limits before sizing calculation."""
        # Check total positions limit
        if len([p for p in positions if not p.is_flat]) >= self.config.max_total_positions:
            return {
                "allowed": False,
                "reason": f"Maximum total positions {self.config.max_total_positions} reached"
            }
        
        # Check daily new positions limit
        if self.daily_new_positions >= self.config.max_new_positions_per_day:
            return {
                "allowed": False,
                "reason": f"Daily new positions limit {self.config.max_new_positions_per_day} reached"
            }
        
        # Check symbol positions limit
        symbol_count = self.symbol_counts.get(signal.symbol, 0)
        if symbol_count >= self.config.max_positions_per_symbol:
            return {
                "allowed": False,
                "reason": f"Maximum positions per symbol {self.config.max_positions_per_symbol} reached for {signal.symbol}"
            }
        
        return {"allowed": True}

    def _calculate_fixed_percentage_size(self, signal: TradingSignal,
                                       portfolio_value: float) -> Dict[str, Any]:
        """Calculate position size using fixed percentage."""
        position_value = portfolio_value * self.config.max_position_size_percent
        quantity = int(position_value / signal.price)
        
        return {
            "method": "fixed_percentage",
            "value": position_value,
            "quantity": quantity,
            "risk_amount": position_value,
            "confidence": 0.8,
            "warnings": []
        }

    def _calculate_risk_based_size(self, signal: TradingSignal,
                                 portfolio_value: float,
                                 volatility: float) -> Dict[str, Any]:
        """Calculate position size based on risk."""
        risk_amount = portfolio_value * self.config.max_position_risk_percent
        position_value = risk_amount / volatility  # Simplified risk-based sizing
        quantity = int(position_value / signal.price)
        
        return {
            "method": "risk_based",
            "value": position_value,
            "quantity": quantity,
            "risk_amount": risk_amount,
            "confidence": 0.9,
            "warnings": []
        }

    def _calculate_kelly_size(self, signal: TradingSignal,
                            portfolio_value: float) -> Dict[str, Any]:
        """Calculate position size using Kelly criterion."""
        # Simplified Kelly calculation
        win_rate = 0.6  # Assume 60% win rate
        avg_win = 0.05  # Assume 5% average win
        avg_loss = 0.03  # Assume 3% average loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        position_value = portfolio_value * kelly_fraction * self.config.kelly_fraction
        quantity = int(position_value / signal.price)
        
        return {
            "method": "kelly_criterion",
            "value": position_value,
            "quantity": quantity,
            "risk_amount": position_value,
            "confidence": 0.7,
            "warnings": ["Kelly criterion sizing is experimental"]
        }

    def _apply_final_limits(self, sizing: Dict[str, Any], signal: TradingSignal,
                          available_cash: float) -> Dict[str, Any]:
        """Apply final limits to position size."""
        # Ensure minimum size
        if sizing["value"] < self.config.min_position_size:
            sizing["value"] = self.config.min_position_size
            sizing["quantity"] = int(sizing["value"] / signal.price)
            sizing["warnings"].append("Position size increased to minimum")
        
        # Ensure maximum size
        if sizing["value"] > self.config.max_position_size:
            sizing["value"] = self.config.max_position_size
            sizing["quantity"] = int(sizing["value"] / signal.price)
            sizing["warnings"].append("Position size reduced to maximum")
        
        # Check available cash
        required_cash = sizing["value"]
        if required_cash > available_cash:
            sizing["value"] = available_cash * 0.95  # Leave 5% buffer
            sizing["quantity"] = int(sizing["value"] / signal.price)
            sizing["warnings"].append("Position size reduced due to cash constraints")
        
        # Ensure minimum quantity
        if sizing["quantity"] < 1:
            sizing["quantity"] = 0
            sizing["value"] = 0.0
            sizing["warnings"].append("Position size too small, recommended quantity is 0")
        
        return sizing

    def _check_position_size_limit(self, signal: TradingSignal,
                                 portfolio_value: float) -> PositionLimit:
        """Check position size limit."""
        position_value = signal.price * 100  # Assume 100 quantity
        position_percent = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        is_violated = position_percent > self.config.max_position_size_percent * 100
        violation_level = LimitViolation.BLOCK if is_violated else None
        
        return PositionLimit(
            limit_type=LimitType.POSITION_SIZE,
            current_value=position_percent,
            limit_value=self.config.max_position_size_percent * 100,
            utilization_percent=(position_percent / (self.config.max_position_size_percent * 100)) * 100,
            is_violated=is_violated,
            violation_level=violation_level,
            recommendation="Reduce position size" if is_violated else "Position size within limits",
            symbol=signal.symbol
        )

    def _check_concentration_limit(self, signal: TradingSignal,
                                 portfolio_value: float,
                                 positions: List[BrokerPosition]) -> PositionLimit:
        """Check concentration limit."""
        # Calculate current concentration
        position_values = [float(pos.market_value) for pos in positions if not pos.is_flat]
        position_values.sort(reverse=True)
        
        largest_position_percent = 0.0
        if position_values:
            largest_position_percent = (position_values[0] / portfolio_value * 100) if portfolio_value > 0 else 0
        
        is_violated = largest_position_percent > self.config.max_concentration_percent * 100
        violation_level = LimitViolation.WARNING if is_violated else None
        
        return PositionLimit(
            limit_type=LimitType.CONCENTRATION,
            current_value=largest_position_percent,
            limit_value=self.config.max_concentration_percent * 100,
            utilization_percent=(largest_position_percent / (self.config.max_concentration_percent * 100)) * 100,
            is_violated=is_violated,
            violation_level=violation_level,
            recommendation="Diversify positions" if is_violated else "Concentration within limits"
        )

    def _check_symbol_count_limit(self, signal: TradingSignal,
                                positions: List[BrokerPosition]) -> PositionLimit:
        """Check symbol count limit."""
        symbol_count = self.symbol_counts.get(signal.symbol, 0)
        
        is_violated = symbol_count >= self.config.max_positions_per_symbol
        violation_level = LimitViolation.BLOCK if is_violated else None
        
        return PositionLimit(
            limit_type=LimitType.SYMBOL_COUNT,
            current_value=symbol_count,
            limit_value=self.config.max_positions_per_symbol,
            utilization_percent=(symbol_count / self.config.max_positions_per_symbol) * 100,
            is_violated=is_violated,
            violation_level=violation_level,
            recommendation="Reduce positions in this symbol" if is_violated else "Symbol count within limits",
            symbol=signal.symbol
        )

    def _check_total_positions_limit(self, positions: List[BrokerPosition]) -> PositionLimit:
        """Check total positions limit."""
        total_positions = len([p for p in positions if not p.is_flat])
        
        is_violated = total_positions >= self.config.max_total_positions
        violation_level = LimitViolation.BLOCK if is_violated else None
        
        return PositionLimit(
            limit_type=LimitType.SYMBOL_COUNT,
            current_value=total_positions,
            limit_value=self.config.max_total_positions,
            utilization_percent=(total_positions / self.config.max_total_positions) * 100,
            is_violated=is_violated,
            violation_level=violation_level,
            recommendation="Close some positions" if is_violated else "Total positions within limits"
        )

    def _check_daily_positions_limit(self) -> PositionLimit:
        """Check daily new positions limit."""
        is_violated = self.daily_new_positions >= self.config.max_new_positions_per_day
        violation_level = LimitViolation.BLOCK if is_violated else None
        
        return PositionLimit(
            limit_type=LimitType.SYMBOL_COUNT,
            current_value=self.daily_new_positions,
            limit_value=self.config.max_new_positions_per_day,
            utilization_percent=(self.daily_new_positions / self.config.max_new_positions_per_day) * 100,
            is_violated=is_violated,
            violation_level=violation_level,
            recommendation="Wait until tomorrow" if is_violated else "Daily positions within limits"
        )

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified)."""
        # Simplified sector mapping
        if "NIFTY" in symbol:
            return "INDEX"
        elif "BANK" in symbol:
            return "BANKING"
        else:
            return "OTHER"
