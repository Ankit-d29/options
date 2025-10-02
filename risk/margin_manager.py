"""
Margin management system for the options trading system.

This module provides comprehensive margin monitoring, management,
and risk controls to ensure proper capital allocation and leverage management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from broker.base_broker import BrokerPosition, BrokerOrder
from utils.logging_utils import get_logger

# Setup logging
margin_logger = get_logger(__name__)


class MarginStatus(Enum):
    """Margin status enumeration."""
    ADEQUATE = "ADEQUATE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MARGIN_CALL = "MARGIN_CALL"


class MarginType(Enum):
    """Margin type enumeration."""
    INITIAL = "INITIAL"
    MAINTENANCE = "MAINTENANCE"
    ADDITIONAL = "ADDITIONAL"
    VARIATION = "VARIATION"


@dataclass
class MarginConfig:
    """Configuration for margin management."""
    # Margin requirements
    initial_margin_percent: float = 0.20      # 20% initial margin
    maintenance_margin_percent: float = 0.15  # 15% maintenance margin
    additional_margin_percent: float = 0.25   # 25% additional margin buffer
    
    # Risk limits
    max_margin_utilization_percent: float = 0.80  # 80% max utilization
    warning_margin_percent: float = 0.70          # 70% warning threshold
    critical_margin_percent: float = 0.90         # 90% critical threshold
    
    # Leverage controls
    max_leverage: float = 5.0                     # 5x max leverage
    enable_leverage_limits: bool = True
    
    # Auto-management
    enable_auto_reduce: bool = True               # Auto-reduce positions on margin call
    enable_auto_close: bool = False               # Auto-close positions on margin call
    margin_call_timeout: int = 300                # 5 minutes to meet margin call
    
    # Margin calculation
    use_portfolio_margin: bool = True             # Use portfolio margin
    correlation_factor: float = 0.5               # Correlation reduction factor
    volatility_adjustment: bool = True            # Adjust for volatility


@dataclass
class MarginRequirement:
    """Margin requirement data structure."""
    margin_type: MarginType
    required_amount: float
    current_amount: float
    utilization_percent: float
    is_satisfied: bool
    deficit_amount: float = 0.0
    symbol: Optional[str] = None
    position_id: Optional[str] = None


@dataclass
class MarginStatusResult:
    """Margin status data structure."""
    timestamp: datetime
    total_margin_required: float
    total_margin_available: float
    margin_utilization_percent: float
    margin_status: MarginStatus
    excess_margin: float
    deficit_margin: float
    
    # Position-level margins
    position_margins: List[MarginRequirement] = field(default_factory=list)
    
    # Risk metrics
    leverage_ratio: float = 0.0
    portfolio_risk: float = 0.0
    concentration_risk: float = 0.0
    
    # Alerts and warnings
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MarginManager:
    """
    Margin management system.
    
    This class provides comprehensive margin monitoring, calculation,
    and management to ensure proper capital allocation and risk control.
    """

    def __init__(self, config: MarginConfig):
        """
        Initialize margin manager.
        
        Args:
            config: Margin management configuration
        """
        self.config = config
        self.margin_history: List[MarginStatus] = []
        self.margin_calls: List[Dict[str, Any]] = []
        self.auto_reduce_triggered = False
        
        margin_logger.info(f"Initialized MarginManager with config: {config}")

    def calculate_margin_requirements(self, positions: List[BrokerPosition],
                                    available_capital: float,
                                    portfolio_value: float) -> MarginStatusResult:
        """
        Calculate margin requirements for current positions.
        
        Args:
            positions: Current positions
            available_capital: Available capital
            portfolio_value: Current portfolio value
            
        Returns:
            Margin status
        """
        try:
            # Calculate position-level margins
            position_margins = []
            total_margin_required = 0.0
            
            for position in positions:
                if position.is_flat:
                    continue
                
                margin_req = self._calculate_position_margin(position, portfolio_value)
                position_margins.append(margin_req)
                total_margin_required += margin_req.required_amount
            
            # Calculate portfolio-level adjustments
            if self.config.use_portfolio_margin:
                portfolio_adjustment = self._calculate_portfolio_adjustment(
                    positions, total_margin_required
                )
                total_margin_required *= portfolio_adjustment
            
            # Calculate leverage ratio
            leverage_ratio = portfolio_value / available_capital if available_capital > 0 else 0
            
            # Determine margin status
            margin_utilization = (total_margin_required / available_capital * 100) if available_capital > 0 else 0
            margin_status = self._determine_margin_status(margin_utilization)
            
            # Calculate excess/deficit
            excess_margin = max(0, available_capital - total_margin_required)
            deficit_margin = max(0, total_margin_required - available_capital)
            
            # Generate warnings and recommendations
            warnings, recommendations = self._generate_warnings_and_recommendations(
                margin_utilization, leverage_ratio, deficit_margin
            )
            
            # Create margin status
            margin_status_obj = MarginStatusResult(
                timestamp=datetime.now(),
                total_margin_required=total_margin_required,
                total_margin_available=available_capital,
                margin_utilization_percent=margin_utilization,
                margin_status=margin_status,
                excess_margin=excess_margin,
                deficit_margin=deficit_margin,
                position_margins=position_margins,
                leverage_ratio=leverage_ratio,
                portfolio_risk=self._calculate_portfolio_risk(positions),
                concentration_risk=self._calculate_concentration_risk(positions, portfolio_value),
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Add to history
            self.margin_history.append(margin_status_obj)
            
            # Check for margin calls
            if margin_status == MarginStatus.MARGIN_CALL:
                # Note: In a real implementation, this would need to be handled asynchronously
                # For now, we'll just log the margin call
                logger.warning(f"Margin call detected: {margin_status_obj.total_margin_required} required, {margin_status_obj.total_margin_available} available")
            
            return margin_status_obj
            
        except Exception as e:
            margin_logger.error(f"Error calculating margin requirements: {e}")
            raise

    def check_order_margin(self, order: BrokerOrder, available_capital: float,
                          current_positions: List[BrokerPosition]) -> Dict[str, Any]:
        """
        Check margin requirements for a new order.
        
        Args:
            order: Order to check
            available_capital: Available capital
            current_positions: Current positions
            
        Returns:
            Margin check result
        """
        try:
            # Calculate additional margin required for this order
            order_margin = self._calculate_order_margin(order)
            
            # Calculate current margin utilization
            current_margin = sum(
                self._calculate_position_margin(pos, 0).required_amount
                for pos in current_positions if not pos.is_flat
            )
            
            # Calculate total margin after order
            total_margin_after = current_margin + order_margin
            total_margin_utilization = (total_margin_after / available_capital * 100) if available_capital > 0 else 0
            
            # Check if order is allowed
            allowed = total_margin_utilization <= self.config.max_margin_utilization_percent * 100
            
            result = {
                "allowed": allowed,
                "order_margin": order_margin,
                "current_margin": current_margin,
                "total_margin_after": total_margin_after,
                "margin_utilization_after": total_margin_utilization,
                "excess_margin": max(0, available_capital - total_margin_after),
                "deficit_margin": max(0, total_margin_after - available_capital)
            }
            
            if not allowed:
                result["reason"] = f"Margin utilization {total_margin_utilization:.2f}% exceeds limit {self.config.max_margin_utilization_percent * 100:.2f}%"
            else:
                result["reason"] = "Margin check passed"
            
            return result
            
        except Exception as e:
            margin_logger.error(f"Error checking order margin: {e}")
            return {
                "allowed": False,
                "reason": f"Margin check error: {e}",
                "order_margin": 0.0,
                "current_margin": 0.0,
                "total_margin_after": 0.0,
                "margin_utilization_after": 0.0
            }

    def get_margin_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get margin summary for specified hours.
        
        Args:
            hours: Hours to include in summary
            
        Returns:
            Margin summary
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [m for m in self.margin_history if m.timestamp >= cutoff_time]
            
            if not recent_history:
                return {"error": "No margin data available"}
            
            latest = recent_history[-1]
            
            # Calculate statistics
            avg_utilization = sum(m.margin_utilization_percent for m in recent_history) / len(recent_history)
            max_utilization = max(m.margin_utilization_percent for m in recent_history)
            min_utilization = min(m.margin_utilization_percent for m in recent_history)
            
            margin_calls_count = len([m for m in recent_history if m.margin_status == MarginStatus.MARGIN_CALL])
            
            return {
                "current_status": latest.margin_status.value,
                "current_utilization": latest.margin_utilization_percent,
                "average_utilization": avg_utilization,
                "max_utilization": max_utilization,
                "min_utilization": min_utilization,
                "excess_margin": latest.excess_margin,
                "deficit_margin": latest.deficit_margin,
                "leverage_ratio": latest.leverage_ratio,
                "margin_calls": margin_calls_count,
                "warnings": latest.warnings,
                "recommendations": latest.recommendations,
                "data_points": len(recent_history)
            }
            
        except Exception as e:
            margin_logger.error(f"Error getting margin summary: {e}")
            return {"error": str(e)}

    def _calculate_position_margin(self, position: BrokerPosition,
                                 portfolio_value: float) -> MarginRequirement:
        """Calculate margin requirement for a position."""
        try:
            # Base margin calculation
            position_value = float(position.quantity * position.last_price)
            
            # Initial margin requirement
            initial_margin = position_value * self.config.initial_margin_percent
            
            # Maintenance margin requirement
            maintenance_margin = position_value * self.config.maintenance_margin_percent
            
            # Use the higher of initial or maintenance
            required_margin = max(initial_margin, maintenance_margin)
            
            # Volatility adjustment
            if self.config.volatility_adjustment:
                volatility_factor = 1.0 + (abs(position.day_change_percent) / 100)
                required_margin *= volatility_factor
            
            # Determine if satisfied (simplified - assume adequate capital)
            is_satisfied = True  # This would be determined by actual capital check
            
            return MarginRequirement(
                margin_type=MarginType.INITIAL,
                required_amount=required_margin,
                current_amount=required_margin,  # Simplified
                utilization_percent=100.0,
                is_satisfied=is_satisfied,
                symbol=position.symbol
            )
            
        except Exception as e:
            margin_logger.error(f"Error calculating position margin: {e}")
            return MarginRequirement(
                margin_type=MarginType.INITIAL,
                required_amount=0.0,
                current_amount=0.0,
                utilization_percent=0.0,
                is_satisfied=True
            )

    def _calculate_order_margin(self, order: BrokerOrder) -> float:
        """Calculate margin requirement for an order."""
        try:
            # Calculate order value
            order_value = float(order.price * order.quantity) if order.price else 0.0
            
            # Calculate margin requirement
            margin_requirement = order_value * self.config.initial_margin_percent
            
            return margin_requirement
            
        except Exception as e:
            margin_logger.error(f"Error calculating order margin: {e}")
            return 0.0

    def _calculate_portfolio_adjustment(self, positions: List[BrokerPosition],
                                      total_margin: float) -> float:
        """Calculate portfolio margin adjustment factor."""
        try:
            if not positions:
                return 1.0
            
            # Simple portfolio adjustment based on diversification
            num_positions = len([p for p in positions if not p.is_flat])
            
            if num_positions <= 1:
                return 1.0
            elif num_positions <= 3:
                return 0.9  # 10% reduction
            elif num_positions <= 5:
                return 0.8  # 20% reduction
            else:
                return 0.7  # 30% reduction
                
        except Exception as e:
            margin_logger.error(f"Error calculating portfolio adjustment: {e}")
            return 1.0

    def _determine_margin_status(self, utilization_percent: float) -> MarginStatus:
        """Determine margin status based on utilization."""
        if utilization_percent >= self.config.critical_margin_percent * 100:
            return MarginStatus.MARGIN_CALL
        elif utilization_percent >= self.config.warning_margin_percent * 100:
            return MarginStatus.WARNING
        elif utilization_percent >= self.config.max_margin_utilization_percent * 100:
            return MarginStatus.CRITICAL
        else:
            return MarginStatus.ADEQUATE

    def _calculate_portfolio_risk(self, positions: List[BrokerPosition]) -> float:
        """Calculate portfolio risk metric."""
        try:
            if not positions:
                return 0.0
            
            # Simple risk calculation based on position sizes and volatility
            total_risk = 0.0
            for position in positions:
                if position.is_flat:
                    continue
                
                position_risk = abs(float(position.day_change_percent)) / 100
                total_risk += position_risk
            
            return total_risk / len(positions) if positions else 0.0
            
        except Exception as e:
            margin_logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0

    def _calculate_concentration_risk(self, positions: List[BrokerPosition],
                                   portfolio_value: float) -> float:
        """Calculate concentration risk."""
        try:
            if not positions or portfolio_value == 0:
                return 0.0
            
            # Calculate largest position percentage
            largest_position = max(
                (float(pos.market_value) for pos in positions if not pos.is_flat),
                default=0.0
            )
            
            return (largest_position / portfolio_value) * 100
            
        except Exception as e:
            margin_logger.error(f"Error calculating concentration risk: {e}")
            return 0.0

    def _generate_warnings_and_recommendations(self, utilization_percent: float,
                                             leverage_ratio: float,
                                             deficit_margin: float) -> tuple:
        """Generate warnings and recommendations."""
        warnings = []
        recommendations = []
        
        # Utilization warnings
        if utilization_percent >= self.config.warning_margin_percent * 100:
            warnings.append(f"Margin utilization {utilization_percent:.2f}% is high")
            recommendations.append("Consider reducing position sizes")
        
        if utilization_percent >= self.config.critical_margin_percent * 100:
            warnings.append(f"Margin utilization {utilization_percent:.2f}% is critical")
            recommendations.append("Immediately reduce positions or add capital")
        
        # Leverage warnings
        if leverage_ratio > self.config.max_leverage:
            warnings.append(f"Leverage ratio {leverage_ratio:.2f}x exceeds limit")
            recommendations.append("Reduce leverage by closing positions")
        
        # Deficit warnings
        if deficit_margin > 0:
            warnings.append(f"Margin deficit of ${deficit_margin:,.2f}")
            recommendations.append("Add capital or close positions immediately")
        
        return warnings, recommendations

    async def _handle_margin_call(self, margin_status: MarginStatus):
        """Handle margin call situation."""
        try:
            margin_call = {
                "timestamp": datetime.now(),
                "deficit_amount": margin_status.deficit_margin,
                "utilization_percent": margin_status.margin_utilization_percent,
                "status": "ACTIVE",
                "actions_taken": []
            }
            
            self.margin_calls.append(margin_call)
            
            margin_logger.critical(
                f"MARGIN CALL: Deficit ${margin_status.deficit_margin:,.2f}, "
                f"Utilization {margin_status.margin_utilization_percent:.2f}%"
            )
            
            # Auto-reduce positions if enabled
            if self.config.enable_auto_reduce and not self.auto_reduce_triggered:
                await self._auto_reduce_positions(margin_status)
                margin_call["actions_taken"].append("auto_reduce_triggered")
                self.auto_reduce_triggered = True
            
            # Auto-close positions if enabled
            if self.config.enable_auto_close:
                await self._auto_close_positions(margin_status)
                margin_call["actions_taken"].append("auto_close_triggered")
            
        except Exception as e:
            margin_logger.error(f"Error handling margin call: {e}")

    async def _auto_reduce_positions(self, margin_status: MarginStatus):
        """Auto-reduce positions to meet margin requirements."""
        try:
            # This would integrate with the broker to reduce positions
            margin_logger.warning("Auto-reduce positions triggered")
            
        except Exception as e:
            margin_logger.error(f"Error in auto-reduce positions: {e}")

    async def _auto_close_positions(self, margin_status: MarginStatus):
        """Auto-close positions to meet margin requirements."""
        try:
            # This would integrate with the broker to close positions
            margin_logger.warning("Auto-close positions triggered")
            
        except Exception as e:
            margin_logger.error(f"Error in auto-close positions: {e}")

    def reset_auto_reduce_flag(self):
        """Reset auto-reduce flag."""
        self.auto_reduce_triggered = False

    def get_margin_calls_history(self) -> List[Dict[str, Any]]:
        """Get margin calls history."""
        return self.margin_calls.copy()
