"""Total portfolio exposure limit.

:class:`ExposureCheck` ensures that the sum of all open position values, plus
the new order's notional value, does not exceed ``max_exposure_pct`` of total
equity.  Unlike :class:`~risk.position_limit.PositionLimitCheck`, exposure
violations result in a hard rejection rather than an adjustment, because
reducing the new order's size while total exposure is already at the cap would
still leave the portfolio over-exposed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from risk.base import PortfolioState, RiskDecision

if TYPE_CHECKING:
    from signals.base import Signal

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class ExposureCheck:
    """Limits total portfolio exposure (sum of all open position values / equity).

    If ``current_exposure + new_position_notional > max_exposure_pct * balance``
    the order is rejected.

    Args:
        max_exposure_pct: Maximum total notional exposure as a fraction of
            total equity (e.g. ``0.10`` = 10 %).
    """

    def __init__(self, max_exposure_pct: float = 0.10) -> None:
        self._max_exposure_pct = max_exposure_pct

    async def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        requested_quantity: float,
        current_price: float,
    ) -> RiskDecision:
        """Evaluate whether opening the new position would breach the exposure cap.

        Args:
            signal: Originating signal (used only for logging the pair).
            portfolio: Current portfolio snapshot.
            requested_quantity: Asset quantity for the new order.
            current_price: Current market price used to compute new order notional.

        Returns:
            Approved :class:`~risk.base.RiskDecision` if total exposure
            (existing + new) stays within the cap, rejected otherwise.
        """
        current_exposure = sum(portfolio.open_position_values.values())
        new_notional = requested_quantity * current_price
        projected_exposure = current_exposure + new_notional

        balance = max(portfolio.total_balance_usd, 1e-9)
        max_exposure_usd = self._max_exposure_pct * balance
        projected_exposure_pct = projected_exposure / balance

        logger.debug(
            "exposure.evaluate",
            pair=signal.pair,
            current_exposure=current_exposure,
            new_notional=new_notional,
            projected_exposure=projected_exposure,
            projected_exposure_pct=projected_exposure_pct,
            max_exposure_pct=self._max_exposure_pct,
        )

        if projected_exposure <= max_exposure_usd:
            return RiskDecision(
                approved=True,
                reason=f"exposure_ok:{projected_exposure_pct:.4%}",
                checks_passed=["exposure"],
            )

        logger.warning(
            "exposure.limit_breached",
            pair=signal.pair,
            projected_exposure_pct=projected_exposure_pct,
            max_exposure_pct=self._max_exposure_pct,
            current_exposure=current_exposure,
            new_notional=new_notional,
        )
        return RiskDecision(
            approved=False,
            reason=(
                f"exposure_limit_breached:{projected_exposure_pct:.4%}"
                f">{self._max_exposure_pct:.4%}"
            ),
            checks_failed=["exposure"],
        )
