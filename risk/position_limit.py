"""Per-trade size and concurrent position limits.

:class:`PositionLimitCheck` enforces two constraints:

1. **Per-trade size cap**: the notional value of the new order must not exceed
   ``max_position_pct * total_balance_usd``.  If the requested quantity is too
   large it is *reduced* (adjusted) rather than rejected outright, because a
   smaller trade is still meaningful.

2. **Max concurrent positions**: if the portfolio already holds the maximum
   number of allowed open positions the order is **rejected** (not adjusted),
   because there is no safe way to open yet another position.
"""

from __future__ import annotations

# Signals are imported lazily via TYPE_CHECKING to avoid a circular import
# at runtime. At runtime we only need the pair / direction attributes, which
# are plain strings / StrEnum values available on the dataclass instance.
from typing import TYPE_CHECKING

import structlog

from risk.base import PortfolioState, RiskDecision

if TYPE_CHECKING:
    from signals.base import Signal

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class PositionLimitCheck:
    """Enforces per-trade size and concurrent position limits.

    Args:
        max_position_pct: Maximum single-position size as a fraction of total
            equity (e.g. ``0.02`` = 2 %).
        max_open_positions: Hard cap on the number of simultaneously open
            positions.
    """

    def __init__(
        self,
        max_position_pct: float = 0.02,
        max_open_positions: int = 3,
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_open_positions = max_open_positions

    async def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        requested_quantity: float,
        current_price: float,
    ) -> RiskDecision:
        """Evaluate position-size and open-count constraints.

        Args:
            signal: The originating trading signal.  Only ``signal.pair`` is
                used (to check whether this pair already has an open position).
            portfolio: Current portfolio snapshot.
            requested_quantity: Asset quantity requested by the strategy.
            current_price: Current market price used to convert quantity to a
                notional USD value.

        Returns:
            :class:`~risk.base.RiskDecision` with ``approved=True`` when at
            least one of the size or count checks passes.  If quantity is
            adjusted the ``adjusted_quantity`` field carries the capped value.
            Rejected when the concurrent-position cap is already full and this
            pair is not already open.
        """
        # ------------------------------------------------------------------
        # 1. Concurrent open-position count
        # ------------------------------------------------------------------
        open_count = len(portfolio.open_positions)
        pair_already_open = signal.pair in portfolio.open_positions

        if open_count >= self._max_open_positions and not pair_already_open:
            logger.warning(
                "position_limit.too_many_open_positions",
                open_count=open_count,
                max_open=self._max_open_positions,
                pair=signal.pair,
            )
            return RiskDecision(
                approved=False,
                reason=(
                    f"max_open_positions_reached:{open_count}/{self._max_open_positions}"
                ),
                checks_failed=["position_limit_count"],
            )

        # ------------------------------------------------------------------
        # 2. Per-trade notional size cap
        # ------------------------------------------------------------------
        max_notional = self._max_position_pct * portfolio.total_balance_usd
        requested_notional = requested_quantity * current_price

        if requested_notional <= max_notional:
            logger.debug(
                "position_limit.within_size_limit",
                requested_notional=requested_notional,
                max_notional=max_notional,
                pair=signal.pair,
            )
            return RiskDecision(
                approved=True,
                reason="position_limit_ok",
                checks_passed=["position_limit_count", "position_limit_size"],
            )

        # Quantity exceeds the cap — adjust downward instead of rejecting.
        # Guard against zero price to avoid ZeroDivisionError.
        safe_price = max(current_price, 1e-9)
        adjusted_quantity = max_notional / safe_price

        logger.warning(
            "position_limit.quantity_adjusted",
            requested_quantity=requested_quantity,
            adjusted_quantity=adjusted_quantity,
            requested_notional=requested_notional,
            max_notional=max_notional,
            pair=signal.pair,
        )
        return RiskDecision(
            approved=True,
            reason=(
                f"position_size_adjusted:{requested_notional:.2f}"
                f"->{max_notional:.2f}_usd"
            ),
            adjusted_quantity=adjusted_quantity,
            checks_passed=["position_limit_count"],
            checks_failed=["position_limit_size"],
        )
