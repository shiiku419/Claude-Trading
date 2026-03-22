"""Main risk gate orchestrator (P0-2).

:class:`RiskGate` runs a sequential pipeline of risk checks.  Every check must
pass for an order to be approved.  On approval, it produces an
:class:`~risk.base.ApprovedOrder` bearing an HMAC-SHA256 token signed with the
process-local secret.  The Execution Engine must verify this token via
:func:`~risk.base.verify_gate_token` before submitting any order to the
exchange.

Pipeline order
--------------
1. :class:`~risk.kill_switch.KillSwitch` — emergency halt gate (fast path).
2. :class:`~risk.time_policy.TimePolicyCheck` — blocked hours / candle cooldown.
3. :class:`~risk.daily_loss.DailyLossCheck` — daily PnL threshold with hysteresis.
4. :class:`~risk.exposure.ExposureCheck` — total portfolio notional cap.
5. :class:`~risk.position_limit.PositionLimitCheck` — per-trade size + open count.

Any check returning ``approved=False`` short-circuits the pipeline and causes
``evaluate`` to return ``None``.  A size adjustment from step 5 is forwarded to
the :class:`~risk.base.ApprovedOrder` rather than the originally requested
quantity.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog

from risk.base import (
    ApprovedOrder,
    PortfolioState,
    _compute_gate_token,
)
from risk.daily_loss import DailyLossCheck
from risk.exposure import ExposureCheck
from risk.kill_switch import KillSwitch
from risk.position_limit import PositionLimitCheck
from risk.time_policy import TimePolicyCheck

if TYPE_CHECKING:
    from signals.base import Signal

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class RiskGate:
    """Sequential risk check pipeline — the sole producer of :class:`~risk.base.ApprovedOrder`.

    All five checks must pass in the declared order.  A single failure
    short-circuits evaluation and returns ``None``.  Only when every check
    returns ``approved=True`` is an :class:`~risk.base.ApprovedOrder` minted
    with a fresh HMAC token.

    Args:
        kill_switch: Emergency halt check (always evaluated first).
        time_policy: Hour-of-day and candle-cooldown restriction.
        daily_loss: Daily PnL threshold with hysteresis.
        exposure: Total portfolio notional exposure cap.
        position_limit: Per-trade size cap and concurrent-position count limit.
    """

    def __init__(
        self,
        kill_switch: KillSwitch,
        time_policy: TimePolicyCheck,
        daily_loss: DailyLossCheck,
        exposure: ExposureCheck,
        position_limit: PositionLimitCheck,
    ) -> None:
        self._kill_switch = kill_switch
        self._time_policy = time_policy
        self._daily_loss = daily_loss
        self._exposure = exposure
        self._position_limit = position_limit

    async def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        requested_quantity: float,
        current_price: float,
    ) -> ApprovedOrder | None:
        """Run all risk checks and return an approved order token or ``None``.

        The method is the single entry point for the Strategy Controller.
        Callers must treat a ``None`` return as a definitive rejection and
        must NOT submit any order to the exchange.

        Args:
            signal: The originating :class:`~signals.base.Signal` that
                triggered this evaluation.
            portfolio: Current portfolio snapshot provided by the Strategy
                Controller.
            requested_quantity: Asset quantity the strategy wishes to trade.
            current_price: Latest market price used for notional calculations.

        Returns:
            An :class:`~risk.base.ApprovedOrder` with a valid HMAC token when
            all checks pass, or ``None`` when any check rejects the order.
        """
        signal_id = str(uuid.uuid4())
        side = "buy" if signal.direction == "long" else "sell"
        checks_passed: list[str] = []
        effective_quantity = requested_quantity

        log = logger.bind(
            signal_id=signal_id,
            pair=signal.pair,
            direction=str(signal.direction),
            side=side,
            requested_quantity=requested_quantity,
            current_price=current_price,
        )

        # ------------------------------------------------------------------
        # Step 1: Kill switch
        # ------------------------------------------------------------------
        ks_decision = await self._kill_switch.evaluate(portfolio)
        if not ks_decision.approved:
            log.warning(
                "risk_gate.rejected",
                step="kill_switch",
                reason=ks_decision.reason,
            )
            return None
        checks_passed.extend(ks_decision.checks_passed)

        # ------------------------------------------------------------------
        # Step 2: Time policy
        # ------------------------------------------------------------------
        tp_decision = await self._time_policy.evaluate(portfolio)
        if not tp_decision.approved:
            log.info(
                "risk_gate.rejected",
                step="time_policy",
                reason=tp_decision.reason,
            )
            return None
        checks_passed.extend(tp_decision.checks_passed)

        # ------------------------------------------------------------------
        # Step 3: Daily loss
        # ------------------------------------------------------------------
        dl_decision = await self._daily_loss.evaluate(portfolio)
        if not dl_decision.approved:
            log.warning(
                "risk_gate.rejected",
                step="daily_loss",
                reason=dl_decision.reason,
            )
            return None
        checks_passed.extend(dl_decision.checks_passed)

        # ------------------------------------------------------------------
        # Step 4: Exposure
        # ------------------------------------------------------------------
        exp_decision = await self._exposure.evaluate(
            signal, portfolio, effective_quantity, current_price
        )
        if not exp_decision.approved:
            log.warning(
                "risk_gate.rejected",
                step="exposure",
                reason=exp_decision.reason,
            )
            return None
        checks_passed.extend(exp_decision.checks_passed)

        # ------------------------------------------------------------------
        # Step 5: Position limit (may adjust quantity downward)
        # ------------------------------------------------------------------
        pl_decision = await self._position_limit.evaluate(
            signal, portfolio, effective_quantity, current_price
        )
        if not pl_decision.approved:
            log.warning(
                "risk_gate.rejected",
                step="position_limit",
                reason=pl_decision.reason,
            )
            return None
        checks_passed.extend(pl_decision.checks_passed)

        # Accept adjusted quantity if the position-limit check reduced it.
        if pl_decision.adjusted_quantity is not None:
            effective_quantity = pl_decision.adjusted_quantity
            log.info(
                "risk_gate.quantity_adjusted",
                original=requested_quantity,
                adjusted=effective_quantity,
            )

        # ------------------------------------------------------------------
        # Mint ApprovedOrder with HMAC token
        # ------------------------------------------------------------------
        approved_at = time.time()
        gate_token = _compute_gate_token(
            signal_id=signal_id,
            pair=signal.pair,
            side=side,
            quantity=effective_quantity,
            timestamp=approved_at,
        )

        order = ApprovedOrder(
            signal_id=signal_id,
            pair=signal.pair,
            side=side,
            quantity=effective_quantity,
            approved_at=approved_at,
            gate_token=gate_token,
            risk_checks_passed=checks_passed,
            original_signal_strength=signal.strength,
        )

        log.info(
            "risk_gate.approved",
            effective_quantity=effective_quantity,
            checks_passed=checks_passed,
            approved_at=approved_at,
        )
        return order
