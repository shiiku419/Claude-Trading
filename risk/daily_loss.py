"""Daily loss tracking with hysteresis (P1-3).

The :class:`DailyLossCheck` tracks realized and unrealized PnL relative to
total portfolio equity.  To avoid overreacting to transient mark-to-market
spikes, unrealized PnL is smoothed with an exponential moving average (EMA)
before being included in the loss calculation.

Hysteresis (P1-3)
-----------------
The kill switch is triggered only after ``consecutive_breaches`` consecutive
evaluations where the total loss exceeds the *critical* threshold.  A single
spike in unrealized PnL will not halt trading.  As soon as the loss drops back
below the critical threshold the consecutive-breach counter resets to zero.
"""

from __future__ import annotations

import structlog

from risk.base import PortfolioState, RiskDecision
from risk.kill_switch import KillSwitch

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

_EMA_ALPHA_5MIN = 0.1818  # ≈ 2 / (10 + 1), tuned for a 5-minute half-life


# ---------------------------------------------------------------------------
# DailyLossCheck
# ---------------------------------------------------------------------------


class DailyLossCheck:
    """Tracks daily PnL and triggers the kill switch on sustained loss breach.

    The check computes::

        total_loss_pct = (realized_pnl + smoothed_unrealized_pnl) / total_balance

    A negative value means a loss.

    Thresholds:

    - ``warning_pct``: Log a WARNING and let the trade proceed.
    - ``critical_pct``: Increment the consecutive-breach counter.  Once the
      counter reaches ``consecutive_breaches`` the kill switch is activated and
      the check rejects the order.
    - Below ``critical_pct``: Reset the consecutive counter to zero.

    Args:
        warning_pct: Loss fraction (e.g. ``0.03`` = 3 %) that triggers a
            WARNING log but does not block trading.
        critical_pct: Loss fraction that, when breached for
            ``consecutive_breaches`` consecutive evaluations, halts trading.
        consecutive_breaches: Number of consecutive critical evaluations
            required before the kill switch fires.
        kill_switch: Optional :class:`~risk.kill_switch.KillSwitch` to
            activate on sustained breach.  When ``None`` the check will reject
            orders but will not persist the halt to Redis.
    """

    def __init__(
        self,
        warning_pct: float = 0.03,
        critical_pct: float = 0.05,
        consecutive_breaches: int = 3,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._warning_pct = warning_pct
        self._critical_pct = critical_pct
        self._consecutive_breaches = consecutive_breaches
        self._kill_switch = kill_switch

        # Mutable state — reset daily via reset_daily()
        self._consecutive_breach_count: int = 0
        self._smoothed_unrealized_pnl: float | None = None  # None until first observation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ema(self, new_value: float) -> float:
        """Update and return the EMA-smoothed unrealized PnL.

        Uses Wilder's EMA (``alpha ≈ 0.18`` for a ~5-minute half-life).  The
        first call seeds the EMA with the raw value so there is no cold-start
        bias.

        Args:
            new_value: Latest raw unrealized PnL observation.

        Returns:
            Updated smoothed value.
        """
        if self._smoothed_unrealized_pnl is None:
            self._smoothed_unrealized_pnl = new_value
        else:
            self._smoothed_unrealized_pnl = (
                _EMA_ALPHA_5MIN * new_value
                + (1.0 - _EMA_ALPHA_5MIN) * self._smoothed_unrealized_pnl
            )
        return self._smoothed_unrealized_pnl

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def evaluate(self, portfolio: PortfolioState) -> RiskDecision:
        """Evaluate daily loss against configured thresholds.

        Args:
            portfolio: Current portfolio snapshot.  Only
                ``total_balance_usd``, ``daily_realized_pnl``, and
                ``daily_unrealized_pnl`` are consumed.

        Returns:
            Approved :class:`~risk.base.RiskDecision` when below critical
            threshold (or below warning), rejected when the kill switch
            threshold is reached.
        """
        smoothed = self._update_ema(portfolio.daily_unrealized_pnl)
        total_loss = portfolio.daily_realized_pnl + smoothed

        # Guard against zero / negative balance to avoid division errors.
        balance = max(portfolio.total_balance_usd, 1e-9)
        total_loss_pct = total_loss / balance  # negative = loss

        logger.debug(
            "daily_loss.evaluate",
            realized=portfolio.daily_realized_pnl,
            smoothed_unrealized=smoothed,
            total_loss_pct=total_loss_pct,
            consecutive_breach_count=self._consecutive_breach_count,
        )

        # ---- Below warning threshold -----------------------------------------
        if total_loss_pct > -self._warning_pct:
            self._consecutive_breach_count = 0
            return RiskDecision(
                approved=True,
                reason=f"daily_loss_ok:{total_loss_pct:.4%}",
                checks_passed=["daily_loss"],
            )

        # ---- Warning zone (between warning and critical) ---------------------
        if total_loss_pct > -self._critical_pct:
            self._consecutive_breach_count = 0
            logger.warning(
                "daily_loss.warning_threshold_breached",
                total_loss_pct=total_loss_pct,
                warning_pct=self._warning_pct,
            )
            return RiskDecision(
                approved=True,
                reason=f"daily_loss_warning:{total_loss_pct:.4%}",
                checks_passed=["daily_loss"],
            )

        # ---- Critical zone ---------------------------------------------------
        self._consecutive_breach_count += 1
        logger.warning(
            "daily_loss.critical_threshold_breached",
            total_loss_pct=total_loss_pct,
            critical_pct=self._critical_pct,
            consecutive_breach_count=self._consecutive_breach_count,
            threshold=self._consecutive_breaches,
        )

        if self._consecutive_breach_count >= self._consecutive_breaches:
            reason = (
                f"daily_loss_critical:{total_loss_pct:.4%}"
                f" consecutive={self._consecutive_breach_count}"
            )
            logger.critical(
                "daily_loss.kill_switch_triggered",
                reason=reason,
                total_loss_pct=total_loss_pct,
            )
            if self._kill_switch is not None:
                await self._kill_switch.activate(reason)
            return RiskDecision(
                approved=False,
                reason=reason,
                checks_failed=["daily_loss"],
            )

        # Consecutive count has not yet reached the threshold — reject this
        # evaluation but do NOT activate the kill switch yet (hysteresis).
        return RiskDecision(
            approved=False,
            reason=(
                f"daily_loss_critical_pending:{total_loss_pct:.4%}"
                f" breach={self._consecutive_breach_count}/{self._consecutive_breaches}"
            ),
            checks_failed=["daily_loss"],
        )

    def reset_daily(self) -> None:
        """Reset all daily-loss state at UTC midnight.

        Clears the consecutive breach counter and the EMA seed so that the
        new trading day starts with a clean slate.
        """
        logger.info(
            "daily_loss.daily_reset",
            previous_breach_count=self._consecutive_breach_count,
        )
        self._consecutive_breach_count = 0
        self._smoothed_unrealized_pnl = None
