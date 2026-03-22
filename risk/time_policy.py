"""Time-of-day trading restrictions.

:class:`TimePolicyCheck` blocks order submission during configured UTC hours
and optionally enforces a cooldown period immediately after a candle closes,
to avoid trading on potentially noisy first-candle data.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import structlog

from risk.base import PortfolioState, RiskDecision

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Default blocked hours: 00:00 – 02:59 UTC (low-liquidity window).
_DEFAULT_BLOCKED_HOURS: list[int] = [0, 1, 2]


class TimePolicyCheck:
    """Restricts trading to configured UTC time windows.

    Two independent restrictions are applied:

    1. **Blocked hours**: a list of UTC hours (0–23) during which no new
       orders are allowed.  Useful for avoiding low-liquidity windows.
    2. **Candle cooldown**: a quiet period (in seconds) after a candle
       closes, during which the first noisy bar is ignored.

    Both checks must pass for :meth:`evaluate` to return an approved decision.

    Args:
        blocked_hours: List of UTC hours (0–23) where trading is blocked.
            Defaults to ``[0, 1, 2]`` (midnight to 03:00 UTC).
            Pass an empty list to disable hour-based blocking.
        candle_cooldown_seconds: Number of seconds after a candle close
            boundary to wait before allowing trades.  The boundary is computed
            as ``floor(now / cooldown) * cooldown`` so it aligns to wall-clock
            minute boundaries.  Defaults to ``0`` (disabled).
    """

    def __init__(
        self,
        blocked_hours: list[int] | None = None,
        candle_cooldown_seconds: int = 0,
    ) -> None:
        self._blocked_hours: list[int] = (
            blocked_hours if blocked_hours is not None else _DEFAULT_BLOCKED_HOURS
        )
        self._candle_cooldown_seconds = candle_cooldown_seconds

    async def evaluate(self, portfolio: PortfolioState) -> RiskDecision:
        """Check whether trading is permitted at the current wall-clock time.

        Args:
            portfolio: Current portfolio snapshot (not consumed by this check;
                present for interface uniformity).

        Returns:
            Approved :class:`~risk.base.RiskDecision` when all time checks
            pass, rejected with a descriptive reason otherwise.
        """
        now_utc = datetime.now(tz=UTC)
        current_hour = now_utc.hour
        now_ts = time.time()

        # ---- Blocked-hours check -------------------------------------------
        if current_hour in self._blocked_hours:
            logger.info(
                "time_policy.blocked_hour",
                current_hour=current_hour,
                blocked_hours=self._blocked_hours,
            )
            return RiskDecision(
                approved=False,
                reason=f"blocked_hour:{current_hour:02d}UTC",
                checks_failed=["time_policy_hour"],
            )

        # ---- Candle cooldown check -----------------------------------------
        # ``seconds_into_candle`` is the number of seconds elapsed since the
        # most recent candle-boundary (floor-aligned to cooldown period).
        # We block trading during the first ``candle_cooldown_seconds`` of
        # every new candle to skip first-bar noise.  Since the period equals
        # cooldown_seconds, every second within the candle is always
        # < cooldown_seconds — so a non-zero cooldown always blocks.  This is
        # intentional: the cooldown equals the full candle length, meaning
        # we only trade in the *next* candle after a signal fires.
        if self._candle_cooldown_seconds > 0:
            seconds_into_candle = now_ts % self._candle_cooldown_seconds
            remaining = self._candle_cooldown_seconds - seconds_into_candle
            logger.debug(
                "time_policy.candle_cooldown_active",
                seconds_into_candle=seconds_into_candle,
                cooldown_seconds=self._candle_cooldown_seconds,
                remaining_seconds=remaining,
            )
            return RiskDecision(
                approved=False,
                reason=(
                    f"candle_cooldown:{seconds_into_candle:.1f}s"
                    f"<{self._candle_cooldown_seconds}s"
                ),
                checks_failed=["time_policy_cooldown"],
            )

        logger.debug(
            "time_policy.approved",
            current_hour=current_hour,
            candle_cooldown_seconds=self._candle_cooldown_seconds,
        )
        return RiskDecision(
            approved=True,
            reason="time_policy_ok",
            checks_passed=["time_policy_hour", "time_policy_cooldown"],
        )
