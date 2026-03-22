"""Emergency kill switch with Redis-backed persistence.

When active, the kill switch causes :class:`~risk.gate.RiskGate` to reject
ALL new orders immediately, regardless of any other checks.

Design principles
-----------------
- **Fail-closed**: if Redis is unavailable the switch is treated as *active*
  so that a connectivity blip can never accidentally open the gate.
- **Manual reset only**: the switch can be activated programmatically (by
  :class:`~risk.daily_loss.DailyLossCheck` or an operator), but deactivation
  always requires an explicit human-initiated call to :meth:`KillSwitch.deactivate`.
- **Survives restart**: state is persisted in Redis so a process restart does
  not silently reset a live emergency stop.
"""

from __future__ import annotations

import time

import redis.asyncio as aioredis
import structlog

from risk.base import PortfolioState, RiskDecision

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


class KillSwitch:
    """Emergency stop that rejects ALL new orders when active.

    State is persisted across three Redis keys:

    - ``trading:kill_switch`` — ``"1"`` when active, ``"0"`` when inactive.
    - ``trading:kill_switch:reason`` — human-readable trigger description.
    - ``trading:kill_switch:activated_at`` — Unix timestamp string.

    Trigger sources:
    - Daily loss consecutive-breach threshold (via :class:`~risk.daily_loss.DailyLossCheck`).
    - Manual operator call.
    - Exchange connection loss > 5 min (triggered externally by the reconnect monitor).

    Args:
        redis_url: Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
    """

    REDIS_KEY = "trading:kill_switch"
    REDIS_REASON_KEY = "trading:kill_switch:reason"
    REDIS_ACTIVATED_AT_KEY = "trading:kill_switch:activated_at"

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: aioredis.Redis | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open a Redis connection pool.

        Should be called once during application startup before any
        :meth:`evaluate` calls are made.
        """
        self._client = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        logger.info("kill_switch.connected", redis_url=self._redis_url)

    async def close(self) -> None:
        """Close the Redis connection pool gracefully."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("kill_switch.closed")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    async def is_active(self) -> bool:
        """Return ``True`` if the kill switch is currently active.

        Fails closed: any exception (Redis unavailable, timeout, etc.) causes
        this method to return ``True`` so that orders are blocked rather than
        accidentally allowed through.

        Returns:
            ``True`` when the kill switch is active or Redis is unavailable.
        """
        try:
            if self._client is None:
                logger.warning("kill_switch.no_client_fail_closed")
                return True
            value = await self._client.get(self.REDIS_KEY)
            return value == "1"
        except Exception as exc:
            logger.error("kill_switch.redis_error_fail_closed", error=str(exc))
            return True

    async def get_status(self) -> dict[str, str | bool]:
        """Return a diagnostic status dict for monitoring and alerting.

        Returns:
            Dictionary with keys ``active``, ``reason``, and ``activated_at``.
            Values are empty strings when the switch has never been triggered.
        """
        try:
            if self._client is None:
                return {"active": True, "reason": "no_redis_client", "activated_at": ""}
            active_raw = await self._client.get(self.REDIS_KEY)
            reason = await self._client.get(self.REDIS_REASON_KEY) or ""
            activated_at = await self._client.get(self.REDIS_ACTIVATED_AT_KEY) or ""
            return {
                "active": active_raw == "1",
                "reason": reason,
                "activated_at": activated_at,
            }
        except Exception as exc:
            logger.error("kill_switch.get_status_error", error=str(exc))
            return {"active": True, "reason": f"redis_error:{exc}", "activated_at": ""}

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    async def activate(self, reason: str) -> None:
        """Activate the kill switch, blocking all new orders.

        Writes the activation state to Redis in a pipeline so all three keys
        are updated atomically.  Logs a CRITICAL-level event.

        Args:
            reason: Human-readable description of why the switch was triggered.
        """
        try:
            if self._client is None:
                logger.critical(
                    "kill_switch.activated_no_redis",
                    reason=reason,
                    note="state not persisted",
                )
                return
            activated_at = str(time.time())
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.set(self.REDIS_KEY, "1")
                pipe.set(self.REDIS_REASON_KEY, reason)
                pipe.set(self.REDIS_ACTIVATED_AT_KEY, activated_at)
                await pipe.execute()
            logger.critical(
                "kill_switch.activated",
                reason=reason,
                activated_at=activated_at,
            )
        except Exception as exc:
            logger.critical(
                "kill_switch.activate_failed",
                reason=reason,
                error=str(exc),
            )

    async def deactivate(self) -> None:
        """Deactivate the kill switch, re-enabling new orders.

        This must be called manually by an operator.  It cannot be triggered
        automatically.  Logs a WARNING-level event for audit trail.
        """
        try:
            if self._client is None:
                logger.warning("kill_switch.deactivate_no_redis")
                return
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.set(self.REDIS_KEY, "0")
                pipe.delete(self.REDIS_REASON_KEY)
                pipe.delete(self.REDIS_ACTIVATED_AT_KEY)
                await pipe.execute()
            logger.warning("kill_switch.deactivated")
        except Exception as exc:
            logger.error("kill_switch.deactivate_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Risk check interface
    # ------------------------------------------------------------------

    async def evaluate(self, portfolio: PortfolioState) -> RiskDecision:
        """Check if the kill switch is active and return a :class:`~risk.base.RiskDecision`.

        Args:
            portfolio: Current portfolio snapshot (not used in this check, but
                kept for a uniform interface across all risk checks).

        Returns:
            Approved :class:`~risk.base.RiskDecision` if the switch is inactive,
            rejected decision otherwise.
        """
        if await self.is_active():
            status = await self.get_status()
            reason = str(status.get("reason", "unknown"))
            logger.warning("kill_switch.evaluate_rejected", reason=reason)
            return RiskDecision(
                approved=False,
                reason=f"kill_switch_active:{reason}",
                checks_failed=["kill_switch"],
            )

        return RiskDecision(
            approved=True,
            reason="kill_switch_inactive",
            checks_passed=["kill_switch"],
        )
