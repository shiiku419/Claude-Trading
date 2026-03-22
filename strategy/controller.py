"""Strategy controller – bridge between Claude's regime and the signal engine.

[P0-1] Regime TTL contract
--------------------------
Every stored regime carries a creation timestamp.  On each call to
:meth:`StrategyController.get_current_mode` the controller verifies that::

    updated_at + regime_ttl_hours > utcnow()

If the inequality is violated the controller **immediately** returns the
``unknown`` mode and emits a ``CRITICAL`` log event.  This prevents the bot
from trading on stale intelligence, regardless of how long Claude has been
unavailable.

Redis state
-----------
Three keys share the ``trading:regime:`` namespace:

* ``trading:regime:current``    – regime name string (e.g. ``"trending_up"``)
* ``trading:regime:updated_at`` – ISO-8601 UTC timestamp of last update
* ``trading:regime:confidence`` – float string in ``[0, 1]``

All three are written atomically via a Redis pipeline.  If Redis becomes
unavailable the controller fails safe to ``unknown`` mode and logs a warning.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Final

import structlog
from redis import asyncio as aioredis
from redis.exceptions import RedisError

from strategy.modes import MODES, Regime, TradingMode

log: structlog.BoundLogger = structlog.get_logger(__name__)

_UNKNOWN_MODE: Final[TradingMode] = MODES[Regime.UNKNOWN]


class StrategyController:
    """Maps Claude's regime judgment to machine-readable trading modes.

    [P0-1] Regime TTL: Every regime has a hard TTL.  If expired, force
    ``unknown`` mode regardless of what is stored in Redis.

    State persisted in Redis:

    * ``trading:regime:current``    – regime name
    * ``trading:regime:updated_at`` – ISO-8601 UTC timestamp
    * ``trading:regime:confidence`` – float in ``[0, 1]``

    Args:
        redis_url: Redis connection URL, e.g. ``"redis://localhost:6379/0"``.
        regime_ttl_hours: Maximum age (in hours) of a valid regime.
            Defaults to 12 hours per the P0-1 requirement.
    """

    REDIS_KEY: Final[str] = "trading:regime:current"
    REDIS_UPDATED_AT_KEY: Final[str] = "trading:regime:updated_at"
    REDIS_CONFIDENCE_KEY: Final[str] = "trading:regime:confidence"

    def __init__(
        self,
        redis_url: str,
        regime_ttl_hours: float = 12.0,
    ) -> None:
        self._redis_url: str = redis_url
        self._regime_ttl_hours: float = regime_ttl_hours
        self._redis: aioredis.Redis | None = None
        self._connect_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection pool.

        Safe to call multiple times; subsequent calls are no-ops when the
        connection is already open.
        """
        async with self._connect_lock:
            if self._redis is None:
                self._redis = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                log.info("strategy_controller.connected", redis_url=self._redis_url)

    async def close(self) -> None:
        """Close the Redis connection and release resources."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            log.info("strategy_controller.closed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_current_mode(self) -> TradingMode:
        """Return the active :class:`~strategy.modes.TradingMode`.

        [P0-1] Contract: ``updated_at + TTL > utcnow()`` must hold.
        If it does not, this method returns ``unknown`` mode and logs
        at ``CRITICAL`` level.  If Redis is unavailable it also returns
        ``unknown`` and logs at ``WARNING`` level.

        Returns:
            The current :class:`~strategy.modes.TradingMode`, or the
            ``unknown`` mode when the regime is absent, expired, or
            unreadable.
        """
        if self._redis is None:
            log.warning("strategy_controller.redis_not_connected")
            return _UNKNOWN_MODE

        try:
            regime_name, updated_at_str, _ = await asyncio.gather(
                self._redis.get(self.REDIS_KEY),
                self._redis.get(self.REDIS_UPDATED_AT_KEY),
                self._redis.get(self.REDIS_CONFIDENCE_KEY),
            )
        except RedisError:
            log.warning("strategy_controller.redis_read_error", exc_info=True)
            return _UNKNOWN_MODE

        if regime_name is None or updated_at_str is None:
            log.debug("strategy_controller.no_regime_set")
            return _UNKNOWN_MODE

        # [P0-1] TTL check.
        try:
            updated_at = datetime.fromisoformat(updated_at_str)
        except ValueError:
            log.warning(
                "strategy_controller.invalid_timestamp",
                updated_at=updated_at_str,
            )
            return _UNKNOWN_MODE

        age_seconds = self._age_seconds(updated_at)
        ttl_seconds = self._regime_ttl_hours * 3600.0

        if age_seconds >= ttl_seconds:
            log.critical(
                "strategy_controller.regime_ttl_expired",
                regime=regime_name,
                age_seconds=age_seconds,
                ttl_seconds=ttl_seconds,
            )
            return _UNKNOWN_MODE

        mode = MODES.get(regime_name, _UNKNOWN_MODE)
        if mode is _UNKNOWN_MODE and regime_name != Regime.UNKNOWN:
            log.warning(
                "strategy_controller.unknown_regime_name",
                regime=regime_name,
            )

        return mode

    async def update_regime(self, regime: str, confidence: float) -> TradingMode:
        """Persist a new regime decision and return the resulting mode.

        Writes ``regime``, ``updated_at`` (UTC ISO-8601), and ``confidence``
        to Redis atomically via a pipeline.  Logs the old → new transition.

        Args:
            regime: Regime name string; must be a valid :class:`Regime` value.
                If not recognised, ``"unknown"`` is stored instead.
            confidence: Claude's confidence score in ``[0, 1]``.

        Returns:
            The :class:`~strategy.modes.TradingMode` that corresponds to
            the stored regime.

        Raises:
            Nothing – Redis errors are caught and logged; the method still
            returns the in-memory mode object.
        """
        # Normalise to a known regime; fall back to unknown.
        normalised = regime if regime in MODES else Regime.UNKNOWN
        if normalised != regime:
            log.warning(
                "strategy_controller.unrecognised_regime_normalised",
                raw=regime,
                normalised=normalised,
            )

        new_mode = MODES[normalised]
        now_iso = datetime.now(tz=UTC).isoformat()

        if self._redis is not None:
            try:
                old_regime = await self._redis.get(self.REDIS_KEY)

                pipe = self._redis.pipeline(transaction=True)
                pipe.set(self.REDIS_KEY, normalised)
                pipe.set(self.REDIS_UPDATED_AT_KEY, now_iso)
                pipe.set(self.REDIS_CONFIDENCE_KEY, str(confidence))
                await pipe.execute()

                log.info(
                    "strategy_controller.regime_updated",
                    old_regime=old_regime,
                    new_regime=normalised,
                    confidence=confidence,
                )
            except RedisError:
                log.warning(
                    "strategy_controller.redis_write_error",
                    regime=normalised,
                    exc_info=True,
                )
        else:
            log.warning(
                "strategy_controller.update_regime_no_redis",
                regime=normalised,
            )

        return new_mode

    async def should_evaluate_signal(self, pair: str, strategy: str) -> bool:
        """Return ``True`` if *pair* and *strategy* are active in the current mode.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            strategy: Strategy identifier, e.g. ``"momentum"``.

        Returns:
            ``True`` only when both the pair and the strategy appear in the
            current :class:`~strategy.modes.TradingMode`.
        """
        mode = await self.get_current_mode()
        return pair in mode.active_pairs and strategy in mode.active_strategies

    async def get_regime_age_seconds(self) -> float | None:
        """Return how old the stored regime is in seconds.

        Args: (none)

        Returns:
            Age in seconds since the last :meth:`update_regime` call, or
            ``None`` if no regime has been stored or Redis is unavailable.
        """
        if self._redis is None:
            return None

        try:
            updated_at_str: str | None = await self._redis.get(self.REDIS_UPDATED_AT_KEY)
        except RedisError:
            log.warning("strategy_controller.redis_read_error_age", exc_info=True)
            return None

        if updated_at_str is None:
            return None

        try:
            updated_at = datetime.fromisoformat(updated_at_str)
        except ValueError:
            return None

        return self._age_seconds(updated_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _age_seconds(updated_at: datetime) -> float:
        """Return elapsed seconds since *updated_at* (UTC-aware required)."""
        now = datetime.now(tz=UTC)
        # If updated_at is naive, assume UTC.
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        return (now - updated_at).total_seconds()
