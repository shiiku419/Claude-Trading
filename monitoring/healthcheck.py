"""Periodic health-check runner for all trading-system components.

Each check is registered as a callable returning a :class:`HealthStatus`.
The runner executes all checks on a configurable interval and logs a
structured summary after every pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)

# A check function may be sync or async; both return HealthStatus.
CheckFn = Callable[[], "HealthStatus | Coroutine[Any, Any, HealthStatus]"]


@dataclass
class HealthStatus:
    """Result of a single component health check.

    Attributes:
        component: Human-readable component name, e.g. ``"websocket"``.
        healthy: ``True`` if the component is operating normally.
        last_check: UTC timestamp when this status was produced.
        details: Optional free-form description of the failure or state.
    """

    component: str
    healthy: bool
    last_check: datetime
    details: str = ""


class HealthChecker:
    """Periodic health checks for all trading-system components.

    Registered check functions are executed on every tick.  Results are
    stored in :attr:`last_results` and emitted as structured log lines so
    that operators can observe component health without polling an API.

    Checks that should be registered:

    * **websocket** — ``last_message_time`` is less than 30 s ago.
    * **redis** — ``PING`` round-trip succeeds.
    * **database** — async connection check succeeds.
    * **kill_switch** — reports current activation state.
    * **claude** — last successful API call was recent.
    * **event_bus** — queue lag metrics are within bounds.

    Example::

        checker = HealthChecker(check_interval=60)
        checker.register_check("redis", _check_redis)
        asyncio.create_task(checker.run_periodic())

    Args:
        check_interval: Seconds between full check passes (default ``60``).
    """

    def __init__(self, check_interval: int = 60) -> None:
        self._check_interval: int = check_interval
        self._checks: dict[str, CheckFn] = {}
        self._last_results: list[HealthStatus] = []
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_check(self, name: str, check_fn: CheckFn) -> None:
        """Register a health-check function under *name*.

        The function must accept no arguments and return a
        :class:`HealthStatus` (sync or async).  Registering a name a
        second time silently replaces the previous check.

        Args:
            name: Unique identifier for this check, e.g. ``"redis"``.
            check_fn: Callable ``() -> HealthStatus`` (sync or async).
        """
        self._checks[name] = check_fn
        log.debug("health_checker.registered", component=name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def check_all(self) -> list[HealthStatus]:
        """Execute every registered check and return the full results list.

        Each check is awaited if it returns a coroutine; synchronous
        callables are called directly.  Exceptions inside a check are
        caught and converted into an unhealthy :class:`HealthStatus` so
        that one broken check cannot prevent the others from running.

        Returns:
            List of :class:`HealthStatus` objects, one per registered check,
            in registration order.
        """
        results: list[HealthStatus] = []
        for name, fn in self._checks.items():
            try:
                maybe_coro = fn()
                if asyncio.iscoroutine(maybe_coro):
                    status: HealthStatus = await maybe_coro
                else:
                    # fn is synchronous; mypy sees maybe_coro as the union type
                    status = maybe_coro  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                status = HealthStatus(
                    component=name,
                    healthy=False,
                    last_check=datetime.now(tz=UTC),
                    details=f"check raised: {exc!r}",
                )
                log.error(
                    "health_checker.check_raised",
                    component=name,
                    error=repr(exc),
                )
            results.append(status)

        self._last_results = results
        return results

    async def run_periodic(self) -> None:
        """Run all registered checks every :attr:`_check_interval` seconds.

        Logs a ``health_checker.pass`` event after each tick with a
        summary of healthy/unhealthy component counts.  Returns when
        :meth:`stop` is called.
        """
        log.info("health_checker.started", interval_seconds=self._check_interval)
        while not self._stop_event.is_set():
            results = await self.check_all()
            healthy_count = sum(1 for r in results if r.healthy)
            unhealthy = [r.component for r in results if not r.healthy]
            log.info(
                "health_checker.pass",
                total=len(results),
                healthy=healthy_count,
                unhealthy_components=unhealthy,
            )
            if unhealthy:
                for r in results:
                    if not r.healthy:
                        log.warning(
                            "health_checker.component_unhealthy",
                            component=r.component,
                            details=r.details,
                        )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(self._check_interval),
                )
            except TimeoutError:
                pass  # normal — just means the interval elapsed

        log.info("health_checker.stopped")

    async def stop(self) -> None:
        """Signal the periodic runner to exit on its next iteration.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        self._stop_event.set()
        log.debug("health_checker.stop_requested")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def last_results(self) -> list[HealthStatus]:
        """Most-recent :class:`HealthStatus` list from the last :meth:`check_all` call.

        Returns an empty list before the first check has completed.
        """
        return self._last_results
