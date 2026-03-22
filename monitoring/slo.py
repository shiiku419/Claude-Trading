"""SLO definitions and real-time tracking for the trading system.

Four SLOs are tracked:

* **ws_uptime** — WebSocket connection uptime must exceed 99.5 % of total
  elapsed time.
* **order_latency_p99** — 99th-percentile signal-to-order latency must be
  below 2 seconds.
* **claude_success_rate** — Claude API call success rate must exceed 95 %.
* **reconciliation_drift** — Maximum balance reconciliation discrepancy
  recorded today must be 0.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass
from datetime import date

import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass
class SLODefinition:
    """Static description of a single SLO.

    Attributes:
        name: Machine-readable identifier, e.g. ``"ws_uptime"``.
        target: Numeric threshold.  Interpretation varies by SLO:
            for rates this is a fraction (``0.995``); for latency it is
            a maximum in seconds (``2.0``); for drift it is a maximum
            count (``0.0``).
        description: Human-readable summary of what is being measured.
    """

    name: str
    target: float
    description: str


@dataclass
class SLOStatus:
    """Point-in-time status for a single SLO.

    Attributes:
        name: Matches :attr:`SLODefinition.name`.
        target: The configured target value.
        actual: The measured value at query time.
        met: ``True`` when the SLO is currently satisfied.
        window_seconds: Elapsed seconds over which the measurement was taken.
    """

    name: str
    target: float
    actual: float
    met: bool
    window_seconds: float


class SLOTracker:
    """Tracks real-time SLO metrics for the trading system.

    All measurement methods are non-blocking and safe to call from async
    contexts.  Call :meth:`get_status` at any time to retrieve a snapshot.

    Example::

        tracker = SLOTracker()
        tracker.record_ws_connected()
        # ...trading runs...
        tracker.record_ws_disconnected()
        tracker.record_order_latency(0.35)
        tracker.record_claude_call(success=True)
        for status in tracker.get_status():
            print(status.name, status.met)
    """

    SLO_DEFINITIONS: list[SLODefinition] = [
        SLODefinition("ws_uptime", 0.995, "WebSocket connection uptime"),
        SLODefinition("order_latency_p99", 2.0, "Order latency p99 in seconds"),
        SLODefinition("claude_success_rate", 0.95, "Claude API call success rate"),
        SLODefinition("reconciliation_drift", 0.0, "Balance reconciliation discrepancies"),
    ]

    def __init__(self) -> None:
        self._start_time: float = time.monotonic()

        # --- ws_uptime ---
        # Track cumulative connected seconds by recording connect/disconnect walls.
        self._ws_connected: bool = False
        self._ws_connected_seconds: float = 0.0
        self._ws_last_connect_time: float | None = None

        # --- order_latency_p99 ---
        # Keep a sorted list of latencies for O(log n) percentile computation.
        self._latencies: list[float] = []

        # --- claude_success_rate ---
        self._claude_total: int = 0
        self._claude_success: int = 0

        # --- reconciliation_drift ---
        # Reset daily; track today's date and max drift observed.
        self._drift_date: date = date.today()
        self._max_drift: float = 0.0

    # ------------------------------------------------------------------
    # WebSocket uptime
    # ------------------------------------------------------------------

    def record_ws_connected(self) -> None:
        """Mark the WebSocket as connected at the current monotonic time.

        Idempotent: calling while already connected has no effect.
        """
        if self._ws_connected:
            return
        self._ws_connected = True
        self._ws_last_connect_time = time.monotonic()
        log.debug("slo_tracker.ws_connected")

    def record_ws_disconnected(self) -> None:
        """Mark the WebSocket as disconnected and accumulate connected time.

        Idempotent: calling while already disconnected has no effect.
        """
        if not self._ws_connected:
            return
        if self._ws_last_connect_time is not None:
            self._ws_connected_seconds += time.monotonic() - self._ws_last_connect_time
        self._ws_connected = False
        self._ws_last_connect_time = None
        log.debug("slo_tracker.ws_disconnected", connected_seconds=self._ws_connected_seconds)

    def _ws_current_connected_seconds(self) -> float:
        """Return total connected seconds including any in-progress session."""
        total = self._ws_connected_seconds
        if self._ws_connected and self._ws_last_connect_time is not None:
            total += time.monotonic() - self._ws_last_connect_time
        return total

    # ------------------------------------------------------------------
    # Order latency
    # ------------------------------------------------------------------

    def record_order_latency(self, latency_seconds: float) -> None:
        """Record the elapsed time from signal emission to order submission.

        The value is inserted into a sorted list so that p99 can be
        computed with a simple index lookup.

        Args:
            latency_seconds: Non-negative duration in seconds.
        """
        bisect.insort(self._latencies, latency_seconds)
        log.debug("slo_tracker.order_latency_recorded", latency=latency_seconds)

    def _p99_latency(self) -> float | None:
        """Compute the p99 latency from the sorted sample list.

        Returns ``None`` when no samples have been recorded.
        """
        n = len(self._latencies)
        if n == 0:
            return None
        # Index of the 99th percentile using the ceiling method.
        # ceil(0.99 * n) gives the rank (1-based); clamp to valid range.
        idx = min(n - 1, math.ceil(0.99 * n))
        return self._latencies[idx]

    # ------------------------------------------------------------------
    # Claude success rate
    # ------------------------------------------------------------------

    def record_claude_call(self, success: bool) -> None:
        """Record the outcome of a single Claude API call.

        Args:
            success: ``True`` if the call completed without error.
        """
        self._claude_total += 1
        if success:
            self._claude_success += 1
        log.debug(
            "slo_tracker.claude_call_recorded",
            success=success,
            total=self._claude_total,
        )

    # ------------------------------------------------------------------
    # Reconciliation drift
    # ------------------------------------------------------------------

    def record_reconciliation_drift(self, drift: float) -> None:
        """Record a reconciliation discrepancy observation.

        The daily maximum is updated.  The counter resets automatically
        when the calendar date advances.

        Args:
            drift: Absolute discrepancy magnitude (non-negative).
        """
        today = date.today()
        if today != self._drift_date:
            self._drift_date = today
            self._max_drift = 0.0
            log.info("slo_tracker.drift_daily_reset")

        self._max_drift = max(self._max_drift, drift)
        log.debug("slo_tracker.drift_recorded", drift=drift, max_today=self._max_drift)

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    def get_status(self) -> list[SLOStatus]:
        """Return a current snapshot of all SLO statuses.

        Each :class:`SLOStatus` in the returned list reflects the state
        at the moment of the call.

        Returns:
            List of :class:`SLOStatus` objects, one per SLO definition,
            in the order they appear in :attr:`SLO_DEFINITIONS`.
        """
        now = time.monotonic()
        elapsed = now - self._start_time
        statuses: list[SLOStatus] = []

        for defn in self.SLO_DEFINITIONS:
            if defn.name == "ws_uptime":
                connected = self._ws_current_connected_seconds()
                actual = connected / elapsed if elapsed > 0 else 0.0
                met = actual >= defn.target
                statuses.append(SLOStatus(defn.name, defn.target, actual, met, elapsed))

            elif defn.name == "order_latency_p99":
                p99 = self._p99_latency()
                if p99 is None:
                    # No orders yet; treat as met (no violation observed).
                    actual_lat = 0.0
                    met_lat = True
                else:
                    actual_lat = p99
                    met_lat = actual_lat <= defn.target
                statuses.append(
                    SLOStatus(defn.name, defn.target, actual_lat, met_lat, elapsed)
                )

            elif defn.name == "claude_success_rate":
                if self._claude_total == 0:
                    actual_rate = 1.0  # vacuously met
                    met_rate = True
                else:
                    actual_rate = self._claude_success / self._claude_total
                    met_rate = actual_rate >= defn.target
                statuses.append(
                    SLOStatus(defn.name, defn.target, actual_rate, met_rate, elapsed)
                )

            elif defn.name == "reconciliation_drift":
                # Reset check in case the date rolled over since last record call.
                today = date.today()
                if today != self._drift_date:
                    self._drift_date = today
                    self._max_drift = 0.0
                actual_drift = self._max_drift
                met_drift = actual_drift <= defn.target
                statuses.append(
                    SLOStatus(defn.name, defn.target, actual_drift, met_drift, elapsed)
                )

        return statuses
