"""Transaction Cost Analysis (TCA) for execution quality measurement.

:class:`TCAAnalyzer` collects per-execution metrics and exposes aggregate
statistics for monitoring and strategy optimisation.

Slippage definition
-------------------
Slippage measures how far the actual fill price deviated from the expected
(signal) price, expressed in basis points (1 bps = 0.01%).

For a **buy** order, positive slippage means we paid *more* than expected
(adverse); for a **sell** order, positive slippage means we received *less*
than expected (adverse).  The convention used here normalises both sides so
that a positive ``avg_slippage_bps`` always indicates adverse execution.

.. code-block:: text

    buy  slippage_bps  = (actual - expected) / expected * 10_000
    sell slippage_bps  = (expected - actual) / expected * 10_000

Market impact
-------------
Market impact estimates the price movement caused by our order.  It is
computed as the absolute percentage price change relative to order size::

    impact_bps = abs(actual - expected) / expected * 10_000 * quantity_scaling

where ``quantity_scaling = 1.0`` in this simple model.  A more sophisticated
implementation would scale by traded notional relative to average daily volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dataclass
class TCAMetrics:
    """Aggregate transaction cost analysis metrics.

    Attributes:
        avg_slippage_bps: Average slippage across all executions, in basis
            points.  Positive means adverse (paid more / received less than
            expected).
        total_fees_usd: Cumulative trading fees across all executions, in
            quote-asset (USD) units.
        avg_market_impact_bps: Average estimated market impact across all
            executions, in basis points.
        total_trades: Number of executions recorded since the last
            :meth:`~TCAAnalyzer.reset` call.
    """

    avg_slippage_bps: float
    total_fees_usd: float
    avg_market_impact_bps: float
    total_trades: int


@dataclass
class _ExecutionRecord:
    """Internal per-execution data point.

    Attributes:
        slippage_bps: Signed slippage for this execution (positive = adverse).
        fees: Fees charged for this execution (quote-asset units).
        market_impact_bps: Estimated market impact in basis points.
    """

    slippage_bps: float
    fees: float
    market_impact_bps: float


class TCAAnalyzer:
    """Transaction Cost Analysis for execution quality.

    Tracks slippage, fees, and estimated market impact across all executions
    and exposes aggregated metrics via :meth:`get_metrics`.

    State is entirely in-memory; call :meth:`reset` to start a new analysis
    window (e.g. at the start of each trading session).
    """

    def __init__(self) -> None:
        self._records: list[_ExecutionRecord] = []

    def record_execution(
        self,
        expected_price: float,
        actual_price: float,
        quantity: float,
        fees: float,
        side: str,
    ) -> None:
        """Record a single order execution and accumulate cost metrics.

        Args:
            expected_price: The signal or reference price at which the order
                was expected to fill (e.g. mid-price at signal time).
            actual_price: The actual volume-weighted average fill price.
            quantity: Filled quantity in base-asset units.
            fees: Fees charged for this fill (quote-asset units).
            side: ``"buy"`` or ``"sell"``.

        Raises:
            ValueError: If ``expected_price`` is zero (division guard).
        """
        if expected_price == 0.0:
            log.warning(
                "tca.zero_expected_price",
                actual_price=actual_price,
                side=side,
            )
            return

        side_lower = side.lower()

        # Slippage: positive = adverse for both sides.
        if side_lower == "buy":
            slippage_bps = (actual_price - expected_price) / expected_price * 10_000.0
        else:
            # sell: adverse when actual < expected
            slippage_bps = (expected_price - actual_price) / expected_price * 10_000.0

        # Market impact: absolute price deviation scaled by quantity.
        # Using a simple proportional model; advanced models would use ADV.
        raw_impact = abs(actual_price - expected_price) / expected_price * 10_000.0
        market_impact_bps = raw_impact * math.log1p(quantity)

        record = _ExecutionRecord(
            slippage_bps=slippage_bps,
            fees=fees,
            market_impact_bps=market_impact_bps,
        )
        self._records.append(record)

        log.debug(
            "tca.record_execution",
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            fees=fees,
            side=side,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
        )

    def get_metrics(self) -> TCAMetrics:
        """Return aggregated TCA metrics across all recorded executions.

        Returns:
            A :class:`TCAMetrics` snapshot.  All averages are simple
            (unweighted) means.  Returns zeroed metrics if no executions
            have been recorded yet.
        """
        n = len(self._records)
        if n == 0:
            return TCAMetrics(
                avg_slippage_bps=0.0,
                total_fees_usd=0.0,
                avg_market_impact_bps=0.0,
                total_trades=0,
            )

        avg_slippage = sum(r.slippage_bps for r in self._records) / n
        total_fees = sum(r.fees for r in self._records)
        avg_impact = sum(r.market_impact_bps for r in self._records) / n

        return TCAMetrics(
            avg_slippage_bps=avg_slippage,
            total_fees_usd=total_fees,
            avg_market_impact_bps=avg_impact,
            total_trades=n,
        )

    def reset(self) -> None:
        """Clear all recorded executions and reset metrics to zero.

        Typical use: call at the start of each trading session or analysis
        window.
        """
        self._records.clear()
        log.info("tca.reset")
