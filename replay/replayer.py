"""Historical candle replay through the full signal + risk + execution pipeline.

Used for backtesting and validation.  The :class:`Replayer` feeds a pre-loaded
numpy candle array one candle at a time through the same ``CompositeSignal``
logic that runs in live trading, making the replay results directly comparable
to paper-trading logs.

Design notes
------------
* ``replay`` is a pure signal-generation pass — no risk or execution side effects.
* ``replay_with_execution`` exercises the full pipeline and returns aggregated
  metrics that feed :class:`~replay.evaluation.BacktestEvaluator`.
* Both methods are ``async`` so that they can drive the async ``RiskGate`` and
  ``ExecutionEngine`` without blocking the event loop.
* The :class:`~data.feature_store.FeatureStore` is used to feed candle windows
  into the signal generators, maintaining the same data-access contract as live
  trading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from data.feature_store import FeatureStore
from signals.base import Signal
from signals.composite import CompositeSignal

if TYPE_CHECKING:
    from execution.engine import ExecutionEngine
    from risk.base import PortfolioState
    from risk.gate import RiskGate

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Number of candles fed to the feature store before signal generation begins.
# This gives the ring buffer enough history for all indicator warm-up periods.
_MIN_WARMUP_CANDLES: int = 30


class Replayer:
    """Replays historical candle data through the signal engine.

    Used for backtesting and validation.  Feeds candles one at a time through
    the same signal + risk + execution pipeline as live trading.

    The :class:`~data.feature_store.FeatureStore` is populated candle-by-candle
    so that each call to ``CompositeSignal.evaluate`` sees exactly the same
    data window it would see in production, preventing look-ahead bias.

    Args:
        feature_store: Pre-initialised feature store.  The replayer writes
            candles into it and reads windows back for signal generation.
        signal_generators: List of ``(generator, weight)`` pairs forwarded to
            :class:`~signals.composite.CompositeSignal`.
        composite_threshold: Minimum weighted-average strength required for
            the composite signal to fire (default ``0.6``).
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        signal_generators: list[tuple[object, float]],
        composite_threshold: float = 0.6,
    ) -> None:
        self._feature_store = feature_store
        self._composite = CompositeSignal(
            generators=signal_generators,  # type: ignore[arg-type]
            threshold=composite_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def replay(
        self,
        candles: np.ndarray,
        pair: str,
        timeframe: str,
    ) -> list[Signal]:
        """Replay candles and return all generated signals.

        Each candle in *candles* is inserted into the feature store in
        chronological order.  After each insertion, the composite signal is
        evaluated against the accumulated window.  Signals are collected and
        returned when all candles have been processed.

        Args:
            candles: Numpy structured array with ``CANDLE_DTYPE``, ordered
                oldest-first (index 0 = oldest, index -1 = newest).
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            timeframe: Candle timeframe string, e.g. ``"1m"``.

        Returns:
            List of :class:`~signals.base.Signal` objects produced during the
            replay, in chronological order.  May be empty if no signal threshold
            was met.
        """
        signals: list[Signal] = []

        for i, candle in enumerate(candles):
            await self._ingest_candle(candle, pair, timeframe)

            # Need at least _MIN_WARMUP_CANDLES before attempting evaluation
            # so that all indicators have sufficient history.
            if i < _MIN_WARMUP_CANDLES - 1:
                continue

            window = self._feature_store.get_candles(pair, timeframe, _MIN_WARMUP_CANDLES)
            if window is None:
                continue

            signal = self._composite.evaluate(pair, window)
            if signal is not None:
                signals.append(signal)
                log.debug(
                    "replayer.signal_generated",
                    pair=pair,
                    direction=str(signal.direction),
                    strength=signal.strength,
                    candle_index=i,
                )

        log.info(
            "replayer.replay_complete",
            pair=pair,
            timeframe=timeframe,
            total_candles=len(candles),
            signals_generated=len(signals),
        )
        return signals

    async def replay_with_execution(
        self,
        candles: np.ndarray,
        pair: str,
        timeframe: str,
        risk_gate: RiskGate,
        execution_engine: ExecutionEngine,
        portfolio: PortfolioState,
    ) -> dict[str, object]:
        """Full pipeline replay.  Returns metrics dict.

        Each generated signal is evaluated by the risk gate.  Approved orders
        are forwarded to the execution engine.  Aggregated counters are
        collected and returned as a plain ``dict`` suitable for ingestion by
        :class:`~replay.evaluation.BacktestEvaluator`.

        Args:
            candles: Numpy structured array with ``CANDLE_DTYPE``, ordered
                oldest-first.
            pair: Trading pair symbol.
            timeframe: Candle timeframe string.
            risk_gate: Configured :class:`~risk.gate.RiskGate` instance.
            execution_engine: Connected :class:`~execution.engine.ExecutionEngine`.
            portfolio: Portfolio snapshot passed to every risk-gate evaluation.

        Returns:
            Metrics dict with the following keys:

            * ``"total_candles"`` – number of candles processed.
            * ``"total_signals"`` – number of composite signals fired.
            * ``"approved_orders"`` – number of signals approved by the risk gate.
            * ``"rejected_orders"`` – number of signals rejected by the risk gate.
            * ``"executed_orders"`` – number of orders accepted by the exchange.
            * ``"failed_orders"`` – number of approved orders that the exchange
              rejected.
            * ``"order_results"`` – list of raw
              :class:`~execution.base.OrderResult` objects.
        """
        total_signals = 0
        approved_count = 0
        rejected_count = 0
        executed_count = 0
        failed_count = 0
        order_results: list[object] = []

        for i, candle in enumerate(candles):
            await self._ingest_candle(candle, pair, timeframe)

            if i < _MIN_WARMUP_CANDLES - 1:
                continue

            window = self._feature_store.get_candles(pair, timeframe, _MIN_WARMUP_CANDLES)
            if window is None:
                continue

            signal = self._composite.evaluate(pair, window)
            if signal is None:
                continue

            total_signals += 1
            current_price = float(candle["close"])
            # Use a modest default quantity for replay purposes.
            requested_quantity = 0.001

            approved_order = await risk_gate.evaluate(
                signal=signal,
                portfolio=portfolio,
                requested_quantity=requested_quantity,
                current_price=current_price,
            )

            if approved_order is None:
                rejected_count += 1
                log.debug(
                    "replayer.signal_rejected_by_risk",
                    pair=pair,
                    candle_index=i,
                    direction=str(signal.direction),
                )
                continue

            approved_count += 1
            result = await execution_engine.execute(approved_order)

            if result is not None:
                executed_count += 1
                order_results.append(result)
                log.debug(
                    "replayer.order_executed",
                    pair=pair,
                    candle_index=i,
                    client_order_id=result.client_order_id,
                    status=str(result.status),
                )
            else:
                failed_count += 1
                log.debug(
                    "replayer.order_failed",
                    pair=pair,
                    candle_index=i,
                )

        metrics: dict[str, object] = {
            "total_candles": len(candles),
            "total_signals": total_signals,
            "approved_orders": approved_count,
            "rejected_orders": rejected_count,
            "executed_orders": executed_count,
            "failed_orders": failed_count,
            "order_results": order_results,
        }

        log.info("replayer.full_pipeline_complete", pair=pair, **{
            k: v for k, v in metrics.items() if k != "order_results"
        })
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ingest_candle(
        self,
        candle: np.void,
        pair: str,
        timeframe: str,
    ) -> None:
        """Write a single structured candle record into the feature store.

        Args:
            candle: A single row from a ``CANDLE_DTYPE`` structured array.
            pair: Trading pair symbol.
            timeframe: Candle timeframe string.
        """
        await self._feature_store.update(
            pair=pair,
            timeframe=timeframe,
            timestamp_ms=int(candle["timestamp"]),
            o=float(candle["open"]),
            h=float(candle["high"]),
            low=float(candle["low"]),
            c=float(candle["close"]),
            v=float(candle["volume"]),
        )
