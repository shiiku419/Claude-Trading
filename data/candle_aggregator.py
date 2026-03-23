"""Candle aggregator: bridges the event bus and the feature store.

Subscribes to the ``"candle"`` topic on the event bus and forwards each
*closed* candle to the feature store.  Partial (in-progress) candles
are intentionally ignored to avoid polluting the ring buffer with
incomplete OHLCV data.

Typical usage::

    aggregator = CandleAggregator(event_bus, feature_store)
    await aggregator.start()
    # ... bot runs ...
    await aggregator.stop()
"""

from __future__ import annotations

import asyncio

import structlog

from bus.event_bus import EventBus
from bus.events import CandleEvent
from data.feature_store import FeatureStore

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Log a progress line every N closed candles processed.
_LOG_EVERY_N_CANDLES: int = 100


class CandleAggregator:
    """Subscribes to CandleEvents from the event bus and updates the feature store.

    Only processes closed candles (``is_closed=True``) to avoid writing
    partial data into the feature-store ring buffers.

    Attributes:
        _event_bus: The system-wide event bus.
        _feature_store: The feature store that receives OHLCV updates.
        _queue: The subscriber queue allocated by the event bus.
        _run_task: Background asyncio task running the consumer loop.
        _candles_processed: Cumulative count of closed candles forwarded
            to the feature store since this aggregator was started.
        _stop_event: Set by :meth:`stop` to break the consumer loop.
    """

    def __init__(self, event_bus: EventBus, feature_store: FeatureStore) -> None:
        """Initialise the aggregator without starting any I/O.

        Args:
            event_bus: Shared event bus instance.
            feature_store: Feature store instance to receive candle data.
        """
        self._event_bus = event_bus
        self._feature_store = feature_store
        self._queue: asyncio.Queue[CandleEvent] | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._candles_processed: int = 0
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the candle topic and begin processing events.

        Spawns a background :class:`asyncio.Task` that drains the
        subscriber queue and forwards closed candles to the feature store.
        Returns immediately; use :meth:`stop` for graceful shutdown.

        Raises:
            RuntimeError: If the aggregator is already running.
        """
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("CandleAggregator is already running")

        self._stop_event.clear()
        self._queue = self._event_bus.subscribe("candle")  # type: ignore[assignment]

        self._run_task = asyncio.create_task(
            self._consume(),
            name="candle_aggregator",
        )

        log.info("candle_aggregator.started")

    async def stop(self) -> None:
        """Signal the consumer loop to exit and wait for it to finish.

        Safe to call when the aggregator is not running.
        """
        self._stop_event.set()

        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            self._run_task = None

        if self._queue is not None:
            self._event_bus.unsubscribe("candle", self._queue)  # type: ignore[arg-type]
            self._queue = None

        log.info(
            "candle_aggregator.stopped",
            total_candles_processed=self._candles_processed,
        )

    # ------------------------------------------------------------------
    # Internal consumer loop
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        """Drain the subscriber queue and forward closed candles to the store.

        Runs until :meth:`stop` is called or the task is cancelled.  Each
        iteration calls :meth:`asyncio.Queue.get` with a short timeout so
        that the stop event is checked promptly even when no events arrive.
        """
        assert self._queue is not None, "_consume called before queue was created"

        while not self._stop_event.is_set():
            try:
                event: CandleEvent = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
            except TimeoutError:
                # No event within the polling window — check stop flag and loop.
                continue
            except asyncio.CancelledError:
                break

            log.info("candle_aggregator.received_event", pair=event.pair, timeframe=event.timeframe, is_closed=event.is_closed)
            if not event.is_closed:
                # Skip in-progress candles to avoid partial data in the store.
                log.debug("candle_aggregator.skipping_unclosed", pair=event.pair, timeframe=event.timeframe)
                continue

            log.info("candle_aggregator.processing_closed_candle", pair=event.pair, timeframe=event.timeframe)
            try:
                await self._feature_store.update(
                    pair=event.pair,
                    timeframe=event.timeframe,
                    timestamp_ms=event.timestamp_ms,
                    o=event.open,
                    h=event.high,
                    low=event.low,
                    c=event.close,
                    v=event.volume,
                )
            except Exception:
                log.warning(
                    "candle_aggregator.feature_store_error",
                    pair=event.pair,
                    timeframe=event.timeframe,
                    timestamp_ms=event.timestamp_ms,
                    exc_info=True,
                )
                continue

            self._candles_processed += 1

            if self._candles_processed % _LOG_EVERY_N_CANDLES == 0:
                log.info(
                    "candle_aggregator.progress",
                    candles_processed=self._candles_processed,
                )
