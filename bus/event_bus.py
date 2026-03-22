"""Asyncio-based, topic-keyed event bus with per-topic backpressure.

Each topic maintains an independent list of subscriber ``asyncio.Queue``
instances so that slow consumers on one topic cannot block unrelated topics.

Backpressure behaviour is governed by the ``DropPolicy`` associated with each
topic in ``TOPIC_CONFIGS``:

* **LOSSY** — the oldest event is discarded to make room for the new one.
  Appropriate for high-frequency market-data streams where a momentarily slow
  consumer should simply skip stale prices.
* **BLOCKING** — the publisher awaits until space is available (bounded by a
  5-second timeout).  Appropriate for control-plane events (signals, orders,
  fills) that must not be silently lost.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from bus.events import TOPIC_CONFIGS, DropPolicy

log: structlog.BoundLogger = structlog.get_logger(__name__)

_PUT_TIMEOUT: float = 5.0  # seconds before a blocking put is considered hung


class EventBus:
    """Central publish/subscribe hub used by all trading-system components.

    Subscribers receive their own dedicated ``asyncio.Queue`` so that each
    consumer's backlog is isolated.  The bus itself does not retain events
    after delivery — it is purely a routing layer.

    Example::

        bus = EventBus()
        queue = bus.subscribe("candle")

        async def consumer() -> None:
            while True:
                event = await queue.get()
                process(event)

        await bus.publish("candle", candle_event)
    """

    def __init__(self) -> None:
        # topic -> list of subscriber queues
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = {}

        # per-topic operation counters
        self._drop_counts: dict[str, int] = {}
        self._block_counts: dict[str, int] = {}
        self._publish_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, topic: str) -> asyncio.Queue[Any]:
        """Register a new subscriber for *topic* and return its queue.

        The returned queue is pre-sized according to the ``TopicConfig``
        registered for *topic*.  Callers should keep a reference to the queue
        and call :meth:`unsubscribe` when they no longer need events.

        Args:
            topic: Routing key, e.g. ``"candle"`` or ``"signal"``.

        Returns:
            A new ``asyncio.Queue`` that will receive all future events
            published to *topic*.

        Raises:
            KeyError: If *topic* is not present in ``TOPIC_CONFIGS``.
        """
        config = TOPIC_CONFIGS[topic]  # raises KeyError for unknown topics
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=config.maxsize)

        self._subscribers.setdefault(topic, []).append(queue)
        self._drop_counts.setdefault(topic, 0)
        self._block_counts.setdefault(topic, 0)
        self._publish_counts.setdefault(topic, 0)

        n_subs = len(self._subscribers[topic])
        log.debug("event_bus.subscribed", topic=topic, total_subscribers=n_subs)
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        """Remove *queue* from the subscriber list for *topic*.

        It is safe to call this method even if *queue* is no longer in the
        list (the operation is idempotent).

        Args:
            topic: Routing key the queue was subscribed to.
            queue: The exact queue object returned by :meth:`subscribe`.
        """
        subscribers = self._subscribers.get(topic, [])
        try:
            subscribers.remove(queue)
        except ValueError:
            log.warning("event_bus.unsubscribe_noop", topic=topic, reason="queue not found")
        else:
            log.debug(
                "event_bus.unsubscribed",
                topic=topic,
                remaining_subscribers=len(subscribers),
            )

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, topic: str, event: Any) -> None:
        """Route *event* to every subscriber queue registered for *topic*.

        Backpressure behaviour is determined by the ``DropPolicy`` configured
        for *topic*:

        * **LOSSY**: if the queue is full, the *oldest* event is discarded and
          the new event is inserted at the tail.
        * **BLOCKING**: the coroutine waits up to ``_PUT_TIMEOUT`` seconds for
          space.  If the timeout elapses a CRITICAL log is emitted and the
          block-count metric is incremented.  The event is **not** delivered to
          that subscriber.

        Args:
            topic: Routing key matching the event's ``topic`` ClassVar.
            event: The event object to deliver.  Typically a ``BaseEvent``
                subclass, but the bus is intentionally untyped at this layer.
        """
        config = TOPIC_CONFIGS.get(topic)
        if config is None:
            log.error("event_bus.unknown_topic", topic=topic)
            return

        subscribers = self._subscribers.get(topic, [])
        if not subscribers:
            return

        self._publish_counts[topic] = self._publish_counts.get(topic, 0) + 1

        if config.policy is DropPolicy.LOSSY:
            await self._publish_lossy(topic, event, subscribers)
        else:
            await self._publish_blocking(topic, event, subscribers)

    async def _publish_lossy(
        self,
        topic: str,
        event: Any,
        subscribers: list[asyncio.Queue[Any]],
    ) -> None:
        """Deliver *event* to all *subscribers* using the LOSSY policy."""
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest item to make room, then insert the new one.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass  # another coroutine drained it between the two calls
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Extremely unlikely — another producer filled it again.
                    pass

                self._drop_counts[topic] = self._drop_counts.get(topic, 0) + 1
                log.debug(
                    "event_bus.event_dropped",
                    topic=topic,
                    total_drops=self._drop_counts[topic],
                )

    async def _publish_blocking(
        self,
        topic: str,
        event: Any,
        subscribers: list[asyncio.Queue[Any]],
    ) -> None:
        """Deliver *event* to all *subscribers* using the BLOCKING policy."""
        for queue in subscribers:
            try:
                await asyncio.wait_for(queue.put(event), timeout=_PUT_TIMEOUT)
            except TimeoutError:
                self._block_counts[topic] = self._block_counts.get(topic, 0) + 1
                log.critical(
                    "event_bus.put_timeout",
                    topic=topic,
                    timeout_seconds=_PUT_TIMEOUT,
                    total_blocks=self._block_counts[topic],
                    queue_depth=queue.qsize(),
                )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def lag_metrics(self) -> dict[str, dict[str, Any]]:
        """Return per-topic observability metrics.

        The returned mapping has one entry per topic that has at least one
        subscriber or has seen at least one published event.

        Returns:
            A dict keyed by topic name.  Each value contains:

            * ``queue_depths`` — list of current ``qsize()`` for every active
              subscriber queue.
            * ``drop_count`` — cumulative number of events dropped (LOSSY
              topics only).
            * ``block_count`` — cumulative number of put-timeout incidents
              (BLOCKING topics only).
            * ``publish_count`` — total events published to this topic since
              bus creation.
        """
        all_topics = set(self._subscribers) | set(self._publish_counts)
        return {
            topic: {
                "queue_depths": [q.qsize() for q in self._subscribers.get(topic, [])],
                "drop_count": self._drop_counts.get(topic, 0),
                "block_count": self._block_counts.get(topic, 0),
                "publish_count": self._publish_counts.get(topic, 0),
            }
            for topic in sorted(all_topics)
        }
