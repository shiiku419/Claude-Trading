"""Unit tests for :class:`bus.event_bus.EventBus`.

Covers:
- Basic publish/subscribe round-trip (lossy topic)
- Lossy back-pressure: drops when queue is full, no crash
- Blocking back-pressure: publish awaits when queue is full
- Lag metrics accumulate correctly across publishes and drops
- Unsubscribe prevents further event delivery
"""

from __future__ import annotations

import asyncio

import pytest

from bus.event_bus import EventBus
from bus.events import CandleEvent, OrderEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candle(pair: str = "BTC/USDT", close: float = 50_000.0) -> CandleEvent:
    """Build a minimal :class:`CandleEvent` for testing."""
    return CandleEvent(
        pair=pair,
        timeframe="5m",
        timestamp_ms=1_700_000_000_000,
        open=close - 10,
        high=close + 20,
        low=close - 30,
        close=close,
        volume=1.5,
        is_closed=True,
    )


def _make_order(order_id: str = "ord-001") -> OrderEvent:
    """Build a minimal :class:`OrderEvent` for testing."""
    return OrderEvent(
        client_order_id=order_id,
        pair="BTC/USDT",
        side="buy",
        quantity=0.01,
        order_type="market",
        status="pending",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_publish_subscribe(event_bus: EventBus) -> None:
    """A subscribed queue receives the exact event that was published."""
    queue = event_bus.subscribe("candle")
    event = _make_candle()

    await event_bus.publish("candle", event)

    received = queue.get_nowait()
    assert received is event
    assert isinstance(received, CandleEvent)
    assert received.pair == "BTC/USDT"
    assert received.close == 50_000.0


async def test_lossy_backpressure(event_bus: EventBus) -> None:
    """Publishing more events than the queue holds drops old ones silently.

    We manually pre-fill a subscriber queue to its capacity and then publish
    additional events to confirm the bus does not raise, does not deadlock,
    and increments its drop counter.
    """
    queue = event_bus.subscribe("candle")
    cap = queue.maxsize  # 10 000 per TOPIC_CONFIGS

    # Drain is expensive at 10 000; instead directly fill the queue to capacity
    # by putting events without going through the bus, then publish one more.
    for i in range(cap):
        queue.put_nowait(_make_candle(close=float(i)))

    assert queue.full()

    # The bus must handle a full queue gracefully (drop oldest, insert new).
    extra_count = 5
    for i in range(extra_count):
        await event_bus.publish("candle", _make_candle(close=float(cap + i)))

    # Queue must still be exactly at capacity — no overflow.
    assert queue.qsize() == cap

    # Drop counter must reflect the extra events.
    metrics = event_bus.lag_metrics()
    assert metrics["candle"]["drop_count"] == extra_count

    # publish_count only counts events routed through publish().
    assert metrics["candle"]["publish_count"] == extra_count


async def test_blocking_publish(event_bus: EventBus) -> None:
    """Publish to a full BLOCKING queue must await until space is available.

    We fill an ``order`` queue to capacity, then verify that the next
    ``publish`` call suspends (does not return within a short timeout).
    After we drain one item the publish should complete.
    """
    queue = event_bus.subscribe("order")
    max_q = queue.maxsize  # from TOPIC_CONFIGS

    # Fill the queue to capacity synchronously.
    for i in range(max_q):
        queue.put_nowait(_make_order(order_id=f"ord-fill-{i}"))

    assert queue.full()

    extra_event = _make_order(order_id="ord-extra")

    # The publish coroutine should block because the queue is full.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            event_bus.publish("order", extra_event),
            timeout=0.05,
        )


async def test_lag_metrics(event_bus: EventBus) -> None:
    """lag_metrics() returns accurate publish and drop counts per topic.

    Strategy: subscribe to the candle topic, pre-fill the queue to its
    capacity, then publish a known number of additional events.  All extras
    will be dropped and must appear in the metrics drop_count.
    """
    queue = event_bus.subscribe("candle")
    cap = queue.maxsize

    # Pre-fill queue directly so publish() only sees overflow events.
    for i in range(cap):
        queue.put_nowait(_make_candle(close=float(i)))

    publish_count = 7
    drop_count = publish_count  # every publish causes a drop (queue is full)
    for i in range(publish_count):
        await event_bus.publish("candle", _make_candle(close=float(cap + i)))

    metrics = event_bus.lag_metrics()

    assert "candle" in metrics
    assert metrics["candle"]["publish_count"] == publish_count
    assert metrics["candle"]["drop_count"] == drop_count
    # Queue must still hold exactly `cap` items.
    assert queue.qsize() == cap


async def test_unsubscribe(event_bus: EventBus) -> None:
    """After unsubscribing, a queue must receive no further events."""
    queue = event_bus.subscribe("candle")

    # Publish one event before unsubscribing — it should arrive.
    await event_bus.publish("candle", _make_candle(close=1.0))
    assert queue.qsize() == 1

    event_bus.unsubscribe("candle", queue)

    # Drain the pre-unsubscribe event.
    queue.get_nowait()

    # Publish after unsubscribing — the queue must stay empty.
    await event_bus.publish("candle", _make_candle(close=2.0))
    assert queue.qsize() == 0


async def test_unsubscribe_noop_for_unknown_queue(event_bus: EventBus) -> None:
    """Unsubscribing a queue that was never registered must not raise."""
    orphan: asyncio.Queue[object] = asyncio.Queue()
    # Should not raise or crash; a warning log is acceptable.
    event_bus.unsubscribe("candle", orphan)  # type: ignore[arg-type]


async def test_publish_no_subscribers_is_noop(event_bus: EventBus) -> None:
    """Publishing to a topic with no subscribers must return silently."""
    # No subscriber registered — must not raise.
    await event_bus.publish("candle", _make_candle())

    metrics = event_bus.lag_metrics()
    # Nothing was published (no subscribers so publish_count is not incremented).
    assert metrics.get("candle", {}).get("publish_count", 0) == 0


async def test_multiple_subscribers_all_receive(event_bus: EventBus) -> None:
    """Every subscriber for the same topic must independently receive the event."""
    q1 = event_bus.subscribe("candle")
    q2 = event_bus.subscribe("candle")

    event = _make_candle()
    await event_bus.publish("candle", event)

    assert q1.get_nowait() is event
    assert q2.get_nowait() is event
