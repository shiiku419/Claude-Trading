"""Unit tests for the Binance WebSocket data layer.

Covers:
- Symbol conversion helpers (_pair_to_binance_symbol, _binance_symbol_to_pair)
- Stream URL construction for testnet and mainnet
- Kline message parsing (closed and open candles)
- ReconnectManager backoff calculation and attempt-reset behaviour
- CandleAggregator: only closed candles reach the feature store,
  open candles are silently skipped

No actual network calls are made; all I/O dependencies are replaced with
in-process fakes or mocks.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bus.event_bus import EventBus
from bus.events import CandleEvent
from data.candle_aggregator import CandleAggregator
from data.feeds.binance_ws import (
    BinanceKlineFeed,
    _binance_symbol_to_pair,
    _build_stream_url,
    _pair_to_binance_symbol,
)
from execution.reconnect import ReconnectManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_KLINE_MSG = json.dumps(
    {
        "stream": "btcusdt@kline_5m",
        "data": {
            "e": "kline",
            "E": 1234567890123,
            "s": "BTCUSDT",
            "k": {
                "t": 1234567890000,
                "T": 1234567890299,
                "s": "BTCUSDT",
                "i": "5m",
                "f": 100,
                "L": 200,
                "o": "50000.0",
                "h": "50100.0",
                "l": "49900.0",
                "c": "50050.0",
                "v": "10.5",
                "n": 100,
                "x": True,
                "q": "525525.0",
                "V": "5.25",
                "Q": "262762.5",
                "B": "0",
            },
        },
    }
)

_SAMPLE_KLINE_MSG_OPEN = json.dumps(
    {
        "stream": "btcusdt@kline_5m",
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1234567890000,
                "T": 1234567890299,
                "s": "BTCUSDT",
                "i": "5m",
                "o": "50000.0",
                "h": "50100.0",
                "l": "49900.0",
                "c": "50050.0",
                "v": "10.5",
                "x": False,
            },
        },
    }
)


def _make_candle(
    pair: str = "BTC/USDT",
    timeframe: str = "5m",
    is_closed: bool = True,
) -> CandleEvent:
    """Build a minimal CandleEvent for testing."""
    return CandleEvent(
        pair=pair,
        timeframe=timeframe,
        timestamp_ms=1_700_000_000_000,
        open=49_990.0,
        high=50_100.0,
        low=49_800.0,
        close=50_000.0,
        volume=1.5,
        is_closed=is_closed,
    )


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------


def test_pair_to_binance_symbol() -> None:
    """CCXT pair string is converted to lowercase Binance symbol."""
    assert _pair_to_binance_symbol("BTC/USDT") == "btcusdt"
    assert _pair_to_binance_symbol("ETH/USDT") == "ethusdt"
    assert _pair_to_binance_symbol("SOL/USDT") == "solusdt"
    assert _pair_to_binance_symbol("BNB/BTC") == "bnbbtc"


def test_binance_symbol_to_pair() -> None:
    """Uppercase Binance symbol is converted back to CCXT pair string."""
    assert _binance_symbol_to_pair("BTCUSDT") == "BTC/USDT"
    assert _binance_symbol_to_pair("ETHUSDT") == "ETH/USDT"
    assert _binance_symbol_to_pair("SOLUSDT") == "SOL/USDT"
    assert _binance_symbol_to_pair("BNBBTC") == "BNB/BTC"
    assert _binance_symbol_to_pair("ETHBTC") == "ETH/BTC"


def test_pair_roundtrip() -> None:
    """Converting pair -> symbol -> pair should be idempotent."""
    for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        symbol = _pair_to_binance_symbol(pair).upper()
        assert _binance_symbol_to_pair(symbol) == pair


# ---------------------------------------------------------------------------
# Stream URL construction
# ---------------------------------------------------------------------------


def test_build_stream_url_testnet() -> None:
    """Testnet URL uses the testnet.binance.vision host."""
    url = _build_stream_url(["BTC/USDT"], ["5m"], testnet=True)
    assert url.startswith("wss://testnet.binance.vision/stream")
    assert "btcusdt@kline_5m" in url


def test_build_stream_url_mainnet() -> None:
    """Mainnet URL uses the stream.binance.com host on port 9443."""
    url = _build_stream_url(["BTC/USDT", "ETH/USDT"], ["5m", "15m"], testnet=False)
    assert url.startswith("wss://stream.binance.com:9443/stream")
    assert "btcusdt@kline_5m" in url
    assert "btcusdt@kline_15m" in url
    assert "ethusdt@kline_5m" in url
    assert "ethusdt@kline_15m" in url


def test_build_stream_url_format() -> None:
    """Streams are separated by '/' and appended after '?streams='."""
    url = _build_stream_url(["BTC/USDT"], ["5m", "15m"], testnet=False)
    qs = url.split("?streams=", 1)[1]
    streams = qs.split("/")
    assert "btcusdt@kline_5m" in streams
    assert "btcusdt@kline_15m" in streams


# ---------------------------------------------------------------------------
# Kline message parsing
# ---------------------------------------------------------------------------


async def test_parse_kline_message() -> None:
    """A closed kline message is parsed and published as a CandleEvent."""
    bus = EventBus()
    queue = bus.subscribe("candle")

    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )

    # Call the internal message handler directly — no network needed.
    await feed._handle_message(_SAMPLE_KLINE_MSG)

    assert queue.qsize() == 1
    event: CandleEvent = queue.get_nowait()

    assert isinstance(event, CandleEvent)
    assert event.pair == "BTC/USDT"
    assert event.timeframe == "5m"
    assert event.timestamp_ms == 1234567890000
    assert event.open == 50000.0
    assert event.high == 50100.0
    assert event.low == 49900.0
    assert event.close == 50050.0
    assert event.volume == 10.5
    assert event.is_closed is True


async def test_parse_kline_message_not_closed() -> None:
    """An open (in-progress) kline sets is_closed=False on the event."""
    bus = EventBus()
    queue = bus.subscribe("candle")

    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )

    await feed._handle_message(_SAMPLE_KLINE_MSG_OPEN)

    assert queue.qsize() == 1
    event: CandleEvent = queue.get_nowait()
    assert event.is_closed is False


async def test_parse_invalid_json_is_noop() -> None:
    """Malformed JSON is silently discarded; no event is published."""
    bus = EventBus()
    queue = bus.subscribe("candle")

    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )

    await feed._handle_message("not valid json {{{")
    assert queue.qsize() == 0


async def test_parse_non_kline_event_is_noop() -> None:
    """Messages whose event type is not 'kline' are silently discarded."""
    bus = EventBus()
    queue = bus.subscribe("candle")

    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )

    other_msg = json.dumps({"stream": "btcusdt@trade", "data": {"e": "trade"}})
    await feed._handle_message(other_msg)
    assert queue.qsize() == 0


# ---------------------------------------------------------------------------
# ReconnectManager backoff
# ---------------------------------------------------------------------------


def test_backoff_increases() -> None:
    """Backoff delay roughly doubles with each attempt (ignoring jitter)."""
    manager = ReconnectManager(base_delay=1.0, max_delay=3600.0)

    # Strip jitter by subtracting up to 1 second and checking minimum bound.
    delays = [manager._backoff_delay(attempt) for attempt in range(5)]

    # Each raw delay (before jitter) is base * 2^attempt.
    # With jitter in [0,1), the actual delay is in [base*2^n, base*2^n + 1).
    # Check that consecutive delays are non-decreasing (jitter can cause
    # occasional equal values but not strict regression by more than 1 s).
    for i in range(1, len(delays)):
        assert delays[i] >= delays[i - 1] - 1.0, (
            f"Delay at attempt {i} ({delays[i]:.2f}) regressed more than 1 s "
            f"relative to attempt {i - 1} ({delays[i - 1]:.2f})"
        )

    # Verify the exponential structure by checking the jitter-stripped minimum.
    for attempt in range(5):
        raw_min = 1.0 * (2**attempt)
        actual = manager._backoff_delay(attempt)
        assert actual >= raw_min, (
            f"Attempt {attempt}: expected >= {raw_min}, got {actual:.3f}"
        )


def test_backoff_capped_at_max() -> None:
    """Backoff delay never exceeds max_delay + 1 (the jitter ceiling)."""
    max_delay = 30.0
    manager = ReconnectManager(base_delay=1.0, max_delay=max_delay)

    for attempt in range(20):
        delay = manager._backoff_delay(attempt)
        # Upper bound: max_delay + 1 second of jitter
        assert delay <= max_delay + 1.0, (
            f"Attempt {attempt}: delay {delay:.2f} exceeds cap {max_delay + 1}"
        )


def test_backoff_base_delay_respected() -> None:
    """First retry delay is at least base_delay (before jitter)."""
    manager = ReconnectManager(base_delay=5.0, max_delay=60.0)
    delay = manager._backoff_delay(0)
    # attempt 0: base_delay * 2^0 = base_delay
    assert delay >= 5.0


# ---------------------------------------------------------------------------
# ReconnectManager attempt counter
# ---------------------------------------------------------------------------


async def test_attempt_resets_on_success() -> None:
    """_attempt counter resets to 0 after a successful connection."""
    import websockets.exceptions
    import websockets.frames

    manager = ReconnectManager(max_retries=5, base_delay=0.01, max_delay=0.1)

    # Simulate one failed attempt followed by one successful short-lived connection.
    connection_count = 0

    async def connect_fn() -> MagicMock:
        nonlocal connection_count
        connection_count += 1
        if connection_count == 1:
            raise ConnectionError("simulated failure")

        # Second call: real async generator that yields one message then
        # raises ConnectionClosed (the normal end-of-stream signal for websockets).
        ws = MagicMock()

        async def _aiter(self: MagicMock) -> object:  # type: ignore[override]
            yield '{"stream": "test", "data": {"e": "other"}}'
            # ConnectionClosedOK is a subclass of ConnectionClosed; this is the
            # clean way to signal end-of-stream without triggering the async
            # generator StopAsyncIteration restriction.
            raise websockets.exceptions.ConnectionClosedOK(
                websockets.frames.Close(code=1000, reason="normal"),
                None,
            )

        ws.__aiter__ = _aiter
        return ws  # type: ignore[return-value]

    messages_handled: list[str] = []

    async def message_handler(msg: str) -> None:
        messages_handled.append(msg)
        # After handling one message, signal stop so the loop exits cleanly.
        manager.stop()

    await manager.run(connect_fn=connect_fn, message_handler=message_handler)

    # The manager should have reset _attempt to 0 after the successful connect.
    assert manager._attempt == 0


async def test_is_connected_false_before_start() -> None:
    """is_connected is False before run() is called."""
    manager = ReconnectManager()
    assert manager.is_connected is False


async def test_last_message_time_none_before_first_message() -> None:
    """last_message_time is None until the first message is received."""
    manager = ReconnectManager()
    assert manager.last_message_time is None


# ---------------------------------------------------------------------------
# CandleAggregator
# ---------------------------------------------------------------------------


async def test_aggregator_forwards_closed_candles() -> None:
    """Closed candles are forwarded to the feature store."""
    bus = EventBus()

    # Fake feature store — track calls to update()
    feature_store = AsyncMock()
    feature_store.update = AsyncMock()

    aggregator = CandleAggregator(event_bus=bus, feature_store=feature_store)
    await aggregator.start()

    closed_candle = _make_candle(is_closed=True)
    await bus.publish("candle", closed_candle)

    # Give the consumer task time to process
    await asyncio.sleep(0.05)
    await aggregator.stop()

    feature_store.update.assert_called_once_with(
        pair=closed_candle.pair,
        timeframe=closed_candle.timeframe,
        timestamp_ms=closed_candle.timestamp_ms,
        o=closed_candle.open,
        h=closed_candle.high,
        low=closed_candle.low,
        c=closed_candle.close,
        v=closed_candle.volume,
    )


async def test_aggregator_skips_open_candles() -> None:
    """Open (in-progress) candles are NOT forwarded to the feature store."""
    bus = EventBus()
    feature_store = AsyncMock()
    feature_store.update = AsyncMock()

    aggregator = CandleAggregator(event_bus=bus, feature_store=feature_store)
    await aggregator.start()

    open_candle = _make_candle(is_closed=False)
    await bus.publish("candle", open_candle)

    await asyncio.sleep(0.05)
    await aggregator.stop()

    feature_store.update.assert_not_called()


async def test_aggregator_counts_processed_candles() -> None:
    """_candles_processed increments only for closed candles."""
    bus = EventBus()
    feature_store = AsyncMock()
    feature_store.update = AsyncMock()

    aggregator = CandleAggregator(event_bus=bus, feature_store=feature_store)
    await aggregator.start()

    # Publish 3 closed + 2 open
    for _ in range(3):
        await bus.publish("candle", _make_candle(is_closed=True))
    for _ in range(2):
        await bus.publish("candle", _make_candle(is_closed=False))

    await asyncio.sleep(0.1)
    await aggregator.stop()

    assert aggregator._candles_processed == 3


async def test_aggregator_stop_is_idempotent() -> None:
    """Calling stop() multiple times does not raise."""
    bus = EventBus()
    feature_store = AsyncMock()

    aggregator = CandleAggregator(event_bus=bus, feature_store=feature_store)
    await aggregator.start()
    await aggregator.stop()
    await aggregator.stop()  # second call must be a no-op


async def test_aggregator_start_raises_if_already_running() -> None:
    """Starting an already-running aggregator raises RuntimeError."""
    bus = EventBus()
    feature_store = AsyncMock()

    aggregator = CandleAggregator(event_bus=bus, feature_store=feature_store)
    await aggregator.start()

    try:
        with pytest.raises(RuntimeError, match="already running"):
            await aggregator.start()
    finally:
        await aggregator.stop()


# ---------------------------------------------------------------------------
# BinanceKlineFeed integration (no network)
# ---------------------------------------------------------------------------


async def test_feed_stop_before_start_is_safe() -> None:
    """Calling stop() on a feed that was never started must not raise."""
    bus = EventBus()
    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )
    await feed.stop()  # should be a no-op


async def test_feed_is_connected_false_initially() -> None:
    """is_connected is False before the WebSocket is established."""
    bus = EventBus()
    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )
    assert feed.is_connected is False


async def test_feed_start_raises_if_already_running() -> None:
    """Starting an already-running feed raises RuntimeError."""
    bus = EventBus()

    feed = BinanceKlineFeed(
        event_bus=bus,
        pairs=["BTC/USDT"],
        timeframes=["5m"],
        testnet=True,
    )

    # Patch websockets.connect to prevent real network calls
    with patch("data.feeds.binance_ws.websockets.connect", new_callable=AsyncMock) as mock_ws:
        # Return a mock that blocks forever on iteration so the task stays alive
        ws_instance = MagicMock()

        async def _blocking_aiter(self: MagicMock) -> object:
            await asyncio.sleep(9999)
            return
            yield  # make it an async generator

        ws_instance.__aiter__ = _blocking_aiter
        mock_ws.return_value = ws_instance

        await feed.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await feed.start()
        finally:
            await feed.stop()
