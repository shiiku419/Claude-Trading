"""Unit tests for :class:`data.feature_store.FeatureStore`.

These tests exercise only the in-memory ring-buffer logic.  All Redis I/O is
skipped because ``connect()`` is never called (``_redis`` stays ``None``),
which causes the Redis write in ``update()`` to be silently skipped and
``get_features()`` to return ``None``.
"""

from __future__ import annotations

import pytest

from data.feature_store import FeatureStore

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fs() -> FeatureStore:
    """Return a small (10-candle) FeatureStore with no Redis connection."""
    return FeatureStore(redis_url="redis://localhost:6379/0", max_candles=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _write(store: FeatureStore, start: int, count: int) -> None:
    """Write *count* candles starting from sequence number *start*."""
    for i in range(start, start + count):
        await store.update(
            pair="BTC/USDT",
            timeframe="1m",
            timestamp_ms=i * 60_000,
            o=1.0,
            h=1.1,
            low=0.9,
            c=float(i),
            v=10.0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_buffer_returns_none(fs: FeatureStore) -> None:
    """Before any data is written, ``get_candles`` must return ``None``."""
    assert fs.get_candles("BTC/USDT", "1m", 5) is None


def test_unknown_pair_returns_none(fs: FeatureStore) -> None:
    """Querying a pair that has never been written returns ``None``."""
    assert fs.get_candles("ETH/USDT", "1m", 1) is None


async def test_insufficient_data_returns_none(fs: FeatureStore) -> None:
    """Requesting more candles than have been written returns ``None``."""
    await _write(fs, 0, 3)
    assert fs.get_candles("BTC/USDT", "1m", 5) is None


async def test_exact_count_returned(fs: FeatureStore) -> None:
    """Requesting exactly the number of written candles succeeds."""
    await _write(fs, 0, 6)
    candles = fs.get_candles("BTC/USDT", "1m", 6)
    assert candles is not None
    assert len(candles) == 6


async def test_partial_read(fs: FeatureStore) -> None:
    """Requesting fewer candles than written returns the most recent ones."""
    await _write(fs, 0, 8)
    candles = fs.get_candles("BTC/USDT", "1m", 3)
    assert candles is not None
    assert len(candles) == 3
    # Most recent close should be 7 (the 8th candle, 0-indexed)
    assert float(candles[-1]["close"]) == pytest.approx(7.0)


async def test_chronological_order(fs: FeatureStore) -> None:
    """Returned candles must be in ascending (oldest-first) order."""
    await _write(fs, 0, 5)
    candles = fs.get_candles("BTC/USDT", "1m", 5)
    assert candles is not None
    closes = [float(c["close"]) for c in candles]
    assert closes == sorted(closes), "Candles must be in chronological order"


async def test_ring_buffer_wrap_around(fs: FeatureStore) -> None:
    """Writing more than max_candles wraps the ring buffer correctly."""
    # Write 16 candles into a 10-slot buffer (6-candle overflow)
    await _write(fs, 0, 16)
    candles = fs.get_candles("BTC/USDT", "1m", 5)
    assert candles is not None
    # Most recent close should be the last written value (15.0)
    assert float(candles[-1]["close"]) == pytest.approx(15.0)
    # Oldest of the 5 should be 11.0 (16 total, last 5 are 11-15)
    assert float(candles[0]["close"]) == pytest.approx(11.0)


async def test_ring_buffer_full_read_after_wrap(fs: FeatureStore) -> None:
    """Reading all max_candles after an overflow returns the last window."""
    await _write(fs, 0, 25)  # 25 into a 10-slot buffer
    candles = fs.get_candles("BTC/USDT", "1m", 10)
    assert candles is not None
    assert len(candles) == 10
    closes = [float(c["close"]) for c in candles]
    # Last 10 candles written are indices 15..24
    expected = [float(i) for i in range(15, 25)]
    assert closes == pytest.approx(expected)


async def test_returns_copy_not_view(fs: FeatureStore) -> None:
    """Mutating the returned array must not affect the internal ring buffer."""
    await _write(fs, 0, 5)
    c = fs.get_candles("BTC/USDT", "1m", 3)
    assert c is not None
    original_close = float(c[-1]["close"])
    c[-1]["close"] = 99_999.0  # mutate the copy

    c2 = fs.get_candles("BTC/USDT", "1m", 3)
    assert c2 is not None
    assert float(c2[-1]["close"]) == pytest.approx(original_close)


async def test_multiple_pairs_are_isolated(fs: FeatureStore) -> None:
    """Different pair/timeframe combinations use independent buffers."""
    for i in range(5):
        await fs.update("BTC/USDT", "1m", i * 60_000, 1.0, 1.1, 0.9, float(i), 10.0)
        await fs.update("ETH/USDT", "1m", i * 60_000, 2.0, 2.2, 1.8, float(i + 100), 5.0)

    btc = fs.get_candles("BTC/USDT", "1m", 5)
    eth = fs.get_candles("ETH/USDT", "1m", 5)

    assert btc is not None and eth is not None
    assert float(btc[-1]["close"]) == pytest.approx(4.0)
    assert float(eth[-1]["close"]) == pytest.approx(104.0)


async def test_numpy_dtype_fields(fs: FeatureStore) -> None:
    """Returned array has the expected structured dtype fields."""
    await _write(fs, 0, 3)
    candles = fs.get_candles("BTC/USDT", "1m", 1)
    assert candles is not None
    assert set(candles.dtype.names) == {"timestamp", "open", "high", "low", "close", "volume"}


async def test_get_features_without_redis_returns_none(fs: FeatureStore) -> None:
    """``get_features`` must return ``None`` when Redis is not connected."""
    result = await fs.get_features("BTC/USDT")
    assert result is None
