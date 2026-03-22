"""In-memory ring-buffer + Redis feature store for market data.

Each trading pair/timeframe combination owns a fixed-size numpy structured
array that behaves as a circular buffer.  Real-time summaries (last price,
last volume, last update timestamp) are mirrored to Redis so that other
processes can read them without holding a reference to this object.
"""

from __future__ import annotations

import numpy as np
import structlog
from redis import asyncio as aioredis

log: structlog.BoundLogger = structlog.get_logger(__name__)

# dtype shared by every candle ring buffer
_CANDLE_DTYPE = np.dtype(
    [
        ("timestamp", np.int64),
        ("open", np.float64),
        ("high", np.float64),
        ("low", np.float64),
        ("close", np.float64),
        ("volume", np.float64),
    ]
)


class FeatureStore:
    """Thread-safe (asyncio) in-memory + Redis feature store.

    The in-memory store uses one numpy structured array per
    ``{pair}:{timeframe}`` key, written as a ring buffer so that memory
    consumption is bounded to ``max_candles`` entries per key regardless of
    how long the bot runs.

    Redis is used as a lightweight read-through cache of the most recent
    scalar features so that other services (e.g., a monitoring dashboard)
    can query current state without needing a direct reference to this object.

    Attributes:
        _buffers: Mapping of ``"{pair}:{timeframe}"`` to a numpy structured
            array with fields: timestamp (int64), open, high, low, close,
            volume (all float64).
        _buffer_positions: Current write head per buffer key.  Incremented
            modulo ``_max_candles`` after each write.
        _buffer_counts: Number of candles written to each buffer (capped at
            ``_max_candles`` so we can detect partially-filled buffers).
        _redis: Lazy-initialised async Redis connection.
        _redis_url: Connection string passed to the constructor.
        _max_candles: Maximum ring-buffer length; shared across all keys.
    """

    def __init__(self, redis_url: str, max_candles: int = 500) -> None:
        """Initialise the feature store.

        Args:
            redis_url: Redis connection URL, e.g. ``"redis://localhost:6379/0"``.
            max_candles: Maximum number of candles retained per
                pair/timeframe combination.
        """
        self._redis_url: str = redis_url
        self._max_candles: int = max_candles
        self._buffers: dict[str, np.ndarray] = {}
        self._buffer_positions: dict[str, int] = {}
        self._buffer_counts: dict[str, int] = {}
        self._redis: aioredis.Redis | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection pool.

        Safe to call multiple times — subsequent calls are no-ops if the
        connection is already open.
        """
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            log.info("feature_store.connected", redis_url=self._redis_url)

    async def close(self) -> None:
        """Close the Redis connection and free resources."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            log.info("feature_store.closed")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def update(  # noqa: PLR0913
        self,
        pair: str,
        timeframe: str,
        timestamp_ms: int,
        o: float,
        h: float,
        low: float,
        c: float,
        v: float,
    ) -> None:
        """Ingest one OHLCV candle into the ring buffer and update Redis.

        If no buffer exists for the given ``pair``/``timeframe`` combination a
        new zero-filled array of ``max_candles`` rows is created on first call.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            timeframe: Candle timeframe string, e.g. ``"1m"``.
            timestamp_ms: Candle open timestamp in milliseconds (UTC).
            o: Opening price.
            h: Highest price.
            low: Lowest price.
            c: Closing price.
            v: Base-asset volume.
        """
        key = f"{pair}:{timeframe}"

        # Initialise buffer on first write for this key.
        if key not in self._buffers:
            self._buffers[key] = np.zeros(self._max_candles, dtype=_CANDLE_DTYPE)
            self._buffer_positions[key] = 0
            self._buffer_counts[key] = 0
            log.debug("feature_store.buffer_created", key=key, max_candles=self._max_candles)

        pos = self._buffer_positions[key]
        buf = self._buffers[key]

        buf[pos]["timestamp"] = timestamp_ms
        buf[pos]["open"] = o
        buf[pos]["high"] = h
        buf[pos]["low"] = low
        buf[pos]["close"] = c
        buf[pos]["volume"] = v

        self._buffer_positions[key] = (pos + 1) % self._max_candles
        if self._buffer_counts[key] < self._max_candles:
            self._buffer_counts[key] += 1

        log.debug(
            "feature_store.updated",
            key=key,
            pos=pos,
            count=self._buffer_counts[key],
            close=c,
        )

        # Mirror scalars to Redis for cross-process visibility.
        if self._redis is not None:
            try:
                await self._redis.hset(
                    f"features:{pair}",
                    mapping={
                        "last_price": str(c),
                        "last_volume": str(v),
                        "last_update_ms": str(timestamp_ms),
                    },
                )
            except Exception:
                log.warning("feature_store.redis_write_failed", key=key, exc_info=True)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_candles(self, pair: str, timeframe: str, n: int) -> np.ndarray | None:
        """Return the *n* most recent candles for a given pair and timeframe.

        The ring buffer is traversed in chronological order so index 0 of the
        returned array is the *oldest* of the *n* candles and index ``n-1``
        is the most recent.

        Args:
            pair: Trading pair symbol.
            timeframe: Candle timeframe string.
            n: Number of candles to retrieve.  Must be positive and
               ``<= max_candles``.

        Returns:
            A **copy** of the last *n* candles as a numpy structured array,
            or ``None`` if the buffer does not exist or contains fewer than
            *n* valid (non-zero) candles.
        """
        key = f"{pair}:{timeframe}"

        if key not in self._buffers:
            return None

        count = self._buffer_counts[key]
        if count < n:
            return None

        buf = self._buffers[key]
        pos = self._buffer_positions[key]  # next write position = oldest slot when full
        size = self._max_candles

        if count < size:
            # Buffer not yet full: valid data lives in [0, count).
            # The last n entries are buf[count-n : count].
            start = count - n
            result = buf[start:count].copy()
        else:
            # Buffer is full: oldest data starts at `pos`.
            # Concatenate the two halves to get chronological order, then
            # take the last n elements.
            ordered = np.concatenate([buf[pos:], buf[:pos]])
            result = ordered[size - n :].copy()

        return result

    def get_summary(self) -> dict[str, object]:
        """Return a JSON-serialisable market-data summary for all tracked pairs.

        Iterates over all buffered pair/timeframe keys and collects the most
        recent OHLCV values.  Pairs/timeframes with no data yet are skipped.

        Returns:
            A dict with a ``"pairs"`` key mapping each ``"{pair}:{timeframe}"``
            key to its latest close, volume, and candle count.  If no data is
            available, the pairs value is ``{"status": "no_data_yet"}``.
        """
        summary: dict[str, object] = {"note": "MVP summary from in-memory ring buffers"}
        pairs_data: dict[str, object] = {}

        for key, buf in self._buffers.items():
            count = self._buffer_counts.get(key, 0)
            if count == 0:
                continue

            pos = self._buffer_positions[key]
            last_idx = (pos - 1) % self._max_candles
            last = buf[last_idx]

            pairs_data[key] = {
                "last_close": float(last["close"]),
                "last_volume": float(last["volume"]),
                "candles_available": count,
            }

        summary["pairs"] = pairs_data if pairs_data else {"status": "no_data_yet"}
        return summary

    async def get_features(self, pair: str) -> dict[str, str] | None:
        """Read current scalar features for *pair* from Redis.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.

        Returns:
            A dict mapping feature name to string value
            (e.g. ``{"last_price": "45000.0", ...}``),
            or ``None`` if no data has been written for *pair* yet or Redis
            is not connected.
        """
        if self._redis is None:
            log.warning("feature_store.get_features_no_redis", pair=pair)
            return None

        try:
            data: dict[str, str] = await self._redis.hgetall(f"features:{pair}")
        except Exception:
            log.warning("feature_store.redis_read_failed", pair=pair, exc_info=True)
            return None

        return data if data else None
