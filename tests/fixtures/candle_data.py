"""Synthetic candle data generators for unit tests.

All functions return numpy structured arrays with ``CANDLE_DTYPE`` ordered
oldest-first (index 0 = oldest, index -1 = newest).  Data is fully
deterministic — no random state, no I/O.
"""

from __future__ import annotations

import numpy as np

from signals.base import CANDLE_DTYPE

# Base timestamp: 2024-01-01 00:00:00 UTC in milliseconds.
_BASE_TS_MS = 1_704_067_200_000
_ONE_MINUTE_MS = 60_000


def make_candles(
    n: int,
    *,
    start_price: float = 100.0,
    price_delta: float = 0.0,
    volume: float = 1_000.0,
    spread_pct: float = 0.005,
    start_ts_ms: int = _BASE_TS_MS,
    interval_ms: int = _ONE_MINUTE_MS,
) -> np.ndarray:
    """Generate synthetic candle data.

    Each candle has:

    * ``open``  = previous close (or ``start_price`` for first candle)
    * ``close`` = open + price_delta
    * ``high``  = max(open, close) * (1 + spread_pct)
    * ``low``   = min(open, close) * (1 - spread_pct)
    * ``volume`` = constant ``volume``
    * ``timestamp`` = linearly spaced starting from ``start_ts_ms``

    Args:
        n: Number of candles to generate.
        start_price: Opening price of the first candle.
        price_delta: Fixed price change applied per candle.  Positive values
            produce an uptrend; negative values produce a downtrend; zero
            produces flat candles.
        volume: Volume assigned to every candle.
        spread_pct: Fractional spread used to compute high and low.
        start_ts_ms: Timestamp of the first candle in Unix milliseconds.
        interval_ms: Time between successive candles in milliseconds.

    Returns:
        1-D numpy structured array of length ``n`` with dtype
        :data:`~signals.base.CANDLE_DTYPE`.
    """
    arr = np.zeros(n, dtype=CANDLE_DTYPE)
    price = start_price
    for i in range(n):
        open_price = price
        close_price = price + price_delta
        high_price = max(open_price, close_price) * (1.0 + spread_pct)
        low_price = min(open_price, close_price) * (1.0 - spread_pct)
        arr[i]["timestamp"] = start_ts_ms + i * interval_ms
        arr[i]["open"] = open_price
        arr[i]["high"] = high_price
        arr[i]["low"] = low_price
        arr[i]["close"] = close_price
        arr[i]["volume"] = volume
        price = close_price
    return arr


def uptrend_candles(n: int = 100) -> np.ndarray:
    """Generate candles with steadily increasing prices.

    Each candle closes 1.0 higher than it opens, producing a clear uptrend
    that should drive fast EMA above slow EMA and RSI above 50.

    Args:
        n: Number of candles (default 100).

    Returns:
        Uptrending candle array of length ``n``.
    """
    return make_candles(n, start_price=1_000.0, price_delta=1.0, volume=1_000.0)


def downtrend_candles(n: int = 100) -> np.ndarray:
    """Generate candles with steadily decreasing prices.

    Each candle closes 1.0 lower than it opens, producing a clear downtrend
    that should drive fast EMA below slow EMA and RSI below 50.

    Args:
        n: Number of candles (default 100).

    Returns:
        Downtrending candle array of length ``n``.
    """
    return make_candles(n, start_price=1_000.0, price_delta=-1.0, volume=1_000.0)


def ranging_candles(n: int = 100) -> np.ndarray:
    """Generate candles oscillating around a fixed mean price.

    Prices alternate between +0.5 and -0.5 moves, keeping the EMAs and RSI
    close to neutral values.

    Args:
        n: Number of candles (default 100).

    Returns:
        Ranging candle array of length ``n``.
    """
    arr = np.zeros(n, dtype=CANDLE_DTYPE)
    price = 1_000.0
    spread_pct = 0.002
    for i in range(n):
        delta = 0.5 if i % 2 == 0 else -0.5
        open_price = price
        close_price = price + delta
        high_price = max(open_price, close_price) * (1.0 + spread_pct)
        low_price = min(open_price, close_price) * (1.0 - spread_pct)
        arr[i]["timestamp"] = _BASE_TS_MS + i * _ONE_MINUTE_MS
        arr[i]["open"] = open_price
        arr[i]["high"] = high_price
        arr[i]["low"] = low_price
        arr[i]["close"] = close_price
        arr[i]["volume"] = 1_000.0
        price = close_price
    return arr


def volume_spike_candles(n: int = 100) -> np.ndarray:
    """Generate normal-volume candles with a volume spike on the last candle.

    The last candle has volume 10× the baseline and a bullish body
    (close > open) to produce a LONG signal from
    :class:`~signals.volume_spike.VolumeSpikeSignal`.

    Args:
        n: Total number of candles (default 100).  Must be >= 2.

    Returns:
        Candle array of length ``n``.
    """
    if n < 2:
        raise ValueError("n must be >= 2 to include a spike candle.")

    arr = make_candles(n, start_price=1_000.0, price_delta=0.0, volume=500.0)

    # Overwrite the final candle with spike volume and a bullish body.
    last_open = float(arr[-1]["open"])
    last_close = last_open + 5.0  # bullish body
    arr[-1]["close"] = last_close
    arr[-1]["high"] = last_close * 1.005
    arr[-1]["volume"] = 500.0 * 10.0  # 10x normal = well above 2.5x threshold

    return arr
