"""EMA crossover with RSI confirmation signal generator.

Pure function implementation — no I/O, no global state mutations.
"""

from __future__ import annotations

import numpy as np

from signals.base import Signal, SignalDirection

# ---------------------------------------------------------------------------
# Private math helpers
# ---------------------------------------------------------------------------


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute the Exponential Moving Average of a 1-D float array.

    The first ``period`` values are seeded with the simple moving average of
    those same ``period`` values (standard EMA initialisation).  Subsequent
    values apply the standard EMA recurrence:

        ema[t] = close[t] * multiplier + ema[t-1] * (1 - multiplier)
        multiplier = 2 / (period + 1)

    Args:
        values: 1-D array of prices (oldest-first).
        period: EMA lookback period.  Must be >= 1 and <= len(values).

    Returns:
        1-D float64 array of the same length as ``values`` containing EMA
        values.  The first ``period - 1`` entries are set to ``nan`` because
        there is not enough history to compute a meaningful value.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    multiplier = 2.0 / (period + 1)
    # Seed with the SMA of the first `period` values.
    result[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        result[i] = values[i] * multiplier + result[i - 1] * (1.0 - multiplier)
    return result


def _rsi(values: np.ndarray, period: int) -> float:
    """Compute the RSI of the *last* value in a price series.

    Uses Wilder's smoothed (EMA-style) average with alpha = 1/period.

    Args:
        values: 1-D array of prices (oldest-first).  Must contain at least
            ``period + 1`` elements.
        period: RSI lookback period.

    Returns:
        RSI as a float in ``[0, 100]``.  Returns ``50.0`` if there are
        insufficient data points or if average loss is zero (infinite RS).
    """
    if len(values) < period + 1:
        return 50.0

    deltas = np.diff(values.astype(np.float64))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple mean of first `period` deltas.
    avg_gain: float = float(np.mean(gains[:period]))
    avg_loss: float = float(np.mean(losses[:period]))

    # Wilder smoothing over remaining deltas.
    alpha = 1.0 / period
    for i in range(period, len(deltas)):
        avg_gain = gains[i] * alpha + avg_gain * (1.0 - alpha)
        avg_loss = losses[i] * alpha + avg_loss * (1.0 - alpha)

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------


class MomentumSignal:
    """EMA crossover + RSI trend-following signal.

    Generates a directional signal by combining two signals:

    1. **EMA crossover** — a fast EMA crossing above the slow EMA suggests
       bullish momentum; crossing below suggests bearish momentum.
    2. **RSI confirmation** — RSI above 50 confirms bullish momentum;
       below 50 confirms bearish momentum.

    Direction rules:

    * ``LONG``  — fast EMA > slow EMA *and* RSI > 50
    * ``SHORT`` — fast EMA < slow EMA *and* RSI < 50
    * ``NEUTRAL`` — any other combination

    Strength is the absolute percentage difference between the two EMAs,
    clamped to ``[0.0, 1.0]``:

        strength = clamp(|fast_ema - slow_ema| / slow_ema, 0, 1)

    Attributes:
        fast_period: Lookback period for the fast EMA.
        slow_period: Lookback period for the slow EMA.
        rsi_period: Lookback period for the RSI.
    """

    _NAME = "momentum_ema_rsi"

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        rsi_period: int = 14,
    ) -> None:
        """Initialise the MomentumSignal generator.

        Args:
            fast_period: Fast EMA period (default 9).
            slow_period: Slow EMA period (default 21).
            rsi_period: RSI period (default 14).

        Raises:
            ValueError: If ``fast_period >= slow_period`` or any period < 1.
        """
        if fast_period < 1 or slow_period < 1 or rsi_period < 1:
            raise ValueError("All periods must be >= 1.")
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})."
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        """Generate a momentum signal from the supplied candle array.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            candles: Numpy structured array with dtype ``CANDLE_DTYPE``,
                ordered oldest-first.

        Returns:
            A :class:`~signals.base.Signal` instance.  Returns ``NEUTRAL``
            when there are not enough candles to compute all indicators.
        """
        min_candles = max(self.fast_period, self.slow_period, self.rsi_period) + 1
        timestamp = int(candles[-1]["timestamp"]) if len(candles) > 0 else 0

        if len(candles) < min_candles:
            return Signal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                indicator_name=self._NAME,
                timestamp=timestamp,
                metadata={
                    "insufficient_candles": float(len(candles)),
                    "min_required": float(min_candles),
                },
            )

        closes: np.ndarray = candles["close"].astype(np.float64)

        fast_ema_arr = _ema(closes, self.fast_period)
        slow_ema_arr = _ema(closes, self.slow_period)
        rsi_val = _rsi(closes, self.rsi_period)

        fast_ema = float(fast_ema_arr[-1])
        slow_ema = float(slow_ema_arr[-1])

        ema_diff_pct = abs(fast_ema - slow_ema) / slow_ema if slow_ema != 0.0 else 0.0
        strength = float(np.clip(ema_diff_pct, 0.0, 1.0))

        if fast_ema > slow_ema and rsi_val > 50.0:
            direction = SignalDirection.LONG
        elif fast_ema < slow_ema and rsi_val < 50.0:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        return Signal(
            pair=pair,
            direction=direction,
            strength=strength,
            indicator_name=self._NAME,
            timestamp=timestamp,
            metadata={
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "rsi": rsi_val,
                "ema_diff_pct": ema_diff_pct,
            },
        )
