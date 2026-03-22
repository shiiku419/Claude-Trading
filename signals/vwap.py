"""VWAP deviation mean-reversion signal generator.

Pure function implementation — no I/O, no global state mutations.
"""

from __future__ import annotations

import numpy as np

from signals.base import Signal, SignalDirection


class VWAPSignal:
    """VWAP deviation signal for mean-reversion trading.

    Computes the session VWAP using all candles in the supplied array, then
    measures how far the current price deviates from VWAP in units of the
    price standard deviation around VWAP.

    VWAP formula (session-based, using all supplied candles):

        typical_price = (high + low + close) / 3
        vwap = cumsum(typical_price * volume) / cumsum(volume)

    Standard deviation is computed as the population std-dev of
    ``(typical_price - vwap)`` over all candles.

    Direction rules:

    * ``SHORT`` — current price > VWAP + threshold * stddev  (overbought)
    * ``LONG``  — current price < VWAP - threshold * stddev  (oversold)
    * ``NEUTRAL`` — price within the band

    Strength:

        strength = clamp(|price - vwap| / (threshold * stddev), 0, 1)

    Attributes:
        deviation_threshold: Number of standard deviations that define the
            signal band (default 2.0).
    """

    _NAME = "vwap_deviation"
    _MIN_CANDLES = 20

    def __init__(self, deviation_threshold: float = 2.0) -> None:
        """Initialise the VWAPSignal generator.

        Args:
            deviation_threshold: Standard-deviation multiplier that sets the
                overbought/oversold boundary (default 2.0).

        Raises:
            ValueError: If ``deviation_threshold`` is not positive.
        """
        if deviation_threshold <= 0.0:
            raise ValueError("deviation_threshold must be positive.")
        self.deviation_threshold = deviation_threshold

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        """Generate a VWAP deviation signal from the supplied candle array.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            candles: Numpy structured array with dtype ``CANDLE_DTYPE``,
                ordered oldest-first.

        Returns:
            A :class:`~signals.base.Signal` instance.  Returns ``NEUTRAL``
            when there are insufficient candles, zero total volume, or the
            price standard deviation around VWAP is zero.
        """
        timestamp = int(candles[-1]["timestamp"]) if len(candles) > 0 else 0

        if len(candles) < self._MIN_CANDLES:
            return Signal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                indicator_name=self._NAME,
                timestamp=timestamp,
                metadata={
                    "insufficient_candles": float(len(candles)),
                    "min_required": float(self._MIN_CANDLES),
                },
            )

        high = candles["high"].astype(np.float64)
        low = candles["low"].astype(np.float64)
        close = candles["close"].astype(np.float64)
        volume = candles["volume"].astype(np.float64)

        total_volume = float(np.sum(volume))
        if total_volume == 0.0:
            return Signal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                indicator_name=self._NAME,
                timestamp=timestamp,
                metadata={"zero_volume": 1.0},
            )

        typical_price = (high + low + close) / 3.0
        vwap = float(np.sum(typical_price * volume) / total_volume)

        deviations = typical_price - vwap
        stddev = float(np.std(deviations))

        current_price = float(close[-1])
        deviation = current_price - vwap
        z_score = deviation / stddev if stddev > 0.0 else 0.0

        band = self.deviation_threshold * stddev
        if stddev == 0.0 or band == 0.0:
            return Signal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                indicator_name=self._NAME,
                timestamp=timestamp,
                metadata={
                    "vwap": vwap,
                    "price": current_price,
                    "deviation": deviation,
                    "stddev": stddev,
                    "z_score": z_score,
                },
            )

        strength = float(np.clip(abs(deviation) / band, 0.0, 1.0))

        if current_price > vwap + band:
            direction = SignalDirection.SHORT
        elif current_price < vwap - band:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.NEUTRAL

        return Signal(
            pair=pair,
            direction=direction,
            strength=strength,
            indicator_name=self._NAME,
            timestamp=timestamp,
            metadata={
                "vwap": vwap,
                "price": current_price,
                "deviation": deviation,
                "stddev": stddev,
                "z_score": z_score,
            },
        )
