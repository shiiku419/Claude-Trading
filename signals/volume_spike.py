"""Volume spike detection signal generator.

Pure function implementation — no I/O, no global state mutations.
"""

from __future__ import annotations

import numpy as np

from signals.base import Signal, SignalDirection


class VolumeSpikeSignal:
    """Volume spike detection as a confirmation signal.

    Detects abnormally high volume on the most recent candle by comparing it
    against the rolling mean of the preceding ``lookback`` candles.  When a
    spike is detected, the direction is determined by price action on that
    same candle:

    * ``LONG``  — close > open (bullish candle body)
    * ``SHORT`` — close < open (bearish candle body)
    * ``NEUTRAL`` — close == open (doji; treated as no conviction)

    No spike (volume <= ``spike_multiplier * avg_volume``) always returns
    ``NEUTRAL``.

    Strength formula:

        strength = clamp(current_volume / (spike_multiplier * avg_volume), 0, 1)

    The strength saturates at ``1.0`` when the current volume equals exactly
    the spike threshold.  Any volume above that threshold also maps to
    ``1.0``.

    Attributes:
        lookback: Number of preceding candles used to compute the rolling
            average volume (default 20).
        spike_multiplier: Multiplier applied to the average volume to define
            the spike threshold (default 2.5).
    """

    _NAME = "volume_spike"

    def __init__(
        self,
        lookback: int = 20,
        spike_multiplier: float = 2.5,
    ) -> None:
        """Initialise the VolumeSpikeSignal generator.

        Args:
            lookback: Rolling window size for average volume (default 20).
            spike_multiplier: Volume threshold multiplier (default 2.5).

        Raises:
            ValueError: If ``lookback < 1`` or ``spike_multiplier <= 0``.
        """
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if spike_multiplier <= 0.0:
            raise ValueError("spike_multiplier must be positive.")
        self.lookback = lookback
        self.spike_multiplier = spike_multiplier

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        """Generate a volume spike signal from the supplied candle array.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            candles: Numpy structured array with dtype ``CANDLE_DTYPE``,
                ordered oldest-first.

        Returns:
            A :class:`~signals.base.Signal` instance.  Returns ``NEUTRAL``
            when there are insufficient candles, average volume is zero, or
            no spike is detected.
        """
        min_candles = self.lookback + 1
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

        volume = candles["volume"].astype(np.float64)
        # Rolling window: the `lookback` candles immediately before the last.
        window_volume = volume[-min_candles:-1]
        avg_volume = float(np.mean(window_volume))

        if avg_volume == 0.0:
            return Signal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                indicator_name=self._NAME,
                timestamp=timestamp,
                metadata={"zero_avg_volume": 1.0},
            )

        current_volume = float(volume[-1])
        spike_threshold = self.spike_multiplier * avg_volume
        volume_ratio = current_volume / avg_volume

        strength = float(np.clip(current_volume / spike_threshold, 0.0, 1.0))

        last_open = float(candles[-1]["open"])
        last_close = float(candles[-1]["close"])
        if last_close > last_open:
            price_direction = 1.0
        elif last_close < last_open:
            price_direction = -1.0
        else:
            price_direction = 0.0

        if current_volume <= spike_threshold:
            direction = SignalDirection.NEUTRAL
        elif last_close > last_open:
            direction = SignalDirection.LONG
        elif last_close < last_open:
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
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "price_direction": price_direction,
            },
        )
