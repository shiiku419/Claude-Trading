"""Signal protocol and shared data types for the signal engine.

All signal generators must be pure functions: deterministic, no side effects,
no network calls. Given the same candle array, they must always return the
same Signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Candle dtype — must match data.feature_store._CANDLE_DTYPE exactly.
# ---------------------------------------------------------------------------

CANDLE_DTYPE = np.dtype(
    [
        ("timestamp", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "f8"),
    ]
)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class SignalDirection(StrEnum):
    """Enumeration of possible signal directions."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class Signal:
    """Immutable value object representing a trading signal.

    Attributes:
        pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
        direction: Directional bias of the signal.
        strength: Normalised signal confidence in ``[0.0, 1.0]``.
            ``0.0`` means no conviction; ``1.0`` means maximum conviction.
        indicator_name: Human-readable name of the generating indicator.
        timestamp: Unix timestamp in milliseconds of the *most recent* candle
            that produced this signal.
        metadata: Arbitrary float-valued diagnostic data produced by the
            generator (e.g. EMA values, RSI reading).
    """

    pair: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    indicator_name: str
    timestamp: int  # Unix ms
    metadata: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SignalGenerator(Protocol):
    """Structural protocol that all signal generators must satisfy.

    Implementations must be stateless with respect to market data: all
    computation must derive solely from the ``candles`` argument.
    """

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        """Generate a signal from the supplied candle array.

        Args:
            pair: Trading pair symbol.
            candles: Numpy structured array with dtype :data:`CANDLE_DTYPE`,
                ordered oldest-first (index 0 = oldest, index -1 = newest).

        Returns:
            A :class:`Signal` instance.  Must be deterministic: identical
            ``candles`` input always produces an identical ``Signal`` output.
        """
        ...
