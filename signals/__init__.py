"""Signal engine for the crypto auto-trading system.

All signal generators are pure functions: deterministic, no side effects,
no network calls.  Same candle input always produces the same signal output.

Public API
----------
- :class:`~signals.base.Signal`
- :class:`~signals.base.SignalDirection`
- :class:`~signals.base.SignalGenerator`
- :class:`~signals.momentum.MomentumSignal`
- :class:`~signals.vwap.VWAPSignal`
- :class:`~signals.volume_spike.VolumeSpikeSignal`
- :class:`~signals.composite.CompositeSignal`
"""

from signals.base import CANDLE_DTYPE, Signal, SignalDirection, SignalGenerator
from signals.composite import CompositeSignal
from signals.momentum import MomentumSignal
from signals.volume_spike import VolumeSpikeSignal
from signals.vwap import VWAPSignal

__all__ = [
    "CANDLE_DTYPE",
    "Signal",
    "SignalDirection",
    "SignalGenerator",
    "MomentumSignal",
    "VWAPSignal",
    "VolumeSpikeSignal",
    "CompositeSignal",
]
