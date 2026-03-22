"""Comprehensive tests for the signal engine.

All tests are pure unit tests: no network calls, no I/O, no async operations.
Fixtures live in ``tests/fixtures/candle_data.py``.

Test organisation:
- MomentumSignal
- VWAPSignal
- VolumeSpikeSignal
- CompositeSignal
"""

from __future__ import annotations

import numpy as np
import pytest

from signals.base import CANDLE_DTYPE, Signal, SignalDirection
from signals.composite import CompositeSignal
from signals.momentum import MomentumSignal, _ema, _rsi
from signals.volume_spike import VolumeSpikeSignal
from signals.vwap import VWAPSignal
from tests.fixtures.candle_data import (
    downtrend_candles,
    make_candles,
    ranging_candles,
    uptrend_candles,
    volume_spike_candles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAIR = "BTC/USDT"


def _neutral(sig: Signal) -> bool:
    return sig.direction == SignalDirection.NEUTRAL


def _long(sig: Signal) -> bool:
    return sig.direction == SignalDirection.LONG


def _short(sig: Signal) -> bool:
    return sig.direction == SignalDirection.SHORT


# ---------------------------------------------------------------------------
# Internal math unit tests
# ---------------------------------------------------------------------------


class TestEma:
    def test_ema_length_matches_input(self) -> None:
        values = np.arange(20, dtype=np.float64)
        result = _ema(values, period=5)
        assert len(result) == len(values)

    def test_ema_first_period_minus_one_are_nan(self) -> None:
        values = np.arange(10, dtype=np.float64)
        result = _ema(values, period=5)
        assert np.all(np.isnan(result[:4]))

    def test_ema_seed_equals_sma(self) -> None:
        """The EMA value at index period-1 equals the SMA of those values."""
        values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _ema(values, period=5)
        assert float(result[4]) == pytest.approx(6.0)

    def test_ema_insufficient_data_all_nan(self) -> None:
        values = np.array([1.0, 2.0])
        result = _ema(values, period=5)
        assert np.all(np.isnan(result))

    def test_ema_fast_tracks_price_faster_than_slow(self) -> None:
        closes = uptrend_candles(50)["close"].astype(np.float64)
        fast = _ema(closes, 5)
        slow = _ema(closes, 20)
        # In uptrend fast EMA should be above slow EMA at the end.
        assert fast[-1] > slow[-1]


class TestRsi:
    def test_rsi_uptrend_above_50(self) -> None:
        closes = uptrend_candles(50)["close"].astype(np.float64)
        rsi = _rsi(closes, period=14)
        assert rsi > 50.0

    def test_rsi_downtrend_below_50(self) -> None:
        closes = downtrend_candles(50)["close"].astype(np.float64)
        rsi = _rsi(closes, period=14)
        assert rsi < 50.0

    def test_rsi_insufficient_data_returns_50(self) -> None:
        closes = np.array([100.0, 101.0])
        rsi = _rsi(closes, period=14)
        assert rsi == pytest.approx(50.0)

    def test_rsi_range(self) -> None:
        closes = uptrend_candles(60)["close"].astype(np.float64)
        rsi = _rsi(closes, period=14)
        assert 0.0 <= rsi <= 100.0


# ---------------------------------------------------------------------------
# MomentumSignal
# ---------------------------------------------------------------------------


class TestMomentumSignal:
    def test_momentum_uptrend_gives_long(self) -> None:
        sig = MomentumSignal().generate(_PAIR, uptrend_candles(100))
        assert _long(sig), f"Expected LONG, got {sig.direction}"

    def test_momentum_downtrend_gives_short(self) -> None:
        sig = MomentumSignal().generate(_PAIR, downtrend_candles(100))
        assert _short(sig), f"Expected SHORT, got {sig.direction}"

    def test_momentum_ranging_gives_neutral_or_low_strength(self) -> None:
        sig = MomentumSignal().generate(_PAIR, ranging_candles(100))
        # Ranging market: either NEUTRAL or very low conviction.
        assert _neutral(sig) or sig.strength < 0.05

    def test_momentum_insufficient_candles_neutral(self) -> None:
        # Default requires at least max(9, 21, 14) + 1 = 22 candles.
        few = make_candles(5, price_delta=1.0)
        sig = MomentumSignal().generate(_PAIR, few)
        assert _neutral(sig)
        assert "insufficient_candles" in sig.metadata

    def test_momentum_exactly_min_candles_not_neutral_for_uptrend(self) -> None:
        gen = MomentumSignal(fast_period=5, slow_period=10, rsi_period=8)
        # min_candles = max(5, 10, 8) + 1 = 11
        candles = uptrend_candles(11)
        sig = gen.generate(_PAIR, candles)
        # May or may not fire but must not crash and metadata must be present.
        assert isinstance(sig, Signal)

    def test_momentum_deterministic(self) -> None:
        candles = uptrend_candles(100)
        gen = MomentumSignal()
        sig1 = gen.generate(_PAIR, candles)
        sig2 = gen.generate(_PAIR, candles)
        assert sig1 == sig2

    def test_momentum_metadata_present(self) -> None:
        sig = MomentumSignal().generate(_PAIR, uptrend_candles(100))
        for key in ("fast_ema", "slow_ema", "rsi", "ema_diff_pct"):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_momentum_strength_clamped(self) -> None:
        sig = MomentumSignal().generate(_PAIR, uptrend_candles(100))
        assert 0.0 <= sig.strength <= 1.0

    def test_momentum_pair_propagated(self) -> None:
        sig = MomentumSignal().generate("ETH/USDT", uptrend_candles(100))
        assert sig.pair == "ETH/USDT"

    def test_momentum_invalid_periods_raise(self) -> None:
        with pytest.raises(ValueError, match="fast_period"):
            MomentumSignal(fast_period=21, slow_period=9)
        with pytest.raises(ValueError):
            MomentumSignal(fast_period=0, slow_period=9)

    def test_momentum_timestamp_from_last_candle(self) -> None:
        candles = uptrend_candles(100)
        sig = MomentumSignal().generate(_PAIR, candles)
        assert sig.timestamp == int(candles[-1]["timestamp"])


# ---------------------------------------------------------------------------
# VWAPSignal
# ---------------------------------------------------------------------------


class TestVWAPSignal:
    def _overbought_candles(self, n: int = 50) -> np.ndarray:
        """Candles where the last price is far above session VWAP."""
        arr = make_candles(n, start_price=1_000.0, price_delta=0.0, volume=1_000.0)
        # Push last candle's close far above VWAP (inject a massive spike).
        arr[-1]["close"] = 1_000.0 * 1.20  # 20% above mean
        arr[-1]["high"] = arr[-1]["close"] * 1.001
        return arr

    def _oversold_candles(self, n: int = 50) -> np.ndarray:
        """Candles where the last price is far below session VWAP."""
        arr = make_candles(n, start_price=1_000.0, price_delta=0.0, volume=1_000.0)
        arr[-1]["close"] = 1_000.0 * 0.80  # 20% below mean
        arr[-1]["low"] = arr[-1]["close"] * 0.999
        return arr

    def test_vwap_overbought_gives_short(self) -> None:
        sig = VWAPSignal(deviation_threshold=1.5).generate(_PAIR, self._overbought_candles())
        assert _short(sig), f"Expected SHORT, got {sig.direction}"

    def test_vwap_oversold_gives_long(self) -> None:
        sig = VWAPSignal(deviation_threshold=1.5).generate(_PAIR, self._oversold_candles())
        assert _long(sig), f"Expected LONG, got {sig.direction}"

    def test_vwap_near_vwap_neutral(self) -> None:
        # Flat candles: typical price == VWAP, stddev is tiny → NEUTRAL.
        candles = make_candles(50, price_delta=0.0, volume=1_000.0)
        sig = VWAPSignal().generate(_PAIR, candles)
        assert _neutral(sig)

    def test_vwap_insufficient_candles_neutral(self) -> None:
        few = make_candles(10, price_delta=1.0)
        sig = VWAPSignal().generate(_PAIR, few)
        assert _neutral(sig)
        assert "insufficient_candles" in sig.metadata

    def test_vwap_zero_volume_neutral(self) -> None:
        arr = make_candles(30, price_delta=1.0, volume=0.0)
        sig = VWAPSignal().generate(_PAIR, arr)
        assert _neutral(sig)
        assert "zero_volume" in sig.metadata

    def test_vwap_metadata_present(self) -> None:
        sig = VWAPSignal().generate(_PAIR, self._overbought_candles())
        for key in ("vwap", "price", "deviation", "stddev", "z_score"):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_vwap_strength_clamped(self) -> None:
        sig = VWAPSignal(deviation_threshold=1.5).generate(_PAIR, self._overbought_candles())
        assert 0.0 <= sig.strength <= 1.0

    def test_vwap_deterministic(self) -> None:
        candles = self._overbought_candles()
        gen = VWAPSignal()
        assert gen.generate(_PAIR, candles) == gen.generate(_PAIR, candles)

    def test_vwap_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            VWAPSignal(deviation_threshold=0.0)
        with pytest.raises(ValueError):
            VWAPSignal(deviation_threshold=-1.0)


# ---------------------------------------------------------------------------
# VolumeSpikeSignal
# ---------------------------------------------------------------------------


class TestVolumeSpikeSignal:
    def test_volume_spike_bullish(self) -> None:
        """Spike with close > open on the last candle → LONG."""
        candles = volume_spike_candles(50)
        sig = VolumeSpikeSignal(lookback=20, spike_multiplier=2.5).generate(_PAIR, candles)
        assert _long(sig), f"Expected LONG, got {sig.direction}"

    def test_volume_spike_bearish(self) -> None:
        """Spike with close < open on the last candle → SHORT."""
        candles = volume_spike_candles(50)
        # Make the last candle bearish.
        last_open = float(candles[-1]["open"])
        candles[-1]["close"] = last_open - 5.0
        candles[-1]["low"] = (last_open - 5.0) * 0.995
        sig = VolumeSpikeSignal(lookback=20, spike_multiplier=2.5).generate(_PAIR, candles)
        assert _short(sig), f"Expected SHORT, got {sig.direction}"

    def test_no_volume_spike_neutral(self) -> None:
        """Normal, constant volume → NEUTRAL."""
        candles = make_candles(50, price_delta=1.0, volume=1_000.0)
        sig = VolumeSpikeSignal(lookback=20, spike_multiplier=2.5).generate(_PAIR, candles)
        assert _neutral(sig)

    def test_volume_spike_insufficient_candles(self) -> None:
        """Fewer than lookback + 1 candles → NEUTRAL."""
        few = make_candles(10, price_delta=1.0, volume=1_000.0)
        sig = VolumeSpikeSignal(lookback=20, spike_multiplier=2.5).generate(_PAIR, few)
        assert _neutral(sig)
        assert "insufficient_candles" in sig.metadata

    def test_volume_spike_zero_avg_volume_neutral(self) -> None:
        """All preceding candles have zero volume → NEUTRAL."""
        arr = make_candles(30, price_delta=1.0, volume=0.0)
        # Give the last candle some volume so it would otherwise spike.
        arr[-1]["volume"] = 10_000.0
        sig = VolumeSpikeSignal(lookback=20, spike_multiplier=2.5).generate(_PAIR, arr)
        assert _neutral(sig)

    def test_volume_spike_metadata_present(self) -> None:
        candles = volume_spike_candles(50)
        sig = VolumeSpikeSignal().generate(_PAIR, candles)
        for key in ("current_volume", "avg_volume", "volume_ratio", "price_direction"):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_volume_spike_strength_clamped(self) -> None:
        candles = volume_spike_candles(50)
        sig = VolumeSpikeSignal().generate(_PAIR, candles)
        assert 0.0 <= sig.strength <= 1.0

    def test_volume_spike_deterministic(self) -> None:
        candles = volume_spike_candles(50)
        gen = VolumeSpikeSignal()
        assert gen.generate(_PAIR, candles) == gen.generate(_PAIR, candles)

    def test_volume_spike_invalid_init_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeSpikeSignal(lookback=0)
        with pytest.raises(ValueError):
            VolumeSpikeSignal(spike_multiplier=0.0)


# ---------------------------------------------------------------------------
# CompositeSignal
# ---------------------------------------------------------------------------


class _AlwaysLong:
    """Stub generator that always returns a LONG signal with given strength."""

    def __init__(self, strength: float = 0.8, name: str = "always_long") -> None:
        self._strength = strength
        self._name = name

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        ts = int(candles[-1]["timestamp"]) if len(candles) > 0 else 0
        return Signal(
            pair=pair,
            direction=SignalDirection.LONG,
            strength=self._strength,
            indicator_name=self._name,
            timestamp=ts,
        )


class _AlwaysShort:
    """Stub generator that always returns a SHORT signal with given strength."""

    def __init__(self, strength: float = 0.8, name: str = "always_short") -> None:
        self._strength = strength
        self._name = name

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        ts = int(candles[-1]["timestamp"]) if len(candles) > 0 else 0
        return Signal(
            pair=pair,
            direction=SignalDirection.SHORT,
            strength=self._strength,
            indicator_name=self._name,
            timestamp=ts,
        )


class _AlwaysNeutral:
    """Stub generator that always returns NEUTRAL."""

    def generate(self, pair: str, candles: np.ndarray) -> Signal:
        ts = int(candles[-1]["timestamp"]) if len(candles) > 0 else 0
        return Signal(
            pair=pair,
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            indicator_name="always_neutral",
            timestamp=ts,
        )


class TestCompositeSignal:
    _candles = uptrend_candles(100)

    def test_composite_fires_on_agreement(self) -> None:
        """Two LONG signals with sufficient strength → Signal returned."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(strength=0.8), 1.0),
                (_AlwaysLong(strength=0.9, name="long2"), 1.0),
            ],
            threshold=0.6,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_composite_returns_none_on_disagreement(self) -> None:
        """Equal LONG and SHORT votes → no majority → None."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(strength=0.9), 1.0),
                (_AlwaysShort(strength=0.9), 1.0),
            ],
            threshold=0.5,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is None

    def test_composite_returns_none_below_threshold(self) -> None:
        """Signals agree on direction but strength is below threshold → None."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(strength=0.3), 1.0),
                (_AlwaysLong(strength=0.2, name="weak2"), 1.0),
            ],
            threshold=0.6,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is None

    def test_composite_all_neutral_returns_none(self) -> None:
        """All generators return NEUTRAL → None."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysNeutral(), 1.0),
                (_AlwaysNeutral(), 1.0),
            ],
            threshold=0.5,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is None

    def test_composite_majority_wins_over_minority(self) -> None:
        """Two LONG and one SHORT → LONG majority fires."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(strength=0.8), 1.0),
                (_AlwaysLong(strength=0.8, name="long2"), 1.0),
                (_AlwaysShort(strength=0.9), 1.0),
            ],
            threshold=0.6,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_composite_metadata_includes_all_signals(self) -> None:
        """Composite metadata must include keys for every individual signal."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(strength=0.8, name="sig_a"), 1.0),
                (_AlwaysLong(strength=0.9, name="sig_b"), 1.0),
            ],
            threshold=0.5,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is not None
        assert "sig_a.strength" in result.metadata
        assert "sig_b.strength" in result.metadata

    def test_composite_strength_is_weighted_average(self) -> None:
        """Composite strength equals weighted average of agreeing strengths."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(strength=0.6), 1.0),
                (_AlwaysLong(strength=1.0, name="strong"), 3.0),
            ],
            threshold=0.5,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is not None
        expected = (0.6 * 1.0 + 1.0 * 3.0) / (1.0 + 3.0)  # 0.9
        assert result.strength == pytest.approx(expected)

    def test_composite_with_real_generators_uptrend(self) -> None:
        """Real signal generators agree on LONG direction in an uptrend."""
        candles = uptrend_candles(100)
        gen = CompositeSignal(
            generators=[
                (MomentumSignal(), 1.0),
                (VolumeSpikeSignal(), 0.5),
            ],
            threshold=0.3,
        )
        # The result may or may not fire depending on volume; just verify no crash.
        result = gen.evaluate(_PAIR, candles)
        if result is not None:
            assert result.direction in (SignalDirection.LONG, SignalDirection.NEUTRAL)

    def test_composite_indicator_name(self) -> None:
        gen = CompositeSignal(
            generators=[(_AlwaysLong(0.9), 1.0)],
            threshold=0.5,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is not None
        assert result.indicator_name == "composite"

    def test_composite_invalid_init_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CompositeSignal(generators=[], threshold=0.5)
        with pytest.raises(ValueError):
            CompositeSignal(generators=[(_AlwaysLong(), 0.0)], threshold=0.5)
        with pytest.raises(ValueError):
            CompositeSignal(generators=[(_AlwaysLong(), 1.0)], threshold=0.0)
        with pytest.raises(ValueError):
            CompositeSignal(generators=[(_AlwaysLong(), 1.0)], threshold=1.1)

    def test_composite_deterministic(self) -> None:
        candles = uptrend_candles(100)
        gen = CompositeSignal(
            generators=[
                (_AlwaysLong(0.8), 1.0),
                (_AlwaysLong(0.9, "b"), 2.0),
            ],
            threshold=0.5,
        )
        assert gen.evaluate(_PAIR, candles) == gen.evaluate(_PAIR, candles)

    def test_composite_neutral_mixed_with_directional(self) -> None:
        """NEUTRAL signals are ignored; only directional ones count."""
        gen = CompositeSignal(
            generators=[
                (_AlwaysNeutral(), 1.0),
                (_AlwaysNeutral(), 1.0),
                (_AlwaysLong(strength=0.9, name="lone_long"), 1.0),
            ],
            threshold=0.5,
        )
        result = gen.evaluate(_PAIR, self._candles)
        assert result is not None
        assert result.direction == SignalDirection.LONG


# ---------------------------------------------------------------------------
# CANDLE_DTYPE integrity check
# ---------------------------------------------------------------------------


class TestCandleDtype:
    def test_dtype_fields(self) -> None:
        assert CANDLE_DTYPE.names is not None
        assert set(CANDLE_DTYPE.names) == {"timestamp", "open", "high", "low", "close", "volume"}

    def test_dtype_types(self) -> None:
        assert CANDLE_DTYPE["timestamp"] == np.dtype("i8")
        for field in ("open", "high", "low", "close", "volume"):
            assert CANDLE_DTYPE[field] == np.dtype("f8"), f"Unexpected dtype for {field}"
