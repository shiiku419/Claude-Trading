"""Trading mode definitions driven by Claude's regime classification.

Each :class:`TradingMode` is an immutable snapshot of what the bot is
allowed to do under a given market regime.  The :data:`MODES` registry
maps every :class:`Regime` value to a concrete mode object so that the
:class:`~strategy.controller.StrategyController` can do a simple
dictionary lookup after Claude classifies the market.

Regime semantics
----------------
- ``trending_up``    – directional bull market; run momentum + volume-spike
                       strategies at full risk.
- ``trending_down``  – directional bear market; momentum-only on BTC, half risk.
- ``ranging``        – mean-reversion opportunity; VWAP only, reduced risk.
- ``high_volatility``– no trading; risk_multiplier = 0.0.
- ``unknown``        – regime expired or never set; no trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Regime(StrEnum):
    """Canonical market-regime labels produced by Claude.

    The string values are used directly in Redis and in Claude's JSON
    response, so they must never be renamed without a data-migration step.
    """

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TradingMode:
    """Immutable description of bot behaviour for a single market regime.

    Attributes:
        name: Regime label; matches the corresponding :class:`Regime` value.
        active_pairs: Trading pairs the signal engine may evaluate.  An
            empty list means no pairs are eligible (no new positions).
        active_strategies: Strategy identifiers (``"momentum"``,
            ``"vwap"``, ``"volume_spike"``) enabled in this mode.
        signal_weights: Relative weighting for each active strategy when
            the :class:`~signals.composite.CompositeSignal` blends
            individual scores.  Weights need not sum to 1; the composite
            normalises them internally.
        risk_multiplier: Scalar applied to every risk parameter produced
            by the :class:`~risk.risk_gate.RiskGate`.  ``0.0`` effectively
            disables new-order generation.
    """

    name: str
    active_pairs: list[str] = field(default_factory=list)
    active_strategies: list[str] = field(default_factory=list)
    signal_weights: dict[str, float] = field(default_factory=dict)
    risk_multiplier: float = 0.0


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

MODES: dict[str, TradingMode] = {
    Regime.TRENDING_UP: TradingMode(
        name=Regime.TRENDING_UP,
        active_pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        active_strategies=["momentum", "volume_spike"],
        signal_weights={"momentum": 0.7, "volume_spike": 0.3},
        risk_multiplier=1.0,
    ),
    Regime.TRENDING_DOWN: TradingMode(
        name=Regime.TRENDING_DOWN,
        active_pairs=["BTC/USDT"],
        active_strategies=["momentum"],
        signal_weights={"momentum": 1.0},
        risk_multiplier=0.5,
    ),
    Regime.RANGING: TradingMode(
        name=Regime.RANGING,
        active_pairs=["BTC/USDT", "ETH/USDT"],
        active_strategies=["vwap"],
        signal_weights={"vwap": 1.0},
        risk_multiplier=0.3,
    ),
    Regime.HIGH_VOLATILITY: TradingMode(
        name=Regime.HIGH_VOLATILITY,
        active_pairs=[],
        active_strategies=[],
        signal_weights={},
        risk_multiplier=0.0,
    ),
    Regime.UNKNOWN: TradingMode(
        name=Regime.UNKNOWN,
        active_pairs=[],
        active_strategies=[],
        signal_weights={},
        risk_multiplier=0.0,
    ),
}
