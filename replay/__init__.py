"""Replay and backtesting subsystem for the crypto auto-trading bot.

Public API
----------
- :class:`~replay.replayer.Replayer`
- :class:`~replay.evaluation.BacktestEvaluator`
- :class:`~replay.evaluation.BacktestResult`
- :class:`~replay.canary.CanaryComparison`
"""

from replay.canary import CanaryComparison
from replay.evaluation import BacktestEvaluator, BacktestResult
from replay.replayer import Replayer

__all__ = [
    "Replayer",
    "BacktestEvaluator",
    "BacktestResult",
    "CanaryComparison",
]
