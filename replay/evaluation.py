"""Backtest evaluation metrics computed from replay results.

The :class:`BacktestEvaluator` accumulates signal and trade records produced
by :class:`~replay.replayer.Replayer` and computes a standard set of
performance metrics via :meth:`BacktestEvaluator.get_result`.

All computations are pure Python — no I/O, no external state.  The evaluator
is safe to use from synchronous test code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from signals.base import Signal

log: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Aggregated performance metrics for a completed backtest.

    Attributes:
        total_trades: Total number of completed trades recorded.
        winning_trades: Trades with strictly positive PnL.
        losing_trades: Trades with strictly negative PnL.
        win_rate: ``winning_trades / total_trades`` in ``[0.0, 1.0]``;
            ``0.0`` when no trades have been recorded.
        total_pnl: Sum of all per-trade PnL values.
        max_drawdown: Largest peak-to-trough decline in the cumulative PnL
            curve (expressed as a positive number; ``0.0`` if no drawdown
            occurred).
        sharpe_ratio: Annualised Sharpe ratio computed from per-trade returns
            assuming 252 trading days; ``0.0`` when fewer than two trades exist.
        profit_factor: Gross profit divided by gross loss (absolute value).
            ``0.0`` when there is no gross profit; ``inf`` when there is no
            gross loss.
        avg_trade_pnl: Arithmetic mean PnL per trade; ``0.0`` when no trades.
        total_signals: Total number of composite signals generated.
        signals_rejected_by_risk: Signals that the risk gate did not approve.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    total_signals: int
    signals_rejected_by_risk: int


class BacktestEvaluator:
    """Computes backtest metrics from replay results.

    Call :meth:`record_signal` for every signal that fires during replay and
    :meth:`record_trade` for every completed trade.  When the replay is
    finished, call :meth:`get_result` to obtain the full
    :class:`BacktestResult`.

    Example::

        evaluator = BacktestEvaluator()
        for signal in replay_signals:
            approved = risk_gate.evaluate(signal, ...) is not None
            evaluator.record_signal(signal, approved)
        for pnl in trade_pnls:
            evaluator.record_trade(pnl)
        result = evaluator.get_result()
    """

    def __init__(self) -> None:
        self._trade_pnls: list[float] = []
        self._total_signals: int = 0
        self._signals_rejected: int = 0

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def record_signal(self, signal: Signal, approved: bool) -> None:
        """Record the outcome of a risk-gate evaluation for one signal.

        Args:
            signal: The :class:`~signals.base.Signal` that was evaluated.
            approved: ``True`` when the risk gate approved the order;
                ``False`` when it was rejected.
        """
        self._total_signals += 1
        if not approved:
            self._signals_rejected += 1
            log.debug(
                "evaluator.signal_rejected",
                pair=signal.pair,
                direction=str(signal.direction),
                total_rejected=self._signals_rejected,
            )

    def record_trade(self, pnl: float) -> None:
        """Record the realised PnL of one completed trade.

        Args:
            pnl: Signed profit-and-loss for the trade.  Positive values
                indicate a profitable trade; negative values indicate a loss.
        """
        self._trade_pnls.append(pnl)
        log.debug(
            "evaluator.trade_recorded",
            pnl=pnl,
            total_trades=len(self._trade_pnls),
        )

    # ------------------------------------------------------------------
    # Result computation
    # ------------------------------------------------------------------

    def get_result(self) -> BacktestResult:
        """Compute and return the full :class:`BacktestResult`.

        All metrics are derived from the signals and trades recorded so far.
        Calling this method does not reset internal state — it is safe to call
        multiple times.

        Returns:
            A fully populated :class:`BacktestResult` snapshot.
        """
        pnls = self._trade_pnls
        total_trades = len(pnls)

        winning_trades = sum(1 for p in pnls if p > 0.0)
        losing_trades = sum(1 for p in pnls if p < 0.0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_pnl = sum(pnls)
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(p for p in pnls if p > 0.0)
        gross_loss = abs(sum(p for p in pnls if p < 0.0))
        if gross_profit == 0.0:
            profit_factor = 0.0
        elif gross_loss == 0.0:
            profit_factor = math.inf
        else:
            profit_factor = gross_profit / gross_loss

        max_dd = _max_drawdown(pnls)
        sharpe = self.sharpe_ratio(pnls)

        result = BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            total_signals=self._total_signals,
            signals_rejected_by_risk=self._signals_rejected,
        )

        log.info(
            "evaluator.result_computed",
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            total_pnl=result.total_pnl,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
        )
        return result

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
        """Annualised Sharpe ratio from a list of periodic returns.

        Uses the standard formula:

        .. math::

            S = \\frac{\\bar{r} - r_f}{\\sigma_r} \\times \\sqrt{252}

        where :math:`\\bar{r}` is the mean return, :math:`r_f` is the
        risk-free rate, and :math:`\\sigma_r` is the sample standard deviation
        of returns.

        Args:
            returns: List of periodic (e.g. per-trade) returns.  May contain
                negative values.
            risk_free_rate: Risk-free rate per period (default ``0.0``).

        Returns:
            Annualised Sharpe ratio, or ``0.0`` when fewer than two data
            points are available or the standard deviation is zero.
        """
        if len(returns) < 2:
            return 0.0

        n = len(returns)
        mean_r = sum(returns) / n
        excess = mean_r - risk_free_rate

        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_dev = math.sqrt(variance)

        if std_dev == 0.0:
            return 0.0

        # Annualise assuming 252 independent trading periods per year.
        return (excess / std_dev) * math.sqrt(252)


# ---------------------------------------------------------------------------
# Module-level helper (not part of the public API)
# ---------------------------------------------------------------------------


def _max_drawdown(pnls: list[float]) -> float:
    """Compute the maximum peak-to-trough drawdown of the cumulative PnL curve.

    Args:
        pnls: Ordered list of per-trade PnL values.

    Returns:
        Maximum drawdown as a positive number (or ``0.0`` if the curve never
        declines).
    """
    if not pnls:
        return 0.0

    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0

    for pnl in pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd
