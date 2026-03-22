"""PnL computation and in-memory tracking for the ledger subsystem.

:class:`PnLTracker` maintains a FIFO position book per trading pair and
computes both realised and unrealised profit-and-loss in real time.

Position model
--------------
- Each buy creates one or more ``(quantity, entry_price)`` lot entries for the
  pair.
- Each sell matches against the oldest lots first (FIFO), realising PnL for
  the portion consumed.
- Unrealised PnL is computed on demand from the remaining open lots and the
  most recent market price for each pair.

Metrics
-------
- ``max_drawdown``: The maximum observed decline from a cumulative-PnL peak,
  expressed as an absolute dollar amount.  Tracked on a rolling basis across
  all ``record_trade`` calls.
- ``profit_factor``: Gross winning PnL divided by the absolute value of gross
  losing PnL.  ``math.inf`` when there are no losing trades.
- ``win_rate``: Fraction of closed trades that resulted in a net gain.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, date, datetime

import structlog

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dataclass
class PnLSummary:
    """Snapshot of all PnL metrics at a point in time.

    Attributes:
        realized_pnl: Total realised PnL across all closed trades (lifetime,
            or since the last :meth:`~PnLTracker.reset_daily` call).
        unrealized_pnl: Current unrealised PnL across all open positions.
        total_pnl: ``realized_pnl + unrealized_pnl``.
        win_count: Number of trades that closed with a positive PnL.
        loss_count: Number of trades that closed with a negative or zero PnL.
        win_rate: ``win_count / total_trades`` in ``[0.0, 1.0]``, or ``0.0``
            when no trades have been recorded.
        total_trades: Total number of closed trades (``win_count + loss_count``).
        max_drawdown: Maximum observed drawdown from any PnL peak, expressed as
            a positive dollar amount (e.g. ``150.0`` means the PnL fell $150
            below its peak at some point).
        profit_factor: Gross profit divided by gross loss magnitude.
            ``math.inf`` when there are no losing trades and at least one
            winning trade.  ``0.0`` when no trades have been recorded.
    """

    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    total_trades: int
    max_drawdown: float
    profit_factor: float


class PnLTracker:
    """Tracks and computes PnL metrics with in-memory position management.

    Thread-safety: this class is not thread-safe.  It is designed to be used
    exclusively from the asyncio event loop.  For concurrent access, wrap
    operations in an ``asyncio.Lock``.

    The tracker maintains two date-scoped counters (``_daily_realized_pnl`` and
    ``_daily_trade_count``) that are reset by :meth:`reset_daily`.  All other
    state (positions, peak, drawdown, wins/losses) is lifetime-scoped.
    """

    def __init__(self) -> None:
        # pair -> deque of (quantity, entry_price) lots, oldest-first (FIFO)
        self._positions: dict[str, deque[tuple[float, float]]] = defaultdict(deque)

        # pair -> latest known market price for unrealised PnL
        self._market_prices: dict[str, float] = {}

        # Lifetime realised PnL accumulator
        self._realized_pnl: float = 0.0

        # Win/loss counters and gross flows for profit_factor
        self._win_count: int = 0
        self._loss_count: int = 0
        self._gross_profit: float = 0.0   # sum of positive trade PnLs
        self._gross_loss: float = 0.0     # sum of abs(negative trade PnLs)

        # Max-drawdown tracking (rolling peak of cumulative realised PnL)
        self._peak_pnl: float = 0.0
        self._max_drawdown: float = 0.0

        # Daily tracking
        self._daily_realized_pnl: float = 0.0
        self._daily_date: date = datetime.now(UTC).date()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: float,
        fees: float,
    ) -> None:
        """Record a completed trade fill and update all metrics.

        For a ``"buy"`` trade, a new lot ``(quantity, price)`` is appended to
        the FIFO queue for ``pair``.

        For a ``"sell"`` trade, lots are consumed from the front of the queue
        (FIFO), realising PnL for each lot consumed.  The ``fees`` argument is
        subtracted from the realised PnL of the trade as a whole.

        Args:
            pair: CCXT-formatted trading pair (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            quantity: Quantity filled (positive, base asset units).
            price: Actual fill price (quote-asset units per base unit).
            fees: Total fees charged for this fill (quote-asset units).
        """
        side_lower = side.lower()
        if side_lower == "buy":
            self._positions[pair].append((quantity, price))
            log.debug(
                "pnl_tracker.buy",
                pair=pair,
                quantity=quantity,
                price=price,
                fees=fees,
            )
        elif side_lower == "sell":
            trade_pnl = self._realize_sell(pair, quantity, price) - fees
            self._realized_pnl += trade_pnl
            self._daily_realized_pnl += trade_pnl

            # Update win/loss counters and gross flows.
            if trade_pnl > 0.0:
                self._win_count += 1
                self._gross_profit += trade_pnl
            else:
                self._loss_count += 1
                self._gross_loss += abs(trade_pnl)

            # Update max drawdown: peak is max of running realised PnL.
            if self._realized_pnl > self._peak_pnl:
                self._peak_pnl = self._realized_pnl
            current_drawdown = self._peak_pnl - self._realized_pnl
            if current_drawdown > self._max_drawdown:
                self._max_drawdown = current_drawdown

            log.debug(
                "pnl_tracker.sell",
                pair=pair,
                quantity=quantity,
                price=price,
                fees=fees,
                trade_pnl=trade_pnl,
                realized_pnl=self._realized_pnl,
            )
        else:
            log.warning("pnl_tracker.unknown_side", side=side)

    def update_market_price(self, pair: str, price: float) -> None:
        """Update the latest market price used for unrealised PnL computation.

        Args:
            pair: CCXT-formatted trading pair.
            price: Current mid/mark price (quote-asset units).
        """
        self._market_prices[pair] = price

    def get_daily_pnl(self) -> float:
        """Return today's total PnL (daily realised + current unrealised).

        Returns:
            Combined daily PnL in quote-asset units.
        """
        return self._daily_realized_pnl + self._compute_unrealized_pnl()

    def get_summary(self) -> PnLSummary:
        """Compute and return a complete PnL snapshot.

        Returns:
            A :class:`PnLSummary` reflecting all recorded trades and open
            positions.
        """
        unrealized = self._compute_unrealized_pnl()
        total_trades = self._win_count + self._loss_count

        win_rate = self._win_count / total_trades if total_trades > 0 else 0.0

        if self._gross_loss == 0.0:
            profit_factor = math.inf if self._gross_profit > 0.0 else 0.0
        else:
            profit_factor = self._gross_profit / self._gross_loss

        return PnLSummary(
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
            total_pnl=self._realized_pnl + unrealized,
            win_count=self._win_count,
            loss_count=self._loss_count,
            win_rate=win_rate,
            total_trades=total_trades,
            max_drawdown=self._max_drawdown,
            profit_factor=profit_factor,
        )

    def reset_daily(self) -> None:
        """Reset daily PnL tracking counters.

        Call this at UTC midnight to start a fresh daily accounting period.
        Lifetime metrics (positions, wins/losses, drawdown) are unaffected.
        """
        self._daily_realized_pnl = 0.0
        self._daily_date = datetime.now(UTC).date()
        log.info("pnl_tracker.daily_reset", date=str(self._daily_date))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _realize_sell(self, pair: str, sell_qty: float, sell_price: float) -> float:
        """Consume FIFO lots and return gross realised PnL (before fees).

        Args:
            pair: Trading pair whose lots are consumed.
            sell_qty: Total quantity being sold.
            sell_price: Fill price of the sell order.

        Returns:
            Gross realised PnL (positive = profit, negative = loss).
        """
        lots = self._positions[pair]
        remaining = sell_qty
        gross_pnl = 0.0

        while remaining > 0.0 and lots:
            lot_qty, lot_price = lots[0]

            if lot_qty <= remaining:
                # Consume the entire lot.
                gross_pnl += (sell_price - lot_price) * lot_qty
                remaining -= lot_qty
                lots.popleft()
            else:
                # Partially consume the lot.
                gross_pnl += (sell_price - lot_price) * remaining
                lots[0] = (lot_qty - remaining, lot_price)
                remaining = 0.0

        if remaining > 0.0:
            # Sold more than held; treat excess as zero-cost (short not modelled).
            log.warning(
                "pnl_tracker.oversell",
                pair=pair,
                excess_qty=remaining,
            )

        return gross_pnl

    def _compute_unrealized_pnl(self) -> float:
        """Compute unrealised PnL across all open positions.

        For each pair with open lots, computes
        ``sum((current_price - entry_price) * quantity)`` over all lots.
        Pairs with no current market price are ignored.

        Returns:
            Total unrealised PnL in quote-asset units.
        """
        total = 0.0
        for pair, lots in self._positions.items():
            current_price = self._market_prices.get(pair)
            if current_price is None:
                continue
            for qty, entry_price in lots:
                total += (current_price - entry_price) * qty
        return total
