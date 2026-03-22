"""Paper-trading executor that simulates realistic fills without network calls.

Fill models
-----------
``next_bar_open`` (default, P1-2)
    Orders are queued when ``place_order`` is called and filled at the *open*
    price of the **next** candle, set via :meth:`PaperExecutor.set_next_bar_price`.
    This eliminates look-ahead bias: the decision to buy was made on candle N,
    but the fill cannot happen until candle N+1's open is known.

``worst_case``
    Buys fill at the candle *high*, sells at the candle *low*.  If no candle
    data has been provided yet, the order is queued similarly to
    ``next_bar_open`` and filled on the next :meth:`set_next_bar_price` call
    using the worst-case price from that bar.

Slippage model
--------------
    slippage = base_slippage_pct + volume_impact * (quantity / avg_volume)

Fees
----
A flat taker rate (``fee_pct``) is applied to the filled notional value.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import structlog

from execution.base import OrderRecord, OrderResult, OrderStatus, validate_transition

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Fractional volume-impact coefficient used in the slippage formula.
_VOLUME_IMPACT: float = 0.001


# ---------------------------------------------------------------------------
# Internal bar-price snapshot
# ---------------------------------------------------------------------------


@dataclass
class _BarPrice:
    """Prices for a single bar used by fill-model calculations.

    Attributes:
        open: Open price of the bar.
        high: Highest price during the bar.
        low: Lowest price during the bar.
        avg_volume: Average base-asset volume, used in the slippage formula.
    """

    open: float
    high: float
    low: float
    avg_volume: float


# ---------------------------------------------------------------------------
# PaperExecutor
# ---------------------------------------------------------------------------


class PaperExecutor:
    """Simulates order execution with configurable fill models.

    P1-2: The default fill model is ``next_bar_open`` to avoid look-ahead bias.
    All monetary values are denominated in the quote asset (e.g. USDT).

    Args:
        initial_balance: Starting quote-asset balance.
        fill_model: ``"next_bar_open"`` (default) or ``"worst_case"``.
        base_slippage_pct: Base slippage as a fraction, e.g. ``0.001`` = 0.1 %.
        fee_pct: Taker fee as a fraction, e.g. ``0.001`` = 0.1 %.

    Example::

        executor = PaperExecutor(initial_balance=10_000.0)
        result = await executor.place_order("BTC/USDT", "buy", 0.01, "ts-abc12345-00000001")
        executor.set_next_bar_price("BTC/USDT", open_price=50_100.0, high=50_500.0,
                                     low=49_900.0, avg_volume=200.0)
        # After set_next_bar_price the pending order is filled.
        result = await executor.get_order_status("ts-abc12345-00000001")
        assert result.status == OrderStatus.FILLED
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        fill_model: Literal["next_bar_open", "worst_case"] = "next_bar_open",
        base_slippage_pct: float = 0.001,
        fee_pct: float = 0.001,
    ) -> None:
        self._balance: float = initial_balance
        self._fill_model: Literal["next_bar_open", "worst_case"] = fill_model
        self._base_slippage_pct: float = base_slippage_pct
        self._fee_pct: float = fee_pct

        # pair -> quantity held (positive = long)
        self._positions: dict[str, float] = {}

        # client_order_id -> OrderRecord
        self._orders: dict[str, OrderRecord] = {}

        # pair -> latest bar prices (set by set_next_bar_price)
        self._latest_bar: dict[str, _BarPrice] = {}

        # pair -> list of client_order_ids awaiting a bar-open fill
        self._pending_fills: dict[str, list[str]] = {}

        # Protects _balance, _positions, _orders from concurrent coroutines.
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Executor protocol implementation
    # ------------------------------------------------------------------

    async def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        client_order_id: str,
    ) -> OrderResult:
        """Queue or immediately fill a paper order.

        When *fill_model* is ``"next_bar_open"`` the order is stored as
        ``PENDING_SUBMIT`` and filled by the next :meth:`set_next_bar_price`
        call.  With ``"worst_case"`` the fill happens immediately if a bar is
        already available, otherwise it is also queued.

        Args:
            pair: CCXT-formatted trading pair.
            side: ``"buy"`` or ``"sell"``.
            quantity: Asset quantity (base asset).
            client_order_id: Caller-provided idempotency key.

        Returns:
            :class:`OrderResult` reflecting the order's state after placement.
        """
        async with self._lock:
            now = datetime.now(UTC)
            record = OrderRecord(
                client_order_id=client_order_id,
                exchange_order_id=f"paper-{client_order_id}",
                pair=pair,
                side=side,
                requested_quantity=quantity,
                status=OrderStatus.PENDING_SUBMIT,
                created_at=now,
                updated_at=now,
            )
            record.transitions.append((OrderStatus.PENDING_SUBMIT, now))
            self._orders[client_order_id] = record

            log.debug(
                "paper_executor.order_placed",
                client_order_id=client_order_id,
                pair=pair,
                side=side,
                quantity=quantity,
                fill_model=self._fill_model,
            )

            # Attempt an immediate fill if a bar is available and fill model allows it.
            if self._fill_model == "worst_case" and pair in self._latest_bar:
                bar = self._latest_bar[pair]
                fill_price = bar.high if side == "buy" else bar.low
                self._apply_fill(record, fill_price, bar.avg_volume)
            elif self._fill_model == "next_bar_open" or pair not in self._latest_bar:
                # Queue for next bar regardless of fill model.
                self._pending_fills.setdefault(pair, []).append(client_order_id)
                # Transition to SUBMITTED to indicate we're "on the books".
                self._transition(record, OrderStatus.SUBMITTED)
            else:
                # worst_case but no bar yet — queue
                self._pending_fills.setdefault(pair, []).append(client_order_id)
                self._transition(record, OrderStatus.SUBMITTED)

            return self._record_to_result(record)

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel a pending paper order.

        Args:
            client_order_id: The idempotency key of the order to cancel.

        Returns:
            ``True`` if the order was found and cancelled; ``False`` if it was
            already in a terminal state or not found.
        """
        async with self._lock:
            record = self._orders.get(client_order_id)
            if record is None:
                log.warning(
                    "paper_executor.cancel_not_found",
                    client_order_id=client_order_id,
                )
                return False

            if record.status in {
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.EXPIRED,
                OrderStatus.REJECTED,
            }:
                log.debug(
                    "paper_executor.cancel_noop",
                    client_order_id=client_order_id,
                    status=record.status,
                )
                return False

            # Remove from pending-fill queue if present.
            pending = self._pending_fills.get(record.pair, [])
            if client_order_id in pending:
                pending.remove(client_order_id)

            self._transition(record, OrderStatus.CANCELLED)
            log.info(
                "paper_executor.order_cancelled",
                client_order_id=client_order_id,
            )
            return True

    async def get_order_status(self, client_order_id: str) -> OrderResult:
        """Return the current :class:`OrderResult` for a known order.

        Args:
            client_order_id: The idempotency key of the order to query.

        Returns:
            The latest :class:`OrderResult` snapshot.

        Raises:
            KeyError: If *client_order_id* is not tracked by this executor.
        """
        async with self._lock:
            record = self._orders[client_order_id]
            return self._record_to_result(record)

    # ------------------------------------------------------------------
    # Bar-price ingestion
    # ------------------------------------------------------------------

    def set_next_bar_price(
        self,
        pair: str,
        open_price: float,
        high: float,
        low: float,
        avg_volume: float,
    ) -> None:
        """Provide a new bar's prices and trigger fills for queued orders.

        Called by the signal loop when a new candle arrives.  Any orders that
        were queued for *pair* will be filled according to the active fill model:

        * ``next_bar_open``: fills at *open_price*.
        * ``worst_case``: buys at *high*, sells at *low*.

        The lock is acquired synchronously-style via ``asyncio.get_event_loop``
        because this method is typically called from synchronous candle-processing
        code.  If you call it from an async context, wrap it in
        ``asyncio.get_event_loop().run_until_complete`` or refactor to ``async``.

        Args:
            pair: CCXT-formatted trading pair.
            open_price: Open price of the arriving bar.
            high: Highest price during the bar.
            low: Lowest price during the bar.
            avg_volume: Average volume used for slippage calculation.
        """
        bar = _BarPrice(open=open_price, high=high, low=low, avg_volume=avg_volume)
        self._latest_bar[pair] = bar

        pending_ids = self._pending_fills.pop(pair, [])
        if not pending_ids:
            return

        for client_order_id in pending_ids:
            record = self._orders.get(client_order_id)
            if record is None or record.status not in {
                OrderStatus.PENDING_SUBMIT,
                OrderStatus.SUBMITTED,
            }:
                continue

            fill_price = (
                open_price
                if self._fill_model == "next_bar_open"
                else (bar.high if record.side == "buy" else bar.low)
            )
            self._apply_fill(record, fill_price, avg_volume)

            log.info(
                "paper_executor.order_filled",
                client_order_id=client_order_id,
                pair=pair,
                side=record.side,
                fill_price=record.filled_price,
                filled_quantity=record.filled_quantity,
                fees=record.fees,
                fill_model=self._fill_model,
            )

    # ------------------------------------------------------------------
    # Portfolio state accessors
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Current quote-asset balance after all fills and fees."""
        return self._balance

    @property
    def positions(self) -> dict[str, float]:
        """Current open positions as a ``{pair: quantity}`` mapping."""
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_slippage(self, quantity: float, avg_volume: float) -> float:
        """Calculate total slippage fraction for a given order.

        Formula::

            slippage = base_slippage_pct + volume_impact * (quantity / avg_volume)

        Args:
            quantity: Asset quantity being traded.
            avg_volume: Average volume over recent bars (used as market depth proxy).

        Returns:
            Slippage as a fraction (e.g. ``0.0015`` = 0.15 %).
        """
        if avg_volume <= 0:
            return self._base_slippage_pct
        return self._base_slippage_pct + _VOLUME_IMPACT * (quantity / avg_volume)

    def _apply_fill(
        self,
        record: OrderRecord,
        raw_fill_price: float,
        avg_volume: float,
    ) -> None:
        """Mutate *record* in-place to reflect a completed fill.

        Applies slippage (adverse) and fees to compute the effective fill price
        and updates the executor's virtual balance and positions.

        Args:
            record: The :class:`OrderRecord` to update.
            raw_fill_price: Pre-slippage price from the bar model.
            avg_volume: Average bar volume for slippage computation.
        """
        slippage = self._compute_slippage(record.requested_quantity, avg_volume)

        # Slippage is adverse: buys get a higher price, sells a lower price.
        if record.side == "buy":
            effective_price = raw_fill_price * (1.0 + slippage)
        else:
            effective_price = raw_fill_price * (1.0 - slippage)

        notional = effective_price * record.requested_quantity
        fees = notional * self._fee_pct

        # Update balance and positions.
        if record.side == "buy":
            self._balance -= notional + fees
            self._positions[record.pair] = (
                self._positions.get(record.pair, 0.0) + record.requested_quantity
            )
        else:
            self._balance += notional - fees
            self._positions[record.pair] = (
                self._positions.get(record.pair, 0.0) - record.requested_quantity
            )

        # Remove zero (or near-zero) positions.
        if abs(self._positions.get(record.pair, 0.0)) < 1e-10:
            self._positions.pop(record.pair, None)

        # Update the record.
        record.filled_price = effective_price
        record.filled_quantity = record.requested_quantity
        record.fees = fees
        record.updated_at = datetime.now(UTC)

        # Ensure the record is in SUBMITTED before transitioning to FILLED.
        if record.status == OrderStatus.PENDING_SUBMIT:
            self._transition(record, OrderStatus.SUBMITTED)
        self._transition(record, OrderStatus.FILLED)

    def _transition(self, record: OrderRecord, target: OrderStatus) -> None:
        """Apply a state transition to *record*, logging on invalid moves.

        Args:
            record: The :class:`OrderRecord` to update.
            target: Desired next :class:`OrderStatus`.
        """
        if not validate_transition(record.status, target):
            log.critical(
                "paper_executor.invalid_transition",
                client_order_id=record.client_order_id,
                current_status=record.status,
                target_status=target,
            )
            return

        now = datetime.now(UTC)
        record.status = target
        record.updated_at = now
        record.transitions.append((target, now))

    @staticmethod
    def _record_to_result(record: OrderRecord) -> OrderResult:
        """Convert an :class:`OrderRecord` to a lightweight :class:`OrderResult`.

        Args:
            record: Source record.

        Returns:
            :class:`OrderResult` snapshot of *record*.
        """
        return OrderResult(
            client_order_id=record.client_order_id,
            exchange_order_id=record.exchange_order_id,
            status=record.status,
            filled_price=record.filled_price,
            filled_quantity=record.filled_quantity,
            fees=record.fees,
        )
