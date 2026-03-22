"""Structured trade audit logging for the ledger subsystem.

Every trading event — signals, risk decisions, orders, and regime changes —
is written to the database via :class:`TradeLogger`.  All writes are
append-only; existing rows are never deleted.  Order update operations use
``UPDATE`` statements gated on the ``client_order_id`` unique index.

The logger intentionally has no in-memory buffer: every call performs a
real database write so that crash-safe audit invariants (P0-3) hold even
if the process terminates immediately after the call returns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update

from ledger.database import Database
from ledger.models import (
    OrderRecord,
    RegimeRecord,
    RiskDecisionRecord,
    SignalRecord,
)
from risk.base import ApprovedOrder, RiskDecision
from signals.base import Signal

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class TradeLogger:
    """Logs all trading events to the database for a complete audit trail.

    Every signal, risk decision, order, and regime change is recorded with
    timestamps for full traceability.  Methods are intentionally fine-grained
    so that callers can log each stage of the pipeline independently.

    Args:
        database: Initialised :class:`~ledger.database.Database` instance.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    async def log_signal(self, signal: Signal) -> int:
        """Persist a trading signal and return its database record ID.

        Args:
            signal: The :class:`~signals.base.Signal` to persist.

        Returns:
            The auto-generated integer primary key of the inserted row.
        """
        record = SignalRecord(
            pair=signal.pair,
            direction=signal.direction.value,
            strength=signal.strength,
            indicator_name=signal.indicator_name,
            signal_timestamp_ms=signal.timestamp,
            metadata_json=json.dumps(signal.metadata),
        )

        async with self._db.session() as sess:
            sess.add(record)
            await sess.commit()
            await sess.refresh(record)

        log.debug(
            "trade_logger.signal",
            signal_db_id=record.id,
            pair=signal.pair,
            direction=signal.direction.value,
            strength=signal.strength,
        )
        return record.id

    # ------------------------------------------------------------------
    # Risk decision logging
    # ------------------------------------------------------------------

    async def log_risk_decision(
        self, signal_db_id: int, decision: RiskDecision
    ) -> int:
        """Persist a risk gate decision linked to a previously logged signal.

        Args:
            signal_db_id: The database ID returned by :meth:`log_signal`.
            decision: The :class:`~risk.base.RiskDecision` to persist.

        Returns:
            The auto-generated integer primary key of the inserted row.
        """
        record = RiskDecisionRecord(
            signal_id=signal_db_id,
            approved=decision.approved,
            reason=decision.reason,
            adjusted_quantity=decision.adjusted_quantity,
            checks_passed_json=json.dumps(decision.checks_passed),
            checks_failed_json=json.dumps(decision.checks_failed),
        )

        async with self._db.session() as sess:
            sess.add(record)
            await sess.commit()
            await sess.refresh(record)

        log.debug(
            "trade_logger.risk_decision",
            risk_db_id=record.id,
            signal_db_id=signal_db_id,
            approved=decision.approved,
            reason=decision.reason,
        )
        return record.id

    # ------------------------------------------------------------------
    # Order logging
    # ------------------------------------------------------------------

    async def log_order(
        self,
        order: ApprovedOrder,
        client_order_id: str,
        status: str,
    ) -> int:
        """Persist a new order in the initial state.

        This must be called *before* the order is submitted to the exchange
        to satisfy the P0-3 crash-safe checkpoint.

        Args:
            order: The :class:`~risk.base.ApprovedOrder` authorised by the
                risk gate.
            client_order_id: The engine-generated idempotency key.
            status: Initial order status string (typically
                ``"pending_submit"``).

        Returns:
            The auto-generated integer primary key of the inserted row.

        Raises:
            sqlalchemy.exc.IntegrityError: If ``client_order_id`` is already
                present (P0-3 UNIQUE constraint violation).
        """
        record = OrderRecord(
            client_order_id=client_order_id,
            signal_id=order.signal_id,
            pair=order.pair,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=0.0,
            filled_price=0.0,
            status=status,
            fees=0.0,
            transitions_json=json.dumps(
                [[status, datetime.now(UTC).isoformat()]]
            ),
        )

        async with self._db.session() as sess:
            sess.add(record)
            await sess.commit()
            await sess.refresh(record)

        log.debug(
            "trade_logger.order",
            order_db_id=record.id,
            client_order_id=client_order_id,
            pair=order.pair,
            side=order.side,
            status=status,
        )
        return record.id

    async def log_order_update(
        self,
        client_order_id: str,
        status: str,
        filled_qty: float,
        filled_price: float,
        fees: float,
    ) -> None:
        """Update an existing order row with fill information and new status.

        Also appends the new status + timestamp to the ``transitions_json``
        audit trail column.

        Args:
            client_order_id: The idempotency key that identifies the row.
            status: The new :class:`~execution.base.OrderStatus` string value.
            filled_qty: Updated cumulative filled quantity.
            filled_price: Updated volume-weighted average fill price.
            fees: Updated cumulative fees (quote-asset units).
        """
        now_iso = datetime.now(UTC).isoformat()

        async with self._db.session() as sess:
            # Fetch current transitions to append.
            result = await sess.execute(
                select(OrderRecord).where(
                    OrderRecord.client_order_id == client_order_id
                )
            )
            row = result.scalar_one_or_none()

            if row is None:
                log.warning(
                    "trade_logger.order_update.not_found",
                    client_order_id=client_order_id,
                )
                return

            existing: list[list[str]] = json.loads(row.transitions_json)
            existing.append([status, now_iso])

            await sess.execute(
                update(OrderRecord)
                .where(OrderRecord.client_order_id == client_order_id)
                .values(
                    status=status,
                    filled_quantity=filled_qty,
                    filled_price=filled_price,
                    fees=fees,
                    updated_at=datetime.now(UTC),
                    transitions_json=json.dumps(existing),
                )
            )
            await sess.commit()

        log.debug(
            "trade_logger.order_update",
            client_order_id=client_order_id,
            status=status,
            filled_qty=filled_qty,
            filled_price=filled_price,
            fees=fees,
        )

    # ------------------------------------------------------------------
    # Regime logging
    # ------------------------------------------------------------------

    async def log_regime(
        self,
        regime: str,
        confidence: float,
        raw_response: str,
        active_pairs: list[str],
        active_strategies: list[str],
    ) -> int:
        """Persist a market regime classification event.

        Args:
            regime: Regime label (e.g. ``"trending"``, ``"ranging"``).
            confidence: Model confidence in ``[0.0, 1.0]``.
            raw_response: Full raw LLM response text for auditability.
            active_pairs: Trading pairs active under this regime.
            active_strategies: Strategy names active under this regime.

        Returns:
            The auto-generated integer primary key of the inserted row.
        """
        record = RegimeRecord(
            regime=regime,
            confidence=confidence,
            raw_response=raw_response,
            active_pairs_json=json.dumps(active_pairs),
            active_strategies_json=json.dumps(active_strategies),
        )

        async with self._db.session() as sess:
            sess.add(record)
            await sess.commit()
            await sess.refresh(record)

        log.info(
            "trade_logger.regime",
            regime_db_id=record.id,
            regime=regime,
            confidence=confidence,
            active_pairs=active_pairs,
        )
        return record.id
