"""Crash-safe recovery worker for orders stuck in PENDING_SUBMIT.

P0-3 Invariant
--------------
If the process crashes between writing a PENDING_SUBMIT record and updating it
to SUBMITTED, the order is left in an ambiguous state.  On restart (or
periodically while running), :class:`RecoveryWorker` scans for stale
PENDING_SUBMIT orders and reconciles them against the exchange:

- If the exchange knows about the order → update to SUBMITTED (or FILLED if
  the executor reports a fill).
- If the exchange has no record of the order → the submit never reached the
  exchange, so mark it CANCELLED.

The worker never re-submits orders; it only reconciles state.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import structlog

from execution.base import Executor, OrderRecord, OrderStatus, validate_transition
from execution.engine import ExecutionEngine

log: structlog.BoundLogger = structlog.get_logger(__name__)


class RecoveryWorker:
    """Recovers orders stuck in PENDING_SUBMIT state after a process crash.

    P0-3: On startup and periodically (every *check_interval_seconds*):

    1. Scan ``ExecutionEngine._orders`` for PENDING_SUBMIT records older than
       *stale_threshold_seconds*.
    2. For each stale record, query the executor with ``get_order_status``.
    3. If the executor returns a result: update the record to SUBMITTED or
       FILLED accordingly.
    4. If the executor raises ``KeyError`` (order unknown): mark CANCELLED.

    Args:
        engine: The :class:`~execution.engine.ExecutionEngine` whose order
            registry is scanned.
        executor: The :class:`~execution.base.Executor` used to query live
            order state.
        stale_threshold_seconds: Age in seconds beyond which a PENDING_SUBMIT
            order is considered stale and eligible for recovery.
        check_interval_seconds: Interval between periodic recovery scans.

    Example::

        worker = RecoveryWorker(engine=engine, executor=paper)
        await worker.recover_stale_orders()   # one-shot on startup
        asyncio.create_task(worker.run_periodic())  # background loop
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        executor: Executor,
        stale_threshold_seconds: float = 60.0,
        check_interval_seconds: float = 300.0,
    ) -> None:
        self._engine: ExecutionEngine = engine
        self._executor: Executor = executor
        self._stale_threshold: float = stale_threshold_seconds
        self._check_interval: float = check_interval_seconds
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recover_stale_orders(self) -> int:
        """Perform a one-shot recovery pass over all stale PENDING_SUBMIT orders.

        Iterates the engine's in-memory order registry, identifies records in
        ``PENDING_SUBMIT`` that are older than *stale_threshold_seconds*, and
        reconciles each one against the executor.

        Returns:
            The number of orders that were recovered (transitioned out of
            ``PENDING_SUBMIT``).
        """
        now = datetime.now(UTC)
        stale: list[OrderRecord] = [
            record
            for record in self._engine._orders.values()
            if record.status == OrderStatus.PENDING_SUBMIT
            and (now - record.created_at).total_seconds() >= self._stale_threshold
        ]

        if not stale:
            log.debug("recovery_worker.no_stale_orders")
            return 0

        log.info(
            "recovery_worker.recovery_start",
            stale_count=len(stale),
            threshold_seconds=self._stale_threshold,
        )

        recovered = 0
        for record in stale:
            try:
                result = await self._executor.get_order_status(record.client_order_id)
            except KeyError:
                # Order was never received by the exchange.
                log.warning(
                    "recovery_worker.order_unknown_on_exchange",
                    client_order_id=record.client_order_id,
                    action="cancelling",
                )
                self._apply_transition(record, OrderStatus.CANCELLED)
                recovered += 1
            except Exception as exc:
                log.error(
                    "recovery_worker.status_query_failed",
                    client_order_id=record.client_order_id,
                    error=str(exc),
                )
                # Leave in PENDING_SUBMIT; retry on next pass.
            else:
                # Executor returned a result — sync the record's state.
                target = self._map_executor_status(result.status)
                if target is not None:
                    if result.status == OrderStatus.FILLED:
                        record.filled_price = result.filled_price
                        record.filled_quantity = result.filled_quantity
                        record.fees = result.fees
                        record.exchange_order_id = result.exchange_order_id
                        # Move through SUBMITTED first if needed
                        if validate_transition(record.status, OrderStatus.SUBMITTED):
                            self._apply_transition(record, OrderStatus.SUBMITTED)
                    self._apply_transition(record, target)
                    recovered += 1

                    log.info(
                        "recovery_worker.order_recovered",
                        client_order_id=record.client_order_id,
                        new_status=target,
                    )

        log.info(
            "recovery_worker.recovery_complete",
            recovered=recovered,
            total_stale=len(stale),
        )
        return recovered

    async def run_periodic(self) -> None:
        """Run recovery scans periodically until :meth:`stop` is called.

        Intended to be launched as an ``asyncio.Task`` during bot startup.

        The first scan runs immediately (to catch orders stale from a prior
        crash), then repeats every *check_interval_seconds*.
        """
        self._running = True
        log.info(
            "recovery_worker.started",
            check_interval_seconds=self._check_interval,
            stale_threshold_seconds=self._stale_threshold,
        )

        while self._running:
            try:
                await self.recover_stale_orders()
            except Exception as exc:
                log.error(
                    "recovery_worker.unexpected_error",
                    error=str(exc),
                )
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

        log.info("recovery_worker.stopped")

    async def stop(self) -> None:
        """Signal the periodic loop to stop on the next iteration."""
        self._running = False
        log.info("recovery_worker.stop_requested")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_transition(self, record: OrderRecord, target: OrderStatus) -> None:
        """Transition *record* to *target* with validation and audit logging.

        Args:
            record: The :class:`~execution.base.OrderRecord` to update.
            target: Desired next :class:`~execution.base.OrderStatus`.
        """
        if not validate_transition(record.status, target):
            log.critical(
                "recovery_worker.invalid_transition",
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
    def _map_executor_status(status: OrderStatus) -> OrderStatus | None:
        """Map an executor-reported status to the target recovery status.

        SUBMITTED and PARTIAL are treated as SUBMITTED (still open on exchange).
        FILLED is mapped directly.  Terminal states that are not recoverable
        (PENDING_SUBMIT itself, REJECTED) return ``None``.

        Args:
            status: The :class:`OrderStatus` returned by the executor.

        Returns:
            The :class:`OrderStatus` to apply, or ``None`` to skip.
        """
        mapping: dict[OrderStatus, OrderStatus] = {
            OrderStatus.SUBMITTED: OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL: OrderStatus.SUBMITTED,
            OrderStatus.FILLED: OrderStatus.FILLED,
            OrderStatus.CANCELLED: OrderStatus.CANCELLED,
            OrderStatus.EXPIRED: OrderStatus.EXPIRED,
        }
        return mapping.get(status)
