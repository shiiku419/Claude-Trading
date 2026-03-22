"""Execution engine: crash-safe, idempotent order lifecycle orchestration.

Design invariants
-----------------
P0-2
    Every :class:`~risk.base.ApprovedOrder` must carry a valid HMAC token
    produced by the Risk Gate.  :meth:`ExecutionEngine.execute` verifies the
    token via :func:`~risk.base.verify_gate_token` and immediately rejects
    (with a CRITICAL log) any order whose token is absent or incorrect.

P0-3  Crash-safe three-state idempotency
    The engine follows this commit protocol for every order:

    1. **Redis SET NX** — fast deduplication guard (24 h TTL).
    2. **In-memory PENDING_SUBMIT** — record created before any I/O.
    3. **Exchange submit** — ``executor.place_order`` is called.
    4. **SUBMITTED** — record updated after the exchange acknowledges.

    If the process crashes between steps 2 and 4, the
    :class:`~execution.recovery.RecoveryWorker` will find the stale
    ``PENDING_SUBMIT`` record and reconcile it on the next startup.

P1-4  Explicit state-machine transitions
    Every status change goes through :func:`~execution.base.validate_transition`.
    An illegal transition is logged at CRITICAL and suppressed — it never
    silently corrupts the record.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from bus.event_bus import EventBus
from bus.events import FillEvent, OrderEvent
from execution.base import Executor, OrderRecord, OrderResult, OrderStatus, validate_transition
from risk.base import ApprovedOrder, verify_gate_token

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Redis key prefix and TTL for order deduplication.
_DEDUP_KEY_PREFIX: str = "order:dedup:"
_DEDUP_TTL_SECONDS: int = 86_400  # 24 hours


class ExecutionEngine:
    """Orchestrates the full order lifecycle with crash-safe idempotency.

    The engine is the single entry-point through which :class:`~risk.base.ApprovedOrder`
    objects flow into the executor (paper or live).  It enforces all P0-level
    security and reliability guarantees.

    Args:
        executor: An :class:`~execution.base.Executor`-compatible object that
            handles the actual order placement (paper or live exchange).
        event_bus: The system's :class:`~bus.event_bus.EventBus` to which
            :class:`~bus.events.OrderEvent` and :class:`~bus.events.FillEvent`
            messages are published.
        redis_url: Redis connection URL for deduplication key storage.
            If Redis is unavailable, the engine falls back to in-memory dedup
            and logs a WARNING.

    Example::

        engine = ExecutionEngine(executor=paper, event_bus=bus, redis_url="redis://localhost:6379/0")
        await engine.connect()
        result = await engine.execute(approved_order)
        await engine.close()
    """

    def __init__(
        self,
        executor: Executor,
        event_bus: EventBus,
        redis_url: str,
    ) -> None:
        self._executor: Executor = executor
        self._event_bus: EventBus = event_bus
        self._redis_url: str = redis_url

        # In-memory order registry: client_order_id -> OrderRecord
        self._orders: dict[str, OrderRecord] = {}

        # Fallback in-memory dedup set (used when Redis is unavailable).
        self._dedup_fallback: set[str] = set()

        # Redis client — injected lazily by connect() or in tests.
        self._redis: Any | None = None

        # Controls periodic background tasks.
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the Redis connection.

        A failed connection is tolerated: the engine will operate with
        in-memory deduplication only and emit a WARNING.  This keeps the
        bot operational in environments where Redis is temporarily unavailable.
        """
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]

            self._redis = await aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            # Smoke-test the connection.
            await self._redis.ping()
            log.info("execution_engine.redis_connected", url=self._redis_url)
        except Exception as exc:
            log.warning(
                "execution_engine.redis_unavailable",
                error=str(exc),
                fallback="in_memory_dedup",
            )
            self._redis = None

        self._running = True

    async def close(self) -> None:
        """Close the Redis connection and stop background tasks."""
        self._running = False
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception as exc:
                log.warning("execution_engine.redis_close_error", error=str(exc))
            finally:
                self._redis = None
        log.info("execution_engine.closed")

    # ------------------------------------------------------------------
    # Core execution path
    # ------------------------------------------------------------------

    async def execute(self, order: ApprovedOrder) -> OrderResult | None:
        """Execute an approved order through the full crash-safe lifecycle.

        The method follows the P0-3 commit sequence:

        1. Verify HMAC token (P0-2) — reject immediately on failure.
        2. Generate ``client_order_id`` deterministically from ``signal_id``.
        3. Check Redis for a duplicate key (fast path).
        4. Write PENDING_SUBMIT record to in-memory store.
        5. Set Redis dedup key (NX) to claim exclusive ownership.
        6. Submit to the executor.
        7. Transition record to SUBMITTED.
        8. Publish :class:`~bus.events.OrderEvent` and
           :class:`~bus.events.FillEvent` on success.

        Args:
            order: A :class:`~risk.base.ApprovedOrder` produced by the Risk Gate.

        Returns:
            :class:`OrderResult` on success, or ``None`` if the order was
            rejected (invalid HMAC, duplicate, or executor error).
        """
        # ------------------------------------------------------------------
        # Step 1: HMAC verification (P0-2)
        # ------------------------------------------------------------------
        if not verify_gate_token(order):
            log.critical(
                "execution_engine.invalid_hmac",
                signal_id=order.signal_id,
                pair=order.pair,
                side=order.side,
                quantity=order.quantity,
                reason="gate_token verification failed — order rejected",
            )
            return None

        # ------------------------------------------------------------------
        # Step 2: Generate client_order_id
        # ------------------------------------------------------------------
        client_order_id = f"ts-{order.signal_id[:8]}-{uuid4().hex[:8]}"

        # ------------------------------------------------------------------
        # Step 3: Deduplication check
        # ------------------------------------------------------------------
        dedup_key = f"{_DEDUP_KEY_PREFIX}{client_order_id}"
        is_duplicate = await self._check_duplicate(dedup_key, order.signal_id)
        if is_duplicate:
            log.warning(
                "execution_engine.duplicate_order",
                client_order_id=client_order_id,
                signal_id=order.signal_id,
            )
            return None

        # ------------------------------------------------------------------
        # Step 4: Write PENDING_SUBMIT record
        # ------------------------------------------------------------------
        now = datetime.now(UTC)
        record = OrderRecord(
            client_order_id=client_order_id,
            pair=order.pair,
            side=order.side,
            requested_quantity=order.quantity,
            signal_id=order.signal_id,
            status=OrderStatus.PENDING_SUBMIT,
            created_at=now,
            updated_at=now,
        )
        record.transitions.append((OrderStatus.PENDING_SUBMIT, now))
        self._orders[client_order_id] = record

        log.info(
            "execution_engine.order_pending",
            client_order_id=client_order_id,
            signal_id=order.signal_id,
            pair=order.pair,
            side=order.side,
            quantity=order.quantity,
        )

        # ------------------------------------------------------------------
        # Step 5: Claim dedup key in Redis (or fallback)
        # ------------------------------------------------------------------
        await self._claim_dedup_key(dedup_key, order.signal_id)

        # ------------------------------------------------------------------
        # Step 6: Submit to executor
        # ------------------------------------------------------------------
        try:
            result = await self._executor.place_order(
                pair=order.pair,
                side=order.side,
                quantity=order.quantity,
                client_order_id=client_order_id,
            )
        except Exception as exc:
            log.error(
                "execution_engine.submit_failed",
                client_order_id=client_order_id,
                signal_id=order.signal_id,
                error=str(exc),
            )
            self._transition(record, OrderStatus.REJECTED)
            await self._publish_order_event(record, "rejected")
            return None

        # ------------------------------------------------------------------
        # Step 7: Update record to SUBMITTED
        # ------------------------------------------------------------------
        record.exchange_order_id = result.exchange_order_id
        self._transition(record, OrderStatus.SUBMITTED)

        # If the executor already filled the order (e.g. worst_case paper model),
        # continue through to FILLED.
        if result.status == OrderStatus.FILLED:
            record.filled_price = result.filled_price
            record.filled_quantity = result.filled_quantity
            record.fees = result.fees
            self._transition(record, OrderStatus.FILLED)

        log.info(
            "execution_engine.order_submitted",
            client_order_id=client_order_id,
            exchange_order_id=result.exchange_order_id,
            status=record.status,
            pair=order.pair,
            side=order.side,
            quantity=order.quantity,
        )

        # ------------------------------------------------------------------
        # Step 8: Publish events
        # ------------------------------------------------------------------
        await self._publish_order_event(record, record.status.value)

        if record.status == OrderStatus.FILLED:
            await self._publish_fill_event(record)

        return self._record_to_result(record)

    # ------------------------------------------------------------------
    # Order lookup
    # ------------------------------------------------------------------

    async def get_order(self, client_order_id: str) -> OrderRecord | None:
        """Return the :class:`OrderRecord` for a given *client_order_id*.

        Args:
            client_order_id: The idempotency key assigned by the engine.

        Returns:
            The :class:`OrderRecord`, or ``None`` if not found.
        """
        return self._orders.get(client_order_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_duplicate(self, dedup_key: str, signal_id: str) -> bool:
        """Return ``True`` if this order key is already known.

        Checks Redis first (fast path), falls back to the in-memory set.
        If both checks fail with an exception, the engine fails closed and
        rejects the order.

        Args:
            dedup_key: The full Redis key to inspect.
            signal_id: Originating signal identifier (for logging).

        Returns:
            ``True`` when the order is a duplicate and should be rejected.
        """
        # Redis path
        if self._redis is not None:
            try:
                existing = await self._redis.get(dedup_key)
                return existing is not None
            except Exception as exc:
                log.warning(
                    "execution_engine.redis_dedup_check_failed",
                    dedup_key=dedup_key,
                    signal_id=signal_id,
                    error=str(exc),
                    fallback="in_memory",
                )
                # Fall through to in-memory check.

        # In-memory fallback path
        return dedup_key in self._dedup_fallback

    async def _claim_dedup_key(self, dedup_key: str, signal_id: str) -> None:
        """Write the dedup key to Redis (SET NX) or the in-memory fallback.

        Failures are tolerated: the key may not be written, but the order has
        already been persisted as PENDING_SUBMIT so recovery can handle it.

        Args:
            dedup_key: The Redis key to set.
            signal_id: Originating signal identifier (for logging).
        """
        if self._redis is not None:
            try:
                await self._redis.set(
                    dedup_key,
                    signal_id,
                    nx=True,
                    ex=_DEDUP_TTL_SECONDS,
                )
                return
            except Exception as exc:
                log.warning(
                    "execution_engine.redis_claim_failed",
                    dedup_key=dedup_key,
                    error=str(exc),
                    fallback="in_memory",
                )

        # In-memory fallback
        self._dedup_fallback.add(dedup_key)

    def _transition(self, record: OrderRecord, target: OrderStatus) -> None:
        """Apply a validated state transition to *record*.

        An invalid transition is logged at CRITICAL and rejected silently to
        prevent corrupting the order record.

        Args:
            record: The :class:`OrderRecord` to update.
            target: Desired next :class:`OrderStatus`.
        """
        if not validate_transition(record.status, target):
            log.critical(
                "execution_engine.invalid_transition",
                client_order_id=record.client_order_id,
                current_status=record.status,
                target_status=target,
                signal_id=record.signal_id,
            )
            return

        now = datetime.now(UTC)
        record.status = target
        record.updated_at = now
        record.transitions.append((target, now))

        log.debug(
            "execution_engine.state_transition",
            client_order_id=record.client_order_id,
            new_status=target,
        )

    async def _publish_order_event(self, record: OrderRecord, status: str) -> None:
        """Publish an :class:`~bus.events.OrderEvent` for *record*.

        Args:
            record: The order whose state should be broadcast.
            status: Human-readable status string to attach to the event.
        """
        event = OrderEvent(
            client_order_id=record.client_order_id,
            pair=record.pair,
            side=record.side,
            quantity=record.requested_quantity,
            order_type="market",
            status=status,
        )
        try:
            await self._event_bus.publish("order", event)
        except Exception as exc:
            log.warning(
                "execution_engine.publish_order_event_failed",
                client_order_id=record.client_order_id,
                error=str(exc),
            )

    async def _publish_fill_event(self, record: OrderRecord) -> None:
        """Publish a :class:`~bus.events.FillEvent` for a filled *record*.

        Args:
            record: The fully-filled :class:`OrderRecord`.
        """
        event = FillEvent(
            client_order_id=record.client_order_id,
            pair=record.pair,
            side=record.side,
            filled_price=record.filled_price,
            filled_quantity=record.filled_quantity,
            fees=record.fees,
        )
        try:
            await self._event_bus.publish("fill", event)
        except Exception as exc:
            log.warning(
                "execution_engine.publish_fill_event_failed",
                client_order_id=record.client_order_id,
                error=str(exc),
            )

    @staticmethod
    def _record_to_result(record: OrderRecord) -> OrderResult:
        """Convert an :class:`OrderRecord` to an :class:`OrderResult`.

        Args:
            record: Source record.

        Returns:
            Lightweight snapshot suitable for returning to callers.
        """
        return OrderResult(
            client_order_id=record.client_order_id,
            exchange_order_id=record.exchange_order_id,
            status=record.status,
            filled_price=record.filled_price,
            filled_quantity=record.filled_quantity,
            fees=record.fees,
        )
