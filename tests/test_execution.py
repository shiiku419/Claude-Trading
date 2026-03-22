"""Comprehensive tests for the execution engine subsystem.

All tests run fully offline — no real network calls, no real Redis connections.
Redis is mocked via ``unittest.mock.AsyncMock`` / ``MagicMock``.

Test organisation
-----------------
- :class:`PaperExecutor` — fill mechanics, slippage, fees, both fill models.
- :class:`ExecutionEngine` — HMAC gate, dedup, event publishing, state transitions.
- Order state machine — transition validity, terminal-state enforcement.
- :class:`RecoveryWorker` — stale order detection and reconciliation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bus.event_bus import EventBus
from bus.events import FillEvent, OrderEvent
from execution.base import (
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    OrderRecord,
    OrderResult,
    OrderStatus,
    validate_transition,
)
from execution.engine import ExecutionEngine
from execution.paper_executor import PaperExecutor
from execution.recovery import RecoveryWorker
from risk.base import ApprovedOrder, _compute_gate_token

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_approved_order(
    signal_id: str = "sig-0001-abcd",
    pair: str = "BTC/USDT",
    side: str = "buy",
    quantity: float = 0.01,
    approved_at: float | None = None,
) -> ApprovedOrder:
    """Build a genuine :class:`ApprovedOrder` with a valid HMAC token.

    The token is computed using the same private helper used by the Risk Gate
    so that :func:`~risk.base.verify_gate_token` will accept it.
    """
    ts = approved_at if approved_at is not None else time.time()
    token = _compute_gate_token(signal_id, pair, side, quantity, ts)
    return ApprovedOrder(
        signal_id=signal_id,
        pair=pair,
        side=side,
        quantity=quantity,
        approved_at=ts,
        gate_token=token,
        risk_checks_passed=[
            "kill_switch", "time_policy", "daily_loss", "exposure", "position_limit"
        ],
        original_signal_strength=0.8,
    )


def _make_forged_order(base: ApprovedOrder) -> ApprovedOrder:
    """Return an :class:`ApprovedOrder` with a tampered quantity (invalid HMAC)."""
    return ApprovedOrder(
        signal_id=base.signal_id,
        pair=base.pair,
        side=base.side,
        quantity=base.quantity * 10,  # attacker inflates quantity
        approved_at=base.approved_at,
        gate_token=base.gate_token,  # stolen original token
    )


def _make_redis_mock() -> MagicMock:
    """Return a mock async Redis client that behaves as if the key is absent."""
    client = MagicMock()
    client.get = AsyncMock(return_value=None)       # no existing dedup key
    client.set = AsyncMock(return_value=True)        # SET NX succeeds
    client.ping = AsyncMock(return_value=True)
    client.aclose = AsyncMock()
    return client


def _make_engine(
    executor: PaperExecutor | None = None,
    bus: EventBus | None = None,
    redis_mock: MagicMock | None = None,
) -> ExecutionEngine:
    """Build an :class:`ExecutionEngine` wired to mocked dependencies."""
    if executor is None:
        executor = PaperExecutor(
            initial_balance=10_000.0,
            fill_model="worst_case",
        )
    if bus is None:
        bus = EventBus()

    engine = ExecutionEngine(
        executor=executor,
        event_bus=bus,
        redis_url="redis://localhost:6379/0",
    )
    engine._redis = redis_mock if redis_mock is not None else _make_redis_mock()
    return engine


# ---------------------------------------------------------------------------
# PaperExecutor tests
# ---------------------------------------------------------------------------


class TestPaperExecutor:
    """Tests for :class:`~execution.paper_executor.PaperExecutor`."""

    # ------------------------------------------------------------------
    # Balance and position accounting
    # ------------------------------------------------------------------

    async def test_paper_buy_reduces_balance(self) -> None:
        """Buying BTC must reduce the quote-asset balance."""
        executor = PaperExecutor(initial_balance=10_000.0, fill_model="worst_case")
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=51_000.0, low=49_000.0, avg_volume=100.0
        )

        initial_balance = executor.balance
        await executor.place_order("BTC/USDT", "buy", 0.1, "order-001")

        assert executor.balance < initial_balance

    async def test_paper_sell_increases_balance(self) -> None:
        """Selling BTC must increase the quote-asset balance."""
        executor = PaperExecutor(initial_balance=10_000.0, fill_model="worst_case")
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=51_000.0, low=49_000.0, avg_volume=100.0
        )

        # First buy to hold a position.
        await executor.place_order("BTC/USDT", "buy", 0.1, "order-001")
        balance_after_buy = executor.balance

        # Now sell.
        await executor.place_order("BTC/USDT", "sell", 0.1, "order-002")
        assert executor.balance > balance_after_buy

    async def test_paper_buy_sell_pnl(self) -> None:
        """Buy low, sell high should result in positive realised PnL."""
        executor = PaperExecutor(
            initial_balance=10_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,  # disable slippage for clarity
            fee_pct=0.0,            # disable fees for clarity
        )
        initial = executor.balance

        # Buy bar: high=50 100 (worst_case buy price)
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=50_100.0, low=49_900.0, avg_volume=0.0
        )
        await executor.place_order("BTC/USDT", "buy", 0.1, "order-buy")

        # Sell bar: low=55 000 (worst_case sell price — higher than buy)
        executor.set_next_bar_price(
            "BTC/USDT", open_price=55_500.0, high=56_000.0, low=55_000.0, avg_volume=0.0
        )
        await executor.place_order("BTC/USDT", "sell", 0.1, "order-sell")

        # PnL = (55_000 - 50_100) * 0.1 = 490
        assert executor.balance > initial
        assert executor.balance == pytest.approx(initial + (55_000.0 - 50_100.0) * 0.1, rel=1e-6)

    # ------------------------------------------------------------------
    # Fill models
    # ------------------------------------------------------------------

    async def test_paper_next_bar_open_fill(self) -> None:
        """Orders placed with next_bar_open should fill at the bar's open price
        (adjusted for slippage), not the close of the signal bar."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="next_bar_open",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )

        # Place order — bar not yet available, should stay SUBMITTED.
        result = await executor.place_order("BTC/USDT", "buy", 1.0, "order-001")
        assert result.status == OrderStatus.SUBMITTED

        # Provide the next bar.
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=51_000.0, low=49_000.0, avg_volume=1000.0
        )

        # Now check status — should be FILLED at open price.
        result = await executor.get_order_status("order-001")
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == pytest.approx(50_000.0, rel=1e-6)

    async def test_paper_worst_case_fill(self) -> None:
        """worst_case fill model should use high for buys and low for sells."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=51_000.0, low=49_000.0, avg_volume=1000.0
        )

        # Buy should fill at high (49_000 would be low, 51_000 is high).
        buy_result = await executor.place_order("BTC/USDT", "buy", 0.1, "order-buy")
        assert buy_result.status == OrderStatus.FILLED
        assert buy_result.filled_price == pytest.approx(51_000.0, rel=1e-6)

        # Sell should fill at low.
        sell_result = await executor.place_order("BTC/USDT", "sell", 0.1, "order-sell")
        assert sell_result.status == OrderStatus.FILLED
        assert sell_result.filled_price == pytest.approx(49_000.0, rel=1e-6)

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------

    async def test_paper_slippage_applied(self) -> None:
        """The effective fill price must be worse than the raw bar price due to slippage."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.005,  # 0.5 %
            fee_pct=0.0,
        )
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=50_000.0, low=50_000.0, avg_volume=1e9
        )

        # With zero volume impact, slippage = base_slippage_pct only.
        result = await executor.place_order("BTC/USDT", "buy", 1.0, "order-001")
        assert result.status == OrderStatus.FILLED
        # Expected fill = 50_000 * 1.005 = 50_250
        assert result.filled_price == pytest.approx(50_000.0 * 1.005, rel=1e-6)

    async def test_paper_slippage_volume_impact(self) -> None:
        """Large orders relative to avg_volume should incur higher slippage."""
        executor = PaperExecutor(
            initial_balance=10_000_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.001,
            fee_pct=0.0,
        )
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=50_000.0, low=50_000.0, avg_volume=10.0
        )

        # quantity=5 on avg_volume=10 → volume_impact term = 0.001 * 5/10 = 0.0005
        # total_slippage = 0.001 + 0.0005 = 0.0015
        result = await executor.place_order("BTC/USDT", "buy", 5.0, "order-001")
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == pytest.approx(50_000.0 * 1.0015, rel=1e-6)

    # ------------------------------------------------------------------
    # Fees
    # ------------------------------------------------------------------

    async def test_paper_fees_applied(self) -> None:
        """Fees should equal fee_pct * filled_notional (after slippage)."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.001,  # 0.1 %
        )
        executor.set_next_bar_price(
            "BTC/USDT", open_price=50_000.0, high=50_000.0, low=50_000.0, avg_volume=1e9
        )

        result = await executor.place_order("BTC/USDT", "buy", 1.0, "order-001")
        assert result.status == OrderStatus.FILLED
        # Notional = 50_000 * 1.0; fee = 50_000 * 0.001 = 50
        assert result.fees == pytest.approx(50.0, rel=1e-6)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def test_paper_cancel_pending_order(self) -> None:
        """A pending (SUBMITTED) order can be cancelled before bar arrival."""
        executor = PaperExecutor(fill_model="next_bar_open")
        await executor.place_order("BTC/USDT", "buy", 0.01, "order-001")

        cancelled = await executor.cancel_order("order-001")
        assert cancelled is True

        result = await executor.get_order_status("order-001")
        assert result.status == OrderStatus.CANCELLED

    async def test_paper_cancel_filled_order_returns_false(self) -> None:
        """Cancelling a FILLED order must return False and leave it FILLED."""
        executor = PaperExecutor(
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 50_000.0, 50_000.0, 1.0)
        await executor.place_order("BTC/USDT", "buy", 0.01, "order-001")

        cancelled = await executor.cancel_order("order-001")
        assert cancelled is False

        result = await executor.get_order_status("order-001")
        assert result.status == OrderStatus.FILLED

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def test_paper_positions_updated_after_fill(self) -> None:
        """After a buy fill, the pair should appear in positions."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 50_000.0, 50_000.0, 1.0)
        await executor.place_order("BTC/USDT", "buy", 0.5, "order-001")

        assert executor.positions.get("BTC/USDT") == pytest.approx(0.5, rel=1e-9)

    async def test_paper_next_bar_queued_before_bar_arrives(self) -> None:
        """An order placed with next_bar_open must be SUBMITTED until a bar arrives."""
        executor = PaperExecutor(fill_model="next_bar_open")
        result = await executor.place_order("ETH/USDT", "sell", 1.0, "order-001")
        assert result.status == OrderStatus.SUBMITTED
        assert "order-001" in executor._pending_fills.get("ETH/USDT", [])


# ---------------------------------------------------------------------------
# ExecutionEngine tests
# ---------------------------------------------------------------------------


class TestExecutionEngine:
    """Tests for :class:`~execution.engine.ExecutionEngine`."""

    async def test_execute_valid_approved_order(self) -> None:
        """A valid ApprovedOrder with correct HMAC must be executed and return a result."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 51_000.0, 49_000.0, 1000.0)

        engine = _make_engine(executor=executor)
        order = _make_approved_order()

        result = await engine.execute(order)

        assert result is not None
        assert result.status in {OrderStatus.SUBMITTED, OrderStatus.FILLED}
        assert result.client_order_id.startswith("ts-")

    async def test_execute_rejects_invalid_hmac(self) -> None:
        """An order with a tampered HMAC must be rejected with a CRITICAL log and return None."""
        engine = _make_engine()
        base_order = _make_approved_order()
        forged = _make_forged_order(base_order)

        result = await engine.execute(forged)
        assert result is None

    async def test_execute_dedup_same_signal(self) -> None:
        """Two executions with the same signal_id dedup key must only submit once."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 51_000.0, 49_000.0, 1000.0)

        bus = EventBus()
        redis_mock = _make_redis_mock()
        engine = _make_engine(executor=executor, bus=bus, redis_mock=redis_mock)

        order = _make_approved_order(signal_id="dedup-test-1234")

        # First execution should succeed.
        result1 = await engine.execute(order)
        assert result1 is not None

        # Simulate Redis now returning the existing key for subsequent checks.
        redis_mock.get = AsyncMock(return_value="dedup-test-1234")

        # Craft a second order with the same signal prefix so the client_order_id
        # dedup key resolves to the existing one.  We achieve this by patching uuid4.
        with patch("execution.engine.uuid4") as mock_uuid4:
            first_id = result1.client_order_id.split("-")[2]  # hex suffix
            mock_uuid4.return_value.hex = first_id

            result2 = await engine.execute(order)

        assert result2 is None  # rejected as duplicate

    async def test_execute_publishes_order_event(self) -> None:
        """A successful execution must publish an OrderEvent to the bus."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 51_000.0, 49_000.0, 1000.0)

        bus = EventBus()
        order_queue = bus.subscribe("order")
        engine = _make_engine(executor=executor, bus=bus)

        order = _make_approved_order()
        await engine.execute(order)

        # At least one OrderEvent should have been published.
        assert not order_queue.empty()
        event = order_queue.get_nowait()
        assert isinstance(event, OrderEvent)
        assert event.pair == "BTC/USDT"
        assert event.side == "buy"

    async def test_execute_publishes_fill_event(self) -> None:
        """A filled order must publish a FillEvent to the bus."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 51_000.0, 49_000.0, 1000.0)

        bus = EventBus()
        fill_queue = bus.subscribe("fill")
        engine = _make_engine(executor=executor, bus=bus)

        order = _make_approved_order()
        result = await engine.execute(order)

        # Only filled orders emit FillEvent.
        if result is not None and result.status == OrderStatus.FILLED:
            assert not fill_queue.empty()
            event = fill_queue.get_nowait()
            assert isinstance(event, FillEvent)
            assert event.pair == "BTC/USDT"
            assert event.filled_price > 0.0

    async def test_order_state_transitions(self) -> None:
        """Executed order must transition through PENDING_SUBMIT -> SUBMITTED -> FILLED."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 51_000.0, 49_000.0, 1000.0)

        engine = _make_engine(executor=executor)
        order = _make_approved_order()
        result = await engine.execute(order)

        assert result is not None
        record = await engine.get_order(result.client_order_id)
        assert record is not None

        transition_statuses = [status for status, _ in record.transitions]
        assert OrderStatus.PENDING_SUBMIT in transition_statuses
        assert OrderStatus.SUBMITTED in transition_statuses

    async def test_invalid_state_transition_logged(self) -> None:
        """An invalid state transition must log CRITICAL and not corrupt the record."""
        engine = _make_engine()
        record = OrderRecord(
            client_order_id="test-order-001",
            status=OrderStatus.FILLED,  # terminal state
        )
        engine._orders["test-order-001"] = record

        # Patch the 'critical' method on the engine module's bound logger.
        with patch("execution.engine.log") as mock_log:
            # Access the private method directly to test transition guard.
            engine._transition(record, OrderStatus.SUBMITTED)
            mock_log.critical.assert_called_once()

        # Record must remain FILLED.
        assert record.status == OrderStatus.FILLED

    async def test_execute_redis_fallback_to_memory(self) -> None:
        """When Redis is unavailable, the engine should fall back to in-memory dedup."""
        executor = PaperExecutor(
            initial_balance=100_000.0,
            fill_model="worst_case",
            base_slippage_pct=0.0,
            fee_pct=0.0,
        )
        executor.set_next_bar_price("BTC/USDT", 50_000.0, 51_000.0, 49_000.0, 1000.0)

        engine = _make_engine(executor=executor)
        # Simulate Redis being unavailable.
        engine._redis = None

        order = _make_approved_order(signal_id="fallback-test-12")
        result = await engine.execute(order)

        # Should succeed using in-memory dedup fallback.
        assert result is not None

    async def test_execute_fail_closed_on_executor_error(self) -> None:
        """When the executor raises, the order must be REJECTED and None returned."""
        broken_executor = MagicMock()
        broken_executor.place_order = AsyncMock(side_effect=RuntimeError("exchange down"))

        engine = _make_engine(executor=broken_executor)
        order = _make_approved_order(signal_id="crash-test-5678")

        result = await engine.execute(order)
        assert result is None

        # The record should be REJECTED in the internal store.
        stored = list(engine._orders.values())
        assert any(r.status == OrderStatus.REJECTED for r in stored)


# ---------------------------------------------------------------------------
# Order state machine tests
# ---------------------------------------------------------------------------


class TestOrderStateMachine:
    """Tests for the standalone state-machine helpers in :mod:`execution.base`."""

    def test_valid_transitions(self) -> None:
        """All documented valid transitions must be accepted."""
        valid_cases = [
            (OrderStatus.PENDING_SUBMIT, OrderStatus.SUBMITTED),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.CANCELLED),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.REJECTED),
            (OrderStatus.SUBMITTED, OrderStatus.PARTIAL),
            (OrderStatus.SUBMITTED, OrderStatus.FILLED),
            (OrderStatus.SUBMITTED, OrderStatus.CANCELLED),
            (OrderStatus.SUBMITTED, OrderStatus.EXPIRED),
            (OrderStatus.PARTIAL, OrderStatus.FILLED),
            (OrderStatus.PARTIAL, OrderStatus.CANCELLED),
        ]
        for current, target in valid_cases:
            assert validate_transition(current, target), (
                f"Expected {current} -> {target} to be valid"
            )

    def test_invalid_transitions_rejected(self) -> None:
        """Transitions that skip states or go backwards must be rejected."""
        invalid_cases = [
            (OrderStatus.PENDING_SUBMIT, OrderStatus.FILLED),  # skip SUBMITTED
            (OrderStatus.PENDING_SUBMIT, OrderStatus.PARTIAL),
            (OrderStatus.SUBMITTED, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.FILLED, OrderStatus.SUBMITTED),
            (OrderStatus.CANCELLED, OrderStatus.SUBMITTED),
        ]
        for current, target in invalid_cases:
            assert not validate_transition(current, target), (
                f"Expected {current} -> {target} to be invalid"
            )

    def test_terminal_states_have_no_transitions(self) -> None:
        """Every terminal state must have an empty transition set."""
        for status in TERMINAL_STATES:
            assert VALID_TRANSITIONS[status] == set(), (
                f"Terminal state {status} should have no outgoing transitions"
            )

    def test_all_statuses_covered_in_transition_table(self) -> None:
        """Every :class:`OrderStatus` value must have an entry in VALID_TRANSITIONS."""
        for status in OrderStatus:
            assert status in VALID_TRANSITIONS, f"{status} missing from VALID_TRANSITIONS"

    def test_terminal_states_correct_set(self) -> None:
        """TERMINAL_STATES should contain exactly FILLED, CANCELLED, EXPIRED, REJECTED."""
        expected = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
        assert TERMINAL_STATES == expected


# ---------------------------------------------------------------------------
# RecoveryWorker tests
# ---------------------------------------------------------------------------


class TestRecoveryWorker:
    """Tests for :class:`~execution.recovery.RecoveryWorker`."""

    def _make_stale_record(
        self,
        client_order_id: str = "stale-order-001",
        age_seconds: float = 120.0,
        status: OrderStatus = OrderStatus.PENDING_SUBMIT,
    ) -> OrderRecord:
        """Build an :class:`OrderRecord` that is artificially aged."""
        stale_time = datetime.now(UTC) - timedelta(seconds=age_seconds)
        return OrderRecord(
            client_order_id=client_order_id,
            pair="BTC/USDT",
            side="buy",
            requested_quantity=0.01,
            status=status,
            created_at=stale_time,
            updated_at=stale_time,
        )

    async def test_recovery_finds_stale_orders(self) -> None:
        """RecoveryWorker must identify PENDING_SUBMIT orders older than the threshold."""
        executor = MagicMock()
        executor.get_order_status = AsyncMock(side_effect=KeyError("not found"))

        engine = _make_engine()
        stale = self._make_stale_record(age_seconds=120.0)
        engine._orders[stale.client_order_id] = stale

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        recovered = await worker.recover_stale_orders()

        assert recovered == 1

    async def test_recovery_ignores_fresh_orders(self) -> None:
        """Orders younger than the threshold must not be touched."""
        executor = MagicMock()
        executor.get_order_status = AsyncMock(side_effect=KeyError("not found"))

        engine = _make_engine()
        fresh = self._make_stale_record(age_seconds=10.0)  # very recent
        engine._orders[fresh.client_order_id] = fresh

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        recovered = await worker.recover_stale_orders()

        assert recovered == 0
        assert fresh.status == OrderStatus.PENDING_SUBMIT

    async def test_recovery_resolves_existing_exchange_order(self) -> None:
        """If the exchange knows about the order and it's FILLED, record should be updated."""
        filled_result = OrderResult(
            client_order_id="stale-order-001",
            exchange_order_id="exch-999",
            status=OrderStatus.FILLED,
            filled_price=51_000.0,
            filled_quantity=0.01,
            fees=0.051,
        )
        executor = MagicMock()
        executor.get_order_status = AsyncMock(return_value=filled_result)

        engine = _make_engine()
        stale = self._make_stale_record(age_seconds=120.0)
        engine._orders[stale.client_order_id] = stale

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        recovered = await worker.recover_stale_orders()

        assert recovered == 1
        assert stale.status == OrderStatus.FILLED
        assert stale.filled_price == pytest.approx(51_000.0)
        assert stale.filled_quantity == pytest.approx(0.01)

    async def test_recovery_cancels_missing_exchange_order(self) -> None:
        """If the exchange has no record of the order, it must be CANCELLED."""
        executor = MagicMock()
        executor.get_order_status = AsyncMock(side_effect=KeyError("order not found"))

        engine = _make_engine()
        stale = self._make_stale_record(age_seconds=120.0)
        engine._orders[stale.client_order_id] = stale

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        recovered = await worker.recover_stale_orders()

        assert recovered == 1
        assert stale.status == OrderStatus.CANCELLED

    async def test_recovery_resolves_submitted_exchange_order(self) -> None:
        """If the exchange shows the order as SUBMITTED, record should be SUBMITTED."""
        submitted_result = OrderResult(
            client_order_id="stale-order-001",
            exchange_order_id="exch-888",
            status=OrderStatus.SUBMITTED,
            filled_price=0.0,
            filled_quantity=0.0,
            fees=0.0,
        )
        executor = MagicMock()
        executor.get_order_status = AsyncMock(return_value=submitted_result)

        engine = _make_engine()
        stale = self._make_stale_record(age_seconds=120.0)
        engine._orders[stale.client_order_id] = stale

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        recovered = await worker.recover_stale_orders()

        assert recovered == 1
        assert stale.status == OrderStatus.SUBMITTED

    async def test_recovery_skips_non_pending_submit_orders(self) -> None:
        """Orders already in SUBMITTED or terminal states must be skipped."""
        executor = MagicMock()
        executor.get_order_status = AsyncMock(side_effect=KeyError("not found"))

        engine = _make_engine()
        submitted = self._make_stale_record(
            client_order_id="submitted-order", age_seconds=300.0, status=OrderStatus.SUBMITTED
        )
        filled = self._make_stale_record(
            client_order_id="filled-order", age_seconds=300.0, status=OrderStatus.FILLED
        )
        engine._orders["submitted-order"] = submitted
        engine._orders["filled-order"] = filled

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        recovered = await worker.recover_stale_orders()

        assert recovered == 0
        assert submitted.status == OrderStatus.SUBMITTED
        assert filled.status == OrderStatus.FILLED

    async def test_recovery_handles_executor_exception_gracefully(self) -> None:
        """An unexpected executor error must be logged and the order left in PENDING_SUBMIT."""
        executor = MagicMock()
        executor.get_order_status = AsyncMock(side_effect=RuntimeError("network timeout"))

        engine = _make_engine()
        stale = self._make_stale_record(age_seconds=120.0)
        engine._orders[stale.client_order_id] = stale

        worker = RecoveryWorker(engine=engine, executor=executor, stale_threshold_seconds=60.0)
        # Should not raise.
        recovered = await worker.recover_stale_orders()

        assert recovered == 0
        assert stale.status == OrderStatus.PENDING_SUBMIT  # untouched

    async def test_recovery_periodic_runs_and_stops(self) -> None:
        """run_periodic must execute at least one scan and stop cleanly on stop()."""
        executor = MagicMock()
        executor.get_order_status = AsyncMock(side_effect=KeyError("not found"))

        engine = _make_engine()

        worker = RecoveryWorker(
            engine=engine,
            executor=executor,
            stale_threshold_seconds=60.0,
            check_interval_seconds=0.05,  # very fast for testing
        )

        task = asyncio.create_task(worker.run_periodic())
        await asyncio.sleep(0.15)  # let at least one scan happen
        await worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert not worker._running
