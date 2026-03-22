"""Tests for the ledger subsystem: Database, TradeLogger, PnLTracker, TCAAnalyzer.

All database tests run against an in-memory SQLite instance via aiosqlite
(``sqlite+aiosqlite://``).  No files are created on disk and no external
services are required.

Test organisation
-----------------
- :class:`TestDatabase`     — engine lifecycle, table creation, session factory.
- :class:`TestTradeLogger`  — signal, order, order-update, and regime logging.
- :class:`TestPnLTracker`   — FIFO realisation, win/loss stats, drawdown, reset.
- :class:`TestTCAAnalyzer`  — slippage sign convention, fee accumulation, averages.
"""

from __future__ import annotations

import json
import math
import time

import pytest
from sqlalchemy import select

from ledger.database import Database
from ledger.models import OrderRecord, RegimeRecord, RiskDecisionRecord, SignalRecord
from ledger.pnl_tracker import PnLTracker
from ledger.tca import TCAAnalyzer
from ledger.trade_logger import TradeLogger
from risk.base import ApprovedOrder, RiskDecision, _compute_gate_token
from signals.base import Signal, SignalDirection

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

IN_MEMORY_URL = "sqlite+aiosqlite://"


def _make_db() -> Database:
    """Return a fresh in-memory :class:`Database` instance."""
    return Database(IN_MEMORY_URL)


def _make_signal(
    pair: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.LONG,
    strength: float = 0.75,
    indicator_name: str = "ema_crossover",
    timestamp: int = 1_700_000_000_000,
    metadata: dict[str, float] | None = None,
) -> Signal:
    """Build a :class:`~signals.base.Signal` for tests."""
    return Signal(
        pair=pair,
        direction=direction,
        strength=strength,
        indicator_name=indicator_name,
        timestamp=timestamp,
        metadata=metadata or {"ema_fast": 50_100.0, "ema_slow": 49_900.0},
    )


def _make_risk_decision(
    approved: bool = True,
    reason: str = "all checks passed",
    adjusted_quantity: float | None = None,
    checks_passed: list[str] | None = None,
    checks_failed: list[str] | None = None,
) -> RiskDecision:
    """Build a :class:`~risk.base.RiskDecision` for tests."""
    return RiskDecision(
        approved=approved,
        reason=reason,
        adjusted_quantity=adjusted_quantity,
        checks_passed=checks_passed or ["kill_switch", "daily_loss", "position_limit"],
        checks_failed=checks_failed or [],
    )


def _make_approved_order(
    signal_id: str = "sig-test-0001",
    pair: str = "BTC/USDT",
    side: str = "buy",
    quantity: float = 0.01,
) -> ApprovedOrder:
    """Build a genuine :class:`~risk.base.ApprovedOrder` with a valid HMAC token."""
    ts = time.time()
    token = _compute_gate_token(signal_id, pair, side, quantity, ts)
    return ApprovedOrder(
        signal_id=signal_id,
        pair=pair,
        side=side,
        quantity=quantity,
        approved_at=ts,
        gate_token=token,
        risk_checks_passed=["kill_switch", "daily_loss", "position_limit"],
        original_signal_strength=0.75,
    )


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------


class TestDatabase:
    """Tests for :class:`~ledger.database.Database` lifecycle and session factory."""

    async def test_init_creates_tables(self) -> None:
        """After init(), all ORM tables must exist and be writable."""
        db = _make_db()
        await db.init()

        # Verify each table by inserting a row into it.
        async with db.session() as sess:
            for record in [
                SignalRecord(
                    pair="BTC/USDT",
                    direction="long",
                    strength=0.8,
                    indicator_name="test",
                    signal_timestamp_ms=1_700_000_000_000,
                ),
                RegimeRecord(
                    regime="trending",
                    confidence=0.9,
                    raw_response="",
                ),
            ]:
                sess.add(record)
            await sess.commit()

        await db.close()

    async def test_init_is_idempotent(self) -> None:
        """Calling init() twice must not raise or create duplicate tables."""
        db = _make_db()
        await db.init()
        await db.init()  # second call is a no-op
        await db.close()

    async def test_session_raises_before_init(self) -> None:
        """Calling session() before init() must raise RuntimeError."""
        db = _make_db()
        with pytest.raises(RuntimeError, match="init"):
            async with db.session():
                pass

    async def test_session_works_after_init(self) -> None:
        """session() must return a working AsyncSession after init()."""
        db = _make_db()
        await db.init()

        async with db.session() as sess:
            record = SignalRecord(
                pair="ETH/USDT",
                direction="short",
                strength=0.5,
                indicator_name="rsi",
                signal_timestamp_ms=1_700_000_001_000,
            )
            sess.add(record)
            await sess.commit()
            await sess.refresh(record)
            assert record.id is not None

        await db.close()

    async def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times must not raise."""
        db = _make_db()
        await db.init()
        await db.close()
        await db.close()  # second close is a no-op


# ---------------------------------------------------------------------------
# TradeLogger tests
# ---------------------------------------------------------------------------


class TestTradeLogger:
    """Tests for :class:`~ledger.trade_logger.TradeLogger`."""

    async def test_log_signal_returns_id(self) -> None:
        """log_signal() must return a positive integer DB row ID."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        signal = _make_signal()
        record_id = await logger.log_signal(signal)

        assert isinstance(record_id, int)
        assert record_id > 0
        await db.close()

    async def test_log_signal_persists_fields(self) -> None:
        """The persisted SignalRecord must contain all fields from the Signal."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        signal = _make_signal(
            pair="ETH/USDT",
            direction=SignalDirection.SHORT,
            strength=0.6,
            indicator_name="rsi_14",
            timestamp=1_700_000_002_000,
            metadata={"rsi": 72.5},
        )
        record_id = await logger.log_signal(signal)

        async with db.session() as sess:
            row = await sess.get(SignalRecord, record_id)
            assert row is not None
            assert row.pair == "ETH/USDT"
            assert row.direction == "short"
            assert row.strength == pytest.approx(0.6)
            assert row.indicator_name == "rsi_14"
            assert row.signal_timestamp_ms == 1_700_000_002_000
            assert json.loads(row.metadata_json) == {"rsi": 72.5}

        await db.close()

    async def test_log_risk_decision_persists_fields(self) -> None:
        """The persisted RiskDecisionRecord must be linked to the signal ID."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        signal_db_id = await logger.log_signal(_make_signal())
        decision = _make_risk_decision(
            approved=False,
            reason="daily loss limit breached",
            checks_passed=["kill_switch"],
            checks_failed=["daily_loss"],
        )
        risk_id = await logger.log_risk_decision(signal_db_id, decision)

        async with db.session() as sess:
            row = await sess.get(RiskDecisionRecord, risk_id)
            assert row is not None
            assert row.signal_id == signal_db_id
            assert row.approved is False
            assert row.reason == "daily loss limit breached"
            assert json.loads(row.checks_passed_json) == ["kill_switch"]
            assert json.loads(row.checks_failed_json) == ["daily_loss"]

        await db.close()

    async def test_log_order_returns_id(self) -> None:
        """log_order() must return a positive integer DB row ID."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        order = _make_approved_order()
        record_id = await logger.log_order(order, "coid-001", "pending_submit")

        assert isinstance(record_id, int)
        assert record_id > 0
        await db.close()

    async def test_log_order_persists_fields(self) -> None:
        """The persisted OrderRecord must reflect all fields from the ApprovedOrder."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        order = _make_approved_order(
            signal_id="sig-abc-1234",
            pair="BTC/USDT",
            side="buy",
            quantity=0.05,
        )
        record_id = await logger.log_order(order, "coid-abc-001", "pending_submit")

        async with db.session() as sess:
            row = await sess.get(OrderRecord, record_id)
            assert row is not None
            assert row.client_order_id == "coid-abc-001"
            assert row.signal_id == "sig-abc-1234"
            assert row.pair == "BTC/USDT"
            assert row.side == "buy"
            assert row.requested_quantity == pytest.approx(0.05)
            assert row.status == "pending_submit"
            transitions = json.loads(row.transitions_json)
            assert len(transitions) == 1
            assert transitions[0][0] == "pending_submit"

        await db.close()

    async def test_log_order_unique_constraint(self) -> None:
        """Inserting two orders with the same client_order_id must raise IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        order = _make_approved_order()
        await logger.log_order(order, "duplicate-coid", "pending_submit")

        with pytest.raises(IntegrityError):
            await logger.log_order(order, "duplicate-coid", "pending_submit")

        await db.close()

    async def test_log_order_update_status(self) -> None:
        """log_order_update() must update the status and fill fields."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        order = _make_approved_order()
        await logger.log_order(order, "coid-update-001", "pending_submit")

        await logger.log_order_update(
            client_order_id="coid-update-001",
            status="filled",
            filled_qty=0.01,
            filled_price=50_500.0,
            fees=0.505,
        )

        async with db.session() as sess:
            result = await sess.execute(
                select(OrderRecord).where(
                    OrderRecord.client_order_id == "coid-update-001"
                )
            )
            row = result.scalar_one()
            assert row.status == "filled"
            assert row.filled_quantity == pytest.approx(0.01)
            assert row.filled_price == pytest.approx(50_500.0)
            assert row.fees == pytest.approx(0.505)

        await db.close()

    async def test_log_order_update_appends_transitions(self) -> None:
        """Each call to log_order_update() must append an entry to transitions_json."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        order = _make_approved_order()
        await logger.log_order(order, "coid-trans-001", "pending_submit")
        await logger.log_order_update("coid-trans-001", "submitted", 0.0, 0.0, 0.0)
        await logger.log_order_update("coid-trans-001", "filled", 0.01, 50_000.0, 0.5)

        async with db.session() as sess:
            result = await sess.execute(
                select(OrderRecord).where(
                    OrderRecord.client_order_id == "coid-trans-001"
                )
            )
            row = result.scalar_one()
            transitions = json.loads(row.transitions_json)
            statuses = [t[0] for t in transitions]
            assert statuses == ["pending_submit", "submitted", "filled"]

        await db.close()

    async def test_log_order_update_missing_order(self) -> None:
        """log_order_update() on a non-existent order must not raise."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        # Should log a warning and return without raising.
        await logger.log_order_update(
            "nonexistent-coid", "filled", 0.01, 50_000.0, 0.5
        )
        await db.close()

    async def test_log_regime_persists_fields(self) -> None:
        """The persisted RegimeRecord must reflect all arguments."""
        db = _make_db()
        await db.init()
        logger = TradeLogger(db)

        regime_id = await logger.log_regime(
            regime="trending",
            confidence=0.88,
            raw_response="Market shows strong trend...",
            active_pairs=["BTC/USDT", "ETH/USDT"],
            active_strategies=["ema_crossover"],
        )

        async with db.session() as sess:
            row = await sess.get(RegimeRecord, regime_id)
            assert row is not None
            assert row.regime == "trending"
            assert row.confidence == pytest.approx(0.88)
            assert row.raw_response == "Market shows strong trend..."
            assert json.loads(row.active_pairs_json) == ["BTC/USDT", "ETH/USDT"]
            assert json.loads(row.active_strategies_json) == ["ema_crossover"]

        await db.close()


# ---------------------------------------------------------------------------
# PnLTracker tests
# ---------------------------------------------------------------------------


class TestPnLTracker:
    """Tests for :class:`~ledger.pnl_tracker.PnLTracker`."""

    def test_buy_then_sell_profit(self) -> None:
        """Buy at 50_000, sell at 55_000 -> positive realised PnL."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 55_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.realized_pnl == pytest.approx(5_000.0)
        assert summary.win_count == 1
        assert summary.loss_count == 0

    def test_buy_then_sell_loss(self) -> None:
        """Buy at 50_000, sell at 48_000 -> negative realised PnL."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 48_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.realized_pnl == pytest.approx(-2_000.0)
        assert summary.win_count == 0
        assert summary.loss_count == 1

    def test_fees_reduce_realised_pnl(self) -> None:
        """Fees must be subtracted from realised PnL on the sell leg."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 55_000.0, 50.0)

        summary = tracker.get_summary()
        assert summary.realized_pnl == pytest.approx(5_000.0 - 50.0)

    def test_win_rate_two_wins_one_loss(self) -> None:
        """Win rate must be 2/3 when there are two wins and one loss."""
        tracker = PnLTracker()
        # Win 1
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 51_000.0, 0.0)
        # Win 2
        tracker.record_trade("ETH/USDT", "buy", 1.0, 2_000.0, 0.0)
        tracker.record_trade("ETH/USDT", "sell", 1.0, 2_100.0, 0.0)
        # Loss
        tracker.record_trade("BTC/USDT", "buy", 1.0, 52_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 51_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.total_trades == 3
        assert summary.win_count == 2
        assert summary.loss_count == 1
        assert summary.win_rate == pytest.approx(2 / 3)

    def test_win_rate_no_trades(self) -> None:
        """Win rate must be 0.0 when no trades have been recorded."""
        tracker = PnLTracker()
        summary = tracker.get_summary()
        assert summary.win_rate == 0.0
        assert summary.total_trades == 0

    def test_max_drawdown_tracks_peak_decline(self) -> None:
        """Max drawdown must record the largest drop from any PnL peak.

        Sequence:
          Trade 1: +1000  -> peak=1000, dd=0
          Trade 2: -500   -> current=500,  dd=500  -> max_dd=500
          Trade 3: +800   -> current=1300, peak=1300, dd=0
          Trade 4: -1100  -> current=200,  dd=1100 -> max_dd=1100
        """
        tracker = PnLTracker()

        def _round_trip(pair: str, buy: float, sell: float) -> None:
            tracker.record_trade(pair, "buy", 1.0, buy, 0.0)
            tracker.record_trade(pair, "sell", 1.0, sell, 0.0)

        _round_trip("BTC/USDT", 0.0, 1_000.0)   # +1000
        _round_trip("BTC/USDT", 0.0, -500.0)     # -500 (sell below cost)
        _round_trip("BTC/USDT", 0.0, 800.0)      # +800
        _round_trip("BTC/USDT", 0.0, -1_100.0)   # -1100

        summary = tracker.get_summary()
        assert summary.max_drawdown == pytest.approx(1_100.0)

    def test_max_drawdown_no_decline(self) -> None:
        """Max drawdown must be 0.0 if the PnL only ever increases."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 51_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "buy", 1.0, 51_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 52_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.max_drawdown == pytest.approx(0.0)

    def test_profit_factor_wins_and_losses(self) -> None:
        """Profit factor = gross_profit / gross_loss."""
        tracker = PnLTracker()
        # +3000 win
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 53_000.0, 0.0)
        # -1000 loss
        tracker.record_trade("BTC/USDT", "buy", 1.0, 53_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 52_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.profit_factor == pytest.approx(3.0)

    def test_profit_factor_no_losses(self) -> None:
        """Profit factor must be math.inf when there are no losing trades."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 51_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.profit_factor == math.inf

    def test_profit_factor_no_trades(self) -> None:
        """Profit factor must be 0.0 when no trades have been recorded."""
        tracker = PnLTracker()
        summary = tracker.get_summary()
        assert summary.profit_factor == 0.0

    def test_unrealized_pnl_with_open_position(self) -> None:
        """Unrealised PnL must reflect open lots at the current market price."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 2.0, 50_000.0, 0.0)
        tracker.update_market_price("BTC/USDT", 52_000.0)

        summary = tracker.get_summary()
        # (52_000 - 50_000) * 2 = 4_000
        assert summary.unrealized_pnl == pytest.approx(4_000.0)
        assert summary.total_pnl == pytest.approx(4_000.0)

    def test_unrealized_pnl_no_market_price(self) -> None:
        """Unrealised PnL must be 0.0 if no market price has been set."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)

        summary = tracker.get_summary()
        assert summary.unrealized_pnl == pytest.approx(0.0)

    def test_unrealized_pnl_negative(self) -> None:
        """Unrealised PnL must be negative when the market is below entry."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.update_market_price("BTC/USDT", 48_000.0)

        summary = tracker.get_summary()
        assert summary.unrealized_pnl == pytest.approx(-2_000.0)

    def test_fifo_partial_lot_consumption(self) -> None:
        """A sell that partially consumes the first lot must leave a correct remainder."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 3.0, 50_000.0, 0.0)
        # Sell only 1 of the 3 BTC
        tracker.record_trade("BTC/USDT", "sell", 1.0, 55_000.0, 0.0)

        # Remaining open: 2 BTC @ 50_000
        tracker.update_market_price("BTC/USDT", 55_000.0)
        summary = tracker.get_summary()

        assert summary.realized_pnl == pytest.approx(5_000.0)
        assert summary.unrealized_pnl == pytest.approx(10_000.0)  # 2 * 5_000

    def test_fifo_two_lots(self) -> None:
        """A sell spanning two lots must consume them in order (oldest first)."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)  # lot 1
        tracker.record_trade("BTC/USDT", "buy", 1.0, 52_000.0, 0.0)  # lot 2
        # Sell 2 BTC: consumes lot1 first, then lot2
        tracker.record_trade("BTC/USDT", "sell", 2.0, 54_000.0, 0.0)

        summary = tracker.get_summary()
        # PnL = (54k-50k)*1 + (54k-52k)*1 = 4000 + 2000 = 6000
        assert summary.realized_pnl == pytest.approx(6_000.0)

    def test_reset_daily_clears_daily_pnl(self) -> None:
        """reset_daily() must zero the daily realized PnL accumulator."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 51_000.0, 0.0)

        assert tracker.get_daily_pnl() == pytest.approx(1_000.0)

        tracker.reset_daily()
        assert tracker.get_daily_pnl() == pytest.approx(0.0)

    def test_reset_daily_preserves_lifetime_pnl(self) -> None:
        """reset_daily() must not affect the lifetime realized_pnl."""
        tracker = PnLTracker()
        tracker.record_trade("BTC/USDT", "buy", 1.0, 50_000.0, 0.0)
        tracker.record_trade("BTC/USDT", "sell", 1.0, 51_000.0, 0.0)
        tracker.reset_daily()

        summary = tracker.get_summary()
        assert summary.realized_pnl == pytest.approx(1_000.0)


# ---------------------------------------------------------------------------
# TCAAnalyzer tests
# ---------------------------------------------------------------------------


class TestTCAAnalyzer:
    """Tests for :class:`~ledger.tca.TCAAnalyzer`."""

    def test_no_executions_returns_zero_metrics(self) -> None:
        """get_metrics() on an empty analyzer must return zeroed TCAMetrics."""
        tca = TCAAnalyzer()
        metrics = tca.get_metrics()

        assert metrics.avg_slippage_bps == pytest.approx(0.0)
        assert metrics.total_fees_usd == pytest.approx(0.0)
        assert metrics.avg_market_impact_bps == pytest.approx(0.0)
        assert metrics.total_trades == 0

    def test_positive_slippage_buy(self) -> None:
        """A buy that fills above expected price must produce positive slippage_bps."""
        tca = TCAAnalyzer()
        # Expected 50_000, actual 50_010 -> delta=10
        # (50_010 - 50_000) / 50_000 * 10_000 = 2.0 bps
        tca.record_execution(
            expected_price=50_000.0,
            actual_price=50_010.0,
            quantity=1.0,
            fees=0.0,
            side="buy",
        )
        metrics = tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(2.0)
        assert metrics.total_trades == 1

    def test_negative_slippage_buy(self) -> None:
        """A buy that fills below expected price must produce negative slippage_bps."""
        tca = TCAAnalyzer()
        # Expected 50_000, actual 49_995 -> delta=-5
        # (49_995 - 50_000) / 50_000 * 10_000 = -1.0 bps
        tca.record_execution(
            expected_price=50_000.0,
            actual_price=49_995.0,
            quantity=1.0,
            fees=0.0,
            side="buy",
        )
        metrics = tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(-1.0)

    def test_positive_slippage_sell(self) -> None:
        """A sell that fills below expected price must produce positive slippage_bps."""
        tca = TCAAnalyzer()
        # Expected 50_000, actual 49_990 -> delta=-10
        # (50_000 - 49_990) / 50_000 * 10_000 = 2.0 bps adverse on a sell
        tca.record_execution(
            expected_price=50_000.0,
            actual_price=49_990.0,
            quantity=1.0,
            fees=0.0,
            side="sell",
        )
        metrics = tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(2.0)

    def test_negative_slippage_sell(self) -> None:
        """A sell that fills above expected price (lucky) must produce negative slippage_bps."""
        tca = TCAAnalyzer()
        # Expected 50_000, actual 50_010 -> delta=+10
        # (50_000 - 50_010) / 50_000 * 10_000 = -2.0 bps
        tca.record_execution(
            expected_price=50_000.0,
            actual_price=50_010.0,
            quantity=1.0,
            fees=0.0,
            side="sell",
        )
        metrics = tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(-2.0)

    def test_avg_slippage_multiple_executions(self) -> None:
        """avg_slippage_bps must be the simple mean of per-execution slippages."""
        tca = TCAAnalyzer()
        # +40 bps: (50_200 - 50_000) / 50_000 * 10_000
        tca.record_execution(50_000.0, 50_200.0, 1.0, 0.0, "buy")
        # +20 bps: (50_100 - 50_000) / 50_000 * 10_000
        tca.record_execution(50_000.0, 50_100.0, 1.0, 0.0, "buy")
        # 0 bps
        tca.record_execution(50_000.0, 50_000.0, 1.0, 0.0, "buy")

        metrics = tca.get_metrics()
        # mean of [40, 20, 0] = 20 bps
        assert metrics.avg_slippage_bps == pytest.approx(20.0)
        assert metrics.total_trades == 3

    def test_fees_accumulated(self) -> None:
        """total_fees_usd must sum fees across all recorded executions."""
        tca = TCAAnalyzer()
        tca.record_execution(50_000.0, 50_000.0, 1.0, 50.0, "buy")
        tca.record_execution(50_000.0, 50_000.0, 1.0, 25.0, "sell")

        metrics = tca.get_metrics()
        assert metrics.total_fees_usd == pytest.approx(75.0)

    def test_market_impact_positive(self) -> None:
        """avg_market_impact_bps must be positive when there is a price deviation."""
        tca = TCAAnalyzer()
        tca.record_execution(50_000.0, 50_100.0, 1.0, 0.0, "buy")

        metrics = tca.get_metrics()
        assert metrics.avg_market_impact_bps > 0.0

    def test_market_impact_zero_for_perfect_fill(self) -> None:
        """avg_market_impact_bps must be 0.0 when actual == expected."""
        tca = TCAAnalyzer()
        tca.record_execution(50_000.0, 50_000.0, 1.0, 0.0, "buy")

        metrics = tca.get_metrics()
        assert metrics.avg_market_impact_bps == pytest.approx(0.0)

    def test_reset_clears_all_records(self) -> None:
        """After reset(), metrics must return to zero."""
        tca = TCAAnalyzer()
        tca.record_execution(50_000.0, 50_100.0, 1.0, 50.0, "buy")
        tca.reset()

        metrics = tca.get_metrics()
        assert metrics.total_trades == 0
        assert metrics.total_fees_usd == pytest.approx(0.0)

    def test_zero_expected_price_is_ignored(self) -> None:
        """record_execution() with expected_price=0.0 must not append a record."""
        tca = TCAAnalyzer()
        tca.record_execution(0.0, 50_000.0, 1.0, 50.0, "buy")

        metrics = tca.get_metrics()
        assert metrics.total_trades == 0
