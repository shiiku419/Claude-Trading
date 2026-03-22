"""Comprehensive tests for the risk gate pipeline.

All tests are fully offline — no real Redis connections are made.
Redis is mocked via :class:`unittest.mock.AsyncMock` / :class:`unittest.mock.MagicMock`.

Test organisation:
- :class:`KillSwitch` — activation, deactivation, fail-closed behaviour.
- :class:`DailyLossCheck` — thresholds, hysteresis, daily reset.
- :class:`PositionLimitCheck` — size cap, open-count cap.
- :class:`ExposureCheck` — total exposure cap.
- :class:`RiskGate` — full pipeline, HMAC integrity, forged token rejection.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from risk.base import ApprovedOrder, PortfolioState, verify_gate_token
from risk.daily_loss import DailyLossCheck
from risk.exposure import ExposureCheck
from risk.gate import RiskGate
from risk.kill_switch import KillSwitch
from risk.position_limit import PositionLimitCheck
from risk.time_policy import TimePolicyCheck
from signals.base import Signal, SignalDirection

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_signal(
    pair: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.LONG,
    strength: float = 0.8,
) -> Signal:
    """Build a minimal :class:`Signal` for testing."""
    return Signal(
        pair=pair,
        direction=direction,
        strength=strength,
        indicator_name="test_indicator",
        timestamp=1_700_000_000_000,
    )


def _make_portfolio(
    balance: float = 100_000.0,
    open_positions: dict[str, float] | None = None,
    open_position_values: dict[str, float] | None = None,
    daily_realized_pnl: float = 0.0,
    daily_unrealized_pnl: float = 0.0,
) -> PortfolioState:
    """Build a :class:`PortfolioState` with sensible defaults."""
    return PortfolioState(
        total_balance_usd=balance,
        open_positions=open_positions or {},
        open_position_values=open_position_values or {},
        daily_realized_pnl=daily_realized_pnl,
        daily_unrealized_pnl=daily_unrealized_pnl,
    )


def _make_redis_mock(active: bool = False) -> MagicMock:
    """Build a mock aioredis client that returns the given kill-switch state."""
    client = MagicMock()
    # .get() is awaitable and returns "1" or "0"
    client.get = AsyncMock(side_effect=lambda key: "1" if active else "0")
    client.aclose = AsyncMock()

    # pipeline() returns a context manager whose __aenter__ gives a pipe object
    pipe = MagicMock()
    pipe.set = MagicMock(return_value=pipe)
    pipe.delete = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=["OK", "OK", "OK"])
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    client.pipeline = MagicMock(return_value=pipe)
    return client


async def _make_kill_switch(active: bool = False) -> KillSwitch:
    """Build a :class:`KillSwitch` with a mocked Redis client."""
    ks = KillSwitch(redis_url="redis://localhost:6379/0")
    ks._client = _make_redis_mock(active=active)
    return ks


def _make_time_policy(blocked_hours: list[int] | None = None) -> TimePolicyCheck:
    """Build a :class:`TimePolicyCheck` that never blocks (empty blocked_hours)."""
    return TimePolicyCheck(
        blocked_hours=blocked_hours if blocked_hours is not None else [],
        candle_cooldown_seconds=0,
    )


async def _make_gate(
    kill_switch_active: bool = False,
    blocked_hours: list[int] | None = None,
    warning_pct: float = 0.03,
    critical_pct: float = 0.05,
    consecutive_breaches: int = 3,
    max_position_pct: float = 0.02,
    max_open_positions: int = 3,
    max_exposure_pct: float = 0.10,
) -> tuple[RiskGate, KillSwitch, DailyLossCheck]:
    """Build a fully-wired :class:`RiskGate` with mocked Redis."""
    ks = await _make_kill_switch(active=kill_switch_active)
    tp = _make_time_policy(blocked_hours=blocked_hours)
    dl = DailyLossCheck(
        warning_pct=warning_pct,
        critical_pct=critical_pct,
        consecutive_breaches=consecutive_breaches,
        kill_switch=ks,
    )
    exp = ExposureCheck(max_exposure_pct=max_exposure_pct)
    pl = PositionLimitCheck(
        max_position_pct=max_position_pct,
        max_open_positions=max_open_positions,
    )
    gate = RiskGate(
        kill_switch=ks,
        time_policy=tp,
        daily_loss=dl,
        exposure=exp,
        position_limit=pl,
    )
    return gate, ks, dl


# ===========================================================================
# KillSwitch tests
# ===========================================================================


async def test_kill_switch_inactive_by_default() -> None:
    """A fresh KillSwitch with Redis reporting '0' should be inactive."""
    ks = await _make_kill_switch(active=False)
    assert not await ks.is_active()


async def test_kill_switch_activate_and_reject() -> None:
    """After activation the switch reports active and evaluate() rejects."""
    ks = await _make_kill_switch(active=False)

    # Activate — patch the mock so subsequent get() returns "1"
    ks._client.get = AsyncMock(return_value="1")  # type: ignore[union-attr]
    await ks.activate("test_reason")

    assert await ks.is_active()

    portfolio = _make_portfolio()
    decision = await ks.evaluate(portfolio)
    assert not decision.approved
    assert "kill_switch" in decision.checks_failed


async def test_kill_switch_deactivate_allows_trading() -> None:
    """After deactivation the switch should no longer reject orders."""
    ks = await _make_kill_switch(active=True)

    # Simulate deactivation: subsequent get() returns "0"
    ks._client.get = AsyncMock(return_value="0")  # type: ignore[union-attr]
    await ks.deactivate()

    assert not await ks.is_active()

    portfolio = _make_portfolio()
    decision = await ks.evaluate(portfolio)
    assert decision.approved
    assert "kill_switch" in decision.checks_passed


async def test_kill_switch_redis_unavailable_fails_closed() -> None:
    """When Redis raises an exception, is_active() must return True (fail-closed)."""
    ks = KillSwitch(redis_url="redis://localhost:6379/0")
    # Inject a broken client
    broken_client = MagicMock()
    broken_client.get = AsyncMock(side_effect=ConnectionError("redis unreachable"))
    ks._client = broken_client

    assert await ks.is_active() is True

    portfolio = _make_portfolio()
    decision = await ks.evaluate(portfolio)
    assert not decision.approved


async def test_kill_switch_no_client_fails_closed() -> None:
    """With no client at all, is_active() must return True."""
    ks = KillSwitch(redis_url="redis://localhost:6379/0")
    # _client stays None (connect() never called)
    assert await ks.is_active() is True


async def test_kill_switch_get_status_returns_dict() -> None:
    """get_status() must return a dict with the expected keys."""
    ks = await _make_kill_switch(active=False)
    # Multi-key get: side_effect mapping by key name
    async def _get(key: str) -> str | None:
        mapping = {
            KillSwitch.REDIS_KEY: "0",
            KillSwitch.REDIS_REASON_KEY: "test",
            KillSwitch.REDIS_ACTIVATED_AT_KEY: "1234567890.0",
        }
        return mapping.get(key)

    ks._client.get = AsyncMock(side_effect=_get)  # type: ignore[union-attr]
    status = await ks.get_status()

    assert "active" in status
    assert "reason" in status
    assert "activated_at" in status
    assert status["active"] is False


# ===========================================================================
# DailyLossCheck tests
# ===========================================================================


async def test_daily_loss_below_warning_approved() -> None:
    """Zero daily loss should be approved without warnings."""
    dl = DailyLossCheck(warning_pct=0.03, critical_pct=0.05)
    portfolio = _make_portfolio(balance=100_000.0)
    decision = await dl.evaluate(portfolio)
    assert decision.approved
    assert "daily_loss" in decision.checks_passed


async def test_daily_loss_above_warning_still_approved() -> None:
    """Loss between warning and critical threshold: approved but logged."""
    dl = DailyLossCheck(warning_pct=0.03, critical_pct=0.05)
    # Realized loss = 4 % (above warning 3 %, below critical 5 %)
    portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=-4_000.0)
    decision = await dl.evaluate(portfolio)
    assert decision.approved
    assert "warning" in decision.reason


async def test_daily_loss_single_critical_breach_not_killed() -> None:
    """A single critical breach must not activate kill switch (hysteresis P1-3)."""
    ks = await _make_kill_switch(active=False)
    dl = DailyLossCheck(
        warning_pct=0.03,
        critical_pct=0.05,
        consecutive_breaches=3,
        kill_switch=ks,
    )
    # Loss = 6 % (above critical 5 %) — first breach
    portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=-6_000.0)
    decision = await dl.evaluate(portfolio)

    # Must be rejected (single breach rejects) but kill switch NOT activated yet
    assert not decision.approved
    assert dl._consecutive_breach_count == 1
    # Kill switch activation is not called — pipe.execute should not have been invoked
    pipe = ks._client.pipeline.return_value.__aenter__.return_value  # type: ignore[union-attr]
    pipe.execute.assert_not_awaited()


async def test_daily_loss_consecutive_breaches_triggers_kill_switch() -> None:
    """Three consecutive critical breaches must activate the kill switch."""
    ks = await _make_kill_switch(active=False)
    dl = DailyLossCheck(
        warning_pct=0.03,
        critical_pct=0.05,
        consecutive_breaches=3,
        kill_switch=ks,
    )
    portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=-6_000.0)

    # First two breaches: rejected but kill switch not yet fired
    await dl.evaluate(portfolio)
    await dl.evaluate(portfolio)
    assert dl._consecutive_breach_count == 2
    pipe = ks._client.pipeline.return_value.__aenter__.return_value  # type: ignore[union-attr]
    pipe.execute.assert_not_awaited()

    # Third consecutive breach: kill switch fires
    decision = await dl.evaluate(portfolio)
    assert not decision.approved
    assert dl._consecutive_breach_count == 3
    pipe.execute.assert_awaited_once()


async def test_daily_loss_reset_clears_breach_count() -> None:
    """reset_daily() must zero the breach counter and EMA state."""
    dl = DailyLossCheck(warning_pct=0.03, critical_pct=0.05, consecutive_breaches=3)
    portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=-6_000.0)

    await dl.evaluate(portfolio)
    await dl.evaluate(portfolio)
    assert dl._consecutive_breach_count == 2

    dl.reset_daily()
    assert dl._consecutive_breach_count == 0
    assert dl._smoothed_unrealized_pnl is None


async def test_daily_loss_ema_smoothing_reduces_spike_impact() -> None:
    """A single unrealized PnL spike should be smoothed by the EMA."""
    dl = DailyLossCheck(warning_pct=0.03, critical_pct=0.05, consecutive_breaches=3)
    # Seed EMA with a small loss
    portfolio_baseline = _make_portfolio(balance=100_000.0, daily_unrealized_pnl=-1_000.0)
    await dl.evaluate(portfolio_baseline)
    seeded = dl._smoothed_unrealized_pnl
    assert seeded is not None

    # Now a large spike
    portfolio_spike = _make_portfolio(balance=100_000.0, daily_unrealized_pnl=-20_000.0)
    await dl.evaluate(portfolio_spike)
    smoothed = dl._smoothed_unrealized_pnl
    assert smoothed is not None
    # Smoothed value must be much less severe than the raw spike
    assert smoothed > -20_000.0


# ===========================================================================
# PositionLimitCheck tests
# ===========================================================================


async def test_position_within_limit_approved() -> None:
    """Requested quantity within the 2 % cap should be approved unchanged."""
    pl = PositionLimitCheck(max_position_pct=0.02, max_open_positions=3)
    signal = _make_signal()
    portfolio = _make_portfolio(balance=100_000.0)
    # 2 % of 100 000 = 2 000 USD; 0.03 BTC * 60 000 = 1 800 USD < 2 000
    decision = await pl.evaluate(signal, portfolio, requested_quantity=0.03, current_price=60_000.0)

    assert decision.approved
    assert decision.adjusted_quantity is None


async def test_position_over_limit_adjusted() -> None:
    """Quantity exceeding the per-trade cap must be reduced, not rejected."""
    pl = PositionLimitCheck(max_position_pct=0.02, max_open_positions=3)
    signal = _make_signal()
    portfolio = _make_portfolio(balance=100_000.0)
    # 2 % of 100 000 = 2 000 USD; 0.1 BTC * 60 000 = 6 000 USD > 2 000
    decision = await pl.evaluate(signal, portfolio, requested_quantity=0.1, current_price=60_000.0)

    assert decision.approved
    assert decision.adjusted_quantity is not None
    # Adjusted notional should be ≈ 2 000 / 60 000 ≈ 0.03333 BTC
    expected = 2_000.0 / 60_000.0
    assert abs(decision.adjusted_quantity - expected) < 1e-9


async def test_too_many_open_positions_rejected() -> None:
    """At max open positions for a NEW pair, the order must be rejected."""
    pl = PositionLimitCheck(max_position_pct=0.02, max_open_positions=3)
    signal = _make_signal(pair="SOL/USDT")
    # 3 positions already open (none is SOL/USDT)
    portfolio = _make_portfolio(
        open_positions={"BTC/USDT": 0.5, "ETH/USDT": 2.0, "XRP/USDT": 100.0},
    )
    decision = await pl.evaluate(signal, portfolio, requested_quantity=10.0, current_price=150.0)

    assert not decision.approved
    assert "position_limit_count" in decision.checks_failed


async def test_existing_position_pair_not_blocked_by_count() -> None:
    """Adding to an already-open position should not be blocked by the count limit."""
    pl = PositionLimitCheck(max_position_pct=0.02, max_open_positions=3)
    signal = _make_signal(pair="BTC/USDT")
    portfolio = _make_portfolio(
        balance=100_000.0,
        open_positions={"BTC/USDT": 0.5, "ETH/USDT": 2.0, "XRP/USDT": 100.0},
    )
    # 0.01 BTC * 60 000 = 600 USD < 2 000 cap
    decision = await pl.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)

    assert decision.approved


# ===========================================================================
# ExposureCheck tests
# ===========================================================================


async def test_exposure_within_limit_approved() -> None:
    """New order that keeps total exposure under 10 % should be approved."""
    exp = ExposureCheck(max_exposure_pct=0.10)
    signal = _make_signal()
    # Existing exposure: 5 000 / 100 000 = 5 %; new order 3 000 → total 8 %
    portfolio = _make_portfolio(
        balance=100_000.0,
        open_position_values={"ETH/USDT": 5_000.0},
    )
    decision = await exp.evaluate(
        signal, portfolio, requested_quantity=0.05, current_price=60_000.0
    )

    assert decision.approved
    assert "exposure" in decision.checks_passed


async def test_exposure_over_limit_rejected() -> None:
    """Order that would push total exposure above 10 % must be rejected."""
    exp = ExposureCheck(max_exposure_pct=0.10)
    signal = _make_signal()
    # Existing exposure: 9 500 / 100 000 = 9.5 %; new 0.01 BTC * 60 000 = 600 → 10.1 %
    portfolio = _make_portfolio(
        balance=100_000.0,
        open_position_values={"ETH/USDT": 9_500.0},
    )
    decision = await exp.evaluate(
        signal, portfolio, requested_quantity=0.01, current_price=60_000.0
    )

    assert not decision.approved
    assert "exposure" in decision.checks_failed


async def test_exposure_at_exact_limit_approved() -> None:
    """An order that hits exactly the exposure cap (not over) should be approved."""
    exp = ExposureCheck(max_exposure_pct=0.10)
    signal = _make_signal()
    # No existing positions; new order exactly 10 000 USD = 10 %
    portfolio = _make_portfolio(balance=100_000.0)
    decision = await exp.evaluate(
        signal, portfolio, requested_quantity=1.0, current_price=10_000.0
    )

    assert decision.approved


# ===========================================================================
# RiskGate integration tests
# ===========================================================================


async def test_gate_all_pass_produces_approved_order() -> None:
    """All checks passing must produce a non-None ApprovedOrder."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)

    assert order is not None
    assert isinstance(order, ApprovedOrder)
    assert order.pair == "BTC/USDT"
    assert order.side == "buy"
    assert order.quantity == pytest.approx(0.01)


async def test_gate_approved_order_has_valid_hmac() -> None:
    """The gate_token on an ApprovedOrder must pass verify_gate_token()."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)

    assert order is not None
    assert verify_gate_token(order) is True


async def test_gate_forged_token_fails_verification() -> None:
    """Manually altering any field of an ApprovedOrder must invalidate the token."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is not None

    # Tamper with the quantity field
    forged = ApprovedOrder(
        signal_id=order.signal_id,
        pair=order.pair,
        side=order.side,
        quantity=order.quantity * 100,  # attacker inflates quantity
        approved_at=order.approved_at,
        gate_token=order.gate_token,  # stolen original token
        risk_checks_passed=order.risk_checks_passed,
        original_signal_strength=order.original_signal_strength,
    )
    assert verify_gate_token(forged) is False


async def test_gate_forged_token_with_tampered_pair_fails() -> None:
    """Changing the pair field of an ApprovedOrder must also invalidate the token."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is not None

    forged = ApprovedOrder(
        signal_id=order.signal_id,
        pair="ETH/USDT",  # changed pair
        side=order.side,
        quantity=order.quantity,
        approved_at=order.approved_at,
        gate_token=order.gate_token,
        risk_checks_passed=order.risk_checks_passed,
        original_signal_strength=order.original_signal_strength,
    )
    assert verify_gate_token(forged) is False


async def test_gate_kill_switch_rejects_immediately() -> None:
    """An active kill switch must cause the gate to return None without running other checks."""
    gate, ks, _ = await _make_gate(kill_switch_active=True)
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is None


async def test_gate_returns_none_on_rejection() -> None:
    """Any rejected check must cause evaluate() to return None."""
    # Use blocked_hours containing the current hour to force time policy rejection
    current_hour = time.gmtime().tm_hour
    gate, _, _ = await _make_gate(blocked_hours=[current_hour])
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is None


async def test_gate_exposure_rejection_returns_none() -> None:
    """Exposure breach must cause the gate to return None."""
    gate, _, _ = await _make_gate(max_exposure_pct=0.01)  # very tight cap
    signal = _make_signal()
    # Existing exposure already near cap
    portfolio = _make_portfolio(
        balance=100_000.0,
        open_position_values={"ETH/USDT": 900.0},
    )
    # New order notional = 0.01 * 60 000 = 600; total = 1 500 > 1 % * 100 000 = 1 000
    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is None


async def test_gate_daily_loss_rejection_returns_none() -> None:
    """A daily loss breach (consecutive) must cause the gate to return None."""
    gate, _, dl = await _make_gate(consecutive_breaches=1)
    signal = _make_signal()
    portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=-6_000.0)

    # With consecutive_breaches=1 a single critical breach fires the kill switch and rejects
    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is None


async def test_gate_quantity_adjusted_by_position_limit() -> None:
    """When position limit adjusts quantity, the ApprovedOrder must reflect it.

    The exposure check uses the *original* requested quantity.  To exercise the
    position-limit adjustment path we therefore need a requested quantity that:
    - Passes exposure: notional <= 10 % of balance.
    - Exceeds the per-trade size cap: notional > 2 % of balance.

    With balance = 100 000 USD and price = 10 000 USD/BTC:
    - 0.08 BTC * 10 000 =   800 USD →  0.8 % (< 10 %, passes exposure)
    - 0.08 BTC * 10 000 =   800 USD →  0.8 % (> 0.5 %, but we need > 2 %)

    Use a tighter max_position_pct = 0.005 (0.5 %) and price = 10 000:
    - requested 0.08 BTC → 800 USD notional → 0.8 % (passes 10 % exposure cap)
    - 0.5 % cap = 500 USD → adjusted qty = 500 / 10 000 = 0.05 BTC
    """
    gate, _, _ = await _make_gate(max_position_pct=0.005)  # 0.5 % cap
    signal = _make_signal()
    portfolio = _make_portfolio(balance=100_000.0)
    # 0.08 BTC * 10 000 = 800 USD < 10 % exposure cap → passes exposure check
    # But 800 > 0.5 % * 100 000 = 500 → position limit adjusts down to 0.05 BTC
    order = await gate.evaluate(
        signal, portfolio, requested_quantity=0.08, current_price=10_000.0
    )

    assert order is not None
    expected_qty = (0.005 * 100_000.0) / 10_000.0  # 500 / 10 000 = 0.05
    assert order.quantity == pytest.approx(expected_qty, rel=1e-6)
    assert verify_gate_token(order) is True


async def test_gate_short_signal_produces_sell_side() -> None:
    """A SHORT signal must produce an ApprovedOrder with side='sell'."""
    gate, _, _ = await _make_gate()
    signal = _make_signal(direction=SignalDirection.SHORT)
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)

    assert order is not None
    assert order.side == "sell"
    assert verify_gate_token(order) is True


async def test_gate_approved_order_carries_signal_strength() -> None:
    """The ApprovedOrder must carry the original signal's strength value."""
    gate, _, _ = await _make_gate()
    signal = _make_signal(strength=0.75)
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)

    assert order is not None
    assert order.original_signal_strength == pytest.approx(0.75)


async def test_gate_approved_order_lists_all_checks() -> None:
    """risk_checks_passed on the ApprovedOrder must include checks from all five steps."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)

    assert order is not None
    # All check names must appear
    joined = " ".join(order.risk_checks_passed)
    assert "kill_switch" in joined
    assert "time_policy" in joined
    assert "daily_loss" in joined
    assert "exposure" in joined
    assert "position_limit" in joined


# ===========================================================================
# Edge-case / boundary tests
# ===========================================================================


async def test_verify_gate_token_with_genuine_order() -> None:
    """verify_gate_token must return True for an unmodified ApprovedOrder."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.005, current_price=60_000.0)
    assert order is not None
    assert verify_gate_token(order) is True


async def test_verify_gate_token_rejects_empty_token() -> None:
    """A blank gate_token string must fail verification."""
    gate, _, _ = await _make_gate()
    signal = _make_signal()
    portfolio = _make_portfolio()

    order = await gate.evaluate(signal, portfolio, requested_quantity=0.01, current_price=60_000.0)
    assert order is not None

    blank_token_order = ApprovedOrder(
        signal_id=order.signal_id,
        pair=order.pair,
        side=order.side,
        quantity=order.quantity,
        approved_at=order.approved_at,
        gate_token="",  # empty
    )
    assert verify_gate_token(blank_token_order) is False


async def test_daily_loss_consecutive_count_resets_on_recovery() -> None:
    """If loss drops below critical after a breach, the counter resets to 0."""
    dl = DailyLossCheck(warning_pct=0.03, critical_pct=0.05, consecutive_breaches=3)
    critical_portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=-6_000.0)
    ok_portfolio = _make_portfolio(balance=100_000.0, daily_realized_pnl=0.0)

    await dl.evaluate(critical_portfolio)  # breach count → 1
    assert dl._consecutive_breach_count == 1

    # But EMA is seeded with -6000; inject ok portfolio to drive smoothed value
    # towards 0 over several evaluations
    for _ in range(20):
        await dl.evaluate(ok_portfolio)

    # Eventually the smoothed unrealized + realized must fall below critical
    assert dl._consecutive_breach_count == 0


async def test_exposure_check_zero_existing_positions() -> None:
    """With no open positions, any order under the cap should be approved."""
    exp = ExposureCheck(max_exposure_pct=0.10)
    signal = _make_signal()
    portfolio = _make_portfolio(balance=100_000.0)  # no open positions
    # 0.01 * 60 000 = 600 < 10 000
    decision = await exp.evaluate(
        signal, portfolio, requested_quantity=0.01, current_price=60_000.0
    )
    assert decision.approved
