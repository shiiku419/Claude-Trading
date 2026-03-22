"""Tests for the monitoring package.

Coverage:

* :class:`~monitoring.healthcheck.HealthChecker` — registration, check execution,
  unhealthy component propagation, periodic runner shutdown.
* :class:`~monitoring.alert.AlertDispatcher` — no-webhook log-only path,
  rate limiting enforcement, CRITICAL bypass.
* :class:`~monitoring.slo.SLOTracker` — ws_uptime, order_latency_p99,
  claude_success_rate, reconciliation_drift, and combined met/unmet status.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from monitoring.alert import AlertDispatcher, AlertLevel
from monitoring.healthcheck import HealthChecker, HealthStatus
from monitoring.slo import SLOTracker

# ===========================================================================
# HealthChecker
# ===========================================================================


def _healthy_status(component: str = "test") -> HealthStatus:
    """Build a healthy :class:`HealthStatus` for a given component."""
    return HealthStatus(
        component=component,
        healthy=True,
        last_check=datetime.now(tz=UTC),
    )


def _unhealthy_status(component: str = "test", details: str = "down") -> HealthStatus:
    """Build an unhealthy :class:`HealthStatus` for a given component."""
    return HealthStatus(
        component=component,
        healthy=False,
        last_check=datetime.now(tz=UTC),
        details=details,
    )


class TestHealthChecker:
    """Tests for :class:`HealthChecker`."""

    async def test_register_and_check_sync(self) -> None:
        """A registered sync check function is called and its result stored."""
        checker = HealthChecker(check_interval=60)
        checker.register_check("redis", lambda: _healthy_status("redis"))

        results = await checker.check_all()

        assert len(results) == 1
        assert results[0].component == "redis"
        assert results[0].healthy is True

    async def test_register_and_check_async(self) -> None:
        """A registered async check function is awaited and its result stored."""
        checker = HealthChecker(check_interval=60)

        async def _async_check() -> HealthStatus:
            return _healthy_status("websocket")

        checker.register_check("websocket", _async_check)
        results = await checker.check_all()

        assert len(results) == 1
        assert results[0].component == "websocket"
        assert results[0].healthy is True

    async def test_multiple_checks_all_returned(self) -> None:
        """All registered checks are executed and returned in order."""
        checker = HealthChecker()
        checker.register_check("redis", lambda: _healthy_status("redis"))
        checker.register_check("db", lambda: _healthy_status("db"))
        checker.register_check("ws", lambda: _unhealthy_status("ws", "timeout"))

        results = await checker.check_all()

        assert len(results) == 3
        components = [r.component for r in results]
        assert "redis" in components
        assert "db" in components
        assert "ws" in components

    async def test_unhealthy_component_reported(self) -> None:
        """An unhealthy check result is correctly surfaced in last_results."""
        checker = HealthChecker()
        checker.register_check("kill_switch", lambda: _unhealthy_status("kill_switch", "active"))

        results = await checker.check_all()

        assert results[0].healthy is False
        assert results[0].details == "active"
        assert checker.last_results[0].healthy is False

    async def test_check_raising_exception_yields_unhealthy_status(self) -> None:
        """A check that raises must not propagate; result must be unhealthy."""
        checker = HealthChecker()

        def _exploding_check() -> HealthStatus:
            raise RuntimeError("redis unreachable")

        checker.register_check("redis", _exploding_check)
        results = await checker.check_all()

        assert len(results) == 1
        assert results[0].healthy is False
        assert "redis unreachable" in results[0].details

    async def test_last_results_empty_before_first_check(self) -> None:
        """last_results is an empty list before check_all is called."""
        checker = HealthChecker()
        checker.register_check("db", lambda: _healthy_status("db"))
        assert checker.last_results == []

    async def test_last_results_updated_after_check(self) -> None:
        """last_results reflects the most recent check_all invocation."""
        checker = HealthChecker()
        checker.register_check("claude", lambda: _healthy_status("claude"))

        await checker.check_all()
        assert len(checker.last_results) == 1
        assert checker.last_results[0].component == "claude"

    async def test_run_periodic_stops_cleanly(self) -> None:
        """run_periodic returns promptly after stop() is called."""
        checker = HealthChecker(check_interval=60)
        checker.register_check("db", lambda: _healthy_status("db"))

        task = asyncio.create_task(checker.run_periodic())
        # Allow at least one iteration to fire.
        await asyncio.sleep(0.05)
        await checker.stop()

        # Give the task a moment to exit; it should not time out.
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()

    async def test_run_periodic_executes_checks(self) -> None:
        """run_periodic calls check_all at least once before being stopped."""
        checker = HealthChecker(check_interval=1)
        call_count = 0

        def _counting_check() -> HealthStatus:
            nonlocal call_count
            call_count += 1
            return _healthy_status("probe")

        checker.register_check("probe", _counting_check)

        task = asyncio.create_task(checker.run_periodic())
        await asyncio.sleep(0.1)
        await checker.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert call_count >= 1

    async def test_stop_is_idempotent(self) -> None:
        """Calling stop() multiple times must not raise."""
        checker = HealthChecker(check_interval=60)
        await checker.stop()
        await checker.stop()  # second call must be a no-op


# ===========================================================================
# AlertDispatcher
# ===========================================================================


class TestAlertDispatcher:
    """Tests for :class:`AlertDispatcher`."""

    async def test_send_info_no_webhooks_logs_only(self) -> None:
        """With no webhook configured, send() logs but makes no HTTP calls."""
        dispatcher = AlertDispatcher()  # no webhook, no token

        with patch("monitoring.alert.httpx.AsyncClient") as mock_client_cls:
            await dispatcher.send(AlertLevel.INFO, "strategy", "Regime changed to bull")

        mock_client_cls.assert_not_called()

    async def test_send_warning_no_webhooks_logs_only(self) -> None:
        """WARNING with no webhook configured produces no HTTP calls."""
        dispatcher = AlertDispatcher()

        with patch("monitoring.alert.httpx.AsyncClient") as mock_client_cls:
            await dispatcher.send(AlertLevel.WARNING, "websocket", "Reconnecting")

        mock_client_cls.assert_not_called()

    async def test_send_slack_called_when_webhook_configured(self) -> None:
        """When a Slack webhook URL is set, a POST is issued for each alert."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.INFO, "regime", "Bull market detected")

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert "https://hooks.slack.com/test" in call_args.args
        payload = call_args.kwargs["json"]
        assert "[INFO]" in payload["text"]

    async def test_send_telegram_called_when_token_configured(self) -> None:
        """When Telegram credentials are set, a POST is issued for each alert."""
        dispatcher = AlertDispatcher(
            telegram_bot_token="123:TOKEN",
            telegram_chat_id="-100999",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.WARNING, "risk_gate", "Loss threshold crossed")

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert "sendMessage" in call_args.args[0]

    async def test_rate_limit_suppresses_duplicate_non_critical(self) -> None:
        """A second non-CRITICAL alert for the same component within 5 min is dropped."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.WARNING, "websocket", "First alert")
            await dispatcher.send(AlertLevel.WARNING, "websocket", "Second alert (suppressed)")

        # Only one POST should have been made.
        assert mock_client.post.await_count == 1

    async def test_rate_limit_different_components_both_sent(self) -> None:
        """Rate limiting is per-component: two different components both deliver."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.WARNING, "websocket", "WS alert")
            await dispatcher.send(AlertLevel.WARNING, "database", "DB alert")

        assert mock_client.post.await_count == 2

    async def test_critical_bypasses_rate_limit(self) -> None:
        """CRITICAL alerts are never rate-limited, even for the same component."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.CRITICAL, "risk_gate", "Kill switch activated (1)")
            await dispatcher.send(AlertLevel.CRITICAL, "risk_gate", "Kill switch activated (2)")

        # Both CRITICAL alerts must be delivered.
        assert mock_client.post.await_count == 2

    async def test_message_format_contains_level_prefix(self) -> None:
        """Delivered text must start with the bracket-enclosed level label."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.CRITICAL, "kill_switch", "Activated")

        text = mock_client.post.call_args.kwargs["json"]["text"]
        assert text.startswith("[CRITICAL]")
        assert "kill_switch" in text
        assert "Activated" in text

    async def test_slack_http_error_does_not_raise(self) -> None:
        """A failed HTTP call to Slack must not propagate to the caller."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("unreachable"))

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            # Must not raise
            await dispatcher.send(AlertLevel.CRITICAL, "slack_test", "Should not propagate")

    async def test_rate_limit_resets_after_window(self) -> None:
        """After the rate-limit window elapses, subsequent alerts are delivered."""
        dispatcher = AlertDispatcher(slack_webhook="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("monitoring.alert.httpx.AsyncClient", return_value=mock_client):
            await dispatcher.send(AlertLevel.INFO, "strategy", "First")
            # Simulate time having passed beyond the 5-minute window.
            dispatcher._last_sent["strategy"] = time.monotonic() - 301.0
            await dispatcher.send(AlertLevel.INFO, "strategy", "Second after window")

        assert mock_client.post.await_count == 2


# ===========================================================================
# SLOTracker
# ===========================================================================


class TestSLOTracker:
    """Tests for :class:`SLOTracker`."""

    # --- ws_uptime -----------------------------------------------------------

    def test_ws_uptime_no_connection_is_zero(self) -> None:
        """With no WebSocket activity, uptime is 0 %."""
        tracker = SLOTracker()
        # Advance monotonic clock simulation by sleeping minimally.
        time.sleep(0.01)
        statuses = tracker.get_status()
        ws = next(s for s in statuses if s.name == "ws_uptime")
        assert ws.actual == pytest.approx(0.0, abs=0.01)
        assert ws.met is False  # 0 < 99.5 %

    def test_ws_uptime_fully_connected_meets_slo(self) -> None:
        """Connected for the full window should yield uptime ~100 %."""
        tracker = SLOTracker()
        tracker.record_ws_connected()
        time.sleep(0.05)
        statuses = tracker.get_status()
        ws = next(s for s in statuses if s.name == "ws_uptime")
        # Should be close to 1.0 (all time spent connected)
        assert ws.actual > 0.99
        assert ws.met is True

    def test_ws_uptime_disconnected_accumulates(self) -> None:
        """Connected-then-disconnected time is accumulated correctly."""
        tracker = SLOTracker()
        tracker.record_ws_connected()
        time.sleep(0.05)
        tracker.record_ws_disconnected()
        # Sleep a bit more while disconnected
        time.sleep(0.05)
        statuses = tracker.get_status()
        ws = next(s for s in statuses if s.name == "ws_uptime")
        # Uptime should be around 50 % (connected for ~half the window)
        assert 0.3 < ws.actual < 0.8

    def test_ws_uptime_idempotent_connect(self) -> None:
        """Calling record_ws_connected twice does not double-count."""
        tracker = SLOTracker()
        tracker.record_ws_connected()
        tracker.record_ws_connected()  # second call is no-op
        time.sleep(0.05)
        tracker.record_ws_disconnected()
        statuses = tracker.get_status()
        ws = next(s for s in statuses if s.name == "ws_uptime")
        # Should still be < 1.0 (not inflated by double counting)
        assert ws.actual <= 1.0

    # --- order_latency_p99 ---------------------------------------------------

    def test_order_latency_no_samples_met(self) -> None:
        """With no latency samples, the p99 SLO is considered met."""
        tracker = SLOTracker()
        statuses = tracker.get_status()
        lat = next(s for s in statuses if s.name == "order_latency_p99")
        assert lat.actual == 0.0
        assert lat.met is True

    def test_order_latency_within_target_met(self) -> None:
        """All latencies under 2 s results in p99 SLO met."""
        tracker = SLOTracker()
        for i in range(100):
            tracker.record_order_latency(0.1 + i * 0.005)  # max ~0.6 s

        statuses = tracker.get_status()
        lat = next(s for s in statuses if s.name == "order_latency_p99")
        assert lat.actual < 2.0
        assert lat.met is True

    def test_order_latency_p99_exceeds_target_not_met(self) -> None:
        """When p99 exceeds 2 s the SLO is not met."""
        tracker = SLOTracker()
        # 99 fast calls + 1 very slow call → p99 is the slow one
        for _ in range(99):
            tracker.record_order_latency(0.5)
        tracker.record_order_latency(5.0)  # outlier at p99

        statuses = tracker.get_status()
        lat = next(s for s in statuses if s.name == "order_latency_p99")
        assert lat.actual >= 5.0
        assert lat.met is False

    def test_order_latency_single_sample(self) -> None:
        """A single sample is its own p99."""
        tracker = SLOTracker()
        tracker.record_order_latency(1.5)
        statuses = tracker.get_status()
        lat = next(s for s in statuses if s.name == "order_latency_p99")
        assert lat.actual == pytest.approx(1.5)
        assert lat.met is True

    # --- claude_success_rate -------------------------------------------------

    def test_claude_success_rate_no_calls_met(self) -> None:
        """With no Claude calls recorded, success rate is vacuously 100 %."""
        tracker = SLOTracker()
        statuses = tracker.get_status()
        claude = next(s for s in statuses if s.name == "claude_success_rate")
        assert claude.actual == pytest.approx(1.0)
        assert claude.met is True

    def test_claude_success_rate_all_success(self) -> None:
        """100 % success rate satisfies the 95 % target."""
        tracker = SLOTracker()
        for _ in range(50):
            tracker.record_claude_call(success=True)

        statuses = tracker.get_status()
        claude = next(s for s in statuses if s.name == "claude_success_rate")
        assert claude.actual == pytest.approx(1.0)
        assert claude.met is True

    def test_claude_success_rate_below_threshold_not_met(self) -> None:
        """A 90 % success rate does not meet the 95 % target."""
        tracker = SLOTracker()
        for _ in range(90):
            tracker.record_claude_call(success=True)
        for _ in range(10):
            tracker.record_claude_call(success=False)

        statuses = tracker.get_status()
        claude = next(s for s in statuses if s.name == "claude_success_rate")
        assert claude.actual == pytest.approx(0.90)
        assert claude.met is False

    def test_claude_success_rate_exactly_at_threshold(self) -> None:
        """Success rate exactly equal to the target (0.95) is considered met."""
        tracker = SLOTracker()
        for _ in range(95):
            tracker.record_claude_call(success=True)
        for _ in range(5):
            tracker.record_claude_call(success=False)

        statuses = tracker.get_status()
        claude = next(s for s in statuses if s.name == "claude_success_rate")
        assert claude.actual == pytest.approx(0.95)
        assert claude.met is True

    # --- reconciliation_drift ------------------------------------------------

    def test_reconciliation_drift_zero_met(self) -> None:
        """No drift recorded means the SLO (0 discrepancies) is met."""
        tracker = SLOTracker()
        statuses = tracker.get_status()
        drift = next(s for s in statuses if s.name == "reconciliation_drift")
        assert drift.actual == pytest.approx(0.0)
        assert drift.met is True

    def test_reconciliation_drift_nonzero_not_met(self) -> None:
        """Any positive drift violates the zero-discrepancy SLO."""
        tracker = SLOTracker()
        tracker.record_reconciliation_drift(0.01)
        statuses = tracker.get_status()
        drift = next(s for s in statuses if s.name == "reconciliation_drift")
        assert drift.actual == pytest.approx(0.01)
        assert drift.met is False

    def test_reconciliation_drift_tracks_maximum(self) -> None:
        """Multiple drift observations keep only the maximum."""
        tracker = SLOTracker()
        tracker.record_reconciliation_drift(0.5)
        tracker.record_reconciliation_drift(2.0)
        tracker.record_reconciliation_drift(0.1)

        statuses = tracker.get_status()
        drift = next(s for s in statuses if s.name == "reconciliation_drift")
        assert drift.actual == pytest.approx(2.0)

    # --- combined status -----------------------------------------------------

    def test_get_status_returns_all_four_slos(self) -> None:
        """get_status() always returns exactly four SLOStatus objects."""
        tracker = SLOTracker()
        statuses = tracker.get_status()
        names = {s.name for s in statuses}
        assert names == {
            "ws_uptime",
            "order_latency_p99",
            "claude_success_rate",
            "reconciliation_drift",
        }

    def test_slo_met_status_reflects_targets(self) -> None:
        """A healthy system with good metrics shows all SLOs met."""
        tracker = SLOTracker()
        tracker.record_ws_connected()
        time.sleep(0.05)
        for _ in range(20):
            tracker.record_order_latency(0.3)
            tracker.record_claude_call(success=True)

        statuses = tracker.get_status()
        for s in statuses:
            assert s.met is True, f"Expected SLO {s.name!r} to be met (actual={s.actual})"

    def test_slo_status_window_seconds_positive(self) -> None:
        """window_seconds on each status must be a positive number."""
        tracker = SLOTracker()
        time.sleep(0.01)
        for s in tracker.get_status():
            assert s.window_seconds > 0
