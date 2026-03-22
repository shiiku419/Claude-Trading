"""Tests for strategy controller, response parser, regime runner, and modes.

Test organisation
-----------------
- TestStrategyControllerNoRegime   – no regime stored yet
- TestStrategyControllerUpdate     – update_regime persistence and transitions
- TestStrategyControllerTTL        – [P0-1] regime expiry behaviour
- TestStrategyControllerRedisDown  – fail-safe when Redis is unavailable
- TestStrategyControllerSignal     – should_evaluate_signal checks
- TestResponseParser               – JSON extraction and Pydantic validation
- TestRegimeRunner                 – Claude API interaction
- TestModes                        – TradingMode registry invariants

All Redis interactions are mocked with :mod:`unittest.mock`.
The Anthropic client is replaced with a stub that returns a pre-built
message object.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude.regime_runner import RegimeRunner
from claude.response_parser import parse_regime_response
from config.settings import ClaudeConfig
from data.feature_store import FeatureStore
from strategy.controller import StrategyController
from strategy.modes import MODES, Regime, TradingMode

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_controller(
    ttl_hours: float = 12.0,
    redis_url: str = "redis://localhost:6379/0",
) -> StrategyController:
    return StrategyController(redis_url=redis_url, regime_ttl_hours=ttl_hours)


def _iso_now(offset_hours: float = 0.0) -> str:
    """Return UTC ISO-8601 string offset by *offset_hours* from now."""
    dt = datetime.now(tz=UTC) + timedelta(hours=offset_hours)
    return dt.isoformat()


def _make_redis_mock(
    regime: str | None = None,
    updated_at: str | None = None,
    confidence: str | None = "0.85",
) -> MagicMock:
    """Build a mock aioredis.Redis instance with pre-configured responses."""
    mock = MagicMock()
    mock.ping = AsyncMock(return_value=True)
    mock.aclose = AsyncMock()

    # gather() calls these three in sequence
    mock.get = AsyncMock(side_effect=[regime, updated_at, confidence])

    # pipeline mock
    pipe_mock = MagicMock()
    pipe_mock.set = MagicMock(return_value=pipe_mock)
    pipe_mock.execute = AsyncMock(return_value=["OK", "OK", "OK"])
    mock.pipeline = MagicMock(return_value=pipe_mock)

    return mock


def _attach_redis(controller: StrategyController, redis_mock: MagicMock) -> None:
    """Bypass connect() and inject a mock Redis directly."""
    controller._redis = redis_mock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestStrategyControllerNoRegime
# ---------------------------------------------------------------------------


class TestStrategyControllerNoRegime:
    async def test_get_mode_returns_unknown_when_no_regime_set(self) -> None:
        """No keys in Redis → unknown mode returned."""
        controller = _make_controller()
        redis_mock = _make_redis_mock(regime=None, updated_at=None, confidence=None)
        _attach_redis(controller, redis_mock)

        mode = await controller.get_current_mode()

        assert mode.name == Regime.UNKNOWN
        assert mode.risk_multiplier == 0.0
        assert mode.active_pairs == []

    async def test_get_mode_returns_unknown_when_redis_not_connected(self) -> None:
        """No Redis connection at all → unknown mode, no crash."""
        controller = _make_controller()
        # _redis is None by default
        mode = await controller.get_current_mode()
        assert mode.name == Regime.UNKNOWN


# ---------------------------------------------------------------------------
# TestStrategyControllerUpdate
# ---------------------------------------------------------------------------


class TestStrategyControllerUpdate:
    async def test_update_regime_changes_mode(self) -> None:
        """update_regime stores the new regime and returns the correct mode."""
        controller = _make_controller()
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=None)  # old regime fetch
        pipe_mock = MagicMock()
        pipe_mock.set = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(return_value=["OK", "OK", "OK"])
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)
        _attach_redis(controller, redis_mock)

        mode = await controller.update_regime(Regime.TRENDING_UP, confidence=0.9)

        assert mode.name == Regime.TRENDING_UP
        assert mode.risk_multiplier == 1.0
        assert "BTC/USDT" in mode.active_pairs

    async def test_update_regime_unknown_when_invalid_name(self) -> None:
        """An unrecognised regime string is normalised to 'unknown'."""
        controller = _make_controller()
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=None)
        pipe_mock = MagicMock()
        pipe_mock.set = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(return_value=["OK", "OK", "OK"])
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)
        _attach_redis(controller, redis_mock)

        mode = await controller.update_regime("totally_invalid_regime", confidence=0.5)

        assert mode.name == Regime.UNKNOWN

    async def test_mode_transition_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A regime change must emit a log event with old and new regime names."""

        controller = _make_controller()
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=Regime.RANGING)
        pipe_mock = MagicMock()
        pipe_mock.set = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(return_value=["OK", "OK", "OK"])
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)
        _attach_redis(controller, redis_mock)

        import structlog

        events: list[dict[str, Any]] = []

        def capture(logger: Any, method: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
            events.append(event_dict)
            raise structlog.DropEvent()

        structlog.configure(processors=[capture])

        try:
            await controller.update_regime(Regime.TRENDING_UP, confidence=0.8)
        finally:
            structlog.reset_defaults()

        transition_events = [e for e in events if "regime_updated" in e.get("event", "")]
        assert len(transition_events) == 1
        ev = transition_events[0]
        assert ev["new_regime"] == Regime.TRENDING_UP
        assert ev["old_regime"] == Regime.RANGING


# ---------------------------------------------------------------------------
# TestStrategyControllerTTL  [P0-1]
# ---------------------------------------------------------------------------


class TestStrategyControllerTTL:
    async def test_get_mode_after_ttl_expired_returns_unknown(self) -> None:
        """[P0-1] A regime older than TTL must resolve to unknown."""
        controller = _make_controller(ttl_hours=1.0)

        # Timestamp 2 hours in the past → expired under 1-hour TTL.
        stale_ts = _iso_now(offset_hours=-2.0)
        redis_mock = _make_redis_mock(
            regime=Regime.TRENDING_UP,
            updated_at=stale_ts,
            confidence="0.9",
        )
        _attach_redis(controller, redis_mock)

        mode = await controller.get_current_mode()

        assert mode.name == Regime.UNKNOWN
        assert mode.risk_multiplier == 0.0

    async def test_get_mode_just_before_ttl_returns_real_mode(self) -> None:
        """A regime set 30 min ago under a 1-hour TTL is still valid."""
        controller = _make_controller(ttl_hours=1.0)

        recent_ts = _iso_now(offset_hours=-0.5)  # 30 minutes ago
        redis_mock = _make_redis_mock(
            regime=Regime.TRENDING_UP,
            updated_at=recent_ts,
            confidence="0.9",
        )
        _attach_redis(controller, redis_mock)

        mode = await controller.get_current_mode()

        assert mode.name == Regime.TRENDING_UP

    async def test_get_regime_age_seconds_returns_correct_age(self) -> None:
        """get_regime_age_seconds approximates the stored timestamp age."""
        controller = _make_controller()
        thirty_min_ago = _iso_now(offset_hours=-0.5)

        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=thirty_min_ago)
        _attach_redis(controller, redis_mock)

        age = await controller.get_regime_age_seconds()

        assert age is not None
        assert 1750.0 <= age <= 1850.0  # ~1800 s with some tolerance

    async def test_get_regime_age_returns_none_when_not_set(self) -> None:
        """No timestamp in Redis → age returns None."""
        controller = _make_controller()
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=None)
        _attach_redis(controller, redis_mock)

        age = await controller.get_regime_age_seconds()
        assert age is None


# ---------------------------------------------------------------------------
# TestStrategyControllerRedisDown
# ---------------------------------------------------------------------------


class TestStrategyControllerRedisDown:
    async def test_get_mode_redis_unavailable_returns_unknown(self) -> None:
        """RedisError during get_current_mode → unknown mode, no exception."""
        from redis.exceptions import RedisError

        controller = _make_controller()
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(side_effect=RedisError("connection refused"))
        _attach_redis(controller, redis_mock)

        mode = await controller.get_current_mode()

        assert mode.name == Regime.UNKNOWN

    async def test_update_regime_redis_down_still_returns_mode(self) -> None:
        """update_regime must not raise even when Redis write fails."""
        from redis.exceptions import RedisError

        controller = _make_controller()
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=None)
        pipe_mock = MagicMock()
        pipe_mock.set = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(side_effect=RedisError("write failed"))
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)
        _attach_redis(controller, redis_mock)

        mode = await controller.update_regime(Regime.RANGING, confidence=0.6)

        # Mode object still returned even though Redis write failed.
        assert mode.name == Regime.RANGING


# ---------------------------------------------------------------------------
# TestStrategyControllerSignal
# ---------------------------------------------------------------------------


class TestStrategyControllerSignal:
    async def _controller_with_mode(self, regime: str) -> StrategyController:
        controller = _make_controller(ttl_hours=12.0)
        updated_at = _iso_now(offset_hours=-1.0)  # 1 hour ago, well within TTL
        redis_mock = _make_redis_mock(
            regime=regime,
            updated_at=updated_at,
            confidence="0.9",
        )
        _attach_redis(controller, redis_mock)
        return controller

    async def test_should_evaluate_signal_active_pair_and_strategy(self) -> None:
        """trending_up mode: BTC/USDT + momentum → True."""
        controller = await self._controller_with_mode(Regime.TRENDING_UP)
        result = await controller.should_evaluate_signal("BTC/USDT", "momentum")
        assert result is True

    async def test_should_evaluate_signal_inactive_pair_rejected(self) -> None:
        """trending_down mode only has BTC/USDT; ETH/USDT is rejected."""
        controller = await self._controller_with_mode(Regime.TRENDING_DOWN)
        result = await controller.should_evaluate_signal("ETH/USDT", "momentum")
        assert result is False

    async def test_should_evaluate_signal_inactive_strategy_rejected(self) -> None:
        """trending_up mode has no vwap strategy."""
        controller = await self._controller_with_mode(Regime.TRENDING_UP)
        result = await controller.should_evaluate_signal("BTC/USDT", "vwap")
        assert result is False

    async def test_should_evaluate_signal_unknown_mode_always_false(self) -> None:
        """unknown mode rejects every pair/strategy."""
        controller = _make_controller()
        redis_mock = _make_redis_mock(regime=None, updated_at=None, confidence=None)
        _attach_redis(controller, redis_mock)

        result = await controller.should_evaluate_signal("BTC/USDT", "momentum")
        assert result is False

    async def test_should_evaluate_signal_high_volatility_always_false(self) -> None:
        """high_volatility mode has no pairs or strategies."""
        controller = await self._controller_with_mode(Regime.HIGH_VOLATILITY)
        result = await controller.should_evaluate_signal("BTC/USDT", "momentum")
        assert result is False


# ---------------------------------------------------------------------------
# TestResponseParser
# ---------------------------------------------------------------------------


class TestResponseParser:
    def _valid_payload(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "regime": "trending_up",
            "confidence": 0.85,
            "reasoning": "Strong upward momentum with volume confirmation.",
            "suggested_pairs": ["BTC/USDT", "ETH/USDT"],
            "risk_assessment": "low",
        }
        base.update(overrides)
        return base

    def test_parse_valid_regime_response(self) -> None:
        raw = json.dumps(self._valid_payload())
        result = parse_regime_response(raw)

        assert result is not None
        assert result.regime == "trending_up"
        assert result.confidence == pytest.approx(0.85)
        assert result.risk_assessment == "low"
        assert "BTC/USDT" in result.suggested_pairs

    def test_parse_invalid_regime_returns_none(self) -> None:
        raw = json.dumps(self._valid_payload(regime="moon_phase"))
        result = parse_regime_response(raw)
        assert result is None

    def test_parse_confidence_out_of_range_returns_none(self) -> None:
        raw = json.dumps(self._valid_payload(confidence=1.5))
        result = parse_regime_response(raw)
        assert result is None

    def test_parse_confidence_negative_returns_none(self) -> None:
        raw = json.dumps(self._valid_payload(confidence=-0.1))
        result = parse_regime_response(raw)
        assert result is None

    def test_parse_non_json_text_returns_none(self) -> None:
        result = parse_regime_response("This is just plain text with no JSON at all.")
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        result = parse_regime_response("")
        assert result is None

    def test_parse_json_embedded_in_markdown(self) -> None:
        """Claude may wrap JSON in triple-backtick fences."""
        payload = self._valid_payload()
        raw = f"Sure, here is my analysis:\n\n```json\n{json.dumps(payload)}\n```"
        result = parse_regime_response(raw)

        assert result is not None
        assert result.regime == "trending_up"

    def test_parse_json_embedded_in_prose(self) -> None:
        """Claude may prepend a sentence before the JSON object."""
        payload = self._valid_payload(regime="ranging", confidence=0.6)
        raw = f"Based on the data provided: {json.dumps(payload)} Let me know if you need more."
        result = parse_regime_response(raw)

        assert result is not None
        assert result.regime == "ranging"

    def test_parse_missing_required_field_returns_none(self) -> None:
        """'reasoning' is required; missing it fails validation."""
        data = {
            "regime": "ranging",
            "confidence": 0.5,
            # 'reasoning' intentionally omitted
        }
        result = parse_regime_response(json.dumps(data))
        assert result is None

    def test_parse_optional_fields_use_defaults(self) -> None:
        """suggested_pairs and risk_assessment have defaults."""
        data = {
            "regime": "unknown",
            "confidence": 0.1,
            "reasoning": "Insufficient data.",
        }
        result = parse_regime_response(json.dumps(data))
        assert result is not None
        assert result.suggested_pairs == []
        assert result.risk_assessment == "medium"

    def test_regime_response_all_valid_regimes(self) -> None:
        """Every Regime enum value must be accepted by the validator."""
        for regime in Regime:
            data = {
                "regime": regime.value,
                "confidence": 0.5,
                "reasoning": "test",
            }
            result = parse_regime_response(json.dumps(data))
            assert result is not None, f"Expected valid parse for regime={regime.value}"

    def test_parse_confidence_boundary_zero(self) -> None:
        raw = json.dumps(self._valid_payload(confidence=0.0))
        result = parse_regime_response(raw)
        assert result is not None
        assert result.confidence == pytest.approx(0.0)

    def test_parse_confidence_boundary_one(self) -> None:
        raw = json.dumps(self._valid_payload(confidence=1.0))
        result = parse_regime_response(raw)
        assert result is not None
        assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestRegimeRunner
# ---------------------------------------------------------------------------


def _make_anthropic_message(content_text: str) -> MagicMock:
    """Construct a mock anthropic.types.Message with a single TextBlock."""
    block = MagicMock()
    block.text = content_text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _make_regime_runner(
    api_key: str = "test-key",
    regime_ttl_hours: float = 12.0,
) -> tuple[RegimeRunner, StrategyController, FeatureStore]:
    config = ClaudeConfig(
        model="claude-sonnet-4-5-20250929",
        calls_per_day=4,
        regime_ttl_hours=regime_ttl_hours,
        timeout_seconds=10,
        regime_prompt_path="claude/prompts/regime.md",
    )
    feature_store = MagicMock(spec=FeatureStore)
    feature_store._buffers = {}
    feature_store._buffer_counts = {}
    feature_store._buffer_positions = {}
    feature_store._max_candles = 500

    controller = MagicMock(spec=StrategyController)
    controller.update_regime = AsyncMock(return_value=MODES[Regime.TRENDING_UP])

    runner = RegimeRunner(
        api_key=api_key,
        model=config.model,
        controller=controller,
        feature_store=feature_store,
        config=config,
    )
    return runner, controller, feature_store


class TestRegimeRunner:
    async def test_run_once_with_mock_claude(self) -> None:
        """Successful Claude call updates the controller with the parsed regime."""
        runner, controller, _ = _make_regime_runner()

        valid_response = json.dumps(
            {
                "regime": "trending_up",
                "confidence": 0.9,
                "reasoning": "Strong bull trend.",
                "suggested_pairs": ["BTC/USDT"],
                "risk_assessment": "low",
            }
        )
        mock_message = _make_anthropic_message(valid_response)

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        runner._client = mock_client

        result = await runner.run_once()

        assert result == "trending_up"
        controller.update_regime.assert_awaited_once_with("trending_up", 0.9)

    async def test_run_once_timeout_keeps_last_regime(self) -> None:
        """API timeout returns None without clearing the last regime."""
        runner, controller, _ = _make_regime_runner()

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=TimeoutError())
        runner._client = mock_client

        result = await runner.run_once()

        assert result is None
        controller.update_regime.assert_not_awaited()

    async def test_run_once_empty_api_key_returns_none(self) -> None:
        """Empty API key prevents any client creation; run_once returns None."""
        runner, controller, _ = _make_regime_runner(api_key="")

        result = await runner.run_once()

        assert result is None
        controller.update_regime.assert_not_awaited()

    async def test_run_once_invalid_response_returns_none(self) -> None:
        """Claude returns non-JSON → parse fails → None returned."""
        runner, controller, _ = _make_regime_runner()

        mock_message = _make_anthropic_message("Sorry, I cannot help with that.")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        runner._client = mock_client

        result = await runner.run_once()

        assert result is None
        controller.update_regime.assert_not_awaited()

    async def test_run_once_api_error_returns_none(self) -> None:
        """Anthropic APIError is caught; run_once returns None."""
        import anthropic as anthropic_lib

        runner, controller, _ = _make_regime_runner()

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic_lib.APIError(
                message="rate_limit",
                request=MagicMock(),
                body=None,
            )
        )
        runner._client = mock_client

        result = await runner.run_once()

        assert result is None

    async def test_run_once_calls_controller_with_correct_confidence(self) -> None:
        """Confidence value from JSON is passed verbatim to update_regime."""
        runner, controller, _ = _make_regime_runner()

        valid_response = json.dumps(
            {
                "regime": "ranging",
                "confidence": 0.42,
                "reasoning": "Sideways market.",
                "suggested_pairs": [],
                "risk_assessment": "medium",
            }
        )
        mock_message = _make_anthropic_message(valid_response)
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        runner._client = mock_client

        await runner.run_once()

        controller.update_regime.assert_awaited_once_with("ranging", pytest.approx(0.42))


# ---------------------------------------------------------------------------
# TestModes
# ---------------------------------------------------------------------------


class TestModes:
    def test_unknown_mode_has_no_active_pairs(self) -> None:
        mode = MODES[Regime.UNKNOWN]
        assert mode.active_pairs == []
        assert mode.active_strategies == []

    def test_unknown_mode_risk_multiplier_is_zero(self) -> None:
        assert MODES[Regime.UNKNOWN].risk_multiplier == 0.0

    def test_trending_up_mode_has_all_three_pairs(self) -> None:
        mode = MODES[Regime.TRENDING_UP]
        assert "BTC/USDT" in mode.active_pairs
        assert "ETH/USDT" in mode.active_pairs
        assert "SOL/USDT" in mode.active_pairs

    def test_trending_up_mode_full_risk(self) -> None:
        assert MODES[Regime.TRENDING_UP].risk_multiplier == pytest.approx(1.0)

    def test_high_volatility_mode_stops_trading(self) -> None:
        mode = MODES[Regime.HIGH_VOLATILITY]
        assert mode.risk_multiplier == pytest.approx(0.0)
        assert mode.active_pairs == []
        assert mode.active_strategies == []

    def test_trending_down_reduced_risk(self) -> None:
        mode = MODES[Regime.TRENDING_DOWN]
        assert mode.risk_multiplier == pytest.approx(0.5)
        assert mode.active_pairs == ["BTC/USDT"]

    def test_ranging_mode_vwap_only(self) -> None:
        mode = MODES[Regime.RANGING]
        assert mode.active_strategies == ["vwap"]
        assert mode.risk_multiplier == pytest.approx(0.3)

    def test_all_regimes_covered_in_modes(self) -> None:
        """Every Regime enum value must have a corresponding MODES entry."""
        for regime in Regime:
            assert regime.value in MODES, f"Missing MODES entry for {regime.value}"

    def test_trading_mode_is_frozen(self) -> None:
        """TradingMode is frozen (immutable)."""
        mode = MODES[Regime.TRENDING_UP]
        with pytest.raises((AttributeError, TypeError)):
            mode.risk_multiplier = 99.0  # type: ignore[misc]

    def test_signal_weights_sum_to_one_for_trending_up(self) -> None:
        weights = MODES[Regime.TRENDING_UP].signal_weights
        total = sum(weights.values())
        assert total == pytest.approx(1.0)

    def test_modes_entries_are_trading_mode_instances(self) -> None:
        for name, mode in MODES.items():
            assert isinstance(mode, TradingMode), f"MODES[{name!r}] is not a TradingMode"

    def test_mode_name_matches_key(self) -> None:
        """Each TradingMode.name must equal its MODES key."""
        for key, mode in MODES.items():
            assert mode.name == key, f"MODES[{key!r}].name = {mode.name!r}"
