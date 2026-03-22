"""Pydantic Settings v2 configuration system for the trading bot.

Settings are layered: YAML file values are the base, environment variables
(with ``__`` as the nested delimiter) override them, and explicit constructor
arguments take final precedence.

Typical usage::

    settings = Settings.from_yaml("config/paper.yaml")
    print(settings.trading.mode)   # "paper"
    print(settings.risk.max_position_pct)  # 0.02
"""

from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from bus.events import TOPIC_CONFIGS

# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class ExchangeConfig(BaseSettings):
    """Exchange connectivity settings.

    Attributes:
        name: CCXT exchange identifier (``"binance"``, ``"bybit"``, …).
        api_key: REST / WebSocket API key.  Populated from environment in
            production so it is never committed to YAML.
        api_secret: Corresponding API secret.
        testnet: When ``True`` the exchange client targets the testnet
            endpoint.
        rate_limit_per_second: Maximum number of REST requests per second
            before the client self-throttles.
    """

    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit_per_second: int = 10


class TradingConfig(BaseSettings):
    """Which markets and resolutions to trade.

    Attributes:
        pairs: CCXT-formatted symbol list.
        timeframes: Candle resolutions consumed by the signal engine.
        mode: ``"paper"`` routes orders through a local simulator;
            ``"live"`` sends them to the exchange.
    """

    pairs: list[str] = Field(default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframes: list[str] = Field(default=["5m", "15m"])
    mode: Literal["paper", "live"] = "paper"


class RiskConfig(BaseSettings):
    """Risk management thresholds.

    All ``_pct`` fields are expressed as fractions of total portfolio equity
    (e.g. ``0.02`` = 2 %).

    Attributes:
        max_position_pct: Maximum single-position size as a fraction of
            equity.
        daily_loss_warning_pct: Drawdown fraction that triggers a WARNING
            alert.
        daily_loss_critical_pct: Drawdown fraction that triggers a CRITICAL
            alert and may halt trading.
        loss_consecutive_breaches: Number of consecutive threshold breaches
            before the kill-switch is armed.
        max_open_positions: Hard cap on simultaneously open positions.
        max_exposure_pct: Maximum total notional exposure as a fraction of
            equity.
        kill_switch_enabled: When ``True`` the risk gate can halt all new
            orders.
    """

    max_position_pct: float = 0.02
    daily_loss_warning_pct: float = 0.03
    daily_loss_critical_pct: float = 0.05
    loss_consecutive_breaches: int = 3
    max_open_positions: int = 3
    max_exposure_pct: float = 0.10
    kill_switch_enabled: bool = True


class ClaudeConfig(BaseSettings):
    """Anthropic Claude integration settings.

    Attributes:
        model: Anthropic model identifier.
        calls_per_day: Maximum number of Claude API calls to make per
            calendar day.  Enforced by the rate-limiter in the regime
            detector.
        regime_ttl_hours: Regime classification is cached for this many
            hours before a fresh Claude call is triggered.
        timeout_seconds: HTTP timeout for each Claude API request.
        regime_prompt_path: Path to the Markdown prompt template used for
            regime detection.
    """

    model: str = "claude-sonnet-4-5-20250929"
    calls_per_day: int = 4
    regime_ttl_hours: float = 12.0
    timeout_seconds: int = 60
    regime_prompt_path: str = "claude/prompts/regime.md"


class DatabaseConfig(BaseSettings):
    """Persistence layer connection strings.

    Attributes:
        url: SQLAlchemy async database URL.  Use ``aiosqlite`` for local
            development and ``asyncpg`` for production PostgreSQL.
        redis_url: Redis connection URL used for state, idempotency keys,
            and the feature store.
    """

    url: str = "sqlite+aiosqlite:///trading.db"
    redis_url: str = "redis://localhost:6379/0"


class AlertConfig(BaseSettings):
    """Outbound alert channel credentials.

    Leave the fields empty to disable the corresponding channel.

    Attributes:
        slack_webhook: Incoming-webhook URL for Slack notifications.
        telegram_bot_token: Bot token issued by @BotFather.
        telegram_chat_id: Target chat or channel ID for Telegram messages.
    """

    slack_webhook: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


class BusTopicConfig(BaseSettings):
    """Per-topic queue parameters surfaced in the top-level Settings.

    Attributes:
        maxsize: Maximum number of events buffered per subscriber.
        policy: ``"lossy"`` or ``"blocking"``.
    """

    maxsize: int
    policy: str


def _default_bus_topics() -> dict[str, BusTopicConfig]:
    """Build the default topic map from the canonical ``TOPIC_CONFIGS`` registry."""
    return {
        name: BusTopicConfig(maxsize=cfg.maxsize, policy=cfg.policy.value)
        for name, cfg in TOPIC_CONFIGS.items()
    }


class BusConfig(BaseSettings):
    """Event-bus tuning parameters.

    Attributes:
        put_timeout: Seconds a BLOCKING publisher will wait for queue space
            before logging a CRITICAL and giving up.
        topics: Per-topic queue configuration.  Defaults mirror
            ``TOPIC_CONFIGS``.
    """

    put_timeout: float = 5.0
    topics: dict[str, BusTopicConfig] = Field(default_factory=_default_bus_topics)


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root configuration object for the trading system.

    Environment variables override YAML values using ``__`` as the nested
    delimiter.  For example, ``EXCHANGE__API_KEY=abc`` sets
    ``settings.exchange.api_key``.

    Attributes:
        exchange: Exchange connectivity settings.
        trading: Market selection and execution mode.
        risk: Risk management thresholds.
        claude: Claude AI integration settings.
        database: Database and Redis connection strings.
        alerts: Outbound alert channel credentials.
        bus: Event-bus queue configuration.
    """

    model_config = {  # type: ignore[misc]
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    bus: BusConfig = Field(default_factory=BusConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Settings:
        """Construct a ``Settings`` instance from a YAML file.

        The YAML structure must mirror the nested model hierarchy.  After
        loading, environment variables are still applied on top via the
        normal Pydantic Settings resolution order.

        Args:
            path: Filesystem path to the YAML configuration file.

        Returns:
            A fully validated ``Settings`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file cannot be parsed.
            pydantic.ValidationError: If the YAML contents fail validation.
        """
        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(**data)
