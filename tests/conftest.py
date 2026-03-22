"""Pytest configuration and shared fixtures.

All async tests in this suite use the ``asyncio`` backend supplied by
``pytest-asyncio``.  The ``asyncio_mode = "auto"`` setting in
``pyproject.toml`` means every ``async def test_*`` function is automatically
run under asyncio without needing an explicit ``@pytest.mark.asyncio``
decorator.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bus.event_bus import EventBus
from config.settings import Settings

# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Return a fresh :class:`~bus.event_bus.EventBus` for each test.

    Using a separate instance per test ensures that subscriber lists and
    metric counters never leak between test cases.
    """
    return EventBus()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return a :class:`~config.settings.Settings` instance for tests.

    The fixture first attempts to load the real ``config/paper.yaml``.  If
    that file is absent (e.g. in a stripped CI environment) it writes a
    minimal YAML to a temporary directory and loads that instead.

    Args:
        tmp_path: pytest-provided temporary directory, unique per test.

    Returns:
        A validated ``Settings`` object suitable for unit tests.
    """
    real_config = Path("config/paper.yaml")
    if real_config.exists():
        return Settings.from_yaml(str(real_config))

    # Fallback: write a minimal valid config for CI environments.
    minimal_yaml = textwrap.dedent(
        """\
        exchange:
          name: binance
          testnet: true

        trading:
          pairs:
            - BTC/USDT
          timeframes:
            - 1m
          mode: paper

        risk:
          max_position_pct: 0.02
          daily_loss_warning_pct: 0.03
          daily_loss_critical_pct: 0.05
          loss_consecutive_breaches: 3
          max_open_positions: 3
          max_exposure_pct: 0.10
          kill_switch_enabled: true

        database:
          url: "sqlite+aiosqlite:///trading.db"
          redis_url: "redis://localhost:6379/0"
        """
    )
    config_file = tmp_path / "paper.yaml"
    config_file.write_text(minimal_yaml)
    return Settings.from_yaml(str(config_file))
