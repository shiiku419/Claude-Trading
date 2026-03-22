"""Trading system asyncio entry point.

Wires together the event bus, feature store, and future phase components.
Each runnable phase is launched as a concurrent task inside an
``asyncio.TaskGroup`` so that any unexpected exception in one task cancels
the entire group and triggers a clean shutdown.

Usage::

    uv run python main.py config/paper.yaml
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

import structlog
import structlog.dev

from bus.event_bus import EventBus
from config.settings import Settings
from data.feature_store import FeatureStore


def _configure_structlog(log_level: str) -> None:
    """Configure structlog with a JSON renderer suitable for production.

    Args:
        log_level: Minimum log level string, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def main(config_path: str) -> None:
    """Async entry point for the trading bot.

    Loads configuration, wires up infrastructure components, registers OS
    signal handlers for graceful shutdown, and runs all concurrent phase tasks
    inside a single ``asyncio.TaskGroup``.

    Args:
        config_path: Filesystem path to the YAML configuration file.
    """
    # ------------------------------------------------------------------ #
    # Bootstrap: settings + logging                                       #
    # ------------------------------------------------------------------ #
    settings: Settings = Settings.from_yaml(config_path)
    _configure_structlog(log_level="INFO")
    log: structlog.BoundLogger = structlog.get_logger(__name__)

    log.info(
        "trading_system.starting",
        config_path=config_path,
        mode=settings.trading.mode,
        pairs=settings.trading.pairs,
        timeframes=settings.trading.timeframes,
    )

    # ------------------------------------------------------------------ #
    # Infrastructure                                                      #
    # ------------------------------------------------------------------ #
    bus: EventBus = EventBus()  # noqa: F841  — wired to tasks in future phases

    feature_store: FeatureStore = FeatureStore(
        redis_url=settings.database.redis_url,
        max_candles=500,
    )
    await feature_store.connect()

    # ------------------------------------------------------------------ #
    # Graceful-shutdown event                                             #
    # ------------------------------------------------------------------ #
    stop_event: asyncio.Event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals, _frame: Any) -> None:
        log.info("trading_system.signal_received", signal=sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig, None)

    # ------------------------------------------------------------------ #
    # Concurrent tasks                                                    #
    # ------------------------------------------------------------------ #
    async def _shutdown_watcher(tg: asyncio.TaskGroup) -> None:
        """Wait for the stop event then cancel the task group."""
        await stop_event.wait()
        log.info("trading_system.shutting_down")
        # Raising inside the task group cancels all sibling tasks.
        raise asyncio.CancelledError("shutdown requested")

    try:
        async with asyncio.TaskGroup() as tg:
            # Phase 2: market feed WebSocket connection.
            # tg.create_task(market_feed.connect())

            # Phase 6: Claude regime detection scheduler.
            # tg.create_task(regime_runner.run_scheduled())

            # Phase 8: health / heartbeat checker.
            # tg.create_task(health_checker.run_periodic())

            # Phase 3-5: signal generation loop.
            # tg.create_task(signal_loop(bus, feature_store, settings))

            # Always present: watch for shutdown signal.
            tg.create_task(_shutdown_watcher(tg))

    except* asyncio.CancelledError:
        # TaskGroup re-raises CancelledError as an ExceptionGroup; this is
        # the expected path for a graceful shutdown triggered by the watcher.
        pass
    except* Exception as eg:
        log.error(
            "trading_system.task_group_error",
            exc_info=eg,
        )

    # ------------------------------------------------------------------ #
    # Teardown                                                            #
    # ------------------------------------------------------------------ #
    await feature_store.close()
    log.info("trading_system.stopped")


if __name__ == "__main__":
    _config_path: str = sys.argv[1] if len(sys.argv) > 1 else "config/paper.yaml"
    asyncio.run(main(_config_path))
