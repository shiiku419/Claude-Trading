"""Trading system asyncio entry point.

Wires together the event bus, feature store, signal engine, risk gate, and
execution engine.  Each runnable component is launched as a concurrent task
inside an ``asyncio.TaskGroup`` so that any unexpected exception in one task
cancels the entire group and triggers a clean shutdown.

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
from bus.events import CandleEvent
from claude.regime_runner import RegimeRunner
from config.settings import Settings
from data.candle_aggregator import CandleAggregator
from data.feature_store import FeatureStore
from data.feeds.binance_ws import BinanceKlineFeed
from execution.engine import ExecutionEngine
from execution.paper_executor import PaperExecutor
from execution.recovery import RecoveryWorker
from ledger.pnl_tracker import PnLTracker
from ledger.tca import TCAAnalyzer
from monitoring.healthcheck import HealthChecker, HealthStatus
from monitoring.slo import SLOTracker
from risk.base import PortfolioState
from risk.daily_loss import DailyLossCheck
from risk.exposure import ExposureCheck
from risk.gate import RiskGate
from risk.kill_switch import KillSwitch
from risk.position_limit import PositionLimitCheck
from risk.time_policy import TimePolicyCheck
from signals.composite import CompositeSignal
from signals.momentum import MomentumSignal
from signals.volume_spike import VolumeSpikeSignal
from signals.vwap import VWAPSignal
from strategy.controller import StrategyController


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


def _build_composite_signal(controller_mode_name: str, settings: Settings) -> CompositeSignal:
    """Build a :class:`~signals.composite.CompositeSignal` for the current mode.

    Instantiates generator objects for each active strategy name defined in
    ``settings.trading`` and pairs them with the per-strategy weights from the
    current :class:`~strategy.modes.TradingMode`.

    Args:
        controller_mode_name: The ``name`` attribute of the current
            :class:`~strategy.modes.TradingMode`.
        settings: Fully resolved :class:`~config.settings.Settings` instance.

    Returns:
        A :class:`~signals.composite.CompositeSignal` ready to evaluate candles.
    """
    from strategy.modes import MODES, Regime

    mode = MODES.get(controller_mode_name, MODES[Regime.UNKNOWN])

    # Map strategy name -> generator instance.
    _generator_map = {
        "momentum": MomentumSignal(),
        "vwap": VWAPSignal(),
        "volume_spike": VolumeSpikeSignal(),
    }

    generators = []
    for strategy_name in mode.active_strategies:
        gen = _generator_map.get(strategy_name)
        if gen is None:
            continue
        weight = mode.signal_weights.get(strategy_name, 1.0)
        generators.append((gen, weight))

    if not generators:
        # No active strategies — return a neutral-only composite using momentum.
        generators = [(_generator_map["momentum"], 1.0)]

    return CompositeSignal(generators=generators)


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
    bus: EventBus = EventBus()

    feature_store: FeatureStore = FeatureStore(
        redis_url=settings.database.redis_url,
        max_candles=500,
    )
    await feature_store.connect()

    controller: StrategyController = StrategyController(
        redis_url=settings.database.redis_url,
        regime_ttl_hours=settings.claude.regime_ttl_hours,
    )
    await controller.connect()

    # ------------------------------------------------------------------ #
    # Execution layer                                                     #
    # ------------------------------------------------------------------ #
    # Only paper mode is supported for now; live executor is a future phase.
    paper_cfg = settings.paper
    executor: PaperExecutor = PaperExecutor(
        initial_balance=paper_cfg.initial_balance,
        fill_model=paper_cfg.fill_model,  # type: ignore[arg-type]
        base_slippage_pct=paper_cfg.slippage_pct,
        fee_pct=paper_cfg.fee_pct,
    )

    engine: ExecutionEngine = ExecutionEngine(
        executor=executor,
        event_bus=bus,
        redis_url=settings.database.redis_url,
    )
    await engine.connect()

    recovery_worker: RecoveryWorker = RecoveryWorker(engine=engine, executor=executor)

    # ------------------------------------------------------------------ #
    # Risk gate                                                           #
    # ------------------------------------------------------------------ #
    risk_gate: RiskGate = RiskGate(
        kill_switch=KillSwitch(redis_url=settings.database.redis_url),
        time_policy=TimePolicyCheck(),
        daily_loss=DailyLossCheck(
            warning_pct=settings.risk.daily_loss_warning_pct,
            critical_pct=settings.risk.daily_loss_critical_pct,
        ),
        exposure=ExposureCheck(max_exposure_pct=settings.risk.max_exposure_pct),
        position_limit=PositionLimitCheck(
            max_position_pct=settings.risk.max_position_pct,
            max_open_positions=settings.risk.max_open_positions,
        ),
    )

    # ------------------------------------------------------------------ #
    # Market data feed + candle aggregator                               #
    # ------------------------------------------------------------------ #
    market_feed: BinanceKlineFeed = BinanceKlineFeed(
        event_bus=bus,
        pairs=settings.trading.pairs,
        timeframes=settings.trading.timeframes,
        testnet=settings.exchange.testnet,
    )

    aggregator: CandleAggregator = CandleAggregator(
        event_bus=bus,
        feature_store=feature_store,
    )

    # ------------------------------------------------------------------ #
    # Claude regime runner                                                #
    # ------------------------------------------------------------------ #
    regime_runner: RegimeRunner = RegimeRunner(
        api_key=settings.claude.api_key,
        model=settings.claude.model,
        controller=controller,
        feature_store=feature_store,
        config=settings.claude,
    )

    # ------------------------------------------------------------------ #
    # Health checker                                                      #
    # ------------------------------------------------------------------ #
    health_checker: HealthChecker = HealthChecker(
        check_interval=settings.monitoring.health_check_interval_seconds,
    )

    async def _check_market_feed() -> HealthStatus:
        from datetime import UTC, datetime

        return HealthStatus(
            component="market_feed",
            healthy=market_feed.is_connected,
            last_check=datetime.now(tz=UTC),
            details="" if market_feed.is_connected else "websocket disconnected",
        )

    health_checker.register_check("market_feed", _check_market_feed)

    # ------------------------------------------------------------------ #
    # PnL / TCA / SLO trackers                                           #
    # ------------------------------------------------------------------ #
    pnl_tracker: PnLTracker = PnLTracker()
    tca_analyzer: TCAAnalyzer = TCAAnalyzer()
    slo_tracker: SLOTracker = SLOTracker()

    # ------------------------------------------------------------------ #
    # Dashboard                                                           #
    # ------------------------------------------------------------------ #
    dashboard_server: Any = None
    dashboard_state: Any = None

    if settings.dashboard.enabled:
        import uvicorn

        from dashboard.app import DashboardState, create_app

        dashboard_state = DashboardState(
            feature_store=feature_store,
            controller=controller,
            executor=executor,
            engine=engine,
            risk_gate=risk_gate,
            health_checker=health_checker,
            market_feed=market_feed,
            pnl_tracker=pnl_tracker,
            tca_analyzer=tca_analyzer,
            slo_tracker=slo_tracker,
            settings=settings,
        )
        dashboard_app = create_app(dashboard_state)
        dashboard_config = uvicorn.Config(
            dashboard_app,
            host=settings.dashboard.host,
            port=settings.dashboard.port,
            log_level="warning",
        )
        dashboard_server = uvicorn.Server(dashboard_config)
        log.info(
            "dashboard.enabled",
            host=settings.dashboard.host,
            port=settings.dashboard.port,
            url=f"http://{settings.dashboard.host}:{settings.dashboard.port}",
        )

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
    # Signal loop                                                         #
    # ------------------------------------------------------------------ #

    async def signal_loop() -> None:
        """Subscribe to closed candles; evaluate signals, risk, and execute.

        For every closed candle received on the event bus:

        1. Check :meth:`~strategy.controller.StrategyController.should_evaluate_signal`
           for each active strategy.
        2. Build a :class:`~signals.composite.CompositeSignal` from the mode's
           active strategies and weights.
        3. Retrieve candle history from the feature store.
        4. Evaluate the composite signal.
        5. Pass the signal through the :class:`~risk.gate.RiskGate`.
        6. Execute via :class:`~execution.engine.ExecutionEngine`.
        """
        candle_queue: asyncio.Queue[CandleEvent] = bus.subscribe("candle")  # type: ignore[assignment]
        log.info("signal_loop.started")

        try:
            while not stop_event.is_set():
                try:
                    event: CandleEvent = await asyncio.wait_for(
                        candle_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

                if not event.is_closed:
                    continue

                pair = event.pair
                timeframe = event.timeframe

                # Determine current mode and active strategies.
                current_mode = await controller.get_current_mode()
                if not current_mode.active_strategies:
                    log.debug(
                        "signal_loop.no_active_strategies",
                        pair=pair,
                        mode=current_mode.name,
                    )
                    continue

                if pair not in current_mode.active_pairs:
                    continue

                # Check at least one strategy is valid for this pair.
                any_active = False
                for strategy in current_mode.active_strategies:
                    if await controller.should_evaluate_signal(pair, strategy):
                        any_active = True
                        break
                if not any_active:
                    continue

                # Fetch candle history for signal computation.
                candles = feature_store.get_candles(pair, timeframe, n=100)
                if candles is None:
                    log.debug(
                        "signal_loop.insufficient_candles",
                        pair=pair,
                        timeframe=timeframe,
                    )
                    continue

                # Build composite signal for the current mode.
                composite = _build_composite_signal(current_mode.name, settings)
                signal_result = composite.evaluate(pair, candles)

                if signal_result is None:
                    log.debug(
                        "signal_loop.no_signal",
                        pair=pair,
                        timeframe=timeframe,
                        mode=current_mode.name,
                    )
                    continue

                log.info(
                    "signal_loop.signal_generated",
                    pair=pair,
                    direction=signal_result.direction,
                    strength=signal_result.strength,
                    indicator=signal_result.indicator_name,
                )

                # Record signal for dashboard
                if dashboard_state is not None:
                    dashboard_state.record_signal({
                        "pair": pair,
                        "direction": signal_result.direction,
                        "strength": signal_result.strength,
                        "indicator": signal_result.indicator_name,
                    })

                # Build portfolio state from executor.
                current_price = float(candles[-1]["close"])
                positions = executor.positions
                open_position_values = {
                    p: qty * current_price for p, qty in positions.items()
                }
                portfolio = PortfolioState(
                    total_balance_usd=executor.balance,
                    open_positions=positions,
                    open_position_values=open_position_values,
                    daily_realized_pnl=0.0,
                    daily_unrealized_pnl=0.0,
                )

                # Compute requested quantity: risk_multiplier * max_position_pct * balance / price.
                if current_price <= 0:
                    continue
                notional = (
                    settings.risk.max_position_pct
                    * current_mode.risk_multiplier
                    * executor.balance
                )
                requested_qty = notional / current_price

                if requested_qty <= 0:
                    continue

                # Pass through the risk gate.
                approved = await risk_gate.evaluate(
                    signal=signal_result,
                    portfolio=portfolio,
                    requested_quantity=requested_qty,
                    current_price=current_price,
                )

                if approved is None:
                    log.debug(
                        "signal_loop.risk_rejected",
                        pair=pair,
                        direction=signal_result.direction,
                    )
                    continue

                # Execute the approved order.
                result = await engine.execute(approved)
                if result is not None:
                    log.info(
                        "signal_loop.order_executed",
                        pair=pair,
                        side=approved.side,
                        quantity=approved.quantity,
                        status=result.status,
                        client_order_id=result.client_order_id,
                    )

                    # Record order for dashboard + PnL tracking
                    if dashboard_state is not None:
                        dashboard_state.record_order({
                            "pair": pair,
                            "side": approved.side,
                            "quantity": approved.quantity,
                            "price": result.filled_price,
                            "status": result.status,
                            "client_order_id": result.client_order_id,
                        })
                    if result.filled_quantity > 0:
                        pnl_tracker.record_trade(
                            pair=pair,
                            side=approved.side,
                            quantity=result.filled_quantity,
                            price=result.filled_price,
                            fees=result.fees,
                        )
                        tca_analyzer.record_execution(
                            expected_price=current_price,
                            actual_price=result.filled_price,
                            quantity=result.filled_quantity,
                            fees=result.fees,
                            side=approved.side,
                        )

        finally:
            bus.unsubscribe("candle", candle_queue)  # type: ignore[arg-type]
            log.info("signal_loop.stopped")

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
            tg.create_task(market_feed.start(), name="market_feed")
            tg.create_task(aggregator.start(), name="candle_aggregator")
            tg.create_task(
                regime_runner.run_scheduled(
                    interval_hours=settings.claude.regime_ttl_hours / 2.0
                ),
                name="regime_runner",
            )
            tg.create_task(health_checker.run_periodic(), name="health_checker")
            tg.create_task(recovery_worker.run_periodic(), name="recovery_worker")
            tg.create_task(signal_loop(), name="signal_loop")
            if dashboard_server is not None:
                tg.create_task(dashboard_server.serve(), name="dashboard")
            tg.create_task(_shutdown_watcher(tg), name="shutdown_watcher")

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
    await health_checker.stop()
    await regime_runner.stop()
    await recovery_worker.stop()
    await aggregator.stop()
    await market_feed.stop()
    await engine.close()
    await controller.close()
    await feature_store.close()
    log.info("trading_system.stopped")


if __name__ == "__main__":
    _config_path: str = sys.argv[1] if len(sys.argv) > 1 else "config/paper.yaml"
    asyncio.run(main(_config_path))
