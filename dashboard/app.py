"""Real-time monitoring dashboard for the trading system.

Provides read-only API endpoints and an SSE stream for live updates.
Serves a single-page HTML dashboard from ``dashboard/static/``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Maximum recent items kept in memory for signals/orders ring buffers.
_MAX_RECENT: int = 100


@dataclass
class DashboardState:
    """Holds references to all bot components the dashboard reads from.

    Created in main.py and passed to create_app().
    All fields are optional to allow partial setup (e.g., no SLO tracker yet).

    Attributes:
        feature_store: In-memory + Redis feature store
            (``data.feature_store.FeatureStore``).
        controller: Strategy controller that maps Claude's regime to a
            ``TradingMode`` (``strategy.controller.StrategyController``).
        executor: Paper-trading executor with ``balance`` and ``positions``
            properties (``execution.paper_executor.PaperExecutor``).
        engine: Execution engine for crash-safe order lifecycle
            (``execution.engine.ExecutionEngine``).
        risk_gate: Risk gate pipeline; exposes ``_kill_switch``
            (``risk.gate.RiskGate``).
        health_checker: Periodic health-check runner
            (``monitoring.healthcheck.HealthChecker``).
        market_feed: Binance WebSocket kline feed with ``is_connected``
            property (``data.feeds.binance_ws.BinanceKlineFeed``).
        pnl_tracker: FIFO position book and PnL metrics
            (``ledger.pnl_tracker.PnLTracker``).
        tca_analyzer: Transaction cost analysis
            (``ledger.tca.TCAAnalyzer``).
        slo_tracker: Real-time SLO tracking
            (``monitoring.slo.SLOTracker``).
        settings: Root configuration object
            (``config.settings.Settings``).
        recent_signals: Ring buffer of the last ``_MAX_RECENT`` signal dicts.
        recent_orders: Ring buffer of the last ``_MAX_RECENT`` order dicts.
        started_at: Unix timestamp recorded at construction time, used for
            the uptime calculation.
        pnl_history: Ring buffer of up to 500 ``{timestamp, total_pnl,
            daily_pnl}`` dicts for the dashboard PnL chart.
    """

    # Core components (set from main.py)
    feature_store: Any = None  # data.feature_store.FeatureStore
    controller: Any = None  # strategy.controller.StrategyController
    executor: Any = None  # execution.paper_executor.PaperExecutor
    engine: Any = None  # execution.engine.ExecutionEngine
    risk_gate: Any = None  # risk.gate.RiskGate
    health_checker: Any = None  # monitoring.healthcheck.HealthChecker
    market_feed: Any = None  # data.feeds.binance_ws.BinanceKlineFeed

    # Tracking components (instantiated in main.py)
    pnl_tracker: Any = None  # ledger.pnl_tracker.PnLTracker
    tca_analyzer: Any = None  # ledger.tca.TCAAnalyzer
    slo_tracker: Any = None  # monitoring.slo.SLOTracker

    # Settings
    settings: Any = None  # config.settings.Settings

    # In-memory ring buffers for recent activity
    recent_signals: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_MAX_RECENT)
    )
    recent_orders: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_MAX_RECENT)
    )

    # Startup time for uptime calculation
    started_at: float = field(default_factory=time.time)

    # PnL history for charting (list of {timestamp, total_pnl, daily_pnl} dicts)
    pnl_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=500)
    )

    def record_signal(self, signal_data: dict[str, Any]) -> None:
        """Add a signal to the ring buffer with a UTC timestamp.

        Args:
            signal_data: Arbitrary dict describing the signal.  A
                ``"timestamp"`` key (ISO-8601 string) is injected before
                storing.
        """
        signal_data["timestamp"] = datetime.now(tz=UTC).isoformat()
        self.recent_signals.appendleft(signal_data)

    def record_order(self, order_data: dict[str, Any]) -> None:
        """Add an order to the ring buffer with a UTC timestamp.

        Args:
            order_data: Arbitrary dict describing the order.  A
                ``"timestamp"`` key (ISO-8601 string) is injected before
                storing.
        """
        order_data["timestamp"] = datetime.now(tz=UTC).isoformat()
        self.recent_orders.appendleft(order_data)

    def record_pnl_snapshot(self) -> None:
        """Snapshot current PnL into the history ring buffer for charting.

        No-op when ``pnl_tracker`` is ``None``.
        """
        if self.pnl_tracker is None:
            return
        summary = self.pnl_tracker.get_summary()
        self.pnl_history.append(
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "total_pnl": summary.total_pnl,
                "daily_pnl": self.pnl_tracker.get_daily_pnl(),
            }
        )


def create_app(state: DashboardState) -> FastAPI:
    """Create the FastAPI dashboard application.

    Mounts static files from ``dashboard/static/`` when the directory
    exists, registers all read-only REST endpoints and the SSE stream,
    and returns the configured :class:`fastapi.FastAPI` instance.

    Args:
        state: Shared :class:`DashboardState` populated by ``main.py``.

    Returns:
        A fully wired :class:`fastapi.FastAPI` application ready to hand
        to uvicorn.
    """
    app = FastAPI(title="Trading System Dashboard", docs_url=None, redoc_url=None)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # HTML entry point
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the single-page dashboard HTML."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text())
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

    # ------------------------------------------------------------------
    # /api/status
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        """Return overall system status.

        Includes uptime, trading mode, current regime name, regime age,
        WebSocket connection state, and kill-switch state.

        Returns:
            JSON-serialisable status dict.
        """
        result: dict[str, Any] = {
            "uptime_seconds": round(time.time() - state.started_at, 1),
            "mode": "unknown",
            "regime": "unknown",
            "regime_confidence": 0.0,
            "regime_age_seconds": None,
            "ws_connected": False,
            "kill_switch_active": False,
            "kill_switch_reason": "",
        }

        if state.settings is not None:
            result["mode"] = state.settings.trading.mode

        if state.controller is not None:
            try:
                trading_mode = await state.controller.get_current_mode()
                result["regime"] = trading_mode.name
                result["regime_age_seconds"] = (
                    await state.controller.get_regime_age_seconds()
                )
                # Fetch confidence directly from the Redis key the controller manages.
                if state.controller._redis is not None:
                    raw_conf: str | None = await state.controller._redis.get(
                        state.controller.REDIS_CONFIDENCE_KEY
                    )
                    if raw_conf is not None:
                        try:
                            result["regime_confidence"] = float(raw_conf)
                        except ValueError:
                            pass
            except Exception:
                log.warning("dashboard.status_controller_error", exc_info=True)

        if state.market_feed is not None:
            result["ws_connected"] = state.market_feed.is_connected

        if state.risk_gate is not None and state.risk_gate._kill_switch is not None:
            try:
                ks_status = await state.risk_gate._kill_switch.get_status()
                result["kill_switch_active"] = bool(ks_status.get("active", False))
                result["kill_switch_reason"] = str(ks_status.get("reason", ""))
            except Exception:
                log.warning("dashboard.status_kill_switch_error", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # /api/portfolio
    # ------------------------------------------------------------------

    @app.get("/api/portfolio")
    async def get_portfolio() -> dict[str, Any]:
        """Return the current portfolio state.

        Fetches the latest price for each open position from the feature
        store (Redis) to compute per-position market values.

        Returns:
            Dict with ``balance``, ``positions``, ``position_values``, and
            ``total_value`` keys.  Returns zeroed values when no executor is
            configured.
        """
        if state.executor is None:
            return {"balance": 0, "positions": {}, "position_values": {}, "total_value": 0}

        positions: dict[str, float] = state.executor.positions

        # Enrich each position with its current market value using Redis
        # scalar features written by the feature store on every candle.
        position_values: dict[str, float] = {}
        if state.feature_store is not None:
            for pair, qty in positions.items():
                try:
                    features = await state.feature_store.get_features(pair)
                    if features is not None and "last_price" in features:
                        price = float(features["last_price"])
                        position_values[pair] = round(qty * price, 2)
                except Exception:
                    log.warning(
                        "dashboard.portfolio_price_error", pair=pair, exc_info=True
                    )

        balance: float = state.executor.balance
        total_value = round(balance + sum(position_values.values()), 2)

        return {
            "balance": round(balance, 2),
            "positions": {k: round(v, 8) for k, v in positions.items()},
            "position_values": position_values,
            "total_value": total_value,
        }

    # ------------------------------------------------------------------
    # /api/pnl
    # ------------------------------------------------------------------

    @app.get("/api/pnl")
    async def get_pnl() -> dict[str, Any]:
        """Return PnL summary metrics and historical chart data.

        ``profit_factor`` is serialised as the Unicode infinity symbol
        ``"∞"`` when there are no losing trades so that the JSON payload
        remains valid (``Infinity`` is not valid JSON).

        Returns:
            Dict with ``summary`` (metrics dict) and ``history`` (list of
            ``{timestamp, total_pnl, daily_pnl}`` dicts).
        """
        if state.pnl_tracker is None:
            return {"summary": {}, "history": []}

        summary = state.pnl_tracker.get_summary()

        profit_factor: float | str
        if summary.profit_factor == float("inf"):
            profit_factor = "\u221e"
        else:
            profit_factor = round(summary.profit_factor, 2)

        return {
            "summary": {
                "realized_pnl": round(summary.realized_pnl, 2),
                "unrealized_pnl": round(summary.unrealized_pnl, 2),
                "total_pnl": round(summary.total_pnl, 2),
                "daily_pnl": round(state.pnl_tracker.get_daily_pnl(), 2),
                "win_count": summary.win_count,
                "loss_count": summary.loss_count,
                "win_rate": round(summary.win_rate * 100, 1),
                "total_trades": summary.total_trades,
                "max_drawdown": round(summary.max_drawdown, 2),
                "profit_factor": profit_factor,
            },
            "history": list(state.pnl_history),
        }

    # ------------------------------------------------------------------
    # /api/orders
    # ------------------------------------------------------------------

    @app.get("/api/orders")
    async def get_orders() -> dict[str, Any]:
        """Return the most recent orders from the ring buffer.

        Returns:
            Dict with an ``"orders"`` key containing a list of order dicts,
            newest first (ring buffer is prepended on each new order).
        """
        return {"orders": list(state.recent_orders)}

    # ------------------------------------------------------------------
    # /api/signals
    # ------------------------------------------------------------------

    @app.get("/api/signals")
    async def get_signals() -> dict[str, Any]:
        """Return the most recent trading signals from the ring buffer.

        Returns:
            Dict with a ``"signals"`` key containing a list of signal dicts,
            newest first.
        """
        return {"signals": list(state.recent_signals)}

    # ------------------------------------------------------------------
    # /api/health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def get_health() -> dict[str, Any]:
        """Return the most recent component health-check results.

        Reads :attr:`~monitoring.healthcheck.HealthChecker.last_results`
        — the results from the last completed :meth:`HealthChecker.check_all`
        pass.  Returns an empty list before the first pass has completed.

        Returns:
            Dict with a ``"checks"`` list, each entry containing
            ``component``, ``healthy``, ``last_check``, and ``details``.
        """
        if state.health_checker is None:
            return {"checks": []}

        results = state.health_checker.last_results
        return {
            "checks": [
                {
                    "component": r.component,
                    "healthy": r.healthy,
                    "last_check": r.last_check.isoformat() if r.last_check else None,
                    "details": r.details,
                }
                for r in results
            ]
        }

    # ------------------------------------------------------------------
    # /api/slo
    # ------------------------------------------------------------------

    @app.get("/api/slo")
    async def get_slo() -> dict[str, Any]:
        """Return real-time SLO status for all four tracked SLOs.

        SLOs tracked: ``ws_uptime``, ``order_latency_p99``,
        ``claude_success_rate``, ``reconciliation_drift``.

        Returns:
            Dict with an ``"slos"`` list; each entry contains ``name``,
            ``target``, ``actual``, and ``met``.
        """
        if state.slo_tracker is None:
            return {"slos": []}

        statuses = state.slo_tracker.get_status()
        return {
            "slos": [
                {
                    "name": s.name,
                    "target": s.target,
                    "actual": round(s.actual, 4),
                    "met": s.met,
                }
                for s in statuses
            ]
        }

    # ------------------------------------------------------------------
    # /api/tca
    # ------------------------------------------------------------------

    @app.get("/api/tca")
    async def get_tca() -> dict[str, Any]:
        """Return aggregated transaction cost analysis metrics.

        Returns:
            Dict with a ``"metrics"`` key containing ``avg_slippage_bps``,
            ``total_fees_usd``, ``avg_market_impact_bps``, and
            ``total_trades``.  Returns an empty metrics dict when no
            analyzer is configured.
        """
        if state.tca_analyzer is None:
            return {"metrics": {}}

        m = state.tca_analyzer.get_metrics()
        return {
            "metrics": {
                "avg_slippage_bps": round(m.avg_slippage_bps, 2),
                "total_fees_usd": round(m.total_fees_usd, 4),
                "avg_market_impact_bps": round(m.avg_market_impact_bps, 2),
                "total_trades": m.total_trades,
            }
        }

    # ------------------------------------------------------------------
    # /api/market
    # ------------------------------------------------------------------

    @app.get("/api/market")
    async def get_market() -> dict[str, Any]:
        """Return the latest market data snapshot per pair.

        Reads :meth:`~data.feature_store.FeatureStore.get_summary` which
        iterates all in-memory ring buffers.  Keys in the summary follow
        the ``"{pair}:{timeframe}"`` format; this endpoint collapses to
        per-pair by taking the first timeframe seen.

        Returns:
            Dict with a ``"pairs"`` key mapping each pair string to a dict
            containing ``last_price``, ``last_volume``, and
            ``candles_available``.
        """
        if state.feature_store is None:
            return {"pairs": {}}

        summary = state.feature_store.get_summary()
        pairs_data: Any = summary.get("pairs", {})

        # pairs_data may be {"status": "no_data_yet"} when nothing has
        # been ingested yet — guard against non-dict values.
        if not isinstance(pairs_data, dict):
            return {"pairs": {}}

        result: dict[str, Any] = {}
        for key, data in pairs_data.items():
            if not isinstance(data, dict):
                continue
            # Key format is "{pair}:{timeframe}"; extract pair portion.
            pair = key.split(":")[0] if ":" in key else key
            if pair not in result:
                result[pair] = {
                    "last_price": data.get("last_close", 0),
                    "last_volume": data.get("last_volume", 0),
                    "candles_available": data.get("candles_available", 0),
                }

        return {"pairs": result}

    # ------------------------------------------------------------------
    # /api/events  — Server-Sent Events stream
    # ------------------------------------------------------------------

    @app.get("/api/events")
    async def sse_events() -> StreamingResponse:
        """Server-Sent Events stream for real-time dashboard updates.

        Emits a JSON payload every two seconds containing a merged snapshot
        of all monitored subsystems: status, portfolio, PnL, market data,
        and health checks.  Also records a PnL snapshot into the history
        ring buffer on each tick.

        Handles ``asyncio.CancelledError`` for clean client disconnect and
        backs off for five seconds on unexpected errors before resuming.

        Returns:
            A ``text/event-stream`` :class:`~fastapi.responses.StreamingResponse`
            with cache-control and buffering headers set appropriately.
        """

        async def event_generator() -> AsyncGenerator[str, None]:
            while True:
                try:
                    data: dict[str, Any] = {}

                    try:
                        data["status"] = await get_status()
                    except Exception:
                        log.warning("dashboard.sse_status_error", exc_info=True)
                        data["status"] = {}

                    try:
                        data["portfolio"] = await get_portfolio()
                    except Exception:
                        log.warning("dashboard.sse_portfolio_error", exc_info=True)
                        data["portfolio"] = {}

                    try:
                        data["pnl"] = await get_pnl()
                    except Exception:
                        log.warning("dashboard.sse_pnl_error", exc_info=True)
                        data["pnl"] = {}

                    try:
                        data["market"] = await get_market()
                    except Exception:
                        log.warning("dashboard.sse_market_error", exc_info=True)
                        data["market"] = {}

                    try:
                        data["health"] = await get_health()
                    except Exception:
                        log.warning("dashboard.sse_health_error", exc_info=True)
                        data["health"] = {}

                    data["recent_signals_count"] = len(state.recent_signals)
                    data["recent_orders_count"] = len(state.recent_orders)

                    # Snapshot PnL into the history buffer for charting.
                    state.record_pnl_snapshot()

                    yield f"data: {json.dumps(data, default=str)}\n\n"
                    await asyncio.sleep(2)

                except asyncio.CancelledError:
                    # Client disconnected — exit cleanly.
                    break
                except Exception as exc:
                    log.warning("dashboard.sse_error", error=str(exc))
                    await asyncio.sleep(5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
