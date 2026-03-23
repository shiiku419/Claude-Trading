"""Binance WebSocket kline (candlestick) feed.

Connects to the Binance combined stream endpoint and multiplexes all
configured pair/timeframe subscriptions over a single connection.
Parsed candle data is published as :class:`~bus.events.CandleEvent`
objects onto the event bus.

Gap recovery: when the feed reconnects after a disconnect the
``on_disconnect`` callback records the window and triggers a CCXT REST
backfill so the feature store never has holes in its ring buffer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import ccxt.async_support as ccxt
import structlog
import websockets

from bus.event_bus import EventBus
from bus.events import CandleEvent
from execution.reconnect import ReconnectManager

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Binance combined-stream base URLs
_MAINNET_WS = "wss://stream.binance.com:9443/stream"
_TESTNET_WS = "wss://testnet.binance.vision/stream"

# ccxt timeframe strings map to Binance interval identifiers 1:1 for common ones
# e.g. "5m" -> "5m", "15m" -> "15m", "1h" -> "1h"


def _pair_to_binance_symbol(pair: str) -> str:
    """Convert a CCXT-style pair to a lowercase Binance symbol.

    Args:
        pair: CCXT pair string, e.g. ``"BTC/USDT"``.

    Returns:
        Lowercase Binance symbol, e.g. ``"btcusdt"``.
    """
    return pair.replace("/", "").lower()


def _binance_symbol_to_pair(symbol: str) -> str:
    """Convert an uppercase Binance symbol to a CCXT-style pair.

    Binance uses conventional quote assets (USDT, BTC, ETH, BNB, BUSD,
    USDC).  This function inserts a ``/`` before the longest matching
    known quote asset suffix.

    Args:
        symbol: Uppercase Binance symbol, e.g. ``"BTCUSDT"``.

    Returns:
        CCXT pair string, e.g. ``"BTC/USDT"``.
    """
    # Ordered longest-first so e.g. "USDT" is matched before "USD"
    known_quotes = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "USD"]
    upper = symbol.upper()
    for quote in known_quotes:
        if upper.endswith(quote):
            base = upper[: -len(quote)]
            return f"{base}/{quote}"
    # Fallback: return as-is (shouldn't happen with well-formed symbols)
    return upper


def _build_stream_url(
    pairs: list[str],
    timeframes: list[str],
    testnet: bool,
) -> str:
    """Construct the Binance combined-stream WebSocket URL.

    Args:
        pairs: CCXT-style pair list, e.g. ``["BTC/USDT", "ETH/USDT"]``.
        timeframes: Timeframe list, e.g. ``["5m", "15m"]``.
        testnet: Use the Binance testnet endpoint when ``True``.

    Returns:
        A fully-qualified ``wss://`` URL string with all stream names
        appended as a ``?streams=`` query parameter.
    """
    stream_names: list[str] = []
    for pair in pairs:
        symbol = _pair_to_binance_symbol(pair)
        for tf in timeframes:
            stream_names.append(f"{symbol}@kline_{tf}")

    base = _TESTNET_WS if testnet else _MAINNET_WS
    return f"{base}?streams={'/'.join(stream_names)}"


class BinanceKlineFeed:
    """Connects to Binance combined WebSocket streams for kline data.

    Subscribes to kline streams for all configured pairs and timeframes.
    Parses messages and publishes :class:`~bus.events.CandleEvent` objects
    to the event bus.

    Gap recovery is performed automatically on reconnect: the time window
    between disconnect and reconnect is backfilled via ccxt REST so the
    feature store ring buffers stay continuous.

    Attributes:
        _event_bus: The system-wide event bus.
        _pairs: CCXT-style trading pairs to subscribe to.
        _timeframes: Candle resolutions to subscribe to.
        _testnet: Whether to use the Binance testnet WebSocket endpoint.
        _reconnect_manager: Handles backoff and reconnection.
        _run_task: The asyncio task running the reconnect loop.
        _gap_windows: List of ``(disconnect_ts_ms, reconnect_ts_ms)``
            tuples accumulated during outages, drained by the backfill task.
    """

    def __init__(
        self,
        event_bus: EventBus,
        pairs: list[str],
        timeframes: list[str],
        testnet: bool = True,
    ) -> None:
        """Initialise the feed without starting any I/O.

        Args:
            event_bus: Shared event bus instance.
            pairs: CCXT-formatted trading pairs, e.g.
                ``["BTC/USDT", "ETH/USDT", "SOL/USDT"]``.
            timeframes: Candle resolutions, e.g. ``["5m", "15m"]``.
            testnet: Target Binance testnet when ``True`` (default).
        """
        self._event_bus = event_bus
        self._pairs = pairs
        self._timeframes = timeframes
        self._testnet = testnet
        self._reconnect_manager = ReconnectManager()
        self._run_task: asyncio.Task[None] | None = None
        # Each entry: (disconnect_ts_s, reconnect_ts_s) pairs for backfill
        self._gap_windows: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Binance WebSocket and start streaming kline data.

        Spawns a background :class:`asyncio.Task` that manages the
        connection lifecycle via :class:`~execution.reconnect.ReconnectManager`.
        Returns immediately; use :meth:`stop` for graceful shutdown.

        Raises:
            RuntimeError: If the feed is already running.
        """
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("BinanceKlineFeed is already running")

        url = _build_stream_url(self._pairs, self._timeframes, self._testnet)
        log.info(
            "binance_ws.starting",
            pairs=self._pairs,
            timeframes=self._timeframes,
            testnet=self._testnet,
            url=url,
        )

        self._run_task = asyncio.create_task(
            self._reconnect_manager.run(
                connect_fn=self._make_connect_fn(url),
                message_handler=self._handle_message,
                on_connect=self._on_connect,
                on_disconnect=self._on_disconnect,
            ),
            name="binance_kline_feed",
        )

    async def stop(self) -> None:
        """Disconnect gracefully and cancel the background task.

        Safe to call multiple times; subsequent calls are no-ops if the
        feed is already stopped.
        """
        self._reconnect_manager.stop()

        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            self._run_task = None

        log.info("binance_ws.stopped")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` when the underlying WebSocket is currently connected."""
        return self._reconnect_manager.is_connected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_connect_fn(
        self,
        url: str,
    ) -> Any:
        """Build an async callable that opens a new WebSocket connection.

        Args:
            url: The combined-stream URL to connect to.

        Returns:
            An async zero-argument callable that returns a
            :class:`websockets.WebSocketClientProtocol`.
        """

        async def _connect() -> websockets.WebSocketClientProtocol:
            return await websockets.connect(  # type: ignore[return-value]
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )

        return _connect

    async def _on_connect(self) -> None:
        """Callback fired by ReconnectManager after each successful connect."""
        log.info("binance_ws.connected", pairs=self._pairs, timeframes=self._timeframes)

    async def _on_disconnect(self, disconnect_ts: float, reconnect_ts: float) -> None:
        """Callback fired by ReconnectManager on each disconnect.

        Records the gap window and triggers REST backfill to recover any
        candles missed during the outage.

        Args:
            disconnect_ts: Unix seconds when the connection was lost.
            reconnect_ts: Unix seconds when reconnection was attempted.
        """
        gap_seconds = reconnect_ts - disconnect_ts
        log.warning(
            "binance_ws.disconnected",
            disconnect_ts=disconnect_ts,
            reconnect_ts=reconnect_ts,
            gap_seconds=round(gap_seconds, 2),
        )
        self._gap_windows.append((disconnect_ts, reconnect_ts))

        # Fire backfill for every pair/timeframe over the gap window.
        since_ms = int(disconnect_ts * 1000)
        until_ms = int(reconnect_ts * 1000)

        for pair in self._pairs:
            for tf in self._timeframes:
                asyncio.create_task(
                    self._backfill_gaps(pair, tf, since_ms, until_ms),
                    name=f"backfill_{pair}_{tf}",
                )

    async def _handle_message(self, raw: str) -> None:
        """Parse a raw Binance combined-stream message and publish a CandleEvent.

        Silently discards messages that cannot be parsed or that do not
        carry kline data.

        Args:
            raw: Raw JSON text received from the WebSocket.
        """
        log.debug("binance_ws.raw_message_received", raw_len=len(raw))
        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("binance_ws.invalid_json", raw=raw[:200])
            return

        data = payload.get("data", {})
        event_type = data.get("e")
        log.debug("binance_ws.message_parsed", event_type=event_type)
        if event_type != "kline":
            # Not a kline event (e.g. subscription confirmation)
            return

        kline: dict[str, Any] = data.get("k", {})

        try:
            event = CandleEvent(
                pair=_binance_symbol_to_pair(kline["s"]),
                timeframe=kline["i"],
                timestamp_ms=int(kline["t"]),
                open=float(kline["o"]),
                high=float(kline["h"]),
                low=float(kline["l"]),
                close=float(kline["c"]),
                volume=float(kline["v"]),
                is_closed=bool(kline["x"]),
            )
        except (KeyError, ValueError, TypeError):
            log.warning("binance_ws.malformed_kline", kline=kline, exc_info=True)
            return

        log.info("binance_ws.candle_event", pair=event.pair, timeframe=event.timeframe, is_closed=event.is_closed)
        await self._event_bus.publish("candle", event)

    async def _backfill_gaps(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
    ) -> None:
        """Fetch missing candles via ccxt REST and publish them to the bus.

        Uses ``ccxt.binance`` (or ``ccxt.binance`` testnet) synchronously
        wrapped in the async ccxt layer.  Each recovered candle is
        published as a closed :class:`~bus.events.CandleEvent` so the
        feature store ring buffer stays continuous.

        Args:
            pair: CCXT-style pair, e.g. ``"BTC/USDT"``.
            timeframe: Candle resolution, e.g. ``"5m"``.
            since_ms: Start of the gap window in milliseconds (inclusive).
            until_ms: End of the gap window in milliseconds (exclusive).
        """
        gap_seconds = (until_ms - since_ms) / 1000
        log.info(
            "binance_ws.backfill_start",
            pair=pair,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            gap_seconds=round(gap_seconds, 1),
        )

        exchange_kwargs: dict[str, Any] = {}
        if self._testnet:
            exchange_kwargs["options"] = {"defaultType": "spot"}
            exchange_kwargs["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }

        exchange = ccxt.binance(exchange_kwargs)

        try:
            ohlcv: list[list[Any]] = await exchange.fetch_ohlcv(
                pair,
                timeframe=timeframe,
                since=since_ms,
                limit=1000,
            )
        except Exception:
            log.warning(
                "binance_ws.backfill_failed",
                pair=pair,
                timeframe=timeframe,
                exc_info=True,
            )
            return
        finally:
            await exchange.close()

        recovered = 0
        for row in ohlcv:
            ts_ms: int = int(row[0])
            if ts_ms > until_ms:
                break
            event = CandleEvent(
                pair=pair,
                timeframe=timeframe,
                timestamp_ms=ts_ms,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                is_closed=True,
            )
            await self._event_bus.publish("candle", event)
            recovered += 1

        log.info(
            "binance_ws.backfill_complete",
            pair=pair,
            timeframe=timeframe,
            candles_recovered=recovered,
        )


# ---------------------------------------------------------------------------
# Module-level symbol conversion helpers (re-exported for convenience)
# ---------------------------------------------------------------------------

__all__ = [
    "BinanceKlineFeed",
    "_pair_to_binance_symbol",
    "_binance_symbol_to_pair",
    "_build_stream_url",
]
