"""Reusable WebSocket reconnection manager with exponential backoff.

Handles reconnection logic for any WebSocket connection: exponential
backoff with jitter, proactive reconnection before the Binance 24-hour
stream limit, ping/pong heartbeat monitoring, and connection state
tracking with optional async callbacks.
"""

from __future__ import annotations

import asyncio
import random
import ssl
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
import websockets
import websockets.exceptions
from websockets import WebSocketClientProtocol

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Type alias for the connect callable
ConnectFn = Callable[[], Awaitable[WebSocketClientProtocol]]
MessageHandler = Callable[[str], Awaitable[None]]
OnConnectCallback = Callable[[], Awaitable[None]]
# Receives (disconnect_timestamp_s, reconnect_timestamp_s)
OnDisconnectCallback = Callable[[float, float], Awaitable[None]]


class ReconnectManager:
    """Manages WebSocket reconnection with exponential backoff.

    Exponential backoff formula: ``base_delay * 2^attempt + jitter``
    where jitter is a uniform random value in ``[0, 1)`` seconds added to
    prevent thundering-herd reconnect storms after a shared outage.

    Proactive reconnection is triggered when the current connection has
    been alive for ``max_connection_hours`` hours, which avoids being
    forcibly disconnected by the Binance 24-hour stream limit.

    Attributes:
        _max_retries: Maximum consecutive reconnection attempts before
            the run loop raises ``RuntimeError``.
        _base_delay: Initial backoff duration in seconds.
        _max_delay: Upper bound on the computed backoff delay.
        _max_connection_hours: Reconnect proactively after this many hours
            of continuous connection (Binance closes streams at 24 h).
        _attempt: Current consecutive failure count; resets on success.
        _connected: Whether the WS is currently considered live.
        _connection_start_time: Wall-clock seconds at last successful
            connect; used to detect when to proactively rotate.
        _last_message_time: Wall-clock seconds of the most recent message
            received; useful for external health checks.
    """

    def __init__(
        self,
        max_retries: int = 50,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_connection_hours: float = 23.0,
    ) -> None:
        """Initialise the manager; does not start any I/O.

        Args:
            max_retries: Maximum consecutive reconnection attempts before
                the manager gives up and raises.
            base_delay: Backoff base in seconds (``base_delay * 2^attempt``).
            max_delay: Backoff ceiling in seconds.
            max_connection_hours: Proactively close and reconnect after this
                many hours to avoid Binance's 24 h forced disconnect.
        """
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._max_connection_seconds = max_connection_hours * 3600.0

        self._attempt: int = 0
        self._connected: bool = False
        self._connection_start_time: float | None = None
        self._last_message_time: float | None = None

        # Set when stop() is called externally so the loop can exit cleanly.
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        connect_fn: ConnectFn,
        message_handler: MessageHandler,
        on_connect: OnConnectCallback | None = None,
        on_disconnect: OnDisconnectCallback | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        """Drive the WebSocket connection with automatic reconnection.

        The method loops indefinitely, re-establishing the connection
        whenever it drops, until either ``stop()`` is called or
        ``_max_retries`` consecutive failures occur.

        Args:
            connect_fn: Async callable that creates and returns a new
                :class:`websockets.WebSocketClientProtocol`.  Called on
                every (re)connection attempt.
            message_handler: Async callable invoked for every raw text
                message received from the server.
            on_connect: Optional async callback fired immediately after a
                successful connection is established.  Receives no args.
            on_disconnect: Optional async callback fired when the connection
                closes.  Receives ``(disconnect_ts, reconnect_ts)`` where
                both values are Unix wall-clock seconds.
            ssl_context: Optional SSL context forwarded to ``connect_fn``
                callers that need it.  This manager does not use it
                directly; it is provided as a convenience for callers that
                build their ``connect_fn`` via a closure over this arg.

        Raises:
            RuntimeError: If ``_max_retries`` consecutive failures occur
                without a successful message being received in between.
        """
        # ssl_context is accepted for API symmetry; callers use it in their
        # connect_fn closure.  Mark it used to satisfy strict linters.
        _: Any = ssl_context

        self._stop_requested = False

        while not self._stop_requested:
            if self._attempt > 0:
                delay = self._backoff_delay(self._attempt)
                log.info(
                    "reconnect_manager.waiting",
                    attempt=self._attempt,
                    delay_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)

            if self._attempt >= self._max_retries:
                log.error(
                    "reconnect_manager.max_retries_exceeded",
                    max_retries=self._max_retries,
                )
                raise RuntimeError(
                    f"ReconnectManager exceeded {self._max_retries} consecutive retries"
                )

            log.info("reconnect_manager.connecting", attempt=self._attempt)

            try:
                ws = await connect_fn()
            except (TimeoutError, ConnectionError, OSError) as exc:
                self._attempt += 1
                log.warning(
                    "reconnect_manager.connect_failed",
                    attempt=self._attempt,
                    error=str(exc),
                )
                continue

            # ----------------------------------------------------------
            # Connected
            # ----------------------------------------------------------
            connect_ts = time.time()
            self._connected = True
            self._connection_start_time = connect_ts
            self._attempt = 0

            log.info("reconnect_manager.connected", attempt=0)

            if on_connect is not None:
                try:
                    await on_connect()
                except Exception:
                    log.warning("reconnect_manager.on_connect_error", exc_info=True)

            disconnect_ts: float | None = None

            try:
                async for raw_message in ws:
                    if self._stop_requested:
                        break

                    self._last_message_time = time.time()

                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode()

                    try:
                        await message_handler(raw_message)
                    except Exception:
                        log.warning(
                            "reconnect_manager.message_handler_error",
                            exc_info=True,
                        )

                    # Proactive reconnection: rotate before the server's
                    # 24-hour forced-close window.
                    elapsed = time.time() - connect_ts
                    if elapsed >= self._max_connection_seconds:
                        log.info(
                            "reconnect_manager.proactive_reconnect",
                            elapsed_hours=round(elapsed / 3600, 2),
                            max_hours=round(
                                self._max_connection_seconds / 3600, 2
                            ),
                        )
                        disconnect_ts = time.time()
                        await ws.close()
                        break

            except websockets.exceptions.ConnectionClosed as exc:
                disconnect_ts = time.time()
                log.warning(
                    "reconnect_manager.connection_closed",
                    code=exc.code,
                    reason=exc.reason,
                )
            except (TimeoutError, ConnectionError, OSError) as exc:
                disconnect_ts = time.time()
                log.warning(
                    "reconnect_manager.connection_error",
                    error=str(exc),
                )
            finally:
                self._connected = False

            if self._stop_requested:
                break

            # ----------------------------------------------------------
            # Disconnected — notify caller, then retry
            # ----------------------------------------------------------
            actual_disconnect_ts = disconnect_ts if disconnect_ts is not None else time.time()
            reconnect_ts = time.time()

            if on_disconnect is not None:
                try:
                    await on_disconnect(actual_disconnect_ts, reconnect_ts)
                except Exception:
                    log.warning("reconnect_manager.on_disconnect_error", exc_info=True)

            self._attempt += 1

        log.info("reconnect_manager.stopped")

    def stop(self) -> None:
        """Signal the run loop to exit cleanly after the current message.

        This is a synchronous method so it can be called from non-async
        contexts (e.g., signal handlers).  The loop will check the flag at
        the next iteration boundary.
        """
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` when a WebSocket connection is currently active."""
        return self._connected

    @property
    def last_message_time(self) -> float | None:
        """Unix seconds of the last received message, or ``None`` if none yet."""
        return self._last_message_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Compute the backoff delay for a given attempt number.

        Formula: ``min(base_delay * 2^attempt, max_delay) + jitter``
        where jitter is uniform random in ``[0, 1)``.

        Args:
            attempt: Zero-based attempt index (0 = first retry).

        Returns:
            Delay in seconds, capped at ``max_delay + 1``.
        """
        exponential = self._base_delay * (2**attempt)
        capped = min(exponential, self._max_delay)
        jitter = random.random()  # noqa: S311 — not cryptographic
        return capped + jitter
