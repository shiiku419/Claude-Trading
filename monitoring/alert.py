"""Alert dispatcher for Slack webhook and Telegram bot notifications.

Alerts are rate-limited to one message per component per 5 minutes except
for CRITICAL level, which always delivers immediately.  When no webhook or
token is configured the dispatcher logs only and makes no network calls.
"""

from __future__ import annotations

import time
from enum import StrEnum

import httpx
import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)

_RATE_LIMIT_SECONDS: float = 300.0  # 5 minutes


class AlertLevel(StrEnum):
    """Severity level for dispatched alerts.

    Attributes:
        INFO: Regime change, daily summary — low urgency.
        WARNING: Reconnection attempt, degraded performance.
        CRITICAL: Kill switch activated, loss limit breached, or connection
            lost for more than 5 minutes.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertDispatcher:
    """Sends alerts via Slack webhook and/or Telegram bot.

    Rate limiting is applied per ``(component, level)`` pair: non-CRITICAL
    alerts are suppressed if the same component sent an alert fewer than
    5 minutes ago.  CRITICAL alerts always bypass the rate limiter.

    When neither a Slack webhook URL nor a Telegram bot token is provided
    the dispatcher falls back to structured logging only — no network calls
    are made.

    Example::

        dispatcher = AlertDispatcher(
            slack_webhook="https://hooks.slack.com/...",
            telegram_bot_token="123:ABC",
            telegram_chat_id="-100123456",
        )
        await dispatcher.send(AlertLevel.CRITICAL, "risk_gate", "Kill switch activated")

    Args:
        slack_webhook: Incoming webhook URL.  Empty string disables Slack.
        telegram_bot_token: Bot API token from BotFather.  Empty string
            disables Telegram.
        telegram_chat_id: Target chat / channel ID for Telegram messages.
    """

    def __init__(
        self,
        slack_webhook: str = "",
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
    ) -> None:
        self._slack_webhook: str = slack_webhook
        self._telegram_bot_token: str = telegram_bot_token
        self._telegram_chat_id: str = telegram_chat_id
        # Tracks the last delivery time keyed by component name.
        self._last_sent: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, level: AlertLevel, component: str, message: str) -> None:
        """Dispatch an alert at the specified severity level.

        The alert is rate-limited per component unless *level* is
        :attr:`AlertLevel.CRITICAL`.  The formatted text is always
        emitted as a structured log line regardless of whether a webhook
        is configured.

        Args:
            level: Severity of the alert.
            component: Logical component name, e.g. ``"websocket"`` or
                ``"risk_gate"``.  Used as the rate-limit key.
            message: Human-readable description of the event.
        """
        now = time.monotonic()

        if level is not AlertLevel.CRITICAL:
            last = self._last_sent.get(component, 0.0)
            if now - last < _RATE_LIMIT_SECONDS:
                log.debug(
                    "alert_dispatcher.rate_limited",
                    component=component,
                    level=level,
                    seconds_until_next=_RATE_LIMIT_SECONDS - (now - last),
                )
                return

        self._last_sent[component] = now
        text = f"[{level.upper()}] {component}: {message}"

        log.info(
            "alert_dispatcher.sending",
            level=level,
            component=component,
            message=message,
        )

        if self._slack_webhook:
            await self._send_slack(text)

        if self._telegram_bot_token and self._telegram_chat_id:
            await self._send_telegram(text)

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    async def _send_slack(self, text: str) -> None:
        """POST *text* to the configured Slack incoming-webhook URL.

        Failures are logged as warnings and swallowed so that a Slack
        outage never blocks trading operations.

        Args:
            text: Plain-text alert message to deliver.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._slack_webhook, json={"text": text})
                resp.raise_for_status()
                log.debug("alert_dispatcher.slack_sent", status=resp.status_code)
        except Exception as exc:  # noqa: BLE001
            log.warning("alert_dispatcher.slack_error", error=repr(exc))

    async def _send_telegram(self, text: str) -> None:
        """Send *text* via the Telegram Bot API ``sendMessage`` endpoint.

        Failures are logged as warnings and swallowed.

        Args:
            text: Plain-text alert message to deliver.
        """
        url = f"https://api.telegram.org/bot{self._telegram_bot_token}/sendMessage"
        payload = {"chat_id": self._telegram_chat_id, "text": text}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                log.debug("alert_dispatcher.telegram_sent", status=resp.status_code)
        except Exception as exc:  # noqa: BLE001
            log.warning("alert_dispatcher.telegram_error", error=repr(exc))
