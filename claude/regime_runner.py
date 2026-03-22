"""Scheduled Claude API calls for market-regime detection.

:class:`RegimeRunner` is responsible for the one-way data flow::

    FeatureStore (market data)
        → prompt construction
        → Anthropic SDK call
        → :func:`~claude.response_parser.parse_regime_response`
        → :meth:`~strategy.controller.StrategyController.update_regime`

Failure contract
----------------
If a single ``run_once()`` call fails for any reason (API error, timeout,
invalid JSON, Pydantic validation failure) the runner logs the error and
returns ``None``.  It does **not** clear the current regime – the existing
value will eventually expire via the P0-1 TTL.

The ``run_scheduled()`` coroutine runs forever until ``stop()`` is called,
sleeping ``interval_hours`` between each attempt.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anthropic
import structlog

from claude.response_parser import parse_regime_response
from config.settings import ClaudeConfig
from data.feature_store import FeatureStore
from strategy.controller import StrategyController

log: structlog.BoundLogger = structlog.get_logger(__name__)


class RegimeRunner:
    """Calls Claude on a schedule for regime detection.

    Uses the official ``anthropic.AsyncAnthropic`` client with pre-fetched
    market data embedded in the prompt as context.

    Args:
        api_key: Anthropic API key.  When empty the runner will warn on
            construction and return ``None`` immediately on each call.
        model: Anthropic model identifier from :class:`~config.settings.ClaudeConfig`.
        controller: :class:`~strategy.controller.StrategyController` used to
            persist the detected regime.
        feature_store: :class:`~data.feature_store.FeatureStore` used to
            build the market-data summary.
        config: :class:`~config.settings.ClaudeConfig` for timeout and path
            settings.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        controller: StrategyController,
        feature_store: FeatureStore,
        config: ClaudeConfig,
    ) -> None:
        self._api_key: str = api_key
        self._model: str = model
        self._controller: StrategyController = controller
        self._feature_store: FeatureStore = feature_store
        self._config: ClaudeConfig = config
        self._stop_event: asyncio.Event = asyncio.Event()

        if not api_key:
            log.warning("regime_runner.empty_api_key")

        self._client: anthropic.AsyncAnthropic | None = (
            anthropic.AsyncAnthropic(api_key=api_key) if api_key else None
        )

        self._prompt_template: str = self._load_prompt_template(config.regime_prompt_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(self) -> str | None:
        """Execute a single regime-detection call.

        Steps:

        1. Build a market-data summary dict from the feature store.
        2. Render the prompt template with the summary.
        3. Call Claude with a timeout from ``config.timeout_seconds``.
        4. Parse and validate the response.
        5. Persist to the controller on success.

        Returns:
            The regime name string on success, or ``None`` on any failure.
            Failures preserve the last-known regime (which the TTL will
            eventually expire).
        """
        if self._client is None:
            log.warning("regime_runner.no_client_skip")
            return None

        market_summary = self._build_market_summary()
        prompt = self._build_prompt(market_summary)

        try:
            message = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=float(self._config.timeout_seconds),
            )
        except TimeoutError:
            log.error(
                "regime_runner.api_timeout",
                timeout_seconds=self._config.timeout_seconds,
            )
            return None
        except anthropic.APIError as exc:
            log.error("regime_runner.api_error", error=str(exc))
            return None
        except Exception:
            log.error("regime_runner.unexpected_error", exc_info=True)
            return None

        raw_text = self._extract_text(message)
        if raw_text is None:
            log.error("regime_runner.empty_api_response")
            return None

        parsed = parse_regime_response(raw_text)
        if parsed is None:
            log.error(
                "regime_runner.parse_failed",
                preview=raw_text[:200],
            )
            return None

        await self._controller.update_regime(parsed.regime, parsed.confidence)
        log.info(
            "regime_runner.regime_detected",
            regime=parsed.regime,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
        )
        return parsed.regime

    async def run_scheduled(self, interval_hours: float = 6.0) -> None:
        """Run regime detection in a loop until :meth:`stop` is called.

        The first call fires immediately; subsequent calls are delayed by
        *interval_hours*.

        Args:
            interval_hours: Time to sleep between consecutive calls.
        """
        self._stop_event.clear()
        log.info(
            "regime_runner.scheduled_start",
            interval_hours=interval_hours,
        )

        while not self._stop_event.is_set():
            await self.run_once()

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval_hours * 3600.0,
                )
            except TimeoutError:
                pass  # Normal path: interval elapsed, loop continues.

        log.info("regime_runner.scheduled_stopped")

    async def stop(self) -> None:
        """Signal :meth:`run_scheduled` to exit after the current sleep."""
        self._stop_event.set()
        log.info("regime_runner.stop_requested")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, market_summary: dict[str, object]) -> str:
        """Render the regime detection prompt with *market_summary* injected.

        Args:
            market_summary: Aggregated market metrics from the feature store.

        Returns:
            The complete prompt string ready to send to Claude.
        """
        summary_json = json.dumps(market_summary, indent=2)
        return self._prompt_template.replace("{market_summary}", summary_json)

    def _build_market_summary(self) -> dict[str, object]:
        """Aggregate market data from the feature store for prompt context.

        Iterates over all tracked pair/timeframe buffers and collects the
        most recent OHLCV values.  If the store is empty a minimal
        placeholder dict is returned so the prompt is still valid.

        Returns:
            A JSON-serialisable dict suitable for embedding in the prompt.
        """
        summary: dict[str, object] = {"note": "MVP summary from in-memory ring buffers"}
        pairs_data: dict[str, object] = {}

        for key, buf in self._feature_store._buffers.items():
            count = self._feature_store._buffer_counts.get(key, 0)
            if count == 0:
                continue

            pos = self._feature_store._buffer_positions[key]
            last_idx = (pos - 1) % self._feature_store._max_candles
            last = buf[last_idx]

            pairs_data[key] = {
                "last_close": float(last["close"]),
                "last_volume": float(last["volume"]),
                "candles_available": count,
            }

        summary["pairs"] = pairs_data if pairs_data else {"status": "no_data_yet"}
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompt_template(path: str) -> str:
        """Load the prompt markdown template from *path*.

        Falls back to an embedded minimal template if the file is absent so
        that the runner degrades gracefully in unit tests and stripped
        environments.

        Args:
            path: Filesystem path to the ``.md`` template.

        Returns:
            The template string with ``{market_summary}`` placeholder.
        """
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")

        log.warning("regime_runner.prompt_not_found", path=path)
        return (
            'Analyze this crypto market data and return ONLY a JSON object:\n'
            '{market_summary}\n\n'
            'Return: {"regime": "unknown"|"trending_up"|"trending_down"'
            '|"ranging"|"high_volatility", "confidence": 0.0-1.0, '
            '"reasoning": "...", "suggested_pairs": [], "risk_assessment": "medium"}'
        )

    @staticmethod
    def _extract_text(message: anthropic.types.Message) -> str | None:
        """Pull the text content from a :class:`anthropic.types.Message`.

        Args:
            message: Response object from the Anthropic API.

        Returns:
            The concatenated text of all ``TextBlock`` content items, or
            ``None`` if no text blocks are present.
        """
        parts: list[str] = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts) if parts else None
