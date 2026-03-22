"""Strict validation of Claude's regime detection JSON response.

Claude is instructed to return **only** a JSON object, but in practice it
may wrap the payload in markdown fences (```json ... ```) or include a short
preamble sentence.  :func:`parse_regime_response` handles that reality by
first extracting the first ``{...}`` block it can find before running
Pydantic validation.

All parse failures are logged at WARNING level and result in a ``None``
return value – the caller (``RegimeRunner``) then keeps the last-known
regime, which will eventually expire via the P0-1 TTL mechanism.
"""

from __future__ import annotations

import json
import re

import structlog
from pydantic import BaseModel, ValidationError, field_validator

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Regex to locate the first JSON object in a string.  We look for the
# outermost ``{...}`` span, which handles both bare JSON and JSON embedded
# inside markdown code fences or prose.
_JSON_RE: re.Pattern[str] = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


class RegimeResponse(BaseModel):
    """Validated regime response from Claude.

    Strict validation rules:

    * ``regime`` must be a value of :class:`~strategy.modes.Regime`.
    * ``confidence`` must be in ``[0.0, 1.0]``.
    * ``suggested_pairs`` may be empty; no universe check at this layer.
    * ``risk_assessment`` defaults to ``"medium"`` when omitted.

    Attributes:
        regime: Classified market regime name.
        confidence: Claude's confidence score (0 = none, 1 = certain).
        reasoning: Short human-readable explanation.
        suggested_pairs: Optional list of pairs Claude recommends.
        risk_assessment: Qualitative risk level (``"low"``, ``"medium"``,
            or ``"high"``).
    """

    regime: str
    confidence: float
    reasoning: str
    suggested_pairs: list[str] = []
    risk_assessment: str = "medium"

    @field_validator("regime")
    @classmethod
    def validate_regime(cls, v: str) -> str:
        """Ensure *v* is a known :class:`~strategy.modes.Regime` value.

        Args:
            v: Raw regime string from Claude.

        Returns:
            The validated regime string.

        Raises:
            ValueError: If *v* is not a member of :class:`~strategy.modes.Regime`.
        """
        from strategy.modes import Regime

        valid = {r.value for r in Regime}
        if v not in valid:
            raise ValueError(f"Unknown regime: {v!r}. Must be one of {valid}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure *v* is within ``[0.0, 1.0]``.

        Args:
            v: Raw confidence value from Claude.

        Returns:
            The validated confidence float.

        Raises:
            ValueError: If *v* is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return v


def parse_regime_response(raw_text: str) -> RegimeResponse | None:
    """Parse and validate Claude's raw text response.

    The function:

    1. Searches *raw_text* for the first ``{...}`` block (handles markdown
       fences and prose preambles).
    2. Deserialises the block with :mod:`json`.
    3. Validates the payload with :class:`RegimeResponse`.
    4. Returns ``None`` on any failure, logging a WARNING with the reason.

    Args:
        raw_text: The full string returned by the Claude API.

    Returns:
        A validated :class:`RegimeResponse`, or ``None`` if the text could
        not be parsed or failed validation.
    """
    if not raw_text or not raw_text.strip():
        log.warning("response_parser.empty_response")
        return None

    match = _JSON_RE.search(raw_text)
    if match is None:
        log.warning(
            "response_parser.no_json_found",
            preview=raw_text[:200],
        )
        return None

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning(
            "response_parser.json_decode_error",
            error=str(exc),
            json_str=json_str[:200],
        )
        return None

    try:
        response = RegimeResponse.model_validate(data)
    except ValidationError as exc:
        log.warning(
            "response_parser.validation_error",
            errors=exc.errors(),
        )
        return None

    log.debug(
        "response_parser.parsed",
        regime=response.regime,
        confidence=response.confidence,
    )
    return response
