"""Core types for the risk gate system.

This module defines the fundamental data structures that flow through the risk
pipeline.  The most important type is :class:`ApprovedOrder`, which is a
cryptographically signed token produced **only** by :class:`~risk.gate.RiskGate`.

P0-2 Invariant
--------------
The Execution Engine MUST call :func:`verify_gate_token` before submitting any
order.  An ``ApprovedOrder`` whose ``gate_token`` does not pass verification was
either forged or tampered with and must be silently dropped.

The signing secret (``_GATE_SECRET``) is generated fresh from ``os.urandom``
each time the process starts.  It is never persisted, so tokens are
automatically invalidated on restart.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Process-local signing secret
# ---------------------------------------------------------------------------

# Generated at module import time.  A fresh 256-bit secret per process makes
# it impossible to forge an ApprovedOrder from outside the risk gate process.
_GATE_SECRET: bytes = os.urandom(32)


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------


def _compute_gate_token(
    signal_id: str,
    pair: str,
    side: str,
    quantity: float,
    timestamp: float,
) -> str:
    """Compute HMAC-SHA256 token for an approved order.

    The message is a colon-delimited canonical string that includes every
    field that the Execution Engine will act on.  Changing any field after
    signing will cause :func:`verify_gate_token` to return ``False``.

    Args:
        signal_id: Unique identifier of the originating signal.
        pair: Trading pair (e.g. ``"BTC/USDT"``).
        side: ``"buy"`` or ``"sell"``.
        quantity: Asset quantity to trade, formatted to 8 decimal places.
        timestamp: Unix timestamp (``time.time()``) when the order was approved.

    Returns:
        Lowercase hexadecimal HMAC-SHA256 digest string.
    """
    message = f"{signal_id}:{pair}:{side}:{quantity:.8f}:{timestamp:.6f}"
    return hmac.new(_GATE_SECRET, message.encode(), hashlib.sha256).hexdigest()


def verify_gate_token(order: ApprovedOrder) -> bool:
    """Verify that an :class:`ApprovedOrder`'s gate_token is authentic.

    Uses :func:`hmac.compare_digest` for constant-time comparison to prevent
    timing-based side-channel attacks.

    Args:
        order: The :class:`ApprovedOrder` to verify.

    Returns:
        ``True`` if the token is valid, ``False`` otherwise.
    """
    expected = _compute_gate_token(
        order.signal_id,
        order.pair,
        order.side,
        order.quantity,
        order.approved_at,
    )
    return hmac.compare_digest(order.gate_token, expected)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovedOrder:
    """Token produced ONLY by :meth:`~risk.gate.RiskGate.evaluate` when all checks pass.

    The ``gate_token`` is an HMAC-SHA256 of the order fields using a
    process-local secret.  The Execution Engine verifies this token via
    :func:`verify_gate_token` before submitting any order to the exchange.

    Instances are frozen (immutable) so that fields cannot be mutated after
    the HMAC is computed.

    Attributes:
        signal_id: Unique identifier linking this order back to its originating
            :class:`~signals.base.Signal`.
        pair: CCXT-formatted trading pair (e.g. ``"BTC/USDT"``).
        side: ``"buy"`` for long entries / short exits, ``"sell"`` for the
            reverse.
        quantity: Approved asset quantity.  May be lower than the originally
            requested quantity if :class:`~risk.position_limit.PositionLimitCheck`
            adjusted it downward.
        approved_at: Unix timestamp (``time.time()``) recorded at the moment of
            approval.  Included in the HMAC to prevent replay attacks.
        gate_token: HMAC-SHA256 hex digest.  Verified by the Execution Engine.
        risk_checks_passed: Ordered list of check names that passed, for
            auditability.
        original_signal_strength: Strength value from the originating signal
            (``0.0`` to ``1.0``), carried through for downstream logging.
    """

    signal_id: str
    pair: str
    side: str  # "buy" or "sell"
    quantity: float
    approved_at: float  # time.time() when approved
    gate_token: str
    risk_checks_passed: list[str] = field(default_factory=list)
    original_signal_strength: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    """Result of a single risk check.

    Each check in the pipeline returns a :class:`RiskDecision` that indicates
    whether the trade may proceed and, optionally, a size-adjusted quantity.

    Attributes:
        approved: ``True`` when the check passes (or passes with a warning).
        reason: Human-readable explanation of the decision.
        adjusted_quantity: When not ``None``, the check has reduced the
            requested quantity to this value.  The gate will use this value for
            subsequent checks.
        checks_passed: Names of sub-checks that passed within this evaluator.
        checks_failed: Names of sub-checks that failed within this evaluator.
    """

    approved: bool
    reason: str
    adjusted_quantity: float | None = None
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Current portfolio snapshot consumed by every risk check.

    This is a plain data container populated by the Strategy Controller before
    calling :meth:`~risk.gate.RiskGate.evaluate`.  All monetary values are in
    USD.

    Attributes:
        total_balance_usd: Total equity including unrealized PnL.
        open_positions: Map from pair to currently held asset quantity.
        open_position_values: Map from pair to notional USD value of the
            position (quantity * mark price).
        daily_realized_pnl: Realized PnL since the most recent UTC midnight
            (may be negative).
        daily_unrealized_pnl: Unrealized PnL across all open positions.  This
            value is EMA-smoothed by :class:`~risk.daily_loss.DailyLossCheck`
            before being used in loss calculations.
    """

    total_balance_usd: float
    open_positions: dict[str, float]  # pair -> quantity
    open_position_values: dict[str, float]  # pair -> notional USD value
    daily_realized_pnl: float
    daily_unrealized_pnl: float
