"""SQLAlchemy async ORM models for all audit data.

Every domain event — candles, signals, risk decisions, orders, and regime
changes — is persisted to a corresponding table.  All tables use a simple
integer surrogate primary key with ``created_at`` timestamps for ordering.

The ``OrderRecord`` table enforces a UNIQUE constraint on ``client_order_id``
to satisfy the P0-3 dual-layer idempotency requirement: even if the
application crashes and retries, a duplicate row can never be inserted.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ledger ORM models."""


class CandleRecord(Base):
    """Persisted OHLCV candle data.

    Attributes:
        id: Surrogate primary key.
        timestamp_ms: Unix timestamp in milliseconds of the candle open.
        pair: CCXT-formatted trading pair (e.g. ``"BTC/USDT"``).
        timeframe: Timeframe string (e.g. ``"1m"``, ``"5m"``).
        open: Opening price.
        high: Highest price during the candle.
        low: Lowest price during the candle.
        close: Closing price.
        volume: Volume traded during the candle (base asset units).
    """

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    pair: Mapped[str] = mapped_column(String(20), index=True)
    timeframe: Mapped[str] = mapped_column(String(5))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class SignalRecord(Base):
    """Persisted trading signal produced by a signal generator.

    Attributes:
        id: Surrogate primary key.
        created_at: Wall-clock time when the row was inserted.
        pair: CCXT-formatted trading pair.
        direction: Signal direction (``"long"``, ``"short"``, or ``"neutral"``).
        strength: Normalised conviction in ``[0.0, 1.0]``.
        indicator_name: Name of the generator that produced the signal.
        signal_timestamp_ms: Unix timestamp (ms) of the most recent candle
            that produced this signal.
        metadata_json: JSON-serialised ``dict[str, float]`` of diagnostic data.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
    pair: Mapped[str] = mapped_column(String(20), index=True)
    direction: Mapped[str] = mapped_column(String(10))
    strength: Mapped[float] = mapped_column(Float)
    indicator_name: Mapped[str] = mapped_column(String(50))
    signal_timestamp_ms: Mapped[int] = mapped_column(BigInteger)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class RiskDecisionRecord(Base):
    """Persisted risk gate evaluation result linked to an originating signal.

    Attributes:
        id: Surrogate primary key.
        created_at: Wall-clock time when the row was inserted.
        signal_id: FK-style reference to :attr:`SignalRecord.id`.
        approved: ``True`` when all checks passed.
        reason: Human-readable explanation of the decision.
        adjusted_quantity: Size-adjusted quantity if a check reduced it, else
            ``None``.
        checks_passed_json: JSON array of check names that passed.
        checks_failed_json: JSON array of check names that failed.
    """

    __tablename__ = "risk_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
    signal_id: Mapped[int] = mapped_column(Integer, index=True)
    approved: Mapped[bool]
    reason: Mapped[str] = mapped_column(String(500))
    adjusted_quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    checks_passed_json: Mapped[str] = mapped_column(Text, default="[]")
    checks_failed_json: Mapped[str] = mapped_column(Text, default="[]")


class OrderRecord(Base):
    """Persisted order lifecycle record.

    The ``client_order_id`` column carries a UNIQUE constraint to enforce the
    P0-3 crash-safe idempotency guarantee at the database layer.  Any attempt
    to insert a duplicate order will raise an ``IntegrityError`` before it
    reaches the exchange.

    Attributes:
        id: Surrogate primary key.
        created_at: Wall-clock time when the row was first inserted.
        updated_at: Wall-clock time of the most recent update.
        client_order_id: Caller-generated idempotency key (unique).
        exchange_order_id: Exchange-assigned identifier, or ``None`` until
            the order is acknowledged.
        signal_id: Identifier of the originating signal (string form of the
            signal's unique key).
        pair: CCXT-formatted trading pair.
        side: ``"buy"`` or ``"sell"``.
        requested_quantity: Quantity originally requested.
        filled_quantity: Quantity matched so far.
        filled_price: Volume-weighted average fill price.
        status: Current :class:`~execution.base.OrderStatus` string value.
        fees: Cumulative fees paid (quote-asset units).
        transitions_json: JSON array of ``[status, iso_timestamp]`` pairs
            representing the full state-machine audit trail.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[str] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    client_order_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    signal_id: Mapped[str] = mapped_column(String(100), index=True)
    pair: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))
    requested_quantity: Mapped[float] = mapped_column(Float)
    filled_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    filled_price: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), index=True)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    transitions_json: Mapped[str] = mapped_column(Text, default="[]")


class RegimeRecord(Base):
    """Persisted market regime classification produced by the Claude brain.

    Attributes:
        id: Surrogate primary key.
        created_at: Wall-clock time when the regime was recorded.
        regime: Regime label string (e.g. ``"trending"``, ``"ranging"``,
            ``"volatile"``).
        confidence: Model confidence in ``[0.0, 1.0]``.
        raw_response: Raw LLM response text for auditability.
        active_pairs_json: JSON array of trading pairs active in this regime.
        active_strategies_json: JSON array of strategy names active in this
            regime.
    """

    __tablename__ = "regimes"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
    regime: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)
    raw_response: Mapped[str] = mapped_column(Text, default="")
    active_pairs_json: Mapped[str] = mapped_column(Text, default="[]")
    active_strategies_json: Mapped[str] = mapped_column(Text, default="[]")
