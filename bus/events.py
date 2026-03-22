"""Event type definitions and topic configuration for the asyncio event bus.

All events are Pydantic models with a stable event_id, timestamp, and a
ClassVar ``topic`` that determines which bus channel they are published to.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Drop / backpressure policy
# ---------------------------------------------------------------------------


class DropPolicy(StrEnum):
    """Backpressure behaviour when a subscriber queue is full.

    LOSSY: silently drop the oldest event to make room for the new one.
    BLOCKING: await until space is available (with a timeout safeguard).
    """

    LOSSY = "lossy"
    BLOCKING = "blocking"


# ---------------------------------------------------------------------------
# Topic configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicConfig:
    """Queue configuration for a single event bus topic.

    Attributes:
        name: Canonical topic name used as the routing key.
        maxsize: Maximum number of events buffered per subscriber queue.
        policy: What to do when a queue is full.
    """

    name: str
    maxsize: int
    policy: DropPolicy


#: Topic configuration registry.  Keys are the topic routing strings.
TOPIC_CONFIGS: dict[str, TopicConfig] = {
    "candle": TopicConfig(name="candle", maxsize=10_000, policy=DropPolicy.LOSSY),
    "trade": TopicConfig(name="trade", maxsize=10_000, policy=DropPolicy.LOSSY),
    "orderbook": TopicConfig(name="orderbook", maxsize=5_000, policy=DropPolicy.LOSSY),
    "signal": TopicConfig(name="signal", maxsize=1_000, policy=DropPolicy.BLOCKING),
    "order": TopicConfig(name="order", maxsize=1_000, policy=DropPolicy.BLOCKING),
    "fill": TopicConfig(name="fill", maxsize=1_000, policy=DropPolicy.BLOCKING),
    "regime": TopicConfig(name="regime", maxsize=100, policy=DropPolicy.BLOCKING),
    "alert": TopicConfig(name="alert", maxsize=500, policy=DropPolicy.BLOCKING),
}


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Common fields shared by every event flowing through the bus.

    Attributes:
        event_id: Globally unique identifier for deduplication.
        timestamp: Wall-clock time (seconds since epoch) when the event was
            created.
        topic: Routing key used by EventBus.  Must be overridden by each
            concrete subclass as a ``ClassVar``.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)

    # Subclasses must declare:  topic: ClassVar[str] = "<name>"
    topic: ClassVar[str]

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Market-data events  (lossy topics — high-frequency)
# ---------------------------------------------------------------------------


class CandleEvent(BaseEvent):
    """OHLCV candle produced by the market-data feed.

    Attributes:
        pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
        timeframe: Candle resolution string, e.g. ``"5m"``.
        timestamp_ms: Candle open time in milliseconds (UTC).
        open: Opening price.
        high: Highest price during the candle.
        low: Lowest price during the candle.
        close: Closing price.
        volume: Base-asset volume traded during the candle.
        is_closed: ``True`` when the candle is finalised (no further updates).
    """

    topic: ClassVar[str] = "candle"

    pair: str
    timeframe: str
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


# ---------------------------------------------------------------------------
# Control-plane events  (blocking topics — low-frequency, must not be lost)
# ---------------------------------------------------------------------------


class SignalEvent(BaseEvent):
    """Trading signal produced by the signal engine.

    Attributes:
        pair: Trading pair symbol.
        direction: ``"long"``, ``"short"``, or ``"flat"``.
        strength: Normalised signal strength in [0.0, 1.0].
        indicator_name: Name of the indicator that generated the signal.
        signal_timestamp: Exchange time (ms) the signal was computed for.
        metadata: Arbitrary key-value context (indicator values, etc.).
    """

    topic: ClassVar[str] = "signal"

    pair: str
    direction: str
    strength: float
    indicator_name: str
    signal_timestamp: int
    metadata: dict = Field(default_factory=dict)


class OrderEvent(BaseEvent):
    """Lifecycle event for a single order.

    Attributes:
        client_order_id: Idempotency key generated by the risk gate.
        pair: Trading pair symbol.
        side: ``"buy"`` or ``"sell"``.
        quantity: Order quantity in base-asset units.
        order_type: ``"market"``, ``"limit"``, etc.
        status: Order lifecycle state (``"pending"``, ``"open"``,
            ``"filled"``, ``"cancelled"``).
    """

    topic: ClassVar[str] = "order"

    client_order_id: str
    pair: str
    side: str
    quantity: float
    order_type: str
    status: str


class FillEvent(BaseEvent):
    """Execution report produced when an order is (partially) filled.

    Attributes:
        client_order_id: Links back to the originating OrderEvent.
        pair: Trading pair symbol.
        side: ``"buy"`` or ``"sell"``.
        filled_price: Average fill price.
        filled_quantity: Quantity matched in this fill.
        fees: Fees charged for this fill (in quote-asset units).
    """

    topic: ClassVar[str] = "fill"

    client_order_id: str
    pair: str
    side: str
    filled_price: float
    filled_quantity: float
    fees: float


class RegimeEvent(BaseEvent):
    """Market-regime classification produced by the Claude integration.

    Attributes:
        regime: Regime label, e.g. ``"trending"``, ``"ranging"``,
            ``"volatile"``, or ``"unknown"``.
        confidence: Model confidence in [0.0, 1.0].
        active_pairs: Pairs approved for trading under this regime.
        active_strategies: Strategy names activated for this regime.
        reasoning: Human-readable explanation from the model.
    """

    topic: ClassVar[str] = "regime"

    regime: str
    confidence: float
    active_pairs: list[str]
    active_strategies: list[str]
    reasoning: str
