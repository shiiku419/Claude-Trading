"""Data package: market-data ingestion, aggregation, and feature storage."""

from data.candle_aggregator import CandleAggregator
from data.feature_store import FeatureStore
from data.feeds.binance_ws import BinanceKlineFeed

__all__ = [
    "CandleAggregator",
    "FeatureStore",
    "BinanceKlineFeed",
]
