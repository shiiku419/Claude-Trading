"""Monitoring and alerting package for the trading system.

Public surface:

* :class:`~monitoring.healthcheck.HealthChecker` — periodic component health checks.
* :class:`~monitoring.healthcheck.HealthStatus` — result of a single health check.
* :class:`~monitoring.alert.AlertDispatcher` — Slack / Telegram alert dispatcher.
* :class:`~monitoring.alert.AlertLevel` — alert severity enum.
* :class:`~monitoring.slo.SLOTracker` — real-time SLO metric tracker.
* :class:`~monitoring.slo.SLOStatus` — point-in-time SLO status snapshot.
* :class:`~monitoring.slo.SLODefinition` — static SLO target definition.
"""

from monitoring.alert import AlertDispatcher, AlertLevel
from monitoring.healthcheck import HealthChecker, HealthStatus
from monitoring.slo import SLODefinition, SLOStatus, SLOTracker

__all__ = [
    "AlertDispatcher",
    "AlertLevel",
    "HealthChecker",
    "HealthStatus",
    "SLODefinition",
    "SLOStatus",
    "SLOTracker",
]
