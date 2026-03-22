"""Risk gate package.

Public surface
--------------
- :class:`~risk.base.ApprovedOrder` — signed order token produced by :class:`~risk.gate.RiskGate`.
- :func:`~risk.base.verify_gate_token` — HMAC verifier used by the Execution Engine.
- :class:`~risk.base.RiskDecision` — result type returned by each individual check.
- :class:`~risk.base.PortfolioState` — portfolio snapshot consumed by every check.
- :class:`~risk.gate.RiskGate` — main orchestrator; the sole producer of
  :class:`~risk.base.ApprovedOrder`.
- :class:`~risk.kill_switch.KillSwitch` — emergency halt with Redis persistence.
- :class:`~risk.daily_loss.DailyLossCheck` — daily PnL threshold with hysteresis.
- :class:`~risk.exposure.ExposureCheck` — total portfolio exposure cap.
- :class:`~risk.position_limit.PositionLimitCheck` — per-trade size and open-count limits.
- :class:`~risk.time_policy.TimePolicyCheck` — time-of-day trading restrictions.
"""

from risk.base import ApprovedOrder, PortfolioState, RiskDecision, verify_gate_token
from risk.daily_loss import DailyLossCheck
from risk.exposure import ExposureCheck
from risk.gate import RiskGate
from risk.kill_switch import KillSwitch
from risk.position_limit import PositionLimitCheck
from risk.time_policy import TimePolicyCheck

__all__ = [
    "ApprovedOrder",
    "PortfolioState",
    "RiskDecision",
    "verify_gate_token",
    "DailyLossCheck",
    "ExposureCheck",
    "RiskGate",
    "KillSwitch",
    "PositionLimitCheck",
    "TimePolicyCheck",
]
