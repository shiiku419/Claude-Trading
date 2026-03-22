# Trading System

## Architecture
Claude is the "brain" (regime detection), deterministic Python bot is the "hands" (execution).
Claude output NEVER directly triggers orders. Flow: Strategy Controller -> Signal Engine -> Risk Gate (produces ApprovedOrder) -> Execution (verifies ApprovedOrder).

## Key Invariants
- P0-1: Regime TTL (12h default). Expired regime -> unknown mode -> no trading.
- P0-2: ApprovedOrder HMAC token required by Execution. Only Risk Gate produces it.
- P0-3: Dual-layer idempotency (Redis + DB unique constraint). Crash-safe 3-state flow.
- P0-4: Event bus: lossy for market data, blocking for control-plane.

## Stack
- Python 3.12, uv, asyncio
- Binance (testnet first)
- Pydantic Settings v2 for config
- SQLAlchemy async + aiosqlite (dev) / asyncpg (prod)
- Redis for state + feature store
- structlog for structured logging

## Commands
- `uv run python main.py config/paper.yaml` — start bot
- `uv run pytest` — run tests
- `uv run ruff check .` — lint
- `uv run mypy .` — type check
