# Trading System

Claude-powered crypto auto-trading system. Claude analyzes market regime; a deterministic Python bot generates signals, enforces risk limits, and executes orders.

## Architecture

```
Binance WebSocket ──> Event Bus ──> Feature Store (numpy ring buffers + Redis)
                                         |
Claude API ──> Regime Runner ──> Strategy Controller (12h TTL, Redis-persisted)
  (every 6h)                             |
                                   Signal Engine (momentum, VWAP, volume spike)
                                         |
                                    Risk Gate (kill switch, daily loss, exposure)
                                         |  produces ApprovedOrder (HMAC-signed)
                                         v
                                  Execution Engine (verifies HMAC, idempotent)
                                         |
                                   Paper / Live Executor
                                         |
                                  Ledger + PnL + TCA
```

**Key invariant:** Claude's output never directly triggers orders. Data flows through Strategy Controller, Signal Engine, Risk Gate (produces HMAC-signed `ApprovedOrder`), and Execution Engine (verifies HMAC). This is enforced both statically (contract tests) and at runtime.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Redis 7+ (for state, kill switch, idempotency)
- Docker & Docker Compose (optional, for full stack)

## Quick Start (Local)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter project
cd trading-system

# 3. Install dependencies
uv sync --extra dev

# 4. Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine redis-server --appendonly yes

# 5. Set environment variables
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   EXCHANGE_API_KEY=...        (Binance testnet key)
#   EXCHANGE_API_SECRET=...     (Binance testnet secret)

# 6. Run in paper trading mode
uv run python main.py config/paper.yaml
```

## Quick Start (Docker)

```bash
# 1. Set environment variables
cp .env.example .env
# Edit .env with your keys

# 2. Start full stack (bot + Postgres + Redis)
cd docker
docker compose up -d

# 3. View logs
docker compose logs -f bot
```

## Configuration

Configuration is loaded from YAML files with environment variable overrides. Use `__` as the nested delimiter (e.g., `EXCHANGE__API_KEY=abc`).

### `config/paper.yaml` (default)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `exchange.testnet` | `true` | Use Binance testnet |
| `trading.pairs` | `[BTC/USDT, ETH/USDT, SOL/USDT]` | Traded pairs |
| `trading.timeframes` | `[5m, 15m]` | Candle intervals |
| `trading.mode` | `paper` | `paper` or `live` |
| `risk.max_position_pct` | `0.02` | Max 2% per trade |
| `risk.daily_loss_critical_pct` | `0.05` | 5% daily loss triggers kill switch |
| `risk.max_open_positions` | `3` | Max concurrent positions |
| `claude.model` | `claude-sonnet-4-5-20250929` | Claude model for regime detection |
| `claude.regime_ttl_hours` | `12.0` | Regime expires after 12h (P0-1) |
| `paper.initial_balance` | `10000.0` | Starting paper balance (USD) |
| `paper.fill_model` | `next_bar_open` | Fill at next bar's open price (P1-2) |
| `database.url` | `sqlite+aiosqlite:///trading.db` | SQLite for local dev |
| `database.redis_url` | `redis://localhost:6379/0` | Redis connection |

### `config/prod.yaml`

Tighter risk limits: 1% per trade, 3% daily loss, 2 max positions. Uses PostgreSQL and faster alert rate limiting.

## Project Structure

```
trading-system/
  main.py              # asyncio entry point, wires all components
  bus/                  # Event bus (lossy for market data, blocking for control-plane)
  data/                 # Binance WS feed, candle aggregator, feature store
  claude/               # Regime detection via Anthropic SDK
  strategy/             # Mode definitions, regime-to-strategy mapping
  signals/              # Momentum, VWAP, volume spike, composite signal
  risk/                 # Kill switch, daily loss, exposure, position limits
  execution/            # Idempotent order engine, paper executor, recovery worker
  ledger/               # SQLAlchemy models, trade logger, PnL tracker, TCA
  monitoring/           # Health checks, alerts (Slack/Telegram), SLO tracking
  replay/               # Backtesting replayer, evaluator, canary comparison
  config/               # Pydantic Settings + YAML configs
  docker/               # Dockerfile + docker-compose.yml
  tests/                # 328 tests
```

## Safety Invariants (P0)

These must be satisfied before paper trading:

| ID | Invariant | How Enforced |
|----|-----------|--------------|
| P0-1 | **Regime TTL** | Regime expires after 12h. Expired regime forces `unknown` mode (no trading). |
| P0-2 | **Single path to execution** | `ApprovedOrder` HMAC token required by Execution Engine. Only Risk Gate produces it. Contract tests verify no bypass. |
| P0-3 | **Crash-safe idempotency** | 3-state flow: `PENDING_SUBMIT` -> `SUBMITTED` -> `FILLED`. Redis + DB unique constraint. Recovery worker on startup. |
| P0-4 | **Bus backpressure** | Market data queues are lossy (drop oldest). Control-plane queues (`order`, `fill`, `regime`) are blocking (never drop). |

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy .
```

## Kill Switch

The kill switch is a Redis-persisted emergency halt. When active, all new orders are rejected.

```bash
# Activate manually (via Redis CLI)
redis-cli SET trading:kill_switch "true"
redis-cli SET trading:kill_switch:reason "manual halt"

# Deactivate
redis-cli DEL trading:kill_switch trading:kill_switch:reason trading:kill_switch:activated_at
```

The kill switch activates automatically when daily loss exceeds the critical threshold for 3 consecutive evaluations.

## Replay / Backtesting

```python
from replay import Replayer, BacktestEvaluator
from data import FeatureStore
from signals.momentum import MomentumSignal

store = FeatureStore(redis_url="redis://localhost:6379/0")
replayer = Replayer(feature_store=store, signal_generators=[MomentumSignal()])
signals = await replayer.replay(candles, pair="BTC/USDT", timeframe="5m")
```

## Alerts

Configure Slack and/or Telegram in `.env`:

```bash
ALERTS__SLACK_WEBHOOK=https://hooks.slack.com/services/T.../B.../...
ALERTS__TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
ALERTS__TELEGRAM_CHAT_ID=987654321
```

Alert levels:
- **INFO**: Regime change, daily summary
- **WARNING**: WS reconnection, degraded performance
- **CRITICAL**: Kill switch activated, daily loss limit, connection lost >5 min

## Gradual Scaling Plan (Post Paper Trading Validation)

1. **Week 1**: $100, BTC/USDT only, 1 position max
2. **Week 2**: $250, add ETH/USDT
3. **Week 3**: $500, add SOL/USDT
4. **Week 4**: Review metrics, decide on scaling
