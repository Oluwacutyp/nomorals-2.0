# Trading Integration — Mining Report

Mined 2026-10-10. Inside: `nomorals/connectors/exness.py` (1104 lines),
`nomorals/finance/trading_desk.py` (826 lines), `nomorals/tools/trading.py`
(666 lines). Outside: MT5 bridge patterns, Exness Public Trader API docs,
real algo-trader risk frameworks.

## What's already real (keep, don't fork)

### Exness REST connector (`nomorals/connectors/exness.py`)
- **The Exness Public Trader API is REAL** — verified directly at
  exness-api.com/documentation/create-api-key. Ed25519-signed requests,
  per-account trading hosts, idempotency keys.
- `open_position` / `close_position` / `close_all_positions` /
  `place_pending_order` / `modify_order` / `cancel_order` — all
  confirmation-gated (human checkpoint), demo-first.
- Native Ed25519 (RFC 8032), ~7ms per signature, verified against
  independent `cryptography` implementation.
- Clock skew handling (asymmetric 3s tolerance, 1s past bias).

### Trading desk (`nomorals/finance/trading_desk.py`)
- `RiskPolicy`: 1% max risk/trade, 3 max positions, 3% daily loss kill,
  SL required, live unlock gate.
- `size_position`: fixed-fractional sizing, floors to broker step
  (never rounds UP into extra risk), raises on invalid sizing.
- Paper trading: `paper_open` / `paper_close` / `paper_mark` with
  R-multiple tracking.
- `daily_pnl`, `equity_curve`, `max_drawdown`, `kelly_fraction`, `stats`.
- Live mode: `unlock_live()` gate, `live_open` / `live_close`.

## External findings

### MT5 Python package — Windows-only (confirmed)
The official `MetaTrader5` package requires Windows. Linux options:
1. **ZeroMQ bridge** (recommended): MQL5 EA on Windows + pyzmq on Linux.
   Used by Titan, Darwinex-based bridges.
2. **FastAPI bridge microservice**: Windows box runs MT5 + FastAPI,
   Linux calls it over HTTPS. (beaststudi0/mt5-guardrail pattern)
3. **mt5linux**: rpyc bridge to Windows machine.
4. **Wine**: MT5 under Wine on Linux (fragile, updates break it).

**Verdict for AWS Linux:** The Exness REST API is the right bridge.
No Windows needed, no bridge to maintain, native HTTPS. MT5 route
only makes sense if the REST API lacks a needed capability.

### What real algo traders do (mined from omni-full-algo-trading-bot, omega-devin, wolfiesch/evolutionary-algo-trading)

**Position sizing:**
- Base 1% risk, scaled by confidence (0.5% low → 1.5% high)
- `lot = (equity × risk%) / (sl_distance / tick_size × tick_value)`
- Adaptive: win streak ≥3 → +0.25%; loss streak ≥3 → −0.25% (floor 0.5%)

**Kill switches:**
- Daily loss > 3% → stop today
- Weekly loss > 7% → stop week
- Max drawdown > 10-15% → stop and review
- 5 consecutive losses → halve size
- Kill switch FILE (`touch HALT`) → immediate stop

**Trade management:**
- SL: 2x ATR from entry, at logical structure, never moved further
- TP: scale out at 1R/2R/3R, trail after 1R
- Min R:R 2:1 — setups below discarded

### Gaps to build
1. Adaptive risk scaling (streak-based) — not in RiskPolicy
2. Kill switch file — not implemented
3. Weekly loss limit — only daily exists
4. Consecutive-loss size reduction — not implemented
5. Session filters (London/NY kill zones) — not implemented
