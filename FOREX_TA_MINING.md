# Forex Technical Analysis — Mining Report

Mined 2026-10-10. Inside: all of `nomorals/ta/` (12 files, ~6300 lines),
`nomorals/finance/trading_desk.py`, `nomorals/connectors/exness.py`.
Outside: ICT/SMC literature, Kelly/risk math, session volatility research,
S/R detection algorithms.

## What already exists (keep)

- **indicators.py** — RSI, MACD, Bollinger, ATR, ADX, Ichimoku, SAR, VWAP, etc. Textbook math, no lookahead.
- **patterns.py** — 30+ candlestick patterns + trend context + confluence features.
- **strategies.py** — 9 canonical strategies with signal/confidence/gate contract.
- **signals.py** — vote fusion, disagreement filter, cost-aware thresholds.
- **risk.py** — RiskManager with fixed-fractional, quarter-Kelly, vol-target, CPPI, daily halt, max-DD governor. Solid.
- **backtest.py** — event + vector backtesters, walk-forward, Monte Carlo, risk-of-ruin.
- **regime.py** — trend/range/panic detection, playbooks.
- **pipeline.py** — `analyze()`, `trade_plan()`, `analyze_mtf()` (Elder's Triple Screen).
- **ExnessConnector** — full order/position API. Execution exists.

## Gaps (built this pass)

### 1. Market structure — no swing/BOS/CHoCH module
Every price-action framework starts with structure. Missing: fractal swing
detection, HH/HL/LH/LL labeling, BOS/CHoCH events, S/R zones with strength.
Mined algorithm (convergent across TradingView S/R indicators):
pivots confirmed N bars each side → ATR-based clustering → strength =
recency-weighted touches + volume + trend alignment. Zone states:
fresh / inside / broken / flipped.

### 2. SMC/ICT — the Rosetta Stone
Source: algostorm.com ICT/SMC key-concepts crosswalk; forextradelab.com 2026
guide; harshatangirala/smc-ict-research (prospective-detection methodology).

| SMC term | Classic equivalent |
|---|---|
| Order Block | Supply/demand base, Wyckoff accumulation, Rally-Base-Rally |
| Fair Value Gap | Price gap / Market Profile LVN — imbalance, NOT guaranteed fill |
| BOS | Dow Theory HH/HL trend confirmation |
| CHoCH | Dow Theory first failure — alert, not entry trigger |
| Liquidity sweep | Stop hunt / failed breakout with reclaim |
| Premium/Discount | Above/below mean, 50% of dealing range |
| OTE | Fib 61.8–78.6% retracement — pullback filter, not magic |

Critical methodology (smc-ict-research): detection must be PROSPECTIVE —
an event fires on the bar where all conditions are observable. A "sweep"
confirmed by later reversal cannot be traded. Liquidity sweep := penetration
of most recent confirmed swing level by ≥X×ATR, then close back within Z
bars, stamped on the reclaim bar.

Honest framing in code: SMC renames classical concepts. The value is the
systematic vocabulary, not secret institutional knowledge. Order blocks fail
constantly in isolation; FVGs are revisited often but not always.

### 3. Sessions/killzones — no time-of-day module
Convergent across sources (quantvps, analyticsinsight, scribehow):
- London–NY overlap 13:00–17:00 UTC: extreme volatility, best XAUUSD window
- London open 07:00–09:00 UTC: clean structure, breakout momentum
- NY open 13:30–15:00 UTC: US data reactions
- Asian 22:00–06:00 UTC: low vol, chop, fake breakouts
- Late NY post-17:00 UTC: danger zone
ICT killzones (ET): Asian 20:00–00:00, London 02:00–05:00, NY 07:00–10:00.
News events (NFP, CPI, FOMC) override session logic — volatility without direction.

### 4. Risk math — verified, minor gaps
Kelly f* = (p·b − q)/b (Kelly 1956, Bell Labs). Production rule: half or
quarter Kelly — full Kelly is psychologically brutal (10-loss streak ≈ −90%).
Fixed fractional 0.25–1%; above 1% → ruin risk in long tails. Risk of ruin
≈ ((1−edge)/(1+edge))^(1/r). Cornish-Fisher VaR for FX (Gaussian
underestimates 30–50%). Existing RiskManager already implements the core;
added: half-Kelly helper with estimation-error shrinkage, RoR calculator.

### 5. Analyst/executor separation — no explicit contract
`trade_plan()` produces a plan dict, ExnessConnector executes, but nothing
enforces the boundary. Built `analyst.py`: Analyst fuses structure + SMC +
indicators + regime + MTF + sessions into a scored TradeIdea (bias,
confidence, invalidation, targets, session gate). The analyst NEVER places
orders; the executor NEVER does analysis. Contract enforced by module
boundaries + docstring law.

## What profitable retail traders actually do (synthesis)
No single edge dominates. Convergent traits: risk ≤1% per trade, fewer
trades in high-volatility windows only, systematic rules journaled and
backtested, asymmetric payoffs (win rate 30–45% with R:R ≥ 2), drawdown
governors that actually halt trading. Nothing here guarantees profit —
backtest everything, forward-test on demo, then risk small.
