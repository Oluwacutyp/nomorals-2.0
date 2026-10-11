# Quant Strategy Mining: Verified Backtests from Open Source & Pro Communities
*Researched 2026-10-10. Focus: Forex (XAUUSD, majors) and crypto. Skeptical pass applied.*

## Headline takeaways
1. **Honest edges live at PF 1.0–1.4, not 2.0+.** The best independently-verified retail strategies (PMax on gold H1, London breakout on FX) land at profit factors of 1.2–1.5 *with real costs*. Anything far above that without out-of-sample is overfit or a backtester artifact.
2. **Costs are the #1 strategy killer, by an order of magnitude.** London breakout goes PF 1.48 → 0.77 when broker spread goes 0 → 1 pip. FX M15 systems die under commission+spread; gold survives because moves are larger relative to costs.
3. **Delta-neutral carry (funding arb) has the best risk-adjusted in-sample evidence, but it's idle/crowded right now** — multiple independent researchers found trigger conditions rare in 2026's quiet funding regime, and naive cross-exchange variants fail outright at retail costs.
4. **Two instructive FAILED backtests** (G10 carry, triangular arb) are as valuable as the winners: they show exactly where structural logic breaks at retail execution.
5. **Cross-asset correlations are confirmation filters, not entry signals.** DXY↔gold (-0.70) and BTC↔SPX (regime-dependent, negative skew) both have real literature; neither has a verified standalone backtested strategy attached.

---

## Strategy 1 — Delta-neutral funding-rate carry (crypto)
**Core logic:** Long spot + short equal-notional perp → price risk cancels by construction → income is pure funding (longs pay shorts). Cross-exchange variant: short the perp with the richest funding on one venue, long the cheapest on another, pocket the spread. The edge is structural: funding is a cash flow transfer between longs and shorts, not a prediction.

**Evidence it works:**
- nvolpe2007-cell/crypto-bot research synthesis: realistic 12–25% annualized, Sharpe 3–6, max DD typically <5%. Funding averages ~0.05%/8h (~22%/yr) in bull markets. Source: https://github.com/nvolpe2007-cell/crypto-bot/blob/HEAD/RESEARCH_strategies_and_filters.md
- card0re/trading-bot `cmd/carry` (Go, Binance USDT-M): in-sample 2023–2025 (~2.5y), +15–22% consistent gains on all 7 symbols simultaneously, drawdowns 0.15–2.7% — "an order of magnitude calmer than any directional strategy tried." Honest caveat noted: walk-forward on the last 180 days scored near zero — but that's *idle, not losing*: funding too thin to clear the ~0.15% round-trip fee. Source: https://github.com/card0re/trading-bot
- drandromeda/trading-system-io "Crypto Carry": 2020–2026 backtest, Carry BTC +436% total / CAGR +28.4% / Sharpe 11.85. ⚠️ **Skeptical flag:** DD reported as 0.00% which the author admits is an artifact (real DD est. -3..-8%), Sharpe 11.85 is unrealistic for any live carry (basis risk unmodeled), and the backtester uses a stub order function. In-sample only, no OOS. Treat as illustrative, not verified. Source: https://github.com/drandromeda/trading-system-io-

**What market/regime:** Bull/volatile crypto markets with high perp speculation (funding persistently positive). Same-venue spot/perp carry works when annualized funding > round-trip cost hurdle.

**What kills it:**
- Crowding: "the easiest trade retail can do → crowded. Of observations with a ≥20bp arbitrage spread, only ~40% were profitable after transaction costs." Pros moved to cross-exchange; retail cross-exchange variants FAILED honest backtests (see Strategy 3).
- Fee hurdle: taker fees ~17bps round-trip eat the median cross-venue spread.
- Quiet regimes: funding collapses in sideways markets → strategy sits idle (carry earns nothing, which is fine) — but only if you correctly distinguish "idle" from "broken."
- Basis risk: spot/perp divergence during crashes can open real mark-to-market DD despite delta neutrality.

---

## Strategy 2 — OFI-skewed market making (crypto)
**Core logic:** Quote both sides of the book, earn the spread + maker rebates. The alpha that keeps it from losing is **order-flow imbalance (OFI)** — skewing quotes based on short-horizon directional prediction. Edge: microstructure, not price prediction.

**Evidence it works:**
- 2026 academic study across 5 coins (cited in crypto-bot research): OFI is the single strongest predictor of short-horizon (3-second) mid-price moves; spread and VWAP-to-mid deviation rank next. Source: https://github.com/nvolpe2007-cell/crypto-bot/blob/HEAD/RESEARCH_strategies_and_filters.md
- Real backtest via hftbacktest: OFI-skewed quoting achieved Sharpe 5–10 on BTC — **but** return-per-trade decayed from 0.0139% (2023) to 0.0086% (2025). "Maker rebates and fee tier are the difference between profit and loss."
- Hummingbot (the reference open-source MM framework) is honest about backtest limits: queue priority unknown, path dependency means one different fill diverges everything, needs 1-second candles (30+ fills/min typical). Their own docs frame backtesting as parameter research, not a prediction engine. Sources: https://github.com/hummingbot/condor-docs/blob/HEAD/bots/backtesting.mdx, https://hummingbot.substack.com/p/the-bot-pod-episode-4-backtesting

**What market/regime:** Liquid majors (BTC/ETH) in normal, mean-reverting microstructure regimes with tight spreads and stable fee tiers.

**What kills it:**
- Fee tier: the entire edge is inside the fee/rebate structure. Top-tier fees flip it from profitable to losing.
- Crowding + adverse selection: "OFI-based MMs got wiped out in the Oct 10 2025 flash crash (adverse selection). When everyone runs the same imbalance signal it creates self-reinforcing feedback loops." Spread-widening + inventory limits in volatility are mandatory.
- Decay: per-trade return fell ~38% in 2 years. This is the classic alpha-decay curve — documentable edge, shrinking capacity.

---

## Strategy 3 — Triangular arbitrage (crypto) — an instructive FAILURE
**Core logic:** Exploit cross-rate differentials (A→B→C→A) where the loop product ≠ 1. Mathematically real: chrisaacson69 found 264/336 triangular paths showing positive profit on ECB rates, best full-loop 0.071% on $10K.

**Evidence it does NOT work for retail:**
- Harjus (ValtteriL/harjus, MIT, real Binance spot triangular arb bot, full journey documented): "Ultimately, I was not able to make the bot profitable." Source: https://github.com/valtteril/valtteril.github.io/blob/HEAD/content/posts/binance-triangular-arbitrage.md
- zabor-crypto/zabor funding_arb_research (honest PASS/REJECT verdicts, no-look-ahead engine): cross-DEX dispersion at 0.2 APR threshold: 2,423 trades at **-81% net APR**; most selective variant: 45 trades at -1% net APR (closest to breakeven, insufficient data). "The pitch falls apart when you run the numbers: taker fees (~17 bps round-trip) eat the median cross-venue spread, and the rare large spikes mean-revert before you can exit at maker rates." Source: https://github.com/zabor-crypto/zabor/blob/HEAD/scripts/funding_arb_research/README.md
- pattybepatient/crypto-funding-arb (OKX × Hyperliquid cross-exchange): the spread is statistically mean-reverting (ADF p < 0.001) — but the honest backtest lost **-66.6% over 3 months**. "The gap between 'statistically significant pattern' and 'executable strategy' is the entire story." Source: https://github.com/pattybepatient/crypto-funding-arb

**What market/regime:** N/A — infrastructure game, not a market game.

**What kills it:** Latency and competition. "The math works. The question is infrastructure." On-venue triangular arb is arbitraged away by co-located HFT; cross-exchange arb dies on fees + transfer friction + inventory risk (tokens dropping 10–42% wipes arb profit). **Lesson for Devon's build: statistical significance (ADF p<0.001) is not the same as executability.**

---

## Strategy 4 — London open Asian-range breakout (FX)
**Core logic:** Build the Asian-session range (the pre-London lull box); trade the breakout at London open (08:00 BST) when the day's order flow arrives and stops above/below the range get hunted. Edge: liquidity-driven — "smart money needs to trigger stops to fill large orders"; the one-sided volume burst is a real market-structure event, not a chart pattern.

**Evidence it works:**
- forex_python_auto_trade London breakout engine (tick-replay backtester, MT5 tick history, half-spread charged at entry): EURUSD PF 1.48 / GBPUSD PF 1.27 at demo-feed spreads (0–1 point). **But:** at ~0.5 pip spread PF drops to 1.38/1.02, and at ~1 pip broker spread both go NEGATIVE (-$47.79 PF 0.77 / -$35.59 PF 0.82). H1 EMA200 trend gate "raised the minimum profit factor in every London-aligned window." Six-month result: ~$90, 0.09R per trade. Source: https://github.com/nadeemabualarayes/forex_python_auto_trade/blob/HEAD/docs/superpowers/specs/2026-09-06-london-breakout-engine-design.md
- Complementary rigorous data: ~800 backtests on TradingView's PMax system (Exness tick archives, real commission+swap): on FX M15, costs turn a near-zero result into a clear loss (GBPUSD M15: -330 → -2,093 pips with costs). This corroborates: FX intraday edges are paper-thin after costs. Source: http://dev.to/moonthetrain/tradingview-pmax-strategy-best-settings-repaint-backtest-2f4

**What market/regime:** Trending/day-range-expansion days in London session; needs the H1 trend filter (EMA200 gate) to avoid trading breakouts against the daily flow.

**What kills it:** Spread. The whole strategy is a spread play: a 1-pip broker kills a PF-1.48 system. Demo-feed backtests (0–1 point spreads) are fantasy; must re-cost through the actual broker. Fakeouts without the trend filter; fixed 2R targets with box-edge stops won only 26–41% (below the 33% break-even) — ATR stops performed better.

---

## Strategy 5 — PMax trend system on XAUUSD (gold)
**Core logic:** PMax (KAMA-based adaptive trailing stop): trend-following entries on flips, riding gold's large directional moves. Edge: gold's trend moves are large relative to costs, so a mid-grade trend signal survives costs that kill FX M15.

**Evidence it works:**
- moonthetrain's independent ~800-backtest study (Python rewrite of Pine logic, 6.7 years of Exness Raw Spread tick archives for XAUUSD/USDJPY/EURUSD/GBPUSD/NZDUSD, real ask/bid fills + $3.5/lot/side commission + swap; candle-time shifts and 56 parameter sweeps): Gold M15: 1,761 trades, PF 1.23 → **1.21 with real costs** (+$3,219 net). Gold H1: 448 trades, PF 1.38 → **1.37 with real costs** (+$2,638 net). Contrast: USDJPY M15 PF 1.02, GBPUSD M15 PF 0.95, NZDUSD M15 PF 0.91 — all dead under costs. Source: http://dev.to/moonthetrain/tradingview-pmax-strategy-best-settings-repaint-backtest-2f4

**What market/regime:** Gold in trending regimes on M15–H1. Notably robust across 800+ variations (settings sweeps, MA types, candle-time shifts) — the parameter-insensitivity is the real evidence of edge.

**What kills it:** Ranging/choppy gold regimes (whipsawed by the trailing stop); swap costs (gold swaps are large — the study separates them for a reason); applying the same logic to FX M15 where costs dominate; trusting TradingView's built-in backtest (no commission, no spread setting — "gets the signals right and the money too rosy").

**⚠️ Skeptical note:** PF 1.37 on gold H1 is credible precisely because it's modest. This is the single best-documented gold retail strategy found: 6.7 years, tick data, real costs, parameter-sweep robustness. No out-of-sample split is reported — that's the gap to close before trusting it.

---

## Strategy 6 — G10 FX carry (long high-yield, short low-yield) — a FAILED retail implementation of a REAL edge
**Core logic:** Borrow low-yield currencies, invest in high-yield ones; collect the rate differential + the trending appreciation carry flows generate. One of the most documented FX premia in the literature (Lustig & Verdelhan 2007: Sharpe 0.6–1.0 over 30+ years; Chen 2024 200-year study: robust except around the World Wars). Source: https://quantpedia.com/fx-carry-value-momentum-strategies-over-their-200-year-history/

**Evidence it does NOT work as a retail backtest:**
- marcohwlam/quant-zero H37 (G10 currency carry via ETFs, dual-signal carry + momentum filter, 2007–2021 in-sample, full Gate-1 validation): Sharpe **-0.23**, max DD -25.66%, permutation test p = 1.0 (statistically indistinguishable from noise), 0/4 walk-forward windows pass, best parameter sensitivity Sharpe 0.19. Verdict: **FAIL, formally retired.** Source: https://github.com/marcohwlam/quant-zero/blob/HEAD/backtests/h37_g10_carry_gate1_report.md

**What market/regime:** Works in calm risk-on regimes with wide, stable rate differentials (2003–2007 yen carry). Long-horizon academic evidence is real; 200-year robustness is real.

**What kills it:** Carry crashes — 2008 GFC, COVID 2020, CHF floor removal 2015 — dominate returns and wipe years of accrual ("picking up pennies in front of a steamroller"); the post-2008 near-zero-rate era compressed differentials to nothing. Brunnermeier et al. (2008): crash risk premia explain a large share of returns; hedging the crash with OTM puts costs 30–40% of the excess return (Jurek 2014, Farhi et al. 2009). Retail FX accounts can't hedge the tail cheaply, so the unhedged retail backtest fails even though the academic premium is real.

---

## Cross-asset correlations: filters, not signals
- **DXY ↔ XAUUSD (~-0.70):** Real, well-documented inverse correlation (World Gold Council notes it varies over time). Professional usage is a *confirmation framework*, not an entry signal: e.g. long gold only when DXY < 20-EMA(H4) AND XAUUSD > 20-EMA(H4), then wait for an H1 pullback + reversal candle. Breaks down: Asian session/holidays (thin liquidity, noisy link); risk-aversion episodes where both rise as safe havens; Fed-hiking periods where dollar-specific strength diverges from gold. Sources: https://forextradelab.com/blog/gold-dollar-correlation-trading/, https://github.com/prathameshh-patil/trading-intelligence/blob/HEAD/docs/strategy/xauusd-regimes.md
- **BTC ↔ S&P 500 (currently ~0.51, 21-day):** No verified backtested *signal* strategy found — and the literature says why: in high-correlation regimes BTC behaves as "a high-beta version of the SPX with a negative skew, falling with equities while rarely participating equally on the way up" (Wintermute via FXStreet, Oct 2026). Correlation rising = *risk-off filter* (reduce exposure), not an entry signal. A falling correlation alone means "less clean synchronicity," not decoupling (Adler/cryptopotato analysis). Sources: https://www.fxstreet.com/cryptocurrencies/news/bitcoin-may-continue-consolidation-as-rising-equities-correlation-pose-downside-risk-202610062049, https://cryptopotato.com/analyst-bitcoins-sp-correlation-is-not-the-bull-sign-it-looks/

---

## Skeptical red flags encountered (not recommended)
- **drandromeda Crypto Carry (Sharpe 11.85, DD 0%):** artifact of an unmodeled backtester (stub orders, no basis risk), in-sample only. Classic "too good" signal.
- **jtscwhitney BollingerRatchet + "Oracle" ML regime filter (84.6% win rate, +15.47% / 2y on BTC/USDT):** no out-of-sample window; 26 separate ML models on 30-day windows = textbook overfit risk. Interesting architecture, unverified. https://github.com/jtscwhitney/freqtrade-trading-bot/blob/HEAD/BACKTEST_RESULTS_BollingerRatchet_2026-01-28.md
- **Monarch FX ("90% win rate," "40–50 pips daily in Asian session"):** marketing copy for a paid system, no methodology or verifiable data. Ignored.
- **apex-prim/strategy-audit corpus:** a genuinely useful resource — independent out-of-sample audits of public Freqtrade strategies, most showing insignificance (e.g. one audited strategy: p=0.74 in-author-window, loses to buy-and-hold OOS). Good mining target for future "what fails" research. https://github.com/apex-prim/strategy-audit

## What this means for the user's Sentinel work
The fresh backtest of 120 Sentinel strategies (only XAUUSD-H1 breakout +14.77% / Sharpe 1.98 on 60 days; mean reversion 0/20; trend 0/20 on XAU; PF 1.06–1.11 across the board) is **exactly what this literature predicts**:
1. PF 1.06–1.11 is the graveyard zone where honest thin edges live and die — the London breakout data shows PF 1.48 → 0.77 purely from spread. If Sentinel's backtests didn't model *actual broker spread + commission + swap*, every PF in that table is overstated.
2. Mean reversion 0/20 is consistent with the literature: intraday FX mean reversion needs tick-level execution and survives only with a trend filter (ADX<25, 200-EMA slope gate) and tight-spread pairs — unfiltered it fails everywhere.
3. The strongest verified gold logic found here (PMax H1, PF 1.37 with real costs) suggests the user's hypothesis "logics are wrong, not parameters" is right: a *trend-with-adaptive-trailing-stop* logic on gold H1 has independent support that their tested set (mean reversion, fixed-breakout) doesn't.
4. Funding-rate carry is the one structurally different logic worth mining — delta-neutral by construction, best risk-adjusted in-sample evidence — but 2026's quiet funding regime means triggers are currently rare; run it as a scanner that waits rather than a strategy that forces trades.
