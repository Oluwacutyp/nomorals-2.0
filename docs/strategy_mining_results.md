# Strategy Logic Mining: Real Edges with Evidence
*Researched 2026-10-10/11. Three parallel research streams: academic papers, professional quant literature, open-source quant projects with verified backtests. Plus direct research.*

**Method:** Only strategies with named, citable sources. Retail guru content excluded. Instructive failures included — they show where logic breaks.

**Key finding for Sentinel:** The 120 auto-generated strategies are parameter variations of the same templates. The logics below are genuinely different ways of seeing the market — different inefficiencies, different mechanisms, different failure modes.

---

## 1. Time-Series Momentum (TSMOM)

**Core logic:** An asset's own past return predicts its future return. If it's been going up, it keeps going up (and vice versa). The mechanism: initial under-reaction to news (slow information diffusion, institutional herding), followed by delayed over-reaction. This is *not* a moving-average crossover — it's a direct bet on return persistence, the most replicated anomaly in finance.

**Source:** Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum," *Journal of Financial Economics*, Vol 104, Issue 2, pp. 228-250. http://docs.lhpedersen.com/TimeSeriesMomentum.pdf

**Evidence:**
- 58 liquid futures across equities, bonds, currencies, commodities
- 12-month lookback is the classic signal; 1-month and 3-month also work
- Diversified TSMOM portfolio: Sharpe ~1.0
- Returns persist 1-12 months, then partially reverse (under-reaction → over-reaction)
- Independent replication (Vietnamese stocks 2014-2026): Sharpe 1.28, momentum t-stat 11.47
- Crypto-specific hypothesis: TSMOM Sharpe 0.6-1.0 (higher vol amplifies both profits and crash risk)

**Market/regime:** Works across ALL asset classes. Strongest in commodities. Needs trending markets; performs as "crisis alpha" (positive in 2022 bear market while equities fell).

**What kills it:**
- Momentum crashes (Daniel & Moskowitz 2016): violent reversals at regime transitions, when crowded positioning unwinds
- Choppy/range-bound markets: whipsawed by false signals
- High turnover on short lookbacks eats returns after costs
- Crowding: return dispersion collapse + volume spikes on leaders signal elevated crash risk 2-4 weeks ahead

**For XAUUSD/BTC:** Directly applicable. Gold and BTC both trend strongly. Use 12-month lookback for position sizing bias, shorter for timing.

**⚠️ Critical caveats from academic replication:**
- Kim, Tse & Wald deconstruction: the headline alpha is heavily dependent on **volatility scaling**. With unscaled equal-weighted returns, monthly alpha drops from 1.27% to 0.41%, and individual-contract TSMOM beat buy-and-hold on only 29 of 55 contracts. Much of the Sharpe comes from leveraging low-vol instruments.
- **Post-2013 decay:** Ranaldo & Somogyi (2021) find weak performance for carry/momentum in FX specifically from 2013-2019. Crowded trend-following risk is real.
- This is a *daily-bar, 1-12 month horizon* finding. It says nothing about 5-minute or intraday gold trading — borrowing its authority for minute-level rules is a category error.

---

## 2. Intraday Momentum (First Half-Hour → Last Half-Hour)

**Core logic:** The market's return in the first 30 minutes predicts the return in the last 30 minutes. Mechanism: late-informed trading (investors who react to morning news only when they reach their desks), index fund/leveraged-ETF rebalancing flows, and gamma-hedging from options market makers. These are *forced* flows, not predictions — they have to trade regardless of price.

**Source:** Gao, Han, Li & Zhou (2018), "Market Intraday Momentum," *Journal of Financial Economics*, Vol 129, Issue 2, pp. 394-414. https://doi.org/10.1016/j.jfineco.2018.05.009
Extended by: Baltussen, Da, Lammers & Martens (2021), "Hedging Demand and Market Intraday Momentum," *JFE* — gamma-hedging mechanism across 60+ futures.

**Evidence:**
- SPY 1993-2013: first half-hour return predicts last half-hour with t-stat > 3.0 (in-sample), R² OOS ~1.4-2.0%
- Timing Sharpe ~1.08 vs 0.29 buy-and-hold
- Stronger on high-volatility days, high-volume days, NBER recessions, macro news days (FOMC, CPI, NFP)
- Confirmed in Chinese commodity futures including metals (strongest with high volume/volatility openings)
- One implementation notes it works "weaker" on gold futures too

**Market/regime:** Intraday only, flat overnight. Best on high-vol, high-volume, news days. One trade per day.

**What kills it:**
- Quiet/low-volume days: signal too weak to overcome costs
- The effect is small per trade (~3.7% annualized before costs long-short) — needs low-cost execution
- Cannot be scaled by large funds without moving the close (good for retail, bad for institutions)
- Post-2018: some decay as more participants aware, but structural flows (ETF rebalancing) persist

**For XAUUSD/BTC:** Testable on gold during London/NY overlap. For BTC (24/7), adapt the logic: does the Asian session predict the NY session? Does the first hour after a major move predict continuation into the next session?

---

## 3. FX Carry Trade (Forward Premium Puzzle)

**Core logic:** The most documented FX inefficiency. Uncovered Interest Rate Parity says high-rate currencies should depreciate by the rate differential — empirically they don't (Fama 1984). So: borrow in low-rate currencies (JPY, CHF), invest in high-rate ones, collect the rate differential as structural yield. You're being paid to provide insurance against currency crashes.

**Source:** Lustig, Roussanov & Verdelhan (2011), "Common Risk Factors in Currency Markets," *Review of Financial Studies*. Also: Brunnermeier, Nagel & Pedersen (2008) on crash risk; Burnside et al. (2011) on rare-event compensation.

**Evidence:**
- G10 carry: Sharpe 0.6-0.9 before costs (long-run)
- HML-Carry factor: long high-rate, short low-rate, monthly rebalance
- 200-year study (Chen 2024): robust except around World Wars
- Daily G10 carry 1976-2016: Sharpe 0.70, mean 3.59%/yr

**Market/regime:** Low-VIX, risk-on environments with wide, stable rate differentials. Monthly rebalancing.

**What kills it:**
- **Crash risk is the strategy:** Returns are negatively skewed. "Carry crashes 3-5x faster than it builds." Aug 2024 JPY unwind: 12% in 3 weeks, wiping months of carry.
- 2008 GFC, COVID 2020, CHF floor removal 2015 — each wiped years of accrual
- Post-2008: rate differentials compressed globally (near-zero era)
- Hedging the tail with OTM puts costs 30-40% of excess return (Jurek 2014) — retail can't hedge cheaply
- **Honest retail backtest FAILED:** G10 carry via ETFs 2007-2021: Sharpe -0.23, max DD -25.66%, statistically indistinguishable from noise (permutation p=1.0)

**For XAUUSD/BTC:** Does NOT map to gold or crypto — there's no interest-rate differential. This is FX-only. Included because it's the benchmark "real edge" to calibrate expectations against.

**⚠️ Additional honest notes:**
- **No gold-specific academic strategy paper exists.** Gold enters the literature as one instrument inside commodity futures baskets, and via safe-haven studies (which describe gold's *hedging properties*, not tradable momentum logic). Gold's safe-haven spikes (crisis bids that reverse fast) actually fight momentum logic on gold specifically.
- Post-2008 rate compression killed the fuel; weak 2013-2019 per Ranaldo & Somogyi.

---

## 4. Crypto Funding-Rate / Basis Capture (Delta-Neutral Carry)

**Core logic:** Perpetual futures anchor to spot via funding payments. When perps trade at a premium (bullish positioning), longs pay shorts every 8 hours. The structural inefficiency: speculative demand for leveraged longs makes funding *persistently positive*. Trade is fully delta-neutral: long spot + short equal-notional perp → price risk cancels → return = pure funding stream. This is not a prediction — it's harvesting a cash flow.

**Sources:**
- 47-month Binance BTCUSDT backtest (Jan 2022-Nov 2025): https://github.com/sohammahale/crypto-arbitrage-project
- Werapuna et al. (2025), "Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX," *ScienceDirect*
- BIS Working Paper 1087 on crypto carry

**Evidence:**
- 47-month real Binance data: **6.97%/yr average (6.57% after costs), 6.47% CAGR**, best year 11.49% (2024), worst 3.21% (2022 bear), max DD <5%
- "20% returns are unsustainable. Focus on realistic 5-10% annual returns."
- Cross-exchange variant (short rich perp, long cheap perp): largest professional strategy, but retail backtests FAILED (see #11)
- DEX venues showed higher Sharpe (less competition) but limited capacity

**Market/regime:** Bullish/sideways crypto with persistent perp premium and leverage demand. In bear markets, funding flips negative (reverse the trade).

**What kills it:**
- **Fees are the silent killer:** At 0.24% round-trip, a 0.01%/8h funding rate needs ~8 days just to break even. Requires maker-heavy execution.
- Funding can flip negative and stay there (bear regimes)
- Basis risk: spot/perp can diverge during crashes → real mark-to-market DD despite "delta neutrality"
- Liquidation risk on the short perp leg during spikes
- Crowding: "easiest trade retail can do → crowded"; only ~40% of ≥20bp arb observations profitable after costs
- 2026 regime: funding is thin/quiet → strategy sits idle (correct behavior, not a failure)

**For BTC:** **This is the single best-documented crypto strategy with multi-year real data.** The ~5-7% post-cost return is the realistic bar. Anyone pitching 20%+ is selling marketing.

---

## 5. Volatility Risk Premium (Short Vol / Variance Harvesting)

**Core logic:** Implied volatility systematically exceeds subsequently realized volatility. Investors overpay for crash insurance because they're more averse to large losses than the actual probability warrants. Selling that insurance collects the premium in calm periods.

**Source:** Bollerslev, Tauchen & Zhou (2009), "Expected Stock Returns and Variance Risk Premia," *Review of Financial Studies*. Also: Bondarenko on put-writing; CBOE PUT index.

**Evidence:**
- S&P 500 IV minus realized vol averaged ~3 percentage points/month (1990-2007)
- Selling ATM straddles (delta-hedged): Sharpe 0.8-1.5 over long horizons
- CBOE PUT index: competitive returns with lower vol/drawdown than S&P 500
- VRP is countercyclical — highest in crises (insurance most expensive when most demanded)
- FX application: variance risk premia predict directional FX returns, especially EM (Macrosynergy/Academic)

**Market/regime:** Calm to normal markets. The premium is harvested steadily; the risk is episodic.

**What kills it:**
- **Tail events:** "Exchanging many small insurance-premium receipts for a small number of highly concentrated left-tail losses." VXN hit 80 in March 2020. Full-Kelly short-vol goes to zero in 2008, 2018, 2020.
- Must size at half-Kelly or quarter-Kelly (practitioner default)
- Volatility of volatility: the strategy itself becomes the risk in regime shifts
- For FX: needs options on currency pairs (available but wider spreads than equity options)

**For XAUUSD/BTC:** Gold has listed options (COMEX). BTC has Deribit options with a volatility surface. Short-vol on BTC during calm periods harvests the substantial crypto VRP, but BTC's tail events are violent. Position sizing is everything.

---

## 6. London Open Asian-Range Breakout (FX Session Structure)

**Core logic:** The Asian session is a low-volume lull that builds a range. At London open (08:00 BST), institutional order flow arrives and hunts stops above/below the Asian range. This is a *liquidity event*, not a chart pattern — "smart money needs to trigger stops to fill large orders." The one-sided volume burst is structural.

**Source:** Multiple independent implementations. Best documented: forex_python_auto_trade London breakout engine (tick-replay, MT5 tick history). https://github.com/nadeemabualarayes/forex_python_auto_trade

**Evidence:**
- EURUSD: PF 1.48 at demo spreads → **PF 0.77 at 1-pip broker spread** (strategy dies on costs)
- GBPUSD: PF 1.27 → PF 0.82 at 1-pip spread
- H1 EMA200 trend filter "raised the minimum profit factor in every London-aligned window"
- 6-month result: ~$90, 0.09R per trade (thin but real at low costs)
- London handles 35-40% of daily FX volume; 60%+ of major pair breakouts occur during London

**Market/regime:** Trending/day-range-expansion days. Needs the trend filter to avoid trading breakouts against daily flow.

**What kills it:**
- **Spread is the entire game.** A 1-pip broker kills a PF-1.48 system. Demo-feed backtests are fantasy.
- Fakeouts without trend filter (65% of EUR/USD breaks go both sides of 30-min opening range)
- Fixed 2R targets with box-edge stops won only 26-41% (below 33% breakeven) — ATR stops better
- Range-bound days: no expansion, just noise

**For XAUUSD:** Gold is heavily traded during London. The Asian range → London breakout logic maps directly. Gold's larger moves relative to spread make it more viable than FX pairs.

---

## 7. PMax Adaptive Trend System on Gold (XAUUSD)

**Core logic:** PMax uses KAMA (Kaufman Adaptive Moving Average) to build an adaptive trailing stop. Entries on trend flips, riding gold's large directional moves. The edge: gold's trend moves are *large relative to transaction costs*, so a mid-grade trend signal survives costs that kill the same logic on FX.

**Source:** moonthetrain's independent study: ~800 backtests, Python rewrite of Pine logic, 6.7 years Exness Raw Spread tick archives, real ask/bid fills + $3.5/lot/side commission + swap, 56 parameter sweeps. http://dev.to/moonthetrain/tradingview-pmax-strategy-best-settings-repaint-backtest-2f4

**Evidence:**
- Gold M15: 1,761 trades, PF 1.23 → **1.21 with real costs** (+$3,219 net)
- Gold H1: 448 trades, PF 1.38 → **1.37 with real costs** (+$2,638 net)
- Contrast: USDJPY M15 PF 1.02, GBPUSD M15 PF 0.95, NZDUSD M15 PF 0.91 — all dead under costs
- Robust across 800+ variations (parameter-insensitive = real edge signal)

**Market/regime:** Gold in trending regimes, M15-H1. The parameter-insensitivity is the key evidence — it's not curve-fit.

**What kills it:**
- Ranging/choppy gold: whipsawed by the trailing stop
- Swap costs (gold swaps are large)
- Applying the same logic to FX M15 (costs dominate)
- TradingView's built-in backtester (no commission/spread = fantasy numbers)

**For XAUUSD:** **This is the single best-documented gold retail strategy found.** PF 1.37 is credible *because* it's modest. No out-of-sample split reported — that's the gap to close before trusting it live.

---

## 8. Order Flow Imbalance / VPIN (Microstructure)

**Core logic:** Trade direction is embedded in order flow before fully reflected in price. When informed/toxic flow dominates one side, prices continue in that direction because liquidity providers can't replenish fast enough. VPIN (Volume-Synchronized Probability of Informed Trading) quantifies this: high VPIN = follow momentum, low VPIN = safe to mean-revert.

**Source:** Easley, López de Prado & O'Hara (2012) on VPIN. Cont–Kukanov–Stoikov OFI framework. 2026 academic study: OFI is the single strongest predictor of 3-second mid-price moves across 5 crypto assets.

**Evidence:**
- Academic foundation is real; used inside professional HFT/market-making systems
- OFI-skewed quoting backtest (hftbacktest): Sharpe 5-10 on BTC — **but** return-per-trade decayed 38% in 2 years (0.0139% → 0.0086%)
- One replication: taker edge 0.09 tick vs 2.48-tick cost → "zero trades" (costs ate it)

**Market/regime:** Intraday, tick-level. Needs real informed flow and high participation.

**What kills it:**
- **Needs tick data.** Bar-resolution VPIN is broken — a crash + rebound inside one minute averages to "balanced."
- Fees/slippage dominate the micro-edge for retail
- Andersen & Bondarenko (2014) disputed VPIN's predictive content beyond volume/volatility
- Crowding: "OFI-based MMs got wiped out in the Oct 10 2025 flash crash" (adverse selection feedback loops)
- Queue-position adverse selection: ~31% of half-spread given back with *no* informed traders present

**For XAUUSD/BTC:** Not a standalone strategy for this setup. Honest use: as a **regime gate** — detect toxic flow to *pause* mean-reversion strategies, not as an entry signal.

---

## 9. Statistical Arbitrage / Pairs Trading (Cointegration)

**Core logic:** Two economically linked assets that historically move together sometimes diverge on transient shocks. The spread is stationary by construction, so extreme z-scores should revert. Entry at |z|>2, exit at convergence.

**Source:** Gatev, Goetzmann & Rouwenhorst (2006), seminal. Do & Faff (2010) on decay.

**Evidence:**
- GGR (1962-2002, US equities): ~11%/yr on top-5 pairs, ~8% on top-20, market-neutral, unexplained by Fama-French
- **BUT:** Profits decayed sharply post-2000. 2025 replication (S&P 500, 1991-2018): Sharpe ≈ 0 to slightly negative after proper controls
- BTC/ETH pairs: structurally correlated, but same decay — spreads thinner and faster now

**Market/regime:** Range-bound, non-trending markets with stable relationships.

**What kills it:**
- Alpha decay: arbitraged to ~zero in US equities since ~2002
- Pair breakdowns: ~18% of divergences are permanent (fundamental regime change)
- Transaction costs + short-sale friction eat the thin spread

**For XAUUSD/BTC:** Limited direct application (need pairs). Could work on BTC/ETH or gold/silver ratios, but expect thin edge. Best as a small sleeve, not a headline strategy.

---

## 10. News Sentiment Crossover (Event-Driven FX)

**Core logic:** Markets underreact to the *tone* of macro/geopolitical news before it becomes consensus. Track short-term vs long-term news sentiment; trade the crossover (like a moving-average crossover, but on sentiment instead of price).

**Source:** RavenPack "Country News Sentiment Factors Predict Forex Prices" (2010-2012, 7 major pairs). Tarlowski (2016) on macro-announcement reactions. MDPI 2026 systematic review of 80 studies.

**Evidence:**
- RavenPack best variant (1-day vs 3-month sentiment): Information Ratio ~2.2 (2010-2012)
- 2026 review: sentiment integration improves Sharpe 20-60%, cuts max DD up to ~49.6% vs price-only — but gains concentrated in high-volatility/crisis regimes
- FX reacts *faster* than stocks to macro news but retreats to prior levels quickly (Andersen et al. 2003)

**Market/regime:** Macro calendar events — NFP, CPI, central bank statements. High-vol regimes.

**What kills it:**
- Data costs: RavenPack feeds are expensive; free alternatives (social sentiment) are noisy/manipulated
- Latency: EURUSD reacts in milliseconds; a phone-based trader is structurally late to news releases
- The 2026 review's own conclusion: sentiment works best as a **risk filter**, not a primary signal
- Most studies don't include realistic transaction costs

**For XAUUSD:** Gold is highly macro-sensitive (Fed policy, inflation, geopolitics). Sentiment as a regime filter (reduce risk when sentiment vol is high, increase when calm) is the honest application.

---

## 11. INSTRUCTIVE FAILURE: Triangular Arbitrage (Crypto)

**Core logic:** Exploit cross-rate differentials (A→B→C→A) where the loop product ≠ 1. Mathematically real.

**Why it fails for retail:**
- Harjus (real Binance triangular arb bot, documented journey): "Ultimately, I was not able to make the bot profitable."
- Cross-DEX funding arb: 2,423 trades at **-81% net APR**; most selective: 45 trades at -1% net APR
- OKX × Hyperliquid: spread is statistically mean-reverting (ADF p<0.001) but backtest lost **-66.6% over 3 months**
- "The gap between 'statistically significant pattern' and 'executable strategy' is the entire story."

**Lesson:** Statistical significance ≠ executability. Latency, fees, and inventory risk kill it. **For Devon's build: always backtest with real costs before trusting a p-value.**

---

## 12. Cross-Asset Correlations as Filters (Not Signals)

**DXY ↔ Gold (~-0.70):** Real inverse correlation (World Gold Council). Professional use is a *confirmation framework*: long gold only when DXY < 20-EMA AND XAUUSD > 20-EMA, then wait for H1 pullback. Breaks during: Asian session (thin liquidity), risk-aversion episodes (both rise as safe havens), Fed-hiking divergences.

**BTC ↔ S&P 500 (~0.51):** No verified standalone signal strategy exists. In high-correlation regimes, BTC behaves as "high-beta SPX with negative skew" (Wintermute). Rising correlation = **risk-off filter** (reduce exposure), not an entry.

**Lesson:** Correlations are regime indicators and confirmation filters. They don't generate entries on their own.

---

## 13. Crypto Cross-Sectional Momentum + Investor Attention — Liu, Tsyvinski & Wu (2021), *Review of Financial Studies*

**Core logic:** Two effects: (1) **Time-series momentum at daily/weekly frequency** for BTC/ETH/XRP — past 1-4 week returns positively predict next-week returns (+3.3% per one std-dev rise). (2) **Investor attention** (abnormal Google/Twitter mentions) predicts high future returns over 1-2 weeks. Mechanism: crypto has no fundamentals anchor, so attention and herding dominate — a demand-pressure / slow-diffusion story.

**Source:** https://news.yale.edu/2018/08/06/assessing-cryptocurrency-yale-economist-aleh-tsyvinski (Yale summary of the RFS paper)

**Evidence:**
- Cross-sectional long/short momentum: +2.7-4.1%/week (2014-2020, 1827 coins)
- Concentrated in **larger, liquid coins** (+4.2%/week); small coins insignificant or negative
- BTC time-series rule ("buy if +20% last week") reported with very high Sharpe in-sample

**Market/regime:** Attention-driven, retail-dominated regimes. Works when narrative cycles dominate.

**What kills it:**
- **Contested:** Shen et al. (2020) report *negative* payoffs; Zaremba et al. (2021) find results swing on universe construction and sample period
- Liquidity dependency: long/short spread partly unharvestable (can't short small coins in practice)
- Turnover costs brutal on weekly rebalancing; headline Sharpe ignores realistic slippage on short leg
- Breaks in sharp de-risking (narratives collapse faster than momentum can exit)

**For BTC:** Testable but treat with skepticism. The attention factor (Google Trends as demand proxy) is the more novel leg — it's not a price derivative.

---

## Summary: What This Means for the Trading System

| # | Strategy | Type | Honest Edge | Feasible? |
|---|---|---|---|---|
| 1 | Time-Series Momentum | Trend | Sharpe ~1.0 diversified | ✅ |
| 2 | Intraday Momentum | Intraday | Small but real, needs low costs | ✅ (test) |
| 3 | FX Carry | Risk premium | Sharpe 0.5-0.8, crash tails | ✅ (FX only) |
| 4 | Crypto Funding Capture | Structural yield | ~5-7%/yr post-cost | ✅ (best for BTC) |
| 5 | Volatility Risk Premium | Risk premium | Sharpe 0.8-1.5, tail risk | ⚠️ (needs options) |
| 6 | London Breakout | Session structure | PF 1.48→0.77 (spread kills) | ✅ (gold > FX) |
| 7 | PMax on Gold | Adaptive trend | PF 1.37 with real costs | ✅ (best gold evidence) |
| 8 | Order Flow / VPIN | Microstructure | Real but needs tick data | ❌ (as signal) / ✅ (as filter) |
| 9 | Pairs Trading | Cross-sectional MR | ~0 now (decayed) | ⚠️ |
| 10 | News Sentiment | Event-driven | Real, latency-gated | ⚠️ (as filter) |
| 11 | Triangular Arb | — | FAILED (instructive) | ❌ |
| 12 | Cross-asset filters | — | Filters, not signals | ✅ (as overlay) |
| 13 | Crypto Momentum + Attention | Momentum | +3%/week but contested | ⚠️ (skeptical) |

**Critical academic caveats (from replication literature):**
- TSMOM's headline alpha depends heavily on volatility scaling (Kim, Tse & Wald) — unscaled, most individual contracts don't beat buy-and-hold
- FX momentum/carry show documented post-2013 decay (Ranaldo & Somogyi 2021)
- **No gold-specific academic strategy paper exists** — gold appears only inside commodity baskets; its safe-haven spikes actually fight momentum logic
- Crypto momentum is contested (Shen et al. 2020 negative; Zaremba et al. 2021 inconclusive)
- The strongest academic evidence is cross-sectional/diversified, not single-asset — single-instrument 1H technical rules have little academic support

**The user's hypothesis is confirmed:** The 120 Sentinel strategies are parameter variations, not different logics. The table above contains 10 genuinely different market inefficiencies. Each has a different mechanism, different evidence, and different failure modes.

**Recommended starting set for XAUUSD + BTC:**
1. **PMax adaptive trend on gold H1** (best documented gold edge)
2. **Time-series momentum** (12-month lookback for bias, shorter for timing)
3. **Crypto funding-rate capture** (delta-neutral, best BTC evidence)
4. **London session breakout** (with trend filter, on gold not FX)
5. **Volatility regime filter** (using VRP/sentiment as overlay, not entry)

Each needs independent backtesting with real costs before combining. The mixer (fusion.py) can stay — but it should vote across *these* different logics, not 120 variations of the same one.
