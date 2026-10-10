# TA Sweep — External Mining Report

Module: `nomorals/ta/` (12 files). Mined 2026-10-10. Every significant class
was compared against the best implementations outside the repo — the good and
the trash. This report is the spec the implementation below was built from.

---

## 1. `math.py` — core math & performance stats

**Best in class:**
- **empyrical** (Quantopian lineage): the canonical stat set — Sharpe,
  Sortino, Calmar, Omega, max DD, VaR/CVaR, tail ratio, capture ratios.
  Lesson: a serious TA kit needs the *full* risk-adjusted zoo, not just
  Sharpe/Sortino. We only had 4 stats.
- **QuantStats / jquantstats paper**: PSR (Probabilistic Sharpe Ratio) and
  DSR (Deflated Sharpe Ratio) from Bailey & López de Prado — the honest
  answer to "is this backtest lucky?". DSR specifically corrects for the
  number of trials run (the 1111-file sweep reality: we run MANY trials).
  Missing entirely from our kit.
- **Andrew Lo (2002)** autocorrelation correction: Sharpe ratios on
  autocorrelated (e.g. smoothed, illiquid, or high-frequency) returns are
  biased; Lo's annualization factor fixes it. Nobody implements this; it is
  a differentiator.
- **Range-based volatility estimators** (Parkinson 1980, Garman-Klass 1980,
  Yang-Zhang 2000): use high/low (and open) — strictly more efficient than
  close-to-close vol. Our `atr`-only world was leaving information on the
  table. Yang-Zhang is the minimum-variance drift-independent estimator.
- **Ulcer Index / UPI** (Peter Martin): drawdown-pain metric retail
  traders actually understand better than Sharpe.

**Trash mined:** dozens of "trading metrics" gists that annualize with 252
on crypto (wrong — 365), compute Sharpe on non-excess returns without saying
so, or divide by zero std and return inf as a feature. We keep the honest
guards (0 when undefined) and infer annualization from bar spacing.

**What we added:** `cagr`, `calmar`, `sterling`, `ulcer_index`,
`value_at_risk`, `expected_shortfall` (CVaR), `omega_ratio`, `tail_ratio`,
`probabilistic_sharpe`, `deflated_sharpe`, `autocorr_adjusted_sharpe`,
`information_ratio`, `parkinson_vol`, `garman_klass_vol`, `yang_zhang_vol`,
`rolling_sharpe`, `max_drawdown_duration`, `skew`/`kurtosis`,
`monte_carlo_shuffle`, `effective_n` (participation ratio).

## 2. `indicators.py` — indicator library

**Best in class:**
- **finta** (peerchemist): clean hand-written pandas indicators, the
  readability bar. Our hand-written style already matches; finta's breadth
  (MFI, KAMA, Aroon, AO, TEMA/DEMA/HMA, Force Index, CMF, StochRSI,
  Choppiness, Connors RSI, Coppock, SMI, Elder-Ray) is the gap — we had 16,
  the canon is ~35.
- **pandas-ta-classic**: 200+ vectorized indicators; lesson is the
  `.ta`-accessor ergonomics and the "everything returns aligned frames"
  contract, which we already honor.
- **talipp** (nardew): the *incremental* insight — O(1) per-bar updates vs
  O(n) full recompute. For Devon's live bot (phone, ticking bars), recomputing
  2000 bars of EMA/RSI/ATR/MACD every tick is pure waste. EMA/RSI(Wilder)/
  ATR/MACD all have exact recursive forms — no approximation needed.
- **TA-Lib**: the C reference for formula correctness (Wilder smoothing
  conventions, SAR acceleration bounds). Used as the correctness oracle.

**Trash mined:** pandas-ta's original repo (abandoned, dependency hell),
indicator zoos with 33k lines of parameter permutations (we already deleted
ours — correct call).

**What we added:** `supertrend`, `mfi`, `kama`, `aroon`, `awesome_oscillator`,
`wma`, `hma`, `tema`, `dema`, `elder_force`, `chaikin_money_flow`,
`stoch_rsi`, `choppiness`, `connors_rsi`, `coppock`, `smi_ergodic`,
`heikin_ashi`, `elder_ray`, `compute_all` (full feature matrix),
`StreamState` (exact O(1) incremental EMA/SMA/RSI/ATR/MACD + bar update).

## 3. `patterns.py` — NEW — candlestick patterns

**Best in class:**
- **TA-Lib**: 61 canonical candlestick patterns — the reference taxonomy.
  Pure pattern functions return -100/0/+100; no context awareness.
- **eminsk/yfinance-ta-patterns**: the key lesson — *naked patterns are
  noise* (~48-51% win rates); edge comes from **confluence scoring**
  (trend alignment via multi-EMA, RVOL surge, RSI exhaustion, ATR
  expansion). We implement confluence natively: every signal can be filtered
  by trend context.
- **Bulkowski's Encyclopedia of Candlestick Charts**: statistical reversal
  rankings — we encode reliability as qualitative tiers (high/medium/low),
  not fabricated percentages.

**Trash mined:** MQL5 pattern gists with hardcoded 10%-of-range doji
thresholds and no ATR normalization — breaks across volatility regimes.
Ours normalizes body/shadow sizes by ATR.

**What we built:** 30 vectorized patterns with signed signals, per-pattern
reliability tiers, `detect_all`, `pattern_score` (reliability-weighted
confluence), `with_trend_context` filter, `confluence_features` for the
meta model.

## 4. `backtest.py` — backtest engines

**Best in class:**
- **backtesting.py** (kernc): the event-driven readability bar; intrabar
  stop/limit semantics (stop touched intrabar = filled) — we only filled at
  close, which *understates* stop-outs. Added `stop_on_touch`.
- **vectorbt**: speed for sweeps; lesson is the metrics breadth on every
  result. Our vector engine now reports the full stat sheet.
- **backtest-forensics-engine** (agentjdrew): the trust insight — every
  backtest result should carry PSR/DSR and a trade ledger with MAE/MFE.
  MAE/MFE (max adverse/favorable excursion) is *the* professional tool for
  stop/target placement; we had neither.
- **QuantStats**: tear-sheet presentation. Our `tear_sheet()` is the
  terminal-native version — plain text that looks god-tier in chat, not a
  notebook HTML page.

**Trash mined:** vectorized backtests that trade at the signal bar's close
(look-ahead), ignore fees, or report Sharpe with no trade count.

**What we added:** full trade ledger DataFrame (entry/exit, side, pnl,
bars held, **MAE/MFE**), intrabar stop touch, `tear_sheet()` text report,
`monte_carlo_trades` (reshuffle significance), `risk_of_ruin`,
`compare()` multi-result table, PSR/DSR wired into metrics, expectancy by
side.

## 5. `signals.py` — fusion

**Best in class:**
- **Bayesian model averaging / stacking**: the literature answer to "how to
  combine models" — weight by out-of-sample skill, not vibes. Our
  `fuse_stacked` does logistic stacking over strategy votes with a
  TimeSeriesSplit when sklearn exists, graceful fallback otherwise.
- **Risk-parity thinking applied to strategies**: correlated strategies
  should not each get full weight (double-counting the same bet). Our
  `fuse_diversified` penalizes each strategy's weight by its mean
  correlation to the committee — the `effective_n` participation ratio
  exposes how many *independent* bets the committee really holds.
- **QuantStats HHI concentration**: same math, applied to vote mass.

**Trash mined:** "AI signal fusion" repos that average 50 indicators with
equal weights and call it ensemble learning.

**What we added:** `fuse_diversified` (correlation-penalized),
`fuse_stacked` (logistic stacking, sklearn-optional), `explain_vote`
(per-strategy contribution table — the presentation gold),
`vote_quality` (effective_n, HHI, agreement), `min_hold` position
smoothing to kill churn, `strategy_correlation`.

## 6. `strategies.py` — strategy zoo

**Best in class:**
- **freqtrade strategy repo**: the community zoo done right — each strategy
  is one readable file with named parameters. Lesson: canonical named
  strategies beat parameter permutations.
- **TuneTA** (jmrichardson): indicator *parameter optimization* via distance
  correlation to forward returns — the honest way to tune. Our
  `optimize_params` is the grid-search version over the vector engine.
- **Connors RSI(2)** (Larry Connors): the most statistically documented
  short-term mean-reversion edge in equities; belonged in the zoo.
- **Supertrend** (Olivier Seban): the retail-standard ATR trailing trend
  system — conspicuously absent from a "canonical" zoo.

**What we added:** `SupertrendTrend`, `KeltnerBreakout`, `MacdCross`,
`ConnorsRsi2`, `StochCross`, `HeikinAshiTrend`, `PatternConfluence`
(patterns → signals), `optimize_params` grid search, all registered in
`STRATEGIES` (now 16).

## 7. `regime.py` — regime detection

**Best in class:**
- **Gaussian HMM via hmmlearn** (multiple repos: quantwcapital,
  ctrl-jesper, aarushiitrpr): the standard ML approach — learn bull/bear/
  sideways as latent states from (log returns, rolling vol, momentum) with
  Baum-Welch, decode with Viterbi, label states by mean return/vol.
  Probabilistic, temporal (transition matrix), no hand thresholds.
- **adlerlinfoot/hmm-fcl2**: the 4-state vol×trend decomposition
  (low/high vol × up/down trend) with expected durations — more
  trade-actionable than bull/bear/sideways.
- Our existing 5-state quantile engine is the *interpretable, dependency-
  free* counterpart; both now exist, HMM optional behind hmmlearn.

**Trash mined:** regime detectors that fit the HMM on the full series then
"predict" in-sample and report accuracy — textbook leakage.

**What we added:** `HMMRegimeDetector` (hmmlearn-optional, 4-state vol×trend
mapping onto our labels, graceful fallback), `transition_matrix`,
`expected_durations`, `persistence_score`, `regime_playbook` (regime →
favored/avoided strategy kinds + exposure multiplier — the actionability
our detector was missing).

## 8. `risk.py` — risk management

**Best in class:**
- **Vince's Optimal f**: the position-sizing math serious system traders
  use; we had Kelly but not optimal-f.
- **Chandelier exit** (Chuck LeBeau): ATR trailing stop from the highest
  high — the professional alternative to fixed-ATR stops; our stops were
  static ladders only.
- **Risk parity** (Bridgewater/Quantopian lineage): size by inverse
  volatility so each position contributes equal risk — our sizing was
  single-trade only, no portfolio view.
- **Drawdown-shrunk Kelly** (practitioner standard): full Kelly is
  untradable; shrink by drawdown depth. We had quarter-Kelly; now it also
  breathes with the drawdown.

**Trash mined:** "risk managers" that are just a stop-loss constant, and
Kelly calculators that suggest betting 80% of equity.

**What we added:** `Position` + `PositionTracker` (live position state:
breakeven moves, trailing ratchets, time stops, partial targets, heat),
`risk_parity_weights`, `correlation_adjusted_fraction`,
`chandelier_exit`, `optimal_f`, `kelly_drawdown_shrink`,
`portfolio_heat` with per-trade risk, `max_simultaneous_risk` check.

## 9. `meta.py` — meta-labeling

**Best in class:**
- **López de Prado, AFML Ch. 3-4, 10** (via mlfinlab, darufinance
  work-review, limjoony94 experiments): the canonical stack —
  **triple-barrier labeling** (PT/SL/vertical barrier on the actual
  high/low path, ATR-scaled) → **meta-label** (did the primary side win?)
  → secondary classifier → **bet sizing from P(prob)** → purged/embargoed
  CV so overlapping labels never leak. Our old `labeled_matrix` used a
  naive forward-return label — no barriers, no purge. That was the
  weak link; now fixed properly.
- **Bagged trees + OOF predictions** (darufinance): meta-probabilities must
  come from out-of-fold predictions, not in-sample fit.

**What we added:** `triple_barrier_labels` (OHLC path, ATR-scaled,
{-1,0,1}), `purged_kfold_splits` (embargo + label-span purge),
`bet_size_from_prob` (de Prado sizing curve), `oof_predict_proba`,
`permutation_importance` (model-agnostic), `MetaGate.summary`.

## 10. `data.py` — bar data

**Best in class:**
- **DeTime / exchange data adapters**: the integrity checklist — reject
  missing/duplicate/out-of-order bars, split-adjustment policy, session
  info. Our `clean_ohlcv` fixed inversions but never *reported* what it
  found.
- Practitioner rule: never silently repair data — report it.

**What we added:** `quality_report` (gaps with locations, outliers,
stale bars, duplicates, volume anomalies — the "never silently repair"
rule), `detect_outliers`, `align_frames` (multi-symbol common index),
`bar_gaps`.

## 11. `feeds.py` — live feeds

**Best in class:**
- **ccxt**: unified OHLCV across 100+ exchanges; lesson is the *uniform
  frame* contract (which we keep) plus implicit rate-limit handling.
- Public no-key endpoints: **Kraken** (`/0/public/OHLC`) and **Bybit**
  (`/v5/market/kline`) both serve klines with zero authentication — our
  feeds required connector auth even for public data. Now: keyless public
  fetchers + disk cache with TTL, so the phone bot isn't hammering APIs.

**What we added:** `fetch_kraken`, `fetch_bybit` (stdlib urllib, no key),
disk cache (`~/.cache/devon/ta_feeds`, TTL), cache-aware `fetch_ohlcv`,
`SOURCES` extended.

## 12. `pipeline.py` — committee pipeline + presentation

**Best in class:**
- **QuantStats tear sheets**: the presentation bar — one page, everything
  that matters, scannable in 10 seconds. Our `analyze()` returned a raw
  dict; nobody can read a dict in chat.
- **Multi-timeframe confluence** (Elder's Triple Screen — mined from the
  fintrade repo's Analyse.md): higher timeframe decides *permission*,
  lower timeframe decides *timing*. Our pipeline was single-timeframe.

**What we added:** `render_report` (god-tier text briefing: regime banner,
unicode bias meter, committee contribution table, trade-plan box, risk
readout — two themes `rich`/`plain`), `trade_plan` (entry/side/size/
stops/targets/risk in one dict), `analyze_mtf` (higher-TF regime gate),
`explain_committee`.

---

## Style doctrine applied

Every user-facing surface got a presentation pass: `render_report` and
`tear_sheet` are designed for chat/terminal, not notebooks — unicode meters
(▓░), aligned tables, plain-English verdicts. Numbers are rounded for
humans; machines keep the raw dicts.
