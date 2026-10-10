# FINANCE SWEEP — External Mining Report

Mined 2026-10-10 for the true system-wide upgrade of `nomorals/finance/`
(14 files). For every significant class: how the best implementation OUTSIDE
the repo does it, and what gold gets merged. Trash builds were mined too —
several "AI finance trackers" turned out to be prompt-wrappers with no real
analytics; their UX copy was the only thing worth stealing.

Sources: Actual Budget (MIT), YNAB, Rocket Money/Truebill (incl. Rowan, Aug
2026), Firefly III (AGPL — inspiration only), Edgewonk/TradeZella/edgebook,
Van Tharp position-sizing literature, Coinbase Agentic Wallets / MetaMask
Advanced Permissions (ERC-7710), OpenClawCash, fraud-detection literature
(PaySim, Z-score/Isolation-Forest work).

---

## 1. Ledger (`ledger.py`) — best: Actual Budget, Firefly III, smartmoney

**How the best do it:**
- Actual Budget: local-first, integer cents everywhere (we already do kobo —
  good), a **rules engine** that auto-categorizes on import (match on
  payee/amount/date, user corrections become rules), CSV/OFX/QIF import as a
  first-class intake, duplicate detection on import, schedules for recurring.
- Firefly III: transaction **rules with triggers+actions** ("test rule against
  existing transactions"), tags, "piggy banks".
- mrahmadt/smartmoney (Firefly companion): regex bank-SMS parsing + AI
  fallback, **spending anomaly detection**, subscription detection,
  alternative merchant categories with inline reassign.
- Trash mined: several "expense tracker" repos were CRUD over SQLite with
  float money — float money is the classic tell. We keep integer kobo.

**Gold merged:**
- Merchant **learning**: `Ledger.learn(note, category)` persists
  merchant→category corrections; `categorize()` consults learned mappings
  FIRST (Firefly's "alternative categories" pattern). Corrections stick.
- `Ledger.import_csv(path)` — bank statement intake (Actual's core intake
  path) with header sniffing, amount parsing via `parse_amount`, and
  duplicate detection (same amount + merchant within 3 days →
  `duplicates()`), matching firefly-iii-categorizer's duplicate detection.
- `Ledger.search(text)` — full-text search over notes/merchant (needed for
  any real ledger UX).
- `parse_amount`: word suffixes ("5 thousand", "2 million") alongside
  k/m shorthand; cents/kobo decimals kept.
- `Transaction.id` (uuid) + `merchant` extraction helper — every best
  journal keys records by id, not by array position.

**What it SHOULD have that it doesn't:** import is the #1 missing feature
of every manual ledger. Now it has CSV import.

---

## 2. Budgets (`budgets.py`) — best: Actual Budget envelope math, YNAB

**How the best do it:**
- Actual's envelope math (ported in prose by lraulin/planner from
  `loot-core/.../envelope.ts`): `balance(c,m) = assigned + activity +
  carryIn`; **Ready to Assign** = income + from-last-month − assigned −
  buffered; overspend rolls forward only with an explicit carryover flag,
  otherwise it's absorbed by next month's Ready to Assign. Overspending is
  *somebody's problem*, never silent.
- YNAB's 4 rules: give every dollar a job; embrace true expenses (sinking
  funds for big future bills); roll with the punches (move money between
  envelopes, it's progress not failure).
- 50/30/20 framework (bloknayrb/money-money): needs/wants/savings split
  with HCOL adjustments; too coarse alone, great as a starter template.
- Actual's "Quick Budget" presets: average / spent-last-month / scheduled.

**Gold merged:**
- `budget_pace(status, now)` — expected % of month elapsed vs actual used →
  "on pace / ahead / behind" with projected month-end. (YNAB's "underfunded"
  number, generalized.)
- `BudgetStore.copy_forward(month)` — clone last month's budgets into a new
  month (Actual's Quick Budget, manual but real).
- `suggested_budgets(ledger, months)` — per-category average of trailing
  months as a starter template (Actual's "average" preset); plus
  `fifty_thirty_twenty(income_kobo)` template.
- `render_budget_grid(statuses)` — the envelope view the module never had:
  progress bars, pace flags, remaining/day.

**What it SHOULD have:** pace. A budget without a pace projection is a
rear-view mirror. Now every status line shows whether you're on pace.

---

## 3. Alerts (`alerts.py`) — best: TradingView, Rocket Money Rowan (2026)

**How the best do it:**
- TradingView: multi-condition alerts, webhook delivery, alert logs; the
  value is *reliability of evaluation*, not fancy conditions.
- Rocket Money's Rowan (Aug 2026): an AI agent that **texts when it finds
  something** — price increases on subscriptions, unexpected fees, upcoming
  trial-to-paid conversions. The killer feature set is *proactive
  subscription intelligence*, not generic price lines.
- Best subscription trackers (2026 roundup): upcoming-charge lists with due
  dates, price-change notifications, duplicate-subscription spotting.

**Gold merged:**
- New alert kind `bill_due` — fires N days before a detected recurring
  charge's next due date (Rowan's "sign up for a free trial, we text you
  before the first charge" pattern).
- `sync_bill_alerts(store, ledger)` — auto-generates/refreshes bill-due
  alerts from `detect_recurring` output. The watchtower now watches the
  ledger, not just prices.
- Price-hike flag on recurring charges (see insights) feeds alert copy.
- `Alert.snooze_until` (YNAB's "snooze" — the most-loved goal feature,
  generalized to alerts).
- `render_alerts()` — one readable watchtower view.

---

## 4. Insights (`insights.py`) — best: Rocket Money, Quicken Simplifi

**How the best do it:**
- Rocket Money: subscription list with **price-increase detection**,
  upcoming due dates, "vampire subscription" surfacing, per-type grouping.
- Quicken Simplifi: Spending Plan = income − recurring − targets → **"safe
  to spend"** number. That's the single most useful derived number in
  consumer finance and we didn't have it.
- smartmoney: daily/weekly/monthly anomaly analysis with thresholds.

**Gold merged:**
- `RecurringCharge` gains `price_changed` (first vs last amount —
  Rocket Money's rate-hike flag), `next_due_ts` (forecast from cadence),
  `yearly_cost_kobo`.
- `upcoming_bills(ledger, days)` — charges due in the window with amounts.
- `safe_to_spend(ledger)` — Simplifi's number: expected monthly income −
  committed monthly recurring − already spent this month.
- `forecast_month_end(ledger)` — burn-rate projection to month end.
- `spend_series(ledger, months)` → monthly totals for sparklines.

**What it SHOULD have:** safe-to-spend. The answer to "can I buy this?"
Now computed natively.

---

## 5. Guard (`guard.py`) — best: fraud-detection literature, PaySim

**How the best do it:**
- PaySim research: fraud is ~0.1% of transactions (extreme imbalance —
  rules beat ML on tiny ledgers); fraud amounts skew higher; **transaction
  velocity** (frequency in a short window) is a top engineered feature.
- FasterCapital/anomaly literature: z-scores on amount vs history,
  Isolation Forests for unsupervised flagging; custom risk-score formulas
  `Score = Σ wᵢ·signalᵢ` with a 0–100 threshold.
- The MDPI ARIMA paper: model *normal* behavior, flag deviations from the
  predicted baseline (z-score on prediction error).

**Gold merged:**
- `risk_score(to, amount_kobo, ledger)` → 0–100 weighted score:
  amount z-score vs category history (35), new recipient (25), unusual
  hour (15), velocity — ≥3 sends in 24h (15), round-number-to-new-recipient
  (10). Thresholds: ≥60 warn, ≥85 strong warn.
- `check_outgoing` now returns the score + per-signal contributions in
  context (the "Why, and the numbers" pattern from nuthan79/trading-journal).
- Advisory-only posture kept (user's explicit override rule).

**What it SHOULD have:** a score, not just reasons. "Why is this 82/100?"
is now answerable per signal.

---

## 6. Trading desk (`trading_desk.py`) — best: Edgewonk, TradeZella, edgebook, Van Tharp

**How the best do it:**
- Edgewonk/TradeZella/edgebook analytics: win rate, **profit factor**,
  **expectancy** (in R and currency), average R, max drawdown, R-multiple
  distribution, long-vs-short, streaks, equity curve, P&L calendar.
- Van Tharp: `Position Size = Risk $ ÷ (Stop distance × $ per unit)`;
  **R-multiple** = outcome ÷ initial risk (−1R = stopped, +2R = 2× risk);
  **portfolio heat** < 6% (total open risk); Kelly criterion for sizing
  guidance; 100-trade minimum before raising risk.
- 4von/tradetracker-v2: Monte Carlo risk-of-ruin, run-rate calculator.

**Gold merged:**
- Journal records **R-multiple** on every close (planned risk captured at
  open; R = pnl ÷ risk_amount). `PaperPosition` gains `risk_amount`.
- `stats()` adds: profit factor, expectancy (R + currency), average R,
  max drawdown, best/worst trade, streaks, long-vs-short split.
- `equity_curve()` — cumulative R series for sparklines.
- `portfolio_heat()` — Σ open risk ÷ equity (Van Tharp's <6% rule,
  surfaced as a number + flag).
- `kelly_fraction()` — (W − (1−W)/R̄) from journal stats, quarter-Kelly
  reported as the sane default.
- `/trade` chat command (the desk had NO chat surface — the biggest gap).

**What it SHOULD have:** R-multiples. Without R, "win rate 60%" is
meaningless. Now every close is recorded in R.

---

## 7. Mandates (`mandate.py`) — best: Coinbase Agentic Wallets, MetaMask ERC-7710, OpenClawCash

**How the best do it:**
- Coinbase Agentic Wallets (2026): programmable spending policies —
  spending limits, transaction limits, immediate revocation; the agent never
  holds the key.
- MetaMask Advanced Permissions: `wallet_requestExecutionPermissions` shows
  a **human-readable approval screen** (asset, amount, duration,
  constraints); the session redeems via ERC-7710 inside the granted scope.
  Rail vs permission — the permission is the product.
- OpenClawCash: weekly AND monthly limits (we only had per-day).

**Gold merged:**
- Optional `cap_per_week` / `cap_per_month` (OpenClawCash gold), enforced
  in `check_mandate` alongside daily.
- `describe()` — the human-readable approval screen (MetaMask gold):
  "Devon may spend up to ₦50,000/txn, ₦200k/day, ₦1M/month on transfers
  until 2026-11-09. Revocable anytime."
- `remaining()` snapshot — per-window remaining on one line.
- Mandate **spend ledger** reuse: weekly/monthly spend computed from the
  transfer-audit trail (no new storage).

**What it SHOULD have:** weekly/monthly caps. Daily-only caps are trivially
gamed by waiting for midnight. Now covered.

---

## 8. Goals (`goals.py`) — best: YNAB goals, Firefly III piggy banks

**How the best do it:**
- YNAB's three goal types: **Target Balance** (have ₦X by date Y),
  **Monthly Builder** (save ₦X every month), **Spend-by-date** (need ₦X for
  a bill on date Y); "By Date" auto-breaks a big target into monthly
  chunks; **snooze** a goal when life happens; underfunded highlighting.
- Firefly III piggy banks: dead-simple named pots with progress.

**Gold merged:**
- `goal_type`: `target` | `monthly` | `by_date` (YNAB's three, natively).
- `monthly_need` auto-calc per type (the "₦500 in October = ₦70/month"
  math YNAB does for you).
- `snooze_until` (YNAB's beloved snooze).
- Contribution **streaks** — consecutive months with a contribution
  (the behavioral hook every savings product uses).
- `render_goals()` with progress bars + pace flags.

---

## 9. Overview (`overview.py`) — best: Rocket Money aggregation, Copilot

**How the best do it:**
- Rocket Money/Monarch/Copilot: every account in one place via Plaid; the
  differentiator is **trend** (net worth over time) and allocation views,
  not the raw total.
- Honest absent-data handling (we already do "absent ≠ zero" — kept).

**Gold merged:**
- `BalanceCache` — TTL-cached snapshots (no API hammering on every
  `/balances`; each rail fail-soft as before).
- `snapshot_net_worth()` → JSONL history; `net_worth_series(days)` for
  sparkline trend.
- Asset-mix breakdown in `render_overview`: fiat vs crypto vs unconverted.

---

## 10. Send (`send.py`) — best: conversational money UX

Already best-in-class structurally (parse → resolve → mandate → guard →
stage → biometric → execute, fail-closed). Mined MoMo/USSD flows for copy
patterns.

**Gold merged:**
- Confirmation echo shows **masked bank details**
  ("₦5,000 → Mama · GTBank ••4521") — the #1 anti-misdirection UX in
  real money apps.
- `RecipientStore.remove()` / `rename()` (were missing — you could add but
  never fix a recipient).
- Staged transfers get **expiry** (15 min) — a staged-but-unconfirmed
  transfer shouldn't live forever.

---

## 11. Style (NEW `style.py`) — best: textual/rich CLI design, Actual's UI

No output theming existed anywhere in the module. Mined rich/textual
conventions: progress bars, sparklines (▁▂▃▄▅▆▇), tables, semantic color.

**Gold merged (new file, no parallel system):**
- `Theme` dataclass: `ninja` (default — electric-blue, matches the user's
  Termux theme), `plain` (no ANSI — logs/pipes), `minimal`.
- `bar(pct)`, `sparkline(series)`, `money()`, `table()`, `header()`,
  `status_dot()` shared by every render function in the module.
- All `render_*` functions and chat commands route through it; theme
  selectable via `FINANCE_THEME` env.

---

## 12. Tools & commands (`tools.py`, `commands.py`, `digest.py`)

**Gaps found by mining:** no ledger search tool, no import tool, no
recurring-bills tool, and the trading desk had zero chat/tools surface.

**Gold merged:**
- Tools: `finance_search`, `finance_import` (CSV), `finance_recurring`
  (list + price hikes + upcoming), `finance_trade` (size/open/close/
  positions/stats/heat — full desk surface).
- Commands: `/recurring`, `/trade`, `/import`.
- Digest: upcoming-bills section + safe-to-spend line (Simplifi gold).

---

## Deliberately NOT merged

- Firefly III's double-entry model (AGPL; our single-entry kobo ledger is
  the right scope for a chat-first personal ledger).
- ML categorization (no training data on-device; rules + merchant learning
  win on tiny ledgers — matches the fraud literature's "rules beat ML on
  small data").
- Live trading expansion beyond the desk (user's 7B plan / Sentinel work is
  separate; the desk stays discipline-first).
- Actual's sync server (out of scope for this module).
