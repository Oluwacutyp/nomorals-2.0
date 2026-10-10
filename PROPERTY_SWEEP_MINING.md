# PROPERTY_SWEEP_MINING.md — real external gold for nomorals/property

Mined 2026-10-10. Every item below was taken to real code in
`nomorals/property/` (scam.py, passport.py, value.py). Nothing invented.

## 1. Rental scam detection — scam.py

### Fredy (orangecoding/fredy, doc/scam-detection.md) — the richest gold
- **Weighted signals, threshold 3**: each fraud signal has a weight
  (1–3); a listing is flagged at total weight ≥ 3, so one weight-3
  signal fires alone but a single weight-2 signal never does alone.
  Implemented: severity→deduction map extended + explicit
  "corroboration" rule — weak signals only combine.
- **Signals we were missing**: `noViewing` ("a viewing is said to be
  impossible", weight 2), `landlordAbroad` (landlord claims to be abroad,
  weight 2), `keysByPost` (keys promised by post, weight 3),
  `advancePayment` ("money asked for before anyone has seen the flat",
  weight 3), `moneyTransferService` (irreversible route — Western Union,
  gift cards — weight 3), `priceFarBelowMarket` (40%+ under local median,
  weight 2).
- **Deliberately-excluded signals (false-positive control)**: a deposit,
  agency fees, and a phone/email in the description are NORMAL and fired
  on honest listings — they must never be flags by themselves.
  → We downgraded `caution fee` from danger to warn ("verify refundable
  terms in writing") and keep it evidence-based.
- **"A cheap flat is not a scam"**: price anomaly alone should not
  hard-flag; it needs corroboration. → Our price_anomaly at info/warn
  stays; only danger+urgency/personal-account combos hard-stop.

### nilayraut/rentsentry (GitHub) — AI rental-listing fraud detector
- Verdict bands on a 0–100 trust score: 70–100 safe / 40–69 suspicious /
  0–39 likely_scam — matches our existing thresholds (validates them).
- Suspicion categories 0–20 / 21–40 / 41–65 / 66–85 / 86–100.
- Module pattern: never-raises LLM analysis with safe defaults,
  injectable scraper — our fetcher/photo_analyzer seams already match.

### FTC rental-scam guidance (via financeply/curiosityfacts Zillow guides)
- Core red flags: demands for **wire transfers, gift cards, or
  cryptocurrency** — "work like cash, nearly impossible to trace or
  recover". → Added irreversible-payment-route detection
  (crypto/USDT/BTC, gift cards, Western Union/MoneyGram) to the payment
  check.
- "Pressure to book immediately" + immediate payment demand treated
  with real skepticism. → Kept/expanded urgency+payment hard stop.

### LASRERA / Lagos law (legit.ng, joliba.com.ng, nairametrics.com, blog.buyletlive.com)
- Governor Sanwo-Olu: agency fees capped at **10%** under LASRERA law;
  "every other fee attached to rentals is illegal" (legit.ng).
- LASRERA head: "Agency fees must fall within 0 to 10%. Anything beyond
  this is unacceptable"; **unlawful to collect more than one year's rent
  upfront** (joliba.com.ng).
- Lagos Tenancy Law 2011: agency/legal fees capped at 10% of annual
  rent each (nairametrics.com).
- LASRERA recovered ₦478m + 18 properties in ~4 years; 1,700+ fraud
  cases since 2020 — the reporting channel is real.
  → true_cost now warns when >1 year is demanded upfront; agency norm
  10% is already the legal cap, not just a norm.

## 2. Rental passport / true cost — passport.py

### Zillow rental applications (smartscreen.clearscreening.com, digitaltrends.com, financeply.com)
- **Verify once, apply anywhere**: one $29–35 application covers
  unlimited applications to participating properties for **30 days**.
- **Freshness caveat**: credit/background reports are pulled once at the
  start; Day-28 applications show 4-week-old data → landlords should
  request updated reports.
  → Implemented: `attested_at` on the passport, `is_fresh(30d)`,
  freshness note in export, staleness warning.

### OpenRent tenant referencing (help.openrent.co.uk, blog.openrent.co.uk)
- Affordability pass: tenant must earn **≥ 2.5× annual rent**;
  **guarantors 3×** (help.openrent.co.uk). Our Kwaba 33%-of-monthly rule
  is equivalent to 3.03× — documented as OpenRent-consistent.
- **"Passed in conjunction"**: joint tenants may pass referencing
  together when combined incomes cover the rent.
  → Implemented: `combined_affordability()`, `can_guarantee()`
  (guarantor 3× rule).
- Referencing report structure: Affordability + Credit + Fraud/Identity
  + Previous landlord + Income/employment (comprehensive referencing,
  3–5 day turnaround, ~£30/applicant).
  → Implemented: `referencing_readiness()` scoring those five
  components from the passport → completeness % + what's missing.
- Outgoings matter as much as income ("if more is being paid out each
  month than coming in, that's not good" — OpenRent community).
  → true_cost format now shows the amortized **monthly** true cost and,
  when a passport band is set, the true monthly vs band midpoint.

### Kwaba (kwaba.ng via loanspot.ng, nyscinfo.com, technext24.com)
- Required documents: valid **BVN**, recent **utility bill**, work
  details, government ID — matches Nigeria's real KYC stack.
  → Implemented: `NIGERIA_KYC_DOCS` checklist + `missing_kyc_docs()`
  matched against the passport's doc-ref labels.
- Salary-earner minimum ₦80,000/mo for rent-now-pay-later.
- Pay-your-rent-upfront + monthly-installments model → the amortized
  monthly true-cost figure is the operative number in Lagos.

### Monthly housing cost (financeply.com Zillow guides)
- "Monthly affordability should include rent, utilities, parking…;
  compare total monthly housing costs."
  → `CostBreakdown.monthly()` + format shows monthly equivalent.

## 3. Valuation — value.py

### HouseCanary (support.housecanary.com, housecanary.com white papers)
- **Confidence Score = 1 − FSD**, as a percentage (fsd .12 → 88%
  confidence). FSD is the statistical model-uncertainty measure; the
  ±1σ range P×(1±FSD) captures the actual sale price ~68% of the time.
  → Implemented: `ValueEstimate.confidence` property + displayed in
  format ("confidence 72%").
- FSD is trained on the empirical error distribution AND on
  agreement/disagreement among component price estimates — our
  weighted-comps stdev already models disagreement; documented.
- AVM breakdowns are only offered where data density allows; where
  data is thin the breakdown is withheld rather than fabricated.
  → Thin-market wide band (existing) + now labeled with its basis.
- "We surface our underlying data so you can compare sources yourself."
  → `Comp.source_trust` (0–1): seeded norms 0.6 vs live listing 1.0,
  weights comps by similarity × trust, shown in the comp listing.

### Zillow Zestimate accuracy (published figures via houwzer.com, mjmgroupfl.com, nailandkey.com)
- Nationwide median error: **~1.9–2.4% on-market, ~7.2–7.5% off-market**
  (Zillow's own published data). Our seeded-norm comps are
  off-market-class inputs — documented as such in the disclaimer line
  ("seeded norms, not live listings" was already there; now the FSD
  floor for seeded sources is set honestly).
- Zestimate "never stood in the driveway" — thin-market caution kept.

### Redfin (activebeat.com comparison data)
- Margin of error ~1.99% on-market / ~7.95% off-market — same class as
  Zillow; validates our two-tier treatment (live comps vs seeded).
- Redfin-style comp trust: we already had the owner veto/swap pattern
  (pick_comps).

### Zillow Offers post-mortem (scienceshot.com, mjmgroupfl.com)
- $500M+ write-down; "Zillow never bought the average home — it bought
  the homes whose owners said yes." Winner's-curse asymmetry: models
  misprice worst where the other side selects against you.
  → Strengthened DISCLAIMER line; estimate trend/watch added instead of
  any commitment path (already money-gated by design).

### Trend (Zillow guidance: "watch the general direction of your equity over time")
- → `ValueStore.history()` + `trend()`: direction + % change across
  stored estimates for an area/bedroom class.
