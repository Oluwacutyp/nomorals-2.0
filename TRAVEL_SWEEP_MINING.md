# TRAVEL SWEEP — External Mining Report

Module: `nomorals/travel/` (6 files: `__init__`, `watchers`, `itinerary`,
`display`, `whitelabel`, `groups`). Sweep date: 2026-10-10.

For every significant class the question asked was:
**"How does the best implementation of X do it — and what SHOULD this have that it doesn't?"**

## 1. PriceWatcher / PriceWatch / PriceAlert / PricePoint — `watchers.py`

### Sources mined
- **Hopper** (citizendailypost.com/faq, d3.harvard.edu Digital Innovation case,
  editorialge.com, groupon.com): collects billions of price quotes/day; the
  product's core is NOT the alert — it's the **buy-now / wait forecast**
  (claimed up to 95% accuracy): historical data + current prices → "buy now if
  prices expected to rise, wait if a drop is predicted". UX: color-coded
  calendar (green = cheap, red = expensive), price freeze for a fee, and the
  follow-through hook — **alerts that fire when the recommendation shifts to
  "buy now"** (claimed 90% of bookings originate from such pushes).
- **telegram-flight-ai-assistant** (github.com/alessio2405): Crew.ai multi-agent
  system — Search Specialist → Price Analyst → Booking Assistant → Notification
  Specialist; persistent sessions; voice alerts. Lesson: the pipeline should be
  staged (search → analyze → notify), not one blob.
- **PriceTrackerBot** (github.com/nuhmanpk) and **Telegram-Crypto-Alerts**
  (github.com/hschickdevs): `/watch btc drop 50% 14 days persist`,
  `/new_alert BASE PRICE PCTCHG 10.0 1200 1h` — **rise/drop/percent/absolute
  trigger modes, cooldowns, persistent watches, `/showwatches`, `/delete`**.
  Also "from ATH" baselines (all-time-high → here: all-time-low of the route).
- **Skyscanner "Everywhere" search** (editorialge.com): flexibility-first
  discovery when price matters more than destination.

### Gold extracted → what we should have and don't
1. **Price history is the missing organ.** We alert on one-step drops but keep
   no history — no trend, no forecast, no "vs last week". → Add a
   `price_history` table written on every check; derive min/avg/max, 7-day
   moving average, drop-vs-peak.
2. **Hopper's buy-now/wait verdict.** Implement `forecast()` (pure): given
   history + current fare → verdict BUY NOW / WAIT / WATCH with confidence +
   reasons (below 30-day low → buy; rising vs 7d avg → wait). Heuristic, honest
   about limits (dynamic revenue management).
3. **Rise alerts + cooldowns**, not just drops; alert text should say WHY NOW
   (below 30-day low, forecast says buy) — telegram-flight-ai-assistant's
   route/previous→current/savings/scarcity/buttons payload we already follow;
   add trend line + verdict.
4. **Watch list formatting** (`format_watches`) — `/showwatches` equivalent.
   Persist mode is already structural (active watches survive restarts).
5. Price-freeze and ML prediction: out of scope (needs backend/billions of
   quotes) — the honest heuristic layer is the floor here.

## 2. ItineraryBuilder / parse_confirmation / Trip / Flight / HotelStay / CarRental — `itinerary.py`

### Sources mined
- **TripIt** (macworld.com, traveldailynews.asia, executivetraveller.com,
  pcworld.com, tripadvisor threads): the gold standard of confirmation parsing —
  forward emails → master itinerary. Best practices we don't have: (a) **merge
  trips** — separate forwards become one trip when dates overlap ("Options →
  Merge Trips"); (b) **every reservation type** — rail, restaurants, activities,
  cruises, not just flight/hotel/car; (c) **delay/change alerts** (red alert on
  itinerary), **check-in-open alerts**, gate + baggage carousel info, "will I
  make my connection"; (d) share itinerary with friends; (e) notes per
  segment; (f) weather + directions per destination.
- **Wanderlog** (play.google.com, apkmirror.com, androidauthority.com):
  reservations + attractions in one place, day-by-day drag-drop itinerary,
  Google-Maps route view, **expense tracking per trip**, offline access.
- **Layla** (therundown.ai, phocuswire.com, editorialge.com): conversational
  builder that **asks clarifying questions** (pace, diet, must-sees) and
  produces a day-by-day plan with **budget breakdown by category**
  (accommodation/food/activities/transport/buffer) that sums to the budget;
  realistic pacing (buffer time, don't over-schedule).

### Gold extracted
1. **Trip merging**: multiple forwards of the same booking → one Trip.
   Auto-merge fires only on a shared PNR/booking reference (precise, never a
   surprise); date-proximity matches (±3 days, the flight-lands /
   hotel-next-morning case) surface via `find_merge_candidate()` as a chat
   suggestion. Explicit `merge_trips()` folds trips on demand.
2. **Airline name resolution** from flight-number prefix (BA→British Airways;
   Nigerian carriers: Air Peace AP? actually Air Peace is P4; Arik W3; Ibom QI;
   ValueJet VK; Max Air VM; Dana 9J; Overland OJ; Green Africa Q9). Small
   curated IATA→airline map.
3. **Timeline view** with countdowns ("departs in 3d 4h") — the travel-day card
   pattern.
4. **Conflict detection**: overlapping flights, hotel check-in before flight
   arrival, gaps. Real value, cheap to compute.
5. **iCal export** (`.ics` file in the vault) — TripIt syncs to calendar; a
   downloadable .ics works without API keys.
6. **Packing checklist** generated from trip facts (international → passport;
   nights → clothing count; flight → charger/docs).
7. **Trip status** (upcoming / in-progress / past) from dates; `confirmation_hook`
   scoring stays but gets weighted signals.
8. Layla's budget-breakdown-by-category is a chat-brain concern (LLM does the
   categorizing); the itinerary side contributes `estimate` hooks — not forced
   into this class.

## 3. LoyaltyProgram / points_vs_cash / multi_origin_search / enrich_* — `display.py`

### Sources mined
- **point.me** (thepointsguy.com, upgradedpoints.com, prnewswire.com):
  side-by-side cash-vs-points, **cents-per-point (CPP) = (cash − taxes)/miles**,
  ≥1.5¢ is a good deal; step-by-step booking instructions; "never transfer
  speculatively — confirm the award seat first"; 150+ airline programs;
  transferable currencies (Amex MR, Chase UR, Citi, Capital One, Bilt).
- **pointsyeah-api** (github.com/farzanariel): CLI that prints **awards
  side-by-side with cash, sorted by CPP**, green highlight for CPP ≥ 1.5¢ —
  exactly the terminal-first aesthetic Devon wants.
- **Wonderplan/SearchSpot pattern** (already in our docstring): budget-first
  output — total cost in message one.

### Gold extracted
1. **CPP computation + redemption rating tiers** — we show "65k miles + ₦80k"
   but never say whether it's a GOOD deal. Add `cpp()` and
   `rate_redemption()` (great ≥ 2.0¢, good ≥ 1.5¢, fair ≥ 1.0¢, poor < 1.0¢),
   green-highlight style in text (⭐/✅ markers).
2. **Best-program recommendation**: across all programs with balance, pick the
   max-CPP award → "use X first". Plus the point.me rule: warn "confirm the
   award seat before transferring".
3. **Transferable-bank map**: small curated bank → partner programs table
   (Amex MR, Chase UR, Capital One, Citi) so the recommendation can say "move
   Amex MR → Flying Blue".
4. **Output themes** (style mandate): "rich" (current, emoji + lines), "compact"
   (one-liners), "minimal" (numbers only) — theme parameter threading through
   the enrich functions.

## 4. TravelClient / TravelClientStore / viki_template / answer — `whitelabel.py`

### Sources mined
- **GuideGeek** (en.wikipedia.org, mrtechking.com, ideausher.com,
  futureaimind.com, thetoolsverse.com, meetingsmags.com): free AI travel
  assistant inside WhatsApp/Instagram/Messenger/web — **no app to install**;
  RLHF-tuned answers; **white-label monetization via 70+ DMO clients**
  ("Pythia" for Discover Greece, "Rocky Mountain Roamer" for Estes Park);
  conversations in 15+ languages; **image recognition** for suggestions;
  voice input; real-time weather/events; human curation layer; WhatsApp group
  trip planning; three layers — Knowledge (landmarks, culture, transport
  logic), Language (fragments, emotion), Decision (adapts, narrows, refines).
- **point.me Gateway** (prnewswire.com): the same white-label idea in award
  travel — partners live under their own brand in 60 days.

### Gold extracted
1. **Client analytics**: we track spend (#68) but nothing else. Add event log
   (query/booking/escalation) + `client_analytics()` — queries, top topics,
   escalations, spend in one report. A B2B product without analytics is blind.
2. **Template registry beyond airlines**: hotel concierge, tour operator,
   car rental — viki is airline-only; GuideGeek's DMO clients are destinations.
   `TEMPLATES` dict + `template_for(kind, name)`.
3. **Escalation detection**: keyword patterns (refund, complaint, human agent,
   angry signals) → `route_to_human()` suggestion — the human-curation layer.
4. **Onboarding checklist**: `client_onboarding()` — knowledge loaded? channel
   connected? billing set? Every new client gets a readiness report.
5. **Multi-channel + language fields** on the client record (channels list,
   languages) — GuideGeek lives on 4 surfaces, we assume WhatsApp.
6. Knowledge versioning: timestamp + source on docs; `list_knowledge()`.

## 5. GroupTrip / GroupPoll / GroupExpense / TripLeg / TripProposal / GroupTripStore / settle_balances — `groups.py`

### Sources mined
- **Splitwise research** (medium.com debt-simplification deep dive,
  github.com/subhm2004/splitwise, github.com/alvinbanerjee/...,
  github.com/kuldeephumbal/det-admin, github.com/prateekscaler/...): (a) **three
  split modes** — EQUAL, PERCENT, EXACT, computed in integer minor units,
  always summing exactly; (b) min-cash-flow greedy settlement (matches ours;
  note the honest caveat: exact minimum is **NP-hard/subset-sum**, greedy is
  near-optimal — our code should stop implying "minimized" as provable); (c)
  **record payments / partial settlements**; (d) settle-guard on leaving;
  (e) balances recomputed from expenses as source of truth.
- **Troupe** (makeuseof.com, disneyfoodblog.com): destinations + stays +
  activities as votable suggestions, **broadcast messages**, **deadlines AND
  reminders** for polls, notes per item, anyone can suggest.
- **Wanderlog**: budget tracking + expense logging in the same trip, route
  optimization.
- **G8Trip / Plan Harmony**: already embodied (per-traveler legs, proposal vs
  agreed duality) — keep and deepen.

### Gold extracted
1. **Split modes EQUAL/PERCENT/EXACT** — the single biggest gap: real group
   expenses are rarely equal splits. Shares stored per expense; integer-kobo
   math that sums exactly (Splitwise's integer rule).
2. **`record_payment()`** with partial support + payments table; balances =
   expenses − payments. Settle-up becomes trackable ("Ada paid ₦5k of ₦8k").
3. **Poll reminders**: `polls_closing_soon()` + non-voter listing so a nudge can
   go out — Troupe's deadline-reminder pattern.
4. **Trip dates + destination** on GroupTrip (schema migration, additive
   columns) — every real planner has them.
5. **Expense categories** (`#food`, `#stay`, …) + per-category rollup in trip
   stats — Wanderlog's expense tracking.
6. **Per-person cost rollup** (`member_totals`) and a `trip_overview()` with
   budget progress bars.
7. **Settle-guard `leave_trip()`** — can't leave with a nonzero balance.
8. Honest docstring on `settle_balances`: greedy near-optimal, not provably
   minimal.

## Presentation mandate (applies to all classes)

Every chat-facing string should feel god-tier, not functional: timeline cards
with countdowns, budget progress bars, CPP-starred award options, verdict
banners ("🟢 BUY NOW — 23% below 30-day low"), settle-up receipts. Three
**output themes** (rich/compact/minimal) in `display.py`; rich is the default
for chat.

## Implementation order
1. `watchers.py` — price_history table + forecast() + richer alerts +
   rise/cooldown triggers + format_watches.
2. `itinerary.py` — airline map, merge_trips, timeline, conflicts, export_ics,
   packing_list, status.
3. `display.py` — cpp(), rate_redemption(), best_redemption(), transferable
   bank map, themes.
4. `whitelabel.py` — analytics, template registry, escalation, onboarding,
   channels/languages, knowledge listing.
5. `groups.py` — split modes, record_payment, poll reminders, dates/destination,
   categories, member_totals, overview, leave guard.
6. `tests/test_travel_sweep.py` — real tests for all changed behavior.
7. Commit as `sweep(travel): …`, push `two main:main` (rebase-retry, no stash).
