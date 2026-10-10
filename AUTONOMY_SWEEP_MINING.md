# AUTONOMY Sweep — External Mining Report

Module: `nomorals/autonomy/` (6 files: `__init__.py`, `coordinator.py`,
`idle.py`, `weakness.py`, `patterns.py`, `presence.py`).
Mined 2026-10-10. Method: real web search per significant class, best AND
trash implementations compared, then design the upgrade from the gold.

---

## 1. `idle.py` — idle detection

### Best outside the repo

- **WICG Idle Detection API** (https://github.com/WICG/idle-detection):
  two orthogonal states — *user idle/active* × *screen locked/unlocked*,
  configurable threshold, fully **reactive** (`change` event), permission-gated.
  Lesson: idle is not binary — it's a state pair, and listeners react to
  transitions, not polls.
- **glimpse `research/idle.md`**
  (https://github.com/alex-oleshkevich/glimpse/blob/HEAD/research/idle.md):
  multiple idle thresholds (**dim → lock → suspend** graduated states),
  **idle inhibitors** (acquire/release handles — a video player can block
  idle), inhibitor listing. Lesson: graduated idle stages + an inhibitor
  API so long-running organs can *veto* maintenance during heavy work.
- **cpuidle governors** (cnx-software.com, 2026): idle states annotated
  with *target residency* — the minimum dwell time for a state to pay off;
  waking too early wastes energy, too late wastes opportunity. Lesson:
  each idle stage should have a minimum residency before deeper work
  starts (don't run the 10-minute job 30s into a 35s idle window).
- Browser idle skills (coastai-skills): never count time with decrementing
  counters; store an **absolute expiration timestamp** and evaluate against
  it per heartbeat — immune to thread stalls and background throttling.

### Trash / cautionary

- Naive JS polyfills of `requestIdleCallback` with `setTimeout` —
  a timer is *not* idle detection; gating real work on it means "might run
  at the worst possible moment." Ours is DB-timestamp-based (absolute),
  which is the correct shape.
- iDRAC-style server idle (CPU < 20% for 240h) — wrong domain; measuring
  machine load tells you nothing about *user* engagement.

### What weakness.py's class should get from this

1. **Idle stages**: `active → shallow → deep → night` — deeper stages
   unlock heavier maintenance (deep = research ticks, night = full
   corpus/wisdom ingestion). Each stage has a minimum residency.
2. **Idle inhibitors**: organs acquire a named inhibitor while doing
   long work; maintenance skips while inhibitors are held.
3. **Adaptive threshold**: learn the owner's typical gap between
   activities (EMA of inter-activity gaps); idle = gap exceeds the
   learned threshold, not a fixed 900s.
4. **Idle session history**: record each idle window's duration; powers
   the adaptive threshold and residency planning.
5. Stage-transition events on the bus (`system.idle.deep`), so organs
   can subscribe to the depth they care about.

---

## 2. `coordinator.py` — the maintenance-cycle conductor

### Best outside the repo

- **SRE-style budget management**: fixed cycle budget (600s) exists here;
  the gold is *proportional allocation* — organs with more pending work
  get more time — plus **graceful degradation ladders**
  (Zylos Research 2026, via crew-research-council):
  FULL → SIMPLIFIED → CACHED → MINIMAL → REJECT. When a tick blows its
  budget, degrade the cycle (skip optional ticks) instead of overrunning.
- **Backoff on repeated failure**: every mature scheduler (systemd,
  Kubernetes) backs off a failing unit instead of hammering it every
  cycle. A tick that fails N cycles in a row gets skipped with an
  exponential backoff.
- **Cycle SLAs / health ledger**: Prometheus-style "alerts are for
  machines" — the coordinator should produce a machine-readable health
  report per organ (last ok, avg seconds, consecutive failures) so other
  systems can react (e.g. presence notes "research tick failing").

### Trash

- Cron-shaped fixed pipelines (run A, then B, then C, always) — blind to
  pending-work volume. Ours already varies work inside ticks; the missing
  piece is *budget-aware ordering*.

### Gaps to fill

1. Per-organ budget slices scaled by pending queue depth.
2. Priority ordering: weakness/research first (they feed everything else).
3. Failure backoff per organ (skip a consistently failing tick, note it).
4. `coordinator_status()` — machine-readable organ health.
5. **Formatted cycle report** (`cycle_report()`): the "how she LOOKS"
   requirement — a human-readable, themeable summary of each cycle the
   owner can actually read in chat.

---

## 3. `weakness.py` — the self-healing loop

### Best outside the repo

- **Microsoft 2025 — six agent failure categories**
  (via faresrafat3/crew-research-council self-healing.md):
  tool misuse, context loss, goal drift, retry loops, cascading errors,
  silent quality degradation. Lesson: our four kinds are too shallow —
  *retry_loop* and *cascading_failure* are first-class agent diseases we
  don't detect at all.
- **Regenesis / Actian self-healing loop**
  (https://www.actian.com/blog/developer/how-regenesis-built-medshifts-self-healing-loop-with-vectorai-db/):
  three transferable patterns — (a) index failures by **structure, not
  vocabulary** (same root cause through different entry points must match);
  (b) filter memory at **recall**, not at write; (c) **never store a fix
  until an independent verifier confirms the repair held**. Lesson: our
  clustering is exact-string on subject; we need fuzzy *signature*
  clustering (exception type + location + normalized message), and a
  "fixed weaknesses memory" that stops re-opening what was verified.
- **SRE multi-window multi-burn-rate alerting**
  (Google SRE Workbook lineage):
  fast burn (14.4× over 1h+5m) → page; slow burn (6× over 6h+30m) →
  ticket; sustained 1× → review. Static thresholds are either noisy or
  blind. Lesson: weakness detection should score *burn rate*
  (fast/acute vs slow/chronic) per weakness, not just a raw count ≥ 3.
- **LogCluster** (Lin et al., evaluated on Microsoft online services,
  via the self-healing survey arxiv.org/pdf/2403.00455):
  convert rejection sequences to weighted vectors, measure sequence
  similarity, hierarchical-cluster them, pick a representative per
  cluster. Lesson: clustering errors with a similarity measure beats
  exact-match grouping — directly applicable to `error_pattern`.

### Trash

- Self-healing test-automation tools (accelq): element-fingerprint
  substitution — domain-specific (UI locators), not transferable beyond
  the "fingerprint + verify" shape, which we already have.

### Gaps to fill

1. New weakness kinds: `retry_loop`, `cascading_failure`, `silent_degradation`.
2. **Fuzzy signature clustering**: normalize errors → signatures
   (`exc_type@file:line` + normalized message), cluster sightings by
   signature instead of raw subject strings.
3. **Burn-rate severity**: each case gets fast/slow burn score from
   short vs long window rates; priority = severity, not raw count.
4. **Fixed-weakness memory**: resolved cases store their fix; a new
   matching signature first checks the memory (one-step resolve, like
   Regenesis) instead of re-diagnosing.
5. **Acceleration detection**: failure rate rising week-over-week →
   escalation flag.
6. **Stale-case expiry**: bounded growth — auto-dismiss cases with no
   new evidence for N days (currently the table grows forever).
7. **Owner-facing digest** (`weakness_digest()`): themeable report of
   open/proposed cases — the "how she LOOKS" requirement.

---

## 4. `patterns.py` — routine / interest / sequence modeling

### Best outside the repo

- **YAKE** (https://pypi.org/project/yake/0.4.8): lightweight,
  *unsupervised, corpus-independent, single-document* keyword extraction
  on statistical text features (casing, position, frequency, relatedness
  to context, sentence spread) — beats TF-IDF/RAKE/TextRank on 20
  datasets. Lesson: our crude unigram/bigram counter is the weakest link;
  a YAKE-style statistical scorer is implementable dependency-free and
  works on single short chat messages (the "short-text problem" RAKE
  people solve badly).
- **TextRank / PositionRank** (DerwenAI/pytextrank): graph-based phrase
  ranking. Heavier; YAKE's feature approach is the better fit for us.
- **Prophet** (arxiv 1909.00045): models cyclical human activity,
  *adapts gradually to habit change* and tolerates missing data.
  Lesson: our routine histogram is static — it never decays, so a
  changed routine fights the old one forever. Routine cells need
  time-decay (like interests already have).
- **Learning Automata** (arxiv 1608.03507): learns *transition
  probabilities between actions online* — each launch updates the
  transition matrix. This is the mathematically honest version of our
  `likely_next` counter: a proper Markov transition model with
  smoothing, confidence from sample count.
- **CalBehav** (arxiv 1909.04724): personalized, *dominant-behavior*
  identification from calendar+phone logs — behavior during scheduled
  events, not just clock time. Lesson: routines should key on *context*
  (calendar events, active triggers) too, not only hour×dow.
- **ProactiveMobile benchmark** (arxiv 2602.21858v3): proactive agents
  must predict from four dimensions — user profile, device status, world
  info, behavioral trajectories. Lesson: presence context should carry
  all four, not just time+interests.

### Trash

- Generic "AI predicts everything" marketing pieces
  (appperformancelab, sartimsolutions) — no method, no code. Ignored.
- Benchmarks of 7 extraction libs that conclude "validate manually" —
  true but not actionable beyond what we do.

### Gaps to fill

1. **YAKE-style statistical topic scoring** (dependency-free): casing,
   position-in-text, frequency, bigram cohesion — replaces the naive
   counter in `extract_topics`.
2. **Routine decay**: routine cells get time-decay like interests, so
   habits that changed stop haunting predictions.
3. **Transition model with confidence**: `likely_next` gets Laplace
   smoothing + confidence (count + recency-weighted), plus *anti-edges*
   (what reliably does NOT follow a trigger).
4. **Interest co-occurrence**: topics seen together form clusters —
   surfaces become "you're into X+Y lately" instead of single tokens.
5. **Next-activity prediction with confidence intervals**: not just
   top-N windows, but "high confidence you'll be active around 7–8pm".
6. **Snapshot export/import** for the models (portability, debugging).

---

## 5. `presence.py` — the "alive" layer

### Best outside the repo

- **InterruptMe** (Pejovic et al., via arxiv 1711.10171): ML
  interruptibility prediction from context (location, activity,
  engagement), **online learning → stable predictions within a week**,
  ~60% accuracy. Lesson: don't hardcode a 6h cooldown — learn an
  *interruptibility score* from context (daypart, activity recency,
  past surface reactions) and only surface when the score is high.
- **Bounded deferral** (Horvitz, via the same survey): non-urgent
  messages wait for an *opportune moment*, bounded by a max delay.
  Lesson: candidates shouldn't fire immediately or be dropped — they
  queue with a deadline, and the heartbeat delivers when
  interruptibility peaks. That's the batched-digest shape.
- **Serendipity = unexpectedness × relevance** (Ge, Delgado-Battenfeld
  & Jannach, RecSys 2010; soupnet's 2026 research notes):
  unexpectedness measured against a *primitive baseline* (what the user
  would find on their own). Lesson: our surfaces have no surprise
  scoring — "you're deep into X lately" is a tautology, not serendipity.
  Score candidates: novel (not in recent history) × relevant (ties to
  a real interest) × unexpected (not the obvious next thing).
- **ChatGPT Pulse** (securityonline.info, 2025): proactive *curated
  digests* from conversations + interests + calendar — the product
  shape users actually accept: **one batched briefing, not scattered
  pings**. Lesson: presence should compile a *digest* (one message,
  batched candidates) rather than single-shot surfaces.
- **Proactive voice assistants hierarchy** (arxiv 2005.01322):
  L1 push-like, L2 scheduled/contextual, L3 **batched daily routines** —
  messages hierarchically stacked by importance. Lesson: surface
  *priority tiers* — urgent (breaks cooldown), normal (batched),
  background (digest only).
- **Quiet delivery**: the community ambient-assistant thread
  (community.openai.com) demands granular opt-in and *quiet hours* —
  learned from routine (our `predict_active_windows` already knows when
  she sleeps), enforced in code, not prompts.

### Trash

- Brain-sensing "Phylter" headband (sciencealert): real lab work, but
  requires an fNIRS headband — not applicable. The *idea* (sense before
  interrupting) transfers; the hardware doesn't.
- mzarras/ambient: always-listening audio transcription — privacy
  model (100% local) is admirable but the feature (fact detection from
  ambient audio) is out of scope for a chat agent.

### Gaps to fill

1. **Interruptibility score**: daypart weight × activity recency ×
   learned quiet hours × past reaction feedback — replaces the flat
   6h cooldown as the gate.
2. **Quiet hours learned from routine**: never surface at 3am just
   because 6h elapsed; learn sleep from `predict_active_windows`.
3. **Candidate queue with bounded deferral**: surfaces queue with a
   deadline; heartbeat delivers the batch at high interruptibility.
4. **Serendipity scoring**: novelty × relevance × unexpectedness per
   candidate; only top-scoring candidate per batch.
5. **Surface feedback loop**: owner engaged / dismissed / ignored →
   tunes future surfacing per kind (learned weights).
6. **Presence digest** (`digest()`): one batched briefing of queued
   candidates — themeable, human-readable.
7. **Daypart-aware tone**: morning = brief & useful; evening = warmer;
   night = only urgent. (Presentation god-tier.)

---

## 6. `coordinator.py` (conductor) — cross-cutting style requirement

The sweep demands god-tier *presentation*. The coordinator's cycle
stats are currently log-only. Add `cycle_report(theme=...)`: a rendered
cycle summary (organs, seconds, outcomes, health flags) the owner can
read — this is the visible "alive" proof of the nervous system.

---

## Implementation plan (what actually ships)

| File | Additions (extend, never delete) |
|---|---|
| `idle.py` | `IDLE_STAGES` (shallow/deep/night) + stage detection w/ residency; `inhibit_idle`/`release_idle` context API + `inhibitors()`; adaptive threshold via EMA of inter-activity gaps (`learned_threshold`); idle session history table + `idle_history()`; stage-transition bus events |
| `coordinator.py` | per-organ budget slices scaled by pending depth (`_budget_plan`); priority order; per-organ failure backoff (`_backoff`); `coordinator_status()` machine-readable health; `cycle_report(theme)` formatted summary |
| `weakness.py` | new kinds (`retry_loop`, `cascading_failure`, `silent_degradation`); `error_signature()` fuzzy normalization + signature-based clustering in `report_weakness`; burn-rate severity (`severity()` fast/slow); fixed-weakness memory (`weakness_memory` table, `recall_fix`); `acceleration()` trend; stale-case expiry (`expire_stale`); `weakness_digest(theme)` |
| `patterns.py` | YAKE-style `extract_topics` v2 (casing/position/frequency/cohesion, no deps); routine cell time-decay + `routine_confidence`; smoothed transition model w/ confidence + anti-edges in `likely_next`; interest co-occurrence `topic_clusters()`; `next_activity()` w/ confidence; `export_models`/`import_models` |
| `presence.py` | interruptibility score (`interruptibility()`); learned quiet hours (`quiet_hours()`); candidate queue w/ bounded deferral (`queue_candidate`, `due_candidates`); serendipity scoring (`serendipity_score`); feedback loop (`record_surface_feedback`, learned weights); priority tiers; `digest(theme)` batched briefing; daypart-aware tone |

All guarded: existing public API keeps working; new tables are
`CREATE TABLE IF NOT EXISTS`; every new path fail-open (never raises
out of the nervous system).
