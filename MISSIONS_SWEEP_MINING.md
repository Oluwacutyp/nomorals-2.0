# MISSIONS SWEEP — External Mining Report

Module: `nomorals/missions/` (7 files). Mined 2026-10-10 against the best
implementations **outside** this repo: durable-execution engines (Temporal,
Inngest, DBOS, Restate, LangGraph), the Stripe idempotency contract, retry
literature (AWS Architecture Blog "Exponential Backoff And Jitter", tenacity
practice), golden-test discipline, ETA/rate-smoothing practice, saga
compensation, and terminal-UX practice (tqdm/rich). Best AND trash mined;
only the gold is taken.

How each significant class compares, and what it is missing.

---

## 1. MissionRunner — durable plan → execute → checkpoint → resume

**Best in class: Temporal (workflows + activities), Inngest, DBOS, LangGraph.**

- Temporal's articulation is the clearest: a **durable plane** (workflow
  state, step history, signals — persisted) vs an **ephemeral plane**
  (workers/activities — stateless, freely replaceable). A crash costs only
  the in-flight step. Our runner already implements this split (SQLite row =
  durable plane, agent thread = ephemeral). ✅
- **Inngest's model** is step-memoization with **no determinism requirement**:
  each `step.run()` executes once, the result is persisted, completed steps
  are skipped on retry — matched by step-name hash. This is exactly our
  idempotency-key design (mission_step hash). ✅
- **Timeout taxonomy (Temporal activities) — GAP.** Temporal defines FOUR
  timeouts: `ScheduleToClose` (end-to-end incl. retries), `StartToClose`
  (single attempt), `ScheduleToStart` (queue wait — detects dead workers),
  `Heartbeat` (long-running liveness). We have only one: per-attempt
  `timeout_s`. Missing: a step-level end-to-end cap and queue-wait timeout.
- **Heartbeat carries progress payload (Temporal)** — GAP. Our heartbeat is
  `{pid, at}` only; a long step's *progress* (e.g. "3/10 files processed")
  is invisible to the watchdog and to chat.
- **Signals / human-in-the-loop (Temporal signals, FlowMaestro
  user-input-workflow)** — GAP. We have a `blocked_on_approval` stall code
  with **no producer**: nothing in the runner can actually pause for owner
  approval and resume on signal. The best implementations treat approval as
  a first-class step type, not an ad-hoc stall note.
- **Saga compensation (orchestrated saga)** — GAP. When a multi-step
  transaction fails partway, the best practice is not "stop and escalate"
  but *forward compensation*: run each completed step's compensating action
  in reverse order (refund, release, cancel). Rules mined: compensation must
  be idempotent; design for forward recovery (retry) before rollback;
  orchestrator runs compensations in reverse. We have replan-on-failure but
  no undo path — a failed mission leaves side effects dangling.
- **Dry-run / plan preview (Inngest dev server, Temporal replay testing)**
  — GAP. No way to see the plan, its dependency levels, and policy
  warnings without executing.
- **Deterministic replay testing (Temporal `WorkflowReplayer`,
  autumn-harvest's replay harness)** — partial. Golden missions cover the
  machinery but there is no "replay recorded history against new code"
  check; out of scope for this sweep, noted.
- **Versioning running workflows (`workflow.GetVersion`)** — noted, out of
  scope.

**Taken for implementation:** per-step `schedule_timeout_s` (queue wait)
policy, heartbeat progress payload + `heartbeat_progress()` API, approval-gate
steps (`needs_approval` policy → pause → `approve()` resumes), orchestrated
saga compensation (`compensate` per step, reverse-order, idempotent,
ledgered), `dry_run()` plan preview with policy warnings, `render_plan_text`
ASCII dependency tree.

---

## 2. IdempotencyStore / dedupe — exactly-once intent

**Best in class: Stripe's idempotency contract (the industry reference,
battle-tested since 2015), IETF Idempotency-Key draft.**

- Stripe's server path: load key row → missing: insert `started`, execute →
  `finished`: replay cached response → **in-flight: 409 Conflict** →
  **same key, different payload: 400/422 (client bug)**. We collapse
  in-flight duplicates onto one execution (better than 409 for our use) but
  have **no payload-fingerprint check**: a reused key with different step
  content silently replays the old outcome. GAP.
- **Recovery points (Stripe's `rocket-rides-atomic`):** the execution is a
  state machine (`started → ride_created → charge_created → finished`); a
  crash resumes from the last committed recovery point, not from scratch.
  Our steps are atomic (all-or-nothing) — intra-step recovery points are a
  natural next step, but for agent steps the honest unit is the step itself;
  noted, not taken.
- **Retention TTL:** Stripe retains keys 24h, then the key is a new request.
  We have `purge(older_than)` (manual) but no TTL on read and no per-key
  expiry. GAP.
- **Replay tooling (webhook-retry literature):** "replay everything that
  failed since 14:00" — operators need to list failed keys and redrive them.
  We have `purge` but no `failed_since()` / bulk-retry query. GAP.
- **Single-flight pattern (in-memory guard coalescing duplicates)** — we
  already do this with `_IN_FLIGHT` + condition variables. ✅
- **Keyed by (user, key):** per-account scoping. Our scope param covers it. ✅

**Taken:** optional `fingerprint` on `dedupe()` — mismatch on a completed
key raises `IdempotencyConflict` (fail fast on the client bug); `ttl_seconds`
per key with expiry honored on read; `ALTER TABLE ADD COLUMN` for
`expires_at`/`request_json` (additive, guarded, no migration); `failed_since()`
+ `retry_failed()` operator redrive; `peek()` non-claiming status read;
`meta` kind labels + `stats()` grouped by kind.

---

## 3. Retry/backoff policy (`_step_policy`, `_retryable`, `_backoff_delay`)

**Best in class: AWS Architecture Blog "Exponential Backoff And Jitter"
(full jitter: `sleep = random(0, min(cap, base * 2^attempt))`), tenacity
practice, fatsecret/maestro production policies.**

- Full jitter: we implement full jitter ✅ (`random.uniform(0.5, 1.5)` —
  close; the canonical form is `random_between(0, min(cap, base*2^n))`).
- **Honor `Retry-After`:** production SDKs parse the server's `Retry-After`
  (or "retry after Ns" text) and use it instead of computed backoff. GAP —
  we ignore it.
- **Adaptive classification:** error classes `transient / rate_limit /
  server_error / permanent`; rate-limit errors get a **3x delay multiplier**
  and (in fatsecret) an extra attempt. We have a flat transient list; 429 is
  just another marker. GAP.
- **Circuit breaker (pybreaker readiness):** repeated failures to the same
  provider should open a breaker instead of burning retry budget per step.
  Noted; our stall-after-3-consecutive is the coarse version. Taken as a
  small step: per-step consecutive-transient tracking feeds the stall
  message with a breaker-flavored hint.
- **Retry metadata surfaced (maestro's `RunResult.retry_count`):** we store
  `step_attempts` ✅ but never show it in chat status. Taken: surface in
  `detail()`/`render_status_text` (already in `attempts`; make it visible).

**Taken:** `Retry-After`-aware backoff (parse seconds from detail text),
rate-limit error class with 3x multiplier, canonical full-jitter formula
`random(0, min(cap, base*2^n))`, `_classify_error()` returning
`transient|rate_limit|server_error|permanent` used by `_retryable` and the
stall message.

---

## 4. progress.py — ETA, status rendering, milestones, watchers

**Best in class: tqdm (`[elapsed<remaining, rate]`), gmailarchiver's
`ProgressTracker` (EMA α=0.3 rate smoothing, perf_counter, zero-time edge
cases), rich dashboards (tables, panels, live), notification-digest
practice.**

- **Rate smoothing:** we use a trailing window of the last 5 step durations
  (simple mean). Mined practice is an **exponential moving average
  (α=0.3)** — reacts to recent changes without spike whiplash. Taken: EMA
  over the duration series, blended with historical similar-mission timing
  when available (reflections table holds per-mission step data).
- **Historical priors:** nobody in our codebase uses past missions to seed
  ETA. Taken: `estimate_eta` consults the average per-step wall time of
  recently completed missions as a prior when the current mission has no
  timing yet (marked "based on history" so it stays honest).
- **Presentation — GAP, the big one.** `render_status_text` is functional
  plain text. tqdm/rich show the bar: `[██████░░░░] 62%`, rate, ETA in one
  glance; dashboards use tables + panels. Taken: `render_progress_bar()`
  (unicode blocks, width-aware, no-ansi mode), `render_status_card()`
  (boxed card: header, progress bar, budget bar, ETA, stall, attempts),
  `render_status_text(..., style="compact"|"full"|"plain"|"card")` themes,
  `render_mission_table()` for `/mission list`, `render_sparkline()` for
  per-step duration shape. All pure functions, all tested.
- **Digest notifications:** milestone pushes are per-step (cooldown-gated).
  Best practice for long missions is a digest mode (every N steps or
  terminal-only). Taken: `notify_level` on MissionMilestones
  (`all|milestones|terminal`) + `digest()` collapsing step updates.
- **Quiet-hours hold** ✅ already best-in-class; kept.

---

## 5. mission.py — Mission model + MissionStore

**Best in class: Temporal search attributes / memo, Hatchet labels,
Prefect task metadata, DBOS workflow metadata.**

- **Priority + tags (search attributes):** every serious orchestrator lets
  operators prioritize and label runs. We have neither; `list()` filters
  only by status. Taken: `priority` (int, default 0) and `tags` in
  `metadata` (zero-migration, JSON column), exposed as properties,
  `list()` gains `tag`, `min_priority`, `search` filters, priority-aware
  ordering.
- **Parent/child missions (Temporal child workflows, Inngest fan-out):**
  complex goals decompose into sub-missions, but we have no hierarchy —
  `resumable()` is flat. Taken: `parent_id` + `spawn_child()` /
  `children()` / `descendants()` / `tree()`; child failure policy
  (`child_policy: fail_fast|continue`) honored by the parent's settle.
- **Templates:** golden missions ARE the template shapes
  (research→write→verify, build→test→fix, audit→remediate→rescan). Taken:
  `MISSION_TEMPLATES` registry + `create_from_template()` wiring a mission
  with a canned plan skeleton and policies.
- **Retention/archival (Temporal retention, DBOS):** terminal missions
  accumulate forever; `stats()` counts them. Taken: `archive(older_than)`
  bulk-deletes terminal missions + checkpoints + reflections with counts.
- **Bulk ops:** `bulk_set_status`, `bulk_cancel`. Taken.
- **Status card rendering:** `Mission.summary_card()` one-glance box.
  Taken (style).

---

## 6. golden.py — deterministic plan→execute→verify→repair drills

**Best in class: golden-set discipline (version-controlled fixtures,
deterministic assertions before any judge, CI gates on pass-rate
*regression* not perfection), fixture normalization (volatile fields
dropped before comparison).**

- Our golden missions are deterministic ✅ (plain Python steps, fixed
  fixtures, short/long sizes) and comparable over time via BenchmarkDB ✅.
- **GAP: only 3 drills; none exercise the new machinery** (idempotency
  crash-kill, saga compensation, approval gates). Taken: 2 new drills —
  `crash_no_dup` (kill mid-step → resume proves no duplicate side effects
  via idempotency keys) and `saga_undo` (step fails → compensations run in
  reverse and undo files).
- **GAP: regression gate.** Benchmarks are recorded but never compared —
  nothing fails when a drill gets *worse*. Taken: `check_regression(key)`
  comparing the latest run against the committed baseline
  (`GOLDEN_BASELINES`), failing on pass-rate regression.
- **GAP: output normalization.** Step outputs embed volatile data
  (timestamps, pids). Taken: `normalize_output()` dropping volatile keys
  before golden comparison.
- **GAP: report rendering.** Taken: `render_golden_report()` card.

---

## 7. wiring.py + __init__.py

- `wired_runner` is the sanctioned L5→L6 bridge ✅ (lazy dynamic import,
  documented). Taken: `preview_mission()` helper — build runner, plan,
  render, discard; the dry-run UX for chat (`/mission preview`).
- `__init__` exports grow with the new surface.

---

## What was deliberately NOT taken

- **Deterministic replay testing** (Temporal WorkflowReplayer): needs
  recorded event histories; our golden drills + checkpoints cover the
  practical surface. Noted for later.
- **Intra-step recovery points** (Stripe atomic phases): the honest unit
  for agent steps is the step; finer granularity would complicate the
  at-least-once contract without buying much.
- **Workflow versioning** (`GetVersion`): no deployed-version skew problem
  in a single-process personal agent.
- **Full circuit breaker (pybreaker):** the stall-after-consecutive +
  budget machinery is the proportionate version at this scale.
- **Rich-library rendering:** terminal chat is plain-text; the card style
  is hand-rolled unicode (no new dependency — standing order: zero
  mandatory deps, stdlib-first).

---

## Implementation checklist

1. `mission.py`: priority/tags, parent/child, templates, archive, bulk ops,
   search, summary card.
2. `runner.py`: approval gates, saga compensation, dry-run preview,
   schedule_timeout_s, heartbeat progress payload, Retry-After-aware
   backoff, error classification, plan rendering.
3. `progress.py`: EMA ETA + historical priors, progress bar / card / table /
   sparkline rendering, style themes, digest notify level.
4. `idempotency.py`: fingerprint conflicts, TTL, failed_since/redrive,
   peek, kind-labeled stats.
5. `golden.py`: 2 new drills, regression gate, output normalization,
   report card.
6. `wiring.py`: `preview_mission`.
7. `tests/test_missions_sweep.py`: real tests for all changed behavior.
8. Commit `sweep(missions): ...`, push `two main:main` (rebase-retry, no
   stash).
