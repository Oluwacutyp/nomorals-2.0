# Autonomy Mining Report — Phase 9 Slice B

*Date: 2026-10-10. Every autonomy implementation in the repo was read,
traced end-to-end, and checked for who actually calls what.*

---

## 1. What exists (internal mining)

### 1.1 `nomorals/autonomy/idle.py` — IdleMonitor
**Does well:** persisted single-row activity clock in SQLite, cheap
`note_activity()` hook (wired into the partner runtime's message pump and
boot), `system.idle` / `system.active` bus events, background thread with
clean stop, restart-safe (idle state in DB, not RAM).

**Provably broken:** the monitor reads a *different database* than the
writers use. `IdleMonitor._db()` opens `workspace/autonomy.db` via raw
sqlite3, but every `note_activity()` caller passes the main workspace
`Database` — the `autonomy_activity` table is created and updated in the
*main* DB file, which the monitor never opens. Result: `last_activity_ts`
is always 0 in the monitor's file, `is_idle` is always False (requires
`last > 0`), the monitor emits **nothing ever** — the entire idle chain is
dead in production. Fix: the monitor must read the same DB the writers
write to (open `Database(workspace_dir)`).

### 1.2 `nomorals/autonomy/coordinator.py` — IdleCoordinator
**Does well:** subscribes `system.idle`/`system.active`, background thread
per cycle, cycle budget (10 min), graceful stand-down when activity resumes,
per-organ drain loop, exception containment per event.

**Provably broken:** double-drain event loss. `_drain_organ()` drains the
organ's event queue via `organs.drain()` (which marks events *consumed* in
the same transaction), then `_handle_research_event()` constructs
`ResearchOrgan(db)` and calls `tick()` — whose first action,
`_drain_directives()`, drains the same queue *again* (now empty). Every
organ event the coordinator "routes" is **consumed and dropped** without
being processed. The coordinator must not pre-drain: tick each organ once
per cycle and let the organ drain its own queue.

**Missing (docstring promises, code doesn't deliver):**
- Memory consolidation tick (documented step 3) — no code calls it. A
  `MemoryManager.consolidate()` API exists in `nomorals/memory/manager.py`.
- No journaling: the idle cycle's work is invisible — nothing is written to
  the autonomy ledger.
- No completion signal: no `system.idle_cycle` event for downstream systems.

### 1.3 `nomorals/autonomy/weakness.py` — weakness detection
**Does well:** clustering by `(kind, subject)`, threshold-gated escalation
(3 occurrences in 24h), status lifecycle
(open→researching→proposed→approved→resolved/dismissed), evidence as JSON,
`scan_tool_failures()` consumer, and the one live writer in the repo:
`nomorals/agents/partner/tool_loop.py:401` calls `report_weakness()` on
repeated tool failures. Owner approval is required before any fix — the
safety line is correctly placed (detect/research/propose automatic,
build/merge needs go-ahead).

**Provably broken:** the research leg is a dead end. `report_weakness()`
emits organ event `weakness.investigate` to `research`, but
`ResearchOrgan._drain_directives()` only handles `directive.watch` and
`directive.gap` — the `weakness.investigate` kind is consumed and ignored.
Fix (no research-file edits needed): the weakness module translates its own
event into a `directive.gap` the research organ already understands.

**Missing:**
- No path from research findings → `record_proposal()` (nothing ever calls
  it; proposals can only exist if the owner writes them by hand).
- No recovery detection: a `tool_failure` case stays open forever even
  after the tool starts succeeding — no auto-resolution, no
  failure→success streak tracking.
- `approve_weakness()` ends the chain: no sandbox-build mission is launched,
  no scheduler job is queued, no notification goes out. The "weakness →
  improvement mission → scheduler" chain the user demanded doesn't exist.
- No ledger writes for the lifecycle.

### 1.4 `nomorals/autonomy/patterns.py` — models
**Does well:** three clean, learned-not-hardcoded models — hour×dow
routine histogram, exponential-decay interest model (14-day half-life),
trigger→action pattern counts — all persisted in SQLite, all pure functions
with injected timestamps (testable).

**Gap:** the PatternModel has **zero writers** — `record_sequence()` and
`likely_next()` are called by nothing in the repo (only tests). Patterns are
learned in theory only. The interest model and routine model at least get
fed by the partner runtime's message pump (`record_interests`,
`record_activity`). The sequence model needs a writer hook in the runtime's
tool-loop completion path (cross-slice — see "couldn't touch").

**Missing:** bounded-growth maintenance (interests/patterns tables grow
unbounded), no decayed-score pruning, `predict_active_windows` requires ≥20
samples (sane), and `sense()` in presence never consumes `likely_next`.

### 1.5 `nomorals/autonomy/presence.py` — presence
**Does well:** honest architecture — presence *senses* and *proposes*, never
sends; actual sends are owned by the autonomy agent with safety vetoes.
`sense()` snapshot is the best "state of the system" API in the repo
(time, windows, interests, rising topics, weaknesses). 30-min durable
scheduler heartbeat, idempotent registration.

**Gaps:**
- `SERENDIPITY_COOLDOWN` (6h) is defined and **never enforced** — the
  "surface" leg doesn't actually surface anything; it only publishes a bus
  event. No audit table, no cooldown state, no delivery path.
- Never consumes the pattern model (`likely_next`) despite the docstring
  claim that models feed presence.
- No ledger writes; heartbeat work is invisible in the ledger.
- Rising-interest → research emits fire on *every* heartbeat with no
  dedup: the same rising topic re-emits every 30 min (organ queue spam).
  Needs "nudged" bookkeeping.

### 1.6 `nomorals/agents/autonomy.py` — AutonomyAgent (partner proactive)
**Does well:** this is the most production-grade piece in the slice.
Strategy-chain decision (SilenceCheckin/AmbientShare/GroupPost scoring),
safety vetoes applied *after* selection (quiet hours, daily caps,
per-chat intervals, feature flags — never negotiable), adaptive threshold
(bolder on success, cautious on failure/denial, bounded [0.3, 0.9]),
off/suggest/auto modes with approval flow, ledger journaling, per-strategy
exception containment. Failures = proactive silence, always. Externally
this matches best-in-class "suggest-before-act" guardrail practice.

**Missing:**
- No learning from owner *responses*: the threshold adapts on send success
  and denials, but not on whether the owner actually *replied* to a
  proactive DM (the strongest positive signal). Needs an `on_owner_reply`
  hook + `note_outcome()` wired to inbound traffic (cross-slice wiring in
  the runtime's message path).
- Per-strategy performance stats exist only implicitly; no per-strategy
  win/loss counters for later tuning.
- No quiet-hours override awareness of "presence" — fine as is.

### 1.7 `nomorals/agents/autonomy_ledger.py` — the ledger
**Does well:** single durable journal for all autonomous systems, best-effort
writes (never breaks the observed system), per-system rollups with cost
accounting, 90-day purge. This is the "keep a record you can replay"
best practice, done right.

**Gaps:**
- `_LEDGER_SYSTEMS` doesn't include the systems in this slice: `idle`,
  `presence`, `weakness`, `improvement` → their writes would collapse into
  "other". Fix: extend the set.
- Writers so far: scheduler, cognition, pulse, autonomy, (motion_studio's
  unrelated `record_ledger`). Idle coordinator, presence heartbeat, and the
  weakness lifecycle write nothing — the busiest background systems are
  invisible in the ledger.
- No failure-rate query helper that other systems (weakness detection)
  could consume.

### 1.8 Related but not owned (read freely)
- `nomorals/organs.py` — organ event store: `emit`/`drain`, consumed-marked
  in same transaction. Sound; the bug was in how the coordinator used it.
- `nomorals/research/autonomy.py` `ResearchOrgan` — tick drains directives
  (only `directive.watch` / `directive.gap`), runs due watches, works gaps.
  Has `add_watch` — presence's rising-interest nudge could create real
  watches instead of raw event spam.
- `nomorals/wisdom/autonomy.py` `WisdomOrgan` — ingestion loop with budget;
  coordinator can tick it once per cycle (no per-event dispatch needed).
- `nomorals/agents/improvement.py` `ImprovementLoop.tick()` — closed-loop
  benchmark→lever→evolve→gate→verify cycle, gated by
  `settings.improvement.mode`. Already exists; the missing piece is the
  *trigger*: nothing autonomously starts it from weakness findings.
- `nomorals/agents/partner/runtime.py` — already wires idle monitor,
  coordinator, heartbeat job, activity hooks, and pattern feeding. My fixes
  must keep these call sites working (same function names/signatures).

---

## 2. External research (best-in-class)

Sources: dev.to agent-loop deep-dives, ManageEngine agent monitoring best
practices, techtarget guardrails, bioengineer.org self-improvement safety
survey, dev.to zero-trust guardrail write-up.

Consensus patterns to adopt:
1. **Observe→Decide→Act→Check→Repeat**, and each stage must be
   independently inspectable/loggable ("the plan step picked the wrong tool"
   is debuggable; "the agent didn't work" is not). → the ledger + sense().
2. **Tier by cost of failure**: cheap reversible work runs autonomously;
   irreversible/high-blast-radius work needs human approval. → weakness
   detect/research/propose automatic; sandbox build + merge gated on owner
   approval. Already the design; keep it.
3. **Suggest-before-act, then earn autonomy**: proposals first, execution
   after the owner's approval history validates judgment. → autonomy agent
   modes; weakness proposals.
4. **Guardrails in deterministic code, not prompts**: quiet hours, caps,
   intervals, cooldowns enforced by code the model can't negotiate. → keep
   veto-after-selection architecture; add enforced serendipity cooldown.
5. **Learning from outcomes**: successes make the system bolder, failures
   more cautious, bounded. → adaptive threshold; add owner-reply feedback.
6. **Claim guards / idempotency**: one tick must not double-execute. → fix
   the double-drain; add cycle claim keys.
7. **Self-improvement safety paradox** (bioengineer.org survey): agents that
   improve themselves face the core risk of mis-specifying their own
   objectives. Mitigation: benchmark-gated evolution (ImprovementLoop
   already does this — tests + benchmark regression check, revert on
   regression), and *never* silent self-rewrite (owner approval line).

---

## 3. What "god-tier" means for this slice

The user mandate: "The system as a whole should be able to work together
without me tripping commands every now and then."

1. **Idle time becomes productive and *visible*.** Monitor works (DB fix),
   coordinator ticks each organ once per cycle (no event loss), memory
   consolidates, the cycle is journaled, completion events fire.
2. **Weakness → research → proposal → mission → scheduler is a real chain.**
   Detection clusters; research investigates (via `directive.gap`
   translation); proposals are recorded; owner approval launches a sandbox
   improvement mission through the scheduler; resolution is automatic when
   failures stop.
3. **Presence is honest and non-spammy.** Cooldown enforced in code,
   surfacing is audited, rising-interest nudges are deduped, and the pattern
   model feeds anticipation.
4. **The partner agent learns from outcomes**, not just sends/denials.
5. **Every autonomous system writes to the ledger** — "what ran, why, what
   it cost, what it learned" answerable in one place.

Safety line (unchanged): detection, research, proposals, and all
*reversible* background work are automatic. Sandbox builds and merges, and
any outbound send, still require the owner's approval or the configured
mode's standing authorization. Nothing rewrites production code silently.
