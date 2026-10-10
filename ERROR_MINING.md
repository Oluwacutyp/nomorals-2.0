# Error-Handling Mining Report

Researched before building the God-tier error catcher upgrade.
Surveyed: Erlang/OTP, Google SRE, self-healing agent literature, graceful-degradation ADRs, and the repo's own primitives.

## What's already in the repo (do not rebuild)

| Module | What it does | Gap |
|---|---|---|
| `nomorals/core/retry.py` | BackoffPolicy (full jitter), retry_call, CircuitBreaker (3-state), BreakerRegistry | Solid. Not wired to error intelligence. |
| `nomorals/core/error_doctor.py` | AST-level diagnosis (UnboundLocalError branch reconstruction, AttributeError difflib, ModuleNotFoundError pip hints) | Never raises by design. Not consulted by error_intelligence. |
| `nomorals/core/error_intelligence.py` | Classification, knowledge-base patterns, user messages | **Purely analytical. Zero recovery. No memory. No budgets.** |
| `nomorals/core/errors.py` | NoMoralsError with `retryable` flags, `classify()` | Good taxonomy. Underused. |

## Mined patterns (gold taken from each)

### 1. Erlang/OTP supervision trees — "let it crash"
- Supervisors watch workers; on crash, restart per strategy: `one_for_one` (just the child), `one_for_all` (all siblings), `rest_for_one` (child + later siblings).
- Gold: **fault isolation + structured restart beats defensive code everywhere.** Don't try to prevent every failure; contain and restart cleanly.
- Applied: per-subsystem supervisors with restart strategies, not one global try/except.

### 2. Self-healing 5-phase loop (SIGNAL → CONTEXT → HYPOTHESIS → EXECUTION → VALIDATION)
- The loop is worthless without VALIDATION: verify the fix worked, else escalate.
- Gold: **every recovery attempt ends with a verification probe.** A "recovered" system that didn't verify is a liar.
- Applied: `RecoveryAttempt` records outcome; verifier confirms health before closing the incident.

### 3. Regenesis/MedShift memory pattern (agentic self-healing)
Three transferable rules:
1. **Index failures by structure, not vocabulary** — hash (exception type, code location, subsystem), not the message string.
2. **Filter memory at recall, not at write** — record everything, rank by relevance when asked.
3. **Never store a fix before an independent verifier confirms it held** — failed fixes must not resurface as solutions.
- Applied: incident signatures are structural; the fix journal only promotes verified recoveries.

### 4. Google SRE multi-burn-rate alerting
- Burn rate = observed error ratio ÷ budget ratio. Alert on *rate of budget consumption*, not absolute errors.
- Canonical thresholds (30d window): page at 14.4× (5m+1h windows), 6× (30m+6h); ticket at 3× (2h+24h), 1× (6h+72h). **Two-window rule**: both short and long windows must exceed — kills flapping.
- Gold: proportional response. Fast burn = page now; slow burn = ticket for later.
- Applied: per-subsystem budgets with fast/slow burn detection, adapted to a single box (in-memory windows, SQLite persistence).

### 5. Degradation ladders (from agent-reliability ADRs)
- Design the ladder *before* the outage: one rung per capability cut.
- rung 0: full experience → rung 1: fallback provider → rung 2: smaller/cheaper → rung 3: retrieval-only/cached → rung 4: honest failure.
- Each rung needs: a **trigger** (error-rate threshold), a **capability trade** (what's lost), an **honesty clause** (user-visible "live data unavailable"), and **auto-recovery** (probe primary, climb back up).
- Gold: **users forgive slower, not silent.** Degradation must be announced. Manual un-degrading is forgotten within a week — automate the climb back.
- Applied: `DegradationLadder` per subsystem with triggers, honesty messages, recovery probes.

### 6. Existing resilience patterns (circuit breaker, retry, bulkhead)
- Retry with exponential backoff + full jitter (already in repo — AWS-recommended against thundering herds).
- Circuit breaker stops 64 agents turning one dead endpoint into a self-inflicted DoS (already in repo).
- Bulkhead: isolate components so one failure can't take the system (partially in repo via capability gating).
- Gold: these are the *mechanics*; the error catcher is the *brain* that decides when to use which.

### 7. Trust-preserving failure UX
- A fallback message must say three things: what happened, what happens next, what the user can do now.
- Never expose internals (confidence scores, classifier names) unless the audience needs them.
- Every fallback state offers a next step — no dead ends.
- Applied: degraded responses carry `degraded: true` + reason + what's still working.

## What the best systems do that ours doesn't

1. **Attempt recovery, not just classification.** Ours explains; theirs fixes.
2. **Remember.** Ours forgets every incident on restart; theirs builds a fix journal.
3. **Verify.** Ours assumes the suggestion works; theirs probes.
4. **Budget.** Ours has no notion of "too many failures"; theirs pages on burn rate.
5. **Degrade honestly.** Ours fails hard or silently; theirs steps down a ladder and says so.
6. **Isolate.** Ours lets one subsystem's failure cascade; theirs bulkheads.

## Build plan (this report → code)

1. `IncidentJournal` — SQLite-backed, structural signatures, fix journal with verifier gate.
2. `SelfHealingExecutor` — analyze → recover (retry/fallback/degrade) → verify → record. Wires retry.py + error_doctor.py + error_intelligence.py together.
3. `RecoveryStrategy` registry — per-category recovery playbooks.
4. `ErrorBudget` — per-subsystem SLO, multi-window burn-rate alerts (fast=page, slow=ticket).
5. `DegradationLadder` — named rungs, triggers, honesty clauses, auto-recovery probes.
6. Supervisor — per-subsystem restart strategies (OTP-inspired, adapted to threads).

Target: t3.small CPU. SQLite + in-memory windows. Stdlib + existing repo modules only.
