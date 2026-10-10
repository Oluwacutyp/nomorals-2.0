# MISSIONS — Mining Report (Phase 9, Slice A)

Surveyed 2026-10-10. Every mission implementation in the repo, what each does
well, what's missing — then outside best-in-class, then the gap list that
drives the Slice A implementation.

## 1. In-repo survey

### 1a. `nomorals/missions/` — the core (THIS slice)

**`mission.py` — state + persistence.**
Does well: `Mission` dataclass with budgets in the row (spent wall/tokens
persisted, so a resumed mission cannot launder its budget); append-only
checkpoints (crash mid-write can't lose both old and new); `MissionStore`
with `reconcile()` (stale-heartbeat + dead-pid ⇒ running→failed with an
explicit reason, never "running" forever); `live_with_lock()` (refuses a
second concurrent mission on the same lock key — the anti-duplicate-side-
effect guard); `resumable()` for boot recovery; stall tracking with concrete
codes; `normalize_acceptance` failing fast at creation.
Missing: no per-step attempt records; `detail()`/`progress()` don't surface
attempt counts; `resume_all` lives on the runner but the store is fine.

**`runner.py` — execution.**
Does well: dependency-aware execution via Kahn's topological levels (cycle /
dangling dep fails fast at plan time); per-step policies (`optional` ⇒
degraded mode, `on_failure: continue|fail_fast`); bounded replanning
(`replan_policy` auto/always/off, max 1/3/0) with sanitized new steps;
checkpoint after every step (configurable); heartbeat for reconcile;
cooperative cancel; VERIFYING acceptance gate before COMPLETED (never a
silent success); reflection scoring (mechanical base, model only adjusts);
self-heal for dead workers; escalation to owner on terminal failure; honest
at-least-once contract documented in the module docstring.
Missing: **no per-step retries** (a step fails once and the policy ladder
engages — no transient-retry with backoff); **no per-step timeout** (a hung
step hangs the mission; the watchdog only notices after 30 min); **levels
run sequentially** ("the blackboard stays sequential") — independent steps
in one level never fan out; `_execute_step` charges iterations even for
idempotency replays? (no — replay returns stored outcome, still counts an
iteration: acceptable, documented below).

**`progress.py` — chat-visible status.**
Does well: concrete stall codes with unblock hints (never a bare
"waiting"); honest ETA (returns None + a *reason* instead of inventing a
number); `render_status_text` built only from persisted state; milestone
pushes through the existing Notifier (dedupe, feature flag, quiet-hours —
no parallel channel); transition-gated started/stalled/terminal pushes;
cooldown-gated step pushes; `/mission watch` fan-out; everything
telemetry-safe (never raises).
Missing: ETA uses whole-run average (`spent_wall / done`) — best practice
is a trailing window; no per-step timing is persisted at all.

**`golden.py` — deterministic drills.**
Does well: fixed plans that run with zero model (execute → verify → repair
→ re-verify); two sizes (seconds vs minutes); killable (PAUSED, resumable);
benchmark DB recording (`model_id="golden:<key>"`) for regression over
time; the long drill injects deterministic faults (lost fact, off-by-one,
insecure defaults) so the *repair* path is genuinely exercised.
Missing: it is a **parallel drive loop** re-implementing run/checkpoint/
resume instead of riding `MissionRunner`; step reports lack attempt counts
(repair is attempt 2 but the report just says `repaired: true`).

**`idempotency.py` — Wave J.**
Does well: SQLite key→result next to mission state (`CREATE TABLE IF NOT
EXISTS`, no-migration pattern); keys derived from *business intent*
`(kind, params, scope)` — `(step name, goal, role)` scoped to the mission —
not from runtime state; in-process collapse via condition variable (one
execution, waiters share the result); stale-owner steal after `stale_after`;
failed keys are retryable, completed keys never re-execute; `succeeded()`
predicate lets "ran but ok=False" record as failed; `create_mission_once`
for exactly-once mission creation.
Missing: **wired nowhere by default** — `MissionRunner.idempotency=None`
unless the caller passes it; the CLI passes it, but the chat path
(`runtime_mission.py`) and triggers (`triggers/actions.py`) don't, so the
crash-window duplicate-side-effect protection is off in production paths;
no TTL/hygiene (`purge`) for the keys table.

**`wiring.py` — L5→L6 bridge.**
Does well: `wired_runner` attaches the os state-machine hook
(`attach_runner`) so production runs get transitions + the VERIFYING gate;
the upward import is lazy + dynamic inside the function (the sanctioned
pattern, documented as do-not-copy); `set_acceptance` fail-fast.
Missing: doesn't attach the idempotency store (see above).

### 1b. Consumers (read-only for this slice)

- `agents/partner/runtime_mission.py` — Devon's `/mission` chat surface:
  status (with reconcile), stall/clear, pause/resume (idempotent, resume
  reconciles dead workers first), replan (forces policy=always, resets
  budget), cancel (idempotent, never rewrites terminal), retry (fresh row,
  `metadata.retry_of` link, original kept as history), watch. Background
  threads for resume/replan/retry with DB-connection release. **Well built.**
- `cmdline/commands/missions.py` + `mission.py` — `nm missions` CLI; the
  only production path that passes `idempotency=IdempotencyStore(...)`.
- `triggers/actions.py` — triggers can *start* missions (`ACTION_MISSION`);
  the triggers engine consumes `mission.terminal` bus events (scheduler job
  finished → start mission; mission terminal → notify). **The
  mission→other-systems loop already exists via the event bus.**
- `scheduler/scheduler.py` — `consult(mission)` advisory only (L4 must not
  import L6); overlap policies; condition ops for event hooks.
- `self_improvement.py` — `_PipelineRunner(MissionRunner)` subclass:
  idempotent persisted pipeline step handlers. Proof the runner is
  subclass-friendly.
- `os/mission_state.py` (L6) — the formal state machine (PLANNED →
  RUNNING → VERIFYING → COMPLETED/FAILED, + PAUSED/CANCELLED);
  `evaluate_acceptance`; `attach_runner`. VERIFYING lives here, not on the
  `Mission` row (the row stays RUNNING through verification, which is
  synchronous inside `_finish` — deliberate, not a bug).
- `agents/devon.py`, `agents/ops_alerts.py`, morning briefing — start and
  report missions through `wired_runner`.

### 1c. Test coverage (existing — do not duplicate)

`test_missions.py` (incl. real SIGKILL crash-recovery in a subprocess),
`test_missions_deep.py` (topo, policies, degraded, replan, lock keys,
ledger/bus, escalation), `test_missions_f2.py` (status, milestones,
notifier discipline, stalls, rendering), `test_missions_g3.py` (reconcile,
idempotency, stall reasons), `test_missions_wiring.py` (wired runner,
acceptance gate, health, self-heal), `test_mission_control.py` (chat
verbs), `test_mission_replay.py` (replay/redrive), `test_golden_missions.py`,
`test_os_missions.py`, `test_idempotency.py`, `test_coding_mission.py`.

## 2. Outside best-in-class (researched)

- **Durable execution** (Temporal/event-history replay, DBOS/Restate/Inngest
  checkpointed steps, LangGraph `checkpointer` + `thread_id`, Airflow
  explicit task checkpoints): record the *journal of side-effecting steps*,
  not just the latest state; recovery replays recorded results instead of
  re-executing. → Our idempotency store is the SQLite version of this; it
  just needs to be **on by default**.
- **Idempotency keys**: key = `{entity}:{action}:{target}` — business
  intent, never trace ids or timestamps; write the side-effect log *before
  returning*; two logs for two concerns (replay log for determinism within
  a run vs side-effect log global); unsafe step with *unknown* outcome
  after a timeout ⇒ FAILED "outcome unknown", human decides — never guess;
  IN_PROGRESS duplicates collapse/block/poll, never double-execute.
- **Retry policy**: max attempts + total time budget + exponential backoff
  with full jitter + per-call timeout smaller than the deadline; classify
  failures (429/5xx/timeout transient; validation/auth non-retryable).
  → We have none of this per step.
- **Progress honesty**: percentages only from verifiable completed steps
  (a script does the arithmetic, never the model); 100% means done —
  floored, never rounded; ETA from a trailing window, never whole-run
  extrapolation; stall indicators carry cause + unblock hint; small stable
  event vocabulary (`job.started/step.started/tool.started/tool.completed/
  step.failed/job.completed`); the UI never infers completion from elapsed
  time.
- **Golden tests**: deterministic drills surrounding the model with
  deterministic checks; approved-output hashing; execute→verify→repair→
  re-verify; control the three nondeterminism sources (model calls, ids/
  wall-clock, iteration order).
- **Cancellation**: stop admitting new work; propagate; bounded graceful
  checkpointing; never claim a checkpoint committed unless publication
  completed; release resources.

## 3. Gaps → Slice A implementation plan

| # | Gap | Fix (in my files) |
|---|-----|-------------------|
| 1 | No per-step retry: one transient 429 kills the step and engages the replan ladder | `_step_policy` gains `retries`, `retry_backoff_s`, `retry_on` (`transient`/`any`/`none`); transient-error classifier; jittered exponential backoff; attempts persisted in `state["step_attempts"]`; cancel-aware sleep. Defaults: `retries=0` ⇒ behavior unchanged |
| 2 | No per-step timeout: a hung step hangs the mission | `timeout_s` in step payload (else `metadata["step_timeout_s"]`); thread + join; timeout ⇒ failed outcome "step timed out after Ns (agent thread abandoned — honest)"; recorded failed ⇒ retryable |
| 3 | Levels run sequentially; no fan-out for independent steps | `metadata["max_parallel"]` (default 1 = historical path untouched, clamp 1..8); steps in one level run on a thread pool; settle under a lock in plan order (policies/stall semantics preserved); levels stay the barrier |
| 4 | Idempotency off by default in production paths | `wired_runner` attaches `IdempotencyStore(context.db)` unless caller passes `idempotency=` explicitly (incl. explicit `None` to opt out); add `IdempotencyStore.purge(older_than_seconds)` hygiene (never deletes `running`) |
| 5 | ETA uses whole-run average; no per-step timing persisted | `_settle_step` records `state["step_durations"]` (bounded); `estimate_eta` uses trailing-5 mean when available, honest fallback otherwise; `detail()` surfaces `attempts` |
| 6 | `resume_all` aborts the whole batch on one unexpected exception | catch per-mission broadly, log, continue (headline kill -9 recovery must not die on one bad row) |
| 7 | Golden runner is a parallel drive loop; reports lack attempt counts | keep the loop (verify/repair semantics are golden-specific) but record `attempts` in step reports |

Deliberately NOT changed: `resume_all` does not `reconcile` first —
reconciling on boot would mark kill -9 victims FAILED instead of resuming
them, breaking the headline recovery property (proven by
`test_missions.py::CrashRecoveryTests`). `MissionStatus` does not gain
VERIFYING — verification is synchronous inside `_finish` and the formal
state machine (L6) owns that state. `start()` lock-key behavior unchanged.

## 4. Cross-slice integration points (not mine to touch)

- `agents/partner/runtime_mission.py` — could surface `detail()["attempts"]`
  in `/mission status` text (now available without touching that file);
  could pass `idempotency=` explicitly (now defaulted via `wired_runner`).
- `triggers/actions.py::default_start_mission` — no `create_mission_once`
  (double-submit race), no idempotency (now defaulted); a "schedule
  follow-up job on mission.terminal" trigger action would close the
  mission→scheduler loop (bus event already published).
- `scheduler/scheduler.py` — missions cannot yet enqueue scheduler jobs
  directly; the bus→triggers path is the sanctioned route.
- `os/mission_state.py` — VERIFYING + acceptance evaluation live here;
  unchanged by this slice.
