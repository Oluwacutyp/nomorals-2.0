# Planning & Goals & Morning Pulse — Mining Report (Phase 9 Slice C)

Written 2026-10-10 before any code changes. Every implementation in the
slice was read in full (docstrings, control flow, DB schemas, tests).

## 1. In-repo implementations

### 1a. `nomorals/planning/graph.py` — WorldGraph (o9 knowledge-graph pattern)

**What it does well.** Live world model: people/projects/commitments/assets/
schedules/deadlines as nodes, `depends_on`/`blocks`/`owned_by`/`due` edges,
SQLite-backed (in-memory on termux), disruption propagation via BFS over
`depends_on`+`blocks`, `critical_path()` (longest depends_on chain),
idempotent `sync_from_memory()` / `sync_from_tasks()` projections, first-
consumer seam `disruption_alerts()` matching disrupted nodes to upcoming task
labels. Never-raises discipline, honest disclaimer. `test_world_graph.py` exists.

**What's missing.**
- `critical_path()` counts hops, not time: a 2-hop chain of 5-minute jobs
  beats a 5-hop chain of month-long jobs. No duration-aware scheduling.
- No PERT/three-point estimation on nodes: nodes carry `due_ts` in attrs
  but nothing computes expected completion time or project variance.
- No topological "do this first" ordering — `dependents()` gives blast
  radius, not a schedule.
- No goal projection: GoalSystem steps are invisible to disruption alerts
  (the pulse warns about flights, never about a goal that depends on one).
- `sync_from_memory` is keyword-match only (no memory↔graph dedup beyond
  external_ref).

### 1b. `nomorals/planning/route.py` — Predict→Build→Solve pipeline

**What it does well.** The mined insight is right: *cost-function accuracy
dominates solver choice*. Learned per-cell-pair EMAs + global speed/cost
learning in `CostModel`, honest `learned|heuristic` labeling per leg, H3 +
quantized-grid fallback GeoIndex, profile-gated greedy/OR-Tools solver,
`format_route` with per-stop arrival times. `test_route.py` exists.

**What's missing.**
- `Stop.window` exists in the dataclass but the solver **ignores it** —
  a plan can confidently promise arrivals after a hard deadline.
- No infeasibility signal: nothing tells the caller "these 5 stops cannot
  fit before 5pm".
- `_solve_greedy` is nearest-neighbor only; no 2-opt polish pass on the
  greedy path (termux default), though that's a bounded tradeoff.

### 1c. `nomorals/planning/congestion.py` — Multi-agent congestion (Symbotic pattern)

**What it does well.** Resource registration (api_key/endpoint/lock/
rate_limit), acquire/release holds, arrival-vs-service-rate jam prediction
with EMA smoothing on workstation and a cheap queue-depth heuristic on
termux, `advise()` → proceed/stagger/backoff/**switch** (provider failover),
`pre_fanout_check()` orchestrator seam, fail-open "proceed" everywhere.
`test_congestion.py` exists.

**What's missing.**
- No ergonomic hold context manager — fan-out code must hand-roll
  try/finally around acquire/release.
- `advise()` answers one resource at a time; there is no "route this work
  around congestion" planner that takes a set of resources and returns a
  routed assignment (primary + alternates).
- Hold durations aren't tracked (acquired_at is stored but never used for
  service-rate estimation — `_rates` uses event counts, which is fine, but
  stale-hold decay is absent).

### 1d. `nomorals/planning/estimates.py` — Honest banded estimates (Just Eat/Meituan)

**What it does well.** Bands not points, multiplicative widening after
misses (Just Eat US11853909B1 pattern), learned EMA bias correction
(`ema_ratio`), per-segment absolute-error learning, confidence tracking,
`delay_risk()` / `delay_alert()` proactive late warnings, Meituan
conservative deadline quoting, `/eta` chat control, scheduler seams
`eta_for()`/`record_outcome()`. Best-in-class for agent estimates.

**What's missing.**
- No PERT three-point estimator (optimistic/most-likely/pessimistic →
  TE=(O+4M+P)/6, σ=(P−O)/6) — the classical input format for new,
  unrepeatable tasks where no history exists.
- No project-level aggregation: per-step bands exist, but nothing sums
  step bands into a goal-level band with variance added in quadrature.
- `delay_alert` only fires when polled; nothing pushes it.

### 1e. `nomorals/agents/goals.py` — GoalSystem

**What it does well.** Durable goals with executable plans, structured-
plan floor (structuring sub-agent) → LLM decompose → heuristic fallback,
`advance`/`adapt`/`replan` self-correction, mission control
(priority/depends_on with cycle guard/`next_goal`), goal→project cascade,
KG completion knowledge, upstream knowledge injection, reflective
checkpointing, room auto-creation, `tick()` continuous hook.
Comprehensive — this is the slice's strongest file.

**What's missing.**
- **No deadlines**: `Goal` has no due-date field at all; `due_soon()` /
  overdue detection doesn't exist. The #1 feature of every real goal
  tracker (also: test_goals_tracker covers a *different* GoalTracker in
  `nomorals/goals/`).
- **No time estimates**: steps have no duration; a 12-step goal and a
  3-step goal look identical to the scheduler. No integration with
  `planning.estimates`.
- No world-graph projection: goals are invisible to disruption alerts.
- `tick()` advances blindly; nothing skips goals whose `next_goal`
  ordering says "not now".

### 1f. `nomorals/agents/planner.py` — AgentPlanner

**What it does well.** LLM plan generation with *explicit, honest*
template-fallback degradation (`plan_error` carried on Plan and PlanResult
and into the summary — never a silent downgrade), dependency-aware
execution with skip-on-failed-deps, ErrorIntelligence retryable analysis,
cancellation, progress callbacks, pluggable action handlers, registered as
the `plan_goal` tool.

**What's missing.**
- `execute()` runs steps **sequentially** even though steps carry
  `depends_on` — independent steps (search_products ×2) never run in
  parallel. The dependency graph exists but is only used for skipping.
- No plan **validation** before execution: dangling `depends_on` ids,
  unknown actions, and cycles are discovered mid-flight instead of up front.
- No per-step timeout or retry: a retryable error breaks the plan instead
  of being retried (ErrorIntelligence says retryable, nobody retries).
- Action surface is email/calendar/shopping only — fine for its purpose,
  but no seam to the broader tool registry.

### 1g. `nomorals/agents/morning_pulse.py` — Morning pulse

**What it does well.** The autonomous showpiece: news radar → briefing
compose → two-host LLM script → per-turn TTS → OGG → Telegram voice note,
every stage through an exponential-backoff stage runner with ledger
entries, graceful degradation at every joint, `pulse.finished` bus event,
idempotent `ensure_pulse_job` (fire_now / skip-overlap / 30-min cap),
delivery diagnosis instead of silent 0-deliveries. `test_morning_pulse.py`
+ `test_pulse_deep.py` exist.

**What's missing.**
- **No boot catch-up**: `morning_briefing.check_catchup` exists for the
  07:00 briefing, but the pulse has no equivalent — a phone off at 23:00
  relies solely on the scheduler's `fire_now` missed-fire policy; if the
  *delivery* failed (not the fire), nothing retries it.
- No last-run/delivery marker: "did last night's pulse reach the owner?"
  requires digging the autonomy ledger.
- Script covers briefing+news but never mentions **goals** (active goals,
  next actions, due-soon) — the thing the slice is about.
- `_write_script` silently drops the LLM script if `HOST_1_NAME` isn't in
  the text — a brittle format check (falls back to template, which is
  honest, but the check is arbitrary).

### 1h. `nomorals/agents/morning_briefing.py` — Morning briefing

**What it does well.** 10+ pluggable `_Provider`s (alerts, calendar,
weather, people, markets, repos, news, rooms, devon-self, research,
from-your-past, readiness, mood), adaptive section budgets from the word
cap, engagement-based demotion + pins, anti-monopoly per-source caps,
never-drop-alerts truncation, followup deep-links, delivery health
reporting, `check_catchup` boot semantics, `test_briefing.py` exists.

**What's missing.**
- No **goals** section: active goals / progress / due-soon are the most
  "what needs attention today" content a briefing can carry, and it's
  absent (rooms are covered; goals aren't).
- `Disruption alerts` from the world graph aren't surfaced ("your 2pm
  depends on a delayed flight" is computed by `disruption_alerts()` but no
  provider calls it).

### 1i. Neighboring planners (read-only, not owned)

- `agents/plan_mode.py` — coding-agent plan→approve→execute; unrelated to
  runtime planning, no overlap.
- `agents/mission.py` `MissionControl` — mission orchestration; goals.py's
  mission-control naming collides conceptually but they operate at
  different layers (missions = sessions, goals = durable objectives).
- `agents/orchestration/planner_bridge.py` — agentic-loop → orchestrator
  TaskGraph bridge with parallel execution; the pattern `planner.py`
  should steal (topological batching).
- `scheduler/scheduler.py` — already consumes `planning.estimates`
  (`eta_for`, `record_outcome`); the planned `record_outcome` seam for
  goal steps is open.

## 2. Outside research (best-in-class)

### HTN / SHOP planning
HTN planning (Nau et al.; SHOP/SHOP2/JSHOP2): recursive total-order forward
decomposition — compound tasks → methods (preconditioned) → subtasks →
primitive operators; depth-first with backtracking; methods selected by
preconditions against a simulated world state; plan validated by
re-execution. Sources: CMU HTN slides (cs.cmu.edu/~reids/planning/handouts/
HTN.pdf), flyriver HTN overview, sbox-htn (JSHOP2-like precondition query
system with backtracking). **Steal:** method libraries with preconditions
and backtracking search; but the practical steal for this slice is smaller —
*plan validation before execution* (dangling deps, cycles, unknown actions)
and *dependency-graph parallel execution*, which the orchestrator bridge
already does for TaskGraph.

### PERT (US Navy Polaris, 1958)
Three-point estimation: TE = (O + 4M + P) / 6 (beta-distribution mean),
σ = (P − O) / 6; project duration = Σ TE on the critical path, project
variance = Σ σ², giving P(completion by date) via the normal CDF. Sources:
medium.com PERT overview; testbook PERT references. **Steal:** PERT node
estimates feed a duration-aware critical path in the world graph and
project-level goal bands in estimates.py (variances add in quadrature).

### Adaptive estimation (already in-repo, validated by research)
Just Eat US11853909B1 quantile-interval widening after misses and Meituan
conservative deadline quoting are already implemented in estimates.py —
research confirms these are the right patterns; no change needed.

### Daily briefing / morning routines
Best practice from product teardowns (not academic): the briefing is
**triage-first** (alerts > calendar > everything), **actionable** (every
item carries its next action), **bounded** (word cap), and **follow-up-
able** (deep links). The pulse pattern news→briefing→voice→delivery is
already the in-repo architecture and matches the "morning routine"
literature (chain, don't batch-manually). **Steal:** add the *goal triage*
section (goals with next actions + due-soon) and a *disruption* line to
the briefing; add *delivery retry + boot catch-up* to the pulse.

## 3. Implementation plan (what Slice C will build)

Merging into existing classes only — no parallel duplicates:

1. **`planning/graph.py`**: duration-aware scheduling — `schedule_order()`
   (topological, due-date urgency), `critical_path_pert()` (TE/variance per
   PERT, project duration + σ), `link_goal(goal)` / `sync_goal_steps()`
   projecting GoalSystem goals+steps as nodes with sequential depends_on
   edges so disruption alerts cover goals.
2. **`planning/estimates.py`**: `pert(o, m, p)` three-point estimator;
   `aggregate(estimates)` project-band (Σ expected, σ in quadrature);
   `Estimate` gains `std` (band half-width / 2); export in `__init__`.
3. **`planning/route.py`**: window feasibility — annotate legs against
   `Stop.window`, surface `window_violations` on RouteResult + in
   `format_route`.
4. **`planning/congestion.py`**: `hold()` context manager;
   `route_around(names, agents_waiting)` — per-resource routed assignment
   (primary or alternate) for fan-out.
5. **`agents/goals.py`**: `deadline` column (defensive ALTER), `set_deadline`,
   `due_soon()`/`overdue()`, per-step `est_minutes`,
   `estimate_goal()` (banded, via planning.estimates segments),
   `record_step_outcome()` (feeds estimates store), `project_to_graph()`.
6. **`agents/planner.py`**: plan validation (`validate()` — dangling deps,
   cycles, unknown actions) before execution; topological batching with
   parallel `asyncio.gather` per batch; per-step retry for retryable
   errors with backoff; per-step timeout.
7. **`agents/morning_briefing.py`**: `GoalsProvider` (active goals, progress,
   next action, due-soon) + `DisruptionProvider` (world-graph disrupted
   nodes matched to today's items).
8. **`agents/morning_pulse.py`**: `check_pulse_catchup()` (boot retry when
   last pulse didn't deliver), last-run marker sidecar, delivery-failure
   requeue; pulse script gains goal-progress beats from the briefing.

## 4. Cross-slice integration points (couldn't touch)

- `scheduler` invoking `GoalSystem.tick()` on a cadence (slice owns goals;
  the scheduler wiring belongs to whoever owns scheduling).
- Orchestrator `pre_fanout_check` consuming `route_around` output
  (orchestration slice).
- `AgentPlanner` action handlers for the general tool registry
  (capability/tool-loop slice owns the registry).
- TTS voice quality for the pulse (voice slice).
