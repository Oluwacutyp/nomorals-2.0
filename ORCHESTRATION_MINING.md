# Orchestration Mining Report — Phase 9, Slice E

Survey of every orchestration implementation in `nomorals/agents/` + outside
best-in-class research. Written before any code was touched.

Date: 2026-10-10. Branch: main.

---

## 1. What exists in the repo

### 1.1 `supervisor.py` (268 lines) — watchdog + restart policy + budget
**Does well:**
- Escalation ladder done right: restart → restart → escalate → give up. Never
  restarts forever; the failure-window rollover forgets stale failures.
- Budget errors are *not* retried by default (`restart_on_budget=False`) —
  retrying a spent budget is the classic token-bill infinite loop.
- `factory` param builds a *fresh* agent for restarts (no poisoned state reuse).
- `retry_task` / `watch_graph` give supervised retry to `TaskGraph` tasks too.
- Honest telemetry: `snapshot()` with per-subject records, last 20 events.

**Missing:**
- Purely passive: only acts when `run_agent`/`retry_task` is called. No
  liveness/heartbeat watch over long-running agents.
- Supervisor events go to `on_event` callback + log only — nothing posts them
  to the shared Blackboard, so other agents can't observe or react to a
  teammate's restarts/escalations.
- No supervised *parallel* team run (run N agents at once, each watched).

### 1.2 `swarm.py` (270 lines) — plan → parallel legs → fuse
**Does well:**
- Model-driven decomposition with a genuinely good heuristic fallback
  (conjunction splitting, numbered items, seeded perspective rotation —
  reproducible per goal, varied across goals).
- Real thread parallelism, per-leg wall budget, crashed legs become failed
  *records* (never fatal), single model-call synthesis with concat fallback.
- `SwarmResult.ok` is honest (any leg ok, not all-or-nothing).

**Missing:**
- Legs are isolated: each runs a full `DevonAgent` loop with no shared
  context. There is a Blackboard *concept* in the repo but the swarm doesn't
  use it — leg 3 can't see what leg 1 already found (duplicate work, missed
  cross-evidence).
- Worker cap is hardcoded at 5; no profile gating (fanout.py has it, swarm
  doesn't).
- No checkpoint/resume: a 7-minute swarm that dies at minute 6 re-runs
  everything.
- Synthesis has no conflict-resolution step when legs disagree — the prompt
  says "keep conflicts visible" but nothing *decides*.

### 1.3 `subagents.py` (2066 lines) — specialist coding team + `run_parallel`
**Does well:**
- The `AgentRun` record pattern is excellent: failures are *records, not
  raises*, never silently dropped; timeouts become explicit failed runs.
- `MERGERS` registry: deterministic, per-roster-type fusion functions
  (reviewer dedupes flaws by severity, tester merges pass/fail, etc.).
- `run_parallel` validates arity up front, keeps result order stable.

**Missing:**
- The roster is coding-only (Planner/Implementer/Reviewer/Tester/Refactorer/
  DepHunter/ApiDesigner). There is no *general-purpose* team runner — every
  non-coding multi-agent job re-invents fan-out.
- No cancellation of stragglers, no priority, no partial-result streaming.
- Verified: `AgentRun` **is** a `@dataclass`, so `to_dict()` via
  `dataclasses.asdict` is sound — the suspected bug was a false alarm.

### 1.4 `fanout.py` (572 lines) — fan-out/fan-in/map-reduce + Hark compare
**Does well:**
- Injected `worker_fn` makes every pattern fully testable without a model.
- Per-angle budget slicing (`child_budget`) — one runaway angle can't eat
  the run. This is the budget discipline most fan-out implementations lack.
- Three honest merge strategies (`concat_dedupe`, `vote`, `judge`); the judge
  fallback says "kept first" instead of pretending it judged.
- Profile-gated parallelism for the Hark compare path (workstation 36 /
  laptop 12 / termux 4) — the only fan-out in the repo that respects the
  machine it's on.
- `compare()` one-call API: fan out per source → comparison table + markdown.
  Never raises.

**Missing:**
- No per-angle retry (one transient failure = one error dict; supervisor
  exists but isn't wired in).
- `vote` reports ties but still picks `max()` — no tie-break policy.
- No streaming partial results; no early termination when enough angles agree.

### 1.5 `debate.py` (300 lines) — coder/critic + arbitration
**Does well:**
- Anti-stalemate escalation: an issue repeated 2 rounds unaddressed escalates
  instead of looping forever. `unresolved` is a real verdict — never a silent
  accept.
- Full transcript on the Blackboard (reflector + lesson memory can learn).
- Two-level arbitration: judge agent (level 1) → owner brief (level 2), with
  autonomous-mode behavior explicit and logged.

**Missing:**
- Only 2-party (coder vs critic). No N-position panel debate — the pattern
  the research calls "parallelization and voting" / "multi-agent debate"
  (multiple instances reason independently, then critique).
- `arbitrate` in autonomous mode "proceeds with the first position" —
  arbitrary, and first-position bias is a known debate failure mode.
- No score-based consensus across multiple critics; no structured
  claim→evidence→risk format enforcement (research: the mitigation that
  actually stops debate non-termination).

### 1.6 `blackboard.py` (213 lines) — shared scratch space
**Does well:**
- Versioned entries, TTL/pruning, access counting (`most_read` feeds the
  reflector with "what was actually used"), topics, `append` for list keys,
  glob `keys()`, `wait_for`/`gather` coordination helpers.
- Watchers fire *outside* the lock with error isolation — a bad watcher
  can't break a write or deadlock the board.

**Missing:**
- Entirely in-memory: a restart wipes all shared state. No persistence.
- `wait_for` polls (`time.sleep(poll)`) instead of using a condition — fine
  at this scale but wasteful and slow to wake.
- No *reactive* dispatch: `watch` notifies, but nothing runs an agent *in
  response* to a post. Agents can't trigger each other without someone
  manually polling or calling — the headline gap for this slice.
- No write scoping (any agent can overwrite any key; research flags shared-
  memory poisoning as the #1 blackboard failure mode).

### 1.7 `checkpoints.py` (458 lines) — code+convo checkpoints for coding
**Does well:**
- `git stash create` captures the tree *without touching the user's stash
  list*; rewind never moves HEAD or switches branches; every rewind saves a
  pre-rewind safety checkpoint first. Recovery is genuinely reversible.
- Fail-closed everywhere: convo-only checkpoint when git is unavailable,
  rewind reports failures as strings instead of raising.
- Size-bounded untracked capture, path-escape guard on restore.

**Missing:**
- Coding-workdir-centric: there is no checkpointing for *agent work* —
  TaskGraph state, blackboard contents, swarm legs, orchestrator runs.
- No auto-checkpoint policy; nothing in the orchestration layer captures
  state mid-run, so a dead `MasterOrchestrator.run` loses the whole plan.

### 1.8 `mission.py` (393 lines) — portfolio view + EV scoring
**Does well:**
- The EV scoring is genuinely sophisticated: priority base × dependency-depth
  × reliability (heal history) × size × cost factors, plus budget-fit ETA
  math (`steps_today`, `eta_days`). `plan()` partitions ready/blocked/paused/
  done with `waiting_on` attribution.

**Missing:**
- It's a *view*, not glue: nothing here *executes* a mission. The portfolio
  knows what's ready but can't launch it — the "agent mission glue" the
  slice asks for doesn't exist yet.

### 1.9 `orchestrator.py` (939 lines) — MasterOrchestrator
**Does well:**
- The strongest file in the slice: model planning with *mandatory repair*
  (dedup names, drop dangling deps, break cycles), lesson-memory injection at
  the planner choke point, `HybridExecutor` parallel dispatch, per-role
  handlers + telemetry, supervisor retries, blackboard posts per task.
- Mid-flight `reevaluate` (wave F2): abort on failure cascade, revise on
  explicit handler directives, trim superseded steps, course-correct consult
  on failure. Decisions are recorded, emitted, and telemetered. Long runs
  don't blindly continue invalidated plans.
- Handler failures are *named* (`step 'x' (role 'y', handler z) failed: ...`)
  with the original error classification preserved.

**Missing:**
- No debate integration: a plan step can request verification, but nothing
  routes hard decisions through coder→critic rounds.
- No checkpoint/resume: `_checkpoint` re-evaluates the plan but persists
  nothing — a process death loses the run.
- `run()` and a hypothetical `resume()` would duplicate the execute tail;
  needs the execute phase extracted to share it.

### 1.10 `directives.py` (205 lines) — the durable work queue
**Does well:**
- Persisted queue (pending/running/done/failed), journaled, notifier
  reporting, clean subsystem executors (`_download`/`_research`/`_code`/
  `_model`).

**Missing — and this one is a standing-order violation:**
- `_execute` is *exactly* the hardcoded keyword intent routing the user
  banned on 2026-10-09 ("NO hardcoded regex/keyword intent shortcuts for
  capability routing — it either stays dynamic or should not exist"):
  `if "download" in lowered ... elif "research" in lowered ... elif "write
  a" in lowered`. The *intent classification* must come from the brain;
  subsystem executors stay as capabilities.

### 1.11 `orchestration/` dir — agentic ReAct loop + planner bridge
- `loop.py` (693): think→act→observe single-agent loop. `planner_bridge.py`
  (423): loop→MasterOrchestrator escalation for complex goals, with
  tool-mapped steps and lesson feedback. Solid single-agent story; the
  *team* story (agents triggering each other) lives nowhere yet — it will
  live on the Blackboard's new reactive triggers (see plan below).

---

## 2. Outside research (best-in-class, Oct 2026)

**Pattern catalog** (multiple 2026 surveys agree): orchestrator-worker is the
production workhorse; hierarchical for teams-of-teams; sequential pipeline for
fixed auditable flows; parallel fan-out/scatter-gather for latency; debate /
maker-checker when accuracy > speed; swarm/handoff networks for emergent
paths (hardest to debug — always add a decision node); blackboard when many
agents evolve one artifact; mixture-of-agents (layered refinement); market/
auction for candidate selection.

**Failure modes + mitigations that actually work** (kunalganglani.com 2026
swarm-coordination survey; LinkedIn A2A piece; awesome-claude-multi-agent):
- Supervisor over-trusts a worker → require citations + a verifier agent;
  enforce tool approval for risky actions.
- Debate non-termination → judge *must* decide by round N; force structured
  "claim → evidence → risk".
- Map-reduce reducer overload → hierarchical reduce, per-shard validation.
- Blackboard memory poisoning → write-once logs + append-only memory; scoped
  write permissions.
- Mesh swarms with no decision → add a reducer/judge; stop treating it as
  a mesh.

**Checkpoint/resume** (LangGraph lineage): checkpoint per step keyed by
thread/run id; resume by reloading state and re-invoking with completed phases
skipped via sentinel fields; checkpoint save is best-effort and *never
raises*; version the state schema; don't store blobs in state (store URIs).
Trace ID on every handoff (A2A lesson).

**Agent communication** (A2A protocol, OpenAI handoffs): typed, versioned
messages with schemas — not free text; structured state transfer on handoff.

What we're borrowing: per-step checkpointing with run ids, best-effort never-
raise persistence, judge-decides-by-round-N debates, reactive blackboard
triggers as the handoff mechanism, supervisor-verified workers.

---

## 3. Gap analysis → this slice's build list

| # | Gap | Fix (merged into existing classes, no parallel duplicates) |
|---|-----|-----------------------------------------------------------|
| 1 | Agents can't trigger each other | `Blackboard.link()` reactive triggers: pattern → agent callback, dispatched on a worker thread, error-isolated, cancellable subscription. Chain reactions without manual commands. |
| 2 | Blackboard dies with the process; `wait_for` polls | `Blackboard.save()/load()` JSON persistence (best-effort); condition-variable wakeups in `wait_for`. |
| 3 | Supervisor is passive + invisible | `Supervisor.attach_blackboard()` posts restart/escalate/give_up/budget events to the board; `Supervisor.run_all()` supervised parallel team run (each agent watched, restarted within policy). |
| 4 | Swarm legs are isolated; no resume | `SwarmAgent` takes a shared blackboard; legs post digests as they finish and read earlier digests (cross-leg awareness); `SwarmResult` serializes; `SwarmAgent.save_state()/resume()` re-runs only unfinished legs. |
| 5 | Debate is 2-party only; autonomous arbitrate picks first position | `PanelDebate`: N positions × critic scoring → weighted vote → stalemate escalates to `arbitrate`; `Debate.settle()` convenience: unresolved/needs_arbitration auto-arbitrates instead of dying quietly. |
| 6 | No agent-work checkpointing | `CheckpointStore.save_state()/load_state()/list_states()` — generic JSON run-state snapshots (run id, versioned schema). |
| 7 | Orchestrator can't resume; no debate on hard steps | `MasterOrchestrator.run(..., checkpoint_store=..., run_id=...)` persists plan+task states at every mid-flight checkpoint; `MasterOrchestrator.resume(run_id, store, handlers...)` rebuilds the graph, restores DONE/FAILED/SKIPPED, re-runs only PENDING (execute tail extracted to `_execute_graph` so run/resume share it); per-step opt-in `debate: true` payload routes the step's output through coder→critic rounds. |
| 8 | Mission control can't launch | `MissionControl.launch(goal_id)` — builds a Plan from the goal's pending steps, runs it through MasterOrchestrator, writes step outcomes back to the goal system. Portfolio → execution → progress. |
| 9 | Directives uses banned keyword routing | `_execute` becomes brain-first dynamic routing (brain picks subsystem + args as JSON); no brain → honest error, not keyword guessing. URL entity extraction stays (entities ≠ intent). |

**Deliberately not built:** a generic `Team` class (would duplicate
`Supervisor.run_all` + `fanout`); blackboard ACLs/scoped writes (no caller
needs them yet — noted as future); swarm profile-gating (5-cap is a
documented constant; changing it alters production behavior without a
requester).

**Cross-slice notes:** `HybridExecutor` (runtime.py, not mine) already skips
non-PENDING tasks, which is what makes resume work — verified, not changed.
`GoalSystem._set_step` is private but already used cross-module by
mission.py (`gs._steps`); `launch()` follows the same established pattern.
