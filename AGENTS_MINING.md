# AGENTS module — external mining report

Module: `nomorals/agents/` (144 Python files). Method: for every significant class,
"What does the best implementation of X outside this repo do that we don't?"
Then: features missing, features weak, how it should look/feel.

Sources: HF `smolagents` (apache-2.0), `openai/openai-agents-python` (Runner,
handoffs, guardrails, tracing), `langchain-ai/langgraph` (checkpoints, supervisor,
plan-and-execute), `microsoft/AutoGen` (group chat, blackboard, code execution),
OpenAI `swarm` (handoffs), Erlang/OTP supervision trees, Microsoft PyRIT
(multi-turn red-team orchestration: RedTeamingOrchestrator, Crescendo, TAP),
promptfoo (CI-gated red-team reports), NotDiamond / langchain-notdiamond (model
routing with cost/latency/quality tradeoffs + real-time feedback), Du et al.
2023 "Improving Factuality and Reasoning through Multiagent Debate" (arXiv:2305.14325),
the "Durable Agent Checklist" (LangGraph practice: stable thread ids, idempotent
steps, small serializable state, retention policy).

---

## 1. `partner/tool_loop.py` — ToolCallingLoop (ReAct spine)

**External gold:**
- OpenAI Agents SDK `Runner`: turn loop with **handoffs** (agent switches mid-run),
  **guardrails** (input/output/tool), **human-in-the-loop pause/resume**, tool timeouts,
  `MaxTurnsExceeded`, tracing (trace → span per agent/generation/tool), and
  `run_streamed()` for live partial results + lifecycle hooks.
- smolagents `MultiStepAgent`: **planning interval** (periodic re-plan step),
  `managed_agents` (agents-as-tools, each with own context window, only final
  report flows back), parallel tool execution in `ToolCallingAgent`,
  `step_callbacks` (post-step), iteration cap, structured memory (ActionStep /
  PlanningStep / FinalAnswerStep).
- Durable-agent practice: stable run ids, idempotency, small state.

**Gaps (features):**
1. No event/streaming hooks — a live UI can't show progress. → add `on_event`
   lifecycle callbacks (turn/tool/final), mirroring OpenAI's hooks + streamed runner.
2. No stall detection — the same (tool, args) can repeat until budget dies.
   → detect N consecutive identical calls and degrade honestly.
3. Per-turn calls run sequentially in the pool. → parallel per-turn execution
   (opt-in) like smolagents.
4. No renderable trace — `ToolLoopTrace` is debug-only. → `render_transcript()`
   markdown for chat display (god-tier presentation).
5. No plan step / mid-run re-plan injection.

**Style:** difflib imported mid-function; observation text is raw dict dumps.

## 2. `orchestration/loop.py` — AgenticLoop (code-first ReAct)

**External gold:** smolagents `CodeAgent` (~30% fewer steps via code),
LangGraph interrupts (human approval), checkpoint-after-step.

**Gaps:** no checkpoint-per-step (crash loses the whole run), no human-in-the-loop
interrupt, no skill-distillation hook wiring for the code path (db exists but
code-mode results never distill). → add optional per-step checkpoint snapshot
and expose memory snapshot save/load (LangGraph thread-id semantics).

**Style:** confidence UX appends a tag to the answer; good, but no visual
progress for long code-mode runs.

## 3. `debate.py` — Debate / PanelDebate

**External gold:** Du et al. 2023 protocol: (a) each agent answers INDEPENDENTLY
first, (b) each round every agent sees all others' answers and revises its own,
(c) final answer from converged consensus — 3 agents × 2 rounds ≈ +15pp on
GSM8K/MMLU, fewer hallucinations on biographies. AutoGen group chat for turn
management.

**Gaps (features):**
1. Current `Debate` is coder-vs-critic, NOT the paper's peer reveal-and-revise.
   → add `SymmetricDebate`: independent drafts → reveal-and-revise rounds →
   convergence detection (all answers agree → stop early) → judge verdict.
2. No round-transcript rendering — `transcript` is a dict list. →
   `render_debate()` markdown with per-agent, per-round panels.
3. No async parallel debate (sequential calls = slow).

**Style:** scores as floats in logs; no human-readable verdict presentation.

## 4. `orchestrator.py` — MasterOrchestrator (DAG over TaskGraph)

**External gold:** LangGraph StateGraph (typed state, checkpointed edges),
Temporal (durable workflows, retries, human signals), Prefect (retries,
caching, result persistence). All of them: durable step results, human
approval interrupts, run resumption from stable ids.

**Gaps:** debate step and model critic exist (good); missing: per-step durable
checkpoints tied to `checkpoints.py` (so `resume()` is truly crash-safe),
partial-result degradation per step (LangGraph graceful degradation), run
visualization (Mermaid-style graph render for the user).

**Style:** plan parse is text-based; a rendered plan table would help.

## 5. `checkpoints.py` — CheckpointStore (git-based + state payloads)

**External gold:** LangGraph `SqliteSaver`/`PostgresSaver`: stable thread id,
small serializable state, retention policy, resume-from-last-good.

**Gaps:** no `verify()` (integrity check of stored files/git commit), no `diff()`
between checkpoints (what changed?), no human `describe()` summary. →
add all three (from the durable-agent checklist); these are the difference
between "saved" and "trustworthy".

**Style:** `short()` exists; add a rendered checkpoint card.

## 6. `router_select.py` — TaskRouter (model selection)

**External gold:** NotDiamond: per-prompt model selection over cost/latency/
quality tradeoffs, **real-time feedback loop** (session outcomes personalize
routing), custom routers trained from eval datasets. langchain `NotDiamondRunnable`
returns the pick as a plain string the app applies.

**Gaps (features):**
1. No learning: `ReliabilityPenalty` exists but is static per-refresh.
   → add `record_outcome(model, task_kind, ok, latency_ms)` feeding a
   persistent per-model reliability score (NotDiamond-style feedback loop).
2. No `explain()`: decision() gives scores, but no human sentence.
   → `explain(model)` → "picked X because …".
3. No exploration: pure argmax gets stuck. → epsilon-greedy exploration toggle.

**Style:** decision dict is fine; explanation should be prose.

## 7. `blackboard.py` — Blackboard (Hayes-Roth style)

**External gold:** classic blackboard architectures, AutoGen shared context.
Best versions: pattern subscriptions, TTL, provenance (who wrote what, when),
snapshots, and queryable history.

**Gaps:** `most_read()` exists (nice); missing: `history(key)` (audit of who
changed what), `export()`/`import()` JSON snapshot for persistence across
restarts, and a rendered board view. → add all three.

**Style:** no human-readable board dump.

## 8. `redteam*.py` — RedTeam / catalog / sandbox / scenarios

**External gold:** PyRIT: **multi-turn orchestrators** (RedTeamingOrchestrator:
attacker + scorer loop), **Crescendo** (start benign, escalate so each turn
looks reasonable in isolation), **TAP** (branch multiple attack lines, expand
the promising, prune the dead), **converters** (base64/translation/ASCII-art
mutation), memory of every turn. promptfoo: CI-gated actionable reports.

**Gaps (features):**
1. Current suite is single-shot scenarios with scripted LLMs. → add a
   **Crescendo-style multi-turn strategy**: benign → escalate → scorer judges
   progress each turn → refine or prune (PyRIT's composable-primitive shape,
   adapted to the existing `AttackScenario` harness).
2. No per-turn progress scoring → attacks can't adapt. → `AttackTurn` +
   progress scorer callback.
3. Report is data-only. → `render_report()` markdown (promptfoo-style:
   pass/fail per probe, severity, evidence, recommendation).

**Style:** `RedTeamReport.summary()` exists; make the report beautiful and
actionable.

## 9. `supervisor.py` — Supervisor (restart policies, run_all)

**External gold:** Erlang/OTP supervision trees: one_for_one / one_for_all /
rest_for_one restart strategies, intensity limits (max R restarts in T seconds),
escalation. Kubernetes: backoff, liveness.

**Gaps:** `RestartPolicy.delay_for` exists; missing: **restart intensity**
(max attempts within a window before giving up — OTP's `intensity/period`),
and **rest_for_one** semantics (restart dependents after a failed dependency).
→ add intensity window + dependent-restart to `run_all`/`watch_graph`.

**Style:** events exist; render a supervision tree status view.

## 10. `subagents.py` — Subagent (Planner/Implementer/etc.)

**External gold:** OpenAI Agents SDK `handoff` + agents-as-tools; smolagents
managed agents (own context window, report-back template, arbitrary nesting).

**Gaps:** no handoff semantics (context carried between agents), no
arbitrary-depth delegation reporting. → add `HandoffResult` with lineage +
report template; make `spawn` in `base.py` carry budget fraction (already
does — good) and add depth caps to prevent runaway nesting.

## 11. `swarm.py` — SwarmAgent (decompose → legs → synthesize)

**External gold:** OpenAI Swarm (routines + handoffs), AutoGen group chat,
map-reduce fan-out.

**Gaps:** resume exists (good); missing: leg-level retry with different
angle (TAP-style expand promising, drop dead), synthesis with source
attribution (which leg said what), progress events during long runs.

**Style:** `_synthesize` returns text; add a rendered run report (legs as
sections with status icons).

## 12. `planner.py` / `plan_spec.py` / `plan_mode.py` — AgentPlanner

**External gold:** LangGraph plan-and-execute, AutoGPT goal decomposition.
Best: validate-then-repair loops, dependency-aware batching (exists via
execution_batches — good), plan diff on replan.

**Gaps:** no **replan diff** (what changed between plan v1 and v2 — the user
wants to SEE adaptation), no plan rendering as a table. → `diff_plans()`
+ `render_plan()` markdown table.

**Style:** plans are dicts; render them as beautiful tables.

## 13. `reasoning.py` — ReasoningEngine (CoT/decompose/hypothesize/critique/tree)

**External gold:** CoT, self-consistency, Tree-of-Thoughts, ReAct.

**Gaps:** no **self-consistency voting** (sample N, majority vote — the
cheapest reasoning upgrade in the literature), no confidence per step.
→ add `self_consistency` strategy: N samples at temp>0, vote; attach
per-step confidence from `confidence.py`.

**Style:** `ReasoningStep` → render as numbered reasoning trace.

## 14. `fanout.py` — fan-out / fan-in / map-reduce

**External gold:** standard scatter-gather; best versions add progress
callbacks, partial-failure tolerance with per-shard retries, and a merged
provenance trail.

**Gaps:** add progress callback + per-item retry policy; render a
shard-status table.

## 15. `autonomy.py` / `autonomy_ledger.py` — AutonomyAgent

**External gold:** Claude's scheduled tasks / memory; proactive assistants.
Best: proposal → policy check → adaptive thresholds (exists — good), plus
a rendered daily ledger of what the agent did on its own.

**Gaps:** ledger exists; add `render_ledger()` daily digest for the user
("while you were away" — the morning-pulse tie-in).

## 16. `skill_*.py` — skill evolution / distillation / proving / canary

**External gold:** Voyager (MineDojo skill library), Eureka. Best: skills
as versioned artifacts with tests (proving exists — good), canary rollout
(exists), distillation from traces (exists).

**Gaps:** no skill **diff view** (what did evolution change), no
human-readable skill card render. → `render_skill()` card.

## 17. `confidence.py` — assess_confidence

Solid heuristic. Upgrade path (noted in-module): LLM-judge assessment.
Add: expose the reason codes as user-facing prose (map already half-there
via `_CHECK_SUGGESTIONS`) — `explain(conf)` sentence. Style: god-tier
uncertainty line is already good; keep.

## 18. `morning_briefing.py` / `morning_pulse.py` / `notifier.py` / `brief.py`

Presentation layer for the user. **Style mandate applies hardest here:**
every digest should read like a beautiful briefing, not a log dump. →
route through a shared renderer (new `render.py`) with sections, tables,
and status icons.

## 19. `watchers.py` / `watcher.py` / `monitor.py` / `ops_alerts.py`

Best-in-class (Datadog/PagerDuty): alert grouping, severity, quiet hours,
escalation. Gaps: dedupe/grouping of repeated alerts, rendered alert cards.

## 20. `research_*.py` / `search/*` — research loop + swarm

Best: Exa/Tavily-style search with trust scoring (trust.py exists — good),
curate/summarize pipeline (exists). Gaps: research digest rendering (exists
in research_digest.py — upgrade its presentation), source-conflict flagging.

## 21. `scheduler.py` — scheduler

Best: cron + one-shot + natural-language parsing; the known gap is
"alert me in X minutes" not recognized as a scheduler request (memory,
2026-10-06). Fix lives in routing, but the scheduler can expose a
`parse_relative("in 20 minutes")` helper the router can call.

## 22. `devon.py` — the agent persona class (1849 lines)

The face of the system. Style mandate: every user-facing string should feel
Devon — direct, warm, zero corporate. Audit user-facing strings for
"AI assistant" boilerplate.

## 23. `evolution.py` / `improvement.py` — self-improvement

Best: Voyager-style curriculum + skill library. Gaps: render the improvement
queue as a prioritized, human-readable roadmap.

## 24. `benchmark.py` — agent benchmarks

Best: SWE-bench harness style, inspect-evals. Gaps: rendered benchmark
report card with trend (delta vs last run).

---

## What ships in this sweep (agents module)

Prioritized by value × risk (no deletions, extend real classes):

1. **NEW `nomorals/agents/render.py`** — shared presentation primitives
   (banner, section, kv-table, status icons, plan tables, shard tables,
   report cards). Used by notifier/brief/debate/swarm/benchmark/redteam.
2. **`partner/tool_loop.py`** — event hooks (`on_event`), stall detection,
   parallel per-turn execution (opt-in), `render_transcript()`.
3. **`debate.py`** — `SymmetricDebate` (Du et al. 2023 reveal-and-revise with
   convergence early-stop + judge) + `render_debate()` transcript.
4. **`router_select.py`** — `record_outcome()` feedback learning +
   `explain()` prose + epsilon-greedy exploration.
5. **`checkpoints.py`** — `verify()`, `diff()`, `describe()` (+ rendered card).
6. **`blackboard.py`** — `history()`, `export()`/`import()`, `render_board()`.
7. **`redteam.py`** — `CrescendoStrategy` multi-turn adaptive attack
   (escalate + scorer feedback + prune) + `render_report()` markdown.
8. **`supervisor.py`** — restart intensity window (OTP) + dependent restart.
9. **`reasoning.py`** — `self_consistency` strategy + per-step confidence.
10. **`planner.py`** — `diff_plans()` + `render_plan()` table.
11. **`swarm.py`** — leg progress events + rendered run report.
12. **`fanout.py`** — progress callback + per-shard retry + shard table render.
13. **`scheduler.py`** — `parse_relative()` ("in 20 minutes") helper.
14. **`autonomy_ledger.py`** — `render_ledger()` daily digest.
15. Style pass: user-facing strings, mid-function imports, consistent result
    rendering in `notifier.py`/`brief.py`/`morning_briefing.py` via render.py.

Tests: `tests/test_agents_sweep.py` covering every new behavior.
