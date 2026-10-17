# Research-loop proposal #2 — Foundations: distilled spec

Source proposal: `upg_01M554VTN0WAKB80ZKY5VT2YMQ`, survey arxiv 2508.11126v1.
Scope: ONE distilled spec. Two claims verified in code, then only REAL gaps built.

## Claim (a): iterative loop, not one-shot (plan → execute → feedback → revise)

### Verified-wired (no work — evidence)

| Path | Feedback → revision | Evidence |
|---|---|---|
| Tool loop / dispatch setup | Dispatch setup crash retried with bounded budget + backoff (read-only organs retry more freely; write organs conservative — no duplicated side effects) | `nomorals/agents/coremind.py:1850-1880` (`_dispatch`, `_dispatch_budget`) |
| Code exec | Draft → run → exact stderr fed back into next draft (`last_error`), plus error-recall injection of proven fix patterns; session distilled into a `fix-*` skill | `nomorals/agents/coding.py:853-1104` (attempt loop), `coding.py:1568-1575` (`_draft` feeds `last_error`), `coding.py:1606` (`_recall_error_fixes`) |
| Missions | Step failure → `_settle_outcome` → `_maybe_replan` (bounded, policy-gated) produces a revised graph; run continues on the replanned graph | `nomorals/missions/runner.py:1226-1279` (`_settle_outcome`), `runner.py:1454+` (`_maybe_replan`) |
| Orchestrator | Failed tasks get one supervised retry; mid-flight `reevaluate` revises the plan | `nomorals/agents/orchestrator.py:511-533` (`_execute_graph`), `orchestrator.py:857+` (`reevaluate`) |
| Scheduled research loop | Per-topic exception is logged and the topic stays eligible next cycle (only successful topics land in `topics_done`, so anti-repeat does not bury a failed topic) | `nomorals/agents/research_loop.py:598` + `topics_done` bookkeeping |

### REAL gaps (in `ResearchSwarm.run` — the research organ itself)

**G1 — Failed angles are dropped, never revised.**
`nomorals/agents/research_swarm.py:695-714`: a worker timeout / exception /
"no search results" lands in `report.failed_angles` and the run moves on.
No retry with the error, no rephrased angle, no fallback. The angle's
coverage is silently lost (only noted in the synthesis footer).

**G2 — Supervision critique is computed, then ignored.**
`research_swarm.py:728`: `report.critique = self._supervise(...)`.
`_supervise` calls `ReasoningAgent.course_correct`, which can return
`should_pivot=True` + a concrete pivot — but `run()` never acts on it:
no follow-up angle, no re-synthesis. Feedback without revision.

## Claim (b): explicit memory module for long-term context (not just activation)

### Verified-wired (no work — evidence)

- `MemoryManager` is a real explicit store (episodic/semantic tiers,
  FTS + vector recall, `build_context`): `nomorals/memory/manager.py:310+`,
  `recall()` at `manager.py:570`, `build_context()` at `manager.py:1029`.
- Memory IS read on live paths: companion conversation recalls
  (`nomorals/agents/partner/brain.py:523`, `partner/responder.py:978`),
  coding recalls past fix patterns (`coding.py:1606`), digest checks the
  KG (`research_digest.py:285`).
- Memory IS written on the chat research path: `coremind.py:1991,2936`
  and `partner/runtime_build.py:63` all call `swarm.run(..., save_memory=True)`.

### REAL gap (in `ResearchSwarm.run` — the research organ itself)

**G3 — The swarm never consults memory before researching.**
`run()` has no recall: no prior-findings lookup, no dedup against what
the system already knows. Every run re-derives from scratch, and the
scheduled-loop path (`ResearchPipeline.run`, `research_digest.py:1037`)
calls `swarm.run(query)` without `save_memory`, so on that path the
report is not even filed. Memory written (chat path) is never read back
by the organ that wrote it.

## Spec — behavior

All changes live in `nomorals/agents/research_swarm.py` (+ tests).
No new modules, no new dependencies, no change to any other organ's
behavior when the new paths find nothing to do.

**G1 — Bounded failed-angle retry.**
After the first worker pass, each failed angle gets EXACTLY ONE retry:
- error contains "no search results" → retry with the broadened query
  `"<angle> overview"` (rephrased, not identical);
- timeout / any other error → retry with the same angle once (transient).
The retry's findings are relabelled under the ORIGINAL angle (domain
tagging preserved). A recovered angle leaves `failed_angles` and lands in
`recovered_angles`; `retried_angles` records every angle retried.
A twice-failed angle stays in `failed_angles`. Total worker passes ≤ 2
for the original angles (retry pass is one extra pool run, no recursion).

**G2 — Critique-driven pivot pass.**
`_supervise` returns `(critique_text, pivot)` instead of text only.
When the pivot is a researchable angle — 3–120 chars, ≥ 2 content words,
not a duplicate of an existing angle, and NOT a meta-instruction
(`proceed…`, `stop:…`, `change approach:…`) — `run()` executes EXACTLY
ONE follow-up pass on that angle, merges its findings, re-dedupes,
re-detects conflicts, and re-synthesizes once. The pivot angle lands in
`pivot_angles`. No re-supervision of the pivot pass (no recursion).
Non-researchable pivots are recorded in the critique text only, as today.

**G3 — Memory recall before research; dedupe against memory.**
At the start of `run()`, best-effort recall from `context.memory`
(`recall(query, limit=5)`, never raises, never blocks the run):
keep only records whose `source` starts with `research_swarm` AND whose
content matches the swarm's own fact format `[research: …] <claim>
(source: …)` — i.e. prior swarm FACTS, not episode summaries.
They land in `report.prior_findings` (angle `"prior memory"`).
Any NEW finding whose claim matches a prior finding (`_same_claim`)
is dropped as already-known and counted in `report.prior_overlap`.
Prior findings are never re-filed by `to_memory` (they are not in
`report.findings`), and the scheduled-loop path still does not write
unless the caller passes `save_memory=True` (no behavior change there).

**Report shape (additive, backward-compatible).**
`SwarmReport` gains `prior_findings: list[SwarmFinding]`,
`prior_overlap: int = 0`, `retried_angles: list[str]`,
`recovered_angles: list[str]`, `pivot_angles: list[str]` — all exposed
in `to_dict()` and noted in `to_text()` when non-empty. Existing fields
keep their meaning; `failed_angles` now means "failed after retry".

## Failing tests (written first, in `tests/test_research_swarm_foundations.py`)

1. `test_failed_angle_retried_once_with_rephrasing` — worker fails the
   original angle with "no search results", succeeds on the rephrased
   query → finding recovered under the original angle, `failed_angles`
   empty, `recovered_angles == [angle]`, worker called exactly 2×.
2. `test_failed_angle_stays_failed_after_retry_exhausted` — worker
   always fails → angle in both `failed_angles` and `retried_angles`,
   worker called exactly 2× (bounded, no loop).
3. `test_critique_pivot_runs_one_followup_pass` — stubbed `_supervise`
   returns a researchable pivot → pivot angle researched, its finding
   merged, `pivot_angles == [pivot]`, synthesis regenerated.
4. `test_non_researchable_pivot_is_not_an_angle` — pivot
   `"change approach: try a different tool"` → no extra worker call,
   `pivot_angles == []`.
5. `test_memory_recall_seeds_prior_and_dedupes` — fake memory returns a
   prior swarm fact whose claim matches a new finding → `prior_findings`
   non-empty, duplicate dropped from `findings`, `prior_overlap == 1`,
   `memory.recall` called with the query.
6. `test_memory_failure_never_breaks_run` — `memory.recall` raises →
   run still succeeds, `prior_findings == []`.
7. `test_no_memory_context_still_works` — context without `memory` →
   run works exactly as before.

## Regression

- `tests/test_swarm_specialists.py` (stubbed `_research_angle` through `run()`)
- `tests/test_research_digest.py`, `tests/test_research_cycle.py`,
  `tests/test_research_loop.py` (pipeline / loop consumers of the swarm)
