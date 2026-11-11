---
id: multi-cursor-bench
title: The multi-cursor bench — parallel cursors, one monitor, no shared unfinished work
summary: How several coding cursors work one job in parallel: fenced claims, staged candidates, an independent verifier's veto, contract-first decomposition for coupled work, and serial promotion.
topics: bench, multi-cursor, parallel agents, verification, contracts, swarm
triggers: multi-cursor bench, parallel cursors, bench run, verifier cursor, contract generation, spawn tasks
layer: agents
version: 1
---

# The multi-cursor bench

The bench exists to eliminate one failure mode: an agent building on
another agent's unfinished edits. Everything below is mechanism, not
convention — if a rule is not enforced by a store, it is not a rule.

## The organs

- **Monitor** (`agents/bench/monitor.py`) — the only writer of bench
  state. Cursors never talk to each other; they claim through the
  monitor. A claim is one transaction: a task lease plus one file
  lease per write-set entry, generation-fenced by `core/leases`. A
  claim whose write-set overlaps a live claim is refused *naming the
  holder*. A task is claimable only when every dep is **promoted** —
  not done, not submitted, promoted.
- **Staging** (`agents/bench/staging.py`) — at claim time the cursor
  gets an isolated checkout (git worktree, clone fallback) pinned to
  the promoted base. There are exactly three read views: promoted
  revisions, your own staging, a promoted dep's artifact at its
  promotion revision. A peer's staging has no API. Submit *captures*
  the staging diff as an immutable candidate commit; it stays
  invisible to peers until promotion.
- **Verifier** (`agents/bench/verifier.py`) — a standing cursor
  whose only job is judging. It receives the spec, the base revision
  and the candidate ref — there is no field for the builder's
  self-report. It materialises the candidate alone, reads the full
  diff, re-runs the acceptance and the adjoining suites itself, and
  applies the per-class check table (a bugfix's proving test must
  fail on base and pass on the candidate; format evidence must run
  against a real stored artifact; `migrations.py` is hash-frozen).
  Its verdict is written to `bench_verdicts` and the delegation
  ledger — a verifier can never verify its own work; the store
  refuses. `contradicted` returns the task with the evidence bundle;
  exhausting the attempt budget escalates to the owner as arbiter.
- **Integration queue** (`agents/bench/integration.py`) — serial,
  monitor-owned. Only `verifying` tasks whose latest verdict for the
  *current* candidate ref is `supported` promote, one at a time,
  exactly once. A candidate that does not apply becomes a
  `conflicted` event naming the files — never a silent merge.
- **Verifier pool** (`agents/bench/pool.py`) — hands submitted
  tasks to verifier instances that are not their worker, on a
  JobRegistry cadence; a task whose only verifier is its worker
  waits, named `no_independent_verifier`.
- **Wiring** (`agents/bench/wiring.py`) — `build_bench(db, repo,
  root)` composes all of it; `BenchRuntime.cycle()` runs one full
  pass (publish contracts → verify → promote → publish again). Every
  run also registers as a mission, so `/runs` shows bench runs,
  task states, verdicts and open escalations (ticket refs
  `bench:<run_id>` and `bench-task:<task_id>`).

## Task specs

A spec without a write-set or acceptance is refused — the monitor
never invents either. Write-set entries are repo-relative paths or
subtree globs (`nomorals/agents/**` serialises a layer; use
sparingly). Acceptance entries the verifier can execute itself:
`test:<path>`, `cmd:<command>`, `artifact:<path>` (format class),
`repin:<test id>` (refactor class); anything else is prose it notes
but cannot check.

## Contract-first, for coupled work

When cursors build against one shared interface, decide it once:

1. `contracts.define(run, key, artifact_path)` registers the
   contract as a bench task whose write-set is the artifact alone.
2. Implementation tasks `contracts.bind(...)` to it and list it in
   their `deps` — unclaimable until generation 1 promotes.
3. The contract author claims, writes, submits; the verifier
   judges; promotion publishes **generation 1**. Bound tasks build
   against that frozen generation (`contracts.current()` reads its
   bytes at its promotion revision).
4. A cursor that finds the contract wrong files
   `contracts.challenge(...)` — it never edits the contract itself.
   Bound tasks suspend (claims refuse naming the contract), the
   contract task requeues for revision, and when the revision
   promotes, generation 2 publishes and every binding **rebases**
   onto it, journaled per task. Old generations stay readable via
   `contracts.at_generation(...)`.

## Swarm spawning

A cursor subdivides its own task: `monitor.spawn(run, parent,
spawner, generation, [(key, spec), ...])`. The monitor refuses —
before registering anything — a child whose write-set escapes the
parent's, children whose write-sets overlap each other (naming the
collision), and roles the registry does not know. Children depend
on the parent, so they claim only after the parent's candidate
promotes and its leases release.

## Planning

`MasterOrchestrator.bench_plan(goal, monitor)` is planner-only: it
decomposes and registers bench tasks (steps need `writes` and
`acceptance` payloads; a step missing either is skipped with the
reason, never repaired by invention) and executes nothing itself.
Execution state belongs to the monitor alone.

## The laws

- No verdict for the current candidate ref, no promotion.
- A resubmission is a new candidate; old verdicts have no power
  over it.
- Conflict is a named event, never a merge strategy.
- Dead claims expire at read, requeue, and fence their late
  submits by generation — the store refuses, not politeness.
- `storage/migrations.py` is frozen; bench tables are
  runtime-created DDL beside it, never in it.
