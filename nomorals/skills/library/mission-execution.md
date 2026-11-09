---
id: mission-execution
title: Mission execution — durable long work
summary: How long-running work is planned, driven, checkpointed and finished as a mission.
topics: missions, autonomy, long-running, planning
triggers: mission, long task, background job, multi-step task
layer: missions
version: 1
---

# Mission execution

A mission is the unit of work that outlives a single run. It persists its
plan, progress and budget to SQLite, so a crash costs one step, not the
whole job.

## Shape

- Create through `MissionStore`, drive with `MissionRunner`
  (`nomorals.missions`). The runner is composed by
  `agency.wiring.compose_agency` / `missions.wiring.wired_runner` — never
  hand-roll a second runner.
- Every plan step names an explicit role from the real role catalog
  (`agents.roles.ROLES`); a step with no role, or an unknown role, is a
  plan defect — there is no silent default role.
- Steps declare dependencies; the runner executes level by level and
  journals every transition to the autonomy ledger.

## Rules that bite

- Acceptance is verified at the terminal point: a mission whose
  acceptance checks fail ends FAILED with the reason, never DONE.
- Delegated work under a work contract gates completion the same way —
  open contracts refuse the DONE verdict (see accountable-delegation).
- Side effects go through the idempotency store: durable-before-effect,
  lease-fenced, exactly-once. Never perform an external effect inline in
  a step without a claim.
- Budgets are real: wall-clock and step budgets stop the mission; report
  the stop honestly instead of shrinking the goal.
