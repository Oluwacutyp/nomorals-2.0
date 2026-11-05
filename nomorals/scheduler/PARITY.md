# Scheduling parity ledger — Echelon Wave 5

What the Wave 5 consolidation found, what it changed, and what is
proven where. Base: `a024911` (Devon main). Proof suites:
`tests/test_cron_proof.py` (engine vs independent oracles) and
`tests/test_scheduler_facade_parity.py` (facade vs incumbents on
twin stores).

## Survey truth

* **One real cron engine already existed**: `recurrence.py`
  (`CronSpec` / `next_cron` / `prev_cron` / `RRule`). Every surface
  delegates to it — the user-facing `CronParser`
  (`scheduler.py`), the agents scheduler's `parse_schedule_spec`,
  and the unified scheduler's `_cron_next`. No second next-fire
  implementation was hiding in the in-scope surfaces (one out-of-scope
  exception below).
* **The facade existed but had zero production consumers** (only a
  skiplist test referenced it), and its docstring overstated it:
  `parse_schedule_spec` was described as a passthrough to the agent
  parser while the code ran a local minimal parser with divergent
  result shapes. The Wave 5 work made the docstring true: the full
  parser is injectable (`spec_parser=`), the agent scheduler is
  injectable (`agent_scheduler=`) with its policy surface delegated,
  and the divergence is pinned by tests instead of prose.
* **The unverified part was the proof standard**, as scoped: there
  was no exhaustive cron suite. There is now (two independent
  oracles, grammar sweep, DST battery, leap/month-end battery).

## Engine defects found by probing the pristine base — all fixed

1. **Leap horizon**: `next_cron` searched at most ~366 days, so
   `0 0 29 2 *` raised "never matches within a year" even though it
   fires (2028, and 2104 across the non-leap year 2100). Search is
   now day-granular over 2930 days — the proven maximum gap for any
   satisfiable 5-field expression (Feb 29 across a non-leap century:
   2096 → 2104 = 2922 days) — so exhaustion means *impossible*, and
   impossible expressions raise the dedicated
   `CronNeverMatchesError` instead of sharing a message with the
   merely-far-away.
2. **Silent timezone fallback**: an unknown zone was ignored and the
   job computed in UTC. Unknown zones now raise `CronTimezoneError`
   (message mirrors the incumbent `unknown timezone: {name!r}`) from
   `next_cron` / `prev_cron` / `validate_cron`, and the user-facing
   `_validate_tz` raises the same type.
3. **DST gap phantom**: in the spring-forward gap the engine
   returned an instant whose wall time did not satisfy the
   expression. Gap wall times are now skipped — no firing that day.
4. **DST overlap past-instant**: after a pivot inside the second
   pass of a fall-back overlap, the engine returned the *first*
   occurrence — an instant in the past. Overlap semantics are now:
   first occurrence, unless the pivot already sits between the two
   occurrences, in which case the second is next (and `prev_cron`
   mirrors it). The returned instant localized always satisfies the
   expression (pinned as an invariant across the proof suite).
5. **Grammar gaps vs the module's own documented grammar**:
   `*/n` steps were refused in the day-of-month / day-of-week
   fields, and `a/n` was silently read as `{a}` instead of Vixie
   `a..max step n`. Both now follow classic cron.

Schedule-time honesty: `validate_cron` (parse + zone + proof
search) is called at construction/schedule time by
`UnifiedJob.__post_init__` (cron recurrences), the user-facing
`Scheduler.schedule_cron` and its reschedule path, and the facade
passthrough — garbage, impossible, and bad-zone expressions are
refused by name before a job exists.

## Facade consolidation status

| Surface | Status | Evidence |
|---|---|---|
| Recurrence engine delegation | All in-scope surfaces delegate to `recurrence.py` | imports + `test_cron_proof.py` |
| `parse_spec` parity (injected full parser) | Exact, incl. identical refusals | `ParseSpecParityTests` corpus (19 specs) |
| `parse_spec` uninjected | Documented minimal subset; missing agent scheduler raises by name | `ParseSpecParityTests.test_uninjected_*` |
| Agent jobs via facade | add outcomes, stored fields, policy validation identical | `AgentJobParityTests` |
| Missed-fire policies `fire_now` / `skip` / `next_only` | Identical outcomes on twin DBs | `AgentJobParityTests.test_missed_fire_outcomes_agree` |
| Overlap policy | Round-trips identically via facade `add` (agent surface) | `AgentJobParityTests` (`pol` case + validation) |
| Unified jobs via facade | First `next_run` exact; missed verdicts `catch_up_once` / `skip` / `backfill` exact (jitter pinned 0) | `UnifiedParityTests` |
| Cron refusal via facade | `CronNeverMatchesError` / `CronTimezoneError` propagate | `CronHonestyTests` |

Nothing was deleted. Incumbents keep their own APIs; the facade
fronts them in place.

## Ranked gaps carried (not fixed here)

1. **CLOSED (Wave 5 follow-up) — `nomorals/storage/queue.py`'s
   divergent sixth cron engine is retired.** It had its own
   `parse_cron` / `next_cron_fire` with AND semantics for dom+dow
   (classic cron is OR), local-time only, its own minute-stepping,
   and a silent `or 0.0` fallback when no firing was found. Queue's
   cron surface now delegates to `recurrence.py`: `parse_cron`
   returns the canonical `CronSpec`, `next_cron_fire` /
   `schedule_recurring` / `tick_recurring` take an explicit
   `timezone` (IANA name; the empty default keeps the caller's
   system-local footing, never a silent UTC), `schedule_recurring`
   proves schedulability at registration via `validate_cron`, and
   never-matching expressions raise `CronNeverMatchesError` by name
   instead of being stored dead. `tick_recurring` heals legacy rows
   in place (schema unchanged): dead-zero rows are recomputed or
   reported by name in the failure ledger; AND-computed next fires
   are tightened to the canonical OR value. The layer cost is one
   pin: `nomorals.scheduler.recurrence` is pinned at L2 in
   `tests/test_layering.py` (a stdlib-only leaf; the pin rule
   0 ≤ 2 ≤ 2 holds once storage.queue imports it). Survey facts:
   the recurring surface had no production callers (only tests),
   and no persisted database on the build host holds a
   `recurring_jobs` row — the EC2 store, if any, is covered by the
   tick-time healing. Proof: `tests/test_queue_cron_parity.py`
   (corpus parity vs the canonical engine, delta pins, healing
   tests) + the updated sweep pins.
2. **Two classes named `UnifiedScheduler`** — `scheduler/
   unified_scheduler.py` (the facade's backend) and
   `autonomy/unified.py` (organ ticks / missions). Name collision
   only; no shared state. Renaming is a wider refactor than this
   wave's surface.
3. **Dead code**: `agents/scheduler.py::_parse_cron_field` has no
   callers (definition site only). Left in place per the
   never-delete rule; flagged here.
4. **`UnifiedJob.from_dict` silently drops unknown keys** — normal
   schema filtering for a machine store, recorded for visibility;
   behaviour unchanged.

## Markdown-store invariant

The Wave 5 plan row mentions the markdown-store unknown-lines
invariant "where applicable". It is **not applicable**: every
scheduling store here is JSON (`UnifiedScheduler`) or SQLite
(user-facing and agents schedulers). No markdown-backed schedule
store exists, so there is nothing to preserve unknown lines in.

## Verification status

* Proven in tests (this wave): engine next/prev vs two independent
  oracles across the grammar sweep, DST transitions for
  America/Denver + Australia/Lord Howe (30-minute shift) +
  Africa/Lagos anchor, leap/month-end edges, named refusals;
  facade parity on twin stores for all rows above; the incumbent
  scheduler neighbourhood suite passes unchanged against the
  pristine-base baseline.
* Not verified live: EC2 behaviour after the owner's pull — as with
  every wave, "shipped" means pushed + verified on the remote;
  "live" waits for the pull + restart.
