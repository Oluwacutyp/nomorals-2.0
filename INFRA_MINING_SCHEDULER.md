# INFRA MINING — Scheduler (Section B)

Mined 2026-10-09. Sources: APScheduler 3.x/4.x, Quartz/Quartz.NET, systemd
timers (systemd.timer(5)), Kubernetes CronJob (batch/v1), AWS EventBridge
Scheduler, Celery Beat, gocron, Temporal, RFC 5545 RRULE (Google Calendar /
Outlook), node-cron / node-schedule, plus the two in-repo schedulers.

Rule: take the gold from everywhere (even trash builds), leave the dirt.

## The gold, per system

### APScheduler (Python — the reference)
- **misfire_grace_time** (per-job, default from job_defaults): a firing
  missed by less than the grace still runs; older than the grace is a
  misfire. Distinct from "missed entirely".
- **coalesce**: N missed firings collapse into ONE run (no flood on
  restart). We have `fire_now` coalescing — matches.
- **max_instances**: per-job overlap cap. `max_instances=1` = our
  `overlap_policy=skip`/`queue` family; "concurrent" = unbounded.
- **Job stores / executors split**: persistence (SQLAlchemy/Mongo/Redis)
  is separate from execution (thread/process/asyncio pools). Our two
  schedulers both run inline in the tick thread — the pool is the gap.
- **Event listeners**: `EVENT_JOB_SUBMITTED / EXECUTED / ERROR / MISSED`
  — observability hooks as a first-class API, not just logs. Our agents
  scheduler has bus events; the user-facing one has nothing.
- **Timezone-aware triggers** everywhere; `replace_existing=True` on add.

### Quartz (Java — the enterprise reference)
- **Per-trigger-type misfire instructions + "smart policy"**: different
  trigger kinds get different sensible missed-fire behavior by default.
  (We: one `missed_fire_policy` per job — good; smart defaults per kind
  is the extra 10%.)
- **Exclusion calendars**: named holiday calendars a trigger can opt out
  of ("don't fire on public holidays"). Nothing in-repo does this.
- **@DisallowConcurrentExecution**: declarative no-overlap. (We have it
  as data — better.)
- **TriggerListener.vetoJobExecution()**: a listener can cancel a single
  run — observability that can *act*. Gold for future autonomy work.
- **Clustered JDBC job store**: distributed lock so N instances don't
  double-fire. Out of scope single-node, but the *idempotency* lesson
  stands (K8s says it outright: jobs "should be idempotent").

### systemd timers (the OS-level reference)
- **OnCalendar expressiveness**: `~` (days from end of month), ranges,
  repetitions, lists, shorthands — calendar-grade recurrence without a
  second syntax. Cron's `L`/`W`/`#`/`?` are the portable subset.
- **Persistent=true**: last-trigger timestamp stamped on disk; a missed
  window fires ONCE on next activation. Our `catch_up_on_startup` is the
  same idea — ours is per-job policy-driven, which is strictly better.
- **RandomizedDelaySec**: fresh random delay per elapse to de-congest a
  fleet; **FixedRandomDelay**: stable per-timer delay (machine-id hash).
- **AccuracySec**: coalesce wakeups inside a window (power saving) — the
  *opposite* of RandomizedDelaySec, and systemd documents that they do
  opposite jobs. Both knobs, not one.
- **Multiple triggers accumulate** (earliest wins); relative triggers
  (OnUnitActiveSec) — "run X after Y finished", a poor-man's dependency.

### Kubernetes CronJob (the orchestrator reference)
- **concurrencyPolicy: Allow | Forbid | Replace**. *Replace* (kill the
  running one, start fresh) is the policy our `OVERLAP_POLICIES` lacks —
  worth adding.
- **startingDeadlineSeconds**: a missed start older than the deadline
  counts as FAILED, not silently skipped. Missed-fire with teeth.
- **successfulJobsHistoryLimit / failedJobsHistoryLimit**: run retention
  as a first-class field. Our `schedule_runs` grows forever — real gap.
- **suspend**: pause scheduling without touching running executions.
  (Our `set_enabled(False)` covers it.)
- **timeZone** field (GA 1.27); **backoffLimit** + **activeDeadlineSeconds**
  (retry budget + kill deadline per job — our `max_retries` + per-job
  `run_timeout_s` match).

### AWS EventBridge Scheduler
- **Flexible time windows**: "run within 15 min of the nominal time" —
  jitter as a scheduling primitive for de-congestion.
- **Retry policy** (max attempts + max event age) → **dead-letter queue**.
  DLQ as a *destination*, not just a flag: our notifier dead-letters,
  the schedulers should too (failed job → dead-letter row, inspectable).
- **One-time schedules** as a first-class schedule type.

### Celery Beat
- **crontab / interval / solar** schedule objects; `enable_utc`; beat
  keeps a "last run" per entry so only one beat instance fires
  (poor-man's leader election via shelve). django-celery-beat = DB-backed
  dynamic schedules (our DB tables already do this).

### gocron (Go)
- **Singleton mode**, **start/end datetimes** (a job that only lives in a
  window — our `at` + `max_runs` approximates; explicit not-before /
  not-after is cleaner), **tags** for grouping, per-job event listeners.

### Temporal (the durability reference)
- Workflows are *code*; **retry policies with backoff+jitter**,
  **timeouts** (schedule-to-start vs start-to-close — we have only the
  latter), cron schedules as sugar. The lesson: separate "time to start"
  from "time to finish" budgets.

### RFC 5545 RRULE (Google Calendar / Outlook — calendar-grade recurrence)
- `FREQ=DAILY|WEEKLY|MONTHLY|YEARLY`, `INTERVAL`, `COUNT`, `UNTIL`,
  `BYDAY` (with ordinals: `BYDAY=2TU`, `-1FR`), `BYMONTHDAY`, `BYMONTH`,
  `BYSETPOS` ("last Friday": `FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1`),
  `WKST`, `EXDATE`/`RDATE` (exceptions). This is what "calendar-grade
  recurrence" means — cron can't say "2nd Tuesday" or "last weekday".

## The dirt (what to avoid — found in trash builds and bug reports)
- **Memory-only schedules** (node-schedule): everything lost on restart.
- **Naive setInterval drift**: re-arming from "now" accumulates skew;
  always compute next fire from the nominal schedule, not from
...[truncated 2980 chars]