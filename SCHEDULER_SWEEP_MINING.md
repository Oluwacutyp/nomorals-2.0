# Scheduler Sweep — External Mining

Module: `nomorals/scheduler/` (`__init__.py`, `recurrence.py`, `scheduler.py`).
Method: mined first, then built. Every item below cites a real, verified source.
Nothing here is invented.

## 1. APScheduler — misfire handling, overlap, listeners
**Source:** APScheduler 3.x docs + real-world ADRs
(https://github.com/getdango/dango/blob/HEAD/docs/decisions/ADR-002-apscheduler-over-celery.md,
https://github.com/solidrhino/ocm/blob/HEAD/CLAUDE.md)

- `misfire_grace_time` — how long after the scheduled time a job may still run;
  beyond it the run is skipped. Repo equivalent exists (`stale_after_s`) ✓.
- `coalesce=True` — multiple missed runs fold into ONE execution on recovery
  (prevents flood after downtime). Repo equivalent exists (`fire_now`) ✓.
- `max_instances=1` — same job never runs concurrently; a long run makes the
  next trigger skip. Repo equivalent exists (`overlap_policy="skip"`) ✓.
- Event listeners (`EVENT_JOB_EXECUTED`, `EVENT_JOB_ERROR`, `EVENT_JOB_MISSED`,
  `EVENT_JOB_MAX_INSTANCES`) for execution-history tracking. Repo has
  `fired/failed/missed/dead/deferred` ✓.
- **`replace_existing=True` — gap → gold:** safe reload / idempotent job add.
  Repo currently lets a duplicate `task_id` die with a raw
  `sqlite3.IntegrityError`. Sweep adds a clear `ValueError("task_id … already
  exists")`.
- Liveness lesson from a real incident writeup
  (https://github.com/mikeotoole/pullbackup/commit/a9b2f40f9863322130aaf3d7ceff8dd6df6cdf54):
  a job can show a healthy *future* `next_run` while nothing actually executes
  (admission denied at execution, trigger keeps advancing). The signal that
  catches it is **elapsed silence vs cadence**, not next_run. → Sweep adds
  per-task silence info to `health()` (`last_ok_run_at` age vs cadence).

## 2. Quartz — misfire instructions, trigger model
**Source:** Quartz docs/tutorials
(https://github.com/quartz-scheduler/quartz-scheduler.org-site/blob/HEAD/documentation/quartz-1.8.6/tutorials/TutorialLesson06.md,
https://www.quartz-scheduler.net/documentation/quartz-4.x/tutorial/crontriggers.html,
https://howtodoinjava.com/spring-boot/spring-boot-quartz-scheduler/)

- CronTrigger misfire instructions: `SmartPolicy` (= `FireOnceNow`), `DoNothing`,
  `IgnoreMisfires` (replay every missed firing). Repo has fire_now/skip ✓.
- One job, **multiple triggers** ("register both triggers to run the same job"
  for complex schedules). Repo: one recurrence per task — acceptable; the sweep
  instead closes the expressiveness gap with NL→cron and human descriptions.
- `L`/`W`/`#` cron specials — repo's `recurrence.py` already implements all ✓.

## 3. Temporal Schedules — overlap policies, jitter, catchup, pause-on-failure
**Source:** Temporal CLI docs
(https://github.com/atighe/temporal-documentation/blob/HEAD/docs/cli/schedule.mdx,
https://github.com/atighe/temporal-documentation/blob/HEAD/docs/cli/cmd-options.mdx)

- Overlap policies: `Skip, BufferOne, BufferAll, CancelOther, TerminateOther,
  AllowAll`. Repo has `concurrent/skip/queue` (queue ≈ BufferAll) ✓.
- **`--jitter` — gap → gold:** "Max difference in time from the specification.
  Vary the start time randomly within this amount." Prevents thundering-herd
  when many schedules share a slot. Sweep adds `jitter_s` to cron/rrule jobs.
- `catchup_window` — repo has `stale_after_s` ✓.
- **`--pause-on-failure` — gap → gold:** pause the schedule after a failure so a
  human can inspect instead of retry-spamming. Sweep adds `pause_on_failure`.

## 4. Kubernetes CronJob — history limits, deadlines, idempotency
**Source:** K8s guides
(https://github.com/sandeepk24/learn-devops-playbook/blob/HEAD/ckad/kubernetes_cronjob_101_guide.md,
https://dev.to/4thwithme/cron-jobs-the-tiny-line-that-runs-half-your-backend-3d5m,
https://github.com/r97221004/k8s-tutorial/blob/HEAD/docs/core-objects/job-cronjob.md)

- `startingDeadlineSeconds` — missed starts older than this are skipped.
  Repo equivalent (`stale_after_s`) ✓.
- `concurrencyPolicy: Allow|Forbid|Replace` — repo ✓ (no Replace; killing a
  running thread is unsafe in-process — deliberately not ported).
- **`successfulJobsHistoryLimit` / `failedJobsHistoryLimit` /
  `ttlSecondsAfterFinished` — gap → gold:** bounded history of *finished*
  objects. Repo bounds per-task *run* rows (`run_history_limit`) but finished
  *task* rows (completed/cancelled/missed/dead) grow forever. Sweep adds
  `prune_terminal(older_than_s)`.
- `activeDeadlineSeconds` (kill hung runs) — repo has `timeout_s` ✓.
- "CronJobs are at least once — write jobs so running twice is harmless" →
  documented on `schedule_once`/`schedule_cron`.

## 5. systemd.timer — jitter, persistence, accuracy
**Source:** systemd.timer skill refs + man-page summaries
(https://github.com/pvillega/claude-templates/blob/HEAD/plugins/ct/skills/systemd/SKILL.md,
https://github.com/rojman1984/securebot/blob/HEAD/skills/systemd-timer/SKILL.md,
http://dev.to/lyraalishaikh/stop-fighting-cron-practical-systemdtimer-units-on-linux-1d5m)

- **`RandomizedDelaySec=` — gap → gold:** "add a fresh random delay in
  `[0, value]` before each firing" to spread fleet-wide stampedes
  (`FixedRandomDelay=` makes it stable per machine). Same gold as Temporal
  jitter; sweep implements uniform `[0, jitter_s]` per firing.
- `Persistent=true` — fire once on boot for a missed calendar elapse.
  Repo has `catch_up_on_startup` ✓.
- `AccuracySec=` — coalescing window for power saving (repo ticks every
  second; deliberately not ported — a personal scheduler wants accuracy).

## 6. dateutil rruleset — exdate/rdate, between()
**Source:** dateutil docs
(https://dateutil.readthedocs.io/en/stable/_modules/dateutil/rrule.html,
https://github.com/dateutil/dateutil/blob/master/docs/rrule.rst)

- **`rruleset.exdate()` / `.rdate()` — gap → gold:** exclusion dates win over
  every inclusive rule ("Dates which are part of the given recurrence rules
  will not be generated, even if some inclusive rrule or rdate matches them").
  This is how Google Calendar deletes one instance. Sweep adds
  `exdates`/`rdates` to `RRule` (+ a `parse_rrule_set` for multi-line
  `RRULE:/EXDATE:/RDATE:` text) and per-job `skip_dates` in the scheduler.
- **`between(after, before)` — gap → gold:** bounded occurrence listing, the
  primitive behind "show me the next 5". Sweep adds `RRule.between()`,
  `RRule.next_n()`, and `Scheduler.preview(task_id, n)`.

## 7. cron-descriptor — human-readable schedules
**Source:** cron-expression-descriptor ports
(https://github.com/illegitimis/cron-expression-descriptor,
https://github.com/monksoul/cron-expression-descriptor)

- `GetDescription("30 11 * * 1-5")` → "At 11:30 AM, Monday through Friday".
  The canonical UX gold: never show a raw cron string to a user when you can
  describe it. **Gap → gold:** repo has zero humanization. Sweep adds
  `describe_cron()` / `describe_rrule()` (implemented natively, stdlib-only —
  no new dependency, per the repo's zero-mandatory-deps rule) plus
  `Scheduler.describe(task_id)` producing chat-ready lines like
  "⏰ Every weekday at 9:00 AM · Africa/Lagos · next: today 09:00".

## 8. parsedatetime / dateparser — natural-language datetimes
**Source:** (https://github.com/bear/parsedatetime,
https://github.com/Workable/python-dateparser)

- `cal.parse("tomorrow")` / `parse("in 20 minutes")` — the pattern users
  actually type ("alert me in X minutes" is a live routing gap on the phone
  bot). **Gap → gold:** repo accepts only `datetime`/cron/RRULE. Sweep adds
  stdlib-only `parse_natural_datetime()` ("in 20 minutes", "tomorrow at 8am",
  "next monday at 9", "tonight at 10pm", …) and `parse_natural_schedule()`
  ("every weekday at 9am" → cron, "every 2nd tuesday" → rrule), wired into
  `create_reminder(text, due_at="in 20 minutes")` and
  `schedule_cron(task_id, cron_expr="every weekday at 9am")`.

## 9. Reminder/hook UX gold (Due app, PagerDuty, MongoDB)
- **Nag / auto-snooze** (Due app "Auto Snooze"): a reminder that re-fires every
  N minutes until acknowledged. **Gap → gold:** `nag_every_s` + `max_nags` on
  `create_reminder`.
- **Do-Not-Disturb / quiet hours** (PagerDuty suppression, phone DND):
  **gap → gold:** `quiet_hours=("23:00","07:00")` on reminders/one-time tasks —
  a firing inside the window is shifted to the window end, recorded, and a
  `deferred` event fires.
- **Hook debounce** (every webhook platform ever): **gap → gold:**
  `cooldown_s` on event hooks — at most one firing per window; excess
  triggers are recorded as throttled, not dropped silently.
- **MongoDB query combinators** (`$or`/`$and`/`$not`): **gap → gold** in
  `evaluate_conditions` — currently AND-only across keys.

## What was deliberately NOT ported
- K8s `Replace` concurrency policy (killing a running in-process thread is
  unsafe; `skip`/`queue` cover the need).
- systemd `AccuracySec` coalescing (personal scheduler wants exactness).
- Airflow `catchup=True` full replay (repo's coalesce-once is the safer
  default for a personal agent; replay-everything is a footgun here).
- Quartz multi-trigger-per-job (complexity without a user ask; NL + RRULE
  cover the expressiveness).
