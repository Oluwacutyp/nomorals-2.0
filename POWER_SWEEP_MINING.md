# POWER_SWEEP_MINING.md

External mining for the `nomorals/power` sweep (battery/thermal/degradation-aware
scheduling: `PowerMonitor`, `PowerStatus`, `PowerAwareScheduler`, `PowerTask`,
`PowerError`). Sources are real, verified in Oct 2026. Gold that lands below
is implemented; gold that doesn't fit the layer contract (L5, no upward
imports, zero mandatory deps) is noted.

## 1. Linux kernel — Energy Aware Scheduling (EAS)

Source: https://docs.kernel.org/next/scheduler/sched-energy.html

- The kernel keeps an **Energy Model (EM)** per performance domain
  (per-cluster OPP table of capacity vs power cost, visible at
  `/sys/kernel/debug/energy_model/`).
- `find_energy_efficient_cpu()` places a waking task on the CPU with the
  **highest spare capacity** (capacity − utilization) in each domain, then
  `compute_energy()` simulates the placement against the EM to check the
  migration actually saves energy vs. leaving the task where it ran.
- PELT (per-entity load tracking) signals feed the decision; schedutil
  picks the DVFS operating point from the same utilization signal.

Gold taken → `EnergyModel` in `nomorals/power/scheduler.py`: per-task-type
historical energy/duration ledger (`energy_wh`, `duration_s` learned from
`complete()`), `estimated_cost(task_type)` advisory, and poll-time ordering
of the batch by ascending estimated cost (cheapest runnable work first —
userspace analog of spare-capacity placement). Also `energy_per_hour()`
projection from the monitor's drain rate so dispatch can warn "this batch
will drain ~X% at current rate".

## 2. Android WorkManager / JobScheduler constraints

Sources:
- https://developer.android.com (WorkManager `Constraints`: requiresCharging,
  requiresBatteryNotLow, requiresDeviceIdle, requiredNetworkType)
- https://medium.com/@promode7/22-28-workmanager-isnt-magic-it-s-jobscheduler-underneath-010a91d9e43e
- https://github.com/wrsilva/reis-mobile/blob/HEAD/skills/mobile-performance/references/background.md

- Contract: work is **guaranteed eventually, never immediately**; the system
  owns the second clock ("when conditions are met").
- Tasks declare constraints; the OS batches them and runs during **Doze
  maintenance windows**. Expedited work (`setExpedited`) is the escape hatch
  for user-waiting work.
- Rule of thumb from the mobile-performance skill: "Say what the work
  needs, let the system choose when." Never poll on a timer.

Gold taken → `PowerTask.constraints` (`requires_charging`,
`requires_battery_not_low`, `requires_idle`): poll-time gating against the
live `PowerStatus` (charging flag, battery %, idle hint). Deferred tasks are
requeued with jittered backoff instead of dropped — the same
guaranteed-eventually contract this module already promises, now with
constraints like the real platform APIs.

## 3. Kubernetes QoS classes + priority preemption

Sources:
- https://github.com/fmarani/blog/blob/HEAD/posts/kubernetes-qos-and-priority-classes.md
- https://github.com/srinivassarkar/cka_prep/blob/HEAD/notes/30_pod_priority.md

- QoS (Guaranteed/Burstable/BestEffort) is a **runtime eviction** construct:
  under memory pressure the kubelet evicts BestEffort → Burstable →
  Guaranteed, independent of priority.
- Priority decides scheduling order AND preemption (a higher-priority pod
  may evict lower-priority ones); `preemptionPolicy: Never` marks work that
  prefers waiting over disrupting.

Gold taken → the tier ladder already mirrors eviction order (bulk shed
first, critical never). Added: **degradation transition events** on the
global bus (`power.degradation.changed`), an explicit `preemptible` flag on
PowerTask (non-preemptible critical tasks are never requeued for budget
pressure — the `preemptionPolicy: Never` analog inverted), and
`deferred_tasks()` inspection so operators can see what's waiting.

## 4. psutil — real battery + thermal sampling primitives

Source: https://github.com/giampaolo/giampaolo.github.io/blob/HEAD/content/blog/2017/psutil-510-system-temperatures-battery-cpu-freq.rst

- `psutil.sensors_battery()` → `sbattery(percent, secsleft, power_plugged)`.
- `psutil.sensors_temperatures()` → per-chip `shwtemp(label, current, high,
  critical)` (e.g. `coretemp` per-core).

Gold taken → `nomorals/power/monitor.py::local_sampler()`: a real,
dependency-optional sampler. Prefers psutil (import-guarded); falls back to
**sysfs** (`/sys/class/power_supply/*/capacity|status`, per the kernel ABI
documented in `Documentation/ABI/testing/sysfs-class-power`); maps the
max core temperature against `high`/`critical` thresholds to
nominal/warm/hot. Degrades to "unknown" (fail-open) when neither is
available — zero mandatory deps preserved.

## 5. UPower / sysfs battery interfaces (Linux)

Sources:
- https://github.com/marrionesa/tauri-plugin-power-monitor/blob/HEAD/RESEARCH.md
- https://github.com/dara-oladapo/power-helper/issues/4 (sysfs layout)

- UPower D-Bus: `org.freedesktop.UPower.Device` on
  `/org/freedesktop/UPower/devices/DisplayDevice` — `Percentage`,
  `State` (1 charging, 2 discharging, 4 fully charged...),
  `TimeToEmpty`/`TimeToFull`, `PropertiesChanged` signals.
- sysfs fallback: enumerate `/sys/class/power_supply/*` by `type`
  (`Battery`/`Mains`), read `capacity`, `status`
  (Charging/Discharging/Full/Not charging), `online`, `energy_now`+`power_now`
  for real time-remaining. Don't hardcode `BAT0`.

Gold taken → the sysfs fallback in `local_sampler()` follows exactly this
enumeration (type-file driven, multi-battery aware). Not taken: D-Bus
subscription — needs a D-Bus client lib and a listener thread; out of scope
for the layer contract, and polling already matches the engine's heartbeat
model.

## 6. AWS "Exponential Backoff and Jitter" (Marc Brooker)

Sources:
- https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- https://docs.aws.amazon.com/sdkref/latest/guide/feature-retry-behavior.html
  (`delay = random(0,1) × min(cap, base_delay × 2^retry)`, 20s cap)
- https://github.com/ivan-podgurskiy/jitter (No/Full/Equal/Decorrelated)

- Full jitter: `sleep = random_between(0, min(cap, base * 2 ** attempt))` —
  recommended default; expected retry rate stays flat under contention.
- Decorrelated jitter: `sleep = min(cap, random_between(base, prev * 3))` —
  better when retries last minutes.
- Fixed delays don't break the herd; exponential spreads them; jitter
  dissolves the schedule.

Gold taken → `nomorals/power/backoff.py`: `BackoffPolicy` with full/equal/
decorrelated jitter, base/cap, and **injectable RNG** for deterministic
tests (the ivan-podgurskiy/jitter pattern). Replaces the module's fixed
30s/60s deferral backoffs. Per-task deferral counters feed the attempt
number, so a task deferred 10 times backs off exponentially, capped.

## 7. Aging / starvation prevention (classic scheduling theory)

Classic multilevel-feedback / aging result: a waiting task's effective
priority must rise with wait time or low tiers starve under sustained
pressure. Kubernetes' own docs warn BestEffort workloads can wait
indefinitely.

Gold taken → deferral ledger on each task (`deferred` count persisted in
the job payload): `max_deferrals` per dispatch; past the cap the task
**escalates** — emitted as `power.task.escalated`, optionally tier-bumped,
and poll admits it as overdue (`force` flag). Nothing deferred forever.

## 8. Battery drain history → local forecast

The drain-rate regressions in battery-logging practice (coulomb counting
from `energy_now`, linear fits on recent samples).

Gold taken → `PowerMonitor.record()` keeps a bounded rolling history of
(percent, charging) samples; `drain_rate_pct_per_h` is computed locally
when the sampler's advisory lacks it, and `can_sustain(pct_budget,
minutes)` answers "can a heavy batch of this size finish before the
battery gives out". Hysteresis: band transitions require N consecutive
consistent readings (`stability_reads`) so a single noisy sample can't
flap the deferral plan.

## 9. Human-readable status cards

Not from one paper — from operational practice (UPower `upower -i` output,
`kubectl describe`, psutil's `scripts/sensors.py`): operators read a
rendered card, not a dict.

Gold taken → `PowerStatus.format()` renders a compact text card (battery
bar, thermal, degradation ladder, tier flow, time-to-empty) for chat/CLI
surfaces, and `PowerAwareScheduler.summary()` renders queue state
(pending/deferred per class, escalations, energy ledger).

## Not taken (deliberate)

- UPower D-Bus push signals: needs a D-Bus client + thread; heartbeat
  polling matches this engine.
- DVFS/freq control (`schedutil` analog): writing sysfs governors needs
  root and is a node-management concern, not a task-scheduling one.
- True coulomb-counted energy per task: needs per-process energy telemetry
  (RAPL) — the historical-duration ledger is the portable stand-in.
