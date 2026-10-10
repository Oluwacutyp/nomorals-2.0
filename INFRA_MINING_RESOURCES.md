# Infrastructure Mining — Resource Management (Section C)

Mined 2026-10-09 before the Section C build. Best AND trash builds studied;
the gold below is what got implanted. Full-capability design: the
ResourceManager exposes the FULL signal surface on every platform, and the
profile system (`resource_profile`) decides what to use where — nothing here
is designed down to any deployment target.

## 1. Pressure Stall Information (PSI) — the best kernel signal

`/proc/pressure/{cpu,memory,io}` (kernel 4.20+, may need `psi=1` on the
cmdline on some distros). Each file reports stall time — the share of
wall-clock time tasks were DELAYED waiting for a resource — as `some` and
`full` percentages over avg10/avg60/avg300 windows:

- `some` = at least one task stalled while others kept running
- `full` = ALL runnable tasks stalled simultaneously (zero productive cycles;
  only memory/io have `full`, cpu can't because any running task = progress)

Gold: PSI measures *experienced* stalls, not utilization. A box at 98% CPU
can be healthy; a box at 15% CPU can be frozen on I/O. Utilization lies;
PSI doesn't. `memory full avg10` in double digits = the box is effectively
down. systemd-oomd already watches PSI (kills at ~50% over 20s on a slice).
Sources: kernel PSI docs via netdata's collector notes; dev.to incident
runbooks confirming PSI beats `%wa`/loadavg for "who is hurting".

Implanted: `sample_psi()` parses all three files into `PressureStall`
(some/full × avg10/60/300); memory PSI feeds the composite score and the
predictive layer.

## 2. MemAvailable is predictive, PSI is reactive — need both

From the `oma` admission-control build (baulehrer/omarag, a small shell-side
memory guard for llama.cpp, public commit):

- Two numbers answer different questions. **MemAvailable is predictive** —
  "does this load fit?" **PSI is reactive** — "is anything stalling right
  now?" With heavy swap, MemAvailable can look calm while everything
  thrashes, and PSI catches it.
- **Admission control, not just throttling**: `admit.py` refuses to start
  work that won't fit ("load only what fits, give it back when asked").
- **Act below the killer**: it releases idle at PSI 10% over 5s and stops
  starting work at 25% over 10s — well below systemd-oomd's 50%/20s, so it
  finishes letting go before the kernel has to choose a victim.
- **Reserve margin**: default 6% of RAM, never under a floor (2 GiB there) —
  the same guard behaves on a laptop and a workstation.

Implanted: `ResourceBudgets.acquire()` is admission control ("does it fit,
including the reserve?"); reserve defaults 6% of effective RAM with a
configurable floor; consult warns below the oomd band instead of at it.

## 3. lmkd-linux — composite score, trend, damped state machine

`aquilesorei/lmkd-linux` (userspace memory-pressure manager, MIT-ish
project) is the best small-system design found:

- **Composite pressure score** instead of one metric: 55% PSI-memory stall
  + 20% swap saturation + 15% UMA/GPU residency + 10% swap I/O rate.
  Catches fast-onset thrashing before the 10s PSI average reacts.
- **Pressure trend (dP/dt)**: velocity of pressure change accelerates
  reaction to spikes while ignoring transient blips.
- **Damped state machine**: upward transitions need pressure to persist for
  ≥2 ticks (unless a critical spike); downward transitions need SUSTAINED
  calm (1–2 min) to prevent oscillation/flapping.
- **Escalating action ladder**: ELEVATED → compact/freeze low-priority;
  HIGH → SIGTERM expendables; CRITICAL → checkpoint-or-kill; EMERGENCY →
  SIGKILL non-critical. Never binary ok/broken.

Implanted: `composite_pressure()` (PSI-memory 55% / swap saturation 20% /
mem utilization 15% / swap-I/O rate 10%); `trend()` (least-squares slope
over sample history); `DegradationLadder` with damped up/down transitions
and 5 levels; consult emits `degradation_level`.

## 4. CPU steal time — the virtualization signal

`/proc/stat`'s `cpu` line carries `steal` (jiffies the vCPU was ready but
the hypervisor scheduled elsewhere). Delta-sampled steal % is the direct
signal for host contention; on burstable families (AWS T2/T3) sustained
steal + sustained load ≈ credit depletion (CloudWatch `CPUCreditBalance`
is the ground truth but needs network + credentials; steal is the local
proxy). Also: `loadavg` counts runnable + UNINTERRUPTIBLE tasks, so high
load ≠ CPU busy — check D-state / pair with real utilization from
idle/total deltas. Sources: AWS re:Post on T3 credit monitoring; SRE
incident writeups on steal spikes; Linux runbook guidance.

Implanted: delta-based `cpu_util_percent` (idle/total, more truthful than
loadavg on burstable/steal) AND `steal_percent`; consult flags sustained
steal as its own reason ("hypervisor contention — likely credit-exhausted
on burstable types").

## 5. Cgroup awareness — /proc/meminfo lies in containers

`/proc/meminfo` is not namespaced: inside a container it reports the
HOST's RAM. Real-world failure (crawlee-python issue #2095): autoscaler
scaled past the container limit and got OOM-killed because budgets came
from host RAM. The correct probe order (spiceai spec, crawlee-JS
implementation, flapi proposal):

1. **v2**: walk `/proc/self/cgroup` (`0::<path>`), take min of
   `memory.max` from leaf up through ancestors (ancestor `MemoryMax=` on
   the slice binds even when the leaf says `max`). Literal `max` or values
   ≥ 2^62 = unlimited, skip.
2. **v1**: same walk over `memory.limit_in_bytes` (sentinel LONG_MAX =
   unlimited).
3. Fallback: host `MemTotal`.
4. Effective = `min(cgroup_limit, host_ram)`.

Same idea for CPU: v2 `cpu.max` = `<quota> <period>`; v1
`cpu.cfs_quota_us / cpu.cfs_period_us`; effective cores =
`min(quota_cores, cpu_count())`.

Implanted: `effective_memory_mb()` / `effective_cpu_count()` with exactly
this walk; all budget sizing uses effective resources, never raw meminfo.

## 6. Battery: status + drain rate, not just capacity

Kernel ABI (`Documentation/ABI/testing/sysfs-class-power`): per supply,
`type` (Mains/Battery), `status` (Charging/Discharging/Not
charging/Full/Unknown), `capacity`, `power_now` (µW), `energy_now`,
`energy_full`. A raw capacity % is nearly useless for scheduling: what
matters is whether it's CHARGING (throttle rules relax) and the drain
rate. BatteryScope/batrun pattern: estimated time-to-empty = remaining
energy / rolling-average power draw (rolling, not instantaneous, so it's
stable under spikes); charging periods excluded from drain averages.

Implanted: `battery_status` on the sample; `ResourceManager.battery_forecast()`
computes drain %/h and time-to-empty from the sample history; monitor's
`PowerStatus` carries `charging`, `drain_rate_pct_per_h`,
`time_to_empty_min`. Plugged-in (charging) relaxes deferral.

## 7. Degradation tiers & load shedding

Industry pattern (several independent writeups converge): priority buckets
CRITICAL > DEGRADED/IMPORTANT > BEST_EFFORT > BULK/BACKGROUND; shed
progressively by CPU thresholds; critical paths stay thin; plus slow-start
recovery (after recovery, re-accept work on a ramp, e.g. 10%→100% over
minutes) so a recovering system isn't stampeded.

Implanted: 5-level ladder (FULL → LIGHT_SHED → DEGRADED → SURVIVAL →
ESSENTIAL) with per-level allowed task tiers; damped recovery =
slow-start: levels step down one at a time after sustained calm.

## 8. Trash builds — what NOT to do

- **Loadavg-only monitors**: `load1/ncpu` as the single CPU signal. Blind
  to steal, blind to D-state, and on burstable VMs it punishes the guest
  for the host's scheduling. Reacts after the stall.
- **Host-RAM budgets in containers**: the crawlee-python failure above —
  any budget computed from `/proc/meminfo` without the cgroup walk is
  wrong under Docker/Kubernetes.
- **Battery % without status**: throttling a phone that's plugged in and
  charging at 25% is pure waste; throttling a laptop at 25% discharging
  is survival. Same number, opposite action.
- **Binary ok/throttled with no hysteresis**: flaps under oscillating load;
  deferral queues churn. The damped machine fixes this.
- **No per-subsystem accounting**: one global "memory is fine" while the
  LLM and the media pipeline each assume they own all of it.

## 9. Environment detection signals (stdlib-only)

- Cloud: `/sys/devices/virtual/dmi/id/{sys_vendor,product_name}` —
  "Amazon EC2", "Google Compute Engine", "Microsoft Corporation"+`Virtual
  Machine`; also `/sys/hypervisor/uuid` starting with `ec2` on Xen-based
  EC2. Board vendor + bios version as corroboration.
- Container: `/.dockerenv`, `/run/.containerenv`, `/proc/1/cgroup` or
  `/proc/self/cgroup` containing `docker`/`kubepods`/`containerd`.
- Termux/Android: `PREFIX` env containing `com.termux`; `platform.system()
  == "Android"` (PEP 738: Termux Python 3.13+ reports `android`).
- Virtualization: `hypervisor` flag in `/proc/cpuinfo`, `/sys/hypervisor/
...[truncated 1166 chars]