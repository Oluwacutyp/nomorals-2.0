"""The Workspace (wave 84): the machine's virtual CPU farm.

The Workspace owns the set of Virtual CPUs the system runs on:

* **profile-aware sizing** — the count comes from the detected
  environment (termux / mobile / pc / vps / workstation / embedded)
  and its min/target/max envelope, overridable by config;
* **smart dispatch** — every submit is assigned to the best available
  VCPU by a composite score (kind affinity, smoothed busy fraction,
  queue pressure, even-wear tiebreak);
* **autoscaling** — a periodic tick grows the pool when the fleet
  runs hot (sustained high busy fraction, queue pressure) and shrinks
  it when the fleet idles, always staying inside the profile
  envelope and never killing a VCPU with live work (scale-down drains);
* **full observability** — :meth:`status` is the whole farm in one
  dict, and the workspace emits ``workspace.*`` events on the context
  bus so the main AI (and the watch loop) can see it live.

The Workspace is deliberately transport-agnostic: a task is any
callable; results come back as standard Futures, so the Orchestrator,
the mission runner, and any tool can submit work without knowing
which physical core will run it.
"""
from __future__ import annotations

import collections
import math
import threading
import time
import weakref
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import Any

from ..core.profile import EnvironmentProfile, resolve_profile
from .vcpu import KINDS, VcpuStatus, VirtualCPU

__all__ = ["Workspace"]

#: autoscale triggers (HPA-style: proportional, deadbanded, asymmetric)
_SCALE_UP_BUSY = 0.7      # fleet busy fraction that earns new cores
_SCALE_DOWN_BUSY = 0.1    # fleet busy fraction that loses one
_SCALE_DOWN_IDLE_S = 45.0  # must be near-idle this long before shrinking
_SCALE_TOLERANCE = 0.10    # deadband: |ratio - 1| <= tol → no action
_SCALE_UP_MAX_STEP = 2     # rate limit: at most +2 cores per decision
_SCALE_DOWN_MAX_STEP = 1   # ... and -1 per decision (conservative shrink)
_SCALE_UP_PRESSURE = 0.5   # queue pressure that earns new cores
_SCALE_DOWN_PRESSURE = 0.05


class Workspace:
    """The central execution farm."""

    def __init__(self, context: Any = None, *, profile: str = "",
                 vcpus: int = 0, min_vcpus: int = 0, max_vcpus: int = 0,
                 autoscale: bool = True, autoscale_interval: float = 30.0,
                 max_queue: int = 256, auto_start: bool = True,
                 memory_pressure: str = "") -> None:
        self.context = context
        self.profile = resolve_profile(profile)
        #: wave 86: profile-aware memory posture (aggressive|moderate|relaxed).
        #: "aggressive" (phones, tiny boxes) means the autoscaler grows more
        #: reluctantly and shrinks more readily — RAM is the scarcest thing.
        self.memory_pressure = str(memory_pressure or "").strip().lower()
        self.max_queue = max(1, int(max_queue))

        lo = min_vcpus or self.profile.min_vcpus
        hi = max_vcpus or self.profile.max_vcpus
        lo = max(1, min(lo, hi))
        target = vcpus if vcpus else self.profile.target_vcpus
        self.min_vcpus = lo
        self.max_vcpus = max(lo, hi)
        self.target_vcpus = max(lo, min(target, self.max_vcpus))

        self._vcpus: dict[str, VirtualCPU] = {}
        self._next_index = 0
        self._lock = threading.RLock()
        self._autoscale = bool(autoscale)
        self.autoscale_interval = max(5.0, float(autoscale_interval))
        self._last_scale = 0.0
        self._hot_since: float | None = None
        self._idle_since: float | None = None
        self.last_scale_reason = ""
        self.stats = {"scaled_up": 0, "scaled_down": 0,
                      "dispatched": 0, "dispatch_errors": 0,
                      "since": time.time()}
        self._thread: threading.Thread | None = None
        self._stop = False
        self._wake = threading.Event()
        #: recent scale decisions, newest last (observability)
        self._scale_history: collections.deque[dict[str, Any]] = \
            collections.deque(maxlen=32)
        #: (t, busy, pressure) samples for predictive scale-up
        self._metric_history: collections.deque[tuple[float, float, float]] = \
            collections.deque(maxlen=16)

        with self._lock:
            for _ in range(self.target_vcpus):
                self._spawn("balanced")
        if auto_start and self._autoscale:
            self._thread = threading.Thread(
                target=_autoscale_worker, args=(weakref.ref(self),),
                name="nm-workspace-autoscale", daemon=True)
            self._stop = False
            self._thread.start()

    # ── VCPU management ──────────────────────────────────────────────────────
    def _spawn(self, kind: str) -> VirtualCPU:
        index = self._next_index
        self._next_index += 1
        vcpu = VirtualCPU(index, kind=kind, max_queue=self.max_queue)
        self._vcpus[vcpu.name] = vcpu
        self._emit("workspace.vcpu.started",
                   vcpu=vcpu.name, kind=kind, index=index)
        return vcpu

    def _retire(self, vcpu: VirtualCPU, *, drain: bool = True) -> None:
        vcpu.stop(drain=drain, timeout=5.0)
        with self._lock:
            self._vcpus.pop(vcpu.name, None)
        self._emit("workspace.vcpu.stopped", vcpu=vcpu.name)

    def vcpus(self) -> list[VirtualCPU]:
        with self._lock:
            return list(self._vcpus.values())

    def get(self, name: str) -> VirtualCPU | None:
        with self._lock:
            return self._vcpus.get(name)

    def scale_to(self, n: int, *, drain: bool = True,
                 reason: str = "") -> int:
        """Resize the farm to exactly n VCPUs (clamped to the envelope)."""
        n = max(self.min_vcpus, min(int(n), self.max_vcpus))
        with self._lock:
            current = len(self._vcpus)
            now = time.time()
            why = reason or f"scaled to {n} (requested)"
            if n > current:
                for _ in range(n - current):
                    self._spawn("balanced")
                self._last_scale = now
                self.last_scale_reason = why
                self.stats["scaled_up"] += n - current
                self._record_scale("up", n - current, why)
            elif n < current:
                for _ in range(current - n):
                    victim = self._least_loaded()
                    if victim is None:
                        break
                    self._retire(victim, drain=drain)
                self._last_scale = now
                self.last_scale_reason = why
                self.stats["scaled_down"] += current - n
                self._record_scale("down", current - n, why)
            return len(self._vcpus)

    def scale_up(self, n: int = 1, *, reason: str = "") -> int:
        with self._lock:
            return self.scale_to(len(self._vcpus) + n,
                                 reason=reason or "scale up (requested)")

    def scale_down(self, n: int = 1, *, drain: bool = True,
                   reason: str = "") -> int:
        with self._lock:
            return self.scale_to(len(self._vcpus) - n, drain=drain,
                                 reason=reason or "scale down (requested)")

    def _record_scale(self, direction: str, n: int, reason: str) -> None:
        self._scale_history.append({
            "at": time.time(), "direction": direction, "n": n,
            "reason": reason, "vcpus": len(self._vcpus),
        })
        self._emit("workspace.scaled", direction=direction, n=n,
                   reason=reason, count=len(self._vcpus))

    def scale_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Recent scale decisions, newest last."""
        return list(self._scale_history)[-max(1, limit):]

    def kind_counts(self) -> dict[str, int]:
        """How many VCPUs of each kind are live."""
        counts: dict[str, int] = {}
        with self._lock:
            for v in self._vcpus.values():
                counts[v.kind] = counts.get(v.kind, 0) + 1
        return counts

    def ensure_kind(self, kind: str, n: int) -> dict[str, int]:
        """Dedicated lane: keep exactly ``n`` VCPUs of ``kind`` live.

        Celery-style queue routing needs somewhere to route *to* — this
        builds the io/cpu lanes the affinity scorer prefers.  Retires the
        least-loaded of the kind first; never breaches the farm envelope.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown VCPU kind {kind!r}; expected {KINDS}")
        n = max(0, int(n))
        with self._lock:
            of_kind = sorted(
                (v for v in self._vcpus.values() if v.kind == kind),
                key=lambda v: (v.load, v.stats["tasks_run"]))
            if len(of_kind) < n:
                room = self.max_vcpus - len(self._vcpus)
                # the farm envelope wins over the lane request
                for _ in range(min(n - len(of_kind), max(0, room))):
                    self._spawn(kind)
            elif len(of_kind) > n:
                for victim in of_kind[:len(of_kind) - n]:
                    if len(self._vcpus) <= self.min_vcpus:
                        break
                    self._retire(victim, drain=True)
        return self.kind_counts()

    def pause_all(self) -> int:
        """Pause every live VCPU (queues keep accepting work)."""
        n = 0
        for v in self.vcpus():
            v.pause()
            n += 1
        return n

    def resume_all(self) -> int:
        n = 0
        for v in self.vcpus():
            v.resume()
            n += 1
        return n

    def add_vcpu(self, *, kind: str = "balanced", name: str = "") -> VirtualCPU:
        """Add one core of a specific kind (outside the auto envelope)."""
        with self._lock:
            vcpu = self._spawn(kind)
            if name:
                self._vcpus[name] = vcpu
                self._vcpus.pop(vcpu.name, None)
                vcpu.name = name
            return vcpu

    def remove_vcpu(self, name: str, *, drain: bool = True) -> bool:
        with self._lock:
            vcpu = self._vcpus.get(name)
            if vcpu is None:
                return False
            if len(self._vcpus) <= self.min_vcpus:
                return False
            self._retire(vcpu, drain=drain)
            return True

    def _least_loaded(self) -> VirtualCPU | None:
        candidates = [v for v in self._vcpus.values()
                      if v.status in (VcpuStatus.IDLE, VcpuStatus.BUSY,
                                      VcpuStatus.PAUSED)]
        if not candidates:
            return None
        return min(candidates, key=lambda v: (v.load, v.stats["tasks_run"]))

    # ── dispatch ─────────────────────────────────────────────────────────────
    def pick_vcpu(self, affinity: str = "balanced",
                  priority: int = 0) -> VirtualCPU:
        """The best core for this kind of work right now.

        Score (lower = better): load, kind-mismatch penalty, and a tiny
        even-wear term so cores age together.  Paused cores are only
        used as a last resort (their work will wait).
        """
        with self._lock:
            alive = [v for v in self._vcpus.values()
                     if v.status in (VcpuStatus.IDLE, VcpuStatus.BUSY)]
            if not alive:  # all paused or offline: use any non-offline
                alive = [v for v in self._vcpus.values()
                         if v.status != VcpuStatus.OFFLINE]
            if not alive:
                raise RuntimeError("workspace has no live VCPUs")

        def score(v: VirtualCPU) -> tuple:
            mismatch = 0.15 if v.kind != affinity and affinity in KINDS \
                else 0.0
            paused = 0.5 if v.status == VcpuStatus.PAUSED else 0.0
            wear = (v.stats["tasks_run"] % 1000) / 1000.0 * 0.01
            return (v.load + mismatch + paused + wear, v.index)

        return min(alive, key=score)

    def submit(self, fn: Callable[..., Any], *args: Any,
               affinity: str = "balanced", priority: int = 0,
               vcpu: str = "", name: str = "",
               timeout: float = 0.0) -> tuple[str, Future]:
        """Assign work to the best core.  Returns ``(vcpu_name, future)``.

        ``affinity`` = "io" | "cpu" | "balanced" hints at the kind of
        work (a dedicated io/cpu core is preferred when free);
        ``vcpu`` pins a specific core by name (ops tooling);
        ``name`` labels the task; ``timeout`` is a soft per-task limit
        in seconds (see :meth:`VirtualCPU.submit`).
        """
        self.stats["dispatched"] += 1
        try:
            if vcpu:
                target = self.get(vcpu)
                if target is None:
                    raise RuntimeError(f"unknown VCPU {vcpu!r}")
            else:
                target = self.pick_vcpu(affinity, priority)
        except Exception as exc:  # noqa: BLE001
            self.stats["dispatch_errors"] += 1
            raise exc
        future = target.submit(fn, *args, priority=priority, name=name,
                               timeout=timeout)
        return target.name, future

    def map(self, fn: Callable[..., Any], iterable: Iterable[Any], *,
            affinity: str = "balanced", priority: int = 0,
            timeout: float = 0.0) -> list[Future]:
        """Submit ``fn(x)`` for every ``x``.  Returns Futures in order."""
        futures: list[Future] = []
        for x in iterable:
            _, fut = self.submit(fn, x, affinity=affinity, priority=priority,
                                 timeout=timeout,
                                 name=getattr(fn, "__name__", "map"))
            futures.append(fut)
        return futures

    def rebalance(self) -> int:
        """Work-stealing pass (Tokio-style): idle cores steal half the
        queued work of the busiest core.  Returns tasks moved.

        Cheaper than scaling — the autoscaler runs this before deciding
        to grow, so a lopsided queue doesn't buy a core it doesn't need.
        """
        with self._lock:
            vcpus = list(self._vcpus.values())
        idle = [v for v in vcpus
                if v.status == VcpuStatus.IDLE and v.queue_depth() == 0]
        donors = sorted(
            (v for v in vcpus
             if v.status in (VcpuStatus.IDLE, VcpuStatus.BUSY)
             and v.queue_depth() > 1),
            key=lambda v: v.queue_pressure, reverse=True)
        if not idle or not donors:
            return 0
        moved = 0
        for thief in idle:
            if not donors:
                break
            donor = donors[0]
            if donor is thief or donor.queue_depth() <= 1:
                continue
            moved += thief.steal_from(donor)
            donors.sort(key=lambda v: v.queue_pressure, reverse=True)
        if moved:
            self._emit("workspace.rebalanced", moved=moved)
        return moved

    # ── autoscaling ──────────────────────────────────────────────────────────
    def _fleet_busy(self) -> float:
        with self._lock:
            alive = [v for v in self._vcpus.values()
                     if v.status != VcpuStatus.OFFLINE]
        if not alive:
            return 0.0
        return sum(v.busy_fraction for v in alive) / len(alive)

    def _fleet_pressure(self) -> float:
        with self._lock:
            alive = [v for v in self._vcpus.values()
                     if v.status != VcpuStatus.OFFLINE]
        if not alive:
            return 0.0
        return sum(v.queue_pressure for v in alive) / len(alive)

    def _note_metrics(self, now: float, busy: float, pressure: float) -> None:
        self._metric_history.append((now, busy, pressure))

    def _predict_pressure(self, now: float) -> float:
        """Linear extrapolation of queue pressure one interval ahead.

        Catches a queue that's *climbing* before busy_fraction saturates —
        the autoscaler grows on the forecast, not just the present.
        """
        hist = list(self._metric_history)
        if len(hist) < 2:
            return 0.0
        (t0, _, p0), (t1, _, p1) = hist[0], hist[-1]
        dt = max(1e-3, t1 - t0)
        rate = (p1 - p0) / dt
        if rate <= 0:
            return p1
        return min(1.0, p1 + rate * self.autoscale_interval)

    def autoscale_tick(self) -> str:
        """One growth/shrink decision.  Returns the action taken
        ('' = no change).  Safe to call from the watch loop too.

        HPA-style: the desired size is *proportional* to load
        (``ceil(current × signal)``), a 10% deadband stops flapping,
        scale-up is fast and multi-core, scale-down is slow and single —
        and a work-stealing rebalance runs first so a lopsided queue
        doesn't buy cores it doesn't need.
        """
        if not self._autoscale:
            return ""
        now = time.time()
        if now - self._last_scale < self.autoscale_interval:
            return ""
        busy = self._fleet_busy()
        pressure = self._fleet_pressure()
        self._note_metrics(now, busy, pressure)
        # steal before scaling: free capacity hiding on idle cores
        if pressure > 0.2:
            self.rebalance()
            pressure = self._fleet_pressure()
        # Profile-aware thresholds: an aggressive (low-RAM) posture earns a
        # new core only at higher load, and gives cores back earlier.
        up_busy, down_busy = _SCALE_UP_BUSY, _SCALE_DOWN_BUSY
        if self.memory_pressure == "aggressive":
            up_busy = min(0.95, up_busy + 0.15)
            down_busy = min(0.30, down_busy + 0.15)
        elif self.memory_pressure == "relaxed":
            up_busy = max(0.5, up_busy - 0.05)
        with self._lock:
            current = len(self._vcpus)

        # combined hot signal: busy ratio, queue-pressure ratio, forecast
        forecast = self._predict_pressure(now)
        hot = max(busy / up_busy, pressure / _SCALE_UP_PRESSURE,
                  forecast / _SCALE_UP_PRESSURE)
        if hot > 1.0 + _SCALE_TOLERANCE:
            if self._hot_since is None:
                self._hot_since = now
            if now - self._hot_since >= self.autoscale_interval \
                    and current < self.max_vcpus:
                desired = min(self.max_vcpus,
                              max(current + 1, math.ceil(current * hot)))
                step = min(_SCALE_UP_MAX_STEP, desired - current)
                why = (f"autoscale: hot (busy {busy:.0%}, pressure "
                       f"{pressure:.0%}, forecast {forecast:.0%})")
                self.scale_up(step, reason=why)
                self._hot_since = None
                return f"scaled up +{step} → {len(self._vcpus)} ({why})"
        else:
            self._hot_since = None

        if busy < down_busy and pressure < _SCALE_DOWN_PRESSURE:
            if self._idle_since is None:
                self._idle_since = now
            if now - self._idle_since >= _SCALE_DOWN_IDLE_S \
                    and current > self.min_vcpus:
                idle_for = now - (self._idle_since or now)
                why = (f"autoscale: idle {idle_for:.0f}s "
                       f"(busy {busy:.0%})")
                self.scale_down(_SCALE_DOWN_MAX_STEP, reason=why)
                self._idle_since = None
                return f"scaled down → {len(self._vcpus)} ({why})"
        else:
            self._idle_since = None
        return ""


    # ── reporting ────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """The whole farm in one dict — for the AI, the CLI, the bus."""
        vcpu_rows = [v.to_dict() for v in self.vcpus()]
        alive = [v for v in vcpu_rows
                 if v["status"] != VcpuStatus.OFFLINE]
        busy = sum(v["busy_fraction"] for v in alive) / len(alive) \
            if alive else 0.0
        queued = sum(v["queue"] for v in vcpu_rows)
        return {
            "profile": self.profile.to_dict(),
            "vcpus": len(vcpu_rows),
            "kinds": self.kind_counts(),
            "min_vcpus": self.min_vcpus,
            "target_vcpus": self.target_vcpus,
            "max_vcpus": self.max_vcpus,
            "fleet_busy": round(busy, 3),
            "fleet_pressure": round(self._fleet_pressure(), 3),
            "queued_total": queued,
            "autoscale": self._autoscale,
            "last_scale": self.last_scale_reason,
            "scale_history": self.scale_history(5),
            "stats": {
                "scaled_up": self.stats["scaled_up"],
                "scaled_down": self.stats["scaled_down"],
                "dispatched": self.stats["dispatched"],
                "dispatch_errors": self.stats["dispatch_errors"],
            },
            "details": vcpu_rows,
        }

    def summary_line(self) -> str:
        st = self.status()
        states: dict[str, int] = {}
        for v in st["details"]:
            states[v["status"]] = states.get(v["status"], 0) + 1
        label = " ".join(f"{n}{k[0].upper()}{k[1:]}"
                         for k, n in sorted(states.items()))
        kinds = " ".join(f"{k}:{n}" for k, n in sorted(st["kinds"].items()))
        return (f"workspace: {st['vcpus']} vcpu(s) [{label}] "
                f"busy {st['fleet_busy']:.0%} pressure {st['fleet_pressure']:.0%} "
                f"queued {st['queued_total']} kinds({kinds}) "
                f"profile={st['profile']['kind']} "
                f"envelope {st['min_vcpus']}-{st['target_vcpus']}-{st['max_vcpus']}")

    @staticmethod
    def _bar(frac: float, width: int = 10) -> str:
        filled = int(round(min(1.0, max(0.0, frac)) * width))
        return "█" * filled + "░" * (width - filled)

    def render_status(self) -> str:
        """God-tier farm view: one ASCII table for the CLI/chat.

        Pure (returns a string, never prints) so the AI, the CLI, and
        tests all share it.
        """
        st = self.status()
        rows = st["details"]
        lines = [
            f"⚙ workspace farm · profile={st['profile']['kind']} · "
            f"{st['vcpus']} vcpu(s) · busy {st['fleet_busy']:.0%} · "
            f"pressure {st['fleet_pressure']:.0%} · queued {st['queued_total']}",
            f"  envelope {st['min_vcpus']}-{st['target_vcpus']}-{st['max_vcpus']}"
            f" · autoscale {'on' if st['autoscale'] else 'off'}"
            + (f" · last: {st['last_scale']}" if st["last_scale"] else ""),
            "┌──────────┬──────────┬────────┬──────────────┬──────┬───────┬───────┐",
            "│ vcpu     │ kind     │ status │ load         │ busy │ queue │ tasks │",
            "├──────────┼──────────┼────────┼──────────────┼──────┼───────┼───────┤",
        ]
        glyph = {"idle": "○", "busy": "●", "paused": "⏸",
                 "offline": "✕", "error": "⚠"}
        for v in sorted(rows, key=lambda r: r["index"]):
            s = v["stats"]
            tasks = s["tasks_run"] + s["tasks_failed"]
            lines.append(
                f"│ {v['id'][:8]:<8} │ {v['kind'][:8]:<8} │ "
                f"{glyph.get(v['status'], '?')} {v['status'][:6]:<6} │ "
                f"{self._bar(v['load'])} │ {v['busy_fraction']:>4.0%} │ "
                f"{v['queue']:>3}/{v['max_queue']:<3} │ {tasks:>5} │")
        lines.append(
            "└──────────┴──────────┴────────┴──────────────┴──────┴───────┴───────┘")
        hist = self.scale_history(3)
        if hist:
            lines.append("  scale history:")
            for h in hist:
                arrow = "▲" if h["direction"] == "up" else "▼"
                lines.append(f"    {arrow} {h['direction']} {h['n']:+d} → "
                             f"{h['vcpus']} vcpus — {h['reason'][:70]}")
        return "\n".join(lines)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def shutdown(self, *, drain: bool = False) -> None:
        self._stop = True
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        for vcpu in self.vcpus():
            vcpu.stop(drain=drain, timeout=2.0)
        with self._lock:
            self._vcpus.clear()

    def __del__(self):
        # Last-resort cleanup: any workspace an owner forgot to shut down
        # (e.g. one spawned ad hoc by a tool or chat handler) must not keep
        # its VCPU threads alive forever — a leaked fleet of threads
        # exhausts fork headroom and breaks sandboxed code execution.
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001 - GC must never raise
            pass

    def _emit(self, event: str, **data: Any) -> None:
        bus = getattr(self.context, "bus", None) if self.context else None
        if bus is None:
            return
        try:
            bus.emit(event, **data)
        except Exception:  # noqa: BLE001 - telemetry is a bonus
            pass


def _autoscale_worker(ws_ref: "weakref.ref") -> None:
    """Thread target for the autoscale ticker.

    Holds only a WEAK reference to its workspace during the long wait —
    just the Event and the interval, neither of which references the
    workspace. A strong reference (bound-method target or a frame local
    held across the wait) would keep a dead workspace alive forever, so
    its __del__ never ran and its VCPU thread pool never stopped: every
    leaked fleet would permanently eat fork headroom.
    """
    while True:
        ws = ws_ref()
        if ws is None:
            return
        event, interval = ws._wake, ws.autoscale_interval
        ws = None  # release before the long wait
        if event.wait(timeout=interval):
            event.clear()
        ws = ws_ref()
        if ws is None or ws._stop:
            return
        try:
            action = ws.autoscale_tick()
            if action:
                ws._emit("workspace.scaled", action=action,
                         count=len(ws.vcpus()))
        except Exception:  # noqa: BLE001 - autoscale must never die
            pass
        ws = None  # release before the next wait


def build_workspace(context: Any = None) -> "Workspace":
    """Build a Workspace from the context's settings (or defaults)."""
    settings = getattr(context, "settings", None) if context else None
    ws = getattr(settings, "workspace", None) if settings else None
    tune = (getattr(context, "extras", None) or {}).get("tune") if context else None
    return Workspace(
        context,
        profile=str(getattr(ws, "profile", "") or ""),
        vcpus=int(getattr(ws, "vcpus", 0) or 0),
        min_vcpus=int(getattr(ws, "min_vcpus", 0) or 0),
        max_vcpus=int(getattr(ws, "max_vcpus", 0) or 0),
        autoscale=bool(getattr(ws, "autoscale", True)),
        autoscale_interval=float(getattr(ws, "autoscale_interval", 30.0)
                                 or 30.0),
        memory_pressure=str(getattr(tune, "memory_pressure", "") or ""),
    )
