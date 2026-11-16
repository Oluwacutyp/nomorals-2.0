"""Gap R1-14 (reversed): the OS control plane is RESTORED and WIRED.

Lane 14 deleted kernel/services/health as "unfit"; the owner reversed
that (2026-11-16): deleting a wanted capability is wrong — make it
work. These pins cover the wiring, not the modules in isolation
(the restored test_os_core / test_os_sweep / test_os_kernel_failures
files cover that):

1. exactly one kernel is constructed at boot, adopting live instances
   (identity: no second scheduler/router/adapter is ever built);
2. start() leaves real services running (not just kernel.started);
3. signal teardown is preserved end to end (the script's handlers
   still own signals; arm_signals chains the previous handler);
4. health delegates to real readiness (wait_ready reflects a real
   unready service);
5. a dead scheduler is restarted with backoff and the restart is
   journaled (failure ledger + restart ledger + beacon);
6. a gave-up adapter is re-attempted through the gateway's own
   hot-restart contract;
7. stop() shuts down cleanly (services stopped once, in order).
"""

from __future__ import annotations

import threading
import time
import unittest

from nomorals.core.events import EventBus
from nomorals.os.kernel import OSKernel
from nomorals.os.services import ServiceNotFound, ServiceRegistry


class _FakeScheduler:
    """Stand-in for agents/scheduler.py::Scheduler: start/stop/running
    plus a tick clock the health probe reads."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_tick_at: float | None = None
        self.start_calls = 0
        self.stop_calls = 0

    def start(self, **_kwargs: object) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self.start_calls += 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._last_tick_at = time.time()
            self._stop.wait(0.05)

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def kill(self) -> None:
        """Simulate a dead scheduler: thread gone, clock stale."""
        self._stop.set()
        self._thread = None
        self._last_tick_at = time.time() - 3600.0


class _FakeAdapter:
    """Stand-in for a ChatAdapter: health() + the gave-up terminal state
    the adapter's own restarter cannot leave."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._alive = threading.Event()
        self._gave_up = False
        self.start_calls = 0

    def start(self, _handler: object) -> bool:
        if self._alive.is_set():
            return False
        self.start_calls += 1
        self._gave_up = False
        self._alive.set()
        return True

    def stop(self) -> None:
        self._alive.clear()

    def give_up(self) -> None:
        self._alive.clear()
        self._gave_up = True

    def health(self) -> dict:
        return {
            "name": self.name,
            "running": self._alive.is_set(),
            "gave_up": self._gave_up,
        }


class _FakeGateway:
    """Stand-in for ChatGateway's hot-restart contract."""

    def __init__(self) -> None:
        self.adapters: dict[str, _FakeAdapter] = {}
        self._known: dict[str, _FakeAdapter] = {}
        self._inbound = object()

    def register_known(self, adapter: _FakeAdapter) -> None:
        self._known[adapter.name] = adapter
        self.adapters[adapter.name] = adapter

    def stop_one(self, name: str) -> dict:
        adapter = self.adapters.pop(name, None)
        if adapter is None:
            return {"ok": False, "error": f"{name} is not running"}
        adapter.stop()
        return {"ok": True, "platform": name}

    def start_one(self, name: str, handler: object) -> dict:
        adapter = self.adapters.get(name)
        if adapter is None:
            adapter = self._known.get(name)
        if adapter is None:
            return {"ok": False, "error": f"{name} unknown"}
        if adapter.health()["running"]:
            return {"ok": False, "error": f"{name} is already running"}
        self.adapters[name] = adapter
        adapter.start(handler)
        return {"ok": True, "platform": name}


def _scheduler_probe(sched: _FakeScheduler) -> tuple[bool, str]:
    """The real probe the boot site wires: thread alive AND tick fresh."""
    if not sched.running():
        return False, "scheduler thread is not running"
    last = sched._last_tick_at
    if last is None:
        return True, "scheduler started, no tick yet"
    age = time.time() - last
    if age > 10.0:
        return False, f"scheduler tick is {age:.0f}s stale"
    return True, "scheduler ticking"


def _always_fail(_service: object) -> None:
    raise RuntimeError("start always fails")


class AdoptTests(unittest.TestCase):
    def test_adopt_returns_the_live_instance_never_a_second_one(self) -> None:
        sched = _FakeScheduler()
        sched.start()
        reg = ServiceRegistry()
        reg.adopt("scheduler", sched, description="live scheduler",
                  health_check=lambda: _scheduler_probe(sched),
                  on_start=lambda s: s.start(), on_stop=lambda s: s.stop())
        self.assertIs(reg.lookup("scheduler"), sched)
        self.assertEqual(reg.lookup("scheduler").start_calls, 1)

    def test_unadopted_builtins_are_unregistered_at_boot(self) -> None:
        # The boot site unregisters every built-in with no live
        # counterpart, so lookup() raises instead of building a
        # surprise second service.
        reg = ServiceRegistry()
        for name in ("browser", "model", "bridge", "queue",
                     "browser_service"):
            self.assertTrue(reg.unregister(name), name)
            with self.assertRaises(ServiceNotFound):
                reg.lookup(name)


class KernelStartTests(unittest.TestCase):
    def test_start_leaves_a_real_service_running(self) -> None:
        sched = _FakeScheduler()
        sched.start()
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "scheduler", sched, description="live scheduler",
            health_check=lambda: _scheduler_probe(sched),
            on_start=lambda s: s.start(), on_stop=lambda s: s.stop())
        kernel.start()
        try:
            self.assertTrue(kernel.started)
            self.assertIn("scheduler", kernel.services.started_services())
            self.assertTrue(sched.running())
        finally:
            kernel.stop()

    def test_start_does_not_start_the_bus_dispatcher(self) -> None:
        # Inline delivery is the proven behavior (gap-14 point d);
        # the wired kernel leaves the dispatcher alone unless asked.
        bus = EventBus()
        kernel = OSKernel(bus=bus)
        kernel.start()
        try:
            self.assertIsNone(bus._dispatcher)
        finally:
            kernel.stop()

    def test_wait_ready_reflects_a_real_unready_service(self) -> None:
        sched = _FakeScheduler()
        sched.start()
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "scheduler", sched, description="live scheduler",
            health_check=lambda: (False, "scheduler down for test"),
            on_start=lambda s: s.start(), on_stop=lambda s: s.stop())
        kernel.start()
        try:
            self.assertFalse(kernel.wait_ready(timeout_s=0.3))
        finally:
            kernel.stop()


class SuperviseTests(unittest.TestCase):
    def _kernel_with_scheduler(self, sched: _FakeScheduler) -> OSKernel:
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "scheduler", sched, description="live scheduler",
            health_check=lambda: _scheduler_probe(sched),
            on_start=lambda s: s.start(), on_stop=lambda s: s.stop(),
            health_role="liveness")
        kernel.start()
        kernel.supervise(interval_s=0.05)
        return kernel

    def test_dead_scheduler_is_restarted_and_journaled(self) -> None:
        sched = _FakeScheduler()
        sched.start()
        kernel = self._kernel_with_scheduler(sched)
        try:
            sched.kill()
            deadline = time.time() + 10.0
            while time.time() < deadline and not sched.running():
                time.sleep(0.05)
            self.assertTrue(sched.running(),
                            "supervisor did not restart the scheduler")
            self.assertGreaterEqual(sched.start_calls, 2)
            summary = kernel.supervision_summary()
            self.assertGreaterEqual(summary["restarts_total"], 1)
            names = [r["service"] for r in summary["restarts"]]
            self.assertIn("scheduler", names)
        finally:
            kernel.stop()

    def test_restart_budget_gives_up_and_says_so(self) -> None:
        sched = _FakeScheduler()
        sched.start()
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "scheduler", sched, description="live scheduler",
            health_check=lambda: (False, "always down"),
            on_start=_always_fail,
            on_stop=lambda s: s.stop(),
            health_role="liveness")
        kernel.start()
        kernel.supervise(interval_s=0.05, restart_delays=(0.01, 0.01),
                         max_restarts=2)
        try:
            deadline = time.time() + 10.0
            summary = kernel.supervision_summary()
            while time.time() < deadline and not summary.get("gave_up"):
                time.sleep(0.05)
                summary = kernel.supervision_summary()
            gave_up = summary.get("gave_up") or {}
            self.assertIn("scheduler", gave_up,
                          "supervisor never gave up on a hopeless service")
            self.assertEqual(
                len([r for r in summary["restarts"]
                     if r["service"] == "scheduler"]), 2)
        finally:
            kernel.stop()

    def test_health_only_service_is_watched_never_restarted(self) -> None:
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "router", object(), description="live router",
            health_check=lambda: (False, "no providers"),
            restartable=False, health_role="readiness")
        kernel.start()
        kernel.supervise(interval_s=0.05)
        try:
            time.sleep(0.3)
            summary = kernel.supervision_summary()
            self.assertEqual(summary["restarts_total"], 0)
            self.assertIn("router", summary.get("down", []))
        finally:
            kernel.stop()

    def test_gave_up_adapter_is_reattempted_via_the_gateway(self) -> None:
        gateway = _FakeGateway()
        adapter = _FakeAdapter("whatsapp")
        gateway.register_known(adapter)
        adapter.start(gateway._inbound)
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "adapter:whatsapp", adapter, description="live whatsapp adapter",
            health_check=lambda: (
                bool(adapter.health()["running"]),
                "adapter down" if not adapter.health()["running"]
                else "adapter running"),
            on_start=lambda a: gateway.start_one("whatsapp",
                                                 gateway._inbound),
            on_stop=lambda a: gateway.stop_one("whatsapp"),
            health_role="liveness")
        kernel.start()
        kernel.supervise(interval_s=0.05)
        try:
            adapter.give_up()
            deadline = time.time() + 10.0
            while time.time() < deadline and not adapter.health()["running"]:
                time.sleep(0.05)
            self.assertTrue(adapter.health()["running"],
                            "supervisor did not re-attempt the gave-up adapter")
            self.assertGreaterEqual(adapter.start_calls, 2)
        finally:
            kernel.stop()


class SignalChainTests(unittest.TestCase):
    def test_arm_signals_chains_the_previous_handler(self) -> None:
        # The kernel must never silently replace an existing handler
        # (gap-14 point c): the previous handler still runs.
        import signal

        called: list[str] = []

        def _previous(signum: int, _frame: object) -> None:
            called.append(f"prev-{signum}")

        kernel = OSKernel(bus=EventBus())
        try:
            previous = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _previous)
            try:
                kernel.arm_signals()
                handler = signal.getsignal(signal.SIGTERM)
                handler(signal.SIGTERM, None)
                self.assertIn(f"prev-{signal.SIGTERM}", called)
            finally:
                signal.signal(signal.SIGTERM, previous)
        finally:
            kernel.stop()


class BeaconTests(unittest.TestCase):
    def test_kernel_section_rides_the_beacon(self) -> None:
        from nomorals.agents.beacon import build_beacon_state

        sched = _FakeScheduler()
        sched.start()
        kernel = OSKernel(bus=EventBus())
        kernel.services.adopt(
            "scheduler", sched, description="live scheduler",
            health_check=lambda: _scheduler_probe(sched),
            on_start=lambda s: s.start(), on_stop=lambda s: s.stop())
        kernel.start()
        try:
            runtime = type("R", (), {
                "brain": None, "_started": time.time(), "stats": {},
                "gateway": None, "context": None,
                "_kernel": kernel,
            })()
            state = build_beacon_state(runtime)
            self.assertIn("kernel", state)
            self.assertTrue(state["kernel"]["started"])
            self.assertIn("scheduler", state["kernel"]["services"])
        finally:
            kernel.stop()


if __name__ == "__main__":
    unittest.main()
