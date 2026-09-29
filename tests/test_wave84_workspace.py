"""Wave 84: the Workspace + Virtual CPU farm.

Profile detection, VCPU lifecycle (queue, pause/resume, drain, errors),
affinity dispatch, autoscaling, and the integrations (executor IO lane,
parallel mission resume, tools, CLI, watch tick).  All hermetic."""

from __future__ import annotations

import json
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.missions import MissionRunner, MissionStatus, MissionStore
from nomorals.workspace import (EnvironmentProfile, VirtualCPU, Workspace,
                                detect_profile, resolve_profile)


class _Base(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w84-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)


# ── profile detection ────────────────────────────────────────────────────────

class ProfileTests(unittest.TestCase):
    def test_android_is_termux(self):
        p = detect_profile(cpu=8, mem_mb=6 * 1024, system="Android")
        self.assertEqual(p.kind, "termux")
        self.assertEqual((p.min_vcpus, p.target_vcpus, p.max_vcpus), (1, 2, 3))

    def test_tiny_box_is_embedded(self):
        p = detect_profile(cpu=1, mem_mb=512, system="Linux")
        self.assertEqual(p.kind, "embedded")
        self.assertEqual(p.max_vcpus, 2)

    def test_big_ram_is_workstation(self):
        p = detect_profile(cpu=16, mem_mb=64 * 1024, system="Linux")
        self.assertEqual(p.kind, "workstation")
        self.assertEqual((p.min_vcpus, p.target_vcpus, p.max_vcpus),
                         (4, 8, 16))

    def test_small_box_is_vps(self):
        p = detect_profile(cpu=2, mem_mb=4 * 1024, system="Linux")
        self.assertEqual(p.kind, "vps")

    def test_normal_box_is_pc(self):
        p = detect_profile(cpu=8, mem_mb=32 * 1024 - 1, system="Linux")
        self.assertEqual(p.kind, "pc")

    def test_config_pins_the_profile(self):
        p = resolve_profile("termux")
        self.assertEqual(p.kind, "termux")
        self.assertFalse(p.detected)
        # unknown names fall back to pc, never crash
        self.assertEqual(resolve_profile("mars").kind, "pc")


# ── VirtualCPU ───────────────────────────────────────────────────────────────

class VcpuTests(unittest.TestCase):
    def test_runs_and_accounts(self):
        v = VirtualCPU(0)
        futs = [v.submit(lambda i=i: i * 3, i) for i in range(5)]
        self.assertEqual([f.result(timeout=5) for f in futs],
                         [0, 3, 6, 9, 12])
        self.assertEqual(v.stats["tasks_run"], 5)
        time.sleep(0.05)
        self.assertEqual(v.status, "idle")
        v.stop()
        self.assertEqual(v.status, "offline")
        with self.assertRaises(RuntimeError):
            v.submit(lambda: 1)

    def test_workspace_runs_in_parallel_across_cores(self):
        ws = Workspace(None, profile="pc", autoscale=False, vcpus=4)
        self.addCleanup(ws.shutdown)
        t0 = time.time()
        futs = [ws.submit(lambda: time.sleep(0.3)) for _ in range(4)]
        for name, f in futs:
            f.result(timeout=10)
        wall = time.time() - t0
        # 4 x 0.3s of work on 4 cores ≈ one 0.3s wave, not 1.2s serial
        self.assertLess(wall, 1.0)
        self.assertGreaterEqual(len({n for n, _ in futs}), 3)

    def test_task_exception_is_data(self):
        v = VirtualCPU(0)
        f = v.submit(lambda: 1 / 0)
        self.assertIsNotNone(f.exception(timeout=5))
        self.assertEqual(v.stats["tasks_failed"], 1)
        self.assertEqual(v.status, "idle")  # the core is fine
        v.stop()

    def test_pause_holds_work_until_resume(self):
        v = VirtualCPU(0)
        v.pause()
        self.assertEqual(v.status, "paused")
        ran = []
        f = v.submit(lambda: ran.append(1))
        time.sleep(0.4)
        self.assertEqual(ran, [])           # queued, not running
        v.resume()
        f.result(timeout=5)
        self.assertEqual(ran, [1])
        v.stop()

    def test_priority_runs_first(self):
        v = VirtualCPU(0)
        order = []
        v.pause()
        v.submit(lambda: order.append("normal"))
        v.submit(lambda: order.append("urgent"), priority=10)
        v.resume()
        time.sleep(0.5)
        self.assertEqual(order, ["urgent", "normal"])
        v.stop()

    def test_abort_pending_drops_the_queue(self):
        v = VirtualCPU(0)
        v.pause()
        f1 = v.submit(lambda: 1)
        f2 = v.submit(lambda: 2)
        dropped = v.abort_pending()
        self.assertEqual(dropped, 2)
        self.assertTrue(f1.cancelled() or f1.exception(timeout=1) is not None)
        self.assertEqual(v.queue_depth(), 0)
        v.stop()

    def test_stop_drain_finishes_queued_work(self):
        v = VirtualCPU(0)
        done = []
        v.pause()
        f = v.submit(lambda: done.append(1))
        v.stop(drain=True)
        self.assertEqual(done, [1])        # the work ran, not lost
        self.assertEqual(v.status, "offline")


# ── Workspace ────────────────────────────────────────────────────────────────

class WorkspaceTests(unittest.TestCase):
    def _ws(self, **kw):
        ws = Workspace(None, profile="pc", autoscale=False, **kw)
        self.addCleanup(ws.shutdown)
        return ws

    def test_profile_sized_farm(self):
        ws = self._ws()
        self.assertEqual(len(ws.vcpus()), 4)          # pc target
        self.assertEqual(ws.min_vcpus, 2)
        self.assertEqual(ws.max_vcpus, 8)

    def test_scale_clamped_to_envelope(self):
        ws = self._ws()
        self.assertEqual(ws.scale_to(99), 8)          # clamped to max
        self.assertEqual(ws.scale_to(0), 2)           # clamped to min
        self.assertEqual(ws.scale_up(1), 3)
        self.assertEqual(ws.scale_down(1), 2)
        # 4 up (4→8), 6 down (8→2), 1 up, 1 down
        self.assertEqual(ws.stats["scaled_up"], 5)
        self.assertEqual(ws.stats["scaled_down"], 7)

    def test_affinity_picks_the_matching_kind(self):
        ws = self._ws(vcpus=2)
        io_core = ws.add_vcpu(kind="io")
        name, f = ws.submit(lambda: 42, affinity="io")
        self.assertEqual(f.result(timeout=5), 42)
        self.assertEqual(name, io_core.name)

    def test_work_spreads_across_cores(self):
        ws = self._ws(vcpus=3)
        names = set()
        futs = []
        for i in range(12):
            name, f = ws.submit(lambda i=i: i, i)
            names.add(name)
            futs.append(f)
        for f in futs:
            f.result(timeout=10)
        self.assertGreaterEqual(len(names), 2)
        self.assertEqual(ws.stats["dispatched"], 12)

    def test_scale_down_drains_the_victim(self):
        ws = self._ws(vcpus=2, min_vcpus=1)
        done = []
        # back up one core's queue so it is the least-loaded victim
        busy_f = ws.submit(lambda: time.sleep(0.4), affinity="balanced")
        time.sleep(0.05)
        ws.scale_down(1)
        self.assertEqual(len(ws.vcpus()), 1)
        busy_f[1].result(timeout=5)
        self.assertEqual(done, [])  # nothing lost: the task completed

    def test_status_and_summary(self):
        ws = self._ws(vcpus=2)
        st = ws.status()
        for key in ("profile", "vcpus", "fleet_busy", "min_vcpus",
                    "target_vcpus", "max_vcpus", "autoscale", "stats",
                    "details"):
            self.assertIn(key, st)
        self.assertEqual(st["profile"]["kind"], "pc")
        self.assertIn("workspace:", ws.summary_line())
        line = ws.summary_line()
        self.assertIn("vcpu(s)", line)

    def test_paused_core_is_avoided_but_not_lost(self):
        ws = self._ws(vcpus=2)
        core = ws.vcpus()[0]
        core.pause()
        name, f = ws.submit(lambda: 7)
        self.assertEqual(name, ws.vcpus()[1].name)  # routed around the pause
        f.result(timeout=5)
        core.stop()


# ── autoscaling ──────────────────────────────────────────────────────────────

class AutoscaleTests(unittest.TestCase):
    def _ws(self, **kw):
        ws = Workspace(None, profile="pc", autoscale=False,
                       autoscale_interval=1.0, **kw)
        ws._autoscale = True                        # enable ticking, no thread
        self.addCleanup(ws.shutdown)
        return ws

    def test_hot_fleet_scales_up(self):
        ws = self._ws(vcpus=2)
        for v in ws.vcpus():
            v._busy_ema = 0.95                      # everything saturated
        ws.autoscale_interval = 1.0
        now = time.time()
        ws._last_scale = 0
        ws._hot_since = now - 2.0                   # been hot past the window
        action = ws.autoscale_tick()
        self.assertIn("scaled up", action)
        self.assertEqual(len(ws.vcpus()), 3)

    def test_cold_fleet_scales_down(self):
        ws = self._ws(vcpus=3)
        for v in ws.vcpus():
            v._busy_ema = 0.0
        ws.autoscale_interval = 1.0
        now = time.time()
        ws._last_scale = 0
        ws._idle_since = now - 60.0                 # been idle for a while
        action = ws.autoscale_tick()
        self.assertIn("scaled down", action)
        self.assertEqual(len(ws.vcpus()), 2)

    def test_never_leaves_the_envelope(self):
        ws = self._ws()
        for v in ws.vcpus():
            v._busy_ema = 1.0
        ws._last_scale = 0
        ws._hot_since = time.time() - 10
        ws.autoscale_tick()
        ws.scale_to(ws.max_vcpus)
        ws._hot_since = time.time() - 10
        ws._last_scale = 0
        ws.autoscale_tick()
        self.assertEqual(len(ws.vcpus()), ws.max_vcpus)  # stayed clamped


# ── integrations ─────────────────────────────────────────────────────────────

class IntegrationTests(_Base):
    def test_context_boots_the_farm(self):
        ws = self.context.workspace
        self.assertIsNotNone(ws)
        self.assertGreaterEqual(len(ws.vcpus()), 1)
        self.assertIn(ws.profile.kind,
                      {"termux", "mobile", "pc", "vps", "workstation",
                       "embedded"})

    def test_executor_io_lane_runs_on_vcpus(self):
        from nomorals.agents.runtime import HybridExecutor

        from nomorals.agents.tasks import Task, TaskGraph, TaskKind

        def step(i):
            time.sleep(0.05)
            return i * 10

        graph = TaskGraph()
        for i in range(6):
            task = graph.add_task(f"s{i}", (lambda i=i: step(i)))
            task.kind = TaskKind.IO
        before = self.context.workspace.stats["dispatched"]
        executor = HybridExecutor(workspace=self.context.workspace)
        report = executor.run(graph)
        after = self.context.workspace.stats["dispatched"]
        self.assertEqual(report.failed, 0)
        self.assertEqual(report.done, 6)
        self.assertGreaterEqual(after - before, 6)

    def test_parallel_mission_resume(self):
        store = MissionStore(self.context.db)
        runner = MissionRunner(self.context)
        missions = []
        for i in range(2):
            m = store.create_new(f"parallel mission {i}")
            m.status = MissionStatus.RUNNING
            m.state["plan"] = [{
                "name": "gather", "goal": f"find thing {i}",
                "role": "research", "kind": "io", "depends_on": []}]
            store.save(m)
            missions.append(m)
        before = self.context.workspace.stats["dispatched"]
        results = runner.resume_all(parallel=True, max_iterations=3)
        after = self.context.workspace.stats["dispatched"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status == MissionStatus.DONE
                            for r in results))
        self.assertGreaterEqual(after - before, 2)

    def test_tools_inspect_and_control(self):
        status = self.context.tools.call("workspace_status")
        self.assertTrue(status.ok)
        self.assertIn("summary", status.value)
        up = self.context.tools.call("workspace_control", action="scale_up")
        self.assertTrue(up.ok)
        n = up.value["vcpus"]
        down = self.context.tools.call("workspace_control",
                                       action="scale_down")
        self.assertEqual(down.value["vcpus"], n - 1)

    def test_cli_workspace(self):
        from nomorals.cli import _cmd_workspace
        args = types.SimpleNamespace(scale=0, scale_up=False,
                                     scale_down=False, pause="",
                                     resume_vcpu="", add="",
                                     remove_vcpu="", json=False)
        buf = StringIO()
        with redirect_stdout(buf):
            rc = _cmd_workspace(args, self.context)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("workspace:", out)
        self.assertIn("profile", out)

    def test_cli_workspace_json_scale(self):
        from nomorals.cli import _cmd_workspace
        args = types.SimpleNamespace(scale=2, scale_up=False,
                                     scale_down=False, pause="",
                                     resume_vcpu="", add="",
                                     remove_vcpu="", json=True)
        buf = StringIO()
        with redirect_stdout(buf):
            rc = _cmd_workspace(args, self.context)
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["vcpus"], 2)

    def test_watch_tick_reports_the_farm(self):
        from nomorals.agents.watcher import WatchLoop
        loop = WatchLoop(self.context, interval=1)
        result = loop.tick()
        self.assertIn("workspace", result)
        self.assertIn("vcpu(s)", result["workspace"])


if __name__ == "__main__":
    unittest.main()
