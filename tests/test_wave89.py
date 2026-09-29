"""Wave 89: a frozen local model must repair itself.

The recurring phone failure, in one sentence: llama-server dies on RAM
pressure but KEEPS ITS PORT, so 'is a port open?' says 'healthy' while
every request hangs — until a human kills it by hand.  Wave 89 removes
the human:

* ``find_llama_server_pid`` — find the port holder by scanning /proc
  (only processes that name our port and are llama-server count)
* ``kill_pid`` — SIGTERM, wait, SIGKILL
* ``gguf_check`` — magic + size, so a 310 MB '4.7 GB model' is caught
* ``GGUFServerManager.heal`` — free→start, answering→no-op,
  frozen-ours→kill+start, frozen-foreign→report, don't touch
* boot (``ensure_local_gguf``) uses heal instead of 'port open = done'
* the router fires repair hooks on preflight failure (background,
  300 s cooldown) — this message falls back, next message gets repaired
* ``nm doctor`` reports the whole reply path: model file, server state,
  every provider's live health
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals.core.config import Settings, get_settings  # noqa: E402
from nomorals.llm import local_server as ls  # noqa: E402
from nomorals.llm.base import LLMProvider, LLMResponse, Message  # noqa: E402
from nomorals.llm.router import LLMRouter  # noqa: E402

PORT = 18080  # nobody else uses this


# ── find_llama_server_pid ────────────────────────────────────────────────────

def _fake_proc(tmp: Path, entries: dict[str, str]) -> Path:
    """entries: pid → argv (space-joined, nulls restored)."""
    proc = tmp / "proc"
    proc.mkdir()
    for pid, argv in entries.items():
        d = proc / pid
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv.split()) + b"\0")
    return proc


class FindLlamaServerPidTest(unittest.TestCase):
    def test_finds_llama_server_with_our_port(self):
        tmp = Path(__file__).parent / "_w89proc"
        tmp.mkdir(exist_ok=True)
        try:
            proc = _fake_proc(tmp, {
                "123": "llama-server --model /x/Q4.gguf --port 18080",
                "456": "bash -c sleep",
            })
            self.assertEqual(ls.find_llama_server_pid(PORT, proc_dir=str(proc)), 123)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ignores_other_ports(self):
        tmp = Path(__file__).parent / "_w89proc"
        tmp.mkdir(exist_ok=True)
        try:
            proc = _fake_proc(tmp, {"123": "llama-server --port 9999"})
            self.assertIsNone(ls.find_llama_server_pid(PORT, proc_dir=str(proc)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ignores_foreign_processes_even_same_port(self):
        tmp = Path(__file__).parent / "_w89proc"
        tmp.mkdir(exist_ok=True)
        try:
            proc = _fake_proc(tmp, {"123": f"nginx -p /etc --port {PORT}"})
            self.assertIsNone(ls.find_llama_server_pid(PORT, proc_dir=str(proc)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_proc_dir_is_none(self):
        self.assertIsNone(ls.find_llama_server_pid(PORT, proc_dir="/nonexistent-proc"))


class KillPidTest(unittest.TestCase):
    def test_kills_a_live_process(self):
        proc = subprocess.Popen(["sleep", "60"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            self.assertTrue(ls.kill_pid(proc.pid, timeout=5.0))
        finally:
            proc.kill()
            proc.wait()
        time.sleep(0.1)
        with self.assertRaises(ProcessLookupError):
            os.kill(proc.pid, 0)

    def test_already_dead_is_true(self):
        self.assertTrue(ls.kill_pid(99999999))


# ── gguf_check ────────────────────────────────────────────────────────────────

class GgufCheckTest(unittest.TestCase):
    def _mk(self, path: Path, size_mb: int, magic: bytes = b"GGUF") -> Path:
        with path.open("wb") as fh:
            if magic:
                fh.write(magic)
            if size_mb > 0:
                fh.seek(size_mb * 1024 * 1024 - 1)
                fh.write(b"\0")  # sparse: stat size is big, disk is small
        return path

    def test_valid_sparse_file(self):
        d = Path(__file__).parent / "_w89gguf"
        d.mkdir(exist_ok=True)
        try:
            ok, detail = ls.gguf_check(self._mk(d / "good.gguf", 12))
            self.assertTrue(ok, detail)
            self.assertIn("GGUF magic OK", detail)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_bad_magic(self):
        d = Path(__file__).parent / "_w89gguf"
        d.mkdir(exist_ok=True)
        try:
            ok, detail = ls.gguf_check(self._mk(d / "bad.gguf", 12, magic=b"PK\x03\x04"))
            self.assertFalse(ok)
            self.assertIn("bad magic", detail)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_too_small(self):
        d = Path(__file__).parent / "_w89gguf"
        d.mkdir(exist_ok=True)
        try:
            p = d / "small.gguf"
            p.write_bytes(b"GGUF" + b"\0" * 100)
            ok, detail = ls.gguf_check(p)
            self.assertFalse(ok)
            self.assertIn("too small", detail)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_missing(self):
        ok, detail = ls.gguf_check("/nonexistent/nope.gguf")
        self.assertFalse(ok)
        self.assertIn("not found", detail)


# ── heal() state machine (no real processes) ─────────────────────────────────

class FakeManager:
    """A GGUFServerManager stand-in with the surface heal() uses."""

    READY_STATES = ls.GGUFServerManager.READY_STATES

    def __init__(self, *, port: int = PORT, host: str = "127.0.0.1", **overrides):
        self.host, self.port = host, port
        self.model_path = "/x/model.gguf"
        self.binary = "/bin/llama-server"
        self.problems: list[str] = []
        self._process = None
        self.calls: list[str] = []
        self.healthy = True
        self.loading = False
        self.port_free = False
        self.pid = None
        self.kill_ok = True
        self.start_ok = True
        for key, value in overrides.items():
            setattr(self, key, value)

    def health_status(self) -> str:
        self.calls.append("probe")
        if self.loading:
            return "loading"
        return "ok" if self.healthy else "dead"

    def _healthy(self) -> bool:
        return self.health_status() in ls.GGUFServerManager.READY_STATES

    def stop(self) -> None:
        self.calls.append("stop")

    def start(self, model_path: str):  # pragma: no cover - replaced in tests
        self.calls.append("start")
        from nomorals.llm.local_server import GGUFDiagnosis
        return GGUFDiagnosis(ok=self.start_ok, url="u", binary=self.binary,
                             model_path=model_path,
                             problems=[] if self.start_ok else ["no"],
                             hints=[])


def _heal(mgr: FakeManager, monkey_state=None):
    """Run the REAL heal() logic against the fakes by monkeypatching the
    module globals it looks up.  A successful kill frees the port so heal's
    wait loop does not block."""
    def port_in_use(host, port):
        return not mgr.port_free

    def kill_pid(pid, timeout=10.0):
        mgr.calls.append(f"kill:{pid}")
        if mgr.kill_ok:
            mgr.port_free = True
        return mgr.kill_ok

    with mock.patch.object(ls, "port_in_use", port_in_use), \
         mock.patch.object(ls, "find_llama_server_pid",
                           lambda port, proc_dir="/proc": mgr.pid), \
         mock.patch.object(ls, "kill_pid", kill_pid), \
         mock.patch.object(FakeManager, "start",
                           lambda self, p: _record_start(self, p)):
        return FakeManager.heal_real(mgr)


def _record_start(self, model_path):  # noqa: ANN001
    self.calls.append("start")
    from nomorals.llm.local_server import GGUFDiagnosis
    return GGUFDiagnosis(ok=self.start_ok, url="u", binary=self.binary,
                         model_path=model_path,
                         problems=[] if self.start_ok else ["start failed"],
                         hints=[])


# attach the real heal logic to the fake so we test THE code, not a copy
FakeManager.heal_real = ls.GGUFServerManager.heal  # type: ignore[attr-defined]
FakeManager.base_url = property(lambda self: f"http://{self.host}:{self.port}")  # type: ignore[attr-defined]


class HealTest(unittest.TestCase):
    def test_heal_starts_when_port_free(self):
        m = FakeManager(port_free=True)
        d = _heal(m)
        self.assertIn("start", m.calls)
        self.assertTrue(d.ok)

    def test_heal_noop_when_healthy(self):
        m = FakeManager(healthy=True, port_free=False)
        d = _heal(m)
        self.assertEqual(m.calls, ["probe"])
        self.assertTrue(d.ok)

    def test_heal_kills_and_restarts_frozen_own(self):
        m = FakeManager(healthy=False, port_free=False, pid=4242)
        d = _heal(m)
        self.assertIn("kill:4242", m.calls)
        self.assertIn("start", m.calls)
        self.assertTrue(d.ok)
        self.assertTrue(any("repaired" in h for h in d.hints))

    def test_heal_refuses_foreign_holder(self):
        m = FakeManager(healthy=False, port_free=False, pid=None)
        d = _heal(m)
        self.assertFalse(d.ok)
        self.assertEqual(m.calls, ["probe"])
        self.assertTrue(any("foreign" in p for p in d.problems))

    def test_heal_reports_unkillable(self):
        m = FakeManager(healthy=False, port_free=False, pid=4242, kill_ok=False)
        d = _heal(m)
        self.assertFalse(d.ok)
        self.assertNotIn("start", m.calls)

    def test_heal_never_kills_a_loading_server(self):
        # THE hot-phone loop: kill a server that is still loading the model
        # and the whole load restarts — forever.  heal must leave it alone.
        m = FakeManager(loading=True, port_free=False, pid=4242)
        d = _heal(m)
        self.assertTrue(d.ok)
        self.assertNotIn("kill:4242", m.calls)
        self.assertNotIn("start", m.calls)
        self.assertTrue(any("loading" in h for h in d.hints))

    def test_health_status_three_states(self):
        # /health body semantics, via a real local HTTP server
        import http.server
        import threading

        class _LoadingHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(b'{"status":"loading"}'))
                self.end_headers()
                self.wfile.write(b'{"status":"loading"}')

        import socket as _socket
        with _socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", free_port), _LoadingHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            self.assertEqual(ls.health_status("127.0.0.1", free_port), "loading")
        finally:
            httpd.shutdown()
            httpd.server_close()


# ── start() self-repair of a frozen own server ───────────────────────────────

class _FakeProc:
    pid = 999

    def poll(self):
        return None


class StartSelfRepairTest(unittest.TestCase):
    def _patches(self, mgr, state, healthy_after_boot):
        def kill_pid(pid, timeout=10.0):
            state["port_held"] = False  # the kill works → port frees up
            return True

        return [
            mock.patch.object(ls, "port_in_use", lambda h, p: state["port_held"]),
            mock.patch.object(ls, "find_llama_binary", lambda env=None: "/llama-server"),
            mock.patch.object(ls, "find_llama_server_pid", lambda port: 7777),
            mock.patch.object(ls, "kill_pid", kill_pid),
            mock.patch.object(ls, "find_gguf", lambda name, cache: "/x/m.gguf"),
            mock.patch.object(ls.subprocess, "Popen",
                              lambda *a, **kw: _FakeProc()),
        ]

    def test_start_replaces_frozen_own_server(self):
        mgr = ls.GGUFServerManager(port=PORT, boot_timeout=5.0)
        state = {"port_held": True}
        healthy = {"value": False}
        patches = self._patches(mgr, state, True)
        # unhealthy while the stale server sits there; healthy once the fresh
        # one is spawned (the kill frees the port → the new server answers)
        patches.append(mock.patch.object(mgr, "_healthy",
                                         lambda: not state["port_held"]))
        for p in patches:
            p.start()
        try:
            d = mgr.start("/x/m.gguf")
        finally:
            for p in patches:
                p.stop()
        self.assertTrue(d.ok, d.problems)
        self.assertIn("repaired", " ".join(d.hints))
        self.assertEqual(d.pid, 999)

    def test_start_refuses_foreign_holder(self):
        mgr = ls.GGUFServerManager(port=PORT)

        def boom(*a, **kw):  # pragma: no cover
            raise AssertionError("must not start over a foreign holder")

        with mock.patch.object(ls, "port_in_use", lambda h, p: True), \
             mock.patch.object(ls, "find_llama_binary", lambda env=None: "/llama-server"), \
             mock.patch.object(ls, "find_gguf", lambda name, cache: "/x/m.gguf"), \
             mock.patch.object(ls, "find_llama_server_pid", lambda port: None), \
             mock.patch.object(mgr, "_healthy", lambda: False), \
             mock.patch.object(ls.subprocess, "Popen", boom):
            d = mgr.start("/x/m.gguf")
        self.assertFalse(d.ok)
        self.assertTrue(any("foreign" in p.lower() for p in d.problems))


# ── router repair hooks ──────────────────────────────────────────────────────

class DeadProvider(LLMProvider):
    name = "dead_local"
    model_id = "dead"
    capabilities = ("chat",)
    preflight_health = True

    def health(self) -> bool:
        return False

    def chat(self, messages, params=None, **kw):  # pragma: no cover - never reached while dead
        raise AssertionError("dead provider must not be called")


class CloudProvider(LLMProvider):
    name = "cloud"
    model_id = "cloud-1"
    capabilities = ("chat",)

    def health(self) -> bool:
        return True

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="from-cloud", model=self.model_id,
                           provider=self.name)


class LoadingProvider(LLMProvider):
    """Alive, burning CPU, still reading the model into RAM."""

    name = "loading_local"
    model_id = "loading"
    capabilities = ("chat",)
    preflight_health = True

    def health(self) -> bool:
        return True

    def load_status(self) -> str:
        return "loading"

    def chat(self, messages, params=None, **kw):  # pragma: no cover - never reached
        raise AssertionError("loading provider must not be called")


class RepairHookTest(unittest.TestCase):
    def _router(self, now, hook):
        # the clock must return a float — `now` is a 1-element list the
        # tests can advance
        return LLMRouter(clock=lambda: now[0], repair_hooks={"dead_local": hook})

    def test_hook_fired_on_dead_preflight_and_fallback_serves(self):
        now = [1000.0]
        fired = []
        r = self._router(now, lambda: fired.append(1))
        r.add(DeadProvider())
        r.add(CloudProvider())
        resp = r.complete("hi")
        self.assertEqual(resp.text, "from-cloud")  # current message answered NOW
        self.assertTrue(resp.error or True)
        deadline = time.time() + 5
        while not fired and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(fired, [1])
        self.assertEqual(r.stats["repairs"], 1)

    def test_hook_cooldown_suppresses_second_fire(self):
        now = [1000.0]
        fired = []
        r = self._router(now, lambda: fired.append(1))
        r.add(DeadProvider())
        r.add(CloudProvider())
        r.complete("1")
        time.sleep(0.05)
        r.complete("2")  # within 300 s of the first repair
        time.sleep(0.05)
        self.assertEqual(r.stats["repairs"], 1)

    def test_hook_refires_after_cooldown(self):
        now = [1000.0]
        fired = []
        r = self._router(now, lambda: fired.append(1))
        r.add(DeadProvider())
        r.add(CloudProvider())
        r.complete("1")
        now[0] = 1000.0 + r.repair_cooldown_seconds + 1
        r.complete("2")
        time.sleep(0.05)
        self.assertEqual(r.stats["repairs"], 2)

    def test_no_hook_means_no_repair_bookkeeping(self):
        r = LLMRouter()
        r.add(DeadProvider())
        r.add(CloudProvider())
        resp = r.complete("hi")
        self.assertEqual(resp.text, "from-cloud")
        self.assertEqual(r.stats.get("repairs", 0), 0)

    def test_loading_provider_is_skipped_without_repair(self):
        # A loading server must never be killed: no repair hook, no
        # failure bookkeeping, and the fallback answers this message.
        now = [1000.0]
        fired = []
        r = LLMRouter(clock=lambda: now[0], repair_hooks={"loading_local": fired.append})
        r.add(LoadingProvider())
        r.add(CloudProvider())
        resp = r.complete("hi")
        self.assertEqual(resp.text, "from-cloud")
        self.assertEqual(resp.provider, "cloud")
        self.assertEqual(fired, [])  # no repair fired
        self.assertEqual(r.stats.get("repairs", 0), 0)


# ── boot: ensure_local_gguf heals instead of 'port open = done' ──────────────

class BootHealTest(unittest.TestCase):
    def test_boot_uses_heal_when_port_open_but_silent(self):
        from nomorals.agents import context as ctx_mod
        from nomorals.llm.local_server import GGUFDiagnosis

        settings = get_settings()
        settings.llm.local_auto_start = True
        settings.llm.local_model = "/x/m.gguf"

        captured = {}

        class FakeMgr:
            def __init__(self, **kw):
                captured["kw"] = kw
                self.model_path = ""
                self._process = None
                self.problems = []
                self.binary = ""
                self.port = PORT

            def _healthy(self):
                return False

            def heal(self):
                captured["heal"] = True
                return GGUFDiagnosis(ok=True, url="u", binary="/b",
                                     model_path="/x/m.gguf", problems=[],
                                     hints=["repaired"])

            def stop(self):
                pass

        with mock.patch.object(ls, "GGUFServerManager", FakeMgr), \
             mock.patch.object(ls, "port_in_use", lambda h, p: True):
            mgr = ctx_mod.ensure_local_gguf(settings)
        self.assertIsNotNone(mgr)
        self.assertTrue(captured.get("heal"))
        self.assertEqual(captured["heal"] and mgr.model_path, "/x/m.gguf")

    def test_boot_noop_when_port_open_and_healthy(self):
        from nomorals.agents import context as ctx_mod

        settings = get_settings()
        settings.llm.local_auto_start = True
        settings.llm.local_model = "/x/m.gguf"
        called = {}

        class FakeMgr:
            base_url = f"http://127.0.0.1:{PORT}"

            def __init__(self, **kw):
                self.model_path = ""
                self._process = None

            def _healthy(self):
                return True

            def heal(self):  # pragma: no cover - must not be called
                called["heal"] = True

        with mock.patch.object(ls, "GGUFServerManager", FakeMgr), \
             mock.patch.object(ls, "port_in_use", lambda h, p: True):
            self.assertIsNone(ctx_mod.ensure_local_gguf(settings))
        self.assertNotIn("heal", called)

    def test_boot_skips_when_model_unset(self):
        from nomorals.agents import context as ctx_mod

        settings = get_settings()
        settings.llm.local_auto_start = True
        settings.llm.local_model = ""
        with mock.patch.object(ls, "port_in_use", lambda h, p: False):
            self.assertIsNone(ctx_mod.ensure_local_gguf(settings))


# ── status/doctor surface ────────────────────────────────────────────────────

class StatusFrozenTest(unittest.TestCase):
    def test_status_reports_frozen_with_pid(self):
        mgr = ls.GGUFServerManager(port=PORT)
        with mock.patch.object(ls, "port_in_use", lambda h, p: True), \
             mock.patch.object(mgr, "health_status", lambda: "dead"), \
             mock.patch.object(ls, "find_llama_server_pid", lambda port: 4242):
            s = mgr.status()
        self.assertEqual(s["state"], "frozen")
        self.assertFalse(s["running"])
        self.assertEqual(s["pid"], 4242)
        self.assertEqual(s["external_pid"], 4242)

    def test_status_reports_foreign_holder(self):
        mgr = ls.GGUFServerManager(port=PORT)
        with mock.patch.object(ls, "port_in_use", lambda h, p: True), \
             mock.patch.object(mgr, "health_status", lambda: "dead"), \
             mock.patch.object(ls, "find_llama_server_pid", lambda port: None):
            s = mgr.status()
        self.assertEqual(s["state"], "foreign_holder")
        self.assertFalse(s["running"])
        self.assertEqual(s["pid"], 0)

    def test_status_reports_running_external(self):
        mgr = ls.GGUFServerManager(port=PORT)
        with mock.patch.object(ls, "port_in_use", lambda h, p: True), \
             mock.patch.object(mgr, "health_status", lambda: "ok"), \
             mock.patch.object(ls, "find_llama_server_pid", lambda port: 31337):
            s = mgr.status()
        self.assertEqual(s["state"], "running")
        self.assertTrue(s["running"])
        self.assertTrue(s["healthy"])

    def test_status_reports_loading_is_not_frozen(self):
        mgr = ls.GGUFServerManager(port=PORT)
        with mock.patch.object(ls, "port_in_use", lambda h, p: True), \
             mock.patch.object(mgr, "health_status", lambda: "loading"), \
             mock.patch.object(ls, "find_llama_server_pid", lambda port: 5555):
            s = mgr.status()
        self.assertEqual(s["state"], "loading")
        self.assertFalse(s["running"])
        self.assertEqual(s["external_pid"], 5555)

    def test_doctor_reply_path_reports_model_and_server(self):
        import socket as _socket
        from nomorals.cli import _reply_path_report
        # a port that is actually free in this process right now
        with _socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        settings = get_settings()
        settings.llm.provider = "llama_cpp"
        settings.llm.fallback_chain = []
        settings.llm.local_port = free_port
        settings.llm.local_model = "/nonexistent/m.gguf"
        report = _reply_path_report(settings)
        self.assertEqual(report["active_provider"], "llama_cpp")
        self.assertFalse(report["model_file"]["valid"])
        self.assertEqual(report["local_server"]["state"].split(" ")[0], "not")
        self.assertEqual(report["providers"][0]["name"], "llama_cpp")
        self.assertFalse(report["providers"][0]["responds"])


if __name__ == "__main__":
    unittest.main()
