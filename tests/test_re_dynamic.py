"""RE dynamic rung — Frida + GDB under the namespace sandbox.

The gate is dissolved (owner order 2026-11-12): the dynamic rung
is built, and its sandbox is part of the build, not a question.
These tests pin the machinery around the real engines the way
the Pass 2 provision tests do: no test downloads an engine
(downloaders/runners are injected, artifacts are tiny fakes), and
the real-engine runs below execute a REAL compiled target — a
small C program built with gcc into the test's own tmp dir —
never a mock. Real-run tests skip when the engine is not ready
on THIS machine (a provisionable Frida or a missing GDB is an
honest skip, not a failure); Frida's real runs also skip where
the sandbox has no fresh /proc — Frida cannot attach there, and
GDB runs reduced (starti + entry registers) — the verify area
proves the real provision + real runs end to end.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.reverse import dynamic, provision
from nomorals.agents.reverse.dynamic import DynamicError
from nomorals.storage.db import Database

_TARGET_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

int add(int a, int b) { return a + b; }
int greet(int n) { return add(n, 1) * 2; }

int main(int argc, char **argv) {
    int total = 0;
    for (int i = 0; i < 5; i++) total += greet(i);
    printf("total=%d\n", total);
    FILE *f = fopen("marker.txt", "w");
    if (f) { fputs("hello", f); fclose(f); }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "sleep") == 0 && i + 1 < argc) {
            int secs = atoi(argv[i + 1]);
            printf("sleeping %d\n", secs);
            fflush(stdout);
            sleep((unsigned)secs);
        }
        if (strcmp(argv[i], "connect") == 0 && i + 1 < argc) {
            int port = atoi(argv[i + 1]);
            int s = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons((unsigned short)port);
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            int rc = connect(s, (struct sockaddr *)&addr, sizeof(addr));
            printf("connect %s\n", rc == 0 ? "CONNECT_OK" : "CONNECT_FAIL");
            fflush(stdout);
            close(s);
        }
    }
    printf("done\n");
    return 0;
}
"""


def _compile_target(workdir: str) -> str:
    src = os.path.join(workdir, "target.c")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(_TARGET_C)
    binary = os.path.join(workdir, "target")
    done = subprocess.run(
        ["gcc", "-g", "-O0", "-o", binary, src],
        capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise unittest.SkipTest(f"gcc could not build the target: {done.stderr}")
    return binary


def _frida_ready() -> bool:
    try:
        return bool(provision.frida_state()["available"])
    except Exception:  # noqa: BLE001 - readiness is best-effort
        return False


def _gdb_ready() -> bool:
    try:
        return bool(provision.gdb_state()["available"])
    except Exception:  # noqa: BLE001 - readiness is best-effort
        return False


def _sandbox_ready() -> bool:
    try:
        return bool(dynamic.sandbox_state()["available"])
    except Exception:  # noqa: BLE001 - readiness is best-effort
        return False


def _proc_confined() -> bool:
    """True when the sandbox mounts a fresh /proc — the condition
    Frida's attach and GDB's breakpoints both need."""
    try:
        return bool(dynamic.sandbox_state()["proc_confined"])
    except Exception:  # noqa: BLE001 - readiness is best-effort
        return False


class _TmpRoot:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-dynamic-")
        return Path(self.dir)

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


class StateTest(unittest.TestCase):
    def test_frida_state_unavailable_on_empty_root(self):
        with _TmpRoot() as root:
            state = provision.frida_state(root)
            self.assertFalse(state["available"])
            self.assertIn(state["state"],
                          ("provisionable", "unavailable"))
            self.assertTrue(state["detail"])
            self.assertEqual(state["engine"], "frida")

    def test_gdb_state_matches_the_system(self):
        state = provision.gdb_state()
        has_gdb = shutil.which("gdb") is not None
        self.assertEqual(state["available"], has_gdb)
        if has_gdb:
            self.assertEqual(state["state"], "ready")
            self.assertTrue(state["version"])
        else:
            self.assertTrue(state["detail"])

    def test_gdb_state_unavailable_when_no_gdb(self):
        with mock.patch(
                "nomorals.agents.reverse.provision.shutil.which",
                return_value=None):
            state = provision.gdb_state()
            self.assertFalse(state["available"])
            self.assertIn("gdb", state["detail"].lower())

    def test_sandbox_state_names_the_backend(self):
        state = dynamic.sandbox_state()
        self.assertIn(state["backend"], ("bwrap", "unshare", "rlimit"))
        self.assertEqual(state["available"],
                         state["backend"] in ("bwrap", "unshare"))
        self.assertTrue(state["detail"])

    def test_dynamic_state_is_probe_true(self):
        with _TmpRoot() as root:
            state = dynamic.dynamic_state(root)
            self.assertEqual(state["frida"]["available"],
                             provision.frida_state(root)["available"]
                             and state["sandbox"]["proc_confined"])
            self.assertEqual(state["gdb"]["available"],
                             provision.gdb_state()["available"])
            self.assertEqual(
                state["available"],
                state["sandbox"]["available"]
                and (state["frida"]["available"]
                     or state["gdb"]["available"]))
            self.assertIn(state.get("rung", ""),
                          ("", "frida", "gdb"))
        # On the default root Frida may be provisioned yet still
        # unavailable where the sandbox has no fresh /proc — the
        # detail must name that wall, not just say "ready".
        default = dynamic.dynamic_state()
        self.assertEqual(default["frida"]["available"],
                         provision.frida_state()["available"]
                         and default["sandbox"]["proc_confined"])
        if (provision.frida_state()["available"]
                and not default["sandbox"]["proc_confined"]):
            self.assertIn("proc", default["frida"]["detail"].lower())


class ProvisionTest(unittest.TestCase):
    def test_frida_pins_file_exists_and_covers_the_pin(self):
        pins = provision.frida_pins_path()
        self.assertTrue(pins.is_file(), pins)
        text = pins.read_text(encoding="utf-8")
        self.assertIn("frida==17.26.2", text)
        self.assertIn("--hash=sha256:", text)
        descriptor = provision.DESCRIPTORS["frida"]
        self.assertEqual(descriptor["version"], "17.26.2")
        self.assertEqual(descriptor["kind"], "pip-venv")

    def test_provision_frida_with_fake_runner_lands_and_probes(self):
        # The machinery, pinned the Pass 2 way: the venv is a tiny
        # fake shaped like the real one (a python that answers the
        # frida import probe), pip is faked to success, and the
        # install is only trusted because the probe answers.
        with _TmpRoot() as root:
            def fake_run(argv, *, env=None, timeout=30.0):
                if argv[1:3] == ["-m", "venv"]:
                    venv = Path(argv[3])
                    bindir = venv / "bin"
                    bindir.mkdir(parents=True, exist_ok=True)
                    fake_py = bindir / "python"
                    fake_py.write_text("#!/bin/sh\nexit 0\n",
                                       encoding="utf-8")
                    fake_py.chmod(0o755)
                    return (0, "")
                if "pip" in argv:
                    self.assertIn("--require-hashes", argv)
                    return (0, "installed")
                if argv[1:3] == ["-c", "import frida; print(frida.__version__)"]:
                    # the venv probe answers; the system probe
                    # (argv[0] is the system python) does not —
                    # otherwise the initial state check would
                    # short-circuit as "already provisioned".
                    if "frida" in str(argv[0]):
                        return (0, "17.26.2\n")
                    return (0, "")
                return (0, "")

            row = provision.provision_engine(
                "frida", root=root, runner=fake_run)
            self.assertTrue(row["available"], row)
            self.assertEqual(row["version"], "17.26.2")
            self.assertTrue(any("probe passed" in a
                                for a in row["attempts"]))

    def test_provision_gdb_with_fake_runner(self):
        with _TmpRoot() as root:
            probes = {"n": 0}

            def fake_run(argv, *, env=None, timeout=30.0):
                if argv[:2] == ["apt-get", "update"]:
                    return (0, "")
                if argv[:2] == ["apt-get", "install"]:
                    self.assertIn("gdb", argv)
                    return (0, "Setting up gdb")
                if (Path(argv[0]).name == "gdb"
                        and argv[1:] == ["--version"]):
                    # first probe (pre-install): no gdb yet;
                    # later probes answer the installed version
                    probes["n"] += 1
                    if probes["n"] == 1:
                        return (0, "")
                    return (0, "GNU gdb (Ubuntu) 15.1\n")
                return (0, "")

            with mock.patch(
                    "nomorals.agents.reverse.provision.shutil.which",
                    side_effect=lambda name: "/usr/bin/gdb"
                    if name == "gdb" else None):
                with mock.patch.object(
                        provision, "_apt_available",
                        return_value=True):
                    row = provision.provision_engine(
                        "gdb", root=root, runner=fake_run)
            self.assertTrue(row["available"], row)
            self.assertEqual(row["version"], "15.1")
            self.assertTrue(any("apt-get install" in a
                                for a in row["attempts"]))

    def test_provision_gdb_refuses_without_root(self):
        # gdb_state is mocked unavailable: on a box with a real
        # gdb the drive would short-circuit as already provisioned
        # before ever reaching the root check.
        with mock.patch.object(
                provision, "gdb_state",
                return_value={"engine": "gdb", "state": "provisionable",
                              "available": False, "detail": "no gdb"}):
            with mock.patch.object(provision.os, "geteuid",
                                   return_value=1000):
                with mock.patch.object(
                        provision, "_apt_available", return_value=True):
                    with self.assertRaises(provision.ProvisionError) as ctx:
                        provision.provision_engine("gdb", root=Path("/tmp"))
                    self.assertTrue(any("root" in a.lower()
                                        for a in ctx.exception.attempts))


class FridaTraceTest(unittest.TestCase):
    @unittest.skipUnless(_frida_ready() and _sandbox_ready()
                         and _proc_confined(),
                         "frida rung, namespace sandbox, or fresh "
                         "/proc not ready")
    def test_frida_traces_real_target_functions(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            # no root: the real Frida lives at the default tools
            # root (provisioned for real); the tmp root holds only
            # the compiled target
            res = dynamic.run_frida(binary, timeout=60.0)
            self.assertFalse(res.get("error"), res)
            self.assertEqual(res["rung"], "frida")
            self.assertEqual(res["exit_code"], 0, res)
            self.assertFalse(res["timed_out"])
            seen = {e["function"] for e in res["events"]}
            for name in ("main", "add", "greet"):
                self.assertIn(name, seen, seen)
            self.assertIn("total=30", res["stdout"])
            self.assertEqual(res["sandbox"]["backend"], "unshare")
            self.assertTrue(res["sandbox"]["network_enforced"])

    @unittest.skipUnless(_frida_ready() and _sandbox_ready()
                         and _proc_confined(),
                         "frida rung, namespace sandbox, or fresh "
                         "/proc not ready")
    def test_frida_timeout_kills_the_target(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            res = dynamic.run_frida(binary, args=("sleep", "30"),
                                    timeout=5.0)
            self.assertTrue(res["timed_out"], res)
            self.assertLess(res["duration_s"], 20.0)

    @unittest.skipUnless(_frida_ready() and _sandbox_ready()
                         and _proc_confined(),
                         "frida rung, namespace sandbox, or fresh "
                         "/proc not ready")
    def test_frida_run_has_no_network(self):
        # A listener on the host loopback: the target can only
        # reach it when it shares our network namespace. Inside
        # the sandbox it must fail — that is the proof the net
        # namespace is what ran, not a flag that was passed.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def _serve():
            try:
                conn, _ = listener.accept()
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        try:
            with _TmpRoot() as root:
                binary = _compile_target(str(root))
                res = dynamic.run_frida(
                    binary, args=("connect", str(port)),
                    timeout=60.0)
            self.assertFalse(res.get("error"), res)
            self.assertIn("CONNECT_FAIL", res["stdout"], res)
        finally:
            listener.close()

    @unittest.skipUnless(_frida_ready() and _sandbox_ready()
                         and not _proc_confined(),
                         "only meaningful where Frida is provisioned "
                         "but the sandbox has no fresh /proc")
    def test_frida_unavailable_without_fresh_proc(self):
        # The honest-unavailable path on this box: Frida is
        # provisioned, the sandbox runs, but attach cannot resolve
        # the namespace pid through the host's /proc — so the run
        # refuses and names the wall instead of failing inside.
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            with self.assertRaises(DynamicError) as ctx:
                dynamic.run_frida(binary)
            self.assertIn("proc", str(ctx.exception).lower())

    def test_frida_unavailable_is_an_error_not_a_fake_trace(self):
        with _TmpRoot() as root:
            with mock.patch.object(
                    provision, "frida_state",
                    return_value={"engine": "frida", "state": "unavailable",
                                  "available": False, "detail": "no frida"}):
                with self.assertRaises(DynamicError) as ctx:
                    dynamic.run_frida("/bin/true", root=root)
                self.assertIn("frida", str(ctx.exception).lower())


class GdbSessionTest(unittest.TestCase):
    @unittest.skipUnless(_gdb_ready() and _sandbox_ready(),
                         "gdb rung or namespace sandbox not ready")
    def test_gdb_breaks_at_main_and_reads_frames(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            res = dynamic.run_gdb(binary, root=root, timeout=60.0)
            self.assertFalse(res.get("error"), res)
            self.assertEqual(res["rung"], "gdb")
            self.assertEqual(res["exit_code"], 0, res)
            self.assertTrue(res["stops"], res)
            first = res["stops"][0]
            if _proc_confined():
                self.assertEqual(first["function"], "main")
                frames = [f["function"] for f in first["frames"]]
                self.assertIn("main", frames)
            else:
                # No fresh /proc on this box: the session is
                # honestly reduced to starti + entry registers —
                # a real stop at the entry point, never a faked
                # breakpoint.
                self.assertEqual(first["function"], "_start")
            self.assertTrue(first["registers"])
            self.assertEqual(res["sandbox"]["backend"], "unshare")
            self.assertTrue(res["sandbox"]["network_enforced"])

    @unittest.skipUnless(_gdb_ready() and _sandbox_ready(),
                         "gdb rung or namespace sandbox not ready")
    def test_gdb_timeout_kills_the_target(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            res = dynamic.run_gdb(binary, root=root, args=("sleep", "30"),
                                  timeout=5.0)
            self.assertTrue(res["timed_out"], res)
            self.assertLess(res["duration_s"], 20.0)

    def test_gdb_unavailable_is_an_error(self):
        with _TmpRoot() as root:
            with mock.patch.object(
                    provision, "gdb_state",
                    return_value={"engine": "gdb", "state": "unavailable",
                                  "available": False, "detail": "no gdb"}):
                with self.assertRaises(DynamicError):
                    dynamic.run_gdb("/bin/true", root=root)


class SandboxEnforcementTest(unittest.TestCase):
    def test_dynamic_refuses_without_a_namespace_sandbox(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            with mock.patch(
                    "nomorals.agents.reverse.dynamic.detect_backend",
                    return_value="rlimit"):
                with self.assertRaises(DynamicError) as ctx:
                    dynamic.run_dynamic(binary, root=root, rung="gdb")
                self.assertIn("sandbox", str(ctx.exception).lower())

    def test_dynamic_runs_a_copy_never_the_original(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            before = Path(binary).read_bytes()
            res = dynamic.run_dynamic(binary, root=root, rung="gdb")
            self.assertFalse(res.get("error"), res)
            self.assertEqual(Path(binary).read_bytes(), before)
            self.assertNotEqual(res["run_path"], binary)
            self.assertTrue(Path(res["run_path"]).is_file())


class EvidenceTest(unittest.TestCase):
    def _engine(self):
        from nomorals.agents.reverse import ReEngine

        return ReEngine(db=Database(":memory:"))

    @unittest.skipUnless(_sandbox_ready()
                         and dynamic.dynamic_state()["rung"],
                         "no dynamic rung ready on this machine")
    def test_run_dynamic_folds_evidence_with_the_rung_named(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            engine = self._engine()
            # the rung that can actually run HERE (Frida needs a
            # fresh /proc; where it has none, GDB is the rung)
            rung = dynamic.dynamic_state()["rung"]
            # no root: the real Frida lives at the default tools
            # root; GDB is a system binary either way
            res = engine.run_dynamic(binary, rung=rung)
            self.assertFalse(res.get("error"), res)
            self.assertEqual(res["rung"], rung)
            self.assertTrue(res["evidence_ids"], res)
            rec = engine.store.get(res["evidence_ids"][0])
            self.assertIsNotNone(rec)
            self.assertEqual(rec.provider_id, dynamic.PROVIDER_ID)
            self.assertIn(rung, rec.predicate_type)
            self.assertEqual(rec.authority, "controlled-replay")
            self.assertEqual(rec.confidence, "observed")
            self.assertTrue(rec.limitations)
            self.assertIn("card_text", res)

    def test_run_dynamic_unknown_rung_is_an_error(self):
        engine = self._engine()
        res = engine.run_dynamic("/bin/true", rung="strace")
        self.assertIn("error", res)

    def test_run_dynamic_missing_file_is_an_error(self):
        engine = self._engine()
        res = engine.run_dynamic("/no/such/file", rung="gdb")
        self.assertIn("error", res)


class MatrixTest(unittest.TestCase):
    def test_capability_matrix_has_real_dynamic_rows(self):
        from nomorals.agents.reverse import capability_matrix

        rows = {r["engine"]: r for r in capability_matrix()}
        self.assertNotIn("frida-gdb", rows)
        for name in ("frida", "gdb", "dynamic-sandbox"):
            self.assertIn(name, rows)
            self.assertTrue(rows[name]["note"])
        self.assertEqual(rows["gdb"]["available"],
                         provision.gdb_state()["available"])
        self.assertEqual(rows["frida"]["available"],
                         provision.frida_state()["available"]
                         and dynamic.sandbox_state()["proc_confined"])


class ToolTest(unittest.TestCase):
    def test_re_inspect_dynamic_action(self):
        from nomorals.agents.reverse import register
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(db=Database(":memory:"))
        registry = ToolRegistry(context, enforce=False)
        register(registry)
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            if not (_gdb_ready() and _sandbox_ready()):
                self.skipTest("gdb rung or namespace sandbox not ready")
            out = registry.call("re_inspect", action="dynamic",
                                path=binary, rung="gdb")
            self.assertTrue(out.ok, out)
            out = out.value
            self.assertFalse(out.get("error"), out)
            self.assertEqual(out["rung"], "gdb")
            self.assertTrue(out["evidence_ids"])


if __name__ == "__main__":
    unittest.main()
