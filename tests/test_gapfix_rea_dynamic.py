"""Lane 96 — REA dynamic rung live proof: GDB exit codes are octal.

FINDING (proven live 2026-11-18 on this box, GDB 15.1): GDB renders
the exit status in OCTAL in its "[Inferior 1 (process N) exited with
code NNN]" line — 052 is 42, 0177 is 127, 0377 is 255 — while
``_parse_gdb_log`` read the token as decimal, so every GDB session
whose target exited >= 8 filed the WRONG exit code in its Evidence
record (42 filed as 52) and showed it wrong on the /re dynamic card.
The existing suite never caught it because its live target returns
0, and exit 0 prints no "exited with code" line at all (only
"$1 = 0" from ``print $_exitcode``).

The table below is measured, not assumed: real GDB 15.1 sessions
against targets returning 0/1/7/8/42/127/255 (proof in
~/workspace/reaproof-verify/exp-exitcodes). The live tests here run
a target returning 42 through the real rung — run_gdb, run_dynamic
auto, ReEngine folding, and the /re dynamic chat verb — so the true
exit code is proven at every layer, not just the parser.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.reverse import dynamic
from nomorals.agents.reverse.dynamic import _parse_gdb_log
from nomorals.storage.db import Database

_TARGET_42_C = r"""
#include <stdio.h>
#include <string.h>

static int add(int a, int b) { return a + b; }

int main(int argc, char **argv) {
    int total = add(40, 2);
    printf("total=%d\n", total);
    if (argc > 1) {
        printf("arg=%s len=%zu\n", argv[1], strlen(argv[1]));
    }
    puts("done");
    return 42;
}
"""

# The reduced-mode log shape, verbatim from a real GDB 15.1 session
# on this box (starti + entry registers, no fresh /proc) — see
# ~/workspace/reaproof-verify/proof-gdb.json.
_REDUCED_LOG = """
Program stopped.
0x00007ffff7fe4530 in _start () from /lib64/ld-linux-x86-64.so.2
rax            0x0                 0
rip            0x7ffff7fe4530      0x7ffff7fe4530 <_start>
rsp            0x7fffffffeb70      0x7fffffffeb70
total=42
arg=hello len=5
done
[Inferior 1 (process 8) exited with code 052]
$1 = 42
"""

# The full-mode log shape (a device whose sandbox mounts a fresh
# /proc): breakpoint at main, args/locals/registers, exit line.
_FULL_LOG = """
Breakpoint 1, main (argc=2, argv=0x7fffffffeb78) at known.c:10
argc            = 2
argv            = (char **) 0x7fffffffeb78
total           = 42
#0  main (argc=2, argv=0x7fffffffeb78) at known.c:10
rax            0x2a                42
rip            0x5555555551ba      0x5555555551ba <main+13>
[Inferior 1 (process 8) exited with code 052]
$1 = 42
"""


def _compile_target(workdir: str) -> str:
    src = os.path.join(workdir, "target42.c")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(_TARGET_42_C)
    binary = os.path.join(workdir, "target42")
    done = subprocess.run(
        ["gcc", "-g", "-O0", "-o", binary, src],
        capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise unittest.SkipTest(
            f"gcc could not build the target: {done.stderr}")
    return binary


def _gdb_ready() -> bool:
    try:
        from nomorals.agents.reverse import provision

        return bool(provision.gdb_state()["available"])
    except Exception:  # noqa: BLE001 - readiness is best-effort
        return False


def _sandbox_ready() -> bool:
    try:
        return bool(dynamic.sandbox_state()["available"])
    except Exception:  # noqa: BLE001 - readiness is best-effort
        return False


def _live_ready() -> bool:
    return _gdb_ready() and _sandbox_ready() and shutil.which("gcc") is not None


class _TmpRoot:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="rea-dynamic-96-")
        return Path(self.dir)

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


class ParseExitCodeTest(unittest.TestCase):
    # GDB's token is octal: (token, true exit code), measured.
    OCTAL_TABLE = [
        ("01", 1), ("07", 7), ("010", 8), ("052", 42),
        ("0177", 127), ("0377", 255),
    ]

    def test_exited_with_code_token_is_octal(self):
        for token, want in self.OCTAL_TABLE:
            parsed = _parse_gdb_log(
                f"[Inferior 1 (process 8) exited with code {token}]\n")
            self.assertEqual(parsed["exit_code"], want,
                             f"token {token!r} parsed as decimal, not octal")

    def test_reduced_mode_log_reports_42_not_52(self):
        parsed = _parse_gdb_log(_REDUCED_LOG)
        self.assertEqual(parsed["exit_code"], 42, parsed)
        self.assertIsNone(parsed["signal"])

    def test_full_mode_log_reports_42_not_52(self):
        parsed = _parse_gdb_log(_FULL_LOG)
        self.assertEqual(parsed["exit_code"], 42, parsed)
        self.assertEqual(len(parsed["stops"]), 1)
        self.assertEqual(parsed["stops"][0]["function"], "main")

    def test_exit_zero_comes_from_exitcode_variable(self):
        # Exit 0 prints no "exited with code" line at all — only
        # "$1 = 0" from `print $_exitcode`.
        parsed = _parse_gdb_log("$1 = 0\n")
        self.assertEqual(parsed["exit_code"], 0)

    def test_reduced_mode_log_parses_starti_stop(self):
        parsed = _parse_gdb_log(_REDUCED_LOG)
        self.assertEqual(len(parsed["stops"]), 1)
        stop = parsed["stops"][0]
        self.assertEqual(stop["function"], "_start")
        self.assertIn("rip", stop["registers"])
        self.assertIn("rsp", stop["registers"])

    def test_signal_is_recorded_not_an_exit_code(self):
        parsed = _parse_gdb_log(
            "Program received signal SIGSEGV, Segmentation fault.\n"
            "0x00005555555551a7 in main () at known.c:10\n")
        self.assertEqual(parsed["signal"], "SIGSEGV")
        self.assertIsNone(parsed["exit_code"])


class LiveGdbExitCodeTest(unittest.TestCase):
    @unittest.skipUnless(_live_ready(),
                         "gdb rung, namespace sandbox, or gcc not ready")
    def test_live_gdb_reports_real_exit_code_42(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            res = dynamic.run_gdb(binary, root=root, args=["hello"],
                                  timeout=60.0)
            self.assertEqual(res["rung"], "gdb")
            self.assertEqual(res["exit_code"], 42, res)
            self.assertFalse(res["timed_out"])
            self.assertTrue(res["stops"], res)
            first = res["stops"][0]
            if dynamic.sandbox_state()["proc_confined"]:
                self.assertEqual(first["function"], "main")
            else:
                # No fresh /proc on this box: honestly reduced to
                # starti + entry registers — a real stop, never a
                # faked breakpoint.
                self.assertEqual(first["function"], "_start")
            self.assertIn("rip", first["registers"])
            self.assertIn("total=42", res["stdout"])
            self.assertIn("arg=hello len=5", res["stdout"])

    @unittest.skipUnless(_live_ready(),
                         "gdb rung, namespace sandbox, or gcc not ready")
    def test_live_auto_picks_gdb_and_reports_42(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            res = dynamic.run_dynamic(binary, root=root, rung="auto",
                                      args=["hello"], timeout=60.0)
            self.assertEqual(res["rung"], "gdb", res)
            self.assertEqual(res["exit_code"], 42, res)

    @unittest.skipUnless(_live_ready(),
                         "gdb rung, namespace sandbox, or gcc not ready")
    def test_live_engine_folds_true_exit_code(self):
        from nomorals.agents.reverse import ReEngine

        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            engine = ReEngine(db=Database(":memory:"))
            res = engine.run_dynamic(binary, rung="gdb",
                                     args=["hello"], timeout=60.0)
            self.assertFalse(res.get("error"), res)
            self.assertEqual(res["exit_code"], 42, res)
            self.assertTrue(res["evidence_ids"], res)
            self.assertEqual(res["unknowns"], [])
            rec = engine.store.get(res["evidence_ids"][0])
            self.assertIsNotNone(rec)
            self.assertEqual(rec.predicate_type, "dynamic.gdb.session")
            self.assertEqual(rec.result["exit_code"], 42)
            self.assertEqual(rec.confidence, "observed")
            self.assertIn("exit code: 42", res["card_text"])
            self.assertNotIn("exit code: 52", res["card_text"])


class VerbTest(unittest.TestCase):
    def _runtime(self):
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

        class _RT(RuntimeGamesMixin):
            pass

        rt = _RT()
        rt.context = SimpleNamespace(db=Database(":memory:"),
                                     memory=None, settings=None)
        return rt

    @unittest.skipUnless(_live_ready(),
                         "gdb rung, namespace sandbox, or gcc not ready")
    def test_control_re_dynamic_card_shows_true_exit_code(self):
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            rt = self._runtime()
            card = rt._control_re(f"dynamic {binary} gdb")
            self.assertIn("dynamic run (gdb)", card)
            self.assertIn("exit code: 42", card)
            self.assertNotIn("exit code: 52", card)
            self.assertIn("residual unknowns: none raised by this run",
                          card)

    @unittest.skipUnless(_live_ready(),
                         "gdb rung, namespace sandbox, or gcc not ready")
    def test_control_re_dynamic_frida_refusal_names_the_wall(self):
        # Frida is provisioned but this box's sandbox has no fresh
        # /proc, so the verb must refuse naming the wall — never a
        # fake trace, never a crash.
        if dynamic.sandbox_state()["proc_confined"]:
            self.skipTest("fresh /proc here: frida can attach")
        with _TmpRoot() as root:
            binary = _compile_target(str(root))
            rt = self._runtime()
            card = rt._control_re(f"dynamic {binary} frida")
            self.assertIn("proc", card.lower(), card)


if __name__ == "__main__":
    unittest.main()
