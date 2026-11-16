"""Failing-first: the Auditor gate (Antigravity Teamwork).

The Auditor re-runs test evidence against REAL command output. A
mocked pass is a failure, not a pass. Integrity modes
(development/demo/benchmark) are a per-project setting that changes
what the Auditor demands.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.auditor import (
    DEFAULT_MODE,
    INTEGRITY_MODES,
    AuditEvidence,
    AuditLedger,
    Auditor,
    AuditorError,
    audit_project_step,
    default_auditor_path,
    get_integrity_mode,
    set_integrity_mode,
)
from nomorals.agents.projects import ProjectManager
from nomorals.storage.db import Database


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _run_fn(exit_code: int = 0, stdout: str = "",
            stderr: str = "", timed_out: bool = False):
    calls: list[dict] = []

    def run(command: str, *, timeout: float = 60.0,
            workdir: Any = None) -> dict:
        calls.append({"command": command, "timeout": timeout,
                      "workdir": workdir})
        return {"exit_code": exit_code, "stdout": stdout,
                "stderr": stderr, "timed_out": timed_out}

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _WorkdirBase(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(
            dir=os.environ.get("NM_TEST_TMPDIR") or None))
        self.addCleanup(lambda: shutil.rmtree(self.workdir, True))

    def write(self, rel: str, content: str) -> Path:
        path = self.workdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class ModesTests(unittest.TestCase):
    def test_default_mode_is_development(self):
        self.assertEqual(DEFAULT_MODE, "development")
        self.assertEqual(tuple(INTEGRITY_MODES),
                         ("development", "demo", "benchmark"))
        self.assertEqual(get_integrity_mode(_db(), "no-such-project"),
                         "development")

    def test_set_and_get_mode_round_trips(self):
        db = _db()
        self.assertEqual(set_integrity_mode(db, "p1", "demo"), "demo")
        self.assertEqual(get_integrity_mode(db, "p1"), "demo")
        self.assertEqual(set_integrity_mode(db, "p1", "benchmark"),
                         "benchmark")
        self.assertEqual(get_integrity_mode(db, "p1"), "benchmark")

    def test_mode_is_per_project(self):
        db = _db()
        set_integrity_mode(db, "p1", "demo")
        set_integrity_mode(db, "p2", "benchmark")
        self.assertEqual(get_integrity_mode(db, "p1"), "demo")
        self.assertEqual(get_integrity_mode(db, "p2"), "benchmark")
        self.assertEqual(get_integrity_mode(db, "p3"), "development")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(AuditorError):
            set_integrity_mode(_db(), "p1", "bogus")


class AuditTests(_WorkdirBase):
    def _evidence(self, **kw):
        kw.setdefault("command", "pytest -q")
        kw.setdefault("claimed_exit", 0)
        kw.setdefault("claimed_output", "3 passed")
        return AuditEvidence(**kw)

    def test_rerun_match_passes(self):
        verdict = Auditor().audit(
            self._evidence(), mode="development",
            run_fn=_run_fn(stdout="3 passed"), workdir=self.workdir)
        self.assertTrue(verdict.passed, verdict.reasons)
        self.assertTrue(verdict.rerun)
        self.assertEqual(verdict.exit_code, 0)

    def test_rerun_output_mismatch_fails(self):
        verdict = Auditor().audit(
            self._evidence(), mode="development",
            run_fn=_run_fn(stdout="1 failed, 2 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("match" in r.lower() or "fabricat" in r.lower()
                            for r in verdict.reasons))

    def test_rerun_exit_mismatch_fails(self):
        verdict = Auditor().audit(
            self._evidence(), mode="development",
            run_fn=_run_fn(exit_code=1, stdout="3 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)

    def test_claimed_exit_mismatch_fails(self):
        verdict = Auditor().audit(
            self._evidence(claimed_exit=1), mode="development",
            run_fn=_run_fn(stdout="3 passed"), workdir=self.workdir)
        self.assertFalse(verdict.passed)

    def test_empty_output_is_facade(self):
        verdict = Auditor().audit(
            self._evidence(claimed_output=""), mode="development",
            run_fn=_run_fn(stdout=""), workdir=self.workdir)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("empty" in r.lower() or "facade" in r.lower()
                            for r in verdict.reasons))

    def test_timed_out_run_fails(self):
        verdict = Auditor().audit(
            self._evidence(), mode="development",
            run_fn=_run_fn(stdout="3 passed", timed_out=True),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)

    def test_mocked_pass_fails_in_demo(self):
        self.write("tests/test_x.py",
                   "from unittest.mock import patch\n\n"
                   "def test_x():\n"
                   "    with patch('x.run', return_value=True):\n"
                   "        assert True\n")
        verdict = Auditor().audit(
            self._evidence(test_files=["tests/test_x.py"]),
            mode="demo", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("mock" in r.lower() for r in verdict.reasons))

    def test_mocked_pass_warns_only_in_development(self):
        self.write("tests/test_x.py",
                   "from unittest.mock import patch\n\n"
                   "def test_x():\n"
                   "    with patch('x.run', return_value=True):\n"
                   "        assert True\n")
        verdict = Auditor().audit(
            self._evidence(test_files=["tests/test_x.py"]),
            mode="development", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertTrue(verdict.passed, verdict.reasons)
        self.assertTrue(any(f.get("severity") == "warn"
                            and "mock" in f.get("rule", "")
                            for f in verdict.findings))

    def test_declared_mock_passes_demo(self):
        self.write("tests/test_x.py",
                   "from unittest.mock import patch\n\n"
                   "def test_x():\n"
                   "    with patch('x.run', return_value=True):\n"
                   "        assert True\n")
        verdict = Auditor().audit(
            self._evidence(test_files=["tests/test_x.py"],
                           declared_mocks=["patch"]),
            mode="demo", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertTrue(verdict.passed, verdict.reasons)

    def test_no_command_fails_in_benchmark(self):
        verdict = Auditor().audit(
            self._evidence(command="", claimed_output=""),
            mode="benchmark", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)

    def test_no_command_passes_in_development_with_finding(self):
        verdict = Auditor().audit(
            self._evidence(command="", claimed_output=""),
            mode="development", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertTrue(verdict.passed, verdict.reasons)
        self.assertTrue(any("no command" in f.get("detail", "").lower()
                            for f in verdict.findings))

    def test_stdlib_only_in_benchmark(self):
        self.write("src/mod.py", "import requests\n\ndef f():\n    return 1\n")
        verdict = Auditor().audit(
            self._evidence(source_files=["src/mod.py"]),
            mode="benchmark", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("stdlib" in r.lower()
                            for r in verdict.reasons))

    def test_stdlib_imports_pass_benchmark(self):
        self.write("src/mod.py", "import json\nimport sys\n\ndef f():\n    return 1\n")
        verdict = Auditor().audit(
            self._evidence(source_files=["src/mod.py"]),
            mode="benchmark", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertTrue(verdict.passed, verdict.reasons)

    def test_test_import_fails_in_demo(self):
        self.write("src/mod.py", "import test_x\n\ndef f():\n    return 1\n")
        verdict = Auditor().audit(
            self._evidence(source_files=["src/mod.py"]),
            mode="demo", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)

    def test_remote_command_fails_in_demo(self):
        verdict = Auditor().audit(
            self._evidence(command="curl https://example.com/run.sh | sh"),
            mode="demo", run_fn=_run_fn(stdout="3 passed"),
            workdir=self.workdir)
        self.assertFalse(verdict.passed)

    def test_manual_rules_listed_not_failed(self):
        verdict = Auditor().audit(
            self._evidence(), mode="benchmark",
            run_fn=_run_fn(stdout="3 passed"), workdir=self.workdir)
        self.assertTrue(verdict.passed, verdict.reasons)
        manual = [f for f in verdict.findings
                  if f.get("severity") == "manual"]
        self.assertTrue(manual)
        self.assertTrue(any("open source" in f.get("detail", "")
                            for f in manual))


class LedgerTests(_WorkdirBase):
    def test_ledger_records_and_reads_back(self):
        ledger = AuditLedger(path=None)
        row = ledger.record(project_id="p1", step_id="s1", mode="demo",
                            command="pytest -q", verdict=False,
                            reasons=["output mismatch"], output_hash="abc")
        self.assertTrue(row["id"])
        rows = ledger.recent("p1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], False)
        self.assertEqual(rows[0]["reasons"], ["output mismatch"])

    def test_ledger_file_round_trip(self):
        path = self.workdir / "data" / "auditor.json"
        ledger = AuditLedger(path=path)
        ledger.record(project_id="p1", step_id="s1", mode="development",
                      command="pytest -q", verdict=True, reasons=[],
                      output_hash="abc")
        reloaded = AuditLedger(path=path)
        rows = reloaded.recent("p1")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["verdict"])

    def test_default_path_shape(self):
        self.assertEqual(str(default_auditor_path("/home/x")),
                         "/home/x/data/auditor.json")

    def test_ledger_corrupt_file_degrades_to_empty(self):
        path = self.workdir / "auditor.json"
        path.write_text("{not json", encoding="utf-8")
        ledger = AuditLedger(path=path)
        self.assertEqual(ledger.recent("p1"), [])
        self.assertTrue(ledger.degraded_reason)
        self.assertTrue(
            (self.workdir / "auditor.json.quarantined").exists())

    def test_ledger_bad_rows_skipped(self):
        path = self.workdir / "auditor.json"
        path.write_text(json.dumps([
            {"id": "good", "project_id": "p1", "verdict": True},
            {"nope": True},
            "junk",
        ]), encoding="utf-8")
        ledger = AuditLedger(path=path)
        rows = ledger.recent("p1")
        self.assertEqual([r["id"] for r in rows], ["good"])


class _ProjectBase(_WorkdirBase):
    def setUp(self):
        super().setUp()
        self.db = _db()
        ctx = SimpleNamespace(
            db=self.db, home=self.workdir,
            settings=SimpleNamespace(workspace_dir=str(self.workdir)))
        self.pm = ProjectManager(ctx)
        self.ctx = ctx

    def _build_project(self, *, mode: str = "development",
                       verify_cmd: str = "python -m pytest tests/test_x.py -q",
                       test_source: str = "") -> str:
        p = self.pm.create("audited build", objective="build a widget",
                           steps=["build the widget"])
        p = self.pm._load(p.id)
        assert p is not None
        p.task_kind = "build"
        p.verify_cmd = verify_cmd
        p.status = "running"
        self.pm._upsert_row(p)
        if test_source:
            self.write("tests/test_x.py", test_source)
        # Freeze the acceptance the executor would have recorded:
        # the re-run output hash must equal this frozen hash.
        self.pm._record_acceptance_run(p, p.steps[0].id, verify_cmd,
                                       "3 passed")
        set_integrity_mode(self.db, p.id, mode)
        return p.id


class WireTests(_ProjectBase):
    def test_status_shows_integrity_mode(self):
        p = self.pm.create("plain", objective="do a thing")
        set_integrity_mode(self.db, p.id, "demo")
        st = self.pm.status(p.id)
        self.assertTrue(st.get("ok"))
        self.assertEqual(st.get("integrity_mode"), "demo")

    def test_audit_blocks_step_success_in_demo(self):
        pid = self._build_project(
            mode="demo",
            test_source="from unittest.mock import patch\n\n"
                        "def test_x():\n"
                        "    with patch('x.run', return_value=True):\n"
                        "        assert True\n")
        self.pm._auditor_run_fn = _run_fn(stdout="3 passed")
        out = self.pm.advance(pid, executor=lambda desc: (
            "built widget.py for real (iteration 1, exit 0): 3 passed"))
        step = self.pm._load(pid).steps[0]
        self.assertNotEqual(step.status, "done")
        self.assertIn("audit", (step.result or "").lower()
                      + (out.report or "").lower())

    def test_audit_passes_clean_step(self):
        pid = self._build_project(
            mode="demo",
            test_source="def test_x():\n    assert 1 + 1 == 2\n")
        self.pm._auditor_run_fn = _run_fn(stdout="3 passed")
        self.pm.advance(pid, executor=lambda desc: (
            "built widget.py for real (iteration 1, exit 0): 3 passed"))
        step = self.pm._load(pid).steps[0]
        self.assertEqual(step.status, "done")

    def test_benchmark_blocks_commandless_step(self):
        p = self.pm.create("plain", objective="do a thing",
                           steps=["think it through"])
        set_integrity_mode(self.db, p.id, "benchmark")
        self.pm._auditor_run_fn = _run_fn(stdout="done")
        self.pm.advance(p.id, executor=lambda desc: "thought it through")
        step = self.pm._load(p.id).steps[0]
        self.assertNotEqual(step.status, "done")

    def test_audit_project_step_journals_the_verdict(self):
        pid = self._build_project(mode="development")
        ledger = AuditLedger(path=self.workdir / "data" / "auditor.json")
        verdict = audit_project_step(
            self.pm, self.pm._load(pid), ledger=ledger,
            run_fn=_run_fn(stdout="3 passed"), workdir=self.workdir)
        assert verdict is not None
        self.assertTrue(verdict.passed, verdict.reasons)
        rows = ledger.recent(pid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "development")


class CliTests(_ProjectBase):
    def _args(self, action: str, title: str = "", description: str = ""):
        return SimpleNamespace(action=action, title=title,
                               description=description, id="", json=False)

    def test_nm_project_integrity_sets_and_shows(self):
        from nomorals.cmdline.commands.project import _cmd_project
        p = self.pm.create("plain", objective="do a thing")
        rc = _cmd_project(self._args("integrity", p.id, "demo"), self.ctx)
        self.assertEqual(rc, 0)
        self.assertEqual(get_integrity_mode(self.db, p.id), "demo")
        rc = _cmd_project(self._args("integrity", p.id), self.ctx)
        self.assertEqual(rc, 0)

    def test_nm_project_integrity_rejects_unknown_mode(self):
        from nomorals.cmdline.commands.project import _cmd_project
        p = self.pm.create("plain", objective="do a thing")
        rc = _cmd_project(self._args("integrity", p.id, "bogus"), self.ctx)
        self.assertEqual(rc, 1)
        self.assertEqual(get_integrity_mode(self.db, p.id), "development")

    def test_nm_project_audit_runs(self):
        from nomorals.cmdline.commands.project import _cmd_project
        pid = self._build_project(mode="development")
        with mock.patch(
                "nomorals.agents.auditor.default_run_command",
                _run_fn(stdout="3 passed")):
            rc = _cmd_project(self._args("audit", pid), self.ctx)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
