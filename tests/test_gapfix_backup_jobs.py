"""Gapfix 2 — scheduled backups: the daily backup + monthly restore drill
actually fire (audit gap R1-13).

Failing-first: on the pristine base this module cannot even be collected
(``nomorals.backup.jobs`` and ``nomorals.tools.backup`` do not exist, and
``python -m nomorals.backup`` has no ``ship`` verb).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.backup.coordinator import (  # noqa: E402
    DAILY_CRON,
    MONTHLY_DRILL_CRON,
    register_jobs,
    schedule_specs,
)
from nomorals.backup.jobs import (  # noqa: E402
    BACKUP_DAILY_JOB,
    BACKUP_DRILL_JOB,
    ensure_backup_jobs,
)


class FakeScheduler:
    """Stand-in for the live agent Scheduler (add/list_jobs only)."""

    def __init__(self) -> None:
        self.jobs: list[dict] = []

    def list_jobs(self) -> list[dict]:
        return list(self.jobs)

    def add(self, name, spec, payload_kind, payload, **kwargs):
        job = {"id": f"job-{len(self.jobs) + 1}", "name": name,
               "spec": spec, "payload_kind": payload_kind,
               "payload": payload, "kwargs": kwargs}
        self.jobs.append(job)
        return job


class FakeFacade:
    """Stand-in for a SchedulerFacade / UnifiedScheduler (add_job only)."""

    def __init__(self) -> None:
        self.jobs: list = []

    def add_job(self, job):
        if any(j.job_id == job.job_id for j in self.jobs):
            raise ValueError(f"job_id {job.job_id!r} already exists")
        self.jobs.append(job)
        return job


class FakeRegistry:
    """Stand-in for the ToolRegistry (register-as-decorator only)."""

    def __init__(self, context=None) -> None:
        self.context = context
        self._tools: dict = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self._tools[name] = fn
            return fn
        return deco


class FakeCoordinator:
    """Stand-in for BackupCoordinator (run_backup / run_drill only)."""

    def __init__(self, result=None, drill_ok=True) -> None:
        self.result = result or {
            "backup_id": "backup-20261116-000000",
            "shipped": True, "queued": False,
            "total_encrypted_bytes": 1234,
        }
        self.drill_ok = drill_ok
        self.backup_calls: list[dict] = []
        self.drill_calls: list[dict] = []

    def run_backup(self, *, label="", prune=True):
        self.backup_calls.append({"label": label, "prune": prune})
        return dict(self.result)

    def run_drill(self, *, backup_id=""):
        from nomorals.backup.verify import DrillReport, DrillStep

        self.drill_calls.append({"backup_id": backup_id})
        return DrillReport(
            backup_id=self.result["backup_id"], ok=self.drill_ok,
            steps=[DrillStep("manifest_present", True, "ok")],
            started_at=0.0, finished_at=1.0,
        )


class RegisterJobsTests(unittest.TestCase):
    def test_register_jobs_installs_both_specs(self) -> None:
        facade = FakeFacade()
        out = register_jobs(facade)
        self.assertEqual(len(out), 2)
        self.assertEqual([j.job_id for j in facade.jobs],
                         ["github-backup-daily", "github-backup-drill-monthly"])
        by_id = {j.job_id: j for j in facade.jobs}
        self.assertEqual(by_id["github-backup-daily"].recurrence,
                         {"type": "cron", "spec": DAILY_CRON})
        self.assertEqual(by_id["github-backup-drill-monthly"].recurrence,
                         {"type": "cron", "spec": MONTHLY_DRILL_CRON})
        self.assertEqual(
            by_id["github-backup-daily"].payload,
            {"handler": "nomorals.backup.coordinator:run_scheduled_backup"})
        self.assertEqual(
            by_id["github-backup-drill-monthly"].payload,
            {"handler": "nomorals.backup.coordinator:run_scheduled_drill"})
        self.assertTrue(all(r["installed"] for r in out))

    def test_register_jobs_idempotent(self) -> None:
        facade = FakeFacade()
        register_jobs(facade)
        out = register_jobs(facade)
        self.assertEqual(len(facade.jobs), 2)
        self.assertTrue(all(not r["installed"] for r in out))

    def test_schedule_specs_are_the_single_source_of_truth(self) -> None:
        specs = schedule_specs()
        self.assertEqual([s["job_id"] for s in specs],
                         ["github-backup-daily", "github-backup-drill-monthly"])
        self.assertEqual(specs[0]["recurrence"]["spec"], "0 2 * * *")
        self.assertEqual(specs[1]["recurrence"]["spec"], "0 3 1 * *")


class EnsureBackupJobsTests(unittest.TestCase):
    def test_installs_two_tool_jobs_on_the_live_scheduler(self) -> None:
        sched = FakeScheduler()
        out = ensure_backup_jobs(sched)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r.get("scheduled") for r in out))
        by_name = {j["name"]: j for j in sched.jobs}
        self.assertEqual(set(by_name), {BACKUP_DAILY_JOB, BACKUP_DRILL_JOB})
        daily = by_name[BACKUP_DAILY_JOB]
        self.assertEqual(daily["spec"], "cron " + DAILY_CRON)
        self.assertEqual(daily["payload_kind"], "tool")
        self.assertEqual(daily["payload"],
                         {"tool": "__backup_tick__", "args": {}})
        drill = by_name[BACKUP_DRILL_JOB]
        self.assertEqual(drill["spec"], "cron " + MONTHLY_DRILL_CRON)
        self.assertEqual(drill["payload"],
                         {"tool": "__backup_drill__", "args": {}})
        # A backup generation is heavy and must never run twice at once.
        self.assertTrue(daily["kwargs"].get("heavy"))
        self.assertEqual(drill["kwargs"].get("overlap_policy"), "skip")

    def test_ensure_is_idempotent(self) -> None:
        sched = FakeScheduler()
        ensure_backup_jobs(sched)
        out = ensure_backup_jobs(sched)
        self.assertEqual(len(sched.jobs), 2)
        self.assertTrue(all(r.get("already_scheduled") for r in out))

    def test_job_names_match_schedule_specs(self) -> None:
        names = {s["name"] for s in schedule_specs()}
        self.assertEqual(names, {BACKUP_DAILY_JOB, BACKUP_DRILL_JOB})


class BackupToolTests(unittest.TestCase):
    def _context(self, tmp: Path):
        settings = SimpleNamespace(
            home_path=tmp, backup_dir=tmp / "backups", db_path=tmp / "db.db")
        return SimpleNamespace(settings=settings, db=None, extras={})

    def _registered(self, context):
        from nomorals.tools import backup as backup_tools

        reg = FakeRegistry(context)
        backup_tools.register(reg)
        return reg, backup_tools

    def test_both_tools_registered(self) -> None:
        reg, _ = self._registered(self._context(Path(tempfile.mkdtemp())))
        self.assertIn("__backup_tick__", reg._tools)
        self.assertIn("__backup_drill__", reg._tools)

    def test_tick_runs_the_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coord = FakeCoordinator()
            reg, tools = self._registered(self._context(Path(tmp)))
            with mock.patch.object(tools, "_build_coordinator",
                                   return_value=coord):
                result = reg._tools["__backup_tick__"]()
        self.assertTrue(result["ok"])
        self.assertEqual(result["backup_id"], "backup-20261116-000000")
        self.assertTrue(result["shipped"])
        self.assertEqual(len(coord.backup_calls), 1)
        self.assertEqual(coord.backup_calls[0]["label"], "scheduled")

    def test_drill_runs_the_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coord = FakeCoordinator()
            reg, tools = self._registered(self._context(Path(tmp)))
            with mock.patch.object(tools, "_build_coordinator",
                                   return_value=coord):
                result = reg._tools["__backup_drill__"]()
        self.assertTrue(result["ok"])
        self.assertIn("PASS", result["summary"])
        self.assertEqual(len(coord.drill_calls), 1)

    def test_tick_fails_closed_when_the_coordinator_cannot_start(self) -> None:
        from nomorals.backup.errors import BackupError

        with tempfile.TemporaryDirectory() as tmp:
            reg, tools = self._registered(self._context(Path(tmp)))
            recorded: list = []
            with mock.patch.object(
                    tools, "_build_coordinator",
                    side_effect=BackupError("no vault configured")), \
                    mock.patch.object(
                        tools, "record_failure",
                        side_effect=lambda site, exc: recorded.append(
                            (site, str(exc)))), \
                    mock.patch.object(tools, "_report_failure_best_effort",
                                      return_value=None):
                result = reg._tools["__backup_tick__"]()
        self.assertFalse(result["ok"])
        self.assertIn("no vault", result["error"])
        self.assertEqual([site for site, _ in recorded], ["backup.tick"])

    def test_tick_fails_closed_when_the_run_itself_fails(self) -> None:
        from nomorals.backup.errors import BackupError

        with tempfile.TemporaryDirectory() as tmp:
            coord = FakeCoordinator()
            coord.run_backup = mock.Mock(
                side_effect=BackupError("disk full"))
            reg, tools = self._registered(self._context(Path(tmp)))
            recorded: list = []
            with mock.patch.object(tools, "_build_coordinator",
                                   return_value=coord), \
                    mock.patch.object(
                        tools, "record_failure",
                        side_effect=lambda site, exc: recorded.append(
                            (site, str(exc)))):
                result = reg._tools["__backup_tick__"]()
        self.assertFalse(result["ok"])
        self.assertIn("disk full", result["error"])
        self.assertEqual([site for site, _ in recorded], ["backup.tick"])


class BackupCliShipTests(unittest.TestCase):
    def test_ship_runs_one_generation(self) -> None:
        from nomorals.backup import __main__ as backup_cli

        with tempfile.TemporaryDirectory() as tmp:
            coord = FakeCoordinator()
            with mock.patch.object(
                    backup_cli, "_build_cli_coordinator",
                    return_value=coord):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = backup_cli.main(
                        ["ship", "--work-dir", tmp, "--label", "manual"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(coord.backup_calls), 1)
            self.assertEqual(coord.backup_calls[0]["label"], "manual")
            printed = json.loads(buf.getvalue())
            self.assertEqual(printed["backup_id"], "backup-20261116-000000")
            self.assertTrue(printed["shipped"])

    def test_ship_fails_closed_when_no_coordinator(self) -> None:
        from nomorals.backup import __main__ as backup_cli
        from nomorals.backup.errors import BackupError

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                    backup_cli, "_build_cli_coordinator",
                    side_effect=BackupError("no Devon database here")):
                err = io.StringIO()
                with redirect_stderr(err):
                    rc = backup_cli.main(["ship", "--work-dir", tmp])
            self.assertEqual(rc, 1)
            self.assertIn("no Devon database", err.getvalue())


class BackupStatusHandlerTests(unittest.TestCase):
    def test_backup_status_still_works(self) -> None:
        from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp) / "backups"
            (backup_dir / "backup-20261116-000000").mkdir(parents=True)
            settings = SimpleNamespace(
                backup_dir=backup_dir, db_path=Path(tmp) / "missing.db")
            context = SimpleNamespace(settings=settings)
            mixin = object.__new__(RuntimeReachMixin)
            mixin.context = context
            text = mixin._control_backup("status")
        self.assertIn("backups in", text)
        self.assertIn("1 generation(s)", text)
        self.assertIn("backup-20261116-000000", text)


if __name__ == "__main__":
    unittest.main()
