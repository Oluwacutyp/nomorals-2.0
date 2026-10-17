"""Scheduled jobs must reference tools that exist in the tool registry.

Regression test for the ``presence-heartbeat`` job failing every 30 min
with ``unknown tool '__presence_heartbeat__'``: the job was stored with
``payload_kind="tool"`` but the tool was never registered in the tool
registry (``ensure_heartbeat_job`` only called ``register_action``,
which targets a different scheduler class). A renamed/deleted tool must
never silently break a job again — this test exercises the real
job-registration functions and asserts every payload tool resolves.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


def _registry_with_context(workspace_dir: str):
    """Full tool registry with a context pointing at ``workspace_dir``."""
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=workspace_dir),
        db=None,
    )
    reg.register_builtins()
    return reg


def _tmp_ctx(test):
    """Scheduler context backed by a throwaway DB (mirrors test_briefing)."""
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="sched-job-tools-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp, pulse=None)
    return SimpleNamespace(db=db, settings=settings)


class FakeScheduler:
    """Captures ``scheduler.add`` calls for ``ensure_heartbeat_job``."""

    def __init__(self):
        self.added = []
        self.actions = {}

    def register_action(self, name, handler):
        self.actions[name] = handler

    def list_jobs(self):
        return []

    def add(self, *args, **kwargs):
        # ensure_heartbeat_job calls add(name=..., spec=...,
        # payload_kind=..., payload=...).
        payload = kwargs.get("payload")
        if payload is None and len(args) >= 4:
            payload = args[3]
        self.added.append(payload)
        return {"id": "fake-1"}


class HeartbeatToolTests(unittest.TestCase):
    def test_heartbeat_tool_registered_and_callable(self):
        tmp = tempfile.mkdtemp(prefix="heartbeat-tool-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        reg = _registry_with_context(tmp)
        self.assertIn("__presence_heartbeat__", reg._tools)
        out = reg.call("__presence_heartbeat__")
        self.assertTrue(out.ok, f"heartbeat tool call failed: {out.error}")
        value = out.value
        self.assertTrue(value.get("ok"))
        hb = value.get("heartbeat") or {}
        # heartbeat() returns what it did: noticed / prepared / surfaced.
        self.assertIn("noticed", hb)
        self.assertIn("prepared", hb)
        self.assertIn("surfaced", hb)


class ScheduledJobToolsExistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="sched-job-tools-reg-")
        cls.registry = _registry_with_context(cls._tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _assert_tool_exists(self, tool_name):
        # assertTrue (not assertIn) so a failure doesn't dump the whole
        # registry dict into the output.
        self.assertTrue(
            tool_name in self.registry._tools,
            f"scheduled job references tool {tool_name!r} which is NOT "
            "in the tool registry — the job will fail with 'unknown tool' "
            "on every run",
        )

    def test_heartbeat_job_tool_exists(self):
        from nomorals.autonomy.presence import ensure_heartbeat_job

        fake = FakeScheduler()
        ensure_heartbeat_job(fake)
        self.assertEqual(len(fake.added), 1)
        self.assertEqual(fake.added[0].get("tool"), "__presence_heartbeat__")
        self._assert_tool_exists(fake.added[0]["tool"])

    def _job_tool_from_real_scheduler(self, ensure_fn):
        """Run an ensure_*_job fn against a real Scheduler, return its tool.

        ``list_jobs()`` omits the payload column, so the payload is
        captured at ``Scheduler.add`` time via a spy wrapper that
        delegates to the real method.
        """
        from unittest.mock import patch

        from nomorals.agents.scheduler import Scheduler

        ctx = _tmp_ctx(self)
        captured = []
        orig_add = Scheduler.add

        def spy_add(scheduler_self, *args, **kwargs):
            captured.append((args, kwargs))
            return orig_add(scheduler_self, *args, **kwargs)

        with patch.object(Scheduler, "add", spy_add):
            ensure_fn(ctx)
        self.assertTrue(
            captured, f"{ensure_fn.__name__} called Scheduler.add never")
        args, kwargs = captured[0]
        payload = kwargs.get("payload")
        if payload is None and len(args) >= 4:
            payload = args[3]
        self.assertIsInstance(payload, dict)
        return payload.get("tool")

    def test_pulse_job_tool_exists(self):
        from nomorals.agents.morning_pulse import ensure_pulse_job

        self._assert_tool_exists(
            self._job_tool_from_real_scheduler(ensure_pulse_job))

    def test_briefing_job_tool_exists(self):
        from nomorals.agents.morning_briefing import ensure_briefing_job

        self._assert_tool_exists(
            self._job_tool_from_real_scheduler(ensure_briefing_job))

    def test_afternoon_pulse_job_tool_exists(self):
        from nomorals.agents.afternoon_pulse import (
            ensure_afternoon_pulse_job,
        )

        self._assert_tool_exists(
            self._job_tool_from_real_scheduler(ensure_afternoon_pulse_job))

    def test_evening_pulse_job_tool_exists(self):
        from nomorals.agents.evening_pulse import ensure_evening_pulse_job

        self._assert_tool_exists(
            self._job_tool_from_real_scheduler(ensure_evening_pulse_job))

    def test_mocked_scheduler_job_tool_exists(self):
        """The MagicMock-Scheduler pattern (test_morning_pulse) still holds."""
        from unittest.mock import patch

        from nomorals.agents.morning_pulse import ensure_pulse_job

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.workspace_dir = "/tmp"
        ctx.settings.pulse = None
        sched = MagicMock()
        sched.list_jobs.return_value = []
        sched.add.return_value = {"id": "job-1"}
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            ensure_pulse_job(ctx)
        args, _kwargs = sched.add.call_args
        self._assert_tool_exists(args[3]["tool"])


if __name__ == "__main__":
    unittest.main()
