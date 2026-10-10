"""Morning pulse deep upgrades: stage retries, ledger, bus events,
ensure idempotency, scheduler policies."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import morning_pulse as pulse_mod
from nomorals.agents.morning_pulse import (
    PULSE_JOB_NAME,
    PULSE_POLICIES,
    _pulse_spec_matches,
    _run_stage,
    ensure_pulse_job,
    run_pulse,
)
from nomorals.core.events import EventBus
from nomorals.storage.db import Database


def _ctx(**kw):
    db = Database(":memory:")
    db.migrate()
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw2: SimpleNamespace(
            ok=True, value="ok", error=None))
    ctx = SimpleNamespace(db=db, tools=tools, memory=None,
                          settings=SimpleNamespace(pulse=None),
                          extras={})
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


class StageRunnerTests(unittest.TestCase):
    def test_success_first_try(self):
        ok, value, secs, attempts = _run_stage("s", lambda: 42)
        self.assertTrue(ok)
        self.assertEqual(value, 42)
        self.assertEqual(attempts, 1)
        self.assertGreaterEqual(secs, 0.0)

    def test_retry_then_success(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("transient")
            return "recovered"

        ok, value, secs, attempts = _run_stage(
            "s", flaky, attempts=3, base_delay_s=0.01)
        self.assertTrue(ok)
        self.assertEqual(value, "recovered")
        self.assertEqual(attempts, 2)

    def test_exhaustion_never_raises(self):
        def always_fails():
            raise RuntimeError("permanent")

        ok, value, secs, attempts = _run_stage(
            "s", always_fails, attempts=2, base_delay_s=0.01)
        self.assertFalse(ok)
        self.assertIsNone(value)
        self.assertEqual(attempts, 2)


class SpecMatchTests(unittest.TestCase):
    def test_bare_spec_matches(self):
        self.assertTrue(_pulse_spec_matches("23:00", "23:00"))

    def test_timezone_suffixed_spec_matches(self):
        # the historical bug: "23:00 America/Denver" vs "23:00" caused
        # delete/recreate on every boot
        self.assertTrue(_pulse_spec_matches("23:00 America/Denver", "23:00"))

    def test_different_time_does_not_match(self):
        self.assertFalse(_pulse_spec_matches("07:00 America/Denver", "23:00"))

    def test_empty_spec(self):
        self.assertFalse(_pulse_spec_matches("", "23:00"))


class PulsePoliciesTests(unittest.TestCase):
    def test_policies_shape(self):
        self.assertEqual(PULSE_POLICIES["missed_fire_policy"], "fire_now")
        self.assertEqual(PULSE_POLICIES["overlap_policy"], "skip")
        self.assertEqual(PULSE_POLICIES["run_timeout_s"], 1800.0)

    def test_ensure_passes_policies_to_scheduler(self):
        from unittest.mock import MagicMock

        sched = MagicMock()
        sched.list_jobs.return_value = []
        sched.add.return_value = {"id": "job-1"}
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            ensure_pulse_job(_ctx())
        _, kwargs = sched.add.call_args
        self.assertEqual(kwargs.get("missed_fire_policy"), "fire_now")
        self.assertEqual(kwargs.get("overlap_policy"), "skip")
        self.assertEqual(kwargs.get("run_timeout_s"), 1800.0)

    def test_ensure_idempotent_when_policies_match(self):
        from unittest.mock import MagicMock

        sched = MagicMock()
        sched.list_jobs.return_value = [{
            "id": "job-1", "name": PULSE_JOB_NAME,
            "spec": "23:00 America/Denver", **PULSE_POLICIES}]
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            res = ensure_pulse_job(_ctx())
        self.assertTrue(res.get("already_scheduled"))
        sched.add.assert_not_called()

    def test_ensure_replaces_on_policy_drift(self):
        from unittest.mock import MagicMock

        sched = MagicMock()
        sched.list_jobs.return_value = [{
            "id": "job-1", "name": PULSE_JOB_NAME, "spec": "23:00",
            "missed_fire_policy": "skip", "overlap_policy": "concurrent",
            "run_timeout_s": 0.0}]
        sched.add.return_value = {"id": "job-2"}
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            res = ensure_pulse_job(_ctx())
        self.assertTrue(res.get("scheduled"))
        sched.remove.assert_called_once_with("job-1")
        sched.add.assert_called_once()


class PipelineTests(unittest.TestCase):
    def _quiet_context(self):
        ctx = _ctx()
        return ctx

    def test_total_failure_still_returns_ok_with_fallback(self):
        """All stages failing → fallback text, no raise; ok keeps its
        historical meaning (pipeline ran to completion)."""
        ctx = self._quiet_context()
        with patch.object(pulse_mod, "_fetch_news",
                          side_effect=RuntimeError("news down")), \
             patch.object(pulse_mod, "_compose_briefing",
                          side_effect=RuntimeError("brain down")), \
             patch.object(pulse_mod, "_synthesize",
                          side_effect=RuntimeError("tts down")), \
             patch.object(pulse_mod, "_deliver_text", return_value=False), \
             patch.object(pulse_mod, "_deliver_audio", return_value=False):
            res = run_pulse(ctx)
        self.assertTrue(res["ok"])
        self.assertFalse(res["delivered_text"])
        self.assertFalse(res["delivered_audio"])
        self.assertIn("deliver", res["stages"])

    def test_voice_failure_degrades_to_text(self):
        ctx = self._quiet_context()
        with patch.object(pulse_mod, "_fetch_news", return_value=[]), \
             patch.object(pulse_mod, "_compose_briefing",
                          return_value="tonight's brief"), \
             patch.object(pulse_mod, "_synthesize",
                          side_effect=RuntimeError("tts down")), \
             patch.object(pulse_mod, "_deliver_text", return_value=True), \
             patch.object(pulse_mod, "_deliver_audio", return_value=False):
            res = run_pulse(ctx)
        self.assertTrue(res["delivered_text"])
        self.assertNotIn("voice", res["stages"])
        self.assertEqual(res["audio_path"], "")

    def test_stages_land_in_ledger(self):
        from nomorals.agents.autonomy_ledger import AutonomyLedger

        ctx = self._quiet_context()
        with patch.object(pulse_mod, "_fetch_news", return_value=[]), \
             patch.object(pulse_mod, "_compose_briefing",
                          return_value="brief"), \
             patch.object(pulse_mod, "_synthesize", return_value=""), \
             patch.object(pulse_mod, "_deliver_text", return_value=True), \
             patch.object(pulse_mod, "_deliver_audio", return_value=False):
            run_pulse(ctx)
        rows = AutonomyLedger(ctx.db).recent(system="pulse")
        kinds = [r["kind"] for r in rows]
        self.assertIn("stage", kinds)
        self.assertIn("run", kinds)
        run_row = next(r for r in rows if r["kind"] == "run")
        self.assertTrue(run_row["ok"])
        self.assertGreaterEqual(run_row["cost_seconds"], 0)

    def test_pulse_finished_bus_event(self):
        import nomorals.core.events as events_mod

        bus = EventBus()
        seen = []
        bus.subscribe("pulse.finished", lambda e: seen.append(e.data),
                      sync=True)
        orig = events_mod.global_bus
        events_mod.global_bus = bus
        try:
            ctx = self._quiet_context()
            with patch.object(pulse_mod, "_fetch_news", return_value=[]), \
                 patch.object(pulse_mod, "_compose_briefing",
                              return_value="brief"), \
                 patch.object(pulse_mod, "_synthesize", return_value=""), \
                 patch.object(pulse_mod, "_deliver_text",
                              return_value=True), \
                 patch.object(pulse_mod, "_deliver_audio",
                              return_value=False):
                run_pulse(ctx)
        finally:
            events_mod.global_bus = orig
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0]["delivered_text"])
        self.assertIn("deliver", seen[0]["stages"])


if __name__ == "__main__":
    unittest.main()
