"""Tests for the autonomous morning pulse (news → briefing → voice → delivery)."""

import unittest
from unittest.mock import MagicMock, patch


def _ctx(**kw):
    ctx = MagicMock()
    ctx.settings = MagicMock()
    ctx.settings.workspace_dir = "/tmp"
    # no settings.pulse namespace → sidecar/defaults path
    ctx.settings.pulse = None
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


class PulsePrefsTests(unittest.TestCase):
    def test_default_time_is_2300(self):
        from nomorals.agents.morning_pulse import pulse_time
        self.assertEqual(pulse_time(_ctx()), "23:00")

    def test_default_timezone_is_denver(self):
        from nomorals.agents.morning_pulse import pulse_timezone
        self.assertEqual(pulse_timezone(_ctx()), "America/Denver")

    def test_job_name(self):
        from nomorals.agents.morning_pulse import PULSE_JOB_NAME
        self.assertEqual(PULSE_JOB_NAME, "morning-pulse")


class EnsurePulseJobTests(unittest.TestCase):
    def test_registers_daily_2300_denver(self):
        from nomorals.agents.morning_pulse import ensure_pulse_job
        sched = MagicMock()
        sched.list_jobs.return_value = []
        sched.add.return_value = {"id": "job-1"}
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            res = ensure_pulse_job(_ctx())
        self.assertTrue(res.get("scheduled"))
        # check the add call: name, spec, kind, payload, timezone kwarg
        args, kwargs = sched.add.call_args
        self.assertEqual(args[0], "morning-pulse")
        self.assertEqual(args[1], "daily 23:00")
        self.assertEqual(kwargs.get("timezone"), "America/Denver")
        payload = args[3]
        self.assertEqual(payload["tool"], "pulse")
        self.assertEqual(payload["args"], {"action": "run"})

    def test_idempotent_when_already_scheduled(self):
        from nomorals.agents.morning_pulse import ensure_pulse_job
        sched = MagicMock()
        sched.list_jobs.return_value = [
            {"id": "job-1", "name": "morning-pulse", "spec": "23:00"}]
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            res = ensure_pulse_job(_ctx())
        self.assertTrue(res.get("already_scheduled"))
        sched.add.assert_not_called()

    def test_replaces_stale_time(self):
        from nomorals.agents.morning_pulse import ensure_pulse_job
        sched = MagicMock()
        sched.list_jobs.return_value = [
            {"id": "old-1", "name": "morning-pulse", "spec": "07:00"}]
        sched.add.return_value = {"id": "new-1"}
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            ensure_pulse_job(_ctx())
        sched.remove.assert_called_once_with("old-1")
        sched.add.assert_called_once()


class ScriptTests(unittest.TestCase):
    def test_fallback_script_when_no_router(self):
        from nomorals.agents.morning_pulse import _write_script, HOST_1_NAME
        ctx = _ctx()
        ctx.router = None
        script = _write_script(ctx, "Test briefing content here.", [])
        self.assertIn(HOST_1_NAME + ":", script)
        self.assertIn("Test briefing content", script)

    def test_script_never_empty(self):
        from nomorals.agents.morning_pulse import _write_script
        ctx = _ctx()
        ctx.router = None
        script = _write_script(ctx, "", [])
        self.assertTrue(script.strip())


class RunPulseTests(unittest.TestCase):
    def _failing_ctx(self):
        """Context where every provider raises — pulse must not raise."""
        ctx = _ctx()
        ctx.router = None
        return ctx

    def test_never_raises_on_total_failure(self):
        from nomorals.agents.morning_pulse import run_pulse
        # NewsAgent, BriefingComposer, TTS, Notifier all unavailable
        with patch.dict("sys.modules", {}):
            try:
                res = run_pulse(self._failing_ctx())
            except Exception as exc:  # noqa: BLE001
                self.fail(f"run_pulse raised: {exc}")
        self.assertTrue(res.get("ok"))
        self.assertIn("deliver", res.get("stages", []))

    def test_stages_logged(self):
        from nomorals.agents import morning_pulse
        ctx = self._failing_ctx()
        # Make news + compose succeed minimally, voice fail, deliver fail
        fake_agent = MagicMock()
        fake_agent.run.return_value = {"ok": True}
        fake_agent.recent.return_value = [{"title": "t"}]
        fake_composer = MagicMock()
        fake_briefing = MagicMock()
        fake_briefing.render_text.return_value = "briefing text"
        fake_composer.compose.return_value = fake_briefing
        with patch("nomorals.agents.news.NewsAgent",
                   return_value=fake_agent), \
             patch("nomorals.agents.morning_briefing.BriefingComposer",
                   return_value=fake_composer), \
             patch.object(morning_pulse, "_synthesize",
                          return_value=""), \
             patch.object(morning_pulse, "_deliver_text",
                          return_value=False):
            res = morning_pulse.run_pulse(ctx)
        stages = res.get("stages", [])
        for expected in ("news", "compose", "script", "deliver"):
            self.assertIn(expected, stages, f"missing stage {expected}")
        # voice failed → not in stages, but pulse still ok
        self.assertNotIn("voice", stages)
        self.assertTrue(res["ok"])


class DeliveryDiagnosisTests(unittest.TestCase):
    def test_diagnosis_no_gateway(self):
        from nomorals.agents.morning_pulse import _delivery_diagnosis
        ctx = _ctx()
        with patch("nomorals.agents.notifier.resolve_gateway",
                   return_value=None):
            diag = _delivery_diagnosis(ctx)
        self.assertIn("no live gateway", diag)

    def test_diagnosis_empty_owner_chats(self):
        from nomorals.agents.morning_pulse import _delivery_diagnosis
        ctx = _ctx()
        ctx.settings.partner = MagicMock()
        ctx.settings.partner.owner_chats = ""
        with patch("nomorals.agents.notifier.resolve_gateway",
                   return_value=MagicMock()):
            diag = _delivery_diagnosis(ctx)
        self.assertIn("owner_chats is empty", diag)


class PulseToolTests(unittest.TestCase):
    def test_register_hook_exists(self):
        from nomorals.agents import morning_pulse
        self.assertTrue(callable(morning_pulse.register))

    def test_config_action(self):
        from nomorals.agents import morning_pulse
        registry = MagicMock()
        registry.context = _ctx()
        captured = {}

        def fake_register(name, **kw):
            def deco(fn):
                captured[name] = fn
                return fn
            return deco
        registry.register = fake_register
        morning_pulse.register(registry)
        self.assertIn("pulse", captured)
        res = captured["pulse"](action="config")
        self.assertEqual(res["time"], "23:00")
        self.assertEqual(res["timezone"], "America/Denver")
        self.assertIn("hosts", res)


if __name__ == "__main__":
    unittest.main()
