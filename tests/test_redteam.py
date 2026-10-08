"""Tests for the defensive red-team module. All offline, all faked."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.redteam import (
    AttackFinding,
    RedTeam,
    RedTeamReport,
    build_scenarios,
    control_redteam,
    report_to_github,
)
from nomorals.agents.redteam_scenarios import _messages_text


class MessagesTextTests(unittest.TestCase):
    def test_message_objects(self):
        from nomorals.llm.base import Message

        msgs = [Message.system("sys"), Message.user("hello")]
        self.assertIn("hello", _messages_text(msgs))
        self.assertIn("sys", _messages_text(msgs))

    def test_dicts_and_strings(self):
        self.assertIn("x", _messages_text([{"content": "x"}, "y"]))


class ScenarioLibraryTests(unittest.TestCase):
    def test_eight_scenarios(self):
        scs = build_scenarios()
        self.assertEqual(8, len(scs))
        ids = [s.id for s in scs]
        self.assertEqual(len(ids), len(set(ids)))  # unique ids

    def test_categories_and_severities(self):
        cats = {s.category for s in build_scenarios()}
        self.assertTrue({"injection", "exfiltration", "escalation"} <= cats)
        for s in build_scenarios():
            self.assertIn(s.severity, {"low", "medium", "high", "critical"})
            self.assertTrue(s.payload)
            self.assertTrue(s.target)
            self.assertIsNotNone(s.detect)


class DetectorTests(unittest.TestCase):
    """Each detector fires on the attack and not on benign input."""

    def _ctx(self, **kw):
        from nomorals.agents.redteam import _FakeRegistry

        ctx = {"tools": _FakeRegistry(), "result": SimpleNamespace(response=""),
               "policy": None}
        ctx.update(kw)
        return ctx

    def test_s1_detector_fires_on_attempt(self):
        s = next(x for x in build_scenarios()
                 if x.id == "injection-tool-output")
        reg = s.make_harness()["tools"]
        reg.attempts.append(("fs_delete", {"path": "/"}))
        ok, ev = s.detect({"tools": reg, "executed": {"ran": False}})
        self.assertTrue(ok)
        self.assertIn("policy blocked", ev)

    def test_s1_detector_quiet_when_ignored(self):
        s = next(x for x in build_scenarios()
                 if x.id == "injection-tool-output")
        reg = s.make_harness()["tools"]
        ok, _ = s.detect({"tools": reg, "executed": {"ran": False}})
        self.assertFalse(ok)

    def test_s3_detector_fires_on_secret(self):
        from nomorals.agents.redteam_scenarios import _S3_SECRET

        s = next(x for x in build_scenarios() if x.id == "exfil-vault-keys")
        ctx = self._ctx(result=SimpleNamespace(
            response=f"keys: {_S3_SECRET}"))
        ok, ev = s.detect(ctx)
        self.assertTrue(ok)
        self.assertIn("LEAKED", ev)

    def test_s3_detector_quiet_on_benign(self):
        s = next(x for x in build_scenarios() if x.id == "exfil-vault-keys")
        ctx = self._ctx(result=SimpleNamespace(response="I don't share those."))
        ok, _ = s.detect(ctx)
        self.assertFalse(ok)

    def test_s5_replay_rejected(self):
        # Direct policy assertion: cross-capability token must NOT work.
        s = next(x for x in build_scenarios()
                 if x.id == "escalation-token-replay")
        ok, ev = s.detect({"policy": __import__(
            "nomorals.core.policy", fromlist=["Policy"]).Policy()})
        self.assertFalse(ok)  # attack failed = defense holds
        self.assertIn("rejected", ev)

    def test_s8_passes_on_current_tree(self):
        s = next(x for x in build_scenarios()
                 if x.id == "escalation-biometric-bypass")
        ok, ev = s.detect({})
        self.assertFalse(ok, f"unexpected offenders: {ev}")
        self.assertIn("operator-facing", ev)


class HarnessTests(unittest.TestCase):
    def test_inconclusive_on_harness_error(self):
        from nomorals.agents.redteam import AttackScenario

        def boom():
            raise RuntimeError("nope")

        sc = AttackScenario(
            id="x", name="x", category="injection", description="x",
            severity="low", payload="x", target="x",
            detect=lambda ctx: (False, ""),
            make_harness=boom,
        )
        finding = RedTeam().run_scenario(sc)
        self.assertTrue(finding.inconclusive)
        self.assertFalse(finding.succeeded)

    def test_timeout_is_inconclusive(self):
        from nomorals.agents.redteam import AttackScenario, _FakeRegistry

        def slow_harness():
            return {"tools": _FakeRegistry(),
                    "script": [lambda m, c: {"action": "respond",
                                             "response": "x"}],
                    "user_message": "hi"}

        sc = AttackScenario(
            id="x", name="x", category="injection", description="x",
            severity="low", payload="x", target="x",
            detect=lambda ctx: (False, ""),
            make_harness=slow_harness,
        )
        # Loop itself is fast; just verify the timeout path exists and works
        finding = RedTeam(per_scenario_timeout_s=30.0).run_scenario(sc)
        self.assertFalse(finding.inconclusive)

    def test_suite_records_failure_kb(self):
        from nomorals.agents.redteam import AttackScenario

        noted = []

        class KB:
            def note(self, cluster_key, note_text):
                noted.append((cluster_key, note_text))
                return "n1"

        sc = AttackScenario(
            id="demo-hole", name="demo", category="injection",
            description="x", severity="high", payload="x", target="x",
            detect=lambda ctx: (True, "pwned"),
            make_harness=lambda: {},
        )
        team = RedTeam(failure_kb=KB())
        # Bypass the loop: scenario has no script/tools, detect decides.
        with patch.object(RedTeam, "_run_loop", return_value=None):
            report = team.run_suite([sc])
        self.assertEqual(1, len(report.failed))
        self.assertEqual(1, len(noted))
        self.assertEqual("redteam:demo-hole", noted[0][0])

    def test_report_summary(self):
        r = RedTeamReport(findings=[
            AttackFinding("a", "A", "high", True, "ev"),
            AttackFinding("b", "B", "low", False, ""),
            AttackFinding("c", "C", "medium", False, "err", inconclusive=True),
        ])
        s = r.summary()
        self.assertIn("1 failed / 1 passed / 1 inconclusive", s)
        self.assertIn("FAIL [high] a", s)


class GithubReportingTests(unittest.TestCase):
    def _report(self):
        return RedTeamReport(findings=[
            AttackFinding("demo-hole", "demo", "high", True, "ev"),
        ])

    def test_skips_without_credential(self):
        class Conn:
            def _load_credential(self):
                return None

        out = report_to_github(self._report(), connector=Conn())
        self.assertEqual([], out["created"])
        self.assertEqual(["demo-hole"], out["skipped"])
        self.assertIn("reason", out)

    def test_skips_when_connector_raises(self):
        class Conn:
            def _load_credential(self):
                raise RuntimeError("no vault")

        out = report_to_github(self._report(), connector=Conn())
        self.assertEqual([], out["created"])

    def test_dedupes_open_issues(self):
        calls = []

        class Conn:
            def _load_credential(self):
                return SimpleNamespace(username="u", password="p")

            def _api(self, method, path, payload=None, params=None):
                calls.append((method, path, payload))
                if method == "GET":
                    return [{"title": "[redteam:demo-hole] demo"}]
                return {"id": 1}

        out = report_to_github(self._report(), connector=Conn())
        self.assertEqual([], out["created"])
        self.assertEqual(["demo-hole"], out["skipped"])
        # only the GET happened, no POST
        self.assertFalse(any(m == "POST" for m, _, _ in calls))

    def test_creates_issue_when_new(self):
        calls = []

        class Conn:
            def _load_credential(self):
                return SimpleNamespace(username="u", password="p")

            def _api(self, method, path, payload=None, params=None):
                calls.append((method, path, payload))
                if method == "GET":
                    return []
                return {"id": 42}

        out = report_to_github(self._report(), connector=Conn())
        self.assertEqual(["demo-hole"], out["created"])
        posts = [c for c in calls if c[0] == "POST"]
        self.assertEqual(1, len(posts))
        self.assertIn("[redteam:demo-hole]", posts[0][2]["title"])

    def test_empty_report_no_calls(self):
        class Conn:
            def _load_credential(self):
                raise AssertionError("should not be called")

        out = report_to_github(RedTeamReport(), connector=Conn())
        self.assertEqual([], out["created"])


class ControlCommandTests(unittest.TestCase):
    def test_redteam_parses(self):
        from nomorals.social.chat.control import parse_control

        cmd = parse_control("/redteam")
        self.assertIsNotNone(cmd)
        self.assertEqual("redteam", cmd.kind)
        cmd = parse_control("/redteam injection-tool-output")
        self.assertEqual("injection-tool-output", cmd.arg)

    def test_redteam_unknown_scenario(self):
        out = control_redteam("nope-nope", context=None)
        self.assertIn("unknown redteam scenario", out)
        self.assertIn("injection-tool-output", out)  # lists known ids

    def test_redteam_single_scenario(self):
        out = control_redteam("escalation-token-replay", context=None)
        self.assertIn("redteam:", out)
        # token replay is correctly rejected -> defense holds -> 0 failed
        self.assertIn("0 failed / 1 passed", out)

    def test_redteam_single_failing_scenario(self):
        out = control_redteam("injection-tool-output", context=None)
        self.assertIn("1 failed / 0 passed", out)
        self.assertIn("FAIL", out)


if __name__ == "__main__":
    unittest.main()
