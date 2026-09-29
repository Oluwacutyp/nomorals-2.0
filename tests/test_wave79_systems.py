"""Wave 79 systems: photo→recovery bridge, orchestrator self-tuning,
mission pivots that survive resume.  All hermetic — no model, no
network (vision is stubbed with a fake router)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from unittest import mock

from nomorals.agents.base import AgentResult
from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.skills import SkillLibrary
from nomorals.core import barcode as bc
from nomorals.core.config import Settings
from nomorals.llm.base import LLMResponse
from nomorals.missions import MissionRunner, MissionStatus, MissionStore
from nomorals.tools import giftcard as gc


# ── recover_luhn ───────────────────────────────────────────────────────────

class RecoverLuhnTests(unittest.TestCase):
    def test_complete_valid_pan(self):
        rep = bc.recover_luhn("4111111111111111")
        self.assertEqual(rep["recovered"], "4111111111111111")
        self.assertEqual(rep["holes"], 0)

    def test_complete_invalid_pan(self):
        rep = bc.recover_luhn("4111111111111112")
        self.assertEqual(rep["recovered"], "")
        self.assertFalse(rep["candidates"])
        self.assertIn("fails Luhn", rep["note"])

    def test_one_hole_is_unique(self):
        # the Luhn check digit fixes a single missing digit exactly
        rep = bc.recover_luhn("411111111111111?")
        self.assertEqual(len(rep["candidates"]), 1)
        self.assertEqual(rep["recovered"], "4111111111111111")

    def test_two_holes_true_value_present(self):
        true_value = "5500005555555559"
        template = true_value[:10] + "??" + true_value[12:]
        rep = bc.recover_luhn(template)
        self.assertIn(true_value, [c["value"] for c in rep["candidates"]])
        self.assertLessEqual(len(rep["candidates"]), 10)
        for c in rep["candidates"]:
            self.assertTrue(bc.luhn_valid(c["value"]))

    def test_too_many_holes_rejected(self):
        with self.assertRaises(ValueError):
            bc.recover_luhn("??????????????")


# ── photo_recover (the deterministic half of the bridge) ───────────────────

class PhotoRecoverTests(unittest.TestCase):
    def test_structured_ean_with_hole(self):
        rep = gc.photo_recover("NUMBER: 5193?45678901")
        self.assertEqual(len(rep["candidates"]), 1)
        self.assertTrue(rep["recovered"].startswith("5193"))
        self.assertTrue(rep["recovered"].endswith("45678901"))

    def test_structured_complete_ean(self):
        full = "519345678901" + str(bc.ean_check_digit("519345678901"))
        rep = gc.photo_recover(f"NUMBER: {full}")
        self.assertEqual(rep["recovered"], full)

    def test_luhn_pan_with_holes(self):
        rep = gc.photo_recover("NUMBER: 4111 1111 1111 ?111")
        self.assertTrue(rep["candidates"])
        self.assertTrue(all(c.get("luhn_ok") for c in rep["candidates"]))

    def test_scanner_runs_line(self):
        bits = bc.encode_ean13("519345678901")["bits"]
        runs = []
        cur, count = bits[0], 1
        for ch in bits[1:]:
            if ch == cur:
                count += 1
            else:
                runs.append(count)
                cur, count = ch, 1
        runs.append(count)
        rep = gc.photo_recover("SCANNER: " + ",".join(map(str, runs)))
        top = rep["candidates"][0]
        self.assertEqual(top["symbology"], "ean13")
        self.assertEqual(rep["recovered"], top["value"])

    def test_payload_with_holes(self):
        rep = gc.photo_recover("PAYLOAD: GIFT?519")
        self.assertTrue(rep["candidates"])
        for c in rep["candidates"]:
            self.assertTrue(c["value"].startswith("GIFT"))

    def test_free_text_number(self):
        full = "519345678901" + str(bc.ean_check_digit("519345678901"))
        rep = gc.photo_recover(f"the card reads {full} at the bottom")
        self.assertEqual(rep["recovered"], full)

    def test_known_overrides_vision(self):
        rep = gc.photo_recover("NUMBER: 1234 5678 9012",
                               known="5193?45678901")
        self.assertEqual(rep["templates"]["number"], "5193?45678901")
        self.assertTrue(rep["candidates"])

    def test_garbage_reading(self):
        rep = gc.photo_recover("a blurry photo of a table")
        self.assertEqual(rep["candidates"], [])
        self.assertIn("no card number", rep["note"])

    def test_misread_full_number_is_flagged(self):
        # all digits, but the check digit is wrong
        rep = gc.photo_recover("NUMBER: 5193456789012")
        self.assertNotEqual(rep["recovered"], "5193456789012")
        self.assertIn("fails every checksum", rep["note"])


# ── photo tool through a stubbed vision router ─────────────────────────────

class _FakeVisionRouter:
    def __init__(self, text: str):
        self.text = text

    def describe_image(self, image, prompt="", params=None, **kw):
        return LLMResponse(text=self.text, model="fake-vision",
                           provider="fake-vision")


def _png_bytes() -> bytes:
    # minimal valid PNG header (1x1) — image_metadata only reads headers
    return (b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
            + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00")


class PhotoToolTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w79-photo-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()
        self.img = f"{self.home}/workspace/card.png"
        import os
        os.makedirs(os.path.dirname(self.img), exist_ok=True)
        with open(self.img, "wb") as fh:
            fh.write(_png_bytes())

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_photo_recovers_scratched_number(self):
        self.context.router = _FakeVisionRouter("NUMBER: 5193?45678901")
        rep = gc.photo(self.context, self.img)
        self.assertEqual(rep["image"]["format"], "png")
        self.assertEqual(rep["provider"], "fake-vision")
        recovery = rep["recovery"]
        self.assertTrue(recovery["candidates"])
        self.assertTrue(recovery["recovered"].startswith("5193"))

    def test_photo_mock_vision_reports_unavailable(self):
        # the mock provider's placeholder must NOT count as having seen
        # the image — the report must say so explicitly
        self.context.router = _FakeVisionRouter("[mock vision] no real model")
        rep = gc.photo(self.context, self.img)
        self.assertIn("vision unavailable", rep["recovery"]["note"])

    def test_photo_rejects_non_image(self):
        path = f"{self.home}/workspace/notes.txt"
        with open(path, "w") as fh:
            fh.write("not an image")
        with self.assertRaises(Exception):
            gc.photo(self.context, path)


# ── orchestrator self-tuning ───────────────────────────────────────────────

class _LearningBase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w79-learn-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()
        self.orch = MasterOrchestrator(self.context, max_steps=4)
        self.lib = SkillLibrary(self.context.db)

    def tearDown(self):
        self.context.__exit__(None, None, None)


class LearningTests(_LearningBase):
    def test_successful_route_is_recorded(self):
        plan = Plan(goal="g", steps=[PlanStep(name="dig", goal="find it",
                                              role="research")])
        self.orch.run("find it", plan=plan, reflect=False,
                      handlers={"research": lambda t: {"found": "x"}})
        skill = self.lib.get_by_name("route:research")
        self.assertIsNotNone(skill)
        self.assertGreaterEqual(skill.success_count, 1)
        self.assertEqual(skill.failure_count, 0)

    def test_failed_route_is_recorded(self):
        plan = Plan(goal="g", steps=[PlanStep(name="dig", goal="find it",
                                              role="research")])

        def boom(task):
            raise RuntimeError("always broken")

        self.orch.run("find it", plan=plan, reflect=False,
                      handlers={"research": boom})
        skill = self.lib.get_by_name("route:research")
        self.assertIsNotNone(skill)
        self.assertGreaterEqual(skill.failure_count, 1)

    def test_arbitration_outcome_is_recorded(self):
        graph_plan = Plan(goal="g", steps=[
            PlanStep(name="worker", goal="claim", role="execution"),
            PlanStep(name="critic", goal="check", role="critic"),
        ])

        def worker(task):
            return {"verdict": "yes"}

        def critic(task):
            return {"verdict": "no"}

        result = self.orch.run("g", plan=graph_plan, reflect=False,
                               handlers={"execution": worker,
                                         "critic": critic})
        self.assertTrue(result.conflicts)
        conflict = result.conflicts[0]
        winner = conflict["winner"]
        loser = next(s for s in conflict["steps"] if s != winner)
        winner_skill = self.lib.get_by_name(f"verdict:{winner}")
        loser_skill = self.lib.get_by_name(f"verdict:{loser}")
        self.assertIsNotNone(winner_skill)
        self.assertIsNotNone(loser_skill)
        self.assertGreaterEqual(winner_skill.success_count, 1)
        self.assertGreaterEqual(loser_skill.failure_count, 1)

    def test_prior_rewards_proven_routes(self):
        # seed three clean wins for route:coding
        skill = self.lib.save("route:coding", kind="routing",
                              description="Routing experience for the coding role",
                              body="build the widget")
        for _ in range(3):
            self.lib.record_use(skill.id, success=True, task="t",
                                outcome="ok")
        boost = self.orch._role_prior_adjustment("execution", "coding",
                                                 "build the widget")
        self.assertGreaterEqual(boost, 0.2)

    def test_prior_taxed_by_chronic_failures(self):
        skill = self.lib.save("route:coding", kind="routing",
                              description="Routing experience for the coding role",
                              body="build the widget")
        for _ in range(3):
            self.lib.record_use(skill.id, success=False, task="t",
                                outcome="broken")
        penalty = self.orch._role_prior_adjustment("execution", "coding",
                                                   "build the widget")
        self.assertLessEqual(penalty, -0.15)


# ── mission pivots survive resume ──────────────────────────────────────────

class _PivotResumeBase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w79-mission-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)
        self.runner = MissionRunner(self.context, store=self.store)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    @staticmethod
    def _one_step(mission, name="gather", goal="find the card"):
        mission.state["plan"] = [{
            "name": name, "goal": goal, "role": "research",
            "kind": "io", "depends_on": [],
        }]
        return mission


class PivotPersistenceTests(_PivotResumeBase):
    def test_pivot_is_baked_into_the_persisted_plan(self):
        class _Fake:
            def run(self, prompt):
                if "[PIVOT]" in str(prompt):
                    return AgentResult(agent_id="f", role="research",
                                       output={"text": "ok"}, ok=True)
                return AgentResult(agent_id="f", role="research", output="",
                                   ok=False, error="ValueError: wrong approach")

        with mock.patch("nomorals.agents.roles.build_agent",
                        lambda role, **kw: _Fake()):
            mission = self.store.create_new("recover the card")
            self._one_step(mission)
            self.store.save(mission)
            result = self.runner.run(mission, max_iterations=2, reflect=False)

        self.assertEqual(result.status, MissionStatus.DONE)
        reloaded = self.store.get(result.mission_id)
        plan_entry = reloaded.state["plan"][0]
        self.assertTrue(plan_entry.get("pivoted"))
        self.assertIn("[PIVOT]", plan_entry["goal"])

    def test_resume_uses_the_pivoted_plan(self):
        # gather: fails once, recovers via pivot; ship: always fails on the
        # first run, succeeds on the second.  After resume, the mission
        # must finish — and 'gather' must not be re-executed at all.
        calls = {"gather": 0, "ship": 0}
        ship_ok = {"v": False}

        class _Fake:
            def run(self, prompt):
                if "[PIVOT]" in str(prompt) and \
                        "Current step: ship" not in str(prompt):
                    calls["gather"] += 1
                    return AgentResult(agent_id="f", role="research",
                                       output={"text": "ok"}, ok=True)
                if "Current step: ship" in str(prompt):
                    calls["ship"] += 1
                    if ship_ok["v"]:
                        return AgentResult(agent_id="f", role="research",
                                           output={"text": "shipped"}, ok=True)
                    return AgentResult(agent_id="f", role="research",
                                       output="", ok=False,
                                       error="ValueError: carrier down")
                calls["gather"] += 1
                return AgentResult(agent_id="f", role="research", output="",
                                   ok=False, error="ValueError: wrong approach")

        def factory(role, **kw):
            return _Fake()

        with mock.patch("nomorals.agents.roles.build_agent", factory):
            mission = self.store.create_new("recover then ship")
            mission.state["plan"] = [
                {"name": "gather", "goal": "find the card", "role": "research",
                 "kind": "io", "depends_on": []},
                {"name": "ship", "goal": "ship it", "role": "execution",
                 "kind": "io", "depends_on": ["gather"]},
            ]
            self.store.save(mission)
            first = self.runner.run(mission, max_iterations=4, reflect=False)

        self.assertEqual(first.status, MissionStatus.FAILED)
        # the pivot survived into the plan of record
        plan_entry = self.store.get(mission.id).state["plan"][0]
        self.assertIn("[PIVOT]", plan_entry["goal"])

        # operator fixes the carrier and resumes the failed mission
        ship_ok["v"] = True
        row = self.store.get(mission.id)
        row.status = MissionStatus.RUNNING
        self.store.save(row)
        with mock.patch("nomorals.agents.roles.build_agent", factory):
            second = self.runner.resume(mission.id, max_iterations=4,
                                        reflect=False)
        self.assertEqual(second.status, MissionStatus.DONE)
        # gather ran exactly twice in total (original attempt + the
        # pivoted retry) — the resume skipped it entirely
        self.assertEqual(calls["gather"], 2)
        self.assertGreaterEqual(calls["ship"], 1)


class PivotCapTests(_PivotResumeBase):
    def test_pivots_are_capped_per_step(self):
        class _Fake:
            def run(self, prompt):
                return AgentResult(agent_id="f", role="research", output="",
                                   ok=False, error="ValueError: dead end")

        with mock.patch("nomorals.agents.roles.build_agent",
                        lambda role, **kw: _Fake()):
            mission = self.store.create_new("impossible task")
            self._one_step(mission)
            self.store.save(mission)
            for _ in range(3):
                row = self.store.get(mission.id)
                row.status = MissionStatus.RUNNING
                self.store.save(row)
                self.runner.resume(mission.id, max_iterations=6,
                                   reflect=False)

        reloaded = self.store.get(mission.id)
        pivots = [p for p in (reloaded.state.get("pivots") or [])
                  if p["step"] == "gather"]
        self.assertEqual(len(pivots), MissionRunner.MAX_PIVOTS)
        self.assertIn("approach exhausted",
                      reloaded.state.get("last_error", ""))


if __name__ == "__main__":
    unittest.main()
