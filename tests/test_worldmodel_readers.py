"""World model readers + feeders (2026-11-10).

The world model (agents/worldmodel.py) shipped in the delegation
wave with NO production readers and NO feeders — tests only.  This
suite pins the closing wiring:

* the plan seam (agents/plan_contracts) feeds it: a settled step's
  contract outcome and a step failure land as OBSERVED facts, via
  the model's own API, on a persistent board at a per-test tmp path;
* uninjected, the seam behaves exactly as before (no sink, no fact);
* the feed is fail-open: a sink that raises leaves the settlement's
  outcome untouched;
* /worldmodel renders the recorded state honestly — the empty map
  says so in words, a stale fact is labelled stale, an absent topic
  is an absence of record, and internal ids never print;
* the verb is owner-only: registered in CONTROL_COMMANDS, absent
  from PUBLIC_COMMANDS, and the town game keeps its /world name.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.agents import plan_contracts
from nomorals.agents.blackboard import Blackboard
from nomorals.agents.delegation import DelegationStore
from nomorals.agents.world_board import render_topic, render_world
from nomorals.agents.worldmodel import (
    KIND_OBSERVED,
    WorldModel,
    default_world,
    plan_event_sink,
)
from nomorals.social.chat.control import CONTROL_COMMANDS, PUBLIC_COMMANDS
from nomorals.storage.db import Database


def _tmp(test: unittest.TestCase) -> Path:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    return Path(tmp.name)


class FeederTests(unittest.TestCase):
    def setUp(self):
        self.root = _tmp(self)
        db = Database(str(self.root / "delegation.db"))
        db.migrate()
        self.store = DelegationStore(db)
        self.board_path = self.root / "worldmodel-board.json"
        self.world = default_world(path=self.board_path)
        plan_contracts.install_world_sink(plan_event_sink(self.world))
        self.addCleanup(plan_contracts.clear_world_sink)
        self.step = {"name": "write", "goal": "write the file",
                     "role": "execution", "writes": ["out.txt"],
                     "payload": {"writes": ["out.txt"]}}

    def _settle(self, result):
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        return plan_contracts.settle_step_contract(
            self.store, contract, step=self.step, result=result,
            mission_id="run-x")

    def test_settled_step_lands_as_an_observed_fact(self):
        outcome = self._settle({"did": "write"})
        self.assertTrue(outcome["completed"], outcome)
        fact = self.world.read_fact("plan-step:run-x/write")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.kind, KIND_OBSERVED)
        self.assertEqual(fact.value["goal"], "write the file")
        self.assertEqual(fact.value["outcome"], "completed")
        self.assertEqual(fact.value["step"], "write")
        self.assertIn("plan step settlement", fact.source)

    def test_fact_persists_to_the_board_file(self):
        self._settle({"did": "write"})
        self.assertTrue(self.board_path.exists())
        reloaded = default_world(path=self.root / "second.json")
        self.assertIsNone(reloaded.read_fact("plan-step:run-x/write"))
        board = Blackboard()
        board.load(self.board_path)
        world2 = WorldModel(board)
        fact = world2.read_fact("plan-step:run-x/write")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.value["outcome"], "completed")

    def test_failed_step_lands_as_a_failure_fact(self):
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        plan_contracts.fail_step_contract(
            self.store, contract, detail="executor blew up")
        facts = [f for f in self.world.facts()
                 if f.value.get("outcome") == "failed"]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].value["detail"], "executor blew up")
        self.assertIn("failure", facts[0].source)

    def test_feed_renders_on_the_board_without_internal_ids(self):
        self._settle({"did": "write"})
        text = render_world(self.world)
        self.assertIn("write the file", text)
        self.assertIn("completed", text)
        self.assertIn("[observed]", text)
        # The run id and the contract id never print.
        self.assertNotIn("run-x", text)
        self.assertNotIn("plan-step:", text)


class FailOpenTests(unittest.TestCase):
    def setUp(self):
        self.root = _tmp(self)
        db = Database(str(self.root / "delegation.db"))
        db.migrate()
        self.store = DelegationStore(db)
        self.step = {"name": "write", "goal": "write the file",
                     "role": "execution", "writes": ["out.txt"],
                     "payload": {"writes": ["out.txt"]}}

    def tearDown(self):
        plan_contracts.clear_world_sink()

    def test_raising_sink_leaves_the_settlement_untouched(self):
        def boom(event):
            raise RuntimeError("world is down")

        plan_contracts.install_world_sink(boom)
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        outcome = plan_contracts.settle_step_contract(
            self.store, contract, step=self.step,
            result={"did": "write"}, mission_id="run-x")
        self.assertTrue(outcome["completed"], outcome)
        fresh = self.store.get_contract(contract.id)
        self.assertEqual(fresh.status, "completed")
        # The failure path is equally untouched.
        contract2 = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        plan_contracts.fail_step_contract(
            self.store, contract2, detail="still fine")

    def test_uninjected_seam_writes_no_fact(self):
        plan_contracts.clear_world_sink()
        world = default_world(path=self.root / "untouched.json")
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        plan_contracts.settle_step_contract(
            self.store, contract, step=self.step,
            result={"did": "write"}, mission_id="run-x")
        self.assertEqual(world.facts(), [])
        self.assertFalse((self.root / "untouched.json").exists())


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.world = WorldModel(Blackboard())

    def test_empty_map_reads_as_an_empty_map(self):
        text = render_world(self.world)
        self.assertIn("holds no facts yet", text)
        self.assertIn("absence of record", text)

    def test_stale_fact_is_labelled_stale(self):
        self.world.assert_fact(
            "repo:head", "abc123", kind=KIND_OBSERVED,
            source="git rev-parse")
        fresh = render_world(self.world)
        self.assertNotIn("(stale)", fresh)
        later = render_world(
            self.world, now=time.time() + 2 * 86400)
        self.assertIn("(stale)", later)
        self.assertIn("head", later)

    def test_invalidated_fact_shows_its_tombstone(self):
        self.world.assert_fact(
            "api:shape", {"fields": ["id"]}, kind=KIND_OBSERVED,
            source="openapi dump")
        self.world.invalidate("api:shape", reason="api v2 shipped")
        text = render_world(self.world)
        self.assertIn("INVALIDATED: api v2 shipped", text)

    def test_absent_topic_is_an_absence_of_record(self):
        text = render_topic(self.world, "plan-step")
        self.assertIn("No facts recorded", text)
        self.assertIn("absence of record", text)

    def test_topic_renders_full_detail(self):
        self.world.assert_fact(
            "repo:head", "abc123", kind=KIND_OBSERVED,
            source="git rev-parse", author="scout")
        self.world.assert_fact(
            "repo:head", "def456", kind=KIND_OBSERVED,
            source="git rev-parse", author="scout")
        text = render_topic(self.world, "repo")
        self.assertIn("head", text)
        self.assertIn("versions on record: 2", text)
        self.assertIn("def456", text)


class GatingTests(unittest.TestCase):
    def test_worldmodel_is_owner_only_and_world_stays_the_game(self):
        self.assertIn("worldmodel", CONTROL_COMMANDS)
        self.assertNotIn("worldmodel", PUBLIC_COMMANDS)
        # The town game keeps its name — no hijack.
        self.assertIn("world", CONTROL_COMMANDS)
        details = __import__(
            "nomorals.social.chat.control",
            fromlist=["COMMAND_DETAILS"]).COMMAND_DETAILS
        self.assertIn("continuing town", details["world"]["what"])
        self.assertIn("world model", details["worldmodel"]["what"])


if __name__ == "__main__":
    unittest.main()
