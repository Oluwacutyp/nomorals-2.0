"""missions wiring: wired_runner, the VERIFYING acceptance gate, health/self-heal.

Unit tier: fully offline. Real MissionStore on an in-memory database, an
injected ArtifactStore on a temp dir; the MissionRunner is driven with a
stubbed plan so no LLM or agent is ever constructed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from nomorals.cmdline.commands.missions import _acceptance_from_args
from nomorals.core.errors import NotFound, ValidationError
from nomorals.missions import (
    ACCEPTANCE_STATE_KEY,
    MissionRunner,
    MissionStatus,
    MissionStore,
    mission_acceptance,
    normalize_acceptance,
    set_acceptance,
    wired_runner,
)
from nomorals.missions.runner import STUCK_AFTER_SECONDS
from nomorals.os.mission_state import current_state
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database


def make_db():
    db = Database(":memory:")
    db.migrate()
    return db


def make_artifact_store(test):
    tmp = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = make_db()
    test.addCleanup(db.close)
    return ArtifactStore(db, BlobStore(db, Path(tmp) / "blobs"))


def make_runner(db, **kwargs):
    kwargs.setdefault("milestones", False)
    ctx = SimpleNamespace(db=db, memory=None)
    return wired_runner(ctx, **kwargs)


def transition_log(mission):
    return [(e["from"], e["to"])
            for e in mission.state.get("transition_log", [])]


class WiredRunnerTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)

    def test_hook_is_attached(self):
        runner = make_runner(self.db, store=self.store)
        self.assertIsNotNone(runner._os_transition_hook)

    def test_full_run_moves_through_the_state_machine(self):
        runner = make_runner(self.db, store=self.store)
        runner._plan = lambda mission: []  # stub: no agent ever built
        result = runner.start("ship it", reflect=False)
        self.assertTrue(result.ok)
        done = self.store.get(result.mission_id)
        self.assertEqual(current_state(done), "COMPLETED")
        self.assertEqual(
            transition_log(done),
            [("CREATED", "PLANNED"), ("PLANNED", "RUNNING"),
             ("RUNNING", "COMPLETED")])

    def test_kwargs_pass_through(self):
        from nomorals.missions import IdempotencyStore

        runner = make_runner(self.db, store=self.store,
                             idempotency=IdempotencyStore(self.db),
                             checkpoint_every=3)
        self.assertIsInstance(runner.idempotency, IdempotencyStore)
        self.assertEqual(runner.checkpoint_every, 3)
        self.assertIs(runner.store, self.store)

    def test_store_defaults_to_context_db(self):
        ctx = SimpleNamespace(db=self.db, memory=None)
        runner = wired_runner(ctx, milestones=False)
        self.assertIsInstance(runner.store, MissionStore)

    def test_raw_runner_stays_unwired(self):
        # Documents why wired_runner exists: a directly constructed runner
        # has no hook, so its runs never touch the state machine.
        ctx = SimpleNamespace(db=self.db, memory=None)
        runner = MissionRunner(ctx, store=self.store, milestones=False)
        self.assertIsNone(runner._os_transition_hook)


class AcceptanceModelTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)

    def test_normalize_canonicalizes_criteria(self):
        out = normalize_acceptance({
            "criteria": [{"name": "q", "spec": {"metric": "quality", "gte": 0.8}}],
            "required_artifact_types": ["report"],
        })
        self.assertEqual(out, {
            "criteria": [{"name": "q", "description": "",
                          "spec": {"metric": "quality", "gte": 0.8},
                          "required": True}],
            "required_artifact_types": ["report"],
        })

    def test_normalize_rejects_bad_shapes(self):
        bad = [
            "not-a-dict",
            {"criteria": []},                                   # empty
            {"criteria": "nope"},
            {"criteria": [{"spec": {"metric": "x"}}]},           # no name
            {"criteria": [{"name": "  "}]},                      # blank name
            {"criteria": [{"name": "x", "spec": "nope"}]},       # spec not dict
            {"required_artifact_types": "report"},               # not a list
            {"required_artifact_types": ["ok", "  "]},           # blank type
        ]
        for spec in bad:
            with self.subTest(spec=spec):
                with self.assertRaises(ValidationError):
                    normalize_acceptance(spec)

    def test_create_new_with_acceptance(self):
        m = self.store.create_new(
            "goal",
            acceptance={"criteria": [{"name": "q",
                                      "spec": {"metric": "quality",
                                               "gte": 0.8}}]})
        self.assertEqual(m.state[ACCEPTANCE_STATE_KEY]["criteria"][0]["name"], "q")
        self.assertEqual(self.store.get(m.id).acceptance["criteria"][0]["name"], "q")

    def test_create_new_rejects_bad_acceptance(self):
        with self.assertRaises(ValidationError):
            self.store.create_new("goal", acceptance={"criteria": []})

    def test_set_acceptance(self):
        m = self.store.create_new("goal")
        out = set_acceptance(
            self.store, m.id,
            {"required_artifact_types": ["report"]})
        self.assertEqual(out.state[ACCEPTANCE_STATE_KEY],
                         {"criteria": [],
                          "required_artifact_types": ["report"]})
        self.assertEqual(mission_acceptance(self.store.get(m.id))["required_artifact_types"],
                         ["report"])

    def test_set_acceptance_invalidates_old_verdict(self):
        m = self.store.create_new("goal")
        set_acceptance(self.store, m.id, {"required_artifact_types": ["a"]})
        m = self.store.get(m.id)
        m.state["verification"] = {"passed": True}
        self.store.save(m)
        set_acceptance(self.store, m.id, {"required_artifact_types": ["b"]})
        self.assertNotIn("verification", self.store.get(m.id).state)

    def test_set_acceptance_terminal_raises(self):
        m = self.store.create_new("goal")
        self.store.set_status(m.id, MissionStatus.DONE)
        with self.assertRaises(ValidationError):
            set_acceptance(self.store, m.id,
                           {"required_artifact_types": ["report"]})

    def test_set_acceptance_unknown_raises(self):
        with self.assertRaises(NotFound):
            set_acceptance(self.store, "nope",
                           {"required_artifact_types": ["report"]})

    def test_acceptance_property_defaults_to_none(self):
        m = self.store.create_new("goal")
        self.assertIsNone(m.acceptance)
        self.assertIsNone(mission_acceptance(m))


class AcceptanceGateTests(unittest.TestCase):
    """The runner actually verifies: RUNNING -> VERIFYING -> COMPLETED/FAILED."""

    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)
        self.arts = make_artifact_store(self)

    def _runner(self):
        runner = make_runner(self.db, store=self.store,
                             artifact_store=self.arts)
        runner._plan = lambda mission: []  # stub: no agent ever built
        return runner

    def _run_with(self, state_updates, acceptance):
        m = self.store.create_new("gated")
        set_acceptance(self.store, m.id, acceptance)
        m = self.store.get(m.id)
        m.state.update(state_updates)
        self.store.save(m)
        runner = self._runner()
        runner._os_transition(m.id, "PLANNED", "created")  # what start() does
        result = runner.run(self.store.get(m.id), reflect=False)
        return result, self.store.get(m.id)

    def test_pass_goes_through_verifying(self):
        result, m = self._run_with(
            {"metrics": {"quality": 0.95}},
            {"criteria": [{"name": "q",
                           "spec": {"metric": "quality", "gte": 0.8}}]})
        self.assertTrue(result.ok)
        self.assertEqual(m.status, MissionStatus.DONE)
        self.assertEqual(current_state(m), "COMPLETED")
        self.assertEqual(
            transition_log(m),
            [("CREATED", "PLANNED"), ("PLANNED", "RUNNING"),
             ("RUNNING", "VERIFYING"), ("VERIFYING", "COMPLETED")])
        verification = m.state.get("verification", {})
        self.assertTrue(verification.get("passed"))
        self.assertEqual(verification["criteria"][0]["name"], "q")
        self.assertTrue(verification["criteria"][0]["passed"])

    def test_fail_goes_to_failed_never_silent_success(self):
        result, m = self._run_with(
            {"metrics": {"quality": 0.1}},
            {"criteria": [{"name": "q",
                           "spec": {"metric": "quality", "gte": 0.8}}]})
        self.assertFalse(result.ok)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertEqual(m.status, MissionStatus.FAILED)
        self.assertEqual(current_state(m), "FAILED")
        self.assertIn("acceptance verification failed", result.error)
        self.assertIn("q", result.error)
        self.assertEqual(
            transition_log(m),
            [("CREATED", "PLANNED"), ("PLANNED", "RUNNING"),
             ("RUNNING", "VERIFYING"), ("VERIFYING", "FAILED")])
        self.assertFalse(m.state["verification"]["passed"])
        self.assertEqual(
            m.state["verification"]["required_criteria_failed"], ["q"])

    def test_no_acceptance_skips_verifying(self):
        runner = self._runner()
        result = runner.start("plain", reflect=False)
        m = self.store.get(result.mission_id)
        self.assertTrue(result.ok)
        states = [to for _, to in transition_log(m)]
        self.assertNotIn("VERIFYING", states)
        self.assertNotIn("verification", m.state)

    def test_required_artifact_type_missing_fails(self):
        result, m = self._run_with(
            {}, {"required_artifact_types": ["report"]})
        self.assertFalse(result.ok)
        self.assertEqual(current_state(m), "FAILED")
        self.assertIn("missing artifact types: report", result.error)
        self.assertEqual(
            m.state["verification"]["artifact_types"]["missing"], ["report"])

    def test_required_artifact_type_present_passes(self):
        m = self.store.create_new("gated")
        set_acceptance(self.store, m.id, {"required_artifact_types": ["report"]})
        self.arts.put_text("report body", type="report", mission_id=m.id)
        runner = self._runner()
        runner._os_transition(m.id, "PLANNED", "created")
        result = runner.run(self.store.get(m.id), reflect=False)
        done = self.store.get(m.id)
        self.assertTrue(result.ok)
        self.assertEqual(current_state(done), "COMPLETED")
        self.assertTrue(done.state["verification"]["passed"])

    def test_broken_verifier_fails_closed_not_crash(self):
        # Criteria that can never evaluate (unknown spec kind) fail the
        # mission instead of raising out of the runner.
        result, m = self._run_with(
            {}, {"criteria": [{"name": "x", "spec": {"bogus": 1}}]})
        self.assertFalse(result.ok)
        self.assertEqual(m.status, MissionStatus.FAILED)
        self.assertEqual(current_state(m), "FAILED")

    def test_start_acceptance_passthrough(self):
        runner = self._runner()
        result = runner.start(
            "gated start", reflect=False,
            acceptance={"criteria": [{"name": "q",
                                      "spec": {"metric": "quality",
                                               "gte": 0.8}}]})
        m = self.store.get(result.mission_id)
        self.assertEqual(m.acceptance["criteria"][0]["name"], "q")
        # no metrics recorded -> verification fails -> FAILED, not silent DONE
        self.assertEqual(m.status, MissionStatus.FAILED)

    def test_start_rejects_bad_acceptance_up_front(self):
        runner = self._runner()
        with self.assertRaises(ValidationError):
            runner.start("bad", reflect=False, acceptance={"criteria": []})


class CliAcceptanceTests(unittest.TestCase):
    def test_neither_flag_gives_none(self):
        self.assertIsNone(
            _acceptance_from_args(Namespace(accept="", require_artifact=[])))

    def test_accept_json(self):
        out = _acceptance_from_args(Namespace(
            accept='{"criteria": [{"name": "q", "spec": {"metric": "m", "gte": 1}}]}',
            require_artifact=[]))
        self.assertEqual(out["criteria"][0]["name"], "q")

    def test_require_artifact_merges(self):
        out = _acceptance_from_args(Namespace(
            accept='{"required_artifact_types": ["a"]}',
            require_artifact=["b", "c"]))
        self.assertEqual(out["required_artifact_types"], ["a", "b", "c"])

    def test_require_artifact_alone(self):
        out = _acceptance_from_args(Namespace(accept="", require_artifact=["vid"]))
        self.assertEqual(out, {"criteria": [], "required_artifact_types": ["vid"]})

    def test_bad_json_exits_2(self):
        with self.assertRaises(SystemExit) as cm:
            _acceptance_from_args(Namespace(accept="{nope", require_artifact=[]))
        self.assertEqual(cm.exception.code, 2)

    def test_non_object_json_exits_2(self):
        with self.assertRaises(SystemExit) as cm:
            _acceptance_from_args(Namespace(accept="[1,2]", require_artifact=[]))
        self.assertEqual(cm.exception.code, 2)

    def test_bad_spec_exits_2(self):
        with self.assertRaises(SystemExit) as cm:
            _acceptance_from_args(Namespace(accept='{"criteria": []}',
                                            require_artifact=[]))
        self.assertEqual(cm.exception.code, 2)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)
        self.runner = make_runner(self.db, store=self.store)

    def _live_mission(self, name, *, heartbeat_age, checkpoint_age=None):
        m = self.store.create_new(name)
        self.store.set_status(m.id, MissionStatus.RUNNING)
        m = self.store.get(m.id)
        m.state["heartbeat"] = {"pid": os.getpid(),
                                "at": time.time() - heartbeat_age}
        self.store.save(m)
        if checkpoint_age is not None:
            point = self.store.checkpoint(self.store.get(m.id), label="step")
            self.db.execute(
                "UPDATE mission_checkpoints SET created_at = ? WHERE id = ?",
                (time.time() - checkpoint_age, point.id))
        return m.id

    def test_live_mission_is_not_stuck(self):
        mid = self._live_mission("live", heartbeat_age=5, checkpoint_age=10)
        h = self.runner.health()
        by_id = {e["id"]: e for e in h["active"]}
        self.assertIn(mid, by_id)
        self.assertFalse(by_id[mid]["stuck"])
        self.assertEqual(h["stuck"], [])

    def test_dead_worker_is_stuck(self):
        mid = self._live_mission("dead", heartbeat_age=STUCK_AFTER_SECONDS + 60,
                                 checkpoint_age=STUCK_AFTER_SECONDS + 60)
        h = self.runner.health()
        by_id = {e["id"]: e for e in h["active"]}
        self.assertTrue(by_id[mid]["stuck"])
        self.assertEqual(h["stuck"], [mid])

    def test_never_checkpointed_without_heartbeat_is_stuck(self):
        m = self.store.create_new("ghost")
        self.store.set_status(m.id, MissionStatus.RUNNING)
        h = self.runner.health()
        by_id = {e["id"]: e for e in h["active"]}
        self.assertTrue(by_id[m.id]["stuck"])
        self.assertEqual(h["stuck"], [m.id])

    def test_entry_shape(self):
        mid = self._live_mission("shaped", heartbeat_age=5)
        (entry,) = [e for e in self.runner.health()["active"] if e["id"] == mid]
        for key in ("id", "mission_id", "name", "status",
                    "checkpoint_age_seconds", "heartbeat_age_seconds",
                    "stuck", "stall"):
            self.assertIn(key, entry)
        self.assertEqual(entry["id"], entry["mission_id"])

    def test_health_shape_keys(self):
        h = self.runner.health()
        self.assertEqual(sorted(h.keys()),
                         ["active", "stuck", "stuck_after_seconds"])
        self.assertEqual(h["stuck_after_seconds"], STUCK_AFTER_SECONDS)

    def test_health_is_read_only(self):
        mid = self._live_mission("ro", heartbeat_age=5)
        before = self.store.get(mid).to_dict()
        self.runner.health()
        self.assertEqual(self.store.get(mid).to_dict(), before)


class SelfHealTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)
        self.runner = make_runner(self.db, store=self.store)
        self.runner._plan = lambda mission: []  # stub: no agent ever built

    def _stuck_mission(self):
        m = self.store.create_new("stuck")
        self.store.set_status(m.id, MissionStatus.RUNNING)
        m = self.store.get(m.id)
        # pid 2**30 cannot exist: the worker is provably dead.
        m.state["heartbeat"] = {"pid": 2 ** 30,
                                "at": time.time() - STUCK_AFTER_SECONDS - 60}
        return self.store.save(m).id

    def test_heal_dead_worker_resumes(self):
        mid = self._stuck_mission()
        out = self.runner.self_heal(mid, background=False)
        self.assertTrue(out["attempted"])
        self.assertEqual(out["mission_id"], mid)
        done = self.store.get(mid)
        self.assertEqual(done.status, MissionStatus.DONE)
        self.assertIn("self_heal", done.state)

    def test_heal_live_worker_not_attempted(self):
        m = self.store.create_new("live")
        self.store.set_status(m.id, MissionStatus.RUNNING)
        m = self.store.get(m.id)
        m.state["heartbeat"] = {"pid": os.getpid(), "at": time.time()}
        self.store.save(m)
        out = self.runner.self_heal(m.id, background=False)
        self.assertFalse(out["attempted"])
        self.assertEqual(out["reason"], "worker appears alive")
        self.assertEqual(self.store.get(m.id).status, MissionStatus.RUNNING)

    def test_heal_terminal_not_attempted(self):
        m = self.store.create_new("done")
        self.store.set_status(m.id, MissionStatus.DONE)
        out = self.runner.self_heal(m.id)
        self.assertFalse(out["attempted"])
        self.assertIn("done", out["reason"])

    def test_heal_unknown_mission_does_not_raise(self):
        out = self.runner.self_heal("no-such-mission")
        self.assertFalse(out["attempted"])
        self.assertIn("error", out)

    def test_heal_never_raises(self):
        # A runner whose store blows up still returns a dict, never raises.
        broken = make_runner(self.db, store=self.store)
        broken.store = None  # type: ignore[assignment]
        out = broken.self_heal("x")
        self.assertFalse(out["attempted"])


if __name__ == "__main__":
    unittest.main()
