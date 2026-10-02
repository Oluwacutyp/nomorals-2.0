"""Wave J — timeline replay and mission redrive.

Unit tier: fully offline. Real Timeline (in-memory), real MissionStore and
ArtifactStore on an in-memory Database; the replay is built from scripted
events and the redrive is exercised against real persisted missions.
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.core.errors import NotFound
from nomorals.core.events import Event
from nomorals.missions import MissionStore
from nomorals.os.mission_state import (
    COMPLETED,
    FAILED,
    PLANNED,
    RUNNING,
    current_state,
    transition,
)
from nomorals.os.replay import (
    RedriveRefused,
    redrive_mission,
    replay_mission,
)
from nomorals.os.timeline import Timeline
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


def evt(topic, mission_id, **data):
    return Event(topic=topic, data={"mission_id": mission_id, **data},
                 source="test")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline(None)
        self.addCleanup(self.timeline.close)
        self.mid = "01JTESTMISSION000000000001"

    def _script_mission(self):
        tl = self.timeline
        mid = self.mid
        tl.record(evt("mission.transition", mid, from_state="CREATED",
                      to_state="PLANNED", note="planned"))
        tl.record(evt("task.started", mid, task="fetch",
                      summary="fetching source data"))
        tl.record(evt("artifact.created", mid, artifact_id="art_fetch",
                      uri="artifact://art_fetch", type="json",
                      creator="fetch-agent"))
        tl.record(evt("task.completed", mid, task="fetch", status="ok"))
        tl.record(evt("mission.transition", mid, from_state="PLANNED",
                      to_state="RUNNING", note="run started"))
        tl.record(evt("task.started", mid, task="write",
                      summary="writing the report"))
        tl.record(evt("artifact.created", mid, artifact_id="art_report",
                      uri="artifact://art_report", type="text",
                      creator="write-agent"))
        tl.record(evt("mission.verify", mid, verdict="passed",
                      details="all criteria met"))
        tl.record(evt("mission.transition", mid, from_state="RUNNING",
                      to_state="COMPLETED", note="done"))
        # An unrelated mission's events must not leak into this replay.
        tl.record(evt("mission.transition", "01JOTHERMISSION0000000002",
                      from_state="CREATED", to_state="PLANNED"))

    def test_reconstructs_transitions_tasks_artifacts(self):
        self._script_mission()
        report = replay_mission(self.timeline, self.mid)
        self.assertEqual(report.event_count, 9)
        self.assertEqual(
            [(t["from"], t["to"]) for t in report.transitions],
            [("CREATED", "PLANNED"), ("PLANNED", "RUNNING"),
             ("RUNNING", "COMPLETED")],
        )
        self.assertEqual(report.transitions[0]["note"], "planned")
        self.assertEqual(len(report.tasks), 3)
        self.assertTrue(all(t["topic"].startswith("task.")
                            for t in report.tasks))
        self.assertEqual(
            [(a["artifact_id"], a["type"]) for a in report.artifacts],
            [("art_fetch", "json"), ("art_report", "text")],
        )
        self.assertEqual(report.artifacts[0]["creator"], "fetch-agent")
        self.assertEqual(len(report.verifications), 1)
        self.assertEqual(report.verifications[0]["verdict"], "passed")
        # Oldest event first.
        self.assertEqual(report.transitions[0]["to"], "PLANNED")
        self.assertEqual(report.transitions[-1]["to"], "COMPLETED")
        self.assertIsNotNone(report.covered_from)
        self.assertGreaterEqual(report.covered_to or 0,
                                report.covered_from or 0)

    def test_narrative_and_json(self):
        self._script_mission()
        report = replay_mission(self.timeline, self.mid)
        text = report.narrative()
        self.assertIn("replay of 9 timeline events", text)
        self.assertIn("CREATED -> PLANNED", text)
        self.assertIn("RUNNING -> COMPLETED", text)
        self.assertIn("artifact://art_report", text)
        self.assertIn("task activity (3)", text)
        self.assertIn("verdict=passed", text)
        # The JSON form must round-trip through the stdlib encoder.
        payload = json.loads(json.dumps(report.to_dict(), default=str))
        self.assertEqual(payload["mission_id"], self.mid)
        self.assertEqual(len(payload["transitions"]), 3)
        self.assertEqual(len(payload["artifacts"]), 2)

    def test_empty_mission_replays_cleanly(self):
        report = replay_mission(self.timeline, "01JNOMISSION000000000003")
        self.assertEqual(report.event_count, 0)
        self.assertEqual(report.transitions, [])
        self.assertIn("replay of 0 timeline events", report.narrative())

    def test_unknown_topics_land_in_other_not_dropped(self):
        self.timeline.record(
            evt("session.heartbeat", self.mid, note="still alive"))
        report = replay_mission(self.timeline, self.mid)
        self.assertEqual(report.event_count, 1)
        self.assertEqual(len(report.other), 1)
        self.assertEqual(report.other[0]["topic"], "session.heartbeat")


class RedriveTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)
        self.artifacts = make_artifact_store(self)

    def _failed_mission(self, goal="write the quarterly report"):
        mission = self.store.create_new(
            goal, budget_wall=600.0,
            state={"plan": [
                {"name": "fetch", "goal": "fetch data", "role": "io"},
                {"name": "write", "goal": "write report", "role": "writer"},
            ]})
        transition(self.store, mission.id, PLANNED)
        transition(self.store, mission.id, RUNNING)
        transition(self.store, mission.id, FAILED, note="provider blew up")
        art = self.artifacts.put_text(
            "partial data", type="text", creator="fetch-agent",
            mission_id=mission.id)
        return mission, art

    def test_refuses_without_confirm(self):
        mission, _ = self._failed_mission()
        with self.assertRaises(RedriveRefused):
            redrive_mission(self.store, mission.id, confirm=False,
                            artifact_store=self.artifacts)

    def test_refuses_running_mission(self):
        import os

        mission = self.store.create_new("still going")
        transition(self.store, mission.id, PLANNED)
        transition(self.store, mission.id, RUNNING)
        # A genuinely live runner: fresh heartbeat, this process is alive.
        live = self.store.get(mission.id)
        live.state["heartbeat"] = {"pid": os.getpid(), "at": time.time()}
        self.store.save(live)
        with self.assertRaises(RedriveRefused) as ctx:
            redrive_mission(self.store, mission.id, confirm=True,
                            artifact_store=self.artifacts)
        self.assertIn("running", str(ctx.exception).lower())
        # Still running afterwards — the redrive changed nothing.
        self.assertEqual(self.store.get(mission.id).status, "running")

    def test_refuses_paused_mission(self):
        mission = self.store.create_new("on hold")
        transition(self.store, mission.id, PLANNED)
        transition(self.store, mission.id, RUNNING)
        self.store.set_status(mission.id, "paused", "operator paused it")
        with self.assertRaises(RedriveRefused):
            redrive_mission(self.store, mission.id, confirm=True)

    def test_reconciled_zombie_is_redrivable(self):
        mission = self.store.create_new("zombie run")
        mission.status = "running"  # claims running…
        mission.state["heartbeat"] = {"pid": 999999999,
                                      "at": time.time() - 100_000}
        self.store.save(mission)  # …but the runner died long ago
        report = redrive_mission(self.store, mission.id, confirm=True,
                                 artifact_store=self.artifacts)
        self.assertEqual(
            self.store.get(mission.id).status, "failed")  # reconciled
        self.assertNotEqual(report.new_mission_id, mission.id)

    def test_redrive_creates_linked_mission(self):
        mission, art = self._failed_mission()
        report = redrive_mission(self.store, mission.id, confirm=True,
                                 artifact_store=self.artifacts,
                                 note="second attempt")
        self.assertNotEqual(report.new_mission_id, mission.id)
        self.assertEqual(report.original_mission_id, mission.id)
        self.assertEqual(report.plan_steps, 2)

        new = self.store.get(report.new_mission_id)
        self.assertEqual(new.goal, mission.goal)
        self.assertEqual(new.metadata.get("derived_from_mission"), mission.id)
        self.assertEqual(new.state["redrive"]["of"], mission.id)
        self.assertEqual(new.state["redrive"]["original_status"], "failed")
        # The recorded plan rides along, deep-copied.
        self.assertEqual(
            [p["name"] for p in new.state["plan"]], ["fetch", "write"])
        self.assertEqual(new.state["plan"], mission.state["plan"])
        self.assertIsNot(new.state["plan"], mission.state["plan"])
        # Budgets are copied; nothing is marked complete on the new mission.
        self.assertEqual(new.budget_wall, 600.0)
        self.assertEqual(new.state.get("completed_steps"), [])
        # The new mission is PLANNED in the os state machine, ready to run.
        self.assertEqual(current_state(new), PLANNED)

        # The artifact graph links the redrive record derived_from the
        # original mission's artifacts.
        record = self.artifacts.get(report.artifact_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.mission_id, report.new_mission_id)
        self.assertIn(art.id, record.provenance.derived_from)
        self.assertEqual(report.derived_from_artifact_ids, [art.id])
        self.assertEqual(record.metadata.get("derived_from_mission"),
                         mission.id)

    def test_redrive_without_artifact_store_still_links_metadata(self):
        mission, _ = self._failed_mission()
        report = redrive_mission(self.store, mission.id, confirm=True)
        self.assertEqual(report.artifact_id, "")
        new = self.store.get(report.new_mission_id)
        self.assertEqual(new.metadata.get("derived_from_mission"), mission.id)

    def test_original_mission_is_untouched(self):
        mission, _ = self._failed_mission()
        redrive_mission(self.store, mission.id, confirm=True,
                        artifact_store=self.artifacts)
        original = self.store.get(mission.id)
        self.assertEqual(original.status, "failed")
        self.assertEqual(current_state(original), FAILED)

    def test_unknown_mission_raises_not_found(self):
        with self.assertRaises(NotFound):
            redrive_mission(self.store, "01JNOPE00000000000000000001",
                            confirm=True)

    def test_to_dict_is_json_serializable(self):
        mission, _ = self._failed_mission()
        report = redrive_mission(self.store, mission.id, confirm=True,
                                 artifact_store=self.artifacts)
        payload = json.loads(json.dumps(report.to_dict(), default=str))
        self.assertEqual(payload["original_mission_id"], mission.id)
        self.assertEqual(payload["os_state"], "PLANNED")

    def test_completed_mission_redrives_too(self):
        mission = self.store.create_new("finished work",
                                        state={"plan": []})
        transition(self.store, mission.id, PLANNED)
        transition(self.store, mission.id, RUNNING)
        transition(self.store, mission.id, COMPLETED)
        report = redrive_mission(self.store, mission.id, confirm=True)
        self.assertNotEqual(report.new_mission_id, mission.id)
        self.assertEqual(report.plan_steps, 0)


if __name__ == "__main__":
    unittest.main()
