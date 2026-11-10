"""Doc 18 P2 — the ticket envelope as a read-computed projection.

Pins: (a) a delegation-backed ticket carries the contract's
fencing + evidence kinds VERBATIM and the PG1-mapped grade
(domain-evidence → G4; unmapped kinds carry PG1's reasons, never a
guessed grade); (b) a browser run's ticket carries P1's durable
liveness + typed cancel state — an unhonored cancel reads
unhonored; (c) media jobs likewise (cancelled → confirmed,
timestamped request on a live job → requested); (d) fields no
source record backs are typed-absent WITH reasons, never defaults
— leases absent everywhere, contract cancellation absent because
delegation has none; (e) every populated field cites its organ in
``sources`` and every field sits in exactly one of
sources/absent; (f) unknown references return the typed not-found,
never an empty ticket; (g) the coverage listing counts a planted
mixed population correctly.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from nomorals.agents.delegation import DelegationStore
from nomorals.agents.run_ticket import (
    TICKET_FIELDS,
    ticket_coverage,
    ticket_for,
)
from nomorals.browser.jobs import BrowserRunRecordStore
from nomorals.core.jobs import JobRegistry
from nomorals.core.proof_grade import DELEGATION_UNMAPPED_REASON
from nomorals.core.tasks import AcceptanceCriterion, Task, TaskGraph
from nomorals.media_edit.jobs import MediaJobRecordStore
from nomorals.missions.mission import Mission, MissionStore
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

OLD = time.time() - 7200.0


def _file_db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "tickets.db"))
    db.migrate()
    return db


def _browser_snapshot(run_id: str, status: str, *,
                      cancel_state: str = "") -> dict:
    now = time.time()
    return {"run_id": run_id, "url": "https://example.com/a",
            "goal": "check the thing", "status": status, "steps": 2,
            "stopped_reason": "", "final_url": "https://example.com/a",
            "wall_outcome": "no wall", "error": "",
            "cancel_state": cancel_state,
            "cancel_requested_at": OLD if cancel_state else 0.0,
            "created_at": now - 7200.0, "started_at": now - 7200.0,
            "finished_at": 0.0, "updated_at": OLD}


def _media_snapshot(job_id: str, status: str, *,
                    cancel_requested_at: float = 0.0) -> dict:
    now = time.time()
    return {"job_id": job_id, "kind": "video", "label": "trim intro",
            "backend": "ffmpeg", "input_ref": "/media/in.mp4",
            "output_ref": "", "status": status, "error": "",
            "cancel_requested_at": cancel_requested_at,
            "created_at": now - 7200.0, "started_at": now - 7200.0,
            "finished_at": 0.0, "updated_at": OLD}


def _assert_partition(test: unittest.TestCase, ticket: dict) -> None:
    """Every envelope field is cited or absent-with-reason —
    exactly one of the two, for every field."""
    for name in TICKET_FIELDS:
        in_sources = name in ticket["sources"]
        in_absent = name in ticket["absent"]
        test.assertNotEqual(
            in_sources, in_absent,
            f"field {name} must be in exactly one of sources/absent "
            f"(sources={ticket['sources']}, absent={ticket['absent']})")
        if in_absent:
            test.assertTrue(ticket["absent"][name],
                            f"absent field {name} needs a reason")
            test.assertIsNone(ticket[name])


class ContractTicketTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.store = DelegationStore(self.db)
        self.contract = self.store.create_contract(
            goal="verify the thing", worker="worker-a",
            verifier="verifier-b", witness="witness-c",
            specialist_class="coder", consequential=True,
            required_evidence=("domain-evidence", "verdict",
                               "test run"),
            scope=("files", "tests"), step_budget=10, deadline=0.0)

    def test_fencing_and_evidence_verbatim_with_pg1_grade(self):
        ticket = ticket_for(f"contract:{self.contract.id}", self.db)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "contract")
        self.assertEqual(ticket["fencing"],
                         {"generation": self.contract.generation,
                          "base_revision": self.contract.base_revision})
        evidence = ticket["evidence"]
        self.assertEqual(evidence["kinds"],
                         ["domain-evidence", "verdict", "test run"])
        self.assertEqual(evidence["grade_by_kind"]["domain-evidence"],
                         "G4")
        self.assertIsNone(evidence["grade_by_kind"]["verdict"])
        self.assertIsNone(evidence["grade_by_kind"]["test run"])
        self.assertEqual(evidence["required_grade"], "G4")
        # Unmapped kinds carry PG1's reasons — never a guessed grade.
        self.assertIn("judgment OVER evidence",
                      evidence["unmapped"]["verdict"])
        self.assertEqual(evidence["unmapped"]["test run"],
                         DELEGATION_UNMAPPED_REASON)
        self.assertEqual(ticket["sources"]["fencing"], "delegation")
        self.assertEqual(ticket["sources"]["evidence"], "delegation")
        self.assertEqual(ticket["sources"]["purpose"], "delegation")
        self.assertEqual(ticket["purpose"], "verify the thing")
        self.assertEqual(ticket["actors"]["worker"], "worker-a")
        self.assertEqual(ticket["actors"]["verifier"], "verifier-b")
        self.assertEqual(ticket["capabilities"]["specialist_class"],
                         "coder")
        self.assertEqual(ticket["capabilities"]["scope"],
                         ["files", "tests"])
        self.assertEqual(ticket["budget"]["step_budget"], 10)
        _assert_partition(self, ticket)

    def test_cancel_and_leases_absent_with_reasons(self):
        ticket = ticket_for(f"contract:{self.contract.id}", self.db)
        self.assertIn("no cancellation path", ticket["absent"]["cancel"])
        self.assertIn("P5", ticket["absent"]["cancel"])
        self.assertIn("P3", ticket["absent"]["leases"])
        self.assertIn("acceptance", ticket["absent"])
        self.assertIn("owner_chat", ticket["absent"])

    def test_contract_without_evidence_kinds(self):
        bare = self.store.create_contract(
            goal="plain work", worker="w", verifier="v")
        ticket = ticket_for(f"contract:{bare.id}", self.db)
        self.assertEqual(ticket["evidence"]["kinds"], [])
        self.assertIsNone(ticket["evidence"]["required_grade"])
        self.assertEqual(ticket["sources"]["evidence"], "delegation")
        _assert_partition(self, ticket)


class MissionTicketTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.missions = MissionStore(self.db)
        self.mission = self.missions.create(
            Mission(goal="survey the archive", name="survey"))

    def test_mission_without_contract_states_absences(self):
        ticket = ticket_for(f"mission:{self.mission.id}", self.db)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["purpose"], "survey the archive")
        self.assertEqual(ticket["sources"]["purpose"], "missions")
        self.assertIn("no delegation contract is attached",
                      ticket["absent"]["evidence"])
        self.assertIn("actors", ticket["absent"])
        self.assertIn("acceptance", ticket["absent"])
        self.assertEqual(ticket["cancel"]["state"], "none")
        self.assertEqual(ticket["sources"]["cancel"], "missions")
        self.assertEqual(ticket["budget"]["budget_wall"],
                         self.mission.budget_wall)
        _assert_partition(self, ticket)

    def test_mission_with_contract_joins_evidence_not_fencing(self):
        DelegationStore(self.db).create_contract(
            goal="attached work", worker="w", verifier="v",
            mission_id=self.mission.id,
            required_evidence=("domain-evidence",))
        ticket = ticket_for(f"mission:{self.mission.id}", self.db)
        self.assertEqual(ticket["evidence"]["kinds"],
                         ["domain-evidence"])
        self.assertEqual(ticket["evidence"]["required_grade"], "G4")
        self.assertEqual(ticket["sources"]["evidence"], "delegation")
        # Fencing stays on the contract — the mission ticket says
        # so instead of collapsing generations into one number.
        self.assertIn("1 attached", ticket["absent"]["fencing"])
        self.assertTrue(any(r.startswith("contract:")
                            for r in ticket["related_refs"]))
        _assert_partition(self, ticket)

    def test_mission_acceptance_record_rides_verbatim(self):
        self.mission.state["acceptance"] = {"criteria": ["x"]}
        self.missions.save(self.mission)
        ticket = ticket_for(f"mission:{self.mission.id}", self.db)
        self.assertEqual(ticket["acceptance"], {"criteria": ["x"]})
        self.assertEqual(ticket["sources"]["acceptance"], "missions")

    def test_cancelled_mission_cancel_state(self):
        self.mission.status = "cancelled"
        self.missions.save(self.mission)
        ticket = ticket_for(f"mission:{self.mission.id}", self.db)
        self.assertEqual(ticket["cancel"]["state"], "cancelled")
        self.assertEqual(ticket["sources"]["cancel"], "missions")


class BrowserTicketTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.records = BrowserRunRecordStore(self.db)

    def test_unhonored_cancel_reads_unhonored(self):
        self.records.record(_browser_snapshot(
            "run-x", "done", cancel_state="unhonored"))
        ticket = ticket_for("browser:run-x", self.db)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "browser_run")
        self.assertEqual(ticket["cancel"]["state"], "unhonored")
        self.assertEqual(ticket["cancel"]["requested_at"], OLD)
        self.assertEqual(ticket["sources"]["cancel"], "browser")
        self.assertEqual(ticket["purpose"], "check the thing")
        self.assertEqual(ticket["liveness"]["steps"], 2)
        self.assertGreater(ticket["liveness"]["age_s"], 7000)
        # The label is id-free: host + goal, never the run id.
        self.assertIn("example.com", ticket["label"])
        self.assertNotIn("run-x", ticket["label"])
        _assert_partition(self, ticket)

    def test_requested_cancel_state_verbatim(self):
        self.records.record(_browser_snapshot(
            "run-y", "running", cancel_state="requested"))
        ticket = ticket_for("browser:run-y", self.db)
        self.assertEqual(ticket["cancel"]["state"], "requested")
        self.assertEqual(ticket["status"], "running")


class MediaTicketTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)
        self.records = MediaJobRecordStore(self.db)

    def test_cancelled_media_job_confirmed(self):
        self.records.record(_media_snapshot(
            "job-a", "cancelled", cancel_requested_at=OLD))
        ticket = ticket_for("render:job-a", self.db)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "media_job")
        self.assertEqual(ticket["cancel"]["state"], "confirmed")
        self.assertEqual(ticket["sources"]["cancel"], "render")
        self.assertEqual(ticket["detail"]["kind"], "video")
        self.assertEqual(ticket["detail"]["backend"], "ffmpeg")
        _assert_partition(self, ticket)

    def test_requested_but_running_media_job(self):
        self.records.record(_media_snapshot(
            "job-b", "running", cancel_requested_at=OLD))
        ticket = ticket_for("render:job-b", self.db)
        self.assertEqual(ticket["cancel"]["state"], "requested")
        self.assertEqual(ticket["cancel"]["requested_at"], OLD)


class JobTicketTests(unittest.TestCase):

    def _registry_with_job(self):
        registry = JobRegistry()
        gate = threading.Event()
        job = registry.submit(lambda: gate.wait(30), tool="fetch_page")
        self.addCleanup(gate.set)
        return registry, job

    def test_job_ticket_with_tool_spec_resolution(self):
        registry, job = self._registry_with_job()
        tools = ToolRegistry()
        tools.register("fetch_page", lambda url: url,
                       capability="net.fetch", kind="io")
        self.assertTrue(registry.cancel(job.id))
        ticket = ticket_for(f"job:{job.id}", registry=registry,
                            tool_registry=tools)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["capabilities"]["tool"], "fetch_page")
        self.assertEqual(ticket["capabilities"]["capability"],
                         "net.fetch")
        self.assertEqual(ticket["sources"]["capabilities"],
                         "tools.registry")
        self.assertEqual(ticket["cancel"]["state"], "requested")
        self.assertEqual(ticket["sources"]["liveness"], "core.jobs")
        self.assertIsNotNone(ticket["liveness"]["last_heartbeat_at"])
        _assert_partition(self, ticket)

    def test_job_ticket_without_tool_registry(self):
        registry, job = self._registry_with_job()
        ticket = ticket_for(f"job:{job.id}", registry=registry)
        self.assertEqual(ticket["capabilities"], {"tool": "fetch_page"})
        self.assertEqual(ticket["sources"]["capabilities"], "core.jobs")
        self.assertEqual(ticket["cancel"]["state"], "none")


class TaskTicketTests(unittest.TestCase):

    def test_task_ticket_acceptance_verbatim(self):
        graph = TaskGraph(name="g1")
        task = Task(name="build the widget", role="coder",
                    acceptance=[AcceptanceCriterion(
                        name="artifact exists",
                        spec={"artifact": True})])
        graph.add(task)
        ticket = ticket_for(f"task:{task.id}", graph=graph)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "task")
        criteria = ticket["acceptance"]["criteria"]
        self.assertEqual(criteria[0]["name"], "artifact exists")
        self.assertEqual(criteria[0]["spec"], {"artifact": True})
        self.assertIsNone(ticket["acceptance"]["results"])
        self.assertEqual(ticket["sources"]["acceptance"], "core.tasks")
        self.assertEqual(ticket["capabilities"], {"role": "coder"})
        _assert_partition(self, ticket)

    def test_task_ticket_requires_graph(self):
        ticket = ticket_for("task:whatever")
        self.assertFalse(ticket["found"])
        self.assertIn("in-memory", ticket["reason"])

    def test_task_without_criteria_states_the_empty_bar(self):
        graph = TaskGraph(name="g2")
        task = Task(name="plain step")
        graph.add(task)
        ticket = ticket_for(f"task:{task.id}", graph=graph)
        self.assertIn("acceptance", ticket["absent"])
        self.assertIn("no acceptance criteria",
                      ticket["absent"]["acceptance"])


class NotFoundTests(unittest.TestCase):

    def setUp(self) -> None:
        self.db = _file_db(self)

    def test_unknown_reference_shapes(self):
        for ref in ("contract:nope", "mission:nope", "browser:nope",
                    "render:nope", "bogus:1", "nocolon", ""):
            ticket = ticket_for(ref, self.db,
                                registry=JobRegistry())
            self.assertFalse(ticket["found"], ref)
            self.assertTrue(ticket["reason"], ref)
            self.assertNotIn("sources", ticket)

    def test_unknown_job(self):
        ticket = ticket_for("job:missing", registry=JobRegistry())
        self.assertFalse(ticket["found"])


class CoverageTests(unittest.TestCase):

    def test_coverage_counts_planted_population(self):
        db = _file_db(self)
        DelegationStore(db).create_contract(
            goal="c", worker="w", verifier="v")
        MissionStore(db).create(Mission(goal="m", name="m"))
        BrowserRunRecordStore(db).record(
            _browser_snapshot("run-c", "running"))
        MediaJobRecordStore(db).record(
            _media_snapshot("job-c", "queued"))
        registry = JobRegistry()
        gate = threading.Event()
        self.addCleanup(gate.set)
        registry.submit(lambda: gate.wait(30), tool="fetch_page")
        coverage = ticket_coverage(db, registry=registry)
        self.assertEqual(coverage["operations"], 5)
        self.assertEqual(coverage["by_source"]["contract"], 1)
        self.assertEqual(coverage["by_source"]["mission"], 1)
        self.assertEqual(coverage["by_source"]["browser_run"], 1)
        self.assertEqual(coverage["by_source"]["media_job"], 1)
        self.assertEqual(coverage["by_source"]["job"], 1)
        # Leases are absent on every ticket — stated, not hidden.
        self.assertEqual(coverage["fields_absent"]["leases"], 5)
        self.assertEqual(coverage["complete_tickets"], 0)
        # Cancellation is absent exactly on the delegation-backed
        # ticket (contracts are the only source with no cancel).
        self.assertEqual(
            coverage["absent_by_source"]["contract"]["cancel"], 1)
        self.assertNotIn(
            "cancel",
            coverage["absent_by_source"].get("browser_run", {}))
        joined = "\n".join(coverage["summary"])
        self.assertIn("5 operations", joined)
        self.assertIn("leases: absent on all 5", joined)


if __name__ == "__main__":
    unittest.main()
