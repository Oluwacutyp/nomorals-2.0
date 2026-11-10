"""Doc 18 §6 acceptance battery — the whole programme, one narrative
over a temp home (P6, the programme's last pass).

The narrative: a planned multi-step task runs under the
orchestrator; its consequential step gets a delegation contract
BEFORE it executes (P6/G9 over the P2/P5 substrate) and holds a
leased resource while it works (P3); a CoreMind house is saturated
so the next job QUEUES with a record instead of shedding (P4) and
is then cancelled mid-flight through the one facade, minting its
receipt (P5); a help request is raised on the contract and answered
(P5); the registry renders the brain's tools grouped by state with
the G7 declarations (P6); and at the end the run ticket for the
contract shows every field sourced or honestly absent (P2's
invariant), with the coverage read over the battery's population.

§6 checklist (doc §6 + eval §4):
  [x] planned multi-step task, consequential step contract + ticket
  [x] a leased resource, held by one ref, refused to a second
  [x] a queued coremind job behind a full house (nothing shed)
  [x] a cancellation mid-flight with its receipt
  [x] a help request raised + answered
  [x] the final ticket: every field sourced or honestly absent
  [x] the coverage read over the battery's population
  [x] registry-derived enumeration: every tool in the shipped
      catalog has a real handler and exactly one listing group;
      every action type the agentic loop declares has a dispatch
      branch in the loop (structural pin over the loop's own
      vocabulary)
  [ ] live integration on a real deployment — owner-gated by the
      doc itself (§6's own boundary); not run here
  [ ] prompt-injection negative fixture — named exclusion: not a
      runtime-substrate behavior. The seam reads a step result's
      EMPTINESS for its defect rule, never its content; hostile text
      in a result is data on the ledger, not an instruction path.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.coremind import CoreMind, Intent
from nomorals.agents.delegation import DelegationStore
from nomorals.agents.help_protocol import HelpStore
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.run_ticket import (
    TICKET_FIELDS,
    ticket_coverage,
    ticket_for,
)
from nomorals.agents.runtime import HybridExecutor
from nomorals.core.cancellation import cancel_with_receipt
from nomorals.core.leases import LeaseStore
from nomorals.core.policy import CapabilitySet
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


class _MindSettings:
    def __init__(self, root, mind):
        self._root = root
        self.mind = mind

    def resolve(self, key):
        return os.path.join(self._root, key)


class _MindContext:
    def __init__(self, root, mind):
        self.settings = _MindSettings(root, mind)
        self.extras = {}
        self.memory = None
        self.router = None
        self.db = None


class AcceptanceBatteryTests(unittest.TestCase):
    def test_full_programme_narrative(self):
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "battery.db"))
        db.migrate()
        store = DelegationStore(db)
        leases = LeaseStore(db)

        # ── registry leg (G7): grouped listing over planted tools ──
        reg = ToolRegistry()

        def _fn(x: str = "") -> str:
            return x

        reg.register("read_notes", _fn, description="read notes")
        reg.register("gpu_render", _fn, description="render a scene",
                     cancellation="cooperative", lease_needs=["gpu"])
        reg.register("admin_wipe", _fn, description="wipe a store",
                     capability="admin.danger", cancellation="none")
        listing = reg.grouped_listing(capabilities=CapabilitySet.none())
        self.assertIn("## available now (1)", listing)
        self.assertIn("## requires resource/lease (1)", listing)
        self.assertIn("## rejected unless authority (1)", listing)
        self.assertIn("[leases: gpu]", listing)
        self.assertIn("[cancellation: none]", listing)

        # ── the planned run (G9): trivial step + consequential step ──
        ctx = SimpleNamespace(
            db=db, settings=SimpleNamespace(), router=None,
            executor=None, blackboard=None, emit=lambda *a, **k: None)
        orch = MasterOrchestrator(ctx, executor=HybridExecutor())
        observed: dict = {}

        def handler(task):
            if task.name == "write":
                contracts = store.contracts_for_mission(orch._run_id)
                # The contract exists BEFORE the step executes.
                self.assertEqual(len(contracts), 1)
                self.assertEqual(contracts[0].status, "open")
                cid = contracts[0].id
                observed["contract_id"] = cid
                # The step holds its declared resource while it works;
                # a second holder is refused with the contention named.
                grant = leases.acquire(
                    "file", f"contract:{cid}", ttl_s=600)
                self.assertTrue(grant["granted"], grant)
                observed["lease_id"] = grant["lease"]["lease_id"]
                observed["lease_generation"] = grant["lease"]["generation"]
                refused = leases.acquire(
                    "file", "task:intruder", ttl_s=600)
                self.assertFalse(refused["granted"])
                self.assertEqual(refused["holder_ref"], f"contract:{cid}")
            return {"did": task.name}

        plan = Plan(goal="write the report", steps=[
            PlanStep(name="look", goal="gather notes", role="research"),
            PlanStep(name="write", goal="write report.txt",
                     role="execution", depends_on=["look"],
                     payload={"writes": ["report.txt"]}),
        ])
        result = orch.run("write the report", plan=plan,
                          default_handler=handler, reflect=False,
                          run_id="battery-run")
        self.assertTrue(result.ok)
        cid = observed["contract_id"]
        contract = store.get_contract(cid)
        self.assertEqual(contract.status, "completed")
        self.assertTrue(contract.consequential)

        # ── coremind leg (P4 + P5): full house, queue, cancel ──
        mind_cfg = SimpleNamespace(
            max_inflight=1, inflight_acquire_timeout_s=0.05,
            queue_overdue_s=0)
        mind = CoreMind(_MindContext(tmp.name, mind_cfg))
        gate = threading.Event()
        self.addCleanup(gate.set)

        def job_a():
            gate.wait(timeout=5)
            return "A done"

        id_a = mind._new_job(Intent(kind="research", confidence=1.0,
                                    target="alpha",
                                    route="research_swarm"))
        mind._send_async("k:console", job_a, id_a, "started A",
                         kind="research")
        self.assertTrue(_wait_for(lambda: mind._inflight_now == 1))
        id_b = mind._new_job(Intent(kind="research", confidence=1.0,
                                    target="bravo",
                                    route="research_swarm"))
        reply_b = mind._send_async("k:console", lambda: "B done", id_b,
                                   "started B", kind="research")
        self.assertIn("queued at position 1", reply_b)
        self.assertEqual(mind.job_record(id_b)["status"], "queued")
        self.assertEqual(mind._send_shed, 0)

        out = cancel_with_receipt(db, f"mind:{id_b}", coremind=mind,
                                  requested_by="owner",
                                  reason="battery cancellation")
        self.assertIsNone(out["receipt_error"])
        self.assertEqual(out["receipt"]["receipt_state"], "confirmed")
        self.assertIn("never run", out["receipt"]["detail"])
        self.assertEqual(mind.job_record(id_b)["status"], "cancelled")
        gate.set()
        self.assertTrue(_wait_for(
            lambda: mind.job_record(id_a)["status"] == "done"
            and mind._inflight_now == 0))

        # ── help leg (P5): raised on the contract, answered ──
        help_store = HelpStore(db)
        request = help_store.request_help(
            f"contract:{cid}", requester="backend:execution",
            kind="blocked", detail="report.txt template missing")
        self.assertEqual(request["state"], "open")
        answered = help_store.answer_help(
            request["id"], by="owner", answer="use the Q3 template")
        self.assertEqual(answered["state"], "answered")

        # ── the final ticket (P2): sourced or honestly absent ──
        ticket = ticket_for(f"contract:{cid}", db, coremind=mind)
        self.assertTrue(ticket["found"])
        for name in TICKET_FIELDS:
            self.assertTrue(
                (name in ticket["sources"]) != (name in ticket["absent"]),
                f"{name}: sources={ticket['sources']} "
                f"absent={ticket['absent']}")
        self.assertEqual(ticket["sources"]["purpose"], "delegation")
        self.assertEqual(ticket["sources"]["evidence"], "delegation")
        # The lease the step held rides the ticket from the P3 store
        # (the projection is id-free: lease id + resource + generation;
        # the holder linkage itself was proven at the store above).
        self.assertEqual(ticket["sources"]["leases"], "core.leases")
        self.assertIn(observed["lease_id"], str(ticket["leases"]))
        self.assertIn("file", str(ticket["leases"]))
        # The standing honesty: no shipped record carries an owner chat.
        self.assertIn("owner_chat", ticket["absent"])
        # The answered help request is visible on the ticket.
        helps = ticket["detail"]["help_requests"]
        self.assertEqual(len(helps), 1)
        self.assertEqual(helps[0]["state"], "answered")

        # ── coverage over the battery's population (P2) ──
        coverage = ticket_coverage(db, coremind=mind)
        self.assertGreaterEqual(coverage["operations"], 3)
        for entry in coverage["operations_detail"]:
            absent = set(entry["absent_fields"])
            self.assertTrue(absent <= set(TICKET_FIELDS), entry)

        # Release the lease the step held; the resource frees at the
        # next generation.
        released = leases.release(
            observed["lease_id"], generation=observed["lease_generation"])
        self.assertTrue(released["released"], released)

    def test_registry_and_loop_enumeration(self):
        # Every tool in the shipped catalog: a real handler, exactly
        # one grouped-listing home (the G7 suite pins the catalog
        # itself; here the enumeration is the §6 form).
        reg = ToolRegistry().register_builtins()
        listing = reg.grouped_listing(capabilities=CapabilitySet.all())
        seen = [ln[2:].split("(", 1)[0]
                for ln in listing.splitlines() if ln.startswith("- ")]
        expected = sorted(s.name for s in reg._tools.values()
                          if not s.deprecated)
        self.assertEqual(sorted(seen), expected)
        # Every action type the agentic loop declares in its own
        # prompt vocabulary has a dispatch branch in the loop.
        loop_src = Path("nomorals/agents/orchestration/loop.py").read_text(
            encoding="utf-8")
        self.assertIn("<tool|tools|plan|respond|ask>", loop_src)
        for action in ("tool", "tools", "plan", "respond", "ask"):
            self.assertIn(f'action == "{action}"', loop_src, action)


if __name__ == "__main__":
    unittest.main()
