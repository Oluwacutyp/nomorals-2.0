"""Public-surface fix 5 — the shelfware cluster gets consumers.

The census (public-surface Item 6, HIGH rows) found the doc-18 /
doc-20 ticket cluster shipped with zero production callers. This
suite pins the wiring, through the REAL surfaces:

* the /runs board renders live-shaped records as tickets through
  the real control handler (a delegation contract, a mission, a
  browser run and an incident each appear with their projection
  fields; absent fields show the projection's own reason text;
  internal ids never print);
* detail resolves by board ordinal;
* a stop issued from the surface routes through the ONE
  cancellation facade — a CancelReceipt is minted AND the
  underlying work actually stops (the organ's native mechanism);
* a lease-declaring tool acquires + releases through the real
  registry dispatch seam, and a held lease fences a second
  contender with generation semantics (doc 18 P3);
* the partner tool_loop's full-set listing and the context
  engine's tools section come from grouped_listing() — a ToolSpec
  with cancellation="none" shows it in the brain's view, and a
  lease-declaring tool lands in the lease group.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.delegation import DelegationStore
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.agents.partner.tool_loop import ToolCallingLoop
from nomorals.browser.jobs import BrowserJobManager
from nomorals.context.engine import ContextEngine
from nomorals.core.cancellation import receipts_for_ref
from nomorals.core.incidents import IncidentJournal
from nomorals.core.leases import LeaseStore, live_lease_for_resource
from nomorals.missions.mission import Mission, MissionStatus, MissionStore
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _text(reply):
    """A board render's text — boards return actionable replies
    since button lane 5; read the text half here."""
    return reply.text if hasattr(reply, "text") else reply


def _ordinal_of(board_text: str, marker: str) -> int:
    for line in _text(board_text).splitlines():
        if marker in line:
            head = line.split(".", 1)[0].strip()
            if head.isdigit():
                return int(head)
    raise AssertionError(f"no board row carries {marker!r}:\n{board_text}")


class _Runtime(RuntimeSystemMixin):
    def __init__(self, ctx):
        self.context = ctx


class RunsSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmp, "nm.db"))
        self.db.migrate()
        settings = SimpleNamespace(
            resolve=lambda rel, default=None: os.path.join(self.tmp, rel))
        self.ctx = SimpleNamespace(db=self.db, settings=settings,
                                   extras={})
        self.runtime = _Runtime(self.ctx)
        # The plants: one of each live-shaped record family.
        self.contract = DelegationStore(self.db).create_contract(
            goal="wire the shelfware cluster", worker="builder",
            verifier="checker")
        store = MissionStore(self.db)
        now = time.time()
        self.mission = store.create(Mission(
            goal="keep the board honest", name="board-keeper",
            status=MissionStatus.RUNNING,
            created_at=now, updated_at=now))
        self.journal_path = os.path.join(
            self.tmp, "data", "error_journal.db")
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        journal = IncidentJournal(self.journal_path)
        try:
            raise ValueError("planted failure for the board")
        except ValueError as exc:
            self.incident = journal.record_incident(
                exc, subsystem="shelfware-test", severity="high",
                context={"origin": "fix5-test"})

    # ── the board ────────────────────────────────────────────────

    def test_board_renders_every_family_through_the_handler(self):
        board = _text(self.runtime._control_runs(""))
        self.assertIn("wire the shelfware cluster", board)
        self.assertIn("board-keeper", board)
        self.assertIn("[run · contract]", board)
        self.assertIn("[run · mission]", board)
        self.assertIn("[heal · incident]", board)
        self.assertIn("shelfware-test", board)
        # Internal ids never print — the doc-15 law.
        self.assertNotIn(self.contract.id, board)
        self.assertNotIn(self.mission.id, board)

    def test_detail_by_ordinal_with_absent_reasons(self):
        board = self.runtime._control_runs("")
        n = _ordinal_of(board, "wire the shelfware cluster")
        detail = self.runtime._control_runs(str(n))
        self.assertIn("purpose", detail)
        self.assertIn("evidence", detail)
        # owner_chat is absent on delegation records — the
        # projection's reason text renders, verbatim in substance.
        self.assertIn("not recorded:", detail)
        self.assertIn("owner/chat attribution", detail)
        self.assertNotIn(self.contract.id, detail)

    def test_healing_detail_shows_steward_absent_reason(self):
        board = self.runtime._control_runs("")
        n = _ordinal_of(board, "[heal · incident]")
        detail = self.runtime._control_runs(str(n))
        self.assertIn("steward — not recorded:", detail)
        self.assertIn("claim", detail)

    def test_coverage_rollup_renders(self):
        text = self.runtime._control_runs("coverage")
        self.assertIn("executions", text)
        self.assertIn("healing", text)
        self.assertIn("operations", text)

    def test_bad_ordinal_is_a_usage_answer(self):
        reply = self.runtime._control_runs("stop 9999")
        self.assertIn("isn't on the board", reply)
        self.assertIn("usage", self.runtime._control_runs("bogus"))

    # ── stops through the facade ─────────────────────────────────

    def test_stop_contract_mints_receipt_and_stops_natively(self):
        board = self.runtime._control_runs("")
        n = _ordinal_of(board, "wire the shelfware cluster")
        reply = self.runtime._control_runs(f"stop {n} board test stop")
        self.assertIn("Receipt: confirmed", reply)
        record = DelegationStore(self.db).get_contract(self.contract.id)
        self.assertEqual(record.status, "cancelled")
        receipts = receipts_for_ref(
            self.db, f"contract:{self.contract.id}")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["receipt_state"], "confirmed")
        self.assertEqual(receipts[0]["organ"], "delegation")
        self.assertEqual(receipts[0]["requested_by"], "owner")
        # The receipt rides the ticket's cancel field afterwards.
        detail = self.runtime._control_runs(str(
            _ordinal_of(self.runtime._control_runs(""),
                        "wire the shelfware cluster")))
        self.assertIn("receipt", detail)
        self.assertIn("confirmed", detail)

    def test_stop_mission_mints_receipt_and_flips_status(self):
        board = self.runtime._control_runs("")
        n = _ordinal_of(board, "board-keeper")
        reply = self.runtime._control_runs(f"stop {n}")
        self.assertIn("Receipt: confirmed", reply)
        mission = MissionStore(self.db).get(self.mission.id)
        self.assertEqual(str(mission.status), "cancelled")
        receipts = receipts_for_ref(
            self.db, f"mission:{self.mission.id}")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["native_state"],
                         "running->cancelled")

    def test_stop_browser_run_requests_and_the_run_honors_it(self):
        mgr = BrowserJobManager(db=self.db)

        def runner(job):
            while not job.cancel_requested():
                time.sleep(0.005)
            return {"stopped_reason": "cancelled: owner stop",
                    "steps": 1}

        job, _dup = mgr.start("https://example.test/page",
                              "read the page", runner)
        self.runtime._browser_run_jobs = lambda: mgr
        # The durable record lands asynchronously — wait for the
        # board to see the run.
        deadline = time.time() + 10
        board = ""
        while time.time() < deadline:
            board = _text(self.runtime._control_runs(""))
            if "read the page" in board:
                break
            time.sleep(0.05)
        self.assertIn("read the page", board)
        n = _ordinal_of(board, "read the page")
        reply = self.runtime._control_runs(f"stop {n}")
        self.assertIn("Receipt: requested", reply)
        # The native mechanism was invoked: the runner's own flag.
        self.assertTrue(job.cancel_requested())
        receipts = receipts_for_ref(self.db, f"browser:{job.job_id}")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["receipt_state"], "requested")
        # The run honors the request and lands cancelled on its
        # durable record — the final word the receipt pointed at.
        deadline = time.time() + 10
        row = None
        while time.time() < deadline:
            row = self.db.query_one(
                "SELECT status, cancel_state FROM browser_run_records"
                " WHERE run_id = ?", (job.job_id,))
            if row is not None and row["status"] == "cancelled":
                break
            time.sleep(0.05)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["cancel_state"], "confirmed")

    def test_stop_healing_ticket_refused_in_words(self):
        board = self.runtime._control_runs("")
        n = _ordinal_of(board, "[heal · incident]")
        reply = self.runtime._control_runs(f"stop {n}")
        self.assertIn("aren't stoppable work", reply)
        self.assertIn("/immune", reply)
        self.assertEqual(
            receipts_for_ref(self.db, f"incident:{self.incident.id}"),
            [])


class LeaseSeatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmp, "nm.db"))
        self.db.migrate()
        self.seen: dict = {}

        def leased_fn(**kwargs):
            live = live_lease_for_resource(self.db, "gpu")
            self.seen["during"] = live
            return "ran"

        self.leased_fn = leased_fn

    def _registry(self, with_db=True):
        ctx = SimpleNamespace(db=self.db) if with_db else SimpleNamespace()
        reg = ToolRegistry(context=ctx)
        reg.register("gpu_render", self.leased_fn,
                     description="renders on the gpu",
                     cancellation="cooperative", lease_needs=["gpu"])
        return reg

    def test_call_acquires_and_releases_through_dispatch(self):
        reg = self._registry()
        outcome = reg.call("gpu_render", actor="test")
        self.assertTrue(outcome.ok, outcome)
        during = self.seen["during"]
        self.assertIsNotNone(during)
        self.assertEqual(during["holder_ref"], "tool:gpu_render")
        self.assertEqual(during["generation"], 1)
        # Released when the call returned.
        self.assertIsNone(live_lease_for_resource(self.db, "gpu"))

    def test_held_lease_fences_a_second_contender(self):
        store = LeaseStore(self.db)
        grant = store.acquire("gpu", "job:other-holder", ttl_s=600)
        self.assertTrue(grant["granted"])
        reg = self._registry()
        outcome = reg.call("gpu_render", actor="test")
        self.assertFalse(outcome.ok)
        self.assertIn("gpu", str(outcome.error))
        self.assertIn("job:other-holder", str(outcome.error))
        self.assertNotIn("during", self.seen)  # the tool never ran
        # The fenced holder is untouched; after it releases, the
        # next call takes generation 2 — fencing is monotonic.
        store.release(grant["lease"]["lease_id"],
                      grant["lease"]["generation"])
        outcome = reg.call("gpu_render", actor="test")
        self.assertTrue(outcome.ok, outcome)
        self.assertEqual(self.seen["during"]["generation"], 2)
        self.assertIsNone(live_lease_for_resource(self.db, "gpu"))

    def test_no_db_context_calls_unfenced_but_working(self):
        reg = self._registry(with_db=False)
        outcome = reg.call("gpu_render", actor="test")
        self.assertTrue(outcome.ok, outcome)
        self.assertIsNone(self.seen["during"])

    def test_undeclared_tool_never_touches_the_lease_store(self):
        reg = self._registry()
        reg.register("plain", lambda **kw: "plain-ran",
                     description="no declarations")
        outcome = reg.call("plain", actor="test")
        self.assertTrue(outcome.ok, outcome)
        LeaseStore(self.db)  # construction creates the (empty) table
        rows = self.db.query(
            "SELECT COUNT(*) AS n FROM resource_leases")
        self.assertEqual(rows[0]["n"], 0)


class GroupedListingConsumerTests(unittest.TestCase):
    def _registry(self):
        reg = ToolRegistry(context=SimpleNamespace())
        reg.register("plain_tool", lambda **kw: "x",
                     description="a plain tool")
        reg.register("final_tool", lambda **kw: "x",
                     description="cannot be cancelled",
                     cancellation="none")
        reg.register("gpu_render", lambda **kw: "x",
                     description="renders on the gpu",
                     cancellation="cooperative", lease_needs=["gpu"])
        return reg

    def test_tool_loop_full_listing_comes_from_grouped_listing(self):
        loop = ToolCallingLoop(None, self._registry())
        listing = loop._tool_listing()
        self.assertIn("grouped by state", listing)
        self.assertIn("## requires resource/lease (1)", listing)
        self.assertIn("[leases: gpu]", listing)
        self.assertIn("[cancellation: none]", listing)

    def test_context_engine_tools_section_uses_grouped_listing(self):
        engine = ContextEngine()
        section = engine._tools_section(self._registry())
        self.assertIn("grouped by state", section.content)
        self.assertIn("[cancellation: none]", section.content)
        self.assertIn("## requires resource/lease (1)", section.content)

    def test_grouped_fallback_keeps_flat_listing_for_old_registries(self):
        class FlatOnly:
            def ranked_listing(self, query, capabilities=None, limit=40):
                return ""

            def prompt_listing(self, capabilities=None):
                return "- flat_tool: the flat fallback"

        loop = ToolCallingLoop(None, FlatOnly())
        self.assertEqual(loop._tool_listing(), "- flat_tool: the flat fallback")


class CatalogTests(unittest.TestCase):
    def test_runs_is_owner_only_and_documented(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            PUBLIC_COMMANDS,
        )

        self.assertIn("runs", CONTROL_COMMANDS)
        self.assertNotIn("runs", PUBLIC_COMMANDS)
        detail = COMMAND_DETAILS["runs"]
        self.assertIn("/runs", detail["usage"])
        self.assertTrue(detail["what"])
        self.assertTrue(detail["example"])


if __name__ == "__main__":
    unittest.main()
