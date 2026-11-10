"""Doc 15 P6 — the unified inspection surfaces.

Chat (/notify verbs + /cases), the outcome_steward tool actions and
the nm outcomes / nm notify CLI all read the same journals and the
same case board. These tests pin:

* a delivered alert's why-chain renders completely from the surface
  alone (producer, fact, decision, routing, delivery ack) with no
  internal ids anywhere;
* a suppressed alert is explained, not invisible;
* a notification stored without the decision machinery gets the
  honest no-record answer;
* the /notify board's pre-P6 behaviour is byte-preserved;
* case list/show render the evidence trail readably, no ids;
* /cases verify drives a REAL investigation cycle (delegation
  contract and all) on a planted case;
* ack records + quiets sweep re-surfacing + NEVER resolves, and a
  material change re-surfaces an acknowledged case.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.agents.notifier import Notifier
from nomorals.agents.outcome_cases import ack_changed, ack_snapshot
from nomorals.agents.outcome_steward import OutcomeSteward
from nomorals.agents.partner.runtime_schedule import RuntimeScheduleMixin
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.core.jobs import JobRegistry
from nomorals.missions.mission import Mission, MissionStatus, MissionStore
from nomorals.storage.db import Database

# ── harness (P2/P3 shapes) ────────────────────────────────────────────


class _SendResult:
    ok = True
    message_id = "m1"


class FakeGateway:
    def __init__(self, live=("telegram",)):
        self.live = set(live)
        self.sent: list[tuple] = []

    def status(self):
        return {p: {"running_in_session": p in self.live}
                for p in ("telegram", "whatsapp")}

    def send(self, platform, chat_ref, text, buttons=None):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult()


def _resolve(*args, **kwargs):
    if "default" in kwargs:
        return kwargs["default"]
    return args[-1] if args else None


def make_ctx(gateway=None):
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "nm.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram:111", proactive_enabled=True,
        proactive_briefing=True, proactive_watchers=True,
        quiet_start=0, quiet_end=0, timezone="UTC")
    settings = SimpleNamespace(workspace_dir=tmp, partner=partner,
                               resolve=_resolve)
    ctx = SimpleNamespace(
        db=db, settings=settings,
        extras={"gateway": gateway} if gateway else {})
    return ctx, db


class ScheduleRuntime(RuntimeScheduleMixin):
    def __init__(self, ctx, gateway):
        self.context = ctx
        self.gateway = gateway


class SystemRuntime(RuntimeSystemMixin):
    def __init__(self, ctx):
        self.context = ctx


def _mission(store, *, status, state=None, goal="do the thing"):
    now = time.time()
    mission = Mission(goal=goal, status=status,
                      state=dict(state or {}),
                      created_at=now, updated_at=now)
    return store.create(mission)


def _liveness_failed(store, goal="render the report"):
    state = {"heartbeat": {"at": time.time() - 100000.0,
                           "pid": 999999},
             "last_error": ("runner died — no heartbeat for 27h "
                            "(pid 999999 is gone)")}
    state["status_note"] = f"reconciled: {state['last_error']}"
    return _mission(store, status=MissionStatus.FAILED, state=state,
                    goal=goal)


_DEPLOY_META = {
    "subject": "deploy", "outcome": "succeeded",
    "prev_revision": "rev-1", "current_revision": "rev-2",
    "result_refs": ["artifact:build-9"], "impact": "v2 live",
}


class NotifySurfaceTests(unittest.TestCase):
    def setUp(self):
        self.gw = FakeGateway()
        self.ctx, self.db = make_ctx(self.gw)
        self.notifier = Notifier(self.ctx, self.gw)
        self.runtime = ScheduleRuntime(self.ctx, self.gw)

    def _publish_deploy(self):
        return self.notifier.publish(
            "deploy", "deploy finished", "v2 is live",
            output_meta=dict(_DEPLOY_META))

    def test_why_chain_for_delivered_alert_from_surface_alone(self):
        res = self._publish_deploy()
        self.assertTrue(res["delivered"])
        text = self.runtime._control_notify("why 1")
        # The whole chain, in owner language.
        self.assertIn("deploy finished", text)
        self.assertIn("it asserted: deploy — succeeded", text)
        self.assertIn("revision rev-1 → rev-2", text)
        self.assertIn("decision: paged you", text)
        self.assertIn("routing: owner", text)
        self.assertIn("delivery: delivered", text)
        # No internal ids: not the notification id, not the event id.
        self.assertNotIn(str(res["id"]), text)
        from nomorals.agents.alert_policy import AlertPolicy
        row = AlertPolicy.candidate_for_notification(self.db, res["id"])
        self.assertIsNotNone(row)
        self.assertNotIn(row["event_id"], text)

    def test_decisions_board_and_coverage(self):
        self._publish_deploy()
        board = self.runtime._control_notify("decisions")
        self.assertIn("paged you", board)
        self.assertIn("deploy", board)
        coverage = self.runtime._control_notify("coverage")
        self.assertIn("candidates recorded", coverage)
        self.assertIn("decisions:", coverage)

    def test_bare_board_byte_shape_preserved(self):
        self._publish_deploy()
        board = self.runtime._control_notify("")
        self.assertTrue(board.startswith("notifications (1):"))
        self.assertIn("states: ✓sent", board)
        numbered = self.runtime._control_notify("5")
        self.assertTrue(numbered.startswith("notifications (1):"))
        garbage = self.runtime._control_notify("frobnicate")
        self.assertTrue(garbage.startswith("notifications (1):"))

    def test_suppressed_tick_is_explained_not_invisible(self):
        self.notifier.publish(
            "jobwatch", "job tick", "still running",
            output_meta={"subject": "job:report",
                         "current_revision": "running"})
        self.assertEqual(self.gw.sent, [])
        text = self.runtime._control_notify("why decision 1")
        self.assertIn("suppressed", text)
        self.assertIn("unchanged tick", text)
        self.assertIn("never sent", text)
        board = self.runtime._control_notify("decisions")
        self.assertIn("suppressed", board)

    def test_why_for_directly_stored_row_is_honest(self):
        # The production direct-store path (progress notes): no
        # candidate/decision record exists for the row.
        self.notifier._store("missions", "render progress",
                             "half-way there", 0,
                             delivery_state="held-quiet-hours")
        text = self.runtime._control_notify("why 1")
        self.assertIn("no decision record exists", text)
        self.assertIn("held", text)
        self.assertNotIn("decision: paged", text)


class CasesSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.db = make_ctx()
        self.missions = MissionStore(self.db)
        self.registry = JobRegistry()
        self.steward = OutcomeSteward(self.db, registry=self.registry)
        self.runtime = SystemRuntime(self.ctx)

    def _open_mission_case(self):
        _liveness_failed(self.missions)
        self.steward.sweep()
        return self.steward.cases.list_cases()[0]

    def test_board_and_show_render_evidence_without_ids(self):
        case = self._open_mission_case()
        board = self.runtime._control_cases("")
        self.assertIn("render the report", board)
        self.assertIn("outcome unknown", board)
        self.assertNotIn(case.id, board)
        self.assertNotIn("oc-", board)
        show = self.runtime._control_cases("1")
        self.assertIn("evidence trail:", show)
        self.assertIn("outcome unknown", show)
        self.assertNotIn(case.id, show)
        self.assertNotIn("ocr-", show)

    def test_verify_drives_a_real_investigation(self):
        case = self._open_mission_case()
        text = self.runtime._control_cases("verify 1")
        self.assertIn("still honestly unknown", text)
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.state, "closed-unknown")
        self.assertEqual(closed.verdict, "outcome_unknown")
        # The investigation ran on the delegation substrate: a
        # contract exists and completed.
        self.assertTrue(closed.contract_id)
        from nomorals.agents.delegation import DelegationStore
        contract = DelegationStore(self.db).get_contract(
            closed.contract_id)
        self.assertEqual(contract.status, "completed")
        self.assertIn("evidence trail:", text)

    def test_ack_records_quiets_and_never_resolves(self):
        case = self._open_mission_case()
        text = self.runtime._control_cases("ack 1")
        self.assertIn("acknowledged", text)
        after = self.steward.cases.get(case.id)
        # Ack is a recorded act, not a resolution.
        self.assertEqual(after.state, "open")
        self.assertEqual(after.verdict, "outcome_unknown")
        self.assertIsNotNone(ack_snapshot(after))
        kinds = [r.kind for r in self.steward.cases.receipts(case.id)]
        self.assertIn("acknowledged", kinds)
        # The sweep leaves an acknowledged, unchanged case alone.
        report = self.steward.run_pending()
        self.assertEqual(report["acknowledged_held"], 1)
        self.assertEqual(report["ran"], 0)
        still = self.steward.cases.get(case.id)
        self.assertEqual(still.state, "open")
        # The board parks it in the quiet tail.
        board = self.runtime._control_cases("")
        self.assertIn("acknowledged (quiet unless they change)",
                      board)
        # A material change re-surfaces it: an explicit verify runs
        # the case to closed-unknown — state moved past the ack
        # snapshot, so it leaves the quiet section.
        self.steward.run_case(case.id)
        moved = self.steward.cases.get(case.id)
        self.assertTrue(ack_changed(moved))
        board = self.runtime._control_cases("")
        self.assertNotIn("acknowledged (quiet unless they change)",
                         board)
        self.assertIn("closed-unknown", board)

    def test_verify_resolves_from_readback_evidence(self):
        # A done-but-uncollected job: the effect is in the records.
        # Planted in the SHARED registry because the chat surface's
        # steward reads it — the production shape.
        from nomorals.core.jobs import shared_registry

        registry = shared_registry()
        job = registry.submit(lambda: {"report": "landed"},
                              tool="report")
        registry.collect(job.id, timeout=10)
        job.ended_at = time.time() - 700.0
        OutcomeSteward(self.db).sweep()
        case = self.steward.cases.list_cases()[0]
        text = self.runtime._control_cases("verify 1")
        self.assertIn("resolved: succeeded", text)
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.verdict, "succeeded")
        self.assertIsNotNone(closed.evidence_receipt_id)
        self.assertNotIn(closed.evidence_receipt_id, text)
        self.assertNotIn(case.id, text)


class ToolAndCliParityTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.db = make_ctx()
        self.missions = MissionStore(self.db)
        self.steward = OutcomeSteward(self.db, registry=JobRegistry())
        _liveness_failed(self.missions)
        self.steward.sweep()
        self.case = self.steward.cases.list_cases()[0]

    def _tool(self):
        from nomorals.agents.outcome_steward import register

        captured = {}

        class FakeRegistry:
            context = self.ctx

            def register(self, name, **kwargs):
                def deco(fn):
                    captured["fn"] = fn
                    return fn
                return deco

        register(FakeRegistry())
        return captured["fn"]

    def test_tool_inspect_show_verify_ack(self):
        fn = self._tool()
        listing = fn(action="inspect")
        self.assertTrue(listing["ok"])
        self.assertEqual(len(listing["cases"]), 1)
        self.assertEqual(listing["cases"][0]["id"], self.case.id)
        shown = fn(action="show", case_id=self.case.id)
        self.assertEqual(shown["subject"], "render the report")
        acked = fn(action="ack", case_id=self.case.id)
        self.assertTrue(acked["ok"])
        after = self.steward.cases.get(self.case.id)
        self.assertEqual(after.state, "open")  # ack never resolves
        verified = fn(action="verify", case_id=self.case.id)
        self.assertEqual(verified["outcome"], "closed-unknown")
        self.assertEqual(verified["verdict"], "outcome_unknown")

    def test_cli_outcomes_parity(self):
        from nomorals.cmdline.commands.outcomes import _cmd_outcomes

        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_outcomes(argparse.Namespace(
                outcomes_action="list", json=False), self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("render the report", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_outcomes(argparse.Namespace(
                outcomes_action="show", n=1, json=False), self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("evidence trail:", out.getvalue())
        self.assertNotIn(self.case.id, out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_outcomes(argparse.Namespace(
                outcomes_action="status", json=False), self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("outcome cases:", out.getvalue())
        self.assertIn("unavailable", out.getvalue())

    def test_cli_notify_parity(self):
        from nomorals.cmdline.commands.notify import _cmd_notify

        gw = FakeGateway()
        ctx, _db = make_ctx(gw)
        Notifier(ctx, gw).publish(
            "deploy", "deploy finished", "v2 is live",
            output_meta=dict(_DEPLOY_META))
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_notify(argparse.Namespace(
                notify_action="decisions", n=15, json=False), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("paged you", out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _cmd_notify(argparse.Namespace(
                notify_action="why", n=1, json=False), ctx)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("decision: paged you", text)
        self.assertIn("delivery: delivered", text)


if __name__ == "__main__":
    unittest.main()
