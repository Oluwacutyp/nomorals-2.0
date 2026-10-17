"""Unified /proposals: research upgrade tickets + proactive actions.

Regression: the research loop files proposals into ``upgrade_proposals``
and notifies "N proposal(s)", but /proposals used to read only
``proactive_log`` — so the owner checked /proposals and it was ALWAYS
empty for research proposals. /proposals must list both queues, and
/approve /deny must route upgrade ids to the upgrade queue (not fail
with "unknown proposal" on the proactive path).
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import research_loop as rl
from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.agents.upgrade_queue import UpgradePipeline, UpgradeQueue
from nomorals.core.config import load_settings
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None):
    return SimpleNamespace(db=db or _db(), settings=SimpleNamespace(
        partner=SimpleNamespace(proactive_enabled=True, quiet_start=22,
                                quiet_end=8)))


def _ticket(title):
    # Full ticket shape that passes gate_ticket (problem, approach,
    # risk, verify steps — each substantive, proposed change naming
    # the suggested file) plus the concrete-proposal bar: behavior,
    # verified file:line gap evidence, failing test.
    return {
        "title": title,
        "rationale": "The research loop surfaced a concrete improvement "
                     "with testable acceptance criteria and a bounded "
                     "blast radius.",
        "domain": "systems",
        "suggested_files": ["nomorals/agents/research_loop.py"],
        "confidence": 0.9,
        "expected_tests": ["test_a", "test_b"],
        "acceptance_criteria": ["a", "b", "c"],
        "claim_ids": ["claim-1"],
        "behavior": "List both upgrade and proactive proposals in the "
                    "/proposals output",
        "gap_evidence": "nomorals/agents/research_loop.py:340 — checked "
                        "anchor: /proposals reads only the proactive "
                        "queue, so research upgrade tickets never show",
        "failing_test": "test_proposals_lists_upgrade_and_proactive — "
                        "fails on the current tree when /proposals reads "
                        "only the proactive queue; passes once both "
                        "queues are listed.",
        "problem": "Notification counts and the proposals list disagree, "
                   "so owners cannot find the tickets they were told "
                   "about and the review queue looks broken.",
        "proposed_change": "Unify the proposals listing in "
                           "nomorals/agents/research_loop.py callers by "
                           "reading both queues, and route approvals by "
                           "identifier to the correct queue.",
        "risk": "Listing and routing only; no execution changes. "
                "Worst case is a misrouted approval, guarded by the "
                "existing owner-only gate and status checks.",
        "verify_steps": ["run pytest on tests/test_proposals_unified.py and "
                         "confirm all unified proposal tests pass cleanly",
                         "run pytest on tests/test_research_loop_g2.py and "
                         "confirm the research loop notification tests pass"],
    }


def _runtime(ctx):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = ctx
    rt._autonomy = None
    rt.gateway = SimpleNamespace(send=lambda *a, **k: SimpleNamespace(
        ok=True, error=None))
    return rt


def _proactive_row(db, pid="prop-1"):
    db.execute(
        "INSERT INTO proactive_log (id, kind, platform, chat_id, content,"
        " status, reason, decided_at, acted_at)"
        " VALUES (?, 'dm', 'telegram', '123', 'hello?', 'pending', '', 0, 0)",
        (pid,))


class UnifiedProposalsTests(unittest.TestCase):
    def test_empty_when_nothing_pending(self):
        ctx = _ctx()
        self.assertEqual(_runtime(ctx).proposals(), [])

    def test_research_proposal_visible_via_proposals(self):
        """The owner symptom: file via the research-loop path, then
        /proposals must return it (not empty)."""
        ctx = _ctx()
        pid = UpgradePipeline(ctx).propose_from_ticket(
            _ticket("adopt vector-clock conflict resolution for sync"),
            source="research_loop")
        self.assertTrue(pid.startswith("upg"))
        rows = _runtime(ctx).proposals()
        self.assertEqual([r["id"] for r in rows], [pid])
        self.assertEqual(rows[0]["queue"], "upgrade")

    def test_proactive_and_upgrade_both_listed(self):
        ctx = _ctx()
        pid = UpgradeQueue(ctx).propose(
            title="add dark mode toggle to the settings page",
            rationale="rationale " * 10, patch_plan={}, files=[],
            tests=[], claim_ids=[], source="research_loop")
        _proactive_row(ctx.db)
        rows = _runtime(ctx).proposals()
        self.assertEqual(len(rows), 2)
        by_queue = {r["queue"]: r for r in rows}
        self.assertEqual(by_queue["upgrade"]["id"], pid)
        self.assertEqual(by_queue["proactive"]["id"], "prop-1")

    def test_proposals_handler_lists_upgrade_section(self):
        tmp = tempfile.TemporaryDirectory(prefix="nm-proposals-")
        try:
            settings = load_settings(overrides={
                "home": tmp.name, "partner.platforms": "local",
                "chat.local_enabled": "true"})
            ctx = build_context(settings, with_executor=False,
                                with_tools=False)

            class _Gateway:
                def send(self, *a, **k):
                    return SimpleNamespace(ok=True, error=None)

            rt = PartnerRuntime(ctx, gateway=_Gateway())
            pid = UpgradeQueue(ctx).propose(
                title="cache compiled regexes in the router hot path",
                rationale="rationale " * 10, patch_plan={}, files=[],
                tests=[], claim_ids=[], source="research_loop")
            out = rt.handle_control("/proposals", "local:console")
            self.assertNotIn("no pending proposals", out)
            self.assertIn(pid, out)
            self.assertIn("cache compiled regexes", out)
            self.assertIn("/upgrade", out)
        finally:
            tmp.cleanup()

    def test_approve_routes_upgrade_id_to_implement(self):
        ctx = _ctx()
        pid = UpgradeQueue(ctx).propose(
            title="add dark mode toggle to the settings page",
            rationale="rationale " * 10, patch_plan={}, files=[],
            tests=[], claim_ids=[], source="research_loop")
        with mock.patch.object(
                UpgradePipeline, "approve_and_implement",
                return_value={"id": pid, "status": "implemented"}) as impl:
            res = _runtime(ctx).approve(pid)
        self.assertTrue(res["ok"])
        self.assertEqual(res["status"], "implemented")
        impl.assert_called_once()

    def test_approve_unknown_id_still_fails_honestly(self):
        ctx = _ctx()
        res = _runtime(ctx).approve("no-such-id")
        self.assertFalse(res["ok"])
        self.assertIn("no proposal", res["error"])

    def test_deny_upgrade_requires_reason(self):
        ctx = _ctx()
        pid = UpgradeQueue(ctx).propose(
            title="add dark mode toggle to the settings page",
            rationale="rationale " * 10, patch_plan={}, files=[],
            tests=[], claim_ids=[], source="research_loop")
        res = _runtime(ctx).deny(pid)
        self.assertFalse(res["ok"])
        self.assertIn("reason", res["error"])
        # still proposed — the no-reason deny must not consume it
        self.assertEqual(UpgradeQueue(ctx).get(pid)["status"], "proposed")

    def test_deny_upgrade_with_reason(self):
        ctx = _ctx()
        pid = UpgradeQueue(ctx).propose(
            title="add dark mode toggle to the settings page",
            rationale="rationale " * 10, patch_plan={}, files=[],
            tests=[], claim_ids=[], source="research_loop")
        res = _runtime(ctx).deny(pid, "not now")
        self.assertTrue(res["ok"])
        self.assertEqual(res["status"], "denied")
        self.assertEqual(UpgradeQueue(ctx).get(pid)["status"], "denied")

    def test_deny_proactive_needs_no_reason(self):
        ctx = _ctx()
        _proactive_row(ctx.db)
        res = _runtime(ctx).deny("prop-1")
        self.assertTrue(res["ok"])
        self.assertEqual(res["status"], "denied")


class NotifyPointerTests(unittest.TestCase):
    def test_digest_points_at_proposals(self):
        ctx = _ctx()
        loop = rl.ResearchLoop(ctx)
        fake_notifier = mock.Mock()
        fake_notifier.publish.return_value = {"delivered": True,
                                              "deduped": False, "id": "n1"}
        record = {"findings_count": 15, "proposals_created": 5,
                  "signals": {}}
        with mock.patch("nomorals.agents.research_loop.Notifier",
                        return_value=fake_notifier):
            loop._notify(["brief body"], record)
        _kind, _title, body = fake_notifier.publish.call_args[0][:3]
        self.assertIn("/proposals", body)
        self.assertIn("/upgrade list", body)


if __name__ == "__main__":
    unittest.main()
