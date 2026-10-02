"""Tests for nomorals.agents.upgrade_queue (unittest, runs under make test)."""
import unittest
from types import SimpleNamespace

from nomorals.agents.upgrade_queue import (
    UpgradePipeline,
    UpgradeQueue,
    register,
)
from nomorals.core.errors import ToolError
from nomorals.storage.db import Database


def make_context():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db)


def sample_kwargs(**overrides):
    kw = dict(
        title="Add exponential backoff to market data retries",
        rationale="Market data fetches fail transiently under load; retries "
                "with backoff reduce failed quotes during volatility spikes.",
        patch_plan={"edits": ["wrap fetch in retry loop"], "budget": 2},
        files=["nomorals/integrations/market_data.py"],
        tests=["retry succeeds after two transient failures"],
        claim_ids=["c1", "c2"],
        source="research",
    )
    kw.update(overrides)
    return kw


def sample_ticket(**overrides):
    t = dict(
        title="Add exponential backoff to market data retries",
        rationale="Market data fetches fail transiently under load; retries "
                "with backoff reduce failed quotes during volatility spikes.",
        patch_plan={"edits": ["wrap fetch in retry loop"]},
        suggested_files=["nomorals/integrations/market_data.py"],
        test_plan=["retry succeeds after two transient failures"],
        expected_tests=[
            "test_market_data_retry_backoff_succeeds",
            "test_market_data_retry_backoff_exhausts",
        ],
        acceptance_criteria=[
            "nomorals/integrations/market_data.py read before editing",
            "retry test fails before, passes after",
            "module suite green before and after",
        ],
        domain="integrations",
        claim_ids=["c1"],
        confidence=0.8,
    )
    t.update(overrides)
    return t


class FakeEvolution:
    """Records plan/apply calls; configurable to raise."""

    def __init__(self, fail_plan=False, fail_apply=False):
        self.plan_calls = []
        self.apply_calls = []
        self.fail_plan = fail_plan
        self.fail_apply = fail_apply

    def plan(self, instruction, **kw):
        self.plan_calls.append(instruction)
        if self.fail_plan:
            raise RuntimeError("plan exploded")
        return {"id": "evo_1"}

    def apply(self, proposal_id, **kw):
        self.apply_calls.append(proposal_id)
        if self.fail_apply:
            raise RuntimeError("apply exploded")
        return {"ok": True, "proposal_id": proposal_id}


class FakeRegistry:
    def __init__(self):
        self.context = make_context()
        self.tools = {}

    def register(self, name, **kw):
        def deco(fn):
            self.tools[name] = (fn, kw)
            return fn
        return deco


class QueueBasicsTest(unittest.TestCase):
    def setUp(self):
        self.ctx = make_context()
        self.q = UpgradeQueue(self.ctx)

    def test_propose_validation_rejects_short_title(self):
        with self.assertRaises(ValueError):
            self.q.propose(**sample_kwargs(title="short"))

    def test_propose_validation_rejects_short_rationale(self):
        with self.assertRaises(ValueError):
            self.q.propose(**sample_kwargs(rationale="too short"))

    def test_propose_returns_id_list_shows_it_get_roundtrips_json(self):
        pid = self.q.propose(**sample_kwargs())
        self.assertTrue(pid.startswith("upg_"))

        listed = self.q.list()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], pid)
        self.assertEqual(listed[0]["status"], "proposed")

        row = self.q.get(pid)
        self.assertIsNotNone(row)
        self.assertEqual(row["title"], sample_kwargs()["title"])
        self.assertEqual(row["patch_plan"], sample_kwargs()["patch_plan"])
        self.assertEqual(row["files"], sample_kwargs()["files"])
        self.assertEqual(row["tests"], sample_kwargs()["tests"])
        self.assertEqual(row["claim_ids"], ["c1", "c2"])
        self.assertEqual(row["source"], "research")

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.q.get("upg_nope"))

    def test_list_newest_first_and_status_filter(self):
        first = self.q.propose(**sample_kwargs(title="First proposal title here"))
        second = self.q.propose(**sample_kwargs(title="Second proposal title here"))
        self.assertEqual([r["id"] for r in self.q.list()], [second, first])
        self.q.approve(first)
        self.assertEqual(self.q.list(status="approved")[0]["id"], first)
        self.assertTrue(all(r["status"] == "proposed" for r in self.q.list()))


class ApproveDenyTest(unittest.TestCase):
    def setUp(self):
        self.ctx = make_context()
        self.q = UpgradeQueue(self.ctx)
        self.pid = self.q.propose(**sample_kwargs())

    def test_approve_happy_path(self):
        out = self.q.approve(self.pid, by="owner")
        self.assertEqual(out["status"], "approved")
        self.assertEqual(out["decided_by"], "owner")
        self.assertGreater(out["decided_at"], 0)

    def test_approve_twice_raises(self):
        self.q.approve(self.pid)
        with self.assertRaises(ToolError):
            self.q.approve(self.pid)

    def test_approve_unknown_id_raises(self):
        with self.assertRaises(ToolError):
            self.q.approve("upg_nope")

    def test_deny_requires_reason(self):
        with self.assertRaises(ValueError):
            self.q.deny(self.pid, "")
        with self.assertRaises(ValueError):
            self.q.deny(self.pid, "   ")

    def test_deny_stores_reason(self):
        out = self.q.deny(self.pid, "too risky for this release")
        self.assertEqual(out["status"], "denied")
        self.assertEqual(out["reason"], "too risky for this release")
        self.assertEqual(out["decided_by"], "owner")

    def test_deny_after_approve_raises(self):
        self.q.approve(self.pid)
        with self.assertRaises(ToolError):
            self.q.deny(self.pid, "changed my mind")

    def test_deny_unknown_id_raises(self):
        with self.assertRaises(ToolError):
            self.q.deny("upg_nope", "nope")

    def test_record_implemented_failed_on_ok_false(self):
        self.q.approve(self.pid)
        out = self.q.record_implemented(self.pid, {"ok": False, "error": "tests red"})
        self.assertEqual(out["status"], "failed")


class TicketTest(unittest.TestCase):
    def test_low_confidence_ticket_rejected(self):
        ctx = make_context()
        pipe = UpgradePipeline(ctx)
        out = pipe.propose_from_ticket(sample_ticket(confidence=0.2))
        self.assertEqual(out, "")
        self.assertEqual(UpgradeQueue(ctx).list(), [])

    def test_good_ticket_proposed(self):
        ctx = make_context()
        pipe = UpgradePipeline(ctx)
        pid = pipe.propose_from_ticket(sample_ticket(), source="digest")
        self.assertTrue(pid)
        row = UpgradeQueue(ctx).get(pid)
        self.assertEqual(row["status"], "proposed")
        self.assertEqual(row["source"], "digest")
        self.assertEqual(row["files"], ["nomorals/integrations/market_data.py"])


class ImplementTest(unittest.TestCase):
    def setUp(self):
        self.ctx = make_context()
        self.q = UpgradeQueue(self.ctx)
        self.pid = self.q.propose(**sample_kwargs())

    def test_approve_and_implement_happy_path(self):
        fake = FakeEvolution()
        pipe = UpgradePipeline(self.ctx, evolution=fake)
        out = pipe.approve_and_implement(self.pid)

        self.assertEqual(out["status"], "implemented")
        self.assertEqual(out["evolution_proposal_id"], "evo_1")
        self.assertTrue(out["applied_result"]["ok"])
        self.assertTrue(fake.plan_calls)
        self.assertIn("market data", fake.plan_calls[0])
        self.assertEqual(fake.apply_calls, ["evo_1"])
        self.assertEqual(self.q.get(self.pid)["status"], "implemented")

    def test_plan_failure_records_failed_and_reraises(self):
        fake = FakeEvolution(fail_plan=True)
        pipe = UpgradePipeline(self.ctx, evolution=fake)
        with self.assertRaises(RuntimeError):
            pipe.approve_and_implement(self.pid)
        row = self.q.get(self.pid)
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["applied_result"]["ok"])
        self.assertIn("plan exploded", row["applied_result"]["error"])

    def test_apply_failure_records_failed_and_reraises(self):
        fake = FakeEvolution(fail_apply=True)
        pipe = UpgradePipeline(self.ctx, evolution=fake)
        with self.assertRaises(RuntimeError):
            pipe.approve_and_implement(self.pid)
        row = self.q.get(self.pid)
        self.assertEqual(row["status"], "failed")
        self.assertIn("apply exploded", row["applied_result"]["error"])


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.reg = FakeRegistry()
        register(self.reg)
        self.fn, self.meta = self.reg.tools["upgrade_queue"]

    def test_tool_registered(self):
        self.assertIn("upgrade_queue", self.reg.tools)
        self.assertTrue(self.meta["capability"])

    def test_list_empty(self):
        out = self.fn(action="list")
        self.assertTrue(out["ok"])
        self.assertEqual(out["proposals"], [])

    def test_propose_get_deny_flow(self):
        kw = sample_kwargs()
        created = self.fn(
            action="propose", title=kw["title"], rationale=kw["rationale"],
            patch_plan=kw["patch_plan"], files=kw["files"], tests=kw["tests"])
        self.assertTrue(created["ok"])
        self.assertTrue(created["id"])
        got = self.fn(action="get", id=created["id"])
        self.assertEqual(got["proposal"]["id"], created["id"])
        denied = self.fn(action="deny", id=created["id"],
                         reason="not needed right now")
        self.assertEqual(denied["proposal"]["status"], "denied")

    def test_deny_without_reason_raises(self):
        created = self.fn(action="propose", title=sample_kwargs()["title"],
                          rationale=sample_kwargs()["rationale"])
        with self.assertRaises(ValueError):
            self.fn(action="deny", id=created["id"])

    def test_unknown_action_raises(self):
        with self.assertRaises(ToolError):
            self.fn(action="bogus")

    def test_get_unknown_id_raises(self):
        with self.assertRaises(ToolError):
            self.fn(action="get", id="upg_nope")


if __name__ == "__main__":
    unittest.main()
