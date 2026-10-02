"""Tests for the /upgrade chat UX: research→approve→evolve loop in chat.

Covers the pure renderers in nomorals.agents.upgrade_chat, the proposal
resolver, the _control_upgrade dispatch (with a mock pipeline), and the
owner-only gating that protects every upgrade command.
"""
import unittest
from types import SimpleNamespace

from nomorals.agents.upgrade_chat import (
    UPGRADE_USAGE,
    render_applied_digest,
    render_upgrade_diff,
    render_upgrade_list,
    render_upgrade_show,
    resolve_proposal,
)
from nomorals.agents.upgrade_queue import UpgradeQueue
from nomorals.storage.db import Database


def make_context():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db)


def sample_proposal(**overrides):
    p = dict(
        id="upg_abc123",
        title="Add exponential backoff to market data retries",
        rationale="Market data fetches fail transiently under load; retries "
                  "with backoff reduce failed quotes during volatility spikes.",
        patch_plan={"steps": ["wrap fetch in retry loop", "add jitter"],
                    "risk": "low — retries are idempotent"},
        files=["nomorals/integrations/market_data.py"],
        tests=["retry succeeds after two transient failures"],
        claim_ids=["c1"],
        source="research",
        status="proposed",
    )
    p.update(overrides)
    return p


def propose_sample(ctx, **overrides):
    kw = dict(
        title="Add exponential backoff to market data retries",
        rationale="Market data fetches fail transiently under load; retries "
                  "with backoff reduce failed quotes during volatility spikes.",
        patch_plan={"steps": ["wrap fetch in retry loop", "add jitter"]},
        files=["nomorals/integrations/market_data.py"],
        tests=["retry succeeds after two transient failures"],
        claim_ids=["c1"],
        source="research",
    )
    kw.update(overrides)
    pid = UpgradeQueue(ctx).propose(**kw)
    return UpgradeQueue(ctx).get(pid)


class FakePipeline:
    """Mock UpgradePipeline: records decisions, never touches the tree."""

    def __init__(self, ctx):
        self.queue = UpgradeQueue(ctx)
        self.approved = []
        self.denied = []
        self.apply_result = None

    def approve_and_implement(self, pid, by="owner"):
        self.approved.append((pid, by))
        result = (self.apply_result if self.apply_result is not None else
                  {"applied": True, "edits": ["nomorals/x.py"],
                   "verified": True, "commit": "deadbeef1234",
                   "branch": "main-2.0"})
        return self.queue.record_implemented(pid, result)

    def deny_with_reason(self, pid, reason, by="owner"):
        self.denied.append((pid, reason, by))
        return self.queue.deny(pid, reason, by=by)


def make_runtime(ctx):
    from nomorals.agents.partner_runtime import PartnerRuntime
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = ctx
    return rt


class TestResolveProposal(unittest.TestCase):
    def setUp(self):
        self.ctx = make_context()
        self.queue = UpgradeQueue(self.ctx)
        self.p1 = propose_sample(self.ctx)
        self.p2 = propose_sample(
            self.ctx,
            title="Cache knowledge-graph embeddings between heartbeats",
            rationale="Embedding recomputation on every heartbeat wastes "
                      "tokens; a short-lived cache cuts the cost sharply.",
            source="digest")

    def test_exact_id(self):
        p, err = resolve_proposal(self.queue, self.p1["id"])
        self.assertEqual(err, "")
        self.assertEqual(p["id"], self.p1["id"])

    def test_id_prefix(self):
        p, err = resolve_proposal(self.queue, self.p1["id"][:-2])
        self.assertEqual(err, "")
        self.assertEqual(p["id"], self.p1["id"])

    def test_title_substring(self):
        p, err = resolve_proposal(self.queue, "embeddings between")
        self.assertEqual(err, "")
        self.assertEqual(p["id"], self.p2["id"])

    def test_unknown(self):
        p, err = resolve_proposal(self.queue, "upg_nope")
        self.assertIsNone(p)
        self.assertIn("no upgrade proposal", err)

    def test_empty(self):
        p, err = resolve_proposal(self.queue, "   ")
        self.assertIsNone(p)
        self.assertIn("/upgrade list", err)

    def test_ambiguous(self):
        p, err = resolve_proposal(self.queue, "upg_")
        self.assertIsNone(p)
        self.assertIn("ambiguous", err)


class TestRenderList(unittest.TestCase):
    def test_one_line_each(self):
        p1, p2 = sample_proposal(), sample_proposal(
            id="upg_def456", title="Second proposal title here",
            source="digest")
        out = render_upgrade_list([p1, p2])
        self.assertIn("pending upgrades (2)", out)
        self.assertIn("upg_abc123", out)
        self.assertIn("[research]", out)
        self.assertIn("Add exponential backoff", out)
        self.assertIn("upg_def456", out)
        self.assertIn("[digest]", out)

    def test_empty(self):
        self.assertIn("no pending upgrade proposals", render_upgrade_list([]))


class TestRenderShow(unittest.TestCase):
    def test_full_ticket(self):
        out = render_upgrade_show(sample_proposal())
        self.assertIn("upg_abc123", out)
        self.assertIn("Add exponential backoff", out)
        self.assertIn("problem:", out)
        self.assertIn("fail transiently under load", out)
        self.assertIn("patch plan:", out)
        self.assertIn("nomorals/integrations/market_data.py", out)
        self.assertIn("retry succeeds after two transient failures", out)
        self.assertIn("1. wrap fetch in retry loop", out)
        self.assertIn("risk: low — retries are idempotent", out)
        self.assertIn("source: research", out)
        self.assertIn("/upgrade diff upg_abc123", out)
        self.assertIn("/upgrade approve upg_abc123", out)

    def test_unstated_risk_is_honest(self):
        p = sample_proposal(patch_plan={"steps": ["do the thing"]})
        out = render_upgrade_show(p)
        self.assertIn("risk: unstated in the ticket", out)

    def test_evolution_reference_shown(self):
        p = sample_proposal(patch_plan={"evolution_proposal_id": "evo_99"})
        out = render_upgrade_show(p)
        self.assertIn("evolution proposal: evo_99", out)


class TestRenderDiff(unittest.TestCase):
    def test_steps_rendered(self):
        out = render_upgrade_diff(sample_proposal())
        self.assertIn("🔍 diff preview", out)
        self.assertIn("1. wrap fetch in retry loop", out)
        self.assertIn("files in scope:", out)

    def test_real_hunks_from_evolution_proposal(self):
        evo = SimpleNamespace(
            id="evo_7",
            edits=[{"path": "nomorals/x.py",
                    "old": "a = 1\nb = 2\n",
                    "new": "a = 1\nb = 3\n"}])
        out = render_upgrade_diff(sample_proposal(), evo_proposal=evo)
        self.assertIn("evolution plan evo_7", out)
        self.assertIn("📄 nomorals/x.py", out)
        self.assertIn("-b = 2", out)
        self.assertIn("+b = 3", out)
        self.assertIn("/upgrade approve upg_abc123", out)

    def test_new_file_hunk(self):
        evo = SimpleNamespace(
            id="evo_8",
            edits=[{"path": "nomorals/new_mod.py", "old": "",
                    "new": "X = 1\n"}])
        out = render_upgrade_diff(sample_proposal(), evo_proposal=evo)
        self.assertIn("(new file)", out)
        self.assertIn("+ X = 1", out)

    def test_missing_evolution_plan_falls_back(self):
        p = sample_proposal(patch_plan={"evolution_proposal_id": "evo_gone"})
        out = render_upgrade_diff(p, evo_proposal=None)
        self.assertIn("evo_gone", out)
        self.assertIn("falls back to the ticket text", out)

    def test_staged_skill_edit_noted(self):
        p = sample_proposal(patch_plan={"skill_edit_id": "sk_1"})
        out = render_upgrade_diff(p)
        self.assertIn("staged skill edit sk_1", out)


class TestRenderAppliedDigest(unittest.TestCase):
    def test_evolution_success(self):
        p = sample_proposal(
            status="implemented",
            applied_result={"applied": True, "status": "applied",
                            "proposal": "evo_7",
                            "edits": ["nomorals/x.py", "nomorals/y.py"],
                            "verified": True, "commit": "deadbeef1234",
                            "branch": "main-2.0"})
        out = render_applied_digest(p)
        self.assertIn("✅ upgrade applied", out)
        self.assertIn("files touched (2): nomorals/x.py, nomorals/y.py", out)
        self.assertIn("test gate: passed", out)
        self.assertIn("commit: deadbeef1234", out)
        self.assertIn("main-2.0", out)

    def test_evolution_reverted(self):
        p = sample_proposal(
            status="failed",
            applied_result={"applied": False, "status": "reverted",
                            "proposal": "evo_7",
                            "reason": "verification failed — working tree restored exactly",
                            "report": "FAILED tests/test_x.py::test_thing\n12 passed, 1 failed"})
        out = render_applied_digest(p)
        self.assertIn("❌ upgrade failed", out)
        self.assertIn("verification failed", out)
        self.assertIn("test_thing", out)
        self.assertIn("recorded as failed", out)

    def test_exception_path(self):
        p = sample_proposal(
            status="failed",
            applied_result={"ok": False,
                            "error": "ToolError: working tree is dirty"})
        out = render_applied_digest(p)
        self.assertIn("❌ upgrade failed", out)
        self.assertIn("working tree is dirty", out)

    def test_staged_skill_edit_path(self):
        p = sample_proposal(
            status="implemented",
            applied_result={"ok": True, "edit": "sk_1",
                            "status": "applied"})
        out = render_applied_digest(p)
        self.assertIn("✅ upgrade applied", out)
        self.assertIn("sk_1", out)


class TestControlUpgrade(unittest.TestCase):
    def setUp(self):
        self.ctx = make_context()
        self.rt = make_runtime(self.ctx)
        self.pipe = FakePipeline(self.ctx)
        self.p1 = propose_sample(self.ctx)

    def test_list(self):
        out = self.rt._control_upgrade("list")
        self.assertIn("pending upgrades (1)", out)
        self.assertIn(self.p1["id"], out)

    def test_list_default_verb(self):
        out = self.rt._control_upgrade("")
        self.assertIn("pending upgrades", out)

    def test_show(self):
        out = self.rt._control_upgrade(f"show {self.p1['id']}")
        self.assertIn("problem:", out)
        self.assertIn("patch plan:", out)

    def test_show_unknown(self):
        out = self.rt._control_upgrade("show upg_nope")
        self.assertIn("no upgrade proposal", out)

    def test_diff(self):
        out = self.rt._control_upgrade(f"diff {self.p1["id"][:-2]}")
        self.assertIn("🔍 diff preview", out)
        self.assertIn("wrap fetch in retry loop", out)

    def test_approve_wires_to_pipeline(self):
        out = self.rt._control_upgrade(f"approve {self.p1['id']}",
                                       _pipeline=self.pipe)
        self.assertEqual(self.pipe.approved, [(self.p1["id"], "owner")])
        self.assertIn("✅ upgrade applied", out)
        self.assertIn("files touched", out)
        self.assertIn("commit: deadbeef1234", out)
        # the proposal really moved to implemented
        row = UpgradeQueue(self.ctx).get(self.p1["id"])
        self.assertEqual(row["status"], "implemented")

    def test_approve_refuses_non_proposed(self):
        self.rt._control_upgrade(f"approve {self.p1['id']}",
                                 _pipeline=self.pipe)
        out = self.rt._control_upgrade(f"approve {self.p1['id']}",
                                       _pipeline=self.pipe)
        self.assertIn("only 'proposed' tickets can be approved", out)

    def test_approve_failure_surfaced(self):
        def raising(pid, by="owner"):
            # mirrors UpgradePipeline: record failed, then re-raise
            self.pipe.queue.record_implemented(
                pid, {"ok": False, "error": "RuntimeError: boom"})
            raise RuntimeError("boom")
        self.pipe.approve_and_implement = raising
        out = self.rt._control_upgrade(f"approve {self.p1['id']}",
                                       _pipeline=self.pipe)
        self.assertIn("❌ apply failed and was recorded as failed", out)
        row = UpgradeQueue(self.ctx).get(self.p1["id"])
        self.assertEqual(row["status"], "failed")

    def test_deny_requires_reason(self):
        out = self.rt._control_upgrade(f"deny {self.p1['id']}",
                                       _pipeline=self.pipe)
        self.assertIn("usage: /upgrade deny <id> <reason>", out)
        self.assertEqual(self.pipe.denied, [])

    def test_deny_wires_to_pipeline(self):
        out = self.rt._control_upgrade(
            f"deny {self.p1['id']} not worth the churn", _pipeline=self.pipe)
        self.assertEqual(
            self.pipe.denied,
            [(self.p1["id"], "not worth the churn", "owner")])
        self.assertIn("🚫 upgrade denied", out)
        self.assertIn("not worth the churn", out)
        row = UpgradeQueue(self.ctx).get(self.p1["id"])
        self.assertEqual(row["status"], "denied")

    def test_applied(self):
        self.rt._control_upgrade(f"approve {self.p1['id']}",
                                 _pipeline=self.pipe)
        out = self.rt._control_upgrade("applied")
        self.assertIn("applied upgrades (1)", out)
        self.assertIn("✅ upgrade applied", out)

    def test_applied_empty(self):
        out = self.rt._control_upgrade("applied")
        self.assertIn("no upgrades applied yet", out)

    def test_unknown_verb_shows_usage(self):
        out = self.rt._control_upgrade("frobnicate")
        self.assertIn("/upgrade list", out)
        self.assertEqual(out, UPGRADE_USAGE)


class TestUpgradeCommandCatalog(unittest.TestCase):
    def test_registered_and_parsed(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS, CONTROL_COMMANDS, LIST_GROUPS, LIST_ONELINERS,
            parse_control,
        )
        self.assertIn("upgrade", CONTROL_COMMANDS)
        self.assertIn("upgrade", COMMAND_DETAILS)
        self.assertIn("upgrade", LIST_ONELINERS)
        grouped = {c for _, cs in LIST_GROUPS for c in cs}
        self.assertIn("upgrade", grouped)
        cmd = parse_control("/upgrade diff upg_abc")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "upgrade")
        self.assertEqual(cmd.tail, "diff upg_abc")
        self.assertIn("/upgrade", COMMAND_DETAILS["upgrade"]["usage"])

    def test_help_text_lists_upgrade(self):
        from nomorals.social.chat.control import help_text
        self.assertIn("/upgrade", help_text())


class TestOwnerGating(unittest.TestCase):
    """Every /upgrade verb rides handle_control, which only fires for
    operator chats — a slash from anyone else never reaches dispatch."""

    def test_upgrade_dispatch_lives_behind_operator_gate(self):
        from nomorals.social.chat.control import parse_control
        # the command must exist so it routes into handle_control…
        self.assertEqual(parse_control("/upgrade approve upg_x").kind,
                         "upgrade")
        # …and handle_control is only invoked from the _is_operator branch.
        rt = make_runtime(make_context())
        rt._owner_chats = {"telegram:1"}
        stranger = SimpleNamespace(chat=SimpleNamespace(key="telegram:999"))
        owner = SimpleNamespace(chat=SimpleNamespace(key="telegram:1"))
        self.assertFalse(rt._is_operator(stranger))
        self.assertTrue(rt._is_operator(owner))

    def test_console_is_always_owner(self):
        from nomorals.social.chat.base import is_owner_chat
        console = SimpleNamespace(key="local:console")
        self.assertTrue(is_owner_chat(console, owner_chats=set()))


if __name__ == "__main__":
    unittest.main()
