"""Wave G2 — owner-only gating for every /upgrade verb + rollback digest.

GATING: each of the six /upgrade verbs (list, show, diff, approve, deny,
applied) is invoked from a non-owner chat and must be denied BEFORE any
queue read/write.  The queue and pipeline are replaced with exploding
doubles: if the gate were bypassed, the test errors instead of passing —
a green run proves the gate fires first, not just that the reply is
empty.

DIGEST: the post-apply digest states what changed (files + captured
hunks) and how to roll back — ``/evolve revert <evo-id>`` for evolution
applies (committed or on-disk), the exact ``git revert`` when only a
commit hash is recorded, and an honest "no rollback available" line when
the apply path has no revert.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.agents.upgrade_chat import render_applied_digest
from nomorals.agents.upgrade_queue import UpgradeQueue
from nomorals.storage.db import Database


def make_context():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db)


def make_runtime(ctx):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = ctx
    rt._owner_chats = {"telegram:1"}
    return rt


OWNER_CHAT = SimpleNamespace(key="telegram:1")
STRANGER_CHAT = SimpleNamespace(key="telegram:999")

UPGRADE_DENIAL = "owner-only: /upgrade is not available in this chat."


def propose_sample(ctx):
    pid = UpgradeQueue(ctx).propose(
        title="Add exponential backoff to market data retries",
        rationale="Market data fetches fail transiently under load; retries "
                  "with backoff reduce failed quotes during volatility spikes.",
        patch_plan={"steps": ["wrap fetch in retry loop", "add jitter"]},
        files=["nomorals/integrations/market_data.py"],
        tests=["retry succeeds after two transient failures"],
        claim_ids=["c1"],
        source="research",
    )
    return UpgradeQueue(ctx).get(pid)


class ExplodingQueue:
    """Stands in for UpgradeQueue: any instantiation or attribute use
    means the owner gate was bypassed — the test errors loudly."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        ExplodingQueue.instances.append((args, kwargs))
        raise AssertionError("UpgradeQueue touched from a non-owner chat")

    def __getattr__(self, name):
        raise AssertionError(
            f"UpgradeQueue.{name} touched from a non-owner chat")


class ExplodingPipeline:
    """Stands in for UpgradePipeline: same tripwire as ExplodingQueue."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        ExplodingPipeline.instances.append((args, kwargs))
        raise AssertionError("UpgradePipeline touched from a non-owner chat")


def explode_queue_and_pipeline(test):
    """Context manager patching both queue and pipeline with tripwires."""
    return mock.patch.multiple(
        "nomorals.agents.upgrade_queue",
        UpgradeQueue=ExplodingQueue,
        UpgradePipeline=ExplodingPipeline,
    )


class TestUpgradeOwnerGating(unittest.TestCase):
    """Every /upgrade verb is unreachable from non-owner chats."""

    def setUp(self):
        self.ctx = make_context()
        self.rt = make_runtime(self.ctx)
        self.p1 = propose_sample(self.ctx)
        ExplodingQueue.instances.clear()
        ExplodingPipeline.instances.clear()

    def _deny(self, tail, chat=STRANGER_CHAT):
        with explode_queue_and_pipeline(self):
            return self.rt._control_upgrade(tail, _chat=chat)

    def test_all_six_verbs_denied_for_non_owner(self):
        tails = [
            "list",
            f"show {self.p1['id']}",
            f"diff {self.p1['id']}",
            f"approve {self.p1['id']}",
            f"deny {self.p1['id']} not worth it",
            "applied",
        ]
        for tail in tails:
            with self.subTest(verb=tail.split()[0]):
                out = self._deny(tail)
                self.assertEqual(out, UPGRADE_DENIAL)
        # the tripwires never fired: the gate ran before any queue
        # construction, let alone a read or write.
        self.assertEqual(ExplodingQueue.instances, [])
        self.assertEqual(ExplodingPipeline.instances, [])

    def test_missing_chat_denied_fail_closed(self):
        with explode_queue_and_pipeline(self):
            out = self.rt._control_upgrade("list")
        self.assertEqual(out, UPGRADE_DENIAL)
        self.assertEqual(ExplodingQueue.instances, [])

    def test_approve_is_noop_for_non_owner(self):
        out = self._deny(f"approve {self.p1['id']}")
        self.assertEqual(out, UPGRADE_DENIAL)
        row = UpgradeQueue(self.ctx).get(self.p1["id"])
        self.assertEqual(row["status"], "proposed")

    def test_deny_is_noop_for_non_owner(self):
        out = self._deny(f"deny {self.p1['id']} malicious reason")
        self.assertEqual(out, UPGRADE_DENIAL)
        row = UpgradeQueue(self.ctx).get(self.p1["id"])
        self.assertEqual(row["status"], "proposed")

    def test_owner_chat_reaches_verbs(self):
        # positive control: the gate lets the owner through, and the
        # real queue is used (not the tripwire).
        out = self.rt._control_upgrade("list", _chat=OWNER_CHAT)
        self.assertIn("pending upgrades (1)", out)
        self.assertIn(self.p1["id"], out)

    def test_handle_control_threads_chat_for_stranger(self):
        msg = SimpleNamespace(chat=STRANGER_CHAT, text="/upgrade list")
        with explode_queue_and_pipeline(self):
            out = self.rt.handle_control("/upgrade list", "telegram:999",
                                         message=msg)
        self.assertEqual(out, UPGRADE_DENIAL)
        self.assertEqual(ExplodingQueue.instances, [])

    def test_handle_control_threads_chat_for_owner(self):
        msg = SimpleNamespace(chat=OWNER_CHAT, text="/upgrade list")
        out = self.rt.handle_control("/upgrade list", "telegram:1",
                                     message=msg)
        self.assertIn("pending upgrades (1)", out)

    def test_is_operator_is_still_the_first_layer(self):
        stranger = SimpleNamespace(chat=STRANGER_CHAT)
        owner = SimpleNamespace(chat=OWNER_CHAT)
        self.assertFalse(self.rt._is_operator(stranger))
        self.assertTrue(self.rt._is_operator(owner))


def applied_proposal(**overrides):
    p = dict(
        id="upg_abc123",
        title="Add exponential backoff to market data retries",
        status="implemented",
        evolution_proposal_id="",
    )
    p.update(overrides)
    return p


def evo_result(**overrides):
    r = dict(
        applied=True,
        proposal_id="evo_7",
        edits=["nomorals/x.py", "nomorals/y.py"],
        hunks=[
            {"path": "nomorals/x.py",
             "diff": ["@@ -1,2 +1,2 @@", "-b = 2", "+b = 3"]},
            {"path": "nomorals/y.py",
             "diff": ["+ NEW_LINE = 1"]},
        ],
        verified=True,
        commit="deadbeef1234abcd",
        branch="main-2.0",
    )
    r.update(overrides)
    return r


class TestUpgradeDigestRollback(unittest.TestCase):
    """The post-apply digest names the real rollback — or admits none."""

    def test_evolution_committed_shows_revert_command(self):
        out = render_applied_digest(
            applied_proposal(applied_result=evo_result()))
        self.assertIn("✅ upgrade applied", out)
        self.assertIn("files touched (2): nomorals/x.py, nomorals/y.py", out)
        self.assertIn("📄 nomorals/x.py", out)
        self.assertIn("-b = 2", out)
        self.assertIn("+b = 3", out)
        self.assertIn("test gate: passed", out)
        self.assertIn("commit: deadbeef1234", out)
        # the repo's own rollback path — EvolutionAgent.revert does the
        # git revert and re-verifies the gate.
        self.assertIn("↩️ rollback: /evolve revert evo_7", out)

    def test_evolution_on_disk_still_has_rollback(self):
        # applied on disk without a commit is NOT "no rollback":
        # /evolve revert restores the touched paths via file checkout.
        res = evo_result()
        del res["commit"]
        del res["branch"]
        out = render_applied_digest(applied_proposal(applied_result=res))
        self.assertIn("commit: on disk (not committed)", out)
        self.assertIn("↩️ rollback: /evolve revert evo_7", out)
        self.assertNotIn("no rollback available", out)

    def test_commit_without_proposal_id_falls_back_to_git_revert(self):
        res = evo_result()
        del res["proposal_id"]
        out = render_applied_digest(applied_proposal(applied_result=res))
        self.assertIn("↩️ rollback: git revert deadbeef1234abcd", out)

    def test_no_commit_no_proposal_is_honest(self):
        res = evo_result()
        del res["proposal_id"]
        del res["commit"]
        out = render_applied_digest(applied_proposal(applied_result=res))
        self.assertIn("no rollback available", out)
        self.assertIn("without a recorded commit", out)

    def test_row_evolution_proposal_id_fallback(self):
        # record_implemented stores evolution_proposal_id on the row even
        # when the result dict itself lacks it.
        res = evo_result()
        del res["proposal_id"]
        del res["commit"]
        out = render_applied_digest(applied_proposal(
            applied_result=res, evolution_proposal_id="evo_row9"))
        self.assertIn("↩️ rollback: /evolve revert evo_row9", out)

    def test_skill_edit_path_is_honest(self):
        out = render_applied_digest(applied_proposal(applied_result={
            "ok": True, "edit": "sk_1", "status": "applied",
            "skill": "my-skill"}))
        self.assertIn("✅ upgrade applied", out)
        self.assertIn("staged skill edit sk_1 committed (my-skill)", out)
        # no chat command reaches SkillEvolutionLoop.revert_edit, so the
        # digest must not invent one.
        self.assertIn("no rollback available", out)

    def test_hunks_capped(self):
        hunks = [{"path": f"f{i}.py",
                  "diff": [f"+ line{j}" for j in range(30)]}
                 for i in range(10)]
        res = evo_result(hunks=hunks)
        out = render_applied_digest(applied_proposal(applied_result=res))
        self.assertIn("📄 f0.py", out)
        self.assertIn("… +6 more files", out)
        self.assertIn("… +20 more", out)
        self.assertNotIn("📄 f4.py", out)

    def test_failure_digest_has_no_rollback_line(self):
        out = render_applied_digest(applied_proposal(
            status="failed",
            applied_result={"applied": False,
                            "reason": "verification failed"}))
        self.assertIn("❌ upgrade failed", out)
        self.assertNotIn("rollback:", out)


class TestCompactHunks(unittest.TestCase):
    """The apply path captures chat-sized hunks (evolution.py)."""

    def test_basic_diff(self):
        from nomorals.agents.evolution import _compact_hunks
        hunks = _compact_hunks([{"path": "a.py",
                                 "old": "a = 1\nb = 2\n",
                                 "new": "a = 1\nb = 3\n"}])
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]["path"], "a.py")
        joined = "\n".join(hunks[0]["diff"])
        self.assertIn("-b = 2", joined)
        self.assertIn("+b = 3", joined)

    def test_new_file(self):
        from nomorals.agents.evolution import _compact_hunks
        hunks = _compact_hunks([{"path": "n.py", "old": "",
                                 "new": "X = 1\nY = 2\n"}])
        joined = "\n".join(hunks[0]["diff"])
        self.assertIn("+ X = 1", joined)
        self.assertIn("+ Y = 2", joined)

    def test_caps(self):
        from nomorals.agents.evolution import _compact_hunks
        edits = [{"path": f"f{i}.py", "old": "",
                  "new": "\n".join(f"L{j}" for j in range(50))}
                 for i in range(10)]
        hunks = _compact_hunks(edits)
        self.assertEqual(len(hunks), 6)
        self.assertLessEqual(len(hunks[0]["diff"]), 13)
        self.assertTrue(hunks[0]["diff"][-1].startswith("… +"))

    def test_applied_summary_carries_hunks(self):
        from nomorals.agents.evolution import (
            EvolutionAgent, EvolutionProposal)
        agent = EvolutionAgent.__new__(EvolutionAgent)
        prop = EvolutionProposal(
            id="evo_t", instruction="tweak",
            edits=[{"path": "a.py", "old": "a = 1\n", "new": "a = 2\n"}])
        out = agent._applied_summary(prop, False)
        self.assertTrue(out["applied"])
        self.assertEqual(out["proposal"], "evo_t")
        self.assertEqual(out["edits"], ["a.py"])
        self.assertEqual(len(out["hunks"]), 1)
        self.assertIn("-a = 1", "\n".join(out["hunks"][0]["diff"]))


if __name__ == "__main__":
    unittest.main()
