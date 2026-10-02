"""Wave G1 — ULID/short-prefix resolution hardening.

Every resolver that accepts a short id prefix follows one contract:

* empty / whitespace-only reference → never match-all
* exact full-id match → resolves, even when it is also a prefix of
  another id (exact wins)
* prefix matching exactly one entity → resolves
* prefix matching zero entities → not-found
* prefix matching 2+ entities → ambiguity error listing the candidates
  (id + human label) and the minimum id-prefix length that
  disambiguates them — the resolver NEVER picks one

Covers the shared helper (``nomorals.core.ids.resolve_id_prefix``) and
the resolvers in each domain:

* upgrade proposals — ``nomorals.agents.upgrade_chat.resolve_proposal``
* missions — ``PartnerRuntime._find_mission``
* research proposals — ``ResearchAgent.resolve_proposal_detailed``
  (the research → approve → evolve loop feeds the upgrade queue, and its
  old ``LIKE ref%`` + ``query_one`` silently took the first match)

Game sessions were surveyed: the game engine resolves rooms by exact
``chat_key`` (``GameEngine.live``), invites by exact code
(``GameRelay``), and the engine's ``_by_id`` room index is write-only —
there is no id-prefix resolver in the games domain to harden.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.core.errors import AmbiguousRef
from nomorals.core.ids import (
    min_unique_prefix_len,
    resolve_id_prefix,
)


# ── shared helper ────────────────────────────────────────────────────────────

class TestResolveIdPrefixHelper(unittest.TestCase):
    def test_empty_never_matches(self):
        for ref in ("", "   ", "\t\n"):
            res = resolve_id_prefix(ref, ["abc", "abd"])
            self.assertEqual(res.outcome, "empty")
            self.assertEqual(res.matches, ())

    def test_no_match(self):
        res = resolve_id_prefix("zzz", ["abc", "abd"])
        self.assertEqual(res.outcome, "none")
        self.assertEqual(res.matches, ())

    def test_unique_prefix(self):
        res = resolve_id_prefix("ab", ["abc", "zzz"])
        self.assertEqual(res.outcome, "unique")
        self.assertEqual(res.matches, ("abc",))

    def test_exact_wins_even_when_prefix_of_another(self):
        # "abc" is a full id AND a prefix of "abcdef" — exact must win,
        # not report ambiguity
        res = resolve_id_prefix("abc", ["abcdef", "abc", "zzz"])
        self.assertEqual(res.outcome, "exact")
        self.assertEqual(res.matches, ("abc",))

    def test_ambiguous_never_picks(self):
        res = resolve_id_prefix("ab", ["abc", "abd", "zzz"])
        self.assertEqual(res.outcome, "ambiguous")
        self.assertEqual(set(res.matches), {"abc", "abd"})
        self.assertGreater(res.min_unique_len, len("ab"))

    def test_case_insensitive(self):
        res = resolve_id_prefix("ab", ["ABC", "zzz"])
        self.assertEqual(res.outcome, "unique")
        self.assertEqual(res.matches, ("ABC",))
        res = resolve_id_prefix("ABC", ["abcdef", "abc"])
        self.assertEqual(res.outcome, "exact")
        self.assertEqual(res.matches, ("abc",))

    def test_duplicate_ids_deduped(self):
        res = resolve_id_prefix("ab", ["abc", "abc"])
        self.assertEqual(res.outcome, "unique")

    def test_min_unique_prefix_len(self):
        self.assertEqual(min_unique_prefix_len(["abcdef", "abcxyz"]), 4)
        self.assertEqual(min_unique_prefix_len(["abc", "abcdef"]), 4)
        self.assertEqual(min_unique_prefix_len(["only"]), 0)
        self.assertEqual(min_unique_prefix_len([]), 0)
        # every candidate's first-4 chars are unique here
        ids = ["abcdef", "abcxyz"]
        n = min_unique_prefix_len(ids)
        shorts = [i[:n].lower() for i in ids]
        self.assertEqual(len(set(shorts)), len(ids))


# ── upgrade proposals ────────────────────────────────────────────────────────

def _make_upgrade_ctx():
    from nomorals.storage.db import Database
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db)


def _insert_proposal(ctx, pid, title, status="proposed"):
    ctx.db.execute(
        "INSERT INTO upgrade_proposals (id, title, rationale, patch_plan, "
        "files, tests, claim_ids, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, title, "rationale " * 10, json.dumps({"steps": ["s1"]}),
         json.dumps([]), json.dumps([]), json.dumps([]), status,
         time.time()),
    )


class TestUpgradeProposalPrefix(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.upgrade_chat import resolve_proposal
        from nomorals.agents.upgrade_queue import UpgradeQueue
        self.resolve_proposal = resolve_proposal
        self.ctx = _make_upgrade_ctx()
        self.queue = UpgradeQueue(self.ctx)
        # two proposals sharing the id prefix "upg_AAAX", plus an unrelated one
        _insert_proposal(self.ctx, "upg_AAAX1", "First colliding proposal title")
        _insert_proposal(self.ctx, "upg_AAAX2", "Second colliding proposal title")
        _insert_proposal(self.ctx, "upg_BBB9", "Unrelated proposal title here")

    def test_exact_id(self):
        p, err = self.resolve_proposal(self.queue, "upg_AAAX1")
        self.assertEqual(err, "")
        self.assertEqual(p["id"], "upg_AAAX1")

    def test_exact_wins_when_prefix_of_another(self):
        # craft: full id "upg_EX" is also a prefix of "upg_EX9"
        _insert_proposal(self.ctx, "upg_EX", "Short exact proposal title")
        _insert_proposal(self.ctx, "upg_EX9", "Longer proposal title here")
        p, err = self.resolve_proposal(self.queue, "upg_EX")
        self.assertEqual(err, "")
        self.assertEqual(p["id"], "upg_EX")

    def test_unique_prefix(self):
        p, err = self.resolve_proposal(self.queue, "upg_BBB")
        self.assertEqual(err, "")
        self.assertEqual(p["id"], "upg_BBB9")

    def test_ambiguous_prefix_lists_candidates_and_min_length(self):
        p, err = self.resolve_proposal(self.queue, "upg_AAAX")
        self.assertIsNone(p)
        self.assertIn("ambiguous", err)
        self.assertIn("upg_AAAX1", err)
        self.assertIn("upg_AAAX2", err)
        self.assertNotIn("upg_BBB9", err)
        self.assertIn("at least 9 characters", err)

    def test_no_match(self):
        p, err = self.resolve_proposal(self.queue, "upg_nope")
        self.assertIsNone(p)
        self.assertIn("no upgrade proposal", err)

    def test_empty(self):
        p, err = self.resolve_proposal(self.queue, "   ")
        self.assertIsNone(p)
        self.assertIn("/upgrade list", err)

    def test_approve_never_attaches_to_wrong_proposal(self):
        """Regression: an ambiguous prefix on /upgrade approve must not
        approve the first match — the pipeline must see nothing."""
        from nomorals.agents.partner_runtime import PartnerRuntime

        approved = []

        class Pipe:
            def __init__(self, ctx):
                from nomorals.agents.upgrade_queue import UpgradeQueue
                self.queue = UpgradeQueue(ctx)

            def approve_and_implement(self, pid, by="owner"):
                approved.append(pid)
                return self.queue.record_implemented(
                    pid, {"applied": True, "edits": [], "verified": True})

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = self.ctx
        out = rt._control_upgrade("approve upg_AAAX", _pipeline=Pipe(self.ctx))
        self.assertIn("ambiguous", out)
        self.assertEqual(approved, [])
        # both proposals are still pending — nothing was attached anywhere
        for pid in ("upg_AAAX1", "upg_AAAX2"):
            row = self.ctx.db.query_one(
                "SELECT status FROM upgrade_proposals WHERE id = ?", (pid,))
            self.assertEqual(row["status"], "proposed")


# ── missions ─────────────────────────────────────────────────────────────────

def _make_mission_ctx(test=None):
    from nomorals.storage.db import Database
    tmp = tempfile.mkdtemp(prefix="mission-prefix-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=db, settings=settings, extras={}), tmp


def _make_mission_with_id(store, pid, name, goal="test goal"):
    from nomorals.missions import Mission
    m = Mission(goal=goal, name=name, id=pid)
    store.create(m)
    return m


class _MissionRuntimeStub:
    """Drives _control_mission without a full PartnerRuntime."""

    def __init__(self, ctx):
        from nomorals.agents.partner_runtime import PartnerRuntime
        self.context = ctx
        self._control_mission = PartnerRuntime._control_mission.__get__(self)
        self._find_mission = PartnerRuntime._find_mission


class TestMissionPrefix(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.missions import MissionStore
        self.ctx, _ = _make_mission_ctx(self)
        self.store = MissionStore(self.ctx.db)
        self._find_mission = PartnerRuntime._find_mission
        # two missions sharing the id prefix "01AAAX", plus an unrelated one
        _make_mission_with_id(self.store, "01AAAX1000000000000000001",
                              "mission alpha")
        _make_mission_with_id(self.store, "01AAAX2000000000000000002",
                              "mission beta")
        _make_mission_with_id(self.store, "01ZZZ90000000000000000003",
                              "mission gamma")

    def test_exact_id(self):
        m = self._find_mission(self.store, "01AAAX1000000000000000001")
        self.assertEqual(m.name, "mission alpha")

    def test_exact_wins_when_prefix_of_another(self):
        _make_mission_with_id(self.store, "01EX", "short id mission")
        _make_mission_with_id(self.store, "01EX900000000000000000009",
                              "longer id mission")
        m = self._find_mission(self.store, "01EX")
        self.assertEqual(m.name, "short id mission")

    def test_unique_prefix(self):
        m = self._find_mission(self.store, "01ZZZ9")
        self.assertEqual(m.name, "mission gamma")

    def test_no_match_returns_none(self):
        self.assertIsNone(self._find_mission(self.store, "01NOPE"))

    def test_empty_returns_default_active(self):
        # documented [id|name] UX: /mission status with no ref shows the
        # active mission — preserved, and it never goes through prefix
        # matching
        m = self._find_mission(self.store, "   ")
        self.assertIsNotNone(m)

    def test_ambiguous_prefix_raises_with_candidates(self):
        with self.assertRaises(AmbiguousRef) as cm:
            self._find_mission(self.store, "01AAAX")
        exc = cm.exception
        ids = [c[0] for c in exc.candidates]
        self.assertIn("01AAAX1000000000000000001", ids)
        self.assertIn("01AAAX2000000000000000002", ids)
        self.assertNotIn("01ZZZ90000000000000000003", ids)
        self.assertGreater(exc.min_prefix_len, len("01AAAX"))
        text = str(exc)
        self.assertIn("ambiguous", text)
        self.assertIn("mission alpha", text)
        self.assertIn(str(exc.min_prefix_len), text)

    def test_ambiguous_name_matches_raise(self):
        # name/goal substring fallback must not silently take the first
        # match either
        _make_mission_with_id(self.store, "01Q1AAAAAAAAAAAAAAAAAAAAA",
                              "deploy the thing")
        _make_mission_with_id(self.store, "01Q2BBBBBBBBBBBBBBBBBBBBB",
                              "deploy the other thing")
        with self.assertRaises(AmbiguousRef):
            self._find_mission(self.store, "deploy the")

    def test_pause_never_attaches_to_wrong_mission(self):
        """Regression: /mission pause with an ambiguous prefix must pause
        nothing — previously the first prefix match was paused silently."""
        rt = _MissionRuntimeStub(self.ctx)
        out = rt._control_mission("pause 01AAAX", chat_key="telegram:9")
        self.assertIn("ambiguous", out)
        for pid in ("01AAAX1000000000000000001", "01AAAX2000000000000000002"):
            m = self.store.get(pid)
            self.assertNotEqual(m.status, "paused", pid)


# ── research proposals ───────────────────────────────────────────────────────

def _insert_research(ctx, rid, topic, status="pending", created_at=None):
    ctx.db.execute(
        "INSERT INTO research_log (id, domain, topic, digest, suggestion, "
        "sources, delivered, created_at, score, score_detail, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (rid, "tech", topic, "digest", "suggestion " * 10, "[]", 0,
         created_at if created_at is not None else time.time(),
         0.9, "{}", status),
    )


class TestResearchProposalPrefix(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.researcher import ResearchAgent
        self.ctx = _make_upgrade_ctx()
        self.agent = ResearchAgent(self.ctx)
        _insert_research(self.ctx, "01RAAAX1000000000000000001",
                         "First colliding research topic", created_at=1000.0)
        _insert_research(self.ctx, "01RAAAX2000000000000000002",
                         "Second colliding research topic", created_at=2000.0)
        _insert_research(self.ctx, "01RZZZ90000000000000000003",
                         "Unrelated research topic here", created_at=3000.0)

    def test_exact_id(self):
        row = self.agent.resolve_proposal("01RAAAX1000000000000000001")
        self.assertEqual(row["topic"], "First colliding research topic")

    def test_exact_wins_when_prefix_of_another(self):
        _insert_research(self.ctx, "01REX", "Short exact research topic")
        _insert_research(self.ctx, "01REX900000000000000000009",
                         "Longer research topic here")
        row = self.agent.resolve_proposal("01REX")
        self.assertEqual(row["topic"], "Short exact research topic")

    def test_unique_prefix(self):
        row = self.agent.resolve_proposal("01RZZZ9")
        self.assertEqual(row["topic"], "Unrelated research topic here")

    def test_latest_keyword(self):
        row = self.agent.resolve_proposal("latest")
        self.assertEqual(row["id"], "01RZZZ90000000000000000003")

    def test_no_match(self):
        self.assertIsNone(self.agent.resolve_proposal("01RNOPE"))

    def test_empty(self):
        self.assertIsNone(self.agent.resolve_proposal("   "))

    def test_ambiguous_detailed_lists_candidates_and_min_length(self):
        row, err = self.agent.resolve_proposal_detailed("01RAAAX")
        self.assertIsNone(row)
        self.assertIn("ambiguous", err)
        self.assertIn("01RAAAX1000000000000000001", err)
        self.assertIn("01RAAAX2000000000000000002", err)
        self.assertNotIn("01RZZZ90000000000000000003", err)
        self.assertIn("characters", err)

    def test_like_wildcards_matched_literally(self):
        # "_" must not act as a single-char wildcard and attach to a
        # different row
        _insert_research(self.ctx, "01W_AX10000000000000000001",
                         "Underscore research topic here")
        row, err = self.agent.resolve_proposal_detailed("01W%AX1")
        self.assertIsNone(row)
        self.assertIn("no such proposal", err)

    def test_approve_never_attaches_to_wrong_proposal(self):
        """Regression: /research approve with an ambiguous prefix must not
        approve the first match — the old LIKE+query_one did exactly that."""
        out = self.agent.approve("01RAAAX")
        self.assertIn("ambiguous", out)
        for rid in ("01RAAAX1000000000000000001", "01RAAAX2000000000000000002"):
            row = self.ctx.db.query_one(
                "SELECT status FROM research_log WHERE id = ?", (rid,))
            self.assertEqual(row["status"], "pending")


if __name__ == "__main__":
    unittest.main()
