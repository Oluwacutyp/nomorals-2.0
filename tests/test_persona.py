"""Prompt 11 — persona depth: memory that deepens into a person-shaped model.

Covers: UserModel rebuild (identity/preferences/interests/routines/goals),
routine proposal at 5+ episodes with confirmation-before-acting, PeopleGraph,
PersonaGuide (≤10 lines, no raw episodes, no private records), ProactiveRecall
anti-creep rules, MemoryCurator (dedup, contradiction→superseded_by + one
gentle note, archive + 30-day undo), private-flag exclusion from recall and
training, owner-controlled memory with no content refusals, the correction
path ("forget that"), and the `nm memory` CLI surface.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.memory.base import MemoryKind
from nomorals.memory.persona import (
    MemoryCurator,
    PeopleGraph,
    PersonaGuide,
    ProactiveRecall,
    UserModel,
)


def temp_dir():
    return tempfile.mkdtemp(prefix="persona-test-")


class PersonaTestBase(unittest.TestCase):
    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.memory = self.context.memory

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def remember(self, content, kind=MemoryKind.EPISODE, **kw):
        return self.memory.remember(content, kind=kind, source="test", **kw)


# ── 1. routine detection: 5+ episodes → proposed, confirmation required ──

class RoutineDetectionTests(PersonaTestBase):
    def _seed_market_checks(self, n):
        for i in range(n):
            self.remember(
                f"morning market check {i}: asked for BTC and ETH prices "
                f"with a plain-language brief")

    def test_five_episodes_propose_a_routine_with_evidence(self):
        self._seed_market_checks(6)
        model = UserModel.rebuild(self.memory)
        self.assertTrue(model.routines, "expected at least one routine")
        routine = model.routines[0]
        self.assertGreaterEqual(routine.supporting_episodes, 5)
        self.assertEqual(routine.status, "established")
        self.assertTrue(routine.evidence, "routine must carry evidence ids")

    def test_two_episodes_only_propose_never_assert(self):
        self._seed_market_checks(2)
        model = UserModel.rebuild(self.memory)
        if model.routines:
            self.assertEqual(model.routines[0].status, "proposed")

    def test_acting_before_confirmation_is_refused(self):
        # the model proposes; nothing auto-executes.  ProactiveRecall only
        # produces *hints* for the agent's context — read-only.
        self._seed_market_checks(3)
        model = UserModel.rebuild(self.memory)
        recall = ProactiveRecall(model)
        hints = recall.anticipate("check BTC")
        # proposed (not established) routines never surface as hints
        self.assertEqual(hints, [])

    def test_established_routine_surfaces_as_hint_not_action(self):
        self._seed_market_checks(6)
        model = UserModel.rebuild(self.memory)
        recall = ProactiveRecall(model)
        hints = recall.anticipate("check BTC price this morning")
        self.assertTrue(hints, "established routine should hint")
        # hints are synthesized descriptions, never raw episode text
        for h in hints:
            self.assertNotIn("morning market check 0", h)


# ── 2. user model synthesis ────────────────────────────────────────────

class UserModelTests(PersonaTestBase):
    def test_identity_from_explicit_facts_only(self):
        self.remember("my name is death", kind=MemoryKind.FACT)
        self.remember("my timezone is Africa/Lagos", kind=MemoryKind.FACT)
        model = UserModel.rebuild(self.memory)
        self.assertIn("name", model.identity)
        self.assertIn("timezone", model.identity)
        self.assertEqual(model.identity["name"].value, "death")
        self.assertGreaterEqual(model.identity["name"].confidence, 0.8)
        self.assertTrue(model.identity["name"].evidence)

    def test_preferences_carry_confidence_and_evidence(self):
        rid = self.remember("I prefer short, direct answers with numbers first",
                            kind=MemoryKind.PREFERENCE, importance=0.9)
        model = UserModel.rebuild(self.memory)
        self.assertTrue(model.preferences)
        pref = model.preferences[0]
        self.assertIn(rid, pref.evidence)
        self.assertGreaterEqual(pref.confidence, 0.45)
        self.assertTrue(pref.actionable)

    def test_interests_have_heat_scores(self):
        for i in range(4):
            self.remember(f"working on the devon trading bot, sentinel "
                           f"strategy {i}")
        self.remember("bought yams at the market")
        model = UserModel.rebuild(self.memory)
        topics = {i.topic: i for i in model.interests}
        self.assertIn("trading", topics)
        self.assertGreater(topics["trading"].heat, 0)
        self.assertLessEqual(topics["trading"].heat, 1.0)

    def test_goals_in_flight_from_decisions(self):
        self.remember("decided to ship the Devon upgrade week this Friday",
                      kind=MemoryKind.DECISION)
        model = UserModel.rebuild(self.memory)
        self.assertTrue(model.goals_in_flight)
        self.assertIn("ship", model.goals_in_flight[0].value.lower())

    def test_every_attribute_has_confidence_and_evidence(self):
        self.remember("my name is death", kind=MemoryKind.FACT)
        self.remember("I like terse replies", kind=MemoryKind.PREFERENCE)
        model = UserModel.rebuild(self.memory)
        for attr in list(model.identity.values()) + model.preferences:
            self.assertGreaterEqual(attr.confidence, 0.0)
            self.assertLessEqual(attr.confidence, 1.0)
            self.assertTrue(attr.evidence)


# ── 3. contradiction handling ──────────────────────────────────────────

class ContradictionTests(PersonaTestBase):
    def test_contradiction_supersedes_without_deleting(self):
        old_id = self.remember("I prefer verbose explanations",
                               kind=MemoryKind.PREFERENCE, importance=0.8)
        new_id = self.remember("I do not prefer verbose explanations, "
                               "keep them terse",
                               kind=MemoryKind.PREFERENCE, importance=0.8)
        curator = MemoryCurator(self.memory)
        report = curator.check_contradiction(new_id)
        self.assertEqual(report.get("superseded"), old_id)
        self.assertEqual(report.get("by"), new_id)
        self.assertIn("note", report, "owner gets one gentle note")
        # old record still exists, linked — not deleted
        old = self.memory.get(old_id)
        self.assertIsNotNone(old)
        self.assertEqual(old.metadata.get("superseded_by"), new_id)
        self.assertIsNotNone(self.memory.get(new_id))

    def test_no_nag_loop_on_repeat_check(self):
        old_id = self.remember("I prefer verbose explanations",
                               kind=MemoryKind.PREFERENCE)
        new_id = self.remember("I do not prefer verbose explanations",
                               kind=MemoryKind.PREFERENCE)
        curator = MemoryCurator(self.memory)
        first = curator.check_contradiction(new_id)
        self.assertIn("note", first)
        second = curator.check_contradiction(new_id)
        self.assertNotIn("note", second,
                         "already-superseded records must not re-note")

    def test_superseded_decisions_leave_goals_in_flight(self):
        old_id = self.remember("decided to learn Go", kind=MemoryKind.DECISION)
        new_id = self.remember("decided to not learn Go, sticking with Python",
                               kind=MemoryKind.DECISION)
        # decisions don't go through check_contradiction (facts/prefs only)
        curator = MemoryCurator(self.memory)
        self.assertEqual(curator.check_contradiction(new_id), {})
        self.assertIsNotNone(self.memory.get(old_id))


# ── 4. correction path: "forget that" ─────────────────────────────────

class CorrectionPathTests(PersonaTestBase):
    def test_forget_removes_and_confirms(self):
        rid = self.remember("I like pineapple on pizza",
                            kind=MemoryKind.PREFERENCE)
        curator = MemoryCurator(self.memory)
        curator._archive(rid, reason="test-forget")
        n = self.memory.forget(rid)
        self.assertEqual(n, 1)
        self.assertIsNone(self.memory.get(rid))
        # archived for 30-day undo
        rows = self.memory.db.query(
            "SELECT id FROM memory_archive WHERE id = ?", (rid,))
        self.assertTrue(rows)

    def test_forgotten_record_leaves_proactive_recall(self):
        rid = self.remember("I like pineapple on pizza",
                            kind=MemoryKind.PREFERENCE, importance=0.9)
        model = UserModel.rebuild(self.memory)
        self.assertTrue(any("pineapple" in p.value
                            for p in model.preferences))
        self.memory.forget(rid)
        model2 = UserModel.rebuild(self.memory)
        self.assertFalse(any("pineapple" in p.value
                             for p in model2.preferences))

    def test_update_correction_path(self):
        rid = self.remember("my name is deat", kind=MemoryKind.FACT)
        n = self.memory.update(rid, content="my name is death")
        self.assertEqual(n, 1)
        self.assertEqual(self.memory.get(rid).content, "my name is death")

    def test_archive_restore_within_30_days(self):
        rid = self.remember("temporary note", kind=MemoryKind.EPISODE)
        curator = MemoryCurator(self.memory)
        curator._archive(rid, reason="test")
        self.memory.forget(rid)
        self.assertTrue(curator.restore(rid))
        found = self.memory.recall("temporary note", limit=5)
        self.assertTrue(any("temporary note" in r.content
                            for r in found.records))


# ── 5. owner memory: no content-based refusals ─────────────────────────
# The owner decides what is remembered.  Sensitive attributes store exactly
# like any other fact; the private-metadata flag (not a refusal) is the
# tool for keeping records out of model context.

class OwnerMemoryTests(PersonaTestBase):
    BAITS = [
        ("I was diagnosed with diabetes last year", MemoryKind.FACT),
        ("I voted for the Labour party", MemoryKind.FACT),
        ("I am a practicing Muslim", MemoryKind.FACT),
        ("I prefer dating men", MemoryKind.PREFERENCE),
        ("my therapist says I have anxiety", MemoryKind.EPISODE),
    ]

    def test_sensitive_facts_store_like_any_other(self):
        for text, kind in self.BAITS:
            rid = self.memory.remember(text, kind=kind, source="test")
            self.assertTrue(rid, f"must store, not refuse: {text[:40]}")
        # stored records are retrievable through normal recall
        blob = " ".join(
            r.content for r in self.memory.recall("", limit=50)).lower()
        for needle in ("diabetes", "labour", "muslim", "therapist",
                       "dating men"):
            self.assertIn(needle, blob)
        # and the user model reflects what it models (preferences here)
        model = UserModel.rebuild(self.memory)
        self.assertIn("dating men",
                      json.dumps(model.to_dict()).lower())

    def test_private_flag_still_excludes_from_context(self):
        rid = self.memory.remember("my secret project", kind=MemoryKind.FACT,
                                   source="test")
        self.memory.mark_private(rid)
        model = UserModel.rebuild(self.memory)
        blob = json.dumps(model.to_dict()).lower()
        self.assertNotIn("secret project", blob)

    def test_non_sensitive_facts_still_store(self):
        rid = self.remember("my name is death", kind=MemoryKind.FACT)
        self.assertTrue(rid)


# ── 6. private flag: recall + training exclusion ──────────────────────

class PrivateFlagTests(PersonaTestBase):
    def test_private_excluded_from_recall_by_default(self):
        pub = self.remember("public note about trading")
        priv = self.remember("private note about my salary")
        self.memory.mark_private(priv)
        results = self.memory.recall("note", limit=10)
        ids = {r.id for r in results.records}
        self.assertIn(pub, ids)
        self.assertNotIn(priv, ids)

    def test_private_visible_with_explicit_opt_in(self):
        priv = self.remember("private note about my salary")
        self.memory.mark_private(priv)
        results = self.memory.recall("note", limit=10, include_private=True)
        self.assertIn(priv, {r.id for r in results.records})

    def test_private_excluded_from_build_context(self):
        self.remember("private note about my salary")
        priv = self.remember("another private secret")
        self.memory.mark_private(priv)
        ctx = self.memory.build_context("salary")
        self.assertNotIn("another private secret", ctx)

    def test_private_excluded_from_training_pipeline(self):
        pub = self.remember("public trading lesson", kind=MemoryKind.LESSON)
        priv = self.remember("private salary lesson", kind=MemoryKind.LESSON)
        self.memory.mark_private(priv)
        corpus = self.memory.for_training()
        ids = {r.id for r in corpus}
        self.assertIn(pub, ids)
        self.assertNotIn(priv, ids,
                         "training pipelines must never see private records")

    def test_private_excluded_from_proactive_recall(self):
        self.remember("I prefer private briefings", kind=MemoryKind.PREFERENCE,
                      importance=0.95)
        priv = self.remember("I prefer verbose dumps", kind=MemoryKind.PREFERENCE,
                             importance=0.95)
        self.memory.mark_private(priv)
        model = UserModel.rebuild(self.memory)
        values = " ".join(p.value for p in model.preferences)
        self.assertIn("private briefings", values)
        self.assertNotIn("verbose dumps", values)

    def test_mark_public_clears_flag(self):
        priv = self.remember("temporarily private")
        self.memory.mark_private(priv)
        self.assertEqual(self.memory.mark_public(priv), 1)
        results = self.memory.recall("temporarily", limit=5)
        self.assertIn(priv, {r.id for r in results.records})


# ── 7+8. PersonaGuide discipline ───────────────────────────────────────

class PersonaGuideTests(PersonaTestBase):
    def test_guide_capped_at_ten_lines(self):
        for i in range(8):
            self.remember(f"I prefer style number {i}",
                          kind=MemoryKind.PREFERENCE, importance=0.9)
        model = UserModel.rebuild(self.memory)
        guide = PersonaGuide.build(model)
        self.assertLessEqual(len(guide.lines), 10)

    def test_guide_contains_no_raw_episodes_or_private(self):
        self.remember("yesterday I bought 3 yams and argued with the seller",
                      kind=MemoryKind.EPISODE)
        priv = self.remember("I prefer verbose dumps", kind=MemoryKind.PREFERENCE,
                             importance=0.95)
        self.memory.mark_private(priv)
        self.remember("my name is death", kind=MemoryKind.FACT)
        model = UserModel.rebuild(self.memory)
        guide = PersonaGuide.build(model)
        text = guide.text().lower()
        self.assertNotIn("yams", text, "no raw episode content in guide")
        self.assertNotIn("verbose dumps", text, "no private records in guide")
        self.assertIn("death", text)

    def test_guide_empty_model_has_fallback(self):
        guide = PersonaGuide.build(UserModel())
        self.assertTrue(guide.lines)
        self.assertLessEqual(len(guide.lines), 10)


# ── people graph ───────────────────────────────────────────────────────

class PeopleGraphTests(PersonaTestBase):
    def test_graph_builds_from_relationship_records(self):
        self.remember("Ada is my sister, she runs a boutique in Lagos",
                      kind=MemoryKind.RELATIONSHIP,
                      metadata={"person": "Ada", "role": "sister"})
        self.remember("Ada recommended the Jumia vendor",
                      kind=MemoryKind.RELATIONSHIP,
                      metadata={"person": "Ada"})
        graph = PeopleGraph.build(self.memory)
        entry = graph.lookup("Ada")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.role, "sister")
        self.assertEqual(entry.mention_count, 2)
        self.assertTrue(entry.evidence)

    def test_private_relationships_excluded(self):
        priv = self.remember("X is my secret contact",
                             kind=MemoryKind.RELATIONSHIP,
                             metadata={"person": "X"})
        self.memory.mark_private(priv)
        graph = PeopleGraph.build(self.memory)
        self.assertIsNone(graph.lookup("X"))

    def test_subgraph_is_synthesized_not_raw(self):
        self.remember("Ada is my sister", kind=MemoryKind.RELATIONSHIP,
                      metadata={"person": "Ada", "role": "sister"})
        graph = PeopleGraph.build(self.memory)
        sub = graph.subgraph("Ada")
        self.assertEqual(sub["role"], "sister")
        self.assertIn("mention_count", sub)


# ── proactive recall anti-creep ────────────────────────────────────────

class ProactiveRecallTests(PersonaTestBase):
    def test_never_surfaces_raw_episode_text(self):
        self.remember("on Tuesday I lost 50000 naira on a bad BTC trade",
                      kind=MemoryKind.EPISODE)
        model = UserModel.rebuild(self.memory)
        recall = ProactiveRecall(model)
        block = recall.for_context_block("check BTC")
        self.assertNotIn("50000", block)
        self.assertNotIn("Tuesday", block)

    def test_never_surfaces_below_confidence_floor(self):
        self.remember("I might like blue", kind=MemoryKind.PREFERENCE,
                      importance=0.0)
        model = UserModel.rebuild(self.memory)
        low = [p for p in model.preferences if not p.actionable]
        self.assertTrue(low, "test needs a low-confidence preference")
        recall = ProactiveRecall(model)
        hints = recall.anticipate("blue")
        self.assertFalse(any("might like blue" in h for h in hints))

    def test_read_only_over_memory(self):
        self.remember("I prefer terse replies", kind=MemoryKind.PREFERENCE,
                      importance=0.9)
        before = self.memory.counts_by_kind()
        model = UserModel.rebuild(self.memory)
        ProactiveRecall(model).anticipate("reply tersely")
        after = self.memory.counts_by_kind()
        self.assertEqual(before, after, "anticipation must not write")


# ── curation: dedup + scheduled pass ───────────────────────────────────

class CurationTests(PersonaTestBase):
    def test_dedup_merges_near_duplicates_keeping_best(self):
        a = self.remember("I prefer short answers", kind=MemoryKind.PREFERENCE,
                          importance=0.9)
        b = self.remember("I prefer short answers!", kind=MemoryKind.PREFERENCE,
                          importance=0.4)
        report = MemoryCurator(self.memory).deduplicate(
            kind=MemoryKind.PREFERENCE)
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["merged"], 1)
        # highest-confidence (a) survives
        self.assertIsNotNone(self.memory.get(a))
        self.assertIsNone(self.memory.get(b))
        # merged id linked on the survivor
        survivor = self.memory.get(a)
        self.assertIn(b, survivor.metadata.get("merged_ids", []))
        # loser archived, not vaporized
        rows = self.memory.db.query(
            "SELECT id FROM memory_archive WHERE id = ?", (b,))
        self.assertTrue(rows)

    def test_scheduled_pass_archives_before_forgetting(self):
        rid = self.remember("cruft", kind=MemoryKind.EPISODE, importance=0.01)
        # force age so recency * importance < threshold
        self.memory.db.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (time.time() - 90 * 86400, rid))
        report = MemoryCurator(self.memory).scheduled_pass()
        self.assertGreaterEqual(report["forgotten"], 1)
        rows = self.memory.db.query(
            "SELECT id FROM memory_archive WHERE id = ?", (rid,))
        self.assertTrue(rows, "forgotten records must be archived first")

    def test_facts_never_auto_forgotten(self):
        rid = self.remember("my name is death", kind=MemoryKind.FACT)
        self.memory.db.execute(
            "UPDATE memories SET created_at = ?, importance = 0.01 "
            "WHERE id = ?", (time.time() - 90 * 86400, rid))
        MemoryCurator(self.memory).scheduled_pass()
        self.assertIsNotNone(self.memory.get(rid))


# ── CLI surface ────────────────────────────────────────────────────────

class CLITests(PersonaTestBase):
    def _ns(self, **kw):
        base = dict(json=False, memory_action=None, query="", kind="",
                    limit=20, include_private=False, record_id="",
                    text="", yes=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def _run(self, **kw):
        from nomorals.cli import _cmd_memory_action
        return _cmd_memory_action(self._ns(**kw), self.context)

    def test_show_renders_confidence_and_evidence(self):
        self.remember("my name is death", kind=MemoryKind.FACT)
        self.remember("I prefer terse replies", kind=MemoryKind.PREFERENCE,
                      importance=0.9)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run(memory_action="show")
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("confidence", out)
        self.assertIn("death", out)

    def test_list_and_forget(self):
        rid = self.remember("I like pineapple", kind=MemoryKind.PREFERENCE)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._run(memory_action="list", kind="preference")
        self.assertIn("pineapple", buf.getvalue())
        self.assertEqual(self._run(memory_action="forget", record_id=rid), 0)
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            self._run(memory_action="list", kind="preference")
        self.assertNotIn("pineapple", buf2.getvalue())

    def test_forget_kind_requires_confirmation(self):
        self.remember("ephemeral", kind=MemoryKind.EPISODE)
        rc = self._run(memory_action="forget-kind", kind="episode")
        self.assertEqual(rc, 2, "must refuse without --yes")
        self.assertEqual(
            self._run(memory_action="forget-kind", kind="episode", yes=True),
            0)

    def test_private_and_export(self):
        rid = self.remember("salary talk", kind=MemoryKind.EPISODE)
        self.assertEqual(
            self._run(memory_action="private", record_id=rid), 0)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._run(memory_action="export")
        dump = json.loads(buf.getvalue())
        self.assertIn("user_model", dump)
        self.assertIn("people", dump)
        self.assertIn("records", dump)
        rec = next(r for r in dump["records"] if r["id"] == rid)
        self.assertTrue(rec["private"])


if __name__ == "__main__":
    unittest.main()
