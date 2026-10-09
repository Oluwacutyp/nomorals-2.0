"""Phase 2 Section 4 — memory deep improvement.

Covers the new systems layered on the trusted recall path:
- consolidation cadence (scheduled, observable, additive)
- memory scopes (no cross-space leakage)
- contradiction strategy chain (detect + additive resolve)
- deep recall (multi-hop, associative, temporal)
- backup / restore / import
- native-first embeddings + profile gating

The trusted scoring math is never touched here — the parity suite in
test_memory_health.py guards it, and these tests assert the new layers
degrade to (never replace) the trusted path.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.memory.base import MemoryRecord, score_memory
from nomorals.memory.embeddings import Embedder, select_for_profile


def temp_dir() -> str:
    return tempfile.mkdtemp(prefix="nm-memdeep-")


class Harness(unittest.TestCase):
    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        # Deterministic learning-worker shutdown (runs before the context
        # exit above — cleanups run LIFO).  The llm/learning.py daemon
        # thread segfaults intermittently when it is mid-DB-write during
        # interpreter teardown (pre-existing race in that subsystem, not
        # in memory/): flush the queue and detach the worker here so no
        # background write is in flight when the process winds down.
        self.addCleanup(self._stop_learning_worker)
        self.memory = self.context.memory
        self.memory.embedder = Embedder(provider="hashing", dimensions=512)
        self.memory.consolidate_every = 10 ** 9

    def _stop_learning_worker(self) -> None:
        try:
            router = getattr(self.context, "router", None)
            learning = getattr(router, "_learning", None)
            if learning is None:
                return
            try:
                learning.flush(timeout=10.0)
            except Exception:  # noqa: BLE001
                pass
            learning.detach()
        except Exception:  # noqa: BLE001
            pass

    def remember(self, content, kind="fact", **kw):
        kw.setdefault("source", "user:test")
        return self.memory.remember(content, kind=kind, **kw)

    def backdate(self, record_id: str, created_at: float) -> None:
        self.memory.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (created_at, created_at, record_id))


# ── scopes ─────────────────────────────────────────────────────────────────

class ScopeTests(Harness):
    def test_scoped_write_and_recall(self):
        gid = self.remember("the sky is blue")
        aid = self.remember("arena boss is weak to fire", scope="project:devon-arena")
        hid = self.remember("buy yams", scope="project:home")

        arena = self.memory.recall("boss fire", limit=10,
                                   scope="project:devon-arena")
        ids = {r.id for r in arena.records}
        self.assertIn(aid, ids)   # own space visible
        self.assertIn(gid, ids)   # global visible everywhere
        self.assertNotIn(hid, ids)  # other space never leaks

    def test_global_recall_unchanged(self):
        # no scope requested → historical behaviour: everything visible
        aid = self.remember("arena boss is weak to fire", scope="project:devon-arena")
        everything = self.memory.recall("boss fire", limit=10)
        self.assertIn(aid, {r.id for r in everything.records})

    def test_scope_normalization(self):
        from nomorals.memory.scopes import normalize_scope, scope_tag
        self.assertEqual(normalize_scope("Devon Arena"), "devon-arena")
        self.assertEqual(normalize_scope("project:Devon Arena"),
                         "project:devon-arena")
        self.assertEqual(scope_tag(""), "")
        self.assertEqual(scope_tag("project:devon-arena"),
                         "scope:project:devon-arena")

    def test_recent_path_respects_scope(self):
        aid = self.remember("arena scoped note", scope="project:devon-arena")
        hid = self.remember("home scoped note", scope="project:home")
        recents = self.memory.recall("", limit=10, scope="project:devon-arena")
        ids = {r.id for r in recents.records}
        self.assertIn(aid, ids)
        self.assertNotIn(hid, ids)

    def test_scopes_summary(self):
        from nomorals.memory.scopes import scopes_summary
        self.remember("global fact")
        self.remember("arena fact", scope="project:devon-arena")
        self.remember("home fact", scope="project:home")
        summary = scopes_summary(self.memory)
        self.assertEqual(summary["<global>"], 1)
        self.assertEqual(summary["project:devon-arena"], 1)
        self.assertEqual(summary["project:home"], 1)

    def test_remember_many_scope(self):
        ids = self.memory.remember_many(
            [("scoped one", "fact"), ("scoped two", "fact")],
            source="user:test", scope="project:x")
        self.assertEqual(len(ids), 2)
        scoped = self.memory.recall("", limit=10, scope="project:x")
        self.assertEqual({r.id for r in scoped.records}, set(ids))


# ── contradictions ─────────────────────────────────────────────────────────

class ContradictionTests(Harness):
    _n = 0

    def _rec(self, content, kind="fact"):
        ContradictionTests._n += 1
        return MemoryRecord(id=f"test-{ContradictionTests._n}", kind=kind,
                            content=content)

    def test_negation_flip(self):
        from nomorals.memory.contradictions import detect_against
        old = self._rec("the owner prefers dark mode interfaces always")
        new = self._rec("the owner does not prefer dark mode interfaces")
        found = detect_against(new, [old])
        self.assertTrue(found)
        self.assertEqual(found[0].strategy, "negation_flip")
        self.assertGreaterEqual(found[0].confidence, 0.5)

    def test_value_change(self):
        from nomorals.memory.contradictions import detect_against
        old = self._rec("the project deadline is oct 20")
        new = self._rec("the project deadline moved to oct 25")
        found = detect_against(new, [old])
        self.assertTrue(found)
        self.assertEqual(found[0].strategy, "value_change")

    def test_preference_flip(self):
        from nomorals.memory.contradictions import detect_against
        old = self._rec("i prefer dark mode interfaces always",
                        kind="preference")
        new = self._rec("i prefer light mode interfaces always",
                        kind="preference")
        found = detect_against(new, [old])
        self.assertTrue(found)
        self.assertEqual(found[0].strategy, "preference_flip")

    def test_no_false_positive_on_agreement(self):
        from nomorals.memory.contradictions import detect_against
        old = self._rec("the project deadline is oct 20")
        new = self._rec("the project deadline is oct 20, confirmed again")
        self.assertEqual(detect_against(new, [old]), [])

    def test_no_false_positive_on_unrelated(self):
        from nomorals.memory.contradictions import detect_against
        old = self._rec("the owner likes strong coffee in the morning")
        new = self._rec("the server migration finished last night")
        self.assertEqual(detect_against(new, [old]), [])

    def test_episodes_never_contradict(self):
        from nomorals.memory.contradictions import detect_against
        old = self._rec("went to the gym", kind="episode")
        new = self._rec("did not go to the gym", kind="episode")
        self.assertEqual(detect_against(new, [old]), [])

    def test_resolve_is_additive(self):
        from nomorals.memory.contradictions import detect_for, resolve
        old_id = self.remember("the project deadline is oct 20")
        new_id = self.remember("the project deadline moved to oct 25")
        found = detect_for(self.memory, new_id)
        self.assertTrue(found)
        outcome = resolve(self.memory, found[0])
        self.assertTrue(outcome["ok"])
        # old record still exists — marked, never deleted
        old = self.memory.get(old_id)
        self.assertIsNotNone(old)
        self.assertEqual((old.metadata or {}).get("superseded_by"), new_id)
        # excluded from default recall, visible with include_superseded
        default_ids = {r.id for r in
                       self.memory.recall("deadline", limit=10).records}
        self.assertNotIn(old_id, default_ids)
        full_ids = {r.id for r in
                    self.memory.recall("deadline", limit=10,
                                       include_superseded=True).records}
        self.assertIn(old_id, full_ids)
        # and the chain is intact
        chain = self.memory.supersession_chain(new_id)
        self.assertEqual([r.id for r in chain], [old_id, new_id])

    def test_resolve_dry_run_changes_nothing(self):
        from nomorals.memory.contradictions import detect_for, resolve
        old_id = self.remember("the project deadline is oct 20")
        new_id = self.remember("the project deadline moved to oct 25")
        found = detect_for(self.memory, new_id)
        self.assertTrue(found)
        outcome = resolve(self.memory, found[0], dry_run=True)
        self.assertTrue(outcome["ok"])
        self.assertNotIn("superseded_by",
                         self.memory.get(old_id).metadata or {})


# ── deep recall ────────────────────────────────────────────────────────────

class DeepRecallTests(Harness):
    def test_multihop_never_worse_than_single(self):
        from nomorals.memory.deep_recall import recall_deep
        target = self.remember("the database password is hunter2")
        single = self.memory.recall("database password", limit=5)
        deep = recall_deep(self.memory, "database password", limit=5)
        self.assertIn(target, {r.id for r in deep.records})
        # hop-1 hits are all present in the fused result
        single_ids = {r.id for r in single.records}
        self.assertTrue(single_ids <= {r.id for r in deep.records})

    def test_hop2_expands_vocabulary(self):
        from nomorals.memory.deep_recall import recall_deep
        # "credentials" never appears in the query, but hop-1 teaches hop-2
        self.remember("the wifi network is called HomeNet")
        self.remember("wifi credentials are taped under the router")
        deep = recall_deep(self.memory, "wifi network name", limit=5)
        texts = " ".join(deep.texts).lower()
        self.assertIn("credentials", texts)
        self.assertEqual(deep.hops, 2)
        self.assertTrue(deep.expansion_terms)

    def test_related(self):
        from nomorals.memory.deep_recall import related
        seed = self.remember("planning the lagos trip in december",
                             tags="travel")
        other = self.remember("lagos flights are cheapest on tuesdays",
                              tags="travel")
        _unrelated = self.remember("the database password is hunter2")
        rel = related(self.memory, seed, limit=5)
        ids = [r.id for r in rel]
        self.assertIn(other, ids)
        self.assertNotIn(seed, ids)

    def test_timeline_oldest_first_includes_superseded(self):
        from nomorals.memory.deep_recall import timeline
        now = time.time()
        old_id = self.remember("we decided to use postgres",
                               kind="decision")
        self.backdate(old_id, now - 30 * 86400)
        new_id = self.remember("we decided to use sqlite",
                               kind="decision")
        self.backdate(new_id, now - 5 * 86400)
        # supersede the old one — timeline must still show the evolution
        self.memory.supersede(old_id, "we decided to use sqlite",
                              kind="decision")
        items = timeline(self.memory, "database decision", limit=10)
        self.assertGreaterEqual(len(items), 2)
        stamps = [r.created_at for r in items]
        self.assertEqual(stamps, sorted(stamps))
        self.assertIn(old_id, {r.id for r in items})

    def test_decisions_about_window(self):
        from nomorals.memory.deep_recall import decisions_about
        now = time.time()
        recent = self.remember("we decided to launch in june",
                               kind="decision")
        self.backdate(recent, now - 10 * 86400)
        ancient = self.remember("we decided to launch in january",
                                kind="decision")
        self.backdate(ancient, now - 200 * 86400)
        found = decisions_about(self.memory, "launch", days=30)
        ids = {r.id for r in found}
        self.assertIn(recent, ids)
        self.assertNotIn(ancient, ids)

    def test_recall_window(self):
        from nomorals.memory.deep_recall import recall_window
        now = time.time()
        new_id = self.remember("fresh note about the garden")
        old_id = self.remember("old note about the garden")
        self.backdate(old_id, now - 90 * 86400)
        result = recall_window(self.memory, "garden", limit=10,
                               since=now - 7 * 86400)
        ids = {r.id for r in result.records}
        self.assertIn(new_id, ids)
        self.assertNotIn(old_id, ids)


# ── consolidation cadence ──────────────────────────────────────────────────

class CadenceTests(Harness):
    def test_consolidate_now_is_additive(self):
        from nomorals.memory.cadence import consolidate_now
        ids = [self.remember(f"episode number {i} about the gym routine",
                             kind="episode")
               for i in range(6)]
        report = consolidate_now(self.memory)
        self.assertTrue(report.get("ok"))
        self.assertGreaterEqual(report.get("summaries", 0), 1)
        # nothing deleted — every episode still recallable
        for rid in ids:
            self.assertIsNotNone(self.memory.get(rid))
        # ...but marked as distilled
        for rid in ids:
            md = self.memory.get(rid).metadata or {}
            self.assertIn("distilled_into", md)

    def test_second_run_skips_distilled(self):
        from nomorals.memory.cadence import consolidate_now
        for i in range(6):
            self.remember(f"episode number {i} about the gym routine",
                          kind="episode")
        first = consolidate_now(self.memory)
        second = consolidate_now(self.memory)
        self.assertTrue(first.get("ok") and second.get("ok"))
        self.assertEqual(second.get("episodes"), 0)
        self.assertGreater(second.get("skipped_distilled", 0), 0)

    def test_maybe_run_respects_interval(self):
        from nomorals.memory.cadence import maybe_run, meta_set
        for i in range(6):
            self.remember(f"episode number {i} about the gym routine",
                          kind="episode")
        first = maybe_run(self.memory)
        self.assertTrue(first.get("ran"))
        # immediately again → not due
        second = maybe_run(self.memory)
        self.assertFalse(second.get("ran"))
        self.assertEqual(second.get("reason"), "not due")
        # fake the clock back → due again
        meta_set(self.memory.db, "consolidation.last_run_at", "1")
        third = maybe_run(self.memory)
        self.assertTrue(third.get("ran"))

    def test_status_is_observable(self):
        from nomorals.memory.cadence import consolidate_now, status
        for i in range(6):
            self.remember(f"episode number {i} about the gym routine",
                          kind="episode")
        before = status(self.memory)
        self.assertTrue(before["enabled"])
        self.assertIsNone(before["last_run_at"])
        self.assertGreater(before["undistilled_episodes"], 0)
        consolidate_now(self.memory)
        after = status(self.memory)
        self.assertIsNotNone(after["last_run_at"])
        self.assertEqual(after["runs"], 1)
        self.assertEqual(after["undistilled_episodes"], 0)
        self.assertIn("summaries", after["last_report"])

    def test_health_reports_cadence(self):
        health = self.memory.health()
        self.assertIn("consolidation_cadence", health)
        cad = health["consolidation_cadence"]
        self.assertTrue(cad["enabled"])
        self.assertEqual(cad["mode"], "additive")

    def test_ensure_job_idempotent(self):
        from nomorals.memory.cadence import ensure_consolidation_job
        from nomorals.agents.scheduler import Scheduler
        first = ensure_consolidation_job(self.context)
        second = ensure_consolidation_job(self.context)
        self.assertTrue(first.get("scheduled") or
                        first.get("already_scheduled"))
        self.assertTrue(second.get("already_scheduled"))
        jobs = [j for j in Scheduler(self.context).list_jobs()
                if j.get("name") == "memory-consolidation"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["payload_kind"], "tool")
        self.assertEqual(jobs[0]["spec"], "every 3600s")


# ── backup / restore / import ──────────────────────────────────────────────

class BackupTests(Harness):
    def test_backup_verify_roundtrip(self):
        from nomorals.memory.backup import (backup_to, restore_from,
                                            verify_backup)
        self.remember("back me up", kind="fact")
        dest = tempfile.mkdtemp(prefix="nm-backup-")
        self.addCleanup(shutil.rmtree, dest, ignore_errors=True)
        report = backup_to(self.memory, dest, label="test")
        self.assertTrue(report.get("ok"))
        backup_dir = report["backup_dir"]
        check = verify_backup(backup_dir)
        self.assertTrue(check["ok"])
        self.assertGreater(check["checked"], 0)
        # restore into a fresh store
        home2 = temp_dir()
        self.addCleanup(shutil.rmtree, home2, ignore_errors=True)
        ctx2 = build_context(Settings(home=home2))
        ctx2.__enter__()
        self.addCleanup(ctx2.__exit__, None, None, None)
        ctx2.memory.embedder = Embedder(provider="hashing", dimensions=512)
        restore = restore_from(backup_dir, ctx2.memory)
        self.assertTrue(restore.get("ok"))
        self.assertTrue(restore.get("verified"))

    def test_verify_catches_tampering(self):
        from nomorals.memory.backup import backup_to, verify_backup
        self.remember("back me up", kind="fact")
        dest = tempfile.mkdtemp(prefix="nm-backup-")
        self.addCleanup(shutil.rmtree, dest, ignore_errors=True)
        report = backup_to(self.memory, dest)
        backup_dir = report["backup_dir"]
        # tamper with the memories db copy
        import pathlib
        db_copy = pathlib.Path(backup_dir) / "memories.db"
        if db_copy.is_file():
            with open(db_copy, "ab") as fh:
                fh.write(b"tamper")
            check = verify_backup(backup_dir)
            self.assertFalse(check["ok"])
            self.assertTrue(check["bad"])

    def test_import_dedupes_and_restores_private(self):
        from nomorals.memory.backup import import_records
        self.remember("already here", kind="fact")
        bundle = [
            {"content": "already here", "kind": "fact", "importance": 0.5},
            {"content": "brand new imported fact", "kind": "fact",
             "importance": 0.8, "private": True, "tags": "imported"},
            {"content": "   ", "kind": "fact"},  # empty → failed
        ]
        report = import_records(self.memory, bundle, source="test")
        self.assertTrue(report["ok"])
        self.assertEqual(report["skipped_duplicate"], 1)
        self.assertEqual(report["imported"], 1)
        self.assertEqual(report["failed"], 1)
        new_id = report["ids"][0]
        rec = self.memory.get(new_id)
        self.assertTrue((rec.metadata or {}).get("private"))
        # and it stays out of default recall
        ids = {r.id for r in
               self.memory.recall("brand new imported", limit=5).records}
        self.assertNotIn(new_id, ids)
        # dry run changes nothing
        before = self.memory.repo.count()
        dry = import_records(self.memory, bundle, dry_run=True)
        self.assertTrue(dry["ok"])
        self.assertEqual(self.memory.repo.count(), before)


# ── native-first embeddings ────────────────────────────────────────────────

class NativeEmbeddingTests(unittest.TestCase):
    def test_native_without_server_falls_back_to_hashing(self):
        emb = Embedder(provider="native")
        vec = emb.embed("the quick brown fox")
        self.assertEqual(len(vec), emb.dimensions)
        # L2-normalized
        self.assertAlmostEqual(sum(v * v for v in vec) ** 0.5, 1.0,
                               places=5)
        # deterministic — same text, same vector
        self.assertEqual(vec, emb.embed("the quick brown fox"))

    def test_native_never_touches_router(self):
        class ExplodingRouter:
            def embed(self, texts):
                raise AssertionError("router must not be called")
        emb = Embedder(provider="native", router=ExplodingRouter())
        vecs = emb.embed_many(["a", "b", "c"])
        self.assertEqual(len(vecs), 3)

    def test_select_for_profile(self):
        self.assertEqual(select_for_profile("termux"), "hashing")
        # no local server in CI → hashing, the honest fallback
        self.assertEqual(select_for_profile("laptop"), "hashing")
        self.assertEqual(select_for_profile("workstation"), "hashing")
        self.assertEqual(select_for_profile("mystery-box"), "auto")
        self.assertEqual(select_for_profile(None), "auto")


# ── extraction contradiction wiring ────────────────────────────────────────

class ExtractionContradictionTests(Harness):
    def test_extraction_supersedes_on_contradiction(self):
        from nomorals.memory.extract import MemoryExtractor

        class FakeRouter:
            def __init__(self, text):
                self._text = text

            def chat(self, messages, params=None, **kw):
                from nomorals.llm.base import LLMResponse
                return LLMResponse(text=self._text)

        self.context.settings.memory.extract_llm = True
        old_id = self.remember("the project deadline is oct 20",
                               kind="fact")
        llm_json = ('[{"kind": "fact", "content": '
                    '"the project deadline moved to oct 25", '
                    '"importance": 0.7}]')
        self.context.router = FakeRouter(llm_json)
        actions = MemoryExtractor(self.context).extract_turn(
            "quick heads up about the schedule")
        stored = [a for a in actions if a["action"] == "stored"]
        self.assertTrue(stored)
        # the contradiction was detected and the old fact superseded —
        # additively: the old record still exists
        old = self.memory.get(old_id)
        self.assertIsNotNone(old)
        self.assertIn("superseded_by", old.metadata or {})
        self.assertTrue(any("contradictions" in a for a in stored))


# ── trusted-path parity guard ──────────────────────────────────────────────

class TrustedPathUntouchedTests(Harness):
    def test_scoring_math_identical(self):
        import random
        from nomorals.memory.manager import _explain_score
        rng = random.Random(20261009)
        for _ in range(300):
            record = MemoryRecord(
                id="x", kind=rng.choice(["episode", "fact", "preference"]),
                content="c", importance=rng.random(),
                access_count=rng.randint(0, 50), created_at=0.0)
            sem, lex = rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 1.5)
            weights = {"recency": rng.random(), "importance": rng.random(),
                       "semantic": rng.random(), "lexical": rng.random()}
            expected = score_memory(record, semantic=sem, lexical=lex,
                                    weights=weights, now=1000.0)
            got, _ = _explain_score(record, semantic=sem, lexical=lex,
                                    weights=weights, now=1000.0)
            self.assertAlmostEqual(got, expected, places=9)

    def test_single_hop_deep_equals_trusted(self):
        from nomorals.memory.deep_recall import recall_deep
        ids = [self.remember(f"memory number {i} about gardening")
               for i in range(5)]
        trusted = [r.id for r in
                   self.memory.recall("gardening", limit=5).records]
        deep = recall_deep(self.memory, "gardening", limit=5, max_hops=1)
        self.assertEqual([r.id for r in deep.records], trusted)

    def test_scope_does_not_change_unscoped_ranking(self):
        texts = [f"unscoped memory {i} about Lagos traffic" for i in range(4)]
        for t in texts:
            self.remember(t)
        plain = [r.id for r in
                 self.memory.recall("Lagos traffic", limit=4).records]
        scoped = [r.id for r in
                  self.memory.recall("Lagos traffic", limit=4,
                                     scope="project:other").records]
        self.assertEqual(plain, scoped)


if __name__ == "__main__":
    unittest.main()
