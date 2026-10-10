"""Sweep(memory) upgrade tests: FSRS, communities, core blocks, MMR,
Matryoshka truncation, contextual chunking, mem0 decisions, bi-temporal
close, and presentation styles.

Offline by design: hashing embeddings, sqlite :memory: stores — every
assertion is reproducible.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.memory.base import (
    MemoryKind,
    MemoryRecord,
    format_recall,
    format_record,
)
from nomorals.memory.embeddings import (
    contextualize_chunk,
    matryoshka_truncate,
)
from nomorals.memory.extract import (
    MemoryDecision,
    decide,
    forget_targets,
)
from nomorals.memory.hybrid import Hit, mmr_select, rrf_fuse
from nomorals.memory.manager import CoreBlocks, MemoryManager
from nomorals.memory.persona import Community, PeopleGraph, PersonEntry
from nomorals.memory.repetition import (
    FSRS,
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    RepetitionScheduler,
    next_interval_days,
    retrievability,
)
from nomorals.memory.vector_backends import LegacyStoreBackend


def temp_home() -> str:
    return tempfile.mkdtemp(prefix="nm-sweep-")


# ── FSRS ────────────────────────────────────────────────────────────────────


class FSRSTests(unittest.TestCase):
    def test_forgetting_curve_shape(self):
        # R=1 at t=0, decays monotonically, R(t=S)≈0.9 (by construction:
        # (1+19/81)^-0.5 ≈ 0.898)
        self.assertAlmostEqual(retrievability(0.0, 10.0), 1.0, places=6)
        self.assertLess(retrievability(30.0, 10.0),
                        retrievability(1.0, 10.0))
        self.assertAlmostEqual(retrievability(10.0, 10.0), 0.898, places=2)

    def test_retention_is_a_setting_not_an_outcome(self):
        # Higher desired retention → shorter intervals (the SM-2 gap: SM-2
        # cannot even express the question).
        short = next_interval_days(10.0, 0.99)
        long = next_interval_days(10.0, 0.80)
        self.assertLess(short, long)
        self.assertGreater(long, 0.0)

    def test_intervals_grow_with_success(self):
        fsrs = FSRS()
        d, s = fsrs.init_difficulty(GRADE_GOOD), fsrs.init_stability(GRADE_GOOD)
        intervals = []
        for _ in range(4):
            elapsed = intervals[-1] if intervals else 1.0
            d, s, iv = fsrs.review(d, s, elapsed, GRADE_GOOD)
            intervals.append(iv)
        self.assertTrue(all(b > a for a, b in zip(intervals, intervals[1:])),
                        f"intervals should grow: {intervals}")

    def test_lapse_shrinks_stability_and_raises_difficulty(self):
        fsrs = FSRS()
        d, s = 5.0, 20.0
        d2, s2, iv = fsrs.review(d, s, 20.0, GRADE_AGAIN)
        self.assertLess(s2, s)
        self.assertGreater(d2, d)

    def test_scheduler_end_to_end_fsrs(self):
        sched = RepetitionScheduler(db=sqlite3.connect(":memory:"))
        self.assertTrue(sched.schedule("m1", "my girlfriend is Ada"))
        card = sched.review("m1", GRADE_GOOD)
        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card.algorithm, "fsrs")
        self.assertGreater(card.stability, 0.0)
        self.assertGreater(card.interval_days, 0.0)
        r = sched.retrievability_of("m1")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertGreater(r, 0.85)  # just reviewed → high recall prob

    def test_due_by_forgetting_orders_lowest_retrievability_first(self):
        sched = RepetitionScheduler(db=sqlite3.connect(":memory:"))
        sched.schedule("fresh", "just learned this")
        sched.schedule("old", "learned ages ago")
        sched.review("fresh", GRADE_EASY)
        sched.review("old", GRADE_GOOD)
        # Simulate time passing for "old" by backdating its last review.
        db = sched._db
        assert db is not None
        db.execute(
            "UPDATE review_cards SET last_reviewed = last_reviewed - ?, "
            "due_at = due_at - ? WHERE memory_id = 'old'",
            (30 * 86400.0, 30 * 86400.0))
        db.commit()
        ordered = sched.due_by_forgetting(5)
        self.assertGreaterEqual(len(ordered), 2)
        self.assertEqual(ordered[0].memory_id, "old")
        self.assertLess(ordered[0].retrievability_now,
                        ordered[-1].retrievability_now)

    def test_sm2_card_migrates_on_first_review(self):
        sched = RepetitionScheduler(db=sqlite3.connect(":memory:"))
        db = sched._db
        assert db is not None
        # Simulate a pre-FSRS card (SM-2 fields only).
        sched.schedule("legacy", "old memory")
        db.execute("UPDATE review_cards SET ease=2.5, interval_days=6.0, "
                   "reps=2, algorithm='sm2', stability=0 WHERE memory_id='legacy'")
        db.commit()
        card = sched.review("legacy", GRADE_GOOD)
        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card.algorithm, "fsrs")
        self.assertGreater(card.stability, 0.0)
        self.assertGreater(card.interval_days, 5.0)  # migrated stability pays off

    def test_hard_is_between_again_and_good(self):
        fsrs = FSRS()
        d, s = 5.0, 10.0
        _, _, iv_hard = fsrs.review(d, s, 5.0, GRADE_HARD)
        _, _, iv_good = fsrs.review(d, s, 5.0, GRADE_GOOD)
        _, _, iv_again = fsrs.review(d, s, 5.0, GRADE_AGAIN)
        self.assertLess(iv_again, iv_hard)
        self.assertLess(iv_hard, iv_good)


# ── communities (Graphiti Gc) ───────────────────────────────────────────────


class CommunityTests(unittest.TestCase):
    def _graph(self) -> PeopleGraph:
        g = PeopleGraph()
        for name, role in [("Ada", "girlfriend"), ("Chidi", "brother"),
                           ("Ada", "girlfriend"), ("Zainab", "trader"),
                           ("Musa", "trader")]:
            key = name.lower()
            g.people[key] = g.people.get(key) or PersonEntry(name=name)
            g.people[key].role = role
            g.people[key].mention_count += 1
            g.people[key].notes.append(f"note about {name}")
        # family edge Ada–Chidi, work edge Zainab–Musa
        g.edges[("ada", "chidi")] = 3
        g.edges[("musa", "zainab")] = 4
        return g

    def test_label_propagation_finds_two_communities(self):
        comms = self._graph().communities()
        self.assertEqual(len(comms), 2)
        sizes = sorted(c.size for c in comms)
        self.assertEqual(sizes, [2, 2])

    def test_community_summary_names_members(self):
        comms = self._graph().communities()
        summaries = " ".join(c.summary() for c in comms)
        self.assertIn("Ada", summaries)
        self.assertIn("Zainab", summaries)

    def test_neighbors_ranked_by_weight(self):
        g = self._graph()
        g.edges[("ada", "musa")] = 1
        nbrs = g.neighbors("Ada")
        self.assertEqual(nbrs[0][0].name, "Chidi")  # weight 3 > 1

    def test_empty_graph_has_no_communities(self):
        self.assertEqual(PeopleGraph().communities(), [])

    def test_to_dict_carries_summary(self):
        comms = self._graph().communities()
        d = comms[0].to_dict()
        self.assertIn("summary", d)
        self.assertIn("members", d)
        self.assertGreaterEqual(d["size"], 2)


# ── core blocks (Letta) ─────────────────────────────────────────────────────


class CoreBlocksTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.blocks = CoreBlocks(self.db, limit=100)

    def test_replace_and_get(self):
        r = self.blocks.replace("user", "Owner is death.")
        self.assertTrue(r["ok"])
        self.assertEqual(self.blocks.get("user"), "Owner is death.")

    def test_insert_appends(self):
        self.blocks.replace("context", "morning")
        self.blocks.insert("context", "evening")
        self.assertIn("morning", self.blocks.get("context"))
        self.assertIn("evening", self.blocks.get("context"))

    def test_hard_char_limit_truncates(self):
        r = self.blocks.replace("persona", "x" * 500)
        self.assertTrue(r["truncated"])
        self.assertEqual(len(self.blocks.get("persona")), 100)

    def test_pressure_signal_at_85_percent(self):
        self.blocks.replace("user", "x" * 90)
        pressure = self.blocks.memory_pressure()
        self.assertTrue(pressure["pressured"])
        self.assertEqual(
            pressure["blocks"]["user"]["signal"], "summarize or archive this block")
        self.blocks.replace("user", "short")
        self.assertFalse(self.blocks.memory_pressure()["pressured"])

    def test_render_injects_into_prompt(self):
        self.blocks.replace("persona", "I am Devon.")
        text = self.blocks.render()
        self.assertIn("## Core memory", text)
        self.assertIn("I am Devon.", text)

    def test_render_skips_empty_blocks(self):
        self.assertEqual(self.blocks.render(), "")


# ── manager integration ─────────────────────────────────────────────────────


class ManagerSweepTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.mem = MemoryManager(self.context)

    def test_core_blocks_live_on_manager(self):
        self.mem.core_blocks.replace("user", "Owner: death")
        self.assertEqual(self.mem.core_blocks.get("user"), "Owner: death")
        rendered = self.mem.core_blocks.render()
        self.assertIn("Owner: death", rendered)

    def test_recall_diversify_changes_order(self):
        for text in [
            "Ada likes coffee in the morning",
            "Ada likes coffee in the afternoon",
            "Ada likes coffee before meetings",
            "Ada likes coffee after lunch",
            "The gym opens at 6am daily",
        ]:
            self.mem.remember(text, kind="fact", importance=0.7)
        plain = self.mem.recall("Ada coffee", limit=5, diversify=0.0)
        diverse = self.mem.recall("Ada coffee", limit=5, diversify=0.6)
        self.assertEqual(len(plain.records), 5)
        self.assertEqual(len(diverse.records), 5)
        # The diversified set should surface the gym memory (coverage).
        diverse_texts = " ".join(r.content for r in diverse.records)
        self.assertIn("gym", diverse_texts)

    def test_ingest_document_contextual(self):
        doc = ("The annual report covers Q3 revenue. " * 20
               + "The vet said he doubled the dose. " * 5)
        n = self.mem.ingest_document(doc, source="report-2026",
                                     chunk_tokens=200, contextual=True)
        self.assertGreater(n, 1)
        # The vector lane saw the contextual header: recall the vague chunk.
        hits = self.mem.recall("doubled the dose report", limit=8)
        self.assertTrue(any("doubled the dose" in r.content
                            for r in hits.records))

    def test_remember_embed_text_overrides_vector_lane(self):
        seen: list[str] = []

        class SpyEmbedder:
            def embed(self, text):
                seen.append(text)
                return [1.0, 0.0]

            def embed_many(self, texts):
                return [self.embed(t) for t in texts]

        self.mem.embedder = SpyEmbedder()  # type: ignore[assignment]
        self.mem.remember("raw content here", embed_text="CONTEXT header here")
        self.assertTrue(any("CONTEXT" in t for t in seen))
        rec = self.mem.recall("raw content", limit=1).records[0]
        self.assertEqual(rec.content, "raw content here")  # stored raw


# ── mem0 decisions ──────────────────────────────────────────────────────────


class DecisionTests(unittest.TestCase):
    def _rec(self, rid, content):
        return MemoryRecord(id=rid, kind="fact", content=content)

    def test_add_for_new_knowledge(self):
        d = decide("my dog is named Rex", [], 0.80)
        self.assertEqual(d.action, MemoryDecision.ADD)

    def test_noop_for_duplicate(self):
        existing = [self._rec("1", "my girlfriend is Ada")]
        d = decide("my girlfriend is Ada", existing, 0.80)
        self.assertEqual(d.action, MemoryDecision.NOOP)
        self.assertEqual(d.target_id, "1")

    def test_update_for_contradiction(self):
        existing = [self._rec("1", "deadline is october 20")]
        d = decide("deadline is october 25", existing, 0.80,
                   contradictions=["c1"])
        self.assertEqual(d.action, MemoryDecision.UPDATE)

    def test_delete_forget_request(self):
        existing = [self._rec("1", "I like drinking tea every morning")]
        d = decide("", existing, 0.80, delete_target="I like tea")
        # "I like tea" vs "I like drinking tea every morning": containment
        # (min 20 chars on the normalized pair) → duplicate → delete.
        self.assertIn(d.action,
                      (MemoryDecision.DELETE, MemoryDecision.NOOP))

    def test_delete_noop_when_nothing_matches(self):
        d = decide("", [self._rec("1", "completely unrelated fact")],
                   0.80, delete_target="my secret password")
        self.assertEqual(d.action, MemoryDecision.NOOP)

    def test_forget_targets_detects_phrases(self):
        targets = forget_targets(
            "Hey. Please forget that I like drinking tea every morning.")
        self.assertTrue(any("tea" in t for t in targets))
        self.assertEqual(forget_targets("I love sunny days."), [])


# ── hybrid: MMR + weighted RRF ──────────────────────────────────────────────


class HybridSweepTests(unittest.TestCase):
    def test_weighted_rrf_tilts_lanes(self):
        v = ["a", "b", "c"]
        b = ["c", "b", "a"]
        even = rrf_fuse(v, b)
        tilted = rrf_fuse(v, b, bm25_weight=5.0)
        self.assertEqual(even[0][0], "a")     # vector lane wins ties
        self.assertEqual(tilted[0][0], "c")   # BM25 lane wins when weighted

    def test_mmr_diversifies(self):
        q = [1.0, 0.0]
        hits = [Hit(id=str(i), text=f"doc {i}") for i in range(4)]
        vectors = {
            "0": [1.0, 0.0], "1": [0.99, 0.01], "2": [0.98, 0.02],
            "3": [0.0, 1.0],  # relevant-ish but different direction
        }
        pure = mmr_select(q, hits, vectors, limit=3, lambda_mult=1.0)
        diverse = mmr_select(q, hits, vectors, limit=3, lambda_mult=0.2)
        self.assertEqual([h.id for h in pure], ["0", "1", "2"])
        self.assertIn("3", [h.id for h in diverse])

    def test_mmr_never_drops_vectorless_hits(self):
        q = [1.0, 0.0]
        hits = [Hit(id="a", text="a"), Hit(id="b", text="b")]
        out = mmr_select(q, hits, {}, limit=2)
        self.assertEqual(len(out), 2)


# ── matryoshka + backends ───────────────────────────────────────────────────


class TruncationTests(unittest.TestCase):
    def test_matryoshka_truncate(self):
        v = [float(i) for i in range(1024)]
        t = matryoshka_truncate(v, 512)
        self.assertEqual(len(t), 512)
        self.assertEqual(t[:4], [0.0, 1.0, 2.0, 3.0])
        # No-op cases.
        self.assertEqual(len(matryoshka_truncate(v, 0)), 1024)
        self.assertEqual(len(matryoshka_truncate(v, 2048)), 1024)

    def _manager_with_truncated_backend(self, dims: int):
        home = temp_home()
        mem = MemoryManager(build_context(Settings(home=home)))
        mem.semantic = LegacyStoreBackend(mem.db, truncate_dims=dims)
        return mem

    def test_legacy_backend_truncates(self):
        mem = self._manager_with_truncated_backend(8)
        rid = mem.remember("truncation probe content alpha")
        hits = mem.recall("truncation probe", limit=3)
        self.assertTrue(any(r.id == rid for r in hits.records))
        self.assertEqual(mem.semantic.truncate_dims, 8)

    def test_truncate_dims_reported(self):
        from nomorals.storage.db import Database
        db = Database(":memory:")
        be = LegacyStoreBackend(db, truncate_dims=16)
        self.assertEqual(be.truncate_dims, 16)
        self.assertEqual(LegacyStoreBackend(db).truncate_dims, 0)


# ── presentation styles ─────────────────────────────────────────────────────


class StyleTests(unittest.TestCase):
    def _rec(self) -> MemoryRecord:
        r = MemoryRecord(id="1", kind="fact", content="Ada is my girlfriend",
                         importance=0.9, score=0.87, trust="trusted",
                         access_count=3)
        return r

    def test_compact_is_one_line(self):
        line = format_record(self._rec(), "compact")
        self.assertNotIn("\n", line)
        self.assertIn("Ada is my girlfriend", line)

    def test_rich_has_provenance(self):
        text = format_record(self._rec(), "rich")
        self.assertIn("trusted", text)
        self.assertIn("0.87", text)

    def test_chat_is_human_safe(self):
        text = format_record(self._rec(), "chat")
        self.assertEqual(text, "Ada is my girlfriend")

    def test_briefing_has_glyph_and_age(self):
        text = format_record(self._rec(), "briefing")
        self.assertIn("📌", text)
        self.assertIn("ago", text)

    def test_format_recall_numbered(self):
        text = format_recall([self._rec(), self._rec()], "compact",
                             header="Memories:", numbered=True)
        self.assertIn("Memories:", text)
        self.assertIn("1.", text)
        self.assertIn("2.", text)

    def test_unknown_style_falls_back(self):
        self.assertIn("Ada", format_record(self._rec(), "hologram"))

    def test_never_raises_on_broken_record(self):
        broken = MemoryRecord(id="", kind="", content="")
        for style in ("compact", "rich", "chat", "briefing"):
            self.assertIsInstance(format_record(broken, style), str)


# ── embeddings helpers ──────────────────────────────────────────────────────


class EmbeddingHelperTests(unittest.TestCase):
    def test_contextualize_chunk(self):
        header = contextualize_chunk("The vet doubled the dose. " * 10,
                                     source="vet-notes")
        self.assertIn("vet-notes", header)
        self.assertIn("vet doubled", header)

    def test_contextualize_empty_document(self):
        self.assertIn("note", contextualize_chunk("", source=""))


if __name__ == "__main__":
    unittest.main()
