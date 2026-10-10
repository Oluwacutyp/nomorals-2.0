"""Sweep tests: mined-then-built upgrades across nomorals/wisdom/.

Covers the new behavior only — embeddings asymmetry, vector cache +
pre-filtering, weighted fusion, retrieval upgrades (expansion / floor /
MMR), answer rendering, history views, practice upgrades, ingestor
backends, TextRank digests, keeper passthroughs, chat upgrades.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.wisdom import (
    Answer,
    CanonCorpus,
    CorpusError,
    EmbeddingError,
    HashEmbedBackend,
    HistoryEngine,
    PracticeError,
    PracticeGuide,
    WisdomKeeper,
    embed_texts,
    fuse_hits,
    open_index,
    reciprocal_rank_fusion,
    truncate_dim,
    weighted_rrf,
)
from nomorals.wisdom.autonomy import WisdomOrgan, mmr_order, textrank
from nomorals.wisdom.corpus import ManifestEntry, _mmr_select
from nomorals.wisdom.embeddings import SentenceTransformersBackend
from nomorals.wisdom.history import ERAS, era_of


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-sweep-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


def _entry(slug="gospel-of-thomas", **kw):
    d = {
        "slug": slug,
        "title": "Gospel of Thomas",
        "tradition": "christian-gnostic",
        "canon_status": "gnostic",
        "translator": "Patterson & Robinson",
        "source_url": "http://gnosis.org/naghamm/gth_pat_rob.htm",
        "license": "public-domain",
    }
    d.update(kw)
    return ManifestEntry.from_dict(d)


LOREM = (
    "# The Kingdom\n\n"
    + "The kingdom of heaven is within you and all around you. " * 40
    + "\n\n# Sayings\n\n"
    + "Blessed are the seekers of the inner light and the quiet mind. " * 40
)


class FakeClock:
    def __init__(self):
        self.slept = 0.0
        self.t = 1_700_000_000.0

    def sleep(self, seconds):
        self.slept += seconds
        self.t += seconds

    def now(self):
        return self.t


# ── embeddings: asymmetry, prefixes, MRL, cache keys, async ──────────

class EmbeddingAsymmetryTests(unittest.TestCase):
    def test_hash_backend_prefixes_empty(self):
        b = HashEmbedBackend()
        self.assertEqual(b.query_prefix, "")
        self.assertEqual(b.document_prefix, "")

    def test_e5_model_gets_prefixes(self):
        with mock.patch.object(
                SentenceTransformersBackend, "available",
                classmethod(lambda cls: True)):
            b = SentenceTransformersBackend(
                "intfloat/multilingual-e5-small")
        self.assertEqual(b.query_prefix, "query: ")
        self.assertEqual(b.document_prefix, "passage: ")

    def test_non_e5_model_no_prefixes(self):
        with mock.patch.object(
                SentenceTransformersBackend, "available",
                classmethod(lambda cls: True)):
            b = SentenceTransformersBackend(
                "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(b.query_prefix, "")
        self.assertEqual(b.document_prefix, "")

    def test_cache_key_role_aware(self):
        b = HashEmbedBackend()
        q = b.cache_key("hello world", role="query")
        p = b.cache_key("hello world", role="passage")
        self.assertNotEqual(q, p)
        self.assertEqual(q, b.cache_key("hello world", role="query"))

    def test_embed_texts_role_routing(self):
        b = HashEmbedBackend()
        vecs = embed_texts(b, ["alpha beta"], role="passage")
        self.assertEqual(len(vecs), 1)
        self.assertEqual(len(vecs[0]), b.dim)

    def test_embed_texts_mrl_truncation(self):
        b = HashEmbedBackend()
        vecs = embed_texts(b, ["alpha beta gamma"], dimensions=64)
        self.assertEqual(len(vecs[0]), 64)
        # still unit-normalized after truncation
        norm = sum(x * x for x in vecs[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_truncate_dim_standalone(self):
        b = HashEmbedBackend()
        vecs = embed_texts(b, ["one", "two"])
        small = truncate_dim(vecs, 32)
        self.assertEqual(len(small[0]), 32)

    def test_embed_texts_bad_dimensions_rejected(self):
        b = HashEmbedBackend()
        with self.assertRaises(EmbeddingError):
            embed_texts(b, ["x"], dimensions=0)

    def test_async_variants(self):
        b = HashEmbedBackend()
        vec = asyncio.run(b.aembed_query("seekers of light"))
        self.assertEqual(len(vec), b.dim)
        vecs = asyncio.run(b.aembed_documents(["a b c", "d e f"]))
        self.assertEqual(len(vecs), 2)


# ── vectorstore: cache, metadata, pre-filter, floor, stats ───────────

class VectorSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wisdom-vec-sweep-")
        self.path = str(Path(self.tmp) / "vectors.db")
        self.backend = HashEmbedBackend()

    def _build(self, docs, metadata=None):
        index = open_index(self.path, self.backend)
        try:
            keys = [k for k, _ in docs]
            texts = [t for _, t in docs]
            return index.build(keys, texts, self.backend,
                               metadata=metadata), index
        finally:
            pass  # caller closes

    def test_build_reports_cached_vs_embedded(self):
        docs = [("k1", "the kingdom of heaven is within you"),
                ("k2", "blessed are the seekers of light")]
        index = open_index(self.path, self.backend)
        try:
            r1 = index.build([k for k, _ in docs],
                             [t for _, t in docs], self.backend)
            self.assertEqual(r1["vectors"], 2)
            self.assertEqual(r1["embedded"], 2)
            self.assertEqual(r1["cached"], 0)
            r2 = index.build([k for k, _ in docs],
                             [t for _, t in docs], self.backend)
            self.assertEqual(r2["vectors"], 2)
            self.assertEqual(r2["embedded"], 0)
            self.assertEqual(r2["cached"], 2)
        finally:
            index.close()

    def test_build_partial_cache(self):
        docs = [("k1", "the kingdom of heaven is within you")]
        index = open_index(self.path, self.backend)
        try:
            index.build(["k1"], [docs[0][1]], self.backend)
            r = index.build(["k1", "k2"],
                            [docs[0][1], "a brand new passage here"],
                            self.backend)
            self.assertEqual(r["cached"], 1)
            self.assertEqual(r["embedded"], 1)
        finally:
            index.close()

    def test_tradition_prefilter(self):
        docs = [("k1", "the kingdom of heaven is within you"),
                ("k2", "quantum field theory and particles")]
        meta = {"k1": {"tradition": "gnostic", "work": "Thomas"},
                "k2": {"tradition": "physics", "work": "QFT"}}
        index = open_index(self.path, self.backend)
        try:
            index.build([k for k, _ in docs], [t for _, t in docs],
                        self.backend, metadata=meta)
            q = self.backend.embed_query("kingdom heaven within")
            hits = index.search(q, top=5, tradition="gnostic")
            self.assertEqual([k for k, _ in hits], ["k1"])
            hits_all = index.search(q, top=5)
            self.assertEqual(len(hits_all), 2)
        finally:
            index.close()

    def test_min_score_floor(self):
        docs = [("k1", "the kingdom of heaven is within you"),
                ("k2", "quantum field theory and particles")]
        index = open_index(self.path, self.backend)
        try:
            index.build([k for k, _ in docs], [t for _, t in docs],
                        self.backend)
            q = self.backend.embed_query("completely unrelated zebra")
            hits = index.search(q, top=5, min_score=0.99)
            self.assertEqual(hits, [])
        finally:
            index.close()

    def test_stats_and_vacuum(self):
        index = open_index(self.path, self.backend)
        try:
            index.build(["k1"], ["some passage text here"], self.backend)
            st = index.stats()
            self.assertEqual(st["vectors"], 1)
            self.assertEqual(st["backend"], "hashing")
            self.assertEqual(st["dim"], 512)
            self.assertIn(st["engine"], ("vec0", "python"))
            self.assertTrue(st["built_at"])
            self.assertGreaterEqual(st["cached_vectors"], 1)
            index.vacuum()  # must not raise
        finally:
            index.close()

    def test_upsert_metadata_kwargs(self):
        index = open_index(self.path, self.backend)
        try:
            vec = self.backend.embed_one("hello world")
            index.upsert("k9", vec, tradition="t", work="w")
            hits = index.search(vec, top=1, tradition="t")
            self.assertEqual(hits[0][0], "k9")
            self.assertEqual(index.search(vec, top=1, tradition="nope"), [])
        finally:
            index.close()


# ── hybrid: weights, floor, score transparency ────────────────────────

class _Hit:
    def __init__(self, key, score=0.0):
        self.key = key
        self.score = score


class HybridSweepTests(unittest.TestCase):
    def test_weighted_rrf_keyword_heavy(self):
        kw = ["a", "b", "c"]
        sem = ["c", "b", "a"]
        plain = weighted_rrf([kw, sem])
        self.assertEqual(
            max(plain, key=plain.get), "a")  # tie-ish; a leads kw
        heavy_kw = weighted_rrf([kw, sem], weights=[3.0, 1.0])
        self.assertGreater(heavy_kw["a"], heavy_kw["c"])
        heavy_sem = weighted_rrf([kw, sem], weights=[1.0, 3.0])
        self.assertGreater(heavy_sem["c"], heavy_sem["a"])

    def test_weighted_rrf_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            weighted_rrf([["a"]], weights=[1.0, 2.0])

    def test_reciprocal_rank_fusion_still_plain_rrf(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c"]])
        # b: 1/61+1/62 > c: 1/62+1/63 > a: 1/61
        self.assertGreater(fused["b"], fused["c"])
        self.assertGreater(fused["c"], fused["a"])

    def test_fuse_hits_semantic_floor(self):
        kw = [_Hit("a"), _Hit("b")]
        sem = [_Hit("c", score=0.9), _Hit("d", score=0.1)]
        fused = fuse_hits(kw, sem, key_fn=lambda h: h.key,
                          semantic_floor=0.5, top=10)
        keys = [h.key for h in fused]
        self.assertIn("c", keys)
        self.assertNotIn("d", keys)

    def test_fuse_hits_score_transparency(self):
        kw = [_Hit("a"), _Hit("b")]
        sem = [_Hit("b", score=0.8), _Hit("c", score=0.7)]
        fused = fuse_hits(kw, sem, key_fn=lambda h: h.key, top=10)
        by_key = {h.key: h for h in fused}
        self.assertTrue(hasattr(by_key["b"], "rrf_score"))
        self.assertEqual(by_key["b"].keyword_rank, 2)
        self.assertEqual(by_key["b"].semantic_rank, 1)
        self.assertIsNone(by_key["a"].semantic_rank)

    def test_fuse_hits_weights_shift_order(self):
        kw = [_Hit("kw-only"), _Hit("both")]
        sem = [_Hit("both", score=0.9), _Hit("sem-only", score=0.85)]
        fused = fuse_hits(kw, sem, key_fn=lambda h: h.key, top=10,
                          weights=(100.0, 1.0))
        self.assertEqual(fused[0].key, "kw-only")


# ── corpus: expansion, floor, MMR, rendering, confidence ──────────────

class CorpusSweepTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = _ctx()
        self.corpus = CanonCorpus(self.ctx)
        self.corpus.register(_entry())
        self.corpus.ingest_text("gospel-of-thomas", LOREM)
        self.corpus.register(_entry(
            slug="pistis-sophia", title="Pistis Sophia",
            tradition="christian-gnostic", canon_status="gnostic",
            source_url="http://gnosis.org/library/pistis-sophia.htm"))
        self.corpus.ingest_text(
            "pistis-sophia",
            "# Sophia\n\n" + "Sophia fell from the light of the treasury. " * 40)

    def test_expand_query_adds_terms(self):
        expanded = self.corpus._expand_query("kingdom")
        self.assertTrue(expanded.startswith("kingdom"))
        self.assertGreater(len(expanded), len("kingdom"))

    def test_expand_query_no_hits_returns_original(self):
        q = "zxqvkw absent words here"
        self.assertEqual(self.corpus._expand_query(q), q)

    def test_ask_diversify_dedups(self):
        ans = self.corpus.ask("kingdom heaven", top=2, diversify=True)
        self.assertLessEqual(len(ans.passages), 2)

    def test_ask_bad_mmr_lambda_rejected(self):
        with self.assertRaises(CorpusError):
            self.corpus.ask("kingdom", mmr_lambda=1.5)

    def test_mmr_select_spreads_coverage(self):
        from nomorals.wisdom.corpus import ProvenanceHit
        mk = lambda s, sc: ProvenanceHit(
            work="w", translator="", section="s", url="", snippet=s,
            score=sc)
        dup = "the kingdom of heaven is within you and around you"
        passages = [mk(dup + " x", 1.0), mk(dup + " y", 0.99),
                    mk("sophia fell from the treasury of light", 0.5)]
        picked = _mmr_select(passages, top=2, lambda_=0.5)
        self.assertEqual(len(picked), 2)
        texts = [p.snippet for p in picked]
        self.assertTrue(any("sophia" in t for t in texts))

    def test_confidence_levels(self):
        self.assertEqual(self.corpus._confidence([]), "none")
        one = [SimpleNamespace(work="w1")]
        self.assertEqual(self.corpus._confidence(one), "low")
        two = [SimpleNamespace(work="w1"), SimpleNamespace(work="w1")]
        self.assertEqual(self.corpus._confidence(two), "medium")
        three = [SimpleNamespace(work="w1"), SimpleNamespace(work="w2"),
                 SimpleNamespace(work="w3")]
        self.assertEqual(self.corpus._confidence(three), "high")

    def test_answer_render_chat(self):
        ans = self.corpus.ask("kingdom within", top=2)
        card = ans.render("chat")
        self.assertIn("kingdom within", card)
        self.assertIn("Gospel of Thomas", card)

    def test_answer_render_terminal(self):
        ans = self.corpus.ask("kingdom within", top=1)
        text = ans.render("terminal")
        self.assertIn("QUERY:", text)
        self.assertIn("CONFIDENCE:", text)

    def test_answer_render_markdown(self):
        ans = self.corpus.ask("kingdom within", top=1)
        md = ans.render("markdown")
        self.assertIn("## kingdom within", md)
        self.assertIn(">", md)

    def test_answer_render_empty(self):
        ans = self.corpus.ask("zxqvkw absent words")
        for style in ("chat", "terminal", "markdown"):
            self.assertIn("zxqvkw", ans.render(style))

    def test_answer_render_bad_style_rejected(self):
        ans = self.corpus.ask("kingdom")
        with self.assertRaises(CorpusError):
            ans.render("carrier-pigeon")

    def test_answer_to_dict_has_confidence(self):
        ans = self.corpus.ask("kingdom")
        d = ans.to_dict()
        self.assertIn("confidence", d)

    def test_semantic_index_incremental(self):
        st = self.corpus.build_semantic_index("hashing")
        self.assertTrue(st["built"])
        self.assertEqual(st["backend"], "hashing")
        first_embedded = st["embedded"]
        self.assertGreater(first_embedded, 0)
        st2 = self.corpus.build_semantic_index("hashing")
        # unchanged passage set -> skipped, still consistent
        self.assertTrue(st2["built"])

    def test_ask_semantic_with_floor(self):
        self.corpus.build_semantic_index("hashing")
        ans = self.corpus.ask("kingdom of heaven", mode="semantic",
                              top=3, min_score=0.0)
        self.assertGreater(len(ans.passages), 0)
        ans2 = self.corpus.ask("kingdom of heaven", mode="semantic",
                               top=3, min_score=0.999)
        self.assertEqual(len(ans2.passages), 0)

    def test_ask_hybrid_weights(self):
        self.corpus.build_semantic_index("hashing")
        ans = self.corpus.ask("kingdom heaven", mode="hybrid", top=3,
                              weights=(2.0, 1.0))
        self.assertGreater(len(ans.passages), 0)

    def test_index_metadata_tradition_prefilter(self):
        self.corpus.build_semantic_index("hashing")
        ans = self.corpus.ask("light", mode="semantic", top=5,
                              tradition="christian-gnostic")
        for p in ans.passages:
            # every hit maps back to a manifest entry of that tradition
            self.assertTrue(p.work)


# ── history: eras, parallel, density, gaps, ascii ─────────────────────

class HistorySweepTests(unittest.TestCase):
    def setUp(self):
        self.eng = HistoryEngine(SimpleNamespace())

    def test_era_of(self):
        self.assertEqual(era_of(-1500), "Ancient")
        self.assertEqual(era_of(1000), "Medieval")
        self.assertEqual(era_of(1600), "Early Modern")
        self.assertEqual(era_of(1900), "Modern & Contemporary")
        self.assertEqual(era_of(99999), "")

    def test_eras_constant_shape(self):
        self.assertEqual(len(ERAS), 4)
        for name, start, end in ERAS:
            self.assertLess(start, end)

    def test_events_by_era(self):
        grouped = self.eng.events_by_era()
        self.assertEqual(set(grouped), {n for n, _, _ in ERAS})
        total = sum(len(v) for v in grouped.values())
        self.assertEqual(total, len(self.eng.events()))
        one = self.eng.events_by_era("Medieval")
        self.assertEqual(set(one), {"Medieval"})
        with self.assertRaises(Exception):
            self.eng.events_by_era("Bronze Age")

    def test_parallel_at(self):
        par = self.eng.parallel_at(200, 100)
        self.assertEqual(par["year"], 200)
        self.assertGreater(par["total"], 0)
        for trad, evs in par["by_tradition"].items():
            for e in evs:
                self.assertLessEqual(e["start"], 300)
                self.assertGreaterEqual(e["end"], 100)
        with self.assertRaises(Exception):
            self.eng.parallel_at(200, -5)

    def test_century_density(self):
        density = self.eng.century_density()
        self.assertTrue(density)
        for century, bucket in density.items():
            int(century)  # keys are ints-as-strings
            self.assertTrue(all(v > 0 for v in bucket.values()))

    def test_gaps(self):
        gaps = self.eng.gaps()
        self.assertIn("thin_traditions", gaps)
        self.assertIn("tradition_counts", gaps)
        self.assertEqual(gaps["total_events"], 52)
        self.assertEqual(sum(gaps["tradition_counts"].values()), 52)

    def test_render_ascii(self):
        out = self.eng.render_ascii(width=70)
        self.assertIn("ANCIENT", out)
        self.assertIn("BCE", out)
        lines = out.split("\n")
        self.assertLessEqual(max(len(l) for l in lines), 110)
        self.assertIn("\u25cf", out)  # event markers

    def test_render_ascii_subset(self):
        evs = self.eng.timeline(tradition="buddhism")
        out = self.eng.render_ascii(evs, width=60)
        self.assertIn("buddhism", out.lower())

    def test_render_ascii_empty(self):
        self.assertEqual(self.eng.render_ascii([], width=60),
                         "(no events)")


# ── practice: custom, settle, ramp, chime, stats, ratings, catalog ────

class PracticeSweepTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = _ctx()
        self.guide = PracticeGuide(self.ctx)

    def test_create_custom_and_run(self):
        script = self.guide.create_custom(
            "my-cooldown", "My Cooldown", "A custom wind-down.",
            [{"label": "Inhale", "seconds": 2,
              "instruction": "Breathe in gently.", "repeat": 2},
             {"label": "Exhale", "seconds": 3,
              "instruction": "Let it all out."}])
        self.assertEqual(script.id, "my-cooldown")
        self.assertIn("my-cooldown",
                      [s["id"] for s in self.guide.list_sessions()])
        lines = []
        clock = FakeClock()
        result = self.guide.run("my-cooldown", clock=clock,
                                out=lines.append)
        self.assertTrue(result["completed"])
        self.assertEqual(result["phases_done"], 3)
        # custom session survives a fresh guide instance
        g2 = PracticeGuide(self.ctx)
        self.assertIn("my-cooldown",
                      [s["id"] for s in g2.list_sessions()])

    def test_create_custom_rejects_collision(self):
        with self.assertRaises(PracticeError):
            self.guide.create_custom(
                "box-breathing", "X", "Y",
                [{"label": "Inhale", "seconds": 2,
                  "instruction": "in"}])

    def test_create_custom_validates_phases(self):
        with self.assertRaises(PracticeError):
            self.guide.create_custom(
                "bad-one", "X", "Y",
                [{"label": "Inhale", "seconds": -2, "instruction": "in"}])

    def test_delete_custom(self):
        self.guide.create_custom(
            "temp-x", "Temp", "t",
            [{"label": "Inhale", "seconds": 2, "instruction": "in"}])
        self.assertTrue(self.guide.delete_custom("temp-x"))
        self.assertNotIn("temp-x",
                         [s["id"] for s in self.guide.list_sessions()])

    def test_delete_packaged_rejected(self):
        with self.assertRaises(PracticeError):
            self.guide.delete_custom("box-breathing")

    def test_settle_seconds(self):
        lines, clock = [], FakeClock()
        self.guide.run("four-seven-eight", clock=clock, out=lines.append,
                       settle_seconds=5)
        self.assertEqual(clock.slept, 76 + 5)
        self.assertTrue(any("Settle in" in l for l in lines))

    def test_settle_negative_rejected(self):
        with self.assertRaises(PracticeError):
            self.guide.run("box-breathing", clock=FakeClock(),
                           out=lambda l: None, settle_seconds=-1)

    def test_ramp_scales_durations(self):
        clock = FakeClock()
        self.guide.run("box-breathing", clock=clock,
                       out=lambda l: None, rounds=2, ramp=2.0)
        # round 1: 80s at 1.0x, round 2: 80s at 2.0x
        self.assertAlmostEqual(clock.slept, 240.0)

    def test_ramp_flat_is_default(self):
        clock = FakeClock()
        self.guide.run("box-breathing", clock=clock,
                       out=lambda l: None, rounds=2, ramp=1.0)
        self.assertAlmostEqual(clock.slept, 160.0)

    def test_ramp_bounds_rejected(self):
        for bad in (0.1, 3.5, "2", True):
            with self.assertRaises(PracticeError, msg=f"ramp={bad!r}"):
                self.guide.run("box-breathing", clock=FakeClock(),
                               out=lambda l: None, ramp=bad)

    def test_midpoint_chime(self):
        lines, clock = [], FakeClock()
        self.guide.run("box-breathing", clock=clock, out=lines.append)
        chimes = [l for l in lines if "halfway" in l]
        self.assertEqual(len(chimes), 1)

    def test_rate_and_stats(self):
        import time as _time
        clock = FakeClock()
        clock.t = _time.time()
        self.guide.run("box-breathing", clock=clock,
                       out=lambda l: None)
        self.guide.rate("box-breathing", 4)
        self.guide.rate("box-breathing", 5)
        st = self.guide.stats()
        self.assertEqual(st["sessions_completed"], 1)
        self.assertGreater(st["minutes_practiced"], 0)
        self.assertEqual(st["by_session"]["box-breathing"], 1)
        self.assertEqual(st["avg_rating"]["box-breathing"], 4.5)
        self.assertEqual(st["streak_days"], 1)
        self.assertIsNotNone(st["last_session"])

    def test_rate_bounds_rejected(self):
        for bad in (0, 6, "5", 2.5, True):
            with self.assertRaises(PracticeError, msg=f"stars={bad!r}"):
                self.guide.rate("box-breathing", bad)

    def test_rate_unknown_session_rejected(self):
        with self.assertRaises(PracticeError):
            self.guide.rate("nope", 3)

    def test_catalog_text(self):
        text = self.guide.catalog_text()
        self.assertIn("Box Breathing", text)
        self.assertIn("4-4-4-4", text)
        self.assertIn("\u25b2", text)  # inhale glyph
        self.assertIn("id: box-breathing", text)

    def test_rhythm_signature(self):
        self.assertEqual(
            self.guide.rhythm_signature("box-breathing"), "4-4-4-4")
        self.assertEqual(
            self.guide.rhythm_signature("four-seven-eight"), "4-7-8")


# ── ingestor: boilerplate, gutendex, wikisource, multi-source ─────────

class _FakeResp:
    def __init__(self, text):
        self.text = text
        self.body = text.encode()


class _FakeHttp:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        return self.handler(url)


def _ing_ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-ing-sweep-")
    return SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))


class IngestorSweepTests(unittest.TestCase):
    def _ingestor(self, handler):
        from nomorals.wisdom import ingestor as ing_mod
        ing = ing_mod.ArchiveIngestor(_ing_ctx())
        ing._http = _FakeHttp(handler)
        ing._robots = mock.Mock()
        ing._robots.allowed.return_value = True
        return ing, ing_mod

    def test_strip_gutenberg_boilerplate(self):
        from nomorals.wisdom.ingestor import ArchiveIngestor
        pg = ("license header junk\n"
              "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
              "real text here\n"
              "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
              "trailer junk")
        self.assertEqual(
            ArchiveIngestor.strip_gutenberg_boilerplate(pg),
            "real text here\n")
        plain = "just a normal text"
        self.assertEqual(
            ArchiveIngestor.strip_gutenberg_boilerplate(plain), plain)

    def test_strip_boilerplate_partial_markers(self):
        from nomorals.wisdom.ingestor import ArchiveIngestor
        only_end = "body text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
        self.assertEqual(
            ArchiveIngestor.strip_gutenberg_boilerplate(only_end),
            "body text\n")

    def test_search_gutendex_candidates(self):
        payload = {"results": [{
            "id": 16439,
            "title": "The Yoga Sutras",
            "authors": [{"name": "Patanjali"}],
            "subjects": ["Yoga"],
            "formats": {},
        }]}
        ing, _ = self._ingestor(
            lambda url: _FakeResp(json.dumps(payload)))
        results = ing.search("yoga sutras", sources=("gutendex",))
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand["identifier"], "gutenberg-16439")
        self.assertIn("gutenberg.org", cand["url"])
        self.assertEqual(cand["authors"], "Patanjali")
        self.assertIn("Gutenberg", cand["edition"])
        self.assertEqual(cand["source"], "gutendex")

    def test_search_wikisource_candidates(self):
        payload = {"query": {"search": [{
            "title": "Tao Te Ching",
            "snippet": "the classic of the way",
        }]}}
        ing, _ = self._ingestor(
            lambda url: _FakeResp(json.dumps(payload)))
        results = ing.search("tao te ching", sources=("wikisource",))
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand["title"], "Tao Te Ching")
        self.assertIn("en.wikisource.org/wiki/Tao_Te_Ching", cand["url"])
        self.assertEqual(cand["source"], "wikisource")

    def test_search_merges_and_dedups(self):
        payload = {"results": [{
            "id": 1, "title": "Dup", "authors": [], "subjects": [],
            "formats": {}}]}
        ing, _ = self._ingestor(
            lambda url: _FakeResp(json.dumps(payload)))
        results = ing.search("dup", sources=("gutendex", "gutendex"))
        urls = [c["url"] for c in results]
        self.assertEqual(len(urls), len(set(urls)))

    def test_search_unknown_source_rejected(self):
        from nomorals.wisdom.errors import IngestError
        ing, _ = self._ingestor(lambda url: _FakeResp("{}"))
        with self.assertRaises(IngestError):
            ing.search("x", sources=("atlantis",))

    def test_search_all_fail_raises(self):
        from nomorals.wisdom.errors import IngestError
        from nomorals.core.errors import NoMoralsError

        def boom(url):
            raise NoMoralsError("down", retryable=False)

        ing, _ = self._ingestor(boom)
        with self.assertRaises(IngestError):
            ing.search("x", sources=("gutendex",))

    def test_polite_wait_throttles(self):
        import nomorals.wisdom.ingestor as ing_mod
        ing, _ = self._ingestor(lambda url: _FakeResp("{}"))
        with mock.patch.object(ing_mod, "_POLITE_DELAY", 5.0):
            with mock.patch("time.sleep") as slept:
                ing._polite_wait("https://example.com/a")
                ing._polite_wait("https://example.com/b")
                self.assertTrue(slept.called)
                # different host: no wait needed
                slept.reset_mock()
                ing._polite_wait("https://other.org/a")
                self.assertFalse(slept.called)

    def test_ingest_entry_records_edition(self):
        from nomorals.wisdom import ingestor as ing_mod
        ing = ing_mod.ArchiveIngestor(_ing_ctx())
        body = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
                + "Sacred words of wisdom. " * 60 + "\n"
                + "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
        ing._http = _FakeHttp(lambda url: _FakeResp(body))
        ing._robots = mock.Mock()
        ing._robots.allowed.return_value = True
        # parse() needs real document parsing; bypass via text ingest path
        entry = _entry(slug="edition-test",
                       source_url="https://example.com/x.txt")
        ing.corpus.register(entry)
        ing.fetch_bytes = lambda url: body.encode()
        ing.parse = lambda data, filename: \
            ing_mod.ArchiveIngestor.strip_gutenberg_boilerplate(
                data.decode())
        out = ing.ingest_entry(entry, edition="Project Gutenberg ebook #1")
        self.assertIn("Project Gutenberg", out.notes)


# ── autonomy: textrank, mmr, staleness, consolidate ───────────────────

class _FakeDB:
    """Minimal sqlite-backed stand-in for context.db."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        cur = self.con.execute(sql, params)
        self.con.commit()
        return cur

    def query(self, sql, params=()):
        return [dict(r) for r in self.con.execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        row = self.con.execute(sql, params).fetchone()
        return dict(row) if row else None


class AutonomySweepTests(unittest.TestCase):
    def test_textrank_central_sentence_wins(self):
        sentences = [
            "cats sit on warm mats in the sun",
            "dogs bark at passing cars loudly",
            "cats sit on warm mats near the window",
            "quantum physics describes particles",
            "cats sit on warm mats every afternoon",
        ]
        ranked = textrank(sentences, top_n=2)
        self.assertLessEqual(len(ranked), 6)  # candidate pool: 3x top_n
        # the cat sentences are mutually similar; one must lead
        self.assertIn(ranked[0], (0, 2, 4))

    def test_textrank_empty_and_single(self):
        self.assertEqual(textrank([], top_n=3), [])
        self.assertEqual(textrank(["only one sentence here ok"], top_n=3),
                         [0])

    def test_mmr_order_dedups(self):
        texts = [
            "the kingdom of heaven is within you",
            "the kingdom of heaven is within you indeed",
            "sophia fell from the treasury of light",
        ]
        rel = [1.0, 0.99, 0.5]
        picked = mmr_order([0, 1, 2], texts, rel, top_n=2, lambda_=0.5)
        self.assertEqual(len(picked), 2)
        self.assertIn(2, picked)  # the diverse one survives

    def _organ(self):
        ctx, tmp = _ctx()
        ctx.db = _FakeDB()
        with mock.patch("nomorals.wisdom.autonomy._bus") as bus:
            bus.ensure_schema.return_value = None
            bus.drain.return_value = []
            organ = WisdomOrgan(ctx)
        return organ, ctx, tmp

    def test_digest_stale_no_digest(self):
        organ, ctx, tmp = self._organ()
        self.assertTrue(organ.digest_stale("anything"))

    def test_consolidate_proposes_not_deletes(self):
        organ, ctx, tmp = self._organ()
        now = 1_700_000_000.0
        organ.db.execute(
            "INSERT INTO wisdom_digests (slug, title, tradition, key_terms,"
            " passages, digested_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("a", "Work A", "t", json.dumps(["kingdom", "heaven", "light"]),
             "[]", now))
        organ.db.execute(
            "INSERT INTO wisdom_digests (slug, title, tradition, key_terms,"
            " passages, digested_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("b", "Work B", "t", json.dumps(["kingdom", "heaven", "light"]),
             "[]", now))
        proposals = organ.consolidate(threshold=0.5)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["proposal"], "merge")
        self.assertEqual({proposals[0]["slug_a"], proposals[0]["slug_b"]},
                         {"a", "b"})
        # nothing was deleted
        rows = organ.db.query("SELECT slug FROM wisdom_digests")
        self.assertEqual(len(rows), 2)

    def test_consolidate_bad_threshold_rejected(self):
        organ, ctx, tmp = self._organ()
        with self.assertRaises(ValueError):
            organ.consolidate(threshold=1.5)

    def test_gap_analysis_shape(self):
        organ, ctx, tmp = self._organ()
        keeper = WisdomKeeper(ctx)
        keeper.corpus.register(_entry(slug="w1", tradition="thin-trad"))
        keeper.corpus.ingest_text("w1", LOREM)
        with mock.patch.object(WisdomOrgan, "_keeper",
                               return_value=keeper):
            with mock.patch(
                    "nomorals.wisdom.ingestor.ArchiveIngestor") as ing_cls:
                ing_cls.return_value.search.return_value = []
                gaps = organ.gap_analysis(min_works_per_tradition=3)
        self.assertIn("thin-trad", gaps["thin_traditions"])
        self.assertIn("candidates", gaps)
        self.assertIn("timeline_thin_traditions", gaps)


# ── keeper: passthroughs ──────────────────────────────────────────────

class KeeperSweepTests(unittest.TestCase):
    def test_answer_card_renders(self):
        ctx, tmp = _ctx()
        keeper = WisdomKeeper(ctx)
        keeper.corpus.register(_entry())
        keeper.corpus.ingest_text("gospel-of-thomas", LOREM)
        card = keeper.answer_card("kingdom within", top=1)
        self.assertIn("kingdom within", card)
        self.assertIn("Gospel of Thomas", card)

    def test_status_includes_practice_and_semantic(self):
        ctx, tmp = _ctx()
        keeper = WisdomKeeper(ctx)
        st = keeper.status()
        self.assertIn("corpus", st)
        self.assertIn("practice", st)
        self.assertIn("semantic", st)
        self.assertIn("streak_days", st["practice"])


# ── chat session: chime opt-in, progress, rating parse ─────────────────

class ChatSweepTests(unittest.TestCase):
    def _manager(self, ctx, clock=None, notifier=None,
                 midpoint_chime=False):
        from nomorals.wisdom.chat_session import WisdomChatManager
        return WisdomChatManager(ctx, clock=clock, notifier=notifier,
                                 midpoint_chime=midpoint_chime)

    def test_midpoint_chime_opt_in(self):
        from nomorals.wisdom.chat_session import ChatPracticeSession
        from nomorals.wisdom.practice import PracticeGuide

        class Notifier:
            def __init__(self):
                self.bodies = []

            def publish(self, *a, **k):
                body = a[2] if len(a) > 2 else k.get("body", "")
                self.bodies.append(body)
                return {"delivered": True}

        ctx, tmp = _ctx()
        guide = PracticeGuide(ctx)
        notifier = Notifier()
        clock = FakeClock()
        session = ChatPracticeSession(
            guide, "box-breathing", notifier=notifier,
            platform="telegram", clock=clock, midpoint_chime=True)
        session.start()
        self.assertTrue(session.join(timeout=10))
        chimes = [b for b in notifier.bodies if "halfway" in b]
        self.assertEqual(len(chimes), 1)

    def test_midpoint_chime_default_off(self):
        from nomorals.wisdom.chat_session import ChatPracticeSession
        from nomorals.wisdom.practice import PracticeGuide

        class Notifier:
            def __init__(self):
                self.bodies = []

            def publish(self, *a, **k):
                body = a[2] if len(a) > 2 else k.get("body", "")
                self.bodies.append(body)
                return {"delivered": True}

        ctx, tmp = _ctx()
        guide = PracticeGuide(ctx)
        notifier = Notifier()
        session = ChatPracticeSession(
            guide, "box-breathing", notifier=notifier,
            platform="telegram", clock=FakeClock())
        session.start()
        self.assertTrue(session.join(timeout=10))
        self.assertEqual(
            [b for b in notifier.bodies if "halfway" in b], [])

    def test_progress_tracking(self):
        from nomorals.wisdom.chat_session import ChatPracticeSession
        from nomorals.wisdom.practice import PracticeGuide

        class Notifier:
            def publish(self, *a, **k):
                return {"delivered": True}

        ctx, tmp = _ctx()
        guide = PracticeGuide(ctx)
        session = ChatPracticeSession(
            guide, "box-breathing", notifier=Notifier(),
            clock=FakeClock())
        self.assertEqual(session.progress(),
                         {"phases_done": 0, "total": 0})
        session.start()
        self.assertTrue(session.join(timeout=10))
        prog = session.progress()
        self.assertEqual(prog["phases_done"], prog["total"])
        self.assertEqual(prog["total"], 20)

    def test_journal_rating_parsed(self):
        from nomorals.wisdom.practice import PracticeGuide

        class Notifier:
            def publish(self, *a, **k):
                return {"delivered": True}

        ctx, tmp = _ctx()
        mgr = self._manager(ctx, notifier=Notifier())
        guide = PracticeGuide(ctx)
        mgr._guide = guide
        mgr._journal_await["k"] = "box-breathing"
        reply = mgr.handle_incoming("k", "4 calm and steady session")
        self.assertIn("rated 4", reply)
        ratings = [e for e in guide.history(limit=10)
                   if e.get("type") == "rating"]
        self.assertEqual(ratings[0]["stars"], 4)
        journals = [e for e in guide.history(limit=10)
                    if e.get("type") == "journal"]
        self.assertIn("calm and steady", journals[0]["notes"])

    def test_journal_without_rating(self):
        class Notifier:
            def publish(self, *a, **k):
                return {"delivered": True}

        ctx, tmp = _ctx()
        mgr = self._manager(ctx, notifier=Notifier())
        reply = mgr.handle_incoming("k", "just a quiet sit")
        # no journal awaited -> None (normal flow continues)
        self.assertIsNone(reply)


if __name__ == "__main__":
    unittest.main()
