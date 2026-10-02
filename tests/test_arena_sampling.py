"""Adaptive arena sampler: profile + coverage + difficulty.

Covers nomorals/agents/arena/sampling.py and the activity.py keyword
extension:
- sample_challenge returns (category, topic_text, full_entry) with the
  challenge schema keys, persists sampling_state, is deterministic
  with a seeded rng
- the fallback entry schema when the bank lookup misses
- coverage boosts flow into sampling_state weights
- difficulty targets steer forced-category sampling
- CATEGORY_KEYWORDS covers every new challenge category and
  match_category resolves each one
- never raises on db=None
"""
from __future__ import annotations

import random
import unittest
from unittest import mock

from nomorals.agents.arena import activity, sampling, scoring, topics
from nomorals.storage.db import Database


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


#: One distinctive keyword per new challenge category (must resolve
#: via match_category to exactly that category).
NEW_CATEGORY_PROBES = {
    "systems_design": "system design",
    "distributed": "raft",
    "concurrency": "deadlock",
    "databases": "query plan",
    "networking": "bgp",
    "compilers": "codegen",
    "runtimes": "garbage collection",
    "packaging": "wheel",
    "web_fullstack": "fullstack",
    "apis": "graphql",
    "auth": "oauth",
    "realtime": "sse",
    "mobile": "flutter",
    "termux": "termux",
    "performance": "throughput",
    "memory": "allocator",
    "security_eng": "hardening",
    "threat_modeling": "stride",
    "secure_defaults": "default deny",
    "ml_ops": "mlops",
    "evals": "leaderboard",
    "finetune": "qlora",
    "rag": "rerank",
    "agents": "multi-agent",
    "ux_chat": "chatbot",
    "latency": "p99",
    "routing": "load balancer",
    "media_pipelines": "ffmpeg",
    "ocr": "tesseract",
    "vision_edit": "inpainting",
    "research_methods": "methodology",
    "source_trust": "provenance",
    "distillation": "distillation",
}


# ── sample_challenge ──────────────────────────────────────────────────────

class SampleChallengeTests(unittest.TestCase):
    def test_returns_triple_with_entry_schema(self):
        db = make_db()
        cat, text, entry = sampling.sample_challenge(
            db, rng=random.Random(7), anti_repeat=0)
        self.assertIsInstance(cat, str)
        self.assertIsInstance(text, str)
        self.assertTrue(text)
        self.assertIsInstance(entry, dict)
        for key in ("t", "d", "tags", "verify", "kind"):
            self.assertIn(key, entry, key)
        self.assertEqual(entry["t"], text)
        self.assertIn(int(entry["d"]), (1, 2, 3))

    def test_persists_sampling_state(self):
        db = make_db()
        cat, text, entry = sampling.sample_challenge(
            db, rng=random.Random(7), anti_repeat=0)
        state = sampling.sampling_state(db)
        self.assertEqual(state["category"], cat)
        self.assertIn(state["difficulty"], (1, 2, 3))
        self.assertIn("ts", state)
        self.assertIsInstance(state["weights"], dict)
        self.assertIn(cat, state["weights"])

    def test_deterministic_with_seeded_rng(self):
        first = sampling.sample_challenge(
            make_db(), rng=random.Random(42), anti_repeat=0)
        second = sampling.sample_challenge(
            make_db(), rng=random.Random(42), anti_repeat=0)
        self.assertEqual((first[0], first[1]), (second[0], second[1]))
        self.assertEqual(first[2]["t"], second[2]["t"])

    def test_fallback_entry_when_bank_lookup_misses(self):
        db = make_db()
        with mock.patch.object(topics, "topics_in", return_value=[]):
            cat, text, entry = sampling.sample_challenge(
                db, rng=random.Random(3), anti_repeat=0)
        self.assertEqual(entry["t"], text)
        self.assertEqual(entry["tags"], ())
        self.assertEqual(entry["verify"], "")
        self.assertEqual(entry["kind"], "code")
        self.assertIn(int(entry["d"]), (1, 2, 3))

    def test_coverage_boost_flows_into_weights(self):
        db = make_db()
        for i in range(4):
            scoring.record_score(db, topic=f"w{i}", category="web",
                                 tests_passed=True)
        sampling.sample_challenge(db, rng=random.Random(9), anti_repeat=0)
        weights = sampling.sampling_state(db)["weights"]
        self.assertAlmostEqual(weights["web"], 1.0)
        self.assertAlmostEqual(weights["ai"], 3.0)

    def test_explicit_profile_is_base_signal(self):
        db = make_db()
        profile = {"security": 5.0}
        cat, text, entry = sampling.sample_challenge(
            db, profile=profile, rng=random.Random(11), anti_repeat=0)
        weights = sampling.sampling_state(db)["weights"]
        self.assertGreater(weights["security"], weights["ai"])

    def test_forced_category_difficulty_target(self):
        db = make_db()
        # web is aced (avg 1.0 over 3 runs) → difficulty 3
        for i in range(3):
            scoring.record_score(db, topic=f"w{i}", category="web",
                                 research_usefulness=1.0,
                                 build_compiled=True, tests_passed=True,
                                 edit_precision=1.0, latency_s=1.0)
        cat, text, entry = sampling.sample_challenge(
            db, category="web", rng=random.Random(5), anti_repeat=0)
        state = sampling.sampling_state(db)
        self.assertEqual(state["difficulty"], 3)
        # topic came from the web bank and matches its text
        bank_texts = {topics.topic_text(e) for e in topics.topics_in(cat)}
        self.assertIn(text, bank_texts)

    def test_never_raises_on_none_db(self):
        cat, text, entry = sampling.sample_challenge(
            None, rng=random.Random(1), anti_repeat=0)
        self.assertTrue(cat and text and entry)
        self.assertEqual(sampling.sampling_state(None), {})


# ── sampling_state ────────────────────────────────────────────────────────

class SamplingStateTests(unittest.TestCase):
    def test_empty_when_absent(self):
        self.assertEqual(sampling.sampling_state(make_db()), {})

    def test_roundtrip(self):
        db = make_db()
        self.assertTrue(sampling._persist_state(
            db, {"ts": 1.0, "category": "ai", "difficulty": 2,
                 "weights": {"ai": 1.5}}))
        state = sampling.sampling_state(db)
        self.assertEqual(state["category"], "ai")
        self.assertEqual(state["difficulty"], 2)
        self.assertAlmostEqual(state["weights"]["ai"], 1.5)

    def test_persist_never_raises_on_none_db(self):
        self.assertFalse(sampling._persist_state(None, {}))


# ── activity keywords ─────────────────────────────────────────────────────

class CategoryKeywordTests(unittest.TestCase):
    def test_every_new_category_covered(self):
        for cat in NEW_CATEGORY_PROBES:
            self.assertIn(cat, activity.CATEGORY_KEYWORDS, cat)
            kws = activity.CATEGORY_KEYWORDS[cat]
            self.assertGreaterEqual(len(kws), 6, cat)
            self.assertLessEqual(len(kws), 10, cat)
            for kw in kws:
                self.assertEqual(kw, kw.lower(), (cat, kw))

    def test_match_category_resolves_each(self):
        for cat, probe in NEW_CATEGORY_PROBES.items():
            self.assertEqual(activity.match_category(probe), cat,
                             (cat, probe))

    def test_word_boundary_safety(self):
        # "ram" must not fire on "programming"; "go" not on "golang"
        self.assertNotEqual(activity.match_category("programming"), "memory")
        self.assertEqual(activity.match_category("golang generics"), "languages")

    def test_new_verbs(self):
        self.assertEqual(activity.VERB_CATEGORY["benchmark"], "performance")
        self.assertEqual(activity.VERB_CATEGORY["finetune"], "finetune")
        self.assertEqual(activity.VERB_CATEGORY["evaluate"], "evals")
        self.assertEqual(activity.VERB_CATEGORY["route"], "routing")
        self.assertEqual(activity.VERB_CATEGORY["distill"], "distillation")
        self.assertEqual(activity.VERB_CATEGORY["profile"], "performance")


if __name__ == "__main__":
    unittest.main()
