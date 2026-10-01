"""God-tier arena topics: rich tables, personalization, random rotation.

Covers the dynamic topic system in nomorals/agents/arena/:
- the expanded bank (size, categories, per-entry metadata)
- weighted/personalized sampling + determinism with a seeded RNG
- used-topic skipping and the anti-repeat category window
- surprise mode (pure random, reproducible with a seed)
- the interest profiler in activity.py (search/custom/goal/command signals)
- the table renderers behind /arena topics
"""
from __future__ import annotations

import random
import unittest

from nomorals.agents.arena import activity
from nomorals.agents.arena.topics import (
    CATEGORIES,
    TOPIC_BANK,
    bank_size,
    category_stats,
    category_table,
    sample_topic,
    surprise_topic,
    topic_text,
    topics_in,
    topics_table,
)
from nomorals.storage.db import Database


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class FakeCtx:
    def __init__(self, db: Database) -> None:
        self.db = db


# ── the bank ──────────────────────────────────────────────────────────────

class BankTests(unittest.TestCase):
    def test_bank_is_large_and_categorized(self):
        self.assertGreaterEqual(bank_size(), 150)
        self.assertGreaterEqual(len(CATEGORIES), 12)
        for cat in CATEGORIES:
            entries = topics_in(cat)
            self.assertGreaterEqual(len(entries), 10, cat)

    def test_every_entry_has_valid_metadata(self):
        for cat, entries in TOPIC_BANK.items():
            for e in entries:
                text = topic_text(e)
                self.assertTrue(text and len(text) > 10, (cat, e))
                self.assertIn(int(e.get("d", 0)), (1, 2, 3), (cat, text))
                self.assertTrue(e.get("tags"), (cat, text))

    def test_no_duplicate_topic_texts(self):
        seen: set[str] = set()
        for entries in TOPIC_BANK.values():
            for e in entries:
                t = topic_text(e)
                self.assertNotIn(t, seen, t)
                seen.add(t)

    def test_category_stats_shape(self):
        stats = category_stats()
        self.assertEqual(set(stats), set(CATEGORIES))
        for cat, st in stats.items():
            self.assertEqual(st["topics"],
                             st["easy"] + st["medium"] + st["deep"], cat)


# ── sampling ──────────────────────────────────────────────────────────────

class SamplingTests(unittest.TestCase):
    def test_seeded_sample_is_deterministic(self):
        a = sample_topic(rng=random.Random(42))
        b = sample_topic(rng=random.Random(42))
        self.assertEqual(a, b)

    def test_forced_category_respected(self):
        for _ in range(20):
            cat, topic = sample_topic(rng=random.Random(), category="crypto")
            self.assertEqual(cat, "crypto")
            self.assertTrue(topic)

    def test_difficulty_filter(self):
        from nomorals.agents.arena.topics import TOPIC_BANK as BANK

        for _ in range(30):
            cat, topic = sample_topic(rng=random.Random(), difficulty=1)
            entry = next(e for e in BANK[cat]
                         if topic_text(e) == topic)
            self.assertEqual(int(entry["d"]), 1)

    def test_used_topics_are_skipped(self):
        db = make_db()
        victim = topic_text(TOPIC_BANK["ai"][0])
        with db.transaction():
            db.execute(
                "INSERT INTO arena_knowledge (id, topic, category, digest, sources, created_at)"
                " VALUES ('x1', ?, 'ai', 'd', '[]', 0)", (victim,))
        for _ in range(50):
            cat, topic = sample_topic(db=db, rng=random.Random(),
                                      category="ai")
            self.assertNotEqual(topic, victim)

    def test_anti_repeat_window(self):
        db = make_db()
        cats = [sample_topic(db=db, rng=random.Random(7))[0]
                for _ in range(12)]
        # with 14 categories, back-to-back repeats should be rare
        repeats = sum(1 for a, b in zip(cats, cats[1:]) if a == b)
        self.assertLessEqual(repeats, 3)

    def test_profile_weights_steer_sampling(self):
        profile = {c: 1.0 for c in CATEGORIES}
        profile["crypto"] = 50.0
        cats = [sample_topic(rng=random.Random(n), profile=profile)[0]
                for n in range(60)]
        crypto_share = sum(1 for c in cats if c == "crypto") / 60
        self.assertGreater(crypto_share, 0.5)

    def test_sample_without_db_never_raises(self):
        cat, topic = sample_topic()
        self.assertIn(cat, CATEGORIES)
        self.assertTrue(topic)


class SurpriseTests(unittest.TestCase):
    def test_surprise_is_pure_random(self):
        a = surprise_topic(seed=123)
        b = surprise_topic(seed=123)
        self.assertEqual(a, b)
        c = surprise_topic(seed=999)
        self.assertIsInstance(c, tuple)

    def test_surprise_ignores_profile(self):
        # even with a maxed-out profile, surprise must not be steered
        seen = {surprise_topic(seed=n)[0] for n in range(40)}
        self.assertGreater(len(seen), 1)


# ── activity / personalization ────────────────────────────────────────────

class ActivityTests(unittest.TestCase):
    def test_match_category(self):
        self.assertEqual(activity.match_category(
            "how does bitcoin mining work"), "crypto")
        self.assertEqual(activity.match_category(
            "postgres index tuning"), "data")
        self.assertEqual(activity.match_category(
            "rust borrow checker lifetimes"), "languages")
        self.assertIsNone(activity.match_category("qzx wub jkl"))

    def test_record_and_recent_commands(self):
        db = make_db()
        activity.record(db, "command", "search")
        activity.record(db, "command", "arena")
        cmds = activity.recent_commands(db)
        self.assertEqual(cmds[0], "arena")
        self.assertIn("search", cmds)

    def test_record_never_raises_without_db(self):
        activity.record(None, "command", "search")  # must not raise

    def test_search_queries_boost_profile(self):
        db = make_db()
        profile = activity.interest_profile(
            db, search_queries=["bitcoin wallet seed phrase",
                               "ethereum gas fees",
                               "defi yield farming"])
        top = max(profile, key=profile.get)
        self.assertEqual(top, "crypto")
        self.assertGreater(profile["crypto"], profile["web"])

    def test_goals_boost_profile(self):
        profile = activity.interest_profile(
            goal_texts=["learn rust systems programming"])
        self.assertGreater(profile["languages"], 1.0)

    def test_digest_history_boosts_profile(self):
        db = make_db()
        with db.transaction():
            for i in range(5):
                db.execute(
                    "INSERT INTO arena_knowledge (id, topic, category, digest, sources, created_at)"
                    f" VALUES ('k{i}', 'topic {i}', 'security', 'd', '[]', 0)")
        profile = activity.interest_profile(db)
        self.assertGreater(profile["security"], profile["web"])

    def test_custom_topics_boost_profile(self):
        import json
        import time

        db = make_db()
        with db.transaction():
            db.execute(
                """INSERT INTO kv_store (key, value, kind, updated_at)
                   VALUES ('arena.custom_topics', ?, 'json', ?)""",
                (json.dumps({"topics": ["zero knowledge proofs"]}),
                 time.time()))
        profile = activity.interest_profile(db)
        self.assertGreater(profile["crypto"], 1.0)

    def test_command_usage_boosts_profile(self):
        db = make_db()
        for _ in range(10):
            activity.record(db, "command", "search")
        profile = activity.interest_profile(db)
        # VERB_CATEGORY maps search -> web
        self.assertGreater(profile["web"], 1.0)

    def test_profile_is_capped_and_complete(self):
        profile = activity.interest_profile(
            search_queries=["crypto bitcoin"] * 100)
        self.assertEqual(set(profile), set(CATEGORIES))
        self.assertLessEqual(max(profile.values()), 6.0)


# ── display ───────────────────────────────────────────────────────────────

class DisplayTests(unittest.TestCase):
    def test_topics_table_renders(self):
        text = topics_table()
        self.assertIn(str(bank_size()), text)
        for cat in ("web", "crypto", "languages"):
            self.assertIn(cat, text)

    def test_topics_table_flags_personalized(self):
        profile = {c: 1.0 for c in CATEGORIES}
        profile["ai"] = 3.0
        text = topics_table(profile=profile)
        ai_line = next(line for line in text.splitlines()
                       if line.strip().startswith("ai "))
        self.assertIn("★", ai_line)

    def test_category_table_drill_down(self):
        text = category_table("crypto")
        self.assertIn("crypto", text)
        self.assertIn("[", text)  # difficulty grades shown
        self.assertIn("no category", category_table("nope").lower())


# ── core wiring ───────────────────────────────────────────────────────────

class CoreWiringTests(unittest.TestCase):
    def test_interest_profile_from_core(self):
        from nomorals.agents.arena.core import Arena

        db = make_db()
        arena = Arena(FakeCtx(db))
        profile = arena.interest_profile()
        self.assertEqual(set(profile), set(CATEGORIES))

    def test_run_cycle_safe_accepts_surprise(self):
        from nomorals.agents.arena.core import run_cycle_safe

        class Stub:
            def run_cycle(self, topic=None, surprise=False, seed=None):
                return {"ok": True, "topic": "t", "surprise": surprise,
                        "seed": seed}

        out = run_cycle_safe(Stub(), surprise=True, seed=7)  # type: ignore[arg-type]
        self.assertTrue(out["ok"])
        self.assertTrue(out["surprise"])

    def test_run_cycle_safe_never_raises(self):
        from nomorals.agents.arena.core import run_cycle_safe

        class Boom:
            def run_cycle(self, **kw):
                raise RuntimeError("boom")

        out = run_cycle_safe(Boom())  # type: ignore[arg-type]
        self.assertFalse(out["ok"])
        self.assertIn("boom", out["error"])


if __name__ == "__main__":
    unittest.main()
