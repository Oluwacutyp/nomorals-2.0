"""God-tier arena topics: rich tables, personalization, random rotation.

Covers the dynamic topic system in nomorals/agents/arena/:
- the expanded bank (size, categories, per-entry metadata)
- weighted/personalized sampling + determinism with a seeded RNG
- used-topic skipping and the anti-repeat category window
- the topic anti-repeat window (persisted, reshuffles on exhaustion)
- topic packs: registration, weights, replacement, removal
- surprise mode (pure random, reproducible with a seed)
- the interest profiler in activity.py (search/custom/goal/command signals)
- the table renderers behind /arena topics
"""
from __future__ import annotations

import os
import random
import tempfile
import unittest

from nomorals.agents.arena import activity
from nomorals.agents.arena.topics import (
    CATEGORIES,
    DEFAULT_ANTI_REPEAT,
    TOPIC_BANK,
    TopicPack,
    all_categories,
    anti_repeat_window,
    bank_size,
    category_stats,
    category_table,
    clear_recent_topics,
    recent_topics,
    register_topic_pack,
    sample_topic,
    set_anti_repeat_window,
    surprise_topic,
    topic_packs,
    topic_text,
    topics_in,
    topics_table,
    unregister_topic_pack,
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


# ── topic packs ─────────────────────────────────────────────────────────

class PackTests(unittest.TestCase):
    def tearDown(self):
        for name in ("pack_a", "pack_b", "tiny", "strpack"):
            unregister_topic_pack(name)

    def test_register_adds_topics_and_category(self):
        before = bank_size()
        pack = register_topic_pack("pack_a", {
            "packcat": [
                {"t": "pack topic one about testing", "d": 1, "tags": ("x",)},
                "pack topic two, plain string entry",
            ],
        })
        self.assertIsInstance(pack, TopicPack)
        self.assertEqual(pack.name, "pack_a")
        self.assertIn("pack_a", topic_packs())
        self.assertIn("packcat", all_categories())
        self.assertEqual(bank_size(), before + 2)
        entries = topics_in("packcat")
        self.assertEqual(len(entries), 2)
        # string entry normalized: difficulty 2, no tags
        plain = next(e for e in entries
                     if topic_text(e) == "pack topic two, plain string entry")
        self.assertEqual(int(plain["d"]), 2)
        self.assertEqual(tuple(plain["tags"]), ())

    def test_pack_weights_steer_sampling(self):
        register_topic_pack("pack_b", {
            "packcat": [{"t": f"weighted pack topic {i}", "d": 2,
                         "tags": ()} for i in range(6)],
        }, weights={"packcat": 40.0})
        cats = [sample_topic(rng=random.Random(n))[0] for n in range(40)]
        share = sum(1 for c in cats if c == "packcat") / 40
        self.assertGreater(share, 0.5)

    def test_reregister_replaces_pack(self):
        register_topic_pack("pack_a", {"packcat": ["only topic alpha"]})
        register_topic_pack("pack_a", {"packcat": ["only topic beta"]})
        texts = [topic_text(e) for e in topics_in("packcat")]
        self.assertEqual(texts, ["only topic beta"])

    def test_unregister_removes_pack(self):
        register_topic_pack("pack_a", {"packcat": ["temporary topic"]})
        self.assertTrue(unregister_topic_pack("pack_a"))
        self.assertNotIn("pack_a", topic_packs())
        self.assertNotIn("packcat", all_categories())
        self.assertFalse(unregister_topic_pack("pack_a"))  # twice → False

    def test_core_pack_is_protected(self):
        self.assertFalse(unregister_topic_pack("core"))
        self.assertIn("core", topic_packs())

    def test_cross_pack_duplicates_deduped(self):
        victim = topic_text(TOPIC_BANK["ai"][0])
        before = bank_size()
        register_topic_pack("pack_a", {"ai": [victim, "brand new topic xyz"]})
        self.assertEqual(bank_size(), before + 1)  # dup skipped, new kept

    def test_register_validates_input(self):
        with self.assertRaises(ValueError):
            register_topic_pack("", {"c": ["t"]})
        with self.assertRaises(ValueError):
            register_topic_pack("pack_a", {})
        with self.assertRaises(ValueError):
            register_topic_pack("pack_a", {"c": []})
        with self.assertRaises(ValueError):
            register_topic_pack("pack_a", {"c": [""]})
        with self.assertRaises(ValueError):
            register_topic_pack("pack_a", {"c": [123]})
        with self.assertRaises(ValueError):
            register_topic_pack("pack_a", {"c": ["ok"]},
                                weights={"c": "heavy"})


# ── topic anti-repeat window ──────────────────────────────────────────────

class AntiRepeatTests(unittest.TestCase):
    def tearDown(self):
        unregister_topic_pack("tiny")

    def test_no_topic_repeats_within_window(self):
        db = make_db()
        window = 8
        seq = [sample_topic(db=db, rng=random.Random(n),
                            anti_repeat=window)[1] for n in range(30)]
        for i, t in enumerate(seq):
            self.assertNotIn(t, seq[max(0, i - window):i],
                             f"topic repeated within window at {i}: {t}")

    def test_window_is_configurable_per_call(self):
        db = make_db()
        seq = [sample_topic(db=db, rng=random.Random(n), anti_repeat=3)[1]
               for n in range(20)]
        for i, t in enumerate(seq):
            self.assertNotIn(t, seq[max(0, i - 3):i])

    def test_window_persisted_via_kv(self):
        db = make_db()
        self.assertTrue(set_anti_repeat_window(db, 4))
        self.assertEqual(anti_repeat_window(db), 4)
        seq = [sample_topic(db=db, rng=random.Random(n))[1]
               for n in range(16)]
        for i, t in enumerate(seq):
            self.assertNotIn(t, seq[max(0, i - 4):i])

    def test_window_zero_disables_tracking(self):
        db = make_db()
        for n in range(5):
            sample_topic(db=db, rng=random.Random(n), anti_repeat=0)
        self.assertEqual(recent_topics(db), [])

    def test_recent_topics_recorded_newest_first(self):
        db = make_db()
        seen = [sample_topic(db=db, rng=random.Random(n))[1]
                for n in range(4)]
        self.assertEqual(recent_topics(db), seen[::-1])

    def test_default_window_is_sensible(self):
        self.assertGreaterEqual(DEFAULT_ANTI_REPEAT, 8)
        self.assertLessEqual(DEFAULT_ANTI_REPEAT, 12)

    def test_reshuffle_on_exhaustion(self):
        db = make_db()
        register_topic_pack("tiny", {"tinycat": ["tiny topic one",
                                                 "tiny topic two"]})
        got = [sample_topic(db=db, rng=random.Random(n), category="tinycat",
                            anti_repeat=5)[1] for n in range(3)]
        # 2 topics, window 5: the 3rd sample must reshuffle, not fail
        self.assertEqual(set(got[:2]), {"tiny topic one", "tiny topic two"})
        self.assertIn(got[2], {"tiny topic one", "tiny topic two"})
        # after reshuffle the window holds just the newest pick
        self.assertEqual(len(recent_topics(db)), 1)

    def test_true_exhaustion_falls_back(self):
        db = make_db()
        register_topic_pack("tiny", {"tinycat": ["tiny topic one",
                                                 "tiny topic two"]})
        # A dry forced category falls back to another category's topic —
        # the cycle must never fail and never serve a used topic.
        with db.transaction():
            for i, t in enumerate(["tiny topic one", "tiny topic two"]):
                db.execute(
                    "INSERT INTO arena_knowledge (id, topic, category, digest, sources, created_at)"
                    " VALUES (?, ?, 'tinycat', 'd', '[]', 0)",
                    (f"e{i}", t))
        cat, topic = sample_topic(db=db, rng=random.Random(1),
                                  category="tinycat", anti_repeat=5)
        self.assertNotEqual(cat, "tinycat")
        self.assertTrue(topic)
        self.assertNotIn(topic, {"tiny topic one", "tiny topic two"})

    def test_bank_fully_digested_never_fails(self):
        db = make_db()
        with db.transaction():
            i = 0
            for entries in TOPIC_BANK.values():
                for e in entries:
                    db.execute(
                        "INSERT INTO arena_knowledge (id, topic, category, digest, sources, created_at)"
                        " VALUES (?, ?, 'c', 'd', '[]', 0)",
                        (f"u{i}", topic_text(e)))
                    i += 1
        cat, topic = sample_topic(db=db, rng=random.Random(1))
        self.assertEqual((cat, topic), (CATEGORIES[0], "general computing"))

    def test_clear_recent_topics(self):
        db = make_db()
        sample_topic(db=db, rng=random.Random(1))
        sample_topic(db=db, rng=random.Random(2))
        self.assertGreater(clear_recent_topics(db), 0)
        self.assertEqual(recent_topics(db), [])


# ── persistence across restarts ───────────────────────────────────────────

class PersistenceTests(unittest.TestCase):
    def test_recent_history_survives_restart(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            db = Database(path)
            db.migrate()
            seen = [sample_topic(db=db, rng=random.Random(n))[1]
                    for n in range(6)]
            before = recent_topics(db)
            self.assertEqual(before, seen[::-1])
            db.close()

            # "restart": fresh handle on the same file
            db2 = Database(path)
            db2.migrate()
            self.assertEqual(recent_topics(db2), before)
            # and the window is still honored after the restart
            nxt = sample_topic(db=db2, rng=random.Random(99))[1]
            self.assertNotIn(nxt, before)
            db2.close()
        finally:
            os.unlink(path)

    def test_window_setting_survives_restart(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            db = Database(path)
            db.migrate()
            set_anti_repeat_window(db, 7)
            db.close()
            db2 = Database(path)
            db2.migrate()
            self.assertEqual(anti_repeat_window(db2), 7)
            db2.close()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
