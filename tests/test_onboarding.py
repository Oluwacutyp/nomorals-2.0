"""Invariants for new-player onboarding boost + early quests + fresh start.

Covers nomorals/games/onboarding.py, nomorals/games/early_quests.py
and nomorals/games/fresh_start.py:

* boost is 7 days exactly, multipliers exact, ends on day 8
* tutorial chain order enforced, each step pays immediately
* early-quest chest only on full completion, only once
* boost once per verified identity (same fingerprint, second key: no boost)
* fresh start never auto-imports the old DB; backup never deletes
* profile shows "Day X/7 boost" + end date
"""

from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timezone

from nomorals.games import early_quests
from nomorals.games import fresh_start as fresh
from nomorals.games import onboarding
from nomorals.storage.db import Database

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400


def make_db() -> Database:
    return Database(":memory:")


def make_rewards():
    calls: list[tuple[str, int, int, str]] = []

    def reward(player_key: str, xp: int, coins: int, reason: str) -> None:
        calls.append((player_key, xp, coins, reason))

    return calls, reward


# ── boost: 7 days exactly ───────────────────────────────────────────────────

class BoostLengthTests(unittest.TestCase):
    def test_seven_days_exactly_active_then_ends_day_8(self):
        db = make_db()
        res = onboarding.start_boost(db, "telegram:1", now=T0)
        self.assertTrue(res["ok"])
        for day_offset in range(7):
            boost = onboarding.get_boost(db, "telegram:1", now=T0 + day_offset * DAY)
            assert boost is not None
            self.assertTrue(boost["active"], f"day {day_offset + 1} must be active")
            self.assertEqual(boost["day"], day_offset + 1)
            self.assertEqual(boost["xp_multiplier"], 2.0)
            self.assertEqual(boost["coin_multiplier"], 1.5)
        # Day 8: over. Multipliers back to 1.0.
        boost = onboarding.get_boost(db, "telegram:1", now=T0 + 7 * DAY)
        assert boost is not None
        self.assertFalse(boost["active"])
        self.assertEqual(boost["day"], 8)
        self.assertEqual(boost["xp_multiplier"], 1.0)
        self.assertEqual(boost["coin_multiplier"], 1.0)
        self.assertEqual(boost["days_left"], 0)

    def test_multipliers_pure_function(self):
        for day in range(1, 8):
            self.assertEqual(onboarding.xp_multiplier_for_day(day), 2.0)
            self.assertEqual(onboarding.coin_multiplier_for_day(day), 1.5)
        for day in (0, 8, 9, 100, -1):
            self.assertEqual(onboarding.xp_multiplier_for_day(day), 1.0)
            self.assertEqual(onboarding.coin_multiplier_for_day(day), 1.0)

    def test_apply_boost_math(self):
        out = onboarding.apply_boost(100, 100, day=1)
        self.assertEqual(out["total_xp"], 200)
        self.assertEqual(out["total_coins"], 150)
        self.assertEqual(out["bonus_xp"], 100)
        self.assertEqual(out["bonus_coins"], 50)
        out8 = onboarding.apply_boost(100, 100, day=8)
        self.assertEqual(out8["total_xp"], 100)
        self.assertEqual(out8["total_coins"], 100)
        self.assertEqual(out8["bonus_xp"], 0)

    def test_decaying_daily_cap_bounds_bonus_not_base(self):
        # Day 1 bonus cap 1000; day 7 bonus cap 400.
        self.assertEqual(onboarding.daily_boost_cap_xp(1), 1000)
        self.assertEqual(onboarding.daily_boost_cap_xp(7), 400)
        self.assertEqual(onboarding.daily_boost_cap_xp(8), 0)
        out = onboarding.apply_boost(100_000, 100_000, day=1)
        self.assertEqual(out["bonus_xp"], 1000)
        self.assertEqual(out["total_xp"], 100_000 + 1000)
        out7 = onboarding.apply_boost(100_000, 100_000, day=7)
        self.assertEqual(out7["bonus_xp"], 400)
        # Cap already partly consumed today reduces the room left.
        out_used = onboarding.apply_boost(100, 100, day=1,
                                           already_boosted_xp=950)
        self.assertEqual(out_used["bonus_xp"], 50)

    def test_first_win_of_day_once_per_day_only_while_boosted(self):
        db = make_db()
        onboarding.start_boost(db, "telegram:1", now=T0)
        calls, reward = make_rewards()
        first = onboarding.claim_first_win_of_day(db, "telegram:1", reward, now=T0)
        self.assertTrue(first["ok"])
        self.assertEqual((first["xp"], first["coins"]), (50, 100))
        self.assertEqual(len(calls), 1)
        again = onboarding.claim_first_win_of_day(db, "telegram:1", reward, now=T0 + 60)
        self.assertFalse(again["ok"])
        self.assertEqual(again["reason"], "already_claimed_today")
        self.assertEqual(len(calls), 1)
        # Next day, still inside the boost: claimable again.
        nxt = onboarding.claim_first_win_of_day(db, "telegram:1", reward, now=T0 + DAY)
        self.assertTrue(nxt["ok"])
        # Day 8: boost over, no first-win bonus.
        late = onboarding.claim_first_win_of_day(db, "telegram:1", reward, now=T0 + 7 * DAY)
        self.assertFalse(late["ok"])
        self.assertEqual(late["reason"], "boost_not_active")


# ── boost once per verified identity ────────────────────────────────────────

class IdentityTests(unittest.TestCase):
    def test_same_fingerprint_second_player_gets_no_boost(self):
        db = make_db()
        fp = "telegram:123456"
        first = onboarding.start_boost(db, "telegram:alice", fingerprint=fp, now=T0)
        self.assertTrue(first["ok"])
        second = onboarding.start_boost(db, "telegram:bob", fingerprint=fp, now=T0)
        self.assertFalse(second["ok"])
        self.assertEqual(second["reason"], "fingerprint_already_used")
        self.assertIsNone(onboarding.get_boost(db, "telegram:bob", now=T0))
        # The first player's boost is untouched.
        self.assertIsNotNone(onboarding.get_boost(db, "telegram:alice", now=T0))

    def test_same_player_same_fingerprint_is_idempotent(self):
        db = make_db()
        fp = "telegram:123456"
        first = onboarding.start_boost(db, "telegram:alice", fingerprint=fp, now=T0)
        second = onboarding.start_boost(db, "telegram:alice", fingerprint=fp, now=T0 + 10)
        self.assertTrue(second["ok"])
        self.assertTrue(second["already"])
        self.assertEqual(second["boost"]["started_at"], first["boost"]["started_at"])

    def test_different_fingerprint_different_player_ok(self):
        db = make_db()
        self.assertTrue(onboarding.start_boost(db, "telegram:alice",
                                               fingerprint="telegram:1", now=T0)["ok"])
        self.assertTrue(onboarding.start_boost(db, "telegram:bob",
                                               fingerprint="telegram:2", now=T0)["ok"])

    def test_fingerprint_is_hashed_never_stored_raw(self):
        db = make_db()
        onboarding.start_boost(db, "telegram:alice",
                               fingerprint="telegram:123456", now=T0)
        rows = db.query("SELECT fingerprint_hash FROM onboarding_identities")
        self.assertEqual(len(rows), 1)
        stored = rows[0]["fingerprint_hash"]
        self.assertNotIn("telegram:123456", stored)
        self.assertEqual(stored, onboarding.normalize_fingerprint("telegram:123456"))


# ── starter gear + reroll ───────────────────────────────────────────────────

class StarterPackTests(unittest.TestCase):
    def test_starter_gear_cosmetic_only_never_power(self):
        for item in onboarding.starter_gear_pack():
            self.assertEqual(item["power_attack"], 0)
            self.assertEqual(item["power_defense"], 0)
            self.assertEqual(item["power"], 0)
            self.assertEqual(item["cash_value"], 0)
            self.assertFalse(item["tradeable"])
        total_xp = sum(i["xp_bonus_pct"] for i in onboarding.starter_gear_pack())
        self.assertLessEqual(total_xp, 15.0, "starter XP bonus must stay small")

    def test_starter_gear_granted_exactly_once(self):
        db = make_db()
        granted: list[dict] = []

        def grant_fn(player_key, item):
            granted.append(item)

        first = onboarding.grant_starter_gear(db, "telegram:1", grant_fn)
        self.assertTrue(first["ok"])
        self.assertFalse(first["already"])
        self.assertEqual(len(first["granted"]), len(onboarding.STARTER_GEAR_PACK))
        self.assertEqual(len(granted), len(onboarding.STARTER_GEAR_PACK))
        second = onboarding.grant_starter_gear(db, "telegram:1", grant_fn)
        self.assertTrue(second["already"])
        self.assertEqual(second["granted"], [])
        self.assertEqual(len(granted), len(onboarding.STARTER_GEAR_PACK))

    def test_free_reroll_exactly_once(self):
        db = make_db()
        self.assertTrue(onboarding.free_reroll_available(db, "telegram:1"))
        self.assertTrue(onboarding.claim_free_reroll(db, "telegram:1")["ok"])
        self.assertFalse(onboarding.free_reroll_available(db, "telegram:1"))
        again = onboarding.claim_free_reroll(db, "telegram:1")
        self.assertFalse(again["ok"])
        self.assertEqual(again["reason"], "already_claimed")


# ── tutorial chain ──────────────────────────────────────────────────────────

class TutorialTests(unittest.TestCase):
    def test_order_is_math_mix_duel_party_quest(self):
        steps = onboarding.tutorial_steps()
        self.assertEqual([s["step_id"] for s in steps],
                         ["tutorial_math", "tutorial_mix", "tutorial_duel",
                          "tutorial_party", "tutorial_quest"])
        self.assertEqual([s["command"] for s in steps],
                         ["/math", "/mix", "duel", "party", "quest"])

    def test_out_of_order_refused(self):
        db = make_db()
        res = onboarding.complete_tutorial_step(db, "telegram:1", "tutorial_duel")
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "out_of_order")
        self.assertEqual(res["expected"], "tutorial_math")
        res2 = onboarding.complete_tutorial_step(db, "telegram:1", "tutorial_quest")
        self.assertFalse(res2["ok"])

    def test_each_step_pays_immediately_in_order(self):
        db = make_db()
        calls, reward = make_rewards()
        for step in onboarding.tutorial_steps():
            res = onboarding.complete_tutorial_step(
                db, "telegram:1", step["step_id"], reward, now=T0)
            self.assertTrue(res["ok"], step["step_id"])
            # Paid immediately: reward_fn called synchronously in this call.
            self.assertEqual(len(calls), step["position"])
            self.assertEqual(calls[-1], ("telegram:1", step["reward_xp"],
                                         step["reward_coins"],
                                         f"tutorial:{step['step_id']}"))
            self.assertEqual(res["paid"],
                             {"xp": step["reward_xp"], "coins": step["reward_coins"]})
            # Ledger entry exists right now, not deferred.
            kinds = [e["kind"] for e in onboarding.ledger_entries(db, "telegram:1")]
            self.assertIn(f"tutorial:{step['step_id']}", kinds)
        progress = onboarding.tutorial_progress(db, "telegram:1")
        self.assertTrue(progress["done"])
        self.assertEqual(progress["remaining"], 0)

    def test_replay_pays_nothing(self):
        db = make_db()
        calls, reward = make_rewards()
        onboarding.complete_tutorial_step(db, "telegram:1", "tutorial_math", reward)
        again = onboarding.complete_tutorial_step(db, "telegram:1", "tutorial_math", reward)
        self.assertFalse(again["ok"])
        self.assertEqual(again["reason"], "already_completed")
        self.assertEqual(len(calls), 1)

    def test_unknown_step_refused(self):
        db = make_db()
        res = onboarding.complete_tutorial_step(db, "telegram:1", "tutorial_nope")
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "unknown_step")


# ── profile line ────────────────────────────────────────────────────────────

class ProfileLineTests(unittest.TestCase):
    def test_profile_shows_day_and_end_date(self):
        db = make_db()
        onboarding.start_boost(db, "telegram:1", now=T0)
        line = onboarding.boost_profile_line(db, "telegram:1", quests_left=3, now=T0)
        self.assertIn("Day 1/7 boost", line)
        self.assertIn("XP x2", line)
        self.assertIn("3 quests left", line)
        self.assertIn("ends 2026-10-23", line)
        # No dark-pattern language.
        lowered = line.lower()
        for phrase in ("hurry", "running out", "last chance", "don't miss", "only"):
            self.assertNotIn(phrase, lowered)

    def test_profile_day_advances_and_ends_cleanly(self):
        db = make_db()
        onboarding.start_boost(db, "telegram:1", now=T0)
        line3 = onboarding.boost_profile_line(db, "telegram:1", now=T0 + 2 * DAY)
        self.assertIn("Day 3/7 boost", line3)
        line8 = onboarding.boost_profile_line(db, "telegram:1", now=T0 + 7 * DAY)
        self.assertIn("ended 2026-10-23", line8)
        self.assertNotIn("Day 8/7", line8)

    def test_profile_summary_bundles_everything(self):
        db = make_db()
        onboarding.start_boost(db, "telegram:1", now=T0)
        summary = onboarding.onboarding_status(db, "telegram:1", now=T0)
        self.assertIn("Day 1/7 boost", summary["profile_line"])
        self.assertEqual(summary["tutorial"]["remaining"], 5)
        self.assertTrue(summary["free_reroll_available"])
        self.assertFalse(summary["starter_gear_granted"])


# ── early quests ────────────────────────────────────────────────────────────

class EarlyQuestTests(unittest.TestCase):
    def test_catalog_days_and_objectives(self):
        quests = early_quests.early_quests()
        self.assertEqual([q["day"] for q in quests], [1, 2, 3])
        self.assertEqual([o["objective_id"] for o in quests[0]["objectives"]],
                         ["d1_math_easy", "d1_trivia", "d1_party_join"])
        self.assertEqual([o["objective_id"] for o in quests[1]["objectives"]],
                         ["d2_duel_win", "d2_lesson", "d2_invite"])
        self.assertEqual([o["objective_id"] for o in quests[2]["objectives"]],
                         ["d3_streak", "d3_math_hard", "d3_digest"])
        # Chests are never investment / cash-out.
        for quest in quests:
            chest = quest["chest"]
            self.assertIn("never an investment", chest["disclaimer"])
            self.assertIn("never a cash-out promise", chest["disclaimer"])
            self.assertEqual(chest["cosmetic"]["power"], 0)

    def test_chest_only_on_full_completion(self):
        db = make_db()
        calls, reward = make_rewards()
        # Partial progress: chest refused, nothing paid.
        early_quests.record_progress(db, "telegram:1", "d1_math_easy")
        early_quests.record_progress(db, "telegram:1", "d1_trivia")
        res = early_quests.claim_chest(db, "telegram:1", "early_day1", reward)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "not_complete")
        self.assertEqual(res["missing"], ["d1_party_join"])
        self.assertEqual(calls, [])
        self.assertEqual(early_quests.ledger_entries(db, "telegram:1"), [])
        # Finish, then claim pays immediately.
        early_quests.record_progress(db, "telegram:1", "d1_party_join")
        progress = early_quests.quest_progress(db, "telegram:1", "early_day1")
        self.assertTrue(progress["chest_ready"])
        res = early_quests.claim_chest(db, "telegram:1", "early_day1", reward)
        self.assertTrue(res["ok"])
        self.assertEqual(res["paid"]["xp"], 60)
        self.assertEqual(res["paid"]["coins"], 60)
        self.assertEqual(res["paid"]["cosmetic"]["slug"], "day1-ribbon")
        self.assertEqual(calls, [("telegram:1", 60, 60, "chest:early_day1")])
        self.assertEqual(len(early_quests.ledger_entries(db, "telegram:1")), 1)

    def test_chest_claimed_only_once(self):
        db = make_db()
        calls, reward = make_rewards()
        for obj in ("d1_math_easy", "d1_trivia", "d1_party_join"):
            early_quests.record_progress(db, "telegram:1", obj)
        self.assertTrue(early_quests.claim_chest(db, "telegram:1", "early_day1", reward)["ok"])
        again = early_quests.claim_chest(db, "telegram:1", "early_day1", reward)
        self.assertFalse(again["ok"])
        self.assertEqual(again["reason"], "already_claimed")
        self.assertEqual(len(calls), 1)

    def test_progress_capped_at_target(self):
        db = make_db()
        # d3_streak target is 3: recording 5 times still lands on 3.
        for _ in range(5):
            res = early_quests.record_progress(db, "telegram:1", "d3_streak")
        self.assertEqual(res["progress"], 3)
        self.assertTrue(res["objective_done"])
        self.assertFalse(res["quest_completed"])  # other day-3 objectives open

    def test_unknown_objective_and_quest_refused(self):
        db = make_db()
        self.assertEqual(
            early_quests.record_progress(db, "telegram:1", "nope")["reason"],
            "unknown_objective")
        self.assertEqual(
            early_quests.claim_chest(db, "telegram:1", "nope")["reason"],
            "unknown_quest")


# ── fresh start ─────────────────────────────────────────────────────────────

class FreshStartTests(unittest.TestCase):
    def test_fresh_start_is_schema_v86_with_no_import_marker(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "game.db"
            res = fresh.fresh_start(path)
            self.assertTrue(res["ok"])
            self.assertEqual(res["schema_version"], 86)
            check = fresh.verify_clean(path)
            self.assertTrue(check["ok"], check["reasons"])
            self.assertEqual(check["schema_version"], 86)
            self.assertEqual(check["imported_from"], "none")

    def test_fresh_start_never_auto_imports_old_db(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.db"
            conn = sqlite3.connect(str(old))
            conn.execute("CREATE TABLE old_players (player_key TEXT, xp INTEGER)")
            conn.execute("INSERT INTO old_players VALUES ('telegram:ghost', 99999)")
            conn.commit()
            conn.close()
            # Fresh start takes only the new path — there is no old-DB input.
            new = Path(tmp) / "new.db"
            res = fresh.fresh_start(new)
            self.assertTrue(res["ok"])
            conn2 = sqlite3.connect(str(new))
            tables = {r[0] for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            conn2.close()
            self.assertNotIn("old_players", tables)
            self.assertNotIn("telegram:ghost", str(tables))
            self.assertTrue(fresh.verify_clean(new)["ok"])
            # The old file is untouched: still there with its row.
            conn3 = sqlite3.connect(str(old))
            rows = conn3.execute("SELECT * FROM old_players").fetchall()
            conn3.close()
            self.assertEqual(rows, [("telegram:ghost", 99999)])

    def test_fresh_start_refuses_to_overwrite_existing_db(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "game.db"
            path.write_bytes(b"not empty")
            res = fresh.fresh_start(path)
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "already_exists")
            self.assertEqual(path.read_bytes(), b"not empty")

    def test_backup_old_copies_and_never_deletes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.db"
            old.write_bytes(b"old-data-bytes")
            res = fresh.backup_old(old)
            self.assertTrue(res["ok"])
            self.assertEqual(res["bytes"], len(b"old-data-bytes"))
            backup = Path(res["backup_path"])
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_bytes(), b"old-data-bytes")
            # Source still exists, untouched.
            self.assertTrue(old.exists())
            self.assertEqual(old.read_bytes(), b"old-data-bytes")
            self.assertNotEqual(str(backup), str(old))

    def test_backup_old_missing_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            res = fresh.backup_old(Path(tmp) / "missing.db")
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "not_found")

    def test_verify_clean_rejects_imported_marker(self):
        db = make_db()
        db.execute("PRAGMA user_version=86")
        db.execute("CREATE TABLE fresh_start_meta (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("INSERT INTO fresh_start_meta VALUES ('imported_from', 'old.db')")
        check = fresh.verify_clean(db)
        self.assertFalse(check["ok"])
        self.assertTrue(any("imported_from" in r for r in check["reasons"]))


if __name__ == "__main__":
    unittest.main()
