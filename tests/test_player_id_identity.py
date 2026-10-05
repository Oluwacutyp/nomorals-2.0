"""Stable Telegram user-ID game identity with legacy name-key merge.

Covers:
- ``ChatMessage.sender_id`` is plumbed from both Telegram endpoints.
- ``_game_player`` keys off ``sender_id`` when present (one human, one
  profile across renames and across the telegram/telegram-bot split).
- ``PlayerStore.get()`` folds legacy display-name-keyed profiles
  (``telegram:Mary``) into the ID key (``telegram:12345``):
  - xp: keep the HIGHER (no double-counted levels)
  - coins/points/wins/losses/draws/games_played: summed
  - gear: all instances preserved (the gifted item survives)
  - skills: union by slug, no dupes
  - attributes: max per column
  - titles: union, active title preserved
  - game_stats: summed counters, max best_score
  - pure rename when no ID row exists yet
  - no-op when there is no legacy row or the key isn't an ID key
"""
from __future__ import annotations

import unittest

from nomorals.games.players import PlayerStore
from nomorals.social.chat.base import ChatMessage, ChatRef, ChatKind
from nomorals.storage.db import Database


def make_store() -> PlayerStore:
    db = Database(":memory:")
    db.migrate()
    return PlayerStore(db)


def seed_player(store: PlayerStore, key: str, *, xp: int = 0,
                coins: int = 0, wins: int = 0, name: str = "") -> None:
    db = store.db
    with db.transaction():
        db.execute(
            "INSERT OR REPLACE INTO game_players "
            "(player_key, platform, display, coins, xp, wins, losses, draws, "
            "games_played, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, 1.0, 2.0)",
            (key, "telegram", name or key, coins, xp, wins, wins))


def seed_gear(store: PlayerStore, key: str, instance_id: str,
              slug: str = "katana_legendary") -> None:
    db = store.db
    with db.transaction():
        db.execute(
            "INSERT OR REPLACE INTO game_gear "
            "(id, player_key, slug, durability, max_durability) "
            "VALUES (?, ?, ?, 100, 100)",
            (instance_id, key, slug))


def seed_skill(store: PlayerStore, key: str, slug: str) -> None:
    db = store.db
    with db.transaction():
        db.execute(
            "INSERT OR REPLACE INTO game_skills "
            "(player_key, slug) VALUES (?, ?)", (key, slug))


def has_table(store: PlayerStore, table: str) -> bool:
    rows = store.db.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)) or []
    return bool(rows)


class SenderIdPlumbingTests(unittest.TestCase):
    def test_chat_message_defaults_sender_id(self):
        msg = ChatMessage(chat=ChatRef(platform="telegram", chat_id="1",
                                       kind=ChatKind.DM),
                          incoming=True, text="hi", sender="Mary")
        self.assertEqual(msg.sender_id, "")

    def test_chat_message_carries_sender_id(self):
        msg = ChatMessage(chat=ChatRef(platform="telegram", chat_id="1",
                                       kind=ChatKind.DM),
                          incoming=True, text="hi", sender="Mary",
                          sender_id="12345")
        self.assertEqual(msg.sender_id, "12345")
        self.assertEqual(msg.to_dict()["sender_id"], "12345")

    def test_game_player_uses_sender_id(self):
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
        msg = ChatMessage(chat=ChatRef(platform="telegram", chat_id="1",
                                       kind=ChatKind.DM),
                          incoming=True, text="/duel", sender="Mary",
                          sender_id="12345")
        player = RuntimeGamesMixin._game_player(msg)
        self.assertEqual(player.key, "telegram:12345")
        self.assertEqual(player.name, "Mary")

    def test_game_player_falls_back_to_sender_without_id(self):
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
        msg = ChatMessage(chat=ChatRef(platform="telegram", chat_id="1",
                                       kind=ChatKind.DM),
                          incoming=True, text="/duel", sender="Mary")
        player = RuntimeGamesMixin._game_player(msg)
        self.assertEqual(player.key, "telegram:Mary")

    def test_game_player_folds_bot_platform(self):
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
        msg = ChatMessage(chat=ChatRef(platform="telegram-bot", chat_id="1",
                                       kind=ChatKind.DM),
                          incoming=True, text="/duel", sender="chfjdhx",
                          sender_id="12345")
        player = RuntimeGamesMixin._game_player(msg)
        # Same human as the userbot sighting above: one key.
        self.assertEqual(player.key, "telegram:12345")


class LegacyMergeTests(unittest.TestCase):
    def test_pure_rename_when_no_id_row(self):
        store = make_store()
        seed_player(store, "telegram:Mary", xp=500, coins=100, name="Mary")
        prof = store.get("telegram:12345", name="Mary", platform="telegram")
        self.assertEqual(prof.xp, 500)
        self.assertEqual(prof.coins, 100)
        # Legacy row is gone.
        legacy = store.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            ("telegram:Mary",))
        self.assertIsNone(legacy)

    def test_xp_keeps_higher_coins_sum(self):
        store = make_store()
        # Legacy low-level profile (the "Mary" side).
        seed_player(store, "telegram:Mary", xp=500, coins=100, wins=3,
                    name="Mary")
        # Existing ID-keyed profile with higher XP.
        seed_player(store, "telegram:12345", xp=9000, coins=250, wins=10,
                    name="Mary")
        prof = store.get("telegram:12345", name="Mary", platform="telegram")
        self.assertEqual(prof.xp, 9000)  # higher kept, not summed
        self.assertEqual(prof.coins, 350)  # summed
        self.assertEqual(prof.wins, 13)  # summed

    def test_user_case_mary_gift_survives_merge(self):
        """The user's real scenario: level-24 chfjdhx profile + lower-level
        Mary profile that holds a gifted item.  Nothing is lost."""
        store = make_store()
        # High-level profile under the old bot-username key.
        seed_player(store, "telegram:chfjdhx", xp=20000, coins=500, wins=20,
                    name="chfjdhx")
        # Lower-level Mary profile holding the gifted gear.
        seed_player(store, "telegram:Mary", xp=800, coins=50, wins=2,
                    name="Mary")
        if has_table(store, "game_gear"):
            seed_gear(store, "telegram:Mary", "gift-inst-1",
                      "katana_legendary")
            seed_gear(store, "telegram:chfjdhx", "own-inst-1",
                      "armor_epic")
        # BotFather sighting merges chfjdhx into the ID key...
        prof = store.get("telegram:12345", name="chfjdhx",
                         platform="telegram-bot")
        self.assertEqual(prof.xp, 20000)
        # ...then the userbot sighting merges Mary into the same key.
        prof = store.get("telegram:12345", name="Mary", platform="telegram")
        self.assertEqual(prof.xp, 20000)  # level 24 kept
        self.assertEqual(prof.coins, 550)  # 500 + 50
        self.assertEqual(prof.wins, 22)  # 20 + 2
        if has_table(store, "game_gear"):
            slugs = {r["slug"] for r in store.db.query(
                "SELECT slug FROM game_gear WHERE player_key = ?",
                ("telegram:12345",))}
            self.assertIn("katana_legendary", slugs)  # the gift survived
            self.assertIn("armor_epic", slugs)
        # Both legacy rows are gone; one profile remains.
        for legacy in ("telegram:Mary", "telegram:chfjdhx"):
            self.assertIsNone(store.db.query_one(
                "SELECT * FROM game_players WHERE player_key = ?",
                (legacy,)))

    def test_skills_union_no_dupes(self):
        store = make_store()
        if not has_table(store, "game_skills"):
            self.skipTest("game_skills table missing")
        seed_player(store, "telegram:Mary", name="Mary")
        seed_player(store, "telegram:12345", name="Mary")
        seed_skill(store, "telegram:Mary", "war_cry")
        seed_skill(store, "telegram:Mary", "slaying_force")
        seed_skill(store, "telegram:12345", "war_cry")  # dupe
        store.get("telegram:12345", name="Mary", platform="telegram")
        slugs = [r["slug"] for r in store.db.query(
            "SELECT slug FROM game_skills WHERE player_key = ?",
            ("telegram:12345",))]
        self.assertEqual(sorted(slugs), ["slaying_force", "war_cry"])

    def test_noop_without_legacy_row(self):
        store = make_store()
        prof = store.get("telegram:12345", name="Zed", platform="telegram")
        self.assertEqual(prof.xp, 0)
        self.assertEqual(prof.name, "Zed")

    def test_noop_for_non_id_key(self):
        store = make_store()
        seed_player(store, "telegram:Mary", xp=500, name="Mary")
        # Name-keyed lookup must NOT trigger the ID merge path.
        prof = store.get("telegram:Mary", name="Mary", platform="telegram")
        self.assertEqual(prof.xp, 500)

    def test_merge_is_idempotent(self):
        store = make_store()
        seed_player(store, "telegram:Mary", xp=500, coins=100, name="Mary")
        seed_player(store, "telegram:12345", xp=9000, coins=250, name="Mary")
        first = store.get("telegram:12345", name="Mary", platform="telegram")
        second = store.get("telegram:12345", name="Mary", platform="telegram")
        self.assertEqual(first.xp, second.xp)
        self.assertEqual(first.coins, second.coins)
        self.assertEqual(second.xp, 9000)
        self.assertEqual(second.coins, 350)


class MultiNameMergeTests(unittest.TestCase):
    """The 'lost level 24' regression: one human, several legacy names.

    ``telegram:Vrede peace`` (level 24), ``telegram:Mary`` (katana +
    coins) and ``telegram:chfjdhx`` all belong to sender 5478650254.
    The old single-name merge only folded the row matching the current
    sighting name; the others stayed orphaned.  The sighting registry
    fixes it: every name ever sighted for the sender_id is swept.
    """

    def test_rename_does_not_orphan_older_profiles(self):
        store = make_store()
        seed_player(store, "telegram:Vrede peace", xp=20000, coins=100,
                    name="Vrede peace")
        seed_player(store, "telegram:Mary", xp=500, coins=5076, name="Mary")
        seed_gear(store, "telegram:Mary", "gear-1", "katana_legendary")
        # Sighting under the first name merges only that row...
        p = store.get("telegram:5478650254", name="Mary",
                      platform="telegram")
        self.assertEqual(p.coins, 5076)
        # ...but the second sighting recovers the level-24 profile too.
        p = store.get("telegram:5478650254", name="Vrede peace",
                      platform="telegram")
        self.assertEqual(p.xp, 20000)  # higher XP kept
        self.assertEqual(p.coins, 5176)  # summed
        gear = store.db.query(
            "SELECT slug FROM game_gear WHERE player_key = ?",
            ("telegram:5478650254",)) or []
        self.assertEqual([r["slug"] for r in gear], ["katana_legendary"])
        leftovers = store.db.query(
            "SELECT player_key FROM game_players "
            "WHERE player_key LIKE 'telegram:%' "
            "AND player_key != 'telegram:5478650254'") or []
        self.assertEqual(leftovers, [])

    def test_sighting_registry_accumulates_names(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        store.get("telegram:999", name="Vrede peace",
                  platform="telegram-bot")
        names = store._known_names("telegram", "999")
        self.assertEqual(sorted(names), ["Mary", "Vrede peace"])

    def test_stranger_name_rows_are_never_touched(self):
        store = make_store()
        seed_player(store, "telegram:Bob", xp=700, coins=50, name="Bob")
        p = store.get("telegram:111", name="Alice", platform="telegram")
        self.assertEqual(p.xp, 0)
        bob = store.db.query_one(
            "SELECT xp FROM game_players WHERE player_key = ?",
            ("telegram:Bob",))
        self.assertEqual(bob["xp"], 700)

    def test_merge_legacy_names_recovery_entry(self):
        store = make_store()
        seed_player(store, "telegram:Vrede peace", xp=20000, coins=100,
                    name="Vrede peace")
        seed_player(store, "telegram:Mary", xp=500, coins=5076, name="Mary")
        result = store.merge_legacy_names(
            "telegram:5478650254", ["Vrede peace", "Mary", "Nobody"])
        self.assertEqual(sorted(result["merged"]),
                         ["telegram:Mary", "telegram:Vrede peace"])
        p = store.get("telegram:5478650254")
        self.assertEqual(p.xp, 20000)
        self.assertEqual(p.coins, 5176)

    def test_bot_platform_legacy_key_is_swept(self):
        store = make_store()
        seed_player(store, "telegram-bot:OldName", xp=300, coins=25,
                    name="OldName")
        p = store.get("telegram:4242", name="OldName",
                      platform="telegram-bot")
        self.assertEqual(p.xp, 300)
        self.assertEqual(p.coins, 25)

    def test_fuzzy_emoji_name_matches_legacy_row(self):
        # --name "Vrede peace" folds "telegram:Vrede peace 🥷🟫🖤"
        store = make_store()
        seed_player(store, "telegram:Vrede peace 🥷🟫🖤", xp=22625,
                    coins=24208, name="Vrede peace 🥷🟫🖤")
        result = store.merge_legacy_names("telegram:5478650254",
                                          ["Vrede peace"])
        self.assertEqual(result["merged"],
                         ["telegram:Vrede peace 🥷🟫🖤"])
        p = store.get("telegram:5478650254")
        self.assertEqual(p.xp, 22625)
        self.assertEqual(p.coins, 24208)

    def test_like_wildcards_escaped_no_false_merge(self):
        store = make_store()
        seed_player(store, "telegram:100% legit", xp=10, coins=10,
                    name="100% legit")
        # underscore/% in the name must be treated literally, not as
        # LIKE wildcards — nothing should match a nonsense query.
        result = store.merge_legacy_names("telegram:4242", ["zzz_nobody"])
        self.assertEqual(result["merged"], [])
        p = store.get("telegram:100% legit")
        self.assertEqual(p.xp, 10)


if __name__ == "__main__":
    unittest.main()

class ProfileLineDisplayNameTests(unittest.TestCase):
    """_profile_line prefers the live sender name over a stale DB name."""

    def test_live_name_preferred(self):
        from types import SimpleNamespace
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
        prof = SimpleNamespace(
            name="chfjdhx", key="telegram:5478650254",
            wins=190, losses=20, draws=12, streak=10,
            games_played=222, coins=29578, points=29925, items={},
        )
        line = RuntimeGamesMixin._profile_line(prof, display_name="Peacethefirst")
        self.assertIn("Peacethefirst", line)
        self.assertNotIn("chfjdhx", line)

    def test_falls_back_to_db_name(self):
        from types import SimpleNamespace
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
        prof = SimpleNamespace(
            name="chfjdhx", key="telegram:5478650254",
            wins=1, losses=0, draws=0, streak=0,
            games_played=1, coins=10, points=5, items={},
        )
        line = RuntimeGamesMixin._profile_line(prof)
        self.assertIn("chfjdhx", line)


class LeaderboardNameOverrideTests(unittest.TestCase):
    """Leaderboard prefers live sender names via name_overrides."""

    def _make_board_with(self, entries):
        """entries: list of (key, name, games_played, points)."""
        from nomorals.games.players import PlayerStore, Leaderboard
        import tempfile, os
        from nomorals.storage import Database
        tmp = tempfile.mkdtemp()
        db = Database(os.path.join(tmp, "t.db"))
        db.migrate()
        store = PlayerStore(db)
        for key, name, gp, pts in entries:
            store.get(key, name=name, platform="telegram")
            db.execute(
                "UPDATE game_players SET games_played = ?, points = ? "
                "WHERE player_key = ?",
                (gp, pts, key),
            )
        return Leaderboard(store)

    def test_override_replaces_stale_name(self):
        board = self._make_board_with([("telegram:5478650254", "chfjdhx", 5, 100)])
        rows = board.top(10, name_overrides={"telegram:5478650254": "Peacethefirst"})
        self.assertEqual(rows[0]["name"], "Peacethefirst")

    def test_no_override_uses_db_name(self):
        board = self._make_board_with([("telegram:5478650254", "chfjdhx", 5, 100)])
        rows = board.top(10)
        self.assertEqual(rows[0]["name"], "chfjdhx")

    def test_render_uses_override(self):
        board = self._make_board_with([("telegram:5478650254", "chfjdhx", 5, 100)])
        text = board.render(10, name_overrides={"telegram:5478650254": "Peacethefirst"})
        self.assertIn("Peacethefirst", text)
        self.assertNotIn("chfjdhx", text)

    def test_other_players_unaffected(self):
        board = self._make_board_with([
            ("telegram:111", "Alice", 5, 200),
            ("telegram:222", "Bob", 5, 100),
        ])
        rows = board.top(10, name_overrides={"telegram:222": "Bobby"})
        names = [r["name"] for r in rows]
        self.assertIn("Alice", names)  # untouched
        self.assertIn("Bobby", names)  # overridden
        self.assertNotIn("Bob", names)


class UsernameBasedMergeTests(unittest.TestCase):
    """Legacy profiles merged by Telegram username match.

    Catches profiles created via the name-fallback path before
    sightings existed: e.g. ``telegram:chfjdhx`` for a human whose
    ID key is ``telegram:7541672134``.  When the ID key is looked up
    with the username, the legacy row is folded in even though the
    name was never sighted under the ID.
    """

    def test_username_merge_folds_unsighted_legacy_profile(self):
        store = make_store()
        # Legacy name-keyed profile from the pre-sightings era, with
        # the username stored on it.
        seed_player(store, "telegram:chfjdhx", xp=116, coins=5076,
                    wins=1, name="chfjdhx")
        store.db.execute(
            "UPDATE game_players SET username = ? WHERE player_key = ?",
            ("chfjdhx", "telegram:chfjdhx"))
        # ID-keyed lookup with the same username merges it.
        prof = store.get("telegram:7541672134", name="Mary",
                         platform="telegram", username="chfjdhx")
        self.assertEqual(prof.coins, 5076)
        self.assertEqual(prof.username, "chfjdhx")
        # Legacy row is gone.
        row = store.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            ("telegram:chfjdhx",))
        self.assertIsNone(row)

    def test_username_merge_skips_other_humans_id_keys(self):
        store = make_store()
        # Another human's ID-keyed profile that happens to share a
        # username value must NOT be folded (different human).
        seed_player(store, "telegram:999", xp=10, coins=100,
                    wins=0, name="someone")
        store.db.execute(
            "UPDATE game_players SET username = ? WHERE player_key = ?",
            ("chfjdhx", "telegram:999"))
        prof = store.get("telegram:7541672134", name="Mary",
                         platform="telegram", username="chfjdhx")
        # The other human's ID-keyed profile is untouched.
        other = store.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            ("telegram:999",))
        self.assertIsNotNone(other)
        self.assertEqual(other["coins"], 100)

    def test_username_merge_case_insensitive(self):
        store = make_store()
        seed_player(store, "telegram:chfjdhx", xp=0, coins=250,
                    wins=0, name="chfjdhx")
        store.db.execute(
            "UPDATE game_players SET username = ? WHERE player_key = ?",
            ("ChFjDhX", "telegram:chfjdhx"))
        prof = store.get("telegram:7541672134", name="Mary",
                         platform="telegram", username="CHFJDHX")
        self.assertEqual(prof.coins, 250)
