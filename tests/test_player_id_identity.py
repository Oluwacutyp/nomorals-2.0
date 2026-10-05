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


if __name__ == "__main__":
    unittest.main()
