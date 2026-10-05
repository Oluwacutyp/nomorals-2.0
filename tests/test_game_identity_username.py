"""Game identity: username storage, @mention lookup, soft-delete, untangle.

Covers the identity repair wave:
- ``ChatMessage.sender_username`` is plumbed from both Telegram endpoints.
- ``PlayerStore`` stores the Telegram username per profile; the userbot
  is the authority (bot endpoint only fills when empty).
- ``resolve_recipient`` matches by username first: ``/gift @chfjdhx``
  finds Mary's profile, ``/gift @peacethefirst`` finds the main profile.
- Soft-delete: ``/game delete`` marks the row; playing again within 48h
  restores it untouched; after 48h the row is purged and a fresh
  profile starts.  Deleted profiles never appear on the leaderboard or
  in gift lookup.
- The untangle repair (migration 79): the main profile
  (``telegram:5478650254``) gets username ``peacethefirst`` (never
  ``chfjdhx``), and the alt (``telegram:7541672134``) is restored as
  its own profile with its pre-merge values.
"""
from __future__ import annotations

import time
import types
import unittest

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.gifting import resolve_recipient
from nomorals.games.players import (
    SOFT_DELETE_RESTORE_S,
    Player,
    PlayerStore,
)
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.storage.db import Database
from nomorals.storage.migrations import _apply_game_identity_untangle


def make_store() -> PlayerStore:
    db = Database(":memory:")
    db.migrate()
    return PlayerStore(db)


def seed_profile(store: PlayerStore, key: str, **kw: object) -> None:
    import json as _json
    db = store.db
    cols = ["player_key", "platform", "display", "username", "coins",
            "points", "wins", "losses", "draws", "games_played", "xp"]
    vals = [key, kw.get("platform", "telegram"), kw.get("name", ""),
            kw.get("username", ""), kw.get("coins", 0), kw.get("points", 0),
            kw.get("wins", 0), kw.get("losses", 0), kw.get("draws", 0),
            kw.get("games_played", 0), kw.get("xp", 0)]
    items_json = _json.dumps(kw.get("items", {}))
    with db.transaction():
        db.execute(
            "INSERT OR REPLACE INTO game_players "
            "(player_key, platform, display, username, coins, points, wins, "
            "losses, draws, games_played, xp, per_game, items, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, "
            "1.0, 2.0)",
            tuple(vals) + (items_json,))


def make_msg(sender: str, sender_id: str, username: str,
             platform: str = "telegram") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id="1", kind=ChatKind.DM),
        incoming=True, text="/game", sender=sender, sender_id=sender_id,
        sender_username=username)


class UsernamePlumbingTests(unittest.TestCase):
    def test_game_player_carries_username(self):
        p = RuntimeGamesMixin._game_player(
            make_msg("Mary", "7541672134", "chfjdhx"))
        self.assertEqual(p.key, "telegram:7541672134")
        self.assertEqual(p.username, "chfjdhx")

    def test_game_player_strips_at(self):
        p = RuntimeGamesMixin._game_player(
            make_msg("Mary", "7541672134", "@chfjdhx"))
        self.assertEqual(p.username, "chfjdhx")

    def test_bot_platform_folds_key_keeps_username(self):
        p = RuntimeGamesMixin._game_player(
            make_msg("chfjdhx", "7541672134", "chfjdhx",
                     platform="telegram-bot"))
        self.assertEqual(p.key, "telegram:7541672134")
        self.assertEqual(p.username, "chfjdhx")


class UsernameAuthorityTests(unittest.TestCase):
    def test_userbot_sets_username(self):
        store = make_store()
        prof = store.get("telegram:7541672134", name="Mary",
                         platform="telegram", username="chfjdhx")
        self.assertEqual(prof.username, "chfjdhx")

    def test_userbot_overwrites_wrong_username(self):
        # The chfjdhx-on-main mix-up: the userbot sighting corrects it.
        store = make_store()
        seed_profile(store, "telegram:5478650254", name="Peacethefirst",
                     username="chfjdhx", coins=100)
        prof = store.get("telegram:5478650254", name="Peacethefirst",
                         platform="telegram", username="peacethefirst")
        self.assertEqual(prof.username, "peacethefirst")

    def test_bot_endpoint_only_fills_empty_username(self):
        store = make_store()
        seed_profile(store, "telegram:7541672134", name="Mary",
                     username="chfjdhx")
        # Bot reports the same handle: no-op, keeps the value.
        prof = store.get("telegram:7541672134", name="chfjdhx",
                         platform="telegram-bot", username="chfjdhx")
        self.assertEqual(prof.username, "chfjdhx")
        # Bot must never clobber a userbot-set username.
        prof = store.get("telegram:7541672134", name="chfjdhx",
                         platform="telegram-bot", username="hacker")
        self.assertEqual(prof.username, "chfjdhx")

    def test_bot_fills_missing_username(self):
        store = make_store()
        seed_profile(store, "telegram:7541672134", name="Mary")
        prof = store.get("telegram:7541672134", name="chfjdhx",
                         platform="telegram-bot", username="chfjdhx")
        self.assertEqual(prof.username, "chfjdhx")

    def test_find_by_username(self):
        store = make_store()
        seed_profile(store, "telegram:7541672134", name="Mary",
                     username="chfjdhx", coins=50)
        seed_profile(store, "telegram:5478650254", name="Peacethefirst",
                     username="peacethefirst", coins=100)
        hit = store.find_by_username("chfjdhx")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.key, "telegram:7541672134")
        # Case-insensitive, @ tolerated.
        hit = store.find_by_username("@PEACETHEFIRST")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.key, "telegram:5478650254")
        self.assertIsNone(store.find_by_username("nobody"))


class GiftUsernameLookupTests(unittest.TestCase):
    def _store(self) -> PlayerStore:
        store = make_store()
        seed_profile(store, "telegram:5478650254", name="Peacethefirst",
                     username="peacethefirst", coins=100)
        seed_profile(store, "telegram:7541672134", name="Mary",
                     username="chfjdhx", coins=50)
        return store

    def test_gift_at_chfjdhx_finds_mary(self):
        prof, err = resolve_recipient(self._store(), "@chfjdhx")
        self.assertEqual(err, "")
        self.assertEqual(prof.key, "telegram:7541672134")

    def test_gift_at_peacethefirst_finds_main(self):
        prof, err = resolve_recipient(self._store(), "@peacethefirst")
        self.assertEqual(err, "")
        self.assertEqual(prof.key, "telegram:5478650254")

    def test_username_beats_display_name(self):
        # Two profiles share a display name; the username disambiguates.
        store = make_store()
        seed_profile(store, "telegram:111", name="Mary", username="mary_one")
        seed_profile(store, "telegram:222", name="Mary", username="mary_two")
        prof, err = resolve_recipient(store, "@mary_two")
        self.assertEqual(err, "")
        self.assertEqual(prof.key, "telegram:222")

    def test_unknown_username_still_helpful(self):
        prof, err = resolve_recipient(self._store(), "@ghost")
        self.assertIsNone(prof)
        self.assertIn("no player found", err)


class SoftDeleteTests(unittest.TestCase):
    def test_delete_hides_from_all_and_leaderboard(self):
        store = make_store()
        seed_profile(store, "telegram:1", name="Ada", username="ada",
                     coins=100, games_played=5)
        seed_profile(store, "telegram:2", name="Bob", username="bob",
                     coins=10, games_played=2)
        self.assertTrue(store.soft_delete("telegram:1"))
        keys = [p.key for p in store.all()]
        self.assertNotIn("telegram:1", keys)
        self.assertIn("telegram:2", keys)
        # Gift lookup skips deleted too.
        self.assertIsNone(store.find_by_username("ada"))
        prof, err = resolve_recipient(store, "@ada")
        self.assertIsNone(prof)

    def test_play_within_48h_restores_untouched(self):
        store = make_store()
        seed_profile(store, "telegram:1", name="Ada", username="ada",
                     coins=100, xp=500, games_played=5)
        store.soft_delete("telegram:1")
        prof = store.get("telegram:1", name="Ada", platform="telegram",
                         username="ada")
        self.assertEqual(prof.coins, 100)
        self.assertEqual(prof.xp, 500)
        self.assertEqual(prof.games_played, 5)
        row = store.db.query_one(
            "SELECT deleted_at FROM game_players WHERE player_key = ?",
            ("telegram:1",))
        self.assertEqual(float(row["deleted_at"]), 0.0)

    def test_after_48h_profile_purged_and_fresh(self):
        store = make_store()
        seed_profile(store, "telegram:1", name="Ada", username="ada",
                     coins=100, xp=500, games_played=5)
        store.soft_delete("telegram:1")
        # Age the deletion past the window.
        old = time.time() - SOFT_DELETE_RESTORE_S - 60
        with store.db.transaction():
            store.db.execute(
                "UPDATE game_players SET deleted_at = ? WHERE player_key = ?",
                (old, "telegram:1"))
        prof = store.get("telegram:1", name="Ada", platform="telegram",
                         username="ada")
        self.assertEqual(prof.coins, 0)
        self.assertEqual(prof.xp, 0)
        self.assertEqual(prof.games_played, 0)
        # Username is re-learned from the fresh sighting.
        self.assertEqual(prof.username, "ada")

    def test_delete_missing_profile_returns_false(self):
        store = make_store()
        self.assertFalse(store.soft_delete("telegram:nobody"))

    def test_double_delete_is_noop(self):
        store = make_store()
        seed_profile(store, "telegram:1", name="Ada")
        self.assertTrue(store.soft_delete("telegram:1"))
        self.assertFalse(store.soft_delete("telegram:1"))


class _FakeEngine:
    def __init__(self, store: PlayerStore) -> None:
        self.store = store
        self.games: dict = {}

    def live(self, chat_key: str):  # type: ignore[no-untyped-def]
        return None


class _DeleteHarness(RuntimeGamesMixin):
    def __init__(self, store: PlayerStore) -> None:
        self._engine = _FakeEngine(store)
        self.context = types.SimpleNamespace(db=None)

    def _game_engine(self):  # type: ignore[no-untyped-def]
        return self._engine


def _player(key: str, name: str, username: str = "") -> Player:
    return Player(key=key, platform="telegram", name=name, username=username)


class GameDeleteCommandTests(unittest.TestCase):
    def _harness(self) -> _DeleteHarness:
        store = make_store()
        seed_profile(store, "telegram:1", name="Ada", username="ada",
                     coins=100, xp=500, games_played=5, wins=3, losses=2)
        return _DeleteHarness(store)

    def test_delete_asks_for_confirmation(self):
        h = self._harness()
        out = h._control_game("delete", "telegram:dm:1",
                              player=_player("telegram:1", "Ada", "ada"))
        self.assertIn("confirm", out)
        self.assertIn("48 hours", out)
        # Not deleted yet.
        self.assertIn("telegram:1",
                      [p.key for p in h._engine.store.all()])

    def test_delete_confirm_wipes(self):
        h = self._harness()
        h._control_game("delete", "telegram:dm:1",
                        player=_player("telegram:1", "Ada", "ada"))
        out = h._control_game("delete confirm", "telegram:dm:1",
                              player=_player("telegram:1", "Ada", "ada"))
        self.assertIn("deleted", out.lower())
        self.assertNotIn("telegram:1",
                         [p.key for p in h._engine.store.all()])

    def test_delete_cancel_keeps_profile(self):
        h = self._harness()
        h._control_game("delete", "telegram:dm:1",
                        player=_player("telegram:1", "Ada", "ada"))
        out = h._control_game("delete cancel", "telegram:dm:1",
                              player=_player("telegram:1", "Ada", "ada"))
        self.assertIn("safe", out.lower())
        self.assertIn("telegram:1",
                      [p.key for p in h._engine.store.all()])

    def test_delete_confirm_without_prompt(self):
        h = self._harness()
        out = h._control_game("delete confirm", "telegram:dm:1",
                              player=_player("telegram:1", "Ada", "ada"))
        self.assertIn("nothing to confirm", out)

    def test_delete_empty_profile(self):
        h = self._harness()
        out = h._control_game("delete", "telegram:dm:1",
                              player=_player("telegram:999", "Ghost"))
        self.assertIn("don't have a profile", out)


class UntangleRepairTests(unittest.TestCase):
    """Migration 79: the chfjdhx/peacethefirst mix-up repair."""

    def _seed_merged_state(self, store: PlayerStore) -> None:
        # The phone's post-merge state: everything folded into main,
        # main wrongly carrying the alt's username.
        seed_profile(store, "telegram:5478650254", name="chfjdhx",
                     username="chfjdhx", coins=29578, points=29925,
                     wins=190, losses=20, games_played=222, xp=22625)
        db = store.db
        with db.transaction():
            db.execute(
                "INSERT INTO game_gear (id, player_key, slug, durability, "
                "max_durability, equipped, created_at) VALUES "
                "('gear-kat-1', 'telegram:5478650254', 'katana_legendary', "
                "100, 100, 0, 1.0), "
                "('gear-broad-1', 'telegram:5478650254', 'broadsword_common', "
                "25, 25, 0, 1.0)")

    def test_repair_untangles(self):
        store = make_store()
        self._seed_merged_state(store)
        _apply_game_identity_untangle(store.db)

        main = store.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            ("telegram:5478650254",))
        alt = store.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            ("telegram:7541672134",))

        # Main: correct username, alt's coins removed, XP untouched.
        self.assertEqual(main["username"], "peacethefirst")
        self.assertEqual(main["display"], "Peacethefirst")
        self.assertEqual(main["coins"], 29578 - 5076)
        self.assertEqual(main["xp"], 22625)
        self.assertEqual(main["points"], 29925 - 85)
        self.assertEqual(main["wins"], 190 - 1)
        self.assertEqual(main["losses"], 20 - 2)
        self.assertEqual(main["games_played"], 222 - 3)

        # Alt: restored as its own profile with pre-merge values.
        self.assertIsNotNone(alt)
        self.assertEqual(alt["username"], "chfjdhx")
        self.assertEqual(alt["display"], "Mary")
        self.assertEqual(alt["coins"], 5076)
        self.assertEqual(alt["xp"], 116)
        self.assertEqual(alt["points"], 85)
        self.assertEqual((alt["wins"], alt["losses"], alt["games_played"]),
                         (1, 2, 3))

        # The gifted katana moved back to the alt.
        gear_main = [r["slug"] for r in store.db.query(
            "SELECT slug FROM game_gear WHERE player_key = ?",
            ("telegram:5478650254",))]
        gear_alt = [r["slug"] for r in store.db.query(
            "SELECT slug FROM game_gear WHERE player_key = ?",
            ("telegram:7541672134",))]
        self.assertEqual(gear_alt, ["katana_legendary"])
        self.assertEqual(gear_main, ["broadsword_common"])

    def test_repair_idempotent(self):
        store = make_store()
        self._seed_merged_state(store)
        _apply_game_identity_untangle(store.db)
        _apply_game_identity_untangle(store.db)
        main = store.db.query_one(
            "SELECT coins, username FROM game_players WHERE player_key = ?",
            ("telegram:5478650254",))
        self.assertEqual(main["coins"], 29578 - 5076)
        self.assertEqual(main["username"], "peacethefirst")

    def test_repair_skips_when_main_missing(self):
        store = make_store()
        # No main row at all: repair must not invent one.
        _apply_game_identity_untangle(store.db)
        alt = store.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            ("telegram:7541672134",))
        self.assertIsNone(alt)

    def test_repair_never_clobbers_correct_username(self):
        store = make_store()
        self._seed_merged_state(store)
        with store.db.transaction():
            store.db.execute(
                "UPDATE game_players SET username = ? WHERE player_key = ?",
                ("peacethefirst", "telegram:5478650254"))
        _apply_game_identity_untangle(store.db)
        main = store.db.query_one(
            "SELECT username FROM game_players WHERE player_key = ?",
            ("telegram:5478650254",))
        self.assertEqual(main["username"], "peacethefirst")

    def test_repair_skips_alt_with_real_data(self):
        # If the alt already built a genuine profile since, don't
        # overwrite it — just ensure identity fields are sane.
        store = make_store()
        self._seed_merged_state(store)
        seed_profile(store, "telegram:7541672134", name="Mary",
                     username="chfjdhx", coins=999, xp=2000,
                     games_played=50)
        _apply_game_identity_untangle(store.db)
        alt = store.db.query_one(
            "SELECT coins, xp FROM game_players WHERE player_key = ?",
            ("telegram:7541672134",))
        self.assertEqual(alt["coins"], 999)
        self.assertEqual(alt["xp"], 2000)


class GetForTests(unittest.TestCase):
    """PlayerStore.get_for() must pass username through so the
    username-based legacy merge actually runs.

    Regression: the game engine called store.get(player.key) with just
    the key, dropping the username — so _merge_all_legacy_names never
    saw it and Mary's orphaned telegram:chfjdhx profile was never
    folded into telegram:7541672134.
    """

    def test_get_for_merges_unsighted_username_profile(self):
        store = make_store()
        # Mary's orphaned Tgbot profile: name-keyed, never sighted under
        # the numeric ID, holds her coins + katana.
        seed_profile(store, "telegram:chfjdhx", name="chfjdhx",
                     username="chfjdhx", coins=6076, points=85,
                     wins=1, losses=2, games_played=3, xp=116,
                     items={"katana_legendary": 1})
        # What _game_player builds for Mary's Tgbot /game message.
        player = Player.from_sender("telegram-bot", "7541672134",
                                    "Mary", username="chfjdhx")
        prof = store.get_for(player)
        self.assertEqual(prof.key, "telegram:7541672134")
        self.assertEqual(prof.coins, 6076)
        self.assertEqual(prof.items.get("katana_legendary"), 1)
        # Legacy row is gone.
        rows = store.db.query("SELECT player_key FROM game_players")
        self.assertEqual([r["player_key"] for r in rows],
                         ["telegram:7541672134"])

    def test_get_for_passes_identity_fields(self):
        store = make_store()
        player = Player.from_sender("telegram", "5478650254",
                                    "Vrede peace", username="peacethefirst")
        prof = store.get_for(player)
        self.assertEqual(prof.key, "telegram:5478650254")
        self.assertEqual(prof.username, "peacethefirst")
        self.assertEqual(prof.name, "Vrede peace")


def make_group_msg(sender: str, sender_id: str, username: str = "",
                   platform: str = "telegram") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id="-5223197263",
                     kind=ChatKind.GROUP),
        incoming=True, text="/game", sender=sender, sender_id=sender_id,
        sender_username=username)


class GamePlayerGroupGuardTests(unittest.TestCase):
    """_game_player must never mint name-keyed profiles for group/channel
    senders without a numeric sender_id.

    Regression: a channel post (sender = channel, no sender_id) created
    `telegram:xauusd_sentinel_signal` — groups/channels must never get
    game profiles. Human senders in groups (with sender_id) still get
    their normal ID-keyed profile.
    """

    def test_group_channel_post_returns_none(self):
        # Channel post: sender is the channel name, no numeric ID.
        msg = make_group_msg("xauusd_sentinel_signal", "")
        player = RuntimeGamesMixin._game_player(msg)
        self.assertIsNone(player)

    def test_group_human_sender_uses_id(self):
        # Mary sends /game in a group: her ID-keyed profile is used.
        msg = make_group_msg("Mary", "7541672134", "chfjdhx")
        player = RuntimeGamesMixin._game_player(msg)
        self.assertIsNotNone(player)
        self.assertEqual(player.key, "telegram:7541672134")

    def test_dm_name_fallback_still_works(self):
        # DM without sender_id keeps the legacy name-fallback path
        # (the telegram.py DM fallback should prevent this in practice,
        # but the guard must not break DMs).
        msg = make_msg("Someone", "", "")
        player = RuntimeGamesMixin._game_player(msg)
        self.assertIsNotNone(player)
        self.assertEqual(player.key, "telegram:Someone")


class PhantomCleanupMigrationTests(unittest.TestCase):
    """Migration 80 deletes phantom game profiles: the orphaned
    `telegram:Mary` row and any zero-activity non-numeric telegram
    profiles (channel/group phantoms like
    `telegram:xauusd_sentinel_signal`).
    """

    def test_deletes_orphaned_mary_and_channel_phantom(self):
        from nomorals.storage.migrations import _apply_game_phantom_cleanup
        store = make_store()
        # The real merged profile — must survive.
        seed_profile(store, "telegram:7541672134", name="Mary",
                     username="chfjdhx", coins=6076, games_played=3, xp=116)
        # Orphaned name-keyed Mary (0 activity) — must go.
        seed_profile(store, "telegram:Mary", name="Mary")
        # Channel phantom (0 activity) — must go.
        seed_profile(store, "telegram:xauusd_sentinel_signal",
                     name="xauusd_sentinel_signal")
        # A legacy profile WITH activity — must survive (not a phantom).
        seed_profile(store, "telegram:OldBob", name="OldBob", coins=100,
                     games_played=5)
        _apply_game_phantom_cleanup(store.db)
        rows = store.db.query(
            "SELECT player_key FROM game_players ORDER BY player_key")
        keys = [r["player_key"] for r in rows]
        self.assertIn("telegram:7541672134", keys)
        self.assertIn("telegram:OldBob", keys)
        self.assertNotIn("telegram:Mary", keys)
        self.assertNotIn("telegram:xauusd_sentinel_signal", keys)

    def test_never_deletes_known_good_accounts(self):
        from nomorals.storage.migrations import _apply_game_phantom_cleanup
        store = make_store()
        # Even with zero activity, the two real accounts are protected.
        seed_profile(store, "telegram:5478650254", name="Vrede peace",
                     username="peacethefirst")
        seed_profile(store, "telegram:7541672134", name="Mary",
                     username="chfjdhx")
        _apply_game_phantom_cleanup(store.db)
        rows = store.db.query(
            "SELECT player_key FROM game_players ORDER BY player_key")
        keys = [r["player_key"] for r in rows]
        self.assertIn("telegram:5478650254", keys)
        self.assertIn("telegram:7541672134", keys)


if __name__ == "__main__":
    unittest.main()
