"""Two Telegram bots: gated (telegram-bot) + personal (telegram-personal).

Covers the two-bot split end to end:

- Game identity: the same Telegram user id via ``telegram``,
  ``telegram-bot`` and ``telegram-personal`` folds to ONE player_key,
  ONE game_players profile, ONE game_gems row, ONE leaderboard slot —
  coins/gems granted from different endpoints sum, never split.
- Userbot loop guard: ``companion_bot_ids`` is a SET holding both bot
  ids; the userbot refuses to reply to either bot's messages.
- Two-bot build/routing: ``build_both`` yields two adapters with
  distinct names, media dirs, partner_context flags and owner gating;
  ``route`` picks by platform or by bot id.
- Gating per bot: personal bot denies non-owners (no brain, no
  credit); gated bot denies restricted commands to non-owners with
  ``"that one's just for the owner."``; the personal adapter itself
  sends the honest denial and drops the message.
- Media dir separation: the two bots never share a media dir.
- Settings wiring: ``build_adapter`` builds ``telegram-personal``
  from ChatSettings, and the userbot gets both companion ids.
- Brain rule: personal-bot messages carry partner context; gated-bot
  group messages don't (covered via the adapter meta stamping +
  the brain's ``partner_context`` branch — see the meta assertions).
"""
from __future__ import annotations

import types
import unittest

from nomorals.agents.partner.brain import PartnerBrain  # noqa: F401 - import guard
from nomorals.core.config import ChatSettings, PartnerSettings
from nomorals.games.economy import gem_balance, grant_gems
from nomorals.games.players import (
    GAME_IDENTITY_ALIASES,
    Leaderboard,
    Player,
    PlayerStore,
)
from nomorals.social.chat import build_adapter, build_adapters
from nomorals.social.chat.base import ChatRef
from nomorals.social.chat.telegram import TelegramAdapter, TelegramBotAdapter
from nomorals.social.chat.two_bots import (
    DENY_NOT_OWNER,
    DENY_RESTRICTED,
    TwoBotConfig,
    bot_id_from_token,
    build_both,
    companion_bot_ids,
    gate_message,
    route,
)
from nomorals.storage.db import Database

OWNER_ID = "7541672134"
STRANGER_ID = "111222333"


def make_store() -> PlayerStore:
    db = Database(":memory:")
    db.migrate()
    return PlayerStore(db)


def make_bot(**kw) -> TelegramBotAdapter:
    args = {"token": "111:fake-token-for-tests", "media_dir": "/tmp/tb-media"}
    args.update(kw)
    return TelegramBotAdapter(**args)


def personal_bot() -> TelegramBotAdapter:
    return make_bot(
        token="222:personal-fake-token",
        adapter_name="telegram-personal",
        partner_context=True,
        owner_only=True,
        owner_sender_ids={OWNER_ID},
        owner_chat_keys={f"telegram-personal:{OWNER_ID}"},
        media_dir="/tmp/tb-media-personal",
        lock_path="/tmp/tb-personal.lock",
    )


def gated_bot() -> TelegramBotAdapter:
    return make_bot(
        token="111:gated-fake-token",
        adapter_name="telegram-bot",
        partner_context=False,
        owner_only=False,
        media_dir="/tmp/tb-media-gated",
        lock_path="/tmp/tb-gated.lock",
    )


def tg_update(sender_id: str, chat_id: str, text: str = "hello",
              *, chat_type: str = "private") -> dict:
    return {
        "update_id": 1,
        "message": {
            "message_id": 7,
            "date": 1700000000,
            "text": text,
            "from": {"id": int(sender_id), "is_bot": False,
                     "username": "someone", "first_name": "Someone"},
            "chat": {"id": int(chat_id), "type": chat_type},
        },
    }


class TestGameIdentityAlias(unittest.TestCase):
    def test_alias_table_has_personal(self):
        self.assertEqual(GAME_IDENTITY_ALIASES.get("telegram-personal"),
                         "telegram")
        self.assertEqual(GAME_IDENTITY_ALIASES.get("telegram-bot"), "telegram")

    def test_same_user_folds_to_one_key(self):
        keys = {
            Player.from_sender(p, OWNER_ID, name="Owner").key
            for p in ("telegram", "telegram-bot", "telegram-personal")
        }
        self.assertEqual(keys, {f"telegram:{OWNER_ID}"})

    def test_one_profile_one_wallet_one_leaderboard_slot(self):
        store = make_store()
        db = store.db
        # Same human arrives on all three endpoints; wins/coins land once.
        for platform in ("telegram", "telegram-bot", "telegram-personal"):
            player = Player.from_sender(platform, OWNER_ID, name="Owner")
            store.get_for(player)
        store.add_coins(Player.from_sender("telegram-bot", OWNER_ID), 40,
                        "bot win")
        store.add_coins(Player.from_sender("telegram-personal", OWNER_ID), 60,
                        "personal win")
        # Gems granted via two different endpoint keys must sum, not split.
        grant_gems(db, Player.from_sender("telegram-bot", OWNER_ID).key, 5,
                   "bot")
        grant_gems(db, Player.from_sender("telegram-personal", OWNER_ID).key,
                   7, "personal")
        rows = db.query("SELECT * FROM game_players") or []
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_key"], f"telegram:{OWNER_ID}")
        self.assertEqual(store.get(f"telegram:{OWNER_ID}").coins, 100)
        gem_rows = db.query("SELECT * FROM game_gems") or []
        self.assertEqual(len(gem_rows), 1)
        self.assertEqual(gem_balance(db, f"telegram:{OWNER_ID}"), 12)
        # A game played via the personal endpoint lands on the same
        # profile — one leaderboard slot, not a second player.
        store.record_outcome(
            Player.from_sender("telegram-personal", OWNER_ID),
            won=True, game="duel", points=10)
        top = Leaderboard(store).top()
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0]["points"], 10)
        self.assertEqual(top[0]["games"], 1)
        # A different human is still a different player.
        store.get_for(Player.from_sender("telegram-personal", STRANGER_ID,
                                         name="Stranger"))
        self.assertEqual(len(db.query("SELECT * FROM game_players") or []), 2)


class TestCompanionBotIds(unittest.TestCase):
    def test_userbot_ignores_both_bots(self):
        userbot = TelegramAdapter(
            api_id=1, api_hash="x",
            companion_bot_ids={111, 222},
        )
        self.assertEqual(userbot.companion_bot_ids, frozenset({111, 222}))
        for bot_id in (111, 222):
            event = types.SimpleNamespace(sender_id=bot_id, sender=None,
                                          message=None)
            msg = types.SimpleNamespace(sender_id=str(bot_id), from_id=None)
            self.assertTrue(userbot._sender_is_bot(event, msg),
                            f"bot {bot_id} must be detected")
        event = types.SimpleNamespace(sender_id=int(OWNER_ID), sender=None,
                                      message=None)
        msg = types.SimpleNamespace(sender_id=OWNER_ID, from_id=None)
        self.assertFalse(userbot._sender_is_bot(event, msg))

    def test_singular_back_compat_folds_into_set(self):
        userbot = TelegramAdapter(api_id=1, api_hash="x",
                                  companion_bot_id="111")
        self.assertEqual(userbot.companion_bot_ids, frozenset({111}))
        userbot2 = TelegramAdapter(api_id=1, api_hash="x",
                                   companion_bot_id="111",
                                   companion_bot_ids=["222"])
        self.assertEqual(userbot2.companion_bot_ids, frozenset({111, 222}))


class TestTwoBotBuildAndRoute(unittest.TestCase):
    def test_bot_id_from_token(self):
        self.assertEqual(bot_id_from_token("111:secret"), 111)
        self.assertIsNone(bot_id_from_token(""))
        self.assertIsNone(bot_id_from_token("no-colon"))

    def test_build_both(self):
        cfg = TwoBotConfig(
            gated_enabled=True, gated_token="111:gated",
            personal_enabled=True, personal_token="222:personal",
            owner_sender_ids={OWNER_ID},
            owner_chat_keys={f"telegram-personal:{OWNER_ID}"},
        )
        adapters = build_both(cfg)
        gated = adapters["telegram-bot"]
        personal = adapters["telegram-personal"]
        self.assertEqual(gated.name, "telegram-bot")
        self.assertEqual(personal.name, "telegram-personal")
        self.assertEqual(gated._bot_id, 111)
        self.assertEqual(personal._bot_id, 222)
        self.assertFalse(gated.partner_context)
        self.assertTrue(personal.partner_context)
        self.assertFalse(gated.owner_only)
        self.assertTrue(personal.owner_only)
        self.assertEqual(companion_bot_ids(cfg), {111, 222})
        self.assertEqual(cfg.bot_ids(), {111, 222})

    def test_build_both_skips_disabled(self):
        cfg = TwoBotConfig(gated_enabled=True, gated_token="111:gated")
        adapters = build_both(cfg)
        self.assertIsNotNone(adapters["telegram-bot"])
        self.assertIsNone(adapters["telegram-personal"])

    def test_route_by_platform_and_bot_id(self):
        adapters = build_both(TwoBotConfig(
            gated_enabled=True, gated_token="111:gated",
            personal_enabled=True, personal_token="222:personal",
            owner_sender_ids={OWNER_ID}))
        self.assertIs(route(adapters, ChatRef(platform="telegram-personal",
                                              chat_id=OWNER_ID)),
                      adapters["telegram-personal"])
        self.assertIs(route(adapters, ChatRef(platform="telegram-bot",
                                              chat_id="1")),
                      adapters["telegram-bot"])
        self.assertIs(route(adapters, ChatRef(platform="telegram-personal",
                                              chat_id=OWNER_ID), bot_id=222),
                      adapters["telegram-personal"])
        self.assertIs(route(adapters, ChatRef(platform="telegram-bot",
                                              chat_id="1"), bot_id=111),
                      adapters["telegram-bot"])
        # Userbot platform is NOT routed by this router.
        self.assertIsNone(route(adapters, ChatRef(platform="telegram",
                                                  chat_id=OWNER_ID)))
        self.assertIsNone(route(adapters, ChatRef(platform="telegram-bot",
                                                  chat_id="1"), bot_id=999))

    def test_media_dirs_separate(self):
        adapters = build_both(TwoBotConfig(
            gated_enabled=True, gated_token="111:gated",
            personal_enabled=True, personal_token="222:personal"))
        self.assertNotEqual(adapters["telegram-bot"].media_dir,
                            adapters["telegram-personal"].media_dir)
        self.assertIn("telegram-bot", adapters["telegram-bot"].media_dir)
        self.assertIn("telegram-personal",
                      adapters["telegram-personal"].media_dir)


class TestGatingPerBot(unittest.TestCase):
    def test_personal_denies_stranger_allows_owner(self):
        personal = personal_bot()
        verdict, reply = gate_message(personal, is_owner=False)
        self.assertEqual(verdict, "deny")
        self.assertEqual(reply, DENY_NOT_OWNER)
        verdict, reply = gate_message(personal, is_owner=True)
        self.assertEqual((verdict, reply), ("allow", ""))
        # Even a "restricted" flag doesn't change the personal rule:
        # the owner is always allowed, strangers always denied.
        self.assertEqual(gate_message(personal, is_owner=False,
                                      is_restricted_command=True)[0], "deny")

    def test_gated_denies_restricted_to_stranger(self):
        gated = gated_bot()
        verdict, reply = gate_message(gated, is_owner=False,
                                      is_restricted_command=True)
        self.assertEqual((verdict, reply), ("deny", DENY_RESTRICTED))
        self.assertEqual(gate_message(gated, is_owner=False)[0], "allow")
        self.assertEqual(gate_message(gated, is_owner=True,
                                      is_restricted_command=True)[0], "allow")

    def test_personal_adapter_denies_stranger_with_honest_message(self):
        personal = personal_bot()
        calls: list[tuple] = []
        personal._api = lambda method, **kw: calls.append((method, kw)) or {
            "ok": True}  # type: ignore[method-assign]
        # Stranger DM: dropped, denial sent directly, no brain delivery.
        self.assertIsNone(personal._convert(
            tg_update(STRANGER_ID, STRANGER_ID, "/status")))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "sendMessage")
        self.assertIn("private", calls[0][1]["text"])
        # Stranger in a group the bot sits in: denied too (sender, not
        # chat id, is the gate).
        calls.clear()
        self.assertIsNone(personal._convert(
            tg_update(STRANGER_ID, "-100123", "hi", chat_type="group")))
        self.assertEqual(len(calls), 1)
        # Edited messages from strangers: dropped WITHOUT a re-denial
        # (no spam on every edit).
        calls.clear()
        update = tg_update(STRANGER_ID, STRANGER_ID, "edited")
        update["edited_message"] = update.pop("message")
        self.assertIsNone(personal._convert(update))
        self.assertEqual(calls, [])

    def test_personal_adapter_delivers_owner_with_partner_context(self):
        personal = personal_bot()
        msg = personal._convert(tg_update(OWNER_ID, OWNER_ID, "hi love"))
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.chat.platform, "telegram-personal")
        self.assertTrue(msg.meta.get("partner_context"))
        self.assertEqual(msg.meta.get("bot_role"), "personal")

    def test_gated_adapter_stamps_no_partner_context(self):
        gated = gated_bot()
        msg = gated._convert(tg_update(OWNER_ID, OWNER_ID, "hi"))
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.chat.platform, "telegram-bot")
        self.assertFalse(msg.meta.get("partner_context"))
        self.assertEqual(msg.meta.get("bot_role"), "gated")

    def test_personal_callback_denies_stranger(self):
        personal = personal_bot()
        answers: list[tuple] = []
        personal._answer_callback = lambda q, text="": answers.append((q, text))  # type: ignore[method-assign]
        personal._verify_callback = lambda signed: "status"  # type: ignore[method-assign]
        delivered: list = []
        personal._handle_callback(
            {"id": "q1", "from": {"id": int(STRANGER_ID), "is_bot": False},
             "message": {"chat": {"id": int(STRANGER_ID), "type": "private"}},
             "data": "status|sig"},
            delivered.append)
        self.assertEqual(delivered, [])
        self.assertEqual(answers, [("q1", "Not authorized")])


class TestSettingsWiring(unittest.TestCase):
    def _settings(self) -> types.SimpleNamespace:
        chat = ChatSettings(
            telegram_enabled=True,
            telegram_api_id="12345",
            telegram_api_hash="hash",
            telegram_bot_enabled=True,
            telegram_bot_token="111:gated",
            telegram_bot_chats="",
            telegram_personal_bot_enabled=True,
            telegram_personal_bot_token="222:personal",
            telegram_personal_bot_chats="",
        )
        partner = PartnerSettings(
            platforms="telegram-bot,telegram-personal",
            owner_sender_ids=OWNER_ID,
            owner_chats=f"telegram-personal:{OWNER_ID}",
        )
        return types.SimpleNamespace(chat=chat, partner=partner,
                                     resolve=lambda p: p)

    def test_build_adapter_personal(self):
        adapter = build_adapter(self._settings(), "telegram-personal")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter.name, "telegram-personal")
        self.assertTrue(adapter.owner_only)
        self.assertTrue(adapter.partner_context)
        self.assertIn(OWNER_ID, adapter.owner_sender_ids)
        self.assertIn(f"telegram-personal:{OWNER_ID}",
                      adapter.owner_chat_keys)

    def test_build_adapter_userbot_gets_both_companion_ids(self):
        adapter = build_adapter(self._settings(), "telegram")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter.companion_bot_ids, frozenset({111, 222}))

    def test_build_adapters_includes_personal(self):
        adapters, skipped = build_adapters(self._settings())
        self.assertIn("telegram-bot", adapters)
        self.assertIn("telegram-personal", adapters)
        self.assertEqual(skipped, [])

    def test_build_adapter_userbot_empty_session_defaults_to_file(self):
        # Empty NM_CHAT_TELEGRAM_SESSION must NOT resolve to the home
        # directory (Settings.resolve("") == home_path): the session
        # must be a file path ending in telegram.session.
        settings = self._settings()
        settings.chat.telegram_session = ""
        adapter = build_adapter(settings, "telegram")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertTrue(str(adapter.session_path).endswith("telegram.session"))
        self.assertNotEqual(str(adapter.session_path), "")

    def test_build_adapter_userbot_explicit_session_kept(self):
        settings = self._settings()
        settings.chat.telegram_session = "custom/my.session"
        adapter = build_adapter(settings, "telegram")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(str(adapter.session_path), "custom/my.session")

    def test_build_adapter_whatsapp_empty_outbox_defaults(self):
        # Same empty-path trap: an explicitly-emptied outbox dir must
        # fall back to data/chat/outbox, not resolve to the home dir.
        settings = self._settings()
        settings.chat.whatsapp_enabled = True
        settings.chat.whatsapp_outbox_dir = ""
        adapter = build_adapter(settings, "whatsapp")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertTrue(
            str(adapter.outbox.path).endswith(
                "data/chat/outbox/whatsapp.jsonl"))


if __name__ == "__main__":
    unittest.main()
