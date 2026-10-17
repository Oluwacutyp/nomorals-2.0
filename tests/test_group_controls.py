"""Infra spec P4 (§7): group controls — modes + toggles + per-game.

Decisions D1–D5 use the spec's recommendations:
- D1 quiet mode YES (never messages first, replies when addressed).
- D2 live game ends honestly when games go off, scores kept.
- D3 voice off = text-replies-only, transcription continues.
- D4 new groups all features ON (default).
- D5 per-game toggles, god-tier (floor = games toggle).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.partner import chat_modes


class _EnvStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-test-group-ctl-")
        self.path = os.path.join(self.tmp.name, "chat_modes.json")
        self.env = mock.patch.dict(
            os.environ, {"NOMORALS_CHAT_MODES_PATH": self.path})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class TestQuietProactiveDuality(_EnvStore):
    def test_proactive_off_sets_quiet(self):
        chat_modes.set_feature("telegram:1", "proactive", False)
        self.assertEqual(chat_modes.get_chat_mode("telegram:1"), "quiet")
        self.assertTrue(chat_modes.is_feature_disabled("telegram:1", "proactive"))
        self.assertFalse(chat_modes.proactive_allowed("telegram:1"))

    def test_proactive_on_clears_quiet(self):
        chat_modes.set_feature("telegram:1", "proactive", False)
        chat_modes.set_feature("telegram:1", "proactive", True)
        self.assertEqual(chat_modes.get_chat_mode("telegram:1"), "normal")
        self.assertTrue(chat_modes.proactive_allowed("telegram:1"))

    def test_quiet_mode_sets_proactive_off(self):
        chat_modes.set_chat_mode("telegram:1", "quiet")
        self.assertTrue(chat_modes.is_feature_disabled("telegram:1", "proactive"))
        self.assertFalse(chat_modes.proactive_allowed("telegram:1"))

    def test_normal_keeps_proactive_toggle(self):
        chat_modes.set_chat_mode("telegram:1", "quiet")
        chat_modes.set_chat_mode("telegram:1", "normal")
        # normal + proactive-off behaves like quiet (toggle view).
        self.assertFalse(chat_modes.proactive_allowed("telegram:1"))
        settings = chat_modes.get_chat_settings("telegram:1")
        self.assertIn("proactive", settings["disabled"])

    def test_quiet_does_not_touch_other_toggles(self):
        chat_modes.set_chat_mode("telegram:1", "quiet")
        self.assertFalse(chat_modes.is_feature_disabled("telegram:1", "games"))
        self.assertFalse(chat_modes.is_feature_disabled("telegram:1", "music"))


class TestNewToggles(_EnvStore):
    def test_images_toggle(self):
        self.assertFalse(chat_modes.is_feature_disabled("telegram:1", "images"))
        chat_modes.set_feature("telegram:1", "images", False)
        self.assertTrue(chat_modes.is_feature_disabled("telegram:1", "images"))
        self.assertEqual(chat_modes.feature_for_command("image"), "images")
        self.assertEqual(chat_modes.feature_for_command("look"), "images")
        self.assertEqual(chat_modes.feature_for_command("lens"), "images")
        self.assertEqual(chat_modes.feature_for_command("caption"), "images")

    def test_d4_new_groups_all_on(self):
        for feature in chat_modes.FEATURES:
            self.assertFalse(chat_modes.is_feature_disabled("telegram:new", feature))
        self.assertEqual(chat_modes.get_chat_mode("telegram:new"), "normal")
        self.assertTrue(chat_modes.proactive_allowed("telegram:new"))

    def test_d3_voice_reply_text_says_transcription_continues(self):
        reply = chat_modes.feature_disabled_reply("voice")
        self.assertIn("still transcribe", reply)
        self.assertIn("text", reply)


class TestPerGameToggles(_EnvStore):
    def test_one_game_off_rest_keeps_running(self):
        chat_modes.set_game("telegram:1", "chess", False)
        self.assertTrue(chat_modes.is_game_disabled("telegram:1", "chess"))
        self.assertFalse(chat_modes.is_game_disabled("telegram:1", "checkers"))
        self.assertFalse(chat_modes.is_feature_disabled("telegram:1", "games"))

    def test_games_off_dominates_per_game(self):
        chat_modes.set_game("telegram:1", "chess", True)
        chat_modes.set_chat_mode("telegram:1", "games_off")
        self.assertTrue(chat_modes.is_game_disabled("telegram:1", "chess"))
        self.assertTrue(chat_modes.is_game_disabled("telegram:1", "checkers"))

    def test_per_game_off_never_reenables_games_off(self):
        chat_modes.set_chat_mode("telegram:1", "games_off")
        chat_modes.set_game("telegram:1", "chess", True)
        self.assertTrue(chat_modes.is_game_disabled("telegram:1", "chess"))

    def test_games_only_clears_per_game_offs(self):
        chat_modes.set_game("telegram:1", "chess", False)
        chat_modes.set_chat_mode("telegram:1", "games_only")
        self.assertFalse(chat_modes.is_game_disabled("telegram:1", "chess"))

    def test_per_game_off_drops_games_only(self):
        chat_modes.set_chat_mode("telegram:1", "games_only")
        chat_modes.set_game("telegram:1", "chess", False)
        self.assertEqual(chat_modes.get_chat_mode("telegram:1"), "normal")

    def test_per_game_isolation_between_chats(self):
        chat_modes.set_game("telegram:1", "chess", False)
        self.assertFalse(chat_modes.is_game_disabled("telegram:2", "chess"))

    def test_per_game_reported_in_settings_and_list(self):
        chat_modes.set_game("telegram:1", "chess", False)
        settings = chat_modes.get_chat_settings("telegram:1")
        self.assertEqual(settings["games_disabled"], ["chess"])
        listed = chat_modes.list_chat_settings()
        self.assertEqual(listed["telegram:1"]["games_disabled"], ["chess"])

    def test_clearing_all_prunes_entry(self):
        chat_modes.set_game("telegram:1", "chess", False)
        chat_modes.set_game("telegram:1", "chess", True)
        self.assertEqual(chat_modes.list_chat_settings(), {})

    def test_game_reply_is_honest(self):
        reply = chat_modes.game_disabled_reply("chess")
        self.assertIn("chess", reply)
        self.assertIn("other games", reply)


class TestComposition(_EnvStore):
    def test_games_only_plus_music_off(self):
        # Mode gates reply-vs-silence, toggles gate features: games-only
        # + music off = games work, music doesn't, normal chat silenced.
        chat_modes.set_chat_mode("telegram:1", "games_only")
        chat_modes.set_feature("telegram:1", "music", False)
        self.assertTrue(chat_modes.is_feature_disabled("telegram:1", "music"))
        self.assertFalse(chat_modes.is_feature_disabled("telegram:1", "games"))
        self.assertEqual(chat_modes.get_chat_mode("telegram:1"), "games_only")

    def test_no_toggle_reenables_mode_silence(self):
        chat_modes.set_chat_mode("telegram:1", "games_off")
        chat_modes.set_game("telegram:1", "chess", True)
        self.assertTrue(chat_modes.is_game_disabled("telegram:1", "chess"))


def _runtime(tmp_home: str):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
    from nomorals.social.chat.gateway import ChatGateway

    class FakeAdapter(ChatAdapter):
        def __init__(self):
            super().__init__(media_dir="/tmp/nm-test-media")
        def run(self, handler):
            pass
        def send(self, chat, text, *, reply_to="", buttons=None, **kwargs):
            return SendResult(ok=True, platform="telegram", message_id="m1")

    settings = load_settings(overrides={
        "home": tmp_home,
        "partner.platforms": "telegram",
        "partner.owner_chats": "5478650254",
        "partner.owner_sender_ids": "5478650254",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    gateway = ChatGateway({"telegram": FakeAdapter()}, db=context.db)
    return PartnerRuntime(context, gateway=gateway)


class TestChatmodeCommand(_EnvStore):
    def test_quiet_and_status(self):
        rt = _runtime(self.tmp.name)
        out = rt.handle_control("/chatmode quiet", "telegram:5478650254")
        self.assertIn("quiet", out)
        out = rt.handle_control("/chatmode status", "telegram:5478650254")
        self.assertIn("quiet", out)
        self.assertIn("proactive off", out)

    def test_feature_images_and_proactive(self):
        rt = _runtime(self.tmp.name)
        out = rt.handle_control("/chatmode feature images off",
                                "telegram:5478650254")
        self.assertIn("images is off", out)
        out = rt.handle_control("/chatmode feature proactive off",
                                "telegram:5478650254")
        self.assertIn("proactive is off", out)
        self.assertIn("quiet", out)  # proactive-off == quiet view

    def test_feature_per_game(self):
        rt = _runtime(self.tmp.name)
        out = rt.handle_control("/chatmode feature games.chess off",
                                "telegram:5478650254")
        self.assertIn("games.chess is off", out)
        out = rt.handle_control("/chatmode status", "telegram:5478650254")
        self.assertIn("games off: chess", out)
        out = rt.handle_control("/chatmode list", "telegram:5478650254")
        self.assertIn("games off: chess", out)

    def test_d2_games_off_ends_live_game(self):
        rt = _runtime(self.tmp.name)
        quits: list[str] = []

        class FakeEngine:
            def live(self, chat_key):
                return SimpleNamespace(game="chess")
            def quit(self, chat_key):
                quits.append(chat_key)
                return ["table closed"]

        rt._game_engine = lambda: FakeEngine()  # type: ignore[method-assign]
        out = rt.handle_control("/chatmode games-off", "telegram:5478650254")
        self.assertIn("ending the live chess, scores kept", out)
        self.assertEqual(quits, ["telegram:5478650254"])

    def test_d2_per_game_off_ends_only_matching_game(self):
        rt = _runtime(self.tmp.name)
        quits: list[str] = []

        class FakeEngine:
            def live(self, chat_key):
                return SimpleNamespace(game="checkers")
            def quit(self, chat_key):
                quits.append(chat_key)
                return ["table closed"]

        rt._game_engine = lambda: FakeEngine()  # type: ignore[method-assign]
        out = rt.handle_control("/chatmode feature games.chess off",
                                "telegram:5478650254")
        self.assertNotIn("ending the live", out)
        self.assertEqual(quits, [])

    def test_per_game_gate_refuses_game_command(self):
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
        rt = _runtime(self.tmp.name)
        # Slash game command choke point: /checkers IS a game command;
        # /chess is not (chess starts via /game chess — next test).
        chat_modes.set_game("telegram:5478650254", "checkers", False)
        msg = ChatMessage(chat=ChatRef(platform="telegram",
                                       chat_id="5478650254", kind=ChatKind.DM),
                          incoming=True, text="/checkers", sender="owner",
                          sender_id="5478650254", meta={"is_owner": True})
        out = rt.handle_control("/checkers", "telegram:5478650254", message=msg)
        self.assertIn("checkers", out)
        self.assertIn("turned off", out)
        msg2 = ChatMessage(chat=ChatRef(platform="telegram",
                                        chat_id="5478650254", kind=ChatKind.DM),
                           incoming=True, text="/duel", sender="owner",
                           sender_id="5478650254", meta={"is_owner": True})
        out = rt.handle_control("/duel", "telegram:5478650254", message=msg2)
        self.assertNotIn("turned off", out)

    def test_per_game_gate_refuses_game_start(self):
        rt = _runtime(self.tmp.name)
        # /game <name> choke point: checkers is a real engine game.
        chat_modes.set_game("telegram:5478650254", "checkers", False)
        out = rt.handle_control("/game checkers", "telegram:5478650254")
        self.assertIn("checkers", out)
        self.assertIn("turned off", out)

    def test_images_gate_refuses_image_command(self):
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
        rt = _runtime(self.tmp.name)
        chat_modes.set_feature("telegram:5478650254", "images", False)
        msg = ChatMessage(chat=ChatRef(platform="telegram",
                                       chat_id="5478650254", kind=ChatKind.DM),
                          incoming=True, text="/image a cat", sender="owner",
                          sender_id="5478650254", meta={"is_owner": True})
        out = rt.handle_control("/image a cat", "telegram:5478650254",
                                message=msg)
        self.assertIn("Images are turned off", out)


class TestNotifierProactiveGate(_EnvStore):
    def test_proactive_off_target_is_skipped_not_sent(self):
        from nomorals.agents.notifier import Notifier

        sent: list[tuple] = []

        class FakeGateway:
            def status(self):
                return {"telegram": {"running_in_session": True}}
            def send(self, platform, chat, text):
                sent.append((platform, chat.chat_id))
                return SimpleNamespace(ok=True, message_id="m1")

        settings = SimpleNamespace(
            partner=SimpleNamespace(owner_chats="telegram:1"))
        context = SimpleNamespace(settings=settings, db=None)
        notifier = Notifier(context, gateway=FakeGateway())
        chat_modes.set_feature("telegram:1", "proactive", False)
        outcome = notifier._deliver("watcher", "title", "body")
        self.assertEqual(sent, [])
        self.assertTrue(outcome.attempts)
        self.assertEqual(outcome.attempts[0]["state"], "skipped")
        self.assertIn("proactive", outcome.attempts[0]["error"])

    def test_proactive_on_target_sends(self):
        from nomorals.agents.notifier import Notifier

        sent: list[tuple] = []

        class FakeGateway:
            def status(self):
                return {"telegram": {"running_in_session": True}}
            def send(self, platform, chat, text):
                sent.append((platform, chat.chat_id))
                return SimpleNamespace(ok=True, message_id="m1")

        settings = SimpleNamespace(
            partner=SimpleNamespace(owner_chats="telegram:1"))
        context = SimpleNamespace(settings=settings, db=None)
        notifier = Notifier(context, gateway=FakeGateway())
        outcome = notifier._deliver("watcher", "title", "body")
        self.assertEqual(sent, [("telegram", "1")])
        self.assertEqual(outcome.attempts[0]["state"], "sent")


if __name__ == "__main__":
    unittest.main()
