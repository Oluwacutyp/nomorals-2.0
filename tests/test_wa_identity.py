"""WhatsApp identity, targeting & display (register #29–#31, #34).

Owner-reported from his WhatsApp self-chat, 2026-11-05:

(#29) /stats answered "📊 unknown's attributes" — the bridge
    reports no sender name for the owner's own messages, the
    adapter passed the empty name through, and game resolution
    substituted the literal "unknown".  Fixed at every layer:
    the bridge names the account (status user_name + own_message
    from), the adapter resolves the owner's display name through
    a chain ending at his number (never empty), and game
    resolution folds WhatsApp's two addressing forms (phone JID
    / LID) into ONE player, labelled by id when no name exists —
    never "unknown".
(#30) /chatmode demanded a raw ``whatsapp:<jid>`` target.  A
    target is now a chat NAME, a phone number, or the full key;
    ambiguity lists candidates, exhaustion names what was tried.
(#31) WhatsApp had no progressive display.  The Go bridge grew
    an edit command (whatsmeow BuildEdit, own messages, the
    platform's edit window) behind the same ``edit_progress``
    seam Telegram uses; the edges are recorded BY NAME in
    platform_limits (other people's messages, the Node/Baileys
    rung's missing edit command).
(#34) "What can you do?" got the base model's stock list.  The
    brain's prompt now carries her REAL catalog (capability_brief,
    computed from the same registry /help renders); the stock
    shapes are pinned here as fixtures that must never appear.

All phone numbers / ids in this file are obviously fake fixtures.
Context homes honour NM_TEST_TMPDIR.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.games.players import PlayerStore
from nomorals.partner import chat_modes
from nomorals.partner.chat_targets import (
    chat_title_index,
    note_chat_title,
    resolve_chat_target,
)
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.platform_limits import for_platform
from nomorals.social.chat.progress import ProgressReporter
from nomorals.social.chat.progress_throttle import ProgressThrottler
from nomorals.social.chat.whatsapp import WhatsAppAdapter

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_JID = "2348000000001@s.whatsapp.net"
OWNER_LID = "100000000000001@lid"
OWNER_NAME = "Peace Testname"
STRANGER_JID = "2348000000002@s.whatsapp.net"
GROUP_JID = "120360000000000001@g.us"


class _RecordingGateway:
    adapters: dict = {}

    def __init__(self) -> None:
        self.sent: list = []

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


class _RuntimeCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-waidentity-", dir=_TEST_TMPDIR)
        self._old = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_CHAT_MODES_PATH"] = os.path.join(
            self.tmp.name, "chat_modes.json")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.gateway = _RecordingGateway()
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        if self._old is None:
            os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
        else:
            os.environ["NOMORALS_CHAT_MODES_PATH"] = self._old
        self.tmp.cleanup()


def _wa_msg(text: str, *, sender: str = "", sender_id: str = "",
            chat_id: str = OWNER_JID, kind: ChatKind = ChatKind.DM,
            alt: str = "", title: str = "") -> ChatMessage:
    msg = ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=chat_id, kind=kind,
                     title=title, peer=chat_id),
        incoming=True, text=text, sender=sender, sender_id=sender_id,
        message_id="m-test")
    if alt:
        msg.meta["sender_alt_id"] = alt
    return msg


# ── #29: the owner's /stats name ───────────────────────────────────────────

class AdapterOwnNameTests(unittest.TestCase):
    """The adapter resolves the owner's display name on his turns."""

    def _adapter(self, status_extra: dict | None = None):
        adapter = WhatsAppAdapter()
        msgs: list = []
        status = {"type": "status", "state": "open", "user": OWNER_JID,
                  "lid": OWNER_LID}
        status.update(status_extra or {})
        adapter._dispatch(status, msgs.append)
        return adapter, msgs

    @staticmethod
    def _own(text: str = "/stats", extra: dict | None = None) -> dict:
        obj = {"type": "own_message", "id": "m1",
               "chat": {"id": OWNER_JID, "kind": "dm", "title": ""},
               "text": text, "ts": 1_700_000_000_000}
        obj.update(extra or {})
        return obj

    def test_status_stores_user_name(self) -> None:
        adapter, _ = self._adapter({"user_name": OWNER_NAME})
        self.assertEqual(adapter.user_name, OWNER_NAME)

    def test_old_bridge_status_does_not_clobber_known_name(self) -> None:
        adapter, msgs = self._adapter({"user_name": OWNER_NAME})
        adapter._dispatch(
            {"type": "status", "state": "open", "user": OWNER_JID,
             "lid": OWNER_LID}, msgs.append)
        self.assertEqual(adapter.user_name, OWNER_NAME)

    def test_self_chat_turn_named_from_account_profile(self) -> None:
        """The Go bridge's own_message carries no usable name on
        older builds — the account profile name from status is the
        resolution, not an empty string."""
        adapter, msgs = self._adapter({"user_name": OWNER_NAME})
        adapter._dispatch(self._own(), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].sender, OWNER_NAME)
        self.assertEqual(msgs[0].sender_id, OWNER_JID)

    def test_self_chat_turn_named_from_event_from(self) -> None:
        adapter, msgs = self._adapter()
        adapter._dispatch(self._own(extra={
            "from": {"id": OWNER_JID, "name": OWNER_NAME}}), msgs.append)
        self.assertEqual(msgs[0].sender, OWNER_NAME)

    def test_self_chat_turn_falls_back_to_the_number(self) -> None:
        """No name anywhere: the honest label is his number —
        never an empty sender for downstream to placeholder."""
        adapter, msgs = self._adapter()
        adapter._dispatch(self._own(), msgs.append)
        self.assertEqual(msgs[0].sender, "2348000000001")

    def test_message_event_records_alt_addressing(self) -> None:
        adapter, msgs = self._adapter()
        adapter._dispatch(
            {"type": "message", "id": "m9",
             "chat": {"id": GROUP_JID, "kind": "group", "title": "Crew"},
             "from": {"id": OWNER_LID, "name": OWNER_NAME,
                      "alt": OWNER_JID},
             "text": "gm", "ts": 1_700_000_000_000}, msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].sender_id, OWNER_LID)
        self.assertEqual(msgs[0].meta.get("sender_alt_id"), OWNER_JID)


class GamePlayerIdentityTests(_RuntimeCase):
    """Game resolution: one WhatsApp human, one player, real name."""

    def test_owner_self_chat_stats_show_his_name(self) -> None:
        msg = _wa_msg("/stats", sender=OWNER_NAME, sender_id=OWNER_JID,
                      title="Message yourself")
        player = self.runtime._game_player(msg)
        self.assertIsNotNone(player)
        self.assertEqual(player.name, OWNER_NAME)
        self.assertEqual(player.key, f"whatsapp:{OWNER_JID}")
        reply = self.runtime._control_stats("", player=player)
        self.assertIn(f"{OWNER_NAME}'s attributes", reply)
        self.assertNotIn("unknown", reply)

    def test_lid_and_number_forms_are_one_player(self) -> None:
        by_lid = self.runtime._game_player(_wa_msg(
            "/stats", sender=OWNER_NAME, sender_id=OWNER_LID,
            alt=OWNER_JID))
        by_number = self.runtime._game_player(_wa_msg(
            "/stats", sender=OWNER_NAME, sender_id=OWNER_JID,
            alt=OWNER_LID))
        self.assertEqual(by_lid.key, f"whatsapp:{OWNER_JID}")
        self.assertEqual(by_number.key, by_lid.key)

    def test_lid_keyed_profile_folds_into_the_number_key(self) -> None:
        store = PlayerStore(self.context.db)
        lid_player = store.get(f"whatsapp:{OWNER_LID}",
                               name=OWNER_NAME, platform="whatsapp")
        from nomorals.games.players import Player
        store.add_coins(
            Player(key=f"whatsapp:{OWNER_LID}", platform="whatsapp",
                   name=OWNER_NAME), 50)
        self.assertEqual(lid_player.coins, 0)  # stale copy; fold is on read
        msg = _wa_msg("/stats", sender=OWNER_NAME, sender_id=OWNER_LID,
                      alt=OWNER_JID)
        player = self.runtime._game_player(msg)
        prof = store.get_for(player)
        self.assertEqual(prof.key, f"whatsapp:{OWNER_JID}")
        self.assertEqual(prof.coins, 50)
        gone = self.context.db.query_one(
            "SELECT * FROM game_players WHERE player_key = ?",
            (f"whatsapp:{OWNER_LID}",))
        self.assertIsNone(gone)

    def test_nameless_member_labelled_by_number_never_unknown(self) -> None:
        msg = _wa_msg("/stats", sender="", sender_id=STRANGER_JID,
                      chat_id=STRANGER_JID)
        player = self.runtime._game_player(msg)
        self.assertIsNotNone(player)
        self.assertEqual(player.name, "2348000000002")
        reply = self.runtime._control_stats("", player=player)
        self.assertIn("2348000000002's attributes", reply)
        self.assertNotIn("unknown", reply)


# ── #30: chat targeting ────────────────────────────────────────────────────

class ChatTargetResolutionTests(_RuntimeCase):
    def test_no_target_is_this_chat(self) -> None:
        res = resolve_chat_target("", current_key="whatsapp:x")
        self.assertEqual(res.key, "whatsapp:x")

    def test_full_key_verbatim(self) -> None:
        res = resolve_chat_target("telegram:555", current_key="telegram:9",
                                  current_platform="telegram")
        self.assertEqual(res.key, "telegram:555")

    def test_phone_number_names_a_whatsapp_dm(self) -> None:
        res = resolve_chat_target("+234 800 000 0002",
                                  current_key=f"whatsapp:{OWNER_JID}",
                                  current_platform="whatsapp")
        self.assertEqual(res.key, "whatsapp:2348000000002@s.whatsapp.net")

    def test_bare_digits_from_telegram_are_a_telegram_id(self) -> None:
        res = resolve_chat_target("123456789", current_key="telegram:9",
                                  current_platform="telegram")
        self.assertEqual(res.key, "telegram:123456789")

    def test_name_resolves_from_the_title_index(self) -> None:
        note_chat_title(self.context.db, ChatRef(
            platform="whatsapp", chat_id=GROUP_JID, kind=ChatKind.GROUP,
            title="Think Creativity"))
        res = resolve_chat_target(
            "think creativity", current_key=f"whatsapp:{OWNER_JID}",
            current_platform="whatsapp", db=self.context.db)
        self.assertEqual(res.key, f"whatsapp:{GROUP_JID}")
        self.assertEqual(res.title, "Think Creativity")

    def test_name_resolves_from_adapter_listings(self) -> None:
        class _Lister:
            name = "whatsapp"

            def chats(self):
                return [{"id": GROUP_JID, "title": "Harbour Crew",
                         "kind": "group"}]

            def groups(self):
                return []

        res = resolve_chat_target(
            "Harbour", current_key=f"whatsapp:{OWNER_JID}",
            current_platform="whatsapp", adapters=[_Lister()])
        self.assertEqual(res.key, f"whatsapp:{GROUP_JID}")

    def test_ambiguous_name_lists_candidates(self) -> None:
        for cid in ("111@g.us", "222@g.us"):
            note_chat_title(self.context.db, ChatRef(
                platform="whatsapp", chat_id=cid, kind=ChatKind.GROUP,
                title="Crew"))
        res = resolve_chat_target(
            "Crew", current_key=f"whatsapp:{OWNER_JID}",
            current_platform="whatsapp", db=self.context.db)
        self.assertFalse(res.resolved)
        self.assertTrue(res.ambiguous)
        text = res.describe("Crew")
        self.assertIn("whatsapp:111@g.us", text)
        self.assertIn("whatsapp:222@g.us", text)

    def test_unknown_name_reports_what_was_tried(self) -> None:
        res = resolve_chat_target(
            "Nowhere", current_key=f"whatsapp:{OWNER_JID}",
            current_platform="whatsapp", db=self.context.db)
        self.assertFalse(res.resolved)
        text = res.describe("Nowhere")
        self.assertIn("no chat matches 'Nowhere'", text)
        self.assertIn("chat name", text)

    def test_title_index_roundtrip(self) -> None:
        ref = ChatRef(platform="whatsapp", chat_id=GROUP_JID,
                      kind=ChatKind.GROUP, title="Harbour Crew")
        note_chat_title(self.context.db, ref)
        self.assertIn((f"whatsapp:{GROUP_JID}", "Harbour Crew"),
                      chat_title_index(self.context.db))


class ChatmodeTargetingTests(_RuntimeCase):
    def _owner_msg(self, text: str, chat_key: str) -> ChatMessage:
        platform, _, chat_id = chat_key.partition(":")
        return ChatMessage(
            chat=ChatRef(platform=platform, chat_id=chat_id,
                         kind=ChatKind.DM),
            incoming=True, text=text, sender="owner",
            sender_id="424242", meta={"is_owner": True})

    def test_target_by_phone_number_from_whatsapp(self) -> None:
        key = f"whatsapp:{OWNER_JID}"
        out = self.runtime.handle_control(
            "/chatmode games-only +234 800 000 0002", key,
            message=self._owner_msg("/chatmode games-only", key))
        self.assertTrue(
            chat_modes.is_games_only("whatsapp:2348000000002@s.whatsapp.net"))
        self.assertIn("whatsapp:2348000000002@s.whatsapp.net", out)
        self.assertFalse(chat_modes.is_games_only(key))

    def test_target_by_chat_name(self) -> None:
        note_chat_title(self.context.db, ChatRef(
            platform="whatsapp", chat_id=GROUP_JID, kind=ChatKind.GROUP,
            title="Think Creativity"))
        key = f"whatsapp:{OWNER_JID}"
        out = self.runtime.handle_control(
            "/chatmode quiet Think Creativity", key,
            message=self._owner_msg("/chatmode quiet", key))
        self.assertEqual(
            chat_modes.get_chat_mode(f"whatsapp:{GROUP_JID}"), "quiet")
        self.assertIn("Think Creativity", out)
        self.assertIn(f"whatsapp:{GROUP_JID}", out)

    def test_feature_target_by_name(self) -> None:
        note_chat_title(self.context.db, ChatRef(
            platform="whatsapp", chat_id=GROUP_JID, kind=ChatKind.GROUP,
            title="Think Creativity"))
        key = f"whatsapp:{OWNER_JID}"
        out = self.runtime.handle_control(
            "/chatmode feature music off Think Creativity", key,
            message=self._owner_msg("/chatmode feature", key))
        settings = chat_modes.get_chat_settings(f"whatsapp:{GROUP_JID}")
        self.assertIn("music", settings["disabled"])
        self.assertIn("Think Creativity", out)

    def test_ambiguous_name_applies_nothing(self) -> None:
        for cid in ("111@g.us", "222@g.us"):
            note_chat_title(self.context.db, ChatRef(
                platform="whatsapp", chat_id=cid, kind=ChatKind.GROUP,
                title="Crew"))
        key = f"whatsapp:{OWNER_JID}"
        out = self.runtime.handle_control(
            "/chatmode games-only Crew", key,
            message=self._owner_msg("/chatmode games-only Crew", key))
        self.assertIn("matches more than one chat", out)
        self.assertFalse(chat_modes.is_games_only("whatsapp:111@g.us"))
        self.assertFalse(chat_modes.is_games_only("whatsapp:222@g.us"))
        self.assertFalse(chat_modes.is_games_only(key))

    def test_unknown_name_is_an_honest_error(self) -> None:
        key = f"whatsapp:{OWNER_JID}"
        out = self.runtime.handle_control(
            "/chatmode games-only Nowhere", key,
            message=self._owner_msg("/chatmode games-only Nowhere", key))
        self.assertIn("no chat matches 'Nowhere'", out)
        self.assertFalse(chat_modes.is_games_only(key))

    def test_short_garbage_target_still_errors(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode games-only 555", "telegram:999",
            message=self._owner_msg("/chatmode games-only 555",
                                    "telegram:999"))
        self.assertIn("no chat matches '555'", out)
        self.assertFalse(chat_modes.is_games_only("telegram:999"))


# ── #31: progressive display ───────────────────────────────────────────────

class ProgressiveDisplayTests(unittest.TestCase):
    def _adapter(self, responder):
        adapter = WhatsAppAdapter()
        adapter.connected.set()
        calls: list = []

        def _cmd(cmd, timeout=25.0):
            calls.append(cmd)
            return responder(cmd)

        adapter._send_cmd = _cmd  # type: ignore[method-assign]
        adapter._progress = ProgressThrottler(min_interval_s=0)
        return adapter, calls

    _CHAT = ChatRef(platform="whatsapp", chat_id=GROUP_JID,
                    kind=ChatKind.GROUP)

    def test_edit_progress_sends_the_bridge_edit_command(self) -> None:
        adapter, calls = self._adapter(
            lambda cmd: {"ok": True, "message_id": "M1"})
        out = adapter.edit_progress(self._CHAT, "M1", "halfway there")
        self.assertIsNotNone(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["cmd"], "edit")
        self.assertEqual(calls[0]["chat"], GROUP_JID)
        self.assertEqual(calls[0]["message_id"], "M1")
        self.assertIn("halfway there", calls[0]["text"])

    def test_edit_progress_bridge_refusal_is_none_not_a_raise(self) -> None:
        adapter, _ = self._adapter(lambda cmd: {
            "ok": False,
            "error": "edit target M9 is not one of this bridge's own "
                     "messages in that chat (only own messages can "
                     "be edited)"})
        self.assertIsNone(adapter.edit_progress(self._CHAT, "M9", "x"))

    def test_edit_progress_disconnected_is_none(self) -> None:
        adapter = WhatsAppAdapter()
        self.assertIsNone(adapter.edit_progress(self._CHAT, "M1", "x"))

    def test_edit_progress_throttles_unchanged_text(self) -> None:
        adapter, calls = self._adapter(
            lambda cmd: {"ok": True, "message_id": "M1"})
        adapter._progress = ProgressThrottler(min_interval_s=60)
        self.assertIsNotNone(adapter.edit_progress(self._CHAT, "M1", "a"))
        self.assertIsNone(adapter.edit_progress(self._CHAT, "M1", "a"))
        self.assertIsNone(adapter.edit_progress(self._CHAT, "M1", "b"))
        self.assertEqual(len(calls), 1)

    def test_progress_reporter_flows_over_whatsapp(self) -> None:
        adapter, calls = self._adapter(
            lambda cmd: {"ok": True, "message_id": "M1"})
        reporter = ProgressReporter(
            lambda text: adapter.edit_progress(self._CHAT, "M1", text),
            throttler=ProgressThrottler(min_interval_s=0))
        reporter.milestone("working — real step one")
        reporter.finish("done — real result")
        texts = [c["text"] for c in calls if c["cmd"] == "edit"]
        self.assertTrue(any("real step one" in t for t in texts))
        self.assertTrue(any("real result" in t for t in texts))

    def test_platform_limits_record_the_edit_truth(self) -> None:
        wa = for_platform("whatsapp")
        self.assertTrue(wa.supports_text_edit)
        self.assertIn("own messages only", wa.text_edit_scope)
        self.assertIn("other people's messages", wa.text_edit_scope)
        self.assertIn("Baileys", wa.text_edit_scope)
        self.assertTrue(for_platform("telegram-bot").supports_text_edit)
        tg = for_platform("telegram")
        self.assertFalse(tg.supports_text_edit)
        self.assertIn("edit_progress", tg.text_edit_scope)


# ── #34: capability grounding ──────────────────────────────────────────────

# The stock shapes from the owner's screenshot, assembled from
# fragments so the verbatim boilerplate never sits in this file
# as one literal (same discipline as test_surface_laws).
_D = chr(8212)  # em dash
_STOCK_SHAPES = (
    f"Answer questions {_D} factual, technical, creative",
    f"Write and debug code {_D} any language, any framework",
    f"Explain concepts {_D} from simple to advanced",
    f"Creative writing {_D} stories, poems, scripts, lyrics",
    "What are you working on?",
)


class CapabilityBriefTests(unittest.TestCase):
    def test_brief_is_computed_from_the_registry(self) -> None:
        from nomorals.social.chat.control import CONTROL_COMMANDS, LIST_GROUPS, capability_brief
        brief = capability_brief()
        self.assertTrue(brief.startswith(
            f"{len(CONTROL_COMMANDS)} commands across "
            f"{len(LIST_GROUPS)} systems:"))
        for _title, kinds in LIST_GROUPS:
            self.assertIn(f"/{kinds[0]}", brief)

    def test_brief_names_real_devon_systems(self) -> None:
        from nomorals.social.chat.control import capability_brief
        brief = capability_brief()
        # The brief is a map — the surfaced picks per system, with
        # the remainder counted, not hidden.
        for needle in ("/status", "/search", "/wisdom", "/code",
                       "/music", "/game", "/help", "more)"):
            self.assertIn(needle, brief)

    def test_brief_has_no_stock_shapes(self) -> None:
        from nomorals.social.chat.control import capability_brief
        brief = capability_brief()
        for shape in _STOCK_SHAPES:
            self.assertNotIn(shape, brief)

    def test_owner_prompt_carries_the_catalog_not_boilerplate(self) -> None:
        from nomorals.partner.context import PartnerContextBuilder
        from nomorals.partner.mood import MoodEngine
        from nomorals.partner.persona import default_persona
        from nomorals.partner.relationship import Relationship

        persona = default_persona()
        engine = MoodEngine(persona.baselines, now=1_000_000.0)
        system = PartnerContextBuilder().build(
            persona=persona, mood=engine, relationship=Relationship(),
            platform="telegram")
        self.assertIn("your real systems", system.content)
        self.assertIn("/wisdom", system.content)
        self.assertIn("/status", system.content)
        from nomorals.social.chat.control import CONTROL_COMMANDS
        self.assertIn(f"{len(CONTROL_COMMANDS)} commands",
                      system.content)
        for shape in _STOCK_SHAPES:
            self.assertNotIn(shape, system.content)

    def test_restricted_prompt_omits_the_catalog(self) -> None:
        from nomorals.partner.context import PartnerContextBuilder
        from nomorals.partner.mood import MoodEngine
        from nomorals.partner.persona import default_persona
        from nomorals.partner.relationship import Relationship

        persona = default_persona()
        engine = MoodEngine(persona.baselines, now=1_000_000.0)
        system = PartnerContextBuilder().build(
            persona=persona, mood=engine, relationship=Relationship(),
            platform="telegram", include_capabilities=False)
        self.assertNotIn("your real systems", system.content)


if __name__ == "__main__":
    unittest.main()
