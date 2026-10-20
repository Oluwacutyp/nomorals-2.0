"""Pass #56 — conversation state + context assembly.

Pins the scope contract the whole context layer stands on: the brain's
conversation scope is the DETERMINISTIC conversation id (the chat key,
``platform:chat_id`` — thread-aware), never the OS Session's random
record id. Before the fix the gateway attached ``session.id``
(``sess_…``), which silently killed the double-reply guard, starved
per-chat profile mining, orphaned all history when a session ended,
and split turns written during a session-store hiccup into a
``transient-…`` scope no reader ever loaded. Also pinned: the history
window is a hard limit (it returned up to 2x), and the message being
answered is not duplicated into the prompt (persisted inbound +
appended live turn showed it twice).
"""

from __future__ import annotations

import random
import tempfile
import threading
import unittest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse
from nomorals.os.session_bridge import SessionBridge
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.gateway import ChatGateway


class _CapRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[list] = []

    def chat(self, messages, params=None, **kw):
        self.calls += 1
        self.seen.append(list(messages))
        return LLMResponse(text="mhm. yeah, i'm around", model="fake")


class _FakeAdapter:
    def start(self, handler):
        return True


def _make(tmp: str):
    settings = load_settings(overrides={
        "home": tmp, "partner.platforms": "local",
        "chat.local_enabled": "true"})
    context = build_context(settings, with_executor=False, with_tools=False)
    router = _CapRouter()
    context.router = router
    brain = PartnerBrain(context)
    brain.presence_rng = random.Random(34)
    return context, brain, router


def _tg(text: str, chat_id: str = "123", thread_id=None,
        sender: str = "Pelumi", platform: str = "telegram-bot") -> ChatMessage:
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM,
                   peer="Pelumi", thread_id=thread_id)
    return ChatMessage(chat=chat, incoming=True, text=text, sender=sender)


class _ScopeBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ctx56-")
        self.context, self.brain, self.router = _make(self.tmp.name)
        self.db = self.context.db
        self.bridge = SessionBridge(db=self.db)
        self.gw = ChatGateway({"telegram-bot": _FakeAdapter()}, db=self.db,
                              session_bridge=self.bridge,
                              owner_chats={"telegram-bot:123"})
        self.gw._inbound = lambda m: None  # attach side effects only

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def attach(self, msg: ChatMessage) -> str:
        self.gw._on_inbound(msg)
        return str(msg.meta.get("os_session_id") or "")


class ScopeAttachTests(_ScopeBase):
    def test_gateway_attaches_conversation_id_not_session_record_id(self):
        msg = _tg("hey")
        scope = self.attach(msg)
        session = self.bridge.session_for_message(_tg("probe"), is_owner=True)
        self.assertEqual(scope, "telegram-bot:123")
        self.assertEqual(scope, msg.chat.key)
        self.assertEqual(scope, session.conversation_id)
        self.assertNotEqual(scope, session.id)
        self.assertFalse(scope.startswith("sess_"))

    def test_double_reply_guard_fires_under_production_scope(self):
        msg = _tg("first message")
        scope = self.attach(msg)
        t0 = self.brain._persist_inbound(msg)
        self.brain._persist_outbound(msg.chat, ["a reply"], "fake",
                                     session_id=scope)
        # A stale redelivery of an older inbound: the guard (called with
        # the chat key in handle_message) must see the newer reply.
        self.assertTrue(self.brain._already_addressed(msg.chat.key, t0 - 10.0))
        self.assertFalse(self.brain._already_addressed(msg.chat.key, t0 + 100.0))

    def test_profile_mining_sees_the_chats_history(self):
        from nomorals.partner.chat_profile import refresh_profile
        msg = _tg("we should book the lisbon flights soon")
        self.attach(msg)
        self.brain._persist_inbound(msg)
        profile = refresh_profile(self.db, msg.chat.key, force=True)
        self.assertEqual(profile.get("participants"), {"Pelumi": 1})

    def test_history_survives_session_end_and_restart(self):
        msg = _tg("remember the lisbon plan")
        self.attach(msg)
        self.brain._persist_turn(msg, ["noted, love"], "fake")
        # An inbound that was never answered (process died mid-turn).
        pending = _tg("and book the train too")
        self.attach(pending)
        self.brain._persist_inbound(pending)
        # Session ends; brand-new bridge + brain (a process restart).
        session = self.bridge.session_for_message(_tg("probe"), is_owner=True)
        self.bridge.end_session(session.id)
        bridge2 = SessionBridge(db=self.db)
        gw2 = ChatGateway({"telegram-bot": _FakeAdapter()}, db=self.db,
                          session_bridge=bridge2,
                          owner_chats={"telegram-bot:123"})
        gw2._inbound = lambda m: None
        brain2 = PartnerBrain(self.context)
        nxt = _tg("what did i ask you to book?")
        gw2._on_inbound(nxt)
        scope = str(nxt.meta.get("os_session_id") or "")
        history = brain2._history(scope, limit=50)
        self.assertEqual(
            [(h.role, h.content) for h in history],
            [("user", "remember the lisbon plan"),
             ("assistant", "noted, love"),
             ("user", "and book the train too")])

    def test_transient_store_failure_turn_joins_real_history(self):
        from nomorals.os.session import SessionStore
        good = _tg("before the hiccup")
        scope = self.attach(good)
        self.brain._persist_inbound(good)
        orig = SessionStore.list_active

        def _boom(self):
            raise RuntimeError("database is locked")

        SessionStore.list_active = _boom
        try:
            during = _tg("during the hiccup")
            transient_scope = self.attach(during)
        finally:
            SessionStore.list_active = orig
        self.assertEqual(transient_scope, "telegram-bot:123")
        self.brain._persist_inbound(during)
        history = self.brain._history(scope, limit=50)
        self.assertEqual([h.content for h in history],
                         ["before the hiccup", "during the hiccup"])

    def test_cross_chat_isolation_same_platform(self):
        a = _tg("chat A secret", chat_id="111")
        b = _tg("chat B secret", chat_id="222")
        self.attach(a)
        self.attach(b)
        self.brain._persist_turn(a, ["reply A"], "fake")
        self.brain._persist_turn(b, ["reply B"], "fake")
        hist_a = self.brain._history(a.chat.key, limit=50)
        hist_b = self.brain._history(b.chat.key, limit=50)
        self.assertEqual([h.content for h in hist_a],
                         ["chat A secret", "reply A"])
        self.assertEqual([h.content for h in hist_b],
                         ["chat B secret", "reply B"])

    def test_main_bot_and_personal_bot_do_not_share_history(self):
        # Same Telegram user id on both bots: scopes must stay separate.
        main = _tg("on the main bot", chat_id="123", platform="telegram-bot")
        personal = _tg("on the personal bot", chat_id="123",
                       platform="telegram-personal")
        self.brain._persist_turn(main, ["main reply"], "fake")
        self.brain._persist_turn(personal, ["personal reply"], "fake")
        self.assertEqual(
            [h.content for h in self.brain._history(main.chat.key, limit=50)],
            ["on the main bot", "main reply"])
        self.assertEqual(
            [h.content for h in self.brain._history(personal.chat.key, limit=50)],
            ["on the personal bot", "personal reply"])

    def test_forum_topics_have_separate_scopes(self):
        a = _tg("topic A", chat_id="555", thread_id="7")
        b = _tg("topic B", chat_id="555", thread_id="9")
        sa = self.bridge.session_for_message(a, is_owner=False)
        sb = self.bridge.session_for_message(b, is_owner=False)
        self.assertNotEqual(sa.conversation_id, sb.conversation_id)
        a.meta["os_session_id"] = sa.conversation_id
        self.brain._persist_inbound(a)
        self.assertEqual(
            [h.content for h in self.brain._history(sa.conversation_id, limit=50)],
            ["topic A"])
        self.assertEqual(self.brain._history(sb.conversation_id, limit=50), [])


class HistoryAssemblyTests(_ScopeBase):
    def test_history_window_is_a_hard_limit(self):
        scope = "telegram-bot:777"
        for i in range(30):
            m = _tg(f"seed {i}", chat_id="777")
            m.meta["os_session_id"] = scope
            self.brain._persist_inbound(m)
            self.brain._persist_outbound(m.chat, [f"ans {i}"], "fake",
                                         session_id=scope)
        history = self.brain._history(scope, limit=10)
        self.assertEqual(len(history), 10)
        # The NEWEST ten, in conversation order.
        self.assertEqual(history[0].content, "seed 25")
        self.assertEqual(history[-1].content, "ans 29")

    def test_current_turn_is_not_duplicated_in_the_prompt(self):
        chat = ChatRef(platform="local", chat_id="owner", kind=ChatKind.DM,
                       peer="you")
        msg = ChatMessage(chat=chat, incoming=True,
                          text="what did we say about the lisbon trip?",
                          sender="you")
        self.brain.handle_message(msg)
        self.assertGreaterEqual(self.router.calls, 1)
        last = self.router.seen[-1]
        hits = sum(1 for m in last
                   if getattr(m, "role", "") == "user"
                   and "lisbon trip" in (getattr(m, "content", "") or ""))
        self.assertEqual(hits, 1)

    def test_exclusion_is_by_row_not_by_content(self):
        # Two identical texts in a row: excluding the CURRENT turn must
        # not also exclude the earlier identical one.
        scope = "telegram-bot:888"
        first = _tg("ok", chat_id="888")
        first.meta["os_session_id"] = scope
        self.brain._persist_inbound(first)
        self.brain._persist_outbound(first.chat, ["kk"], "fake",
                                     session_id=scope)
        second = _tg("ok", chat_id="888")
        second.meta["os_session_id"] = scope
        self.brain._persist_inbound(second)
        history = self.brain._history(
            scope, limit=50,
            exclude_id=str(second.meta.get("_inbound_row_id") or ""))
        self.assertEqual([h.content for h in history], ["ok", "kk"])

    def test_role_and_name_fidelity_roundtrip(self):
        scope = "telegram-bot:999"
        m = _tg("hello from ada", chat_id="999", sender="Ada Lovelace")
        m.meta["os_session_id"] = scope
        self.brain._persist_inbound(m)
        self.brain._persist_outbound(m.chat, ["hi ada"], "fake",
                                     session_id=scope)
        echo = _tg("owner echo", chat_id="999")
        echo.meta["os_session_id"] = scope
        echo.meta["own_message"] = True
        self.brain._persist_inbound(echo)
        history = self.brain._history(scope, limit=50)
        self.assertEqual(
            [(h.role, h.name, h.content) for h in history],
            [("user", "Ada Lovelace", "hello from ada"),
             ("assistant", self.brain.persona.name, "hi ada"),
             ("user", "__owner__", "owner echo")])


class ConcurrencyTests(_ScopeBase):
    def test_concurrent_turns_two_chats_no_loss_no_bleed(self):
        errors: list[BaseException] = []

        def drive(chat_id: str, tag: str) -> None:
            try:
                for i in range(20):
                    m = _tg(f"{tag} {i}", chat_id=chat_id)
                    m.meta["os_session_id"] = m.chat.key
                    self.brain._persist_inbound(m)
                    self.brain._persist_outbound(
                        m.chat, [f"{tag} ans {i}"], "fake",
                        session_id=m.chat.key)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=drive, args=("111", "aaa")),
                   threading.Thread(target=drive, args=("222", "bbb"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        for chat_id, tag in (("111", "aaa"), ("222", "bbb")):
            rows = self.db.query(
                "SELECT content FROM messages WHERE conversation_id = ?",
                (f"telegram-bot:{chat_id}",))
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(tag in (r["content"] or "") for r in rows))


class HandoffAndMigrationTests(_ScopeBase):
    def test_handoff_moves_history_to_the_new_binding(self):
        msg = _tg("we planned lisbon here")
        self.attach(msg)
        self.brain._persist_turn(msg, ["we did"], "fake")
        session = self.bridge.session_for_message(_tg("probe"), is_owner=True)
        moved = self.bridge.handoff_session(session.id, "whatsapp", "999")
        self.assertIsNotNone(moved)
        history = self.brain._history("whatsapp:999", limit=50)
        self.assertEqual([h.content for h in history],
                         ["we planned lisbon here", "we did"])
        self.assertEqual(self.brain._history("telegram-bot:123", limit=50), [])

    def test_migration_87_registered(self):
        from nomorals.storage.migrations import MIGRATIONS, latest_version
        self.assertEqual(latest_version(), 87)
        mig = [m for m in MIGRATIONS if m.version == 87]
        self.assertEqual(len(mig), 1)
        self.assertEqual(mig[0].name, "conversation_scope_rekey")

    def test_migration_rekeys_legacy_session_scoped_rows(self):
        from nomorals.storage.migrations import _apply_conversation_scope_rekey
        db = self.db
        # Legacy state, as the pre-fix gateway wrote it: an os_sessions
        # row mapping a random record id to the real conversation id,
        # with history + conversation rows under BOTH the record id and
        # a transient fallback id.
        db.execute(
            "INSERT INTO os_sessions (id, principal, frontend, project_id,"
            " conversation_id, artifact_scope, state, created_at, updated_at,"
            " ended_at, name, attach_count, last_activity_at) VALUES"
            " ('sess_legacy1', 'owner', 'telegram-bot', '',"
            " 'telegram-bot:321', '[]', '{}', 1.0, 1.0, NULL, '', 0, 1.0)")
        with db.transaction():
            db.execute(
                "INSERT INTO conversations (id, title, agent, channel,"
                " created_at, updated_at) VALUES ('sess_legacy1', 't',"
                " 'partner', 'telegram-bot', 0, 1.0)")
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content,"
                " name, model, created_at) VALUES ('m1', 'sess_legacy1',"
                " 'user', 'legacy hello', 'Pelumi', '', 1.0)")
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content,"
                " name, model, created_at) VALUES ('m2', 'sess_legacy1',"
                " 'assistant', 'legacy reply', 'Devon', 'fake', 2.0)")
            db.execute(
                "INSERT INTO conversations (id, title, agent, channel,"
                " created_at, updated_at) VALUES"
                " ('transient-telegram-bot:654', 't', 'partner',"
                " 'telegram-bot', 0, 1.0)")
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content,"
                " name, model, created_at) VALUES ('m3',"
                " 'transient-telegram-bot:654', 'user', 'hiccup hello',"
                " 'Pelumi', '', 3.0)")
        _apply_conversation_scope_rekey(db)
        hist = self.brain._history("telegram-bot:321", limit=50)
        self.assertEqual([h.content for h in hist],
                         ["legacy hello", "legacy reply"])
        hist_t = self.brain._history("telegram-bot:654", limit=50)
        self.assertEqual([h.content for h in hist_t], ["hiccup hello"])
        self.assertEqual(self.brain._history("sess_legacy1", limit=50), [])
        # Idempotent: a second run changes nothing.
        _apply_conversation_scope_rekey(db)
        self.assertEqual(
            len(self.brain._history("telegram-bot:321", limit=50)), 2)


if __name__ == "__main__":
    unittest.main()
