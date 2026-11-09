"""PARITY #14 — TELEGRAM USERBOT MESSAGE LIFECYCLE
(platform parity census bundle #14, 2026-11-07).

The census graded both rows ABSENT on the Telegram
personal (userbot) surface:

  * Scheduled messages — "send path carries no schedule
    date (grep: only unrelated 'scheduled' comments)".
  * Delete a chat / clear history — "no
    DeleteHistoryRequest in adapter".

The build lands the platform's real verbs, verified
against the Telethon 1.45 sources: send_message /
send_file both carry a native ``schedule`` date, and
``messages.DeleteHistoryRequest(peer, max_id, just_clear,
revoke)`` is the one history verb.  This suite pins every
seam, following the truth-pass standard of the bundles
before it:

  * adapter: a scheduled text rides send_message with an
    aware-UTC schedule datetime; a scheduled video note
    rides send_file with its form flag AND the date; a
    past instant and an empty text refuse BEFORE traffic;
    view-once scheduled media gets send_media's honest
    refusal (MTProto has no view-once outside secret
    chats); DeleteHistory fires with just_clear/revoke
    shaped per verb and per chat kind, and Telegram's
    refusals return verbatim.
  * the base adapter and the BOT adapter report the
    absence truthfully — no fake scheduling anywhere.
  * /sendlater + /deletechat + /clearhistory composed:
    owner command → handle_control → runtime_chatops →
    the real TelegramAdapter → a recording fake Telethon
    client.  Landed acts journal (message_scheduled /
    chat_deleted / history_cleared); the owner's standing
    stop gates all three before any traffic; the bot and
    WhatsApp surfaces answer their platform truth in
    words, never a faked queue or deletion.  The kind is
    sendlater, not schedule — /schedule is the pre-existing
    cron surface and stays exactly that (the makepoll
    vs /poll precedent), pinned below.
  * the when-grammar is a pure parser (relative offsets
    and absolute UTC instants), pinned without a client.

Fake Telethon modules ride the sys.modules seam like the
rest of the family — no network, no real Telethon.
"""
from __future__ import annotations

import asyncio
import calendar
import sys
import time
import types as _types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import (
    ChatKind,
    ChatMessage,
    ChatRef,
    MediaRef,
)

ROOT = Path(__file__).resolve().parents[1]
TG_OWNER = "4242424242"        # the owner's Telegram id (fixture)
ENTITY = SimpleNamespace(id=int(TG_OWNER), first_name="Owner")


# ── fake Telethon seam ─────────────────────────────────────

class _TLReq:
    """One fake TL request: kwargs land as attributes so
    the pins assert exact fields on the exact types."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)

    def __repr__(self):
        fields = ", ".join(
            f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"{type(self).__name__}({fields})"


class DeleteHistoryRequest(_TLReq):
    def __init__(self, peer, max_id, just_clear=None,
                 revoke=None):
        super().__init__(peer=peer, max_id=max_id,
                         just_clear=just_clear, revoke=revoke)


@pytest.fixture()
def fake_telethon(monkeypatch):
    messages_mod = _types.ModuleType(
        "telethon.tl.functions.messages")
    messages_mod.DeleteHistoryRequest = DeleteHistoryRequest
    for name, mod in (
            ("telethon", _types.ModuleType("telethon")),
            ("telethon.tl", _types.ModuleType("telethon.tl")),
            ("telethon.tl.functions",
             _types.ModuleType("telethon.tl.functions")),
            ("telethon.tl.functions.messages", messages_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return messages_mod


# ── fake client: recorded sends + TL asks ──────────────────

class _FakeClient:
    def __init__(self):
        self.requests: list = []
        self.sent_messages: list = []  # (entity, text, kwargs)
        self.sent_files: list = []     # (entity, path, caption, kwargs)
        self.delete_error: Exception | None = None
        self.send_error: Exception | None = None

    async def __call__(self, request):
        self.requests.append(request)
        if type(request).__name__ == "DeleteHistoryRequest":
            if self.delete_error is not None:
                raise self.delete_error
            return SimpleNamespace(pts_count=3, offset=0)
        raise AssertionError(type(request).__name__)

    async def send_message(self, entity, message, **kwargs):
        if self.send_error is not None:
            raise self.send_error
        self.sent_messages.append((entity, message, kwargs))
        return SimpleNamespace(id=77001, views=None)

    async def send_file(self, entity, path, caption=None,
                        **kwargs):
        if self.send_error is not None:
            raise self.send_error
        self.sent_files.append((entity, path, caption, kwargs))
        return SimpleNamespace(id=77002, views=None)


def _adapter(client, tmp_path):
    """A real TelegramAdapter (real __init__) whose client,
    resolver and loop-runner are the fakes — the parity11
    fixture shape."""
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    ad._client = client  # type: ignore[attr-defined]
    ad._loop = object()  # type: ignore[attr-defined]

    async def _resolve(chat):
        return ENTITY

    ad._resolve = _resolve  # type: ignore[method-assign]
    ad._run_on_loop = (  # type: ignore[method-assign]
        lambda coro, timeout=60.0: asyncio.run(coro))
    return ad


def _chat(chat_id=TG_OWNER, kind=ChatKind.DM):
    return ChatRef(platform="telegram", chat_id=chat_id,
                   kind=kind, peer=chat_id)


def _future(hours=2.0):
    return time.time() + hours * 3600


# ── catalog registration ───────────────────────────────────

def test_catalog_registers_the_family():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    for kind in ("sendlater", "deletechat", "clearhistory"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert any(kind in names for _t, names in LIST_GROUPS)
        # owner-only personal-account acts — never public
        assert kind not in PUBLIC_COMMANDS
    # /schedule stays the cron surface — sendlater is a
    # distinct kind, and the cron entries keep their own
    # meaning in every table
    assert "schedule" not in CHATOPS_COMMANDS
    assert "schedule" in CONTROL_COMMANDS
    assert "cron" in COMMAND_DETAILS["schedule"]["what"]
    assert COMMAND_DETAILS["sendlater"]["what"] != (
        COMMAND_DETAILS["schedule"]["what"])


# ── the when-grammar (pure) ────────────────────────────────

NOW = float(calendar.timegm((2026, 11, 9, 12, 0, 0)))


def test_parse_schedule_relative_offsets():
    from nomorals.social.chat.chatops import parse_schedule_args

    out = parse_schedule_args("+2h see you there", NOW)
    assert out["ok"] and out["at"] == NOW + 7200
    assert out["text"] == "see you there"
    out = parse_schedule_args("+45s ping", NOW)
    assert out["at"] == NOW + 45
    out = parse_schedule_args("+30m", NOW)
    assert out["ok"] and out["text"] == ""  # media form: caption optional
    out = parse_schedule_args("+1d tomorrow's post", NOW)
    assert out["at"] == NOW + 86400


def test_parse_schedule_absolute_utc():
    from nomorals.social.chat.chatops import (
        format_schedule_instant,
        parse_schedule_args,
    )

    out = parse_schedule_args(
        "2026-11-10 14:30 happy birthday", NOW)
    assert out["ok"]
    assert out["at"] == float(
        calendar.timegm((2026, 11, 10, 14, 30, 0)))
    assert out["text"] == "happy birthday"
    out = parse_schedule_args("2026-11-10 14:30:05 hi", NOW)
    assert out["at"] == float(
        calendar.timegm((2026, 11, 10, 14, 30, 5)))
    assert format_schedule_instant(out["at"]) == (
        "2026-11-10 14:30 UTC")


def test_parse_schedule_refusals():
    from nomorals.social.chat.chatops import parse_schedule_args

    out = parse_schedule_args("", NOW)
    assert out["ok"] is False and "usage:" in out["error"]
    out = parse_schedule_args("whenever hello", NOW)
    assert out["ok"] is False
    assert "isn't a when I speak" in out["error"]
    out = parse_schedule_args("2026-11-09 11:59 too late", NOW)
    assert out["ok"] is False
    assert "already past" in out["error"]
    out = parse_schedule_args("+0m right now", NOW)
    assert out["ok"] is False
    assert "already past" in out["error"]


# ── adapter: scheduled sends ───────────────────────────────

def test_send_scheduled_rides_the_native_date(fake_telethon,
                                              tmp_path):
    client = _FakeClient()
    ad = _adapter(client, tmp_path)
    at = _future(2)
    res = ad.send_scheduled(_chat(), "see you then", at)
    assert res.ok and res.message_id == "77001"
    (entity, text, kwargs), = client.sent_messages
    assert entity is ENTITY and text == "see you then"
    assert kwargs["schedule"] == datetime.fromtimestamp(
        at, tz=UTC)
    assert kwargs["schedule"].tzinfo is not None


def test_send_scheduled_refusals_before_traffic(fake_telethon,
                                                tmp_path):
    client = _FakeClient()
    ad = _adapter(client, tmp_path)
    res = ad.send_scheduled(_chat(), "too late",
                            time.time() - 60)
    assert not res.ok and "future time" in res.error
    res = ad.send_scheduled(_chat(), "   ", _future())
    assert not res.ok and "needs its words" in res.error
    res = ad.send_scheduled(_chat(), "words", 0)
    assert not res.ok and "future time" in res.error
    assert client.sent_messages == []


def test_send_scheduled_disconnected(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    res = ad.send_scheduled(_chat(), "hello", _future())
    assert not res.ok and res.error == "not connected"
    res = ad.send_scheduled_media(
        _chat(), MediaRef(path="/x.jpg", kind="image"),
        _future())
    assert not res.ok and res.error == "not connected"


def test_send_scheduled_media_forms(fake_telethon, tmp_path):
    client = _FakeClient()
    ad = _adapter(client, tmp_path)
    at = _future(3)
    media = MediaRef(path="/vault/clip.mp4", kind="video_note")
    res = ad.send_scheduled_media(_chat(), media, at,
                                  caption="round one")
    assert res.ok and res.message_id == "77002"
    (entity, path, caption, kwargs), = client.sent_files
    assert entity is ENTITY and path == "/vault/clip.mp4"
    assert caption == "round one"
    assert kwargs["video_note"] is True
    assert kwargs["schedule"] == datetime.fromtimestamp(
        at, tz=UTC)
    # a plain photo carries no form flags, still the date
    res = ad.send_scheduled_media(
        _chat(), MediaRef(path="/vault/pic.jpg", kind="image"),
        at)
    assert res.ok
    _e, _p, cap2, kwargs2 = client.sent_files[-1]
    assert cap2 is None and "video_note" not in kwargs2
    assert kwargs2["schedule"] == datetime.fromtimestamp(
        at, tz=UTC)


def test_send_scheduled_media_refusals(fake_telethon,
                                       tmp_path):
    client = _FakeClient()
    ad = _adapter(client, tmp_path)
    res = ad.send_scheduled_media(
        _chat(), MediaRef(path="/x.jpg", kind="image"),
        time.time() - 5)
    assert not res.ok and "future time" in res.error
    # view-once is not an MTProto form — scheduled or not,
    # the answer is the refusal, never a permanent copy
    res = ad.send_scheduled_media(
        _chat(),
        MediaRef(path="/x.jpg", kind="image", view_once=True),
        _future())
    assert not res.ok
    assert "no view-once media outside secret chats" in (
        res.error)
    assert client.sent_files == []


def test_send_scheduled_server_refusal_surfaces(fake_telethon,
                                                tmp_path):
    client = _FakeClient()
    client.send_error = RuntimeError("SCHEDULE_TOO_MANY")
    ad = _adapter(client, tmp_path)
    res = ad.send_scheduled(_chat(), "one too many", _future())
    assert not res.ok and "SCHEDULE_TOO_MANY" in res.error


# ── adapter: delete chat / clear history ───────────────────

def _history_reqs(client):
    return [r for r in client.requests
            if type(r).__name__ == "DeleteHistoryRequest"]


def test_clear_history_request_shape(fake_telethon, tmp_path):
    client = _FakeClient()
    ad = _adapter(client, tmp_path)
    verdict = ad.clear_history(_chat())
    assert verdict["ok"] is True
    (req,) = _history_reqs(client)
    assert req.peer is ENTITY
    assert req.max_id == 0
    assert req.just_clear is True
    assert req.revoke is False  # my side only, by default
    verdict = ad.clear_history(_chat(), revoke=True)
    assert verdict["ok"] is True
    assert _history_reqs(client)[-1].revoke is True


def test_delete_chat_revoke_follows_the_kind(fake_telethon,
                                             tmp_path):
    client = _FakeClient()
    ad = _adapter(client, tmp_path)
    # a DM deletes for both sides — the client's own
    # delete-chat semantics
    verdict = ad.delete_chat(_chat(kind=ChatKind.DM))
    assert verdict["ok"] is True
    req = _history_reqs(client)[-1]
    assert req.just_clear is False and req.revoke is True
    # a group / channel leaves from the owner's side only
    for kind in (ChatKind.GROUP, ChatKind.CHANNEL):
        verdict = ad.delete_chat(
            _chat(chat_id="-1001234567890", kind=kind))
        assert verdict["ok"] is True
        req = _history_reqs(client)[-1]
        assert req.just_clear is False
        assert req.revoke is False


def test_delete_history_refusal_is_verbatim(fake_telethon,
                                            tmp_path):
    client = _FakeClient()
    client.delete_error = RuntimeError(
        "CHAT_ADMIN_REQUIRED: only the creator can purge")
    ad = _adapter(client, tmp_path)
    verdict = ad.delete_chat(
        _chat(chat_id="-1001234567890", kind=ChatKind.CHANNEL))
    assert verdict["ok"] is False
    assert "CHAT_ADMIN_REQUIRED" in verdict["error"]


def test_delete_history_disconnected(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    for verdict in (ad.clear_history(_chat()),
                    ad.delete_chat(_chat())):
        assert verdict["ok"] is False
        assert verdict["error"] == (
            "telegram userbot not connected")


# ── the absence, stated honestly on the other adapters ─────

def test_base_adapter_reports_the_truth():
    from nomorals.social.chat.base import ChatAdapter

    class _Bare(ChatAdapter):
        name = "bare"

        def run(self, handler):  # pragma: no cover - unused
            raise NotImplementedError

        def send(self, chat, text, **kwargs):  # pragma: no cover
            from nomorals.social.chat.base import SendResult

            return SendResult(ok=False, platform=self.name,
                              error="unused in this test")

    ad = _Bare()
    chat = _chat()
    for res in (ad.send_scheduled(chat, "hi", _future()),
                ad.send_scheduled_media(
                    chat, MediaRef(path="/x.jpg"), _future())):
        assert res.ok is False
        assert "not supported" in res.error


def test_bot_adapter_schedule_is_an_honest_absence():
    from nomorals.social.chat.telegram import TelegramBotAdapter

    ad = TelegramBotAdapter(
        token="123456:TEST", chat_allow="",
        media_dir="data/media/telegram-bot-test")
    chat = ChatRef(platform="telegram-bot",
                   chat_id="-1001234567890",
                   kind=ChatKind.GROUP, peer="-1001234567890")
    res = ad.send_scheduled(chat, "hi", _future())
    assert res.ok is False
    assert "not supported" in res.error
    # and no delete-history verb exists on the bot adapter
    assert not hasattr(ad, "delete_chat")
    assert not hasattr(ad, "clear_history")


# ── chatop branches that answer before any adapter work ────

def _bare_mixin():
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )

    return RuntimeChatOpsMixin()


def _msg(platform, chat_id="-1001234567890", kind=ChatKind.GROUP,
         title="Some Room", text="x"):
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    return ChatMessage(chat=chat, incoming=True, text=text,
                       sender="Owner", sender_id="1")


def test_chatop_schedule_bot_and_whatsapp_truth():
    mixin = _bare_mixin()
    out = mixin._chatops_schedule(
        "+2h hi", _msg("telegram-bot"))
    assert "Bot API has no schedule parameter" in out
    out = mixin._chatops_schedule(
        "+2h hi", _msg("whatsapp", chat_id="123@g.us"))
    assert "WhatsApp has no scheduled send" in out


def test_chatop_deletechat_bot_absence():
    mixin = _bare_mixin()
    mixin._adapter_for = (  # type: ignore[method-assign]
        lambda platform: None)
    out = mixin._chatops_deletechat(
        "deletechat", "", _msg("telegram-bot"))
    assert "personal-account verb" in out
    out = mixin._chatops_deletechat(
        "clearhistory", "", _msg("telegram-bot"))
    assert "personal-account verb" in out


# ── composed chain: the real runtime + userbot adapter ─────

@pytest.fixture()
def rig(fake_telethon, tmp_path):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    client = _FakeClient()
    adapter = _adapter(client, tmp_path)
    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
        "partner.owner_sender_ids": TG_OWNER,
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    gateway = ChatGateway(
        {"local": LocalAdapter(), "telegram": adapter},
        db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    rec = sup.registry.upsert(AccountRecord(
        id=f"telegram:{TG_OWNER}", platform="telegram",
        identity=TG_OWNER, creds_dir="", status="linked",
        created_at=1.0))
    svc = runtime._pairing_personal()
    env = {"runtime": runtime, "client": client,
           "adapter": adapter, "sup": sup, "rec": rec,
           "svc": svc, "personal": svc["personal"],
           "vault": svc["personal"].vault, "tmp": tmp_path}
    yield env
    runtime.stop()
    runtime.gateway.stop()


def _op(rig, text, *, chat_id=TG_OWNER, kind=ChatKind.DM,
        title="", media=None):
    chat = ChatRef(platform="telegram", chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=TG_OWNER,
        message_id=f"op-{time.time()}",
        ts=time.time(), meta={"owner_command": True},
        media=list(media or []))
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_rows(rig, kind):
    return [r for r in rig["vault"].journal(rig["rec"].id)
            if r.get("kind") == kind]


def test_composed_schedule_fires_and_journals(rig):
    before = time.time()
    out = _op(rig, "/sendlater +2h see you at the venue")
    assert out.startswith("scheduled — your message "
                          "lands here ")
    assert out.endswith(" UTC.")
    (entity, text, kwargs), = rig["client"].sent_messages
    assert text == "see you at the venue"
    landed = kwargs["schedule"].timestamp()
    assert before + 7100 < landed < time.time() + 7300
    rows = _journal_rows(rig, "message_scheduled")
    assert rows and rows[-1]["source_id"] == "77001"
    assert rows[-1]["counterpart_id"] == TG_OWNER


def test_composed_schedule_absolute_and_media(rig):
    pic = rig["tmp"] / "venue.jpg"
    pic.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 16)
    stamp = time.strftime(
        "%Y-%m-%d %H:%M", time.gmtime(time.time() + 90000))
    media = [MediaRef(path=str(pic), kind="image")]
    out = _op(rig, f"/sendlater {stamp} the venue",
              media=media)
    assert out.startswith("scheduled — that media "
                          "lands here ")
    (entity, path, caption, kwargs), = rig["client"].sent_files
    assert path == str(pic) and caption == "the venue"
    assert "schedule" in kwargs


def test_composed_schedule_bad_when_and_stop(rig):
    out = _op(rig, "/sendlater whenever hello")
    assert "isn't a when I speak" in out
    assert rig["client"].sent_messages == []
    rig["sup"].registry.set_owner_stopped("telegram", True)
    try:
        out = _op(rig, "/sendlater +2h held back")
        assert "standing stop" in out
        assert rig["client"].sent_messages == []
    finally:
        rig["sup"].registry.set_owner_stopped(
            "telegram", False)


def test_composed_deletechat_fires_and_journals(rig):
    out = _op(rig, "/deletechat")
    assert out.startswith("deleted ")
    (req,) = _history_reqs(rig["client"])
    assert req.just_clear is False
    assert req.revoke is True  # the fixture chat is a DM
    rows = _journal_rows(rig, "chat_deleted")
    assert rows and rows[-1]["counterpart_id"] == TG_OWNER


def test_composed_clearhistory_keeps_the_chat(rig):
    out = _op(rig, "/clearhistory")
    assert "cleared the history of" in out
    assert "the chat itself stays" in out
    (req,) = _history_reqs(rig["client"])
    assert req.just_clear is True and req.revoke is False
    rows = _journal_rows(rig, "history_cleared")
    assert rows and rows[-1]["counterpart_id"] == TG_OWNER


def test_composed_deletechat_honors_owner_stop(rig):
    rig["sup"].registry.set_owner_stopped("telegram", True)
    try:
        out = _op(rig, "/deletechat")
        assert "standing stop" in out
        out = _op(rig, "/clearhistory")
        assert "standing stop" in out
        assert rig["client"].requests == []
    finally:
        rig["sup"].registry.set_owner_stopped(
            "telegram", False)


def test_composed_deletechat_server_refusal_named(rig):
    rig["client"].delete_error = RuntimeError(
        "CHAT_ADMIN_REQUIRED")
    out = _op(rig, "/deletechat")
    assert "couldn't do it" in out
    assert "CHAT_ADMIN_REQUIRED" in out
    assert _journal_rows(rig, "chat_deleted") == []
