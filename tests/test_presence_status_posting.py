"""Presence pack completion (2026-11-06): status POSTING, the
status-media miss journal, bridge presence re-assertion, and
the WhatsApp XMD menu's live routing.

The census behind this file: the presence pack (37c3bd5)
shipped viewing / saving / reacting and always-online, but the
SEND half of the status family was never built on any surface
— no bridge verb, no adapter method, no command.  And a status
whose media failed to download vanished silently in the saving
path (the adapter dropped the bridge's pathless media row
without a trace — the same silent-drop class the photo fix
killed for ordinary messages).

Everything is faked at the adapter/bridge seam, like the
presence suites: the verbs' wire shapes are pinned here, the
bridges' own implementations are pinned by source shape (the
Go/Node code is the deliverable for bridge-layer behavior),
and the command is driven through the real chatops handler on
the walive bare-runtime harness.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.pairing import (
    AccountRecord,
    PairingRegistry,
)
from nomorals.social.personal import PersonalSurfaces, PersonalVault


def make_personal(tmp_path, platform="whatsapp", **surface_kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault, **surface_kw)
    rec = registry.upsert(AccountRecord(
        id=f"{platform}:2348012345678", platform=platform,
        identity="2348012345678", status="linked", created_at=1.0))
    return personal, registry, vault, rec


def _media_file(tmp_path, name="pic.jpg"):
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8jpeg")
    return str(p)


# ── 1. the WhatsApp adapter's status_send verb ─────────────────────

def _wa_adapter_capturing():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter.connected.set()
    sent = []

    def fake_send_cmd(cmd, *, timeout=25.0):
        sent.append(dict(cmd))
        return {"ok": True, "message_id": "STATUS1"}

    adapter._send_cmd = fake_send_cmd
    return adapter, sent


def test_whatsapp_adapter_send_status_text_shape():
    adapter, sent = _wa_adapter_capturing()
    res = adapter.send_status("building while others flex")
    assert res["ok"] is True
    assert sent[-1]["cmd"] == "status_send"
    assert sent[-1]["text"] == "building while others flex"
    assert sent[-1]["path"] == ""
    assert "audience" not in sent[-1]


def test_whatsapp_adapter_send_status_media_shape():
    adapter, sent = _wa_adapter_capturing()
    res = adapter.send_status(
        "", media_path="/media/p.jpg", caption="day one",
        audience=["111@s.whatsapp.net"])
    assert res["ok"] is True
    assert sent[-1]["cmd"] == "status_send"
    assert sent[-1]["path"] == "/media/p.jpg"
    assert sent[-1]["caption"] == "day one"
    assert sent[-1]["audience"] == ["111@s.whatsapp.net"]


def test_whatsapp_adapter_send_status_empty_refuses_offline():
    adapter, sent = _wa_adapter_capturing()
    res = adapter.send_status("   ")
    assert res["ok"] is False
    assert sent == []  # refused BEFORE any bridge traffic


def test_whatsapp_adapter_send_status_disconnected_is_honest():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    res = adapter.send_status("hello")
    assert res["ok"] is False
    assert "not connected" in res["error"]


# ── 2. the bridges carry the verb (source pins) ────────────────────

_REPO = Path(__file__).resolve().parent.parent


def test_go_bridge_has_status_send_and_presence_reassert():
    messages = (_REPO / "bridge" / "gobridge"
                / "messages.go").read_text(encoding="utf-8")
    main = (_REPO / "bridge" / "gobridge"
            / "main.go").read_text(encoding="utf-8")
    assert '"status_send"' in messages
    assert "cmdStatusSend" in messages
    # whatsmeow resolves the status audience itself
    assert "types.StatusBroadcastJID" in messages
    # presence survives reconnects: the want is remembered by
    # the verb and re-asserted by the connect path
    assert "presenceWant" in messages
    assert "presenceWant" in main
    assert "SendPresence" in main


def test_node_bridge_has_status_send_and_presence_reassert():
    mjs = (_REPO / "bridge"
           / "whatsapp-bridge.mjs").read_text(encoding="utf-8")
    assert "case 'status_send'" in mjs
    # the Baileys shape: an explicit status audience, or nobody
    # is encrypted for — never a silent post to no one
    assert "statusJidList" in mjs
    assert "lastPresenceState" in mjs
    assert "sendPresenceUpdate(lastPresenceState)" in mjs


# ── 3. status media that never lands is journaled, not dropped ─────

def test_adapter_stamps_status_media_missing():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    got = []
    base = {"type": "status_post", "id": "st1",
            "from": {"id": "111@s.whatsapp.net", "name": "Ada"},
            "text": "", "ts": int(time.time() * 1000)}
    # the bridge's download-failure shape: a media row with an
    # empty path (both engines emit exactly this)
    adapter._dispatch(
        dict(base, media=[{"path": "", "mime": "image/jpeg",
                           "kind": "image"}]),
        handler=got.append)
    assert got[0].meta["service"] == "status_post"
    assert got[0].meta["status_media_missing"] is True
    got.clear()
    src = "/media/ok.jpg"
    adapter._dispatch(
        dict(base, media=[{"path": src, "mime": "image/jpeg",
                           "kind": "image"}]),
        handler=got.append)
    assert "status_media_missing" not in got[0].meta
    assert got[0].media[0].path == src


def test_organ_journals_status_missed_once(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, status_saving=True)
    personal.observe_status_post(
        "whatsapp", poster_id="111@s.whatsapp.net",
        poster_name="Ada", source_id="st1", media=[],
        media_missing=True, ts=10.0)
    personal.observe_status_post(
        "whatsapp", poster_id="111@s.whatsapp.net",
        poster_name="Ada", source_id="st1", media=[],
        media_missing=True, ts=11.0)
    rows = [r for r in vault.journal(rec.id)
            if r["kind"] == "status_missed"]
    assert len(rows) == 1
    assert rows[0]["counterpart_id"] == "111@s.whatsapp.net"
    assert rows[0]["source_id"] == "st1"


def test_organ_status_miss_silent_when_saving_off(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    personal.observe_status_post(
        "whatsapp", poster_id="111@s.whatsapp.net",
        source_id="st1", media=[], media_missing=True, ts=10.0)
    rows = vault.journal(rec.id)
    # The MISS stays silent when saving is off (the pin's
    # original contract: no status_missed row).  The rows
    # present are the status_view_trace (status fix,
    # 2026-11-07: a trace row for EVERY observed post) and,
    # before it, the status_arrival (truth fix, 2026-11-08:
    # every observed post enters the arrival ledger before
    # any toggle is read, so the funnel reconciles) — what
    # the journal must never contain here is a miss row.
    assert [r["kind"] for r in rows] == [
        "status_arrival", "status_view_trace"]


# ── 4. the runtime turn carries the miss into the journal ──────────

def _beat_runtime(tmp_path):
    """PartnerRuntime on the walive2 bare harness (mirrors
    test_pairing_presence): real provision mixin + real
    gateway, supervision sweep stubbed."""
    from types import SimpleNamespace as NS

    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter

    chat = NS(whatsapp_enabled=True, whatsapp_host="127.0.0.1",
              whatsapp_port=8787, whatsapp_outbox_dir="",
              telegram_api_id="", telegram_api_hash="")
    settings = NS(chat=chat, workspace_dir=str(tmp_path),
                  resolve=lambda rel: Path(tmp_path) / rel)
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.settings = settings
    rt.context = NS(settings=settings)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    rt._bump = lambda *a, **k: None
    rt._harvest_play_downloads = lambda *a, **k: None
    rt._drive_webshot_watches = None
    svc = rt._pairing_personal()
    svc["supervision"] = NS(sweep=lambda: None)
    return rt, svc


def test_runtime_turn_journals_status_miss(tmp_path):
    from nomorals.social.chat.base import (
        ChatKind,
        ChatMessage,
        ChatRef,
    )

    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked",
        created_at=1.0))
    registry.set_toggles(rec.id, status_saving=True)
    msg = ChatMessage(
        chat=ChatRef(platform="whatsapp",
                     chat_id="status@broadcast",
                     kind=ChatKind.DM, title="status",
                     peer="111@s.whatsapp.net"),
        incoming=True, text="", sender="Ada",
        sender_id="111@s.whatsapp.net", message_id="st1",
        media=[], ts=time.time(),
        meta={"service": "status_post",
              "status_media_missing": True})
    assert rt._personal_surface_turn(msg) is True
    rows = [r for r in svc["personal"].vault.journal(rec.id)
            if r["kind"] == "status_missed"]
    assert len(rows) == 1


def test_runtime_turn_saves_status_media(tmp_path):
    from nomorals.social.chat.base import (
        ChatKind,
        ChatMessage,
        ChatRef,
        MediaRef,
    )

    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked",
        created_at=1.0))
    registry.set_toggles(rec.id, status_saving=True)
    src = _media_file(tmp_path, "status.jpg")
    msg = ChatMessage(
        chat=ChatRef(platform="whatsapp",
                     chat_id="status@broadcast",
                     kind=ChatKind.DM, title="status",
                     peer="111@s.whatsapp.net"),
        incoming=True, text="", sender="Ada",
        sender_id="111@s.whatsapp.net", message_id="st2",
        media=[MediaRef(path=src, mime="image/jpeg",
                        kind="image")],
        ts=time.time(), meta={"service": "status_post"})
    assert rt._personal_surface_turn(msg) is True
    rows = svc["personal"].vault.saved_entries(rec.id)
    assert len(rows) == 1
    assert rows[0]["kind"] == "status"
    assert Path(rows[0]["path"]).exists()


# ── 5. /poststatus: the command path ───────────────────────────────

class FakeWAStatusAdapter:
    def __init__(self, verdict=None):
        self.calls: list[dict] = []
        self.verdict = verdict or {"ok": True,
                                    "message_id": "STATUS1"}

    def send_status(self, text, *, media_path="", caption="",
                    audience=None):
        self.calls.append({"text": text, "media_path": media_path,
                           "caption": caption})
        return self.verdict


class FakeTGStoryAdapter:
    def __init__(self, result=True):
        self.calls: list[tuple] = []
        self.result = result

    def send_story(self, media_path, caption=""):
        self.calls.append((media_path, caption))
        return self.result


def _command_runtime(tmp_path, platform, adapter):
    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id=f"{platform}:2348012345678", platform=platform,
        identity="2348012345678", status="linked",
        created_at=1.0))
    rt.gateway.adapters[platform] = adapter
    return rt, svc, rec


def _command_message(platform, media=None, meta=None):
    return SimpleNamespace(
        chat=SimpleNamespace(platform=platform),
        media=media or [], meta=meta or {})


def test_poststatus_whatsapp_text_posts_and_journals(tmp_path):
    fake = FakeWAStatusAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "poststatus", "building while others flex",
        _command_message("whatsapp"))
    assert reply.startswith("posted to your WhatsApp status")
    assert fake.calls == [{
        "text": "building while others flex", "media_path": "",
        "caption": ""}]
    rows = [r for r in svc["personal"].vault.journal(rec.id)
            if r["kind"] == "status_posted"]
    assert len(rows) == 1
    assert rows[0]["source_id"] == "STATUS1"


def test_poststatus_whatsapp_media_from_caption_message(tmp_path):
    fake = FakeWAStatusAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    src = _media_file(tmp_path, "shot.jpg")
    msg = _command_message(
        "whatsapp",
        media=[SimpleNamespace(path=src, kind="image")])
    reply = rt._control_chatops("poststatus", "day one", msg)
    assert reply.startswith("posted to your WhatsApp status")
    assert fake.calls[0]["media_path"] == src
    assert fake.calls[0]["caption"] == "day one"


def test_poststatus_whatsapp_quoted_media(tmp_path):
    fake = FakeWAStatusAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    src = _media_file(tmp_path, "quoted.jpg")
    msg = _command_message(
        "whatsapp",
        meta={"replied_media_local": {
            "path": src, "kind": "image", "mime": "image/jpeg"}})
    reply = rt._control_chatops("poststatus", "", msg)
    assert reply.startswith("posted to your WhatsApp status")
    assert fake.calls[0]["media_path"] == src


def test_poststatus_usage_when_nothing_to_post(tmp_path):
    fake = FakeWAStatusAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "poststatus", "", _command_message("whatsapp"))
    assert reply.startswith("usage: /poststatus")
    assert fake.calls == []


def test_poststatus_bridge_refusal_is_honest(tmp_path):
    fake = FakeWAStatusAdapter(
        verdict={"ok": False, "error": "bridge down"})
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "poststatus", "hello", _command_message("whatsapp"))
    assert "couldn't do it" in reply and "bridge down" in reply
    rows = [r for r in svc["personal"].vault.journal(rec.id)
            if r["kind"] == "status_posted"]
    assert rows == []


def test_poststatus_owner_stop_halts(tmp_path):
    fake = FakeWAStatusAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    svc["registry"].set_owner_stopped("whatsapp", True)
    reply = rt._control_chatops(
        "poststatus", "hello", _command_message("whatsapp"))
    assert "standing stop" in reply
    assert fake.calls == []


def test_poststatus_telegram_text_only_gets_platform_truth(
        tmp_path):
    fake = FakeTGStoryAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "telegram", fake)
    reply = rt._control_chatops(
        "poststatus", "hello", _command_message("telegram"))
    assert "media posts" in reply
    assert fake.calls == []


def test_poststatus_telegram_media_posts_story(tmp_path):
    fake = FakeTGStoryAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "telegram", fake)
    src = _media_file(tmp_path, "story.jpg")
    msg = _command_message(
        "telegram",
        media=[SimpleNamespace(path=src, kind="image")])
    reply = rt._control_chatops("poststatus", "gm", msg)
    assert reply == "posted to your Telegram story."
    assert fake.calls == [(src, "gm")]
    rows = [r for r in svc["personal"].vault.journal(rec.id)
            if r["kind"] == "status_posted"]
    assert len(rows) == 1


def test_poststatus_unpaired_account_says_so(tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    reply = rt._control_chatops(
        "poststatus", "hello", _command_message("whatsapp"))
    assert "no paired whatsapp account" in reply


def test_poststatus_registered_owner_only_kind():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
        parse_control,
    )

    assert "poststatus" in CONTROL_COMMANDS
    assert "poststatus" in CHATOPS_COMMANDS
    assert "poststatus" in COMMAND_DETAILS
    assert "poststatus" in LIST_ONELINERS
    assert "poststatus" not in PUBLIC_COMMANDS
    cmd = parse_control("/poststatus building while others flex")
    assert cmd is not None and cmd.kind == "poststatus"


# ── 6. the Telegram adapter's story verb (fake Telethon) ───────────

class _RecordingRequest:
    def __init__(self, **kw):
        self.kw = kw


class FakeTGStoryClient:
    def __init__(self):
        self.requests = []
        self.uploads = []

    async def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace()

    async def get_input_entity(self, ref):
        return ("input-peer", ref)

    async def upload_file(self, path):
        self.uploads.append(path)
        return ("input-file", path)


@pytest.fixture()
def fake_telethon_story(monkeypatch):
    stories_mod = SimpleNamespace(
        SendStoryRequest=lambda **kw: _RecordingRequest(**kw))
    types_mod = SimpleNamespace(
        InputPrivacyValueAllowAll=lambda: ("privacy", "allow-all"),
        InputMediaUploadedPhoto=lambda file: ("photo", file),
        InputMediaUploadedDocument=lambda **kw: (
            "document", kw),
        DocumentAttributeVideo=lambda **kw: ("video-attr", kw))
    monkeypatch.setitem(sys.modules, "telethon", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "telethon.tl",
                        SimpleNamespace())
    monkeypatch.setitem(sys.modules, "telethon.tl.functions",
                        SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "telethon.tl.functions.stories", stories_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.types",
                        types_mod)
    return stories_mod


def _run_sync(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_telegram_adapter_send_story_uses_send_story_request(
        tmp_path, fake_telethon_story):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    adapter._run_on_loop = lambda coro, timeout=30.0: _run_sync(
        coro)
    src = _media_file(tmp_path, "story.jpg")
    assert adapter.send_story(src, "gm") is False  # not connected
    client = FakeTGStoryClient()
    adapter._client = client
    adapter._loop = object()
    assert adapter.send_story(src, "gm") is True
    assert client.uploads == [src]
    req = client.requests[-1]
    assert req.kw["peer"] == ("input-peer", "me")
    assert req.kw["media"] == ("photo", ("input-file", src))
    assert req.kw["privacy_rules"] == [("privacy", "allow-all")]
    assert req.kw["caption"] == "gm"
    # a missing file refuses before any traffic
    client.requests.clear()
    assert adapter.send_story(str(tmp_path / "gone.jpg")) is False
    assert client.requests == []


# ── 7. the XMD menu's live routing (owner complaint) ───────────────

def _wa_menu_adapter():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter.user_jid = "2348012345678@s.whatsapp.net"
    adapter.user_lid = ""
    return adapter


def _own_event(chat_id, kind, text):
    return {
        "type": "own_message", "id": "m-menu-1",
        "chat": {"id": chat_id, "kind": kind, "title": ""},
        "from": {"id": "2348012345678@s.whatsapp.net",
                 "name": "Owner"},
        "text": text, "ts": int(time.time() * 1000),
    }


def test_whatsapp_self_chat_menu_turn_is_owner_turn():
    adapter = _wa_menu_adapter()
    got = []
    adapter._dispatch(
        _own_event("2348012345678@s.whatsapp.net", "dm", "/menu"),
        handler=got.append)
    assert len(got) == 1
    turn = got[0]
    assert turn.incoming is True
    assert turn.meta.get("self_chat") is True
    assert turn.text == "/menu"


def test_whatsapp_group_owner_menu_turn_is_owner_command():
    adapter = _wa_menu_adapter()
    got = []
    adapter._dispatch(
        _own_event("120363000000000000@g.us", "group", "/menu"),
        handler=got.append)
    assert len(got) == 1
    assert got[0].meta.get("owner_command") is True


def test_menu_command_on_whatsapp_surface_serves_xmd(tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    adapter = _wa_menu_adapter()
    got = []
    adapter._dispatch(
        _own_event("2348012345678@s.whatsapp.net", "dm", "/menu"),
        handler=got.append)
    turn = got[0]
    reply = rt.handle_control("/menu", turn.chat.key,
                              message=turn)
    assert isinstance(reply, str)
    # the XMD frame: boxed header + the registry's real catalog
    assert "╭━━〔 🤖" in reply
    assert "ᴛᴏᴛᴀʟ ᴄᴏᴍᴍᴀɴᴅs" in reply
    assert "/poststatus" in reply
    assert len(reply) > 10000
