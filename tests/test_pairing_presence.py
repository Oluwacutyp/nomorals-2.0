"""Presence pack (§15): always-online, auto status viewing, and
Telegram view-once parity.

The three owner-named capabilities the personal surfaces were
missing: holding a paired account ONLINE (bridge presence on
WhatsApp, Telethon account status on Telegram), VIEWING contacts'
statuses/stories as the owner (status read receipts / ReadStories
— viewing only, independent of saving), and Telegram view-once
capture reaching parity with WhatsApp — including the honest
``view_once_missed`` journal when the ephemeral media could not
be downloaded before it expired (previously a TTL message whose
download failed was dropped at the adapter with no record at all).

Everything is faked at the adapter/bridge seam: toggle off means
ZERO presence/view/capture calls, an owner stop defeats the
keepalive absolutely, and episodes journal once per transition.
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.pairing import (
    PERSONAL_TOGGLES,
    AccountRecord,
    PairingRegistry,
)
from nomorals.social.personal import PersonalSurfaces, PersonalVault


# ── organ fakes ─────────────────────────────────────────────────────

class FakePresenceSetter:
    def __init__(self, result=True):
        self.calls: list[tuple[str, str, bool]] = []
        self.result = result

    def __call__(self, platform, rec, online):
        self.calls.append((platform, rec.id, bool(online)))
        return self.result


class FakeStatusViewer:
    def __init__(self, result=True):
        self.calls: list[dict] = []
        self.result = result

    def __call__(self, rec, item):
        self.calls.append(dict(item))
        return self.result


class FakeStoryViewer:
    def __init__(self, result=True):
        self.calls: list[tuple[str, int]] = []
        self.result = result

    def __call__(self, rec, poster_id, max_story_id):
        self.calls.append((poster_id, int(max_story_id)))
        return self.result


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


# ── 1. toggles ──────────────────────────────────────────────────────

def test_new_toggles_exist_default_off_and_settable(tmp_path):
    assert "always_online" in PERSONAL_TOGGLES
    assert "status_viewing" in PERSONAL_TOGGLES
    personal, registry, vault, rec = make_personal(tmp_path)
    assert rec.toggle("always_online") is False
    assert rec.toggle("status_viewing") is False
    registry.set_toggles(rec.id, always_online=True,
                         status_viewing=True)
    got = registry.get(rec.id)
    assert got.toggle("always_online") is True
    assert got.toggle("status_viewing") is True
    # the originals are untouched and still default off
    assert got.toggle("status_saving") is False
    with pytest.raises(ValueError):
        registry.set_toggles(rec.id, always_onlin=True)
    with pytest.raises(ValueError):
        personal.set_toggle(rec.id, "status_viewin", True)


# ── 2. always-online (organ) ────────────────────────────────────────

def test_presence_asserts_keepalive_journals_once(tmp_path):
    setter = FakePresenceSetter()
    personal, registry, vault, rec = make_personal(
        tmp_path, presence_setter=setter)
    registry.set_toggles(rec.id, always_online=True)
    states = personal.cycle_presence()
    assert setter.calls == [("whatsapp", rec.id, True)]
    assert states[rec.id] == "online"
    assert personal.presence_state(rec.id) == "online"
    # second beat: keepalive re-asserts, but the EPISODE journals once
    personal.cycle_presence()
    assert setter.calls == [("whatsapp", rec.id, True)] * 2
    rows = [r for r in vault.journal() if r["kind"] == "presence"]
    assert len(rows) == 1
    assert rows[0]["note"] == "online asserted"
    assert rows[0]["account"] == rec.id


def test_presence_toggle_off_releases_once(tmp_path):
    setter = FakePresenceSetter()
    personal, registry, vault, rec = make_personal(
        tmp_path, presence_setter=setter)
    registry.set_toggles(rec.id, always_online=True)
    personal.cycle_presence()
    registry.set_toggles(rec.id, always_online=False)
    states = personal.cycle_presence()
    assert setter.calls[-1] == ("whatsapp", rec.id, False)
    assert states[rec.id] == "off — toggle off"
    rows = [r for r in vault.journal() if r["kind"] == "presence"]
    assert [r["note"] for r in rows] == [
        "online asserted", "online released — toggle off"]
    # third beat: nothing left to release
    personal.cycle_presence()
    assert len(setter.calls) == 2


def test_presence_owner_stop_defeats_absolutely(tmp_path):
    setter = FakePresenceSetter()
    personal, registry, vault, rec = make_personal(
        tmp_path, presence_setter=setter)
    registry.set_toggles(rec.id, always_online=True)
    personal.cycle_presence()
    assert setter.calls == [("whatsapp", rec.id, True)]
    registry.set_owner_stopped("whatsapp", True)
    states = personal.cycle_presence()
    # released once, and never asserted again while the stop stands
    assert setter.calls[-1] == ("whatsapp", rec.id, False)
    assert states[rec.id] == "off — stopped by the owner"
    personal.cycle_presence()
    personal.cycle_presence()
    assert [c for c in setter.calls if c[2]] == [
        ("whatsapp", rec.id, True)]  # the one pre-stop assert only
    rows = [r for r in vault.journal() if r["kind"] == "presence"]
    assert rows[-1]["note"] == "online released — stopped by the owner"


def test_presence_owner_stop_from_cold_asserts_nothing(tmp_path):
    setter = FakePresenceSetter()
    personal, registry, vault, rec = make_personal(
        tmp_path, presence_setter=setter)
    registry.set_toggles(rec.id, always_online=True)
    registry.set_owner_stopped("whatsapp", True)
    personal.cycle_presence()
    assert setter.calls == []
    assert personal.presence_state(rec.id) == \
        "off — stopped by the owner"


def test_presence_not_linked_asserts_nothing(tmp_path):
    setter = FakePresenceSetter()
    personal, registry, vault, rec = make_personal(
        tmp_path, presence_setter=setter)
    registry.set_toggles(rec.id, always_online=True)
    registry.set_status(rec.id, "down")
    states = personal.cycle_presence()
    assert setter.calls == []
    assert states[rec.id] == "off — account down"


def test_presence_refused_assert_is_not_reported_online(tmp_path):
    setter = FakePresenceSetter(result=False)
    personal, registry, vault, rec = make_personal(
        tmp_path, presence_setter=setter)
    registry.set_toggles(rec.id, always_online=True)
    states = personal.cycle_presence()
    assert setter.calls == [("whatsapp", rec.id, True)]
    assert states[rec.id].startswith("wanted, but the surface")
    assert [r for r in vault.journal() if r["kind"] == "presence"] == []


def test_presence_no_verb_wired_reports_absence(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, always_online=True)
    states = personal.cycle_presence()
    assert "no presence verb is wired" in states[rec.id]
    assert [r for r in vault.journal() if r["kind"] == "presence"] == []


# ── 3. WhatsApp status viewing (organ) ──────────────────────────────

def test_status_viewing_marks_once_independent_of_saving(tmp_path):
    viewer = FakeStatusViewer()
    personal, registry, vault, rec = make_personal(
        tmp_path, status_viewer=viewer)
    registry.set_toggles(rec.id, status_viewing=True)
    out = personal.observe_status_post(
        "whatsapp", poster_id="111@s.whatsapp.net", poster_name="Ada",
        text="hello", source_id="S1", ts=5.0)
    assert out == []                      # saving off: nothing kept
    assert vault.saved_entries() == []
    assert len(viewer.calls) == 1
    assert viewer.calls[0]["poster_id"] == "111@s.whatsapp.net"
    assert viewer.calls[0]["source_id"] == "S1"
    # the same post never views twice
    personal.observe_status_post(
        "whatsapp", poster_id="111@s.whatsapp.net", text="hello",
        source_id="S1", ts=6.0)
    assert len(viewer.calls) == 1


def test_status_viewing_off_no_calls_saving_unaffected(tmp_path):
    viewer = FakeStatusViewer()
    personal, registry, vault, rec = make_personal(
        tmp_path, status_viewer=viewer)
    registry.set_toggles(rec.id, status_saving=True)
    src = _media_file(tmp_path)
    out = personal.observe_status_post(
        "whatsapp", poster_id="111", media=[{"path": src,
                                             "kind": "image"}],
        source_id="S2", ts=5.0)
    assert len(out) == 1 and out[0]["kind"] == "status"
    assert viewer.calls == []


def test_status_viewing_and_saving_together(tmp_path):
    viewer = FakeStatusViewer()
    personal, registry, vault, rec = make_personal(
        tmp_path, status_viewer=viewer)
    registry.set_toggles(rec.id, status_saving=True,
                         status_viewing=True)
    src = _media_file(tmp_path)
    out = personal.observe_status_post(
        "whatsapp", poster_id="111", media=[{"path": src,
                                             "kind": "image"}],
        source_id="S3", ts=5.0)
    assert len(out) == 1
    assert len(viewer.calls) == 1


def test_status_viewing_refusal_stays_unviewed_not_faked(tmp_path):
    viewer = FakeStatusViewer(result=False)
    personal, registry, vault, rec = make_personal(
        tmp_path, status_viewer=viewer)
    registry.set_toggles(rec.id, status_viewing=True)
    personal.observe_status_post(
        "whatsapp", poster_id="111", source_id="S4", ts=5.0)
    personal.observe_status_post(
        "whatsapp", poster_id="111", source_id="S4", ts=6.0)
    # a refused view is NOT marked seen — the next sighting retries
    assert len(viewer.calls) == 2


# ── 4. Telegram story viewing (organ) ───────────────────────────────

def _story(poster, sid, path=""):
    return {"poster_id": poster, "poster_name": "Ada",
            "story_id": sid, "path": path, "kind": "image",
            "ts": 10.0}


def test_story_viewing_per_peer_newest_id_once(tmp_path):
    viewer = FakeStoryViewer()
    feed = [_story("777", "3"), _story("777", "5"),
            _story("888", "2")]
    personal, registry, vault, rec = make_personal(
        tmp_path, platform="telegram",
        story_fetcher=lambda r: list(feed), story_viewer=viewer)
    registry.set_toggles(rec.id, status_viewing=True)
    personal.cycle_stories("telegram")
    # one ReadStories per peer, at its NEWEST story id
    assert sorted(viewer.calls) == [("777", 5), ("888", 2)]
    # a second sweep over the same feed views nothing again
    personal.cycle_stories("telegram")
    assert len(viewer.calls) == 2


def test_story_viewing_off_saving_still_works(tmp_path):
    viewer = FakeStoryViewer()
    src = _media_file(tmp_path, "story.jpg")
    personal, registry, vault, rec = make_personal(
        tmp_path, platform="telegram",
        story_fetcher=lambda r: [_story("777", "3", path=src)],
        story_viewer=viewer)
    registry.set_toggles(rec.id, status_saving=True)
    out = personal.cycle_stories("telegram")
    assert len(out) == 1 and out[0]["kind"] == "story"
    assert viewer.calls == []


def test_story_cycle_no_toggles_never_fetches(tmp_path):
    fetches = []
    personal, registry, vault, rec = make_personal(
        tmp_path, platform="telegram",
        story_fetcher=lambda r: fetches.append(1) or [],
        story_viewer=FakeStoryViewer())
    assert personal.cycle_stories("telegram") == []
    assert fetches == []


# ── 5. view-once missed media (organ) ───────────────────────────────

def test_view_once_media_missing_is_journaled(tmp_path):
    personal, registry, vault, rec = make_personal(
        tmp_path, platform="telegram")
    registry.set_toggles(rec.id, view_once=True)
    out = personal.observe_message(
        "telegram", chat_key="k", counterpart_id="999",
        message_id="m1", media=[], ttl_seconds=30,
        media_missing=True, ts=5.0)
    assert out == []
    rows = vault.journal(rec.id)
    # the miss row, then (2026-11-08 truth fix) the arrival
    # trace every view-once arrival leaves
    assert [r["kind"] for r in rows] == [
        "view_once_missed", "view_once_trace"]
    assert rows[0]["platform"] == "telegram"
    assert "could not be downloaded" in rows[0]["note"]


def test_view_once_media_missing_toggle_off_traces_arrival(tmp_path):
    """Was "journals nothing" — the exact silent hole the
    2026-11-08 truth fix closes: with capture OFF an arrival
    used to leave no row of any kind, indistinguishable from
    nothing ever arriving.  What stays forbidden is a SAVE or
    a MISS row (she did not try); what the journal now holds
    is the arrival trace naming the toggle state it met."""
    personal, registry, vault, rec = make_personal(
        tmp_path, platform="telegram")
    personal.observe_message(
        "telegram", chat_key="k", message_id="m1", media=[],
        ttl_seconds=30, media_missing=True, ts=5.0)
    rows = vault.journal(rec.id)
    assert [r["kind"] for r in rows] == ["view_once_trace"]
    assert rows[0]["view_once_toggle"] is False
    assert rows[0]["outcome"] == "skipped_toggle_off"


# ── 6. WhatsApp adapter seam ────────────────────────────────────────

class FakeCmdAdapter:
    pass


def _wa_adapter_capturing():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter.connected.set()
    sent = []

    def fake_send_cmd(cmd, *, timeout=25.0):
        sent.append(dict(cmd))
        return {"ok": True}

    adapter._send_cmd = fake_send_cmd
    return adapter, sent


def test_whatsapp_adapter_presence_and_status_view_verbs():
    adapter, sent = _wa_adapter_capturing()
    assert adapter.set_presence(True)["ok"] is True
    assert sent[-1]["cmd"] == "presence"
    assert sent[-1]["state"] == "available"
    assert adapter.set_presence(False)["ok"] is True
    assert sent[-1]["state"] == "unavailable"
    res = adapter.view_status("111@s.whatsapp.net", "MSGID42", ts=7.5)
    assert res["ok"] is True
    assert sent[-1]["cmd"] == "status_view"
    # The target rides "message_id" (truth build): "id" is the
    # request correlation id _send_cmd writes on the real wire —
    # this pin once asserted the target in "id", the shape that
    # never survived the transport.
    assert sent[-1]["message_id"] == "MSGID42"
    assert sent[-1]["from"] == "111@s.whatsapp.net"
    # No ts on the wire (status fix, 2026-11-07): this pin used
    # to assert ts == 7500 — the post's timestamp plumbed into
    # MarkRead, divergence D5 of the status mining.  The
    # bridges now stamp the receipt at READ time, like the
    # phone; the adapter's ts argument is accepted and
    # deliberately not sent.
    assert "ts" not in sent[-1]
    # missing parts refuse BEFORE any bridge traffic
    before = len(sent)
    assert adapter.view_status("", "X")["ok"] is False
    assert adapter.view_status("111", "")["ok"] is False
    assert len(sent) == before


def test_whatsapp_adapter_disconnected_presence_is_honest():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    res = adapter.set_presence(True)
    assert res["ok"] is False
    assert "not connected" in res["error"]


def test_whatsapp_adapter_view_once_missing_stamp():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    got = []
    base = {"type": "message", "id": "m1",
            "chat": {"id": "111@s.whatsapp.net", "kind": "dm",
                     "title": "Ada"},
            "from": {"id": "111@s.whatsapp.net", "name": "Ada"},
            "text": "", "view_once": True, "ts": 1700000000000}
    adapter._dispatch(dict(base, media=[]), handler=got.append)
    assert got[0].meta["view_once"] is True
    assert got[0].meta["view_once_media_missing"] is True
    got.clear()
    adapter._dispatch(
        dict(base, media=[{"path": "/tmp/y.jpg", "mime": "image/jpeg",
                           "kind": "image"}]),
        handler=got.append)
    assert got[0].meta["view_once"] is True
    assert "view_once_media_missing" not in got[0].meta


# ── 7. Telegram adapter seam (fake Telethon modules) ────────────────

class _RecordingRequest:
    def __init__(self, **kw):
        self.kw = kw


class FakeTGClientForAdapter:
    def __init__(self):
        self.requests = []
        self.entities = []

    async def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace()

    async def get_input_entity(self, ref):
        self.entities.append(ref)
        return ("input-peer", ref)


@pytest.fixture()
def fake_telethon(monkeypatch):
    account_mod = SimpleNamespace(
        UpdateStatusRequest=lambda offline: _RecordingRequest(
            offline=offline))
    stories_mod = SimpleNamespace(
        ReadStoriesRequest=lambda peer, max_id: _RecordingRequest(
            peer=peer, max_id=max_id))
    telethon_mod = SimpleNamespace()
    tl_mod = SimpleNamespace()
    functions_mod = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "telethon", telethon_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl", tl_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.functions",
                        functions_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.functions.account",
                        account_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.functions.stories",
                        stories_mod)
    return SimpleNamespace(account=account_mod, stories=stories_mod)


def _run_sync(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _tg_adapter(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    adapter._run_on_loop = lambda coro, timeout=30.0: _run_sync(coro)
    return adapter


def test_telegram_adapter_set_presence_uses_update_status(
        tmp_path, fake_telethon):
    adapter = _tg_adapter(tmp_path)
    assert adapter.set_presence(True) is False  # not connected
    client = FakeTGClientForAdapter()
    adapter._client = client
    adapter._loop = object()
    assert adapter.set_presence(True) is True
    assert client.requests[-1].kw == {"offline": False}
    assert adapter.set_presence(False) is True
    assert client.requests[-1].kw == {"offline": True}


def test_telegram_adapter_view_stories_uses_read_stories(
        tmp_path, fake_telethon):
    adapter = _tg_adapter(tmp_path)
    assert adapter.view_stories("777", 5) is False  # not connected
    client = FakeTGClientForAdapter()
    adapter._client = client
    adapter._loop = object()
    assert adapter.view_stories("777", 5) is True
    assert client.entities == [777]
    req = client.requests[-1]
    assert req.kw["peer"] == ("input-peer", 777)
    assert req.kw["max_id"] == 5


# ── 8. Telegram inbound: TTL media that never lands ─────────────────

class FakeDownloadClient:
    def __init__(self, mode, path=""):
        self.mode = mode
        self.path = path

    async def download_media(self, message, file=None):
        if self.mode == "raise":
            raise RuntimeError("file reference expired")
        return self.path


def _ttl_event(tmp_path, download_mode, src=""):
    user = SimpleNamespace(id=999, first_name="Ada", last_name="",
                           username="ada", bot=False, photo=None)
    media = SimpleNamespace(ttl_seconds=30, document=None)
    message = SimpleNamespace(
        id=42, raw_text="", media=media, voice=None,
        date=datetime.now(timezone.utc), reply_to=None,
        from_id=SimpleNamespace(user_id=999), peer_id=None)
    event = SimpleNamespace(
        message=message, chat_id=999, sender_id=999,
        is_group=False, is_channel=False, chat=user, sender=user,
        input_chat=None, input_sender=None)
    return event


def _deliver_inbound(tmp_path, download_mode, src=""):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    got = []
    event = _ttl_event(tmp_path, download_mode, src)
    client = FakeDownloadClient(download_mode, src)
    asyncio.new_event_loop().run_until_complete(
        adapter._handle_inbound(event, client, got.append,
                                allow_check=False))
    return got


def test_telegram_ttl_download_failure_still_delivers_for_journal(
        tmp_path):
    got = _deliver_inbound(tmp_path, "raise")
    assert len(got) == 1
    msg = got[0]
    assert msg.meta["ttl_seconds"] == 30
    assert msg.meta["view_once_media_missing"] is True
    assert msg.media == []


def test_telegram_ttl_download_success_captures_normally(tmp_path):
    src = _media_file(tmp_path, "vo.jpg")
    got = _deliver_inbound(tmp_path, "ok", src)
    assert len(got) == 1
    msg = got[0]
    assert msg.meta["ttl_seconds"] == 30
    assert "view_once_media_missing" not in msg.meta
    assert len(msg.media) == 1


def test_telegram_plain_textless_medialess_still_dropped(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    got = []
    user = SimpleNamespace(id=999, first_name="Ada", last_name="",
                           username="ada", bot=False, photo=None)
    message = SimpleNamespace(
        id=43, raw_text="", media=None, voice=None,
        date=datetime.now(timezone.utc), reply_to=None,
        from_id=SimpleNamespace(user_id=999), peer_id=None)
    event = SimpleNamespace(
        message=message, chat_id=999, sender_id=999,
        is_group=False, is_channel=False, chat=user, sender=user,
        input_chat=None, input_sender=None)
    asyncio.new_event_loop().run_until_complete(
        adapter._handle_inbound(event, FakeDownloadClient("ok"),
                                got.append, allow_check=False))
    assert got == []


# ── 9. runtime wiring (beat + personal turn, walive2 harness shape) ─

def _beat_runtime(tmp_path):
    """PartnerRuntime on the walive2 bare harness: real provision
    mixin + real gateway, only the supervision sweep stubbed (its
    probes have their own suites; here it must not flip states)."""
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


class FakeSurfaceAdapter:
    """A gateway adapter with the presence verbs, recording calls."""

    def __init__(self):
        self.presence_calls: list[bool] = []

    def set_presence(self, online):
        self.presence_calls.append(bool(online))
        return {"ok": True}


def test_runtime_beat_drives_presence_and_stop_releases(tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        status="linked", created_at=1.0))
    registry.set_toggles(rec.id, always_online=True)
    fake = FakeSurfaceAdapter()
    rt.gateway.adapters["telegram"] = fake
    svc["last_cycle"] = 0.0
    rt._housekeeping_beat()
    assert fake.presence_calls == [True]
    assert svc["personal"].presence_state(rec.id) == "online"
    # the owner's stop: the next beat releases and never re-asserts
    registry.set_owner_stopped("telegram", True)
    svc["last_cycle"] = 0.0
    rt._housekeeping_beat()
    assert fake.presence_calls == [True, False]
    svc["last_cycle"] = 0.0
    rt._housekeeping_beat()
    assert fake.presence_calls == [True, False]


def test_runtime_beat_toggle_off_makes_no_presence_calls(tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        status="linked", created_at=1.0))
    fake = FakeSurfaceAdapter()
    rt.gateway.adapters["telegram"] = fake
    svc["last_cycle"] = 0.0
    rt._housekeeping_beat()
    assert fake.presence_calls == []


def test_runtime_turn_captures_telegram_view_once(tmp_path):
    from nomorals.social.chat.base import (
        ChatKind,
        ChatMessage,
        ChatRef,
        MediaRef,
    )

    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        status="linked", created_at=1.0))
    registry.set_toggles(rec.id, view_once=True)
    src = _media_file(tmp_path, "vo.jpg")
    msg = ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999",
                     kind=ChatKind.DM, title="Ada", peer="999"),
        incoming=True, text="", sender="Ada", sender_id="999",
        message_id="42",
        media=[MediaRef(path=src, mime="image/jpeg", kind="image")],
        ts=time.time(), meta={"ttl_seconds": 30})
    rt._personal_surface_turn(msg)
    rows = svc["personal"].vault.saved_entries(rec.id)
    assert len(rows) == 1
    assert rows[0]["kind"] == "view_once"
    assert rows[0]["platform"] == "telegram"
    assert Path(rows[0]["path"]).exists()


def test_runtime_turn_journals_telegram_view_once_miss(tmp_path):
    from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

    rt, svc = _beat_runtime(tmp_path)
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        status="linked", created_at=1.0))
    registry.set_toggles(rec.id, view_once=True)
    msg = ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999",
                     kind=ChatKind.DM, title="Ada", peer="999"),
        incoming=True, text="", sender="Ada", sender_id="999",
        message_id="43", media=[], ts=time.time(),
        meta={"ttl_seconds": 30, "view_once_media_missing": True})
    rt._personal_surface_turn(msg)
    rows = svc["personal"].vault.journal(rec.id)
    assert [r["kind"] for r in rows] == [
        "view_once_missed", "view_once_trace"]
