"""Channel stories (2026-11-07): POST + VIEW channel stories on
the personal surfaces, both platforms.

The owner's order: "WhatsApp channels and Telegram channels now
support stories" — Devon gains the capability through her
personal surfaces.  What the libraries actually carry (mined,
not assumed):

- whatsmeow (Go bridge pin): SendMessage routes a newsletter
  JID to sendNewsletter (send.go) — the channel post IS the
  WhatsApp channel story format; UploadNewsletter for media;
  NewsletterMarkViewed is the view verb; GetNewsletterMessages
  + GetSubscribedNewsletters are the feed.
- Baileys 6.7.24: sendMessage posts; newsletterFetchMessages
  fetches (raw iq node); NO mark-viewed verb in
  lib/Socket/newsletter.d.ts — the view leg is a named wall on
  that rung, answered ok:false, never faked.
- Telethon: stories.SendStoryRequest takes a peer — a channel
  the account admins is a real story post (boost refusals come
  back in Telegram's own words); the story feed gains a
  per-channel GetPeerStories half so followed channels are
  actually in the cycle's reach.

Pins below: bridge source shapes (the bridges' own code is the
deliverable at that layer), the WhatsApp adapter's wire shapes,
the personal organ's cycle (save/view/dedupe/failure journal),
the /chanstory command through the real chatops handler on the
walive bare-runtime harness, and the Telegram adapter against
fake Telethon modules (the house shape for that adapter).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.pairing import AccountRecord, PairingRegistry
from nomorals.social.personal import PersonalSurfaces, PersonalVault

_REPO = Path(__file__).resolve().parent.parent
NL = "120363000111222333@newsletter"       # a channel JID


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


# ── 1. the bridges carry the verbs (source pins) ─────────────────

def test_go_bridge_carries_channel_story_verbs():
    groups = (_REPO / "bridge" / "gobridge"
              / "groups.go").read_text(encoding="utf-8")
    for verb in ('"channel_send"', '"channel_view"',
                 '"channel_fetch"'):
        assert verb in groups
    # the whatsmeow verbs the Go cases ride
    assert "NewsletterMarkViewed" in groups
    assert "GetNewsletterMessages" in groups
    assert "GetSubscribedNewsletters" in groups
    assert "UploadNewsletter" in groups
    # the post's public identity rides the send reply
    assert '"server_id"' in groups


def test_baileys_bridge_carries_channel_story_verbs():
    admin = (_REPO / "bridge" / "whatsapp-groups-admin.mjs"
             ).read_text(encoding="utf-8")
    assert "channel_send:" in admin
    assert "channel_fetch:" in admin
    assert "channel_view:" in admin
    assert "newsletterFetchMessages" in admin
    # the mark-viewed wall is named in the refusal, not hidden
    assert "no newsletter mark-viewed verb" in admin


# ── 2. the WhatsApp adapter's wire shapes ────────────────────────

def _wa_adapter_capturing(responder=None):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter.connected.set()
    sent = []

    def fake_send_cmd(cmd, *, timeout=25.0):
        sent.append(dict(cmd))
        if responder is not None:
            return responder(cmd)
        return {"ok": True, "message_id": "WAMID1",
                "server_id": "12345"}

    adapter._send_cmd = fake_send_cmd
    return adapter, sent


def test_wa_channel_send_text_shape_and_verdict():
    adapter, sent = _wa_adapter_capturing()
    res = adapter.channel_send(NL, "gold update")
    assert res["ok"] is True
    assert res["server_id"] == "12345"
    cmd = sent[-1]
    assert cmd["cmd"] == "channel_send"
    assert cmd["chat"] == NL
    assert cmd["text"] == "gold update"
    assert cmd["path"] == ""
    assert cmd["message_id"]  # the deterministic stanza id rides


def test_wa_channel_send_media_shape():
    adapter, sent = _wa_adapter_capturing()
    res = adapter.channel_send(NL, "", media_path="/media/p.jpg",
                               caption="day one")
    assert res["ok"] is True
    assert sent[-1]["path"] == "/media/p.jpg"
    assert sent[-1]["caption"] == "day one"


def test_wa_channel_send_refuses_before_traffic():
    adapter, sent = _wa_adapter_capturing()
    # not a channel JID
    assert adapter.channel_send("12345@g.us", "hi")["ok"] is False
    # nothing to post
    assert adapter.channel_send(NL, "   ")["ok"] is False
    assert sent == []


def test_wa_channel_post_id_is_deterministic_per_content():
    adapter, _ = _wa_adapter_capturing()
    a = adapter._channel_post_id(NL, "gold update", "")
    b = adapter._channel_post_id(NL, "gold update", "")
    c = adapter._channel_post_id(NL, "silver update", "")
    d = adapter._channel_post_id(NL, "gold update", "/media/p.jpg")
    assert a == b                      # a double-fire collapses
    assert a != c and a != d           # new content, new id


def test_wa_channel_view_posts_shape():
    adapter, sent = _wa_adapter_capturing(
        responder=lambda cmd: {"ok": True})
    res = adapter.channel_view_posts(NL, ["12345", "12346"])
    assert res["ok"] is True
    cmd = sent[-1]
    assert cmd["cmd"] == "channel_view"
    assert cmd["chat"] == NL
    assert cmd["server_ids"] == ["12345", "12346"]
    # no ids → refused before traffic
    assert adapter.channel_view_posts(NL, [])["ok"] is False
    assert len(sent) == 1


def test_wa_fetch_channel_posts_enumerated_rows_normalized():
    rows = [{"chat": NL, "name": "Gold Signals WA",
             "server_id": "777", "message_id": "wamid.x",
             "text": "gold update", "kind": "", "path": "",
             "media_missing": False, "views": 42,
             "ts": 1794100000000}]

    def responder(cmd):
        if cmd["cmd"] == "channel_fetch" and "chat" not in cmd:
            return {"ok": True, "posts": rows}
        return {"ok": False, "error": "unexpected"}

    adapter, sent = _wa_adapter_capturing(responder)
    out = adapter.fetch_channel_posts()
    assert out == [{
        "poster_id": NL, "poster_name": "Gold Signals WA",
        "story_id": "777", "text": "gold update", "path": "",
        "kind": "", "media_missing": False, "views": 42,
        "ts": 1794100000.0}]


def test_wa_fetch_channel_posts_per_channel_and_fallback():
    def responder(cmd):
        if cmd["cmd"] == "channel_fetch" and "chat" not in cmd:
            # this engine cannot enumerate (the Baileys shape)
            return {"ok": False,
                    "error": "missing channel: pass a channel JID"}
        if cmd["cmd"] == "chats":
            return {"ok": True, "chats": [
                {"id": NL, "title": "Gold Signals WA",
                 "kind": "channel", "ts": 1},
                {"id": "12345@g.us", "title": "Crew",
                 "kind": "group", "ts": 2}]}
        if cmd["cmd"] == "channel_fetch":
            return {"ok": True, "posts": [{
                "chat": cmd["chat"], "name": "", "server_id": "9",
                "message_id": "", "text": "hi", "kind": "",
                "path": "", "media_missing": False, "views": 0,
                "ts": 1000}]}
        return {"ok": False, "error": "unexpected"}

    adapter, sent = _wa_adapter_capturing(responder)
    out = adapter.fetch_channel_posts()
    assert [r["story_id"] for r in out] == ["9"]
    fetches = [c for c in sent if c.get("cmd") == "channel_fetch"
               and c.get("chat") == NL]
    assert len(fetches) == 1          # only the channel chat
    # an explicit channel list skips enumeration entirely
    sent.clear()
    out = adapter.fetch_channel_posts(channels=[NL])
    assert [r["story_id"] for r in out] == ["9"]
    assert not [c for c in sent if c.get("cmd") == "chats"]


# ── 3. the personal organ's channel cycle ────────────────────────

def _post(server_id, *, text="gold update", path="", kind="",
          media_missing=False, ts=100.0):
    return {"poster_id": NL, "poster_name": "Gold Signals WA",
            "story_id": server_id, "text": text, "path": path,
            "kind": kind, "media_missing": media_missing,
            "views": 3, "ts": ts}


def test_cycle_saves_text_and_media_posts_once(tmp_path):
    media = _media_file(tmp_path, "chan.jpg")
    calls = {"fetch": 0}

    def fetcher(rec):
        calls["fetch"] += 1
        return [_post("101"),
                _post("102", text="", path=media, kind="image")]

    personal, registry, vault, rec = make_personal(
        tmp_path, channel_fetcher=fetcher)
    registry.set_toggles(rec.id, status_saving=True)
    out = personal.cycle_channel_posts("whatsapp")
    assert len(out) == 2
    rows = vault.saved_entries(rec.id)
    assert [r["kind"] for r in rows] == ["channel_post"] * 2
    assert rows[0]["counterpart_id"] == NL
    assert rows[0]["text"] == "gold update"
    assert Path(rows[1]["path"]).exists()
    # a second sweep of the same feed saves nothing twice
    assert personal.cycle_channel_posts("whatsapp") == []
    assert len(vault.saved_entries(rec.id)) == 2
    assert calls["fetch"] == 2


def test_cycle_marks_viewed_once_per_post(tmp_path):
    viewed = []

    def viewer(rec, item):
        viewed.append(item["source_id"])
        return True

    personal, registry, vault, rec = make_personal(
        tmp_path, channel_fetcher=lambda r: [_post("201")],
        channel_viewer=viewer)
    registry.set_toggles(rec.id, status_viewing=True)
    personal.cycle_channel_posts("whatsapp")
    personal.cycle_channel_posts("whatsapp")
    assert viewed == ["201"]
    # viewing alone saves nothing
    assert vault.saved_entries(rec.id) == []


def test_cycle_refused_view_journals_failure_once(tmp_path):
    personal, registry, vault, rec = make_personal(
        tmp_path, channel_fetcher=lambda r: [_post("301")],
        channel_viewer=lambda r, item: False)
    registry.set_toggles(rec.id, status_viewing=True)
    personal.cycle_channel_posts("whatsapp")
    personal.cycle_channel_posts("whatsapp")
    rows = [r for r in vault.journal(rec.id)
            if r["kind"] == "channel_view_failed"]
    assert len(rows) == 1
    assert rows[0]["source_id"] == "301"
    assert rows[0]["counterpart_id"] == NL


def test_cycle_media_miss_is_journaled_not_vanished(tmp_path):
    personal, registry, vault, rec = make_personal(
        tmp_path,
        channel_fetcher=lambda r: [_post(
            "401", text="", kind="image", media_missing=True)])
    registry.set_toggles(rec.id, status_saving=True)
    out = personal.cycle_channel_posts("whatsapp")
    assert [r["kind"] for r in out] == ["channel_post_missed"]
    personal.cycle_channel_posts("whatsapp")
    rows = [r for r in vault.journal(rec.id)
            if r["kind"] == "channel_post_missed"]
    assert len(rows) == 1


def test_cycle_toggles_off_never_fetches(tmp_path):
    calls = {"fetch": 0}

    def fetcher(rec):
        calls["fetch"] += 1
        return [_post("501")]

    personal, registry, vault, rec = make_personal(
        tmp_path, channel_fetcher=fetcher,
        channel_viewer=lambda r, item: True)
    assert personal.cycle_channel_posts("whatsapp") == []
    assert calls["fetch"] == 0
    assert vault.journal(rec.id) == []


def test_note_channel_posted_journals_the_act(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    entry = personal.note_channel_posted(
        "whatsapp", account_id=rec.id, channel_id=NL,
        channel_name="Gold Signals WA", text="gold update",
        source_id="12345")
    assert entry["kind"] == "channel_posted"
    rows = [r for r in vault.journal(rec.id)
            if r["kind"] == "channel_posted"]
    assert len(rows) == 1
    assert rows[0]["counterpart_id"] == NL
    assert rows[0]["poster_name"] == "Gold Signals WA"
    assert rows[0]["source_id"] == "12345"


# ── 4. the /chanstory command (real handler, walive harness) ────

def _beat_runtime(tmp_path):
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
    svc["supervision"] = SimpleNamespace(sweep=lambda: None)
    return rt, svc


class FakeWAChannelAdapter:
    def __init__(self, verdict=None, chats_rows=None):
        self.calls: list[dict] = []
        self.resolves: list[str] = []
        self.verdict = verdict or {"ok": True,
                                    "message_id": "WAMID1",
                                    "server_id": "12345"}
        self.chats_rows = chats_rows or []

    def channel_send(self, chat, text="", *, media_path="",
                     caption=""):
        self.calls.append({"chat": str(chat), "text": text,
                           "media_path": media_path,
                           "caption": caption})
        return self.verdict

    def channel_resolve_invite(self, code):
        self.resolves.append(code)
        if code == "AbC123xYz":
            return {"ok": True, "chat": NL,
                    "name": "Gold Signals WA"}
        return {"ok": False,
                "error": "channel invite did not resolve to a "
                         "channel"}

    def chats(self, limit=30):
        return list(self.chats_rows)


class FakeTGChannelAdapter:
    def __init__(self, verdict=None):
        self.calls: list[tuple] = []
        self.verdict = verdict or {"ok": True}

    def send_channel_story(self, channel_ref, media_path,
                           caption=""):
        self.calls.append((channel_ref, media_path, caption))
        return self.verdict


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


def test_chanstory_whatsapp_text_posts_and_journals(tmp_path):
    fake = FakeWAChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "chanstory", f"{NL} gold update", _command_message("whatsapp"))
    assert reply == "posted to your channel on WhatsApp."
    assert fake.calls == [{"chat": NL, "text": "gold update",
                           "media_path": "", "caption": ""}]
    rows = [r for r in svc["personal"].vault.journal(rec.id)
            if r["kind"] == "channel_posted"]
    assert len(rows) == 1
    assert rows[0]["counterpart_id"] == NL
    assert rows[0]["source_id"] == "12345"   # server_id journaled


def test_chanstory_whatsapp_invite_link_resolves(tmp_path):
    fake = FakeWAChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "chanstory",
        "https://whatsapp.com/channel/AbC123xYz gold update",
        _command_message("whatsapp"))
    assert fake.resolves == ["AbC123xYz"]
    assert fake.calls[0]["chat"] == NL
    assert reply == "posted to “Gold Signals WA” on WhatsApp."


def test_chanstory_whatsapp_media_from_caption_message(tmp_path):
    fake = FakeWAChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    media = [SimpleNamespace(path=_media_file(tmp_path),
                             kind="image")]
    reply = rt._control_chatops(
        "chanstory", f"{NL} day one",
        _command_message("whatsapp", media=media))
    assert reply.startswith("posted to")
    assert fake.calls[0]["media_path"] == media[0].path
    assert fake.calls[0]["caption"] == "day one"


def test_chanstory_usage_and_unknown_channel(tmp_path):
    fake = FakeWAChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "chanstory", "", _command_message("whatsapp"))
    assert reply.startswith("usage: /chanstory")
    reply = rt._control_chatops(
        "chanstory", "No Such Channel hello",
        _command_message("whatsapp"))
    # an unresolvable reference posts nothing and says why
    assert fake.calls == []
    assert "couldn't do it" in reply


def test_chanstory_whatsapp_channel_name_form(tmp_path):
    fake = FakeWAChannelAdapter(chats_rows=[
        {"id": NL, "title": "Gold Signals WA", "kind": "channel",
         "last_ts": 1.0},
        {"id": "12345@g.us", "title": "Gold Crew", "kind": "group",
         "last_ts": 2.0}])
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "chanstory", "Gold Signals WA gold update",
        _command_message("whatsapp"))
    # the multi-word name resolves; the rest is the post text
    assert reply == "posted to “Gold Signals WA” on WhatsApp."
    assert fake.calls == [{"chat": NL, "text": "gold update",
                           "media_path": "", "caption": ""}]


def test_chanstory_owner_stop_halts(tmp_path):
    fake = FakeWAChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    svc["registry"].set_owner_stopped("whatsapp", True)
    reply = rt._control_chatops(
        "chanstory", f"{NL} gold update",
        _command_message("whatsapp"))
    assert "standing stop" in reply
    assert fake.calls == []


def test_chanstory_whatsapp_refusal_surfaces_verbatim(tmp_path):
    fake = FakeWAChannelAdapter(verdict={
        "ok": False,
        "error": "not-authorized: the account is not a channel admin"})
    rt, svc, rec = _command_runtime(tmp_path, "whatsapp", fake)
    reply = rt._control_chatops(
        "chanstory", f"{NL} gold update",
        _command_message("whatsapp"))
    assert reply == ("couldn't do it — not-authorized: the account "
                     "is not a channel admin")
    assert svc["personal"].vault.journal(rec.id) == []


def test_chanstory_telegram_media_posts_and_journals(tmp_path):
    fake = FakeTGChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "telegram", fake)
    media = [SimpleNamespace(path=_media_file(tmp_path),
                             kind="image")]
    reply = rt._control_chatops(
        "chanstory", "@goldchan day one",
        _command_message("telegram", media=media))
    assert reply == "posted to the channel's story on Telegram."
    assert fake.calls == [("@goldchan", media[0].path, "day one")]
    rows = [r for r in svc["personal"].vault.journal(rec.id)
            if r["kind"] == "channel_posted"]
    assert len(rows) == 1
    assert rows[0]["counterpart_id"] == "@goldchan"


def test_chanstory_telegram_text_only_is_platform_truth(tmp_path):
    fake = FakeTGChannelAdapter()
    rt, svc, rec = _command_runtime(tmp_path, "telegram", fake)
    reply = rt._control_chatops(
        "chanstory", "@goldchan words only",
        _command_message("telegram"))
    assert "media posts" in reply
    assert fake.calls == []


def test_chanstory_telegram_boost_refusal_surfaces(tmp_path):
    fake = FakeTGChannelAdapter(verdict={
        "ok": False, "error": "BOOSTS_REQUIRED (caused by "
                              "SendStoryRequest)"})
    rt, svc, rec = _command_runtime(tmp_path, "telegram", fake)
    media = [SimpleNamespace(path=_media_file(tmp_path),
                             kind="image")]
    reply = rt._control_chatops(
        "chanstory", "@goldchan day one",
        _command_message("telegram", media=media))
    assert "BOOSTS_REQUIRED" in reply
    assert svc["personal"].vault.journal(rec.id) == []


# ── 5. the Telegram adapter against fake Telethon modules ───────

class _RecordingRequest:
    def __init__(self, **kw):
        self.kw = kw


class _GetAllStoriesRequest(_RecordingRequest):
    pass


class _GetPeerStoriesRequest(_RecordingRequest):
    pass


@pytest.fixture()
def fake_telethon_channel(monkeypatch):
    stories_mod = SimpleNamespace(
        SendStoryRequest=lambda **kw: _RecordingRequest(**kw),
        GetAllStoriesRequest=_GetAllStoriesRequest,
        GetPeerStoriesRequest=_GetPeerStoriesRequest,
        ReadStoriesRequest=lambda **kw: _RecordingRequest(**kw))
    types_mod = SimpleNamespace(
        InputPrivacyValueAllowAll=lambda: ("privacy", "allow-all"),
        InputMediaUploadedPhoto=lambda file: ("photo", file),
        InputMediaUploadedDocument=lambda **kw: ("document", kw),
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

    def fake_get_peer_id(obj):
        if getattr(obj, "_kind", "") == "channel":
            return -1000000000000 - int(obj.id)
        return int(getattr(obj, "id", 0) or 0)

    utils_mod = SimpleNamespace(get_peer_id=fake_get_peer_id)
    monkeypatch.setitem(sys.modules, "telethon.utils", utils_mod)
    return stories_mod


def _run_sync(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeTGClient:
    def __init__(self):
        self.requests: list = []
        self.uploads: list = []
        self.entities: list = []

    async def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace()

    async def get_input_entity(self, ref):
        self.entities.append(ref)
        return ("input-peer", ref)

    async def upload_file(self, path):
        self.uploads.append(path)
        return ("input-file", path)


def _tg_adapter(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    adapter._run_on_loop = lambda coro, timeout=30.0: _run_sync(coro)
    return adapter


def test_tg_send_channel_story_targets_the_channel(
        tmp_path, fake_telethon_channel):
    adapter = _tg_adapter(tmp_path)
    src = _media_file(tmp_path, "story.jpg")
    # not connected → honest refusal dict, no traffic
    assert adapter.send_channel_story("@goldchan", src)["ok"] is False
    client = FakeTGClient()
    adapter._client = client
    adapter._loop = object()
    out = adapter.send_channel_story("@goldchan", src, "day one")
    assert out == {"ok": True}
    assert client.entities == ["@goldchan"]
    req = client.requests[-1]
    assert req.kw["peer"] == ("input-peer", "@goldchan")
    assert req.kw["caption"] == "day one"
    # a numeric channel id resolves as an int peer ref
    out = adapter.send_channel_story("-1001234567890", src)
    assert out == {"ok": True}
    assert client.entities[-1] == -1001234567890
    # a missing file refuses before any traffic
    client.requests.clear()
    out = adapter.send_channel_story(
        "@goldchan", str(tmp_path / "gone.jpg"))
    assert out["ok"] is False
    assert client.requests == []


def test_tg_send_channel_story_refusal_carries_reason(
        tmp_path, fake_telethon_channel):
    adapter = _tg_adapter(tmp_path)
    src = _media_file(tmp_path, "story.jpg")

    class BoostRefusingClient(FakeTGClient):
        async def __call__(self, request):
            raise RuntimeError(
                "BOOSTS_REQUIRED (caused by SendStoryRequest)")

    adapter._client = BoostRefusingClient()
    adapter._loop = object()
    out = adapter.send_channel_story("@goldchan", src)
    assert out["ok"] is False
    assert "BOOSTS_REQUIRED" in out["error"]


class FakeTGFeedClient:
    """The fetcher rig: an empty contacts feed + one channel
    dialog whose PeerStories carry one story."""

    def __init__(self, media_dir):
        self.media_dir = media_dir
        self.requests: list = []

    async def __call__(self, request):
        self.requests.append(request)
        if isinstance(request, _GetAllStoriesRequest):
            return SimpleNamespace(peer_stories=[], users=[],
                                   chats=[])
        if isinstance(request, _GetPeerStoriesRequest):
            import datetime
            peer = SimpleNamespace(_kind="channel", id=777)
            chat = SimpleNamespace(_kind="channel", id=777,
                                   title="Gold Chan")
            story = SimpleNamespace(
                id=42,
                media=SimpleNamespace(document=None),
                date=datetime.datetime.fromtimestamp(1794100000))
            return SimpleNamespace(
                stories=SimpleNamespace(peer=peer, stories=[story]),
                users=[], chats=[chat])
        raise AssertionError(f"unexpected request {request!r}")

    async def get_dialogs(self):
        return [SimpleNamespace(is_channel=True,
                                input_entity="chan-input"),
                SimpleNamespace(is_channel=False,
                                input_entity="user-input")]

    async def download_media(self, media, file=None):
        return file


def test_tg_fetch_includes_followed_channel_stories(
        tmp_path, fake_telethon_channel):
    adapter = _tg_adapter(tmp_path)
    client = FakeTGFeedClient(str(tmp_path / "media"))
    adapter._client = client
    adapter._loop = object()
    rows = adapter.fetch_active_stories()
    assert len(rows) == 1
    row = rows[0]
    assert row["poster_id"] == str(-1000000000000 - 777)
    assert row["poster_name"] == "Gold Chan"
    assert row["story_id"] == "42"
    peer_reqs = [r for r in client.requests
                 if isinstance(r, _GetPeerStoriesRequest)]
    assert len(peer_reqs) == 1
    assert peer_reqs[0].kw["peer"] == "chan-input"


# ── 6. catalog registration ──────────────────────────────────────

def test_chanstory_registered_everywhere():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
    )

    assert "chanstory" in CHATOPS_COMMANDS
    assert "chanstory" in CONTROL_COMMANDS
    assert "chanstory" in COMMAND_DETAILS
    assert "chanstory" in LIST_ONELINERS
    grouped = {name for _title, names in LIST_GROUPS
               for name in names}
    assert "chanstory" in grouped
    # §33 posture: an owner-only account op, never public
    from nomorals.social.chat.control import PUBLIC_COMMANDS
    assert "chanstory" not in PUBLIC_COMMANDS


def test_chanstory_not_before_this_build():
    # red-on-base witness is procedural (the whole file is new);
    # this pin keeps the catalog claim honest if the command is
    # ever renamed: details must carry the boost truth.
    from nomorals.social.chat.control import COMMAND_DETAILS

    assert "boost" in COMMAND_DETAILS["chanstory"]["what"]
