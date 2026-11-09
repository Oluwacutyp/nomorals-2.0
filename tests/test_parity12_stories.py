"""Parity bundle #12 — Telegram stories completion (census §7
row 12): react to a story on the personal surface, and bot
story posting where the platform actually permits a bot.

Platform truth the build stands on (Bot API, verified against
core.telegram.org/bots/api):

- MTProto carries a real story-reaction verb —
  ``stories.SendReactionRequest`` (peer, story_id, reaction) —
  so the personal surface's story cycle gains a react leg
  driven by the same status_react toggle/pool/stop discipline
  as WhatsApp status reactions.
- The Bot API's only story verb is ``postStory``, and it takes
  a *business_connection_id*: a bot posts a story only on
  behalf of a connected Telegram Business account whose bot
  rights include can_manage_stories.  There is NO bot
  channel-story method at all (no chat_id story verb exists) —
  channel stories stay a personal-account post (/chanstory),
  and the bot surfaces say exactly that instead of faking one.

Pins: the catalog registration (no new kinds — poststatus and
chanstory already exist; the bot branches ride them), the
userbot adapter's request shape + refusal honesty against a
fake Telethon tree (the house shape — Telethon is not
installed here), the personal organ's react cycle
(toggle/stop/dedupe/journal/pool), the bot adapter's
business-connection tracking + postStory wire shapes against
a fake session, and the chatop branches through the real
chatops handlers on the walive bare-runtime harness.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.pairing import AccountRecord, PairingRegistry
from nomorals.social.personal import PersonalSurfaces, PersonalVault

_REPO = Path(__file__).resolve().parent.parent


# ── 1. catalog: the commands already exist; the bot route is
#       documented on the one that gains it ─────────────────

def test_catalog_carries_the_story_commands():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    for kind in ("poststatus", "chanstory"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert any(kind in names for _t, names in LIST_GROUPS)
        # owner-only account acts — never public commands
        assert kind not in PUBLIC_COMMANDS
    # The bot's business route is part of the documented
    # surface, not a hidden branch.
    assert "Business" in COMMAND_DETAILS["poststatus"]["what"]
    assert "postStory" in COMMAND_DETAILS["poststatus"]["what"]


# ── 2. the userbot story reaction (fake Telethon tree) ──────

class _FakeTLType:
    """A Telethon TL type stand-in: kwargs become attributes."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture()
def tg_reactor(monkeypatch, tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    stories_mod = _types.ModuleType(
        "telethon.tl.functions.stories")
    for cls_name in ("SendReactionRequest",):
        setattr(stories_mod, cls_name,
                type(cls_name, (_FakeTLType,), {}))
    types_mod = _types.ModuleType("telethon.tl.types")
    for cls_name in ("ReactionEmoji",):
        setattr(types_mod, cls_name,
                type(cls_name, (_FakeTLType,), {}))
    for name, mod in (
            ("telethon", _types.ModuleType("telethon")),
            ("telethon.tl", _types.ModuleType("telethon.tl")),
            ("telethon.tl.types", types_mod),
            ("telethon.tl.functions",
             _types.ModuleType("telethon.tl.functions")),
            ("telethon.tl.functions.stories", stories_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    requests: list = []
    entities: list = []

    class _Client:
        async def get_input_entity(self, ref):
            entities.append(ref)
            return f"entity:{ref}"

        async def __call__(self, request):
            requests.append(request)
            return object()

    ad._client = _Client()  # type: ignore[attr-defined]
    ad._loop = object()  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ad, "_run_on_loop",
        lambda coro, timeout=60.0: asyncio.run(coro))
    ad.requests = requests  # type: ignore[attr-defined]
    ad.entities = entities  # type: ignore[attr-defined]
    return ad


def test_react_to_story_request_shape(tg_reactor):
    verdict = tg_reactor.react_to_story("555", 7, "🔥")
    assert verdict == {"ok": True}
    (req,) = tg_reactor.requests
    assert type(req).__name__ == "SendReactionRequest"
    assert req.peer == "entity:555"
    assert req.story_id == 7
    assert type(req.reaction).__name__ == "ReactionEmoji"
    assert req.reaction.emoticon == "🔥"
    # a numeric poster id resolves as an int peer reference
    assert tg_reactor.entities == [555]


def test_react_to_story_name_poster_passes_through(tg_reactor):
    verdict = tg_reactor.react_to_story("ada_lovelace", "9", "❤️")
    assert verdict == {"ok": True}
    assert tg_reactor.entities == ["ada_lovelace"]
    (req,) = tg_reactor.requests
    assert req.story_id == 9  # the string id coerces


def test_react_to_story_refusals_before_traffic(tg_reactor):
    assert "poster" in tg_reactor.react_to_story(
        "", 7, "🔥")["error"]
    assert "numeric id" in tg_reactor.react_to_story(
        "555", "abc", "🔥")["error"]  # type: ignore[arg-type]
    assert "emoji" in tg_reactor.react_to_story(
        "555", 7, " ")["error"]
    assert tg_reactor.requests == []


def test_react_to_story_not_connected(monkeypatch, tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    verdict = ad.react_to_story("555", 7, "🔥")
    assert verdict["ok"] is False
    assert verdict["error"] == "telegram userbot not connected"


def test_react_to_story_api_refusal_carries_its_words(
        tg_reactor, monkeypatch):
    class _Refusing:
        async def get_input_entity(self, ref):
            return ref

        async def __call__(self, request):
            raise RuntimeError("STORY_ID_INVALID: story expired")

    tg_reactor._client = _Refusing()  # type: ignore[attr-defined]
    verdict = tg_reactor.react_to_story("555", 7, "🔥")
    assert verdict["ok"] is False
    assert "STORY_ID_INVALID" in verdict["error"]


# ── 3. the organ: the story cycle's react leg ───────────────

def _make_personal(tmp_path, **surface_kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault, **surface_kw)
    rec = registry.upsert(AccountRecord(
        id="telegram:555000", platform="telegram",
        identity="555000", status="linked", created_at=1.0))
    return personal, registry, vault, rec


def _story(sid="7", poster="555", name="Ada"):
    return {"story_id": sid, "poster_id": poster,
            "poster_name": name, "ts": 10.0, "path": "",
            "kind": ""}


def _journal(vault, kind):
    return [r for r in vault.journal() if r.get("kind") == kind]


def test_cycle_reacts_journals_and_dedupes(tmp_path):
    calls: list = []

    def reactor(rec, poster, sid, emoji):
        calls.append((poster, sid, emoji))
        return True

    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_reactor=reactor)
    registry.set_toggles(rec.id, status_react=True)
    personal.cycle_stories("telegram")
    personal.cycle_stories("telegram")  # second sweep: no re-fire
    assert calls == [("555", 7, "❤️")]
    rows = _journal(vault, "story_react")
    assert len(rows) == 1
    assert rows[0]["counterpart_id"] == "555"
    assert rows[0]["source_id"] == "7"
    assert rows[0]["note"] == "reacted ❤️"


def test_cycle_react_never_refires_after_restart(tmp_path):
    calls: list = []

    def reactor(rec, poster, sid, emoji):
        calls.append((poster, sid))
        return True

    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_reactor=reactor)
    registry.set_toggles(rec.id, status_react=True)
    personal.cycle_stories("telegram")
    assert len(calls) == 1
    # A fresh organ over the same vault seeds the landed
    # reaction from the journal — no second reaction, ever.
    personal2, registry2, vault2, rec2 = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_reactor=reactor)
    registry2.set_toggles(rec2.id, status_react=True)
    personal2.cycle_stories("telegram")
    assert len(calls) == 1


def test_cycle_react_owner_stop_suppresses_the_act(tmp_path):
    calls: list = []

    def reactor(rec, poster, sid, emoji):
        calls.append((poster, sid))
        return True

    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_reactor=reactor)
    registry.set_toggles(rec.id, status_react=True)
    registry.set_owner_stopped("telegram", True)
    personal.cycle_stories("telegram")
    assert calls == []
    assert _journal(vault, "story_react") == []


def test_cycle_react_toggle_off_and_seam_absent(tmp_path):
    calls: list = []

    def reactor(rec, poster, sid, emoji):
        calls.append((poster, sid))
        return True

    # Toggle off: the fetcher still runs under saving, the
    # reactor never fires.
    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_reactor=reactor)
    registry.set_toggles(rec.id, status_saving=True)
    personal.cycle_stories("telegram")
    assert calls == []
    # No seam wired: absence, silently — nothing to react with.
    (tmp_path / "b").mkdir()
    personal2, registry2, vault2, rec2 = _make_personal(
        tmp_path / "b",
        story_fetcher=lambda r: [_story()])
    registry2.set_toggles(rec2.id, status_react=True)
    personal2.cycle_stories("telegram")
    assert _journal(vault2, "story_react") == []


def test_cycle_react_refusal_journals_once(tmp_path):
    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_reactor=lambda r, p, s, e: False)
    registry.set_toggles(rec.id, status_react=True)
    personal.cycle_stories("telegram")
    personal.cycle_stories("telegram")
    rows = _journal(vault, "story_react_failed")
    assert len(rows) == 1
    assert "did not land" in rows[0]["note"]


def test_cycle_react_is_independent_of_view(tmp_path):
    calls: list = []

    def reactor(rec, poster, sid, emoji):
        calls.append((poster, sid, emoji))
        return True

    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story()],
        story_viewer=lambda r, p, m: False,   # the view refuses
        story_reactor=reactor)
    registry.set_toggles(rec.id, status_viewing=True,
                         status_react=True)
    personal.cycle_stories("telegram")
    # The refused view never blocks the reaction — view and
    # react are separate legs of the same sweep.
    assert calls == [("555", 7, "❤️")]
    assert len(_journal(vault, "story_view_failed")) == 1
    assert len(_journal(vault, "story_react")) == 1


def test_cycle_react_draws_from_the_pool_rotation(tmp_path):
    calls: list = []

    def reactor(rec, poster, sid, emoji):
        calls.append(emoji)
        return True

    personal, registry, vault, rec = _make_personal(
        tmp_path,
        story_fetcher=lambda r: [_story("7"), _story("8")],
        story_reactor=reactor,
        status_react_pool=lambda r: ["😀", "😎"])
    rec.status_react_rotate = True
    registry.upsert(rec)
    registry.set_toggles(rec.id, status_react=True)
    personal.cycle_stories("telegram")
    # Each story draws the NEXT face — never the same twice
    # in a row while the pool holds more than one.
    assert calls == ["😀", "😎"]


# ── 4. the bot adapter: business connections + postStory ────

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeBotSession:
    """The core HttpClient's surface, recording: post_json for
    _api, post_multipart for _api_upload."""

    def __init__(self, result=None, error=None):
        self.calls: list = []
        self.uploads: list = []
        self._result = result if result is not None else {"id": 42}
        self._error = error

    def post_json(self, url, payload, timeout=None):
        self.calls.append((url, payload))
        if self._error:
            return _FakeResponse(
                {"ok": False, "description": self._error})
        return _FakeResponse({"ok": True, "result": self._result})

    def post_multipart(self, url, fields, files, timeout=None):
        self.uploads.append((url, fields, files))
        if self._error:
            return _FakeResponse(
                {"ok": False, "description": self._error})
        return _FakeResponse({"ok": True, "result": self._result})


def _bot(session=None):
    from nomorals.social.chat.telegram import TelegramBotAdapter

    return TelegramBotAdapter(
        token="123456:TEST", chat_allow="",
        media_dir="data/media/telegram-bot-test",
        session=session or _FakeBotSession())


_CONN = {"id": "bc-1", "user": {"id": 777, "first_name": "Boss"},
         "is_enabled": True,
         "rights": {"can_manage_stories": True}}


def test_business_connection_update_is_recorded_not_delivered():
    ad = _bot()
    delivered: list = []
    ad._dispatch_update(
        {"update_id": 5, "business_connection": dict(_CONN)},
        lambda msg: delivered.append(msg))
    assert delivered == []  # adapter state, never chat traffic
    (row,) = ad.business_connections()
    assert row["id"] == "bc-1"
    assert row["user_id"] == "777"
    assert row["is_enabled"] is True
    assert row["can_manage_stories"] is True
    # A later update can revoke the right / disable the link.
    ad._dispatch_update(
        {"update_id": 6,
         "business_connection": {
             "id": "bc-1", "user": {"id": 777},
             "is_enabled": False, "rights": {}}},
        lambda msg: None)
    (row,) = ad.business_connections()
    assert row["is_enabled"] is False
    assert row["can_manage_stories"] is False


def test_post_story_no_connection_names_the_platform_shape():
    session = _FakeBotSession()
    ad = _bot(session)
    verdict = ad.post_story(file_id="FID", media_kind="photo")
    assert verdict["ok"] is False
    assert "Business account" in verdict["error"]
    assert "no channel-story method" in verdict["error"]
    assert session.calls == [] and session.uploads == []


def test_post_story_connection_without_the_right_refuses():
    ad = _bot()
    ad._note_business_connection(
        {"id": "bc-2", "user": {"id": 1}, "is_enabled": True,
         "rights": {"can_manage_stories": False}})
    verdict = ad.post_story(file_id="FID", media_kind="photo")
    assert verdict["ok"] is False
    assert "can_manage_stories" in verdict["error"]


def test_post_story_file_id_wire_shape():
    session = _FakeBotSession(result={"id": 99})
    ad = _bot(session)
    ad._note_business_connection(dict(_CONN))
    verdict = ad.post_story(caption="launch day", file_id="PHOTO1",
                            media_kind="photo", active_period=43200)
    assert verdict == {"ok": True, "story_id": "99"}
    (url, payload), = session.calls
    assert url.endswith("/postStory")
    assert payload["business_connection_id"] == "bc-1"
    assert payload["content"] == {"type": "photo",
                                  "photo": "PHOTO1"}
    assert payload["active_period"] == 43200
    assert payload["caption"] == "launch day"


def test_post_story_file_id_needs_an_explicit_kind():
    ad = _bot()
    ad._note_business_connection(dict(_CONN))
    verdict = ad.post_story(file_id="SOMEID")
    assert verdict["ok"] is False
    assert "media_kind" in verdict["error"]


def test_post_story_upload_wire_shape(tmp_path):
    session = _FakeBotSession(result={"id": 7})
    ad = _bot(session)
    ad._note_business_connection(dict(_CONN))
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\x00\x00ftyp")
    verdict = ad.post_story(str(media), caption="demo")
    assert verdict == {"ok": True, "story_id": "7"}
    (url, fields, files), = session.uploads
    assert url.endswith("/postStory")
    assert fields["business_connection_id"] == "bc-1"
    assert json.loads(fields["content"]) == {
        "type": "video", "video": "attach://story_media"}
    assert fields["active_period"] == "86400"
    assert fields["caption"] == "demo"
    (file_field, file_path, _mime, _name), = files
    assert file_field == "story_media"
    assert Path(file_path) == media


def test_post_story_refusals_before_traffic(tmp_path):
    session = _FakeBotSession()
    ad = _bot(session)
    ad._note_business_connection(dict(_CONN))
    # A lifetime Telegram does not offer.
    verdict = ad.post_story(file_id="F", media_kind="photo",
                            active_period=3600)
    assert verdict["ok"] is False
    assert "active period" in verdict["error"]
    # A gone file, and a kind a story cannot be.
    verdict = ad.post_story(str(tmp_path / "gone.jpg"))
    assert "not found" in verdict["error"]
    odd = tmp_path / "notes.txt"
    odd.write_text("hi")
    verdict = ad.post_story(str(odd))
    assert "photo" in verdict["error"]
    # Nothing at all.
    verdict = ad.post_story()
    assert "media post" in verdict["error"]
    assert session.calls == [] and session.uploads == []


def test_post_story_api_refusal_carries_telegrams_words():
    session = _FakeBotSession(
        error="Bad Request: BUSINESS_CONNECTION_INVALID")
    ad = _bot(session)
    ad._note_business_connection(dict(_CONN))
    verdict = ad.post_story(file_id="F", media_kind="photo")
    assert verdict["ok"] is False
    assert "BUSINESS_CONNECTION_INVALID" in verdict["error"]


def test_poll_loop_asks_for_business_connection_updates():
    src = (_REPO / "nomorals" / "social" / "chat"
           / "telegram.py").read_text(encoding="utf-8")
    assert '"business_connection",' in src
    assert "allowed_updates" in src


# ── 5. the chatop branches (bare-runtime harness) ───────────

def _beat_runtime(tmp_path):
    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter

    chat = SimpleNamespace(
        whatsapp_enabled=True, whatsapp_host="127.0.0.1",
        whatsapp_port=8787, whatsapp_outbox_dir="",
        telegram_api_id="", telegram_api_hash="")
    settings = SimpleNamespace(
        chat=chat, workspace_dir=str(tmp_path),
        resolve=lambda rel: Path(tmp_path) / rel)
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.settings = settings
    rt.context = SimpleNamespace(settings=settings)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    rt._bump = lambda *a, **k: None
    rt._harvest_play_downloads = lambda *a, **k: None
    rt._drive_webshot_watches = None
    svc = rt._pairing_personal()
    svc["supervision"] = SimpleNamespace(sweep=lambda: None)
    return rt, svc


class _FakeBotStoryAdapter:
    def __init__(self, verdict=None):
        self.calls: list = []
        self.verdict = verdict or {"ok": True, "story_id": "3"}

    def post_story(self, media_path="", caption="", **kw):
        self.calls.append((media_path, caption))
        return self.verdict


def _bot_message(media=None):
    return SimpleNamespace(
        chat=SimpleNamespace(platform="telegram-bot"),
        media=media or [], meta={})


def test_poststatus_from_bot_chat_posts_through_business(
        tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    fake = _FakeBotStoryAdapter()
    rt.gateway.adapters["telegram-bot"] = fake
    media = [SimpleNamespace(path="/tmp/launch.jpg",
                             kind="photo")]
    reply = rt._chatops_poststatus("launch day",
                                   _bot_message(media))
    assert reply == ("posted to your story through the bot's "
                     "business connection.")
    assert fake.calls == [("/tmp/launch.jpg", "launch day")]


def test_poststatus_from_bot_chat_relays_the_refusal(tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    fake = _FakeBotStoryAdapter(
        verdict={"ok": False,
                 "error": "no business account is connected"})
    rt.gateway.adapters["telegram-bot"] = fake
    media = [SimpleNamespace(path="/tmp/launch.jpg",
                             kind="photo")]
    reply = rt._chatops_poststatus("", _bot_message(media))
    assert reply == ("couldn't do it — no business account is "
                     "connected")


def test_poststatus_from_bot_chat_without_media_explains(
        tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    rt.gateway.adapters["telegram-bot"] = _FakeBotStoryAdapter()
    reply = rt._chatops_poststatus("text only", _bot_message())
    assert "media posts" in reply
    assert "Business connection" in reply


def test_chanstory_from_bot_chat_answers_the_platform_truth(
        tmp_path):
    rt, svc = _beat_runtime(tmp_path)
    # No paired telegram account exists in this harness at
    # all — the bot branch answers BEFORE the personal gate,
    # so the reply is the platform truth, not a pairing nag.
    reply = rt._chatops_chanstory("@chan hello",
                                  _bot_message())
    assert "no channel-story method" in reply
    assert "personal Telegram" in reply
    assert "/poststatus" in reply


# ── 6. provision wiring (source pins, the house grain) ──────

def test_provision_wires_the_story_reactor():
    src = (_REPO / "nomorals" / "agents" / "partner"
           / "runtime_provision.py").read_text(encoding="utf-8")
    assert "story_reactor=_story_reactor" in src
    assert 'getattr(adapter, "react_to_story", None)' in src
    # The stale claim is gone: MTProto DOES carry the verb.
    assert "exposes no status-reaction verb" not in src
