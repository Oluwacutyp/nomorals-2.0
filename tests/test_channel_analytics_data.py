"""Channel analytics data plane (groups/channels census #3).

The census found the analytics pipeline plumbed to a source that
did not exist: ``note_channel_views`` was fed from
``SendResult.views`` and no adapter ever populated ``views``;
member counts sampled on the bot adapter only; no MTProto stats
calls anywhere.  This build gives all three surfaces real reads:

- the userbot adapter gains ``admin_member_count`` (full-chat
  reads — channels AND groups), ``channel_post_views`` (the
  counters re-read from the published posts themselves) and
  ``channel_broadcast_stats`` (stats.getBroadcastStats, with
  Telegram's refusal for small channels as a first-class state);
- the drive samples all three on the count cadence (claimed +
  analytics on only), feeding the surface stores;
- the console renders the newly real data with sampled-at
  stamps, and keeps absence, sampled-0 and refused-with-reason
  as three distinct lines.

Everything is faked at the adapter/Telethon seam: fake TL
function modules in sys.modules (the house pattern), a fake
client answering canned requests, fake gateway adapters for
the drive.  No network, no real Telethon.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatRef
from nomorals.social.chat.group_surface import GroupStore, \
    GroupSurface

NOW = time.time()
CHAN = ChatRef(platform="telegram", chat_id="-100222",
               kind="channel")
UKEY = "telegram:-100333"      # a userbot-held channel


# ── fake Telethon seam ──────────────────────────────────────────

class _Req:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.kw = kw


@pytest.fixture()
def fake_telethon(monkeypatch):
    channels_mod = SimpleNamespace(
        GetFullChannelRequest=lambda channel: _Req(
            "full_channel", channel=channel))
    messages_mod = SimpleNamespace(
        GetFullChatRequest=lambda chat_id: _Req(
            "full_chat", chat_id=chat_id),
        GetMessagesViewsRequest=lambda peer, id, increment: _Req(
            "views", peer=peer, id=id, increment=increment))
    stats_mod = SimpleNamespace(
        GetBroadcastStatsRequest=lambda channel, dark=False: _Req(
            "stats", channel=channel, dark=dark))
    functions_mod = SimpleNamespace(
        channels=channels_mod, messages=messages_mod,
        stats=stats_mod)
    tl_mod = SimpleNamespace(functions=functions_mod)
    telethon_mod = SimpleNamespace(tl=tl_mod)
    monkeypatch.setitem(sys.modules, "telethon", telethon_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl", tl_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.functions",
                        functions_mod)
    monkeypatch.setitem(
        sys.modules, "telethon.tl.functions.channels",
        channels_mod)
    monkeypatch.setitem(
        sys.modules, "telethon.tl.functions.messages",
        messages_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.functions.stats",
                        stats_mod)
    return functions_mod


class _FakeClient:
    """Answers TL requests from canned state; records the asks."""

    def __init__(self, *, full_count=None, views=None, stats=None,
                 stats_error=None, views_short=False,
                 messages=None):
        self.requests = []
        self._full_count = full_count
        self._views = views or {}
        self._stats = stats
        self._stats_error = stats_error
        self._views_short = views_short
        self._messages = messages or {}

    async def __call__(self, request):
        self.requests.append(request)
        if request.kind in ("full_channel", "full_chat"):
            return SimpleNamespace(
                full_chat=SimpleNamespace(
                    participants_count=self._full_count),
                chats=[SimpleNamespace(
                    participants_count=self._full_count)])
        if request.kind == "views":
            ids = list(request.kw["id"])
            if self._views_short:
                ids = ids[:-1]
            return SimpleNamespace(views=[
                SimpleNamespace(views=self._views.get(i),
                                forwards=None)
                for i in ids])
        if request.kind == "stats":
            if self._stats_error is not None:
                raise self._stats_error
            return self._stats
        raise AssertionError(request.kind)

    async def get_messages(self, entity, ids=None):
        return [
            SimpleNamespace(id=i, views=self._messages[i])
            if i in self._messages else None
            for i in (ids or [])]


def _userbot(client, entity=None):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._client = client
    ent = entity if entity is not None else SimpleNamespace(
        broadcast=True, megagroup=False, id=222,
        participants_count=None)

    async def _resolve(chat):
        return ent

    adapter._resolve = _resolve  # type: ignore[method-assign]

    def _run(coro, timeout=60.0):
        return asyncio.new_event_loop().run_until_complete(coro)

    adapter._run_on_loop = _run  # type: ignore[method-assign]
    return adapter


def _pair(current, previous):
    return SimpleNamespace(current=current, previous=previous)


def _broadcast_stats(**over):
    stats = SimpleNamespace(
        followers=_pair(1200.0, 1100.0),
        views_per_post=_pair(812.4, 700.0),
        shares_per_post=_pair(34.0, 30.0),
        reactions_per_post=_pair(91.0, 80.0),
        enabled_notifications=SimpleNamespace(part=3, total=4))
    for k, v in over.items():
        setattr(stats, k, v)
    return stats


# ── adapter: member counts over MTProto ─────────────────────────

def test_userbot_member_count_channel_route(fake_telethon):
    client = _FakeClient(full_count=4321)
    adapter = _userbot(client)
    out = adapter.admin_member_count(CHAN)
    assert out == {"ok": True, "count": 4321, "error": ""}
    assert client.requests[-1].kind == "full_channel"


def test_userbot_member_count_group_route(fake_telethon):
    client = _FakeClient(full_count=87)
    adapter = _userbot(client, entity=SimpleNamespace(id=111))
    out = adapter.admin_member_count(
        ChatRef(platform="telegram", chat_id="111", kind="group"))
    assert out["ok"] and out["count"] == 87
    assert client.requests[-1].kind == "full_chat"


def test_userbot_member_count_unexposed_is_refusal_not_zero(
        fake_telethon):
    client = _FakeClient(full_count=None)
    adapter = _userbot(client)  # entity participants_count None too
    out = adapter.admin_member_count(CHAN)
    assert out["ok"] is False
    assert "did not expose" in out["error"]


# ── adapter: post views re-read ─────────────────────────────────

def test_userbot_post_views_reads_current_counters(fake_telethon):
    client = _FakeClient(views={501: 1200, 502: None, 503: 45})
    adapter = _userbot(client)
    out = adapter.channel_post_views(CHAN, ["501", "502", "503"])
    # 502 came back with no counter: absent, never a 0
    assert out == {"ok": True, "views": {"501": 1200, "503": 45}}
    req = client.requests[-1]
    assert req.kind == "views" and req.kw["increment"] is False


def test_userbot_post_views_short_vector_falls_back_by_id(
        fake_telethon):
    client = _FakeClient(views={501: 10, 502: 20}, views_short=True,
                         messages={501: 700, 502: 800})
    adapter = _userbot(client)
    out = adapter.channel_post_views(CHAN, ["501", "502"])
    # positional pairing on a short vector would misattribute;
    # the by-id re-read names each message itself
    assert out == {"ok": True, "views": {"501": 700, "502": 800}}


def test_userbot_post_views_failure_is_honest(fake_telethon):
    class _Boom(_FakeClient):
        async def __call__(self, request):
            raise RuntimeError("flood wait")

    adapter = _userbot(_Boom())
    out = adapter.channel_post_views(CHAN, ["501"])
    assert out["ok"] is False and "flood wait" in out["error"]


# ── adapter: broadcast stats ────────────────────────────────────

def test_userbot_broadcast_stats_parses_pairs(fake_telethon):
    client = _FakeClient(stats=_broadcast_stats())
    adapter = _userbot(client)
    out = adapter.channel_broadcast_stats(CHAN)
    assert out["ok"] is True
    assert out["followers"] == 1200
    assert out["followers_prev"] == 1100
    assert out["views_per_post"] == 812.4
    assert out["views_per_post_prev"] == 700.0
    assert out["shares_per_post"] == 34.0
    assert out["reactions_per_post"] == 91.0
    assert out["enabled_notifications_pct"] == 75.0


def test_userbot_broadcast_stats_refusal_is_the_result(
        fake_telethon):
    client = _FakeClient(
        stats_error=RuntimeError("STATS_MIGRATED"))
    adapter = _userbot(client)
    out = adapter.channel_broadcast_stats(CHAN)
    assert out["ok"] is False
    assert "STATS_MIGRATED" in out["error"]


# ── surface: stats + views sample states ────────────────────────

def _surface(tmp_path: Path) -> GroupSurface:
    surface = GroupSurface(
        GroupStore(tmp_path / "g.json").load())
    surface.note_channel_seen(UKEY, title="Gold Signals")
    return surface


_STATS = {"followers": 900, "followers_prev": 850,
          "views_per_post": 500.0, "views_per_post_prev": 450.0,
          "shares_per_post": 10.0, "reactions_per_post": 25.0,
          "enabled_notifications_pct": 50.0}


def test_surface_stats_refusal_then_sample(tmp_path):
    surface = _surface(tmp_path)
    a = surface.channel_analytics(UKEY)
    assert a["stats"] is None and a["stats_unavailable"] is None
    assert a["stats_checked_at"] == 0
    # a refusal is a state with a reason — and it stamps the beat
    surface.note_channel_stats_unavailable(
        UKEY, "broadcast stats failed: STATS_MIGRATED", now=NOW)
    a = surface.channel_analytics(UKEY)
    assert a["stats"] is None
    assert "STATS_MIGRATED" in a["stats_unavailable"]
    assert a["stats_checked_at"] == NOW
    assert not surface.channel_stats_due(UKEY, now=NOW + 60)
    assert surface.channel_stats_due(UKEY, now=NOW + 7 * 3600)
    # a later granted sample lands and clears the refusal
    surface.note_channel_stats(UKEY, _STATS, now=NOW + 7 * 3600)
    a = surface.channel_analytics(UKEY)
    assert a["stats"]["followers"] == 900
    assert a["stats"]["followers_delta"] == 50
    assert a["stats"]["views_per_post"] == 500.0
    assert a["stats"]["sampled_at"] == NOW + 7 * 3600
    assert a["stats_unavailable"] is None


def test_surface_stats_sampled_zero_is_zero(tmp_path):
    surface = _surface(tmp_path)
    zero = dict(_STATS, followers=0, followers_prev=0)
    surface.note_channel_stats(UKEY, zero, now=NOW)
    a = surface.channel_analytics(UKEY)
    assert a["stats"]["followers"] == 0        # a real 0, not None
    assert a["stats"]["followers_delta"] == 0


def test_surface_views_read_beat_updates_not_sums(tmp_path):
    surface = _surface(tmp_path)
    assert surface.channel_views_due(UKEY, now=NOW)
    landed = surface.note_channel_views_read(
        UKEY, {"501": 1200}, now=NOW)
    assert landed == 1
    a = surface.channel_analytics(UKEY)
    assert a["views_total"] == 1200
    assert a["views_sampled_at"] == NOW
    assert not surface.channel_views_due(UKEY, now=NOW + 60)
    assert surface.channel_views_due(UKEY, now=NOW + 7 * 3600)
    # the next beat carries CURRENT counters: replaced, not added
    surface.note_channel_views_read(
        UKEY, {"501": 1500, "502": 30}, now=NOW + 7 * 3600)
    a = surface.channel_analytics(UKEY)
    assert a["views_total"] == 1530


# ── runtime drive: sampling end to end ──────────────────────────

class _Chat:
    def __init__(self, key, platform="telegram-bot", kind="dm",
                 title=""):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat("telegram-bot:99")
        self.meta = {}
        self.sender_id = kw.get("sender_id", "99")
        self.sender = kw.get("sender", "Boss")
        self.sender_username = "boss"
        self.text = kw.get("text", "")
        self.message_id = "55"
        self.reply_to = ""
        self.ts = 0.0

    def first_media(self):
        return None


class _Gateway:
    def __init__(self, adapters, send_views=None):
        self.adapters = adapters
        self.sent = []
        self._seq = 100
        self._send_views = send_views

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, str(
            getattr(chat, "key", chat)), text))
        self._seq += 1
        out = SimpleNamespace(ok=True, message_id=str(self._seq),
                              error="")
        if self._send_views is not None:
            out.views = self._send_views
        return out

    def send_file(self, platform, chat, path, **kw):
        return self.send(platform, chat, f"FILE {path}", **kw)


class _BareAdapter:
    _bot_id = 4242

    def admin_member(self, chat, user_id):
        status = "administrator" if str(user_id) in (
            "99", "4242") else "member"
        return {"ok": True, "result": {"status": status}}


class _AnalyticsAdapter:
    """The userbot shape after this build: count + views +
    stats verbs, with a call ledger."""

    def __init__(self, *, count=900, views=None, stats=None,
                 stats_error=""):
        self.calls = {"count": 0, "views": 0, "stats": 0}
        self._count = count
        self._views = views or {}
        self._stats = stats
        self._stats_error = stats_error

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "administrator"}}

    def admin_member_count(self, chat):
        self.calls["count"] += 1
        return {"ok": True, "count": self._count}

    def channel_post_views(self, chat, message_ids):
        self.calls["views"] += 1
        return {"ok": True, "views": {
            m: self._views[m] for m in message_ids
            if m in self._views}}

    def channel_broadcast_stats(self, chat):
        self.calls["stats"] += 1
        if self._stats_error:
            return {"ok": False, "error": self._stats_error}
        return {"ok": True, **(self._stats or {})}


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter, send_views=None):
        self.sent = []
        self.gateway = _Gateway(
            {"telegram-bot": adapter}, send_views=send_views)
        self._operator = True
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return True


def _dm_msg():
    chat = _Chat("telegram-bot:99", kind="dm")
    return _Msg(chat=chat)


def _claimed_userbot_channel(surface):
    surface.note_channel_seen(UKEY, title="Gold Signals")
    surface.set_designated(UKEY, True, by="owner")
    assert surface.claim_state(UKEY)["claimed"]
    return surface


_DRIVE_STATS = {"followers": 1500, "followers_prev": 1400,
                "views_per_post": 640.0, "views_per_post_prev": 600.0,
                "shares_per_post": 12.0, "reactions_per_post": 40.0,
                "enabled_notifications_pct": 62.5}


def test_drive_samples_views_and_stats_on_the_cycle(tmp_path):
    adapter = _AnalyticsAdapter(views={"101": 4321},
                                stats=_DRIVE_STATS)
    rt = _Runtime(tmp_path, _BareAdapter())
    rt.gateway.adapters["telegram"] = adapter
    surface = _claimed_userbot_channel(rt._group_surface_obj)
    surface.channel_enqueue(UKEY, {
        "kind": "once", "text": "gold update", "due": NOW - 5,
        "source": "dm"})
    # first drive: the count + stats beats land; the post goes
    # out (no published ids existed when the views beat ran)
    posts = rt._channel_drive(UKEY, now=NOW, force=True)
    assert len(posts) == 1 and posts[0]["message_id"] == "101"
    a = surface.channel_analytics(UKEY)
    assert a["subscribers"] == 900
    assert a["stats"]["followers"] == 1500
    assert adapter.calls == {"count": 1, "views": 0, "stats": 1}
    # second drive, a sample gap later: the views re-read feeds
    # note_channel_views with the post's CURRENT counter
    rt._channel_drive(UKEY, now=NOW + 7 * 3600)
    a = surface.channel_analytics(UKEY)
    assert a["views_total"] == 4321
    assert a["views_sampled_at"] == NOW + 7 * 3600
    assert adapter.calls["views"] == 1
    assert adapter.calls["stats"] == 2
    # a drive inside the gap samples nothing new — no poll storm
    rt._channel_drive(UKEY, now=NOW + 7 * 3600 + 60)
    assert adapter.calls["views"] == 1
    assert adapter.calls["stats"] == 2
    # the console card shows the real numbers with their stamps
    # (open the console home first — that is what issues the
    # c-handles, as in the console suite's own flow)
    rt._control_console("", _dm_msg())
    out = getattr(rt._control_console("c1 analytics", _dm_msg()),
                  "text", "")
    assert "subscribers: 900" in out
    assert "post views: 4,321 total" in out and "last read" in out
    assert "broadcast stats" in out
    assert "followers 1,500" in out and "+100" in out
    assert "640 views/post" in out


def test_drive_stats_refusal_renders_reason_without_storm(
        tmp_path):
    adapter = _AnalyticsAdapter(
        stats_error="broadcast stats failed: STATS_MIGRATED")
    rt = _Runtime(tmp_path, _BareAdapter())
    rt.gateway.adapters["telegram"] = adapter
    surface = _claimed_userbot_channel(rt._group_surface_obj)
    rt._channel_drive(UKEY, now=NOW, force=True)
    a = surface.channel_analytics(UKEY)
    assert a["stats"] is None
    assert "STATS_MIGRATED" in a["stats_unavailable"]
    # the refusal stamped the beat: the next drive does not
    # re-ask — no error storm against a too-small channel
    rt._channel_drive(UKEY, now=NOW + 60, force=True)
    assert adapter.calls["stats"] == 1
    lines = rt._channel_analytics_lines(surface, UKEY)
    joined = "\n".join(lines)
    assert "does not grant" in joined
    assert "STATS_MIGRATED" in joined
    assert "not a zero" in joined


def test_drive_send_receipt_views_seed_the_series(tmp_path):
    # a gateway result that carries views (the userbot adapter
    # now populates SendResult.views) feeds note_channel_views
    # through the drive's existing _send path — no views verb
    # on the adapter at all
    rt = _Runtime(tmp_path, _BareAdapter(), send_views=321)
    surface = _claimed_userbot_channel(rt._group_surface_obj)
    surface.channel_enqueue(UKEY, {
        "kind": "once", "text": "seeded", "due": NOW - 5,
        "source": "dm"})
    posts = rt._channel_drive(UKEY, now=NOW, force=True)
    assert len(posts) == 1
    a = surface.channel_analytics(UKEY)
    assert a["views_total"] == 321