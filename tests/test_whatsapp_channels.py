"""WhatsApp channels (groups/channels census gap #2): the channel
system reaches WhatsApp newsletters.

Before this build the WhatsApp adapter declared only DM/GROUP,
both bridges labeled newsletter traffic "dm", and the Go bridge
never said "channel" at all — so a channel post would have been
filed as a DM conversation, and the channel registry / claim /
posting queue / console could not see a WhatsApp channel at all.

Pins:
- a newsletter JID classifies as ChatKind.CHANNEL in the
  adapter, whatever kind the bridge claimed (an older bridge's
  "dm" label cannot get a channel post conversed with);
- the §16 posture holds on WhatsApp: a channel post gets no
  reply, no catalog, no brain — the sighting is only noted into
  the same registry Telegram channels live in;
- claim is by owner designation (there is no WhatsApp bot
  presence to claim from); once designated, the queue posts
  through the gateway to the newsletter JID, a failed send
  stays due, and the follower count samples from the bridge's
  channel_info (newsletter metadata);
- analytics absence stays absence: a bridge whose metadata
  carries no count samples nothing, and per-engine refusals
  (the Go bridge's missing delete/rename/description verbs)
  surface verbatim, never as success.

Rigs follow the house shapes: the fake-bridge socket rig from
test_whatsapp.py, the posture rig from test_channel_posture.py,
and the RuntimeGroupMixin stubs from test_group_channels_boards.py.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.whatsapp import WhatsAppAdapter

NL = "120363000111222333@newsletter"       # a channel JID
WKEY = f"whatsapp:{NL}"                    # its registry key
G = "telegram-bot:-100111"                 # a group beside it
C = "telegram-bot:-100222"                 # a Telegram channel


# ── adapter classification (bridge events, no socket) ─────────

def _dispatch(adapter, obj):
    msgs: list = []
    adapter._dispatch(obj, msgs.append)
    return msgs


def _message_event(kind_label, chat_id=NL):
    return {
        "type": "message", "id": "wamid.1",
        "chat": {"id": chat_id, "kind": kind_label,
                 "title": "Gold Signals WA"},
        "from": {"id": chat_id, "name": ""},
        "text": "gold update", "media": [], "reply_to": "",
        "mentioned": False, "ts": int(time.time() * 1000),
    }


def test_newsletter_message_classifies_as_channel():
    adapter = WhatsAppAdapter()
    msgs = _dispatch(adapter, _message_event("channel"))
    assert len(msgs) == 1
    assert msgs[0].chat.kind == ChatKind.CHANNEL
    assert msgs[0].chat.key == WKEY
    assert msgs[0].chat.platform == "whatsapp"


def test_old_bridge_dm_label_cannot_converse_with_a_channel():
    # A bridge build from before the channel work labels
    # newsletter traffic "dm" (its kindOf knew two shapes).
    # The JID is protocol truth: the adapter must still file
    # the post as a channel, or the brain would answer it.
    adapter = WhatsAppAdapter()
    msgs = _dispatch(adapter, _message_event("dm"))
    assert len(msgs) == 1
    assert msgs[0].chat.kind == ChatKind.CHANNEL


def test_own_message_in_channel_classifies_as_channel():
    adapter = WhatsAppAdapter()
    msgs = _dispatch(adapter, {
        "type": "own_message", "id": "wamid.2",
        "chat": {"id": NL, "kind": "dm", "title": ""},
        "text": "posted from the phone",
        "ts": int(time.time() * 1000)})
    assert len(msgs) == 1
    assert msgs[0].chat.kind == ChatKind.CHANNEL
    assert msgs[0].incoming is False


def test_group_and_dm_labels_survive():
    adapter = WhatsAppAdapter()
    group = _dispatch(adapter, _message_event("group",
                                              "12345-678@g.us"))
    dm = _dispatch(adapter, _message_event(
        "dm", "2348012345678@s.whatsapp.net"))
    assert group[0].chat.kind == ChatKind.GROUP
    assert dm[0].chat.kind == ChatKind.DM


def test_adapter_declares_channel_kind():
    assert ChatKind.CHANNEL in WhatsAppAdapter.supported_kinds


# ── posture + registry on the real runtime ────────────────────

@pytest.fixture()
def wa_rig():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="nm-wa-channel-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError(
                "the brain must not be called for a channel post")

    context.router = _Router()
    sends: list = []

    class _Gateway:
        adapters: dict = {}
        dry_run = False

        def status(self):
            return {"local": {"running_in_session": True}}

        def set_rate_limit(self, *a, **k):
            return None

        def typing(self, *a, **k):
            return True

        def send(self, platform, chat, text, **kw):
            sends.append(text)
            return SimpleNamespace(ok=True)

    runtime = PartnerRuntime(context, gateway=_Gateway())
    return runtime, sends


def _wa_channel_msg(text):
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=NL,
                     kind=ChatKind.CHANNEL,
                     title="Gold Signals WA"),
        incoming=True, text=text, sender="Gold Signals WA",
        sender_id=NL)


def test_whatsapp_channel_post_is_silent_and_only_noted(wa_rig):
    runtime, sends = wa_rig
    runtime._process_inner(_wa_channel_msg("/help"))
    runtime._process_inner(_wa_channel_msg("big move incoming"))
    assert sends == []
    surface = runtime._group_surface()
    # the sighting landed in the SAME registry Telegram
    # channels live in — unclaimed (no bot, no designation)
    assert surface.is_channel(WKEY)
    assert surface.channel_title(WKEY) == "Gold Signals WA"
    assert surface.claim_state(WKEY)["claimed"] is False


# ── runtime drive + console with a fake WhatsApp adapter ──────

class _Chat:
    def __init__(self, key=WKEY, platform="whatsapp",
                 kind="channel", title="Gold Signals WA"):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""
        self.ts = kw.get("ts", 0.0)

    def first_media(self):
        return None


class _WAAdapter:
    """The WhatsApp adapter's console-facing shape: no bot id
    (there is no WhatsApp bot), a real count when the bridge
    metadata carries one."""

    def __init__(self, count=777):
        self._count = count

    def admin_member_count(self, chat):
        if self._count is None:
            return {"ok": False,
                    "error": "the channel metadata carried no "
                             "follower count"}
        return {"ok": True, "count": self._count}


class _Gateway:
    def __init__(self, adapters, fail=False):
        self.adapters = adapters
        self.sent = []
        self.fail = fail

    def send(self, platform, chat, text, **kw):
        if self.fail:
            return SimpleNamespace(ok=False, message_id="",
                                   error="bridge refused")
        self.sent.append((platform, str(
            getattr(chat, "key", chat)), text))
        return SimpleNamespace(ok=True, message_id="1",
                               error="")

    def send_file(self, platform, chat, path, **kw):
        return self.send(platform, chat,
                         f"FILE {path} :: {kw.get('caption', '')}")


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapters):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self._operator = True
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return self._operator


def _dm_msg():
    chat = _Chat(key="whatsapp:2348012345678@s.whatsapp.net",
                 platform="whatsapp", kind="dm", title="")
    return _Msg(chat=chat, sender_id="99", sender="Boss",
                sender_username="boss")


def _text(out):
    return getattr(out, "text", out)


def _designated_wa_channel(rt):
    surface = rt._group_surface_obj
    surface.note_channel_seen(WKEY, title="Gold Signals WA")
    # no bot exists on WhatsApp — the owner's designation is
    # the claim, exactly the userbot pattern
    surface.set_designated(WKEY, True, by="owner")
    assert surface.claim_state(WKEY)["source"] == "designated"
    return surface


def test_designated_wa_channel_posts_and_samples(tmp_path):
    rt = _Runtime(tmp_path, {"whatsapp": _WAAdapter(count=777)})
    surface = _designated_wa_channel(rt)
    surface.channel_enqueue(WKEY, {
        "kind": "once", "text": "gold update for the week",
        "due": time.time() - 5, "source": "dm"})
    rt._channel_note_and_drive(_Msg(text="a channel post"))
    assert len(rt.gateway.sent) == 1
    platform, key, text = rt.gateway.sent[0]
    assert platform == "whatsapp" and key == WKEY
    assert "gold update for the week" in text
    # the drive sampled the adapter's real follower count
    analytics = surface.channel_analytics(WKEY)
    assert analytics["subscribers"] == 777
    assert analytics["posts"] == 1


def test_unclaimed_wa_channel_posts_nothing(tmp_path):
    rt = _Runtime(tmp_path, {"whatsapp": _WAAdapter()})
    surface = rt._group_surface_obj
    surface.note_channel_seen(WKEY, title="Gold Signals WA")
    surface.channel_enqueue(WKEY, {
        "kind": "once", "text": "must not go out",
        "due": time.time() - 5, "source": "dm"})
    rt._channel_note_and_drive(_Msg(text="a channel post"))
    assert rt.gateway.sent == []
    assert len(surface.channel_queue(WKEY)) == 1


def test_failed_send_to_wa_channel_stays_due(tmp_path):
    rt = _Runtime(tmp_path, {"whatsapp": _WAAdapter()})
    rt.gateway.fail = True
    surface = _designated_wa_channel(rt)
    surface.channel_enqueue(WKEY, {
        "kind": "once", "text": "retry me",
        "due": time.time() - 5, "source": "dm"})
    rt._channel_note_and_drive(_Msg(text="a channel post"))
    # the entry is still due — never journaled as posted
    assert len(surface.channel_queue(WKEY)) == 1
    assert surface.channel_analytics(WKEY)["posts"] == 0


def test_console_lists_wa_channel_beside_telegram(tmp_path):
    rt = _Runtime(tmp_path, {"whatsapp": _WAAdapter(count=777),
                             "telegram-bot": _WAAdapter()})
    surface = _designated_wa_channel(rt)
    surface.note_channel_seen(C, title="Gold Signals")
    surface.store.group(G)
    home = _text(rt._control_console("", _dm_msg()))
    assert "Your channels" in home
    assert "Gold Signals WA" in home and "Gold Signals" in home
    # open the WhatsApp channel (c2 once the telegram one is c1 —
    # resolve by scanning the session map the home just built)
    sessions = rt._console_session_map()
    token = next(t for t, k in
                 list(sessions.values())[-1]["channels"].items()
                 if k == WKEY)
    # a channel-activity beat drives the cycle, which samples
    # the count before it would post anything
    rt._channel_note_and_drive(_Msg(text="a channel post"))
    out = _text(rt._control_console(f"{token} analytics",
                                    _dm_msg()))
    assert "777" in out and "subscribers" in out


def test_console_analytics_absent_when_metadata_has_no_count(
        tmp_path):
    rt = _Runtime(tmp_path, {"whatsapp": _WAAdapter(count=None)})
    _designated_wa_channel(rt)
    rt._control_console("", _dm_msg())
    sessions = rt._console_session_map()
    token = next(t for t, k in
                 list(sessions.values())[-1]["channels"].items()
                 if k == WKEY)
    out = _text(rt._control_console(f"{token} analytics",
                                    _dm_msg()))
    assert "777" not in out and "0 subscribers" not in out


# ── adapter ↔ bridge (scripted socket bridge) ─────────────────

class FakeBridge:
    """A scripted JSON-lines bridge on 127.0.0.1:<ephemeral>
    (the test_whatsapp.py rig shape)."""

    def __init__(self, script=None):
        self.script = script or {}
        self.received: list = []
        self._srv = socket.socket(socket.AF_INET,
                                  socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET,
                             socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve,
                                        daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        buf = b""
        try:
            with conn:
                conn.settimeout(5)
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                    while b"\n" in buf:
                        line, _, buf = buf.partition(b"\n")
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
                        self.received.append(obj)
                        resp = self._respond(obj)
                        if resp is not None:
                            conn.sendall(
                                (json.dumps(resp) + "\n").encode())
        except OSError:
            pass

    def _respond(self, obj):
        handler = self.script.get(obj.get("cmd"))
        if handler is None:
            return {"id": obj.get("id"), "ok": False,
                    "error": "unknown command"}
        if callable(handler):
            return handler(obj)
        return {"id": obj.get("id"), **handler}

    def close(self):
        try:
            self._srv.close()
        except OSError:
            pass


def make_adapter(bridge, **kw):
    adapter = WhatsAppAdapter(host="127.0.0.1", port=bridge.port,
                              **kw)
    sock = socket.create_connection(("127.0.0.1", bridge.port),
                                    timeout=5)
    adapter._sock = sock
    handler_msgs: list = []
    reader = threading.Thread(
        target=adapter._reader_loop, args=(handler_msgs.append,),
        daemon=True)
    reader.start()
    adapter.connected.set()
    return adapter, handler_msgs


@pytest.fixture()
def bridge_rig(tmp_path):
    bridges = []

    def make(script):
        bridge = FakeBridge(script)
        bridges.append(bridge)
        adapter, _ = make_adapter(
            bridge, media_dir=str(tmp_path / "media"),
            outbox_dir=str(tmp_path / "outbox"))
        return adapter, bridge

    yield make
    for b in bridges:
        b.close()


def _channel_ref():
    return ChatRef(platform="whatsapp", chat_id=NL,
                   kind=ChatKind.CHANNEL)


def test_channel_info_and_count_from_bridge_metadata(bridge_rig):
    adapter, bridge = bridge_rig({
        "channel_info": {"ok": True, "name": "Gold Signals WA",
                         "description": "daily gold",
                         "subscribers": 4321}})
    info = adapter.channel_info(_channel_ref())
    assert info == {"ok": True, "name": "Gold Signals WA",
                    "description": "daily gold",
                    "subscribers": 4321}
    assert adapter.admin_member_count(_channel_ref()) == {
        "ok": True, "count": 4321}
    asks = [r for r in bridge.received
            if r.get("cmd") == "channel_info"]
    assert asks and asks[0]["chat"] == NL


def test_count_is_absence_when_metadata_carries_none(bridge_rig):
    adapter, _ = bridge_rig({
        "channel_info": {"ok": True, "name": "Gold Signals WA",
                         "description": "", "subscribers": None}})
    info = adapter.channel_info(_channel_ref())
    assert info["ok"] is True and info["subscribers"] is None
    count = adapter.admin_member_count(_channel_ref())
    assert count["ok"] is False        # never a fabricated 0


def test_group_count_refuses_without_a_bridge_call(bridge_rig):
    adapter, bridge = bridge_rig({})
    group = ChatRef(platform="whatsapp", chat_id="12345-678@g.us",
                    kind=ChatKind.GROUP)
    out = adapter.admin_member_count(group)
    assert out["ok"] is False
    assert not [r for r in bridge.received
                if r.get("cmd") == "channel_info"]


def test_go_bridge_channel_delete_refusal_surfaces(bridge_rig):
    # The Go engine's whatsmeow pin has no newsletter-delete
    # verb; its honest ok:false is the adapter's answer too —
    # per-engine truth, never a fake success.
    adapter, _ = bridge_rig({
        "channel_delete": {
            "ok": False,
            "error": "this bridge's whatsmeow build has no "
                     "newsletter delete verb — channel deletion "
                     "is unavailable"}})
    out = adapter.channel_delete(_channel_ref())
    assert out["ok"] is False
    assert "no newsletter delete verb" in out["error"]


def test_channel_post_goes_out_as_a_send_to_the_jid(bridge_rig):
    adapter, bridge = bridge_rig({
        "send": {"ok": True, "message_id": "wamid.post1"}})
    out = adapter.channel_post(_channel_ref(), "gold update")
    assert out["ok"] is True
    sends = [r for r in bridge.received if r.get("cmd") == "send"]
    assert sends and sends[0]["chat"] == NL
