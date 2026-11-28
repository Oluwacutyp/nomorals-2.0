"""Status view + react fix (gapfix166) — alt-JID carry-through and full-key reacts.

The two live failures, pinned at every link they cross:

1. ALT JID DROPPED (outside survey §1.3.2 / §2.3.2 — Devon is the
   outlier: every outside bot preserves the poster's alternate
   addressing form, Devon discards it).  The Go bridge emits
   ``from.alt`` (LID <-> phone) and the Baileys bridge's keys carry
   ``participantAlt``; the adapter's status_post branch never read
   either.  A status keyed under one form while the receipt/react
   carries the other is a silent no-op that still returns ok.
   Fix: the alt rides ``poster_alt_id`` from the wire event through
   the adapter meta, the personal-surface call, the organ's view/
   react items and trace row, the seams, and the bridge verbs —
   which prefer it as the receipt/react participant when present.
   The primary form is never replaced: sender_id stays from.id,
   and the trace records both forms so the live trace can still
   arbitrate which form WhatsApp keyed the status under.

2. REACT KEY REBUILT FROM BRIDGE MEMORY (outside survey §2.3.1 —
   no outside bot does this).  The adapter sent only
   {chat, message_id, emoji} and each bridge rebuilt the key from
   its in-memory cache — a restart, an old message, or a message
   that arrived while the bridge was down all refuse honestly
   instead of reacting.  Fix: the bridges emit the message's full
   original key in the event (``key``: remoteJid/id/participant/
   participantAlt/fromMe), the adapter stamps it on the message
   (``wa_key``), the acknowledge path passes it back through the
   gateway, and both bridges prefer the supplied key — memory is
   only the fallback when no key arrives.

3. BAILEYS receipt_mode MAPPING (outside survey §1.3.3): the mjs
   rung mapped any non-'all' privacy value to 'read-self' where
   Go maps three ways (All -> read, None -> read-self, anything
   else -> unknown).  A failed privacy fetch must read "unknown",
   never "receipts off".  Pinned in the bridge source
   (tests/test_statusfix.py reads the mjs file); the behavior
   itself is verified by node --check + review — the bridge file
   is a top-level script that logs into WhatsApp on import, so
   no test harness can execute it.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.pairing import AccountRecord, PairingRegistry

_REPO = Path(__file__).resolve().parent.parent
PN = "2348012345678@s.whatsapp.net"
LID = "123456789012345@lid"
POST_TS = 1794100000.0
GROUP = "120363025000000000@g.us"
KEY_GROUP = {"remoteJid": GROUP, "id": "m1", "participant": PN,
             "participantAlt": LID, "fromMe": False}
KEY_DM = {"remoteJid": PN, "id": "m2", "fromMe": False}


# ── adapter harness ───────────────────────────────────────────

def _wa_adapter_capturing(responder=None):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter.connected.set()
    sent = []

    def fake_send_cmd(cmd, *, timeout=25.0):
        sent.append(dict(cmd))
        if responder is not None:
            return responder(cmd)
        return {"ok": True}

    adapter._send_cmd = fake_send_cmd
    return adapter, sent


def _dispatch_status(adapter, wire_from, **extra):
    delivered = []
    event = {"type": "status_post", "id": "POST1",
             "from": wire_from, "text": "", "media": [],
             "ts": 1794100000000}
    event.update(extra)
    adapter._dispatch(event, delivered.append)
    assert len(delivered) == 1
    return delivered[0]


def _dispatch_message(adapter, event):
    delivered = []
    adapter._dispatch(event, delivered.append)
    assert len(delivered) == 1
    return delivered[0]


def _message_event(key=None, **overrides):
    event = {"type": "message", "id": "m1",
             "chat": {"id": GROUP, "kind": "group", "title": "g"},
             "from": {"id": PN, "name": "Poster"},
             "text": "hi", "media": [],
             "ts": int(time.time() * 1000)}
    if key is not None:
        event["key"] = key
    event.update(overrides)
    return event


# ── 1. the alt rides the status_post wire event ───────────────

def test_status_post_alt_stamped_in_meta():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_status(adapter, {"id": LID, "alt": PN,
                                     "name": "Poster"})
    # sender_id stays the wire from.id verbatim (the existing
    # pin); the alt rides alongside it in meta, never swapping in.
    assert msg.sender_id == LID
    assert msg.chat.peer == LID
    assert msg.meta["poster_alt_id"] == PN


def test_status_post_without_alt_has_no_alt_meta():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_status(adapter, {"id": PN, "name": "Poster"})
    assert "poster_alt_id" not in msg.meta


def test_message_event_key_stamped_in_meta():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_message(adapter, _message_event(KEY_GROUP))
    assert msg.meta["wa_key"] == KEY_GROUP


def test_message_event_without_key_has_no_wa_key():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_message(adapter, _message_event())
    assert "wa_key" not in msg.meta


# ── 2. the adapter's bridge payloads carry the alt / key ─────

def test_view_status_sends_from_alt():
    adapter, sent = _wa_adapter_capturing()
    adapter.view_status(LID, "POST1", ts=POST_TS, poster_alt_id=PN)
    assert sent == [{"cmd": "status_view", "message_id": "POST1",
                     "from": LID, "from_alt": PN}]


def test_react_status_sends_from_alt():
    adapter, sent = _wa_adapter_capturing()
    adapter.react_status(LID, "POST1", "❤️", poster_alt_id=PN)
    assert sent == [{"cmd": "status_react", "message_id": "POST1",
                     "from": LID, "from_alt": PN, "emoji": "❤️"}]


def test_react_message_sends_supplied_key():
    adapter, sent = _wa_adapter_capturing()
    chat = ChatRef(platform="whatsapp", chat_id=GROUP,
                   kind=ChatKind.GROUP)
    adapter.react_message(chat, "m1", "🔥", key=KEY_GROUP)
    assert sent == [{"cmd": "react", "chat": GROUP,
                     "message_id": "m1", "emoji": "🔥",
                     "key": KEY_GROUP}]


def test_react_message_refuses_malformed_key():
    adapter, sent = _wa_adapter_capturing()
    chat = ChatRef(platform="whatsapp", chat_id=GROUP,
                   kind=ChatKind.GROUP)
    res = adapter.react_message(chat, "m1", "🔥",
                                key={"id": "m1"})
    assert res["ok"] is False
    assert "key" in str(res.get("error") or "")
    assert sent == []  # nothing half-shaped goes on the wire


def test_react_threads_key_to_react_message():
    adapter, sent = _wa_adapter_capturing()
    chat = ChatRef(platform="whatsapp", chat_id=GROUP,
                   kind=ChatKind.GROUP)
    res = adapter.react(chat, "m1", "🔥", key=KEY_GROUP)
    assert res.ok is True
    assert sent[0]["key"] == KEY_GROUP


# ── 3. the gateway threads the key, old adapters unaffected ──

def test_gateway_react_threads_key():
    from nomorals.social.chat.base import SendResult
    from nomorals.social.chat.gateway import ChatGateway

    calls = []

    class KeyAdapter:
        name = "whatsapp"

        def react(self, chat, message_id, emoji, key=None):
            calls.append((str(chat.chat_id), message_id, emoji,
                          key))
            return SendResult(ok=True, platform="whatsapp",
                              message_id=message_id)

    gw = ChatGateway(adapters={"whatsapp": KeyAdapter()})
    chat = ChatRef(platform="whatsapp", chat_id=GROUP,
                   kind=ChatKind.GROUP)
    res = gw.react("whatsapp", chat, "m1", "🔥", key=KEY_GROUP)
    assert res.ok is True
    assert calls == [(GROUP, "m1", "🔥", KEY_GROUP)]


def test_gateway_react_falls_back_for_three_arg_adapter():
    from nomorals.social.chat.base import SendResult
    from nomorals.social.chat.gateway import ChatGateway

    calls = []

    class OldAdapter:
        name = "telegram"

        def react(self, chat, message_id, emoji):
            calls.append((str(chat.chat_id), message_id, emoji))
            return SendResult(ok=True, platform="telegram",
                              message_id=message_id)

    gw = ChatGateway(adapters={"telegram": OldAdapter()})
    chat = ChatRef(platform="telegram", chat_id="777",
                   kind=ChatKind.DM)
    res = gw.react("telegram", chat, "m1", "🔥", key=KEY_GROUP)
    assert res.ok is True
    assert calls == [("777", "m1", "🔥")]


# ── 4. the organ carries the alt end to end ───────────────────

def _personal(tmp_path, **seams):
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault, **seams)
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    return personal, registry, vault, rec


def _rows(personal, rec_id, kind):
    return [r for r in personal.vault.journal(rec_id)
            if r["kind"] == kind]


def test_observe_carries_alt_to_view_react_and_trace(tmp_path):
    view_items = []
    react_items = []
    personal, registry, _vault, rec = _personal(
        tmp_path,
        status_viewer=lambda r, item: (
            view_items.append(dict(item)), True)[1],
        status_reactor=lambda r, item: (
            react_items.append(dict(item)), True)[1])
    registry.set_toggles(rec.id, status_viewing=True,
                         status_react=True)
    personal.observe_status_post(
        "whatsapp", poster_id=LID, poster_name="Poster",
        poster_alt_id=PN, source_id="POST1", ts=POST_TS)
    assert len(view_items) == 1
    assert view_items[0]["poster_id"] == LID
    assert view_items[0]["poster_alt_id"] == PN
    assert len(react_items) == 1
    assert react_items[0]["poster_id"] == LID
    assert react_items[0]["poster_alt_id"] == PN
    traces = _rows(personal, rec.id, "status_view_trace")
    assert len(traces) == 1
    assert traces[0]["seam_poster_id"] == LID
    assert traces[0]["seam_poster_alt_id"] == PN
    arrivals = _rows(personal, rec.id, "status_arrival")
    assert len(arrivals) == 1
    assert arrivals[0]["poster_alt_id"] == PN


def test_cycle_statuses_replays_alt_from_ledger(tmp_path):
    view_items = []
    personal, registry, _vault, rec = _personal(
        tmp_path,
        status_viewer=lambda r, item: (
            view_items.append(dict(item)), True)[1])
    # Observed while the toggle was OFF: the arrival (with its
    # alt) is ledgered, nothing attempted.  The post's ts is
    # NOW — the sweep only replays posts inside their 24h life,
    # and the shared POST_TS fixture is older than that.
    now = time.time()
    personal.observe_status_post(
        "whatsapp", poster_id=LID, poster_name="Poster",
        poster_alt_id=PN, source_id="POST1", ts=now)
    assert view_items == []
    assert len(_rows(personal, rec.id, "status_arrival")) == 1
    registry.set_toggles(rec.id, status_viewing=True)
    personal.cycle_statuses("whatsapp")
    assert len(view_items) == 1
    assert view_items[0]["poster_id"] == LID
    assert view_items[0]["poster_alt_id"] == PN


# ── 5. the composed seams pass the alt, old fakes survive ────

def _runtime(tmp_path, adapter, *, viewing=True):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    rt = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=viewing)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    rt._adapter_for = lambda platform: adapter
    svc = rt._pairing_personal()
    return rt, svc, registry.get(rec.id)


class _FakeViewAdapter:
    """The adapter's view_status, scripted — 4-arg shape."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def view_status(self, poster_id, message_id, ts=0.0,
                    poster_alt_id=""):
        self.calls.append((poster_id, message_id, ts,
                           poster_alt_id))
        return dict(self.verdict)


class _FakeViewAdapter3:
    """The adapter's view_status, old 3-arg shape."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def view_status(self, poster_id, message_id, ts=0.0):
        self.calls.append((poster_id, message_id, ts))
        return dict(self.verdict)


def test_status_viewer_passes_alt_to_adapter(tmp_path):
    adapter = _FakeViewAdapter({"ok": True,
                                "receipt_mode": "read"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    personal = svc["personal"]
    personal.observe_status_post(
        "whatsapp", poster_id=LID, poster_name="Poster",
        poster_alt_id=PN, source_id="POST1", ts=POST_TS)
    assert adapter.calls == [(LID, "POST1", POST_TS, PN)]
    traces = _rows(personal, rec.id, "status_view_trace")
    assert traces[0]["seam_poster_alt_id"] == PN


def test_status_viewer_falls_back_to_three_arg_adapter(tmp_path):
    adapter = _FakeViewAdapter3({"ok": True,
                                 "receipt_mode": "read"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    personal = svc["personal"]
    personal.observe_status_post(
        "whatsapp", poster_id=LID, poster_name="Poster",
        poster_alt_id=PN, source_id="POST1", ts=POST_TS)
    assert adapter.calls == [(LID, "POST1", POST_TS)]
    assert (rec.id, "POST1") in personal._viewed_status


class _FakeReactAdapter:
    """The adapter's react_status, scripted — 4-arg shape."""

    def __init__(self):
        self.calls = []

    def react_status(self, poster_id, message_id, emoji,
                     poster_alt_id=""):
        self.calls.append((poster_id, message_id, emoji,
                           poster_alt_id))
        return {"ok": True}


def test_status_reactor_passes_alt_to_adapter(tmp_path):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    rt = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_react=True)
    adapter = _FakeReactAdapter()
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    rt._adapter_for = lambda platform: adapter
    svc = rt._pairing_personal()
    personal = svc["personal"]
    personal.observe_status_post(
        "whatsapp", poster_id=LID, poster_name="Poster",
        poster_alt_id=PN, source_id="POST1", ts=POST_TS)
    assert adapter.calls == [(LID, "POST1", "❤️", PN)]


def test_personal_surface_turn_passes_alt_from_message_meta(
        tmp_path):
    # The runtime's personal-surface turn reads the alt off the
    # delivered message's meta and hands it to the organ — the
    # wire event's alt must survive the adapter -> runtime hop.
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = _FakeViewAdapter({"ok": True,
                                "receipt_mode": "read"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    rt._housekeeping_beat = lambda: None
    wa = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_status(wa, {"id": LID, "alt": PN,
                                "name": "Poster"})
    assert rt._personal_surface_turn(msg) is True
    assert adapter.calls == [(LID, "POST1", POST_TS, PN)]


# ── 6. the Baileys bridge source carries the fix ──────────────

def test_mjs_status_view_prefers_from_alt_and_maps_mode_three_way():
    src = (_REPO / "bridge" / "whatsapp-bridge.mjs").read_text(
        encoding="utf-8")
    # The status_view receipt's participant is the alt form when
    # the adapter sends one, else the poster's id as before.
    assert "req.from_alt || req.from" in src
    # Receipt mode matches Go's three-way: 'all' -> read,
    # 'none' -> read-self, anything else / fetch failure ->
    # unknown — never "everything else is receipts off".
    assert "readreceipts === 'all' ? 'read'" in src
    assert "readreceipts === 'none' ? 'read-self'" in src
    assert "readreceipts === 'all' ? 'read' : 'read-self'" \
        not in src


def test_mjs_status_post_emits_alt_and_react_prefers_supplied_key():
    src = (_REPO / "bridge" / "whatsapp-bridge.mjs").read_text(
        encoding="utf-8")
    # The status_post feed carries the poster's alternate form
    # alongside the id, like the Go bridge's from.alt.
    assert "participantAlt" in src
    # The chat react prefers a caller-supplied full key over
    # the bridge's own message cache.
    assert "req.key" in src


def test_go_status_verbs_prefer_from_alt_and_react_prefers_key():
    src = (_REPO / "bridge" / "gobridge" / "messages.go").read_text(
        encoding="utf-8")
    assert '"from_alt"' in src
    # cmdReact takes a caller-supplied full key before it ever
    # consults bridge memory.
    assert '"key"' in src
