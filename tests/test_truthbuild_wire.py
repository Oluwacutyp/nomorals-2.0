"""PERSONAL-SURFACE TRUTH BUILD (owner-adopted 2026-11-08).

Four repairs, each pinned at the seam where it used to lie:

1. THE STATUS WIRE.  The adapter's status_view / status_react
   sent the TARGET status id in the JSON field "id";
   ``_send_cmd`` writes ``{**cmd, "id": req_id}``, so the
   request correlation id overwrote the target on the wire and
   both bridges aimed every view/reaction at a random 12-char
   request id — and answered ok.  reject_call had the same
   shape with its call id.  The target now travels as
   "message_id" (a call target as "call_id") end-to-end:
   Python adapter, Node bridge, Go bridge.  These tests drive
   the REAL ``_send_cmd`` against a fake bridge SOCKET and
   assert on the actual JSON bytes — ``_send_cmd`` is never
   mocked here, because mocking it is exactly how the old
   tests stayed green while the feature was dead.

2. /vv REPLY TARGETING (command audit Finding E).  Reply
   resolution matched only source_id across the whole vault;
   message ids repeat across chats, so a reply in one chat
   could reveal (and name) another chat's capture.  A reply
   now resolves inside its own conversation.

3. /vv MEDIA KIND (Finding F).  The reveal passed no kind, so
   photos landed as documents.  The record's media_kind now
   rides the gateway call, and the gateway hands the adapter
   a MediaRef of that kind.

4. GROUP DIAGNOSTICS (Finding A).  /gsettings shows the claim
   facts, economy state, ledger health and the exact unmet
   condition — the platform claim difference is a rule, and
   the panel now says which part of it is unmet.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parent.parent

POSTER = "2347055554444@s.whatsapp.net"
OWNER_JID = "2348012345678@s.whatsapp.net"
REC_ID = "whatsapp:2348012345678"
CHAT_A = "2347111111111@s.whatsapp.net"
CHAT_B = "2347222222222@s.whatsapp.net"
CHAT_C = "2347333333333@s.whatsapp.net"


# ── 1. the wire, through the real _send_cmd ─────────────────


class _BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        self.wfile.write(json.dumps({
            "type": "status", "state": "open", "user": OWNER_JID,
            "user_name": "Owner"}).encode() + b"\n")
        self.wfile.flush()
        try:
            for raw in self.rfile:
                line = raw.strip()
                if not line:
                    continue
                with srv.lock:
                    srv.raw_lines.append(line)
                try:
                    cmd = json.loads(line)
                except ValueError:
                    continue
                cid = cmd.get("id")
                if not cid:
                    continue
                reply = {"id": cid, "ok": True}
                if cmd.get("cmd") == "status_view":
                    reply["receipt_mode"] = "read"
                self.wfile.write(
                    json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _FakeBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.raw_lines: list[bytes] = []
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def lines_for(self, cmd_name: str) -> list[dict]:
        with self.lock:
            raw = list(self.raw_lines)
        out = []
        for line in raw:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("cmd") == cmd_name:
                out.append(obj)
        return out

    def close(self):
        self.shutdown()
        self.server_close()


@pytest.fixture()
def wire(tmp_path):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    bridge = _FakeBridge()
    adapter = WhatsAppAdapter(
        port=bridge.port, outbox_enabled=False,
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))
    thread = threading.Thread(
        target=adapter.run, args=(lambda msg: None,), daemon=True)
    thread.start()
    assert adapter.wait_for_bridge(timeout=15)
    yield adapter, bridge
    adapter.stop()
    bridge.close()


def _assert_correlation(line: dict, target: str) -> None:
    """The wire's "id" is the correlation id — 12 hex chars,
    never the target — and the response round-tripped (the
    caller saw ok), which is what made the old shape so
    believable."""
    assert line["id"] != target
    assert len(line["id"]) == 12
    assert all(c in "0123456789abcdef" for c in line["id"])


def test_status_view_wire_carries_target_as_message_id(wire):
    adapter, bridge = wire
    result = adapter.view_status(POSTER, "STATUS-TARGET-1")
    assert result["ok"] is True
    lines = bridge.lines_for("status_view")
    assert len(lines) == 1
    line = lines[0]
    _assert_correlation(line, "STATUS-TARGET-1")
    assert line["message_id"] == "STATUS-TARGET-1"
    assert line["from"] == POSTER


def test_status_react_wire_carries_target_as_message_id(wire):
    adapter, bridge = wire
    result = adapter.react_status(POSTER, "STATUS-TARGET-2", "❤️")
    assert result["ok"] is True
    lines = bridge.lines_for("status_react")
    assert len(lines) == 1
    line = lines[0]
    _assert_correlation(line, "STATUS-TARGET-2")
    assert line["message_id"] == "STATUS-TARGET-2"
    assert line["from"] == POSTER
    assert line["emoji"] == "❤️"


def test_reject_call_wire_carries_call_id(wire):
    adapter, bridge = wire
    result = adapter.reject_call("CALL-ID-9", POSTER)
    assert result["ok"] is True
    lines = bridge.lines_for("reject_call")
    assert len(lines) == 1
    line = lines[0]
    _assert_correlation(line, "CALL-ID-9")
    assert line["call_id"] == "CALL-ID-9"
    assert line["from"] == POSTER


# ── 1b. the bridges read the new fields (source contracts) ──


def test_go_bridge_reads_targets_from_their_own_fields():
    text = (_REPO / "bridge" / "gobridge"
            / "messages.go").read_text(encoding="utf-8")
    # status_view + status_react read the target as message_id;
    # reject_call reads call_id; the ONLY remaining req.str("id")
    # is the dispatcher reading the correlation id itself.
    assert text.count('req.str("message_id")') >= 2
    assert text.count('req.str("call_id")') == 1
    assert text.count('req.str("id")') == 1


def test_node_bridge_reads_targets_from_their_own_fields():
    text = (_REPO / "bridge"
            / "whatsapp-bridge.mjs").read_text(encoding="utf-8")
    assert "req.message_id" in text
    assert "req.call_id" in text
    # No handler reads a TARGET out of the correlation field.
    assert "String(req.id)" not in text


def test_adapter_docstring_states_the_true_wire_shape():
    text = (_REPO / "nomorals" / "social" / "chat"
            / "whatsapp.py").read_text(encoding="utf-8")
    assert '"cmd":"status_view","message_id"' in text
    assert '"cmd":"reject_call","call_id"' in text


# ── 2+3. /vv reply scoping + media kind ─────────────────────


def _chatops_runtime(tmp_path, **toggles):
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )
    from nomorals.agents.partner.runtime_provision import (
        RuntimeProvisionMixin,
    )
    from nomorals.social.pairing import (
        AccountRecord,
        PairingRegistry,
    )

    class RT(RuntimeProvisionMixin, RuntimeChatOpsMixin):
        pass

    rt = RT.__new__(RT)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    if toggles:
        registry.set_toggles(rec.id, **toggles)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    svc = rt._pairing_personal()
    return rt, svc


def _media_file(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return str(path)


def _seed_two_chats_same_message_id(tmp_path, svc):
    """The Finding E shape: the same message id "7" captured
    in two different chats; chat B's is NEWER, so an unscoped
    newest-first lookup hands a reply in chat A the wrong
    chat's capture."""
    personal = svc["personal"]
    personal.observe_message(
        "whatsapp", chat_key=f"whatsapp:{CHAT_A}",
        counterpart_id=CHAT_A, sender_id=CHAT_A,
        sender_name="Ada", message_id="7", text="",
        media=[{"path": _media_file(tmp_path, "a.jpg"),
                "kind": "image"}],
        view_once=True, ts=time.time() - 100)
    personal.observe_message(
        "whatsapp", chat_key=f"whatsapp:{CHAT_B}",
        counterpart_id=CHAT_B, sender_id=CHAT_B,
        sender_name="Bola", message_id="7", text="",
        media=[{"path": _media_file(tmp_path, "b.jpg"),
                "kind": "image"}],
        view_once=True, ts=time.time())
    saved = [r for r in personal.vault.saved_entries()
             if r.get("kind") == "view_once"]
    assert len(saved) == 2
    return {r["counterpart_id"]: r for r in saved}


def _dm_message(chat_jid, reply_to=""):
    from nomorals.social.chat.base import (
        ChatKind,
        ChatMessage,
        ChatRef,
    )

    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=chat_jid,
                     kind=ChatKind.DM, peer=chat_jid),
        incoming=True, text="/vv", sender="Owner",
        sender_id=OWNER_JID, reply_to=reply_to,
        message_id=f"cmd-{time.time()}", ts=time.time())


def _recording_gateway(sink):
    def send_file(platform, chat, path, caption="", kind=""):
        sink.append({"platform": platform, "path": path,
                     "kind": kind})
        return SimpleNamespace(ok=True)

    return SimpleNamespace(send_file=send_file)


def test_vv_reply_resolves_its_own_chat_not_the_newer_one(tmp_path):
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    rows = _seed_two_chats_same_message_id(tmp_path, svc)
    sent: list = []
    rt.gateway = _recording_gateway(sent)
    reply = rt._chatops_viewonce(
        "vv", "", _dm_message(CHAT_A, reply_to="7"))
    assert "view-once from Ada" in reply
    assert [s["path"] for s in sent] == [rows[CHAT_A]["path"]]
    # Finding F: the record's kind reached the gateway.
    assert sent[0]["kind"] == "image"


def test_vv_reply_in_the_other_chat_resolves_that_chat(tmp_path):
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    rows = _seed_two_chats_same_message_id(tmp_path, svc)
    sent: list = []
    rt.gateway = _recording_gateway(sent)
    reply = rt._chatops_viewonce(
        "vv", "", _dm_message(CHAT_B, reply_to="7"))
    assert "view-once from Bola" in reply
    assert [s["path"] for s in sent] == [rows[CHAT_B]["path"]]


def test_vv_reply_in_a_chat_without_that_capture_explains(tmp_path):
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    _seed_two_chats_same_message_id(tmp_path, svc)
    sent: list = []
    rt.gateway = _recording_gateway(sent)
    reply = rt._chatops_viewonce(
        "vv", "", _dm_message(CHAT_C, reply_to="7"))
    assert "no view-once capture from this chat" in reply
    assert sent == []


def test_vv_browsing_forms_stay_intentionally_global(tmp_path):
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    rows = _seed_two_chats_same_message_id(tmp_path, svc)
    sent: list = []
    rt.gateway = _recording_gateway(sent)
    # Bare form from a third chat still opens the newest —
    # deliberate browsing was never the defect.
    reply = rt._chatops_viewonce("viewonce", "", _dm_message(CHAT_C))
    assert "view-once from Bola" in reply
    assert [s["path"] for s in sent] == [rows[CHAT_B]["path"]]
    assert sent[0]["kind"] == "image"


def test_gateway_hands_the_adapter_a_mediaref_of_the_kind(tmp_path):
    """The transport verb seam: through the REAL gateway, a
    declared image kind reaches the adapter's send_media as a
    MediaRef of kind "image" — not "file"/"document", which is
    what made Telegram answer sendDocument."""
    from nomorals.social.chat.base import ChatRef, SendResult
    from nomorals.social.chat.gateway import ChatGateway

    class RecordingAdapter:
        name = "telegram"

        def __init__(self):
            self.media: list = []

        def send_media(self, chat, media, caption=""):
            self.media.append(media)
            return SendResult(ok=True, platform="telegram",
                              message_id="m1")

    adapter = RecordingAdapter()
    gateway = ChatGateway({"telegram": adapter}, db=None)
    path = _media_file(tmp_path, "pic.jpg")
    chat = ChatRef(platform="telegram", chat_id="777")
    result = gateway.send_file("telegram", chat, path,
                               caption="hi", kind="image")
    assert result.ok
    assert [m.kind for m in adapter.media] == ["image"]


# ── 4. group diagnostics ────────────────────────────────────


def _surface(tmp_path):
    from nomorals.social.chat.group_surface import (
        GroupStore,
        GroupSurface,
    )

    return GroupSurface(
        GroupStore(tmp_path / "groups.json").load())


WA_KEY = "whatsapp:-100200"
TG_BOT_KEY = "telegram-bot:-100300"


def test_diagnostics_fresh_whatsapp_room_names_the_unmet(tmp_path):
    surface = _surface(tmp_path)
    diag = surface.system_diagnostics(WA_KEY)
    assert diag["allowed"] is False
    assert diag["designated"] is False
    assert diag["bot_platform"] is False
    assert diag["economy_enabled"] is True
    assert diag["ledger_frozen"] is None
    assert diag["ledger_entries"] == 0
    unmet = " ".join(diag["unmet"])
    assert "has not designated" in unmet
    assert "not a verified admin" in unmet


def test_diagnostics_telegram_bot_room_claims_itself(tmp_path):
    surface = _surface(tmp_path)
    diag = surface.system_diagnostics(TG_BOT_KEY)
    assert diag["bot_platform"] is True
    assert diag["allowed"] is True
    assert diag["unmet"] == []


def test_diagnostics_designated_room_reads_claimed(tmp_path):
    surface = _surface(tmp_path)
    surface.set_designated(WA_KEY, True, by="owner")
    diag = surface.system_diagnostics(WA_KEY)
    assert diag["allowed"] is True
    assert diag["source"] == "designated"
    assert diag["unmet"] == []


def test_diagnostics_corrupt_ledger_reports_frozen(tmp_path):
    surface = _surface(tmp_path)
    ledger = surface.economy._path(WA_KEY, "ledger")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{not a ledger", encoding="utf-8")
    diag = surface.system_diagnostics(WA_KEY)
    assert diag["ledger_frozen"]
    assert diag["ledger_entries"] is None


def _gsettings_panel(tmp_path, surface, key, platform):
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin,
    )
    from nomorals.social.chat.base import ChatKind, ChatRef

    class RT(RuntimeGroupMixin):
        def _group_admin_ok(self, message):
            return True

    rt = RT.__new__(RT)
    rt._group_surface_obj = surface
    chat = ChatRef(platform=platform,
                   chat_id=key.split(":", 1)[1],
                   kind=ChatKind.GROUP, title="Crew")
    message = SimpleNamespace(chat=chat, sender_id="99",
                              sender="Owner")
    out = rt._control_gsettings("", message)
    return str(getattr(out, "text", out))


def test_gsettings_panel_shows_the_diagnostics(tmp_path):
    surface = _surface(tmp_path)
    text = _gsettings_panel(tmp_path, surface, WA_KEY, "whatsapp")
    assert "System output" in text
    assert "system output: OFF" in text
    assert "claim facts:" in text
    assert "designated no" in text
    assert "ledger: healthy" in text
    assert "unmet:" in text
    assert "has not designated" in text


def test_gsettings_panel_designated_room_shows_on(tmp_path):
    surface = _surface(tmp_path)
    surface.set_designated(WA_KEY, True, by="owner")
    text = _gsettings_panel(tmp_path, surface, WA_KEY, "whatsapp")
    assert "system output: ON" in text
    assert "unmet:" not in text
