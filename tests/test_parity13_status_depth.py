"""PARITY #13 — WHATSAPP STATUS DEPTH (platform parity
census bundle #13, 2026-11-07).

The census graded both rows ABSENT: choosing the status
audience (the send payload carried text/media only — no
surface could read or set the account's status privacy
list) and seeing who viewed a status (no viewer-list verb
in either bridge).  This suite pins the build at every
seam:

  * bridges: `status_audience` / `status_audience_set` /
    `status_viewers` on BOTH WhatsApp engines.  The Go
    verbs are unit-pinned in bridge/gobridge/
    statusdepth_test.go (node shape, verdict
    normalization, read-back-wins, the bounded viewer
    memory); here the dispatch, the Node payload and the
    rebuilt binaries are pinned.  A write is proven by
    reading the list back — the fake bridge can trim the
    applied list, and the reply + journal must carry the
    READ-BACK values.
  * Go `status_send` refuses a per-post audience in words
    (whatsmeow resolves recipients from the account list);
    the Node rung resolves the account list when no
    explicit audience is given and refuses when it cannot
    be read — neither engine posts wider than asked.
  * adapter: shape rules run before traffic; viewers
    carry the honest coverage line; the base adapter
    reports the truth for surfaces without the verbs.
  * /statusaud + /statusviewers: the composed chain
    (owner command → runtime → identity home → real
    WhatsAppAdapter → recording fake bridge).  The show
    and the viewers are READS — the standing stop does
    not halt them; the SET is an act — the stop gates it
    and a landed change journals.  People resolve by
    exact name, unique substring or number; ambiguity
    answers with candidates and changes nothing.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

PORT = 19471
OWNER = "2348012345678"
ADA = "2348011111111@s.whatsapp.net"
BOLA = "2348022222222@s.whatsapp.net"
PN_POSTER = "2347099999999@s.whatsapp.net"
PN_TWO = "2347055555555@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
LID_VIEWER = "123456789012345@lid"
NEW_STATUS_ID = "STATUS-NEW-1"
ROOT = Path(__file__).resolve().parents[1]

TITLES = {
    ADA: "Ada Lovelace",
    BOLA: "Bola Ahmed",
    PN_POSTER: "Poster Pal",
    PN_TWO: "Poster Pal Two",
    GROUP: "Test Group",
}

VIEWER_ROWS = [
    {"jid": ADA, "ts": 1762390800000},
    {"jid": LID_VIEWER, "ts": 1762394400000},
]


def _default_audience():
    return {"mode": "only", "list": [ADA, BOLA]}


# ── recording fake bridge (the bridges' wire protocol) ─────

class _BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        self.wfile.write(json.dumps({
            "type": "status", "state": "open", "user": OWNER,
            "user_name": "Owner"}).encode() + b"\n")
        self.wfile.flush()
        with srv.lock:
            srv.clients.append(self.wfile)
        try:
            for raw in self.rfile:
                line = raw.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except ValueError:
                    continue
                with srv.lock:
                    srv.commands.append(cmd)
                cid = cmd.get("id")
                if not cid:
                    continue
                reply = {"id": cid, "ok": True}
                verb = cmd.get("cmd")
                if verb == "chats":
                    reply = {"id": cid, "ok": True, "chats": [
                        {"id": jid, "title": title,
                         "kind": ("group" if jid.endswith("@g.us")
                                  else "dm"),
                         "ts": 1762390800000}
                        for jid, title in TITLES.items()]}
                elif verb == "status_audience":
                    reply = {"id": cid, "ok": True,
                             "mode": srv.audience["mode"],
                             "list": list(srv.audience["list"]),
                             "list_known": True}
                elif verb == "status_audience_set":
                    applied = list(cmd.get("list") or [])
                    if srv.trim_one and applied:
                        applied = applied[:-1]
                    srv.audience = {
                        "mode": str(cmd.get("mode") or ""),
                        "list": applied}
                    reply = {"id": cid, "ok": True,
                             "mode": srv.audience["mode"],
                             "list": list(applied),
                             "list_known": True,
                             "written": True}
                elif verb == "status_viewers":
                    mid = str(cmd.get("message_id") or "")
                    rows = srv.viewers.get(mid, [])
                    reply = {"id": cid, "ok": True,
                             "message_id": mid,
                             "viewers": [dict(r) for r in rows],
                             "count": len(rows),
                             "posted_here": mid in srv.sent_ids}
                elif verb == "status_send":
                    if cmd.get("audience"):
                        reply = {"id": cid, "ok": False,
                                 "error": (
                                     "a per-post status "
                                     "audience can't be set "
                                     "on this engine — "
                                     "whatsmeow posts to "
                                     "your account's status "
                                     "privacy audience; "
                                     "change it with "
                                     "status_audience_set "
                                     "instead")}
                    else:
                        srv.sent_ids.add(NEW_STATUS_ID)
                        reply = {"id": cid, "ok": True,
                                 "message_id": NEW_STATUS_ID}
                self.wfile.write(
                    json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port=PORT):
        self.commands: list[dict] = []
        self.clients: list = []
        self.lock = threading.Lock()
        self.reset()
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    def reset(self):
        with self.lock:
            self.commands = []
        self.audience = _default_audience()
        self.viewers = {NEW_STATUS_ID: [dict(r)
                                        for r in VIEWER_ROWS]}
        self.sent_ids: set = set()
        self.trim_one = False

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands
                    if c.get("cmd") == name]

    def close(self):
        self.shutdown()
        self.server_close()


def wait_for(pred, timeout=25.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("parity13")
    bridge = FakeBridge()
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "partner.owner_sender_ids": OWNER,
        "chat.local_enabled": "true",
        "chat.whatsapp_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    ws = Path(settings.workspace_dir)
    rec = sup.registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp",
        identity=OWNER,
        creds_dir=str(ws / "data" / "whatsapp_sessions"
                      / f"wa-{OWNER}"),
        port=PORT, status="linked", created_at=1.0))

    def beat():
        runtime.on_message(ChatMessage(
            chat=ChatRef(platform="local", chat_id="console",
                         kind=ChatKind.DM, title="console",
                         peer="owner"),
            incoming=True, text="hello", sender="owner",
            sender_id="owner",
            message_id=f"beat-{time.time()}", ts=time.time()))

    beat()
    assert wait_for(
        lambda: runtime.gateway.adapters.get("whatsapp")
        is not None)
    adapter = runtime.gateway.adapters["whatsapp"]
    assert wait_for(lambda: adapter.connected.is_set())
    svc = runtime._pairing_personal()
    env = {"runtime": runtime, "bridge": bridge,
           "adapter": adapter, "sup": sup, "rec": rec,
           "svc": svc, "personal": svc["personal"],
           "vault": svc["personal"].vault,
           "tmp": tmp, "beat": beat}
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


@pytest.fixture(autouse=True)
def _reset_bridge(rig):
    rig["bridge"].reset()
    yield


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        kind=ChatKind.DM, title="", sender_id=None, mid=""):
    if sender_id is None:
        sender_id = OWNER
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=mid or f"op-{time.time()}",
        ts=time.time(), meta={"owner_command": True})
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_rows(rig, kind):
    return [r for r in rig["vault"].journal(rig["rec"].id)
            if r.get("kind") == kind]


# ── bridge source pins (both engines + the binaries) ───────

def test_go_bridge_carries_the_status_depth_verbs():
    src = (ROOT / "bridge" / "gobridge"
           / "messages.go").read_text()
    for verb in ("status_audience", "status_audience_set",
                 "status_viewers"):
        assert f'case "{verb}"' in src, verb
    # the write rides the status-namespace IQ through the
    # dangerous internals, and every write reads back
    assert "statusPrivacyWire" in src
    assert 'DangerousInfoQueryType("set")' in src
    assert "GetStatusPrivacy" in src
    # the per-post refusal and the receipt-fed viewer memory
    assert "a per-post status audience can't be set" in src
    assert "noteStatusView" in src
    assert "types.StatusBroadcastJID" in src


def test_baileys_bridge_carries_the_status_depth_verbs():
    src = (ROOT / "bridge"
           / "whatsapp-bridge.mjs").read_text()
    for verb in ("status_audience", "status_audience_set",
                 "status_viewers"):
        assert f"case '{verb}'" in src, verb
    # the same status-namespace wire via sock.query
    assert "statusPrivacyGet" in src
    assert "xmlns: 'status'" in src
    # viewers come from the receipt handler's participant
    assert "noteStatusView" in src
    assert "key?.participant" in src
    # an unreadable account list refuses the post
    assert "could not read your status audience setting" in src


def test_rebuilt_binaries_carry_the_verbs():
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"status_audience_set" in data, name
        assert b"status_viewers" in data, name
        assert b"a per-post status audience can't be set" in data, name


def test_catalog_registers_the_family():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    flat_groups = [k for _name, keys in LIST_GROUPS
                   for k in keys]
    for kind in ("statusaud", "statusviewers"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        # owner-only acts on his own account: never public
        assert kind not in PUBLIC_COMMANDS
        assert kind in flat_groups
    runtime_src = (ROOT / "nomorals" / "agents" / "partner"
                   / "runtime.py").read_text()
    assert '"statusaud", "statusviewers"' in runtime_src


# ── parser + formatters (pure) ─────────────────────────────

def test_parse_statusaud_args():
    from nomorals.social.chat.chatops import parse_statusaud_args

    assert parse_statusaud_args("") == {
        "ok": True, "action": "show"}
    assert parse_statusaud_args("contacts") == {
        "ok": True, "action": "set", "mode": "contacts",
        "people": []}
    got = parse_statusaud_args("only Ada Lovelace, +234801")
    assert got["ok"] and got["mode"] == "only"
    assert got["people"] == ["Ada Lovelace", "+234801"]
    assert not parse_statusaud_args("everyone")["ok"]
    assert not parse_statusaud_args("only")["ok"]
    assert not parse_statusaud_args("contacts Ada")["ok"]


def test_format_status_audience_renders_people_not_ids():
    from nomorals.social.chat.chatops import format_status_audience

    names = {ADA: "Ada Lovelace", BOLA: "Bola Ahmed"}
    out = format_status_audience(
        {"ok": True, "mode": "only", "list": [ADA, BOLA],
         "list_known": True}, names=names)
    assert "shared only with" in out
    assert "Ada Lovelace" in out and "Bola Ahmed" in out
    assert "@s.whatsapp.net" not in out
    # an unnamed person renders as their number
    out = format_status_audience(
        {"ok": True, "mode": "except", "list": [ADA],
         "list_known": True}, names={})
    assert "+2348011111111" in out
    # an unknown list is worded, never printed as empty
    out = format_status_audience(
        {"ok": True, "mode": "only", "list": [],
         "list_known": False}, names={})
    assert "isn't readable" in out
    # a refusal passes its reason through
    out = format_status_audience(
        {"ok": False, "error": "bridge is down"})
    assert "bridge is down" in out


def test_format_status_viewers_words_zero_and_hides_lids():
    from nomorals.social.chat.chatops import format_status_viewers

    out = format_status_viewers(
        {"ok": True, "viewers": [], "count": 0,
         "coverage": "views seen since it connected"})
    assert "no views seen" in out
    assert "nobody" not in out
    out = format_status_viewers(
        {"ok": True, "viewers": VIEWER_ROWS, "count": 2,
         "message_id": NEW_STATUS_ID,
         "coverage": "views seen since it connected"},
        names={ADA: "Ada Lovelace"})
    assert "2 people have viewed" in out
    assert "Ada Lovelace" in out
    # the LID viewer is worded, never dumped raw
    assert "@lid" not in out
    assert "123456789012345" not in out
    assert "whose name isn't known" in out
    # the coverage line prints, the message id never does
    assert "(counting views seen since it connected)" in out
    assert NEW_STATUS_ID not in out


# ── adapter seams (real adapter → recording fake bridge) ───

def test_adapter_audience_read_passes_the_verdict(rig):
    verdict = rig["adapter"].status_audience()
    assert verdict["ok"] and verdict["mode"] == "only"
    assert verdict["list"] == [ADA, BOLA]
    assert rig["bridge"].cmds("status_audience")


def test_adapter_set_shape_refuses_before_traffic(rig):
    adapter = rig["adapter"]
    out = adapter.set_status_audience("contacts", [ADA])
    assert not out["ok"] and "takes no list" in out["error"]
    out = adapter.set_status_audience("only", [])
    assert not out["ok"]
    assert "needs at least one person" in out["error"]
    out = adapter.set_status_audience("only", [GROUP])
    assert not out["ok"] and "holds people, not" in out["error"]
    out = adapter.set_status_audience("everyone", [])
    assert not out["ok"]
    assert "contacts, except or only" in out["error"]
    # none of it reached the bridge
    assert rig["bridge"].cmds("status_audience_set") == []


def test_adapter_set_payload_and_readback(rig):
    verdict = rig["adapter"].set_status_audience(
        "except", [ADA])
    assert verdict["ok"] and verdict["mode"] == "except"
    sent = rig["bridge"].cmds("status_audience_set")
    assert len(sent) == 1
    assert sent[0]["mode"] == "except"
    assert sent[0]["list"] == [ADA]


def test_adapter_viewers_payload_and_coverage(rig):
    out = rig["adapter"].status_viewers("")
    assert not out["ok"]
    assert rig["bridge"].cmds("status_viewers") == []
    verdict = rig["adapter"].status_viewers(NEW_STATUS_ID)
    assert verdict["ok"] and verdict["count"] == 2
    assert verdict["posted_here"] is False
    assert "no server-side" in verdict["coverage"]
    sent = rig["bridge"].cmds("status_viewers")
    assert sent[0]["message_id"] == NEW_STATUS_ID


def test_adapter_status_send_audience_refusal_surfaces(rig):
    # the Go engine's honesty: a per-post audience list is
    # refused in words, never silently posted wider
    verdict = rig["adapter"].send_status(
        "hello", audience=[ADA])
    assert not verdict["ok"]
    assert "per-post status audience" in verdict["error"]
    sent = rig["bridge"].cmds("status_send")
    assert sent and sent[0]["audience"] == [ADA]


def test_base_adapter_reports_the_truth():
    from nomorals.social.chat.local import LocalAdapter

    adapter = LocalAdapter()
    for verdict in (adapter.status_audience(),
                    adapter.set_status_audience("only", [ADA]),
                    adapter.status_viewers(NEW_STATUS_ID)):
        assert not verdict["ok"]
        assert "not supported on this adapter" in verdict["error"]


# ── composed chain: /statusaud + /statusviewers ────────────

def test_statusviewers_without_a_post_says_so(rig):
    # runs before any /poststatus in this module: the
    # journal holds no status of the owner's to check
    out = _op(rig, "/statusviewers")
    assert "don't have a status post" in out
    assert rig["bridge"].cmds("status_viewers") == []


def test_statusaud_show_renders_names(rig):
    out = _op(rig, "/statusaud")
    assert "shared only with" in out
    assert "Ada Lovelace" in out and "Bola Ahmed" in out
    assert "@s.whatsapp.net" not in out


def test_statusaud_set_by_exact_name_journals_readback(rig):
    out = _op(rig, "/statusaud only Poster Pal")
    assert "set:" in out and "Poster Pal" in out
    sent = rig["bridge"].cmds("status_audience_set")
    assert len(sent) == 1
    assert sent[0]["mode"] == "only"
    assert sent[0]["list"] == [PN_POSTER]
    rows = _journal_rows(rig, "status_audience_set")
    assert len(rows) == 1
    assert rows[0]["mode"] == "only"
    assert rows[0]["audience"] == [PN_POSTER]


def test_statusaud_set_reports_the_readback_not_the_ask(rig):
    # the server trims the applied list: the reply and the
    # journal must carry what the account confirmed
    rig["bridge"].trim_one = True
    out = _op(rig, "/statusaud only Ada Lovelace, Bola Ahmed")
    assert "Ada Lovelace" in out
    assert "Bola Ahmed" not in out
    rows = _journal_rows(rig, "status_audience_set")
    assert rows[-1]["audience"] == [ADA]


def test_statusaud_set_by_number(rig):
    out = _op(rig, "/statusaud except +234 811 222 3333")
    assert "except:" in out
    assert "+2348112223333" in out
    sent = rig["bridge"].cmds("status_audience_set")
    assert sent[0]["list"] == ["2348112223333@s.whatsapp.net"]


def test_statusaud_ambiguous_name_changes_nothing(rig):
    out = _op(rig, "/statusaud only Poster")
    assert "more than one" in out
    assert "Poster Pal Two" in out
    assert "not changed" in out
    assert rig["bridge"].cmds("status_audience_set") == []


def test_statusaud_unknown_name_changes_nothing(rig):
    out = _op(rig, "/statusaud only Nobody Here")
    assert "can't find" in out and "not changed" in out
    assert rig["bridge"].cmds("status_audience_set") == []


def test_statusaud_contacts_clears_the_list(rig):
    out = _op(rig, "/statusaud contacts")
    assert "all your contacts" in out
    sent = rig["bridge"].cmds("status_audience_set")
    assert sent[0]["mode"] == "contacts"
    assert sent[0].get("list") in (None, [])


def test_poststatus_then_statusviewers_default_target(rig):
    out = _op(rig, "/poststatus building while others flex")
    assert rig["bridge"].cmds("status_send")
    out = _op(rig, "/statusviewers")
    assert "2 people have viewed" in out
    assert "Ada Lovelace" in out
    assert "@lid" not in out
    assert "(counting views this bridge has seen" in out
    sent = rig["bridge"].cmds("status_viewers")
    assert sent[-1]["message_id"] == NEW_STATUS_ID


def test_status_family_honors_the_owner_stop(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        # the SET is an act: gated, nothing sent, no journal
        before = len(_journal_rows(rig, "status_audience_set"))
        out = _op(rig, "/statusaud only Poster Pal")
        assert "standing stop" in out
        assert "not changed" in out
        assert rig["bridge"].cmds("status_audience_set") == []
        assert len(_journal_rows(
            rig, "status_audience_set")) == before
        # the READS answer under the same stop (the /find
        # doctrine: the stop gates acts, not reads)
        out = _op(rig, "/statusaud")
        assert "shared only with" in out
        out = _op(rig, f"/statusviewers {NEW_STATUS_ID}")
        assert "2 people have viewed" in out
    finally:
        rig["sup"].registry.set_owner_stopped(
            "whatsapp", False)


def test_statusaud_on_telegram_answers_platform_truth(rig):
    # The handlers' own platform gate, driven directly: a
    # Telegram surface (bot or personal) gets the platform
    # truth in words, and nothing reaches the bridge.
    # (Through handle_control a telegram-bot chat is the
    # public persona's floor, not the owner's control
    # surface, so the composed pin lives at this seam —
    # the same reason #11 pins the bot adapter directly.)
    chat = ChatRef(platform="telegram-bot", chat_id="4242",
                   kind=ChatKind.DM, title="", peer="4242")
    msg = ChatMessage(
        chat=chat, incoming=True, text="/statusaud",
        sender="Owner", sender_id=OWNER,
        message_id="tg-1", ts=time.time(),
        meta={"owner_command": True})
    runtime = rig["runtime"]
    out = runtime._chatops_statusaud("", msg)
    assert "WhatsApp shape" in out
    out = runtime._chatops_statusviewers("", msg)
    assert "WhatsApp shape" in out
    assert rig["bridge"].cmds("status_audience") == []
    assert rig["bridge"].cmds("status_viewers") == []
