"""PARITY G3 — WhatsApp OWN-IDENTITY + POLL SENDING (platform
parity census bundles #5 and #6, 2026-11-07).

The census graded both ABSENT: the account's push name could
be read off inbound messages but nothing on the account could
be changed (no bridge verb existed on either engine for the
profile name, the About text, the profile photo, or any
privacy setting), and zero poll-send code existed anywhere —
which is also why /pollres sat UNPROVEN since the truth pass:
nothing could produce a poll for it to read.

This suite drives the real composed chain — owner command →
handle_control → runtime_chatops → the real WhatsAppAdapter →
a recording fake bridge speaking the bridges' wire protocol —
plus in-bridge pins, and a capability counts only when its
verb FIRES with the right payload (the truth-pass standard):

  * /myprofile: profile_get renders the bridges' read (name,
    about, photo presence — an unreadable about is NAMED, not
    invented); name/about/photo setters fire profile_set with
    the exact field and value, journal profile_changed, and
    the owner's standing stop gates them.
  * /privacy: the read renders per-category plain words and
    names unreadable categories; a set fires privacy_set with
    the exact (category, value) pair and journals; an
    out-of-vocabulary pair refuses in words BEFORE traffic
    (the protocol's own vocabulary, shared across the command
    layer, the adapter and both bridges).
  * /makepoll: send_poll fires with the exact chat, question,
    options and selectable count (1, or len(options) for
    'multi'); bad shapes refuse before traffic; a landed poll
    journals poll_posted with its message id.
  * /pollres — the truth-pass remainder CLOSES here: the fake
    bridge tallies votes the way the real bridges do (bridge
    memory over decrypted votes), so a poll produced by THIS
    send path reads back with real counts, a re-vote replaces,
    and a poll outside bridge memory refuses in words instead
    of inventing numbers.

Bridge internals are pinned where they live: the Go verbs'
call shapes have their own Go tests (bridge/gobridge/
profile_test.go — whatsmeow BuildSettingPushName/SetStatus
Message/SetGroupPhoto/SetPrivacySetting/BuildPollCreation/
DecryptPollVote), the Node verbs are source-pinned against
Baileys' exported socket surface (updateProfileName /
updateProfileStatus / updateProfilePicture / the per-category
privacy updaters / sendMessage {poll}), and the rebuilt Go
binaries carry the new verbs' refusal strings.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

PORT = 19442
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
ROOT = Path(__file__).resolve().parents[1]

PROFILE_READ = {
    "name": "Bridge Owner",
    "about": "building while others flex",
    "picture_id": "pic-9",
    "has_photo": True,
}
PRIVACY_READ = {
    "groupadd": "all",
    "lastseen": "contacts",
    "status": "all",
    "profile": "contacts",
    "readreceipts": "all",
    "online": "all",
    "calladd": "known",
}


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
                if verb == "send":
                    reply["message_id"] = "WAMID-TEST-1"
                elif verb == "profile_get":
                    reply.update(PROFILE_READ)
                    if srv.profile_info_error:
                        reply.pop("about")
                        reply.pop("picture_id")
                        reply["info_error"] = "usync refused"
                elif verb == "profile_set":
                    reply["field"] = cmd.get("field")
                    if cmd.get("field") == "name":
                        reply["name"] = cmd.get("value")
                    elif cmd.get("field") == "about":
                        reply["about"] = cmd.get("value")
                    elif cmd.get("field") == "photo":
                        if cmd.get("remove"):
                            reply["removed"] = True
                        else:
                            reply["picture_id"] = "pic-10"
                elif verb == "privacy_get":
                    reply["settings"] = dict(srv.privacy)
                elif verb == "privacy_set":
                    srv.privacy[cmd.get("category")] = \
                        cmd.get("value")
                    reply["category"] = cmd.get("category")
                    reply["value"] = cmd.get("value")
                    reply["settings"] = dict(srv.privacy)
                elif verb == "send_poll":
                    srv.polls[cmd.get("chat")] = {
                        "question": cmd.get("question"),
                        "options": list(cmd.get("options") or []),
                        "votes": {},
                    }
                    reply["message_id"] = srv.poll_id
                elif verb == "poll_results":
                    poll = None
                    for chat, state in srv.polls.items():
                        if chat == cmd.get("chat"):
                            poll = state
                    if poll is None:
                        reply = {
                            "id": cid, "ok": False,
                            "error": "that poll is not in bridge "
                                     "memory — results live on "
                                     "polls this bridge has seen "
                                     "since it started"}
                    else:
                        counts = [0] * len(poll["options"])
                        for selected in poll["votes"].values():
                            for name in selected:
                                counts[poll["options"].index(name)] += 1
                        reply["question"] = poll["question"]
                        reply["options"] = [
                            {"text": name, "voters": counts[i]}
                            for i, name in
                            enumerate(poll["options"])]
                        reply["total_voters"] = len(poll["votes"])
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
        self.privacy = dict(PRIVACY_READ)
        self.polls: dict[str, dict] = {}
        self.poll_id = "POLL-1"
        self.profile_info_error = False
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands
                    if c.get("cmd") == name]

    def cast_vote(self, chat, voter, *option_names):
        """One decrypted vote landing in bridge memory — the
        fake's model of the real bridges' decrypt-and-tally."""
        with self.lock:
            self.polls[chat]["votes"][voter] = list(option_names)

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

    tmp = tmp_path_factory.mktemp("parityg3")
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
    env = {
        "runtime": runtime, "bridge": bridge, "adapter": adapter,
        "sup": sup, "rec": rec, "svc": svc,
        "personal": svc["personal"], "vault": svc["personal"].vault,
        "tmp": tmp, "beat": beat,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        kind=ChatKind.DM, title="", sender_id=None, mid="",
        reply_to="", media=None, meta_extra=None):
    if sender_id is None:
        sender_id = OWNER
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    meta = {"owner_command": True}
    if meta_extra:
        meta.update(meta_extra)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=mid or f"op-{time.time()}",
        ts=time.time(), meta=meta, reply_to=reply_to,
        media=list(media or []))
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_rows(rig, account_id):
    return rig["vault"].journal(account_id)


# ── bridge source pins (both engines + the rebuilt binaries) ──

def test_go_bridge_identity_and_poll_verbs():
    src = (ROOT / "bridge" / "gobridge"
           / "profile.go").read_text()
    # identity: the whatsmeow calls that actually exist
    assert "BuildSettingPushName" in src
    assert "SetStatusMessage" in src
    assert "SetGroupPhoto" in src  # own-JID profile picture IQ
    assert "TryFetchPrivacySettings" in src
    assert "SetPrivacySetting" in src
    # polls: creation + vote decryption + bridge-memory tally
    assert "BuildPollCreation" in src
    assert "DecryptPollVote" in src
    assert "PollUpdateMessage" in src
    assert "sha256" in src  # option identity is its hash
    dispatch = (ROOT / "bridge" / "gobridge"
                / "messages.go").read_text()
    for verb in ("profile_get", "profile_set", "privacy_get",
                 "privacy_set", "send_poll", "poll_results"):
        assert f'case "{verb}"' in dispatch, verb


def test_baileys_identity_and_poll_verbs():
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs").read_text()
    assert "updateProfileName" in src
    assert "updateProfileStatus" in src
    assert "updateProfilePicture" in src
    assert "removeProfilePicture" in src
    assert "fetchPrivacySettings" in src
    assert "updateLastSeenPrivacy" in src
    assert "updateReadReceiptsPrivacy" in src
    assert "updateGroupsAddPrivacy" in src
    # polls: the {poll} send payload + vote capture/decryption
    assert "selectableCount" in src
    assert "pollUpdateMessage" in src
    assert "capturePollMessage" in src
    assert "aes-256-gcm" in src  # the decryptPollVote mirror
    assert "messageSecret" in src


def test_rebuilt_binaries_carry_the_verbs():
    # Only strings unique to this build's code are pinned —
    # switch-case names do not survive contiguously in a Go
    # binary (see the G2 suite's note).
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"a WhatsApp poll needs 2" in data, name
        assert b"that poll is not in bridge memory" in data, name
        assert b"WhatsApp profile photos must be" in data, name


def test_catalog_registers_the_family():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
    )

    for kind in ("myprofile", "privacy", "makepoll"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert any(kind in names for _t, names in LIST_GROUPS)
    # /poll stays the group-economy poll — makepoll is distinct
    assert "poll" in CONTROL_COMMANDS
    # the pollres details no longer claim WhatsApp can't answer
    assert "exposes none to the bridge" not in \
        COMMAND_DETAILS["pollres"]["what"]


# ── adapter-level pins (payloads, refusals before traffic) ──

@pytest.fixture()
def adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19999)
    calls: list[tuple] = []

    def fake_admin(cmd, payload, *, timeout=30.0):
        calls.append((cmd, payload))
        if cmd == "send_poll":
            return {"ok": True, "message_id": "POLL-9"}
        return {"ok": True}

    monkeypatch.setattr(ad, "_admin_cmd", fake_admin)
    ad.calls = calls  # type: ignore[attr-defined]
    return ad


def _chat(chat_id=GROUP):
    return ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=ChatKind.GROUP, peer=chat_id)


def test_adapter_send_poll_payload(adapter):
    res = adapter.send_poll(_chat(), "Lunch?",
                            ["Pizza", "Sushi"])
    assert res.ok and res.message_id == "POLL-9"
    assert adapter.calls == [("send_poll", {
        "chat": GROUP, "question": "Lunch?",
        "options": ["Pizza", "Sushi"], "selectable": 1})]
    res = adapter.send_poll(_chat(), "Pick any",
                            ["A", "B", "C"], multiple=True)
    assert res.ok
    assert adapter.calls[-1][1]["selectable"] == 3


def test_adapter_send_poll_refusals_before_traffic(adapter):
    # a quiz cannot ride WhatsApp polls — refusing in words
    # beats silently posting an opinion poll instead
    res = adapter.send_poll(_chat(), "2+2?", ["3", "4"],
                            quiz=True, correct_option=1)
    assert not res.ok and "no quiz mode" in res.error
    res = adapter.send_poll(_chat(), "Q?", ["only"])
    assert not res.ok and "2–12" in res.error
    res = adapter.send_poll(_chat(), "", ["a", "b"])
    assert not res.ok
    res = adapter.send_poll(_chat(), "Q?",
                            [f"opt-{i}" for i in range(13)])
    assert not res.ok
    assert adapter.calls == []  # nothing reached the bridge


def test_adapter_identity_payloads(adapter):
    adapter.set_profile_name("  New Name  ")
    assert adapter.calls[-1] == (
        "profile_set", {"field": "name", "value": "New Name"})
    adapter.set_about("hello world")
    assert adapter.calls[-1] == (
        "profile_set", {"field": "about", "value": "hello world"})
    adapter.set_profile_photo("/vault/pic.jpg")
    assert adapter.calls[-1] == (
        "profile_set", {"field": "photo", "path": "/vault/pic.jpg"})
    adapter.set_profile_photo(remove=True)
    assert adapter.calls[-1] == (
        "profile_set", {"field": "photo", "remove": True})
    adapter.set_privacy("LastSeen", "Contacts")
    assert adapter.calls[-1] == (
        "privacy_set", {"category": "lastseen", "value": "contacts"})
    adapter.get_profile()
    assert adapter.calls[-1] == ("profile_get", {})
    adapter.get_privacy()
    assert adapter.calls[-1] == ("privacy_get", {})
    adapter.poll_results(_chat(), "POLL-9")
    assert adapter.calls[-1] == (
        "poll_results", {"chat": GROUP, "message_id": "POLL-9"})
    # empties refuse before traffic
    before = len(adapter.calls)
    assert adapter.set_profile_name("  ")["ok"] is False
    assert adapter.set_profile_photo("")["ok"] is False
    assert adapter.poll_results(_chat(), "")["ok"] is False
    assert len(adapter.calls) == before


def test_privacy_vocabulary_matches_the_protocol():
    from nomorals.social.chat.chatops import PRIVACY_CATEGORIES, privacy_pair_error

    # the values whatsmeow documents per category
    assert privacy_pair_error("readreceipts", "all") == ""
    assert privacy_pair_error("readreceipts", "contacts") != ""
    assert privacy_pair_error("online", "match_last_seen") == ""
    assert privacy_pair_error("online", "none") != ""
    assert privacy_pair_error("calladd", "known") == ""
    assert privacy_pair_error("defense", "on_standard") == ""
    assert privacy_pair_error("stickers", "contact_allowlist") == ""
    assert privacy_pair_error("bogus", "all") != ""
    assert set(PRIVACY_CATEGORIES) == {
        "groupadd", "lastseen", "status", "profile",
        "readreceipts", "online", "calladd", "messages",
        "defense", "stickers"}
    # the Go bridge carries the same category names (the wire
    # vocabulary is one vocabulary)
    src = (ROOT / "bridge" / "gobridge"
           / "profile.go").read_text()
    for category in PRIVACY_CATEGORIES:
        assert f'"{category}"' in src, category


# ── /profile composed ──────────────────────────────────────

def test_profile_read_renders_bridge_truth(rig):
    out = _op(rig, "/myprofile")
    assert "Bridge Owner" in out
    assert "building while others flex" in out
    assert "photo: set" in out
    assert rig["bridge"].cmds("profile_get")


def test_profile_read_names_unreadable_about(rig):
    rig["bridge"].profile_info_error = True
    try:
        out = _op(rig, "/myprofile")
    finally:
        rig["bridge"].profile_info_error = False
    assert "couldn't read it just now" in out
    assert "usync refused" in out


def test_profile_name_set_fires_and_journals(rig):
    out = _op(rig, "/myprofile name Devon Prime")
    assert "Devon Prime" in out
    sets = [c for c in rig["bridge"].cmds("profile_set")
            if c.get("field") == "name"]
    assert sets and sets[-1]["value"] == "Devon Prime"
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "name"]
    assert rows and rows[-1]["text"] == "Devon Prime"


def test_profile_about_set_and_clear(rig):
    out = _op(rig, "/myprofile about shipping daily")
    assert "shipping daily" in out
    sets = [c for c in rig["bridge"].cmds("profile_set")
            if c.get("field") == "about"]
    assert sets[-1]["value"] == "shipping daily"
    out = _op(rig, "/myprofile about clear")
    assert "cleared" in out
    sets = [c for c in rig["bridge"].cmds("profile_set")
            if c.get("field") == "about"]
    assert sets[-1]["value"] == ""


def test_profile_photo_from_replied_media(rig, tmp_path):
    pic = tmp_path / "me.jpg"
    pic.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 32)
    out = _op(rig, "/myprofile photo", meta_extra={
        "replied_media_local": {"path": str(pic),
                                "kind": "photo"}})
    assert "photo is up" in out
    sets = [c for c in rig["bridge"].cmds("profile_set")
            if c.get("field") == "photo"]
    assert sets[-1]["path"] == str(pic)
    out = _op(rig, "/myprofile photo remove")
    assert "removed" in out
    sets = [c for c in rig["bridge"].cmds("profile_set")
            if c.get("field") == "photo"]
    assert sets[-1].get("remove") is True


def test_profile_photo_without_media_refuses_in_words(rig):
    before = len(rig["bridge"].cmds("profile_set"))
    out = _op(rig, "/myprofile photo")
    assert "reply to a picture" in out
    assert len(rig["bridge"].cmds("profile_set")) == before


# ── /privacy composed ──────────────────────────────────────

def test_privacy_read_plain_words(rig):
    out = _op(rig, "/privacy")
    assert "lastseen: my contacts (contacts)" in out
    assert "readreceipts: everyone (all)" in out
    assert "defense: couldn't read" in out  # unread, named as such
    assert rig["bridge"].cmds("privacy_get")


def test_privacy_set_fires_and_journals(rig):
    out = _op(rig, "/privacy lastseen none")
    assert "lastseen is now nobody" in out
    sets = rig["bridge"].cmds("privacy_set")
    assert sets[-1]["category"] == "lastseen"
    assert sets[-1]["value"] == "none"
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "privacy:lastseen"]
    assert rows and rows[-1]["text"] == "none"


def test_privacy_bad_pair_refuses_before_traffic(rig):
    before = len(rig["bridge"].cmds("privacy_set"))
    out = _op(rig, "/privacy readreceipts contacts")
    assert "readreceipts can be: all, none" in out
    out = _op(rig, "/privacy nonsense all")
    assert "privacy categories are:" in out
    assert len(rig["bridge"].cmds("privacy_set")) == before


def test_identity_setters_honor_owner_stop(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(rig["bridge"].cmds("profile_set"))
        out = _op(rig, "/myprofile name Halted")
        assert "standing stop" in out
        out = _op(rig, "/privacy lastseen all")
        assert "standing stop" in out
        assert len(rig["bridge"].cmds("profile_set")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)


# ── /makepoll + /pollres composed (bundle #6 + #16 closes) ─

def test_makepoll_fires_and_journals(rig):
    out = _op(rig, "/makepoll Lunch today? | Pizza | Sushi",
              chat_id=GROUP, kind=ChatKind.GROUP, title="Gold Room")
    assert "poll posted" in out and "Lunch today?" in out
    sends = rig["bridge"].cmds("send_poll")
    assert sends[-1]["chat"] == GROUP
    assert sends[-1]["question"] == "Lunch today?"
    assert sends[-1]["options"] == ["Pizza", "Sushi"]
    assert sends[-1]["selectable"] == 1
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "poll_posted"]
    assert rows and rows[-1]["source_id"] == "POLL-1"
    assert rows[-1]["counterpart_id"] == f"whatsapp:{GROUP}"


def test_makepoll_multi_selectable(rig):
    _op(rig, "/makepoll multi | Toppings? | Cheese | Olives | Tuna",
        chat_id=GROUP, kind=ChatKind.GROUP, title="Gold Room")
    sends = rig["bridge"].cmds("send_poll")
    assert sends[-1]["selectable"] == 3
    assert sends[-1]["question"] == "Toppings?"


def test_makepoll_bad_shape_refuses_before_traffic(rig):
    before = len(rig["bridge"].cmds("send_poll"))
    out = _op(rig, "/makepoll Lonely? | Only", chat_id=GROUP,
              kind=ChatKind.GROUP, title="Gold Room")
    assert "at least two options" in out
    assert len(rig["bridge"].cmds("send_poll")) == before


def test_pollres_reads_back_a_poll_this_path_posted(rig):
    # a fresh poll through the real send path…
    _op(rig, "/makepoll Dinner? | Pizza | Sushi",
        chat_id=GROUP, kind=ChatKind.GROUP, title="Gold Room")
    bridge = rig["bridge"]
    # …votes land the way the real bridges land them (decrypted
    # into bridge memory), including a CHANGED vote
    bridge.cast_vote(GROUP, "voter-a", "Pizza")
    bridge.cast_vote(GROUP, "voter-b", "Pizza")
    bridge.cast_vote(GROUP, "voter-c", "Sushi")
    bridge.cast_vote(GROUP, "voter-c", "Pizza")  # changed mind
    out = _op(rig, "/pollres", chat_id=GROUP,
              kind=ChatKind.GROUP, title="Gold Room",
              reply_to="POLL-1")
    assert "Dinner?" in out
    assert "Pizza — 3 votes" in out
    assert "Sushi — 0 votes" in out
    assert "3 voters in total" in out or "3 votes in total" in out \
        or "Total: 3" in out
    asks = rig["bridge"].cmds("poll_results")
    assert asks[-1]["chat"] == GROUP
    assert asks[-1]["message_id"] == "POLL-1"


def test_pollres_unknown_poll_refuses_in_words(rig):
    out = _op(rig, "/pollres", chat_id="120363009999999999@g.us",
              kind=ChatKind.GROUP, title="Elsewhere",
              reply_to="POLL-NEVER-SEEN")
    assert "not in bridge memory" in out


def test_pollres_needs_a_reply_target(rig):
    out = _op(rig, "/pollres", chat_id=GROUP,
              kind=ChatKind.GROUP, title="Gold Room")
    assert "reply to the poll" in out


def test_makepoll_honors_owner_stop(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(rig["bridge"].cmds("send_poll"))
        out = _op(rig, "/makepoll Q? | A | B", chat_id=GROUP,
                  kind=ChatKind.GROUP, title="Gold Room")
        assert "standing stop" in out
        assert len(rig["bridge"].cmds("send_poll")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)
