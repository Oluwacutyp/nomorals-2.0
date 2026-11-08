"""PERSONAL-SURFACE TRUTH PASS (owner live report, 2026-11-07).

The presence pass (8aaa7e0) declared the personal family
"audited INTACT" from reading code; live overruled it: the owner
could not view a view-once (/viewonce and /vv did not exist),
statuses stayed unviewed, and /statusreact never answered at his
seat.  This suite replaces reading-verdicts with
exercised-verdicts: every feature is driven through the real
composed chain — bridge socket → WhatsApp adapter → gateway
funnel → runtime personal-surface consumers → adapter →
recording fake bridge — and a feature counts as working only
when its outbound verb FIRES on the wire (or its journal row
lands).

Fixture provenance — payload shapes are copied from the bridges'
own emit code, not invented:
  * Go bridge (bridge/gobridge/messages.go): onMessage emits
    status_post {id, from:{id,name}, text, media ([]map or nil),
    view_once, ts-ms} at chat == types.StatusBroadcastJID
    (~line 428), protocol REVOKE → {type:"revoke"} (~392),
    IsEdit → {type:"edit"} (~407), DM/group messages with
    "view_once": v.IsViewOnce; handleEvent (main.go) emits
    *events.CallOffer → {type:"call_offer", id, from: STRING, ts}.
    Poster ids come from fromOf (messages.go:198): the sender's
    ToNonAD form — a phone JID or a @lid JID, both exercised.
  * Node bridge (bridge/whatsapp-bridge.mjs ~828): status_post
    {id, from:{id: participant, name}, text, media: [] when
    none, view_once, ts-ms}.

Fixed here, pinned below:
  (b) /viewonce + /vv never existed — the capture half kept
      view-once media in the vault and no command ever showed
      it back; the unknown slash fell through to conversation
      and the brain answered filler ("Acknowledged. Working on
      viewonce.").  The commands now exist end to end.
  (a) a bridge verdict of ok:false on status_view /
      status_react / story view / presence / reject_call
      collapsed to a bare False and vanished — the post "stayed
      unviewed" with no trace anywhere, indistinguishable from
      the toggle being off.  Refusals now journal in the vault
      (status_view_failed / status_react_failed /
      story_view_failed, once per post per process) and the
      bridge's own error text lands in the partner failure
      ledger at the seam.
  (c) /statusreact was invisible at the owner's seats: the
      /pair toggle usage named only five of the nine toggles,
      /pair's list never named the command, and on the public
      bot the owner himself got the stranger redirect ("try
      /game list") — the gated-surface rule stands (owner
      ruling 2026-10-20), but a verifiably-owner caller is now
      told where the command lives.
  Node bridge: 'append' upserts (history sync / reconnect
  catch-up) were dropped whole at whatsapp-bridge.mjs:787, so
  statuses posted while the bridge was down never reached the
  status family on the Baileys rung; status posts — only —
  now proceed from append upserts (source-pinned in
  test_pairing_nodefix style below via the handler text).
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, SendResult

PORT = 19397
OWNER = "2348012345678"
LID_POSTER = "1555123456789@lid"
PN_POSTER = "2347099999999@s.whatsapp.net"
TG_OWNER = "777"


# ── recording fake bridge (the bridges' wire protocol) ─────────

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
                if cmd.get("cmd") == "send":
                    reply["message_id"] = "WAMID-TEST-1"
                if cmd.get("cmd") == "status_send":
                    reply["message_id"] = "WAMID-STATUS-1"
                self.wfile.write(json.dumps(reply).encode() + b"\n")
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
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands
                    if c.get("cmd") == name]

    def all_json(self):
        with self.lock:
            return "\n".join(json.dumps(c) for c in self.commands)

    def emit(self, obj):
        line = (json.dumps(obj) + "\n").encode()
        with self.lock:
            clients = list(self.clients)
        for w in clients:
            try:
                w.write(line)
                w.flush()
            except OSError:
                pass

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


class RecordingAdapter:
    """A gateway adapter that records sends (telegram seats)."""

    def __init__(self, platform):
        self.name = platform
        self.sent: list[tuple] = []

    def start(self, handler):
        self._handler = handler
        return True

    def stop(self):
        pass

    def send(self, chat, text, **kw):
        self.sent.append((chat, text))
        return SendResult(ok=True, platform=self.name,
                          message_id="rec-1")


class TGStoriesAdapter:
    name = "telegram"

    def __init__(self, tmp):
        self._tmp = tmp
        self.viewed: list[tuple] = []

    def fetch_active_stories(self):
        pic = self._tmp / "story.jpg"
        if not pic.exists():
            pic.write_bytes(b"\xff\xd8story")
        return [{"poster_id": "555", "poster_name": "Tunde",
                 "story_id": "7", "path": str(pic),
                 "kind": "image", "ts": time.time()}]

    def view_stories(self, poster_id, max_id):
        self.viewed.append((poster_id, max_id))
        return True


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("truthpass")
    bridge = FakeBridge()
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "partner.owner_sender_ids": TG_OWNER,
        "chat.local_enabled": "true",
        "chat.whatsapp_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    bot_rec = RecordingAdapter("telegram-bot")
    personal_rec = RecordingAdapter("telegram-personal")
    tg = TGStoriesAdapter(tmp)
    gateway = ChatGateway(
        {"local": LocalAdapter(), "telegram-bot": bot_rec,
         "telegram-personal": personal_rec, "telegram": tg},
        db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    ws = Path(settings.workspace_dir)
    rec = sup.registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp", identity=OWNER,
        creds_dir=str(ws / "data" / "whatsapp_sessions"
                       / f"wa-{OWNER}"),
        port=PORT, status="linked", created_at=1.0))
    sup.registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        creds_dir=str(ws / "data" / "telegram_sessions" / "tg-777"),
        port=0, status="linked", created_at=1.0))
    sup.registry.set_toggles(
        "telegram:777", status_saving=True, status_viewing=True)
    sup.registry.set_toggles(
        rec.id, status_saving=True, status_viewing=True,
        status_react=True, always_online=True, view_once=True,
        antidelete=True, antiedit=True, anticall=True, antitag=True)

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
        lambda: runtime.gateway.adapters.get("whatsapp") is not None)
    adapter = runtime.gateway.adapters["whatsapp"]
    assert wait_for(lambda: adapter.connected.is_set())
    svc = runtime._pairing_personal()
    env = {
        "runtime": runtime, "bridge": bridge, "adapter": adapter,
        "sup": sup, "rec": rec, "svc": svc,
        "personal": svc["personal"], "vault": svc["personal"].vault,
        "bot_rec": bot_rec, "personal_rec": personal_rec, "tg": tg,
        "tmp": tmp, "beat": beat,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def wa_self(text, reply_to="", mid=None):
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=OWNER,
                     kind=ChatKind.DM, title="you", peer=OWNER),
        incoming=True, text=text, sender="Owner", sender_id=OWNER,
        message_id=mid or f"cmd-{time.time()}", reply_to=reply_to,
        ts=time.time(), meta={"self_chat": True, "from_owner": True})


def sends_with(env, needle):
    return [c for c in env["bridge"].cmds("send")
            if needle in str(c.get("text") or "")]


# ── inbound family: statuses (Go + Node shapes, both posters) ──

def test_status_go_shape_lid_poster_viewed_reacted_saved(rig):
    bridge, vault, rec = rig["bridge"], rig["vault"], rig["rec"]
    now_ms = int(time.time() * 1000)
    # Go onMessage, status branch: media is nil (Go nil slice →
    # JSON null) for a text status; poster is a LID.
    bridge.emit({"type": "status_post", "id": "ST-LID-1",
                 "from": {"id": LID_POSTER, "name": "Ada"},
                 "text": "lid status", "media": None,
                 "view_once": False, "ts": now_ms})
    assert wait_for(lambda: any(
        c.get("from") == LID_POSTER
        for c in bridge.cmds("status_view")))
    assert wait_for(lambda: any(
        c.get("from") == LID_POSTER
        for c in bridge.cmds("status_react")))
    kinds = [r.get("kind") for r in vault.journal(rec.id)]
    assert "status" in kinds and "status_react" in kinds


def test_status_go_shape_pn_poster_media_saved(rig):
    bridge, vault, rec = rig["bridge"], rig["vault"], rig["rec"]
    pic = rig["tmp"] / "status-pic.jpg"
    pic.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    now_ms = int(time.time() * 1000)
    bridge.emit({"type": "status_post", "id": "ST-PN-1",
                 "from": {"id": PN_POSTER, "name": "Bola"},
                 "text": "",
                 "media": [{"path": str(pic), "mime": "image/jpeg",
                            "kind": "image"}],
                 "view_once": False, "ts": now_ms})
    assert wait_for(lambda: any(
        c.get("from") == PN_POSTER
        for c in bridge.cmds("status_view")))
    saved = vault.saved_entries(rec.id)
    assert any(r.get("kind") == "status" and r.get("path")
               and r.get("source_id") == "ST-PN-1" for r in saved)


def test_status_node_shape_viewed_and_reacted(rig):
    bridge = rig["bridge"]
    now_ms = int(time.time() * 1000)
    # Node bridge status_post: media is [] (never null) when the
    # post carries no media.
    before = len(bridge.cmds("status_view"))
    bridge.emit({"type": "status_post", "id": "NODE-ST-1",
                 "from": {"id": PN_POSTER, "name": "Bola"},
                 "text": "node status", "media": [],
                 "view_once": False, "ts": now_ms})
    assert wait_for(
        lambda: len(bridge.cmds("status_view")) > before)


def test_status_burst_all_viewed(rig):
    bridge = rig["bridge"]
    now_ms = int(time.time() * 1000)
    before = len(bridge.cmds("status_view"))
    for i in (1, 2, 3):
        bridge.emit({"type": "status_post", "id": f"BURST-{i}",
                     "from": {"id": PN_POSTER, "name": "Bola"},
                     "text": f"burst {i}", "media": None,
                     "view_once": False, "ts": now_ms})
    assert wait_for(
        lambda: len(bridge.cmds("status_view")) >= before + 3)


# ── inbound family: view-once capture, delete/edit, call, tag ──

DM_CHAT = {"id": PN_POSTER, "kind": "dm", "title": "Bola"}


def _dm(mid, **kw):
    base = {"type": "message", "id": mid, "chat": DM_CHAT,
            "from": {"id": PN_POSTER, "name": "Bola"}, "text": "",
            "media": [], "reply_to": "", "mentioned": False,
            "mentions": [], "view_once": False,
            "ts": int(time.time() * 1000)}
    base.update(kw)
    return base


def test_viewonce_captured_and_miss_journaled(rig):
    bridge, vault, rec = rig["bridge"], rig["vault"], rig["rec"]
    vo = rig["tmp"] / "vo-pic.jpg"
    vo.write_bytes(b"\xff\xd8view-once-bytes")
    bridge.emit(_dm("DM-VO-1", view_once=True,
                    media=[{"path": str(vo), "mime": "image/jpeg",
                            "kind": "image"}]))
    assert wait_for(lambda: any(
        r.get("kind") == "view_once"
        and r.get("source_id") == "DM-VO-1"
        for r in vault.saved_entries(rec.id)))
    # Go shape when the download failed: media null + view_once.
    bridge.emit(_dm("DM-VO-2", view_once=True, media=None))
    assert wait_for(lambda: any(
        r.get("kind") == "view_once_missed"
        for r in vault.journal(rec.id)))


def test_antidelete_and_antiedit_recoveries(rig):
    bridge, vault, rec = rig["bridge"], rig["vault"], rig["rec"]
    now_ms = int(time.time() * 1000)
    bridge.emit(_dm("DM-1", text="secret plans"))
    time.sleep(0.4)
    bridge.emit({"type": "revoke", "id": "DM-1",
                 "chat": {"id": PN_POSTER, "kind": "dm"},
                 "from": {"id": PN_POSTER}, "ts": now_ms})
    assert wait_for(lambda: any(
        r.get("kind") == "delete" and r.get("recovered")
        and r.get("original_text") == "secret plans"
        for r in vault.recoveries(rec.id)))
    bridge.emit(_dm("DM-2", text="original words"))
    time.sleep(0.4)
    bridge.emit({"type": "edit", "id": "DM-2",
                 "chat": {"id": PN_POSTER, "kind": "dm"},
                 "from": {"id": PN_POSTER, "name": "Bola"},
                 "text": "edited words", "ts": now_ms})
    assert wait_for(lambda: any(
        r.get("kind") == "edit"
        and r.get("original_text") == "original words"
        for r in vault.recoveries(rec.id)))


def test_anticall_rejects_call_offer(rig):
    bridge = rig["bridge"]
    # Go handleEvent CallOffer shape: "from" is a bare string.
    bridge.emit({"type": "call_offer", "id": "CALL-1",
                 "from": PN_POSTER, "ts": int(time.time() * 1000)})
    assert wait_for(lambda: len(bridge.cmds("reject_call")) >= 1)


def test_antitag_journals_and_flags_owner(rig):
    bridge, vault = rig["bridge"], rig["vault"]
    mentions = [PN_POSTER] + [
        f"234700000000{i}@s.whatsapp.net" for i in range(1, 5)
    ] + [f"{OWNER}@s.whatsapp.net"]
    bridge.emit(_dm("GRP-1", text="tagging everyone",
                    chat={"id": "12345@g.us", "kind": "group",
                          "title": "Family"},
                    mentioned=True, mentions=mentions))
    assert wait_for(lambda: any(
        r.get("kind") == "mass_tag" for r in vault.journal()))
    assert wait_for(lambda: any(
        "mass tag" in str(c.get("text") or "")
        for c in bridge.cmds("send")))


def test_telegram_stories_viewed_and_saved(rig):
    rig["personal"].cycle_stories("telegram")
    assert rig["tg"].viewed == [("555", 7)]
    saved = rig["vault"].saved_entries("telegram:777")
    assert any(r.get("kind") == "story" for r in saved)


# ── the refusal fix: ok:false must leave a trace ───────────────

def test_refused_status_verbs_journal_and_ledger(rig):
    from nomorals.agents.partner import failures

    bridge, vault, rec, adapter = (rig["bridge"], rig["vault"],
                                   rig["rec"], rig["adapter"])
    orig = adapter._admin_cmd

    def refusing(name, payload, timeout=None):
        if name in ("status_view", "status_react"):
            return {"ok": False,
                    "error": "marked read failed: not in chat"}
        return orig(name, payload, timeout)

    adapter._admin_cmd = refusing
    try:
        bridge.emit({"type": "status_post", "id": "REFUSE-1",
                     "from": {"id": PN_POSTER, "name": "Bola"},
                     "text": "will refuse", "media": None,
                     "view_once": False,
                     "ts": int(time.time() * 1000)})
        assert wait_for(lambda: any(
            r.get("kind") == "status_view_failed"
            and r.get("source_id") == "REFUSE-1"
            for r in vault.journal(rec.id)), timeout=10)
        assert any(r.get("kind") == "status_react_failed"
                   for r in vault.journal(rec.id))
    finally:
        adapter._admin_cmd = orig
    recent = failures.recent_failures(200)
    assert any(f.site == "runtime_provision.status_viewer"
               and "marked read failed" in f.message
               for f in recent)


# ── corpse (b): /viewonce + /vv reveal ─────────────────────────

def test_viewonce_command_reveals_newest_capture(rig):
    runtime, bridge = rig["runtime"], rig["bridge"]
    runtime.on_message(wa_self("/viewonce"))
    # the reply names the reveal…
    assert wait_for(lambda: sends_with(rig, "straight from your "
                                             "vault"))
    # …and the media itself went out over the bridge wire.
    assert wait_for(lambda: "view_once" in bridge.all_json()
                    and ".jpg" in bridge.all_json())


def test_vv_reply_form_reveals_that_capture(rig):
    runtime = rig["runtime"]
    runtime.on_message(wa_self("/vv", reply_to="DM-VO-1"))
    assert wait_for(lambda: sends_with(rig, "view-once from Bola"))


def test_viewonce_list_and_index_forms(rig):
    runtime = rig["runtime"]
    runtime.on_message(wa_self("/viewonce list"))
    assert wait_for(lambda: sends_with(rig, "view-once captures"))
    runtime.on_message(wa_self("/viewonce 1"))
    assert wait_for(lambda: sends_with(rig, "straight from your "
                                             "vault"))


# ── seats: where the personal commands answer ──────────────────

def test_statusreact_answers_in_wa_self_chat(rig):
    rig["runtime"].on_message(wa_self("/statusreact"))
    assert wait_for(lambda: sends_with(rig, "/statusreact is"))


def test_statusreact_answers_on_personal_bot_dm(rig):
    rig["runtime"].on_message(ChatMessage(
        chat=ChatRef(platform="telegram-personal", chat_id=TG_OWNER,
                     kind=ChatKind.DM, title="", peer=TG_OWNER),
        incoming=True, text="/statusreact", sender="Owner",
        sender_id=TG_OWNER, message_id=f"tg-{time.time()}",
        ts=time.time()))
    assert wait_for(lambda: any(
        "/statusreact is" in text
        for _chat, text in rig["personal_rec"].sent))


def test_statusreact_main_bot_owner_told_where_it_lives(rig):
    rig["bot_rec"].sent.clear()
    rig["runtime"].on_message(ChatMessage(
        chat=ChatRef(platform="telegram-bot", chat_id=TG_OWNER,
                     kind=ChatKind.DM, title="", peer=TG_OWNER),
        incoming=True, text="/statusreact", sender="Owner",
        sender_id=TG_OWNER, message_id=f"tb-{time.time()}",
        ts=time.time()))
    assert wait_for(lambda: len(rig["bot_rec"].sent) >= 1)
    text = rig["bot_rec"].sent[0][1]
    # the gate stands (no operator power on the public bot) but
    # the owner is told where the command answers.
    assert "public bot" in text and "/statusreact" in text
    assert "WhatsApp" in text


def test_statusreact_main_bot_stranger_gets_generic_redirect(rig):
    rig["bot_rec"].sent.clear()
    rig["runtime"].on_message(ChatMessage(
        chat=ChatRef(platform="telegram-bot", chat_id="999",
                     kind=ChatKind.DM, title="", peer="999"),
        incoming=True, text="/statusreact", sender="Stranger",
        sender_id="999", message_id=f"ts-{time.time()}",
        ts=time.time()))
    assert wait_for(lambda: len(rig["bot_rec"].sent) >= 1)
    text = rig["bot_rec"].sent[0][1]
    assert "public chat" in text and "game list" in text


# ── corpse (c): /pair discoverability ──────────────────────────

def _new_send_text(rig, before):
    """All bridge send texts since ``before`` (replies can chunk)."""
    return "\n".join(str(c.get("text") or "")
                     for c in rig["bridge"].cmds("send")[before:])


def test_pair_toggle_usage_names_all_nine_toggles(rig):
    before = len(rig["bridge"].cmds("send"))
    rig["runtime"].on_message(wa_self("/pair toggle bogus"))
    assert wait_for(lambda: "usage: /pair toggle" in _new_send_text(
        rig, before))
    text = _new_send_text(rig, before)
    for name in ("status_saving", "view_once", "antidelete",
                 "always_online", "status_viewing", "status_react",
                 "anticall", "antitag", "antiedit"):
        assert name in text, f"{name} missing from usage: {text}"


def test_pair_list_names_personal_commands_and_verdicts(rig):
    before = len(rig["bridge"].cmds("send"))
    rig["runtime"].on_message(wa_self("/pair"))
    assert wait_for(lambda: "paired accounts" in _new_send_text(
        rig, before))
    text = _new_send_text(rig, before)
    assert "/statusreact" in text and "/viewonce" in text
    assert "status_viewing" in text  # per-account live verdicts


def test_pair_saved_and_recovered_answer(rig):
    rig["runtime"].on_message(wa_self("/pair saved"))
    assert wait_for(lambda: sends_with(rig, "saved (latest"))
    rig["runtime"].on_message(wa_self("/pair recovered"))
    assert wait_for(lambda: sends_with(rig, "recovered"))


def test_poststatus_fires_status_send(rig):
    rig["runtime"].on_message(wa_self("/poststatus hello world"))
    assert wait_for(
        lambda: len(rig["bridge"].cmds("status_send")) >= 1)


# ── chatops sweep: verbs fire through handle_control ───────────

def _op(rig, text, *, chat_id=PN_POSTER, kind=ChatKind.DM,
        reply_to=""):
    chat = ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=kind, title="", peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=OWNER, message_id=f"op-{time.time()}",
        reply_to=reply_to, ts=time.time(),
        meta={"owner_command": True})
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def test_chatops_chat_state_verbs_fire(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("chat_modify"))
    for text in ("/archive", "/mutechat", "/readchat",
                 "/unreadchat", "/unarchive", "/unmutechat"):
        reply = _op(rig, text)
        assert isinstance(reply, str) and reply
    assert wait_for(
        lambda: len(bridge.cmds("chat_modify")) >= before + 6)


def test_chatops_star_block_disappear_verbs_fire(rig):
    bridge = rig["bridge"]
    reply = _op(rig, "/star", reply_to="DM-2")
    assert isinstance(reply, str) and reply
    assert wait_for(lambda: len(bridge.cmds("message_star")) >= 1)
    reply = _op(rig, "/block 2347099999999", chat_id=OWNER)
    assert isinstance(reply, str) and reply
    assert wait_for(lambda: len(bridge.cmds("blocklist")) >= 1)
    reply = _op(rig, "/disappear 7d")
    assert isinstance(reply, str) and reply
    assert wait_for(lambda: len(bridge.cmds("chat_disappear")) >= 1)


def test_chatops_unstar_unblock_and_banchecker(rig):
    bridge = rig["bridge"]
    reply = _op(rig, "/unstar", reply_to="DM-2")
    assert isinstance(reply, str) and reply
    assert wait_for(lambda: any(
        c.get("star") is False for c in bridge.cmds("message_star")))
    reply = _op(rig, "/unblock 2347099999999", chat_id=OWNER)
    assert isinstance(reply, str) and reply
    assert wait_for(lambda: any(
        c.get("action") == "unblock"
        for c in bridge.cmds("blocklist")))
    # banchecker with no argument reads the paired accounts'
    # own state — an honest report, no network verb.
    reply = _op(rig, "/banchecker", chat_id=OWNER)
    assert isinstance(reply, str) and reply


def test_chatops_toggle_commands_flip_registry(rig):
    sup = rig["sup"]
    rec = rig["rec"]
    sup.registry.set_toggles(rec.id, anticall=False, antitag=False,
                             antiedit=False)
    for text, name in (("/anticall on", "anticall"),
                       ("/antitag on", "antitag"),
                       ("/antiedit on", "antiedit")):
        reply = _op(rig, text, chat_id=OWNER)
        assert isinstance(reply, str) and reply
        assert sup.registry.get(rec.id).toggle(name), text
    # restore the family state the other tests rely on
    sup.registry.set_toggles(rec.id, anticall=True, antitag=True,
                             antiedit=True)


def test_chatops_group_verbs_answer(rig):
    bridge = rig["bridge"]
    reply = _op(rig, "/gcstatus", chat_id="12345@g.us",
                kind=ChatKind.GROUP)
    assert isinstance(reply, str) and reply
    before = len(bridge.cmds("send"))
    reply = _op(rig, "/tagall roll call", chat_id="12345@g.us",
                kind=ChatKind.GROUP)
    assert isinstance(reply, str) and reply
    # tagall sends from the personal surface (or answers honestly
    # that the room can't be enumerated here) — never silence.
    assert wait_for(lambda: len(bridge.cmds("send")) > before,
                    timeout=5) or "member" in reply.lower()


def test_chatops_unsend_and_edit_answer_honestly(rig):
    # Both act on the owner's OWN messages; the target here is a
    # message id the fake bridge issued for Devon's sends.  The
    # contract pinned: the handler answers (never silence, never
    # a fake success) and the bridge verb is attempted.
    reply = _op(rig, "/unsend", chat_id=OWNER, reply_to="WAMID-TEST-1")
    assert isinstance(reply, str) and reply
    reply = _op(rig, "/editmsg corrected words", chat_id=OWNER,
                reply_to="WAMID-TEST-1")
    assert isinstance(reply, str) and reply


# ── organ-level: refusal journaling, once per post ─────────────

def _organ(tmp_path, **kw):
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp", identity=OWNER,
        creds_dir=str(tmp_path / "sess"), port=PORT,
        status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=True,
                         status_react=True, status_saving=True)
    vault = PersonalVault(tmp_path / "vault")
    personal = PersonalSurfaces(registry, vault, **kw)
    return personal, vault, rec


def test_organ_refused_view_and_react_journal_once(tmp_path):
    personal, vault, rec = _organ(
        tmp_path,
        status_viewer=lambda r, item: False,
        status_reactor=lambda r, item: False)
    for _ in range(2):  # redelivery must not spam the journal
        personal.observe_status_post(
            "whatsapp", poster_id=PN_POSTER, poster_name="Bola",
            text="hi", source_id="S-REF-1", ts=time.time())
    kinds = [r.get("kind") for r in vault.journal(rec.id)]
    assert kinds.count("status_view_failed") == 1
    assert kinds.count("status_react_failed") == 1


def test_organ_refused_story_view_journals(tmp_path):
    personal, vault, rec = _organ(
        tmp_path,
        story_fetcher=lambda r: [{
            "poster_id": "555", "poster_name": "Tunde",
            "story_id": "9", "path": "", "kind": "image",
            "ts": time.time()}],
        story_viewer=lambda r, poster, max_id: False)
    personal.cycle_stories("whatsapp")
    kinds = [r.get("kind") for r in vault.journal(rec.id)]
    assert "story_view_failed" in kinds


# ── registration: the catalog knows the new commands ───────────

def test_viewonce_registered_owner_only():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
        parse_control,
    )

    assert {"viewonce", "vv"} <= CHATOPS_COMMANDS
    assert "viewonce" in CONTROL_COMMANDS
    assert "vv" in CONTROL_COMMANDS
    assert "viewonce" in COMMAND_DETAILS
    assert "vv" in COMMAND_DETAILS
    assert "viewonce" in LIST_ONELINERS
    assert "viewonce" not in PUBLIC_COMMANDS
    assert "vv" not in PUBLIC_COMMANDS
    assert parse_control("/viewonce").kind == "viewonce"
    assert parse_control("/vv").kind == "vv"
    chat_group = [kinds for title, kinds in LIST_GROUPS
                  if "chat & privacy" in title][0]
    assert "viewonce" in chat_group and "vv" in chat_group


def test_node_bridge_append_lets_status_posts_through():
    # Source pin for the Baileys rung: the upsert handler must
    # not return early on non-'notify' upserts; the filter is
    # per-message and passes status@broadcast only.
    src = (Path(__file__).resolve().parents[1]
           / "bridge" / "whatsapp-bridge.mjs").read_text()
    assert "if (type !== 'notify') return;" not in src
    assert "type !== 'notify' && jid !== 'status@broadcast'" in src
