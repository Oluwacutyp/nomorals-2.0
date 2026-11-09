"""Parity #16 — WhatsApp poll results, PROVEN (the truth pass).

Census §1 graded WA "Read poll results" PARTIAL with the
note "truth-pass: UNPROVEN — no poll fixture exists in either
bridge's emit shapes", and §7 bundle #16 is the exercise that
closes it: once bundle #6 made polls sendable, read the tally
back end-to-end on the REAL bridge shapes and pin what is
actually true.

What the pass found and where it is proven:

* The wire contract was already honest and identical on both
  engines — {ok, question, options: [{text, voters}],
  total_voters, unreadable_votes?} on success, one shared
  refusal sentence for a poll outside bridge memory.  Pinned
  below against both bridges' sources and the rebuilt Go
  binaries, and driven through the adapter + /pollres with
  fixtures in exactly that shape.

* The TALLY itself had never been exercised: the G3 suite's
  fake bridge re-implements it by option name.  The real
  machinery keys votes by option HASH (sha256 of the name —
  whatsmeow's HashPollOptions identity) per (chat, poll id).
  It is now exercised for real on both engines:
  bridge/gobridge/pollresults_test.go drives the Go state +
  the cmdPollResults wire handler, and the Node test below
  extracts the shipped poll block from whatsapp-bridge.mjs
  VERBATIM and runs real encrypted votes (the mirror's exact
  HMAC/GCM construction, inverted) through capturePollMessage
  into pollResultsPayload.

* Two tally defects the exercise caught, fixed in both
  bridges: (1) a retracted vote (an update whose selection is
  empty — a poll update always carries the voter's WHOLE
  current selection) kept the voter in total_voters with no
  vote anywhere; the phone's tally drops them, and now both
  bridges delete the voter on an empty selection.
  (2) On the Node engine a vote that arrived but would not
  decrypt fell into capturePollMessage's outer catch and
  vanished — never counted as unreadable, so /pollres's
  "not counted" note could never fire there; the Go engine
  counts a DecryptPollVote failure as unreadable, and Node
  now does the same at the decrypt site.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.chatops import format_poll_results

PORT = 19461
OWNER = "2348012345678"
GROUP = "120363001234567890@g.us"
OTHER_GROUP = "120363009999999999@g.us"
ROOT = Path(__file__).resolve().parents[1]

#: The one refusal both bridges emit for a poll outside
#: bridge memory — transcribed from gobridge/profile.go
#: (cmdPollResults) and whatsapp-bridge.mjs (poll_results);
#: the source pins below hold all three in agreement.
REFUSAL = ("that poll is not in bridge memory — results live "
           "on polls this bridge has seen since it started")

#: A success verdict in the exact shape BOTH bridges emit
#: (Go pollResults map / Node pollResultsPayload), for the
#: formatter pins: field names, nesting and types are the
#: bridges', not an imagined contract.
REAL_VERDICT = {
    "ok": True,
    "question": "Dinner?",
    "options": [{"text": "Pizza", "voters": 2},
                {"text": "Sushi", "voters": 1}],
    "total_voters": 3,
}


def _option_hash(name: str) -> str:
    """The protocol's identity for one poll option."""
    return hashlib.sha256(name.encode()).hexdigest()


# ── source pins: both engines speak one contract ───────────

def test_both_bridges_emit_the_same_refusal_and_shape():
    go = (ROOT / "bridge" / "gobridge" / "profile.go"
          ).read_text(encoding="utf-8")
    node = (ROOT / "bridge" / "whatsapp-bridge.mjs"
            ).read_text(encoding="utf-8")
    # Node spells the refusal on one line; the Go source
    # concatenates it from fragments (the full sentence is
    # pinned on the Go WIRE in pollresults_test.go and in
    # the rebuilt binaries below).
    assert REFUSAL in node
    for fragment in ("that poll is not in bridge memory — ",
                     "results live on polls this bridge has "
                     "seen ", "since it started"):
        assert fragment in go
    for src, label in ((go, "go"), (node, "node")):
        assert '"total_voters"' in src or \
            "total_voters:" in src, label
        assert "unreadable_votes" in src, label
    # Per-option rows: {text, voters} on both engines.
    assert '"text": name, "voters": counts[i]' in go
    assert "{ text, voters: counts[i] }" in node
    # Memory keyed by (chat, poll id) on both engines — a
    # pollres in the wrong chat must miss, never borrow.
    assert 'chat.String() + "\\x00" + pollID' in go
    assert "const pollKeyOf = (chat, id) => `${chat}|${id}`" \
        in node


def test_retraction_fix_present_in_both_bridges():
    go = (ROOT / "bridge" / "gobridge" / "profile.go"
          ).read_text(encoding="utf-8")
    node = (ROOT / "bridge" / "whatsapp-bridge.mjs"
            ).read_text(encoding="utf-8")
    # Empty selection = retraction = the voter is deleted,
    # not stored as an empty vote (the #16 tally defect).
    assert "delete(st.votes, voter)" in go
    assert "votes.delete(voter)" in node
    # Node counts a decryption failure as an unreadable vote
    # at the decrypt site, like Go's DecryptPollVote path.
    assert node.count("pollUnreadable.set") >= 2


def test_rebuilt_go_binaries_carry_the_poll_verbs():
    for name in ("whatsapp-bridge.exe",
                 "whatsapp-bridge-linux"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"that poll is not in bridge memory" in data, \
            name
        assert b"poll_results needs a message_id" in data, \
            name


# ── formatter pins on the real verdict shape ───────────────

def test_format_poll_results_on_the_real_shape():
    text = format_poll_results(dict(REAL_VERDICT))
    assert text.splitlines() == [
        "Poll: Dinner?",
        "• Pizza — 2 votes (67%)",
        "• Sushi — 1 vote (33%)",
        "Total: 3 voters",
    ]


def test_format_poll_results_all_retracted_poll():
    # A known poll whose every vote was retracted reads as
    # zeros — it must NOT collapse into the refusal shape.
    verdict = {"ok": True, "question": "Dinner?",
               "options": [{"text": "Pizza", "voters": 0},
                           {"text": "Sushi", "voters": 0}],
               "total_voters": 0}
    text = format_poll_results(verdict)
    assert "• Pizza — 0 votes" in text
    assert "Total: 0 voters" in text


# ── the Node engine's real code, exercised ─────────────────

NODE_PRELUDE = r"""
import { createHash, createHmac, createDecipheriv,
         createCipheriv } from 'node:crypto';
const log = () => {};
let meJid = '2348012345678@s.whatsapp.net';
// Baileys' jidNormalizedUser: device parts are addressing
// detail; identity is the bare user JID.
const jidNormalizedUser = (jid) =>
  String(jid || '').replace(/:\d+@/, '@');
// Minimal PollVoteMessage decoder for the one message the
// bridge decodes: field 1, repeated bytes (selectedOptions).
// The encryption around it is the bridge's real mirror.
const proto = { Message: { PollVoteMessage: { decode(buf) {
  const selectedOptions = [];
  let i = 0;
  while (i < buf.length) {
    const tag = buf[i]; i += 1;
    if (tag !== 0x0a) throw new Error('bad field ' + tag);
    let len = 0; let shift = 0;
    for (;;) { const b = buf[i]; i += 1;
      len |= (b & 0x7f) << shift; shift += 7;
      if (!(b & 0x80)) break; }
    selectedOptions.push(
      Buffer.from(buf.subarray(i, i + len)));
    i += len;
  }
  return { selectedOptions };
} } } };
// The poll-memory state declarations, exactly as the shipped
// file declares them (asserted verbatim by the test); every
// line of LOGIC below them is extracted from the file itself.
const pollCreations = new Map();
const pollVotes = new Map();
const pollUnreadable = new Map();
"""

NODE_SCENARIO = r"""
const chat = '120363001234567890@g.us';
const pollId = 'POLL-NODE-1';
const creator = '2348012345678@s.whatsapp.net';
const secret = Buffer.from(
  Array.from({ length: 32 }, (_, i) => i + 1));
const v1 = '2348011111111@s.whatsapp.net';
const v2 = '2348022222222@s.whatsapp.net';

function encryptVote(pollMsgId, creatorJid, voterJid, names) {
  const toBytes = (v) => Buffer.from(String(v));
  const key0 = createHmac('sha256', Buffer.alloc(32))
    .update(Buffer.from(secret)).digest();
  const sign = Buffer.concat([
    toBytes(pollMsgId), toBytes(creatorJid),
    toBytes(voterJid), toBytes('Poll Vote'),
    new Uint8Array([1])]);
  const decKey = createHmac('sha256', key0).update(sign)
    .digest();
  const parts = [];
  for (const name of names) {
    const h = createHash('sha256').update(String(name))
      .digest();
    parts.push(Buffer.from([0x0a, h.length]), h);
  }
  const iv = Buffer.alloc(12, 7);
  const cipher = createCipheriv('aes-256-gcm', decKey, iv);
  cipher.setAAD(toBytes(`${pollMsgId}\u0000${voterJid}`));
  const ct = Buffer.concat(
    [cipher.update(Buffer.concat(parts)), cipher.final()]);
  return { encPayload: Buffer.concat([ct,
    cipher.getAuthTag()]), encIv: iv };
}

function castVote(voter, names) {
  capturePollMessage(chat,
    { key: { id: 'V' + Math.random(), remoteJid: chat,
             participant: voter, fromMe: false } },
    { pollUpdateMessage: {
        pollCreationMessageKey: { id: pollId,
          participant: creator, remoteJid: chat },
        vote: encryptVote(pollId, creator, voter,
          names) } });
}

// The creation lands the way an upsert delivers it.
capturePollMessage(chat,
  { key: { id: pollId, remoteJid: chat, fromMe: true } },
  { pollCreationMessage: { name: 'Dinner?', options: [
      { optionName: 'Pizza' }, { optionName: 'Sushi' }] },
    messageContextInfo: { messageSecret: secret } });
// A vote that will not decrypt arrives (garbage payload).
capturePollMessage(chat,
  { key: { id: 'V-GARBAGE', remoteJid: chat,
           participant: v1, fromMe: false } },
  { pollUpdateMessage: {
      pollCreationMessageKey: { id: pollId,
        participant: creator, remoteJid: chat },
      vote: { encPayload: Buffer.alloc(48, 9),
              encIv: Buffer.alloc(12, 3) } } });
castVote(v1, ['Pizza']);
castVote(v2, ['Pizza']);
castVote(v2, ['Sushi']);   // changed mind — replaces
console.log('STAGE1 ' +
  JSON.stringify(pollResultsPayload(chat, pollId)));
castVote(v1, []);          // retraction — voter drops out
console.log('STAGE2 ' +
  JSON.stringify(pollResultsPayload(chat, pollId)));
console.log('UNKNOWN ' + JSON.stringify(
  pollResultsPayload(chat, 'POLL-NEVER')));
console.log('WRONGCHAT ' + JSON.stringify(
  pollResultsPayload('999999999999@g.us', pollId)));
"""


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not on this box")
def test_node_poll_tally_exercised_verbatim(tmp_path):
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs"
           ).read_text(encoding="utf-8")
    # The state declarations the harness re-declares must be
    # the file's own, verbatim.
    for decl in ("const pollCreations = new Map();",
                 "const pollVotes = new Map();",
                 "const pollUnreadable = new Map();"):
        assert decl in src
    start = src.index("polls (parity G3): creation/vote "
                      "capture + tally")
    start = src.rindex("//", 0, start)
    end = src.index("Baileys helpers")
    end = src.rindex("//", 0, end)
    block = src[start:end]
    for fn in ("function notePollCreation",
               "function decryptPollVoteMirror",
               "function capturePollMessage",
               "function pollResultsPayload"):
        assert fn in block, fn
    harness = tmp_path / "poll_harness.mjs"
    harness.write_text(NODE_PRELUDE + block + NODE_SCENARIO,
                       encoding="utf-8")
    proc = subprocess.run(
        [shutil.which("node"), str(harness)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    lines = dict(
        line.split(" ", 1) for line in
        proc.stdout.strip().splitlines() if " " in line)
    stage1 = json.loads(lines["STAGE1"])
    assert stage1 == {
        "question": "Dinner?",
        "options": [{"text": "Pizza", "voters": 1},
                    {"text": "Sushi", "voters": 1}],
        "total_voters": 2,
        "unreadable_votes": 1,
    }
    stage2 = json.loads(lines["STAGE2"])
    assert stage2["options"] == [
        {"text": "Pizza", "voters": 0},
        {"text": "Sushi", "voters": 1}]
    assert stage2["total_voters"] == 1
    assert stage2["unreadable_votes"] == 1
    assert json.loads(lines["UNKNOWN"]) is None
    assert json.loads(lines["WRONGCHAT"]) is None


# ── end-to-end: /makepoll → votes → /pollres ───────────────

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
                if verb == "send_poll":
                    with srv.lock:
                        srv.polls[(cmd.get("chat"),
                                   srv.poll_id)] = {
                            "question": cmd.get("question"),
                            "options": list(
                                cmd.get("options") or []),
                            "votes": {},
                            "unreadable": 0}
                    reply["message_id"] = srv.poll_id
                elif verb == "poll_results":
                    reply = srv.results_reply(
                        cid, cmd.get("chat"),
                        cmd.get("message_id"))
                self.wfile.write(
                    json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeBridge(socketserver.ThreadingTCPServer):
    """The bridges' wire protocol with the bridges' tally
    semantics: memory keyed by (chat, poll id), votes stored
    as option HASHES and resolved back to options at read
    time, a re-vote replacing, a retraction deleting the
    voter, unreadable votes counted apart.  Replies are
    built in the bridges' exact verdict shape."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port=PORT):
        self.commands: list[dict] = []
        self.clients: list = []
        self.lock = threading.Lock()
        self.polls: dict[tuple, dict] = {}
        self.poll_id = "POLL-1"
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands
                    if c.get("cmd") == name]

    def cast_vote(self, chat, voter, *option_names):
        """One decrypted vote landing in bridge memory, as
        option hashes — an empty selection is a retraction
        and deletes the voter (the #16-fixed semantics)."""
        with self.lock:
            state = self.polls[(chat, self.poll_id)]
            if not option_names:
                state["votes"].pop(voter, None)
            else:
                state["votes"][voter] = [
                    _option_hash(n) for n in option_names]

    def cast_unreadable(self, chat):
        with self.lock:
            self.polls[(chat, self.poll_id)][
                "unreadable"] += 1

    def results_reply(self, cid, chat, message_id):
        with self.lock:
            state = self.polls.get((chat, message_id))
            if state is None:
                return {"id": cid, "ok": False,
                        "error": REFUSAL}
            counts = [0] * len(state["options"])
            by_hash = {_option_hash(name): i for i, name
                       in enumerate(state["options"])}
            for selected in state["votes"].values():
                for h in selected:
                    if h in by_hash:
                        counts[by_hash[h]] += 1
            out = {"id": cid, "ok": True,
                   "question": state["question"],
                   "options": [
                       {"text": name, "voters": counts[i]}
                       for i, name in
                       enumerate(state["options"])],
                   "total_voters": len(state["votes"])}
            if state["unreadable"] > 0:
                out["unreadable_votes"] = state["unreadable"]
            return out

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

    tmp = tmp_path_factory.mktemp("parity16")
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
    env = {"runtime": runtime, "bridge": bridge,
           "adapter": adapter, "rec": rec}
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, chat_id=GROUP, kind=ChatKind.GROUP,
        title="Gold Room", reply_to=""):
    chat = ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=OWNER,
        message_id=f"op-{time.time()}", ts=time.time(),
        meta={"owner_command": True}, reply_to=reply_to)
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def test_pollres_end_to_end_on_real_shapes(rig):
    out = _op(rig, "/makepoll Dinner? | Pizza | Sushi")
    assert "poll posted" in out
    bridge = rig["bridge"]
    bridge.cast_vote(GROUP, "voter-a", "Pizza")
    bridge.cast_vote(GROUP, "voter-b", "Pizza")
    bridge.cast_vote(GROUP, "voter-c", "Sushi")
    bridge.cast_vote(GROUP, "voter-c", "Pizza")  # changed mind
    bridge.cast_unreadable(GROUP)
    out = _op(rig, "/pollres", reply_to="POLL-1")
    # The whole rendered read-back, line for line.
    assert "Poll: Dinner?" in out
    assert "• Pizza — 3 votes (100%)" in out
    assert "• Sushi — 0 votes (0%)" in out
    assert "Total: 3 voters" in out
    assert ("(1 vote arrived that the bridge couldn't "
            "decrypt — not counted.)") in out
    # The ask that reached the bridge is the real wire shape.
    asks = bridge.cmds("poll_results")
    assert asks[-1]["chat"] == GROUP
    assert asks[-1]["message_id"] == "POLL-1"


def test_pollres_reflects_a_retraction(rig):
    bridge = rig["bridge"]
    bridge.cast_vote(GROUP, "voter-b")  # voter-b retracts
    out = _op(rig, "/pollres", reply_to="POLL-1")
    assert "• Pizza — 2 votes (100%)" in out
    assert "Total: 2 voters" in out


def test_pollres_wrong_chat_refuses_with_bridge_words(rig):
    # The same poll id, asked from another chat: bridge
    # memory is keyed by (chat, id), so the answer is the
    # bridges' own refusal — never the other chat's tally.
    out = _op(rig, "/pollres", chat_id=OTHER_GROUP,
              title="Elsewhere", reply_to="POLL-1")
    assert f"couldn't do it — {REFUSAL}" in out


def test_adapter_poll_results_wire_and_empty_id(rig):
    adapter = rig["adapter"]
    chat = ChatRef(platform="whatsapp", chat_id=GROUP,
                   kind=ChatKind.GROUP, title="Gold Room",
                   peer=GROUP)
    before = len(rig["bridge"].cmds("poll_results"))
    verdict = adapter.poll_results(chat, "")
    assert verdict["ok"] is False
    assert "message id" in verdict["error"]
    # The empty id refused BEFORE traffic.
    assert len(rig["bridge"].cmds("poll_results")) == before
    verdict = adapter.poll_results(chat, "POLL-1")
    assert verdict["ok"] is True
    assert verdict["question"] == "Dinner?"
    assert verdict["total_voters"] == 2
    asks = rig["bridge"].cmds("poll_results")
    assert asks[-1]["chat"] == GROUP
    assert asks[-1]["message_id"] == "POLL-1"
