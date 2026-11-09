"""INBOUND VIDEO-NOTE CLASSIFICATION (census re-grade gap,
2026-11-09).

Bundle #10 built video-note SENDING: on the wire out, the
round form has been kind "video_note" ever since.  The
RECEIVE side never learned the shape.  A video note arrives
as a PTV message — in whatsmeow a VideoMessage in the
message's own PtvMessage field (NOT VideoMessage), in Baileys
a VideoMessage under the ptvMessage key (NOT videoMessage) —
and neither bridge's inbound classifier looked there, so a
received note classified as kind "" and its media never even
attempted a download: the adapter, the journal and the save
paths downstream all saw a message with nothing on it.

The fix sits at the two bridges' classification seams (the
Python adapter passes the bridge's kind through verbatim):

  * Go (bridge/gobridge/messages.go): mediaKindOf gains the
    GetPtvMessage case -> "video_note"; mediaPartsOf downloads
    through the PtvMessage handle; mediaExtFor maps the kind
    to .mp4 (an unmapped kind used to land in the cache as
    .bin).  Pinned at the proto level in
    bridge/gobridge/videonote_test.go.
  * Node (bridge/whatsapp-bridge.mjs): mediaKindOf gains the
    ptvMessage key; downloadInbound + quotedPayload read the
    note's bytes from ptvMessage; mediaExtFor maps the kind.
    The classifier functions are extracted VERBATIM from the
    shipped file and exercised under node against real
    Baileys message shapes (the parity #16 pattern).

The PTV field is the discriminator, pinned on both engines:
an ordinary video still classifies "video".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatMessage
from nomorals.social.chat.whatsapp import WhatsAppAdapter

ROOT = Path(__file__).resolve().parents[1]


# ── Go bridge: source + binary pins ────────────────────────

def test_go_bridge_classifies_ptv_on_arrival():
    src = (ROOT / "bridge" / "gobridge" / "messages.go"
           ).read_text(encoding="utf-8")
    # The classification case, the download handle and the
    # cache extension — all three seams, or the note arrives
    # named but unreachable / misfiled.
    assert 'case m.GetPtvMessage() != nil:' in src
    assert 'return "video_note"' in src
    assert 'case "video_note":' in src
    assert 'msg.GetPtvMessage(), msg.GetPtvMessage().GetMimetype()' in src
    assert '"video_note": ".mp4"' in src


def test_go_binaries_carry_the_receive_kind():
    # The base binaries predate the fix and carry no
    # "video_note" literal at all (the send side speaks the
    # ptv boolean, never the kind string): its presence in
    # BOTH rebuilt binaries proves the new classifier is in
    # the shipped artifact, not just the source.
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        blob = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"video_note" in blob, name


# ── Node bridge: verbatim classifier, real shapes ──────────

@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not on this box")
def test_node_classifier_exercised_verbatim(tmp_path):
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs"
           ).read_text(encoding="utf-8")
    start = src.index("function mediaKindOf(m) {")
    end = src.index("function mediaExtFor(kind, inner) {")
    end = src.index("\n}\n", end) + len("\n}\n")
    block = src[start:end]
    # The slice must be the shipped classifier block — the
    # functions under test plus the unwrap/cache helpers that
    # sit between them in the file (never called here).
    for fn in ("function mediaKindOf", "function unwrapViewOnce",
               "function mediaCacheName", "function mediaExtFor"):
        assert fn in block, fn
    assert "if (msg.ptvMessage) return 'video_note';" in block
    assert "video_note: '.mp4'" in block
    scenario = r"""
const shapes = {
  note: { message: { ptvMessage: { mimetype: 'video/mp4' } } },
  plain_video: { message: { videoMessage: { mimetype: 'video/mp4' } } },
  photo: { message: { imageMessage: { mimetype: 'image/jpeg' } } },
  voice: { message: { audioMessage: { ptt: true } } },
  text: { message: { conversation: 'hi' } },
};
const out = {};
for (const [name, m] of Object.entries(shapes)) {
  out[name] = mediaKindOf(m);
}
out.ext_note = mediaExtFor('video_note', {});
out.ext_video = mediaExtFor('video', {});
out.ext_unknown = mediaExtFor('', {});
console.log(JSON.stringify(out));
"""
    harness = tmp_path / "videonote_harness.mjs"
    harness.write_text(block + scenario, encoding="utf-8")
    proc = subprocess.run(
        [shutil.which("node"), str(harness)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip())
    assert out == {
        "note": "video_note",
        "plain_video": "video",
        "photo": "image",
        "voice": "voice",
        "text": "",
        "ext_note": ".mp4",
        "ext_video": ".mp4",
        "ext_unknown": ".bin",
    }


def test_node_download_seams_read_the_ptv_key():
    # downloadInbound / quotedPayload cannot run standalone
    # (they own the socket + download machinery), so their
    # ptv seams are pinned in the shipped source — the same
    # mixed standard the send forms were pinned to in #10.
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs"
           ).read_text(encoding="utf-8")
    assert "|| m.message?.ptvMessage" in src          # downloadInbound
    assert "|| quoted.ptvMessage" in src              # quotedPayload
    assert "video_note: 'video'" in src               # quoted dl type
    assert "|| msg.ptvMessage?.contextInfo" in src    # reply resolution


# ── Python adapter: the wire row lands as video_note ───────

def _dispatch_one(event: dict) -> list[ChatMessage]:
    adapter = WhatsAppAdapter(outbox_enabled=False)
    msgs: list[ChatMessage] = []
    adapter._dispatch(event, msgs.append)
    return msgs


def _note_event(media: list[dict]) -> dict:
    return {
        "type": "message", "id": "WAMID-NOTE-1",
        "chat": {"id": "2348012345678@c.us", "kind": "dm",
                 "title": ""},
        "from": {"id": "2348012345678@c.us", "name": "Ada"},
        "text": "", "media": media,
        "reply_to": "", "mentioned": False, "ts": 1759276800000,
    }


def test_adapter_lands_video_note_kind():
    msgs = _dispatch_one(_note_event(
        [{"path": "/cache/note.mp4", "mime": "video/mp4",
          "kind": "video_note"}]))
    assert len(msgs) == 1
    assert len(msgs[0].media) == 1
    assert msgs[0].media[0].kind == "video_note"
    assert msgs[0].media[0].is_voice is False


def test_adapter_stamps_unseen_video_note_with_its_kind():
    # The bridge could not fetch the bytes in time: the row
    # arrives pathless and the adapter stamps the arrival as
    # media_unavailable — WITH the kind, so the brain records
    # "a video note arrived, unseen", never silence.
    msgs = _dispatch_one(_note_event(
        [{"path": "", "mime": "video/mp4", "kind": "video_note"}]))
    assert len(msgs) == 1
    assert msgs[0].media == []
    assert msgs[0].meta["media_unavailable"] == [
        {"kind": "video_note", "reason": "download_failed"}]


def test_adapter_plain_video_unchanged():
    msgs = _dispatch_one(_note_event(
        [{"path": "/cache/plain.mp4", "mime": "video/mp4",
          "kind": "video"}]))
    assert msgs[0].media[0].kind == "video"


def test_brain_media_notes_name_a_video_note():
    # The brain's media-note branch (parent addition at ship):
    # a received note is context the model can see, named as
    # what it is — before the classification fix the kind was
    # "" and no note existed at all; a plain video keeps its
    # own line.
    from types import SimpleNamespace

    from nomorals.agents.partner.brain import PartnerBrain
    from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef

    brain = PartnerBrain.__new__(PartnerBrain)
    brain.context = SimpleNamespace(tools=None)
    brain._last_stt_failure_reason = ""
    chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
    note_msg = ChatMessage(
        chat=chat, incoming=True, text="", sender="a", sender_id="1",
        media=[MediaRef(path="/cache/note.mp4", kind="video_note",
                         mime="video/mp4", name="note.mp4")])
    assert brain._media_notes(note_msg) == ["video note: note.mp4"]
    video_msg = ChatMessage(
        chat=chat, incoming=True, text="", sender="a", sender_id="1",
        media=[MediaRef(path="/cache/plain.mp4", kind="video",
                         mime="video/mp4", name="plain.mp4")])
    assert brain._media_notes(video_msg) == ["video: plain.mp4"]
