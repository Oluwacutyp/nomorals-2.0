"""No meta-narration (owner rule, 2026-11-01, group build P0 sweep):
Devon never sends a second message narrating her own action — the
artifact IS the message. Status text only when the user asked for a
report or something failed, plainly, once.

The sweep silenced the success receipts at every send site found:
_control_tts ("voice note sent"), _deliver_voice_note ("said it"),
_control_decode ("(file sent)"), the /play send line ("▶ sent … to
this chat"), _control_caption ("captioned video sent"), /ekiti say
("reference audio sent"), /proxy's "sent the clean file" lead-in and
/artifact send's "📎 sent" line — with their pins updated in place.
Failure and degradation texts at all of those sites are unchanged:
they are information, not narration.

Presentation overhaul (owner, 2026-11-05): the background-download
message flow is now ONE message edited in place — the wrapper sends
the ack itself, the finished state REPLACES it by edit, and no
follow-up text posts at all on success.  The old delta helper
(``_play_job_delta_text``) is gone; this file pins its successor
contract at the wrapper: a successful job produces exactly
[ack send, file send, ack edit] — zero receipt sends — and the job
records delivered=True so the harvester never invents a "report
never reached this chat" failure for a delivery that landed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock


class _FakeGateway:
    """Records sends/files/edits; plays the adapter for edit verbs."""

    def __init__(self):
        self.sent = []    # (platform, chat, text)
        self.files = []   # (platform, target, path, caption)
        self.edits = []   # (message_id, text)
        self._n = 0

    def send(self, platform, chat, text, **kw):
        self._n += 1
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id=f"m{self._n}")

    def send_file(self, platform, target, path, caption=""):
        self.files.append((platform, target, path, caption))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id="f1")

    def _adapter_for(self, platform):
        return self

    def edit_message(self, chat, message_id, text, *, buttons=None):
        self.edits.append((str(message_id), text))
        return {"ok": True}

    def edit_progress(self, chat, message_id, text):
        self.edits.append((str(message_id), text))
        return {"ok": True}


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _FakeGateway()
    return rt


def _chat(chat_id):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _collect_play_job(chat_key, title="Song", timeout=10.0):
    from nomorals.core.jobs import shared_registry

    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat_key, "title": title})
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done()
    return job


def test_success_is_ack_file_edit_and_nothing_else():
    rt = _handler()
    chat = _chat("nm-1")
    out = rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: (rt.gateway.send_file(
            "telegram", "telegram:nm-1", "/tmp/song.mp3",
            caption="🎵 Song"), ("🎵 Song", True))[1],
        title="Song")
    # The handler's own return is empty: the ack already posted.
    assert out == ""
    job = _collect_play_job(chat.key)
    assert job.result["delivered"] is True
    assert job.result["ack_id"] == "m1"
    # Exactly: one ack send, one file, one edit of the ack.  No
    # follow-up send, no receipt text anywhere.
    assert [t for _p, _c, t in rt.gateway.sent] == ["⏳ Song"]
    assert len(rt.gateway.files) == 1
    assert rt.gateway.edits == [("m1", "🎵 Song")]
    for _mid, text in rt.gateway.edits:
        assert "to this chat" not in text
        assert "sources tried" not in text


def test_empty_success_text_keeps_ack_as_final_state():
    # A body whose whole outcome is the delivery itself returns
    # ("", True): nothing posts, nothing edits — the ack line
    # stands as the final state and the job records delivered.
    rt = _handler()
    chat = _chat("nm-2")
    rt._play_download_bg(
        chat, chat.key, "⏳ Song", lambda: ("", True), title="Song")
    job = _collect_play_job(chat.key)
    assert job.result["delivered"] is True
    assert [t for _p, _c, t in rt.gateway.sent] == ["⏳ Song"]
    assert rt.gateway.edits == []


def test_failure_edits_the_ack_once():
    rt = _handler()
    chat = _chat("nm-3")
    rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: ("couldn't download the audio: DRM", False),
        title="Song")
    job = _collect_play_job(chat.key)
    assert job.result["sent"] is False
    assert [t for _p, _c, t in rt.gateway.sent] == ["⏳ Song"]
    assert rt.gateway.edits == [
        ("m1", "couldn't download the audio: DRM")]


def test_no_job_id_or_queue_counts_in_any_surface_text():
    rt = _handler()
    chat = _chat("nm-4")
    rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: ("couldn't download the audio: DRM", False),
        title="Song")
    _collect_play_job(chat.key)
    import re

    surfaces = [t for _p, _c, t in rt.gateway.sent]
    surfaces += [t for _m, t in rt.gateway.edits]
    for text in surfaces:
        assert "[job " not in text
        assert not re.search(r"queued \d+ \(queue \d+\)", text)
