"""Play & Downloads Phase 1 — handler-level demonstrator (doc 8 #1).

A song request driven through the REAL chat handler machinery —
``RuntimeMediaMixin._play_download_bg`` + ``_play_send_with_fallthrough``
+ a real ``PlaybackEngine`` + a real ``ToolRegistry`` with the real
``media_download`` tool + the real ``MediaDownloader`` chain + the
real streaming ``ResumeDownloader`` — with fakes ONLY at the network
transports (the yt-dlp seam and the kit's default HTTP transports).

The journey must show the measured stage sequence on the ack message
with real numbers: source 1 streams bytes then dies mid-transfer,
source 2 takes over (attempt/source reset — the old percentage never
pretends to describe the new transfer), the file verifies, uploads,
and is delivered.  Variants: unknown content length (bytes + speed,
NO percentage) and the same stream rendered on the WhatsApp surface.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# A stream of plausible MP3 frame headers (8192 bytes total).
_FRAME = b"\xff\xfb\x90\x00"
GOOD_BODY = _FRAME * 2048
FAIL_BODY = _FRAME * 2048
UNKNOWN_BODY = _FRAME * 1024

_BODIES = {
    "fail.test": FAIL_BODY,
    "good.test": GOOD_BODY,
    "unknown.test": UNKNOWN_BODY,
}


def _fake_head(url: str):
    host = url.split("/")[2]
    if host == "unknown.test":
        return {}  # no length, no validators — honestly unknown
    # Per-host validators: a different source is a different file
    # identity — the kit must discard source 1's partial rather than
    # splice source 2 onto it (and the journey shows exactly that).
    return {"content_length": len(_BODIES[host]), "etag": f"v1-{host}",
            "last_modified": ""}


def _fake_stream(url: str, start: int, end):
    host = url.split("/")[2]
    data = _BODIES[host][start:]
    sent = 0
    for i in range(0, len(data), 2048):
        chunk = data[i:i + 2048]
        if host == "fail.test" and sent >= 4096:
            raise ConnectionError("connection reset mid-stream")
        sent += len(chunk)
        if host != "unknown.test":
            # Real time between chunks, so the tracker's adapter-safe
            # render cadence (2s) passes more than one byte update —
            # the same coalescing a real slow transfer would see.
            time.sleep(0.7)
        yield chunk


class _FakeGateway:
    """Records sends/files/edits; plays the adapter for edit verbs."""

    def __init__(self):
        self.sent = []
        self.files = []
        self.edits = []
        self._n = 0

    def send(self, platform, chat, text, **kw):
        self._n += 1
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id=f"m{self._n}")

    def send_file(self, platform, target, path, caption="", kind=None):
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


def _chat(platform="telegram", chat_id="12345"):
    chat = mock.Mock()
    chat.platform = platform
    chat.chat_id = chat_id
    chat.key = f"{platform}:{chat_id}"
    return chat


def _context(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.tools import media as tools_media
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tmp_path),
        db=db, tools=None, extras={})
    registry = ToolRegistry(ctx)
    tools_media.register(registry)
    ctx.tools = registry
    return ctx


def _handler(tmp_path):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = _context(tmp_path)
    rt.gateway = _FakeGateway()
    return rt


def _collect(chat_key, title="Song", timeout=30.0):
    from nomorals.core.jobs import shared_registry

    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat_key, "title": title})
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


def _patched_net():
    from nomorals.media.download_kit import ResumeDownloader

    return mock.patch.multiple(
        ResumeDownloader,
        _default_head=staticmethod(_fake_head),
        _default_get_range_stream=staticmethod(_fake_stream))


def _item(host, title="Song"):
    return {"kind": "url", "path": f"https://{host}/song.mp3",
            "title": title, "artist": "A", "downloadable": True}


def _drive_fallthrough(rt, chat):
    from nomorals.media.playback import PlaybackEngine

    engine = PlaybackEngine(rt.context)
    return rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: rt._play_send_with_fallthrough(
            engine, _item("fail.test"), [_item("good.test")], [], chat,
            query="Song"),
        title="Song")


def test_song_journey_shows_measured_stages_and_honest_retry(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("yt-dlp seam down")), \
            _patched_net():
        out = _drive_fallthrough(rt, chat)
        assert out == ""  # ack posted by the bg wrapper, handler free
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    sent_path = rt.gateway.files[0][2]
    assert Path(sent_path).stat().st_size == len(GOOD_BODY)

    texts = [t for _mid, t in rt.gateway.edits]
    assert texts, "the ack was never edited with progress"
    # Real measured progress from BOTH sources: 25% of 8192 is one
    # 2048-byte chunk — it must appear for the failing source AND
    # again after the switch (the count restarted honestly).
    pct_25 = [t for t in texts if "25%" in t]
    assert len(pct_25) >= 2, f"progress did not restart per source: {texts}"
    assert any("source 2" in t for t in texts), \
        f"the source switch was never said out loud: {texts}"
    assert any("8.0 KB / 8.0 KB" in t for t in texts), \
        f"the completed transfer was never shown: {texts}"
    # Stage order on the ack: downloading → verifying → uploading.
    def _first(pred, start=0):
        for i in range(start, len(texts)):
            if pred(texts[i]):
                return i
        return -1

    i_dl = _first(lambda t: "Downloading" in t)
    i_vf = _first(lambda t: "Verifying" in t, i_dl + 1)
    i_up = _first(lambda t: "Uploading" in t, i_vf + 1)
    assert 0 <= i_dl < i_vf < i_up, f"stage order wrong: {texts}"
    # The final ack state is the delivered item's own line.
    assert texts[-1].startswith("🎵")


def test_unknown_length_journey_never_shows_a_percentage(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    url = "https://unknown.test/song.mp3"
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("yt-dlp seam down")), \
            _patched_net():
        rt._play_download_bg(
            chat, chat.key, "⏳ song.mp3",
            lambda: rt._download_produce(url, True, chat),
            title="song.mp3")
        job = _collect(chat.key, title="song.mp3")
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    texts = [t for _mid, t in rt.gateway.edits]
    progress_texts = texts[:-1]  # last edit is the delivered line
    assert progress_texts, "no progress was rendered at all"
    assert any("so far" in t for t in progress_texts), \
        f"unknown-length progress missing: {texts}"
    for t in progress_texts:
        assert "%" not in t, f"fabricated percentage for unknown length: {t}"


def test_same_event_stream_renders_on_whatsapp(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat(platform="whatsapp", chat_id="999")
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("yt-dlp seam down")), \
            _patched_net():
        _drive_fallthrough(rt, chat)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    assert rt.gateway.files[0][0] == "whatsapp"
    texts = [t for _mid, t in rt.gateway.edits]
    assert any("Downloading" in t and "%" in t for t in texts), \
        f"whatsapp surface got no measured progress: {texts}"
    assert any("source 2" in t for t in texts)
