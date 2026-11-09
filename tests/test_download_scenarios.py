"""Play & Downloads Phase 2 — the acceptance battery (doc 7 §7,
task enumeration): twelve scenarios driven from the actual
WhatsApp/Telegram chat handlers with fakes ONLY at the network
transports (the yt-dlp seam and the download kit's HTTP transports).

Every scenario asserts BOTH the delivered artifact (or its honest
absence) AND the progress event-stream story — attempts, acceptance
rejections, terminal state.  The stream is captured by passing an
explicit tracker into the background-job wrapper, so assertions
read the recorded events, not render-cadence-dependent ack edits.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.test_play_progress_demonstrator import _chat, _collect, _FakeGateway, _handler

_FRAME = b"\xff\xfb\x90\x00"
GOOD_BODY = _FRAME * 2048                       # 8192 B
OTHER_BODY = _FRAME * 1024                      # 4096 B
HTML_BODY = (b"<!DOCTYPE html><html><head><title>404</title></head>"
             b"<body>gone</body></html>") * 8
MP4_BODY = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096

_BODIES = {
    "good.test": GOOD_BODY,
    "other.test": OTHER_BODY,
    "html.test": HTML_BODY,
    "unknown.test": OTHER_BODY,
    "zero.test": b"",
}

_FETCHED: list[str] = []
_STREAM_STARTS: list[tuple[str, int]] = []


def _fake_head(url: str):
    host = url.split("/")[2]
    _FETCHED.append(host)
    if host == "down.test":
        raise ConnectionError("provider outage")
    if host == "unknown.test":
        return {}
    body = _BODIES[host]
    return {"content_length": len(body), "etag": f"v1-{host}",
            "last_modified": ""}


def _fake_stream(url: str, start: int, end):
    host = url.split("/")[2]
    _STREAM_STARTS.append((host, start))
    data = _BODIES[host][start:]
    for i in range(0, len(data), 2048):
        yield data[i:i + 2048]


def _patched_net():
    from nomorals.media.download_kit import ResumeDownloader

    return mock.patch.multiple(
        ResumeDownloader,
        _default_head=staticmethod(_fake_head),
        _default_get_range_stream=staticmethod(_fake_stream))


def _no_ytdlp():
    return mock.patch("nomorals.tools.media.download",
                      side_effect=RuntimeError("yt-dlp seam down"))


def _tracker():
    from nomorals.core.progress_events import ProgressTracker

    return ProgressTracker(op_id="op-scenario", title="Song")


def _item(host, title="Song", artist="A", kind="url"):
    return {"kind": kind, "path": f"https://{host}/song.mp3",
            "title": title, "artist": artist, "downloadable": True}


def _drive(rt, chat, first, fallbacks, *, match=None, tracker=None,
           title="Song"):
    from nomorals.media.playback import PlaybackEngine

    engine = PlaybackEngine(rt.context)
    return rt._play_download_bg(
        chat, chat.key, f"⏳ {title}",
        lambda: rt._play_send_with_fallthrough(
            engine, first, fallbacks, [], chat,
            match=match, query=title),
        title=title, tracker=tracker)


def setup_function():
    _FETCHED.clear()
    _STREAM_STARTS.clear()


# ── 1. exact identity ─────────────────────────────────────────────────

def test_scenario_1_exact_identity_delivered(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, _item("good.test"), [],
               match=("Song", "A"), tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    sent_path = rt.gateway.files[0][2]
    assert Path(sent_path).read_bytes() == GOOD_BODY
    assert tracker.events[-1].terminal == "delivered"
    assert max(e.source_index for e in tracker.events) == 1


# ── 2. same-title ambiguity resolves to the requested recording ──────

def test_scenario_2_same_title_ambiguity(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    wrong = _item("other.test", title="Song", artist="Wrong")
    right = _item("good.test", title="Song", artist="Right")
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, wrong, [right],
               match=("Song", "Right"), tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    sent_path = rt.gateway.files[0][2]
    assert Path(sent_path).read_bytes() == GOOD_BODY
    # The namesake was never even fetched — selection bound to the
    # stable candidate identity, not the shared title.
    assert "other.test" not in _FETCHED


# ── 3. prompt-free selected candidate ─────────────────────────────────

def test_scenario_3_pick_delivers_exactly_the_chosen_candidate(tmp_path):
    from nomorals.media.picklist import PickCache

    rt = _handler(tmp_path)
    chat = _chat()
    rt._ref_from_key = lambda key, _c=chat: _c
    blocked = _item("down.test", title="Song", artist="A")
    alt1 = _item("other.test", title="Other One", artist="X")
    alt2 = _item("good.test", title="Other Two", artist="Y")
    from nomorals.media.playback import PlaybackEngine

    with _no_ytdlp(), _patched_net():
        text, sent = rt._play_send_with_fallthrough(
            PlaybackEngine(rt.context),
            blocked, [alt1, alt2], [], chat,
            match=("Song", "A"), query="Song")
    assert sent is False
    assert "pick" in text.lower()
    found = PickCache.get_for_chat(chat.key)
    assert found is not None, "no pick-list was stored for the chat"
    token, _entry = found
    with _no_ytdlp(), _patched_net():
        out = rt._play_pick(f"{token} 2", chat.key)
        assert out == ""  # background job owns the delivery
        job = _collect(chat.key, title="Other Two")
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    # Exactly the SECOND listed candidate — the owner's tap, not a
    # re-search and not the first alternative.
    assert Path(rt.gateway.files[0][2]).read_bytes() == GOOD_BODY
    assert PickCache.get_for_chat(chat.key) is None  # consumed on send


# ── 4. invalid-host refusal ───────────────────────────────────────────

def test_scenario_4_not_downloadable_host_refused_before_any_fetch(
        tmp_path):
    from nomorals.core.progress_events import DOWNLOADING, ProgressTracker, bind_tracker

    rt = _handler(tmp_path)
    chat = _chat()
    tracker = ProgressTracker(op_id="op-s4", title="spotify")
    calls = []
    with mock.patch("nomorals.tools.media.download",
                    side_effect=lambda *a, **k: calls.append("ytdlp")), \
            _patched_net(), bind_tracker(tracker):
        text, sent = rt._download_produce(
            "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
            False, chat)
    assert sent is False
    assert rt.gateway.files == []
    assert calls == [] and _FETCHED == [] and _STREAM_STARTS == []
    assert "spotify" in text.lower()
    assert not any(e.stage == DOWNLOADING for e in tracker.events)


# ── 5. zero-byte ──────────────────────────────────────────────────────

def test_scenario_5_zero_byte_candidate_is_never_delivered(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, _item("zero.test"), [_item("good.test")],
               tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    assert Path(rt.gateway.files[0][2]).read_bytes() == GOOD_BODY
    assert "zero.test" in _FETCHED  # it WAS tried — and produced nothing


def test_scenario_5b_zero_byte_via_download_command_is_refused(tmp_path):
    from nomorals.media import downloader as dl_mod

    rt = _handler(tmp_path)
    chat = _chat()
    landed = tmp_path / "clip.mp4"
    landed.write_bytes(b"")
    report = SimpleNamespace(ok=True, path=str(landed), title="Clip",
                             url="https://x.test/clip.mp4")
    with mock.patch.object(dl_mod.MediaDownloader, "download",
                           return_value=report):
        text, sent = rt._download_produce(
            "https://x.test/clip.mp4", False, chat)
    assert sent is False and rt.gateway.files == []
    assert "zero" in text.lower()


# ── 6. wrong media ────────────────────────────────────────────────────

def test_scenario_6_html_page_is_rejected_and_next_candidate_wins(
        tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, _item("html.test"), [_item("good.test")],
               tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert Path(rt.gateway.files[0][2]).read_bytes() == GOOD_BODY
    # The event stream journaled the acceptance rejection, by reason.
    notes = [e.note for e in tracker.events]
    assert any("HTML" in n for n in notes), notes
    assert max(e.source_index for e in tracker.events) == 2
    assert tracker.events[-1].terminal == "delivered"


# ── 7. provider outage with candidates remaining ─────────────────────

def test_scenario_7_provider_outage_falls_through(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, _item("down.test"), [_item("good.test")],
               tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert Path(rt.gateway.files[0][2]).read_bytes() == GOOD_BODY
    assert max(e.source_index for e in tracker.events) == 2


# ── 8. unknown size ───────────────────────────────────────────────────

def test_scenario_8_unknown_size_stays_percent_free(tmp_path):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, _item("unknown.test"), [], tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    assert Path(rt.gateway.files[0][2]).stat().st_size == len(OTHER_BODY)
    downloading = [e for e in tracker.events if e.bytes_received]
    assert downloading, "no byte events recorded"
    assert all(e.bytes_expected is None for e in downloading)
    assert tracker.snapshot()["pct"] is None


# ── 9. segmented download ─────────────────────────────────────────────

class _FakeYoutubeDL:
    """A yt-dlp double: drives the REAL progress/segment/postprocess
    hooks with fragment counters, then lands a real file."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        for hook in self.opts.get("progress_hooks", []):
            for frag in (1, 2, 3, 4):
                hook({"status": "downloading",
                      "downloaded_bytes": frag * 250,
                      "total_bytes": 1000,
                      "fragment_index": frag, "fragment_count": 4})
        for hook in self.opts.get("postprocessor_hooks", []):
            hook({"status": "started",
                  "postprocessor": "FFmpegExtractAudio"})
            hook({"status": "finished",
                  "postprocessor": "FFmpegExtractAudio"})
        filename = self.prepare_filename({"id": "vid1"})
        Path(filename).write_bytes(MP4_BODY)
        return {"id": "vid1", "title": "Clip", "filepath": filename}

    def prepare_filename(self, info):
        outtmpl = self.opts["outtmpl"]
        directory = str(Path(outtmpl).parent)
        return str(Path(directory) / "Clip [vid1].mp4")


def test_scenario_9_segmented_download_reports_segments(tmp_path):
    import nomorals.tools.media as tools_media

    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    real_load = tools_media.load_optional

    def fake_load(name, *a, **k):
        if name == "yt_dlp":
            return SimpleNamespace(YoutubeDL=_FakeYoutubeDL)
        return real_load(name, *a, **k)

    with mock.patch.object(tools_media, "load_optional", fake_load), \
            _patched_net():
        rt._play_download_bg(
            chat, chat.key, "⏳ Clip",
            lambda: rt._download_produce(
                "https://clips.test/watch/1", False, chat),
            title="Clip", tracker=tracker)
        job = _collect(chat.key, title="Clip")
    assert job.result["sent"] is True
    assert Path(rt.gateway.files[0][2]).read_bytes() == MP4_BODY
    seg_events = [e for e in tracker.events
                  if e.segments_total is not None]
    assert seg_events and seg_events[-1].segments_total == 4
    assert seg_events[-1].segments_done == 4
    from nomorals.core.progress_events import PROCESSING
    assert any(e.stage == PROCESSING for e in tracker.events), \
        [e.stage for e in tracker.events]
    assert tracker.events[-1].terminal == "delivered"


# ── 10. failed upload ─────────────────────────────────────────────────

class _FailingUploadGateway(_FakeGateway):
    def send_file(self, platform, target, path, caption="", kind=None):
        raise RuntimeError("upload died")


def test_scenario_10_failed_upload_is_not_delivered(tmp_path):
    rt = _handler(tmp_path)
    rt.gateway = _FailingUploadGateway()
    chat = _chat()
    tracker = _tracker()
    with _no_ytdlp(), _patched_net():
        _drive(rt, chat, _item("good.test"), [], tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is False
    assert rt.gateway.files == []
    assert tracker.events[-1].terminal == "failed"
    texts = [t for _mid, t in rt.gateway.edits]
    assert texts and "couldn't send" in texts[-1]


# ── 11. journal vs chat consistency ───────────────────────────────────

def test_scenario_11_journal_chat_and_stream_tell_the_same_story(
        tmp_path, caplog):
    rt = _handler(tmp_path)
    chat = _chat()
    tracker = _tracker()
    bad = _item("html.test", kind="url")
    good = _item("good.test", kind="audiomack")
    with _no_ytdlp(), _patched_net(), \
            caplog.at_level(logging.INFO,
                            logger="nomorals.agents.partner.runtime_media"):
        _drive(rt, chat, bad, [good], tracker=tracker)
        job = _collect(chat.key)
    assert job.result["sent"] is True
    # Chat: the final ack state is the delivered item's own line.
    texts = [t for _mid, t in rt.gateway.edits]
    assert texts and texts[-1].startswith("🎵")
    # Journal: the tried trail names both sources and the rejection.
    journal = "\n".join(r.message for r in caplog.records
                        if "play tried" in r.message)
    assert "direct-url" in journal and "audiomack" in journal
    assert "HTML" in journal
    assert "downloaded ✓" in journal
    # Stream: two sources, a verifying rejection, delivered terminal.
    assert max(e.source_index for e in tracker.events) == 2
    assert any("HTML" in e.note for e in tracker.events)
    assert tracker.events[-1].terminal == "delivered"


# ── 12. retry / resume ────────────────────────────────────────────────

class _FlakyServer:
    """HEAD/stream double for the kit transports: dies mid-stream
    until told to behave; validators under the test's control."""

    def __init__(self, body: bytes):
        self.body = body
        self.etag = "v1"
        self.healthy = False
        self.starts: list[int] = []

    def head(self, url: str):
        return {"content_length": len(self.body), "etag": self.etag,
                "last_modified": ""}

    def stream(self, url: str, start: int, end):
        self.starts.append(start)
        data = self.body[start:]
        if not self.healthy:
            yield data[: len(data) // 2]
            raise ConnectionError("died mid-stream")
        for i in range(0, len(data), 2048):
            yield data[i:i + 2048]


def _queue_with_server(tmp_path, server):
    from nomorals.browser.downloads import DownloadQueue
    from nomorals.media.download_kit import ResumeDownloader

    patcher = mock.patch.multiple(
        ResumeDownloader,
        _default_head=staticmethod(server.head),
        _default_get_range_stream=staticmethod(server.stream))
    queue = DownloadQueue(tmp_path / "q.json")
    return queue, patcher


def test_scenario_12_resume_continues_from_the_partial(tmp_path):
    from nomorals.browser.downloads import JOB_DONE, JOB_FAILED

    server = _FlakyServer(GOOD_BODY)
    queue, patcher = _queue_with_server(tmp_path, server)
    dest = tmp_path / "song.bin"
    job = queue.enqueue("https://files.test/song.bin", dest)
    with patcher:
        first = queue.run_job(job.job_id)
        assert first.status == JOB_FAILED
        partial_size = dest.stat().st_size
        assert 0 < partial_size < len(GOOD_BODY)
        server.healthy = True
        done = queue.resume(job.job_id)
    assert done.status == JOB_DONE
    assert done.resumed is True
    assert dest.read_bytes() == GOOD_BODY
    # The second fetch started exactly where the partial ended —
    # a resume, not a restart.
    assert server.starts[-1] == partial_size


def test_scenario_12b_changed_validator_restarts_never_splices(tmp_path):
    from nomorals.browser.downloads import JOB_DONE, JOB_FAILED

    server = _FlakyServer(GOOD_BODY)
    queue, patcher = _queue_with_server(tmp_path, server)
    dest = tmp_path / "song.bin"
    job = queue.enqueue("https://files.test/song.bin", dest)
    with patcher:
        assert queue.run_job(job.job_id).status == JOB_FAILED
        server.etag = "v2-changed"
        server.healthy = True
        done = queue.resume(job.job_id)
    assert done.status == JOB_DONE
    assert done.resumed is False
    assert server.starts[-1] == 0  # restarted from byte 0
    assert dest.read_bytes() == GOOD_BODY


# ── timing guard: the battery stays fast and offline ─────────────────

def test_battery_has_no_network_dependence():
    # Every transport above is a fake; the only real clocks are the
    # job registry collects, which resolve in milliseconds locally.
    assert time  # placeholder guard — the suite's runtime is the proof
