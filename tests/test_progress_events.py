"""Play & Downloads Phase 1 — one measured progress stream.

Pins for the honest-measurement contract (doc 8 Priority 1):

* an event carries op id, stage, attempt, monotonic time, bytes
  received, expected bytes when known, and the terminal result;
* unknown content length renders bytes + elapsed + speed — NO
  percentage, ever;
* ETA comes from a rolling window, not the instantaneous rate;
* segments count as segments, never mixed with bytes;
* retrieval complete is NOT delivered;
* an upload with no progress callback is indeterminate — no
  fabricated bytes;
* stalled work reads as waiting with its last-observed time;
* terminal states render promptly, stale byte updates coalesce.

Plus the streaming resume transport: bytes are reported as they
land, a mid-stream failure keeps the partial for an honest resume,
and a changed validator still discards the partial — never splices.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.core.progress_events import (
    CONNECTING,
    DOWNLOADING,
    RESOLVING,
    UPLOADING,
    VERIFYING,
    ProgressTracker,
    bind_tracker,
    current_tracker,
    emit_bytes,
    render_progress_text,
)
from nomorals.media.download_kit import ResumeDownloader


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _tracker(clock=None, renders=None, **kw):
    kw.setdefault("min_render_interval_s", 2.0)
    kw.setdefault("stall_after_s", 20.0)
    return ProgressTracker(
        "op-1", title="Song", clock=clock or _Clock(),
        on_event=(renders.append if renders is not None else None), **kw)


# ── event shape + stage vocabulary ───────────────────────────────────


def test_event_carries_the_full_measured_shape():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("YouTube")
    tr.set_stage(CONNECTING)
    clock.advance(1.0)
    tr.report_bytes(500, 1000)
    tr.finish("delivered")
    assert tr.events, "no events recorded"
    for ev in tr.events:
        assert ev.op_id == "op-1"
        assert ev.attempt >= 1
        assert ev.mono >= 0.0
    dl = [e for e in tr.events if e.stage == DOWNLOADING]
    assert dl and dl[-1].bytes_received == 500
    assert dl[-1].bytes_expected == 1000
    assert tr.events[-1].terminal == "delivered"
    monos = [e.mono for e in tr.events]
    assert monos == sorted(monos)


def test_contextvar_binding_and_noop_emitters():
    assert current_tracker() is None
    assert emit_bytes(10, 100) is False  # no tracker: honest no-op
    tr = _tracker()
    with bind_tracker(tr) as bound:
        assert bound is tr
        assert current_tracker() is tr
        assert emit_bytes(10, 100) is True
        assert tr.snapshot()["bytes_received"] == 10
    assert current_tracker() is None


# ── honest measurement rules ─────────────────────────────────────────


def test_unknown_length_renders_bytes_speed_elapsed_no_percentage():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("direct")
    tr.set_stage(CONNECTING)
    tr.report_bytes(1_000_000, None)
    clock.advance(1.0)
    tr.report_bytes(3_000_000, None)
    text = tr.render_text()
    assert "%" not in text
    assert "▓" not in text and "░" not in text
    assert "so far" in text
    assert "MB" in text  # measured bytes, named as bytes
    assert "/s" in text  # a measured speed
    snap = tr.snapshot()
    assert snap["pct"] is None
    assert snap["eta_s"] is None
    assert snap["speed_bps"] is not None and snap["speed_bps"] > 0


def test_known_length_percentage_and_bar_come_from_real_bytes():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("direct")
    tr.set_stage(CONNECTING)
    tr.report_bytes(2048, 8192)
    text = tr.render_text()
    assert "25%" in text
    assert "2.0 KB / 8.0 KB" in text
    assert "▓" in text
    assert tr.snapshot()["pct"] == 25


def test_eta_comes_from_a_rolling_window_not_the_instant():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("direct")
    tr.set_stage(CONNECTING)
    tr.report_bytes(0, 14_000_000)
    clock.advance(2.0)
    tr.report_bytes(2_000_000, 14_000_000)
    clock.advance(2.0)
    tr.report_bytes(4_000_000, 14_000_000)
    snap = tr.snapshot()
    assert snap["speed_bps"] == pytest.approx(1_000_000, rel=0.05)
    assert snap["eta_s"] == pytest.approx(10.0, abs=0.6)
    # One instantaneous spike must not collapse the ETA: the window
    # still governs (instantaneous rate here would be 50 MB/s).
    clock.advance(0.1)
    tr.report_bytes(9_000_000, 14_000_000)
    snap = tr.snapshot()
    assert snap["speed_bps"] < 3_000_000
    assert snap["eta_s"] is not None and snap["eta_s"] > 1.5


def test_segments_count_as_segments_never_as_bytes():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("hls")
    tr.set_stage(CONNECTING)
    tr.report_segments(3, 10)
    tr.report_bytes(1000, None)
    text = tr.render_text()
    assert "segments 3/10" in text
    # No percentage anywhere: bytes have no expected total and the
    # segment count must not be turned into one either.
    assert "%" not in text
    snap = tr.snapshot()
    assert snap["segments_done"] == 3 and snap["segments_total"] == 10
    assert snap["pct"] is None


def test_upload_without_a_callback_is_indeterminate():
    tr = _tracker()
    tr.begin_source("chat")
    tr.set_stage(UPLOADING)
    text = tr.render_text()
    assert "no live count" in text
    assert "%" not in text
    assert tr.snapshot()["bytes_received"] == 0


def test_retrieval_complete_is_not_delivered():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("direct")
    tr.set_stage(CONNECTING)
    tr.report_bytes(8192, 8192)
    tr.set_stage(VERIFYING)
    snap = tr.snapshot()
    assert snap["terminal"] is None
    assert "delivered" not in tr.render_text()
    tr.set_stage(UPLOADING)
    assert tr.snapshot()["terminal"] is None
    tr.finish("delivered")
    assert "delivered" in tr.render_text()


def test_stalled_work_reads_as_waiting_with_last_observed_time():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("direct")
    tr.set_stage(CONNECTING)
    tr.report_bytes(1000, 9000)
    clock.advance(25.0)  # past the 20s stall threshold, no new bytes
    snap = tr.snapshot()
    assert snap["stalled"] is True
    assert snap["terminal"] is None  # waiting, NOT failed
    assert snap["speed_bps"] is None  # a stale rate is not a rate
    assert snap["last_update_ago_s"] == pytest.approx(25.0)
    text = tr.render_text()
    assert "waiting" in text and "last update" in text


def test_failed_terminal_carries_the_error():
    tr = _tracker()
    tr.begin_source("direct")
    tr.finish("failed", error="ConnectionError: reset by peer")
    text = tr.render_text()
    assert "failed" in text and "ConnectionError" in text
    assert tr.snapshot()["terminal"] == "failed"


# ── attempts, sources, coalescing ────────────────────────────────────


def test_source_switch_resets_the_measurement_and_says_so():
    clock = _Clock()
    tr = _tracker(clock)
    tr.begin_source("first")
    tr.set_stage(CONNECTING)
    tr.report_bytes(5000, 10000)
    assert tr.snapshot()["pct"] == 50
    tr.begin_source("second")
    snap = tr.snapshot()
    assert snap["source_index"] == 2
    assert snap["attempt"] == 1
    assert snap["bytes_received"] == 0
    assert snap["pct"] is None  # the old 50% never describes this try
    assert "source 2" in tr.render_text()
    tr.begin_attempt("route: kit")
    snap = tr.snapshot()
    assert snap["attempt"] == 2 and snap["bytes_received"] == 0


def test_byte_updates_coalesce_but_terminal_is_prompt():
    clock = _Clock()
    renders = []
    tr = _tracker(clock, renders)
    tr.begin_source("direct")            # render 1 (new source)
    tr.set_stage(CONNECTING)             # render 2 (stage change)
    tr.report_bytes(100, 1000)           # render 3 (stage → downloading)
    for n in (200, 300, 400, 500):
        tr.report_bytes(n, 1000)         # inside the cadence: coalesced
    assert len(renders) == 3
    clock.advance(1.0)
    tr.report_bytes(600, 1000)           # still inside 2s of last render
    assert len(renders) == 3
    clock.advance(1.5)
    tr.report_bytes(700, 1000)           # cadence elapsed: renders
    assert len(renders) == 4
    tr.finish("delivered")               # terminal: always prompt
    assert len(renders) == 5
    assert renders[-1].terminal == "delivered"


def test_render_progress_text_is_a_pure_function_of_the_snapshot():
    snap = {
        "op_id": "x", "title": "Clip", "stage": DOWNLOADING, "attempt": 1,
        "source_index": 1, "source_label": "", "bytes_received": 4096,
        "bytes_expected": 8192, "pct": 50, "speed_bps": 2048.0,
        "eta_s": 2.0, "elapsed_s": 2.0, "stage_elapsed_s": 2.0,
        "last_update_ago_s": 0.0, "stalled": False, "segments_done": None,
        "segments_total": None, "terminal": None, "error": "", "note": "",
    }
    text = render_progress_text(snap)
    assert "50%" in text and "4.0 KB / 8.0 KB" in text


# ── streaming resume transport ───────────────────────────────────────

_BODY = bytes(range(256)) * 10  # 2560 deterministic bytes


class _Server:
    """In-memory HTTP-ish server speaking the kit's transport seam."""

    def __init__(self, body=_BODY, *, etag="v1", length=True,
                 fail_after=None):
        self.body = body
        self.etag = etag
        self.length = length
        self.fail_after = fail_after
        self.starts: list[int] = []

    def head(self, url):
        out: dict = {}
        if self.length:
            out["content_length"] = len(self.body)
        if self.etag:
            out["etag"] = self.etag
        return out

    def stream(self, url, start, end):
        self.starts.append(start)
        data = self.body[start:]
        sent = 0
        for i in range(0, len(data), 1000):
            chunk = data[i:i + 1000]
            if self.fail_after is not None \
                    and sent + len(chunk) > self.fail_after:
                keep = self.fail_after - sent
                if keep > 0:
                    yield chunk[:keep]
                raise ConnectionError("connection reset")
            sent += len(chunk)
            yield chunk


def test_streaming_reports_bytes_as_they_land(tmp_path):
    server = _Server()
    kit = ResumeDownloader(head=server.head, get_range_stream=server.stream)
    calls: list = []
    dest = tmp_path / "song.mp3"
    report = kit.download("https://x.test/song.mp3", dest,
                          progress=lambda r, e: calls.append((r, e)))
    assert report.ok, report.error
    total = len(_BODY)
    assert calls == [(1000, total), (2000, total), (total, total)]
    assert dest.read_bytes() == _BODY
    assert report.sha256 == hashlib.sha256(_BODY).hexdigest()
    assert report.size_bytes == total


def test_mid_stream_failure_keeps_partial_then_resumes(tmp_path):
    server = _Server(fail_after=1000)
    kit = ResumeDownloader(head=server.head, get_range_stream=server.stream)
    dest = tmp_path / "song.mp3"
    first = kit.download("https://x.test/song.mp3", dest)
    assert not first.ok and "connection reset" in (first.error or "")
    assert dest.stat().st_size == 1000  # the landed bytes ARE the partial
    assert first.size_bytes == 1000
    assert list(tmp_path.glob("*.resume.json")), "validators not saved"
    server.fail_after = None
    calls: list = []
    second = kit.download("https://x.test/song.mp3", dest,
                          progress=lambda r, e: calls.append((r, e)))
    assert second.ok, second.error
    assert server.starts[-1] == 1000  # resumed where it stopped
    assert second.resumed is True
    assert dest.read_bytes() == _BODY
    # Progress counts bytes ON DISK (prefix included) against the total.
    assert calls[-1] == (len(_BODY), len(_BODY))
    assert all(received >= 1000 for received, _e in calls)


def test_changed_validator_discards_partial_never_splices(tmp_path):
    server = _Server(fail_after=1000)
    kit = ResumeDownloader(head=server.head, get_range_stream=server.stream)
    dest = tmp_path / "song.mp3"
    assert not kit.download("https://x.test/song.mp3", dest).ok
    new_body = b"NEW-CONTENT" * 100
    server.body = new_body
    server.etag = "v2"
    server.fail_after = None
    report = kit.download("https://x.test/song.mp3", dest)
    assert report.ok, report.error
    assert server.starts[-1] == 0  # restarted from byte 0
    assert report.restarted is True
    assert dest.read_bytes() == new_body  # no stale prefix spliced on


def test_unknown_length_stream_reports_no_expected(tmp_path):
    server = _Server(length=False, etag=None)
    kit = ResumeDownloader(head=server.head, get_range_stream=server.stream)
    calls: list = []
    dest = tmp_path / "song.mp3"
    report = kit.download("https://x.test/song.mp3", dest,
                          progress=lambda r, e: calls.append((r, e)))
    assert report.ok, report.error
    assert calls and all(expected is None for _r, expected in calls)
    assert calls[-1][0] == len(_BODY)


def test_legacy_get_range_injection_still_works(tmp_path):
    body = b"legacy-bytes" * 50
    kit = ResumeDownloader(
        head=lambda url: {"content_length": len(body), "etag": "v1"},
        get_range=lambda url, start, end: body[start:])
    calls: list = []
    dest = tmp_path / "song.mp3"
    report = kit.download("https://x.test/song.mp3", dest,
                          progress=lambda r, e: calls.append((r, e)))
    assert report.ok, report.error
    assert dest.read_bytes() == body
    assert calls == [(len(body), len(body))]


class _FakeResponse:
    def __init__(self, body: bytes, *, fail_at: int | None = None):
        self.body = body
        self.pos = 0
        self.fail_at = fail_at
        self.read_sizes: list = []
        self.status = 200
        self.headers = {"Content-Length": str(len(body)), "ETag": "v1"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        self.read_sizes.append(size)
        if self.fail_at is not None and self.pos >= self.fail_at:
            raise ConnectionError("reset mid-body")
        if size is None or size < 0:
            size = len(self.body) - self.pos
        if self.fail_at is not None:
            # The connection delivers exactly fail_at bytes, then dies.
            size = min(size, self.fail_at - self.pos)
        chunk = self.body[self.pos:self.pos + size]
        self.pos += len(chunk)
        return chunk


def test_default_transport_reads_in_bounded_chunks(tmp_path):
    from nomorals.media import download_kit

    body = bytes(range(256)) * 40  # 10240 bytes
    responses: list[_FakeResponse] = []
    requests: list = []

    def fake_urlopen(req, timeout=None):
        requests.append(req)
        if req.get_method() == "HEAD":
            resp = _FakeResponse(b"")
            resp.headers = {"Content-Length": str(len(body)), "ETag": "v1"}
            return resp
        start = 0
        range_hdr = req.get_header("Range")
        if range_hdr:
            start = int(range_hdr.split("=")[1].split("-")[0])
        resp = _FakeResponse(body[start:])
        responses.append(resp)
        return resp

    kit = ResumeDownloader()
    dest = tmp_path / "song.mp3"
    with mock.patch.object(download_kit.urllib.request, "urlopen",
                           fake_urlopen):
        report = kit.download("https://x.test/song.mp3", dest)
    assert report.ok, report.error
    assert dest.read_bytes() == body
    assert responses, "default transport never fetched"
    sizes = responses[0].read_sizes
    assert sizes and all(isinstance(s, int) and s > 0 for s in sizes), \
        f"default transport must read in bounded chunks, got {sizes[:3]}"


def test_default_transport_resume_sends_range_header(tmp_path):
    from nomorals.media import download_kit

    body = bytes(range(256)) * 40
    state = {"fail": True}
    seen_ranges: list = []

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "HEAD":
            resp = _FakeResponse(b"")
            resp.headers = {"Content-Length": str(len(body)), "ETag": "v1"}
            return resp
        range_hdr = req.get_header("Range")
        seen_ranges.append(range_hdr)
        start = int(range_hdr.split("=")[1].split("-")[0]) if range_hdr else 0
        return _FakeResponse(body[start:],
                             fail_at=1000 if state["fail"] else None)

    kit = ResumeDownloader()
    dest = tmp_path / "song.mp3"
    with mock.patch.object(download_kit.urllib.request, "urlopen",
                           fake_urlopen):
        first = kit.download("https://x.test/song.mp3", dest)
        assert not first.ok
        assert dest.stat().st_size == 1000
        state["fail"] = False
        second = kit.download("https://x.test/song.mp3", dest)
    assert second.ok, second.error
    assert "bytes=1000-" in seen_ranges
    assert dest.read_bytes() == body


# ── extractor callback seam (yt-dlp hook figures stay structured) ────


def test_ytdlp_hook_reports_real_bytes_and_segments(tmp_path):
    from nomorals.tools import media as tools_media

    events: list = []
    segments: list = []
    target = tmp_path / "Song [abc].mp3"

    class _YDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            hook = self.options["progress_hooks"][0]
            hook({"status": "downloading", "downloaded_bytes": 100,
                  "total_bytes": 400, "fragment_index": 1,
                  "fragment_count": 4})
            # Estimate-only: the estimate must NOT become an expected
            # total — the transfer's real length is still unknown.
            hook({"status": "downloading", "downloaded_bytes": 50,
                  "total_bytes_estimate": 999})
            hook({"status": "downloading", "downloaded_bytes": 400,
                  "total_bytes": 400})
            hook({"status": "finished", "downloaded_bytes": 400,
                  "total_bytes": 400})
            target.write_bytes(b"x" * 400)
            return {"id": "abc", "title": "Song", "filepath": str(target)}

        def prepare_filename(self, info):
            return str(target)

    fake_module = SimpleNamespace(YoutubeDL=_YDL)
    out = tools_media._download_python_api(
        fake_module, "https://x.test/v", tmp_path, "bestaudio/best", True,
        None, time.perf_counter(), "",
        progress_event=lambda d, t: events.append((d, t)),
        progress_segments=lambda d, t: segments.append((d, t)))
    assert out["path"] == str(target)
    assert events == [(100, 400), (50, None), (400, 400)]
    assert segments == [(1, 4)]


# ── the orchestrator wires the stream end to end ─────────────────────


def test_downloader_chain_emits_the_stage_sequence(tmp_path):
    from nomorals.core.progress_events import bind_tracker
    from nomorals.media.downloader import MediaDownloader

    server = _Server()
    kit = ResumeDownloader(head=server.head, get_range_stream=server.stream)
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tmp_path), db=None)
    dl = MediaDownloader(
        ctx, kit_downloader=kit,
        ledger=SimpleNamespace(append=lambda **kw: True))
    clock = _Clock()
    tr = _tracker(clock, min_render_interval_s=0.0)
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("yt-dlp path down")), \
            bind_tracker(tr):
        report = dl.download("https://cdn.test/song.mp3")
    assert report.ok, report.summary()
    stages = []
    for ev in tr.events:
        if ev.stage and (not stages or stages[-1] != ev.stage):
            stages.append(ev.stage)
    assert stages[:4] == [RESOLVING, CONNECTING, DOWNLOADING, VERIFYING], \
        f"stage sequence was {stages}"
    landed = [e for e in tr.events if e.stage == DOWNLOADING]
    assert landed and landed[-1].bytes_received == len(_BODY)
    assert landed[-1].bytes_expected == len(_BODY)
    # Direct failed, kit succeeded: the route switch is a new attempt.
    assert tr.attempt >= 2
    # Retrieval finished — but nothing here may claim delivery.
    assert tr.snapshot()["terminal"] is None
    assert Path(report.path).read_bytes() == _BODY
