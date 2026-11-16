"""The download swarm: parallel segmented downloading.

Pins (all against fake range transports — no network):

* segment planning: exact splits, tiny files shrink the worker
  count, unknown totals plan nothing;
* reassembly: concurrent range workers land the exact original
  bytes, whole-file sha256 matches, segment files + state are
  cleaned up on success;
* resume: completed segments are re-hashed and kept (only the
  missing ranges are fetched), a validator change restarts the
  whole file, a corrupted completed segment is re-fetched;
* refusal: no Accept-Ranges / no Content-Length / a range request
  answered with the whole file / a file below the swarm threshold
  all refuse cleanly with a named reason and no partial dest;
* per-segment retry, and failure exhaustion that keeps the partial
  state for a later resume;
* the durable queue: jobs survive a new queue instance on the same
  path, and the pool drains a queued job to done;
* the MediaDownloader seam: the kit stage tries the swarm first
  when opted in, and falls back to the single-stream kit when the
  swarm refuses.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.progress_events import ProgressTracker
from nomorals.media import download_swarm as ds
from nomorals.media.download_kit import (
    ExtractorDescriptor,
    ExtractorRegistry,
    ResumeReport,
)
from nomorals.media.downloader import DownloadReport, MediaDownloader

URL = "https://cdn.example.org/files/big.mp4"
DATA = os.urandom(10 << 20)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeRangeServer:
    """A deterministic byte-range server: HEAD + ranged streams.

    ``failures`` maps a (start, end) range to the number of times
    that range's fetch raises before it succeeds.  ``ignore_range``
    answers every range request with the WHOLE body (the 200-to-a-
    Range-request refusal shape).
    """

    def __init__(self, data: bytes, *, accept_ranges: bool = True,
                 etag: str = "etag-1",
                 last_modified: str = "Mon, 01 Jan 2024 00:00:00 GMT",
                 failures: dict | None = None,
                 ignore_range: bool = False,
                 content_length: int | None = None,
                 chunk: int = 1 << 16):
        self.data = data
        self.accept_ranges = accept_ranges
        self.etag = etag
        self.last_modified = last_modified
        self.failures = dict(failures or {})
        self.ignore_range = ignore_range
        self._content_length = len(data) if content_length is None \
            else content_length
        self.chunk = chunk
        self.fetched: list[tuple[int, int | None]] = []
        self._lock = threading.Lock()

    def head(self, url: str) -> dict:
        return {
            "status": 200,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "content_length": str(self._content_length),
            "accept_ranges": "bytes" if self.accept_ranges else "none",
        }

    def get_range_stream(self, url: str, start: int, end: int | None,
                         headers: dict | None = None):
        with self._lock:
            self.fetched.append((start, end))
        key = (start, end)
        if self.failures.get(key, 0) > 0:
            self.failures[key] -= 1
            raise ConnectionError("boom (stubbed)")
        body = self.data if self.ignore_range \
            else self.data[start:(end + 1 if end is not None else None)]
        for i in range(0, len(body), self.chunk):
            yield body[i:i + self.chunk]


def _swarm(server: FakeRangeServer, **kw) -> ds.SwarmDownloader:
    return ds.SwarmDownloader(head=server.head,
                              get_range_stream=server.get_range_stream,
                              **kw)


def _ctx(tmp_path: Path):
    return SimpleNamespace(
        db=None,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)


def _direct_registry() -> ExtractorRegistry:
    return ExtractorRegistry(descriptors=[
        ExtractorDescriptor(
            name="cdn", url_patterns=(r"cdn\.example\.org",),
            extraction_rule="direct")])


# ── planning ──────────────────────────────────────────────────────────


def test_plan_segments_exact_split():
    assert ds.plan_segments(10 << 20, 4) == [
        (0, (10 << 20) // 4 - 1),
        ((10 << 20) // 4, 2 * ((10 << 20) // 4) - 1),
        (2 * ((10 << 20) // 4), 3 * ((10 << 20) // 4) - 1),
        (3 * ((10 << 20) // 4), (10 << 20) - 1),
    ]


def test_plan_segments_cover_without_gaps_or_overlap():
    for total in (1, 7, 1 << 20, (3 << 20) + 123, 100_000_003):
        for workers in (1, 2, 3, 4, 8):
            ranges = ds.plan_segments(total, workers)
            assert ranges[0][0] == 0
            assert ranges[-1][1] == total - 1
            for (_s0, e0), (s1, _e1) in zip(ranges, ranges[1:],
                                           strict=False):
                assert s1 == e0 + 1
            assert sum(e - s + 1 for s, e in ranges) == total


def test_plan_segments_small_file_shrinks_workers():
    # 2.5 MiB at the 1 MiB floor can only use 2 segments.
    ranges = ds.plan_segments((5 << 20) // 2, 8)
    assert len(ranges) == 2
    assert ranges == [(0, (5 << 20) // 4 - 1),
                      ((5 << 20) // 4, (5 << 20) // 2 - 1)]


def test_plan_segments_unknown_total_plans_nothing():
    assert ds.plan_segments(0, 4) == []
    assert ds.plan_segments(-5, 4) == []


def test_resolve_workers_prefers_tune_then_default():
    assert ds.resolve_workers(None) == 4
    assert ds.resolve_workers(None, explicit=2) == 2
    assert ds.resolve_workers(None, explicit=99) == 8
    assert ds.resolve_workers(None, explicit=0) == 1
    ctx = SimpleNamespace(extras={"tune": SimpleNamespace(
        max_concurrent_downloads=2)})
    assert ds.resolve_workers(ctx) == 2
    ctx = SimpleNamespace(extras={"tune": SimpleNamespace(
        max_concurrent_downloads=40)})
    assert ds.resolve_workers(ctx) == 8


# ── happy path + reassembly ───────────────────────────────────────────


def test_swarm_reassembles_exact_bytes_with_hash(tmp_path):
    server = FakeRangeServer(DATA)
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok, report.error
    assert report.path == str(dest)
    assert dest.read_bytes() == DATA
    assert report.size_bytes == len(DATA)
    assert report.sha256 == _sha(DATA)
    assert report.segments_total == 4
    assert report.segments_done == 4
    assert report.resumed is False
    assert report.refused_reason == ""
    # segment files + state are cleaned up on success
    assert list(tmp_path.glob("*.part-*")) == []
    assert not Path(str(dest) + ".swarm.json").exists()
    # every segment was fetched exactly once, as a range
    assert sorted(server.fetched) == sorted(ds.plan_segments(len(DATA), 4))


def test_swarm_reports_progress_on_the_segment_axis(tmp_path):
    server = FakeRangeServer(DATA)
    tracker = ProgressTracker("op-1", title="big.mp4")
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20, tracker=tracker)
    assert report.ok
    snap = tracker.snapshot()
    assert snap["segments_done"] == 4
    assert snap["segments_total"] == 4
    assert snap["bytes_received"] == len(DATA)
    assert snap["bytes_expected"] == len(DATA)
    assert snap["pct"] == 100


# ── resume ────────────────────────────────────────────────────────────


def test_swarm_resume_keeps_completed_segments(tmp_path):
    server = FakeRangeServer(DATA)
    dest = tmp_path / "big.mp4"
    swarm = _swarm(server)
    ranges = ds.plan_segments(len(DATA), 4)
    # seed two completed segments on disk + valid state, as a crashed
    # run would have left them
    state = {
        "url": URL, "total": len(DATA), "etag": "etag-1",
        "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "segments": [],
    }
    for i, (start, end) in enumerate(ranges):
        if i in (0, 2):
            part = dest.parent / (dest.name + f".part-{i}")
            chunk = DATA[start:end + 1]
            part.write_bytes(chunk)
            state["segments"].append({
                "index": i, "start": start, "end": end,
                "sha256": _sha(chunk), "complete": True})
        else:
            state["segments"].append({
                "index": i, "start": start, "end": end,
                "sha256": "", "complete": False})
    Path(str(dest) + ".swarm.json").write_text(json.dumps(state))
    report = swarm.download(URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok, report.error
    assert dest.read_bytes() == DATA
    assert report.resumed is True
    fetched = sorted(server.fetched)
    assert fetched == sorted([ranges[1], ranges[3]]), fetched


def test_swarm_resume_restarts_when_validators_change(tmp_path):
    server = FakeRangeServer(DATA, etag="etag-2")
    dest = tmp_path / "big.mp4"
    ranges = ds.plan_segments(len(DATA), 4)
    state = {
        "url": URL, "total": len(DATA), "etag": "etag-1",
        "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "segments": [],
    }
    for i, (start, end) in enumerate(ranges):
        part = dest.parent / (dest.name + f".part-{i}")
        part.write_bytes(DATA[start:end + 1])
        state["segments"].append({
            "index": i, "start": start, "end": end,
            "sha256": _sha(DATA[start:end + 1]), "complete": True})
    Path(str(dest) + ".swarm.json").write_text(json.dumps(state))
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok, report.error
    assert report.restarted is True
    assert report.resumed is False
    assert dest.read_bytes() == DATA
    assert sorted(server.fetched) == sorted(ranges)


def test_swarm_resume_refetches_a_corrupted_segment(tmp_path):
    server = FakeRangeServer(DATA)
    dest = tmp_path / "big.mp4"
    ranges = ds.plan_segments(len(DATA), 4)
    state = {
        "url": URL, "total": len(DATA), "etag": "etag-1",
        "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "segments": [],
    }
    for i, (start, end) in enumerate(ranges):
        part = dest.parent / (dest.name + f".part-{i}")
        chunk = DATA[start:end + 1]
        if i == 1:
            # corrupted on disk, state claims it complete
            part.write_bytes(b"\x00" * len(chunk))
        else:
            part.write_bytes(chunk)
        state["segments"].append({
            "index": i, "start": start, "end": end,
            "sha256": _sha(chunk), "complete": True})
    Path(str(dest) + ".swarm.json").write_text(json.dumps(state))
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok, report.error
    assert dest.read_bytes() == DATA
    assert sorted(server.fetched) == [ranges[1]]


# ── refusal ───────────────────────────────────────────────────────────


def test_swarm_refuses_when_ranges_unsupported(tmp_path):
    server = FakeRangeServer(DATA, accept_ranges=False)
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok is False
    assert "range" in report.refused_reason.lower()
    assert not dest.exists()
    assert server.fetched == []


def test_swarm_refuses_when_no_content_length(tmp_path):
    server = FakeRangeServer(DATA, content_length=0)
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok is False
    assert report.refused_reason != ""
    assert not dest.exists()


def test_swarm_refuses_when_range_answered_with_whole_file(tmp_path):
    server = FakeRangeServer(DATA, ignore_range=True)
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok is False
    assert "range" in report.refused_reason.lower()
    assert not dest.exists()
    assert list(tmp_path.glob("*.part-*")) == []


def test_swarm_refuses_small_files_below_threshold(tmp_path):
    small = os.urandom(1 << 20)
    server = FakeRangeServer(small)
    dest = tmp_path / "small.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=8 << 20)
    assert report.ok is False
    assert "below" in report.refused_reason.lower()
    assert not dest.exists()
    assert server.fetched == []


# ── retry + failure ───────────────────────────────────────────────────


def test_swarm_retries_a_flaky_segment(tmp_path):
    ranges = ds.plan_segments(len(DATA), 4)
    server = FakeRangeServer(DATA, failures={ranges[2]: 1})
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report.ok, report.error
    assert dest.read_bytes() == DATA
    assert report.segment_attempts >= 5


def test_swarm_failure_keeps_partial_state_for_resume(tmp_path):
    ranges = ds.plan_segments(len(DATA), 4)
    server = FakeRangeServer(DATA, failures={ranges[1]: 99})
    dest = tmp_path / "big.mp4"
    report = _swarm(server).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20,
        segment_retries=1)
    assert report.ok is False
    assert report.error != ""
    assert not dest.exists()
    # the partial state survives: a later run can resume from it
    state_path = Path(str(dest) + ".swarm.json")
    assert state_path.is_file()
    state = json.loads(state_path.read_text())
    done = [s for s in state["segments"] if s["complete"]]
    assert len(done) == 3
    assert state["segments"][1]["complete"] is False
    # and the resume really does finish it
    server2 = FakeRangeServer(DATA)
    report2 = _swarm(server2).download(
        URL, dest, workers=4, min_swarm_bytes=1 << 20)
    assert report2.ok and report2.resumed is True
    assert dest.read_bytes() == DATA


# ── durable queue + pool ──────────────────────────────────────────────


def test_swarm_queue_survives_a_new_instance(tmp_path):
    qpath = tmp_path / "queue.json"
    q1 = ds.SwarmQueue(path=qpath)
    first = q1.enqueue(URL, str(tmp_path / "big.mp4"))
    second = q1.enqueue("https://cdn.example.org/files/two.mp4",
                        str(tmp_path / "two.mp4"))
    assert first != second
    # same url+dest enqueued twice is the same job, not a duplicate
    assert q1.enqueue(URL, str(tmp_path / "big.mp4")) == first
    assert len(q1.list_jobs()) == 2
    q2 = ds.SwarmQueue(path=qpath)
    jobs = q2.list_jobs()
    assert [j["id"] for j in jobs] == [first, second]
    assert all(j["status"] == "queued" for j in jobs)
    q2.mark(first, "running", segments_total=4)
    q3 = ds.SwarmQueue(path=qpath)
    assert q3.get(first)["status"] == "running"
    assert q3.get(first)["segments_total"] == 4
    text = q3.status_text()
    assert "queued" in text and "running" in text
    assert URL in text


def test_swarm_pool_drains_a_queued_job(tmp_path):
    server = FakeRangeServer(DATA)
    qpath = tmp_path / "queue.json"
    queue = ds.SwarmQueue(path=qpath)
    job_id = queue.enqueue(URL, str(tmp_path / "big.mp4"))
    pool = ds.SwarmPool(queue, downloader=_swarm(server), workers=4)
    try:
        done = pool.process_one(min_swarm_bytes=1 << 20)
        assert done is not None and done["id"] == job_id
        job = queue.get(job_id)
        assert job["status"] == "done"
        assert job["segments_done"] == 4
        assert job["segments_total"] == 4
        assert job["sha256"] == _sha(DATA)
        assert Path(job["dest"]).read_bytes() == DATA
        assert pool.process_one(min_swarm_bytes=1 << 20) is None
    finally:
        pool.close()


def test_swarm_pool_marks_a_refused_job_failed(tmp_path):
    server = FakeRangeServer(DATA, accept_ranges=False)
    queue = ds.SwarmQueue(path=tmp_path / "queue.json")
    job_id = queue.enqueue(URL, str(tmp_path / "big.mp4"))
    pool = ds.SwarmPool(queue, downloader=_swarm(server), workers=4)
    try:
        done = pool.process_one(min_swarm_bytes=1 << 20)
        assert done is not None
        job = queue.get(job_id)
        assert job["status"] == "failed"
        assert "range" in (job["error"] or "").lower()
    finally:
        pool.close()


# ── MediaDownloader seam ──────────────────────────────────────────────


class _FakeSwarm:
    """Swarm-shaped stand-in for the downloader seam tests."""

    def __init__(self, data: bytes, *, refused_reason: str = ""):
        self.data = data
        self.refused_reason = refused_reason
        self.calls: list[str] = []

    def download(self, url: str, dest, **kw) -> ds.SwarmReport:
        self.calls.append(url)
        if self.refused_reason:
            return ds.SwarmReport(refused_reason=self.refused_reason)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.data)
        return ds.SwarmReport(ok=True, path=str(dest),
                              size_bytes=len(self.data),
                              segments_total=4, segments_done=4,
                              sha256=_sha(self.data))


def _fake_kit(data: bytes):
    def _download(url, dest, **kw):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
        return ResumeReport(ok=True, path=str(dest),
                            size_bytes=len(data),
                            sha256=_sha(data))
    return SimpleNamespace(download=_download)


def _kit_downloader(tmp_path, *, swarm=None, kit=None):
    return MediaDownloader(
        _ctx(tmp_path), extractors=_direct_registry(),
        kit_downloader=kit, swarm_downloader=swarm)


def test_kit_stage_tries_swarm_first_when_opted_in(tmp_path):
    fake = _FakeSwarm(DATA)
    dl = _kit_downloader(tmp_path, swarm=fake)
    report = DownloadReport(url=URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    assert dl._stage_kit(report, URL, tmp_path, False, budget,
                         swarm=True, swarm_min_bytes=1 << 20) is True
    assert fake.calls == [URL]
    assert report.ok and report.path != ""
    assert report.extractor == "download-swarm"
    assert Path(report.path).read_bytes() == DATA
    assert any(s.stage == "swarm" and s.ok for s in report.stages)


def test_kit_stage_falls_back_to_single_stream_on_refusal(tmp_path):
    fake_swarm = _FakeSwarm(DATA, refused_reason=(
        "server does not serve byte ranges"))
    fake_kit = _fake_kit(DATA)
    dl = _kit_downloader(tmp_path, swarm=fake_swarm, kit=fake_kit)
    report = DownloadReport(url=URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    assert dl._stage_kit(report, URL, tmp_path, False, budget,
                         swarm=True, swarm_min_bytes=1 << 20) is True
    assert fake_swarm.calls == [URL]
    assert report.ok and report.extractor == "download-kit"
    assert Path(report.path).read_bytes() == DATA
    assert any(s.stage == "swarm" and not s.ok for s in report.stages)
    assert any(s.stage == "kit" and s.ok for s in report.stages)


def test_kit_stage_ignores_swarm_when_not_opted_in(tmp_path):
    fake = _FakeSwarm(DATA)
    dl = _kit_downloader(tmp_path, swarm=fake, kit=_fake_kit(DATA))
    report = DownloadReport(url=URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    assert dl._stage_kit(report, URL, tmp_path, False, budget) is True
    assert fake.calls == []
    assert report.extractor == "download-kit"


def test_download_chain_reaches_swarm_end_to_end(tmp_path):
    fake = _FakeSwarm(DATA)
    dl = _kit_downloader(tmp_path, swarm=fake, kit=_fake_kit(DATA))
    dl._stage_direct = lambda *a, **k: False
    report = dl.download(URL, dest_dir=tmp_path, swarm=True,
                         swarm_min_bytes=1 << 20)
    assert report.ok, report.summary()
    assert report.extractor == "download-swarm"
    assert Path(report.path).read_bytes() == DATA
    assert fake.calls == [URL]


# ── /download command paths ───────────────────────────────────────────


def _handler(tmp_path):
    from unittest import mock

    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace(
        extras={"swarm_queue_path": str(tmp_path / "q.json")})
    rt.gateway = mock.Mock()
    rt.gateway.sent = []
    rt.gateway.edits = []
    rt.gateway.files = []
    return rt


def test_download_swarm_status_command(tmp_path):
    rt = _handler(tmp_path)
    out = rt._control_download("swarm")
    assert "download swarm" in out
    assert "nothing queued" in out


def test_download_swarm_flag_reaches_downloader(tmp_path):
    from unittest import mock

    from nomorals.core.jobs import shared_registry

    rt = _handler(tmp_path)
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = "swarm-1"
    chat.key = "telegram:swarm-1"
    rt._ref_from_key = lambda key: chat
    target = tmp_path / "clip.mp3"
    target.write_bytes(b"x" * 2048)
    report = SimpleNamespace(
        ok=True, path=str(target), title="Clip", size_bytes=2048,
        stages=[], hint="")
    with mock.patch(
            "nomorals.media.downloader.MediaDownloader") as MD:
        MD.return_value.download.return_value = report
        out = rt._control_download(
            "https://example.com/clip.mp3 --swarm", chat_key=chat.key)
        assert out == ""
        assert MD.return_value.download.call_args.kwargs["swarm"] is True
    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat.key,
                "title": "https://example.com/clip.mp3"})
    assert snaps, "no play job for the swarm download"
    job = shared_registry().collect(snaps[-1]["id"], timeout=10)
    assert job is not None and job.done()
    assert job.result["delivered"] is True
