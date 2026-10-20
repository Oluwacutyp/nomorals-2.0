"""Pass #53 — downloads + media acquisition chain audit fixes.

Findings pinned here (all reproduced failing-first on the pristine
tree):

F1  ``tools.media._download_python_api`` — when yt-dlp's prepared
    filename did not exist (the normal FFmpegExtractAudio rename), the
    fallback picked the LARGEST file in the shared download directory —
    including files from previous downloads.  A new small download
    "succeeded" with a stale, unrelated file: wrong content delivered
    as success.
F2  ``tools.media._download_cli`` — same largest-file pick after a
    zero exit code.
F3  ``tools.media._download_direct`` — an HTTP 200 HTML error page
    (soft-404) was saved as ``song.mp3`` and reported as a successful
    download.  No content check at all.
F4  ``media.downloader.MediaDownloader`` — the orchestrator trusted
    the stage result dict blindly: a path that does not exist on disk
    (or a fabricated byte count) was reported as a successful download.
F5  ``MediaDownloader._fetch_candidate`` (browser stage) — a candidate
    whose bytes were an HTML page passed whenever the server sent no
    Content-Type (or ``application/octet-stream``): no byte sniffing.
F6  Progress truth — the yt-dlp/direct progress hooks existed in
    ``tools.media`` but ``MediaDownloader`` never passed a progress
    callback, so background download jobs showed a static note for the
    whole multi-minute download.  Now wired to ``core.jobs``
    ``report_progress``.
F7  Cancellation — jobs could not be cancelled at all.  ``JobRegistry``
    gains a cooperative ``cancel()``; the download backends check the
    flag at every progress tick (yt-dlp hook, direct-HTTP chunk hook,
    CLI subprocess poll) and actually stop moving bytes.
F8  Duplicate submissions — ``/play``/``/download`` submitted a second
    background job for the same chat + title while the first was still
    running: the same file downloaded and sent twice.
F9  Restart mid-download — chat download jobs lived only in the
    in-memory registry; a process restart lost them silently and the
    chat was never told.  Submissions are now recorded in the DB and
    the harvester reports previous-boot unfinished downloads once.

Also pinned (healthy behavior, evidence for the audit questions):
Q1  the resolver chain falls through with the REAL per-source error
    recorded, and the candidate sweep isolates a dying source.
Q4  the music-registry HealthProber never fabricates: no pinger wired
    means "unknown", honestly labelled.

No external network: fake yt-dlp module/CLI at the documented seams,
a localhost HTTP server for the direct path, injected fake sessions.
"""

from __future__ import annotations

import http.server
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.core.errors import MediaError


# ── shared helpers ────────────────────────────────────────────────────


def _wait_job(registry, job_id, timeout=15.0):
    job = registry.collect(job_id, timeout=timeout)
    assert job is not None and job.done(), \
        f"job {job_id} never finished (status={job and job.status})"
    return job


# ── F1: yt-dlp python API must never return a pre-existing file ──────


class _FakeYDL:
    """Minimal yt-dlp stand-in: 'downloads' one small new file; the
    prepared filename points at the pre-postprocess name (missing),
    exactly what FFmpegExtractAudio produces in production."""

    writes_file = True

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        if type(self).writes_file:
            out_dir = Path(self.opts["outtmpl"]).parent
            (out_dir / "New Song [abc123].mp3").write_bytes(
                b"\xff\xfb" + b"n" * 96)
        return {"id": "abc123", "title": "New Song"}

    def prepare_filename(self, info):
        return str(Path(self.opts["outtmpl"]).parent
                   / "New Song [abc123].webm")


def _fake_module(writes=True):
    cls = type("FakeYDL", (_FakeYDL,), {"writes_file": writes})
    return SimpleNamespace(YoutubeDL=cls)


def test_python_api_returns_the_file_it_downloaded_not_the_largest_old_one(
        tmp_path):
    from nomorals.tools.media import _download_python_api

    (tmp_path / "Old Big [zzz999].mp3").write_bytes(b"o" * 100_000)
    result = _download_python_api(
        _fake_module(), "https://x.test/watch?v=abc123", tmp_path,
        "bestaudio/best", True, None, time.perf_counter())
    assert result["path"].endswith("New Song [abc123].mp3"), result
    assert result["bytes"] == 98


def test_python_api_no_new_file_is_an_error_not_a_stale_success(tmp_path):
    from nomorals.tools.media import _download_python_api

    (tmp_path / "Old Big [zzz999].mp3").write_bytes(b"o" * 100_000)
    with pytest.raises(MediaError):
        _download_python_api(
            _fake_module(writes=False), "https://x.test/watch?v=abc123",
            tmp_path, "bestaudio/best", True, None, time.perf_counter())


# ── F2: yt-dlp CLI must never return a pre-existing file ──────────────


class _FakeCompleted:
    def __init__(self, stdout=""):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def _cli_write_new(argv):
    out_dir = Path(argv[argv.index("-o") + 1]).parent
    new = out_dir / "Cli Song [def456].mp3"
    new.write_bytes(b"\xff\xfb" + b"c" * 64)
    return new


def _fake_run_factory():
    def _run(argv, **kw):
        new = _cli_write_new(argv)
        return _FakeCompleted(f"[download] Destination: {new}\n")
    return _run


class _FakePopen:
    """Popen stand-in whose communicate() yields the same stdout the
    fake run() produced (the implementation may use either seam)."""

    terminated = False
    killed = False

    def __init__(self, argv, **kw):
        self.argv = argv
        self.returncode = None
        self._stdout = ""

    def poll(self):
        if self.returncode is None:
            new = _cli_write_new(self.argv)
            self._stdout = f"[download] Destination: {new}\n"
            self.returncode = 0
        return self.returncode

    def communicate(self, timeout=None):
        self.poll()
        return self._stdout, ""

    def terminate(self):
        type(self).terminated = True
        self.returncode = -15

    def kill(self):
        type(self).killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_cli_returns_the_file_it_downloaded_not_the_largest_old_one(
        tmp_path):
    from nomorals.tools.media import _download_cli

    (tmp_path / "Old Big [zzz999].mp3").write_bytes(b"o" * 100_000)
    with mock.patch("nomorals.tools.media.which",
                    return_value="/usr/bin/ffmpeg"), \
            mock.patch("subprocess.run", side_effect=_fake_run_factory()), \
            mock.patch("subprocess.Popen", _FakePopen):
        result = _download_cli(
            "https://x.test/watch?v=def456", tmp_path, "bestaudio/best",
            True, 60.0, time.perf_counter())
    assert result["path"].endswith("Cli Song [def456].mp3"), result
    assert result["bytes"] == 66


# ── F3: direct HTTP download must not land an HTML error page ────────


class _DirectHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path == "/song.mp3":
            body = (b"<html><head><title>404 Not Found</title></head>"
                    b"<body><h1>Not Found</h1></body></html>")
            ctype = "text/html"
        elif self.path == "/real.mp3":
            body = b"ID3\x04\x00" + b"\xff\xfb" + b"a" * 512
            ctype = "audio/mpeg"
        else:
            body, ctype = b"nope", "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test server
        pass


@pytest.fixture()
def _direct_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             _DirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


from nomorals.core.http import HttpClient as _RealHttpClient


def _real_client_factory(**kw):
    kw.pop("timeout", None)
    return _RealHttpClient(timeout=30.0, allow_private_ips=True)


def test_direct_download_rejects_html_error_page_saved_as_mp3(
        _direct_server, tmp_path):
    from nomorals.tools.media import _download_direct

    with mock.patch("nomorals.core.http.HttpClient",
                    side_effect=_real_client_factory):
        with pytest.raises(MediaError):
            _download_direct(f"{_direct_server}/song.mp3", tmp_path,
                             None, time.perf_counter())
    assert not (tmp_path / "song.mp3").exists(), \
        "the HTML error page must not be left behind as a media file"


def test_direct_download_accepts_real_media_bytes(
        _direct_server, tmp_path):
    from nomorals.tools.media import _download_direct

    with mock.patch("nomorals.core.http.HttpClient",
                    side_effect=_real_client_factory):
        result = _download_direct(f"{_direct_server}/real.mp3", tmp_path,
                                  None, time.perf_counter())
    assert result["bytes"] == 519
    assert Path(result["path"]).read_bytes().startswith(b"ID3")


# ── F4: the orchestrator verifies the landed file ────────────────────


def _md_ctx(tmp):
    ctx = mock.MagicMock()
    ctx.home = Path(tmp)
    ctx.db = mock.MagicMock()
    ctx.settings = None
    return ctx


def test_orchestrator_rejects_phantom_success_path(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    dl = MediaDownloader(_md_ctx(tmp_path), direct_retries=1)

    def fake_download(url, dest, **kw):
        return {"url": url, "path": str(tmp_path / "ghost.mp3"),
                "bytes": 12345, "title": "Ghost", "extractor": "fake",
                "seconds": 0.1}

    with mock.patch("nomorals.tools.media.download",
                    side_effect=fake_download), \
            mock.patch.object(MediaDownloader, "_browser_session",
                              return_value=None), \
            mock.patch.dict("os.environ", {}, clear=False) as _env:
        _env.pop("NM_VAULT_PASSPHRASE", None)
        report = dl.download("https://x.test/a.mp3",
                             dest_dir=tmp_path)
    assert report.ok is False, report.summary()
    direct_logs = [s for s in report.stages if s.stage == "direct"]
    assert direct_logs and direct_logs[0].error_class == "no_media_found", \
        [(s.stage, s.error_class, s.error) for s in report.stages]


def test_orchestrator_reports_disk_truth_not_claimed_bytes(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    real = tmp_path / "real.mp3"
    real.write_bytes(b"m" * 100)
    dl = MediaDownloader(_md_ctx(tmp_path), direct_retries=1)

    def fake_download(url, dest, **kw):
        return {"url": url, "path": str(real), "bytes": 999999,
                "title": "Real", "extractor": "fake", "seconds": 0.1}

    with mock.patch("nomorals.tools.media.download",
                    side_effect=fake_download):
        report = dl.download("https://x.test/a.mp3", dest_dir=tmp_path)
    assert report.ok is True
    assert report.size_bytes == 100, report.size_bytes


# ── F5: browser stage sniffs the bytes, not just the header ─────────


def _fake_session_fetch(payload, content_type):
    class _Session:
        url = "https://x.test/page"

        def fetch_bytes(self, url, dest):
            Path(dest).write_bytes(payload)
            return {"url": url, "status": 200,
                    "content_type": content_type,
                    "bytes": len(payload), "path": str(dest)}
    return _Session()


def test_browser_stage_rejects_html_bytes_with_no_content_type(tmp_path):
    from nomorals.media.downloader import DownloadReport, MediaDownloader

    dl = MediaDownloader(_md_ctx(tmp_path))
    report = DownloadReport(url="https://x.test/page")
    html = b"<!DOCTYPE html><html><body>sign in to continue</body></html>"
    ok = dl._fetch_candidate(
        report, _fake_session_fetch(html, ""), "https://x.test/song.mp3",
        tmp_path, True, "https://x.test/page")
    assert ok is False
    assert report.ok is False
    assert not (tmp_path / "song.mp3").exists()


def test_browser_stage_accepts_media_bytes_with_no_content_type(tmp_path):
    from nomorals.media.downloader import DownloadReport, MediaDownloader

    dl = MediaDownloader(_md_ctx(tmp_path))
    report = DownloadReport(url="https://x.test/page")
    mp3 = b"ID3\x04\x00" + b"\xff\xfb" + b"a" * 256
    ok = dl._fetch_candidate(
        report, _fake_session_fetch(mp3, ""), "https://x.test/song.mp3",
        tmp_path, True, "https://x.test/page")
    assert ok is True
    assert report.size_bytes == len(mp3)


# ── F6: real progress flows into the running job ─────────────────────


def test_download_progress_reaches_job_heartbeat_note(tmp_path):
    from nomorals.core.jobs import JobRegistry
    from nomorals.media.downloader import MediaDownloader

    real = tmp_path / "song.mp3"
    real.write_bytes(b"m" * 64)
    seen = {}

    def fake_download(url, dest, **kw):
        seen["progress"] = kw.get("progress")
        if callable(seen["progress"]):
            seen["progress"]("42% of 1.00MiB")
        return {"url": url, "path": str(real), "bytes": 64,
                "title": "Song", "extractor": "fake", "seconds": 0.1}

    def body():
        dl = MediaDownloader(_md_ctx(tmp_path), direct_retries=1)
        with mock.patch("nomorals.tools.media.download",
                        side_effect=fake_download):
            return dl.download("https://x.test/song.mp3",
                               dest_dir=tmp_path)

    registry = JobRegistry()
    job = registry.submit(body, tool="play-download")
    done = _wait_job(registry, job.id)
    assert done.result.ok is True
    progress = done.snapshot()["last_progress"]
    assert "42%" in progress, \
        f"job progress note never carried the real figure: {progress!r}"


# ── F7: cooperative cancellation ─────────────────────────────────────


def test_registry_cancel_marks_job_and_refuses_terminal():
    from nomorals.core.jobs import JobRegistry

    registry = JobRegistry()
    release = threading.Event()
    job = registry.submit(lambda: release.wait(10), tool="play-download")
    try:
        assert registry.cancel(job.id) is True
        assert job.snapshot()["cancel_requested"] is True
        assert registry.cancel("job_missing") is False
    finally:
        release.set()
    done = _wait_job(registry, job.id)
    assert done.status == "done"
    assert registry.cancel(job.id) is False, \
        "a finished job cannot be cancelled — and must not claim it can"


class _CancellingYDL(_FakeYDL):
    """Fake yt-dlp that reports progress in a loop until its hook
    raises (cancel) or the loop ends (100 ticks)."""

    ticks = 0

    def extract_info(self, url, download=True):
        hook = (self.opts.get("progress_hooks") or [None])[0]
        for i in range(100):
            type(self).ticks = i + 1
            if hook is not None:
                hook({"status": "downloading",
                      "_percent_str": f"{i}%", "_total_bytes_str": "1MiB"})
            time.sleep(0.01)
        out_dir = Path(self.opts["outtmpl"]).parent
        (out_dir / "Long [abc123].mp3").write_bytes(b"\xff\xfb" + b"n" * 8)
        return {"id": "abc123", "title": "Long"}


def test_cancel_stops_a_running_python_api_download(tmp_path):
    from nomorals.core.jobs import JobRegistry
    from nomorals.tools.media import _download_python_api

    _CancellingYDL.ticks = 0
    module = SimpleNamespace(YoutubeDL=_CancellingYDL)
    registry = JobRegistry()

    def body():
        return _download_python_api(
            module, "https://x.test/watch?v=abc123", tmp_path,
            "bestaudio/best", True, lambda msg: None,
            time.perf_counter())

    job = registry.submit(body, tool="play-download")
    deadline = time.time() + 5
    while _CancellingYDL.ticks < 5 and time.time() < deadline:
        time.sleep(0.02)
    assert registry.cancel(job.id) is True
    done = _wait_job(registry, job.id)
    assert done.status == "failed", done.snapshot()
    assert "cancel" in done.error.lower(), done.error
    assert _CancellingYDL.ticks < 100, \
        "the download ran to completion despite the cancel"


# ── chat-surface helpers (same harness shape as the pass-#6 tests) ───


def _handler(db=None):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace(db=db) if db is not None \
        else SimpleNamespace()
    rt.gateway = mock.Mock()
    rt.gateway.send.return_value = mock.Mock(ok=True)
    rt.gateway.send_file.return_value = mock.Mock(ok=True)
    return rt


def _chat(chat_id):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _job_id_from_ack(ack: str) -> str:
    m = re.search(r"\[job (job[0-9a-f]+)\]", ack or "")
    assert m, f"ack carries no job id: {ack!r}"
    return m.group(1)


# ── F8: duplicate submission is refused ──────────────────────────────


def test_duplicate_chat_download_is_not_submitted_twice():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("p53-dup")
    release = threading.Event()
    calls = []

    def produce():
        calls.append(1)
        release.wait(10)
        return ("done", True)

    ack1 = rt._play_download_bg(chat, chat.key, "⏳ first", produce,
                                title="Same Song")
    assert "downloading in the background" in ack1
    ack2 = rt._play_download_bg(chat, chat.key, "⏳ second", produce,
                                title="Same Song")
    assert "already" in ack2.lower(), ack2
    active = [j for j in shared_registry().list()
              if j["labels"].get("chat") == chat.key
              and j["labels"].get("title") == "Same Song"
              and j["status"] == "running"]
    assert len(active) == 1, active
    release.set()
    _wait_job(shared_registry(), _job_id_from_ack(ack1))
    assert len(calls) == 1, "the download body ran more than once"


def test_different_title_still_submits():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("p53-dup2")
    release = threading.Event()

    def produce():
        release.wait(10)
        return ("done", True)

    ack1 = rt._play_download_bg(chat, chat.key, "⏳ a", produce,
                                title="Song A p53")
    ack2 = rt._play_download_bg(chat, chat.key, "⏳ b", produce,
                                title="Song B p53")
    assert "downloading in the background" in ack2
    assert _job_id_from_ack(ack1) != _job_id_from_ack(ack2)
    release.set()
    _wait_job(shared_registry(), _job_id_from_ack(ack1))
    _wait_job(shared_registry(), _job_id_from_ack(ack2))


# ── F7 (chat surface): /download cancel <job> ────────────────────────


def test_download_cancel_command_requests_cancellation():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("p53-cancel")
    rt._ref_from_key = lambda key: chat
    release = threading.Event()

    def produce():
        release.wait(10)
        return ("done", True)

    ack = rt._play_download_bg(chat, chat.key, "⏳ dl", produce,
                               title="Cancel Me")
    job_id = _job_id_from_ack(ack)
    reply = rt._control_download(f"cancel {job_id}", chat_key=chat.key)
    assert "cancel" in reply.lower(), reply
    job = shared_registry().poll(job_id)
    assert job is not None and job.snapshot()["cancel_requested"] is True
    release.set()
    _wait_job(shared_registry(), job_id)


def test_download_cancel_refuses_another_chats_job():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat_a = _chat("p53-cancel-a")
    chat_b = _chat("p53-cancel-b")
    chats = {chat_a.key: chat_a, chat_b.key: chat_b}
    rt._ref_from_key = lambda key: chats[key]
    release = threading.Event()

    def produce():
        release.wait(10)
        return ("done", True)

    ack = rt._play_download_bg(chat_a, chat_a.key, "⏳ dl", produce,
                               title="Not Yours")
    job_id = _job_id_from_ack(ack)
    reply = rt._control_download(f"cancel {job_id}",
                                 chat_key=chat_b.key)
    job = shared_registry().poll(job_id)
    assert job is not None and job.snapshot()["cancel_requested"] is False, \
        f"chat B cancelled chat A's job; reply was: {reply}"
    release.set()
    _wait_job(shared_registry(), job_id)


# ── F9: a restart mid-download is reported, once ────────────────────


class _SqliteDb:
    """The storage Database surface (execute/query/query_one) over
    in-memory sqlite — one instance shared by two 'processes'."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()

    def execute(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None


def test_restart_mid_download_is_reported_once_not_silently_lost():
    from nomorals.core.jobs import shared_registry

    db = _SqliteDb()
    chat = _chat("p53-restart")
    release = threading.Event()

    rt1 = _handler(db=db)
    ack = rt1._play_download_bg(
        chat, chat.key, "⏳ dl",
        lambda: (release.wait(10), ("done", True))[1],
        title="Lost Song")
    job_id = _job_id_from_ack(ack)
    rows = db.query("SELECT * FROM play_download_jobs WHERE job_id = ?",
                    (job_id,))
    assert rows and rows[0]["status"] == "running", rows

    # A new process: fresh runtime (new boot), same on-disk DB.  The
    # in-memory registry in this test process still holds the parked
    # job, but its boot id is not this process's — exactly the restart
    # signature the harvester must surface.
    rt2 = _handler(db=db)
    rt2._harvest_play_downloads(chat.key, chat)
    sends = [c.args[2] for c in rt2.gateway.send.call_args_list
             if len(c.args) > 2]
    lost = [t for t in sends if "restart" in t.lower()]
    assert len(lost) == 1 and "Lost Song" in lost[0], sends
    # Exactly once — a later harvest must not repeat the notice.
    rt2._harvest_play_downloads(chat.key, chat)
    sends2 = [c.args[2] for c in rt2.gateway.send.call_args_list
              if len(c.args) > 2]
    assert len([t for t in sends2 if "restart" in t.lower()]) == 1
    release.set()
    _wait_job(shared_registry(), job_id)


# ── Q1 evidence: resolver fall-through with real errors ──────────────


def test_resolver_chain_falls_through_with_real_errors_recorded():
    from nomorals.media.resolver import (
        ResolutionError, ResolvedAudio, SourceResolver)

    resolver = SourceResolver(None)
    result = ResolvedAudio()

    def boom():
        raise ResolutionError("netnaija-search", "HTTP 404 from netnaija")

    blocked = ResolvedAudio(
        ok=True, path_or_url="boomplay:track", title="T",
        kind="boomplay", source_name="Boomplay search",
        downloadable=False)
    good = ResolvedAudio(
        ok=True, path_or_url="https://youtube.test/watch?v=x", title="T",
        kind="youtube", source_name="YouTube search", downloadable=True)
    out = resolver._run_chain(result, [
        ("netnaija-search", boom),
        ("boomplay-search", lambda: blocked),
        ("youtube-search", lambda: good),
    ])
    assert out.ok and out.kind == "youtube"
    assert any("404" in e and "netnaija" in e for e in result.errors), \
        result.errors
    assert any("not downloadable" in a for a in result.attempts), \
        result.attempts


def test_candidate_sweep_isolates_a_dying_source():
    from nomorals.media.resolver import ResolvedAudio, SourceResolver

    resolver = SourceResolver(None)
    names = ("netnaija", "notjustok", "tooxclusive", "soundcloud",
             "bandcamp", "youtube", "youtube_music", "archiveorg",
             "musicbrainz", "openverse", "wikimedia", "itunes", "deezer")
    for name in names:
        setattr(resolver, f"_{name}_search_many",
                lambda q, limit: [])
    def dead(q, limit):
        raise RuntimeError("netnaija exploded: HTTP 503")
    resolver._netnaija_search_many = dead
    resolver._youtube_search_many = lambda q, limit: [ResolvedAudio(
        ok=True, path_or_url="https://youtube.test/watch?v=x", title="T",
        artist="A", kind="youtube", source_name="YouTube search")]
    candidates, attempts = resolver.search_candidates_detailed("T A")
    assert [c.kind for c in candidates] == ["youtube"]
    netnaija = [a for a in attempts if a.startswith("netnaija")]
    assert netnaija and "failed" in netnaija[0] \
        and "503" in netnaija[0], attempts


# ── Q4 evidence: the registry prober never fabricates ────────────────


def test_registry_health_prober_is_unknown_without_a_pinger():
    from nomorals.music_sources.health import HealthProber
    from nomorals.music_sources.registry import by_id

    prober = HealthProber()
    source = by_id("jamendo")
    assert source is not None
    status = prober.probe(source, force=True)
    assert status.status == "unknown"
    assert "no pinger" in status.detail.lower()


def test_registry_health_prober_reports_pinger_verdicts():
    from nomorals.music_sources.health import HealthProber
    from nomorals.music_sources.registry import by_id

    source = by_id("jamendo")
    down = HealthProber(ping_fn=lambda s: False).probe(source, force=True)
    assert down.status == "down"
    up = HealthProber(ping_fn=lambda s: True).probe(source, force=True)
    assert up.status == "up"
