"""Tail-107 pins: nomorals/tools/media.py census sites (57, 350, 357, 454).

All four sites are held as EXCLUSIONS. The module downloads media
for chat surfaces; its printed failure contract is that a failed
download RAISES MediaError with honest text, and the guards below
are probes and teardown around those raises — none of them turns a
failure into a success claim.

57 — ``_cancel_requested`` is a cooperative-cancellation POLL:
the job registry cannot kill threads, so backends poll it at every
progress tick. Its docstring prints "Outside a job context this is
always False" and the handler is marked "cancellation is
best-effort": a raising jobs registry degrades to the same False
as no-job. Nothing is claimed — the download simply proceeds,
exactly as it does outside a job. Pinned both ways (raising probe,
no job, set flag, missing attr) and for non-recording.

350 — yt-dlp CLI backend, cancel path: after ``terminate()``, the
drain ``communicate(timeout=5)`` may itself raise; the handler
falls through to ``proc.kill()`` and the function raises
MediaError(CANCELLED_MESSAGE). Teardown escalation, not a swallow:
the honest cancellation error is raised either way. Pinned with a
fake process whose drain raises — kill fires, the message is the
shared cancellation text.

357 — same backend, timeout path: after ``kill()``, the drain
``communicate(timeout=5)`` raising is swallowed so the function
raises MediaError("download timed out after {timeout}s"). The
swallow protects the honest error from being masked by teardown
noise. Pinned: timeout message surfaces, kill was attempted.

454 — FLAGGED-ACCEPTED: direct-download HTML landing check. When
the downloaded bytes are an HTML page, the target file is unlinked
(best-effort cleanup — the unlink handler carries no noqa, so this
site stays in the flagged scan as an accepted remainder) and the
function raises MediaError("...returned an HTML page, not
media..."). A failing unlink strands a junk file but changes no
verdict: the raise is identical with or without the cleanup.
Pinned both ways (unlink raises / unlink succeeds and the file is
gone), plus non-recording.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.core import http as httpmod
from nomorals.core import jobs as jobsmod
from nomorals.tools import media as mediamod


# ── 57: cancellation poll — raising registry degrades to False ───

def test_57_raising_jobs_probe_returns_false():
    log = mock.Mock()
    with mock.patch.object(mediamod, "_log", log), \
            mock.patch.object(jobsmod, "current_job",
                              side_effect=RuntimeError("registry boom")):
        assert mediamod._cancel_requested() is False
    log.debug.assert_not_called()
    log.info.assert_not_called()
    log.warning.assert_not_called()
    log.error.assert_not_called()


def test_57_no_job_returns_false():
    with mock.patch.object(jobsmod, "current_job", return_value=None):
        assert mediamod._cancel_requested() is False


def test_57_cancelled_job_returns_true():
    job = SimpleNamespace(cancel_requested=True)
    with mock.patch.object(jobsmod, "current_job", return_value=job):
        assert mediamod._cancel_requested() is True


def test_57_job_without_flag_returns_false():
    with mock.patch.object(jobsmod, "current_job",
                           return_value=SimpleNamespace()):
        assert mediamod._cancel_requested() is False


# ── 350/357: yt-dlp teardown drains never mask the honest raise ──

class _FakeProc:
    """A subprocess stand-in whose slices always time out and whose
    post-terminate/kill drain communicate() raises."""

    def __init__(self):
        self.terminated = False
        self.killed = False
        self.returncode = None

    def communicate(self, timeout=None):
        if timeout and timeout > 1:  # the 5s drain call
            raise OSError("drain boom")
        raise subprocess.TimeoutExpired(cmd="yt-dlp", timeout=timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _run_cli(cancel_seq, timeout):
    proc = _FakeProc()
    seq = iter(cancel_seq)
    with tempfile.TemporaryDirectory() as td, \
            mock.patch.object(subprocess, "Popen", return_value=proc), \
            mock.patch.object(mediamod, "_cancel_requested",
                              side_effect=lambda: next(seq)), \
            mock.patch("nomorals.media.cookies.ytdlp_cookie_args_for",
                       return_value=[]):
        with pytest.raises(mediamod.MediaError) as ei:
            mediamod._download_cli("http://x/v", Path(td), "best", False,
                                   timeout, 0.0)
    return str(ei.value), proc


def test_350_cancel_drain_failure_falls_to_kill_and_raises_cancelled():
    msg, proc = _run_cli([False, True], 60.0)
    assert msg == mediamod.CANCELLED_MESSAGE
    assert proc.terminated is True
    assert proc.killed is True


def test_357_timeout_drain_failure_still_raises_timeout():
    msg, proc = _run_cli([False, False], 0.0)
    assert "timed out" in msg
    assert proc.killed is True


# ── 454: HTML landing — unlink cleanup never changes the verdict ─

class _FakeResp:
    headers = {"content-type": "text/html"}


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def request(self, method, url, stream_to=None, **kwargs):
        Path(stream_to).write_bytes(b"<html><body>parked</body></html>")
        return _FakeResp()


def test_454_unlink_failure_still_raises_html_error():
    log = mock.Mock()
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(httpmod, "HttpClient", _FakeClient), \
            mock.patch.object(mediamod, "_log", log), \
            mock.patch.object(Path, "unlink",
                              side_effect=OSError("locked")):
        with pytest.raises(mediamod.MediaError) as ei:
            mediamod._download_direct("http://x/page.mp4", Path(tmp),
                                      None, 0.0)
    assert "returned an HTML page" in str(ei.value)
    log.error.assert_not_called()
    log.warning.assert_not_called()


def test_454_control_html_error_raised_and_target_unlinked():
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(httpmod, "HttpClient", _FakeClient):
        with pytest.raises(mediamod.MediaError) as ei:
            mediamod._download_direct("http://x/page.mp4", Path(tmp),
                                      None, 0.0)
        assert "returned an HTML page" in str(ei.value)
        assert [p for p in Path(tmp).iterdir() if p.is_file()] == []
