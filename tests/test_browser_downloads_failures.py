"""Tail-110 pins for nomorals/browser/downloads.py — 3 exclusions, 0 converts.

All three sites live in DownloadQueue._run_fallback, the urllib Range
resume path. Its discipline is printed in the docstring: a partial is
only resumed when saved validators match the server's current ones —
"Never splices across a changed file."

Sites:
* 260 E104 — an unreadable resume-meta file degrades to {} ("no
  proof", per the in-code comment), so on-disk partial bytes are
  treated as unproven and the download restarts from zero. The
  failure direction is fail-SAFE: distrust, never splice. Control:
  a readable meta whose validators match resumes the partial.
* 287 E103 — on the fetch-failure path, persisting the partial +
  validators for a later resume is best-effort: an OSError there is
  swallowed BESIDE the honestly returned {"ok": False, "fetch
  failed: ..."} — pinned both with a working save and a failing one;
  the failure verdict is identical either way.
* 295 E103 — on the success path, unlinking the now-spent resume
  meta is cleanup: an OSError leaves a stale meta file, which the
  next run re-validates against the server's live validators before
  trusting (site 260's discipline). The success verdict — sha256 of
  bytes actually on disk — is already true when the guard fires.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from unittest import mock

from nomorals.browser.downloads import DownloadJob, DownloadQueue

FULL = b"PARTIAL-DATA-COMPLETE"


class _Resp:
    def __init__(self, body: bytes = b"", headers=None, status: int = 200):
        self._body = body
        self.headers = headers or {}
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _head() -> _Resp:
    return _Resp(headers={"ETag": '"v1"', "Last-Modified": "",
                          "Content-Length": str(len(FULL))})


def _queue(tmp_path: Path) -> DownloadQueue:
    return DownloadQueue(queue_path=tmp_path / "queue.json")


def _job(dest: Path) -> DownloadJob:
    return DownloadJob(url="http://x/file.bin", dest=str(dest))


# ── 260: unreadable resume meta = no proof = restart ────────────────────

def test_260_corrupt_meta_restarts_from_zero(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"PARTIAL")
    (tmp_path / "file.bin.resume.json").write_text("not json{",
                                                  encoding="utf-8")

    def opener(req, timeout=0):
        if req.get_method() == "HEAD":
            return _head()
        return _Resp(body=FULL)

    with mock.patch.object(urllib.request, "urlopen", side_effect=opener):
        res = _queue(tmp_path)._run_fallback(_job(dest))
    assert res["ok"] is True and res["resumed"] is False
    assert dest.read_bytes() == FULL


def test_260_control_matching_validators_resume(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"PARTIAL")
    (tmp_path / "file.bin.resume.json").write_text(
        json.dumps({"etag": '"v1"', "last_modified": ""}), encoding="utf-8")

    def opener(req, timeout=0):
        if req.get_method() == "HEAD":
            return _head()
        return _Resp(body=b"-DATA-COMPLETE", status=206)

    with mock.patch.object(urllib.request, "urlopen", side_effect=opener):
        res = _queue(tmp_path)._run_fallback(_job(dest))
    assert res["ok"] is True and res["resumed"] is True
    assert dest.read_bytes() == FULL


# ── 287: resume-state save on the failure path is best-effort ───────────

def _failing_get(req, timeout=0):
    if req.get_method() == "HEAD":
        return _head()
    raise ConnectionError("net down")


def test_287_fetch_failure_reported_honestly(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"PARTIAL")
    with mock.patch.object(urllib.request, "urlopen",
                           side_effect=_failing_get):
        res = _queue(tmp_path)._run_fallback(_job(dest))
    assert res["ok"] is False
    assert res["error"].startswith("fetch failed")


def test_287_save_failure_beside_honest_failure(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"PARTIAL")

    def bad_write(self, data):
        raise OSError("disk full")

    with mock.patch.object(urllib.request, "urlopen",
                           side_effect=_failing_get), \
            mock.patch.object(Path, "write_bytes", bad_write):
        res = _queue(tmp_path)._run_fallback(_job(dest))
    assert res["ok"] is False
    assert res["error"].startswith("fetch failed")


# ── 295: spent-meta cleanup on the success path ─────────────────────────

def test_295_unlink_failure_leaves_success_standing(tmp_path):
    dest = tmp_path / "file.bin"

    def opener(req, timeout=0):
        if req.get_method() == "HEAD":
            return _head()
        return _Resp(body=FULL)

    real_unlink = Path.unlink

    def bad_unlink(self, missing_ok=False):
        if self.name.endswith(".resume.json"):
            raise OSError("locked")
        return real_unlink(self, missing_ok=missing_ok)

    with mock.patch.object(urllib.request, "urlopen", side_effect=opener), \
            mock.patch.object(Path, "unlink", bad_unlink):
        res = _queue(tmp_path)._run_fallback(_job(dest))
    assert res["ok"] is True
    assert res["sha256"] == hashlib.sha256(FULL).hexdigest()
    assert dest.read_bytes() == FULL
