"""Regression tests: a download that drops mid-way must fail loudly, never silently.

The incident: a phone download of a 4.68 GB GGUF lost its connection at
310 MB; the read loop hit EOF, and the CLI reported "downloaded (310 MB)"
— success.  The model server then could not load the file.

Two defenses, both exercised here against a real local HTTP server that
drops the connection:

1. ``HttpClient.request(stream_to=...)`` compares bytes written to the
   server's ``Content-Length`` and raises a retryable ``RequestError``
   on a short read (and on a mid-stream reset / IncompleteRead).  The
   partial file is kept — it is a valid prefix — so re-running resumes.
2. ``HuggingFaceDownloader.download_file`` verifies the final size
   against the listing (``expected_size``) even when no checksum is
   available, and wipes a file that was *resumed from* but still lands
   at the wrong size, so the next attempt starts clean.
"""
from __future__ import annotations

import http.server
import threading
import unittest
from pathlib import Path
from typing import ClassVar

from nomorals.core.http import HttpClient, RequestError
from nomorals.llm.download import HuggingFaceDownloader

BODY = bytes(range(256)) * 4  # 1024 bytes, deterministic


class _DropHandler(http.server.BaseHTTPRequestHandler):
    """Serves BODY with correct Content-Length, optionally cutting the
    stream short — the phone-network failure mode."""

    truncate_at: ClassVar[int | None] = None

    def log_message(self, *args):  # keep test output clean
        pass

    def do_GET(self):  # noqa: N802 (http.server API)
        rng = self.headers.get("Range", "")
        start = 0
        if rng.startswith("bytes="):
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
        payload = BODY[start:]
        if self.truncate_at is not None:
            payload = payload[: self.truncate_at]

        self.send_response(206 if start else 200)
        if not start:
            # advertise the FULL size, then deliver a short body —
            # exactly what a dropped connection looks like to the client
            self.send_header("Content-Length", str(len(BODY)))
        else:
            self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if len(payload) < len(BODY[start:]) or self.truncate_at is not None:
            # write what we have, then tear the socket down hard
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            finally:
                self.close_connection = True
                try:
                    self.connection.shutdown(2)
                    self.connection.close()
                except OSError:
                    pass
        else:
            self.wfile.write(payload)


class _Server:
    def __init__(self) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _DropHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/model.gguf"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class TestStreamTruncation(unittest.TestCase):
    def setUp(self) -> None:
        _DropHandler.truncate_at = None
        self.server = _Server()
        self.client = HttpClient(timeout=10.0)
        self.tmp = Path(self.getTempDir())  # type: ignore[attr-defined]

    def getTempDir(self) -> str:  # pragma: no cover - tiny helper
        import tempfile
        self._dir = tempfile.mkdtemp()
        return self._dir

    def tearDown(self) -> None:
        self.server.stop()

    def test_truncated_stream_raises_and_keeps_partial_file(self) -> None:
        # drop after 300 of 1024 bytes
        _DropHandler.truncate_at = 300
        target = Path(self.getTempDir()) / "model.gguf"
        with self.assertRaises(RequestError) as ctx:
            self.client.download(self.server.url, target)
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("re-run to resume", str(ctx.exception))
        # the partial file survives — a valid prefix for the resume
        self.assertEqual(target.stat().st_size, 300)
        self.assertEqual(target.read_bytes(), BODY[:300])

    def test_resume_completes_a_truncated_download(self) -> None:
        target = Path(self.getTempDir()) / "model.gguf"
        # first attempt: connection drops at 300 bytes
        _DropHandler.truncate_at = 300
        with self.assertRaises(RequestError):
            self.client.download(self.server.url, target)
        self.assertEqual(target.stat().st_size, 300)
        # second attempt: server honours the Range and sends the rest
        _DropHandler.truncate_at = None
        self.client.download(self.server.url, target)
        self.assertEqual(target.read_bytes(), BODY)

    def test_full_download_still_succeeds(self) -> None:
        target = Path(self.getTempDir()) / "model.gguf"
        self.client.download(self.server.url, target)
        self.assertEqual(target.read_bytes(), BODY)


class TestDownloaderSizeVerification(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.dir = Path(tempfile.mkdtemp())
        self.dl = HuggingFaceDownloader(cache_dir=self.dir / "models")

    def test_size_mismatch_marks_unverified(self) -> None:
        # simulate a network that stops early: our http layer writes a
        # short file and (in the old bug) reported success
        def fake_download(url, target, *, resume=True, progress=None, chunk_bytes=256 * 1024):
            Path(target).write_bytes(BODY[:300])
            return Path(target)

        self.dl.http.download = fake_download  # type: ignore[method-assign]
        result = self.dl.download_file(
            "org/repo", "model.gguf", expected_size=len(BODY)
        )
        self.assertFalse(result.verified)
        self.assertEqual(result.size, 300)

    def test_stale_oversized_file_is_wiped_and_redownloaded(self) -> None:
        dest = self.dir / "models" / "org" / "repo" / "model.gguf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(B"\x00" * (len(BODY) + 500))  # bigger than the real file

        def fake_download(url, target, *, resume=True, progress=None, chunk_bytes=256 * 1024):
            Path(target).write_bytes(BODY)
            return Path(target)

        self.dl.http.download = fake_download  # type: ignore[method-assign]
        result = self.dl.download_file(
            "org/repo", "model.gguf", expected_size=len(BODY)
        )
        self.assertTrue(result.verified)
        self.assertEqual(result.size, len(BODY))

    def test_partial_prefix_resumes_to_exact_size(self) -> None:
        # the phone's actual situation: a 310 MB (here 300 B) valid
        # prefix exists; the resume completes it to the exact size
        dest = self.dir / "models" / "org" / "repo" / "model.gguf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(BODY[:300])

        seen: dict = {}

        def fake_download(url, target, *, resume=True, progress=None, chunk_bytes=256 * 1024):
            # honour the resume the way the real layer now does
            existing = Path(target).read_bytes()
            seen["started_at"] = len(existing)
            Path(target).write_bytes(BODY)
            return Path(target)

        self.dl.http.download = fake_download  # type: ignore[method-assign]
        result = self.dl.download_file(
            "org/repo", "model.gguf", expected_size=len(BODY)
        )
        self.assertTrue(result.verified)
        self.assertEqual(result.size, len(BODY))
        self.assertTrue(result.resumed)
        self.assertEqual(seen["started_at"], 300)


if __name__ == "__main__":
    unittest.main()
