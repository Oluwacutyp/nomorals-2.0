"""Pass #99 — Telegram Bot API media download honours the core
``HttpResponse`` contract (and the AudD sibling drift).

The bug: ``TelegramBotAdapter._inbound_media`` and ``download_file``
read ``resp.content`` off the session response. The core ``HttpClient``
(the adapter's one session contract) returns ``HttpResponse``, which
has NO ``.content`` — the bytes live in ``.body`` (status in
``.status``, not ``.status_code``). Every download raised
AttributeError, was swallowed by the best-effort except, and surfaced
as the placeholder ``MediaRef(path="")`` — so EVERY inbound voice
note / photo / document / video failed to download and the brain
honestly reported "no audio file". The old suite missed it because
its fake session returned a requests-shaped ``SimpleNamespace``
carrying ``.content`` — a contract production never returns.

These tests drive the REAL session seam with REAL ``HttpResponse``
objects. On the pristine tree they fail (placeholder, path=""); after
the fix the bytes land on disk byte-exact. The placeholder honesty
behaviour for genuine failures (getFile down, empty file_path, HTTP
error status, transport exception) is pinned too — the fix must not
turn real failures into fake successes.

Sibling drift pinned here as well: ``AudDConnector._api_post`` read
``resp.status_code`` off a core ``HttpResponse`` on its non-JSON
error path — AttributeError instead of the intended AudDError.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from nomorals.connectors.audd import AudDConnector, AudDError
from nomorals.core.http import HttpResponse
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.telegram import TelegramBotAdapter

VOICE_BODY = b"OggS\x00\x02fake-voice-note-bytes\x7f\xff"
PHOTO_BODY = b"\xff\xd8\xff\xe0fake-jpeg-bytes\x00\x01"
DOC_BODY = b"%PDF-1.4 fake document bytes"


class _RealShapeSession:
    """Session-seam stub returning REAL core HttpResponse objects —
    the exact type the production HttpClient returns."""

    def __init__(self, resp: HttpResponse | None = None,
                 exc: Exception | None = None) -> None:
        self._resp = resp
        self._exc = exc
        self.urls: list[str] = []

    def get(self, url: str, timeout: Any = None) -> HttpResponse:
        self.urls.append(url)
        if self._exc is not None:
            raise self._exc
        assert self._resp is not None
        return self._resp


def _adapter(tmp: str, session: _RealShapeSession,
             api: Any = None) -> TelegramBotAdapter:
    adapter = TelegramBotAdapter(token="x" * 40, media_dir=tmp)
    adapter._api = api or (lambda name, **kw: {  # type: ignore[method-assign]
        "file_size": 1234, "file_path": "voice/note.ogg"})
    adapter._sess = lambda: session  # type: ignore[method-assign]
    return adapter


class HttpContractPinTest(unittest.TestCase):
    def test_http_response_shape_is_body_and_status(self) -> None:
        resp = HttpResponse(status=200, body=b"x")
        self.assertEqual(resp.body, b"x")
        self.assertEqual(resp.status, 200)
        # The requests-style attributes must NOT exist — fakes in the
        # test suite are shaped from this contract.
        self.assertFalse(hasattr(resp, "content"))
        self.assertFalse(hasattr(resp, "status_code"))


class InboundMediaRealResponseTest(unittest.TestCase):
    def test_voice_note_downloads_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=VOICE_BODY))
            adapter = _adapter(tmp, session)
            media = adapter._inbound_media({"voice": {"file_id": "v1"}},
                                           ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].kind, "voice")
            self.assertTrue(media[0].is_voice)
            self.assertEqual(media[0].mime, "audio/ogg")
            self.assertNotEqual(media[0].path, "")
            self.assertEqual(Path(media[0].path).read_bytes(), VOICE_BODY)
            self.assertEqual(len(session.urls), 1)

    def test_photo_downloads_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=PHOTO_BODY))
            adapter = _adapter(tmp, session, api=lambda name, **kw: {
                "file_size": 4321, "file_path": "photos/big.jpg"})
            media = adapter._inbound_media(
                {"photo": [{"file_id": "p1", "file_size": 10},
                           {"file_id": "p2", "file_size": 999}]},
                ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].kind, "image")
            self.assertFalse(media[0].is_voice)
            self.assertNotEqual(media[0].path, "")
            self.assertEqual(Path(media[0].path).read_bytes(), PHOTO_BODY)

    def test_document_downloads_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=DOC_BODY))
            adapter = _adapter(tmp, session, api=lambda name, **kw: {
                "file_size": 2222, "file_path": "docs/report.pdf"})
            media = adapter._inbound_media(
                {"document": {"file_id": "d1", "file_name": "report.pdf",
                              "mime_type": "application/pdf"}},
                ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].kind, "document")
            self.assertNotEqual(media[0].path, "")
            self.assertEqual(Path(media[0].path).read_bytes(), DOC_BODY)

    # ── honesty: genuine failures still yield the placeholder ──────

    def test_http_error_status_returns_placeholder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(
                HttpResponse(status=500, body=b"server error"))
            adapter = _adapter(tmp, session)
            media = adapter._inbound_media({"voice": {"file_id": "v1"}},
                                           ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].path, "")
            self.assertEqual(media[0].kind, "voice")
            self.assertTrue(media[0].is_voice)

    def test_transport_exception_returns_placeholder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(exc=ConnectionError("network down"))
            adapter = _adapter(tmp, session)
            media = adapter._inbound_media({"voice": {"file_id": "v1"}},
                                           ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].path, "")
            self.assertEqual(media[0].kind, "voice")
            self.assertTrue(media[0].is_voice)

    def test_empty_file_path_returns_placeholder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=VOICE_BODY))
            adapter = _adapter(tmp, session, api=lambda name, **kw: {
                "file_size": 1234, "file_path": ""})
            media = adapter._inbound_media({"voice": {"file_id": "v1"}},
                                           ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].path, "")
            self.assertEqual(media[0].kind, "voice")
            self.assertTrue(media[0].is_voice)
            self.assertEqual(session.urls, [])

    def test_getfile_failure_returns_placeholder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=VOICE_BODY))
            adapter = _adapter(
                tmp, session,
                api=lambda name, **kw: (_ for _ in ()).throw(
                    RuntimeError("getFile down")))
            media = adapter._inbound_media({"voice": {"file_id": "v1"}},
                                           ChatKind.DM)
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].path, "")
            self.assertEqual(media[0].kind, "voice")
            self.assertTrue(media[0].is_voice)


class DownloadFileRealResponseTest(unittest.TestCase):
    def test_download_file_writes_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=VOICE_BODY))
            adapter = _adapter(tmp, session)
            path = adapter.download_file("fileid123", "clip.ogg")
            self.assertNotEqual(path, "")
            self.assertEqual(Path(path).read_bytes(), VOICE_BODY)

    def test_download_file_http_error_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=404, body=b"nope"))
            adapter = _adapter(tmp, session)
            self.assertEqual(adapter.download_file("fileid123"), "")

    def test_download_file_transport_exception_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(exc=TimeoutError("too slow"))
            adapter = _adapter(tmp, session)
            self.assertEqual(adapter.download_file("fileid123"), "")

    def test_download_file_empty_id_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-p99-") as tmp:
            session = _RealShapeSession(HttpResponse(status=200, body=VOICE_BODY))
            adapter = _adapter(tmp, session)
            self.assertEqual(adapter.download_file(""), "")
            self.assertEqual(session.urls, [])


class AudDStatusDriftTest(unittest.TestCase):
    def test_non_json_error_path_reports_status(self) -> None:
        connector = AudDConnector.__new__(AudDConnector)

        class _Http:
            def post_multipart(self, url: str, **kw: Any) -> HttpResponse:
                return HttpResponse(status=502, body=b"<html>bad gateway")

        connector.http = _Http()  # type: ignore[attr-defined]
        with self.assertRaises(AudDError) as ctx:
            connector._api_post("token", fields={"return": "spotify"})
        self.assertIn("HTTP 502", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
