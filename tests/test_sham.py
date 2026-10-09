"""Tests for /sham music identification (AudD connector + command).

HTTP is fully mocked — no network. Audio files are temp files.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.audd import AudDConnector, AudDError
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self.status_code = status
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self, payload: Any = None, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def post_multipart(self, url: str, fields=None, files=None,
                       **kw: Any) -> FakeResponse:
        self.calls.append({"url": url, "fields": fields, "files": files})
        return FakeResponse(self.status, self.payload)


def _audio_file(size: int = 64 * 1024) -> str:
    fd, path = tempfile.mkstemp(suffix=".ogg")
    with os.fdopen(fd, "wb") as fh:
        fh.write(b"\x00" * size)
    return path


MATCH = {
    "status": "success",
    "result": {
        "artist": "Ayo Maff",
        "title": "Lifestyle (YA MAN)",
        "album": "Lifestyle",
        "release_date": "2024-01-01",
        "label": "Test Label",
        "timecode": "00:12",
        "song_link": "https://lis.tn/test",
        "apple_music": {"url": "https://music.apple.com/test"},
        "spotify": {"external_urls": {"spotify": "https://open.spotify.com/test"}},
    },
}

NO_MATCH = {"status": "success", "result": None}

AUTH_ERROR = {
    "status": "error",
    "error": {"error_code": 900, "error_message": "Invalid API token."},
}


class AudDConnectorTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("audd"), AudDConnector)

    def test_recognize_success(self) -> None:
        http = FakeHttp(MATCH)
        conn = AudDConnector(_vault(), http=http)
        conn._store_credential("u", "tok", credential_type="api_key")
        path = _audio_file()
        try:
            res = conn.recognize(path)
        finally:
            os.unlink(path)
        self.assertTrue(res["ok"])
        self.assertEqual(res["artist"], "Ayo Maff")
        self.assertEqual(res["title"], "Lifestyle (YA MAN)")
        self.assertEqual(res["album"], "Lifestyle")
        # multipart went to the right endpoint with the token + return
        call = http.calls[0]
        self.assertEqual(call["url"], "https://api.audd.io/")
        self.assertEqual(call["fields"]["api_token"], "tok")
        self.assertIn("spotify", call["fields"]["return"])

    def test_recognize_no_match(self) -> None:
        http = FakeHttp(NO_MATCH)
        conn = AudDConnector(_vault(), http=http)
        conn._store_credential("u", "tok", credential_type="api_key")
        path = _audio_file()
        try:
            res = conn.recognize(path)
        finally:
            os.unlink(path)
        self.assertFalse(res["ok"])
        self.assertIn("couldn't identify", res["reason"])

    def test_recognize_no_key(self) -> None:
        conn = AudDConnector(_vault(), http=FakeHttp(MATCH))
        res = conn.recognize(_audio_file())
        self.assertFalse(res["ok"])
        self.assertTrue(res.get("needs_key"))
        self.assertIn("free API", res["reason"])

    def test_recognize_missing_file(self) -> None:
        conn = AudDConnector(_vault(), http=FakeHttp(MATCH))
        conn._store_credential("u", "tok", credential_type="api_key")
        res = conn.recognize("/nonexistent/audio.ogg")
        self.assertFalse(res["ok"])
        self.assertIn("could not read", res["reason"])

    def test_recognize_too_large(self) -> None:
        conn = AudDConnector(_vault(), http=FakeHttp(MATCH))
        conn._store_credential("u", "tok", credential_type="api_key")
        path = _audio_file(size=11 * 1024 * 1024)
        try:
            res = conn.recognize(path)
        finally:
            os.unlink(path)
        self.assertFalse(res["ok"])
        self.assertIn("10 MB", res["reason"])

    def test_recognize_too_short(self) -> None:
        conn = AudDConnector(_vault(), http=FakeHttp(MATCH))
        conn._store_credential("u", "tok", credential_type="api_key")
        path = _audio_file(size=1024)
        try:
            res = conn.recognize(path)
        finally:
            os.unlink(path)
        self.assertFalse(res["ok"])
        self.assertIn("too short", res["reason"])

    def test_recognize_never_raises(self) -> None:
        class BoomHttp(FakeHttp):
            def post_multipart(self, *a: Any, **k: Any) -> FakeResponse:
                raise RuntimeError("network down")

        conn = AudDConnector(_vault(), http=BoomHttp())
        conn._store_credential("u", "tok", credential_type="api_key")
        path = _audio_file()
        try:
            res = conn.recognize(path)
        finally:
            os.unlink(path)
        self.assertFalse(res["ok"])

    def test_status_not_connected(self) -> None:
        conn = AudDConnector(_vault(), http=FakeHttp())
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("not connected", st.detail)

    def test_api_error_raises_audd_error(self) -> None:
        conn = AudDConnector(_vault(), http=FakeHttp(AUTH_ERROR))
        with self.assertRaises(AudDError) as ctx:
            conn._api_post("bad-key", fields={})
        self.assertEqual(ctx.exception.error_code, 900)


class ShamCommandTests(unittest.TestCase):
    """Test the /sham command wiring (audio discovery + formatting)."""

    def _mixin(self) -> Any:
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        class T(RuntimeMediaMixin):
            def __init__(self) -> None:
                self.context = mock.Mock()
                self.context.db = Database(":memory:")
                self.gateway = None

        return T()

    def _msg(self, media: Any = None, meta: Any = None) -> Any:
        m = mock.Mock()
        m.media = media or []
        m.meta = meta or {}
        return m

    def _media(self, kind: str, path: str) -> Any:
        m = mock.Mock()
        m.kind = kind
        m.path = path
        return m

    def test_find_audio_from_message_media(self) -> None:
        t = self._mixin()
        msg = self._msg(media=[self._media("audio", "/tmp/x.ogg")])
        self.assertEqual(t._sham_find_audio(msg), "/tmp/x.ogg")

    def test_find_audio_skips_non_audio(self) -> None:
        t = self._mixin()
        msg = self._msg(media=[self._media("image", "/tmp/x.jpg")])
        self.assertEqual(t._sham_find_audio(msg), "")

    def test_find_audio_replied_telegram(self) -> None:
        t = self._mixin()
        adapter = mock.Mock()
        adapter.download_file = mock.Mock(return_value="/tmp/replied.ogg")
        t.gateway = mock.Mock()
        t.gateway.adapters = {"telegram": adapter}
        msg = self._msg(meta={"replied_audio_file_id": "abc123"})
        self.assertEqual(t._sham_find_audio(msg), "/tmp/replied.ogg")
        adapter.download_file.assert_called_once_with("abc123", "sham.ogg")

    def test_find_audio_no_gateway(self) -> None:
        t = self._mixin()
        msg = self._msg(meta={"replied_audio_file_id": "abc123"})
        self.assertEqual(t._sham_find_audio(msg), "")

    def test_find_audio_never_raises(self) -> None:
        t = self._mixin()
        self.assertEqual(t._sham_find_audio(None), "")
        self.assertEqual(t._sham_find_audio(mock.Mock()), "")

    def test_identify_no_audio(self) -> None:
        t = self._mixin()
        out = t._sham_identify(self._msg())
        self.assertIn("nothing to identify", out)
        self.assertIn("/sham", out)

    def test_identify_no_key(self) -> None:
        t = self._mixin()
        os.environ.pop("NM_VAULT_PASSPHRASE", None)
        msg = self._msg(media=[self._media("audio", "/tmp/x.ogg")])
        out = t._sham_identify(msg)
        self.assertIn("free API", out)

    def test_identify_success_formats_play_link(self) -> None:
        t = self._mixin()
        msg = self._msg(media=[self._media("audio", "/tmp/x.ogg")])
        fake_res = {
            "ok": True, "artist": "Ayo Maff", "title": "Lifestyle",
            "album": "EP", "song_link": "", "spotify": None,
            "apple_music": None,
        }
        with mock.patch(
            "nomorals.connectors.registry.create_connector"
        ) as cc:
            conn = mock.Mock()
            conn.recognize = mock.Mock(return_value=fake_res)
            cc.return_value = conn
            out = t._sham_identify(msg)
        self.assertIn("Ayo Maff — Lifestyle", out)
        self.assertIn("/play Ayo Maff Lifestyle", out)

    def test_control_sham_never_raises(self) -> None:
        t = self._mixin()
        out = t._control_sham("", message=None)
        self.assertIsInstance(out, str)
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
