"""STT upload filename must always carry a Groq-supported extension.

Groq decides the file type from the MULTIPART FILENAME EXTENSION, not
the MIME. The MTProto/userbot download path produces extensionless
temp names (``tg-<millis>-0``) with ``.bin`` appended, so the upload
went out as ``tg-<millis>-0.bin`` and Groq rejected it with HTTP 400
unsupported_audio_format — even though the bytes were valid .ogg Opus.
The MIME fix (audio/ogg) was necessary but NOT sufficient.

Covers:
- .bin file with OggS magic -> upload filename ends .ogg, same bytes
  (no copy, no conversion).
- Extensionless file with OggS magic -> .ogg appended.
- .oga with OggS magic -> .ogg (``.oga`` is NOT in Groq's list).
- .ogg file -> name unchanged.
- .aac / unknown bytes -> converted to OGG via ffmpeg, never sent
  with a lying extension; honest ToolError when ffmpeg is missing.
- post_multipart honors an explicit 4-tuple filename; 3-tuples
  still use the on-disk basename (backward compatible).
- End-to-end: _stt_openai_compat with a .bin ogg file sends
  filename="*.ogg" (never .bin) and would not reproduce the 400.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.core.http import HttpClient
from nomorals.tools import audio as audio_mod

OGG_BYTES = b"OggS" + b"\x00" * 100
WAV_BYTES = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40


class SniffTest(unittest.TestCase):
    def _tmp(self, name: str, data: bytes) -> Path:
        import tempfile

        d = tempfile.mkdtemp()
        p = Path(d) / name
        p.write_bytes(data)
        return p

    def test_ogg_magic(self) -> None:
        self.assertEqual(audio_mod._sniff_audio_ext(
            self._tmp("x.bin", OGG_BYTES)), ".ogg")

    def test_wav_magic(self) -> None:
        self.assertEqual(audio_mod._sniff_audio_ext(
            self._tmp("x.bin", WAV_BYTES)), ".wav")

    def test_unknown_magic(self) -> None:
        self.assertEqual(audio_mod._sniff_audio_ext(
            self._tmp("x.bin", b"garbage-bytes-here")), "")


class UploadTargetTest(unittest.TestCase):
    def _tmp(self, name: str, data: bytes) -> Path:
        import tempfile

        d = tempfile.mkdtemp()
        p = Path(d) / name
        p.write_bytes(data)
        return p

    def test_bin_ogg_gets_ogg_name_same_file(self) -> None:
        # The exact pre-fix failure: tg-<millis>-0.bin holding Ogg Opus.
        p = self._tmp("tg-1792260000000-0.bin", OGG_BYTES)
        upload_path, upload_name = audio_mod._stt_upload_target(p)
        self.assertEqual(upload_path, p)  # no copy, no conversion
        self.assertEqual(upload_name, "tg-1792260000000-0.ogg")

    def test_extensionless_ogg_gets_ogg_name(self) -> None:
        p = self._tmp("tg-1792260000000-0", OGG_BYTES)
        _, upload_name = audio_mod._stt_upload_target(p)
        self.assertTrue(upload_name.endswith(".ogg"))
        self.assertNotIn(".bin", upload_name)

    def test_oga_sniffed_to_ogg(self) -> None:
        p = self._tmp("voice.oga", OGG_BYTES)
        _, upload_name = audio_mod._stt_upload_target(p)
        self.assertEqual(upload_name, "voice.ogg")

    def test_ogg_unchanged(self) -> None:
        p = self._tmp("voice.ogg", OGG_BYTES)
        upload_path, upload_name = audio_mod._stt_upload_target(p)
        self.assertEqual((upload_path, upload_name), (p, "voice.ogg"))

    def test_wav_unchanged(self) -> None:
        p = self._tmp("note.wav", WAV_BYTES)
        _, upload_name = audio_mod._stt_upload_target(p)
        self.assertEqual(upload_name, "note.wav")

    def test_aac_converts_not_relabels(self) -> None:
        # AAC bytes must NOT be relabelled .m4a and hoped for — convert.
        # (Bytes chosen to defeat the magic sniff, so the conversion
        # path is exercised, not the sniff path.)
        p = self._tmp("note.aac", b"not-audio-at-all-bytes-here")
        converted = Path(str(p)[:-4] + ".ogg")
        with mock.patch("nomorals.tools.audio.shutil.which",
                        return_value="/usr/bin/ffmpeg"), \
             mock.patch("nomorals.tools.audio.subprocess.run") as run:
            def _fake_run(*args, **kwargs):
                converted.write_bytes(OGG_BYTES)
                return SimpleNamespace(returncode=0, stderr=b"")
            run.side_effect = _fake_run
            upload_path, upload_name = audio_mod._stt_upload_target(p)
        self.assertEqual(upload_path, converted)
        self.assertEqual(upload_name, "note.ogg")

    def test_unsupported_no_ffmpeg_honest_error(self) -> None:
        p = self._tmp("note.aac", b"not-audio-at-all")
        with mock.patch("nomorals.tools.audio.shutil.which",
                        return_value=None):
            with self.assertRaises(Exception) as ctx:
                audio_mod._stt_upload_target(p)
        self.assertIn("ffmpeg", str(ctx.exception))


class PostMultipartFilenameTest(unittest.TestCase):
    def test_explicit_filename_used(self) -> None:
        captured: dict = {}
        import tempfile

        d = tempfile.mkdtemp()
        real = Path(d) / "tg-1-0.bin"
        real.write_bytes(OGG_BYTES)

        class _FakeResp(SimpleNamespace):
            pass

        def _fake_request(method, url, data=None, headers=None, **kw):
            captured["body"] = data
            return _FakeResp(status=200, ok=True, text="{}")

        client = HttpClient.__new__(HttpClient)
        with mock.patch.object(HttpClient, "request",
                               side_effect=_fake_request):
            client.post_multipart(
                "https://example.test/v1/audio/transcriptions",
                fields={"model": "m"},
                files=[("file", str(real), "audio/ogg",
                        "tg-1-0.ogg")])
        body = captured["body"].decode("utf-8", "replace")
        self.assertIn('filename="tg-1-0.ogg"', body)
        self.assertNotIn(".bin", body)

    def test_three_tuple_still_uses_basename(self) -> None:
        captured: dict = {}
        import tempfile

        d = tempfile.mkdtemp()
        real = Path(d) / "note.wav"
        real.write_bytes(WAV_BYTES)

        def _fake_request(method, url, data=None, headers=None, **kw):
            captured["body"] = data
            return SimpleNamespace(status=200, ok=True, text="{}")

        client = HttpClient.__new__(HttpClient)
        with mock.patch.object(HttpClient, "request",
                               side_effect=_fake_request):
            client.post_multipart(
                "https://example.test/x",
                files=[("file", str(real), "audio/wav")])
        self.assertIn('filename="note.wav"',
                      captured["body"].decode("utf-8", "replace"))


class OpenAICompatUploadTest(unittest.TestCase):
    def test_bin_ogg_file_uploads_as_ogg(self) -> None:
        import tempfile

        d = tempfile.mkdtemp()
        p = Path(d) / "tg-1792260000000-0.bin"
        p.write_bytes(OGG_BYTES)
        captured: dict = {}

        class _FakeClient:
            def __init__(self, timeout=None):
                pass

            def post_multipart(self, url, fields=None, files=None,
                               headers=None):
                captured["files"] = files
                captured["fields"] = fields
                captured["url"] = url
                return SimpleNamespace(ok=True, status=200, text="{}",
                                       json=lambda: {"text": "hello"})

        with mock.patch("nomorals.core.http.HttpClient", _FakeClient):
            text = audio_mod._stt_openai_compat(
                p, "en", base_url="https://api.groq.com/openai",
                api_key="k", model="")
        self.assertEqual(text, "hello")
        self.assertEqual(len(captured["files"]), 1)
        name, path, mime, filename = captured["files"][0]
        self.assertEqual(name, "file")
        self.assertEqual(path, str(p))
        self.assertTrue(filename.endswith(".ogg"), filename)
        self.assertNotIn(".bin", filename)
        self.assertEqual(mime, "audio/ogg")
        self.assertEqual(captured["fields"]["model"],
                         "whisper-large-v3-turbo")
