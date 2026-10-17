"""Telegram inbound audio mapping (Gaps 1–2): audio must reach STT.

Gap 1 (Bot API): audio sent as a file arrives as ``document`` with
``mime_type`` audio/* — it used to map to kind="document", mime="",
matching NO STT check, so the note fell through entirely.
Gap 2 (MTProto): downloads were kind="file" unless the mime looked
like an image, and the MIME map only knew .mp3/.ogg among audio —
.m4a/.opus/.wav/.aac (and extension-less downloads) all failed STT.
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.telegram import (
    TelegramBotAdapter, _kind_for_mime, _mime_for, _mime_for_download)
from nomorals.voice.pingpong import has_voice_media


def _stt_checks(media: Any) -> bool:
    """The exact STT admission checks used by _spoken_command,
    has_voice_media and _media_notes."""
    kind = str(getattr(media, "kind", "") or "")
    mime = str(getattr(media, "mime", "") or "")
    return kind in ("audio", "voice") or mime.startswith("audio")


class MimeMapTest(unittest.TestCase):
    FORMATS = {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".opus": "audio/opus",
        ".oga": "audio/ogg",
        ".wav": "audio/wav",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".amr": "audio/amr",
        ".3gp": "audio/3gpp",
        ".aiff": "audio/aiff",
    }

    def test_every_voice_format_maps_to_audio_mime(self) -> None:
        for ext, want in self.FORMATS.items():
            with self.subTest(ext=ext):
                self.assertEqual(_mime_for(f"note{ext}"), want)

    def test_kind_for_mime(self) -> None:
        self.assertEqual(_kind_for_mime("audio/ogg"), "audio")
        self.assertEqual(_kind_for_mime("audio/mp4"), "audio")
        self.assertEqual(_kind_for_mime("image/jpeg"), "image")
        self.assertEqual(_kind_for_mime("video/mp4"), "video")
        self.assertEqual(_kind_for_mime("application/octet-stream"), "file")

    def test_mtproto_voice_note_in_each_format_passes_stt(self) -> None:
        for ext in self.FORMATS:
            with self.subTest(ext=ext):
                mime = _mime_for(f"note{ext}")
                media = SimpleNamespace(kind=_kind_for_mime(mime), mime=mime)
                self.assertTrue(_stt_checks(media))
                self.assertTrue(has_voice_media(
                    SimpleNamespace(media=[media])))

    def test_extensionless_download_falls_back_to_document_mime(self) -> None:
        self.assertEqual(
            _mime_for_download("tg-123-0.bin", "audio/ogg"), "audio/ogg")
        self.assertEqual(
            _mime_for_download("tg-123-0", "audio/mpeg"), "audio/mpeg")
        # A real extension still wins over the document mime.
        self.assertEqual(
            _mime_for_download("note.ogg", "audio/mpeg"), "audio/ogg")
        # No hint anywhere stays octet-stream.
        self.assertEqual(
            _mime_for_download("tg-123-0.bin", ""),
            "application/octet-stream")


class BotApiInboundMediaTest(unittest.TestCase):
    def _adapter(self, tmp: str) -> TelegramBotAdapter:
        adapter = TelegramBotAdapter(token="x" * 40, media_dir=tmp)
        adapter._api = lambda name, **kw: {  # type: ignore[method-assign]
            "file_size": 1234, "file_path": "voice/note.ogg"}
        resp = SimpleNamespace(content=b"OGGDATA")
        resp.raise_for_status = lambda: None
        adapter._sess = lambda: SimpleNamespace(get=lambda *a, **k: resp)  # type: ignore[method-assign]
        return adapter

    def test_document_with_audio_mime_is_audio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter = self._adapter(tmp)
            msg = {"document": {"file_id": "abc123",
                                "file_name": "voice.ogg",
                                "mime_type": "audio/ogg"}}
            media = adapter._inbound_media(msg, ChatKind.DM)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0].kind, "audio")
        self.assertEqual(media[0].mime, "audio/ogg")
        self.assertTrue(_stt_checks(media[0]))
        self.assertTrue(has_voice_media(SimpleNamespace(media=media)))

    def test_document_with_pdf_mime_stays_document(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter = self._adapter(tmp)
            msg = {"document": {"file_id": "abc123",
                                "file_name": "doc.pdf",
                                "mime_type": "application/pdf"}}
            media = adapter._inbound_media(msg, ChatKind.DM)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0].kind, "document")
        self.assertEqual(media[0].mime, "application/pdf")
        self.assertFalse(_stt_checks(media[0]))

    def test_voice_and_audio_branches_carry_mime(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter = self._adapter(tmp)
            voice = adapter._inbound_media(
                {"voice": {"file_id": "v1"}}, ChatKind.DM)
            self.assertEqual(voice[0].kind, "audio")
            self.assertEqual(voice[0].mime, "audio/ogg")
            self.assertTrue(_stt_checks(voice[0]))
            audio = adapter._inbound_media(
                {"audio": {"file_id": "a1", "mime_type": "audio/mpeg"}},
                ChatKind.DM)
            self.assertEqual(audio[0].kind, "audio")
            self.assertEqual(audio[0].mime, "audio/mpeg")
            self.assertTrue(_stt_checks(audio[0]))


if __name__ == "__main__":
    unittest.main()
