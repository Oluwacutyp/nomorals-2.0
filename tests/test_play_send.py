"""Tests for /play sending audio files in chat instead of local mpv playback.

- PlaybackEngine.download(): resolves queue items to local files
- _control_play chat path: sends via gateway instead of local play
- CLI path (no chat_key): local mpv playback preserved
- All failure paths honest, never raises, never fake success.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_context(tmpdir):
    ctx = MagicMock()
    ctx.home = Path(tmpdir)
    # tools.call for media_download
    return ctx


class DownloadTests(unittest.TestCase):
    def setUp(self):
        from nomorals.media.playback import PlaybackEngine
        self.tmp = tempfile.mkdtemp()
        self.ctx = _make_context(self.tmp)
        # PlaybackEngine needs a real-ish context; mock the db
        self.ctx.db = MagicMock()
        with patch.object(PlaybackEngine, "__init__", lambda s, c, **k: None):
            self.engine = PlaybackEngine.__new__(PlaybackEngine)
            self.engine.context = self.ctx

    def test_download_file_kind_local(self):
        # Create a real temp audio file
        audio = os.path.join(self.tmp, "song.mp3")
        Path(audio).write_bytes(b"fake-audio-data")
        item = {"kind": "file", "path": audio, "title": "Test Song"}
        result = self.engine.download(item)
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], audio)
        self.assertEqual(result["title"], "Test Song")

    def test_download_file_missing(self):
        item = {"kind": "file", "path": "/nonexistent/song.mp3", "title": "Ghost"}
        result = self.engine.download(item)
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["reason"])

    def test_download_spotify_refused(self):
        item = {"kind": "spotify", "path": "spotify:track:abc123",
                "title": "Spotify Song"}
        result = self.engine.download(item)
        self.assertFalse(result["ok"])
        self.assertIn("DRM", result["reason"])
        self.assertIn("youtube:", result["reason"])  # suggests alternative

    def test_download_youtube_uses_cache(self):
        # Mock _youtube_audio to return a cached path
        audio = os.path.join(self.tmp, "yt_cached.mp3")
        Path(audio).write_bytes(b"cached-audio")
        item = {"kind": "youtube", "path": "https://youtube.com/watch?v=abc123",
                "title": "YT Song"}
        with patch.object(self.engine, "_youtube_audio", return_value=audio):
            result = self.engine.download(item)
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], audio)

    def test_download_youtube_failure_honest(self):
        item = {"kind": "youtube", "path": "https://youtube.com/watch?v=bad",
                "title": "Bad YT"}
        with patch.object(self.engine, "_youtube_audio",
                           side_effect=Exception("yt-dlp missing")):
            result = self.engine.download(item)
        self.assertFalse(result["ok"])
        self.assertIn("YouTube download failed", result["reason"])

    def test_download_url_via_media_download(self):
        audio = os.path.join(self.tmp, "dl_song.mp3")
        Path(audio).write_bytes(b"downloaded-audio")
        mock_out = MagicMock()
        mock_out.ok = True
        mock_out.value = {"path": audio}
        self.ctx.tools.call.return_value = mock_out
        item = {"kind": "url", "path": "https://example.com/song.mp3",
                "title": "URL Song"}
        result = self.engine.download(item)
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], audio)
        self.ctx.tools.call.assert_called_once()

    def test_download_url_failure_honest(self):
        mock_out = MagicMock()
        mock_out.ok = False
        mock_out.error = "404 not found"
        self.ctx.tools.call.return_value = mock_out
        item = {"kind": "url", "path": "https://example.com/missing.mp3",
                "title": "Missing"}
        result = self.engine.download(item)
        self.assertFalse(result["ok"])
        self.assertIn("404", result["reason"])

    def test_download_no_tool_registry(self):
        self.ctx.tools = None
        item = {"kind": "url", "path": "https://example.com/song.mp3",
                "title": "No Tools"}
        result = self.engine.download(item)
        self.assertFalse(result["ok"])
        self.assertIn("tool registry", result["reason"])

    def test_download_soundcloud_resolves_then_downloads(self):
        audio = os.path.join(self.tmp, "sc_song.mp3")
        Path(audio).write_bytes(b"sc-audio")
        mock_out = MagicMock()
        mock_out.ok = True
        mock_out.value = {"path": audio}
        self.ctx.tools.call.return_value = mock_out
        item = {"kind": "soundcloud",
                "path": "https://soundcloud.com/artist/track",
                "title": "SC Song"}
        with patch.object(self.engine, "_soundcloud_play_url",
                           return_value="https://stream.url/audio.mp3"):
            result = self.engine.download(item)
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], audio)

    def test_download_never_raises_on_garbage(self):
        for bad in [None, {}, {"kind": None}, {"kind": "file"}]:
            try:
                result = self.engine.download(bad or {})
                self.assertIn("ok", result)
            except Exception as exc:
                self.fail(f"download raised on {bad!r}: {exc}")


class ChatSendTests(unittest.TestCase):
    """_control_play sends files in chat, plays locally on CLI."""

    def _make_runtime(self):
        from nomorals.agents.partner import runtime_media as rm
        rt = rm.__new__(rm.RuntimeMediaMixin if hasattr(rm, "RuntimeMediaMixin") else object)
        return rt

    def test_play_send_in_chat_success(self):
        # Find the class containing _play_send_in_chat
        import nomorals.agents.partner.runtime_media as rm_mod

        # Find the class containing _play_send_in_chat
        cls = None
        for name in dir(rm_mod):
            obj = getattr(rm_mod, name)
            if isinstance(obj, type) and hasattr(obj, "_play_send_in_chat"):
                cls = obj
                break
        self.assertIsNotNone(cls, "_play_send_in_chat not found on any class")

        inst = cls.__new__(cls)
        inst.gateway = MagicMock()
        chat = MagicMock()
        chat.platform = "telegram"
        chat.chat_id = "12345"

        engine = MagicMock()
        engine.download.return_value = {
            "ok": True, "path": "/tmp/song.mp3", "title": "Test Track"}

        result = inst._play_send_in_chat(engine, {"title": "Test Track"},
                                         "queued 1:", chat)
        self.assertIn("sent", result)
        self.assertIn("Test Track", result)
        inst.gateway.send_file.assert_called_once()
        args = inst.gateway.send_file.call_args
        self.assertEqual(args[0][0], "telegram")
        self.assertIn("/tmp/song.mp3", args[0])

    def test_play_send_in_chat_download_fails(self):
        import nomorals.agents.partner.runtime_media as rm_mod
        cls = next(obj for name in dir(rm_mod)
                   if isinstance(obj := getattr(rm_mod, name), type)
                   and hasattr(obj, "_play_send_in_chat"))
        inst = cls.__new__(cls)
        inst.gateway = MagicMock()
        chat = MagicMock()
        chat.platform = "telegram"
        chat.chat_id = "12345"

        engine = MagicMock()
        engine.download.return_value = {"ok": False, "reason": "no yt-dlp"}

        result = inst._play_send_in_chat(engine, {"title": "X"}, "queued:", chat)
        self.assertIn("couldn't send", result)
        self.assertIn("no yt-dlp", result)
        inst.gateway.send_file.assert_not_called()

    def test_play_send_in_chat_send_fails(self):
        import nomorals.agents.partner.runtime_media as rm_mod
        cls = next(obj for name in dir(rm_mod)
                   if isinstance(obj := getattr(rm_mod, name), type)
                   and hasattr(obj, "_play_send_in_chat"))
        inst = cls.__new__(cls)
        inst.gateway = MagicMock()
        inst.gateway.send_file.side_effect = Exception("network down")
        chat = MagicMock()
        chat.platform = "telegram"
        chat.chat_id = "12345"

        engine = MagicMock()
        engine.download.return_value = {
            "ok": True, "path": "/tmp/song.mp3", "title": "T"}

        result = inst._play_send_in_chat(engine, {"title": "T"}, "q:", chat)
        self.assertIn("couldn't send it", result)
        self.assertIn("network down", result)


class SignatureTests(unittest.TestCase):
    def test_control_play_accepts_chat_key(self):
        import inspect
        import nomorals.agents.partner.runtime_media as rm_mod
        cls = next(obj for name in dir(rm_mod)
                   if isinstance(obj := getattr(rm_mod, name), type)
                   and hasattr(obj, "_control_play"))
        sig = inspect.signature(cls._control_play)
        self.assertIn("chat_key", sig.parameters)


if __name__ == "__main__":
    unittest.main()
