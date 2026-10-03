"""Tests for the musicwire track: profile-aware synth backends, shared
streaming wiring, Spotify/YouTube playback resolution.

Offline by default: no network, no fluidsynth, no yt-dlp required.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class SynthBackendChoiceTest(unittest.TestCase):
    def setUp(self):
        for var in ("NM_SYNTH_BACKEND", "NM_SOUNDFONT"):
            os.environ.pop(var, None)

    def tearDown(self):
        for var in ("NM_SYNTH_BACKEND", "NM_SOUNDFONT"):
            os.environ.pop(var, None)

    def test_forced_builtin(self):
        from nomorals.media.synth_backend import choose_synth
        os.environ["NM_SYNTH_BACKEND"] = "builtin"
        choice = choose_synth()
        self.assertEqual(choice.name, "builtin")
        self.assertIn("forced", choice.reason)

    def test_forced_fluidsynth_missing_binary_fails_honest(self):
        from nomorals.media.synth_backend import choose_synth
        os.environ["NM_SYNTH_BACKEND"] = "fluidsynth"
        with mock.patch("nomorals.media.synth_backend._fluidsynth_binary",
                        return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                choose_synth()
        self.assertIn("fluidsynth", str(ctx.exception).lower())

    def test_forced_fluidsynth_missing_soundfont_fails_honest(self):
        from nomorals.media.synth_backend import choose_synth
        os.environ["NM_SYNTH_BACKEND"] = "fluidsynth"
        with mock.patch("nomorals.media.synth_backend._fluidsynth_binary",
                        return_value="/usr/bin/fluidsynth"), \
             mock.patch("nomorals.media.synth_backend.find_soundfont",
                        return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                choose_synth()
        self.assertIn("soundfont", str(ctx.exception).lower())

    def test_light_profile_gets_builtin(self):
        from nomorals.media import synth_backend as sb
        with mock.patch.object(sb, "_profile_kind", return_value="termux"):
            choice = sb.choose_synth()
        self.assertEqual(choice.name, "builtin")
        self.assertEqual(choice.profile, "termux")

    def test_workstation_without_fluidsynth_gets_builtin(self):
        from nomorals.media import synth_backend as sb
        with mock.patch.object(sb, "_profile_kind",
                               return_value="workstation"), \
             mock.patch.object(sb, "_fluidsynth_binary", return_value=""):
            choice = sb.choose_synth()
        self.assertEqual(choice.name, "builtin")
        self.assertEqual(choice.profile, "workstation")

    def test_workstation_with_fluidsynth_no_soundfont_offers(self):
        from nomorals.media import synth_backend as sb
        with mock.patch.object(sb, "_profile_kind",
                               return_value="workstation"), \
             mock.patch.object(sb, "_fluidsynth_binary",
                               return_value="/usr/bin/fluidsynth"), \
             mock.patch.object(sb, "find_soundfont", return_value=""):
            choice = sb.choose_synth()
        self.assertEqual(choice.name, "builtin")  # graceful fallback
        self.assertIn("soundfont install", choice.note)

    def test_workstation_with_both_uses_fluidsynth(self):
        from nomorals.media import synth_backend as sb
        with mock.patch.object(sb, "_profile_kind",
                               return_value="workstation"), \
             mock.patch.object(sb, "_fluidsynth_binary",
                               return_value="/usr/bin/fluidsynth"), \
             mock.patch.object(sb, "find_soundfont",
                               return_value="/x/font.sf2"):
            choice = sb.choose_synth()
        self.assertEqual(choice.name, "fluidsynth")
        self.assertEqual(choice.soundfont, "/x/font.sf2")

    def test_find_soundfont_env(self):
        from nomorals.media.synth_backend import find_soundfont
        with tempfile.NamedTemporaryFile(suffix=".sf2",
                                         delete=False) as fh:
            path = fh.name
        try:
            os.environ["NM_SOUNDFONT"] = path
            self.assertEqual(find_soundfont(), path)
        finally:
            os.unlink(path)

    def test_soundfont_offer_shape(self):
        from nomorals.media.synth_backend import soundfont_offer
        offer = soundfont_offer()
        self.assertTrue(offer["url"].startswith("https://"))
        self.assertGreater(offer["size_mb"], 1)
        self.assertEqual(len(offer["sha256"]), 64)
        self.assertIn("install", offer["install_command"])

    def test_render_falls_back_to_builtin(self):
        """FluidSynth render failure → builtin WAV, never an exception."""
        from nomorals.media import synth_backend as sb
        parts = {"melody": []}
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.wav")
            with mock.patch.object(sb, "choose_synth") as mc, \
                 mock.patch.object(sb, "_render_fluidsynth",
                                   side_effect=RuntimeError("boom")):
                from nomorals.media.synth_backend import SynthChoice
                mc.return_value = SynthChoice(
                    name="fluidsynth", profile="workstation",
                    soundfont="/x/font.sf2", reason="test")
                choice = sb.render_wav(parts, 120.0, 42, "/x/song.mid",
                                       out)
            self.assertEqual(choice.name, "builtin")
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


class YouTubeHelpersTest(unittest.TestCase):
    def test_detect_source_youtube(self):
        from nomorals.media.playback import PlaybackEngine
        self.assertEqual(
            PlaybackEngine.detect_source(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "youtube")
        self.assertEqual(
            PlaybackEngine.detect_source("https://youtu.be/dQw4w9WgXcQ"),
            "youtube")
        self.assertEqual(
            PlaybackEngine.detect_source("youtube:lofi hip hop"),
            "youtube")

    def test_detect_source_unchanged_others(self):
        from nomorals.media.playback import PlaybackEngine
        self.assertEqual(
            PlaybackEngine.detect_source("spotify:track:abc"), "spotify")
        self.assertEqual(
            PlaybackEngine.detect_source(
                "https://soundcloud.com/a/b"), "soundcloud")
        self.assertEqual(
            PlaybackEngine.detect_source("https://example.com/x.mp3"),
            "url")
        self.assertEqual(
            PlaybackEngine.detect_source("song.mp3"), "file")

    def test_youtube_id_extraction(self):
        from nomorals.media.playback import PlaybackEngine
        self.assertEqual(
            PlaybackEngine._youtube_id(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ")
        self.assertEqual(
            PlaybackEngine._youtube_id("https://youtu.be/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ")
        self.assertEqual(
            PlaybackEngine._youtube_id(
                "https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ")
        self.assertEqual(
            PlaybackEngine._youtube_id("youtube:lofi"), "")

    def test_youtube_search_without_ytdlp_fails_honest(self):
        from nomorals.media.playback import PlaybackEngine
        with mock.patch.dict(sys.modules, {"yt_dlp": None}), \
             mock.patch("shutil.which", return_value=""):
            # force the ImportError path for the module check
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "yt_dlp":
                    raise ImportError("no yt_dlp")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(Exception) as ctx:
                    PlaybackEngine._youtube_search_id("lofi")
        self.assertIn("yt-dlp", str(ctx.exception))


class ForcedPrefixTest(unittest.TestCase):
    def test_looks_like_spotify_uri(self):
        from nomorals.agents.partner.runtime_media import (
            _looks_like_spotify_uri)
        self.assertTrue(_looks_like_spotify_uri("spotify:track:4uLU6hMCjMI75M1A2tKUQ"))
        self.assertTrue(_looks_like_spotify_uri(
            "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQ"))
        self.assertFalse(_looks_like_spotify_uri("spotify:never gonna give you up"))
        self.assertFalse(_looks_like_spotify_uri("hotel california"))


class WiringHelpersTest(unittest.TestCase):
    def test_spotify_link_help_honest(self):
        from nomorals.connectors.wiring import spotify_link_help
        msg = spotify_link_help()
        self.assertIn("isn't linked", msg)
        self.assertIn("nm connectors connect", msg)
        self.assertNotIn("token", msg.lower().replace(
            "tokens stay in your vault", ""))

    def test_spotify_not_linked_without_vault(self):
        from nomorals.connectors.wiring import spotify_linked
        os.environ.pop("NM_VAULT_PASSPHRASE", None)
        ctx = mock.Mock()
        ctx.spotify_adapter = None
        self.assertFalse(spotify_linked(ctx))


if __name__ == "__main__":
    unittest.main()
