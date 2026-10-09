"""Tests for the music tool module (spine integration)."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _registry():
    sys.path.insert(0, "/home/hatch/workspace/devon")
    from nomorals.tools.registry import ToolRegistry
    return ToolRegistry().register_builtins()


class MusicToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _registry()

    def test_music_and_dj_registered(self):
        self.assertIn("music", self.r.names())
        self.assertIn("dj", self.r.names())

    def test_music_params(self):
        spec = self.r._tools["music"]
        self.assertIn("action", spec.parameters)

    def test_no_active_draft_paths(self):
        from nomorals.tools import music as m
        self.assertIn("no active draft", m._notebook("testkey-none"))
        self.assertIn("no active draft", m._perform("testkey-none", None))
        self.assertIn("no active draft", m._revise("x", "testkey-none", None))

    def test_song_requires_topic(self):
        from nomorals.tools import music as m
        self.assertIn("topic", m._song("", "pop", "k", None))

    def test_taste_logging(self):
        from nomorals.tools import music as m
        with patch("nomorals.media.taste.TasteStore") as TS:
            store = MagicMock()
            store.profile.liked_styles = []
            store.profile.disliked_styles = []
            TS.return_value = store
            out = m._taste("like afrobeats")
            self.assertIn("noted", out)
            store.save.assert_called_once()

    def test_taste_hint(self):
        from nomorals.tools import music as m
        p = MagicMock()
        p.liked_styles = ["afrobeats"]
        p.disliked_styles = ["slow ballads"]
        hint = m._taste_hint(p)
        self.assertIn("afrobeats", hint)
        self.assertIn("slow ballads", hint)

    def test_draft_taste_hint_param(self):
        import inspect
        from nomorals.media.song_draft import draft_song
        self.assertIn("taste_hint", inspect.signature(draft_song).parameters)

    def test_dj_mix_requires_text(self):
        from nomorals.tools import music as m
        self.assertIn("mix what", m._dj_mix(""))


if __name__ == "__main__":
    unittest.main()
