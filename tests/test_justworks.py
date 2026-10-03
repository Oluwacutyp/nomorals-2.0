"""Tests for the "just works" improvements: fuzzy style matching, proxy_file fix, file_send defaults."""
import unittest

from nomorals.media.music import resolve_style, STYLES


class TestFuzzyStyleMatching(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(resolve_style("afrobeats").name, "afrobeats")
        self.assertEqual(resolve_style("pop").name, "pop")

    def test_case_insensitive(self):
        self.assertEqual(resolve_style("AFROBEATS").name, "afrobeats")
        self.assertEqual(resolve_style("  Pop  ").name, "pop")

    def test_fuzzy_typo(self):
        # "ambent" is a typo for "ambient"
        self.assertEqual(resolve_style("ambent").name, "ambient")

    def test_fuzzy_dream_to_ambient(self):
        # "dream" should resolve to ambient via vibe map
        self.assertEqual(resolve_style("dream").name, "ambient")

    def test_vibe_words(self):
        self.assertEqual(resolve_style("chill").name, "lofi")
        self.assertEqual(resolve_style("dreamy").name, "ambient")
        self.assertEqual(resolve_style("party").name, "dancehall")

    def test_hip_hop_with_space(self):
        self.assertEqual(resolve_style("hip hop").name, "hiphop")

    def test_empty_defaults_to_pop(self):
        self.assertEqual(resolve_style("").name, "pop")
        self.assertEqual(resolve_style(None).name, "pop")

    def test_unknown_still_raises(self):
        from nomorals.core.errors import ToolError
        with self.assertRaises(ToolError):
            resolve_style("xyzzy_nonexistent_style_qqq")


class TestProxyFileFix(unittest.TestCase):
    def test_tool_proxy_file_uses_registry(self):
        """_tool_proxy_file must not call non-existent self._call."""
        import inspect
        from nomorals.agents.devon import DevonAgent
        src = inspect.getsource(DevonAgent._tool_proxy_file)
        self.assertNotIn("self._call(", src)
        self.assertIn("_registry_tool", src)

    def test_no_bare_self_call_anywhere(self):
        """No _tool_* method should reference self._call (doesn't exist)."""
        import inspect
        from nomorals.agents.devon import DevonAgent
        for name in dir(DevonAgent):
            if name.startswith("_tool_"):
                src = inspect.getsource(getattr(DevonAgent, name))
                self.assertNotIn("self._call(", src,
                                 f"{name} uses non-existent self._call")


class TestFileSendDefaults(unittest.TestCase):
    def test_file_send_defaults_to_chat_key(self):
        """_tool_file_send should default platform/chat_id from _last_chat_key."""
        import inspect
        from nomorals.agents.devon import DevonAgent
        src = inspect.getsource(DevonAgent._tool_file_send)
        self.assertIn("_last_chat_key", src)


if __name__ == "__main__":
    unittest.main()
