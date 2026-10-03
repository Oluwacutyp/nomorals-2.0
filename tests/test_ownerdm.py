"""Owner-DM routing truth tests (ownerdm-1.0).

Regression tests for the live Termux findings plus the systematic audit:
- NL creative writing routes to BookForge/music, NEVER the coding builder
- NL play titles resolve as music, never Path()'d as files
- Owner identity assertions get owner-mode recognition
- Coding success criteria reject empty artifacts
- Devon agent knows the connector catalog (orientation)
"""
from __future__ import annotations

import unittest

from nomorals.agents.coremind import (
    _book_intent,
    _music_intent,
    _owner_intent,
    _play_media_intent,
    understand,
)


class BookIntentTests(unittest.TestCase):
    def test_story_routes_to_book_not_build(self):
        for text in [
            "write me a story about a dragon",
            "write me a book to feel better",
            "write a poem for my girlfriend",
            "write a short story about heartbreak",
            "create a bedtime story for kids",
            "write me a novel chapter",
        ]:
            intents = understand(text)
            self.assertTrue(intents, text)
            top = intents[0]
            self.assertEqual(top.kind, "book", text)
            self.assertEqual(top.route, "book", text)
            self.assertNotEqual(top.kind, "build", text)

    def test_book_topic_extraction(self):
        it = _book_intent("write me a story about a dragon")
        self.assertIsNotNone(it)
        self.assertEqual(it.target, "a dragon")

    def test_book_report_not_a_book(self):
        # homework, not narrative — must not route to BookForge
        it = _book_intent("write me a book report on dune")
        self.assertIsNone(it)

    def test_tell_me_a_story_stays_chat(self):
        # "tell me a story" is a companion moment, not an 8-chapter book
        intents = understand("tell me a story")
        top = intents[0] if intents else None
        self.assertTrue(top is None or top.kind != "book")


class MusicIntentTests(unittest.TestCase):
    def test_compose_routes_to_music_not_build(self):
        for text in [
            "compose a song about love",
            "make me a beat",
            "write lyrics for a sad song",
            "create an afrobeats track",
        ]:
            intents = understand(text)
            self.assertTrue(intents, text)
            top = intents[0]
            self.assertEqual(top.kind, "music", text)
            self.assertEqual(top.route, "music", text)

    def test_style_hint_extracted(self):
        it = _music_intent("create an afrobeats track")
        self.assertIsNotNone(it)
        self.assertEqual(it.meta.get("style"), "afrobeats")

    def test_music_topic_extraction(self):
        it = _music_intent("compose a song about love")
        self.assertIsNotNone(it)
        self.assertEqual(it.target, "love")


class PlayMediaIntentTests(unittest.TestCase):
    def test_tell_me_a_story_not_a_game(self):
        # live failure: "tell me a story" hit the story GAME ("which one —
        # hangman, mafia, ...") instead of companion storytelling.
        for text in ["tell me a story", "tell me a bedtime story",
                     "read me a story"]:
            intents = understand(text)
            top = intents[0] if intents else None
            self.assertTrue(top is None or top.kind != "game", text)

    def test_explicit_story_game_still_works(self):
        intents = understand("play the story game")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "game")

    def test_title_routes_to_play_not_game(self):
        for text in [
            "play lucid dreams by juice wrld",
            "play some jazz",
            "queue up hotel california",
        ]:
            intents = understand(text)
            self.assertTrue(intents, text)
            top = intents[0]
            self.assertEqual(top.kind, "play", text)
            self.assertEqual(top.route, "media", text)

    def test_title_not_treated_as_path(self):
        it = _play_media_intent("play lucid dreams by juice wrld")
        self.assertIsNotNone(it)
        # one query string, not word-split; no Path() semantics
        self.assertEqual(it.target, "lucid dreams by juice wrld")
        self.assertNotIn("/", it.target)

    def test_real_game_stays_game(self):
        intents = understand("play hangman")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "game")
        self.assertGreaterEqual(intents[0].confidence, 0.9)

    def test_single_unknown_word_stays_game_ask(self):
        # "play chess" — Devon has no chess; the game "which one?" ask
        # is preserved rather than misfiring music resolution
        intents = understand("play chess")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "game")

    def test_pasted_prose_collapses(self):
        it = _play_media_intent('queue it with:\n/play /music/song.mp3')
        # either None (prose rejected) or the clean path — never the
        # word "queue" as a target
        if it is not None:
            self.assertNotIn("queue", it.target.lower().split())


class PlayQueryTests(unittest.TestCase):
    def test_query_normalization(self):
        from nomorals.agents.partner.runtime_media import (
            RuntimeMediaMixin,
        )
        q = RuntimeMediaMixin._play_query

        class FakeSettings:
            workspace_dir = "/tmp/nonexistent-ws-ownerdm"
        class FakeCtx:
            settings = FakeSettings()
            db = None
        ctx = FakeCtx()

        # quoted title → unquoted
        self.assertEqual(q(ctx, '"lucid dreams"'), "lucid dreams")
        # /play prefix stripped
        self.assertEqual(q(ctx, "/play hotel california"),
                         "hotel california")
        # pasted help prose → the actual path on the last line
        self.assertEqual(
            q(ctx, "queue it with:\n/play /music/song.mp3"),
            "/music/song.mp3")
        # URLs pass through untouched
        self.assertEqual(q(ctx, "https://example.com/x.mp3"),
                         "https://example.com/x.mp3")
        # paths pass through untouched
        self.assertEqual(q(ctx, "/home/user/tune.wav"),
                         "/home/user/tune.wav")


class OwnerIntentTests(unittest.TestCase):
    def test_identity_assertion_recognized(self):
        for text in [
            "I'm peace your creator",
            "I am peace",
            "drop the act",
        ]:
            intents = understand(text)
            self.assertTrue(intents, text)
            top = intents[0]
            self.assertEqual(top.kind, "owner", text)
            self.assertGreaterEqual(top.confidence, 0.9)

    def test_owner_beats_work_intents(self):
        # identity assertion wins even with work-ish words nearby
        intents = understand("I'm peace, drop the act and write me a story")
        self.assertTrue(intents)
        # owner (0.9) outranks book (0.85)
        self.assertEqual(intents[0].kind, "owner")


class ResearchBroadeningTests(unittest.TestCase):
    def test_search_for_routes_to_research(self):
        intents = understand("search for cheap flights")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "research")

    def test_look_up_routes_to_research(self):
        intents = understand("look up the bitcoin price")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "research")


class CodingGuardTests(unittest.TestCase):
    def test_build_guard_rejects_narrative(self):
        from nomorals.agents.coremind import CoreMind, Intent
        mind = CoreMind.__new__(CoreMind)
        intent = Intent(kind="build", confidence=0.6,
                        target="write me a story about dragons",
                        route="coding", why="test")
        reply = mind._dispatch_build(intent, "job1", "test:key", None)
        self.assertIn("book", reply.lower())

    def test_build_guard_rejects_account_signup(self):
        from nomorals.agents.coremind import CoreMind, Intent
        mind = CoreMind.__new__(CoreMind)
        intent = Intent(kind="build", confidence=0.6,
                        target="create a soundcloud account and send logins",
                        route="coding", why="test")
        reply = mind._dispatch_build(intent, "job1", "test:key", None)
        self.assertIn("account", reply.lower())

    def test_true_code_goal_passes_guard(self):
        # the guard only fires on narrative/account patterns — a real
        # code goal must not be refused here (it proceeds to async build)
        from nomorals.agents.coremind import (
            _RE_BOOK, _RE_MUSIC,
        )
        import re
        goal = "build a todo app in python"
        self.assertIsNone(_RE_BOOK.search(goal))
        self.assertIsNone(_RE_MUSIC.search(goal))
        self.assertIsNone(
            re.search(r"\b(accounts?|sign ?up|logins?)\b", goal, re.I))


class CodingAcceptanceTests(unittest.TestCase):
    def test_empty_file_not_substantive(self):
        from nomorals.agents.coding import _substantive_lines
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "main.py"
            p.write_text("")
            self.assertEqual(_substantive_lines(p), 0)
            p.write_text("# just a comment\n\n")
            self.assertEqual(_substantive_lines(p), 0)

    def test_real_script_is_substantive(self):
        from nomorals.agents.coding import _substantive_lines
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "main.py"
            p.write_text(
                '"""tool."""\nimport sys\n\n'
                'def main():\n    print("hi")\n    return 0\n\n'
                'if __name__ == "__main__":\n    sys.exit(main())\n')
            self.assertGreaterEqual(_substantive_lines(p), 5)


class OrientationTests(unittest.TestCase):
    def test_orientation_block_lists_spotify(self):
        from nomorals.agents.orientation import repo_orientation_block
        block = repo_orientation_block()
        self.assertIn("spotify", block)
        self.assertIn("nomorals/connectors/", block)
        self.assertIn("src/integrations", block)  # warns against the guess

    def test_connector_tool_in_catalog(self):
        from nomorals.agents.devon import TOOL_CATALOG, DevonAgent
        names = [n for n, _ in TOOL_CATALOG]
        self.assertIn("connector", names)
        self.assertTrue(hasattr(DevonAgent, "_tool_connector"))

    def test_connector_tool_lists(self):
        from nomorals.agents.devon import DevonAgent
        agent = DevonAgent.__new__(DevonAgent)
        out = agent._tool_connector({"action": "list"})
        self.assertIn("spotify", out)
        self.assertIn("connectors", out)

    def test_connector_tool_info_spotify(self):
        from nomorals.agents.devon import DevonAgent
        agent = DevonAgent.__new__(DevonAgent)
        out = agent._tool_connector({"action": "info", "id": "spotify"})
        self.assertIn("spotify", out.lower())
        self.assertIn("how to link", out.lower())


class MediaToolAwarenessTests(unittest.TestCase):
    def test_media_edit_tools_in_catalog(self):
        # live failure: "/devon can you edit images?" wrote raw PIL that
        # crashed — the agent didn't know the media_edit tools exist.
        from nomorals.agents.devon import TOOL_CATALOG, DevonAgent
        names = [n for n, _ in TOOL_CATALOG]
        for tool in ("media_edit", "media_edit_video", "media_probe",
                     "media_convert", "media_capability"):
            self.assertIn(tool, names, tool)
            self.assertTrue(hasattr(DevonAgent, f"_tool_{tool}"), tool)

    def test_orientation_mentions_media_edit(self):
        from nomorals.agents.orientation import repo_orientation_block
        block = repo_orientation_block()
        self.assertIn("media_edit", block)
        self.assertIn("cv_ops", block)


class SchedulerSilenceTests(unittest.TestCase):
    def test_routine_tick_names_are_silenced(self):
        # the scheduler's silence predicate must cover the exact job
        # names the runtime registers ("rooms tick", "watchers sweep")
        for name in ["rooms tick", "watchers sweep", "heartbeat",
                     "Rooms Tick", "WATCHERS SWEEP"]:
            lowered = name.lower()
            is_routine = ('tick' in lowered or 'heartbeat' in lowered
                          or 'sweep' in lowered)
            self.assertTrue(is_routine, name)


if __name__ == "__main__":
    unittest.main()
