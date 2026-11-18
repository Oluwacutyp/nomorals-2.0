"""MusicForProgramming (#31): focus mixes tied to focus sessions.

The editorial layer over the pieces that already exist: the
``music_sources/`` registry + play spine, the DJ / long-form mix
machinery, and the focus-session ledger. Mixes assemble from the
owner's library and listening history (the taste model is the
owner's), sequence through the DJ engine, and link session ↔ mix in
the ledger. The series itself is seeded from its real RSS feed —
fixture below is modeled on the live feed fetched 2026-11-18
(79 episodes; Episode 79: Corticyte).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.focus.sessions import FocusSession, FocusTimer
from nomorals.music_sources.focus_mixes import (
    MOOD_TAGS,
    FocusMix,
    FocusMixStore,
    FocusTrack,
    assemble_mix,
    default_planner,
    find_mix,
    format_mix_reply,
    parse_focus_mix_request,
    parse_rss,
    refresh_from_series,
    render_mix,
    start_focus_session_with_mix,
    taste_scores,
)

RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0">
<channel>
<title>Music For Programming</title>
<link>https://musicforprogramming.net/</link>
<item>
<title>Episode 79: Corticyte</title>
<link>http://musicforprogramming.net/seventynine</link>
<pubDate>Mon, 24 Aug 2026 17:18:00 GMT</pubDate>
<guid>https://datashat.net/music_for_programming_79-corticyte.mp3</guid>
<itunes:keywords>Electronic, Ambient, Acoustic, Drone, Noise, Instrumental, Study, Concentration, Programming</itunes:keywords>
<itunes:duration>4:00:00</itunes:duration>
<enclosure url="https://datashat.net/music_for_programming_79-corticyte.mp3" length="441077163" type="audio/mpeg"/>
</item>
<item>
<title>Episode 78: Datassette</title>
<link>http://musicforprogramming.net/seventyeight</link>
<pubDate>Mon, 10 Aug 2026 17:18:00 GMT</pubDate>
<guid>https://datashat.net/music_for_programming_78-datassette.mp3</guid>
<itunes:keywords>Electronic, Ambient, Instrumental</itunes:keywords>
<itunes:duration>1:00:00</itunes:duration>
<enclosure url="https://datashat.net/music_for_programming_78-datassette.mp3" length="123456789" type="audio/mpeg"/>
</item>
<item>
<title>Episode 77: Phonaut</title>
<link>http://musicforprogramming.net/seventyseven</link>
<pubDate>Mon, 27 Jul 2026 17:18:00 GMT</pubDate>
<guid>https://datashat.net/music_for_programming_77-phonaut.mp3</guid>
<itunes:keywords>Ambient, Drone</itunes:keywords>
<itunes:duration>59:30</itunes:duration>
<enclosure url="https://datashat.net/music_for_programming_77-phonaut.mp3" length="98765432" type="audio/mpeg"/>
</item>
</channel>
</rss>
"""


def _track(title: str, artist: str = "", duration: float = 180.0,
           energy: float = 0.5) -> FocusTrack:
    return FocusTrack(id=f"ft-{title.lower().replace(' ', '-')}",
                      title=title, artist=artist,
                      duration_s=duration, energy=energy)


class FakeLibrary:
    """The owner's library, duck-typed like MusicLibrary."""

    def __init__(self, rows):
        self._rows = list(rows)

    def favorites(self, limit=200):
        return self._rows[:2]

    def top_played(self, limit=10):
        return sorted(self._rows, key=lambda r: r.get("plays", 0),
                      reverse=True)

    def history(self, limit=50):
        return list(reversed(self._rows))


LIBRARY_ROWS = [
    {"path": "/tunes/night-drive.mp3", "title": "Night Drive",
     "artist": "Neon Coast", "album": "Afterglow", "duration": 240,
     "plays": 12},
    {"path": "/tunes/glass-meadow.mp3", "title": "Glass Meadow",
     "artist": "Aurora Fields", "album": "Afterglow", "duration": 300,
     "plays": 3},
    {"path": "/tunes/dark-static.mp3", "title": "Dark Static",
     "artist": "Void Choir", "album": "Static", "duration": 200,
     "plays": 1},
    {"path": "/tunes/low-orbit.mp3", "title": "Low Orbit",
     "artist": "Neon Coast", "album": "Afterglow", "duration": 280,
     "plays": 8},
]


class ParseTests(unittest.TestCase):
    def test_positive_shapes(self) -> None:
        self.assertEqual(
            parse_focus_mix_request("put on a coding mix"),
            {"kind": "mix", "mood": None, "episode": None})
        self.assertEqual(
            parse_focus_mix_request("focus music, darker"),
            {"kind": "mix", "mood": "darker", "episode": None})
        self.assertEqual(
            parse_focus_mix_request("coding mix, driving"),
            {"kind": "mix", "mood": "driving", "episode": None})
        self.assertEqual(
            parse_focus_mix_request("start a focus session with music"),
            {"kind": "mix", "mood": None, "episode": None})
        self.assertEqual(
            parse_focus_mix_request("put on episode 79"),
            {"kind": "episode", "mood": None, "episode": 79})
        self.assertEqual(
            parse_focus_mix_request("play a focus mix"),
            {"kind": "mix", "mood": None, "episode": None})

    def test_rejections(self) -> None:
        for text in ("play despacito", "start a focus session",
                     "play focus", "/focus start", "how's your day going?",
                     "close my laptop", ""):
            self.assertIsNone(parse_focus_mix_request(text), text)

    def test_mood_words(self) -> None:
        self.assertEqual(MOOD_TAGS["darker"]["energy"], "low")
        self.assertEqual(MOOD_TAGS["driving"]["energy"], "high")
        self.assertEqual(
            parse_focus_mix_request("focus music, dark")["mood"], "darker")


class RssTests(unittest.TestCase):
    def test_parse_real_shaped_feed(self) -> None:
        episodes = parse_rss(RSS_FIXTURE)
        self.assertEqual(len(episodes), 3)
        first = episodes[0]
        self.assertEqual(first["number"], 79)
        self.assertEqual(first["title"], "Episode 79: Corticyte")
        self.assertEqual(
            first["url"],
            "https://datashat.net/music_for_programming_79-corticyte.mp3")
        self.assertEqual(first["duration_s"], 14400.0)
        self.assertIn("ambient", first["tags"])
        self.assertEqual(episodes[2]["duration_s"], 3570.0)

    def test_parse_bad_xml_is_empty(self) -> None:
        self.assertEqual(parse_rss("not xml at all"), [])
        self.assertEqual(parse_rss(""), [])

    def test_refresh_records_episodes_as_mixes(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            result = refresh_from_series(
                store, fetcher=lambda url: RSS_FIXTURE)
            self.assertEqual(result, {"added": 3, "total": 3,
                                      "error": ""})
            mix = store.get("mfp-79")
            self.assertIsNotNone(mix)
            self.assertEqual(mix.title, "Episode 79: Corticyte")
            self.assertEqual(mix.curator, "devon")
            self.assertEqual(mix.source, "musicforprogramming")
            self.assertEqual(len(mix.tracks), 1)
            self.assertEqual(
                mix.tracks[0].url,
                "https://datashat.net/music_for_programming_79-corticyte.mp3")
            self.assertEqual(mix.tracks[0].duration_s, 14400.0)
            again = refresh_from_series(
                store, fetcher=lambda url: RSS_FIXTURE)
            self.assertEqual(again["added"], 0)

    def test_refresh_dead_feed_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)

            def _dead(url):
                raise OSError("network down")

            result = refresh_from_series(store, fetcher=_dead)
            self.assertEqual(result["added"], 0)
            self.assertIn("couldn't reach", result["error"])
            self.assertEqual(store.list(), [])


class StoreTests(unittest.TestCase):
    def test_round_trip_and_remove(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            mix = FocusMix(id="fm-1", title="Test mix",
                           tracks=[_track("A"), _track("B")])
            store.add(mix)
            again = FocusMixStore(home)
            self.assertEqual(len(again.list()), 1)
            self.assertEqual(again.get("fm-1").tracks[1].title, "B")
            self.assertTrue(store.remove("fm-1"))
            self.assertFalse(store.remove("fm-1"))

    def test_record_played_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            store.add(FocusMix(id="fm-1", title="Test mix"))
            store.record_played("fm-1")
            store.record_played("fm-1")
            self.assertEqual(store.get("fm-1").sessions_played, 2)
            self.assertIsNone(store.record_played("nope"))

    def test_corrupt_store_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            path = os.path.join(home, "data", "focus_mixes.json")
            os.makedirs(os.path.join(home, "data"), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            store = FocusMixStore(home)
            self.assertEqual(store.list(), [])
            self.assertTrue(any(
                name.startswith("focus_mixes.json.corrupt-")
                for name in os.listdir(os.path.join(home, "data"))))

    def test_find_mix_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            refresh_from_series(store, fetcher=lambda url: RSS_FIXTURE)
            self.assertIsNotNone(find_mix(store, "mfp-79"))
            self.assertIsNotNone(find_mix(store, "mfp-"))
            self.assertIsNotNone(find_mix(store, "Corticyte"))
            self.assertIsNotNone(find_mix(store, "episode 78"))
            self.assertIsNone(find_mix(store, "episode 99"))
            self.assertIsNone(find_mix(store, ""))


class TasteTests(unittest.TestCase):
    def test_scores_from_feedback_rows(self) -> None:
        rows = [
            {"event": "liked", "title": "Night Drive"},
            {"event": "kept", "track_id": "ft-low-orbit"},
            {"event": "played", "title": "Glass Meadow"},
            {"event": "skipped", "title": "Dark Static"},
            {"event": "rated", "title": "Night Drive"},
        ]
        scores = taste_scores(rows)
        self.assertEqual(scores["night drive"], 1.0)
        self.assertEqual(scores["ft-low-orbit"], 1.0)
        self.assertEqual(scores["glass meadow"], 0.25)
        self.assertEqual(scores["dark static"], -1.0)

    def test_default_planner_arc_is_deterministic(self) -> None:
        tracks = [_track("Low", energy=0.2),
                  _track("Mid", energy=0.5),
                  _track("High", energy=0.9)]
        first, _ = default_planner(tracks, 10.0, {})
        second, _ = default_planner(tracks, 10.0, {})
        self.assertEqual([t.title for t in first],
                         ["Low", "Mid", "High"])
        self.assertEqual([t.title for t in first],
                         [t.title for t in second])

    def test_default_planner_empty(self) -> None:
        self.assertEqual(default_planner([], 60.0, {}), ([], {}))


class AssembleTests(unittest.TestCase):
    def test_assemble_from_library(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            mix = assemble_mix(
                store, library=FakeLibrary(LIBRARY_ROWS),
                feedback_rows=[
                    {"event": "liked", "title": "Night Drive"},
                    {"event": "skipped", "title": "Dark Static"}],
                tags={"mood": "dark", "energy": "low"},
                target_minutes=20.0)
            self.assertEqual(len(mix.tracks), 4)
            self.assertEqual(
                {t.title for t in mix.tracks},
                {"Night Drive", "Glass Meadow", "Dark Static",
                 "Low Orbit"})
            # The liked track leads the arc; the skipped one trails.
            self.assertEqual(mix.tracks[0].title, "Night Drive")
            self.assertEqual(mix.tags["mood"], "dark")
            self.assertEqual(mix.source, "library")
            self.assertEqual(mix.curator, "devon")
            self.assertIs(store.get(mix.id), mix)

    def test_assemble_mood_with_no_match_names_it(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            mix = assemble_mix(
                store, library=FakeLibrary(LIBRARY_ROWS),
                tags={"mood": "driving", "energy": "high"},
                target_minutes=20.0)
            self.assertTrue(any("driving" in n for n in mix.notes))
            self.assertEqual(len(mix.tracks), 4)

    def test_assemble_empty_library_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            with self.assertRaises(ValueError) as caught:
                assemble_mix(store, library=FakeLibrary([]),
                             target_minutes=10.0)
            self.assertIn("no tracks in your library",
                          str(caught.exception))
            self.assertEqual(store.list(), [])

    def test_assemble_with_dj_planner(self) -> None:
        from nomorals.tools.focus_mix import dj_planner

        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            mix = assemble_mix(
                store, library=FakeLibrary(LIBRARY_ROWS),
                feedback_rows=[
                    {"event": "liked", "title": "Night Drive"},
                    {"event": "skipped", "title": "Dark Static"}],
                target_minutes=20.0, planner=dj_planner)
            self.assertEqual(len(mix.tracks), 4)
            transitions = mix.plan.get("transitions") or []
            self.assertEqual(len(transitions), 3)
            self.assertTrue(all(
                t.get("kind") in ("bass_swap", "filter_sweep",
                                  "acapella_overlay", "drop_cut",
                                  "crossfade") for t in transitions))
            text = render_mix(mix)
            self.assertIn("[", text)  # a transition kind rides a line


class SessionLinkTests(unittest.TestCase):
    def test_session_start_links_session_to_mix(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            refresh_from_series(store, fetcher=lambda url: RSS_FIXTURE)
            mix = store.get("mfp-79")
            timer = FocusTimer(os.path.join(home, "data", "focus.json"))
            played: list[FocusTrack] = []

            def _player(track):
                played.append(track)
                return {"status": "sent", "detail": track.title}

            result = start_focus_session_with_mix(
                timer, store, mix, player=_player)
            session = result["session"]
            self.assertEqual(session.mix_id, "mfp-79")
            self.assertEqual(session.mix_title, "Episode 79: Corticyte")
            self.assertEqual(store.get("mfp-79").sessions_played, 1)
            self.assertEqual([t.title for t in played],
                             ["Episode 79: Corticyte"])
            self.assertEqual(result["play"]["status"], "sent")
            # The link survives a reload of the ledger from disk.
            again = FocusTimer(os.path.join(home, "data", "focus.json"))
            reloaded = again.sessions[session.id]
            self.assertEqual(reloaded.mix_id, "mfp-79")
            self.assertEqual(reloaded.mix_title,
                             "Episode 79: Corticyte")
            reply = format_mix_reply(mix, session, result["play"])
            self.assertIn("Episode 79: Corticyte", reply)
            self.assertIn("focus session started", reply)
            self.assertIn("playing: Episode 79: Corticyte", reply)

    def test_start_with_running_session_attaches(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = FocusMixStore(home)
            refresh_from_series(store, fetcher=lambda url: RSS_FIXTURE)
            mix = store.get("mfp-78")
            timer = FocusTimer(os.path.join(home, "data", "focus.json"))
            first = timer.start("coding")
            result = start_focus_session_with_mix(
                timer, store, mix, player=None)
            self.assertEqual(result["session"].id, first.id)
            self.assertEqual(result["session"].mix_id, "mfp-78")
            self.assertEqual(result["play"]["status"], "planned")
            self.assertEqual(len(timer.sessions), 1)

    def test_session_without_mix_round_trips(self) -> None:
        session = FocusSession.from_dict({"id": "fc-old", "label": "old"})
        self.assertEqual(session.mix_id, "")
        self.assertEqual(session.mix_title, "")
        self.assertEqual(session.to_dict()["mix_id"], "")

    def test_render_mix_timestamps(self) -> None:
        mix = FocusMix(
            id="fm-1", title="Test mix",
            tracks=[_track("A", duration=120.0),
                    _track("B", artist="Someone", duration=180.0)])
        text = render_mix(mix)
        self.assertIn("00:00 — A", text)
        self.assertIn("02:00 — Someone — B", text)
        self.assertIn("5m", text)


class RegistryTests(unittest.TestCase):
    def test_musicforprogramming_row(self) -> None:
        from nomorals.music_sources.registry import by_id

        row = by_id("musicforprogramming")
        self.assertIsNotNone(row)
        self.assertEqual(row.name, "Music For Programming")
        self.assertEqual(row.source_class, "open")
        self.assertEqual(row.health_check_hours, 24)
        self.assertIn("rss.xml", row.notes)


class ToolTests(unittest.TestCase):
    def _tool(self, home):
        from nomorals.tools import focus_mix as fm_mod
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(
            settings=SimpleNamespace(home=str(home)))
        reg = ToolRegistry(context=context)
        fm_mod.register(reg)
        return reg._tools["focus_mix"].fn

    def test_tool_registered_by_builtins(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register_builtins()
        self.assertIn("focus_mix", reg._tools)

    def test_tool_list_show_refresh(self) -> None:
        import nomorals.music_sources.focus_mixes as organ

        with tempfile.TemporaryDirectory() as home:
            tool = self._tool(home)
            out = tool(action="list")
            self.assertTrue(out["ok"])
            self.assertEqual(out["count"], 0)
            real_fetch = organ._default_fetcher
            organ._default_fetcher = lambda url, timeout_s=20.0: RSS_FIXTURE
            try:
                refreshed = tool(action="refresh")
            finally:
                organ._default_fetcher = real_fetch
            self.assertTrue(refreshed["ok"])
            self.assertEqual(refreshed["added"], 3)
            shown = tool(action="show", id="episode 79")
            self.assertTrue(shown["ok"])
            self.assertIn("Corticyte", shown["reply"])
            missing = tool(action="show", id="nope")
            self.assertFalse(missing["ok"])

    def test_tool_assemble_empty_library_honest(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            tool = self._tool(home)
            out = tool(action="assemble")
            self.assertFalse(out["ok"])
            self.assertIn("no tracks in your library", out["error"])

    def test_tool_assemble_from_library(self) -> None:
        import nomorals.tools.focus_mix as fm_mod

        with tempfile.TemporaryDirectory() as home:
            tool = self._tool(home)
            fm_mod._library_of = lambda ctx: FakeLibrary(LIBRARY_ROWS)
            fm_mod.read_owner_taste = lambda path=None: [
                {"event": "liked", "title": "Night Drive"}]
            try:
                out = tool(action="assemble", mood="darker", minutes=20)
            finally:
                import importlib
                importlib.reload(fm_mod)
            self.assertTrue(out["ok"], out)
            self.assertEqual(len(out["mix"]["tracks"]), 4)
            # Mood "darker" matches "Dark Static" by title, so the
            # mood-matching track leads the arc; taste re-ranks
            # inside each third, it does not override the mood.
            self.assertEqual(out["mix"]["tracks"][0]["title"],
                             "Dark Static")
            self.assertIn("Night Drive",
                          [t["title"] for t in out["mix"]["tracks"]])

    def test_dj_planner_taste_steers_within_third(self) -> None:
        from nomorals.tools.focus_mix import dj_planner

        tracks = [_track("A"), _track("B"), _track("C"), _track("D")]
        ordered, plan = dj_planner(
            tracks, 12.0,
            feedback_rows=[{"event": "liked", "title": "D"},
                           {"event": "skipped", "title": "C"}])
        # All four share one energy, so the arc thirds are
        # [[A], [B], [C, D]] — taste re-ranks inside the last third:
        # the liked track jumps ahead of the skipped one.
        self.assertEqual([t.title for t in ordered],
                         ["A", "B", "D", "C"])
        self.assertEqual(len(plan.get("transitions") or []), 3)

    def test_tool_play_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            tool = self._tool(home)
            created = tool(action="create", title="My mix",
                           tracks="Alpha — One; Beta — Two")
            self.assertTrue(created["ok"], created)
            mix_id = created["mix"]["id"]
            played = tool(action="play", id=mix_id)
            self.assertTrue(played["ok"], played)
            self.assertEqual(played["session"]["mix_id"], mix_id)
            self.assertEqual(played["play"]["status"], "planned")
            status = tool(action="status")
            self.assertTrue(status["running"])
            self.assertEqual(status["mix_id"], mix_id)
            removed = tool(action="remove", id=mix_id)
            self.assertTrue(removed["ok"])
            self.assertFalse(tool(action="play", id=mix_id)["ok"])
            self.assertFalse(tool(action="explode")["ok"])


class HookTests(unittest.TestCase):
    def _hook(self, home, text, play_calls=None):
        from nomorals.agents.partner.runtime import PartnerRuntime
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

        timer = FocusTimer(os.path.join(home, "data", "focus.json"))

        def _control_play(tail, chat_key=""):
            ack = f"queued {tail}"
            if play_calls is not None:
                play_calls.append(tail)
            return ack

        def _player_maker(chat_key):
            def _play(track):
                query = (str(getattr(track, "url", "") or "").strip()
                         or f"{track.title} — {track.artist}".strip(" —"))
                return {"status": "queued",
                        "detail": _control_play("play " + query,
                                                chat_key=chat_key)}
            return _play

        double = SimpleNamespace(
            context=SimpleNamespace(
                settings=SimpleNamespace(home=home)),
            _focus_timer=lambda: timer,
            _control_play=_control_play,
            _focus_mix_player=_player_maker)
        chat = ChatRef(platform="local", chat_id="console",
                       kind=ChatKind.DM, peer="you")
        message = ChatMessage(chat=chat, incoming=True, text=text,
                              sender="you")
        return PartnerRuntime._focus_mix_hook(double, message)

    def _seed_series(self, home):
        store = FocusMixStore(home)
        refresh_from_series(store, fetcher=lambda url: RSS_FIXTURE)
        return store

    def test_hook_episode_plays(self) -> None:
        import nomorals.tools.focus_mix as fm_mod

        with tempfile.TemporaryDirectory() as home:
            self._seed_series(home)
            calls: list[str] = []
            fm_mod.read_owner_taste = lambda path=None: []
            try:
                reply = self._hook(home, "put on episode 79", calls)
            finally:
                import importlib
                importlib.reload(fm_mod)
            self.assertIsNotNone(reply)
            self.assertIn("Episode 79: Corticyte", reply)
            self.assertIn("focus session started", reply)
            self.assertEqual(len(calls), 1)
            self.assertIn(
                "music_for_programming_79-corticyte.mp3", calls[0])

    def test_hook_mix_falls_back_to_series(self) -> None:
        import nomorals.music_sources.focus_mixes as organ
        import nomorals.tools.focus_mix as fm_mod

        with tempfile.TemporaryDirectory() as home:
            self._seed_series(home)
            fm_mod.read_owner_taste = lambda path=None: []
            real_refresh = organ.refresh_from_series
            organ.refresh_from_series = (
                lambda store, **kw: {"added": 0, "total": 79,
                                     "error": ""})
            try:
                reply = self._hook(home, "put on a coding mix")
            finally:
                organ.refresh_from_series = real_refresh
                import importlib
                importlib.reload(fm_mod)
            self.assertIsNotNone(reply)
            self.assertIn("Episode 79: Corticyte", reply)
            self.assertIn("no library tracks", reply)

    def test_hook_unknown_episode_honest(self) -> None:
        import nomorals.music_sources.focus_mixes as organ

        with tempfile.TemporaryDirectory() as home:
            self._seed_series(home)
            real_refresh = organ.refresh_from_series
            organ.refresh_from_series = (
                lambda store, **kw: {"added": 0, "total": 79,
                                     "error": ""})
            try:
                reply = self._hook(home, "put on episode 99")
            finally:
                organ.refresh_from_series = real_refresh
            self.assertIsNotNone(reply)
            self.assertIn("episode 99 isn't in", reply)

    def test_hook_ignores_non_requests(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            self.assertIsNone(self._hook(home, "play despacito"))
            self.assertIsNone(self._hook(home, "start a focus session"))
            self.assertIsNone(self._hook(home, "how's your day going?"))
            self.assertIsNone(self._hook(home, "/focus start"))


if __name__ == "__main__":
    unittest.main()
