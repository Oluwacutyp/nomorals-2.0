"""Media-core acceptance tests (media-imp wave).

Covers everything the media-core survey touched:

* playback.PlaybackEngine — durable queue, transport, simple-backend
  process hygiene (no orphaned players on next/prev/play), the
  play(index=0) sentinel fix in the ``player`` tool wrapper;
* music.MusicCreator — offline lyric engine (real rhymes, seeded
  determinism), styles/aliases, real SMF output, and the
  router.chat Message/SamplingParams contract for _model_lyrics;
* video.VideoFinder — scoring, relevance, dedup, platform pinning,
  youtube-id extraction, fail-fast download without a registry;
* MediaHub — orchestrator modes, auto_send_target, transcript helpers.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.core.errors import ToolError
from nomorals.llm.base import Message, SamplingParams
from nomorals.media import MediaHub
from nomorals.media.music import STYLES, MusicCreator, resolve_style
from nomorals.media.playback import Backend, PlaybackEngine, detect_backend
from nomorals.media.video import VideoFinder, _relevance, _video_score
from nomorals.storage.db import Database

# ── fixtures ────────────────────────────────────────────────────────────────


def _context(**over: Any) -> SimpleNamespace:
    root = tempfile.mkdtemp(prefix="media_core_")
    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=root),
        db=db, tools=None, router=None, gateway=None,
    )
    for k, v in over.items():
        setattr(ctx, k, v)
    ctx._root = root
    return ctx


def _registry(context: SimpleNamespace) -> SimpleNamespace:
    fns: dict[str, Any] = {}

    def register(name: str, **kw: Any) -> Any:
        def deco(fn: Any) -> Any:
            fns[name] = fn
            return fn
        return deco

    return SimpleNamespace(context=context, register=register, fns=fns)


class MediaCoreBase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _context()
        self.addCleanup(shutil.rmtree, self.ctx._root, True)

    def _clip(self, name: str = "song.mp3") -> str:
        p = Path(self.ctx._root) / name
        p.write_bytes(b"fake-audio")
        return name  # workspace-relative, as the tool expects


# ── playback ────────────────────────────────────────────────────────────────


class PlaybackQueueTests(MediaCoreBase):
    def setUp(self) -> None:
        super().setUp()
        import nomorals.media.playback as pb

        # ffplay exists on some dev machines — pin the console backend so
        # these queue/position assertions are deterministic everywhere
        self._be = mock.patch.object(
            pb, "detect_backend", lambda: Backend("console"))
        self._be.start()
        self.addCleanup(self._be.stop)
        self.eng = PlaybackEngine(self.ctx)
        self.a = self._clip("a.mp3")
        self.b = self._clip("b.mp3")

    def test_add_and_queue_order(self) -> None:
        res = self.eng.add(self.a, self.b, title="pair")
        self.assertEqual(len(res["added"]), 2)
        q = self.eng.queue()
        self.assertEqual([i["title"] for i in q], ["pair", "pair"])
        self.assertEqual(q[0]["index"], 0)
        self.assertEqual(q[0]["kind"], "file")
        self.assertTrue(os.path.exists(q[0]["path"]))

    def test_add_url(self) -> None:
        res = self.eng.add("https://example.com/track.mp3")
        self.assertEqual(res["added"][0]["kind"], "url")

    def test_add_missing_file_raises(self) -> None:
        with self.assertRaises(ToolError):
            self.eng.add("nope/gone.mp3")

    def test_play_empty_queue_raises(self) -> None:
        with self.assertRaises(ToolError):
            self.eng.play()

    def test_play_bad_index_raises(self) -> None:
        self.eng.add(self.a)
        with self.assertRaises(ToolError):
            self.eng.play(7)

    def test_remove_and_clear(self) -> None:
        self.eng.add(self.a, self.b)
        res = self.eng.remove(0)
        self.assertEqual(self.eng.queue()[0]["title"], "b")
        self.assertTrue(res["removed"].endswith("a.mp3"))
        self.eng.clear()
        self.assertEqual(self.eng.queue(), [])
        with self.assertRaises(ToolError):
            self.eng.remove(0)

    def test_volume_clamps(self) -> None:
        self.assertEqual(self.eng.volume(150)["level"], 100)
        self.assertEqual(self.eng.volume(-5)["level"], 0)

    def test_next_prev_wrap(self) -> None:
        self.eng.add(self.a, self.b)
        # console backend: no real transport, but the position must move
        st = self.eng.next()
        self.assertEqual(st["status"], "no-backend")
        q = self.eng.queue()
        self.assertEqual(self.eng.status()["position"], 1)
        self.assertEqual(q[self.eng.status()["position"]]["title"], "b")
        self.eng.next()  # wraps
        self.assertEqual(self.eng.status()["position"], 0)
        self.eng.prev()  # wraps back
        self.assertEqual(self.eng.status()["position"], 1)

    def test_console_backend_is_honest(self) -> None:
        self.eng.add(self.a)
        res = self.eng.play()
        self.assertEqual(res["status"], "no-backend")
        self.assertIn("mpv", res["hint"])
        st = self.eng.status()
        self.assertFalse(st["playing"])
        self.assertEqual(st["backend"]["name"], "console")


class PlaybackSimpleBackendTests(MediaCoreBase):
    """next()/prev()/play() must kill the live player before starting
    a new one — no stacked, orphaned audio processes."""

    def setUp(self) -> None:
        super().setUp()
        self.eng = PlaybackEngine(self.ctx)
        self.eng.backend = Backend("mpg123", "/bin/fake-mpg123")
        self.a = self._clip("a.mp3")
        self.b = self._clip("b.mp3")
        self.spawned: list[int] = []
        self.killed: list[int] = []

        def fake_popen(cmd: Any, **kw: Any) -> SimpleNamespace:
            pid = 1000 + len(self.spawned)
            self.spawned.append(pid)
            return SimpleNamespace(pid=pid)

        import nomorals.media.playback as pb

        self._popen_patch = mock.patch.object(
            pb.subprocess, "Popen", fake_popen)
        self._popen_patch.start()
        self.addCleanup(self._popen_patch.stop)

        self._killpg_patch = mock.patch.object(
            pb.os, "killpg",
            lambda pgid, sig: self.killed.append(pgid))
        self._killpg_patch.start()
        self.addCleanup(self._killpg_patch.stop)

        self._getpgid_patch = mock.patch.object(
            pb.os, "getpgid", lambda pid: pid)
        self._getpgid_patch.start()
        self.addCleanup(self._getpgid_patch.stop)

        # add() probes audio metadata with ffprobe when available; keep
        # this test about transport processes only, not metadata probes
        self._meta_patch = mock.patch.object(
            pb, "read_metadata",
            lambda path: {"title": "", "artist": "", "album": "",
                          "genre": "", "duration": 0.0})
        self._meta_patch.start()
        self.addCleanup(self._meta_patch.stop)

    def test_start_kills_previous_player(self) -> None:
        self.eng.add(self.a, self.b)
        first = self.eng.play()
        self.assertEqual(first["status"], "playing")
        self.assertEqual(len(self.spawned), 1)
        old_pid = self.spawned[0]
        second = self.eng.next()
        self.assertEqual(second["status"], "playing")
        # old process was killed before the new one spawned
        self.assertIn(old_pid, self.killed)
        self.assertEqual(len(self.spawned), 2)

    def test_stop_kills_player_and_clears_state(self) -> None:
        self.eng.add(self.a)
        self.eng.play()
        pid = self.eng._state["player_pid"]
        res = self.eng.stop()
        self.assertEqual(res["status"], "stopped")
        self.assertIn(pid, self.killed)
        self.assertNotIn("player_pid", self.eng._state)

    def test_url_rejected_on_non_streaming_backend(self) -> None:
        self.eng.add("https://example.com/x.mp3")
        res = self.eng.play()
        self.assertEqual(res["status"], "error")
        self.assertIn("stream", res["error"])


class PlayerToolIndexTests(MediaCoreBase):
    """The ``player`` tool wrapper: index=-1 means current; an explicit
    0 must address queue item 0 (it used to be swallowed as falsy)."""

    def setUp(self) -> None:
        super().setUp()
        import nomorals.media.playback as pb

        self._be = mock.patch.object(
            pb, "detect_backend", lambda: Backend("console"))
        self._be.start()
        self.addCleanup(self._be.stop)
        reg = _registry(self.ctx)
        pb.register(reg)
        self.player = reg.fns["player"]
        for n in ("a.mp3", "b.mp3"):
            (Path(self.ctx._root) / n).write_bytes(b"x")
        self.player(action="add", targets="a.mp3|b.mp3")

    def test_explicit_zero_plays_item_zero(self) -> None:
        # move the stored position to 1 first
        self.player(action="next")
        eng = PlaybackEngine(self.ctx)
        self.assertEqual(eng._state["position"], 1)
        res = self.player(action="play", index=0)
        self.assertEqual(res["current"]["title"], "a")
        eng2 = PlaybackEngine(self.ctx)
        self.assertEqual(eng2._state["position"], 0)

    def test_default_plays_current_position(self) -> None:
        self.player(action="next")
        res = self.player(action="play")
        self.assertEqual(res["current"]["title"], "b")

    def test_unknown_action_raises(self) -> None:
        with self.assertRaises(ToolError):
            self.player(action="dance")


# ── music ───────────────────────────────────────────────────────────────────


class MusicStyleTests(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(resolve_style("lo-fi").name, "lofi")
        self.assertEqual(resolve_style("rap").name, "hiphop")
        self.assertEqual(resolve_style("AfroBeat").name, "afrobeats")
        self.assertEqual(resolve_style("").name, "pop")

    def test_unknown_style_raises(self) -> None:
        with self.assertRaises(ToolError):
            resolve_style("klezmer-deathstep")

    def test_all_styles_have_full_spec(self) -> None:
        for name, spec in STYLES.items():
            self.assertTrue(spec.sections, name)
            self.assertTrue(spec.progressions, name)
            self.assertGreater(spec.tempo[1], spec.tempo[0], name)


class MusicComposeTests(MediaCoreBase):
    def test_offline_compose_is_seeded_deterministic(self) -> None:
        c = MusicCreator(self.ctx)
        s1 = c.compose("midnight trains", style="lofi", seed=42)
        s2 = c.compose("midnight trains", style="lofi", seed=42)
        l1 = [ln for sec in s1.sections for ln in sec.lyrics]
        l2 = [ln for sec in s2.sections for ln in sec.lyrics]
        self.assertEqual(l1, l2)
        self.assertEqual(s1.title, s2.title)

    def test_lyrics_carry_topic_and_rhyme(self) -> None:
        c = MusicCreator(self.ctx)
        song = c.compose("midnight trains", style="lofi", seed=7,
                         with_midi=False)
        all_lines = [ln for s in song.sections for ln in s.lyrics]
        self.assertGreater(len(all_lines), 10)
        blob = " ".join(all_lines).lower()
        self.assertIn("midnight", blob)  # topic is woven through
        # verse lines rhyme pairwise: odd lines tail from the even line's
        # rhyme group
        verse = next(s for s in song.sections if s.name == "verse")
        self.assertEqual(len(verse.lyrics), 8)

    def test_sections_have_chords_and_notes(self) -> None:
        c = MusicCreator(self.ctx)
        song = c.compose("sunrise", style="pop", seed=3, with_midi=False)
        self.assertTrue(all(s.chords for s in song.sections))
        intros = [s for s in song.sections if s.name == "intro"]
        self.assertTrue(intros[0].note)

    def test_midi_file_is_real_smf(self) -> None:
        c = MusicCreator(self.ctx)
        song = c.compose("sunrise", style="afrobeats", seed=11)
        self.assertTrue(song.midi_path)
        data = Path(song.midi_path).read_bytes()
        self.assertTrue(data.startswith(b"MThd"))
        self.assertGreater(len(data), 200)

    def test_to_dict_and_markdown(self) -> None:
        c = MusicCreator(self.ctx)
        song = c.compose("city rain", style="rnb", seed=5, with_midi=False)
        d = song.to_dict()
        self.assertEqual(d["key"], song.key)
        self.assertEqual(d["tempo"], song.tempo)
        md = song.to_markdown()
        self.assertIn(song.title, md)
        self.assertIn("CHORUS", md)

    def test_rhyme_helpers(self) -> None:
        c = MusicCreator(self.ctx)
        rng = __import__("random").Random(1)
        grp = c._rhyme_for("chasing the night", rng, None)
        self.assertIsNotNone(grp)
        self.assertIn("night", [g.lower() for g in grp])
        line, _img, _em = c._build_line(
            "chorus", "x", "X", ["train"], "midnight train",
            STYLES["pop"], rng, end_group=("night", "light", "flight"))
        self.assertIn(line.split()[-1].lower(), ("night", "light", "flight"))
        self.assertIsNone(c._rhyme_for("qzxw qzxw", rng, None))

    def test_model_lyrics_uses_message_contract(self) -> None:
        # the model path must call router.chat with real Message objects
        # and SamplingParams — not raw dicts
        seen: dict[str, Any] = {}

        class FakeResp:
            ok = True
            text = "line one\nline two\nline three\nline four"

        class FakeRouter:
            def stats_snapshot(self) -> dict[str, Any]:
                return {"active": "test-model"}

            def chat(self, messages: Any, params: Any = None) -> FakeResp:
                seen["messages"] = messages
                seen["params"] = params
                return FakeResp()

        ctx = _context(router=FakeRouter())
        self.addCleanup(shutil.rmtree, ctx._root, True)
        c = MusicCreator(ctx)
        lines = c._model_lyrics("chorus", "trains", STYLES["pop"])
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(isinstance(m, Message)
                            for m in seen["messages"]),
                        f"expected Message objects, got {seen['messages']!r}")
        self.assertIsInstance(seen["params"], SamplingParams)

    def test_music_writer_tool_wrapper(self) -> None:
        import nomorals.media.music as mu

        reg = _registry(self.ctx)
        mu.register(reg)
        fn = reg.fns["music_writer"]
        styles = fn(action="styles")
        self.assertIn("afrobeats", styles["styles"])
        song = fn(action="compose", topic="harbor lights", seed=9,
                  with_midi=False)
        self.assertIn("harbor", song["topic"])
        self.assertEqual(song["midi_path"], "")
        with self.assertRaises(ToolError):
            fn(action="compose", topic="   ")
        with self.assertRaises(ToolError):
            fn(action="remix")


# ── video ───────────────────────────────────────────────────────────────────


class VideoScoreTests(unittest.TestCase):
    def test_youtube_beats_page(self) -> None:
        s_yt, src = _video_score("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(src, "youtube")
        self.assertGreaterEqual(s_yt, 10)
        s_page, src2 = _video_score("https://example.com/blog")
        self.assertEqual(s_page, 0.0)
        self.assertEqual(src2, "")

    def test_shorts_and_youtu_be(self) -> None:
        for url in ("https://youtube.com/shorts/abc123XYZ99",
                    "https://youtu.be/abc123XYZ99"):
            score, src = _video_score(url)
            self.assertGreaterEqual(score, 10, url)
            self.assertEqual(src, "youtube")

    def test_relevance(self) -> None:
        self.assertAlmostEqual(_relevance("lofi beats to study", "lofi beats"),
                               1.0)
        self.assertEqual(_relevance("cooking show", "lofi beats"), 0.0)
        self.assertEqual(_relevance("x", ""), 0.5)

    def test_youtube_id(self) -> None:
        self.assertEqual(
            VideoFinder._youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ")
        self.assertEqual(
            VideoFinder._youtube_id("https://youtu.be/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ")
        self.assertEqual(
            VideoFinder._youtube_id("https://youtube.com/shorts/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ")
        self.assertEqual(VideoFinder._youtube_id("https://example.com/"), "")


class VideoFindTests(MediaCoreBase):
    FIXTURES = [
        {"title": "best lofi mix 2026", "snippet": "lofi beats to relax",
         "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"},
        {"title": "lofi beats blog post", "snippet": "a page about lofi",
         "url": "https://example.com/lofi-blog"},
        {"title": "duplicate", "snippet": "dup",
         "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"},
        {"title": "vimeo short", "snippet": "lofi",
         "url": "https://vimeo.com/12345678"},
    ]

    def _finder(self) -> VideoFinder:
        f = VideoFinder(self.ctx)
        f._search = lambda *a, **k: list(self.FIXTURES)  # type: ignore[method-assign]
        return f

    def test_rank_dedup_and_count(self) -> None:
        res = self._finder().find("lofi beats", max_results=10)
        self.assertEqual(res["count"], 3)  # duplicate dropped
        self.assertEqual(res["results"][0]["source"], "youtube")
        self.assertTrue(res["results"][0]["is_video_url"])
        self.assertGreater(res["results"][0]["score"],
                           res["results"][-1]["score"])

    def test_max_results_cap(self) -> None:
        res = self._finder().find("lofi beats", max_results=2)
        self.assertEqual(res["count"], 2)

    def test_platform_pin_drops_non_matching(self) -> None:
        res = self._finder().find("lofi", platform="youtube")
        urls = [r["url"] for r in res["results"]]
        # the plain blog page is dropped; genuine video URLs survive the pin
        self.assertNotIn("https://example.com/lofi-blog", urls)
        self.assertIn("https://www.youtube.com/watch?v=aaaaaaaaaaa", urls)
        self.assertEqual(res["count"], 2)  # youtube + vimeo (a video URL)

    def test_empty_query_raises(self) -> None:
        with self.assertRaises(ToolError):
            self._finder().find("   ")

    def test_download_without_registry_fails_fast(self) -> None:
        with self.assertRaises(ToolError):
            self._finder().download("https://www.youtube.com/watch?v=x")

    def test_video_finder_tool_wrapper(self) -> None:
        import nomorals.media.video as vd

        reg = _registry(self.ctx)
        vd.register(reg)
        fn = reg.fns["video_finder"]
        plats = fn(action="platforms")
        self.assertIn("youtube", plats["platforms"])
        with self.assertRaises(ToolError):
            fn(action="teleport")


# ── media hub ───────────────────────────────────────────────────────────────


class MediaHubTests(MediaCoreBase):
    def setUp(self) -> None:
        super().setUp()
        import nomorals.media.playback as pb

        self._be = mock.patch.object(
            pb, "detect_backend", lambda: Backend("console"))
        self._be.start()
        self.addCleanup(self._be.stop)

    def test_run_song_console_backend_reports_playback(self) -> None:
        hub = MediaHub(self.ctx)
        out = hub.run("song", topic="harbor lights", style="lofi", seed=4)
        self.assertEqual(out["mode"], "song")
        self.assertIn("midi_path", out["song"])
        self.assertTrue(Path(out["song"]["midi_path"]).exists())
        self.assertEqual(out["playback"]["status"], "no-backend")

    def test_run_song_play_false(self) -> None:
        hub = MediaHub(self.ctx)
        out = hub.run("song", topic="harbor lights", seed=4, play=False)
        self.assertNotIn("playback", out)

    def test_run_video_no_results_is_honest(self) -> None:
        hub = MediaHub(self.ctx)
        hub.video._search = lambda *a, **k: []  # type: ignore[method-assign]
        out = hub.run("video", query="nothing anywhere ever xyz")
        self.assertIn("error", out)

    def test_run_unknown_mode_raises(self) -> None:
        hub = MediaHub(self.ctx)
        with self.assertRaises(ToolError):
            hub.run("teleport", topic="x")

    def test_run_needs_topic_or_query(self) -> None:
        hub = MediaHub(self.ctx)
        with self.assertRaises(ToolError):
            hub.run("song")

    def test_auto_send_target_no_gateway(self) -> None:
        self.assertEqual(MediaHub(self.ctx).auto_send_target(), ("", ""))

    def test_auto_send_target_disconnected(self) -> None:
        class Gw:
            def status(self) -> dict[str, Any]:
                return {"telegram": {"connected": False}}

        ctx = _context(gateway=Gw())
        self.addCleanup(shutil.rmtree, ctx._root, True)
        self.assertEqual(MediaHub(ctx).auto_send_target(), ("", ""))

    def test_auto_send_target_picks_owner_chat(self) -> None:
        class Gw:
            def status(self) -> dict[str, Any]:
                return {"telegram": {"connected": True}}

        ctx = _context(gateway=Gw())
        self.addCleanup(shutil.rmtree, ctx._root, True)
        ctx.db.execute(
            "INSERT INTO chats (id, platform, chat_id, is_owner, "
            "last_active) VALUES (?,?,?,?,?)",
            ("c1", "telegram", "111", 0, 100.0))
        ctx.db.execute(
            "INSERT INTO chats (id, platform, chat_id, is_owner, "
            "last_active) VALUES (?,?,?,?,?)",
            ("c2", "telegram", "999", 1, 50.0))
        self.assertEqual(MediaHub(ctx).auto_send_target(),
                         ("telegram", "999"))

    def test_podcast_transcribe_no_audio(self) -> None:
        hub = MediaHub(self.ctx)
        out = hub._podcast_transcribe("", "some show")
        self.assertIn("error", out)

    def test_podcast_transcribe_stt_failure_is_reported(self) -> None:
        hub = MediaHub(self.ctx)
        p = Path(self.ctx._root) / "ep.mp3"
        p.write_bytes(b"x" * 100)
        out = hub._podcast_transcribe(str(p), "some show")
        self.assertIn("stt_error", out)
        self.assertIsNone(out["transcript"])

    def test_styles_listing(self) -> None:
        styles = MediaHub(self.ctx).styles()
        self.assertIn("amapiano", styles)
        self.assertEqual(styles["amapiano"]["tempo"], [110, 115])

    def test_summarize_extractive(self) -> None:
        import nomorals.media as M

        text = ("The drums are loud and heavy tonight. The crowd is quiet. "
                "The drums carry the whole song forward. "
                "Nobody remembers the quiet verses.")
        s = M._summarize(text)
        self.assertTrue(s)
        self.assertLessEqual(len(s.split(". ")), 8)

    def test_chapters(self) -> None:
        import nomorals.media as M

        words = " ".join(f"word{i}" for i in range(600))
        ch = M._chapters(words, max_chapters=4)
        self.assertTrue(1 <= len(ch) <= 5)
        self.assertTrue(all(c["title"] and c["start"] for c in ch))
        self.assertEqual(M._chapters(""), [])


class DetectBackendTests(unittest.TestCase):
    def test_detect_backend_returns_backend(self) -> None:
        b = detect_backend()
        self.assertIsInstance(b, Backend)
        self.assertTrue(b.name)
        d = b.describe()
        self.assertIn("name", d)


if __name__ == "__main__":
    unittest.main()
