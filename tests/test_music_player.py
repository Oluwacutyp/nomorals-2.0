"""Music-player acceptance tests (music-1.0 wave).

Covers the full player experience built on PlaybackEngine:

* library.read_metadata — mutagen → ffprobe → filename fallback chain;
* library.MusicLibrary — playlists (CRUD, add/remove/move/clear),
  history (record/list/dedup/clear/top), favorites (like/unlike/list),
  search across every source, stats;
* PlaybackEngine — metadata-enriched add/queue, shuffle (current stays
  first), repeat modes (+ mpv loop mapping), move with playhead follow,
  rich now(), playlist loading, history auto-record on play/next/prev,
  mpv auto-advance position sync in status();
* the ``player`` / ``music_library`` tool wrappers;
* the ``nm music`` CLI dispatcher end to end.
"""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.core.errors import ToolError
from nomorals.media import MediaHub
from nomorals.media.library import MusicLibrary, read_metadata
from nomorals.media import library as lib_mod
from nomorals.media.playback import Backend, PlaybackEngine
from nomorals.media import playback as pb_mod
from nomorals.storage.db import Database


# ── fixtures ────────────────────────────────────────────────────────────────


def _context(**over: Any) -> SimpleNamespace:
    root = tempfile.mkdtemp(prefix="music_player_")
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


class PlayerBase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _context()
        self.addCleanup(shutil.rmtree, self.ctx._root, True)
        be = mock.patch.object(pb_mod, "detect_backend",
                               lambda: Backend("console"))
        be.start()
        self.addCleanup(be.stop)

    def _clip(self, name: str = "song.mp3") -> str:
        p = Path(self.ctx._root) / name
        p.write_bytes(b"fake-audio")
        return name  # workspace-relative, as the tools expect


# ── metadata ────────────────────────────────────────────────────────────────


class MetadataTests(PlayerBase):
    def test_filename_fallback_for_unreadable_file(self) -> None:
        name = self._clip("mystery track.mp3")
        meta = read_metadata(str(Path(self.ctx._root) / name))
        self.assertEqual(meta["title"], "mystery track")
        self.assertEqual(meta["artist"], "")
        self.assertEqual(meta["duration"], 0.0)

    def test_url_never_probed(self) -> None:
        with mock.patch.object(lib_mod.shutil, "which",
                               side_effect=AssertionError("no probing")):
            meta = read_metadata("https://example.com/tunes/cool song.mp3")
        self.assertEqual(meta["title"], "cool song.mp3")

    def test_empty_path_is_blank(self) -> None:
        self.assertEqual(read_metadata("")["title"], "")

    def test_ffprobe_result_wins_over_fallback(self) -> None:
        fake = {"title": "Real Title", "artist": "Real Artist",
                "album": "Real Album", "genre": "jazz", "duration": 183.5}
        with mock.patch.object(lib_mod, "_metadata_mutagen",
                               return_value=None), \
             mock.patch.object(lib_mod, "_metadata_ffprobe",
                               return_value=fake):
            meta = read_metadata("/tmp/whatever.mp3")
        self.assertEqual(meta, fake)

    def test_mutagen_result_wins_over_ffprobe(self) -> None:
        mut = {"title": "Mut Title", "artist": "Mut Artist", "album": "",
               "genre": "", "duration": 200.0}
        probe = {"title": "Probe Title", "artist": "Probe Artist",
                 "album": "", "genre": "", "duration": 199.0}
        with mock.patch.object(lib_mod, "_metadata_mutagen",
                               return_value=mut), \
             mock.patch.object(lib_mod, "_metadata_ffprobe",
                               return_value=probe) as ff:
            meta = read_metadata("/tmp/x.mp3")
        self.assertEqual(meta["title"], "Mut Title")
        ff.assert_not_called()

    def test_broken_readers_still_fall_back(self) -> None:
        with mock.patch.object(lib_mod, "_metadata_mutagen",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(lib_mod, "_metadata_ffprobe",
                               side_effect=RuntimeError("boom")):
            meta = read_metadata("/tmp/fallback.mp3")
        self.assertEqual(meta["title"], "fallback")


# ── playlists ───────────────────────────────────────────────────────────────


class PlaylistTests(PlayerBase):
    def setUp(self) -> None:
        super().setUp()
        self.lib = MusicLibrary(self.ctx)

    def test_create_list_duplicate_and_empty(self) -> None:
        self.assertEqual(self.lib.create_playlist("gym")["created"], "gym")
        names = [p["name"] for p in self.lib.playlists()]
        self.assertIn("gym", names)
        with self.assertRaises(ToolError):
            self.lib.create_playlist("gym")
        with self.assertRaises(ToolError):
            self.lib.create_playlist("   ")

    def test_rename_and_conflicts(self) -> None:
        self.lib.create_playlist("a")
        self.lib.create_playlist("b")
        self.assertEqual(self.lib.rename_playlist("a", "c")["to"], "c")
        with self.assertRaises(ToolError):
            self.lib.rename_playlist("c", "b")  # name clash
        with self.assertRaises(ToolError):
            self.lib.rename_playlist("nope", "x")
        with self.assertRaises(ToolError):
            self.lib.rename_playlist("c", "  ")

    def test_delete(self) -> None:
        self.lib.create_playlist("gone")
        self.lib.playlist_add("gone", self._clip("d.mp3"))
        res = self.lib.delete_playlist("gone")
        self.assertEqual(res["tracks_removed"], 1)
        self.assertEqual(self.lib.playlists(), [])
        with self.assertRaises(ToolError):
            self.lib.delete_playlist("gone")

    def test_add_files_and_url_with_metadata(self) -> None:
        self.lib.create_playlist("mix")
        a, b = self._clip("one.mp3"), self._clip("two.mp3")
        res = self.lib.playlist_add("mix", a, b,
                                    "https://example.com/stream.mp3")
        self.assertEqual(len(res["added"]), 3)
        self.assertEqual(res["tracks"], 3)
        pl = self.lib.playlist("mix")
        self.assertEqual([i["title"] for i in pl["items"]],
                         ["one", "two", "stream.mp3"])
        self.assertEqual(pl["items"][2]["kind"], "url")
        with self.assertRaises(ToolError):
            self.lib.playlist_add("mix", "does-not-exist.mp3")
        with self.assertRaises(ToolError):
            self.lib.playlist_add("nope", a)

    def test_remove_move_clear(self) -> None:
        self.lib.create_playlist("p")
        clips = [self._clip(f"t{i}.mp3") for i in range(3)]
        self.lib.playlist_add("p", *clips)
        self.lib.playlist_move("p", 0, 2)
        titles = [i["title"] for i in self.lib.playlist("p")["items"]]
        self.assertEqual(titles, ["t1", "t2", "t0"])
        with self.assertRaises(ToolError):
            self.lib.playlist_move("p", 9, 0)
        rm = self.lib.playlist_remove("p", 0)
        self.assertEqual(rm["removed"], "t1")
        self.assertEqual(rm["tracks"], 2)
        with self.assertRaises(ToolError):
            self.lib.playlist_remove("p", 7)
        self.assertEqual(self.lib.playlist_clear("p")["cleared"], "p")
        self.assertEqual(self.lib.playlist("p")["tracks"], 0)

    def test_playlists_carry_counts(self) -> None:
        self.lib.create_playlist("counts")
        self.lib.playlist_add("counts", self._clip("x.mp3"))
        pls = self.lib.playlists()
        self.assertEqual(pls[0]["tracks"], 1)

    def test_save_queue_as_playlist_replaces(self) -> None:
        clips = [self._clip("s1.mp3"), self._clip("s2.mp3")]
        eng = PlaybackEngine(self.ctx)
        eng.add(*clips)
        res = self.lib.save_queue_as_playlist("jam", eng.queue())
        self.assertEqual(res["saved"], "jam")
        self.assertEqual(res["tracks"], 2)
        # queue contents, in order, with metadata
        pl = self.lib.playlist("jam")
        self.assertEqual([i["title"] for i in pl["items"]], ["s1", "s2"])
        # second save replaces rather than appends
        eng.add(self._clip("s3.mp3"))
        res = self.lib.save_queue_as_playlist("jam", eng.queue())
        self.assertEqual(self.lib.playlist("jam")["tracks"], 3)
        with self.assertRaises(ToolError):
            self.lib.save_queue_as_playlist("jam", [])
        with self.assertRaises(ToolError):
            self.lib.save_queue_as_playlist("  ", eng.queue())

    def test_export_import_m3u_round_trip(self) -> None:
        self.lib.create_playlist("gym")
        self.lib.playlist_add("gym", self._clip("a.mp3"),
                              "https://example.com/stream.mp3")
        res = self.lib.export_m3u("gym", "gym.m3u")
        self.assertEqual(res["tracks"], 2)
        body = (Path(self.ctx._root) / "gym.m3u").read_text(encoding="utf-8")
        self.assertTrue(body.startswith("#EXTM3U"))
        self.assertIn("#EXTINF:", body)
        self.assertIn("https://example.com/stream.mp3", body)
        # relative workspace path round-trips
        imp = self.lib.import_m3u("fresh", "gym.m3u")
        self.assertEqual(imp["added"], 2)
        self.assertEqual(imp["skipped"], 0)
        pl = self.lib.playlist("fresh")
        self.assertEqual(pl["tracks"], 2)
        self.assertEqual(pl["items"][1]["kind"], "url")
        # stale local entries are skipped, not fatal
        (Path(self.ctx._root) / "stale.m3u").write_text(
            "#EXTM3U\n#EXTINF:0,gone\nmissing.mp3\n", encoding="utf-8")
        imp = self.lib.import_m3u("fresh", "stale.m3u")
        self.assertEqual(imp["added"], 0)
        self.assertEqual(imp["skipped"], 1)
        self.assertEqual(self.lib.playlist("fresh")["tracks"], 2)
        with self.assertRaises(ToolError):
            self.lib.export_m3u("gym", "../escape.m3u")
        with self.assertRaises(ToolError):
            self.lib.export_m3u("nope", "x.m3u")
        with self.assertRaises(ToolError):
            self.lib.import_m3u("x", "no-such-file.m3u")


# ── history & favorites ─────────────────────────────────────────────────────


class HistoryFavoriteTests(PlayerBase):
    def setUp(self) -> None:
        super().setUp()
        self.lib = MusicLibrary(self.ctx)

    def test_record_list_clear(self) -> None:
        self.lib.record_played("/m/a.mp3", title="A", artist="Art")
        self.lib.record_played("/m/b.mp3", title="B", _dedup_window=0)
        hist = self.lib.history()
        self.assertEqual([h["title"] for h in hist], ["B", "A"])  # newest first
        self.assertEqual(self.lib.clear_history()["removed"], 2)
        self.assertEqual(self.lib.history(), [])

    def test_dedup_window(self) -> None:
        r1 = self.lib.record_played("/m/a.mp3", title="A")
        r2 = self.lib.record_played("/m/a.mp3", title="A")
        self.assertTrue(r1["recorded"])
        self.assertFalse(r2["recorded"])
        self.assertEqual(len(self.lib.history()), 1)
        self.lib.record_played("/m/a.mp3", title="A", _dedup_window=0)
        self.assertEqual(len(self.lib.history()), 2)

    def test_top_played(self) -> None:
        for _ in range(3):
            self.lib.record_played("/m/a.mp3", title="A", _dedup_window=0)
        self.lib.record_played("/m/b.mp3", title="B", _dedup_window=0)
        top = self.lib.top_played()
        self.assertEqual(top[0]["path"], "/m/a.mp3")
        self.assertEqual(top[0]["plays"], 3)
        self.assertEqual(top[1]["plays"], 1)

    def test_like_unlike_flow(self) -> None:
        p = str(Path(self.ctx._root) / self._clip("fav.mp3"))
        self.assertFalse(self.lib.is_liked(p))
        liked = self.lib.like(p)
        self.assertTrue(liked["liked"])
        self.assertEqual(liked["title"], "fav")
        self.assertTrue(self.lib.is_liked(p))
        # liking again refreshes rather than duplicating
        self.lib.like(p, title="Fav Song")
        favs = self.lib.favorites()
        self.assertEqual(len(favs), 1)
        self.assertEqual(favs[0]["title"], "Fav Song")
        self.assertEqual(self.lib.unlike(p)["title"], "Fav Song")
        self.assertFalse(self.lib.is_liked(p))
        with self.assertRaises(ToolError):
            self.lib.unlike(p)
        with self.assertRaises(ToolError):
            self.lib.like("  ")

    def test_search_across_sources(self) -> None:
        fav = str(Path(self.ctx._root) / self._clip("loved.mp3"))
        hist_p = str(Path(self.ctx._root) / self._clip("heard.mp3"))
        q_p = self._clip("queued.mp3")
        pl_p = self._clip("mixed.mp3")
        self.lib.like(fav)
        self.lib.record_played(hist_p, title="heard")
        self.lib.create_playlist("mp3mixer")
        self.lib.playlist_add("mp3mixer", pl_p)
        PlaybackEngine(self.ctx).add(q_p)
        res = self.lib.search("mp3")
        self.assertTrue(res["favorites"])
        self.assertTrue(res["history"])
        self.assertIn("mp3mixer", res["playlists"])
        self.assertTrue(res["playlist_tracks"])
        self.assertTrue(res["queue"])
        with self.assertRaises(ToolError):
            self.lib.search("   ")

    def test_stats(self) -> None:
        self.lib.create_playlist("s")
        self.lib.playlist_add("s", self._clip("s1.mp3"), self._clip("s2.mp3"))
        self.lib.like(str(Path(self.ctx._root) / self._clip("s1.mp3")))
        self.lib.record_played("/m/s1.mp3", title="s1", _dedup_window=0)
        st = self.lib.stats()
        self.assertEqual(st["playlists"], 1)
        self.assertEqual(st["playlist_tracks"], 2)
        self.assertEqual(st["favorites"], 1)
        self.assertEqual(st["plays"], 1)
        self.assertEqual(st["unique_tracks_played"], 1)


# ── engine: queue management ────────────────────────────────────────────────


class EngineQueueTests(PlayerBase):
    def setUp(self) -> None:
        super().setUp()
        self.eng = PlaybackEngine(self.ctx)
        self.clips = [self._clip(f"q{i}.mp3") for i in range(4)]
        self.eng.add(*self.clips)

    def test_add_enriches_metadata(self) -> None:
        q = self.eng.queue()
        self.assertEqual(len(q), 4)
        self.assertEqual(q[0]["title"], "q0")
        self.assertIn("artist", q[0])
        self.assertIn("album", q[0])
        self.assertIn("duration", q[0])

    def test_shuffle_keeps_all_current_first(self) -> None:
        first = self.eng.queue()[0]["path"]
        res = self.eng.shuffle(True)
        self.assertTrue(res["shuffle"])
        q = self.eng.queue()
        self.assertEqual(len(q), 4)
        self.assertEqual(q[0]["path"], first)  # current stays first
        self.assertEqual(
            sorted(t["path"] for t in q),
            sorted(str(Path(self.ctx._root) / c) for c in self.clips))
        self.assertEqual(self.eng._state["position"], 0)
        # toggle back off keeps the order (no fake restore)
        res = self.eng.shuffle()
        self.assertFalse(res["shuffle"])
        # explicit off
        self.eng.shuffle(True)
        self.assertFalse(self.eng.shuffle(False)["shuffle"])

    def test_shuffle_empty_raises(self) -> None:
        self.eng.clear()
        with self.assertRaises(ToolError):
            self.eng.shuffle(True)

    def test_shuffle_resyncs_live_mpv(self) -> None:
        self.eng.backend = Backend("mpv", "/bin/fake-mpv")
        with mock.patch.object(self.eng, "_mpv_alive",
                               return_value=True), \
             mock.patch.object(self.eng, "_mpv_play_at",
                               return_value=(True, "")) as mpa:
            res = self.eng.shuffle(True)
        self.assertTrue(res["resynced"])
        mpa.assert_called_once_with(0)

    def test_repeat_modes_and_validation(self) -> None:
        self.assertEqual(self.eng.repeat()["repeat"], "off")
        self.assertEqual(self.eng.repeat("all")["repeat"], "all")
        self.assertEqual(self.eng.repeat()["repeat"], "all")
        self.assertEqual(self.eng.repeat("one")["repeat"], "one")
        self.assertEqual(self.eng.repeat("OFF")["repeat"], "off")
        with self.assertRaises(ToolError):
            self.eng.repeat("forever")

    def test_repeat_drives_mpv_loop_props(self) -> None:
        self.eng.backend = Backend("mpv", "/bin/fake-mpv")
        calls: list[list[Any]] = []
        with mock.patch.object(self.eng, "_mpv_transport",
                               side_effect=lambda c: calls.append(c) or True):
            self.eng.repeat("all")
        self.assertIn(["set_property", "loop-playlist", "inf"], calls)
        self.assertIn(["set_property", "loop-file", "no"], calls)
        calls.clear()
        with mock.patch.object(self.eng, "_mpv_transport",
                               side_effect=lambda c: calls.append(c) or True):
            self.eng.repeat("one")
        self.assertIn(["set_property", "loop-file", "inf"], calls)

    def test_move_reorders_and_follows_playhead(self) -> None:
        # playhead on q0; moving q0 to the end must move the playhead too
        res = self.eng.move(0, 3)
        self.assertEqual(res["from"], 0)
        self.assertEqual(res["to"], 3)
        titles = [t["title"] for t in self.eng.queue()]
        self.assertEqual(titles, ["q1", "q2", "q3", "q0"])
        self.assertEqual(self.eng._state["position"], 3)
        # moving something before the playhead shifts it correctly
        self.eng.move(0, 1)
        self.assertEqual(self.eng._state["position"], 3)
        self.assertEqual(
            [t["title"] for t in self.eng.queue()], ["q2", "q1", "q3", "q0"])
        with self.assertRaises(ToolError):
            self.eng.move(9, 0)
        self.eng.clear()
        with self.assertRaises(ToolError):
            self.eng.move(0, 1)

    def test_now_playing_shape(self) -> None:
        now = self.eng.now()
        self.assertEqual(now["track"]["title"], "q0")
        self.assertEqual(now["position"], 0)
        self.assertEqual(now["queue"], 4)
        self.assertFalse(now["playing"])
        self.assertFalse(now["liked"])
        self.assertEqual(now["repeat"], "off")
        self.assertFalse(now["shuffle"])
        self.assertEqual(now["volume"], 80)
        self.assertEqual(now["backend"], "console")
        # liking flips the flag
        MusicLibrary(self.ctx).like(now["track"]["path"])
        self.assertTrue(self.eng.now()["liked"])

    def test_now_reports_live_mpv_progress(self) -> None:
        self.eng.backend = Backend("mpv", "/bin/fake-mpv")
        props = {"playback-status": "playing", "time-pos": 42.0,
                 "duration": 200.0}
        with mock.patch.object(self.eng, "_mpv_alive",
                               return_value=True), \
             mock.patch.object(self.eng, "_mpv_get_props",
                               return_value=props):
            now = self.eng.now()
        self.assertTrue(now["playing"])
        self.assertEqual(now["time_pos"], 42.0)
        self.assertEqual(now["duration"], 200.0)
        self.assertEqual(now["time_remaining"], 158.0)

    def test_load_playlist(self) -> None:
        lib = MusicLibrary(self.ctx)
        lib.create_playlist("road")
        lib.playlist_add("road", self.clips[2], self.clips[0])
        res = self.eng.load_playlist("road", autoplay=False)
        self.assertEqual(res["loaded"], 2)
        self.assertEqual([t["title"] for t in self.eng.queue()],
                         ["q2", "q0"])
        with self.assertRaises(ToolError):
            self.eng.load_playlist("nope")
        lib.create_playlist("empty")
        with self.assertRaises(ToolError):
            self.eng.load_playlist("empty")

    def test_load_playlist_autoplay_records_history(self) -> None:
        lib = MusicLibrary(self.ctx)
        lib.create_playlist("road")
        lib.playlist_add("road", self.clips[0])
        # console backend: queue loads, nothing actually plays, no history
        res = self.eng.load_playlist("road")
        self.assertEqual(res["status"], "no-backend")
        self.assertEqual(len(self.eng.queue()), 1)
        self.assertEqual(lib.history(), [])

    def test_save_playlist_from_queue(self) -> None:
        res = self.eng.save_playlist("jam")
        self.assertEqual(res["saved"], "jam")
        self.assertEqual(res["tracks"], 4)
        lib = MusicLibrary(self.ctx)
        pl = lib.playlist("jam")
        self.assertEqual([i["title"] for i in pl["items"]],
                         [t["title"] for t in self.eng.queue()])
        # the queue itself is untouched
        self.assertEqual(len(self.eng.queue()), 4)
        fresh = PlaybackEngine(self.ctx)
        fresh.clear()
        with self.assertRaises(ToolError):
            fresh.save_playlist("x")

    def test_add_missing_file_raises_tool_error(self) -> None:
        with self.assertRaises(ToolError):
            self.eng.add("does-not-exist.mp3")


class EngineHistoryTests(PlayerBase):
    """play()/next()/prev() on a working backend log to history."""

    def setUp(self) -> None:
        super().setUp()
        self.eng = PlaybackEngine(self.ctx)
        self.eng.backend = Backend("mpg123", "/bin/fake-mpg123")
        self.a = self._clip("a.mp3")
        self.b = self._clip("b.mp3")
        self.spawned: list[int] = []

        def fake_popen(cmd: Any, **kw: Any) -> SimpleNamespace:
            pid = 2000 + len(self.spawned)
            self.spawned.append(pid)
            return SimpleNamespace(pid=pid)

        p = mock.patch.object(pb_mod.subprocess, "Popen", fake_popen)
        p.start()
        self.addCleanup(p.stop)
        p2 = mock.patch.object(pb_mod.os, "killpg", lambda pg, sig: None)
        p2.start()
        self.addCleanup(p2.stop)
        p3 = mock.patch.object(pb_mod.os, "getpgid", lambda pid: pid)
        p3.start()
        self.addCleanup(p3.stop)

    def test_play_next_prev_record_history(self) -> None:
        lib = MusicLibrary(self.ctx)
        self.eng.add(self.a, self.b)
        self.eng.play()
        self.eng.next()
        self.eng.prev()
        hist = lib.history(limit=10)
        self.assertEqual(len(hist), 3)
        self.assertEqual([h["source"] for h in hist],
                         ["prev", "next", "play"])
        self.assertEqual(hist[0]["title"], "a")

    def test_failed_start_records_nothing(self) -> None:
        lib = MusicLibrary(self.ctx)
        self.eng.add("https://example.com/x.mp3")  # mpg123 can't stream
        res = self.eng.play()
        self.assertEqual(res["status"], "error")
        self.assertEqual(lib.history(), [])


class MpvSyncTests(PlayerBase):
    def test_status_follows_mpv_auto_advance(self) -> None:
        eng = PlaybackEngine(self.ctx)
        eng.backend = Backend("mpv", "/bin/fake-mpv")
        a, b = self._clip("a.mp3"), self._clip("b.mp3")
        eng.add(a, b)
        props = {"playback-status": "playing", "playlist-pos": 1,
                 "time-pos": 3.0, "volume": 80}
        with mock.patch.object(eng, "_mpv_alive", return_value=True), \
             mock.patch.object(eng, "_mpv_get_props", return_value=props):
            st = eng.status()
        self.assertEqual(eng._state["position"], 1)
        self.assertEqual(st["current"], "b")  # filename-stem title
        hist = MusicLibrary(self.ctx).history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["source"], "auto-advance")
        self.assertEqual(hist[0]["title"], "b")

    def test_status_ignores_idle_playlist_pos(self) -> None:
        eng = PlaybackEngine(self.ctx)
        eng.backend = Backend("mpv", "/bin/fake-mpv")
        eng.add(self._clip("a.mp3"))
        props = {"playback-status": "idle", "playlist-pos": None,
                 "time-pos": None, "volume": 80}
        with mock.patch.object(eng, "_mpv_alive", return_value=True), \
             mock.patch.object(eng, "_mpv_get_props", return_value=props):
            st = eng.status()
        self.assertEqual(eng._state["position"], 0)
        self.assertEqual(MusicLibrary(self.ctx).history(), [])


class MediaHubPlayerTests(PlayerBase):
    def test_hub_passthroughs(self) -> None:
        hub = MediaHub(self.ctx)
        hub.add(self._clip("h.mp3"))
        self.assertEqual(hub.now()["track"]["title"], "h")
        self.assertEqual(hub.repeat("all")["repeat"], "all")
        self.assertTrue(hub.shuffle(True)["shuffle"])
        self.assertEqual(hub.move(0, 0)["to"], 0)
        lib = hub.library
        lib.create_playlist("hp")
        lib.playlist_add("hp", self._clip("h2.mp3"))
        res = hub.load_playlist("hp", autoplay=False)
        self.assertEqual(res["loaded"], 1)
        saved = hub.save_playlist("hp2")
        self.assertEqual(saved["tracks"], 1)
        self.assertEqual(hub.library.playlist("hp2")["tracks"], 1)


# ── tool wrappers ───────────────────────────────────────────────────────────


class ToolWrapperTests(PlayerBase):
    def setUp(self) -> None:
        super().setUp()
        reg = _registry(self.ctx)
        pb_mod.register(reg)
        lib_mod.register(reg)
        self.player = reg.fns["player"]
        self.lib_tool = reg.fns["music_library"]
        for n in ("a.mp3", "b.mp3", "c.mp3"):
            (Path(self.ctx._root) / n).write_bytes(b"x")
        self.player(action="add", targets="a.mp3|b.mp3|c.mp3")

    def test_player_new_actions(self) -> None:
        self.assertTrue(self.player(action="shuffle", on="on")["shuffle"])
        self.assertEqual(self.player(action="repeat", mode="all")["repeat"],
                         "all")
        self.assertEqual(self.player(action="repeat")["repeat"], "all")
        moved = self.player(action="move", index=0, to=2)
        self.assertEqual(moved["to"], 2)
        now = self.player(action="now")
        self.assertIn("track", now)
        self.assertIn("liked", now)
        with self.assertRaises(ToolError):
            self.player(action="repeat", mode="bogus")
        with self.assertRaises(ToolError):
            self.player(action="dance")

    def test_player_playlist_play(self) -> None:
        self.lib_tool(action="playlist_create", name="toolpl")
        self.lib_tool(action="playlist_add", name="toolpl",
                      targets="a.mp3|b.mp3")
        res = self.player(action="playlist_play", name="toolpl")
        self.assertEqual(res["playlist"], "toolpl")
        self.assertEqual(res["loaded"], 2)
        self.assertEqual(len(self.player(action="queue")["queue"]), 2)
        with self.assertRaises(ToolError):
            self.player(action="playlist_play", name="")
        with self.assertRaises(ToolError):
            self.player(action="playlist_play", name="missing")

    def test_player_playlist_save(self) -> None:
        res = self.player(action="playlist_save", name="toolsave")
        self.assertEqual(res["saved"], "toolsave")
        self.assertEqual(res["tracks"], 3)
        pl = self.lib_tool(action="playlist", name="toolsave")
        self.assertEqual([i["title"] for i in pl["items"]], ["a", "b", "c"])
        with self.assertRaises(ToolError):
            self.player(action="playlist_save", name="")

    def test_library_tool_export_import(self) -> None:
        t = self.lib_tool
        t(action="playlist_create", name="xport")
        t(action="playlist_add", name="xport", targets="a.mp3")
        res = t(action="playlist_export", name="xport", path="x.m3u")
        self.assertEqual(res["tracks"], 1)
        res = t(action="playlist_import", name="back", path="x.m3u")
        self.assertEqual(res["added"], 1)
        self.assertEqual(t(action="playlist", name="back")["tracks"], 1)
        with self.assertRaises(ToolError):
            t(action="playlist_export", name="xport", path="")
        with self.assertRaises(ToolError):
            t(action="playlist_import", name="back", path="")

    def test_library_tool_full_sweep(self) -> None:
        t = self.lib_tool
        self.assertEqual(t(action="playlist_create", name="sweep",
                           description="d")["created"], "sweep")
        self.assertIn("sweep", [p["name"] for p in
                                t(action="playlists")["playlists"]])
        added = t(action="playlist_add", name="sweep", targets="a.mp3|b.mp3")
        self.assertEqual(len(added["added"]), 2)
        self.assertEqual(t(action="playlist", name="sweep")["tracks"], 2)
        self.assertEqual(t(action="playlist_move", name="sweep", index=0,
                           to=1)["to"], 1)
        self.assertEqual(t(action="playlist_remove", name="sweep",
                           index=0)["tracks"], 1)
        self.assertEqual(t(action="playlist_rename", name="sweep",
                           new_name="sweep2")["to"], "sweep2")
        # history via engine play would need a backend; record directly
        lib = MusicLibrary(self.ctx)
        lib.record_played("/m/x.mp3", title="X tune", _dedup_window=0)
        lib.record_played("/m/x.mp3", title="X tune", _dedup_window=0)
        self.assertEqual(len(t(action="history", limit=5)["history"]), 2)
        self.assertEqual(t(action="top", limit=5)["top"][0]["plays"], 2)
        # like the current track (no path → engine's current)
        liked = t(action="like")
        self.assertTrue(liked["liked"])
        self.assertEqual(t(action="liked")["favorites"][0]["title"], "a")
        un = t(action="unlike")
        self.assertTrue(un["unliked"])
        self.assertEqual(t(action="liked")["favorites"], [])
        # search + stats
        found = t(action="search", query="mp3")
        self.assertTrue(found["queue"])
        stats = t(action="stats")
        self.assertEqual(stats["playlists"], 1)
        self.assertEqual(stats["queue"], 3)
        self.assertEqual(t(action="playlist_clear", name="sweep2")["cleared"],
                         "sweep2")
        self.assertEqual(t(action="playlist_delete", name="sweep2")
                         ["deleted"], "sweep2")
        self.assertEqual(t(action="history_clear")["removed"], 2)
        with self.assertRaises(ToolError):
            t(action="playlist_add", name="sweep2", targets="")
        with self.assertRaises(ToolError):
            t(action="bogus")

    def test_like_needs_nonempty_queue(self) -> None:
        self.player(action="clear")
        with self.assertRaises(ToolError):
            self.lib_tool(action="like")


# ── CLI ─────────────────────────────────────────────────────────────────────


class FakeTools:
    def __init__(self, fns: dict[str, Any]) -> None:
        self.fns = fns

    def call(self, tool: str, **kw: Any) -> SimpleNamespace:
        try:
            return SimpleNamespace(ok=True, value=self.fns[tool](**kw),
                                   error="")
        except Exception as exc:  # noqa: BLE001 - test shim
            return SimpleNamespace(ok=False, value=None, error=str(exc))


def _ns(**over: Any) -> argparse.Namespace:
    base = dict(action="status", topic="", extra=[], title="", style="pop",
                key="", seed="0", name="", mode="", to="", limit="20",
                desc="", json=False)
    base.update(over)
    return argparse.Namespace(**base)


class MusicCliTests(PlayerBase):
    def setUp(self) -> None:
        super().setUp()
        reg = _registry(self.ctx)
        pb_mod.register(reg)
        lib_mod.register(reg)
        self.tools = FakeTools(reg.fns)
        self.ctx.tools = self.tools
        from nomorals.cmdline.commands.music import _cmd_music
        self.cmd = _cmd_music

    def _run(self, **kw: Any) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.cmd(_ns(**kw), self.ctx)
        return rc, buf.getvalue()

    def _clips(self) -> None:
        for n in ("a.mp3", "b.mp3"):
            (Path(self.ctx._root) / n).write_bytes(b"x")

    def test_queue_lifecycle(self) -> None:
        rc, out = self._run(action="queue")
        self.assertEqual(rc, 0)
        self.assertIn("empty", out)
        self._clips()
        rc, out = self._run(action="add", topic="a.mp3", extra=["b.mp3"])
        self.assertEqual(rc, 0)
        self.assertIn("added 2", out)
        rc, out = self._run(action="queue")
        self.assertIn("0. a\n", out)
        rc, out = self._run(action="move", topic="0", extra=["1"])
        self.assertEqual(rc, 0)
        self.assertIn("→ #1", out)
        rc, out = self._run(action="remove", topic="0")
        self.assertEqual(rc, 0)
        rc, out = self._run(action="clear")
        self.assertEqual(rc, 0)
        self.assertIn("cleared", out)

    def test_transport_and_now(self) -> None:
        self._clips()
        self._run(action="add", topic="a.mp3")
        rc, out = self._run(action="play")
        self.assertEqual(rc, 0)
        self.assertIn("no audio backend", out)  # console backend is honest
        rc, out = self._run(action="now")
        self.assertEqual(rc, 0)
        self.assertIn("♪ a\n", out)
        rc, out = self._run(action="status")
        self.assertEqual(rc, 0)
        self.assertIn("stopped", out)
        rc, out = self._run(action="play", topic="0")
        self.assertEqual(rc, 0)
        rc, out = self._run(action="play", topic="b.mp3")
        self.assertEqual(rc, 0)  # file target: add + play
        for act in ("pause", "resume", "stop", "next", "prev"):
            rc, _ = self._run(action=act)
            self.assertEqual(rc, 0, act)

    def test_shuffle_repeat_volume_seek(self) -> None:
        self._clips()
        self._run(action="add", topic="a.mp3", extra=["b.mp3"])
        rc, out = self._run(action="shuffle", topic="on")
        self.assertEqual(rc, 0)
        self.assertIn("shuffle on", out)
        rc, out = self._run(action="shuffle")
        self.assertIn("shuffle off", out)
        rc, out = self._run(action="repeat", topic="all")
        self.assertIn("repeat: all", out)
        rc, _ = self._run(action="repeat", topic="bogus")
        self.assertEqual(rc, 1)
        rc, out = self._run(action="volume", topic="60")
        self.assertIn("volume 60", out)
        rc, out = self._run(action="seek", topic="30")
        self.assertIn("0:30", out)

    def test_playlists_end_to_end(self) -> None:
        self._clips()
        rc, out = self._run(action="playlists")
        self.assertIn("no playlists", out)
        rc, out = self._run(action="playlist-create", topic="gym")
        self.assertIn("gym", out)
        rc, out = self._run(action="playlist-add", topic="gym",
                            extra=["a.mp3", "b.mp3"])
        self.assertIn("added 2", out)
        rc, out = self._run(action="playlist", topic="gym")
        self.assertIn("0. a\n", out)
        rc, out = self._run(action="playlist-rename", topic="gym",
                            extra=["workout"])
        self.assertIn("workout", out)
        rc, out = self._run(action="playlist-remove", topic="workout",
                            extra=["0"])
        self.assertEqual(rc, 0)
        rc, out = self._run(action="playlist-play", topic="workout")
        self.assertIn("playing playlist workout", out)
        rc, out = self._run(action="playlist-delete", topic="workout")
        self.assertIn("deleted playlist workout", out)

    def test_history_favorites_search_stats(self) -> None:
        self._clips()
        self._run(action="add", topic="a.mp3")
        rc, out = self._run(action="like")
        self.assertIn("♥ liked", out)
        rc, out = self._run(action="liked")
        self.assertIn("a\n", out)
        rc, out = self._run(action="search", topic="mp3")
        self.assertIn("queue:", out)
        rc, out = self._run(action="stats")
        self.assertIn("favorites: 1", out)
        rc, out = self._run(action="unlike")
        self.assertEqual(rc, 0)
        rc, out = self._run(action="history")
        self.assertIn("no history", out)
        rc, out = self._run(action="top")
        self.assertIn("no plays", out)

    def test_cli_errors(self) -> None:
        rc, _ = self._run(action="add")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="remove", topic="x")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="move", topic="0")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="playlist-create")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="search")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="bogus-action")
        self.assertEqual(rc, 2)

    def test_playlist_move_clear_cli(self) -> None:
        self._clips()
        self._run(action="playlist-create", topic="gym")
        self._run(action="playlist-add", topic="gym",
                  extra=["a.mp3", "b.mp3"])
        rc, out = self._run(action="playlist-move", topic="gym",
                            extra=["0", "1"])
        self.assertEqual(rc, 0)
        self.assertIn("→ #1", out)
        pl = self.tools.fns["music_library"](action="playlist", name="gym")
        self.assertEqual([i["title"] for i in pl["items"]], ["b", "a"])
        rc, out = self._run(action="playlist-clear", topic="gym")
        self.assertEqual(rc, 0)
        self.assertIn("cleared playlist gym", out)
        self.assertEqual(
            self.tools.fns["music_library"](action="playlist", name="gym")
            ["tracks"], 0)
        rc, _ = self._run(action="playlist-move")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="playlist-clear")
        self.assertEqual(rc, 2)

    def test_history_clear_cli(self) -> None:
        lib = self.tools.fns["music_library"]
        lib(action="history")
        self._clips()
        rc, out = self._run(action="history-clear")
        self.assertEqual(rc, 0)
        self.assertIn("history cleared", out)
        self.assertEqual(lib(action="history")["history"], [])

    def test_playlist_save_cli(self) -> None:
        rc, out = self._run(action="playlist-save", topic="jam")
        self.assertEqual(rc, 1)  # empty queue → tool error
        self._clips()
        self._run(action="add", topic="a.mp3", extra=["b.mp3"])
        rc, out = self._run(action="playlist-save", topic="jam")
        self.assertEqual(rc, 0)
        self.assertIn("saved 2 tracks to playlist jam", out)
        lib = self.tools.fns["music_library"]
        pl = lib(action="playlist", name="jam")
        self.assertEqual([i["title"] for i in pl["items"]], ["a", "b"])
        # save again: replaces, never duplicates
        rc, out = self._run(action="playlist-save", topic="jam")
        self.assertIn("saved 2 tracks to playlist jam", out)
        self.assertEqual(lib(action="playlist", name="jam")["tracks"], 2)
        rc, _ = self._run(action="playlist-save")
        self.assertEqual(rc, 2)

    def test_playlist_export_import_cli(self) -> None:
        self._clips()
        self._run(action="playlist-create", topic="gym")
        self._run(action="playlist-add", topic="gym",
                  extra=["a.mp3", "b.mp3"])
        rc, out = self._run(action="playlist-export", topic="gym",
                            extra=["gym.m3u"])
        self.assertEqual(rc, 0)
        self.assertIn("exported 2 tracks", out)
        body = (Path(self.ctx._root) / "gym.m3u").read_text(encoding="utf-8")
        self.assertTrue(body.startswith("#EXTM3U"))
        self.assertIn("a.mp3", body)
        # import into a fresh playlist — relative paths resolve against
        # the workspace
        rc, out = self._run(action="playlist-import", topic="fresh",
                            extra=["gym.m3u"])
        self.assertEqual(rc, 0)
        self.assertIn("imported 2 tracks into fresh", out)
        lib = self.tools.fns["music_library"]
        pl = lib(action="playlist", name="fresh")
        self.assertEqual([i["title"] for i in pl["items"]], ["a", "b"])
        rc, _ = self._run(action="playlist-export", topic="gym")
        self.assertEqual(rc, 2)
        rc, _ = self._run(action="playlist-import", topic="x")
        self.assertEqual(rc, 2)


# ── offline degradation: skip dead streaming items ────────────────────────


class OfflineDegradationTests(PlayerBase):
    def _sc_queue(self) -> PlaybackEngine:
        eng = PlaybackEngine(self.ctx)
        eng.backend = Backend("mpv", "/bin/fake-mpv")
        eng.add(self._clip("a.mp3"))
        eng._enqueue("https://soundcloud.com/artist/dead", "soundcloud",
                     "dead link")
        eng.add(self._clip("c.mp3"))
        return eng

    def _dead_stream(self, eng: PlaybackEngine) -> Any:
        """SoundCloud resolution always fails (offline/stale link)."""
        return mock.patch.object(
            eng, "_soundcloud_play_url",
            side_effect=ToolError("soundcloud resolve failed: offline"))

    def test_dead_item_is_skipped_with_report(self) -> None:
        eng = self._sc_queue()
        with self._dead_stream(eng), \
             mock.patch.object(eng, "_mpv_play_at",
                               return_value=(True, "")):
            res = eng.play(1)
        self.assertEqual(res["status"], "playing")
        self.assertEqual(res["current"], "c")
        self.assertEqual(len(res["skipped"]), 1)
        self.assertEqual(res["skipped"][0]["index"], 1)
        self.assertEqual(res["skipped"][0]["title"], "dead link")
        self.assertIn("offline", res["skipped"][0]["error"])
        # the playhead followed the skip
        self.assertEqual(eng._state["position"], 2)

    def test_all_dead_raises_with_detail(self) -> None:
        eng = self._sc_queue()
        with mock.patch.object(
                eng, "_start_one",
                side_effect=ToolError("soundcloud resolve failed: offline")):
            with self.assertRaises(ToolError) as cm:
                eng.play(0)
        self.assertIn("nothing in the queue is playable", str(cm.exception))
        self.assertIn("dead link", str(cm.exception))

    def test_next_skips_dead_item(self) -> None:
        eng = self._sc_queue()
        with self._dead_stream(eng), \
             mock.patch.object(eng, "_mpv_play_at",
                               return_value=(True, "")):
            eng.play(0)
            res = eng.next()
        self.assertEqual(res["current"], "c")
        self.assertEqual(len(res["skipped"]), 1)

    def test_mpv_playlist_drops_dead_later_items(self) -> None:
        eng = self._sc_queue()
        with mock.patch.object(eng, "_mpv_alive", return_value=True), \
             mock.patch.object(eng, "_mpv_ipc", return_value=True), \
             mock.patch.object(
                 eng, "_soundcloud_play_url",
                 side_effect=ToolError("soundcloud resolve failed: gone")):
            ok, _ = eng._mpv_play_at(0)
        self.assertTrue(ok)
        # the dead item stays in the durable queue but is absent from
        # mpv's playlist — the current track still starts
        self.assertEqual(eng._state["mpv_map"], [0, 2])
        self.assertEqual(len(eng.queue()), 3)

    def test_mpv_play_at_pos_dead_raises_for_skip_walker(self) -> None:
        eng = self._sc_queue()
        with mock.patch.object(eng, "_mpv_alive", return_value=True), \
             mock.patch.object(eng, "_mpv_ipc", return_value=True), \
             mock.patch.object(
                 eng, "_soundcloud_play_url",
                 side_effect=ToolError("soundcloud resolve failed: gone")):
            with self.assertRaises(ToolError):
                eng._mpv_play_at(1)

    def test_adapter_failures_become_tool_errors(self) -> None:
        fake_sc = mock.Mock()
        fake_sc.resolve.side_effect = RuntimeError("net down")
        fake_sc.search_tracks.side_effect = RuntimeError("offline")
        eng = PlaybackEngine(self.ctx, soundcloud=fake_sc)
        with self.assertRaises(ToolError) as cm:
            eng._add_soundcloud("https://soundcloud.com/a/t")
        self.assertIn("soundcloud resolve failed", str(cm.exception))
        with self.assertRaises(ToolError) as cm:
            eng.play_soundcloud("some query")
        self.assertIn("soundcloud search failed", str(cm.exception))

    def test_spotify_transport_failure_is_clear(self) -> None:
        fake_sp = mock.Mock()
        fake_sp.pause.side_effect = RuntimeError("offline")
        eng = PlaybackEngine(self.ctx, spotify=fake_sp)
        eng._enqueue("spotify:track:xyz", "spotify", "some track")
        with self.assertRaises(ToolError) as cm:
            eng.pause()
        self.assertIn("spotify pause failed", str(cm.exception))

    def test_spotify_start_failure_skips_to_local(self) -> None:
        fake_sp = mock.Mock()
        fake_sp.play.side_effect = RuntimeError("no network")
        eng = PlaybackEngine(self.ctx, spotify=fake_sp)
        eng.backend = Backend("mpv", "/bin/fake-mpv")
        eng._enqueue("spotify:track:xyz", "spotify", "some track")
        eng.add(self._clip("a.mp3"))
        with mock.patch.object(eng, "_mpv_play_at",
                               return_value=(True, "")):
            res = eng.play(0)
        self.assertEqual(res["status"], "playing")
        self.assertEqual(res["current"], "a")
        self.assertEqual(len(res["skipped"]), 1)
        self.assertIn("spotify play failed", res["skipped"][0]["error"])

    def test_fmt_play_reports_skips(self) -> None:
        from nomorals.cmdline.commands.music import _fmt_play
        out = _fmt_play({"status": "playing", "current": "c",
                         "skipped": [{"index": 1, "title": "dead link",
                                      "kind": "soundcloud",
                                      "error": "offline"}]})
        self.assertIn("playing: c", out)
        self.assertIn("skipped 1 unplayable: dead link", out)

    def test_fmt_play_surfaces_stream_url(self) -> None:
        from nomorals.cmdline.commands.music import _fmt_play
        out = _fmt_play({"status": "no-backend", "current": "SC Track",
                         "hint": "h",
                         "stream_url": "https://cf.example/stream.mp3"})
        self.assertIn("queued (no audio backend): SC Track", out)
        self.assertIn("stream: https://cf.example/stream.mp3", out)


# ── CLI streaming dispatch (nm music play ↔ adapters) ───────────────────────


class FakeCliSpotify:
    """Duck-typed Spotify adapter for CLI dispatch tests."""

    def __init__(self) -> None:
        self.played: list[list[str] | None] = []
        self.searched: list[str] = []

    @staticmethod
    def normalize_uri(target: str) -> str:
        text = (target or "").strip()
        if text.startswith("spotify:"):
            parts = text.split(":")
            if len(parts) == 3 and parts[1] in (
                    "track", "album", "playlist", "episode", "show",
                    "artist") and parts[2]:
                return text
        low = text.lower()
        if "open.spotify.com" in low or "play.spotify.com" in low:
            segs = [s for s in text.split("?")[0].rstrip("/").split("/")
                    if s]
            for i, seg in enumerate(segs):
                if seg in ("track", "album", "playlist", "episode",
                           "show", "artist") and i + 1 < len(segs):
                    return f"spotify:{seg}:{segs[i + 1]}"
        raise ValueError(f"{target!r} is not a Spotify URI/link")

    @staticmethod
    def get_track(uri: str) -> dict[str, Any]:
        return {"id": "abc", "uri": uri, "name": "CLI Song",
                "artists": ["CLI Artist"], "album": "CLI Album"}

    def search(self, query: str, *, types: Any = None,
               limit: int = 10) -> dict[str, Any]:
        self.searched.append(query)
        return {"tracks": {"items": [
            {"uri": "spotify:track:found9", "id": "found9",
             "name": "Found Nine",
             "artists": [{"name": "Search Artist"}]}]}}

    def play(self, *, device_id: str = "", context_uri: str = "",
             uris: list[str] | None = None) -> dict[str, Any]:
        self.played.append(uris)
        return {"playing": True, "device_id": "dev9"}

    def pause(self, *, device_id: str = "") -> dict[str, Any]:
        return {"playing": False}

    def now_playing(self) -> dict[str, Any]:
        return {"playing": True, "track": "CLI Song",
                "artists": ["CLI Artist"], "progress_ms": 5000}


class FakeCliSoundCloud:
    """Duck-typed SoundCloud adapter for CLI dispatch tests."""

    def __init__(self) -> None:
        self.resolved: list[str] = []
        self.searched: list[str] = []
        self.streamed: list[Any] = []

    def resolve(self, url: str) -> dict[str, Any]:
        self.resolved.append(url)
        return {"kind": "track",
                "track": {"id": 4242, "title": "CLI SC Track",
                          "artist": "SC Artist",
                          "permalink_url": url,
                          "duration_ms": 120000}}

    def search_tracks(self, query: str, *,
                      limit: int = 10) -> list[dict[str, Any]]:
        self.searched.append(query)
        return [{"id": 777, "title": f"SC hit for {query}",
                 "artist": "Search Artist",
                 "permalink_url": "https://soundcloud.com/x/hit",
                 "duration_ms": 90000}]

    def stream_url(self, track: Any) -> dict[str, Any]:
        self.streamed.append(track)
        return {"url": "https://cf.example.com/cli-stream.mp3",
                "protocol": "progressive",
                "mime_type": "audio/mpeg", "track": {}}

    def playlist_tracks(self, target: Any) -> list[dict[str, Any]]:
        return []

    def user_tracks(self, target: Any, *,
                    limit: int = 50) -> list[dict[str, Any]]:
        return []


class MusicCliStreamingTests(PlayerBase):
    """``nm music play`` routes Spotify URIs/links and SoundCloud URLs."""

    def setUp(self) -> None:
        super().setUp()
        reg = _registry(self.ctx)
        pb_mod.register(reg)
        lib_mod.register(reg)
        self.tools = FakeTools(reg.fns)
        self.ctx.tools = self.tools
        from nomorals.cmdline.commands.music import _cmd_music
        self.cmd = _cmd_music
        self.spotify = FakeCliSpotify()
        self.sc = FakeCliSoundCloud()
        self.ctx.spotify_adapter = self.spotify
        self.ctx.soundcloud_adapter = self.sc

    def _run(self, **kw: Any) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.cmd(_ns(**kw), self.ctx)
        return rc, buf.getvalue()

    def test_play_spotify_uri(self) -> None:
        rc, out = self._run(action="play", topic="spotify:track:abc123")
        self.assertEqual(rc, 0)
        self.assertIn("playing: CLI Artist – CLI Song", out)
        self.assertEqual(self.spotify.played, [["spotify:track:abc123"]])

    def test_play_spotify_open_link(self) -> None:
        rc, out = self._run(
            action="play",
            topic="https://open.spotify.com/track/abc123?si=zzz")
        self.assertEqual(rc, 0)
        self.assertEqual(self.spotify.played, [["spotify:track:abc123"]])

    def test_play_spotify_search_with_flag(self) -> None:
        rc, out = self._run(action="play", topic="bohemian rhapsody",
                            spotify=True)
        self.assertEqual(rc, 0)
        self.assertEqual(self.spotify.searched, ["bohemian rhapsody"])
        self.assertEqual(self.spotify.played, [["spotify:track:found9"]])
        self.assertIn("Found Nine", out)

    def test_play_soundcloud_url(self) -> None:
        rc, out = self._run(action="play",
                            topic="https://soundcloud.com/artist/cli-track")
        self.assertEqual(rc, 0)
        self.assertEqual(self.sc.resolved,
                         ["https://soundcloud.com/artist/cli-track"])
        self.assertIn("queued (no audio backend): CLI SC Track", out)
        # the resolved stream URL is surfaced on backend-less machines
        self.assertIn("stream: https://cf.example.com/cli-stream.mp3", out)

    def test_play_bare_text_uses_soundcloud_search(self) -> None:
        rc, out = self._run(action="play", topic="synthwave mix")
        self.assertEqual(rc, 0)
        self.assertEqual(self.sc.searched, ["synthwave mix"])
        self.assertIn("SC hit for synthwave mix", out)

    def test_play_add_accepts_mixed_targets(self) -> None:
        rc, out = self._run(
            action="add", topic="spotify:track:abc123",
            extra=["https://soundcloud.com/artist/cli-track"])
        self.assertEqual(rc, 0)
        self.assertIn("added 2", out)

    def test_wire_streaming_attaches_keyless_soundcloud(self) -> None:
        from nomorals.cmdline.commands.music import _wire_streaming
        from nomorals.connectors.soundcloud import SoundCloudConnector
        ctx = _context()
        _wire_streaming(ctx)
        self.assertIsInstance(ctx.soundcloud_adapter, SoundCloudConnector)
        # Spotify needs the vault — without one it stays unwired, loudly
        self.assertIsNone(getattr(ctx, "spotify_adapter", None))

    def test_play_spotify_without_adapter_fails_clear(self) -> None:
        ctx = _context()
        reg = _registry(ctx)
        pb_mod.register(reg)
        lib_mod.register(reg)
        ctx.tools = FakeTools(reg.fns)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.cmd(_ns(action="play",
                              topic="spotify:track:abc123"), ctx)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
