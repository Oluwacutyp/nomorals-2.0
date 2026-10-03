"""Streaming integration tests: Spotify + SoundCloud wired into
PlaybackEngine via injected adapters (no connector imports in playback —
the fakes below are duck-typed, exactly like the real wiring).

Covers:
* PlaybackEngine.detect_source classification
* add() handling of Spotify URIs/links and SoundCloud links
  (track / playlist / artist expansion)
* play_spotify() with a URI, a link, and a search query — including the
  fail-fast paths (not wired, no results, no active device)
* play_soundcloud() with a link and a search query
* pause()/resume()/next()/prev()/stop()/status() routing for Spotify
  items, and SoundCloud stream resolution at play time
* the ``player`` tool picking adapters up from context attributes
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import nomorals.media.playback as pb
from nomorals.connectors.spotify import SpotifyError
from nomorals.core.errors import ToolError
from nomorals.media.playback import Backend, PlaybackEngine
from nomorals.storage.db import Database

# ── fixtures ──────────────────────────────────────────────────────────────


def _context(**over: Any) -> SimpleNamespace:
    root = tempfile.mkdtemp(prefix="stream_playback_")
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


class FakeSpotify:
    """Duck-typed stand-in for SpotifyConnector."""

    def __init__(self) -> None:
        self.play_calls: list[list[str] | None] = []
        self.pause_calls = 0
        self.play_fail: Exception | None = None
        self.search_results: list[dict[str, Any]] | None = None

    @staticmethod
    def normalize_uri(target: str) -> str:
        text = target.strip()
        if text.startswith("spotify:"):
            parts = text.split(":")
            if len(parts) == 3 and parts[1] in (
                    "track", "album", "playlist") and parts[2]:
                return text
            raise SpotifyError(f"{target!r} is not a Spotify URI")
        if "open.spotify.com" in text:
            segs = [s for s in text.split("?")[0].rstrip("/").split("/")
                    if s]
            for i, seg in enumerate(segs):
                if seg in ("track", "album", "playlist") and i + 1 < len(segs):
                    return f"spotify:{seg}:{segs[i + 1]}"
        raise SpotifyError(f"{target!r} is not a Spotify URI")

    @staticmethod
    def get_track(uri: str) -> dict[str, Any]:
        return {"id": "t1", "uri": uri, "name": "Test Song",
                "artists": ["Test Artist"], "album": "Test Album"}

    def search(self, query: str, *, types: Any = None,
               limit: int = 10) -> dict[str, Any]:
        items = self.search_results if self.search_results is not None \
            else [{"uri": "spotify:track:found1", "id": "found1",
                   "name": "Found Song", "artists": [{"name": "Found Art"}]}]
        return {"tracks": {"items": items}}

    def play(self, *, device_id: str = "", context_uri: str = "",
             uris: list[str] | None = None) -> dict[str, Any]:
        if self.play_fail is not None:
            raise self.play_fail
        self.play_calls.append(uris)
        return {"playing": True, "device_id": "dev1"}

    def pause(self, *, device_id: str = "") -> dict[str, Any]:
        self.pause_calls += 1
        return {"playing": False}

    def now_playing(self) -> dict[str, Any]:
        return {"playing": True, "track": "Test Song",
                "artists": ["Test Artist"], "progress_ms": 30000,
                "device": "Phone", "uri": "spotify:track:t1"}


def _sc_track(i: int = 111, title: str = "Midnight Drive") -> dict[str, Any]:
    return {
        "id": i, "title": title, "artist": "artist",
        "duration_ms": 180000, "genre": "Synthwave",
        "artwork_url": "", "likes_count": 0, "playback_count": 0,
        "policy": "ALLOW", "streamable": True,
        "permalink_url": f"https://soundcloud.com/artist/track-{i}",
        "kind": "track", "_full": None,
    }


class FakeSoundCloud:
    """Duck-typed stand-in for SoundCloudConnector."""

    def __init__(self) -> None:
        self.stream_calls: list[str] = []
        self.resolve_calls: list[str] = []
        self.stream_fail: Exception | None = None
        self.kind = "track"

    def resolve(self, url: str) -> dict[str, Any]:
        self.resolve_calls.append(url)
        if self.kind == "track":
            return {"kind": "track", "track": _sc_track()}
        if self.kind == "playlist":
            return {"kind": "playlist",
                    "playlist": {"id": 777, "title": "Mix",
                                 "permalink_url": url}}
        return {"kind": "user",
                "user": {"id": 9, "username": "artist",
                         "permalink_url": url}}

    def stream_url(self, track: Any) -> dict[str, Any]:
        if self.stream_fail is not None:
            raise self.stream_fail
        target = track if isinstance(track, str) else track.get(
            "permalink_url", "")
        self.stream_calls.append(target)
        return {"url": "https://cf-media.sndcdn.com/stream.mp3",
                "protocol": "progressive", "mime_type": "audio/mpeg",
                "track": _sc_track()}

    def search_tracks(self, query: str, limit: int = 10
                      ) -> list[dict[str, Any]]:
        return [_sc_track(title=f"Result for {query}")]

    def playlist_tracks(self, target: Any) -> list[dict[str, Any]]:
        return [_sc_track(111, "Mix One"), _sc_track(222, "Mix Two")]

    def user_tracks(self, target: Any, limit: int = 25
                    ) -> list[dict[str, Any]]:
        return [_sc_track(333, "Latest One")]


class StreamingBase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _context()
        self.addCleanup(shutil.rmtree, self.ctx._root, True)
        # pin the console backend: deterministic everywhere
        be = mock.patch.object(
            pb, "detect_backend", lambda: Backend("console"))
        be.start()
        self.addCleanup(be.stop)
        self.spotify = FakeSpotify()
        self.sc = FakeSoundCloud()
        self.eng = PlaybackEngine(
            self.ctx, spotify=self.spotify, soundcloud=self.sc)


# ── source detection ──────────────────────────────────────────────────────


class DetectSourceTests(StreamingBase):
    def test_spotify(self) -> None:
        self.assertEqual(
            PlaybackEngine.detect_source("spotify:track:abc"), "spotify")
        self.assertEqual(
            PlaybackEngine.detect_source(
                "https://open.spotify.com/track/abc?si=x"), "spotify")
        self.assertEqual(
            PlaybackEngine.detect_source(
                "https://play.spotify.com/album/abc"), "spotify")

    def test_soundcloud(self) -> None:
        self.assertEqual(
            PlaybackEngine.detect_source(
                "https://soundcloud.com/artist/track"), "soundcloud")
        self.assertEqual(
            PlaybackEngine.detect_source(
                "https://on.soundcloud.com/xyz"), "soundcloud")

    def test_url_and_file(self) -> None:
        self.assertEqual(
            PlaybackEngine.detect_source("https://example.com/a.mp3"), "url")
        self.assertEqual(PlaybackEngine.detect_source("tunes/a.mp3"), "file")
        self.assertEqual(PlaybackEngine.detect_source(""), "file")


# ── add() ─────────────────────────────────────────────────────────────────


class AddStreamingTests(StreamingBase):
    def test_add_spotify_uri(self) -> None:
        res = self.eng.add("spotify:track:t1")
        self.assertEqual(len(res["added"]), 1)
        item = res["added"][0]
        self.assertEqual(item["kind"], "spotify")
        self.assertEqual(item["path"], "spotify:track:t1")
        # title resolved through the adapter's get_track
        self.assertEqual(item["title"], "Test Artist – Test Song")
        self.assertEqual(item["artist"], "Test Artist")
        q = self.eng.queue()
        self.assertEqual(q[0]["kind"], "spotify")

    def test_add_spotify_link_normalized(self) -> None:
        res = self.eng.add("https://open.spotify.com/track/t1?si=x")
        self.assertEqual(res["added"][0]["path"], "spotify:track:t1")

    def test_add_spotify_explicit_title_kept(self) -> None:
        res = self.eng.add("spotify:track:t1", title="My Label")
        self.assertEqual(res["added"][0]["title"], "My Label")

    def test_add_spotify_without_adapter(self) -> None:
        eng = PlaybackEngine(self.ctx)  # no adapters
        res = eng.add("spotify:track:t1")
        self.assertEqual(res["added"][0]["kind"], "spotify")
        self.assertEqual(res["added"][0]["title"], "spotify:track:t1")
        # …but playing fails fast with a clear message
        with self.assertRaises(ToolError) as ctx:
            eng.play(0)
        self.assertIn("not wired", str(ctx.exception))

    def test_add_spotify_garbage_fails_fast(self) -> None:
        with self.assertRaises(SpotifyError):
            self.eng.add("spotify:track:")

    def test_add_soundcloud_track(self) -> None:
        res = self.eng.add("https://soundcloud.com/artist/track-111")
        self.assertEqual(len(res["added"]), 1)
        item = res["added"][0]
        self.assertEqual(item["kind"], "soundcloud")
        self.assertEqual(item["title"], "Midnight Drive")
        self.assertEqual(item["artist"], "artist")
        self.assertEqual(
            item["path"], "https://soundcloud.com/artist/track-111")

    def test_add_soundcloud_playlist_expands(self) -> None:
        self.sc.kind = "playlist"
        res = self.eng.add("https://soundcloud.com/artist/sets/mix")
        self.assertEqual(len(res["added"]), 2)
        self.assertEqual(res["added"][0]["title"], "Mix One")
        self.assertEqual(res["added"][1]["title"], "Mix Two")
        self.assertTrue(all(a["kind"] == "soundcloud"
                            for a in res["added"]))

    def test_add_soundcloud_artist_expands(self) -> None:
        self.sc.kind = "user"
        res = self.eng.add("https://soundcloud.com/artist")
        self.assertEqual(len(res["added"]), 1)
        self.assertEqual(res["added"][0]["title"], "Latest One")

    def test_add_soundcloud_without_adapter_fails_fast(self) -> None:
        eng = PlaybackEngine(self.ctx)
        with self.assertRaises(ToolError) as ctx:
            eng.add("https://soundcloud.com/artist/track-111")
        self.assertIn("not wired", str(ctx.exception))

    def test_local_files_still_work(self) -> None:
        import pathlib
        pathlib.Path(self.ctx._root, "clip.mp3").write_bytes(b"fake-audio")
        res = self.eng.add("clip.mp3")
        self.assertEqual(res["added"][0]["kind"], "file")


# ── play_spotify ──────────────────────────────────────────────────────────


class PlaySpotifyTests(StreamingBase):
    def test_play_spotify_uri(self) -> None:
        out = self.eng.play_spotify("spotify:track:t1")
        self.assertEqual(out["status"], "playing")
        self.assertEqual(out["via"], "spotify")
        self.assertEqual(self.spotify.play_calls, [["spotify:track:t1"]])
        q = self.eng.queue()
        self.assertEqual(q[0]["kind"], "spotify")

    def test_play_spotify_link(self) -> None:
        out = self.eng.play_spotify(
            "https://open.spotify.com/track/t1?si=x")
        self.assertEqual(out["uri"], "spotify:track:t1")
        self.assertEqual(self.spotify.play_calls, [["spotify:track:t1"]])

    def test_play_spotify_search(self) -> None:
        out = self.eng.play_spotify("never gonna give you up")
        self.assertEqual(out["status"], "playing")
        self.assertEqual(out["uri"], "spotify:track:found1")
        self.assertIn("Found Song", out["title"])
        self.assertEqual(self.spotify.play_calls, [["spotify:track:found1"]])

    def test_play_spotify_no_results_fails_fast(self) -> None:
        self.spotify.search_results = []
        with self.assertRaises(ToolError) as ctx:
            self.eng.play_spotify("zzz no such song qqq")
        self.assertIn("no Spotify results", str(ctx.exception))

    def test_play_spotify_no_active_device_fails_fast(self) -> None:
        self.spotify.play_fail = SpotifyError(
            "no active Spotify device — open Spotify on a phone, "
            "computer, or the web player and start anything once, "
            "then retry (devices seen: none found)")
        # the skip-walker finds nothing else playable, so it fails fast
        # with the adapter's own clear message inside the report
        with self.assertRaises(ToolError) as ctx:
            self.eng.play_spotify("spotify:track:t1")
        self.assertIn("no active Spotify device", str(ctx.exception))
        # nothing was queued as "playing"
        self.assertFalse(self.eng._state.get("playing", False))

    def test_play_spotify_not_wired_fails_fast(self) -> None:
        eng = PlaybackEngine(self.ctx, soundcloud=self.sc)
        with self.assertRaises(ToolError) as ctx:
            eng.play_spotify("spotify:track:t1")
        self.assertIn("not wired", str(ctx.exception))

    def test_play_spotify_empty(self) -> None:
        with self.assertRaises(ToolError):
            self.eng.play_spotify("  ")

    def test_pause_resume_route_to_spotify(self) -> None:
        self.eng.play_spotify("spotify:track:t1")
        paused = self.eng.pause()
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["via"], "spotify")
        self.assertEqual(self.spotify.pause_calls, 1)
        resumed = self.eng.resume()
        self.assertEqual(resumed["status"], "resumed")
        self.assertEqual(resumed["via"], "spotify")
        # resume() calls play() with no uris → second play call, uris None
        self.assertEqual(self.spotify.play_calls[-1], None)

    def test_stop_pauses_spotify(self) -> None:
        self.eng.play_spotify("spotify:track:t1")
        out = self.eng.stop()
        self.assertEqual(out["status"], "stopped")
        self.assertEqual(self.spotify.pause_calls, 1)
        self.assertFalse(self.eng._state.get("playing", False))

    def test_status_reports_spotify_live(self) -> None:
        self.eng.play_spotify("spotify:track:t1")
        st = self.eng.status()
        self.assertEqual(st["via"], "spotify")
        self.assertTrue(st["playing"])
        self.assertEqual(st["live"]["spotify"]["track"], "Test Song")

    def test_next_from_spotify_to_local(self) -> None:
        import pathlib
        pathlib.Path(self.ctx._root, "a.mp3").write_bytes(b"fake-audio")
        self.eng.add("spotify:track:t1")
        self.eng.add("a.mp3")
        self.eng.play(0)
        self.assertEqual(self.spotify.play_calls, [["spotify:track:t1"]])
        nxt = self.eng.next()  # local item on the console backend
        self.assertEqual(nxt["status"], "no-backend")
        self.assertEqual(self.eng.status()["position"], 1)


# ── play_soundcloud ───────────────────────────────────────────────────────


class PlaySoundCloudTests(StreamingBase):
    def test_play_soundcloud_link(self) -> None:
        out = self.eng.play_soundcloud(
            "https://soundcloud.com/artist/track-111")
        self.assertEqual(out["tracks"], 1)
        self.assertEqual(out["title"], "Midnight Drive")
        # console backend: honest no-backend, but the stream URL resolves
        self.assertEqual(out["status"], "no-backend")
        self.assertEqual(out["stream_url"],
                         "https://cf-media.sndcdn.com/stream.mp3")
        self.assertEqual(self.sc.stream_calls,
                         ["https://soundcloud.com/artist/track-111"])

    def test_play_soundcloud_search(self) -> None:
        out = self.eng.play_soundcloud("synthwave mix")
        self.assertEqual(out["status"], "no-backend")
        self.assertIn("synthwave mix", out["title"])
        q = self.eng.queue()
        self.assertEqual(q[0]["kind"], "soundcloud")

    def test_play_soundcloud_playlist_enqueues_all(self) -> None:
        self.sc.kind = "playlist"
        out = self.eng.play_soundcloud(
            "https://soundcloud.com/artist/sets/mix")
        self.assertEqual(out["tracks"], 2)
        self.assertEqual(len(self.eng.queue()), 2)
        self.assertEqual(out["title"], "Mix One")

    def test_play_soundcloud_stream_failure_fails_fast(self) -> None:
        from nomorals.connectors.soundcloud import SoundCloudError
        self.sc.stream_fail = SoundCloudError(
            '"Midnight Drive" has no playable stream (policy=BLOCK)')
        with self.assertRaises(ToolError) as ctx:
            self.eng.play_soundcloud(
                "https://soundcloud.com/artist/track-111")
        self.assertIn("can't stream", str(ctx.exception))

    def test_play_soundcloud_not_wired_fails_fast(self) -> None:
        eng = PlaybackEngine(self.ctx, spotify=self.spotify)
        with self.assertRaises(ToolError) as ctx:
            eng.play_soundcloud("https://soundcloud.com/artist/x")
        self.assertIn("not wired", str(ctx.exception))

    def test_play_soundcloud_empty(self) -> None:
        with self.assertRaises(ToolError):
            self.eng.play_soundcloud("  ")


# ── tool wiring ───────────────────────────────────────────────────────────


class ToolWiringTests(StreamingBase):
    def test_player_tool_uses_context_adapters(self) -> None:
        reg = _registry(self.ctx)
        pb.register(reg)
        self.ctx.spotify_adapter = self.spotify
        self.ctx.soundcloud_adapter = self.sc
        player = reg.fns["player"]
        out = player(action="play_spotify", target="spotify:track:t1")
        self.assertEqual(out["via"], "spotify")
        self.assertEqual(self.spotify.play_calls, [["spotify:track:t1"]])
        out = player(action="play_soundcloud",
                     target="https://soundcloud.com/artist/track-111")
        self.assertEqual(out["stream_url"],
                         "https://cf-media.sndcdn.com/stream.mp3")

    def test_player_tool_missing_target_fails_fast(self) -> None:
        reg = _registry(self.ctx)
        pb.register(reg)
        player = reg.fns["player"]
        with self.assertRaises(ToolError):
            player(action="play_spotify", target="  ")
        with self.assertRaises(ToolError):
            player(action="play_soundcloud", target="")


if __name__ == "__main__":
    unittest.main()
