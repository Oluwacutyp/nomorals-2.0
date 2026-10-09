"""PlaybackEngine — real music playback with full transport control.

Design: the player is a **detached mpv process** (own session) whose IPC
socket lives at a stable workspace path, so control commands work across
CLI/chat/tool calls without a resident daemon:

* **mpv** — the queue IS mpv's playlist. ``play`` syncs the durable queue
  (``media_queue`` table, migration 24) into the playlist, then jumps to
  the chosen index.  mpv auto-advances through the playlist when a track
  ends — no supervision process needed.  Transport (pause/resume/seek/
  volume/next/prev) and live status (playback-status, time-pos) go over
  the JSON IPC socket.
* **ffplay / aplay / mpg123 / sox / afplay** — basic play/stop per track
  (pid persisted; no fine transport, no auto-advance — honestly reported).
* **console** — no audio backend: the queue still works and ``play`` says
  exactly what to install (``pkg install mpv`` on Termux).

Streaming sources ride on top of the same queue. The engine never
imports the connector layer (media is L4, connectors are L5) — the
Spotify/SoundCloud adapters are *injected* (constructor kwargs, or the
``spotify_adapter`` / ``soundcloud_adapter`` context attributes the
``nm music`` CLI wires up):

* **Spotify** — queue items of kind ``spotify`` are played through the
  owner's Spotify account: ``play_spotify`` hands the track URI to the
  injected Spotify adapter's ``play(uris=[...])``, which starts it on
  the user's active Spotify device. Spotify serves no direct audio
  stream, so there is no mpv involved — pause/resume route through the
  adapter too, and the no-active-device case fails fast with the
  connector's own clear message.
* **SoundCloud** — queue items of kind ``soundcloud`` hold the track's
  permalink; the stream URL is resolved fresh at play time through the
  injected SoundCloud adapter (progressive MP3 preferred, HLS fallback)
  and then played by whatever local backend is available, mpv included.
* **YouTube** — queue items of kind ``youtube`` hold the watch URL.  The
  audio is extracted once at play time (yt-dlp, optional dependency),
  cached under the media dir by video id, and then played as a local
  file.  No OAuth needed — but without yt-dlp this source fails
  honestly instead of pretending.

    from nomorals.media.playback import PlaybackEngine
    p = PlaybackEngine(context, spotify=spotify_adapter,
                       soundcloud=sc_adapter)
    p.play_spotify("spotify:track:4uLU6hMCjMI75M1A2tKUQ")
    p.play_spotify("never gonna give you up")   # Spotify search
    p.play_soundcloud("https://soundcloud.com/artist/track")
    p.play_soundcloud("synthwave mix")           # SoundCloud search
    p.play_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    p.play_youtube("lofi hip hop radio")         # YouTube search (yt-dlp)

Volume, position, and the mpv socket path persist in ``kv_store``.

    from nomorals.media.playback import PlaybackEngine
    p = PlaybackEngine(context)
    p.add("workspace/tunes/song.mp3", title="Song")
    p.play()
    p.volume(60); p.seek(30); p.pause(); p.resume(); p.next()
    p.status()          # live: playback-status, time-pos, playlist-index

Registered as the ``player`` tool.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import NoMoralsError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .library import MusicLibrary, read_metadata

_log = get_logger(__name__)

#: Error fragments that mean "this source won't serve the audio"
#: (DRM, geo/private locks) — worth falling back to another source
#: instead of failing outright.
_PROTECTION_FRAGMENTS = ("drm", "protected", "not downloadable",
                         "private", "login required", "age-gated")


def is_protection_error(message: Any) -> bool:
    """True when an error message indicates the source blocked the
    download (DRM / private / login-walled) rather than a transient
    network or dependency failure."""
    low = str(message or "").lower()
    return any(frag in low for frag in _PROTECTION_FRAGMENTS)

__all__ = ["PlaybackEngine", "Backend", "register"]

_STATE_KEY = "media.player_state"

#: YouTube video ids are exactly 11 base64url chars.
_YT_ID_RE = re.compile(r"[A-Za-z0-9_\-]{11}\Z")


def _which(names: tuple[str, ...]) -> str:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return ""


@dataclass
class Backend:
    name: str          # mpv | ffplay | aplay | mpg123 | sox | afplay | console
    binary: str = ""
    ipc: bool = False
    urls: bool = False
    controls: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "binary": self.binary, "ipc": self.ipc,
                "urls": self.urls, "controls": list(self.controls)}


def detect_backend() -> Backend:
    mpv = _which(("mpv",))
    if mpv:
        return Backend("mpv", mpv, ipc=True, urls=True,
                       controls=("play", "pause", "resume", "stop", "seek",
                                 "volume", "next", "prev", "status"))
    ffplay = _which(("ffplay",))
    if ffplay:
        return Backend("ffplay", ffplay, urls=True,
                       controls=("play", "stop", "next", "prev"))
    aplay = _which(("aplay",))
    if aplay:
        return Backend("aplay", aplay, controls=("play", "stop", "next",
                                                 "prev"))
    mpg123 = _which(("mpg123",))
    if mpg123:
        return Backend("mpg123", mpg123, controls=("play", "stop", "next",
                                                   "prev"))
    sox = _which(("play",))
    if sox:
        return Backend("sox", sox, controls=("play", "stop", "next", "prev"))
    afplay = _which(("afplay",))
    if afplay:
        return Backend("afplay", afplay, controls=("play", "stop", "next",
                                                   "prev"))
    return Backend("console")


class PlaybackEngine:
    """Durable queue + detached-player transport."""

    role = "player"

    def __init__(self, context: Any, *,
                 spotify: Any = None, soundcloud: Any = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is None:
            raise RuntimeError("PlaybackEngine needs a context with a database")
        self.backend = detect_backend()
        self._state = self._load_state()
        self._state.setdefault("volume", 80)
        self._state.setdefault("position", 0)
        self._state.setdefault("shuffle", False)
        self._state.setdefault("repeat", "off")
        self._sock_dir = ""
        # Streaming adapters are injected, never imported: playback.py is
        # L4 and the connector layer is L5. The ``nm music`` CLI wires
        # these up; the ``player`` tool picks them up from the context
        # attributes when present.
        self._spotify = spotify if spotify is not None else getattr(
            context, "spotify_adapter", None)
        self._soundcloud = soundcloud if soundcloud is not None else getattr(
            context, "soundcloud_adapter", None)

    # ── streaming sources ─────────────────────────────────────────────
    #: Queue kinds the local backends (mpv/ffplay/...) play directly.
    LOCAL_KINDS = ("file", "url")

    @staticmethod
    def detect_source(target: str) -> str:
        """Classify a target: spotify | soundcloud | youtube | url | file."""
        low = (target or "").strip().lower()
        if low.startswith("spotify:") or "open.spotify.com" in low \
                or "play.spotify.com" in low:
            return "spotify"
        if "soundcloud.com" in low:
            return "soundcloud"
        if "youtube.com" in low or "youtu.be" in low \
                or low.startswith("youtube:"):
            return "youtube"
        if low.startswith(("http://", "https://")):
            return "url"
        return "file"

    def _require_spotify(self) -> Any:
        if self._spotify is None:
            raise ToolError(
                "spotify is not wired into the player — `nm music` wires "
                "it automatically; elsewhere pass spotify=<SpotifyConnector> "
                "to PlaybackEngine or set context.spotify_adapter"
            )
        return self._spotify

    def _require_soundcloud(self) -> Any:
        if self._soundcloud is None:
            raise ToolError(
                "soundcloud is not wired into the player — `nm music` wires "
                "it automatically; elsewhere pass "
                "soundcloud=<SoundCloudConnector> to PlaybackEngine or set "
                "context.soundcloud_adapter"
            )
        return self._soundcloud

    @staticmethod
    def _adapter_call(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a streaming-adapter call with a clear player-level error.

        Connector errors (offline, expired links, no device) surface as
        ToolError naming the failed operation, so `nm music` reports
        *what* failed instead of a raw traceback.
        """
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapters/network are opaque
            raise ToolError(f"{label} failed: {exc}") from exc

    def _spotify_uri(self, target: str) -> str:
        """Normalize a Spotify URI/link.

        Uses the adapter's own normalizer when one is wired; otherwise a
        local parse, so ``add()`` can validate syntax without an adapter.
        """
        adapter = self._spotify
        normalizer = getattr(adapter, "normalize_uri", None) \
            if adapter is not None else None
        if callable(normalizer):
            return str(normalizer(target))
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
        raise ToolError(
            f"{target!r} is not a Spotify URI or open.spotify.com link"
        )

    def _spotify_meta(self, uri: str, title: str) -> tuple[str, str]:
        """(title, artist) for a Spotify URI — adapter lookup, best-effort.

        Falls back to the URI itself when no adapter is wired or the
        lookup fails; metadata must never block queueing.
        """
        if title:
            return title, ""
        if self._spotify is not None:
            try:
                info = self._spotify.get_track(uri)
                artists = ", ".join(info.get("artists", []) or [])
                name = str(info.get("name", "") or "")
                label = f"{artists} – {name}" if artists and name else (
                    name or uri)
                return label, artists
            except Exception:  # noqa: BLE001 - metadata is best-effort
                pass
        return uri, ""

    def _soundcloud_play_url(self, item: dict[str, Any]) -> str:
        """Fresh stream URL for a soundcloud queue item (fail fast)."""
        adapter = self._require_soundcloud()
        try:
            stream = adapter.stream_url(item["path"])
        except Exception as exc:  # noqa: BLE001 - add the track context
            raise ToolError(
                f"can't stream {item.get('title', item['path'])!r} from "
                f"SoundCloud: {exc}"
            ) from exc
        url = str((stream or {}).get("url", ""))
        if not url:
            raise ToolError(
                f"SoundCloud gave no stream URL for "
                f"{item.get('title', item['path'])!r}"
            )
        return url

    def _is_spotify_current(self) -> bool:
        q = self.queue()
        if not q:
            return False
        return q[self._resolve_position()]["kind"] == "spotify"

    def _stop_local_audio(self) -> None:
        """Silence mpv/simple backends without touching Spotify state."""
        if self.backend.name == "mpv":
            sock_path = str(self._state.get("mpv_sock", ""))
            if sock_path and os.path.exists(sock_path):
                self._mpv_ipc(sock_path, ["quit"])
                for _ in range(20):
                    if not os.path.exists(sock_path):
                        break
                    time.sleep(0.1)
            self._state.pop("mpv_sock", None)
            self._state.pop("mpv_map", None)
        self._kill_simple()

    # ── persistent state ──────────────────────────────────────────────────
    def _load_state(self) -> dict[str, Any]:
        try:
            row = self.db.query_one("SELECT value FROM kv_store WHERE key=?",
                                    (_STATE_KEY,))
            if row:
                return json.loads(row["value"])
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _save_state(self) -> None:
        try:
            self.db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) "
                "VALUES (?,?, 'json', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (_STATE_KEY, json.dumps(self._state), time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("player state save failed: %s", exc)

    def _resolve_position(self) -> int:
        q = self.queue()
        if not q:
            return -1
        pos = int(self._state.get("position", 0))
        return max(0, min(pos, len(q) - 1))

    # ── queue (durable) ───────────────────────────────────────────────────
    def _enqueue(self, path: str, kind: str, title: str, *,
                 artist: str = "", album: str = "",
                 duration: float = 0.0) -> dict[str, Any]:
        """Insert one queue row; → the added dict."""
        # Monotonic positions within a batch: ORDER BY position, added_at
        # must reproduce insertion order even for same-second inserts.
        seq = getattr(self, "_enqueue_seq", 0)
        self._enqueue_seq = seq + 1
        self.db.execute(
            "INSERT INTO media_queue (id, position, path, title, kind, "
            "artist, album, duration, added_at) VALUES "
            "(?,?,?,?,?,?,?,?,?)",
            (f"mq-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
             time.time() + seq * 0.001, path, title, kind, artist, album,
             float(duration or 0), time.time()))
        return {"path": path, "kind": kind, "title": title,
                "artist": artist, "album": album,
                "duration": float(duration or 0)}

    def add(self, *targets: str, title: str = "") -> dict[str, Any]:
        """Add file(s)/URL(s) to the queue.  Files may be workspace-relative.

        Spotify URIs/links (``spotify:track:...``, open.spotify.com) are
        queued as ``spotify`` items — played later through the owner's
        Spotify account on their active device. SoundCloud links are
        queued as ``soundcloud`` items: tracks resolve to titles now,
        playlists enqueue every track, artist pages enqueue their latest
        tracks. The playable stream URL is resolved fresh at play time.

        Audio metadata (artist/album/duration) is read best-effort at add
        time so the queue and now-playing always have something to show.
        """
        added = []
        for t in targets:
            t = (t or "").strip()
            if not t:
                continue
            source = self.detect_source(t)
            if source == "spotify":
                uri = self._spotify_uri(t)
                final_title, artist = self._spotify_meta(uri, title)
                added.append(self._enqueue(
                    uri, "spotify", final_title, artist=artist))
                continue
            if source == "soundcloud":
                added.extend(self._add_soundcloud(t, title=title))
                continue
            if source == "youtube":
                added.extend(self._add_youtube(t, title=title))
                continue
            kind = "url" if source == "url" else "file"
            if kind == "file":
                from ..tools.filesystem import safe_path

                try:
                    path = str(safe_path(self.context, t, must_exist=True))
                except NoMoralsError as exc:
                    # NotFound / ValidationError → the player's own error
                    # contract, with the track named
                    raise ToolError(f"can't add {t!r}: {exc}") from exc
            else:
                path = t
            meta = read_metadata(path)
            final_title = title or meta["title"] or os.path.basename(t)
            added.append(self._enqueue(
                path, kind, final_title, artist=meta["artist"],
                album=meta["album"], duration=meta["duration"]))
        return {"added": added, "queue": len(self.queue())}

    def _add_soundcloud(self, target: str, title: str = ""
                        ) -> list[dict[str, Any]]:
        """Enqueue SoundCloud target(s): track, playlist, or artist page."""
        adapter = self._require_soundcloud()
        resolved = self._adapter_call("soundcloud resolve", adapter.resolve,
                                      target)  # fail fast on bad URLs
        kind = resolved["kind"]
        if kind == "track":
            tracks = [resolved["track"]]
        elif kind == "playlist":
            tracks = self._adapter_call("soundcloud playlist tracks",
                                        adapter.playlist_tracks, target)
            if not tracks:
                raise ToolError(
                    f"SoundCloud playlist {target!r} has no tracks")
        elif kind == "user":
            tracks = self._adapter_call("soundcloud user tracks",
                                        adapter.user_tracks, target, limit=25)
            if not tracks:
                raise ToolError(
                    f"SoundCloud user {target!r} has no public tracks")
        else:  # pragma: no cover - resolve() only returns the three kinds
            raise ToolError(f"unsupported SoundCloud kind {kind!r}")
        out = []
        for i, tr in enumerate(tracks):
            out.append(self._enqueue(
                tr["permalink_url"], "soundcloud",
                title if title and i == 0 else tr["title"],
                artist=tr.get("artist", ""),
                duration=(tr.get("duration_ms") or 0) / 1000.0))
        return out

    def play_spotify(self, target: str) -> dict[str, Any]:
        """Play a Spotify URI/link immediately, or a search query.

        A bare search query plays the top track result. Playback starts on
        the owner's active Spotify device via the injected adapter —
        Spotify serves no direct audio stream. Fails fast when Spotify
        isn't wired/connected or no device is active.
        """
        target = (target or "").strip()
        if not target:
            raise ToolError(
                "play_spotify needs a Spotify URI, an open.spotify.com "
                "link, or a search query")
        adapter = self._require_spotify()
        if self.detect_source(target) == "spotify":
            uri = self._spotify_uri(target)
            final_title, artist = self._spotify_meta(uri, "")
        else:
            try:
                results = adapter.search(target, types=["track"], limit=1)
            except Exception as exc:  # noqa: BLE001 - surface cleanly
                raise ToolError(f"spotify search failed: {exc}") from exc
            items = ((results.get("tracks") or {}).get("items") or [])
            if not items:
                raise ToolError(f'no Spotify results for "{target}"')
            top = items[0]
            uri = str(top.get("uri") or
                      f"spotify:track:{top.get('id', '')}")
            artists = ", ".join(
                a.get("name", "") for a in top.get("artists", []))
            name = str(top.get("name", "") or "")
            final_title = f"{artists} – {name}" if artists and name else (
                name or uri)
            artist = artists
        self._enqueue(uri, "spotify", final_title, artist=artist)
        started = self.play(len(self.queue()) - 1)
        return {"uri": uri, "title": final_title, **started}

    def play_soundcloud(self, target: str) -> dict[str, Any]:
        """Play a SoundCloud link now, or a search query.

        Links to playlists enqueue every track; artist pages enqueue
        their latest tracks. A bare query plays the top search result.
        The stream URL is resolved fresh and played through the local
        audio backend (mpv preferred).
        """
        target = (target or "").strip()
        if not target:
            raise ToolError(
                "play_soundcloud needs a soundcloud.com link or a search "
                "query")
        adapter = self._require_soundcloud()
        if self.detect_source(target) == "soundcloud":
            resolved = self._adapter_call("soundcloud resolve",
                                          adapter.resolve,
                                          target)  # fail fast on bad URLs
            kind = resolved["kind"]
            if kind == "track":
                tracks = [resolved["track"]]
            elif kind == "playlist":
                tracks = self._adapter_call("soundcloud playlist tracks",
                                            adapter.playlist_tracks, target)
            elif kind == "user":
                tracks = self._adapter_call("soundcloud user tracks",
                                            adapter.user_tracks, target,
                                            limit=25)
            else:  # pragma: no cover
                raise ToolError(f"unsupported SoundCloud kind {kind!r}")
        else:
            tracks = self._adapter_call("soundcloud search",
                                        adapter.search_tracks, target, limit=1)
        if not tracks:
            raise ToolError(
                f'no playable SoundCloud results for "{target}"')
        first = len(self.queue())
        for tr in tracks:
            self._enqueue(
                tr["permalink_url"], "soundcloud", tr["title"],
                artist=tr.get("artist", ""),
                duration=(tr.get("duration_ms") or 0) / 1000.0)
        started = self.play(first)
        return {"tracks": len(tracks), "title": tracks[0]["title"],
                **started}

    # ── YouTube ───────────────────────────────────────────────────────
    # No adapter needed: search and audio extraction go through yt-dlp
    # (optional dependency).  Audio is downloaded once per video id and
    # cached as a local file, so replays cost nothing.

    @staticmethod
    def _youtube_id(url: str) -> str:
        import re
        m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|youtube:)"
                      r"([A-Za-z0-9_\-]{11})", url or "")
        return m.group(1) if m else ""

    @staticmethod
    def _youtube_watch_url(video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    @classmethod
    def _youtube_search_id(cls, query: str) -> str:
        """Top YouTube video id for a query (yt-dlp ``ytsearch1:``).

        Raises ToolError naming the missing dependency when yt-dlp is
        absent — never a silent empty result.
        """
        query = (query or "").strip()
        if not query:
            raise ToolError("youtube search needs a query")
        # python module first, CLI fallback
        try:
            import yt_dlp  # noqa: F401
            has_module = True
        except ImportError:
            has_module = False
        if has_module:
            import yt_dlp
            import contextlib
            import io
            try:
                # Suppress yt-dlp's stderr — it writes "ERROR:" lines that
                # pollute test output and CI logs. We raise ToolError on
                # failure anyway.
                from .cookies import ytdlp_cookie_opts as _cookie_opts_fn
                _opts = {"quiet": True, "no_warnings": True,
                         "skip_download": True}
                _opts.update(_cookie_opts_fn())
                with contextlib.redirect_stderr(io.StringIO()):
                    with yt_dlp.YoutubeDL(_opts) as ydl:
                        info = ydl.extract_info(f"ytsearch1:{query}",
                                                download=False)
            except Exception as exc:  # noqa: BLE001
                raise ToolError(f"youtube search failed: {exc}") from exc
            entries = ((info or {}).get("entries") or [])
            if entries and entries[0].get("id"):
                return str(entries[0]["id"])
            raise ToolError(f'no YouTube results for "{query}"')
        cli = shutil.which("yt-dlp")
        if cli:
            try:
                from .cookies import ytdlp_cookie_args as _cookie_args_fn
                proc = subprocess.run(
                    [cli, "--print", "id", "--skip-download"]
                    + _cookie_args_fn() + [f"ytsearch1:{query}"],
                    capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as exc:
                raise ToolError("youtube search timed out") from exc
            vid = (proc.stdout or "").strip().splitlines()
            vid = vid[0].strip() if vid else ""
            if proc.returncode == 0 and vid:
                return vid
            raise ToolError(
                f"youtube search failed: "
                f"{(proc.stderr or '').strip()[-200:] or 'no results'}")
        raise ToolError(
            "youtube search needs yt-dlp (pip install yt-dlp) — "
            "not installed here")

    def _youtube_media_dir(self) -> str:
        from ..tools.filesystem import safe_path
        d = safe_path(self.context, "media")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    @classmethod
    def _youtube_search_many(cls, query: str,
                             limit: int = 8) -> list[dict[str, Any]]:
        """Top N YouTube results for a query (yt-dlp ``ytsearchN:``).

        Returns [{video_id, title, duration, uploader}].  Empty list when
        yt-dlp is missing or the search fails — never raises.
        """
        from ..core.logging_setup import get_logger as _get_logger
        _log = _get_logger(__name__)
        query = (query or "").strip()
        limit = max(1, min(int(limit or 8), 25))
        if not query:
            return []
        try:
            import yt_dlp  # noqa: F401
            has_module = True
        except ImportError:
            has_module = False
        entries: list[dict[str, Any]] = []
        if has_module:
            import yt_dlp
            import contextlib
            import io
            try:
                from .cookies import ytdlp_cookie_opts as _cookie_opts_fn2
                _opts2 = {"quiet": True, "no_warnings": True,
                          "skip_download": True}
                _opts2.update(_cookie_opts_fn2())
                with contextlib.redirect_stderr(io.StringIO()):
                    with yt_dlp.YoutubeDL(_opts2) as ydl:
                        info = ydl.extract_info(f"ytsearch{limit}:{query}",
                                                download=False)
                entries = (info or {}).get("entries") or []
            except Exception as exc:  # noqa: BLE001
                _log.info("youtube multi-search failed: %s", exc)
                return []
        else:
            import shutil
            import subprocess
            cli = shutil.which("yt-dlp")
            if not cli:
                return []
            try:
                from .cookies import ytdlp_cookie_args as _cookie_args_fn2
                proc = subprocess.run(
                    [cli, "--print", "%(id)s\t%(title)s\t%(duration)s\t"
                     "%(uploader)s", "--skip-download"]
                    + _cookie_args_fn2() + [f"ytsearch{limit}:{query}"],
                    capture_output=True, text=True, timeout=60)
            except (subprocess.TimeoutExpired, OSError) as exc:
                _log.info("youtube multi-search failed: %s", exc)
                return []
            if proc.returncode != 0:
                return []
            for line in (proc.stdout or "").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip():
                    entries.append({
                        "id": parts[0].strip(),
                        "title": parts[1].strip() if len(parts) > 1 else "",
                        "duration": parts[2].strip() if len(parts) > 2 else "",
                        "uploader": parts[3].strip() if len(parts) > 3 else "",
                    })
        out: list[dict[str, Any]] = []
        for e in entries:
            vid = str((e or {}).get("id") or "").strip()
            if not _YT_ID_RE.match(vid):
                continue
            try:
                dur = float((e or {}).get("duration") or 0)
            except (TypeError, ValueError):
                dur = 0.0
            out.append({
                "video_id": vid,
                "title": str((e or {}).get("title") or "").strip(),
                "duration": dur,
                "uploader": str((e or {}).get("uploader") or "").strip(),
            })
            if len(out) >= limit:
                break
        return out

    def _youtube_audio(self, item: dict[str, Any]) -> str:
        """Local audio file for a youtube queue item (download + cache).

        Cached by video id under the media dir, so the second play of
        the same video never re-downloads.
        """
        video_id = self._youtube_id(str(item.get("path", "")))
        if not video_id:
            raise ToolError(
                f"can't parse a YouTube video id from {item.get('path')!r}")
        media_dir = self._youtube_media_dir()
        for ext in ("mp3", "m4a", "webm", "opus", "ogg", "wav"):
            for cand in sorted(Path(media_dir).glob(f"*[{video_id}].{ext}")):
                if cand.is_file() and cand.stat().st_size > 0:
                    return str(cand)
        tools = getattr(self.context, "tools", None)
        call = getattr(tools, "call", None) if tools else None
        if call is None:
            raise ToolError(
                "youtube audio extraction needs the tool registry "
                "(media_download) — and yt-dlp installed "
                "(pip install yt-dlp)")
        url = self._youtube_watch_url(video_id)
        try:
            # page_url = the watch page itself: the browser stage opens
            # the real page and extracts the media through the session.
            out = call("media_download", url=url, audio_only=True,
                       page_url=url)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"youtube audio download failed: {exc}") from exc
        if not getattr(out, "ok", False):
            raise ToolError(
                f"youtube audio download failed: "
                f"{getattr(out, 'error', 'unknown error')}")
        value = out.value if isinstance(out.value, dict) else {}
        path = str(value.get("path", ""))
        if not path or not os.path.exists(path):
            raise ToolError("youtube download reported success but "
                            "produced no file")
        return path

    def download(self, item: dict[str, Any]) -> dict[str, Any]:
        """Resolve a queue item to a local audio file path.

        Used by chat surfaces (Telegram/WhatsApp) to SEND the audio file
        instead of playing it locally via mpv. Returns {"ok", "path", "title"}
        or {"ok": False, "reason"} — never raises, never fake success.

        - file: already local → path as-is
        - youtube: download + cache via _youtube_audio
        - soundcloud: resolve stream URL, download via media_download
        - audiomack: download via media_download (yt-dlp extractor)
        - netnaija: scrape the post page for the direct MP3, then download
        - url: download via media_download (yt-dlp or direct HTTP)
        - spotify / boomplay: honest refusal (DRM / protected streams)
        """
        try:
            kind = str(item.get("kind", ""))
            title = str(item.get("title", "") or item.get("path", ""))
            if kind == "spotify":
                return {"ok": False, "reason":
                        "Spotify tracks can't be downloaded (DRM) — "
                        "they play on your linked Spotify device, not in chat. "
                        "Try /play youtube:<song> instead."}
            if kind == "boomplay":
                return {"ok": False, "reason":
                        "Boomplay streams are protected — they play in the "
                        "Boomplay app, not in chat. "
                        "Try /play youtube:<song> or /play netnaija:<song> "
                        "instead."}
            if kind == "file":
                path = str(item.get("path", ""))
                if path and os.path.isfile(path):
                    return {"ok": True, "path": path, "title": title}
                return {"ok": False, "reason": f"file not found: {path}"}
            if kind == "youtube":
                try:
                    path = self._youtube_audio(item)
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "reason": f"YouTube download failed: {exc}"}
                return {"ok": True, "path": path, "title": title}
            # soundcloud + audiomack + netnaija + generic url:
            # resolve then download
            url = str(item.get("path", ""))
            if kind == "soundcloud":
                try:
                    url = self._soundcloud_play_url(item)
                except Exception as exc:  # noqa: BLE001
                    # The SoundCloud api-v2 has bad days (rotated client
                    # ids, rate limits, regional blocks).  Instead of dying
                    # here, fall back to yt-dlp on the permalink URL —
                    # yt-dlp extracts SoundCloud natively with no API key,
                    # so the track still downloads.
                    _log.info("soundcloud API failed for %r (%s); "
                              "falling back to yt-dlp",
                              item.get("path"), exc)
                    url = str(item.get("path", ""))
            if kind == "netnaija":
                from .sources import NetNaijaSource, SourceCandidate
                direct = NetNaijaSource().download_url(SourceCandidate(
                    source="netnaija", title=title, url=url))
                if not direct:
                    _log.info("netnaija: no direct MP3 for %r; "
                              "trying YouTube fallback", title)
                    return self._download_youtube_fallback(
                        item, title, "NetNaija download page unreachable",
                        source_label="NetNaija")
                url = direct
                _log.info("netnaija: direct MP3 resolved for %r", title)
            if not url:
                return {"ok": False, "reason": "no URL to download"}
            tools = getattr(self.context, "tools", None)
            call = getattr(tools, "call", None) if tools else None
            if call is None:
                return {"ok": False, "reason":
                        "download needs the tool registry (media_download)"}
            dl_error: Any = None
            out = None
            try:
                # page_url = the source page (permalink / post page) so
                # the browser stage opens the real page when direct and
                # proxy stages fail — not the resolved stream URL.
                page_url = str(item.get("path", "") or "")
                out = call("media_download", url=url, audio_only=True,
                           page_url=page_url if page_url != url else "")
            except Exception as exc:  # noqa: BLE001
                dl_error = exc
            if dl_error is None and not getattr(out, "ok", False):
                dl_error = getattr(out, "error", "unknown error")
            if dl_error is not None:
                err_msg = str(dl_error)
                # SoundCloud/Audiomack/NetNaija tracks sometimes come back
                # DRM/private-locked or unreachable.  Fall back to YouTube
                # once using the known artist/title instead of failing
                # outright.
                if kind in ("soundcloud", "audiomack",
                            "netnaija") and is_protection_error(err_msg):
                    return self._download_youtube_fallback(
                        item, title, err_msg,
                        source_label={"soundcloud": "SoundCloud",
                                      "audiomack": "Audiomack",
                                      "netnaija": "NetNaija"}[kind])
                return {"ok": False, "reason": f"download failed: {err_msg}"}
            value = out.value if isinstance(out.value, dict) else {}
            path = str(value.get("path", ""))
            if not path or not os.path.isfile(path):
                return {"ok": False, "reason":
                        "download reported success but produced no file"}
            res: dict[str, Any] = {"ok": True, "path": path, "title": title}
            stages = value.get("stages")
            if stages:
                res["stages"] = stages  # fallback chain trail, e.g.
                # ["direct"] or ["direct", "proxy", "browser"]
            return res
        except Exception as exc:  # noqa: BLE001 - never raises
            return {"ok": False, "reason": f"download error: {exc}"}

    def _download_youtube_fallback(self, item: dict[str, Any], title: str,
                                   sc_error: str,
                                   source_label: str = "SoundCloud") -> dict[str, Any]:
        """One-shot YouTube fallback after a source protection failure.

        Reuses the item's known artist/title to search YouTube and
        downloads the top result. Never raises — returns the honest
        failure when YouTube can't deliver either.
        """
        artist = str(item.get("artist", "") or "").strip()
        query = f"{artist} {title}".strip() if artist else title.strip()
        _log.info("%s download blocked (%s); trying YouTube for %r",
                  source_label, sc_error[:80], query)
        try:
            video_id = self._youtube_search_id(query)
        except Exception as exc:  # noqa: BLE001
            from .cookies import BOT_COOKIE_HELP, is_bot_detection_error
            if is_bot_detection_error(exc):
                return {"ok": False,
                        "reason": f"{source_label} blocked the download ({sc_error[:80]}). "
                                  f"{BOT_COOKIE_HELP}"}
            return {"ok": False,
                    "reason": f"{source_label} blocked the download ({sc_error[:80]}) "
                              f"and YouTube search failed: {exc}"}
        yt_item = {"kind": "youtube",
                   "path": self._youtube_watch_url(video_id),
                   "title": title}
        try:
            path = self._youtube_audio(yt_item)
        except Exception as exc:  # noqa: BLE001
            from .cookies import BOT_COOKIE_HELP, is_bot_detection_error
            err = str(exc)
            if is_bot_detection_error(err):
                return {"ok": False,
                        "reason": f"{source_label} blocked the download ({sc_error[:80]}). "
                                  f"{BOT_COOKIE_HELP}"}
            return {"ok": False,
                    "reason": f"{source_label} blocked the download ({sc_error[:80]}) "
                              f"and YouTube download failed: {exc}"}
        _log.info("youtube fallback succeeded for %r", query)
        return {"ok": True, "path": path, "title": title,
                "note": f"via YouTube ({source_label} was blocked)"}

    def _add_youtube(self, target: str, title: str = "") -> list[dict[str, Any]]:
        """Enqueue a YouTube URL or search query (audio extracted at play)."""
        target = (target or "").strip()
        if target.lower().startswith("youtube:"):
            query = target.split(":", 1)[1].strip()
            video_id = self._youtube_search_id(query)
            url = self._youtube_watch_url(video_id)
            label = query
        else:
            video_id = self._youtube_id(target)
            if video_id:
                url = self._youtube_watch_url(video_id)
                label = title or f"YouTube {video_id}"
            else:
                # bare query → keyless search
                video_id = self._youtube_search_id(target)
                url = self._youtube_watch_url(video_id)
                label = title or target
        return [self._enqueue(url, "youtube", label)]

    def play_youtube(self, target: str) -> dict[str, Any]:
        """Play a YouTube URL now, or a search query.

        A bare query plays the top search result.  Audio is extracted
        with yt-dlp on first play and cached, so replays are instant.
        Fails fast with an honest message when yt-dlp is missing.
        """
        target = (target or "").strip()
        if not target:
            raise ToolError(
                "play_youtube needs a youtube.com URL or a search query")
        added = self._add_youtube(target)
        started = self.play(len(self.queue()) - 1)
        return {"video": added[0]["path"], "title": added[0]["title"],
                **started}

    def _queue_rows(self) -> list[Any]:
        return self.db.query(
            "SELECT * FROM media_queue ORDER BY position, added_at")

    def queue(self) -> list[dict[str, Any]]:
        rows = self._queue_rows()
        return [{"index": i, "path": r["path"], "title": r["title"],
                 "kind": r["kind"], "artist": r.get("artist", "") or "",
                 "album": r.get("album", "") or "",
                 "duration": r.get("duration", 0) or 0}
                for i, r in enumerate(rows)]

    def remove(self, index: int) -> dict[str, Any]:
        rows = self.db.query("SELECT * FROM media_queue ORDER BY position, "
                             "added_at")
        if not (0 <= int(index) < len(rows)):
            raise ToolError(f"no queue item {index}")
        self.db.execute("DELETE FROM media_queue WHERE id=?",
                        (rows[int(index)]["id"],))
        return {"removed": rows[int(index)]["path"],
                "queue": len(self.queue())}

    def clear(self) -> dict[str, Any]:
        self.db.execute("DELETE FROM media_queue")
        self._state["position"] = 0
        self._save_state()
        return {"cleared": True, "queue": 0}

    # ── queue management: shuffle / repeat / reorder ──────────────────────
    def shuffle(self, on: bool | None = None) -> dict[str, Any]:
        """Shuffle the queue.  The current track stays first; the rest are
        randomized.  Turning shuffle off keeps the current order (there is
        no fake "original order" to restore)."""
        rows = self._queue_rows()
        if not rows:
            raise ToolError("queue is empty")
        current = bool(self._state.get("shuffle", False))
        on = (not current) if on is None else bool(on)
        resynced = False
        if on:
            pos = self._resolve_position()
            cur_id = rows[pos]["id"]
            rest = [r["id"] for i, r in enumerate(rows) if i != pos]
            random.shuffle(rest)
            for new_pos, rid in enumerate([cur_id, *rest]):
                self.db.execute("UPDATE media_queue SET position=? WHERE "
                                "id=?", (new_pos, rid))
            self._state["position"] = 0
            if self.backend.name == "mpv" and self._mpv_alive():
                # keep the live mpv playlist truthful after the reorder
                ok, _ = self._mpv_play_at(0)
                resynced = ok
        self._state["shuffle"] = on
        self._save_state()
        return {"shuffle": on, "resynced": resynced,
                "queue": len(self.queue())}

    _REPEAT_MODES = ("off", "one", "all")

    def repeat(self, mode: str | None = None) -> dict[str, Any]:
        """Repeat mode: off | one (repeat the track) | all (loop the queue).

        With no argument, reports the current mode.  On mpv the mode also
        drives the live loop-playlist/loop-file properties.
        """
        if not mode:
            return {"repeat": str(self._state.get("repeat", "off"))}
        mode = mode.strip().lower()
        if mode not in self._REPEAT_MODES:
            raise ToolError(
                f"unknown repeat mode {mode!r} — choose from "
                f"{list(self._REPEAT_MODES)}")
        self._state["repeat"] = mode
        self._save_state()
        if self.backend.name == "mpv":
            self._mpv_transport(["set_property", "loop-playlist",
                                 "inf" if mode == "all" else "no"])
            self._mpv_transport(["set_property", "loop-file",
                                 "inf" if mode == "one" else "no"])
        return {"repeat": mode}

    def move(self, index: int, to: int) -> dict[str, Any]:
        """Move a queue item to another position.  The playhead keeps
        pointing at the same track."""
        rows = self._queue_rows()
        n = len(rows)
        if not n:
            raise ToolError("queue is empty")
        i, j = int(index), int(to)
        if not (0 <= i < n):
            raise ToolError(f"no queue item {index}")
        j = max(0, min(j, n - 1))
        ids = [r["id"] for r in rows]
        rid = ids.pop(i)
        ids.insert(j, rid)
        for new_pos, rid2 in enumerate(ids):
            self.db.execute("UPDATE media_queue SET position=? WHERE id=?",
                            (new_pos, rid2))
        cur_id = rows[self._resolve_position()]["id"]
        self._state["position"] = ids.index(cur_id)
        self._save_state()
        return {"moved": rows[i]["title"], "from": i, "to": j,
                "queue": len(self.queue())}

    def save_playlist(self, name: str) -> dict[str, Any]:
        """Persist the current queue as a named playlist.

        Creates the playlist when missing, replaces its contents when it
        exists — save semantics. The queue itself is untouched."""
        q = self.queue()
        if not q:
            raise ToolError("the queue is empty — nothing to save")
        return MusicLibrary(self.context).save_queue_as_playlist(name, q)

    # ── playlists → queue ─────────────────────────────────────────────────
    def load_playlist(self, name: str, *, autoplay: bool = True
                      ) -> dict[str, Any]:
        """Replace the queue with a saved playlist and play it."""
        lib = MusicLibrary(self.context)
        pl = lib.playlist(name)  # raises ToolError for an unknown name
        items = pl["items"]
        if not items:
            raise ToolError(f"playlist {name!r} is empty")
        self.db.execute("DELETE FROM media_queue")
        stamp = time.time()
        for i, it in enumerate(items):
            self.db.execute(
                "INSERT INTO media_queue (id, position, path, title, kind, "
                "artist, album, duration, added_at) VALUES "
                "(?,?,?,?,?,?,?,?,?)",
                (f"mq-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
                 stamp + i, it["path"], it["title"], it.get("kind") or "file",
                 it.get("artist") or "", it.get("album") or "",
                 float(it.get("duration") or 0), stamp))
        self._state["position"] = 0
        self._save_state()
        if autoplay:
            started = self.play(0)
            return {"playlist": pl["name"], "loaded": len(items), **started}
        return {"playlist": pl["name"], "loaded": len(items),
                "queue": len(items)}

    # ── history ───────────────────────────────────────────────────────────
    def _record_played(self, item: dict[str, Any], source: str) -> None:
        """Log a started track.  History must never break playback."""
        try:
            MusicLibrary(self.context).record_played(
                item["path"], title=item.get("title", "") or "",
                artist=item.get("artist", "") or "",
                album=item.get("album", "") or "",
                duration=float(item.get("duration") or 0), source=source)
        except Exception as exc:  # noqa: BLE001 - history is auxiliary
            _log.debug("history record failed: %s", exc)

    # ── now playing ───────────────────────────────────────────────────────
    def now(self) -> dict[str, Any]:
        """Rich now-playing status: track metadata, live progress, and the
        full player state in one call."""
        q = self.queue()
        pos = self._resolve_position()
        item = dict(q[pos]) if q else {}
        time_pos: float | None = None
        time_remaining: float | None = None
        duration = float(item.get("duration") or 0)
        playing = False
        paused = False
        via = str(item.get("kind", "")) if item else ""
        spotify_live: dict[str, Any] = {}
        if via == "spotify" and self._spotify is not None:
            # Spotify owns this track: progress comes from the account.
            try:
                spotify_live = self._spotify.now_playing() or {}
            except Exception:  # noqa: BLE001 - now() stays truthful
                spotify_live = {}
            playing = bool(spotify_live.get("playing",
                                            self._state.get("playing", False)))
            paused = not playing and bool(self._state.get("paused", False))
            progress = spotify_live.get("progress_ms")
            if isinstance(progress, (int, float)):
                time_pos = float(progress) / 1000.0
        elif self.backend.name == "mpv" and self._mpv_alive():
            props = self._mpv_get_props(
                ["playback-status", "time-pos", "duration"], timeout=1.0) or {}
            tp = props.get("time-pos")
            if isinstance(tp, (int, float)):
                time_pos = float(tp)
            d = props.get("duration")
            if isinstance(d, (int, float)) and d > 0:
                duration = float(d)
            playing = props.get("playback-status") == "playing"
            paused = props.get("playback-status") == "paused"
        elif self.backend.name != "console":
            playing = bool(self._state.get("playing")) and self._simple_alive()
        if time_pos is not None and duration > 0:
            time_remaining = max(0.0, duration - time_pos)
        liked = (MusicLibrary(self.context).is_liked(item["path"])
                 if item else False)
        return {
            "track": item,
            "position": pos,
            "queue": len(q),
            "time_pos": time_pos,
            "duration": duration,
            "time_remaining": time_remaining,
            "playing": playing,
            "paused": paused,
            "liked": liked,
            "volume": int(self._state.get("volume", 80)),
            "backend": self.backend.name,
            "via": via if via not in ("file", "url") else "local",
            "spotify": spotify_live,
            "repeat": str(self._state.get("repeat", "off")),
            "shuffle": bool(self._state.get("shuffle", False)),
        }

    # ── transport ─────────────────────────────────────────────────────────
    def play(self, index: int | None = None) -> dict[str, Any]:
        q = self.queue()
        if not q:
            raise ToolError("queue is empty — add something first")
        if index is not None:
            if not (0 <= int(index) < len(q)):
                raise ToolError(f"no queue item {index}")
            self._state["position"] = int(index)
        else:
            self._state["position"] = self._resolve_position()
        self._save_state()
        return self._start_at(self._state["position"], source="play")

    def _start_at(self, pos: int, source: str = "play") -> dict[str, Any]:
        """Start playback at ``pos``, skipping unplayable items.

        A queue item is unplayable when its stream can't be resolved or
        started — a SoundCloud link gone stale, Spotify unreachable
        offline, the device gone. Instead of dying on the first bad
        item, the engine walks forward to the next playable track and
        reports every skip, so "offline" degrades to local tracks
        instead of a dead stop. Raises only when nothing in the queue
        plays at all.
        """
        q = self.queue()
        n = len(q)
        skipped: list[dict[str, Any]] = []
        for step in range(n):
            i = (pos + step) % n
            item = q[i]
            try:
                started = self._start_one(i, item, source)
            except (ToolError, NoMoralsError) as exc:
                skipped.append({"index": i, "title": item["title"],
                                "kind": item["kind"], "error": str(exc)})
                continue
            self._state["position"] = i
            self._save_state()
            if skipped:
                started["skipped"] = skipped
            return started
        detail = "; ".join(f"#{s['index']} {s['title']!r}: {s['error']}"
                           for s in skipped)
        raise ToolError("nothing in the queue is playable right now"
                        + (f" — {detail}" if detail else ""))

    def _start_one(self, index: int, item: dict[str, Any],
                   source: str = "play") -> dict[str, Any]:
        if item["kind"] == "spotify":
            return self._spotify_start(item)
        if self.backend.name == "console":
            self._state["playing"] = False
            self._save_state()
            result: dict[str, Any] = {
                "status": "no-backend",
                "backend": "console",
                "current": item,
                "hint": ("no audio backend found — install mpv "
                         "(Termux: pkg install mpv) or ffmpeg for ffplay; "
                         "the queue is saved"),
            }
            if item["kind"] == "soundcloud":
                # nothing can play it here, but the direct stream URL is
                # still useful — resolve it. A resolution failure is fatal
                # (fail fast): the track has no playable stream.
                result["stream_url"] = self._soundcloud_play_url(item)
            if item["kind"] == "youtube":
                # same idea: the watch URL is the playable artifact here
                result["watch_url"] = item["path"]
            return result
        path = item["path"]
        if item["kind"] == "soundcloud":
            # resolve fresh every play: stream URLs expire
            path = self._soundcloud_play_url(item)
        if item["kind"] == "youtube":
            # extract once, cache by video id, then play the local file
            path = self._youtube_audio(item)
        play_item = dict(item, path=path)
        if self.backend.name == "mpv":
            ok, detail = self._mpv_play_at(index)
        else:
            ok, detail = self._simple_start(play_item)
        self._state["playing"] = ok
        self._state["current"] = item["path"]
        self._state.pop("stream_via", None)
        self._save_state()
        if not ok:
            return {"status": "error", "error": detail,
                    "backend": self.backend.name}
        self._record_played(play_item, source)
        return {"status": "playing", "backend": self.backend.name,
                "current": item["title"], "path": item["path"],
                "via": item["kind"]
                if item["kind"] not in ("file", "url") else "local"}

    def _spotify_start(self, item: dict[str, Any]) -> dict[str, Any]:
        """Play a Spotify queue item on the owner's active device.

        Spotify serves no direct audio stream, so this never touches the
        local backends — any local audio is stopped first. The adapter's
        own fail-fast covers the no-active-device case.
        """
        adapter = self._require_spotify()
        self._stop_local_audio()
        result = self._adapter_call("spotify play", adapter.play,
                                    uris=[item["path"]])
        self._state["playing"] = True
        self._state["paused"] = False
        self._state["current"] = item["path"]
        self._state["stream_via"] = "spotify"
        self._save_state()
        self._record_played(item, "spotify")
        return {"status": "playing", "via": "spotify",
                "current": item["title"], "uri": item["path"],
                "device_id": (result or {}).get("device_id", "")}

    # ── mpv (playlist = queue, IPC = transport) ───────────────────────────
    def _mpv_sock_path(self) -> str:
        from ..tools.filesystem import safe_path

        d = safe_path(self.context, "player")
        d.mkdir(parents=True, exist_ok=True)
        return str(d / "ipc.sock")

    def _mpv_alive(self) -> bool:
        path = str(self._state.get("mpv_sock", ""))
        if not path or not os.path.exists(path):
            return False
        props = self._mpv_get_props([
            "playback-status", "playlist-index", "time-pos", "volume"],
            timeout=1.0, sock_path=path)
        if props is None:
            return False
        # A dead mpv leaves a stale socket file behind: every property
        # comes back None. Require at least one real value, otherwise
        # treat mpv as dead (this was the "/play lost the mpv IPC
        # connection" bug on Termux — the phone's OOM killer takes mpv
        # out and the stale socket fooled the alive check).
        if not any(v is not None for v in props.values()):
            try:
                os.unlink(path)
            except OSError:  # noqa: E103 - best-effort stale cleanup
                pass
            self._state.pop("mpv_sock", None)
            self._state.pop("mpv_map", None)
            return False
        return True

    def _mpv_connect(self, sock_path: str,
                     timeout: float = 8.0) -> socket.socket | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(sock_path)
                s.settimeout(None)
                return s
            except (ConnectionRefusedError, FileNotFoundError,
                    socket.timeout, OSError):
                time.sleep(0.15)
        return None

    def _mpv_spawn(self) -> tuple[bool, str, str]:
        """Spawn a detached idle mpv with IPC; → (ok, sock_path, error)."""
        sock_path = self._mpv_sock_path()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:  # noqa: E103 - stale socket may not exist
            pass
        cmd = [self.backend.binary, "--idle=yes", "--really-quiet",
               "--no-terminal", f"--input-ipc-server={sock_path}",
               f"--volume={int(self._state.get('volume', 80))}"]
        # Termux: the default audio output often has no server to talk
        # to (no PulseAudio). opensles is the reliable path on Android —
        # without it mpv starts then dies silently, leaving the stale
        # socket that broke /play.
        if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
            cmd.append("--ao=opensles")
        try:
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except OSError as exc:
            return False, sock_path, f"mpv failed to start: {exc}"
        sock = self._mpv_connect(sock_path)
        if sock is None:
            return False, sock_path, "mpv started but IPC socket never came up"
        sock.close()
        return True, sock_path, ""

    def _mpv_ipc(self, sock_path: str, command: list[Any],
                 timeout: float = 2.0) -> bool:
        s = self._mpv_connect(sock_path, timeout=timeout)
        if s is None:
            return False
        try:
            s.sendall((json.dumps({"command": command}) + "\n").encode())
            s.settimeout(0.4)
            try:
                while s.recv(4096):
                    pass
            except (socket.timeout, OSError):  # noqa: E103 - timeout ends the drain loop by design
                pass
            return True
        except OSError:
            return False
        finally:
            s.close()

    def _mpv_get_props(self, props: list[str], timeout: float = 2.0,
                       sock_path: str | None = None
                       ) -> dict[str, Any] | None:
        """get_property for each prop; → {prop: value} or None if dead."""
        path = sock_path or str(self._state.get("mpv_sock", ""))
        if not path:
            return None
        # one property per short connection: exact pairing, no ambiguity
        out: dict[str, Any] = {}
        for p in props:
            out[p] = self._mpv_get_one(path, p, timeout=timeout)
        return out

    def _mpv_get_one(self, sock_path: str, prop: str,
                     timeout: float = 1.5) -> Any:
        s = self._mpv_connect(sock_path, timeout=timeout)
        if s is None:
            return None
        try:
            s.sendall((json.dumps({"command": ["get_property", prop]})
                       + "\n").encode())
            s.settimeout(1.0)
            buf = b""
            try:
                while b'"error"' not in buf:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            except socket.timeout:  # noqa: E103 - partial response is still usable
                pass
            for line in buf.split(b"\n"):
                try:
                    ev = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if ev.get("error") == "success" and "data" in ev:
                    return ev["data"]
            return None
        finally:
            s.close()

    def _mpv_play_at(self, pos: int) -> tuple[bool, str]:
        q = self.queue()
        if not self._mpv_alive():
            ok, sock_path, err = self._mpv_spawn()
            if not ok:
                return False, err
            self._state["mpv_sock"] = sock_path
            self._save_state()
        else:
            sock_path = str(self._state.get("mpv_sock", ""))
        # Build the mpv playlist: local kinds directly; soundcloud items
        # resolve fresh to stream URLs (from pos onward — earlier ones are
        # re-resolved if/when reached); spotify items are never mpv's
        # business (they play on the user's Spotify device). A soundcloud
        # item that won't resolve (stale link, offline) is left OUT of
        # mpv's playlist when it sits ahead of pos — the current track
        # still starts, and the dead item is retried-or-skipped when the
        # playhead reaches it. Only the item AT pos failing is fatal
        # here; the _start_at skip-walker moves past it.
        entries: list[tuple[int, str]] = []
        for i, item in enumerate(q):
            kind = item["kind"]
            if kind == "spotify":
                continue
            path = item["path"]
            if kind == "soundcloud":
                if i < pos:
                    continue
                try:
                    path = self._soundcloud_play_url(item)
                except (ToolError, NoMoralsError):
                    if i == pos:
                        raise
                    _log.debug("mpv playlist: dropping unresolvable "
                               "soundcloud item %d (%s)", i, item["path"])
                    continue
            entries.append((i, path))
        queue_indexes = [i for i, _ in entries]
        if pos not in queue_indexes:
            return False, f"queue item {pos} is not mpv-playable"
        self._state["mpv_map"] = queue_indexes
        self._save_state()
        # sync the durable queue into the playlist, then jump to pos.
        # If mpv dies mid-sync (phone OOM killer, audio device hiccup),
        # respawn once and retry the whole sync before reporting failure.
        for attempt in range(2):
            failed = False
            for j, (_, path) in enumerate(entries):
                how = "replace" if j == 0 else "append"
                if not self._mpv_ipc(sock_path, ["loadfile", path, how]):
                    failed = True
                    break
            if not failed:
                jump = queue_indexes.index(pos)
                if self._mpv_ipc(sock_path, ["set_property", "playlist-pos",
                                             jump]):
                    return True, ""
                failed = True
            if failed and attempt == 0:
                _log.debug("mpv IPC failed during playlist sync, "
                           "respawning once")
                ok, new_sock, err = self._mpv_spawn()
                if not ok:
                    return False, (
                        "mpv died and wouldn't restart: "
                        f"{err}. On Termux: pkg install mpv")
                self._state["mpv_sock"] = new_sock
                self._save_state()
                sock_path = new_sock
        return False, (
            "lost the mpv IPC connection twice — mpv keeps dying. "
            "On Termux try: pkg install mpv (then /play again). "
            "If it persists, the phone may be killing background audio; "
            "keep the Termux session in the foreground.")

    def _mpv_map_is_identity(self) -> bool:
        """True when mpv's playlist mirrors the queue 1:1 — the only case
        where playlist-next/prev stays in sync with queue positions."""
        q = self.queue()
        return list(self._state.get("mpv_map") or []) == list(range(len(q)))

    def _mpv_transport(self, command: list[Any]) -> bool:
        if self.backend.name != "mpv":
            return False
        sock_path = str(self._state.get("mpv_sock", ""))
        if not sock_path or not self._mpv_alive():
            return False
        return self._mpv_ipc(sock_path, command)

    # ── non-mpv backends ──────────────────────────────────────────────────
    def _kill_simple(self) -> None:
        """Kill the simple-backend player process, if one is tracked.

        Called before starting a new track so the old one never keeps
        playing underneath (and its pid is never orphaned from state).
        """
        pid = int(self._state.get("player_pid", 0) or 0)
        if pid:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):  # noqa: E103 - process already gone
                pass
            self._state.pop("player_pid", None)

    def _simple_start(self, item: dict[str, str]) -> tuple[bool, str]:
        path = item["path"]
        if path.lower().startswith(("http://", "https://")) \
                and not self.backend.urls:
            return False, (f"{self.backend.name} can't stream URLs — "
                           "install mpv or ffmpeg for that")
        self._kill_simple()  # never stack a new track over a live one
        if self.backend.name == "aplay":
            cmd = [self.backend.binary, "-q", path]
        elif self.backend.name == "mpg123":
            cmd = [self.backend.binary, path]
        elif self.backend.name == "sox":
            cmd = ["play", "-q", path]
        elif self.backend.name == "afplay":
            cmd = [self.backend.binary, path]
        else:  # ffplay
            cmd = [self.backend.binary, "-nodisp", "-autoexit", "-loglevel",
                   "quiet", path]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self._state["player_pid"] = proc.pid
            return True, ""
        except OSError as exc:
            return False, f"backend failed to start: {exc}"

    def _simple_alive(self) -> bool:
        pid = int(self._state.get("player_pid", 0) or 0)
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    # ── control surface ───────────────────────────────────────────────────
    def pause(self) -> dict[str, Any]:
        if self._is_spotify_current():
            # Spotify owns this track: pause it on the device. The
            # adapter fails fast when nothing is playing.
            self._adapter_call("spotify pause",
                               self._require_spotify().pause)
            self._state["paused"] = True
            self._save_state()
            return {"status": "paused", "via": "spotify"}
        if self._mpv_transport(["set_property", "pause", True]):
            self._state["paused"] = True
            self._save_state()
            return {"status": "paused"}
        return self._no_transport("pause")

    def resume(self) -> dict[str, Any]:
        if self._is_spotify_current():
            # play() with no URIs resumes on the active device
            self._adapter_call("spotify resume",
                               self._require_spotify().play)
            self._state["paused"] = False
            self._state["playing"] = True
            self._save_state()
            return {"status": "resumed", "via": "spotify"}
        if self._mpv_transport(["set_property", "pause", False]):
            self._state["paused"] = False
            self._save_state()
            return {"status": "resumed"}
        return self._no_transport("resume")

    def seek(self, seconds: float) -> dict[str, Any]:
        if self._mpv_transport(["seek", float(seconds), "absolute"]):
            self._save_state()
            return {"status": "seeked", "seconds": float(seconds)}
        # honest report: nothing seekable here, but echo the request so
        # callers can display what was asked for
        resp = self._no_transport("seek")
        resp["seconds"] = float(seconds)
        return resp

    def volume(self, level: int | float) -> dict[str, Any]:
        level = max(0, min(100, int(level)))
        self._state["volume"] = level
        self._save_state()
        self._mpv_transport(["set_property", "volume", level])
        return {"status": "volume", "level": level}

    def next(self) -> dict[str, Any]:
        q = self.queue()
        if not q:
            raise ToolError("queue is empty")
        old_pos = self._resolve_position()
        self._state["position"] = (old_pos + 1) % len(q)
        self._save_state()
        new_item = q[self._state["position"]]
        if (self.backend.name == "mpv" and self._mpv_alive()
                and new_item["kind"] in self.LOCAL_KINDS
                and q[old_pos]["kind"] in self.LOCAL_KINDS
                and self._mpv_map_is_identity()):
            self._mpv_transport(["playlist-next"])
            self._state["playing"] = True
            self._state["paused"] = False
            self._save_state()
            self._record_played(q[self._state["position"]], "next")
            return {"status": "next",
                    "current": q[self._state["position"]]["title"]}
        # streaming kinds (or a reshaped mpv playlist) need a full
        # restart: SoundCloud URLs resolve fresh, Spotify takes over
        return self._start_at(self._state["position"], source="next")

    def prev(self) -> dict[str, Any]:
        q = self.queue()
        if not q:
            raise ToolError("queue is empty")
        old_pos = self._resolve_position()
        self._state["position"] = (old_pos - 1) % len(q)
        self._save_state()
        new_item = q[self._state["position"]]
        if (self.backend.name == "mpv" and self._mpv_alive()
                and new_item["kind"] in self.LOCAL_KINDS
                and q[old_pos]["kind"] in self.LOCAL_KINDS
                and self._mpv_map_is_identity()):
            self._mpv_transport(["playlist-prev"])
            self._state["playing"] = True
            self._state["paused"] = False
            self._save_state()
            self._record_played(q[self._state["position"]], "prev")
            return {"status": "prev",
                    "current": q[self._state["position"]]["title"]}
        return self._start_at(self._state["position"], source="prev")

    def stop(self) -> dict[str, Any]:
        self._stop_local_audio()
        if self._spotify is not None and self._is_spotify_current():
            try:
                self._spotify.pause()  # best-effort: local audio is stopped
            except Exception:  # noqa: BLE001 - stop must not fail
                pass
        self._state["playing"] = False
        self._state["paused"] = False
        self._state.pop("stream_via", None)
        self._save_state()
        return {"status": "stopped"}

    def status(self) -> dict[str, Any]:
        q = self.queue()
        pos = self._resolve_position()
        live: dict[str, Any] = {}
        playing = False
        paused = False
        via = q[pos]["kind"] if q else ""
        if via == "spotify":
            # Spotify owns this track: live state comes from the account,
            # not from any local backend.
            playing = bool(self._state.get("playing", False))
            paused = bool(self._state.get("paused", False))
            if self._spotify is not None:
                try:
                    live = {"via": "spotify",
                            "spotify": self._spotify.now_playing()}
                    np = live["spotify"] or {}
                    if isinstance(np, dict) and "playing" in np:
                        playing = bool(np["playing"])
                        paused = not playing
                except Exception:  # noqa: BLE001 - status stays truthful
                    live = {"via": "spotify", "spotify": "unreachable"}
        elif self.backend.name == "mpv":
            if self._mpv_alive():
                props = self._mpv_get_props(
                    ["playback-status", "playlist-pos", "time-pos",
                     "volume"], timeout=1.0) or {}
                live = {"mpv": True, **props}
                playing = props.get("playback-status") == "playing"
                paused = props.get("playback-status") == "paused"
                pl_pos = props.get("playlist-pos")
                mpv_map = list(self._state.get("mpv_map") or [])
                # translate the mpv playlist index back to a queue index
                q_pos = (mpv_map[pl_pos] if isinstance(pl_pos, int)
                         and 0 <= pl_pos < len(mpv_map) else pl_pos)
                if (isinstance(q_pos, int) and q and 0 <= q_pos < len(q)
                        and q_pos != pos):
                    # mpv auto-advanced past a track end on its own: follow
                    # it in durable state and log the new track
                    pos = q_pos
                    self._state["position"] = pos
                    self._save_state()
                    self._record_played(q[pos], "auto-advance")
        elif self.backend.name != "console":
            alive = self._simple_alive()
            live = {"alive": alive}
            playing = bool(self._state.get("playing")) and alive
        return {
            "backend": self.backend.describe(),
            "playing": playing,
            "paused": paused,
            "live": live,
            "via": via if via not in ("file", "url") else "local",
            "current": q[pos]["title"] if q else "",
            "path": q[pos]["path"] if q else "",
            "position": pos,
            "queue": len(q),
            "volume": int(self._state.get("volume", 80)),
        }

    def _no_transport(self, wanted: str) -> dict[str, Any]:
        return {"status": "unsupported",
                "error": (f"{wanted} needs the mpv backend "
                          f"(have: {self.backend.name})")}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "player",
        description=(
            "Music playback: queue files/URLs and control the player. "
            "action=add (target[, target2…] via targets, title) | play "
            "(index, -1 = current position; unplayable items — stale "
            "SoundCloud links, unreachable Spotify — are skipped with a "
            "report, failing only when nothing plays) | pause | resume | "
            "stop | seek (seconds) | volume (0-100) | next | prev | queue | "
            "remove (index) | move (index, to) | clear | shuffle "
            "(on=on|off|toggle) | repeat (mode=off|one|all) | now (rich "
            "now-playing) | playlist_play (name) | playlist_save (name) | "
            "status. Spotify URIs/links and "
            "SoundCloud links queue as streaming items when the adapters "
            "are wired (nm music wires them); YouTube URLs queue as "
            "youtube items (audio extracted via yt-dlp at play time). "
            "play_spotify / play_soundcloud / play_youtube play a URI, "
            "link, or search query immediately. mpv gives full transport + "
            "auto-advance; the queue persists across restarts."
        ),
        capability=Capability.FS_READ,
    )
    def player(action: str = "status", target: str = "", targets: str = "",
               title: str = "", index: int = -1, seconds: float = 0.0,
               level: int = 80, mode: str = "", to: int = 0,
               on: str = "toggle", name: str = "") -> dict[str, Any]:
        p = PlaybackEngine(
            context,
            spotify=getattr(context, "spotify_adapter", None),
            soundcloud=getattr(context, "soundcloud_adapter", None),
        )
        if action == "add":
            items = [t for t in (target, *filter(None, targets.split("|")))
                     if t.strip()]
            if not items:
                raise ToolError("player add needs target(s)")
            return p.add(*items, title=title)
        if action == "play":
            # index=-1 (the default) means "current position"; an explicit
            # 0 must play queue item 0, so the sentinel is < 0, not falsy
            return p.play(int(index) if int(index) >= 0 else None)
        if action == "play_spotify":
            if not target.strip():
                raise ToolError(
                    "player play_spotify needs a target: a Spotify URI, "
                    "an open.spotify.com link, or a search query")
            return p.play_spotify(target)
        if action == "play_soundcloud":
            if not target.strip():
                raise ToolError(
                    "player play_soundcloud needs a target: a "
                    "soundcloud.com link or a search query")
            return p.play_soundcloud(target)
        if action == "play_youtube":
            if not target.strip():
                raise ToolError(
                    "player play_youtube needs a target: a youtube.com "
                    "URL or a search query (needs yt-dlp)")
            return p.play_youtube(target)
        if action == "pause":
            return p.pause()
        if action == "resume":
            return p.resume()
        if action == "stop":
            return p.stop()
        if action == "seek":
            return p.seek(seconds)
        if action == "volume":
            return p.volume(level)
        if action == "next":
            return p.next()
        if action == "prev":
            return p.prev()
        if action == "queue":
            return {"queue": p.queue()}
        if action == "remove":
            return p.remove(index)
        if action == "move":
            return p.move(index, to)
        if action == "clear":
            return p.clear()
        if action == "shuffle":
            flag = {"on": True, "off": False}.get(on.strip().lower())
            return p.shuffle(flag)
        if action == "repeat":
            return p.repeat(mode.strip().lower() or None)
        if action == "now":
            return p.now()
        if action == "playlist_play":
            if not name.strip():
                raise ToolError("playlist_play needs a playlist name")
            return p.load_playlist(name)
        if action == "playlist_save":
            if not name.strip():
                raise ToolError("playlist_save needs a playlist name")
            return p.save_playlist(name)
        if action in ("status", ""):
            return p.status()
        raise ToolError(f"unknown player action {action!r}")
