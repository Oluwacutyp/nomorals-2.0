"""MusicLibrary — playlists, play history, favorites, stats, search.

The durable side of the music player: named playlists, a recently-played
history with play counts, liked tracks, and search across all of it.
Storage is the app database (migration 67); audio metadata comes from a
best-effort chain so the library never depends on a single reader:

* **mutagen** — richest tag reading (MP3/FLAC/OGG/M4A/…), when installed.
* **ffprobe** — stream/container metadata + duration, when ffmpeg is around.
* **filename fallback** — always works: title from the file name.

A file that defeats every reader still enters the library with its file
name as the title — metadata extraction never blocks a library write.

    from nomorals.media.library import MusicLibrary, read_metadata
    lib = MusicLibrary(context)
    lib.create_playlist("gym")
    lib.playlist_add("gym", ["workspace/tunes/a.mp3", "workspace/tunes/b.mp3"])
    lib.record_played("workspace/tunes/a.mp3", title="A")
    lib.history(10)          # newest first
    lib.like("workspace/tunes/a.mp3", title="A")
    lib.top_played(5)        # most-played tracks
    lib.search("amapiano")   # across favorites, history, playlists, queue

Registered as the ``music_library`` tool.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

from ..core.errors import NoMoralsError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["MusicLibrary", "read_metadata", "register"]

_META_KEYS = ("title", "artist", "album", "genre", "duration")


def _blank_meta() -> dict[str, Any]:
    return {"title": "", "artist": "", "album": "", "genre": "",
            "duration": 0.0}


def _is_url(path: str) -> bool:
    return path.lower().startswith(("http://", "https://"))


def _metadata_mutagen(path: str) -> dict[str, Any] | None:
    """Tag read via mutagen.  None when mutagen is missing or gives up."""
    try:
        import mutagen  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        audio = mutagen.File(path, easy=True)
    except Exception:  # noqa: BLE001 - any corrupt/odd file: fall through
        return None
    if audio is None:
        return None

    def tag(*keys: str) -> str:
        for k in keys:
            try:
                vals = audio.get(k)
            except Exception:  # noqa: BLE001 - odd tag backends
                continue
            if vals:
                text = str(vals[0]).strip()
                if text:
                    return text
        return ""

    try:
        length = float(getattr(audio.info, "length", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        length = 0.0
    return {"title": tag("title"),
            "artist": tag("artist", "albumartist", "performer"),
            "album": tag("album"),
            "genre": tag("genre"),
            "duration": length}


def _metadata_ffprobe(path: str) -> dict[str, Any] | None:
    """Tag read via ffprobe.  None when ffprobe is missing or fails."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError:
        return None
    fmt = data.get("format") or {}
    raw_tags = fmt.get("tags") or {}
    tags = {str(k).lower(): v for k, v in raw_tags.items()}

    def tag(*keys: str) -> str:
        for k in keys:
            v = tags.get(k)
            if v:
                text = str(v).strip()
                if text:
                    return text
        return ""

    try:
        duration = float(fmt.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        # fall back to the longest stream duration
        for st in data.get("streams") or []:
            try:
                duration = max(duration, float(st.get("duration") or 0))
            except (TypeError, ValueError):
                continue
    return {"title": tag("title"),
            "artist": tag("artist", "album_artist", "performer"),
            "album": tag("album"),
            "genre": tag("genre"),
            "duration": duration}


def read_metadata(path: str) -> dict[str, Any]:
    """Best-effort metadata for an audio file or URL.  Never raises.

    Tries mutagen, then ffprobe, then falls back to the file name as the
    title.  URLs are not probed (no streaming reads) — the title comes
    from the URL's last path segment.
    """
    path = (path or "").strip()
    meta = _blank_meta()
    if not path:
        return meta
    if _is_url(path):
        tail = urlsplit(path).path.rsplit("/", 1)[-1] or path
        meta["title"] = tail
        return meta
    for reader in (_metadata_mutagen, _metadata_ffprobe):
        try:
            got = reader(path)
        except Exception:  # noqa: BLE001 - metadata is best-effort by design
            got = None
        if got:
            for k in _META_KEYS:
                if got.get(k):
                    meta[k] = got[k]
            if meta["title"]:
                return meta
    # filename fallback always works
    base = os.path.basename(path)
    meta["title"] = os.path.splitext(base)[0] or base or path
    return meta


def _slug_id(prefix: str) -> str:
    return f"{prefix}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"


class MusicLibrary:
    """Named playlists, play history, favorites — all persisted in the DB."""

    role = "music_library"

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is None:
            raise RuntimeError("MusicLibrary needs a context with a database")

    # ── playlists ─────────────────────────────────────────────────────────
    def create_playlist(self, name: str, description: str = ""
                        ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ToolError("playlist name can't be empty")
        if len(name) > 120:
            raise ToolError("playlist name is too long (max 120 chars)")
        try:
            self.db.execute(
                "INSERT INTO media_playlists (id, name, description, "
                "created_at, updated_at) VALUES (?,?,?,?,?)",
                (_slug_id("pl"), name, description or "", time.time(),
                 time.time()))
        except Exception as exc:  # noqa: BLE001 - duplicate → friendly error
            raise ToolError(f"playlist {name!r} already exists") from exc
        return {"created": name, "description": description or ""}

    def _playlist_id(self, name: str) -> str:
        row = self.db.query_one("SELECT id FROM media_playlists WHERE name=?",
                                ((name or "").strip(),))
        if not row:
            raise ToolError(f"no playlist named {(name or '').strip()!r}")
        return str(row["id"])

    def delete_playlist(self, name: str) -> dict[str, Any]:
        pid = self._playlist_id(name)
        items = self.db.query_one(
            "SELECT COUNT(*) AS n FROM media_playlist_items WHERE "
            "playlist_id=?", (pid,)) or {"n": 0}
        self.db.execute("DELETE FROM media_playlist_items WHERE "
                        "playlist_id=?", (pid,))
        self.db.execute("DELETE FROM media_playlists WHERE id=?", (pid,))
        return {"deleted": (name or "").strip(), "tracks_removed": items["n"]}

    def rename_playlist(self, name: str, new_name: str) -> dict[str, Any]:
        pid = self._playlist_id(name)
        new_name = (new_name or "").strip()
        if not new_name:
            raise ToolError("new playlist name can't be empty")
        if len(new_name) > 120:
            raise ToolError("playlist name is too long (max 120 chars)")
        try:
            self.db.execute("UPDATE media_playlists SET name=?, updated_at=? "
                            "WHERE id=?", (new_name, time.time(), pid))
        except Exception as exc:  # noqa: BLE001 - name clash → friendly error
            raise ToolError(f"playlist {new_name!r} already exists") from exc
        return {"renamed": (name or "").strip(), "to": new_name}

    def playlists(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT p.id, p.name, p.description, p.created_at, p.updated_at, "
            "COUNT(i.id) AS tracks, "
            "COALESCE(SUM(i.duration), 0) AS seconds "
            "FROM media_playlists p LEFT JOIN media_playlist_items i "
            "ON i.playlist_id = p.id GROUP BY p.id ORDER BY p.name")
        return [{"name": r["name"], "description": r["description"],
                 "tracks": r["tracks"],
                 "seconds": round(float(r["seconds"] or 0), 1),
                 "updated_at": r["updated_at"]} for r in rows]

    def playlist(self, name: str, limit: int = 500) -> dict[str, Any]:
        pid = self._playlist_id(name)
        info = self.db.query_one("SELECT name, description FROM "
                                 "media_playlists WHERE id=?", (pid,))
        rows = self.db.query(
            "SELECT * FROM media_playlist_items WHERE playlist_id=? "
            "ORDER BY position, added_at LIMIT ?", (pid, max(1, limit)))
        items = [{"index": i, "path": r["path"], "title": r["title"],
                  "artist": r["artist"], "album": r["album"],
                  "duration": r["duration"], "kind": r["kind"] or "file"}
                 for i, r in enumerate(rows)]
        return {"name": info["name"], "description": info["description"],
                "tracks": len(items), "items": items}

    def playlist_add(self, name: str,
                     *targets: str) -> dict[str, Any]:
        """Add files/URLs to a playlist.  Files may be workspace-relative."""
        pid = self._playlist_id(name)
        from ..tools.filesystem import safe_path

        added = []
        for t in targets:
            t = (t or "").strip()
            if not t:
                continue
            kind = "url" if _is_url(t) else "file"
            if kind == "file":
                try:
                    path = str(safe_path(self.context, t, must_exist=True))
                except NoMoralsError as exc:
                    # NotFound (missing file) / ValidationError (escapes
                    # the workspace) → the library's own error contract
                    raise ToolError(f"can't add {t!r}: {exc}") from exc
            else:
                path = t
            meta = read_metadata(path)
            self.db.execute(
                "INSERT INTO media_playlist_items (id, playlist_id, path, "
                "title, artist, album, duration, kind, position, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_slug_id("pli"), pid, path, meta["title"],
                 meta["artist"], meta["album"], meta["duration"], kind,
                 time.time() + len(added), time.time()))
            added.append({"path": path, "kind": kind,
                          "title": meta["title"], "artist": meta["artist"],
                          "album": meta["album"],
                          "duration": meta["duration"]})
        self.db.execute("UPDATE media_playlists SET updated_at=? WHERE id=?",
                        (time.time(), pid))
        return {"playlist": (name or "").strip(), "added": added,
                "tracks": len(self.playlist(name)["items"])}

    def _playlist_rows(self, pid: str) -> list[Any]:
        return self.db.query(
            "SELECT * FROM media_playlist_items WHERE playlist_id=? "
            "ORDER BY position, added_at", (pid,))

    def playlist_remove(self, name: str, index: int) -> dict[str, Any]:
        pid = self._playlist_id(name)
        rows = self._playlist_rows(pid)
        if not (0 <= int(index) < len(rows)):
            raise ToolError(
                f"no track {index} in playlist {(name or '').strip()!r}")
        self.db.execute("DELETE FROM media_playlist_items WHERE id=?",
                        (rows[int(index)]["id"],))
        return {"removed": rows[int(index)]["title"],
                "tracks": len(rows) - 1}

    def playlist_move(self, name: str, index: int, to: int) -> dict[str, Any]:
        pid = self._playlist_id(name)
        rows = self._playlist_rows(pid)
        n = len(rows)
        if not n:
            raise ToolError(f"playlist {(name or '').strip()!r} is empty")
        i, j = int(index), int(to)
        if not (0 <= i < n):
            raise ToolError(
                f"no track {index} in playlist {(name or '').strip()!r}")
        j = max(0, min(j, n - 1))
        ids = [r["id"] for r in rows]
        rid = ids.pop(i)
        ids.insert(j, rid)
        for pos, rid2 in enumerate(ids):
            self.db.execute("UPDATE media_playlist_items SET position=? "
                            "WHERE id=?", (pos, rid2))
        self.db.execute("UPDATE media_playlists SET updated_at=? WHERE id=?",
                        (time.time(), pid))
        return {"moved": rows[i]["title"], "from": i, "to": j,
                "tracks": n}

    def playlist_clear(self, name: str) -> dict[str, Any]:
        pid = self._playlist_id(name)
        self.db.execute("DELETE FROM media_playlist_items WHERE playlist_id=?",
                        (pid,))
        self.db.execute("UPDATE media_playlists SET updated_at=? WHERE id=?",
                        (time.time(), pid))
        return {"cleared": (name or "").strip()}

    # ── queue ↔ playlist / M3U ──────────────────────────────────────────
    def save_queue_as_playlist(self, name: str,
                               items: list[dict[str, Any]]) -> dict[str, Any]:
        """Persist the given queue items as a playlist named ``name``.

        Creates the playlist when missing; replaces its contents when it
        exists — "save" semantics, like a desktop player. ``items`` are
        queue dicts (path/title/artist/album/duration/kind).
        """
        clean = (name or "").strip()
        if not clean:
            raise ToolError("playlist name can't be empty")
        if not items:
            raise ToolError("the queue is empty — nothing to save")
        row = self.db.query_one("SELECT id FROM media_playlists WHERE "
                                "name=?", (clean,))
        if row:
            pid = str(row["id"])
            self.db.execute("DELETE FROM media_playlist_items WHERE "
                            "playlist_id=?", (pid,))
        else:
            pid = _slug_id("pl")
            self.db.execute(
                "INSERT INTO media_playlists (id, name, description, "
                "created_at, updated_at) VALUES (?,?, '', ?,?)",
                (pid, clean, time.time(), time.time()))
        stamp = time.time()
        for i, it in enumerate(items):
            self.db.execute(
                "INSERT INTO media_playlist_items (id, playlist_id, path, "
                "title, artist, album, duration, kind, position, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_slug_id("pli"), pid, it.get("path", ""),
                 it.get("title") or os.path.basename(it.get("path", "")),
                 it.get("artist") or "", it.get("album") or "",
                 float(it.get("duration") or 0),
                 it.get("kind") or "file", stamp + i, stamp))
        self.db.execute("UPDATE media_playlists SET updated_at=? WHERE id=?",
                        (time.time(), pid))
        return {"saved": clean, "tracks": len(items)}

    def export_m3u(self, name: str, dest: str) -> dict[str, Any]:
        """Write a playlist as an M3U file.  Paths stay workspace-relative
        when possible so the file survives a workspace move."""
        from ..tools.filesystem import safe_path

        pid = self._playlist_id(name)
        rows = self._playlist_rows(pid)
        if not rows:
            raise ToolError(f"playlist {(name or '').strip()!r} is empty")
        try:
            dest_p = safe_path(self.context, (dest or "").strip() or
                               f"{(name or '').strip()}.m3u")
        except NoMoralsError as exc:
            raise ToolError(f"can't export to {(dest or '').strip()!r}: "
                            f"{exc}") from exc
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        ws_root = str(getattr(getattr(self.context, "settings", None),
                              "workspace_dir", "") or "")
        lines = ["#EXTM3U"]
        for r in rows:
            path = str(r["path"])
            if ws_root and os.path.isabs(path):
                try:
                    rel = os.path.relpath(path, ws_root)
                except ValueError:
                    rel = path  # different drive — keep absolute
                if not rel.startswith(".."):
                    path = rel
            secs = int(float(r["duration"] or 0))
            label = f"{r['artist']} - {r['title']}" if r["artist"] \
                else r["title"]
            lines.append(f"#EXTINF:{secs},{label}")
            lines.append(path)
        dest_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"exported": (name or "").strip(), "file": str(dest_p),
                "tracks": len(rows)}

    def import_m3u(self, name: str, src: str) -> dict[str, Any]:
        """Read an M3U file into a playlist (created when missing).

        Local paths may be workspace-relative or absolute; anything else
        (http(s) lines) is kept as a URL. Titles come from #EXTINF, then
        the file name. Existing tracks are kept — import appends.
        """
        from ..tools.filesystem import safe_path

        try:
            src_p = safe_path(self.context, (src or "").strip(),
                              must_exist=True)
        except NoMoralsError as exc:
            raise ToolError(f"can't import from {(src or '').strip()!r}: "
                            f"{exc}") from exc
        text = src_p.read_text(encoding="utf-8", errors="replace")
        pending_title = ""
        targets: list[tuple[str, str]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("#EXTINF:"):
                payload = line.split(":", 1)[1] if ":" in line else ""
                pending_title = payload.split(",", 1)[1].strip() \
                    if "," in payload else ""
                continue
            if line.startswith("#"):
                continue
            targets.append((line, pending_title))
            pending_title = ""
        if not targets:
            raise ToolError(f"{src_p} has no playable entries")
        row = self.db.query_one("SELECT id FROM media_playlists WHERE "
                                "name=?", ((name or "").strip(),))
        if row:
            pid = str(row["id"])
        else:
            pid = _slug_id("pl")
            self.db.execute(
                "INSERT INTO media_playlists (id, name, description, "
                "created_at, updated_at) VALUES (?,?, '', ?,?)",
                (pid, (name or "").strip(), time.time(), time.time()))
        max_pos = self.db.query_one(
            "SELECT COALESCE(MAX(position), -1) AS m FROM "
            "media_playlist_items WHERE playlist_id=?", (pid,)) or {"m": -1}
        pos = float(max_pos["m"]) + 1
        added = 0
        for raw_path, title in targets:
            kind = "url" if _is_url(raw_path) else "file"
            if kind == "file":
                try:
                    path = str(safe_path(self.context, raw_path,
                                         must_exist=True))
                except NoMoralsError:
                    continue  # stale entry — skip, don't kill the import
            else:
                path = raw_path
            meta = read_metadata(path)
            self.db.execute(
                "INSERT INTO media_playlist_items (id, playlist_id, path, "
                "title, artist, album, duration, kind, position, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_slug_id("pli"), pid, path,
                 title or meta["title"], meta["artist"], meta["album"],
                 meta["duration"], kind, pos, time.time()))
            pos += 1
            added += 1
        self.db.execute("UPDATE media_playlists SET updated_at=? WHERE id=?",
                        (time.time(), pid))
        return {"imported": (name or "").strip(), "file": str(src_p),
                "added": added, "skipped": len(targets) - added}

    # ── history ───────────────────────────────────────────────────────────
    def record_played(self, path: str, *, title: str = "", artist: str = "",
                      album: str = "", duration: float = 0.0,
                      source: str = "", _dedup_window: float = 10.0
                      ) -> dict[str, Any]:
        """Log a play.  The same track within the dedup window is skipped so
        play → status-sync races don't double-count."""
        path = (path or "").strip()
        if not path:
            raise ToolError("record_played needs a path")
        last = self.db.query_one(
            "SELECT path, played_at FROM media_history ORDER BY played_at "
            "DESC LIMIT 1")
        if last and last["path"] == path and \
                time.time() - float(last["played_at"] or 0) < _dedup_window:
            return {"recorded": False, "reason": "dedup"}
        title = title or read_metadata(path)["title"]
        self.db.execute(
            "INSERT INTO media_history (id, path, title, artist, album, "
            "duration, played_at, source) VALUES (?,?,?,?,?,?,?,?)",
            (_slug_id("mh"), path, title, artist or "", album or "",
             float(duration or 0), time.time(), source or ""))
        return {"recorded": True, "title": title}

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM media_history ORDER BY played_at DESC LIMIT ?",
            (max(1, int(limit)),))
        return [{"path": r["path"], "title": r["title"], "artist": r["artist"],
                 "album": r["album"], "duration": r["duration"],
                 "played_at": r["played_at"], "source": r["source"]}
                for r in rows]

    def clear_history(self) -> dict[str, Any]:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM media_history")
        self.db.execute("DELETE FROM media_history")
        return {"cleared": True, "removed": (row or {"n": 0})["n"]}

    def top_played(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT path, MAX(title) AS title, MAX(artist) AS artist, "
            "MAX(album) AS album, COUNT(*) AS plays, MAX(played_at) AS last "
            "FROM media_history GROUP BY path ORDER BY plays DESC, last DESC "
            "LIMIT ?", (max(1, int(limit)),))
        return [{"path": r["path"], "title": r["title"], "artist": r["artist"],
                 "album": r["album"], "plays": r["plays"],
                 "last_played": r["last"]} for r in rows]

    # ── favorites ─────────────────────────────────────────────────────────
    def like(self, path: str, *, title: str = "", artist: str = "",
             album: str = "", duration: float = 0.0) -> dict[str, Any]:
        path = (path or "").strip()
        if not path:
            raise ToolError("like needs a path")
        if not title:
            meta = read_metadata(path)
            title, artist = meta["title"], meta["artist"] or artist
            album, duration = meta["album"] or album, meta["duration"]
        self.db.execute(
            "INSERT INTO media_favorites (path, title, artist, album, "
            "duration, liked_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET title=excluded.title, "
            "artist=excluded.artist, album=excluded.album, "
            "duration=excluded.duration, liked_at=excluded.liked_at",
            (path, title, artist or "", album or "", float(duration or 0),
             time.time()))
        return {"liked": True, "title": title, "path": path}

    def unlike(self, path: str) -> dict[str, Any]:
        path = (path or "").strip()
        row = self.db.query_one("SELECT title FROM media_favorites WHERE "
                                "path=?", (path,))
        if not row:
            raise ToolError(f"{path!r} is not in favorites")
        self.db.execute("DELETE FROM media_favorites WHERE path=?", (path,))
        return {"unliked": True, "title": row["title"]}

    def is_liked(self, path: str) -> bool:
        row = self.db.query_one("SELECT path FROM media_favorites WHERE "
                                "path=?", ((path or "").strip(),))
        return row is not None

    def favorites(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM media_favorites ORDER BY liked_at DESC LIMIT ?",
            (max(1, int(limit)),))
        return [{"path": r["path"], "title": r["title"], "artist": r["artist"],
                 "album": r["album"], "duration": r["duration"],
                 "liked_at": r["liked_at"]} for r in rows]

    # ── search & stats ────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 50) -> dict[str, Any]:
        """Search favorites, history, playlists, and the live queue."""
        q = (query or "").strip()
        if not q:
            raise ToolError("search needs a query")
        like = f"%{q}%"
        lim = max(1, int(limit))
        favs = self.db.query(
            "SELECT path, title, artist, album FROM media_favorites "
            "WHERE title LIKE ? OR artist LIKE ? OR album LIKE ? OR path "
            "LIKE ? LIMIT ?", (like, like, like, like, lim))
        hist = self.db.query(
            "SELECT DISTINCT path, title, artist, album FROM media_history "
            "WHERE title LIKE ? OR artist LIKE ? OR album LIKE ? OR path "
            "LIKE ? LIMIT ?", (like, like, like, like, lim))
        pls = self.db.query(
            "SELECT name FROM media_playlists WHERE name LIKE ? OR "
            "description LIKE ? LIMIT ?", (like, like, lim))
        tracks = self.db.query(
            "SELECT playlist_id, path, title, artist, album FROM "
            "media_playlist_items WHERE title LIKE ? OR artist LIKE ? OR "
            "album LIKE ? OR path LIKE ? LIMIT ?",
            (like, like, like, like, lim))
        names: dict[str, str] = {}
        for r in self.db.query("SELECT id, name FROM media_playlists"):
            names[str(r["id"])] = str(r["name"])
        queue_rows = self.db.query(
            "SELECT path, title, artist, album FROM media_queue WHERE title "
            "LIKE ? OR artist LIKE ? OR album LIKE ? OR path LIKE ? LIMIT ?",
            (like, like, like, like, lim))

        def slim(r: Any) -> dict[str, Any]:
            return {"path": r["path"], "title": r["title"],
                    "artist": r["artist"], "album": r["album"]}

        return {"query": q,
                "favorites": [slim(r) for r in favs],
                "history": [slim(r) for r in hist],
                "playlists": [r["name"] for r in pls],
                "playlist_tracks": [
                    {**slim(r), "playlist": names.get(str(r["playlist_id"]),
                                                     "")} for r in tracks],
                "queue": [slim(r) for r in queue_rows]}

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            row = self.db.query_one(f"SELECT COUNT(*) AS n FROM {table}")
            return int((row or {"n": 0})["n"])

        unique = self.db.query_one(
            "SELECT COUNT(DISTINCT path) AS n FROM media_history") or {"n": 0}
        secs = self.db.query_one(
            "SELECT COALESCE(SUM(duration),0) AS s FROM media_favorites"
        ) or {"s": 0}
        return {"playlists": count("media_playlists"),
                "playlist_tracks": count("media_playlist_items"),
                "favorites": count("media_favorites"),
                "plays": count("media_history"),
                "unique_tracks_played": int(unique["n"]),
                "favorites_seconds": round(float(secs["s"] or 0), 1),
                "queue": count("media_queue")}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "music_library",
        description=(
            "Music library: named playlists (create/delete/rename/list/show/"
            "add/remove/move/clear), play history (history/top/clear), "
            "favorites (like/unlike/liked), search across everything, and "
            "library stats. action=playlist_create | playlist_delete | "
            "playlist_rename | playlists | playlist | playlist_add | "
            "playlist_remove | playlist_move | playlist_clear | "
            "playlist_export (name, path=<m3u file>) | "
            "playlist_import (name, path=<m3u file>) | history | "
            "history_clear | top | like | unlike | liked | search | stats. "
            "like/unlike with an empty path act on the current track."
        ),
        capability=Capability.FS_READ,
    )
    def music_library(action: str = "stats", name: str = "",
                      new_name: str = "", targets: str = "", path: str = "",
                      title: str = "", query: str = "", index: int = -1,
                      to: int = 0, limit: int = 50,
                      description: str = "") -> dict[str, Any]:
        lib = MusicLibrary(context)
        if action == "playlist_create":
            return lib.create_playlist(name, description)
        if action == "playlist_delete":
            return lib.delete_playlist(name)
        if action == "playlist_rename":
            return lib.rename_playlist(name, new_name)
        if action == "playlists":
            return {"playlists": lib.playlists()}
        if action == "playlist":
            return lib.playlist(name, limit)
        if action == "playlist_add":
            items = [t for t in filter(None, targets.split("|")) if t.strip()]
            if not items:
                raise ToolError("playlist_add needs target(s)")
            return lib.playlist_add(name, *items)
        if action == "playlist_remove":
            return lib.playlist_remove(name, index)
        if action == "playlist_move":
            return lib.playlist_move(name, index, to)
        if action == "playlist_clear":
            return lib.playlist_clear(name)
        if action == "playlist_export":
            if not path.strip():
                raise ToolError("playlist_export needs a file — "
                                "nm music playlist-export gym gym.m3u")
            return lib.export_m3u(name, path)
        if action == "playlist_import":
            if not path.strip():
                raise ToolError("playlist_import needs an M3U file — "
                                "nm music playlist-import gym gym.m3u")
            return lib.import_m3u(name, path)
        if action == "history":
            return {"history": lib.history(limit)}
        if action == "history_clear":
            return lib.clear_history()
        if action == "top":
            return {"top": lib.top_played(limit)}
        if action == "like":
            if not path.strip():
                path, title = _current_track(context)
            return lib.like(path, title=title)
        if action == "unlike":
            if not path.strip():
                path, _ = _current_track(context)
            return lib.unlike(path)
        if action == "liked":
            return {"favorites": lib.favorites(limit)}
        if action == "search":
            return lib.search(query, limit)
        if action == "stats":
            return lib.stats()
        raise ToolError(f"unknown music_library action {action!r}")


def _current_track(context: Any) -> tuple[str, str]:
    """Current player track (path, title) — like/unlike with no path."""
    from .playback import PlaybackEngine

    eng = PlaybackEngine(context)
    q = eng.queue()
    if not q:
        raise ToolError("nothing is playing — the queue is empty")
    pos = eng._resolve_position()
    item = q[pos]
    return item["path"], item.get("title", "")
