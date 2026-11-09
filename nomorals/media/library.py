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

__all__ = ["MusicLibrary", "ClipIndex", "read_metadata", "register"]

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


def full_track_verdict(path: str,
                       expected_duration: float) -> tuple[str, str]:
    """Three-valued full-track verdict for a downloaded file.

    Returns ``(state, why)`` with state one of:

    * ``"verified"`` — an expected duration exists and the file's own
      probed duration corroborates it (not preview-length);
    * ``"rejected"`` — the file probed as preview-length against a
      known full-track expectation: it is a clip, not the track;
    * ``"unverified"`` — the check could not run: no expected
      duration to compare against, or the file's own duration is
      unprobeable.  Unverified is NOT a pass — callers must label it
      as unverified wherever they report the delivery, never present
      it as full-track proof (doc 7, §2).

    Never raises.
    """
    try:
        expected = float(expected_duration or 0)
    except (TypeError, ValueError):
        expected = 0.0
    if expected < 90:
        return ("unverified",
                "no full-track duration on record to check against")
    try:
        actual = float(read_metadata(path).get("duration") or 0)
    except Exception:  # noqa: BLE001 - probing is best-effort
        actual = 0.0
    if actual <= 0:
        return ("unverified",
                "the file's own duration could not be probed")
    if actual < min(45.0, expected * 0.5):
        return ("rejected",
                f"the file that came down is only {actual:.0f}s "
                f"long — a preview-length clip, not the full "
                f"{expected:.0f}s track")
    return ("verified",
            f"duration corroborated ({actual:.0f}s of "
            f"{expected:.0f}s expected)")


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

    # ── seam tables (None on the SQLite default) ─────────────────────────
    # Mongo P3: when boot attached a document store, the media tables
    # route through storage.rowstore (documents are carry-v1 rows); the
    # SQLite default runs the native statements below unchanged.
    def _t(self, table: str):
        from ..storage.rowstore import attached_rows
        return attached_rows(self.db, table)

    # ── playlists ─────────────────────────────────────────────────────────
    def create_playlist(self, name: str, description: str = ""
                        ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ToolError("playlist name can't be empty")
        if len(name) > 120:
            raise ToolError("playlist name is too long (max 120 chars)")
        playlists = self._t("media_playlists")
        if playlists is not None:
            if playlists.find_one({"name": name}) is not None:
                raise ToolError(f"playlist {name!r} already exists")
            playlists.put({
                "id": _slug_id("pl"), "name": name,
                "description": description or "", "created_at": time.time(),
                "updated_at": time.time()})
            return {"created": name, "description": description or ""}
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
        playlists = self._t("media_playlists")
        if playlists is not None:
            row = playlists.find_one({"name": (name or "").strip()})
        else:
            row = self.db.query_one(
                "SELECT id FROM media_playlists WHERE name=?",
                ((name or "").strip(),))
        if not row:
            raise ToolError(f"no playlist named {(name or '').strip()!r}")
        return str(row["id"])

    def _playlist_row(self, pid: str) -> dict[str, Any] | None:
        playlists = self._t("media_playlists")
        if playlists is not None:
            return playlists.get(pid)
        return self.db.query_one(
            "SELECT * FROM media_playlists WHERE id=?", (pid,))

    def _touch_playlist(self, pid: str) -> None:
        playlists = self._t("media_playlists")
        if playlists is not None:
            playlists.update(pid, {"updated_at": time.time()})
            return
        self.db.execute("UPDATE media_playlists SET updated_at=? WHERE id=?",
                        (time.time(), pid))

    def _put_item(self, row: dict[str, Any]) -> None:
        items = self._t("media_playlist_items")
        if items is not None:
            items.put(row)
            return
        self.db.execute(
            "INSERT INTO media_playlist_items (id, playlist_id, path, "
            "title, artist, album, duration, kind, position, added_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row["id"], row["playlist_id"], row["path"], row["title"],
             row["artist"], row["album"], row["duration"], row["kind"],
             row["position"], row["added_at"]))

    def _clear_items(self, pid: str) -> None:
        items = self._t("media_playlist_items")
        if items is not None:
            items.delete_where({"playlist_id": pid})
            return
        self.db.execute("DELETE FROM media_playlist_items WHERE "
                        "playlist_id=?", (pid,))

    def delete_playlist(self, name: str) -> dict[str, Any]:
        pid = self._playlist_id(name)
        items_t = self._t("media_playlist_items")
        if items_t is not None:
            removed = items_t.count({"playlist_id": pid})
            items_t.delete_where({"playlist_id": pid})
            playlists = self._t("media_playlists")
            assert playlists is not None  # same pickup rule
            playlists.delete(pid)
            return {"deleted": (name or "").strip(),
                    "tracks_removed": removed}
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
        playlists = self._t("media_playlists")
        if playlists is not None:
            clash = playlists.find_one({"name": new_name})
            if clash is not None and str(clash["id"]) != pid:
                raise ToolError(f"playlist {new_name!r} already exists")
            playlists.update(pid, {"name": new_name,
                                   "updated_at": time.time()})
            return {"renamed": (name or "").strip(), "to": new_name}
        try:
            self.db.execute("UPDATE media_playlists SET name=?, updated_at=? "
                            "WHERE id=?", (new_name, time.time(), pid))
        except Exception as exc:  # noqa: BLE001 - name clash → friendly error
            raise ToolError(f"playlist {new_name!r} already exists") from exc
        return {"renamed": (name or "").strip(), "to": new_name}

    def playlists(self) -> list[dict[str, Any]]:
        playlists = self._t("media_playlists")
        if playlists is not None:
            items = self._t("media_playlist_items")
            assert items is not None  # same pickup rule
            tracks: dict[str, int] = {}
            seconds: dict[str, float] = {}
            for it in items.all():
                key = str(it.get("playlist_id"))
                tracks[key] = tracks.get(key, 0) + 1
                seconds[key] = seconds.get(key, 0.0) + float(
                    it.get("duration") or 0)
            rows = sorted(playlists.all(),
                          key=lambda r: str(r.get("name") or ""))
            return [{"name": r["name"], "description": r["description"],
                     "tracks": tracks.get(str(r["id"]), 0),
                     "seconds": round(seconds.get(str(r["id"]), 0.0), 1),
                     "updated_at": r["updated_at"]} for r in rows]
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
        info = self._playlist_row(pid)
        assert info is not None  # _playlist_id resolved it
        rows = self._playlist_rows(pid)[:max(1, limit)]
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
            self._put_item({
                "id": _slug_id("pli"), "playlist_id": pid, "path": path,
                "title": meta["title"], "artist": meta["artist"],
                "album": meta["album"], "duration": meta["duration"],
                "kind": kind, "position": time.time() + len(added),
                "added_at": time.time()})
            added.append({"path": path, "kind": kind,
                          "title": meta["title"], "artist": meta["artist"],
                          "album": meta["album"],
                          "duration": meta["duration"]})
        self._touch_playlist(pid)
        return {"playlist": (name or "").strip(), "added": added,
                "tracks": len(self.playlist(name)["items"])}

    def _playlist_rows(self, pid: str) -> list[Any]:
        items = self._t("media_playlist_items")
        if items is not None:
            rows = items.find({"playlist_id": pid})
            rows.sort(key=lambda r: (r.get("position") or 0,
                                     r.get("added_at") or 0))
            return rows
        return self.db.query(
            "SELECT * FROM media_playlist_items WHERE playlist_id=? "
            "ORDER BY position, added_at", (pid,))

    def playlist_remove(self, name: str, index: int) -> dict[str, Any]:
        pid = self._playlist_id(name)
        rows = self._playlist_rows(pid)
        if not (0 <= int(index) < len(rows)):
            raise ToolError(
                f"no track {index} in playlist {(name or '').strip()!r}")
        items = self._t("media_playlist_items")
        if items is not None:
            items.delete(str(rows[int(index)]["id"]))
        else:
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
        items = self._t("media_playlist_items")
        for pos, rid2 in enumerate(ids):
            if items is not None:
                items.update(str(rid2), {"position": pos})
            else:
                self.db.execute("UPDATE media_playlist_items SET position=? "
                                "WHERE id=?", (pos, rid2))
        self._touch_playlist(pid)
        return {"moved": rows[i]["title"], "from": i, "to": j,
                "tracks": n}

    def playlist_clear(self, name: str) -> dict[str, Any]:
        pid = self._playlist_id(name)
        self._clear_items(pid)
        self._touch_playlist(pid)
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
        playlists = self._t("media_playlists")
        if playlists is not None:
            row = playlists.find_one({"name": clean})
        else:
            row = self.db.query_one("SELECT id FROM media_playlists WHERE "
                                    "name=?", (clean,))
        if row:
            pid = str(row["id"])
            self._clear_items(pid)
        else:
            pid = _slug_id("pl")
            if playlists is not None:
                playlists.put({"id": pid, "name": clean, "description": "",
                               "created_at": time.time(),
                               "updated_at": time.time()})
            else:
                self.db.execute(
                    "INSERT INTO media_playlists (id, name, description, "
                    "created_at, updated_at) VALUES (?,?, '', ?,?)",
                    (pid, clean, time.time(), time.time()))
        stamp = time.time()
        for i, it in enumerate(items):
            self._put_item({
                "id": _slug_id("pli"), "playlist_id": pid,
                "path": it.get("path", ""),
                "title": it.get("title") or os.path.basename(
                    it.get("path", "")),
                "artist": it.get("artist") or "",
                "album": it.get("album") or "",
                "duration": float(it.get("duration") or 0),
                "kind": it.get("kind") or "file",
                "position": stamp + i, "added_at": stamp})
        self._touch_playlist(pid)
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
        playlists = self._t("media_playlists")
        if playlists is not None:
            row = playlists.find_one({"name": (name or "").strip()})
        else:
            row = self.db.query_one("SELECT id FROM media_playlists WHERE "
                                    "name=?", ((name or "").strip(),))
        if row:
            pid = str(row["id"])
        else:
            pid = _slug_id("pl")
            if playlists is not None:
                playlists.put({"id": pid, "name": (name or "").strip(),
                               "description": "", "created_at": time.time(),
                               "updated_at": time.time()})
            else:
                self.db.execute(
                    "INSERT INTO media_playlists (id, name, description, "
                    "created_at, updated_at) VALUES (?,?, '', ?,?)",
                    (pid, (name or "").strip(), time.time(), time.time()))
        items_t = self._t("media_playlist_items")
        if items_t is not None:
            existing = items_t.find({"playlist_id": pid})
            top = max((float(r.get("position") or 0) for r in existing),
                      default=-1.0)
            pos = top + 1
        else:
            max_pos = self.db.query_one(
                "SELECT COALESCE(MAX(position), -1) AS m FROM "
                "media_playlist_items WHERE playlist_id=?", (pid,)) \
                or {"m": -1}
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
            self._put_item({
                "id": _slug_id("pli"), "playlist_id": pid, "path": path,
                "title": title or meta["title"], "artist": meta["artist"],
                "album": meta["album"], "duration": meta["duration"],
                "kind": kind, "position": pos, "added_at": time.time()})
            pos += 1
            added += 1
        self._touch_playlist(pid)
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
        hist = self._t("media_history")
        if hist is not None:
            rows = hist.all()
            last = max(rows, key=lambda r: r.get("played_at") or 0,
                       default=None)
        else:
            last = self.db.query_one(
                "SELECT path, played_at FROM media_history ORDER BY played_at "
                "DESC LIMIT 1")
        if last and last["path"] == path and \
                time.time() - float(last["played_at"] or 0) < _dedup_window:
            return {"recorded": False, "reason": "dedup"}
        title = title or read_metadata(path)["title"]
        if hist is not None:
            hist.put({"id": _slug_id("mh"), "path": path, "title": title,
                      "artist": artist or "", "album": album or "",
                      "duration": float(duration or 0),
                      "played_at": time.time(), "source": source or ""})
        else:
            self.db.execute(
                "INSERT INTO media_history (id, path, title, artist, album, "
                "duration, played_at, source) VALUES (?,?,?,?,?,?,?,?)",
                (_slug_id("mh"), path, title, artist or "", album or "",
                 float(duration or 0), time.time(), source or ""))
        return {"recorded": True, "title": title}

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        hist = self._t("media_history")
        if hist is not None:
            rows = hist.all()
            rows.sort(key=lambda r: r.get("played_at") or 0, reverse=True)
            rows = rows[:max(1, int(limit))]
        else:
            rows = self.db.query(
                "SELECT * FROM media_history ORDER BY played_at DESC "
                "LIMIT ?", (max(1, int(limit)),))
        return [{"path": r["path"], "title": r["title"], "artist": r["artist"],
                 "album": r["album"], "duration": r["duration"],
                 "played_at": r["played_at"], "source": r["source"]}
                for r in rows]

    def clear_history(self) -> dict[str, Any]:
        hist = self._t("media_history")
        if hist is not None:
            removed = hist.count()
            hist.delete_where()
            return {"cleared": True, "removed": removed}
        row = self.db.query_one("SELECT COUNT(*) AS n FROM media_history")
        self.db.execute("DELETE FROM media_history")
        return {"cleared": True, "removed": (row or {"n": 0})["n"]}

    def top_played(self, limit: int = 10) -> list[dict[str, Any]]:
        hist = self._t("media_history")
        if hist is not None:
            groups: dict[str, dict[str, Any]] = {}
            for r in hist.all():
                g = groups.setdefault(str(r.get("path")), {
                    "path": r.get("path"), "title": "", "artist": "",
                    "album": "", "plays": 0, "last": 0.0})
                g["plays"] += 1
                g["title"] = max(g["title"], str(r.get("title") or ""))
                g["artist"] = max(g["artist"], str(r.get("artist") or ""))
                g["album"] = max(g["album"], str(r.get("album") or ""))
                g["last"] = max(g["last"], float(r.get("played_at") or 0))
            rows = sorted(groups.values(),
                          key=lambda g: (g["plays"], g["last"]),
                          reverse=True)[:max(1, int(limit))]
            return [{"path": g["path"], "title": g["title"],
                     "artist": g["artist"], "album": g["album"],
                     "plays": g["plays"], "last_played": g["last"]}
                    for g in rows]
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
        favs = self._t("media_favorites")
        if favs is not None:
            favs.put({"path": path, "title": title, "artist": artist or "",
                      "album": album or "",
                      "duration": float(duration or 0),
                      "liked_at": time.time()})
            return {"liked": True, "title": title, "path": path}
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
        favs = self._t("media_favorites")
        if favs is not None:
            row = favs.get(path)
            if not row:
                raise ToolError(f"{path!r} is not in favorites")
            favs.delete(path)
            return {"unliked": True, "title": row["title"]}
        row = self.db.query_one("SELECT title FROM media_favorites WHERE "
                                "path=?", (path,))
        if not row:
            raise ToolError(f"{path!r} is not in favorites")
        self.db.execute("DELETE FROM media_favorites WHERE path=?", (path,))
        return {"unliked": True, "title": row["title"]}

    def is_liked(self, path: str) -> bool:
        favs = self._t("media_favorites")
        if favs is not None:
            return favs.get((path or "").strip()) is not None
        row = self.db.query_one("SELECT path FROM media_favorites WHERE "
                                "path=?", ((path or "").strip(),))
        return row is not None

    def favorites(self, limit: int = 200) -> list[dict[str, Any]]:
        favs = self._t("media_favorites")
        if favs is not None:
            rows = favs.all()
            rows.sort(key=lambda r: r.get("liked_at") or 0, reverse=True)
            rows = rows[:max(1, int(limit))]
        else:
            rows = self.db.query(
                "SELECT * FROM media_favorites ORDER BY liked_at DESC "
                "LIMIT ?", (max(1, int(limit)),))
        return [{"path": r["path"], "title": r["title"], "artist": r["artist"],
                 "album": r["album"], "duration": r["duration"],
                 "liked_at": r["liked_at"]} for r in rows]

    # ── search & stats ────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 50) -> dict[str, Any]:
        """Search favorites, history, playlists, and the live queue."""
        q = (query or "").strip()
        if not q:
            raise ToolError("search needs a query")
        lim = max(1, int(limit))
        favs_t = self._t("media_favorites")
        if favs_t is not None:
            needle = q.lower()

            def hit(r: dict[str, Any]) -> bool:
                return any(needle in str(r.get(f) or "").lower()
                           for f in ("title", "artist", "album", "path"))

            favs = [r for r in favs_t.all() if hit(r)][:lim]
            seen: set[tuple] = set()
            hist = []
            hist_t = self._t("media_history")
            assert hist_t is not None  # same pickup rule
            for r in hist_t.all():
                key = (r.get("path"), r.get("title"), r.get("artist"),
                       r.get("album"))
                if hit(r) and key not in seen:
                    seen.add(key)
                    hist.append(r)
            hist = hist[:lim]
            pls_t = self._t("media_playlists")
            assert pls_t is not None  # same pickup rule
            pls = [r for r in pls_t.all()
                   if needle in str(r.get("name") or "").lower()
                   or needle in str(r.get("description") or "").lower()][:lim]
            items_t = self._t("media_playlist_items")
            assert items_t is not None  # same pickup rule
            tracks = [r for r in items_t.all() if hit(r)][:lim]
            names = {str(r["id"]): str(r["name"]) for r in pls_t.all()}
            queue_t = self._t("media_queue")
            assert queue_t is not None  # same pickup rule
            queue_rows = [r for r in queue_t.all() if hit(r)][:lim]

            def slim_s(r: Any) -> dict[str, Any]:
                return {"path": r["path"], "title": r["title"],
                        "artist": r["artist"], "album": r["album"]}

            return {"query": q,
                    "favorites": [slim_s(r) for r in favs],
                    "history": [slim_s(r) for r in hist],
                    "playlists": [r["name"] for r in pls],
                    "playlist_tracks": [
                        {**slim_s(r),
                         "playlist": names.get(str(r["playlist_id"]), "")}
                        for r in tracks],
                    "queue": [slim_s(r) for r in queue_rows]}
        like = f"%{q}%"
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
        playlists = self._t("media_playlists")
        if playlists is not None:
            items = self._t("media_playlist_items")
            favs = self._t("media_favorites")
            hist = self._t("media_history")
            queue = self._t("media_queue")
            assert items is not None and favs is not None  # pickup rule
            assert hist is not None and queue is not None
            hist_rows = hist.all()
            return {"playlists": playlists.count(),
                    "playlist_tracks": items.count(),
                    "favorites": favs.count(),
                    "plays": len(hist_rows),
                    "unique_tracks_played": len(
                        {r.get("path") for r in hist_rows}),
                    "favorites_seconds": round(
                        sum(float(r.get("duration") or 0)
                            for r in favs.all()), 1),
                    "queue": queue.count()}

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


# ---------------------------------------------------------------------------
# Shot-level semantic search (EDIT_GEN_UPGRADE 2.6)
# ---------------------------------------------------------------------------
# Every clip is indexed ONCE — segments, characters, transcript, audio
# energy, motion, aesthetic — then queried in plain language:
# "find shot where he laughs", "widest shot last minute",
# "where music drops". Index entries are plain dicts (JSON-safe) so the
# index survives a restart via to_dict/from_dict. Feature extraction
# reuses scene_intel.score + highlight_model.features — no new
# extractor, no second index.

#: transcript tokens that count as laughter / music-drop cues.
_LAUGH_TOKENS = ("laugh", "laughs", "laughing", "haha", "lol", "giggle",
                 "chuckle")
_MUSIC_DROP_TOKENS = ("music drops", "drop", "beat drops", "bass drops")


class ClipIndex:
    """In-memory shot index with semantic query.

    Entry shape (all fields optional except ``clip_id``/``start``/``end``)::

        {"clip_id", "start", "end", "characters": [char_id, ...],
         "transcript": "he laughs at the joke",
         "audio_energy": [0.1, 0.9, ...] | float (mean),
         "motion": float, "aesthetic": float,
         "framing_width": float (0..1, subject bbox width — wide shot
         when small), "tags": ["laughing", ...], "duration": float}
    """

    def __init__(self, entries: list[dict[str, Any]] | None = None):
        self.entries: list[dict[str, Any]] = [dict(e) for e in (entries or [])]

    def __len__(self) -> int:
        return len(self.entries)

    def index_clip(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Index one clip (one call per clip, not per query)."""
        entry = dict(entry)
        if not entry.get("clip_id"):
            raise ToolError("index_clip needs clip_id")
        entry["start"] = float(entry.get("start", 0.0))
        entry["end"] = float(entry.get("end", entry["start"]))
        if entry["end"] < entry["start"]:
            raise ToolError("index_clip needs end >= start")
        entry.setdefault("duration", round(entry["end"] - entry["start"], 3))
        self.entries.append(entry)
        return entry

    def index_file(self, path: str, *, segments: list[tuple[float, float]] | None = None,
                   transcript: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Index a media file once via scene_intel + highlight features.

        ``segments`` defaults to the file's scene segmentation;
        transcript words are binned into segments. Feature extraction
        failures leave that signal at 0.0 (honest) — the entry still
        indexes with its transcript + characters.
        """
        from .scene_intel.segment import segment_movie
        from .highlight_model.features import extract_features
        indexed: list[dict[str, Any]] = []
        try:
            bounds = segments or [(s.start, s.end) for s in
                                  segment_movie(path).segments
                                  if s.kind == "scene"]
        except Exception:  # noqa: BLE001 - segmentation is best-effort
            _log.debug("library: segmentation failed for %s", path,
                       exc_info=True)
            bounds = [(0.0, 0.0)]
        try:
            feats = extract_features(path)
        except Exception:  # noqa: BLE001
            _log.debug("library: feature extraction failed for %s",
                       path, exc_info=True)
            feats = []
        for i, (s, e) in enumerate(bounds):
            words = [w for w in (transcript or [])
                     if s <= float(w.get("start", 0)) < e]
            text = " ".join(str(w.get("text", "")) for w in words)
            feat = feats[int(s)] if feats and int(s) < len(feats) else None
            entry = {"clip_id": str(path), "start": round(float(s), 3),
                     "end": round(float(e), 3),
                     "transcript": text,
                     "audio_energy": float(feat[0]) if feat else 0.0,
                     "motion": float(feat[1]) if feat else 0.0,
                     "aesthetic": float(sum(feat[:3]) / 3.0) if feat else 0.0,
                     "segment_index": i}
            self.index_clip(entry)
            indexed.append(entry)
        return indexed

    # -- query -----------------------------------------------------------
    def query(self, text: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Semantic shot query. Deterministic token + signal matching."""
        q = (text or "").strip().lower()
        if not q:
            raise ToolError("ClipIndex.query needs a query")
        pool = list(self.entries)
        # "widest shot [last minute]" — smallest subject framing, recent window
        if "widest" in q:
            window = 60.0 if "last minute" in q else None
            if window is not None and pool:
                latest = max(e["end"] for e in pool)
                pool = [e for e in pool if e["end"] >= latest - window]
            pool.sort(key=lambda e: float(e.get("framing_width", 1.0)))
            return pool[:limit]
        # "where music drops" — largest audio-energy fall between neighbours
        if any(t in q for t in _MUSIC_DROP_TOKENS):
            drops: list[tuple[float, dict[str, Any]]] = []
            ordered = sorted(pool, key=lambda e: (str(e.get("clip_id", "")),
                                                  float(e.get("start", 0))))
            prev: dict[str, Any] | None = None
            for e in ordered:
                cur = self._energy(e)
                if prev is not None and str(prev.get("clip_id")) == str(e.get("clip_id")):
                    drop = self._energy(prev) - cur
                    if drop > 0:
                        drops.append((drop, e))
                prev = e
            drops.sort(key=lambda d: -d[0])
            return [e for _d, e in drops[:limit]]
        # laugh / character / transcript / tag text match
        tokens = [t.strip(".,!?\"'") for t in q.split() if len(t) > 2]
        if "laugh" in q or "laughs" in q or "laughing" in q:
            tokens += list(_LAUGH_TOKENS)
        scored: list[tuple[float, dict[str, Any]]] = []
        for e in pool:
            hay = " ".join([
                str(e.get("transcript", "")),
                " ".join(str(c) for c in e.get("characters", [])),
                " ".join(str(t) for t in e.get("tags", [])),
                str(e.get("clip_id", "")),
            ]).lower()
            hits = sum(1 for t in tokens if t in hay)
            if hits:
                scored.append((hits + float(e.get("aesthetic", 0.0)) * 0.1, e))
        scored.sort(key=lambda s: -s[0])
        return [e for _s, e in scored[:limit]]

    @staticmethod
    def _energy(entry: dict[str, Any]) -> float:
        val = entry.get("audio_energy", 0.0)
        if isinstance(val, (list, tuple)):
            return float(sum(val) / len(val)) if val else 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def search_shots(self, text: str, **kw: Any) -> list[dict[str, Any]]:
        """Alias for :meth:`query` (the shot-search front door)."""
        return self.query(text, **kw)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipIndex":
        return cls(entries=list(data.get("entries", [])))


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
