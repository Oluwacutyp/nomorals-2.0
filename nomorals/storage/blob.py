"""Content-addressed blob store.

Binary data (downloaded media, model files, document originals) never goes in the
database. It goes in a directory tree named by SHA-256, and the database holds a
pointer plus metadata. Three properties fall out:

* **Deduplication** — the same bytes downloaded twice cost one copy.
* **Integrity** — the name *is* the checksum, so corruption is detectable.
* **Cheap backup** — blobs are immutable, so a backup only needs to copy new ones.
"""

from __future__ import annotations

import gzip
import hashlib
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator

from ..core.errors import NotFound, StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, styled_table
from .db import Database

__all__ = ["BlobInfo", "BlobStore"]

_log = get_logger(__name__)

CHUNK = 1024 * 1024


@dataclass
class BlobInfo:
    sha256: str
    size: int
    mime: str
    compressed: bool
    stored: int
    refcount: int
    created_at: float

    @property
    def ratio(self) -> float:
        return self.stored / self.size if self.size else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size": self.size,
            "mime": self.mime,
            "compressed": self.compressed,
            "stored": self.stored,
            "refcount": self.refcount,
            "created_at": self.created_at,
            "ratio": round(self.ratio, 4),
        }


class BlobStore:
    """SHA-256 addressed object storage with optional transparent gzip."""

    TABLE = "blobs"

    def __init__(
        self,
        db: Database,
        root: str | os.PathLike[str],
        *,
        compress_above: int = 4096,
        compressible: tuple[str, ...] = (
            "text/",
            "application/json",
            "application/javascript",
            "application/xml",
            "image/svg",
        ),
    ) -> None:
        self.db = db
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.compress_above = compress_above
        self.compressible = compressible
        self.stats = {"puts": 0, "dedup_hits": 0, "gets": 0, "bytes_written": 0}

    # ── paths ────────────────────────────────────────────────────────────────
    def path_for(self, sha256: str, compressed: bool = False) -> Path:
        """Two-level fan-out keeps any single directory small."""
        suffix = ".gz" if compressed else ""
        return self.root / sha256[:2] / sha256[2:4] / f"{sha256}{suffix}"

    # ── policy ───────────────────────────────────────────────────────────────
    def _is_compressible(self, mime: str) -> bool:
        """True for text-like payloads where gzip actually pays for itself.

        Media, archives, and already-compressed formats are stored verbatim —
        re-compressing them costs CPU and saves nothing.
        """
        if not mime:
            return False
        lowered = mime.lower()
        if lowered.startswith(("image/", "video/", "audio/")) and "svg" not in lowered:
            return False
        if lowered in {"application/zip", "application/gzip", "application/x-gzip",
                       "application/x-7z-compressed", "application/x-rar-compressed",
                       "application/x-xz", "application/octet-stream"}:
            return False
        return any(lowered.startswith(prefix) for prefix in self.compressible)

    # ── writes ───────────────────────────────────────────────────────────────
    def put_bytes(
        self,
        data: bytes,
        *,
        mime: str = "",
        refcount: int = 1,
    ) -> BlobInfo:
        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.info(sha256)
        if existing is not None:
            self._bump_refcount(sha256, refcount)
            self.stats["dedup_hits"] += 1
            return self.info(sha256) or existing

        guessed = mime or mimetypes.guess_type("f" + _ext_for(data))[0] or "application/octet-stream"
        should_compress = len(data) >= self.compress_above and self._is_compressible(guessed)

        target = self.path_for(sha256, should_compress)
        target.parent.mkdir(parents=True, exist_ok=True)
        if should_compress:
            with gzip.open(target, "wb", compresslevel=6) as handle:
                handle.write(data)
        else:
            target.write_bytes(data)
        stored = target.stat().st_size
        self._record(sha256, len(data), guessed, should_compress, stored, refcount)
        self.stats["puts"] += 1
        self.stats["bytes_written"] += stored
        return self.info(sha256) or BlobInfo(sha256, len(data), guessed, should_compress, stored, refcount, time.time())

    def put_file(
        self,
        source: str | os.PathLike[str],
        *,
        mime: str = "",
        refcount: int = 1,
        move: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> BlobInfo:
        """Store a file, streaming so large files never sit in memory.

        ``progress(done_bytes, total_bytes)`` is called per chunk when given.
        """
        path = Path(source).expanduser()
        if not path.is_file():
            raise NotFound(f"file not found: {path}")
        digest = hashlib.sha256()
        size = 0
        total = path.stat().st_size
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
                if progress is not None:
                    progress(size, total)
        sha256 = digest.hexdigest()

        existing = self.info(sha256)
        if existing is not None:
            self._bump_refcount(sha256, refcount)
            self.stats["dedup_hits"] += 1
            if move:
                path.unlink()
            return self.info(sha256) or existing

        guessed = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        should_compress = size >= self.compress_above and self._is_compressible(guessed)
        target = self.path_for(sha256, should_compress)
        target.parent.mkdir(parents=True, exist_ok=True)
        if should_compress:
            with path.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, CHUNK)
        else:
            shutil.copy2(path, target)
        if move:
            path.unlink()
        stored = target.stat().st_size
        self._record(sha256, size, guessed, should_compress, stored, refcount)
        self.stats["puts"] += 1
        self.stats["bytes_written"] += stored
        return self.info(sha256) or BlobInfo(sha256, size, guessed, should_compress, stored, refcount, time.time())

    def put_stream(
        self,
        stream: BinaryIO,
        *,
        mime: str = "",
        filename: str = "",
        refcount: int = 1,
        progress: Callable[[int, int], None] | None = None,
    ) -> BlobInfo:
        """Store from a file-like object, spooling to a temp file first.

        ``progress(done_bytes, total_bytes)`` is called per chunk; the total
        is 0 when the stream length is unknown up front.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, dir=str(self.root)) as tmp:
            tmp_path = Path(tmp.name)
            digest = hashlib.sha256()
            size = 0
            total = 0
            if progress is not None:
                try:
                    pos = stream.tell()
                    stream.seek(0, os.SEEK_END)
                    total = stream.tell() - pos
                    stream.seek(pos)
                except (OSError, AttributeError):
                    total = 0
            while chunk := stream.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
                tmp.write(chunk)
                if progress is not None:
                    progress(size, total)
        try:
            sha256 = digest.hexdigest()
            existing = self.info(sha256)
            if existing is not None:
                tmp_path.unlink(missing_ok=True)
                self._bump_refcount(sha256, refcount)
                self.stats["dedup_hits"] += 1
                return self.info(sha256) or existing
            guessed = mime or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            should_compress = size >= self.compress_above and self._is_compressible(guessed)
            target = self.path_for(sha256, should_compress)
            target.parent.mkdir(parents=True, exist_ok=True)
            if should_compress:
                with tmp_path.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                tmp_path.unlink()
            else:
                shutil.move(str(tmp_path), target)
            stored = target.stat().st_size
            self._record(sha256, size, guessed, should_compress, stored, refcount)
            self.stats["puts"] += 1
            self.stats["bytes_written"] += stored
            return self.info(sha256) or BlobInfo(
                sha256, size, guessed, should_compress, stored, refcount, time.time())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # ── reads ────────────────────────────────────────────────────────────────
    def get_bytes(self, sha256: str) -> bytes:
        info = self.info(sha256)
        if info is None:
            raise NotFound(f"blob {sha256} not in store")
        path = self.path_for(sha256, info.compressed)
        if not path.is_file():
            raise StorageError(f"blob {sha256} recorded but missing at {path}")
        self.stats["gets"] += 1
        if info.compressed:
            with gzip.open(path, "rb") as handle:
                return handle.read()
        return path.read_bytes()

    def export(self, sha256: str, destination: str | os.PathLike[str]) -> Path:
        """Write the blob to ``destination``, decompressing if needed."""
        data = self.get_bytes(sha256)
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def open(self, sha256: str) -> BinaryIO:
        import io

        return io.BytesIO(self.get_bytes(sha256))

    def read_range(self, sha256: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``offset`` (HTTP Range semantics).

        Uncompressed blobs seek directly — the media-streaming path. Gzip
        blobs are not seekable, so they stream through the decompressor and
        slice (documented cost; compressible blobs are small by policy).
        """
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be >= 0")
        info = self.info(sha256)
        if info is None:
            raise NotFound(f"blob {sha256} not in store")
        path = self.path_for(sha256, info.compressed)
        if not path.is_file():
            raise StorageError(f"blob {sha256} recorded but missing at {path}")
        self.stats["gets"] += 1
        if info.compressed:
            data = self.get_bytes(sha256)
            return data[offset : offset + length]
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)

    def stream_range(
        self, sha256: str, offset: int, length: int, *, chunk: int = CHUNK
    ) -> Iterator[bytes]:
        """Yield ``length`` bytes from ``offset`` in chunks (streaming reads)."""
        remaining = length
        cursor = offset
        while remaining > 0:
            piece = self.read_range(sha256, cursor, min(chunk, remaining))
            if not piece:
                break
            yield piece
            cursor += len(piece)
            remaining -= len(piece)

    # ── soft delete (trash) ────────────────────────────────────────────────
    #
    # vaultfs-style delete markers: ``trash()`` moves the blob aside instead
    # of destroying it, so an accidental purge is recoverable until
    # ``empty_trash()`` runs. The index row is parked in a JSON sidecar.

    @property
    def _trash_dir(self) -> Path:
        return self.root / ".trash"

    def _trash_meta_path(self, sha256: str) -> Path:
        return self._trash_dir / f"{sha256}.json"

    def trash(self, sha256: str) -> bool:
        """Move a blob to the trash (recoverable). Returns False if unknown."""
        import json

        info = self.info(sha256)
        if info is None:
            return False
        self._trash_dir.mkdir(parents=True, exist_ok=True)
        src = self.path_for(sha256, info.compressed)
        dst = self._trash_dir / src.name
        if src.is_file():
            shutil.move(str(src), dst)
        self._trash_meta_path(sha256).write_text(
            json.dumps({**info.to_dict(), "trashed_at": time.time()}),
            encoding="utf-8",
        )
        self.db.delete(self.TABLE, "sha256 = ?", (sha256,))
        return True

    def list_trash(self) -> list[dict[str, Any]]:
        """Trashed blobs with their metadata (newest first)."""
        import json

        if not self._trash_dir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for meta_path in self._trash_dir.glob("*.json"):
            try:
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda m: m.get("trashed_at", 0), reverse=True)

    def restore_trash(self, sha256: str) -> bool:
        """Restore a trashed blob to the live store. False if not trashed."""
        import json

        meta_path = self._trash_meta_path(sha256)
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        compressed = bool(meta.get("compressed"))
        src = self._trash_dir / f"{sha256}{'.gz' if compressed else ''}"
        if not src.is_file():
            return False
        target = self.path_for(sha256, compressed)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), target)
        self._record(
            sha256,
            int(meta.get("size", 0)),
            str(meta.get("mime", "")),
            compressed,
            target.stat().st_size,
            int(meta.get("refcount", 1)),
        )
        meta_path.unlink(missing_ok=True)
        return True

    def empty_trash(self, *, older_than_s: float | None = None) -> int:
        """Permanently delete trashed blobs (optionally only old ones)."""
        if not self._trash_dir.is_dir():
            return 0
        now = time.time()
        removed = 0
        for entry in self.list_trash():
            sha = entry.get("sha256", "")
            if older_than_s is not None:
                trashed_at = float(entry.get("trashed_at", 0))
                if now - trashed_at < older_than_s:
                    continue
            compressed = bool(entry.get("compressed"))
            (self._trash_dir / f"{sha}{'.gz' if compressed else ''}").unlink(
                missing_ok=True
            )
            self._trash_meta_path(sha).unlink(missing_ok=True)
            removed += 1
        return removed

    # ── cross-store sync ───────────────────────────────────────────────────
    def sync_to(
        self,
        other: Any,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Push every blob missing from ``other`` (same blob API).

        The primitive that backs blob replication: local → S3, phone → VPS.
        Content addressing makes it idempotent — re-running syncs only the
        delta. Returns ``{"pushed": n, "skipped": n, "bytes": n}``.
        """
        rows = self.db.query(f"SELECT sha256, mime FROM {self.TABLE} ORDER BY sha256")
        total = len(rows)
        pushed = skipped = 0
        sent_bytes = 0
        for index, row in enumerate(rows):
            sha = row["sha256"]
            if progress is not None:
                progress(index, total)
            if other.exists(sha):
                skipped += 1
                continue
            info = self.info(sha)
            if info is None:  # pragma: no cover - raced deletion
                skipped += 1
                continue
            # Re-store through the peer's own pipeline so its compression
            # policy and metadata stay canonical there.
            import io as _io

            try:
                if info.compressed:
                    payload: BinaryIO = _io.BytesIO(self.get_bytes(sha))
                else:
                    payload = self.path_for(sha, False).open("rb")
            except (OSError, StorageError) as exc:
                # DB row without a file (trashed mid-sync, partial dir):
                # skip it loudly rather than aborting the whole sync.
                _log.warning("sync_to: source file missing for %s: %s", sha, exc)
                skipped += 1
                continue
            try:
                stored = other.put_stream(payload, mime=info.mime)
            finally:
                payload.close()
            pushed += 1
            sent_bytes += stored.size if hasattr(stored, "size") else info.size
        if progress is not None:
            progress(total, total)
        return {"pushed": pushed, "skipped": skipped, "bytes": sent_bytes}

    def describe(self) -> dict[str, Any]:
        """One-dict health overview of the store."""
        row = self.db.query_one(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS size, "
            f"COALESCE(SUM(stored), 0) AS stored FROM {self.TABLE}"
        ) or {}
        trash_bytes = 0
        trash_count = 0
        if self._trash_dir.is_dir():
            for path in self._trash_dir.iterdir():
                if path.is_file() and not path.name.endswith(".json"):
                    trash_count += 1
                    try:
                        trash_bytes += path.stat().st_size
                    except OSError:
                        pass
        size = int(row.get("size") or 0)
        stored = int(row.get("stored") or 0)
        return {
            "blobs": int(row.get("n") or 0),
            "logical_bytes": size,
            "stored_bytes": stored,
            "dedup_ratio": round(size / stored, 3) if stored else 1.0,
            "trashed_blobs": trash_count,
            "trash_bytes": trash_bytes,
            "root": str(self.root),
            **self.stats,
        }

    def format_stats(self, theme: Any = None) -> str:
        """Human-readable store overview through the shared style layer."""
        theme = theme or active_theme()
        desc = self.describe()
        return "\n".join([
            header("blob store", theme=theme),
            *kv_lines(
                {
                    "root": desc["root"],
                    "blobs": desc["blobs"],
                    "logical": _blob_bytes(desc["logical_bytes"]),
                    "stored": _blob_bytes(desc["stored_bytes"]),
                    "dedup ratio": f"{desc['dedup_ratio']}x",
                    "puts": desc["puts"],
                    "dedup hits": desc["dedup_hits"],
                    "gets": desc["gets"],
                    "trashed": f"{desc['trashed_blobs']} ({_blob_bytes(desc['trash_bytes'])})",
                },
                theme=theme,
            ),
        ])

    # ── metadata ─────────────────────────────────────────────────────────────
    def info(self, sha256: str) -> BlobInfo | None:
        row = self.db.query_one(f"SELECT * FROM {self.TABLE} WHERE sha256 = ?", (sha256,))
        if row is None:
            return None
        return BlobInfo(
            sha256=row["sha256"],
            size=row["size"],
            mime=row["mime"],
            compressed=bool(row["compressed"]),
            stored=row["stored"],
            refcount=row["refcount"],
            created_at=row["created_at"],
        )

    def exists(self, sha256: str) -> bool:
        return self.info(sha256) is not None

    def _record(
        self,
        sha256: str,
        size: int,
        mime: str,
        compressed: bool,
        stored: int,
        refcount: int,
    ) -> None:
        self.db.insert(
            self.TABLE,
            {
                "sha256": sha256,
                "size": size,
                "mime": mime,
                "compressed": 1 if compressed else 0,
                "stored": stored,
                "refcount": refcount,
                "created_at": time.time(),
            },
        )

    def _bump_refcount(self, sha256: str, delta: int) -> None:
        self.db.execute(
            f"UPDATE {self.TABLE} SET refcount = MAX(0, refcount + ?) WHERE sha256 = ?",
            (delta, sha256),
        )

    def release(self, sha256: str, *, delete_at_zero: bool = False) -> int:
        """Decrement the reference count; optionally delete the blob at zero."""
        self._bump_refcount(sha256, -1)
        info = self.info(sha256)
        if info is None:
            return 0
        if delete_at_zero and info.refcount <= 0:
            self.purge(sha256)
            return 0
        return info.refcount

    def purge(self, sha256: str) -> bool:
        """Remove a blob from disk and from the index."""
        info = self.info(sha256)
        if info is None:
            return False
        for path in (self.path_for(sha256, False), self.path_for(sha256, True)):
            path.unlink(missing_ok=True)
        self.db.delete(self.TABLE, "sha256 = ?", (sha256,))
        return True

    # ── maintenance ──────────────────────────────────────────────────────────
    def verify(self, sha256: str | None = None) -> list[str]:
        """Re-hash stored blobs; returns a list of corruption problems."""
        problems: list[str] = []
        if sha256 is not None:
            rows = [self.info(sha256)]
        else:
            rows = [
                BlobInfo(r["sha256"], r["size"], r["mime"], bool(r["compressed"]), r["stored"], r["refcount"], r["created_at"])
                for r in self.db.query(f"SELECT * FROM {self.TABLE}")
            ]
        for info in rows:
            if info is None:
                continue
            path = self.path_for(info.sha256, info.compressed)
            if not path.is_file():
                problems.append(f"{info.sha256}: missing at {path}")
                continue
            data = self.get_bytes(info.sha256)
            actual = hashlib.sha256(data).hexdigest()
            if actual != info.sha256:
                problems.append(f"{info.sha256}: rehashed as {actual}")
            elif len(data) != info.size:
                problems.append(f"{info.sha256}: size {len(data)} != recorded {info.size}")
        return problems

    def orphans(self) -> list[str]:
        """Files on disk with no index entry — leftovers from interrupted writes.

        The ``.trash`` directory is excluded: trashed blobs are accounted for
        by :meth:`list_trash`, not orphans.
        """
        indexed = {r["sha256"] for r in self.db.query(f"SELECT sha256 FROM {self.TABLE}")}
        trash = self._trash_dir.resolve()
        found: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if trash in path.resolve().parents:
                    continue
            except OSError:
                pass
            name = path.stem if path.suffix == ".gz" else path.name
            if len(name) == 64 and name not in indexed:
                found.append(str(path))
        return found

    def sweep(self, *, delete_orphans: bool = True) -> dict[str, int]:
        removed = 0
        for path in self.orphans():
            if delete_orphans:
                Path(path).unlink(missing_ok=True)
            removed += 1
        return {"orphans": removed}

    def total_size(self) -> int:
        return int(self.db.scalar(f"SELECT COALESCE(SUM(stored), 0) FROM {self.TABLE}", default=0))

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "blobs": int(self.db.scalar(f"SELECT COUNT(*) FROM {self.TABLE}", default=0)),
            "bytes": self.total_size(),
            "root": str(self.root),
        }


def _blob_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024
    return f"{num:.1f}TB"  # pragma: no cover


def _ext_for(data: bytes) -> str:
    """Best-effort magic-byte sniff so MIME guessing works on unnamed data."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK\x03\x04"):
        return ".zip"
    if data.startswith(b"\x1f\x8b"):
        return ".gz"
    if data[:4] == b"RIFF":
        return ".webp"
    if data.startswith(b"\x00\x00\x00") and data[4:8] == b"ftyp":
        return ".mp4"
    if data.startswith(b"ID3") or data.startswith(b"\xff\xfb"):
        return ".mp3"
    return ".bin"
