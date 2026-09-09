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
from typing import Any, BinaryIO, Iterable

from ..core.errors import NotFound, StorageError
from ..core.logging_setup import get_logger
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
    ) -> BlobInfo:
        """Store a file, streaming so large files never sit in memory."""
        path = Path(source).expanduser()
        if not path.is_file():
            raise NotFound(f"file not found: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
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
    ) -> BlobInfo:
        """Store from a file-like object, spooling to a temp file first."""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, dir=str(self.root)) as tmp:
            tmp_path = Path(tmp.name)
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
                tmp.write(chunk)
        sha256 = digest.hexdigest()
        existing = self.info(sha256)
        if existing is not None:
            tmp_path.unlink(missing_ok=True)
            self._bump_refcount(sha256, refcount)
            self.stats["dedup_hits"] += 1
            return self.info(sha256) or existing
        guessed = mime or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        target = self.path_for(sha256, False)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), target)
        self._record(sha256, size, guessed, False, size, refcount)
        self.stats["puts"] += 1
        return self.info(sha256) or BlobInfo(sha256, size, guessed, False, size, refcount, time.time())

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
        """Files on disk with no index entry — leftovers from interrupted writes."""
        indexed = {r["sha256"] for r in self.db.query(f"SELECT sha256 FROM {self.TABLE}")}
        found: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
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
