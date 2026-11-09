"""The rendered-asset cache — same discipline as the media cache.

Rendered text assets are content-addressed: the key hashes the
text, the font file's own identity (sha256 of the file, face
index), size, style, colours, wrap width, direction, renderer
version and the shaping engine — so the same request is always
the same file and a different font file is never the same asset.
The assets directory is bounded by a hard cap swept with
``core.diskspace.evict_cache_dir`` (oldest first); stores and
evictions are journaled as JSONL rows in the store root (outside
the swept directory), readable like every other surface fact.
Hits are silent: no journal spam, just an mtime touch so LRU
order reflects real use.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from ...core import diskspace
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["RenderCache", "RENDERER_VERSION", "DEFAULT_CAP_BYTES"]

RENDERER_VERSION = "typography-render-1"
DEFAULT_CAP_BYTES = 128 * 1024 * 1024


class RenderCache:
    def __init__(self, store_dir: str | Path, *,
                 cap_bytes: int = DEFAULT_CAP_BYTES):
        self.store_dir = Path(store_dir)
        self.assets_dir = self.store_dir / "assets"
        self.journal_path = self.store_dir / "journal.jsonl"
        self.cap_bytes = int(cap_bytes)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    # identity ────────────────────────────────────────────────────
    @staticmethod
    def key(material: dict[str, Any]) -> str:
        canonical = json.dumps(material, sort_keys=True,
                               ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.assets_dir / f"{key}.png"

    # journal ─────────────────────────────────────────────────────
    def _journal(self, entry: dict[str, Any]) -> None:
        entry = {"ts": time.time(), **entry}
        try:
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            _log.debug("typography cache journal failed: %s", exc)

    # operations ──────────────────────────────────────────────────
    def get(self, key: str) -> Path | None:
        path = self.path_for(key)
        if path.is_file():
            self._hits += 1
            with suppress(OSError):
                os.utime(path, None)
            return path
        self._misses += 1
        return None

    def put(self, key: str, png: bytes, meta: dict[str, Any]) -> Path:
        path = self.path_for(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(png)
        os.replace(tmp, path)
        self._journal({"kind": "rendered_text_stored", "key": key,
                       "bytes": len(png), **meta})
        self.sweep()
        return path

    def sweep(self) -> dict[str, Any]:
        res = diskspace.evict_cache_dir(
            self.assets_dir, cap_bytes=self.cap_bytes)
        if res["files_evicted"] or res["errors"]:
            self._evictions += int(res["files_evicted"])
            self._journal({
                "kind": "rendered_text_evicted",
                "files_evicted": int(res["files_evicted"]),
                "bytes_freed": int(res["bytes_freed"]),
                "bytes_remaining": int(res["bytes_remaining"]),
                "cap_bytes": self.cap_bytes,
                "errors": int(res["errors"])})
        return res

    def stats(self) -> dict[str, Any]:
        total = files = 0
        for path in self.assets_dir.glob("*.png"):
            try:
                total += path.stat().st_size
                files += 1
            except OSError:
                continue
        return {"files": files, "bytes": total,
                "cap_bytes": self.cap_bytes, "hits": self._hits,
                "misses": self._misses, "evictions": self._evictions}
