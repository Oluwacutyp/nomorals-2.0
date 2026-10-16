"""Offline cache sync — move a followed story between devices.

The reader already caches every fetched chapter on disk. This module
exports that cache (story metadata + cached chapters + bookmarks +
continuations) as one zip, and imports it on another device. No cloud
claim: the zip is the sync mechanism — send it however you move files.

Import is traversal-safe: member names are rejected unless they stay
inside the story directory. Import never deletes the destination
first; it merges, and existing files win (the device's own reads are
never clobbered by an older export).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .reader import StoryReader

_log = get_logger(__name__)

__all__ = ["export_cache", "import_cache", "list_cached"]


def export_cache(context: Any, slug: str, dest: str | Path) -> dict[str, Any]:
    """Zip a followed story's cache to ``dest``. Returns file + counts."""
    reader = StoryReader(context)
    story = reader._load(slug)
    src = reader.story_dir(slug)
    dest_path = Path(dest)
    if dest_path.is_dir():
        dest_path = dest_path / f"{slug}-cache.zip"
    files: list[Path] = [p for p in sorted(src.rglob("*")) if p.is_file()]
    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(src).as_posix())
    return {
        "slug": slug, "title": story.title, "path": dest_path.as_posix(),
        "files": len(files),
        "bytes": dest_path.stat().st_size,
        "chapters_cached": story.chapters_cached,
    }


def import_cache(context: Any, zip_path: str | Path,
                 *, slug: str = "") -> dict[str, Any]:
    """Import a cache zip. Traversal-safe; existing files win."""
    reader = StoryReader(context)
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(f"no cache file at {zip_path}")
    target_slug = slug
    with zipfile.ZipFile(zpath, "r") as zf:
        members = zf.namelist()
        # the story slug is the export filename prefix, or the
        # story.json inside — never trust a member path for it
        if not target_slug:
            for member in members:
                if member.endswith("story.json"):
                    try:
                        meta = json.loads(zf.read(member).decode("utf-8"))
                        target_slug = str(meta.get("slug", ""))
                        break
                    except Exception:  # noqa: BLE001
                        continue
        if not target_slug:
            raise ValueError(
                "cache zip has no story.json — cannot determine its slug")
        dest = reader.story_dir(target_slug)
        dest.mkdir(parents=True, exist_ok=True)
        imported = 0
        skipped = 0
        for member in members:
            # traversal guard: reject absolute + .. members
            pure = Path(member)
            if pure.is_absolute() or ".." in pure.parts:
                skipped += 1
                continue
            out = dest / pure
            if out.exists():
                skipped += 1  # local files win
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))
            imported += 1
    story = reader._load(target_slug)
    return {
        "slug": target_slug, "title": story.title,
        "imported_files": imported, "skipped_files": skipped,
        "chapters_cached": story.chapters_cached,
    }


def list_cached(context: Any) -> list[dict[str, Any]]:
    """Followed stories with their on-disk cache sizes."""
    reader = StoryReader(context)
    out = []
    for entry in reader.following():
        d = reader.story_dir(entry["slug"])
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        out.append({**entry, "cache_bytes": size})
    return out
