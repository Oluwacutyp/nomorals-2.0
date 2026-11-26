"""File-kind taxonomy — one shared answer to "what is this file".

A file's kind decides its shelf. The table is extension-first (fast,
honest about what an extension claims) and falls back to magic bytes
for archives, because a renamed ``.zip`` that is really a tarball is
still a tarball — the Archivist's ``detect_format`` is the authority
there. Kinds are OPEN: new kinds register with :func:`register_kind`
instead of editing this table, and a file that matches nothing is a
``download`` on the ``files-other`` shelf, never an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["kind_for", "register_kind", "KINDS", "shelf_for_kind"]

#: kind -> (shelf, extensions). The shelf is where the file lives in
#: the FileStore; the kind is what the Museum card and the pack
#: manifest carry. Audio splits wav from music because the Museum's
#: downloads shelf does (wav = raw captures, music = finished tracks).
_KIND_TABLE: dict[str, tuple[str, frozenset[str]]] = {
    "audio-wav": ("files-audio",
                  frozenset({".wav"})),
    "audio-music": ("files-music",
                    frozenset({".mp3", ".flac", ".ogg", ".m4a", ".aac",
                               ".opus", ".wma"})),
    "archive": ("files-archives",
                frozenset({".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
                           ".rar", ".tgz"})),
    "document": ("files-documents",
                 frozenset({".pdf", ".docx", ".doc", ".odt", ".rtf",
                            ".txt", ".md", ".csv", ".tsv", ".xls",
                            ".xlsx", ".ppt", ".pptx", ".odp", ".ods"})),
    "book": ("files-books",
             frozenset({".epub", ".mobi", ".azw3", ".fb2", ".cbz",
                        ".cbr"})),
    "image": ("files-images",
              frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp",
                         ".bmp", ".svg", ".tif", ".tiff", ".heic",
                         ".avif"})),
    "video": ("files-video",
              frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm",
                         ".m4v", ".wmv", ".flv"})),
    "code": ("files-code",
             frozenset({".py", ".js", ".ts", ".go", ".rs", ".java",
                        ".c", ".h", ".cpp", ".hpp", ".rb", ".php",
                        ".sh", ".html", ".htm", ".css", ".json",
                        ".yaml", ".yml", ".toml", ".sql"})),
    "data": ("files-data",
             frozenset({".db", ".sqlite", ".sqlite3", ".parquet",
                        ".arrow", ".npy", ".npz", ".pkl", ".joblib"})),
}

#: archive formats the magic sniffer can name, mapped onto the
#: archive kind even when the extension lies.
_ARCHIVE_FORMATS = frozenset({
    "zip", "tar", "tar.gz", "tar.bz2", "tar.xz", "7z", "rar",
    "gzip", "bzip2", "xz",
})

_FALLBACK_KIND = "download"
_FALLBACK_SHELF = "files-other"


def register_kind(kind: str, shelf: str, extensions: Any) -> None:
    """Register a new kind (or extend one) — open taxonomy, never a
    closed list. ``extensions`` is any iterable of ``.ext`` strings;
    they are lowercased and dotted on the way in."""
    kind = (kind or "").strip().lower()
    shelf = (shelf or "").strip()
    if not kind or not shelf:
        raise ValueError("kind and shelf are both required")
    exts = frozenset(
        (e if e.startswith(".") else "." + e).lower()
        for e in (extensions or []))
    if kind in _KIND_TABLE:
        old_shelf, old_exts = _KIND_TABLE[kind]
        _KIND_TABLE[kind] = (old_shelf, old_exts | exts)
    else:
        _KIND_TABLE[kind] = (shelf, exts)


def KINDS() -> dict[str, dict[str, Any]]:
    """The live taxonomy: kind -> {shelf, extensions}."""
    return {kind: {"shelf": shelf, "extensions": sorted(exts)}
            for kind, (shelf, exts) in sorted(_KIND_TABLE.items())}


def shelf_for_kind(kind: str) -> str:
    """The shelf a kind lives on (fallback shelf for unknown kinds)."""
    entry = _KIND_TABLE.get((kind or "").strip().lower())
    return entry[0] if entry else _FALLBACK_SHELF


def kind_for(path: str | Path) -> tuple[str, str]:
    """(kind, shelf) for a path, by extension first, magic bytes for
    archives. A path that matches nothing is a ``download`` — the
    intake kind, never an error."""
    p = Path(path)
    suffix = p.suffix.lower()
    for kind, (shelf, exts) in _KIND_TABLE.items():
        if suffix in exts:
            return kind, shelf
    # Extension lied or is absent: let the bytes speak for archives.
    try:
        if p.is_file():
            from ..archives import detect_format

            if detect_format(p) in _ARCHIVE_FORMATS:
                return "archive", shelf_for_kind("archive")
    except Exception:  # noqa: BLE001 — sniff failure degrades to fallback
        pass
    return _FALLBACK_KIND, _FALLBACK_SHELF
