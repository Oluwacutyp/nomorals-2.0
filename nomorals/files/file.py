"""The File class — one addressable file across the whole system.

A museum card is a dict, a bundle member is a dict, a filesystem
path is a string. Nothing unified them, so no verb could take "that
book" or "that download" and do the same thing with it. ``File`` is
the unification: one identity (kind/shelf/path/hash/size/provenance)
with behaviour — resolve a human ref to a real file, hash it, turn it
into a museum card, read its bytes.

Refs, in order:
* ``museum:<id>`` — a Museum card (exact id or unique prefix, the
  Museum's own rule); resolves to the card's path when the card is
  sendable, and refuses exhibit-only cards in words.
* ``library:<slug>`` — a Library book; resolves to its source file
  when the book has one, else the book directory.
* ``archive:<path>::<member>`` — one member inside an archive; the
  File carries its bytes (read through the bundle seam), never a
  fake path.
* ``downloads:<name>`` — a file in the downloads intake
  (``<home>/data/downloads``), by name or relative path.
* ``files:<shelf>/<name>`` — a shelved file, by shelf and name.
* a plain path — workspace-relative (the filesystem sandbox) or an
  absolute path under one of the home data roots (``data/files``,
  ``data/downloads``, ``data/media``). Anything else is a named
  refusal, never a silent escape.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .kinds import kind_for

__all__ = ["File", "FileError", "resolve", "home_roots"]


class FileError(ValueError):
    """A named file-class refusal — the reason, not a traceback."""


#: hash only this much when a caller asks for a quick identity hash
#: at census time — a music file's size carries its weight, like a
#: TrendOps render's (the Museum's own rule, same number).
_QUICK_HASH_MAX_BYTES = 1024 * 1024


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return Path(home).resolve() if home else None


def home_roots(context: Any) -> list[Path]:
    """The absolute roots a plain path may point under: the workspace
    plus the home data roots the system itself writes (files,
    downloads, media). Everything else is outside the file class."""
    roots: list[Path] = []
    settings = getattr(context, "settings", None) if context else None
    if settings is not None:
        workspace = getattr(settings, "workspace_dir", None)
        if workspace:
            roots.append(Path(workspace).resolve())
    home = _home_path(context)
    if home is not None:
        for rel in ("data/files", "data/downloads", "data/media"):
            roots.append((home / rel).resolve())
    return roots


def _resolve_plain_path(context: Any, ref: str) -> Path:
    """A plain path ref: workspace-relative, or absolute under a home
    root. Traversal out is a named refusal."""
    from ..tools.filesystem import safe_path

    candidate = Path(os.path.expanduser(ref))
    if candidate.is_absolute():
        resolved = candidate.resolve()
        inside = False
        for root in home_roots(context):
            try:
                resolved.relative_to(root)
                inside = True
                break
            except ValueError:
                continue
        if not inside:
            raise FileError(
                f"path {ref!r} is outside the file roots "
                f"({', '.join(str(r) for r in home_roots(context))})")
        if not resolved.exists():
            raise FileError(f"no such file: {ref}")
        return resolved
    return safe_path(context, ref, must_exist=True)


def _resolve_museum(context: Any, ref: str) -> Path:
    from .exhibit import museum_with_files

    museum = museum_with_files(context)
    card = museum.find(ref)
    if card is None:
        raise FileError(f"no museum card matches {ref!r}")
    if not card.get("sendable"):
        raise FileError(
            f"museum card {card['id']!r} is exhibit-only "
            f"({card.get('kind', '')}) — it has no file to take")
    path = Path(str(card.get("path") or ""))
    if not path.is_file():
        raise FileError(
            f"museum card {card['id']!r} points at a missing file: "
            f"{path}")
    return path


def _resolve_library(context: Any, slug: str) -> Path:
    from ..books.library import Library

    lib = Library(context)
    slug = (slug or "").strip()
    if not slug:
        raise FileError("library:<slug> needs a book slug")
    try:
        book_dir = lib.book_dir(slug)
    except Exception as exc:  # noqa: BLE001 — unknown slug, named
        raise FileError(f"no library book {slug!r}") from exc
    meta_path = book_dir / "book.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    source = meta.get("source_file") or ""
    if source:
        candidate = book_dir / source
        if candidate.is_file():
            return candidate
    if book_dir.is_dir():
        return book_dir
    raise FileError(f"library book {slug!r} has no file on disk")


class File:
    """One file, addressable everywhere. ``bytes`` is set only for
    archive members (bytes that live inside an archive, not on disk);
    every other File is a real path."""

    def __init__(self, path: str | Path, *, kind: str = "",
                 shelf: str = "", title: str = "",
                 provenance: dict[str, Any] | None = None,
                 tags: list[str] | None = None,
                 data: bytes | None = None) -> None:
        self.path = Path(path)
        self._data = data
        auto_kind, auto_shelf = kind_for(self.path)
        self.kind = (kind or "").strip().lower() or auto_kind
        self.shelf = (shelf or "").strip() or auto_shelf
        self.title = title or self.path.name
        self.provenance = dict(provenance or {})
        self.tags = list(tags or [])
        self._hash: str | None = None
        self._size: int | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_archive_member(self) -> bool:
        return self._data is not None

    @property
    def id(self) -> str:
        """Stable identity: kind + content hash when hashable, else
        kind + shelf + name."""
        digest = self.hash()
        if digest:
            return f"file:{self.kind}:{digest[:16]}"
        return f"file:{self.kind}:{self.shelf}:{self.name}"

    def size(self) -> int:
        if self._size is None:
            if self._data is not None:
                self._size = len(self._data)
            else:
                try:
                    self._size = self.path.stat().st_size
                except OSError:
                    self._size = 0
        return self._size

    def hash(self, *, quick: bool = False) -> str:
        """SHA-256 of the content (streamed, 1 MiB chunks). ``quick``
        hashes at most the first megabyte — the census-time identity,
        never a verification hash. Empty for directories and for
        missing files."""
        if self._hash is not None and not quick:
            return self._hash
        digest = hashlib.sha256()
        try:
            if self._data is not None:
                data = self._data[:_QUICK_HASH_MAX_BYTES] if quick \
                    else self._data
                digest.update(data)
            elif self.path.is_file():
                with open(self.path, "rb") as fh:
                    if quick:
                        digest.update(fh.read(_QUICK_HASH_MAX_BYTES))
                    else:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            digest.update(chunk)
            else:
                return ""
        except OSError:
            return ""
        out = digest.hexdigest()
        if not quick:
            self._hash = out
        return out

    def read_bytes(self) -> bytes:
        if self._data is not None:
            return self._data
        return self.path.read_bytes()

    def to_card(self) -> dict[str, Any]:
        """The museum card for this file — the same shape
        ``museum_card`` builds, so a File exhibits anywhere a card
        does."""
        from ..museum import museum_card

        return museum_card(
            id=self.id,
            title=self.title,
            kind=self.kind,
            shelf=self.shelf,
            path=str(self.path),
            hash=self.hash(quick=True),
            size=self.size(),
            provenance={"type": "file", "ref": str(self.path),
                        **self.provenance},
            tags=["file", self.kind] + self.tags)

    @property
    def sendable(self) -> bool:
        if self._data is not None:
            return True
        return self.path.is_file()

    def describe(self) -> str:
        return (f"{self.title} — {self.kind} · {self.size():,} B · "
                f"{self.shelf} · {self.path}")


def resolve(context: Any, ref: str) -> File:
    """Resolve a human ref to a File. Raises FileError with the reason
    when the ref names nothing takeable."""
    ref = (ref or "").strip()
    if not ref:
        raise FileError("a file ref is required")
    if ref.startswith("museum:"):
        path = _resolve_museum(context, ref[len("museum:"):])
        return File(path, provenance={"ref": ref, "tool": "files"})
    if ref.startswith("downloads:"):
        settings = getattr(context, "settings", None)
        if settings is None or not hasattr(settings, "resolve"):
            raise FileError(
                "downloads: refs need a context with settings")
        base = Path(settings.resolve("data/downloads"))
        rel = (ref[len("downloads:"):]).strip().strip("/")
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError as exc:
            raise FileError(
                f"downloads ref {ref!r} escapes the intake") from exc
        if not candidate.is_file():
            raise FileError(f"no such download: {rel}")
        return File(candidate,
                    provenance={"ref": ref, "tool": "files",
                                "intake": "downloads"})
    if ref.startswith("files:"):
        settings = getattr(context, "settings", None)
        if settings is None or not hasattr(settings, "resolve"):
            raise FileError(
                "files: refs need a context with settings")
        base = Path(settings.resolve("data/files"))
        rel = (ref[len("files:"):]).strip().strip("/")
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError as exc:
            raise FileError(
                f"files ref {ref!r} escapes the store") from exc
        if not candidate.is_file():
            raise FileError(f"no such shelved file: {rel}")
        return File(candidate,
                    provenance={"ref": ref, "tool": "files",
                                "store": str(base)})
    if ref.startswith("library:"):
        path = _resolve_library(context, ref[len("library:"):])
        f = File(path, provenance={"ref": ref, "tool": "files"})
        if path.is_dir():
            f.kind, f.shelf = "book", "files-books"
        return f
    if ref.startswith("archive:"):
        rest = ref[len("archive:"):]
        if "::" not in rest:
            raise FileError(
                "archive refs look like archive:<path>::<member>")
        archive_path, member = rest.split("::", 1)
        from ..archives import _BundleArchive

        apath = _resolve_plain_path(context, archive_path.strip())
        member = member.strip()
        try:
            with _BundleArchive(apath) as arc:
                entries = {e["name"]: e for e in arc.entries()}
                if member not in entries:
                    raise FileError(
                        f"archive {apath.name} has no member "
                        f"{member!r}")
                if entries[member]["is_dir"]:
                    raise FileError(
                        f"archive member {member!r} is a directory")
                data = arc.read(member)
        except FileError:
            raise
        except Exception as exc:  # noqa: BLE001 — unreadable archive
            raise FileError(
                f"cannot open archive {apath.name}: {exc}") from exc
        f = File(f"{apath}::{member}", kind="archive-member",
                 shelf="files-archives", title=member,
                 provenance={"ref": ref, "tool": "files",
                             "archive": str(apath)},
                 data=data)
        return f
    path = _resolve_plain_path(context, ref)
    if path.is_dir():
        raise FileError(
            f"{ref!r} is a directory — name a file, or pack the "
            f"directory as a bundle with its path")
    return File(path, provenance={"ref": ref, "tool": "files"})
