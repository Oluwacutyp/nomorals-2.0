"""The font catalogue — every usable face on the host, as data.

Discovery roots, in order: ``NM_FONT_DIRS`` (path-separated),
font directories vendored in the install (``fonts/`` beside the
repo root and ``nomorals/media/typography/fonts/`` if a build ever
ships faces), then the platform's font directories (Linux system
+ user, Termux prefix, macOS, Windows incl. the per-user Fonts
folder). Every ``.ttf/.otf/.ttc/.woff/.woff2`` under those roots
is parsed by :mod:`fontparse`; a file that does not parse is
counted in the stats, never silently dropped and never invented.

A disk index (``font_index.json`` in the service store) memoizes
per-file parse results keyed by (path, size, mtime) so repeat
catalogue builds are a stat sweep, not a re-parse of every Noto
TTC on the host. The index is a cache of facts about files; when
a file changes, its entry is rebuilt.

All orderings out of this module are deterministic — same host
state, same catalogue, same search results, every time.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from .fontparse import FontFace, parse_font_file

_log = get_logger(__name__)

__all__ = ["FontCatalogue", "build_catalogue", "default_roots"]

_FONT_EXTS = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}
_INDEX_VERSION = 1


def default_roots(extra: Iterable[str | Path] = ()) -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("NM_FONT_DIRS", "")
    for part in env.split(os.pathsep):
        if part.strip():
            roots.append(Path(part.strip()).expanduser())
    for item in extra:
        roots.append(Path(item).expanduser())
    pkg = Path(__file__).resolve()
    for vendored in (pkg.parent / "fonts", pkg.parents[3] / "fonts"):
        if vendored.is_dir():
            roots.append(vendored)
    prefix = os.environ.get("PREFIX", "")  # Termux
    candidates = [
        Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
        Path.home() / ".fonts", Path.home() / ".local/share/fonts",
        Path("/Library/Fonts"), Path.home() / "Library/Fonts",
        Path("/System/Library/Fonts"),
    ]
    if prefix:
        candidates.append(Path(prefix) / "share/fonts")
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates.append(Path(windir) / "Fonts")
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata:
        candidates.append(Path(localappdata) / "Microsoft/Windows/Fonts")
    for cand in candidates:
        if cand.is_dir():
            roots.append(cand)
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


class FontCatalogue:
    def __init__(self, faces: list[FontFace], stats: dict[str, Any]):
        self._faces = sorted(
            faces, key=lambda f: (f.family.lower(), f.weight, f.italic,
                                  f.path, f.face_index))
        self._stats = stats

    def __len__(self) -> int:
        return len(self._faces)

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def faces(self, family: str | None = None) -> list[FontFace]:
        if family is None:
            return list(self._faces)
        want = family.strip().lower()
        return [f for f in self._faces if f.family.lower() == want]

    def families(self) -> list[str]:
        return sorted({f.family for f in self._faces},
                      key=lambda s: s.lower())

    def covering(self, script: str) -> list[FontFace]:
        """Faces that cover ``script``, best coverage first."""
        hits = [f for f in self._faces if f.covers(script)]
        return sorted(hits, key=lambda f: (-f.scripts.get(script, 0.0),
                                           f.family.lower(), f.weight,
                                           f.path, f.face_index))

    def covering_text(self, text: str) -> list[FontFace]:
        return [f for f in self._faces if f.covers_text(text)]

    def search(self, query: str) -> list[FontFace]:
        """Deterministic name search: exact family, then family
        prefix, then substring over family/full/PostScript names."""
        q = " ".join(str(query or "").lower().split())
        if not q:
            return []
        ranked: list[tuple[int, FontFace]] = []
        for face in self._faces:
            fam = face.family.lower()
            names = (fam, face.full_name.lower(),
                     face.postscript_name.lower())
            if fam == q:
                rank = 0
            elif fam.startswith(q):
                rank = 1
            elif any(q in n for n in names):
                rank = 2
            elif all(tok in " ".join(names) for tok in q.split()):
                rank = 3
            else:
                continue
            ranked.append((rank, face))
        ranked.sort(key=lambda rf: (rf[0], rf[1].family.lower(),
                                    rf[1].weight, rf[1].path,
                                    rf[1].face_index))
        return [face for _rank, face in ranked]


def _load_index(index_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if data.get("version") == _INDEX_VERSION:
            return data.get("files", {})
    except (OSError, ValueError):
        pass
    return {}


def _save_index(index_path: Path, files: dict[str, Any]) -> None:
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": _INDEX_VERSION,
                                   "files": files}),
                       encoding="utf-8")
        os.replace(tmp, index_path)
    except OSError as exc:
        _log.debug("typography: font index save failed: %s", exc)


def build_catalogue(roots: Iterable[str | Path] | None = None,
                    index_path: str | Path | None = None) -> FontCatalogue:
    root_list = [Path(r) for r in (roots if roots is not None
                                   else default_roots())]
    index = _load_index(Path(index_path)) if index_path else {}
    new_index: dict[str, Any] = {}
    faces: list[FontFace] = []
    stats = {"files_seen": 0, "files_failed": 0, "files_parsed": 0,
             "files_indexed": 0, "faces": 0, "families": 0,
             "roots": [str(r) for r in root_list]}
    seen_files: set[str] = set()
    for root in root_list:
        if not root.is_dir():
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            for name in sorted(filenames):
                if Path(name).suffix.lower() not in _FONT_EXTS:
                    continue
                path = Path(dirpath) / name
                key = str(path)
                if key in seen_files:
                    continue
                seen_files.add(key)
                stats["files_seen"] += 1
                try:
                    st = path.stat()
                except OSError:
                    stats["files_failed"] += 1
                    continue
                entry = index.get(key)
                if entry and entry.get("size") == st.st_size and \
                        entry.get("mtime_ns") == st.st_mtime_ns:
                    file_faces = [FontFace.from_dict(d)
                                  for d in entry.get("faces", [])]
                    stats["files_indexed"] += 1
                else:
                    file_faces = parse_font_file(path)
                    stats["files_parsed"] += 1
                if not file_faces:
                    stats["files_failed"] += 1
                new_index[key] = {
                    "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                    "faces": [f.to_dict() for f in file_faces]}
                faces.extend(file_faces)
    if index_path:
        _save_index(Path(index_path), new_index)
    stats["faces"] = len(faces)
    stats["families"] = len({f.family for f in faces})
    return FontCatalogue(faces, stats)
