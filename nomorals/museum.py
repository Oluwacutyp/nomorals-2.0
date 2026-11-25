"""The Museum generalized: one browsable census over EVERY system, not
just ApkForge build outputs.

The card shape is ApkForge's (``id/title/kind/shelf/path/hash/size/
provenance/tags`` — ``agents/apkforge/museum.py``), plus ``sendable``:
True only when the card's path exists and is a real file. Non-file
exhibits (digests, sources) point at their owning store as provenance
and are never sendable — the send path must refuse them with a clear
reason, never ship the whole store file by accident.

Exhibitors read REAL system state on every census (no cached
showcase, no hardcoded list). A raising exhibitor contributes a named
error and never kills the other shelves. New systems join by
registering an exhibitor — the census core never changes for them.

Beyond the census the Museum is the artifact home: ``downloads``
exhibits everything Devon fetched (wavs, downloaded music, files)
under the data-downloads root; ``search`` finds a card by a human
name, kind, or date — never only by a long id; ``zip_card`` /
``unzip_card`` pack and unpack a card on demand (stdlib zipfile, the
Archivist's skip-and-report + bomb-cap contract, because the
Archivist itself is workspace-confined and museum cards live under
the home); companions link a book to its artifacts and ``cite``
resolves them live, so a vanished artifact reports missing instead
of silently dropping out of the story.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

from .museum_failures import record_failure

__all__ = ["Museum", "museum_card", "apkforge_exhibitor",
           "library_exhibitor", "characters_exhibitor",
           "trendops_exhibitor", "downloads_exhibitor",
           "museum_store_exhibitor"]

#: Audio kinds the downloads shelf knows by extension.
_AUDIO_WAV_EXTS = {".wav"}
_AUDIO_MUSIC_EXTS = {".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
                     ".wma"}
_ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}

#: Hash only small files at census time — a music file's size carries
#: its weight, like a TrendOps render's; the hash is the manifest's.
_HASH_MAX_BYTES = 1024 * 1024

#: Zip/unzip bomb caps (the Archivist's contract, same numbers in
#: spirit: a museum archive is refused whole when it trips them).
_ZIP_MAX_FILES = 10_000
_ZIP_MAX_BYTES = 2 * 1024 * 1024 * 1024

#: Unpacked files exhibited per census (sorted, first N).
_UNPACKED_CAP = 200


def museum_card(*, id: str, title: str, kind: str, shelf: str,
                path: str, hash: str = "", size: int = 0,
                provenance: dict[str, Any] | None = None,
                tags: list[str] | None = None,
                sendable: bool | None = None) -> dict[str, Any]:
    """Build a card in the shared shape. ``sendable``: None means
    "decide at census time from the path"; an explicit False always
    wins (exhibit-only by design — digest/source cards use it so a
    store file can never be marked sendable by accident)."""
    card: dict[str, Any] = {
        "id": id, "title": title, "kind": kind, "shelf": shelf,
        "path": path, "hash": hash, "size": size,
        "provenance": provenance or {}, "tags": tags or [],
    }
    if sendable is not None:
        card["sendable"] = sendable
    return card


def _finalize(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute sendable for every card that did not explicitly opt
    out: a real file at the path, and only then. An explicit False
    (digest/source cards) is never overridden — that is the guard
    against shipping a whole store file as if it were an exhibit."""
    for card in cards:
        if card.get("sendable") is False:
            continue
        p = Path(str(card.get("path") or ""))
        card["sendable"] = bool(p.is_file())
    return cards


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    continue
    except OSError as exc:
        record_failure("museum._dir_size", exc)
    return total


# ── exhibitors ────────────────────────────────────────────────────────

def apkforge_exhibitor(root: Path) -> list[dict[str, Any]]:
    """ApkForge's own census, delegated — same cards, same shelves,
    zero duplication. Lazy import: the apkforge package pulls the
    whole forge, which the museum never needs at import time."""
    from .agents.apkforge.museum import list_artifacts

    return list_artifacts(Path(root)).get("artifacts", [])


def library_exhibitor(library: Any) -> list[dict[str, Any]]:
    """Library exhibits: books, digests, sources — the living organ's
    state, not just what was built. Books hash their book.json;
    digests hash their claim; sources carry health in tags."""
    cards: list[dict[str, Any]] = []
    root = library.root()
    db_path = str(library.db_path())
    for book in library.list_books():
        slug = book["slug"]
        book_dir = root / slug
        meta_path = book_dir / "book.json"
        digest = ""
        original = None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(
                meta_path.read_bytes()).hexdigest()[:16]
            candidate = book_dir / (meta.get("source_file") or "")
            if candidate.is_file():
                original = candidate
        except (OSError, ValueError) as exc:
            record_failure("museum.library_exhibitor.book_meta", exc)
        cards.append(museum_card(
            id=f"library-book:{slug}",
            title=f"{book['title']} ({book['chapters']} chapters)",
            kind="library-book", shelf="library-books",
            path=str(original) if original else str(book_dir),
            hash=digest,
            size=_dir_size(book_dir),
            provenance={"type": "ingest", "ref": slug, "tool": "library"},
            tags=["library", "book"] + list(book.get("tags") or [])
            + ([f"date:{book['ingested']}"]
               if book.get("ingested") else [])))
    for d in library.list_digests(status=""):
        cards.append(museum_card(
            id=f"library-digest:{d['record_id']}",
            title=f"digest: {d['claim'][:80]}",
            kind="library-digest", shelf="library-digests",
            path=db_path,
            hash=hashlib.sha256(d["claim"].encode("utf-8")).hexdigest()[:16],
            size=len(d["claim"]),
            provenance={"type": "digest", "ref": d["source_id"],
                        "tool": "library", "scope": d["scope"],
                        "confidence": d["confidence"]},
            tags=["library", "digest", d["status"]]
            + ([f"scope:{d['scope']}"] if d["scope"] else []),
            sendable=False))
    for s in library.list_sources():
        cards.append(museum_card(
            id=f"library-source:{s['source_id']}",
            title=f"source: {s['title'] or s['origin']}",
            kind="library-source", shelf="library-sources",
            path=s["origin"],
            hash="",
            size=0,
            provenance={"type": "source", "ref": s["source_id"],
                        "tool": "library", "kind": s["kind"]},
            tags=["library", "source", s["kind"], s["status"],
                  f"health:{s['health']}"],
            sendable=False))
    return cards


def characters_exhibitor(char_store: Any, learn_store: Any,
                         learn_dir: Path | str | None = None,
                         ) -> list[dict[str, Any]]:
    """Characters' learned state: one card per character with a learn
    doc. Stats come from LearnStore.doc() (public) over ids from
    CharacterStore.all() (public) — no learn.py changes, no private
    access. Characters without learn docs are skipped: no learned
    state means no exhibit, never an empty card.

    The learn dir: pass it explicitly when the LearnStore was built
    with an explicit data_dir (the two must agree — the store does not
    expose its dir publicly). When both are default-constructed, the
    dir derives from the PUBLIC default_character_dir() + "learn",
    which is exactly LearnStore's default resolution. The doc filename
    uses the same sanitizer as learn._safe_id (alnum + -_ , 32 chars);
    the path is used only when the file actually exists, so any future
    divergence degrades to exhibit-only instead of a wrong link."""
    from .characters.store import default_character_dir

    if learn_dir is not None:
        learn_path = Path(learn_dir)
    else:
        learn_path = Path(default_character_dir()) / "learn"
    cards: list[dict[str, Any]] = []
    for char in char_store.all():
        try:
            doc = learn_store.doc(char.id)
        except Exception as exc:  # noqa: BLE001 — one bad doc never kills census
            record_failure("museum.characters_exhibitor.learn_doc", exc)
            continue
        entries = doc.get("entries") or []
        if not entries and not (doc.get("stats") or {}).get("graded"):
            continue
        stats = doc.get("stats") or {}
        mean = stats.get("mean_grade")
        mean_s = f"{mean:.2f}" if isinstance(mean, (int, float)) else "—"
        safe = "".join(c for c in (char.id or "")
                       if c.isalnum() or c in "-_")[:32]
        doc_path = learn_path / f"{safe}.json"
        path = str(doc_path) if doc_path.is_file() else str(learn_path)
        cards.append(museum_card(
            id=f"character-learned:{char.id}",
            title=f"{char.name or char.id} — learned "
                  f"({stats.get('graded', 0)} graded, mean {mean_s})",
            kind="character-learned", shelf="characters-learned",
            path=path,
            hash=hashlib.sha256(
                json.dumps(doc.get("weights") or {},
                           sort_keys=True).encode("utf-8")).hexdigest()[:16],
            size=len(entries),
            provenance={"type": "learn", "ref": char.id,
                        "tool": "characters"},
            tags=["characters", "learned",
                  f"graded:{stats.get('graded', 0)}"]))
    return cards


def _download_kind(path: Path) -> tuple[str, str]:
    """(kind, shelf) for a downloaded file, by extension only."""
    suffix = path.suffix.lower()
    if suffix in _AUDIO_WAV_EXTS:
        return "audio-wav", "downloads-audio"
    if suffix in _AUDIO_MUSIC_EXTS:
        return "audio-music", "downloads-audio"
    if suffix in _ARCHIVE_EXTS:
        return "archive", "downloads-files"
    return "download", "downloads-files"


def _date_tag(mtime: float) -> str:
    if not mtime:
        return ""
    return time.strftime("%Y-%m-%d", time.localtime(mtime))


def downloads_exhibitor(root: Path | str) -> list[dict[str, Any]]:
    """Everything Devon fetched, in its own space: wavs, downloaded
    music, archives, and every other file under the downloads root —
    one card per real file, found by its human filename. A missing
    root exhibits nothing (a fresh home has no downloads yet), never
    an error. Small files hash like library books do; a music file's
    size carries its weight, like a TrendOps render's."""
    base = Path(root)
    if not base.is_dir():
        return []
    try:
        files = sorted(f for f in base.rglob("*") if f.is_file())
    except OSError as exc:
        record_failure("museum.downloads_exhibitor.listdir", exc)
        return []
    cards: list[dict[str, Any]] = []
    for f in files:
        kind, shelf = _download_kind(f)
        rel = f.relative_to(base).as_posix()
        try:
            stat = f.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        digest = ""
        if 0 < size <= _HASH_MAX_BYTES:
            try:
                digest = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            except OSError as exc:
                record_failure("museum.downloads_exhibitor.hash", exc)
        date = _date_tag(mtime)
        tags = ["downloads", f"ext:{f.suffix.lstrip('.').lower() or 'none'}"]
        if date:
            tags.append(f"date:{date}")
        cards.append(museum_card(
            id=f"{shelf}:{rel}", title=f.name, kind=kind, shelf=shelf,
            path=str(f), hash=digest, size=size,
            provenance={"type": "download", "ref": rel,
                        "tool": "downloads"},
            tags=tags))
    return cards


def museum_store_exhibitor(museum_root: Path | str | None,
                           archives_root: Path | str | None,
                           ) -> list[dict[str, Any]]:
    """The Museum's own store: every zip the museum (or ``nm zip`` /
    ``zip_create``) made, and every file an unpack landed. Zips are
    one card each; unpacked files exhibit up to the cap, sorted, so
    a huge unpack can never flood the census."""
    cards: list[dict[str, Any]] = []
    zip_dirs: list[tuple[Path, str]] = []
    if museum_root is not None:
        zip_dirs.append((Path(museum_root) / "zips", "zips"))
    if archives_root is not None:
        zip_dirs.append((Path(archives_root), "archives"))
    for zdir, label in zip_dirs:
        try:
            zips = sorted(zdir.glob("*.zip")) if zdir.is_dir() else []
        except OSError as exc:
            record_failure("museum.museum_store_exhibitor.listdir", exc)
            continue
        for z in zips:
            try:
                size, mtime = z.stat().st_size, z.stat().st_mtime
            except OSError:
                size, mtime = 0, 0.0
            date = _date_tag(mtime)
            tags = ["museum", "zip"]
            if date:
                tags.append(f"date:{date}")
            cards.append(museum_card(
                id=f"museum-zip:{label}:{z.name}", title=z.name,
                kind="museum-zip", shelf="museum-zips", path=str(z),
                hash="", size=size,
                provenance={"type": "archive", "ref": z.name,
                            "tool": "museum"},
                tags=tags))
    if museum_root is not None:
        unpacked = Path(museum_root) / "unpacked"
        if unpacked.is_dir():
            try:
                files = sorted(f for f in unpacked.rglob("*")
                               if f.is_file())[:_UNPACKED_CAP]
            except OSError as exc:
                record_failure(
                    "museum.museum_store_exhibitor.unpacked", exc)
                files = []
            for f in files:
                kind, _shelf = _download_kind(f)
                rel = f.relative_to(unpacked).as_posix()
                try:
                    size, mtime = f.stat().st_size, f.stat().st_mtime
                except OSError:
                    size, mtime = 0, 0.0
                date = _date_tag(mtime)
                tags = ["museum", "unpacked"]
                if date:
                    tags.append(f"date:{date}")
                cards.append(museum_card(
                    id=f"museum-unpacked:{rel}", title=f.name,
                    kind=kind, shelf="museum-unpacked", path=str(f),
                    hash="", size=size,
                    provenance={"type": "unpacked", "ref": rel,
                                "tool": "museum"},
                    tags=tags))
    return cards



# ── the museum ────────────────────────────────────────────────────────

def trendops_exhibitor(root: Path) -> list[dict[str, Any]]:
    """TrendOps renders: one card per MP4 in the renders dir, facts
    from the sibling JSON manifest (written by apply_format) when
    present. A render with no manifest still exhibits — a minimal
    card from the filename, no format guessed from it; a corrupt
    manifest is a named failure, never a dead shelf. The card's hash
    is the manifest's (small, like the library hashing book.json),
    never the MP4's — the file is the payload, its size carries its
    weight. Findable by format (``format:<slug>``), by QA verdict
    (``qa:<status>``) and by draft (``draft:<id>``) through tags."""
    renders = Path(root) / "renders"
    try:
        clips = sorted(renders.glob("*.mp4"))
    except OSError as exc:
        record_failure("museum.trendops_exhibitor.listdir", exc)
        return []
    cards: list[dict[str, Any]] = []
    for clip in clips:
        stem = clip.stem
        manifest_path = clip.with_suffix(".json")
        manifest: dict[str, Any] = {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass  # renders predating manifests still exhibit
        except (OSError, ValueError) as exc:
            record_failure("museum.trendops_exhibitor.manifest", exc)
        slug = str(manifest.get("format") or "")
        name = str(manifest.get("format_name") or slug or stem)
        duration = float(manifest.get("duration") or 0.0)
        review = manifest.get("review") or {}
        qa = review.get("qa") or {}
        qa_status = str(qa.get("status") or "")
        draft_id = str(manifest.get("draft_id") or "")
        digest = ""
        try:
            digest = hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()[:16]
        except OSError:
            digest = ""
        try:
            size = clip.stat().st_size
        except OSError:
            size = 0
        tags = ["trendops"]
        if slug:
            tags.append(f"format:{slug}")
        if qa_status:
            tags.append(f"qa:{qa_status}")
        if draft_id:
            tags.append(f"draft:{draft_id}")
        cards.append(museum_card(
            id=f"trendops-render:{stem}",
            title=(f"{name} cut — {duration:.1f}s" if duration > 0
                   else f"{name} cut"),
            kind="trendops-render", shelf="trendops-renders",
            path=str(clip), hash=digest, size=size,
            provenance={"type": "render", "ref": slug,
                        "tool": "trendops", "qa": qa_status,
                        "draft_id": draft_id},
            tags=tags))
    return cards


class Museum:
    """One census, many systems. Roots are explicit (no guessed paths):
    pass what you have; missing systems simply exhibit nothing."""

    def __init__(self, *, apk_root: Path | str | None = None,
                 library: Any = None, char_store: Any = None,
                 learn_store: Any = None,
                 learn_dir: Path | str | None = None,
                 trendops_root: Path | str | None = None,
                 downloads_root: Path | str | None = None,
                 archives_root: Path | str | None = None,
                 museum_root: Path | str | None = None) -> None:
        self.apk_root = Path(apk_root) if apk_root else None
        self.library = library
        self.char_store = char_store
        self.learn_store = learn_store
        self.learn_dir = learn_dir
        self.trendops_root = (Path(trendops_root)
                              if trendops_root else None)
        self.downloads_root = (Path(downloads_root)
                               if downloads_root else None)
        self.archives_root = (Path(archives_root)
                              if archives_root else None)
        self.museum_root = (Path(museum_root)
                            if museum_root else None)
        self._companions: list[dict[str, Any]] = []
        self._companions_loaded = False
        self._exhibitors: dict[str, Callable[[], list[dict[str, Any]]]] = {}
        if self.apk_root is not None:
            self.register_exhibitor(
                "apkforge", lambda: apkforge_exhibitor(self.apk_root))
        if self.library is not None:
            self.register_exhibitor(
                "library", lambda: library_exhibitor(self.library))
        if self.char_store is not None and self.learn_store is not None:
            self.register_exhibitor(
                "characters",
                lambda: characters_exhibitor(self.char_store,
                                             self.learn_store,
                                             self.learn_dir))
        if self.trendops_root is not None:
            self.register_exhibitor(
                "trendops",
                lambda: trendops_exhibitor(self.trendops_root))
        if self.downloads_root is not None:
            self.register_exhibitor(
                "downloads",
                lambda: downloads_exhibitor(self.downloads_root))
        if self.museum_root is not None or self.archives_root is not None:
            self.register_exhibitor(
                "museum-store",
                lambda: museum_store_exhibitor(self.museum_root,
                                               self.archives_root))

    def register_exhibitor(self, name: str,
                           fn: Callable[[], list[dict[str, Any]]]) -> None:
        """A future system (desk, pulse, anything) joins the census by
        registering a callable returning cards. The core never changes
        for a new system."""
        self._exhibitors[name] = fn

    def census(self) -> dict[str, Any]:
        """Aggregate every exhibitor. A raising exhibitor contributes a
        named error and never kills the other shelves."""
        artifacts: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for name, fn in self._exhibitors.items():
            try:
                artifacts.extend(fn())
            except Exception as exc:  # noqa: BLE001 — named, not hidden
                errors[name] = str(exc)[:300]
        artifacts = _finalize(artifacts)
        by_shelf: dict[str, int] = {}
        for card in artifacts:
            by_shelf[card["shelf"]] = by_shelf.get(card["shelf"], 0) + 1
        return {"artifacts": artifacts, "by_shelf": by_shelf,
                "count": len(artifacts), "errors": errors}

    def find(self, ref: str) -> dict[str, Any] | None:
        """Resolve an artifact by exact id or unique id prefix (the
        apkforge rule, generalized)."""
        ref = (ref or "").strip()
        if not ref:
            return None
        arts = self.census()["artifacts"]
        for card in arts:
            if card["id"] == ref:
                return card
        prefixed = [c for c in arts if c["id"].startswith(ref)]
        return prefixed[0] if len(prefixed) == 1 else None

    def search(self, query: str = "", kind: str = "",
               date: str = "") -> dict[str, Any]:
        """Find cards the way a human asks: words from the title
        (every word must match, across id/title/shelf/kind/tags),
        an exact kind, a ``YYYY-MM-DD`` date (the ``date:`` tag the
        downloads, library-book, and zip cards carry). An empty
        query with no filters is the whole census."""
        words = (query or "").strip().lower().split()
        want_kind = (kind or "").strip().lower()
        want_date = (date or "").strip()
        cards: list[dict[str, Any]] = []
        for card in self.census()["artifacts"]:
            if want_kind and str(card.get("kind") or "").lower() != want_kind:
                continue
            if want_date and f"date:{want_date}" not in (card.get("tags") or []):
                continue
            if words:
                hay = " ".join([
                    str(card.get("id") or ""), str(card.get("title") or ""),
                    str(card.get("shelf") or ""), str(card.get("kind") or ""),
                    " ".join(card.get("tags") or [])]).lower()
                if not all(w in hay for w in words):
                    continue
            cards.append(card)
        by_shelf: dict[str, int] = {}
        for card in cards:
            by_shelf[card["shelf"]] = by_shelf.get(card["shelf"], 0) + 1
        sub = {"artifacts": cards, "by_shelf": by_shelf,
               "count": len(cards), "errors": {}}
        if cards:
            text = self.render(sub)
        else:
            text = (f"no artifacts match {query!r}"
                    + (f" kind={kind!r}" if want_kind else "")
                    + (f" date={date!r}" if want_date else "")
                    + " in the museum")
        return {**sub, "card_text": text}

    # ── zip / unzip: the artifact's own pack and unpack ─────────────
    # Stdlib zipfile, the Archivist's contract (skip-and-report unsafe
    # members, bomb caps, atomic write): the Archivist itself is
    # workspace-confined, and museum cards live under the home, so
    # the museum packs its own artifacts — ``nm zip`` / ``zip_create``
    # stay the workspace path.

    def _resolve_card(self, ref: str) -> dict[str, Any]:
        card = self.find(ref)
        if card is None:
            raise ValueError(
                f"no single card matches {ref!r} — /museum lists "
                "them, use the full id")
        return card

    @staticmethod
    def _card_source(card: dict[str, Any]) -> Path:
        """The real file-or-dir behind a card. Exhibit-only cards
        (digests, sources) are refused in words — never the store
        file behind them — and a card whose file raced away is
        refused, never zipped as an empty name."""
        path = Path(str(card.get("path") or ""))
        if card.get("sendable") is False and not path.is_dir():
            raise ValueError(
                f"that's exhibit-only — {card.get('title')} "
                f"[{card.get('id')}] has no file to pack "
                f"({card.get('kind')} cards point at their owning "
                "store, they never ship it)")
        if not path.exists():
            raise ValueError(
                f"the file for {card.get('title')} "
                f"[{card.get('id')}] is gone from the museum — "
                "nothing left to pack")
        return path

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
        return (slug or "artifact")[:60]

    def _store_root(self, what: str) -> Path:
        if self.museum_root is None:
            raise ValueError(
                f"museum has no store root — {what} needs a museum "
                "root to write into (the tools and the console pass "
                "one; a bare Museum() does not have it)")
        return self.museum_root

    def zip_card(self, ref: str, dest_name: str = "") -> dict[str, Any]:
        """Pack one card — a file, or a whole book directory — into
        ``<museum>/zips/<name>.zip``. The zip is itself a museum card
        on the next census (the museum-store exhibitor), so the
        packed artifact never lives in a side index that drifts."""
        store = self._store_root("zip")
        card = self._resolve_card(ref)
        src = self._card_source(card)
        if src.is_dir():
            members = sorted(f for f in src.rglob("*") if f.is_file())
            arc_names = [str(f.relative_to(src.parent)) for f in members]
        else:
            members, arc_names = [src], [src.name]
        total = 0
        for f in members:
            try:
                total += f.stat().st_size
            except OSError:
                continue
        if len(members) > _ZIP_MAX_FILES or total > _ZIP_MAX_BYTES:
            raise ValueError(
                f"refusing to zip {card.get('title')} — "
                f"{len(members)} files / {total} bytes trips the "
                f"museum caps ({_ZIP_MAX_FILES} files / "
                f"{_ZIP_MAX_BYTES} bytes)")
        zips = store / "zips"
        zips.mkdir(parents=True, exist_ok=True)
        name = self._slug(dest_name or
                          f"{card.get('title') or card.get('id')}-"
                          f"{time.strftime('%Y-%m-%d')}")
        dest = zips / f"{name}.zip"
        tmp = dest.with_name(
            f"{dest.name}.{int(time.time())}.tmp")
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as zf:
                for f, arc in zip(members, arc_names, strict=True):
                    zf.write(f, arc)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, dest)
        return {"card": card["id"], "title": card.get("title"),
                "zip": str(dest), "files": len(members),
                "input_bytes": total,
                "archive_bytes": dest.stat().st_size}

    def unzip_card(self, ref: str, dest_name: str = "") -> dict[str, Any]:
        """Unpack a zip card into ``<museum>/unpacked/<stem>/``. A
        member whose name would escape the destination (absolute,
        ``..`` — backslashes count as separators) is SKIPPED and
        named in ``skipped``; the good members still land. The whole
        unpack is refused only when nothing could be extracted or a
        bomb cap trips — checked before anything is written."""
        store = self._store_root("unzip")
        card = self._resolve_card(ref)
        path = Path(str(card.get("path") or ""))
        if not path.is_file() or path.suffix.lower() != ".zip":
            raise ValueError(
                f"that's not a zip archive — {card.get('title')} "
                f"[{card.get('id')}] is a {card.get('kind')} card "
                "(museum unzip handles .zip; nm zip extract handles "
                "the rest)")
        dest = store / "unpacked" / self._slug(dest_name or path.stem)
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            total = sum(i.file_size for i in infos)
            if len(infos) > _ZIP_MAX_FILES or total > _ZIP_MAX_BYTES:
                raise ValueError(
                    f"refusing to unpack {path.name} — "
                    f"{len(infos)} files / {total} bytes trips the "
                    f"museum caps ({_ZIP_MAX_FILES} files / "
                    f"{_ZIP_MAX_BYTES} bytes)")
            skipped: list[str] = []
            written: list[str] = []
            dest.mkdir(parents=True, exist_ok=True)
            dest_root = dest.resolve()
            for info in infos:
                name = info.filename
                parts = [p for p in re.split(r"[\\/]+", name) if p not in ("", ".")]
                unsafe = (not parts or ".." in parts
                          or name.startswith(("/", "\\"))
                          or (len(parts) > 1 and parts[0].endswith(":")))
                if unsafe:
                    skipped.append(name)
                    continue
                target = (dest_root / "/".join(parts)).resolve()
                try:
                    target.relative_to(dest_root)
                except ValueError:
                    skipped.append(name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src_f, target.open("wb") as dst_f:
                    while True:
                        chunk = src_f.read(1024 * 1024)
                        if not chunk:
                            break
                        dst_f.write(chunk)
                written.append("/".join(parts))
        if not written and skipped:
            shown = ", ".join(repr(s) for s in skipped[:5])
            raise ValueError(
                f"unzip: refusing {path.name}: every file entry was "
                f"a zip-slip / unsafe path ({len(skipped)} skipped: "
                f"{shown}); nothing could be extracted")
        return {"archive": str(path), "card": card["id"],
                "destination": str(dest), "files": len(written),
                "skipped": skipped[:50],
                "skipped_count": len(skipped),
                "paths": written[:200],
                "truncated": len(written) > 200}

    # ── companions: books ↔ artifacts ────────────────────────────────
    # A link store, never a file index: the census never trusts it
    # for file existence — ``cite`` re-resolves every link live, and
    # a vanished artifact reports missing instead of silently
    # dropping out of the book's story.

    def _companions_file(self) -> Path | None:
        if self.museum_root is None:
            return None
        return self.museum_root / "companions.json"

    def _load_companions(self) -> list[dict[str, Any]]:
        if self._companions_loaded:
            return self._companions
        self._companions_loaded = True
        path = self._companions_file()
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._companions = list(data.get("companions") or [])
            except (OSError, ValueError) as exc:
                record_failure("museum.companions.load", exc)
        return self._companions

    def _save_companions(self) -> None:
        path = self._companions_file()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"version": 1, "companions": self._companions}, indent=2),
            encoding="utf-8")
        os.replace(tmp, path)

    def attach_companion(self, book_slug: str, artifact_ref: str,
                         note: str = "") -> dict[str, Any]:
        """Link an artifact to a book — the artifact becomes the
        book's companion (theme music, source render, illustration).
        The book must exist in the attached library and the artifact
        must resolve to one card; re-attaching updates the note."""
        slug = (book_slug or "").strip()
        ref = (artifact_ref or "").strip()
        if not slug or not ref:
            raise ValueError(
                "attach needs a book slug and an artifact id — "
                "/museum attach <slug> <id> [note]")
        if self.library is not None:
            slugs = {b.get("slug") for b in self.library.list_books()}
            if slug not in slugs:
                raise ValueError(
                    f"no book {slug!r} in the library — ingest it "
                    "first, then attach its companions")
        card = self.find(ref)
        if card is None:
            raise ValueError(
                f"no single card matches {ref!r} — /museum lists "
                "them, use the full id")
        comps = self._load_companions()
        note = (note or "").strip()
        for entry in comps:
            if entry.get("book") == slug and \
                    entry.get("artifact") == card["id"]:
                entry["note"] = note
                self._save_companions()
                return {"book": slug, "artifact": card["id"],
                        "note": note, "updated": True}
        comps.append({
            "book": slug, "artifact": card["id"], "note": note,
            "attached_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime())})
        self._save_companions()
        return {"book": slug, "artifact": card["id"], "note": note,
                "attached": True}

    def detach_companion(self, book_slug: str,
                         artifact_ref: str) -> dict[str, Any]:
        slug = (book_slug or "").strip()
        ref = (artifact_ref or "").strip()
        comps = self._load_companions()
        for entry in list(comps):
            if entry.get("book") == slug and \
                    entry.get("artifact") == ref:
                comps.remove(entry)
                self._save_companions()
                return {"book": slug, "artifact": ref, "removed": ref}
        raise ValueError(
            f"{ref!r} is not a companion of {slug!r}")

    def companions_for_book(self, slug: str) -> list[dict[str, Any]]:
        """A book's companions, each with its live card — or
        ``missing`` when the artifact has left the museum."""
        out: list[dict[str, Any]] = []
        for entry in self._load_companions():
            if entry.get("book") != slug:
                continue
            card = self.find(str(entry.get("artifact") or ""))
            out.append({"book": slug,
                        "artifact": str(entry.get("artifact") or ""),
                        "note": str(entry.get("note") or ""),
                        "attached_at": str(entry.get("attached_at") or ""),
                        "card": card, "missing": card is None})
        return out

    def companions_for_artifact(self, card_id: str) -> list[dict[str, Any]]:
        return [{"book": str(e.get("book") or ""),
                 "note": str(e.get("note") or ""),
                 "attached_at": str(e.get("attached_at") or "")}
                for e in self._load_companions()
                if e.get("artifact") == card_id]

    def cite(self, slug: str) -> dict[str, Any]:
        """Everything a book cites: its companion artifacts, resolved
        live from the census. A companion whose file vanished is
        named under ``missing`` — the citation never pretends the
        artifact is still on the shelf."""
        entries = self.companions_for_book(slug)
        return {"book": slug,
                "artifacts": [e["card"] for e in entries
                              if e["card"] is not None],
                "count": sum(1 for e in entries if e["card"] is not None),
                "missing": [e["artifact"] for e in entries
                            if e["missing"]]}

    def render(self, res: dict[str, Any] | None = None) -> str:
        res = res if res is not None else self.census()
        arts = res.get("artifacts") or []
        if not arts:
            return "museum is empty — ingest a book, grow a character, or land a build and its cards land here"
        by_shelf = ", ".join(
            f"{k}: {v}" for k, v in sorted((res.get("by_shelf") or {}).items()))
        lines = [f"🏛️ **Museum** — {len(arts)} ({by_shelf})"]
        for card in arts:
            sendable = "sendable" if card.get("sendable") else "exhibit only"
            lines.append(
                f"  [{card['id']}] {card['title']} — {card['kind']} · "
                f"{sendable} · {card['path']}")
        for name, err in sorted((res.get("errors") or {}).items()):
            lines.append(f"  ⚠️ shelf {name} failed: {err}")
        return "\n".join(lines)
