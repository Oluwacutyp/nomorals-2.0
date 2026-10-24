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
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

__all__ = ["Museum", "museum_card", "apkforge_exhibitor",
           "library_exhibitor", "characters_exhibitor"]


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
    except OSError:
        pass
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
        except (OSError, ValueError):
            pass
        cards.append(museum_card(
            id=f"library-book:{slug}",
            title=f"{book['title']} ({book['chapters']} chapters)",
            kind="library-book", shelf="library-books",
            path=str(original) if original else str(book_dir),
            hash=digest,
            size=_dir_size(book_dir),
            provenance={"type": "ingest", "ref": slug, "tool": "library"},
            tags=["library", "book"] + list(book.get("tags") or [])))
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
        except Exception:  # noqa: BLE001 — one bad doc never kills census
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


# ── the museum ────────────────────────────────────────────────────────

class Museum:
    """One census, many systems. Roots are explicit (no guessed paths):
    pass what you have; missing systems simply exhibit nothing."""

    def __init__(self, *, apk_root: Path | str | None = None,
                 library: Any = None, char_store: Any = None,
                 learn_store: Any = None,
                 learn_dir: Path | str | None = None) -> None:
        self.apk_root = Path(apk_root) if apk_root else None
        self.library = library
        self.char_store = char_store
        self.learn_store = learn_store
        self.learn_dir = learn_dir
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
