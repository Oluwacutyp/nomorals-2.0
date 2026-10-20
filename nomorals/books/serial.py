"""Serials + complete novels — orchestration over the existing engines.

* **Never-ending serials**: chapter-by-chapter via
  :class:`FictionWriter` (or :class:`StoryContinuer` for an existing
  read story). After every chapter the story bible is updated, the
  voiceprint drift is measured against the reference voice, and the
  drift is logged — a serial that drifts is named, never silently
  rewritten.
* **Complete novels**: outline → chapters → craft passes →
  voiceprint → read-aloud → ``for:`` field, via the same engines.
* **Continue past the ending**: a read story's final chapters seed a
  bible; new chapters continue from any chapter, in the story's voice,
  with per-chapter drift scores.
* **WisdomKeeper depth**: ``FictionWriter`` already weaves wisdom
  motifs (``weave_wisdom``); this module passes the theme through and
  records which motifs each chapter used.

This module orchestrates; it does not re-implement writing, bible
building, craft passes, or voiceprints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .bible import BibleBuilder
from .continuation import StoryContinuer
from .fiction import FictionWriter
from .reader import StoryReader
from .voiceprint import Voiceprint, compare, fingerprint

_log = get_logger(__name__)

__all__ = ["SerialManager", "SerialError"]


class SerialError(Exception):
    """Serial / novel orchestration could not proceed."""


def _read_chapter_text(path: str | Path) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 - missing file, honest empty
        return ""
    # strip a leading "# title" + attribution line when present
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


class SerialManager:
    """Start serials / novels, write chapters, continue past endings."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.fiction = FictionWriter(context)
        self.reader = StoryReader(context)
        self.continuer = StoryContinuer(context)
        self.bibles = BibleBuilder(context)

    # -- serial state ---------------------------------------------------------
    def _meta_path(self, slug: str) -> Path:
        return self.fiction.story_dir(slug) / "serial.json"

    def _load_meta(self, slug: str) -> dict[str, Any]:
        path = self._meta_path(slug)
        if not path.exists():
            return {"slug": slug, "drift_log": [], "wisdom_motifs": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"slug": slug, "drift_log": [], "wisdom_motifs": []}

    def _save_meta(self, slug: str, meta: dict[str, Any]) -> None:
        path = self._meta_path(slug)
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # -- starting ---------------------------------------------------------------
    def start_serial(self, premise: str, *, genre: str = "fantasy",
                      for_whom: dict[str, Any] | None = None,
                      wisdom_theme: str = "") -> dict[str, Any]:
        """Start a never-ending serial. ``for_whom`` is the humane
        ``for:`` field (name / relationship / occasion). It is stored
        in the serial meta file (``serial.json``) and reported by
        :meth:`status`; the fiction engine itself does not take it, so
        the field never changes generated prose."""
        result = self.fiction.start(
            premise, genre=genre, mode="serial",
            theme=wisdom_theme or "")
        slug = result["slug"]
        meta = self._load_meta(slug)
        meta.update({
            "kind": "serial",
            "for_whom": for_whom or {},
            "wisdom_theme": wisdom_theme or "",
            "reference_voiceprint": None,
        })
        self._save_meta(slug, meta)
        return {**result, "for_whom": for_whom or {},
                "kind": "serial"}

    def start_novel(self, premise: str, *, genre: str = "fantasy",
                    chapters: int = 0,
                    for_whom: dict[str, Any] | None = None,
                    wisdom_theme: str = "") -> dict[str, Any]:
        """Start a complete novel. ``chapters`` 0 = the book decides
        its own shape from premise complexity; a positive value is the
        owner's explicit shape and is planned exactly."""
        result = self.fiction.start(
            premise, genre=genre, mode="novel",
            chapters=int(chapters or 0),
            theme=wisdom_theme or "")
        slug = result["slug"]
        meta = self._load_meta(slug)
        meta.update({
            "kind": "novel",
            "for_whom": for_whom or {},
            "wisdom_theme": wisdom_theme or "",
            "reference_voiceprint": None,
            "planned_chapters": chapters or result.get("total_planned"),
        })
        self._save_meta(slug, meta)
        return {**result, "for_whom": for_whom or {}, "kind": "novel"}

    # -- chapter loop -------------------------------------------------------------
    def _reference(self, slug: str, chapter_texts: list[str]) -> Voiceprint:
        meta = self._load_meta(slug)
        stored = meta.get("reference_voiceprint")
        if stored:
            try:
                ref = Voiceprint.from_dict(stored)
                if ref.words > 0:
                    return ref
            except Exception:  # noqa: BLE001 - corrupt meta, rebuild
                pass
        ref = fingerprint("\n\n".join(chapter_texts[:3]))
        meta["reference_voiceprint"] = ref.to_dict()
        self._save_meta(slug, meta)
        return ref

    def write_next(self, slug: str, *, direction: str = "") -> dict[str, Any]:
        """Write the next chapter, update the bible, score drift.

        Returns the FictionWriter chapter result plus ``drift`` (voice
        comparison vs the reference voiceprint, told-not-blocked) and
        ``bible_updated``."""
        result = self.fiction.write_next(slug, direction=direction)
        text = result.get("text", "") or _read_chapter_text(result.get("path", ""))
        number = result.get("number", 0)
        # reference voice: first 3 chapters of this story
        state = self.fiction.load(slug)
        ref_texts: list[str] = []
        for n in range(1, min(4, number)):
            p = self.fiction.story_dir(slug) / "chapters" / f"{n:04d}.md"
            ref_texts.append(_read_chapter_text(p))
        ref_texts = [t for t in ref_texts if t]
        drift: dict[str, Any] | None = None
        if ref_texts and text:
            char_names = [c.get("name", "") for c in state.characters
                          if isinstance(c, dict) and c.get("name")]
            drift = compare(fingerprint(text, char_names),
                            self._reference(slug, ref_texts))
        # bible: digest this chapter (best-effort, never crashes writing)
        bible_updated = False
        try:
            title = result.get("title", f"Chapter {number}")
            existing = self.bibles.load(slug)
            if existing is None:
                self.bibles.build(slug, state.title,
                                  [(number, title, text)])
            else:
                self.bibles.update(slug, [(number, title, text)])
            bible_updated = True
        except Exception as exc:  # noqa: BLE001 - bible is advisory
            _log.debug("serial bible update failed for %s: %s", slug, exc)
        meta = self._load_meta(slug)
        if drift is not None:
            meta.setdefault("drift_log", []).append(
                {"chapter": number, "drift": drift["drift"],
                 "verdict": drift["verdict"]})
            self._save_meta(slug, meta)
        return {**result, "drift": drift, "bible_updated": bible_updated}

    def write_novel(self, slug: str, *, max_chapters: int = 0) -> dict[str, Any]:
        """Write a complete novel to its planned end (or ``max_chapters``
        chapters). Returns per-chapter drift + a completion summary."""
        state = self.fiction.load(slug)
        total = max_chapters or state.total_planned or 0
        written: list[dict[str, Any]] = []
        while total == 0 or state.chapters_written < total:
            chapter = self.write_next(slug)
            written.append({"number": chapter.get("number"),
                            "words": chapter.get("words"),
                            "drift": (chapter.get("drift") or {}).get("drift"),
                            "path": chapter.get("path")})
            state = self.fiction.load(slug)
            if total and state.chapters_written >= total:
                break
            if not total and len(written) >= 30:
                # unplanned novel: cap a single run, owner continues
                break
        meta = self._load_meta(slug)
        return {
            "slug": slug, "title": state.title, "kind": "novel",
            "chapters_written": state.chapters_written,
            "total_planned": state.total_planned,
            "for_whom": meta.get("for_whom", {}),
            "chapters": written,
            "drift_log": meta.get("drift_log", []),
            "done": bool(total) and state.chapters_written >= total,
        }

    def status(self, slug: str) -> dict[str, Any]:
        """Serial / novel status: arcs, threads, cast, drift, for:."""
        state = self.fiction.load(slug)
        meta = self._load_meta(slug)
        base = self.fiction.status(slug)
        return {
            **base,
            "kind": meta.get("kind", state.mode),
            "for_whom": meta.get("for_whom", {}),
            "wisdom_theme": meta.get("wisdom_theme", ""),
            "drift_log": meta.get("drift_log", [])[-10:],
        }

    # -- continue past the ending -------------------------------------------------
    def continue_past_ending(self, slug: str, n: int = 1, *,
                             from_chapter: int = 0,
                             direction: str = "") -> dict[str, Any]:
        """Continue a read story past its ending (or from any chapter).

        The story bible is built from the read chapters; each new
        chapter is scored against the reference voice built from the
        story's own early chapters. Drift is reported, never blocking."""
        story = self.reader._load(slug)
        # continue from any chapter: point the reader at from_chapter
        # first, so prepare() + continue_story() write from there
        if from_chapter and from_chapter >= 1:
            try:
                self.reader.set_progress(slug, from_chapter)
                story = self.reader._load(slug)
            except Exception as exc:  # noqa: BLE001 - progress is advisory
                _log.debug("continue from_chapter progress failed: %s", exc)
        bible = self.continuer.prepare(slug)
        # reference voice from the story's own first chapters
        ref_texts: list[str] = []
        for num in range(1, 4):
            try:
                ref_texts.append(self.reader.chapter_text(slug, num))
            except Exception:  # noqa: BLE001 - chapter unavailable
                continue
        ref = fingerprint("\n\n".join(ref_texts)) if ref_texts else None
        result = self.continuer.continue_story(
            slug, n=max(1, n), direction=direction)
        scored = []
        for ch in result["chapters"]:
            drift = None
            if ref is not None and ref.words > 0:
                drift = compare(fingerprint(ch["text"]), ref)
            scored.append({**ch, "drift": drift})
        return {
            "slug": slug, "title": story.title,
            "continued_from_chapter": from_chapter or story.current_chapter,
            "chapters": scored,
            "bible": result.get("bible"),
        }
