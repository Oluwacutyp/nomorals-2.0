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
from .fiction import FictionWriter, StoryCompleteError
from .reader import StoryReader
from .voiceprint import Voiceprint, compare, fingerprint

_log = get_logger(__name__)

__all__ = ["SerialManager", "SerialError"]

#: Chapters written per ``write_novel`` run when no smaller batch is
#: asked for.  A batching limit for ONE run (the caller runs again to
#: continue) — never the story's length, which the story decides.
WRITE_RUN_BATCH = 30


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
        """Start a complete novel. ``chapters`` 0 = the story decides
        its own length (its arcs running out ends it; the premise-
        complexity estimate is recorded as advisory metadata only);
        a positive value is the owner's explicit shape and is planned
        exactly."""
        result = self.fiction.start(
            premise, genre=genre, mode="novel",
            chapters=int(chapters or 0),
            theme=wisdom_theme or "")
        slug = result["slug"]
        state = self.fiction.load(slug)
        meta = self._load_meta(slug)
        meta.update({
            "kind": "novel",
            "for_whom": for_whom or {},
            "wisdom_theme": wisdom_theme or "",
            "reference_voiceprint": None,
            # honest metadata: an explicit owner count, else the
            # advisory estimate (never recorded as the owner's number)
            "planned_chapters": (int(chapters) if chapters
                                 else state.total_planned),
            "shape_source": "owner" if chapters else "estimate",
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
        """Write a novel onward and report honestly what stopped the run.

        Who decides the end is never a silent hardcoded count:
        * a novel ends when its STORY ends — its arcs are all served
          (``StoryCompleteError`` from the writer) → ``done: True``;
        * the owner's explicit chapter count (``shape_source="owner"``)
          is a decision, honored exactly → ``done: True``;
        * the premise-complexity estimate (``shape_source="estimate"``)
          never stops the writing;
        * a serial never ends (never-ending) — one run writes one batch
          and says so plainly → ``done: False``.
        ``max_chapters`` caps a single run on the owner's explicit
        instruction — that is batching, not the story's length.  A run
        that hits the per-run batch limit simply continues next call.
        """
        state = self.fiction.load(slug)
        serial = state.mode == "serial"
        owner_target = (state.total_planned
                        if (not serial and state.shape_source == "owner")
                        else 0)
        batch = max(1, int(max_chapters or 0)) if max_chapters else WRITE_RUN_BATCH
        written: list[dict[str, Any]] = []
        done = False
        stopped_because = ""
        while len(written) < batch:
            if owner_target and state.chapters_written >= owner_target:
                done = True
                stopped_because = (
                    f"owner-set shape reached ({owner_target} chapters)")
                break
            try:
                chapter = self.write_next(slug)
            except StoryCompleteError:
                done = True
                stopped_because = ("the story's arcs are all served — "
                                   "it reached its own end")
                break
            written.append({"number": chapter.get("number"),
                            "words": chapter.get("words"),
                            "drift": (chapter.get("drift") or {}).get("drift"),
                            "path": chapter.get("path")})
            state = self.fiction.load(slug)
        if not done:
            # exact batch edge: the final arc may have closed on the last
            # written chapter — the story is still complete even though
            # the writer never got to say so.
            if (not serial) and self.fiction.story_complete(state):
                done = True
                stopped_because = ("the story's arcs are all served — "
                                   "it reached its own end")
            elif serial:
                stopped_because = ("serial — never-ending; wrote one batch "
                                   f"of {len(written)} chapters, the story "
                                   "continues")
            else:
                stopped_because = (f"run batch limit ({batch} chapters) — "
                                   "the story is not finished; run again "
                                   "to continue")
        meta = self._load_meta(slug)
        return {
            "slug": slug, "title": state.title,
            "kind": meta.get("kind", state.mode),
            "chapters_written": state.chapters_written,
            "total_planned": state.total_planned,
            "planned_is_estimate": (not serial
                                    and state.shape_source != "owner"),
            "for_whom": meta.get("for_whom", {}),
            "chapters": written,
            "drift_log": meta.get("drift_log", []),
            "done": done,
            "stopped_because": stopped_because,
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
        """Continue a followed story past its ending (or from any chapter).

        By default the continuation anchors at the story's TRUE END:
        the library is ingested whole and the bible digested through
        the final chapter before writing.  ``from_chapter`` is the
        explicit fork override — it is handed to the continuer as an
        anchor (the reader's progress is NOT moved to fake one).
        Each new chapter is scored against the reference voice built
        from the story's own early chapters. Drift is reported, never
        blocking."""
        story = self.reader._load(slug)
        slug = story.slug
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
            slug, n=max(1, n), direction=direction,
            from_chapter=from_chapter)
        scored = []
        for ch in result["chapters"]:
            drift = None
            if ref is not None and ref.words > 0:
                drift = compare(fingerprint(ch["text"]), ref)
            scored.append({**ch, "drift": drift})
        return {
            "slug": slug, "title": story.title,
            "continued_from_chapter": result.get(
                "continued_from_chapter", story.current_chapter),
            "chapters": scored,
            "bible": result.get("bible"),
        }
