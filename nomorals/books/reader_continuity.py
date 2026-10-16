"""Reader continuity — series tracking, catch-up digests, chapter
summaries, and the 400-chapter return.

Extends (does not replace) books/reader.py: StoryReader owns follow /
fetch / cache; this module owns *coming back* — what happened, where
you left off, what to re-read.

Summaries are extractive (TextRank-lite sentence centrality + position
bonus over the chapter's own sentences): no model invention, no
hallucinated plot. A catch-up digest for a long-running serial is
therefore a compressed re-read of real chapters, and a 'previously on'
digest pairs the last N chapter summaries with the bible's open threads
when a bible file exists for the story.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "extractive_summary", "chapter_summary", "catch_up_digest",
    "group_by_series", "return_after_break",
]

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an and or of to in on at for with is was were be been are "
    "had has have it its they them he she his her we you your this that "
    "as by from".split())
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"“‘(])")
_VOLUME_RE = re.compile(r"\b(?:book|volume|vol\.?)\s*(\d+)", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    flat = " ".join(p.strip() for p in re.split(r"\n\s*\n", text or "")
                    if p.strip())
    return [s.strip() for s in _SENT_SPLIT.split(flat) if len(s.strip()) > 20]


def _tokens(sentence: str) -> set[str]:
    return {t for t in _TOKEN.findall(sentence.lower()) if t not in _STOP}


def extractive_summary(text: str, *, max_sentences: int = 5) -> str:
    """Extractive summary: most central sentences, in original order.

    Centrality = shared-word overlap with the rest of the chapter
    (TextRank-lite, no model). Returns the chapter's own sentences.
    """
    sentences = _sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    token_sets = [_tokens(s) for s in sentences]
    scores: list[float] = []
    for i, toks in enumerate(token_sets):
        overlap = 0.0
        for j, other in enumerate(token_sets):
            if i == j or not toks or not other:
                continue
            overlap += len(toks & other) / len(toks | other)
        # position bonus: opening sentences carry setup
        position = 0.15 if i < 2 else 0.0
        scores.append(overlap + position)
    ranked = sorted(range(len(sentences)),
                    key=lambda i: (-scores[i], i))[:max_sentences]
    return " ".join(sentences[i] for i in sorted(ranked))


def chapter_summary(text: str, *, chapter: int = 0,
                    title: str = "", max_sentences: int = 5) -> dict[str, Any]:
    """Summary record for one chapter (extractive, citable)."""
    words = len(re.findall(r"[A-Za-z']+", text or ""))
    return {"chapter": chapter, "title": title, "words": words,
            "summary": extractive_summary(text,
                                          max_sentences=max_sentences)}


def _cached_chapters(reader: Any, slug: str) -> list[tuple[int, str]]:
    """Read cached chapter files from the reader's own directory layout
    (``<story>/chapters/NNNNN.md``) — no network, no private API."""
    story_dir = Path(reader.story_dir(slug))
    chapters_dir = story_dir / "chapters"
    out: list[tuple[int, str]] = []
    if not chapters_dir.exists():
        return out
    for path in sorted(chapters_dir.glob("*.md")):
        try:
            number = int(path.stem)
        except ValueError:
            continue
        try:
            out.append((number, path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return sorted(out)


def catch_up_digest(reader: Any, slug: str, *,
                    max_chapters: int = 10,
                    max_sentences: int = 3) -> dict[str, Any]:
    """Catch-up digest: per-chapter summaries for the unread tail.

    Starts after the reader's current chapter (from the follow record)
    and covers up to ``max_chapters`` cached chapters. Chapters that are
    not cached are listed as gaps — never summarized from nothing.
    """
    try:
        story = reader._load(slug)  # same record StoryReader itself uses
        current = int(getattr(story, "current_chapter", 0) or 0)
        title = getattr(story, "title", slug)
        total = int(getattr(story, "total_chapters", 0) or 0)
    except Exception:  # noqa: BLE001 - unknown story → honest error dict
        return {"ok": False, "slug": slug,
                "reason": f"no followed story {slug!r}"}
    cached = dict(_cached_chapters(reader, slug))
    start = current + 1
    end = min(total or start + max_chapters - 1,
              start + max_chapters - 1) if total else start + max_chapters - 1
    summaries: list[dict[str, Any]] = []
    gaps: list[int] = []
    for number in range(start, end + 1):
        if number in cached:
            summaries.append(chapter_summary(
                cached[number], chapter=number,
                title=f"Chapter {number}", max_sentences=max_sentences))
        else:
            gaps.append(number)
    return {"ok": True, "slug": slug, "title": title,
            "current_chapter": current, "total_chapters": total,
            "covered": [s["chapter"] for s in summaries],
            "gaps_not_cached": gaps,
            "summaries": summaries,
            "combined": " ".join(s["summary"] for s in summaries)}


def group_by_series(stories: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group followed stories into series by title prefix.

    Heuristic, stated as such: strips a trailing ``Book/Volume N`` or
    ``: subtitle`` marker; stories sharing the stripped prefix group
    together. A story with no marker is its own single-volume series.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for story in stories:
        title = str(story.get("title", ""))
        base = _VOLUME_RE.sub("", title).strip(" :—-").strip()
        base = base.split(":")[0].strip() or title
        groups.setdefault(base, []).append(story)
    for members in groups.values():
        members.sort(key=lambda s: str(s.get("title", "")))
    return groups


def return_after_break(reader: Any, slug: str, *,
                       last_n: int = 5) -> dict[str, Any]:
    """'Previously on' for returning after a long break (e.g. ch.400).

    Last ``last_n`` cached chapters, summarized, plus the bible's open
    threads and cast when a bible file exists for the story — otherwise
    an explicit ``bible: null`` instead of an invented recap.
    """
    cached = _cached_chapters(reader, slug)
    if not cached:
        return {"ok": False, "slug": slug,
                "reason": "no cached chapters to summarize"}
    tail = cached[-last_n:]
    summaries = [chapter_summary(text, chapter=n,
                                 title=f"Chapter {n}", max_sentences=3)
                 for n, text in tail]
    bible: dict[str, Any] | None = None
    try:
        workspace = Path(reader._workspace())
    except Exception:  # noqa: BLE001
        workspace = Path.cwd() / "workspace"
    bible_path = workspace / "books" / "bibles" / f"{slug}.json"
    if bible_path.exists():
        try:
            bible = json.loads(bible_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            bible = None
    open_threads: list[str] = []
    cast: list[str] = []
    if bible:
        open_threads = [t.get("summary", "") for t in
                        bible.get("threads", [])
                        if t.get("status") == "open"][:10]
        cast = [c.get("name", "") for c in
                bible.get("characters", [])][:20]
    return {"ok": True, "slug": slug,
            "chapters_covered": [s["chapter"] for s in summaries],
            "summaries": summaries,
            "combined": " ".join(s["summary"] for s in summaries),
            "open_threads": open_threads, "cast": cast,
            "bible": bool(bible)}
