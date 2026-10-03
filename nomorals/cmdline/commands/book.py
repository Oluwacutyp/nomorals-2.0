"""``nm book`` — books surfaces."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from ..emit import _emit



def _cmd_book(args: argparse.Namespace, context: Any) -> int:
    """`nm book <list|create|run|status|build>` — the BookForge CLI surface."""
    from ...books import BookForge
    from ...books.model import slugify

    forge = BookForge(context)
    action = getattr(args, "action", "list") or "list"

    if action == "list":
        books = forge.list_books()
        lines = [f"  {b['slug']:<28} {b['title'][:50]:<52} "
                 f"{b['written']}/{b['chapters']} ch · {b['words']}w · {b['status']}"
                 for b in books]
        _emit(args, {"books": books},
              "books:\n" + "\n".join(lines) if lines else "no books yet — nm book create \"<topic>\"")
        return 0

    if action == "create":
        topic = (getattr(args, "topic", "") or "").strip()
        if not topic:
            print("book create needs a topic — nm book create \"<topic>\"", file=sys.stderr)
            return 2
        book = forge.create(
            topic,
            chapters=int(getattr(args, "chapters", 0) or 0),
            words_per_chapter=int(getattr(args, "words", 0) or 0),
            research=not getattr(args, "no_research", False),
        )
        plan_note = (f"({len(book.chapters)} chapters planned)"
                     if not book.organic
                     else "(organic — grows as it's written)")
        _emit(args, {"slug": book.slug, "title": book.display_title,
                      "chapters": len(book.chapters), "organic": book.organic},
              f"created {book.slug} — “{book.display_title}” {plan_note}")
        return 0

    if action == "run":
        topic = (getattr(args, "topic", "") or "").strip()
        if not topic:
            print("book run needs a topic — nm book run \"<topic>\"", file=sys.stderr)
            return 2

        def _progress(result: dict[str, Any]) -> None:
            if not getattr(args, "json", False):
                print(f"  chapter {result.get('chapter')}: "
                      f"{str(result.get('note') or result.get('status') or '')[:80]}")

        result = forge.run(
            topic,
            chapters=int(getattr(args, "chapters", 0) or 0),
            words_per_chapter=int(getattr(args, "words", 0) or 0),
            research=not getattr(args, "no_research", False),
            on_chapter=_progress,
        )
        _emit(args, result,
              f"book built: {result.get('slug')} → {result.get('pdf', result.get('manuscript', ''))}")
        return 0

    # status / build need a slug
    slug = (getattr(args, "slug", "") or "").strip()
    if not slug:
        topic = (getattr(args, "topic", "") or "").strip()
        slug = slugify(topic) if topic else ""
    if not slug:
        print("book status/build needs --slug (or a topic) — nm book status --slug <slug>",
              file=sys.stderr)
        return 2
    try:
        book = forge.load(slug)
    except Exception as exc:  # noqa: BLE001
        print(f"book: {exc}", file=sys.stderr)
        return 1

    if action == "status":
        payload = {"slug": book.slug, "title": book.display_title,
                   "status": book.status, "chapters": len(book.chapters),
                   "written": book.chapters_written, "words": book.total_words}
        lines = [f"  chapter {c.number}: {c.title[:50]:<52} [{c.status}] {c.words}w"
                 for c in book.chapters]
        _emit(args, payload,
              f"{book.slug} — “{book.display_title}” [{book.status}]\n"
              f"chapters: {book.chapters_written}/{len(book.chapters)} written, "
              f"{book.total_words} words\n" + "\n".join(lines))
        return 0

    if action == "build":
        result = forge.build(slug)
        _emit(args, result,
              f"built {slug}: {result.get('pdf', result.get('manuscript', ''))}")
        return 0

    print(f"book: unknown action {action}", file=sys.stderr)
    return 2
