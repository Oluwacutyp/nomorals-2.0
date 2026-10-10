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

    # ── story reader + fiction studio ────────────────────────────────
    from ...books.reader import StoryReader
    reader = StoryReader(context)

    if action == "search-stories":
        query = (getattr(args, "topic", "") or "").strip()
        if not query:
            print("book search-stories needs a title — nm book search-stories \"My Vampire System\"",
                  file=sys.stderr)
            return 2
        hits = reader.search(query, limit=10)
        lines = [f"  [{h['source']}] {h['title'][:60]:<62} {h['url']}" for h in hits]
        _emit(args, {"query": query, "hits": hits},
              f"{len(hits)} hit(s) for {query!r}:\n" + "\n".join(lines)
              if lines else f"no stories found for {query!r}")
        return 0

    if action == "follow":
        target = (getattr(args, "topic", "") or "").strip()
        if not target:
            print("book follow needs a URL or title", file=sys.stderr)
            return 2
        story = reader.follow(target)
        _emit(args, {"story": story.to_dict()},
              f"following {story.slug} — “{story.title}” "
              f"[{story.source}] {story.total_chapters or '?'} chapters")
        return 0

    if action == "following":
        stories = reader.following()
        lines = [f"  {s['slug']:<30} {s['title'][:48]:<50} "
                 f"{s['progress']:<12} [{s['source']}]" for s in stories]
        _emit(args, {"stories": stories},
              "followed stories:\n" + "\n".join(lines) if lines
              else "not following any stories — nm book follow \"<title>\"")
        return 0

    if action in ("read", "next", "prev"):
        rslug = (getattr(args, "slug", "") or "").strip() or \
            (getattr(args, "topic", "") or "").strip()
        if not rslug:
            print(f"book {action} needs --slug", file=sys.stderr)
            return 2
        try:
            if action == "read":
                ch_no = int(getattr(args, "chapters", 0) or 0)
                result = reader.read(rslug, chapter=ch_no)
            elif action == "next":
                result = reader.next(rslug)
            else:
                result = reader.prev(rslug)
        except Exception as exc:  # noqa: BLE001
            print(f"book {action}: {exc}", file=sys.stderr)
            return 1
        text = result.pop("text", "")
        _emit(args, result,
              f"“{result['title']}” ch.{result['chapter']}: "
              f"{result['chapter_title']} ({result['words']}w)\n\n{text[:3000]}")
        return 0

    if action == "sync":
        rslug = (getattr(args, "slug", "") or "").strip()
        result = reader.sync(rslug,
                             chapters=int(getattr(args, "chapters", 0) or 0))
        lines = [f"  {s['slug']}: fetched {s['fetched'] or 'none'} "
                 f"(total {s['new_total'] or '?'})" for s in result["synced"]]
        _emit(args, result, "synced:\n" + "\n".join(lines))
        return 0

    if action == "progress":
        rslug = (getattr(args, "slug", "") or "").strip()
        if not rslug:
            print("book progress needs --slug", file=sys.stderr)
            return 2
        ch_no = int(getattr(args, "chapters", 0) or 0)
        if ch_no >= 1:
            result = reader.set_progress(rslug, ch_no)
            verb = "set"
        else:
            result = reader.resume(rslug)
            verb = "resume"
        _emit(args, result,
              f"{verb}: “{result['title']}” — chapter {result['chapter']} "
              f"of {result['total_chapters'] or '?'}")
        return 0

    if action == "bookmark":
        from ...books.reader import ReaderError
        rslug = (getattr(args, "slug", "") or "").strip()
        if not rslug:
            print("book bookmark needs --slug", file=sys.stderr)
            return 2
        ch_no = int(getattr(args, "chapters", 0) or 0)
        try:
            if ch_no >= 1:
                result = reader.add_bookmark(rslug, ch_no,
                                             label=getattr(args, "topic", ""))
                mark = result["bookmark"]
                _emit(args, result,
                      f"bookmarked #{mark['id']}: ch.{mark['chapter']} "
                      f"“{mark['label']}”")
            else:
                marks = reader.list_bookmarks(rslug)
                lines = [f"  #{m['id']} ch.{m['chapter']} {m['label']}"
                         for m in marks]
                _emit(args, {"bookmarks": marks},
                      "bookmarks:\n" + "\n".join(lines) if lines
                      else "no bookmarks")
        except ReaderError as exc:
            print(f"book bookmark: {exc}", file=sys.stderr)
            return 1
        return 0

    if action == "bible":
        from ...books.continuation import StoryContinuer
        rslug = (getattr(args, "slug", "") or "").strip()
        if not rslug:
            print("book bible needs --slug", file=sys.stderr)
            return 2
        try:
            bible = StoryContinuer(context).prepare(
                rslug, chapters=int(getattr(args, "chapters", 0) or 6))
        except Exception as exc:  # noqa: BLE001
            print(f"book bible: {exc}", file=sys.stderr)
            return 1
        _emit(args, {"bible": bible.to_dict()}, bible.brief()[:3000])
        return 0

    if action == "continue":
        from ...books.continuation import StoryContinuer
        rslug = (getattr(args, "slug", "") or "").strip()
        if not rslug:
            print("book continue needs --slug", file=sys.stderr)
            return 2
        try:
            result = StoryContinuer(context).continue_story(
                rslug, n=int(getattr(args, "chapters", 0) or 1),
                words=int(getattr(args, "words", 0) or 1500),
                direction=getattr(args, "direction", "") or "")
        except Exception as exc:  # noqa: BLE001
            print(f"book continue: {exc}", file=sys.stderr)
            return 1
        for ch in result["chapters"]:
            print(f"── {ch['title']} ({ch['words']}w) → {ch['path']}")
            if not getattr(args, "json", False):
                print(ch["text"][:2500])
                print()
        _emit(args, {"slug": result["slug"],
                     "chapters": [{"number": c["number"], "title": c["title"],
                                   "words": c["words"], "path": c["path"]}
                                  for c in result["chapters"]]},
              "")
        return 0

    if action == "fiction":
        from ...books.fiction import FictionWriter
        premise = (getattr(args, "topic", "") or "").strip()
        if not premise:
            print("book fiction needs a premise — nm book fiction \"...\" --genre mystery --mode novel",
                  file=sys.stderr)
            return 2
        try:
            result = FictionWriter(context).start(
                premise, genre=getattr(args, "genre", "") or "fantasy",
                mode=getattr(args, "mode", "") or "novel",
                theme=getattr(args, "theme", "") or "")
        except Exception as exc:  # noqa: BLE001
            print(f"book fiction: {exc}", file=sys.stderr)
            return 1
        ch1 = result.pop("chapter_1", {})
        print(f"started {result['slug']} — “{result['title']}” "
              f"[{result['genre']}/{result['mode']}]")
        print(f"shape: {result['shape']}")
        if not getattr(args, "json", False):
            print(f"\n── chapter 1 ({ch1.get('words', 0)}w)\n")
            print((ch1.get("text", "") or "")[:3000])
        _emit(args, {**result, "chapter_1_path": ch1.get("path", "")}, "")
        return 0

    if action == "fiction-write":
        from ...books.fiction import FictionWriter
        rslug = (getattr(args, "slug", "") or "").strip()
        if not rslug:
            print("book fiction-write needs --slug", file=sys.stderr)
            return 2
        try:
            result = FictionWriter(context).write_next(
                rslug, direction=getattr(args, "direction", "") or "")
        except Exception as exc:  # noqa: BLE001
            print(f"book fiction-write: {exc}", file=sys.stderr)
            return 1
        print(f"── chapter {result['number']} [{result['arc']}] "
              f"({result['words']}w) → {result['path']}")
        if result.get("validation_notes") and not getattr(args, "json", False):
            print("validation notes:", "; ".join(result["validation_notes"]))
        if not getattr(args, "json", False):
            print()
            print(result["text"][:3000])
        _emit(args, {k: v for k, v in result.items() if k != "text"}, "")
        return 0

    if action == "fiction-status":
        from ...books.fiction import FictionWriter
        rslug = (getattr(args, "slug", "") or "").strip()
        if not rslug:
            print("book fiction-status needs --slug", file=sys.stderr)
            return 2
        try:
            st = FictionWriter(context).status(rslug)
        except Exception as exc:  # noqa: BLE001
            print(f"book fiction-status: {exc}", file=sys.stderr)
            return 1
        lines = [f"  {a['name']:<44} [{a['status']}]" for a in st["arcs"]]
        _emit(args, st,
              f"{st['slug']} — “{st['title']}” [{st['genre']}/{st['mode']}]\n"
              f"chapters: {st['chapters_written']}/{st['total_planned'] or '?'} "
              f"· current arc: {st['current_arc']}\n"
              f"open threads: {len(st['open_threads'])}\n" + "\n".join(lines))
        return 0

    if action not in ("status", "build"):
        print(f"book: unknown action {action}", file=sys.stderr)
        return 2

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
