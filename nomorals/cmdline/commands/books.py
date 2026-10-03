"""``nm books`` — the book library surface: ingest, search, read, and the
reading companion (progress, bookmarks, notes, collections, tags, ratings)."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..emit import _emit


def _library(context: Any) -> Any:
    from ...books.library import Library

    return Library(context)


def _cmd_books(args: argparse.Namespace, context: Any) -> int:
    """`nm books <action>` — library CLI surface."""
    from ...books.library import LibraryError

    lib = _library(context)
    action = (getattr(args, "action", "list") or "list").lower()
    target = (getattr(args, "target", "") or "").strip()

    def need(what: str) -> str:
        if not target:
            print(f"books {action} needs {what} — see nm books --help",
                  file=sys.stderr)
            raise SystemExit(2)
        return target

    try:
        if action == "list":
            tag = (getattr(args, "find", "") or "").strip()
            books = (lib.books_with_tag(tag) if tag
                     else [{"slug": b["slug"], "title": b["title"],
                            "author": b["author"], "chapters": b["chapters"],
                            "words": b["words"], "rating": b["rating"],
                            "tags": b["tags"],
                            "progress": b["progress_percent"]}
                           for b in lib.list_books()])
            if tag:
                lines = [f"  {b['slug']:<28} {b['title'][:60]}"
                         for b in books]
                _emit(args, {"tag": tag, "books": books},
                      f"tag {tag!r}:\n" + "\n".join(lines)
                      if lines else f"no books tagged {tag!r}")
                return 0
            lines = [f"  {b['slug']:<28} {b['title'][:48]:<50} "
                     f"{'★' * b['rating'] or '·':<5} "
                     f"{b['progress']:>5}%  {','.join(b['tags'][:3])}"
                     for b in books]
            _emit(args, {"books": books},
                  "library:\n" + "\n".join(lines)
                  if lines else "library is empty — nm books ingest <file>")
            return 0

        if action == "ingest":
            path = need("a file path")
            res = lib.ingest(path, title=getattr(args, "title", "") or "",
                             author=getattr(args, "author", "") or "")
            _emit(args, res.to_dict(),
                  f"ingested {res.slug} — “{res.title}” "
                  f"({res.chapters} chapters, {res.words} words, "
                  f"{res.strategy})")
            return 0

        if action == "search":
            query = need("a search query")
            hits = lib.search(query, top=int(getattr(args, "limit", 8) or 8))
            payload = {"query": query, "hits": [h.to_dict() for h in hits]}
            lines = [f"  [{h.book_slug}] {h.title} — {h.chapter} "
                     f"(score {h.score:.2f})\n    {h.passage[:160]}"
                     for h in hits]
            _emit(args, payload,
                  f"{len(hits)} hit(s) for {query!r}:\n" + "\n".join(lines)
                  if lines else f"no hits for {query!r}")
            return 0

        if action == "read":
            slug = need("a book slug")
            out = lib.read(slug, chapter=int(getattr(args, "chapter", 0) or 0))
            if "outline" in out:
                lines = [f"  {c['n']:>3}. {c['title'][:60]:<62} {c['words']}w"
                         for c in out["outline"]]
                _emit(args, out,
                      f"{out['title']} — {out['chapters']} chapters, "
                      f"{out['total_words']} words\n" + "\n".join(lines))
            else:
                _emit(args, out,
                      f"chapter {out['chapter']}: {out['chapter_title']}\n"
                      f"{out['text']}"
                      f"\n[…truncated]" if out["truncated"] else "")
            return 0

        if action == "drop":
            slug = need("a book slug")
            out = lib.drop(slug)
            _emit(args, out, f"dropped {slug}")
            return 0

        if action == "resume":
            slug = need("a book slug")
            out = lib.resume(slug)
            _emit(args, out, out["hint"])
            return 0

        if action == "progress":
            slug = need("a book slug")
            chapter = int(getattr(args, "chapter", 0) or 0)
            if chapter >= 1:
                out = lib.set_progress(slug, chapter,
                                       int(getattr(args, "offset", 0) or 0))
                _emit(args, out,
                      f"{slug}: now at chapter {out['chapter']} "
                      f"({out['percent']}%)")
            else:
                out = lib.get_progress(slug)
                _emit(args, out,
                      f"{slug}: "
                      + (f"chapter {out['chapter']} “{out['chapter_title']}” "
                         f"({out['percent']}%)" if out["started"]
                         else "not started"))
            return 0

        if action == "bookmark":
            if getattr(args, "list", False):
                marks = lib.list_bookmarks(target)
                lines = [f"  #{m['id']} [{m['slug']}] ch{m['chapter']} "
                         f"+{m['offset_chars']} — {m['label'] or '(no label)'}"
                         for m in marks]
                _emit(args, {"bookmarks": marks},
                      "bookmarks:\n" + "\n".join(lines)
                      if lines else "no bookmarks")
                return 0
            rm_id = int(getattr(args, "id", 0) or 0)
            if rm_id:
                out = lib.remove_bookmark(rm_id)
                _emit(args, out, f"removed bookmark #{rm_id}")
                return 0
            slug = need("a book slug")
            chapter = int(getattr(args, "chapter", 0) or 0)
            if chapter < 1:
                print("books bookmark needs --chapter — "
                      "nm books bookmark <slug> --chapter N [--label L]",
                      file=sys.stderr)
                return 2
            out = lib.add_bookmark(slug, chapter,
                                   int(getattr(args, "offset", 0) or 0),
                                   getattr(args, "label", "") or "")
            _emit(args, out,
                  f"bookmark #{out['id']}: {slug} ch{chapter}"
                  + (f" — {out['label']}" if out["label"] else ""))
            return 0

        if action == "note":
            if getattr(args, "list", False):
                notes = lib.list_notes(target)
                lines = [f"  #{n['id']} [{n['slug']}] ch{n['chapter']}: "
                         f"{n['note'][:80]}" for n in notes]
                _emit(args, {"notes": notes},
                      "notes:\n" + "\n".join(lines)
                      if lines else "no notes")
                return 0
            rm_id = int(getattr(args, "id", 0) or 0)
            if rm_id:
                out = lib.remove_note(rm_id)
                _emit(args, out, f"removed note #{rm_id}")
                return 0
            slug = need("a book slug")
            chapter = int(getattr(args, "chapter", 0) or 0)
            text = (getattr(args, "note_text", "") or "").strip()
            if chapter < 1 or not text:
                print("books note needs --chapter and --note-text — "
                      "nm books note <slug> --chapter N --note-text \"...\"",
                      file=sys.stderr)
                return 2
            out = lib.add_note(slug, chapter,
                               int(getattr(args, "offset", 0) or 0),
                               quote=getattr(args, "quote", "") or "",
                               note=text)
            _emit(args, out, f"note #{out['id']}: {slug} ch{chapter}")
            return 0

        if action == "shelf":
            if getattr(args, "list", False):
                cols = lib.list_collections()
                lines = [f"  {c['name']:<24} {c['books']} book(s)"
                         for c in cols]
                _emit(args, {"collections": cols},
                      "collections:\n" + "\n".join(lines)
                      if lines else "no collections yet")
                return 0
            name = need("a collection name")
            if getattr(args, "create", False):
                out = lib.create_collection(name)
                _emit(args, out, f"collection {name!r} "
                                 f"{'created' if out['created'] else 'already exists'}")
                return 0
            if getattr(args, "delete", False):
                out = lib.delete_collection(name)
                _emit(args, out, f"deleted collection {name!r}")
                return 0
            add = (getattr(args, "add", "") or "").strip()
            if add:
                out = lib.add_to_collection(name, add)
                _emit(args, out, f"added {add} to {name!r}")
                return 0
            rm = (getattr(args, "remove", "") or "").strip()
            if rm:
                out = lib.remove_from_collection(name, rm)
                _emit(args, out, f"removed {rm} from {name!r}")
                return 0
            out = lib.shelf(name)
            lines = [f"  {b['slug']:<28} {b['title'][:48]:<50} "
                     f"{'★' * b['rating'] or '·':<5} {b['progress_percent']:>5}%"
                     for b in out["books"]]
            _emit(args, out,
                  f"shelf {name!r}:\n" + "\n".join(lines)
                  if lines else f"shelf {name!r} is empty")
            return 0

        if action == "tag":
            if getattr(args, "list", False):
                tags = lib.list_tags()
                lines = [f"  {t['tag']:<24} {t['books']} book(s)"
                         for t in tags]
                _emit(args, {"tags": tags},
                      "tags:\n" + "\n".join(lines) if lines else "no tags yet")
                return 0
            find = (getattr(args, "find", "") or "").strip()
            if find:
                books = lib.books_with_tag(find)
                lines = [f"  {b['slug']:<28} {b['title'][:60]}"
                         for b in books]
                _emit(args, {"tag": find, "books": books},
                      f"tag {find!r}:\n" + "\n".join(lines)
                      if lines else f"no books tagged {find!r}")
                return 0
            slug = need("a book slug")
            set_arg = getattr(args, "set", None)
            if set_arg is not None:
                out = lib.set_tags(slug, set_arg)
                _emit(args, out,
                      f"{slug}: tags → {', '.join(out['tags']) or '(cleared)'}")
                return 0
            tags = lib.get_tags(slug)
            _emit(args, {"slug": slug, "tags": tags},
                  f"{slug}: {', '.join(tags) or 'no tags'}")
            return 0

        if action == "rate":
            slug = need("a book slug")
            stars = int(getattr(args, "stars", 0) or 0)
            if stars:
                out = lib.rate(slug, stars)
                _emit(args, out, f"{slug}: rated {'★' * stars}")
            else:
                r = lib.get_rating(slug)
                _emit(args, {"slug": slug, "stars": r},
                      f"{slug}: {'★' * r if r else 'not rated'}")
            return 0

        print(f"books: unknown action {action}", file=sys.stderr)
        return 2
    except LibraryError as exc:
        print(f"books: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        return int(exc.code or 0)
