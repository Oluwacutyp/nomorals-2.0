"""RuntimeSearchMixin: PartnerRuntime command group (search)."""

from __future__ import annotations

import threading
import time
from .failures import record_failure

#: Chars of chapter text per reading-pager page (the menu session's
#: 3200-char page budget, minus headroom for the header/footer and
#: the send funnel's stamp).
READING_PAGE_CHARS = 3000


def _reading_pager_ok(chat_key: str) -> bool:
    """True when this chat's surface can hold the live reading pager:
    buttons AND text edit, per the platform limits — and never
    WhatsApp (its edit verb is text-only, so its keyboard cannot
    follow the page; the menu session sets the same precedent).
    Everywhere else, reading falls back to the chunked send — the
    honest buttonless/no-edit degradation, never a fake pager."""
    platform = (chat_key or "").partition(":")[0].strip().lower()
    if platform in ("whatsapp", "wa"):
        return False
    try:
        from ...social.chat.platform_limits import for_platform

        limits = for_platform(platform)
    except Exception:  # noqa: BLE001 - unknown platform: no pager
        return False
    return bool(limits is not None and limits.supports_buttons
                and limits.supports_text_edit)


def _page_ref(kind: str, slug: str) -> str:
    """The story reference a pager callback can carry: the full slug
    when ``<kind> page <slug> 888 888`` fits Telegram's 47-byte
    callback budget, else a 24-char prefix — which
    :meth:`StoryReader.resolve_slug` resolves as a unique prefix,
    exactly like a typed reference."""
    if len(f"{kind} page {slug} 888 888".encode()) <= 47:
        return slug
    return slug[:24]


def _chapter_actionable(*, kind: str, slug: str, title: str,
                        chapter: int, chapter_title: str, text: str,
                        page: int, total_chapters: int,
                        has_next: bool, session: str, edit: bool):
    """One page of a chapter as a live-session render (the paging
    law: ONE message, Next/Prev buttons, edited in place — never an
    (n/m) chunk dump).  Buttons ride the control grammar of the
    owning command (``<kind> page …`` / ``<kind> read …``), so a tap
    re-enters the same handler that authored the page."""
    from ...social.chat.chunking import chunk_message
    from ...social.chat.tgbot_buttons import actionable

    pages = chunk_message(text or "", max_text=READING_PAGE_CHARS) or [""]
    page = min(max(1, int(page or 1)), len(pages))
    ref = _page_ref(kind, slug)
    head = f"📖 {title} — chapter {chapter}"
    if chapter_title and chapter_title != f"Chapter {chapter}":
        head += f" · {chapter_title}"
    foot = (f"— page {page} of {len(pages)} · chapter {chapter} "
            f"of {total_chapters or '?'} —")
    rows: list = []
    nav: list = []
    if page > 1:
        nav.append(("◀ Prev", f"{kind} page {ref} {chapter} {page - 1}"))
    if page < len(pages):
        nav.append(("Next ▶", f"{kind} page {ref} {chapter} {page + 1}"))
    if nav:
        rows.append(nav)
    chrow: list = []
    if chapter > 1:
        chrow.append((f"⏮ ch {chapter - 1}",
                      f"{kind} read {ref} {chapter - 1}"))
    if has_next:
        chrow.append((f"ch {chapter + 1} ⏭",
                      f"{kind} read {ref} {chapter + 1}"))
    if chrow:
        rows.append(chrow)
    return actionable(f"{head}\n\n{pages[page - 1]}\n\n{foot}", rows,
                      session=session, session_edit=edit)

class RuntimeSearchMixin:
    """RuntimeSearchMixin for :class:`PartnerRuntime`."""


    def _control_search(self, tail: str, mode: str, chat_key: str) -> str:
        from ..features import feature_enabled
        from ..power import power_mode_for
        from ..search.engine import SearchEngine

        if not feature_enabled(self.context, "search"):
            return "search is off. turn it on: /features search on"
        query = (tail or "").strip()
        if not query:
            return "usage: /search <what to research>"
        if mode == "deep" and not power_mode_for(self.context).active:
            return "deep research is a power-mode capability: /power on <key> first"
        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat,
                               f"⏳ {mode} research: {query[:80]}\nthis takes a moment — the report comes right here.")
        except Exception as exc:  # noqa: BLE001 - progress note is best-effort
            record_failure("runtime_search._control_search", exc)
        started = time.time()
        try:
            report = SearchEngine(self.context).run(query, mode=mode,
                                                    pages=8 if mode == "deep" else 3)
        except Exception as exc:  # noqa: BLE001
            return f"search failed: {exc}"
        elapsed = time.time() - started
        text = self._format_search_report(report, query, mode, elapsed)
        return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks

    @staticmethod
    def _format_search_report(report: dict, query: str, mode: str, elapsed: float) -> str:
        pages_read = report.get("pages_read") or []
        n_pages = len(pages_read) if isinstance(pages_read, list) else int(pages_read or 0)
        sub_queries = report.get("sub_queries") or []
        lines = [
            f"🔎 {mode} research: {query[:90]}",
            f"({n_pages} pages read · {elapsed:.0f}s · "
            f"{'model-sourced' if report.get('model_summary') else 'extractive'} summary)",
        ]
        if len(sub_queries) > 1:
            lines.append("sub-queries: " + " | ".join(s[:48] for s in sub_queries[:5]))
        summary = str(report.get("summary") or "").strip()
        if summary:
            lines.append("")
            lines.append(summary)
        numbered = report.get("sources")
        if numbered:
            lines.append("")
            lines.append("sources:")
            for s in numbered[:8]:
                title = (s.get("title") or "").strip()
                lines.append(f"  [{s.get('n')}] {title[:70]}\n      {s.get('url')}")
        else:
            sources = report.get("results") or []
            if sources:
                lines.append("")
                lines.append("main sources:")
                for s in sources[:6]:
                    lines.append(f"  • {s.get('title') or s.get('url')}\n    {s.get('url')}")
        return "\n".join(lines)

    def _control_search_leads(self) -> str:
        from ..features import feature_enabled
        from ..search.engine import SearchEngine

        if not feature_enabled(self.context, "search"):
            return "search is off. turn it on: /features search on"
        try:
            leads = SearchEngine(self.context).leads()
        except Exception as exc:  # noqa: BLE001
            return f"leads research failed: {exc}"
        if not leads:
            return ("no leads surfaced (endpoints may be unreachable from here).\n"
                    "try again later, or /search <platform> to check one directly.")
        lines = [f"legit paid-task platforms — {len(leads)} (research report, pick one you trust):"]
        for lead in leads[:12]:
            lines.append(f"  • {lead.get('title') or lead.get('domain')} — {lead.get('url')}")
        lines.append("")
        lines.append("to try one: /trial start <name>")
        return "\n".join(lines)

    def _control_search_hist(self, arg: str) -> str:
        from ..features import feature_enabled
        from ..search.engine import SearchEngine

        if not feature_enabled(self.context, "search"):
            return "search is off. turn it on: /features search on"
        limit = int(arg) if arg.isdigit() else 5
        rows = SearchEngine(self.context).history(limit)
        if not rows:
            return "no research runs yet."
        lines = [f"recent research ({len(rows)}):"]
        for row in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(row.get("created_at", 0)))
            lines.append(f"  {when}  {row.get('mode', 'quick')}  {str(row.get('query', ''))[:60]}")
        return "\n".join(lines)

    # ── BookForge ────────────────────────────────────────────────────────────
    def _control_book(self, tail: str, chat_key: str) -> str:
        """/book <topic> [chapters] — writes a real book in the background and
        sends the finished PDF to this chat when done.  Also: list / status /
        build / send for books already on disk."""
        from ...books import BookForge
        from ...books.model import BookError, slugify

        query = (tail or "").strip()
        parts = query.split()
        verb = parts[0].lower() if parts else ""

        def _one_line(slug: str) -> str:
            try:
                b = forge.load(slug)
                return (f"  {slug} — {b.status} · {b.chapters_written}/{len(b.chapters)} ch · "
                        f"{b.total_words} words")
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_search._one_line", exc)
                return f"  {slug}"

        try:
            forge = BookForge(self.context)
        except Exception as exc:  # noqa: BLE001
            return f"book system failed: {exc}"

        try:
            if verb == "list" or not verb:
                books = forge.list_books()
                unreadable = forge.last_list_unreadable
                if not books:
                    if unreadable:
                        return (f"no readable books — {unreadable} book(s) on disk are"
                                " unreadable (corrupt book.json).")
                    return "no books yet. /book <topic> [chapters] writes one and sends the pdf."
                lines = ["books:"]
                for b in books:
                    lines.append(f"  {b['slug']} — {b['status']} · {b['written']}/{b['chapters']} ch · "
                                 f"{b['words']} words")
                if unreadable:
                    lines.append(f"  ({unreadable} book(s) on disk unreadable"
                                 " — corrupt book.json)")
                lines.append("\n/book status [slug] for details · /book build <slug> for the pdf")
                return "\n".join(lines)

            if verb == "status":
                if len(parts) > 1:
                    return _one_line(parts[1])
                books = forge.list_books()
                if not books:
                    if forge.last_list_unreadable:
                        return (f"no readable books — {forge.last_list_unreadable} book(s)"
                                " on disk are unreadable (corrupt book.json).")
                    return "no books on disk yet."
                lines = [f"{b['slug']} — {b['status']} · {b['written']}/{b['chapters']} ch · "
                         f"{b['words']} words" for b in books]
                if forge.last_list_unreadable:
                    lines.append(f"  ({forge.last_list_unreadable} book(s) on disk"
                                 " unreadable — corrupt book.json)")
                return "\n".join(lines)

            if verb == "build":
                if len(parts) < 2:
                    return "usage: /book build <slug>"
                r = forge.build(parts[1])
                return (f"built {r['slug']}: {r['pages']} pages · {r['pdf_bytes']} bytes\n"
                        f"{r['pdf']}")

            if verb == "send":
                if len(parts) < 4:
                    return "usage: /book send <slug> <platform> <chat_id>"
                r = forge.send(parts[1], parts[2], parts[3])
                return f"sent {r['slug']} to {parts[2]}:{parts[3]}"

            if verb == "sources":
                from ...books.descriptors import list_sources
                lines = ["story sources (legal first, licence on every row):"]
                for s in list_sources():
                    mark = "✓" if s["verified"] else "?"
                    lines.append(
                        f"  {mark} {s['id']} — {s['rights_class']} — "
                        f"{s['licence']}"
                        + (" (needs API key)" if s["requires_key"] else ""))
                lines.append("\n/books read <url|slug> [n] follows + reads; "
                             "/books part <slug> turns the page where buttons can't; "
                             "/books contents <slug> lists chapters; "
                             "/books export <slug> [pdf|epub|html] exports the "
                             "held chapters as a stamped file; "
                             "/books continue <slug> [n] continues in its voice")
                return "\n".join(lines)

            if verb == "export":
                # /books export <slug> [pdf|epub|html] — export a
                # FOLLOWED story from its locally held chapters
                # (Doc 16 P5): the file is built through the story's
                # manifest, carries its revision + declared
                # completeness, and declares any missing chapters
                # instead of inventing them.  Authored books export
                # through /book build — this verb is stories only.
                from pathlib import Path as _Path

                from ...books.reader import ReaderError, StoryReader
                if len(parts) < 2:
                    return "usage: /books export <slug> [pdf|epub|html]"
                fmt = parts[2].lower() if len(parts) > 2 else "pdf"
                if fmt not in ("pdf", "epub", "html"):
                    return ("usage: /books export <slug> [pdf|epub|html]"
                            " — formats: pdf, epub, html")
                reader = StoryReader(self.context)
                try:
                    r = reader.export_story(parts[1], fmt)
                except ReaderError as exc:
                    return f"📕 export refused: {exc}"
                except Exception as exc:  # noqa: BLE001 - chat surface
                    record_failure("runtime_search._control_book.export",
                                   exc)
                    return f"📕 export failed: {exc}"
                chat = self._ref_from_key(chat_key)
                name = _Path(r["path"]).name
                result = self.gateway.send_file(
                    chat.platform, chat, r["path"],
                    caption=f"📕 {r['title']} — {r['completeness_line']}",
                    name=name)
                if not getattr(result, "ok", False):
                    return (f"📕 exported but the file send did not go "
                            f"through — it is saved at {r['path']}\n"
                            f"{r['completeness_line']}")
                return (f"📄 {name} sent above.\n"
                        f"{r['completeness_line']}")

            if verb == "read":
                # /books read <url|slug> [n] — follow a URL, or read a
                # followed story. Chapter 0 = resume where you left off.
                # The chapter presents through the live reading pager
                # where the surface holds it (full text, paged in one
                # message); elsewhere the plain send below.
                from ...books.reader import StoryReader
                if len(parts) < 2:
                    return "usage: /books read <url|slug> [chapter] — 0 = resume"
                target = parts[1]
                chapter = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                reader = StoryReader(self.context)
                try:
                    if target.startswith("http"):
                        self._ensure_story_engine()
                        followed = reader.follow(target,
                                                 chat_key=chat_key)
                        slug = followed.slug
                    else:
                        slug = target
                    if _reading_pager_ok(chat_key):
                        return self._novel_chapter_reply(
                            reader, slug, chapter, chat_key,
                            kind="books", session="books")
                    result = reader.read(slug, chapter=chapter)
                except Exception as exc:  # noqa: BLE001 - chat surface
                    return f"read failed: {exc}"
                # No-pager surfaces read through the generalized
                # reader (Doc 16 P2): the chapter's FULL text, one
                # part at a time, this chat's session holding the
                # place, the continuation pointer naming the next
                # part of THIS chapter (the next chapter only once
                # this one ends).  The old render sliced read()'s
                # 12000-char preview to 3500 chars and pointed at
                # the NEXT CHAPTER as the "next part" — stranding
                # chars 3500→end of the current chapter entirely.
                slug = result["slug"]
                chapter_no = int(result.get("chapter", chapter or 1))
                view = self._record_story_position(
                    reader, slug, chapter_no, 1, chat_key)
                if view is None:
                    return self._novel_chapter_reply(
                        reader, slug, chapter_no, chat_key,
                        kind="books", session="books")
                return self._story_part_render(view, slug)

            if verb == "page":
                # Pager continuation for /books read renders:
                # books page <ref> <chapter> <page>.
                from ...books.reader import StoryReader
                if len(parts) < 4 or not parts[2].isdigit() \
                        or not parts[3].isdigit():
                    return "usage: /books page <slug> <chapter> <page>"
                reader = StoryReader(self.context)
                try:
                    return self._novel_chapter_reply(
                        reader, parts[1], int(parts[2]), chat_key,
                        page=int(parts[3]), edit=True,
                        kind="books", session="books")
                except Exception as exc:  # noqa: BLE001 - chat surface
                    return f"read failed: {exc}"

            if verb == "part":
                # /books part <slug> [back] — the plain-transport
                # continuation (Doc 16 P2): move this chat's reader
                # session one page, crossing into the next chapter
                # at a chapter's end (fetching it first — read() is
                # the fetch, as it is everywhere on this surface).
                from ...books.manifest import story_content_resolver
                from ...books.reader import StoryReader
                if len(parts) < 2:
                    return "usage: /books part <slug> [back]"
                try:
                    reader = StoryReader(self.context)
                    slug = reader.resolve_slug(parts[1])
                    svc = self._reader_service()
                    session = svc.session_for(
                        f"chat:{chat_key}", f"story:{slug}")
                    if session is None:
                        return (f"no reading in progress for {slug} "
                                f"— /books read {slug} opens it")
                    resolver = story_content_resolver(reader)
                    back = len(parts) > 2 and \
                        parts[2].lower() == "back"
                    if not back:
                        view = svc.view(session.session_id, resolver)
                        seg = session.segment_id
                        number = int(seg[3:]) if seg.startswith("ch-") \
                            and seg[3:].isdigit() else 0
                        if view.status == "ok" and \
                                not view.has_next_page and number:
                            # the next chapter may exist unheld —
                            # fetch it so it becomes a segment, then
                            # let the navigation cross into it; a
                            # source with no such chapter answers
                            # through the navigation's end note
                            try:
                                reader.read(slug, chapter=number + 1)
                                from ...books.manifest import (
                                    manifest_from_story)
                                session = svc.open(
                                    f"chat:{chat_key}",
                                    manifest_from_story(reader, slug),
                                    command_prefix="books go")
                            except Exception:  # noqa: BLE001
                                pass  # navigate answers honestly
                    view = svc.navigate(
                        session.session_id, resolver,
                        "prev" if back else "next")
                    return self._story_part_render(view, slug)
                except Exception as exc:  # noqa: BLE001 - chat surface
                    return f"read failed: {exc}"

            if verb == "contents":
                # /books contents <slug> — the story's contents from
                # its manifest: held chapters with sizes, declared
                # chapters not yet held named as not held.
                from ...books.manifest import manifest_from_story
                from ...books.reader import StoryReader
                if len(parts) < 2:
                    return "usage: /books contents <slug>"
                try:
                    reader = StoryReader(self.context)
                    slug = reader.resolve_slug(parts[1])
                    svc = self._reader_service()
                    session = svc.session_for(
                        f"chat:{chat_key}", f"story:{slug}")
                    view = svc.contents_view(
                        manifest_from_story(reader, slug),
                        current_segment_id=(
                            session.segment_id if session else ""))
                    return svc.render_contents_text(
                        view,
                        open_hint=f"open one: /books read {slug} "
                                  "<chapter> (an unheld chapter is "
                                  "fetched on read)")
                except Exception as exc:  # noqa: BLE001 - chat surface
                    return f"read failed: {exc}"

            if verb == "go":
                # /books go <token> <verb> — actuate a reader-session
                # token (the generalized reader's callback grammar:
                # n/p = next/prev page, c = contents, j<row> = jump
                # to a contents row, u = switch to the newer
                # revision).  Stale tokens answer stale, in words,
                # and never move the reader.
                from ...books.manifest import story_content_resolver
                from ...books.reader import StoryReader
                if len(parts) < 3:
                    return "usage: /books go <token> <n|p|c|jN|u>"
                svc = self._reader_service()
                session = svc.session_by_token(parts[1])
                if session is None:
                    return ("that reading control has expired — "
                            "/books read <slug> opens the story again")
                if not session.document_id.startswith("story:"):
                    return ("that reading control belongs to a "
                            "different document")
                slug = session.document_id[len("story:"):]
                try:
                    reader = StoryReader(self.context)
                    result = svc.act(
                        parts[1], parts[2],
                        story_content_resolver(reader))
                except Exception as exc:  # noqa: BLE001 - chat surface
                    return f"read failed: {exc}"
                if result.status != "ok":
                    return result.note
                if result.contents is not None:
                    return svc.render_contents_text(
                        result.contents,
                        open_hint=f"open one: /books read {slug} "
                                  "<chapter> (an unheld chapter is "
                                  "fetched on read)")
                return self._story_part_render(result.view, slug)

            if verb == "continue":
                # /books continue <slug> [n] — continue past the ending
                # in the story's voice, with per-chapter drift scores
                from ...books.serial import SerialManager
                if len(parts) < 2:
                    return "usage: /books continue <slug> [n] — continue past the ending"
                n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                try:
                    result = SerialManager(self.context).continue_past_ending(
                        parts[1], max(1, min(n, 5)))
                except Exception as exc:  # noqa: BLE001 - chat surface
                    return f"continue failed: {exc}"
                lines = [f"✍️ {result['title']} — continued "
                         f"{len(result['chapters'])} chapter(s) "
                         f"from chapter {result['continued_from_chapter']}"]
                for ch in result["chapters"]:
                    drift = ch.get("drift") or {}
                    lines.append(
                        f"  ch {ch.get('chapter')} — {ch.get('words', 0)} words · "
                        f"voice drift {drift.get('drift', '—')} "
                        f"({drift.get('verdict', 'no reference voice')})")
                return "\n".join(lines)

            # ── new book ────────────────────────────────────────────────────
            # chapters: explicit number wins, otherwise the book grows
            # organically as it is written (no count decided up front)
            chapters = 0
            if parts and parts[-1].isdigit():
                chapters = max(3, min(int(parts[-1]), 24))
                topic = " ".join(parts[:-1]).strip()
            else:
                topic = query
            if not topic:
                return ("usage: /book <topic> [chapters] — e.g. /book eBPF for system security 8\n"
                        "without a number the book grows organically: chapters emerge as it's "
                        "written until the topic is covered. /book list · /book status")
            slug = slugify(topic)
            with self._queue_guard:
                if slug in self._book_busy:
                    return f"already writing {slug!r} — check /book status {slug}"
                self._book_busy.add(slug)

            chat = self._ref_from_key(chat_key)
            chapter_note = (f"{chapters} chapters" if chapters
                            else "grows organically as it's written")
            try:
                self.gateway.send(
                    chat.platform, chat,
                    f"✍️ writing “{topic[:70]}” — {chapter_note}. "
                    "research → outline → write → pdf → straight to this chat. "
                    "check progress: /book status " + slug,
                )
            except Exception as exc:  # noqa: BLE001 - start note is best-effort
                record_failure("runtime_search._control_book", exc)

            def _work() -> None:
                try:
                    try:
                        book = forge.create(topic, chapters=chapters, research=True)
                        title = book.display_title
                    except BookError as exc:
                        self._notify(chat, f"⚠️ book failed: {exc}")
                        return
                    # progress notes at milestones so a long book doesn't look
                    # dead (chapters can take real minutes; organic books
                    # keep growing, so milestones beat a fixed halfway mark)
                    written = 0
                    next_milestone = 2
                    while True:
                        r = forge.write_next(slug)
                        written += 1
                        if r.get("done") and not r.get("chapter"):
                            break
                        if written >= next_milestone:
                            self._notify(chat, (
                                f"✍️ {title[:60]} — {r.get('chapters_written', written)} "
                                f"chapters, {r.get('total_words', 0)} words so far. "
                                f"{'still growing…' if r.get('organic') and not r.get('concluded') else ''}"))
                            next_milestone += 6
                    built = forge.build(slug)
                    caption = (f"📕 {title} — {built.get('chapters_written')} chapters, "
                               f"{built.get('words')} words, {built.get('pages')} pages")
                    try:
                        self.gateway.send_file(chat.platform,
                                               f"{chat.platform}:{chat.chat_id}",
                                               built["pdf"], caption=caption)
                        self._notify(chat, f"📕 done — “{title[:70]}” is on its way. "
                                           f"/book status {slug} for the record.")
                    except Exception as send_exc:  # noqa: BLE001 - the book exists either way
                        self._notify(chat, (f"📕 “{title[:70]}” is finished but the send "
                                            f"failed ({send_exc}).\n"
                                            f"pdf: {built['pdf']}"))
                except Exception as exc:  # noqa: BLE001
                    self._notify(chat, f"⚠️ book {slug} failed: {exc}")
                finally:
                    with self._queue_guard:
                        self._book_busy.discard(slug)
                    # One-shot thread: don't leak its DB connection.
                    self._release_db_thread()

            threading.Thread(target=_work, name=f"book-{slug}", daemon=True).start()
            return ""  # the start note is already on its way
        except BookError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            return f"book failed: {exc}"

    def _story_chat_notify(self, chat_key: str, text: str) -> None:
        """The reading engine's chat delivery: a ping lands in the
        chat that triggered the read.  Best-effort, never raises."""
        try:
            chat = self._ref_from_key(chat_key)
            self._notify(chat, text)
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_search._story_chat_notify", exc)

    def _ensure_story_engine(self) -> None:
        """Stand the story reading engine up on this runtime's
        context (idempotent — boot wires it first; a handler calling
        this on an unbooted runtime still gets a live consumer)."""
        try:
            from ...books.reading_engine import ensure_reading_engine

            ensure_reading_engine(
                self.context, chat_notify=self._story_chat_notify)
        except Exception as exc:  # noqa: BLE001 - the reply still goes out
            record_failure("runtime_search._ensure_story_engine", exc)

    def _reader_service(self):
        """The generalized long-document reader (Doc 16 P2), bound
        to this runtime's context — per-reader reading sessions over
        P1 manifests, persisted under the workspace."""
        from ...social.chat.reader_service import ReaderService

        return ReaderService.for_context(self.context)

    def _control_read(self, tail: str, chat_key: str) -> str:
        """``/read`` — the routed long outputs' desk (Doc 16 P4).

        Long outputs the router sent to the reader (or offered as
        files) live here: the list reopens them, ``go`` actuates a
        reading token (the reader's own grammar), ``get`` delivers
        the branded HTML edition or the raw original as a file.
        Owner-only by absence from PUBLIC_COMMANDS, exactly like
        /cases.
        """
        from ...social.chat.output_router import OutputRouter

        parts = (tail or "").strip().split()
        verb = parts[0].lower() if parts else "list"
        try:
            router = OutputRouter.for_context(self.context)
        except Exception as exc:  # noqa: BLE001 - chat surface
            record_failure("runtime_search._control_read.router", exc)
            return ("reading is unavailable right now — the output "
                    "store failed to open")
        reader_key = f"chat:{chat_key}"
        svc = router.reader
        try:
            if verb == "go":
                # /read go <token> <verb> — the reader's callback
                # grammar, on routed outputs (text: documents);
                # story tokens answer to /books go, as story
                # controls answer in reverse there.
                if len(parts) < 3:
                    return "usage: /read go <token> <n|p|c|jN|u>"
                session = svc.session_by_token(parts[1])
                if session is None:
                    return ("that reading control has expired — "
                            "/read lists what this chat has open")
                if not session.document_id.startswith("text:"):
                    return ("that reading control belongs to a "
                            "different document — story controls "
                            "answer to /books go")
                result = svc.act(parts[1], parts[2],
                                 router.resolver())
                if result.status != "ok":
                    return result.note
                if result.contents is not None:
                    return svc.render_contents_text(
                        result.contents,
                        open_hint=f"open one: /read go "
                                  f"{session.token} j<line number>")
                return svc.render_text(result.view)
            if verb == "get":
                # /read get <token> <html|txt> — the file actions:
                # the branded HTML edition, or the raw original,
                # delivered as a document through the gateway.
                if len(parts) < 3 or \
                        parts[2].lower() not in ("html", "txt"):
                    return "usage: /read get <token> <html|txt>"
                session = svc.session_by_token(parts[1])
                if session is None:
                    return ("that file control has expired — /read "
                            "lists what this chat has open")
                if not session.document_id.startswith("text:"):
                    return ("that file control belongs to a "
                            "different document")
                found = router.file_for(
                    session.document_id, parts[2].lower())
                if found is None:
                    return ("that output is no longer in the store "
                            "— nothing to send")
                path, name = found
                chat = self._ref_from_key(chat_key)
                caption = (
                    "🌐 the branded HTML edition — complete, with "
                    "its completeness stated on the cover"
                    if parts[2].lower() == "html" else
                    "⬇ the original output, byte for byte")
                result = self.gateway.send_file(
                    chat.platform, chat, str(path), caption=caption,
                    name=name)
                if not getattr(result, "ok", False):
                    return ("⚠️ the file send failed — the gateway "
                            "rejected it; the output itself is "
                            "still in the reader (/read)")
                return f"📄 sent above — {name}"
            if verb == "open":
                # /read open <n> — reopen a routed output from the
                # list, at the reader's current page for it.
                rows = router.recent(reader_key)
                if len(parts) < 2 or not parts[1].isdigit() or \
                        not 1 <= int(parts[1]) <= len(rows):
                    return ("usage: /read open <n> — n from the "
                            "/read list")
                row = rows[int(parts[1]) - 1]
                session = svc.session_for(
                    reader_key, row.get("document_id", ""))
                if session is None:
                    return ("that output's reading session is "
                            "gone — its text is still in the store")
                view = svc.view(session.session_id,
                                router.resolver())
                return svc.render_text(view)
            # bare / /read list — this chat's routed outputs.
            rows = router.recent(reader_key)
            if not rows:
                return ("no routed long outputs in this chat yet — "
                        "when a report or result runs long, it "
                        "lands here with reading controls instead "
                        "of a wall of chunks")
            lines = ["📚 routed outputs in this chat (newest first):"]
            for i, row in enumerate(rows, start=1):
                title = row.get("title") or "long output"
                chars = row.get("chars", 0)
                lines.append(
                    f"{i}. {title} — {chars:,} chars "
                    f"· {row.get('route', '')}")
                lines.append(f"   open: /read open {i}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - chat surface
            record_failure("runtime_search._control_read", exc)
            return f"read failed: {exc}"

    def _record_story_position(self, reader, slug: str, chapter: int,
                               page: int, chat_key: str):
        """Mirror a story read into the generalized reader's
        per-reader session: this chat's place in the story
        (segment = chapter, page within it), keyed to the story
        manifest's revision.  The manifest is derived when the
        session is created or the chapter is new to it; page moves
        on an existing session are position-only.  Best-effort —
        a session-store failure never breaks the render.  Returns
        the positioned page view, or None when recording failed."""
        try:
            from ...books.manifest import (
                manifest_from_story,
                story_content_resolver,
            )

            svc = self._reader_service()
            reader_key = f"chat:{chat_key}"
            segment_id = f"ch-{chapter}"
            session = svc.session_for(reader_key, f"story:{slug}")
            if session is None or \
                    session.pinned().segment(segment_id) is None:
                session = svc.open(
                    reader_key, manifest_from_story(reader, slug),
                    command_prefix="books go")
            return svc.goto(
                session.session_id, story_content_resolver(reader),
                segment_id=segment_id, page_index=max(0, page - 1))
        except Exception as exc:  # noqa: BLE001 - best-effort mirror
            record_failure("runtime_search._record_story_position", exc)
            return None

    def _story_part_render(self, view, slug: str) -> str:
        """One page of a story chapter as plain text (the no-pager
        reading render): header, the page body, and continuation
        pointers that name what they point at — the next part of
        THIS chapter reads differently from the next chapter, and
        the next chapter is named only once this one has ended."""
        seg = view.segment_id or ""
        number = int(seg[3:]) if seg.startswith("ch-") \
            and seg[3:].isdigit() else view.segment_ordinal
        head = f"📖 {view.title or slug} — chapter {number}"
        if view.segment_title and \
                view.segment_title != f"Chapter {number}":
            head += f" · {view.segment_title}"
        if view.status != "ok":
            lines = [head, "", f"⚠️ {view.note}"]
            lines.append(f"📑 contents: /books contents {slug}")
            return "\n".join(lines)
        lines = [head, "", view.body, "",
                 f"— part {view.page_index + 1} of {view.page_count}"
                 f" · chapter {number} —"]
        if view.has_next_page:
            lines.append(
                f"▸ next part of this chapter: /books part {slug}")
        elif view.has_next_segment or view.declared_total > number:
            lines.append(
                f"▸ end of chapter {number} — next chapter: "
                f"/books read {slug} {number + 1}")
        else:
            lines.append(
                f"▸ end of chapter {number} — no further chapters "
                f"held · /books contents {slug} shows what is held")
        if view.has_prev_page:
            lines.append(f"◂ previous part: /books part {slug} back")
        elif view.has_prev_segment and number > 1:
            lines.append(
                f"◂ previous chapter: /books read {slug} {number - 1}")
        lines.append(f"📑 contents: /books contents {slug}")
        if view.update_available:
            lines.append(
                "✦ this story changed since you opened it — "
                f"/books go {view.token} u switches to the new "
                "revision (your place is kept)")
        return "\n".join(lines)

    def _novel_chapter_reply(self, reader, slug: str, chapter: int,
                             chat_key: str, *, page: int = 1,
                             edit: bool = False, kind: str = "novel",
                             session: str = "novel"):
        """Render one chapter of a followed story: through the live
        reading pager where the surface holds it (one message,
        Next/Prev, edited in place), else the chunked send — the
        honest degradation.  Reading updates the owner's progress
        (reader.read), and the body is the chapter's FULL text
        (chapter_text) on BOTH paths — never read()'s 12000-char
        preview, which the old fallback sent as if it were the
        chapter, stranding everything past it.  The read also
        records this chat's place in the generalized reader's
        per-reader session (Doc 16 P2)."""
        from ...books.reader import ReaderError

        data = reader.read(slug, chapter)
        slug = data["slug"]
        try:
            full = reader.chapter_text(slug, int(data["chapter"]))
        except ReaderError:
            full = data.get("text", "")
        if full.startswith("# "):
            full = full.split("\n", 1)[1].lstrip("\n")
        self._record_story_position(
            reader, slug, int(data["chapter"]), page, chat_key)
        if not _reading_pager_ok(chat_key):
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(
                chat.platform, chat,
                f"📖 {data.get('title', slug)} — chapter "
                f"{data.get('chapter', chapter)}\n\n{full}")
        story = reader._load(slug)
        return _chapter_actionable(
            kind=kind, slug=slug, title=data.get("title", slug),
            chapter=int(data["chapter"]),
            chapter_title=data.get("chapter_title", ""),
            text=full, page=page,
            total_chapters=int(story.total_chapters or 0),
            has_next=bool(data.get("next")), session=session,
            edit=edit)

    def _control_novel(self, tail: str, chat_key: str) -> str:
        """``/novel`` — story reader + continuer.

        follow <url|title> | list | sources | read <slug> [chapter] |
        next <slug> | sync <slug> | continue <slug> [n]

        Following is a trigger: the reading engine fetches the whole
        story and reads it herself (library + bible digestion) in the
        background; new chapters are read on discovery.  Reading a
        chapter presents through the live pager where the surface
        holds it.  Continue writes from the story's true final
        chapter, never from the reading position.
        """
        from ...books.reader import StoryReader, ReaderError
        from ...books.continuation import StoryContinuer

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""
        try:
            reader = StoryReader(self.context)
        except Exception as exc:  # noqa: BLE001
            return f"novel system failed: {exc}"

        if verb == "sources":
            # Book-source state from the ONE registry (pass #89):
            # seed vs grown story sites, health, fetch outcomes,
            # candidates pending verification.
            return reader.source_status_text()

        if verb == "list":
            stories = reader.following()
            unreadable = reader.last_following_unreadable
            if not stories:
                if unreadable:
                    return (f"no readable followed stories — {unreadable} "
                            "story(ies) on disk are unreadable "
                            "(corrupt story.json)")
                return ("no stories followed yet. /novel follow <url or title> — "
                        "e.g. /novel follow my vampire system")
            # Her reading progress rides the list (the bible's
            # digested watermark per story) — the owner can SEE that
            # following leads to reading, and how far she has got.
            digested_by_slug: dict[str, int] = {}
            try:
                from ...books.bible import BibleBuilder

                bibles = BibleBuilder(self.context)
                for s in stories:
                    bible = bibles.load(s["slug"])
                    if bible is not None:
                        digested_by_slug[s["slug"]] = int(
                            bible.chapters_digested or 0)
            except Exception as exc:  # noqa: BLE001 - list still renders
                record_failure("runtime_search._control_novel.list", exc)
            lines = ["followed stories:"]
            for s in stories:
                line = (
                    f"  {s['slug']} — {s.get('title', '?')} "
                    f"[{s.get('source', '?')}] ch {s.get('current_chapter', 0)}/"
                    f"{s.get('total_chapters') or '?'}")
                if s.get("ingest_total"):
                    line += (f" · library {s.get('ingested', 0)}/"
                             f"{s['ingest_total']}"
                             + (" ✓" if s.get("library_complete") else ""))
                if digested_by_slug.get(s["slug"]):
                    line += (f" · she's read through "
                             f"{digested_by_slug[s['slug']]}")
                lines.append(line)
            if unreadable:
                lines.append(f"  ({unreadable} story(ies) on disk unreadable"
                             " — corrupt story.json)")
            return "\n".join(lines)

        if verb == "follow":
            if not rest:
                return "usage: /novel follow <url or story title>"
            self._ensure_story_engine()
            try:
                story = reader.follow(rest, chat_key=chat_key)
            except ReaderError as exc:
                return f"follow failed: {exc}"
            except Exception as exc:  # noqa: BLE001
                return f"follow failed: {type(exc).__name__}: {exc}"
            # Following is a trigger, not a download: follow() has
            # published story.followed, and the reading engine is
            # already fetching the story and reading it herself
            # (library + bible).  The reply reports the state as it
            # stands NOW and never claims chapters not yet read.
            state = reader.ingest_state(story.slug)
            digested = 0
            try:
                from ...books.bible import BibleBuilder

                bible = BibleBuilder(self.context).load(story.slug)
                digested = int(bible.chapters_digested or 0) if bible else 0
            except Exception as exc:  # noqa: BLE001 - reply still goes out
                record_failure("runtime_search._control_novel.follow", exc)
            if state["complete"] and digested >= int(state["total"] or 0) \
                    and state["total"]:
                return (f"📖 following “{story.title}” [{story.source}] — "
                        f"{state['total']} chapters, the whole story is "
                        f"in its library and she's read it through. "
                        f"/novel read {story.slug} · "
                        f"/novel continue {story.slug} writes from the "
                        f"true ending.")
            return (f"📖 following “{story.title}” [{story.source}] — "
                    f"{story.total_chapters or '?'} chapters. She's "
                    f"reading it now — fetching the chapters and "
                    f"working through them herself; "
                    f"{state['ingested']} of "
                    f"{state['total'] or story.total_chapters or '?'} "
                    f"are in its library so far, and she's read "
                    f"through {digested}. I'll ping here when she's "
                    f"through them. /novel read {story.slug} works "
                    f"meanwhile.")

        if verb in ("sync", "ingest"):
            if not rest:
                return ("usage: /novel sync <slug> — she fetches and "
                        "reads the rest of the story (resumes where "
                        "she left off)")
            try:
                slug = reader.resolve_slug(rest.split()[0])
            except ReaderError as exc:
                return f"sync failed: {exc}"
            # The same trigger the follow fires, asked for by hand:
            # the reading engine runs the read (fetch + digest) and
            # pings the outcome here.  No private thread — the work
            # rides the standing consumer, so a sync, a follow and a
            # discovery can never double-read the same story.
            self._ensure_story_engine()
            try:
                from ...books.reading_engine import TOPIC_READ_REQUESTED
                from ...core.events import Event, global_bus

                global_bus.publish(Event(
                    topic=TOPIC_READ_REQUESTED,
                    data={"slug": slug, "chat_key": chat_key},
                    source=__name__))
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_search._control_novel.sync", exc)
                return f"sync failed: {type(exc).__name__}: {exc}"
            return ("📚 she's on it — fetching and reading the rest "
                    "of the story, resuming where she left off. "
                    "I'll ping here when she's through.")

        if verb in ("read", "next"):
            if not rest:
                return f"usage: /novel {verb} <slug> [chapter]"
            bits = rest.split()
            slug, ch = bits[0], int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else 0
            try:
                if verb == "next":
                    story = reader._load(slug)
                    ch = (story.current_chapter or 0) + 1
                return self._novel_chapter_reply(reader, slug, ch,
                                                  chat_key)
            except ReaderError as exc:
                return f"read failed: {exc}"
            except Exception as exc:  # noqa: BLE001
                return f"read failed: {type(exc).__name__}: {exc}"

        if verb == "page":
            # Pager continuation: novel page <ref> <chapter> <page> —
            # a button tap (or the typed equivalent) re-renders the
            # live reading message at the asked page, in place.
            bits = rest.split()
            if len(bits) < 3 or not bits[1].isdigit() \
                    or not bits[2].isdigit():
                return "usage: /novel page <story> <chapter> <page>"
            try:
                return self._novel_chapter_reply(
                    reader, bits[0], int(bits[1]), chat_key,
                    page=int(bits[2]), edit=True)
            except ReaderError as exc:
                return f"read failed: {exc}"
            except Exception as exc:  # noqa: BLE001
                return f"read failed: {type(exc).__name__}: {exc}"

        if verb == "continue":
            if not rest:
                return "usage: /novel continue <slug> [n] — writes n new chapters in the story's voice"
            bits = rest.split()
            slug = bits[0]
            n = int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else 1
            n = max(1, min(n, 5))
            chat = self._ref_from_key(chat_key)
            try:
                self.gateway.send(chat.platform, chat,
                                  f"✍️ continuing the story — {n} chapter(s) "
                                  f"in its own voice…")
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_search._control_novel:499", exc)

            def _work() -> None:
                try:
                    continuer = StoryContinuer(self.context)
                    result = continuer.continue_story(slug, n=n)
                    for ch in result.get("chapters", []):
                        self._send_long_checked(
                            chat.platform, chat,
                            f"📖 {ch.get('title', 'new chapter')}\n\n"
                            f"{(ch.get('text') or '')[:5500]}")
                    self._notify(chat, f"✍️ done — {len(result.get('chapters', []))} "
                                       f"new chapter(s) for “{result.get('title', slug)}”.")
                except Exception as exc:  # noqa: BLE001
                    self._notify(chat, f"⚠️ continue failed: {exc}")
                finally:
                    self._release_db_thread()

            import threading
            threading.Thread(target=_work, name=f"novel-continue-{slug}",
                             daemon=True).start()
            return ""

        if verb == "write":
            # /novel write <premise> [genre] [novel|serial]
            if not rest:
                return ("usage: /novel write <premise> [genre] [novel|serial] — "
                        "genres: mystery thriller horror scifi fantasy romance")
            from ...books.fiction import FictionWriter
            bits = rest.split()
            genre, mode, premise = "fantasy", "novel", rest
            # trailing tokens that match a genre/mode are flags, not premise
            genres = {"mystery", "thriller", "horror", "scifi", "sci-fi",
                      "fantasy", "romance"}
            while bits and (bits[-1].lower() in genres
                            or bits[-1].lower() in ("novel", "serial")):
                tok = bits.pop().lower()
                if tok in ("novel", "serial"):
                    mode = tok
                else:
                    genre = "scifi" if tok == "sci-fi" else tok
            premise = " ".join(bits).strip()
            if not premise:
                return "give me a premise — /novel write <premise> [genre]"
            chat = self._ref_from_key(chat_key)
            try:
                self.gateway.send(chat.platform, chat,
                                  f"✍️ shaping your {genre} {mode}…")
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_search._control_novel:548", exc)

            def _wstart() -> None:
                try:
                    fw = FictionWriter(self.context)
                    r = fw.start(premise, genre=genre, mode=mode)
                    ch1 = r.get("chapter_1") or {}
                    self._send_long_checked(
                        chat.platform, chat,
                        f"📖 {r.get('title')} — chapter 1\n\n"
                        f"{(ch1.get('text') or '')[:5500]}")
                    self._notify(
                        chat,
                        f"✍️ “{r.get('title')[:60]}” started [{genre}/{mode}] — "
                        f"{r.get('shape', '')} /novel chapter {r.get('slug')} for more.")
                except Exception as exc:  # noqa: BLE001
                    self._notify(chat, f"⚠️ write failed: {exc}")
                finally:
                    self._release_db_thread()

            import threading
            threading.Thread(target=_wstart, name="novel-write",
                             daemon=True).start()
            return ""

        if verb == "chapter":
            if not rest:
                return "usage: /novel chapter <slug> — writes the next chapter"
            from ...books.fiction import FictionWriter
            slug = rest.split()[0]
            chat = self._ref_from_key(chat_key)
            try:
                fw = FictionWriter(self.context)
                r = fw.write_next(slug)
            except Exception as exc:  # noqa: BLE001
                return f"chapter failed: {exc}"
            return self._send_long_checked(
                chat.platform, chat,
                f"📖 {r.get('title', 'next chapter')}\n\n"
                f"{(r.get('text') or '')[:5500]}")

        if verb == "branch":
            # /novel branch <slug> fork <name> <from_ch> <premise>
            # /novel branch <slug> list
            # /novel branch <slug> merge|abandon <name>
            from ...books.branches import StoryBranch, list_branches
            bits = rest.split(None, 3)
            if len(bits) < 2:
                return ("usage: /novel branch <slug> fork <name> <from_ch> <premise> | "
                        "/novel branch <slug> list | /novel branch <slug> merge|abandon <name>")
            slug, action = bits[0], bits[1].lower()
            if action == "list":
                branches = list_branches(slug, context=self.context)
                if not branches:
                    return f"no branches for '{slug}'"
                return "\n".join(
                    f"  {b['branch']} — from ch {b['forked_from']} "
                    f"({b['chapters']} ch, {b['status']}): {b['premise']}"
                    for b in branches)
            if action == "fork":
                if len(bits) < 4:
                    return "usage: /novel branch <slug> fork <name> <from_ch> <premise>"
                name_ch = bits[2].split(None, 1)
                if len(name_ch) < 2 or not name_ch[1].split(None, 1)[0].isdigit():
                    return "usage: /novel branch <slug> fork <name> <from_ch> <premise>"
                name = name_ch[0]
                from_ch = int(name_ch[1].split(None, 1)[0])
                premise = name_ch[1].split(None, 1)[1] if len(name_ch[1].split(None, 1)) > 1 else bits[3]
                r = StoryBranch(slug, name, context=self.context).fork(from_ch, premise)
                if not r.get("ok"):
                    return f"fork failed: {r.get('reason')}"
                return (f"🌿 branched '{name}' from ch {from_ch}: {premise[:100]} — "
                        f"write into it, merge or abandon when decided.")
            if action in ("merge", "abandon"):
                if len(bits) < 3:
                    return f"usage: /novel branch <slug> {action} <name>"
                r = StoryBranch(slug, bits[2], context=self.context).merge() if action == "merge" \
                    else StoryBranch(slug, bits[2], context=self.context).abandon()
                if not r.get("ok"):
                    return f"{action} failed: {r.get('reason')}"
                return f"🌿 branch '{bits[2]}' {action}ed."
            return f"unknown branch action '{action}'"

        if verb == "publish":
            # /novel publish <slug> start [daily|weekly|manual]
            # /novel publish <slug> release | status
            from ...books.publish import SerialPublication
            bits = rest.split(None, 2)
            if len(bits) < 2:
                return ("usage: /novel publish <slug> start [daily|weekly|manual] | "
                        "/novel publish <slug> release | /novel publish <slug> status")
            slug, action = bits[0], bits[1].lower()
            pub = SerialPublication(slug, context=self.context)
            if action == "start":
                cadence = bits[2].strip().lower() if len(bits) > 2 else "daily"
                r = pub.start(cadence)
                if not r.get("ok"):
                    return f"publish failed: {r.get('reason')}"
                return (f"📰 '{slug}' serializing [{cadence}] — chapters drip on cadence. "
                        f"/novel publish {slug} release to drop the next now.")
            if action == "release":
                r = pub.release()
                if not r.get("ok"):
                    return f"release: {r.get('reason', 'not due')}"
                chat = self._ref_from_key(chat_key)
                self._send_long_checked(
                    chat.platform, chat,
                    f"📰 {r['title']} — chapter {r['chapter']}\n\n"
                    f"{(r['text'] or '')[:5500]}")
                return f"released ch {r['chapter']} ({r['released_count']} total)."
            if action == "status":
                r = pub.status()
                if not r.get("ok"):
                    return f"no publication for '{slug}'"
                return (f"📰 {slug} [{r['cadence']}/{r['status']}]: "
                        f"{r['released']}/{r['total_chapters']} released, "
                        f"{r['followers']} followers, {r['feedback_count']} reactions.")
            return f"unknown publish action '{action}'"

        if verb == "scene":
            # /novel scene <slug> <character> <prompt> — character writes a scene
            if not rest:
                return "usage: /novel scene <slug> <character> <scene prompt>"
            bits = rest.split(None, 2)
            if len(bits) < 3:
                return "usage: /novel scene <slug> <character> <scene prompt>"
            slug, character, prompt = bits
            try:
                from ...characters.store import CharacterStore
                from ...books.collab import CollaborativeSession
                store = CharacterStore()
                char = store.get(character) or store.get_by_name(character)
                if char is None:
                    return f"character '{character}' not found — /character list"
                sess = CollaborativeSession(slug, self.context)
                sess.cast_character(char)
                r = sess.write_scene(character, prompt)
                if not r.get("ok"):
                    return f"scene failed: {r.get('reason')}"
                chat = self._ref_from_key(chat_key)
                return self._send_long_checked(
                    chat.platform, chat,
                    f"🎭 {character} writes:\n\n{(r['text'] or '')[:5500]}")
            except Exception as exc:  # noqa: BLE001
                return f"scene failed: {exc}"

        return ("usage: /novel follow <url|title> | /novel list | "
                "/novel sources | "
                "/novel read <slug> [chapter] | /novel next <slug> | "
                "/novel continue <slug> [n] | /novel write <premise> [genre] | "
                "/novel branch <slug> ... | /novel publish <slug> ... | "
                "/novel scene <slug> <character> <prompt>")
