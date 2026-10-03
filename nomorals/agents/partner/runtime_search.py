"""RuntimeSearchMixin: PartnerRuntime command group (search)."""

from __future__ import annotations

import threading
import time

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
        except Exception:  # noqa: BLE001 - progress note is best-effort
            pass
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
            except Exception:  # noqa: BLE001
                return f"  {slug}"

        try:
            forge = BookForge(self.context)
        except Exception as exc:  # noqa: BLE001
            return f"book system failed: {exc}"

        try:
            if verb == "list" or not verb:
                books = forge.list_books()
                if not books:
                    return "no books yet. /book <topic> [chapters] writes one and sends the pdf."
                lines = ["books:"]
                for b in books:
                    lines.append(f"  {b['slug']} — {b['status']} · {b['written']}/{b['chapters']} ch · "
                                 f"{b['words']} words")
                lines.append("\n/book status [slug] for details · /book build <slug> for the pdf")
                return "\n".join(lines)

            if verb == "status":
                if len(parts) > 1:
                    return _one_line(parts[1])
                books = forge.list_books()
                if not books:
                    return "no books on disk yet."
                return "\n".join([f"{b['slug']} — {b['status']} · {b['written']}/{b['chapters']} ch · "
                                  f"{b['words']} words" for b in books])

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

            # ── new book ────────────────────────────────────────────────────
            # chapters: explicit number wins, otherwise inferred from the topic
            from ...books.forge import infer_chapter_count
            chapters = 0
            if parts and parts[-1].isdigit():
                chapters = max(3, min(int(parts[-1]), 24))
                topic = " ".join(parts[:-1]).strip()
            else:
                topic = query
            if not topic:
                return ("usage: /book <topic> [chapters] — e.g. /book eBPF for system security 8\n"
                        "it researches the topic, plans chapters, writes them, builds a real "
                        "pdf, and sends it here when done. /book list · /book status")
            slug = slugify(topic)
            with self._queue_guard:
                if slug in self._book_busy:
                    return f"already writing {slug!r} — check /book status {slug}"
                self._book_busy.add(slug)

            chat = self._ref_from_key(chat_key)
            chapter_note = (f"{chapters} chapters" if chapters
                            else "chapters inferred from the topic")
            try:
                self.gateway.send(
                    chat.platform, chat,
                    f"✍️ writing “{topic[:70]}” — {chapter_note}. "
                    "research → outline → write → pdf → straight to this chat. "
                    "check progress: /book status " + slug,
                )
            except Exception:  # noqa: BLE001 - start note is best-effort
                pass

            def _work() -> None:
                try:
                    try:
                        book = forge.create(topic, chapters=chapters, research=True)
                        title = book.display_title
                    except BookError as exc:
                        self._notify(chat, f"⚠️ book failed: {exc}")
                        return
                    # mid-run note at the half-chapter mark, so a long book
                    # doesn't look dead (chapters can take real minutes)
                    half = max(1, len(book.chapters) // 2)
                    written = 0
                    while True:
                        r = forge.write_next(slug)
                        written += 1
                        if r.get("done") and not r.get("chapter"):
                            break
                        if written == half:
                            self._notify(chat, (
                                f"✍️ {title[:60]} — halfway: "
                                f"{r.get('chapters_written', written)}/{r.get('total_chapters', len(book.chapters))} "
                                f"chapters, {r.get('total_words', 0)} words so far."))
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

            threading.Thread(target=_work, name=f"book-{slug}", daemon=True).start()
            return ""  # the start note is already on its way
        except BookError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            return f"book failed: {exc}"
