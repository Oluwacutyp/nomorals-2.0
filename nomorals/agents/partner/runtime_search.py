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

    def _control_novel(self, tail: str, chat_key: str) -> str:
        """``/novel`` — story reader + continuer.

        follow <url|title> | list | read <slug> [chapter] |
        next <slug> | continue <slug> [n]
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

        if verb == "list":
            stories = reader.following()
            if not stories:
                return ("no stories followed yet. /novel follow <url or title> — "
                        "e.g. /novel follow my vampire system")
            lines = ["followed stories:"]
            for s in stories:
                lines.append(
                    f"  {s['slug']} — {s.get('title', '?')} "
                    f"[{s.get('source', '?')}] ch {s.get('current_chapter', 0)}/"
                    f"{s.get('total_chapters') or '?'}")
            return "\n".join(lines)

        if verb == "follow":
            if not rest:
                return "usage: /novel follow <url or story title>"
            try:
                story = reader.follow(rest)
            except ReaderError as exc:
                return f"follow failed: {exc}"
            except Exception as exc:  # noqa: BLE001
                return f"follow failed: {type(exc).__name__}: {exc}"
            return (f"📖 following “{story.title}” [{story.source}] — "
                    f"{story.total_chapters or '?'} chapters. "
                    f"/novel read {story.slug}")

        if verb in ("read", "next"):
            if not rest:
                return f"usage: /novel {verb} <slug> [chapter]"
            bits = rest.split()
            slug, ch = bits[0], int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else 0
            try:
                data = reader.next(slug) if verb == "next" else reader.read(slug, ch)
            except ReaderError as exc:
                return f"read failed: {exc}"
            except Exception as exc:  # noqa: BLE001
                return f"read failed: {type(exc).__name__}: {exc}"
            text = data.get("text") or data.get("chapter_text") or ""
            title = data.get("title") or f"chapter {data.get('number', '?')}"
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(
                chat.platform, chat,
                f"📖 {title}\n\n{text[:5500]}")

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
            except Exception:  # noqa: BLE001
                pass

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
            except Exception:  # noqa: BLE001
                pass

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
                branches = list_branches(slug)
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
                r = StoryBranch(slug, name).fork(from_ch, premise)
                if not r.get("ok"):
                    return f"fork failed: {r.get('reason')}"
                return (f"🌿 branched '{name}' from ch {from_ch}: {premise[:100]} — "
                        f"write into it, merge or abandon when decided.")
            if action in ("merge", "abandon"):
                if len(bits) < 3:
                    return f"usage: /novel branch <slug> {action} <name>"
                r = StoryBranch(slug, bits[2]).merge() if action == "merge" \
                    else StoryBranch(slug, bits[2]).abandon()
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
            pub = SerialPublication(slug)
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
                "/novel read <slug> [chapter] | /novel next <slug> | "
                "/novel continue <slug> [n] | /novel write <premise> [genre] | "
                "/novel branch <slug> ... | /novel publish <slug> ... | "
                "/novel scene <slug> <character> <prompt>")
