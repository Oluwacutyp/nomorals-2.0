"""RuntimeWisdomMixin: PartnerRuntime command group (wisdom keeper)."""

from __future__ import annotations

import logging

from ...wisdom import PracticeError, WisdomKeeper
from .failures import record_failure

_log = logging.getLogger(__name__)

_WISDOM_USAGE = (
    "usage:\n"
    "  /wisdom ask <query>                  ask the corpus (answer + provenance)\n"
    "  /wisdom read <work> [chapter] [page]  read the actual full text\n"
    "  /wisdom traditions [tradition]       what each tradition is + its works\n"
    "  /wisdom seeds                        every seed source + its works\n"
    "  /wisdom <tradition>                 list one tradition's works\n"
    "  /wisdom practice list                list guided sessions\n"
    "  /wisdom practice <session-id>        start a guided session in this chat\n"
    "  /wisdom practice stop                stop the running session\n"
    "  /wisdom timeline [tradition] [n]     history timeline (full summaries)\n"
    "  /wisdom compare <topic>              cross-tradition comparison\n"
    "  /wisdom oral submit <id> [--lang L] [--pack P] [--restriction R]\n"
    "      [--granted-by NAME --scope S] — then the transcript on the next lines\n"
    "  /wisdom oral verify <id> [--notes ...] [--reject]\n"
    "  /wisdom oral ingest <id> | list [--status S] | read <id>\n"
    "  /wisdom status                       corpus status with per-work receipts"
)


class RuntimeWisdomMixin:
    """RuntimeWisdomMixin for :class:`PartnerRuntime`."""

    # ── chat practice sessions (one manager per runtime) ──────────────
    def _wisdom_chat_manager(self):
        """Lazily-built :class:`WisdomChatManager` for this runtime."""
        mgr = self.__dict__.get("_wisdom_chat_manager_inst")
        if mgr is None:
            from ...wisdom.chat_session import WisdomChatManager
            mgr = WisdomChatManager(self.context)
            self.__dict__["_wisdom_chat_manager_inst"] = mgr
        return mgr

    def _wisdom_incoming(self, chat_key: str, text: str):
        """Route a plain incoming message to a live practice session.

        Returns the reply to send, or None when no session owns this
        message (normal flow continues). Never raises — a routing
        failure must not eat the owner's message.
        """
        try:
            return self._wisdom_chat_manager().handle_incoming(chat_key, text)
        except Exception:  # noqa: BLE001
            _log.warning("wisdom incoming routing failed", exc_info=True)
            return None

    # ── /wisdom ───────────────────────────────────────────────────────
    def _control_wisdom(self, tail: str, *, chat_key: str,
                        message: object = None) -> str:
        """Dispatch ``/wisdom`` (and its ``/wis`` alias)."""
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        keeper = WisdomKeeper(self.context)
        chat = (getattr(message, "chat", None)
                if message is not None else None)
        if chat is None:
            chat = self._ref_from_key(chat_key)

        if verb in ("", "help"):
            return _WISDOM_USAGE
        if verb == "status":
            return self._control_wisdom_status(keeper)
        if verb == "seed":
            return self._control_wisdom_seed(keeper, chat)
        if verb == "ask":
            return self._control_wisdom_ask(keeper, rest, chat)
        if verb == "read":
            return self._control_wisdom_read(keeper, rest, chat)
        if verb == "traditions":
            return self._control_wisdom_traditions(keeper, rest, chat)
        if verb == "seeds":
            return self._control_wisdom_seeds(keeper, chat)
        if verb == "practice":
            return self._control_wisdom_practice(keeper, rest,
                                                 chat_key=chat_key, chat=chat)
        if verb == "timeline":
            return self._control_wisdom_timeline(keeper, rest, chat)
        if verb == "compare":
            return self._control_wisdom_compare(keeper, rest, chat)
        if verb == "oral":
            return self._control_wisdom_oral(keeper, rest, chat)
        # Bare tradition: `/wisdom christian-gnostic` lists its works.
        if not rest:
            from ...wisdom.reading import normalize_tradition
            known = {normalize_tradition(e.tradition)
                     for e in keeper.corpus.list()}
            known |= {normalize_tradition(t)
                      for t in keeper.history.traditions()}
            if normalize_tradition(verb) in known:
                return self._control_wisdom_traditions(keeper, verb, chat)
        # Unknown verb → say so (plus usage). Silently treating the
        # tail as a question turns a typo'd verb (e.g. /wisdom stauts)
        # into a nonsense question instead of a correction.
        return f"unknown /wisdom verb {verb!r}\n\n{_WISDOM_USAGE}"

    def _control_wisdom_status(self, keeper) -> str:
        try:
            full = keeper.status()
            detail = keeper.corpus.traditions_detail()
        except Exception as exc:  # noqa: BLE001 - fail fast, honestly
            return f"⚠️ wisdom status failed: {type(exc).__name__}: {exc}"
        c = full["corpus"]
        lines = [f"📚 corpus: {c['ingested']}/{c['texts']} texts ingested",
                 f"🎙️ oral tradition: "
                 f"{full.get('oral', {}).get('ingested', 0)} ingested draft(s)",
                 "A \"tradition\" = a family of texts + teachings that grew "
                 "together around shared scriptures and practices."]
        for trad in detail:
            lines.append(f"\n{trad['label']} ({trad['tradition']}): "
                         f"{trad['ingested']}/{trad['total']} ingested")
            for w in trad["works"]:
                state = (f"ingested {w['ingested_date']}, "
                         f"{w['chars']:,} chars, {w['pages']} pp"
                         if w["ingested"] else "PENDING")
                lines.append(f"  {'✓' if w['ingested'] else '○'} {w['title']} "
                             f"— seed: {w['seed']} — {state}")
                lines.append(f"      {w['read_command']} · {w['source_url']}")
        return "\n".join(lines)

    def _control_wisdom_read(self, keeper, rest: str, chat) -> str:
        from ...wisdom.reading import parse_read_args
        tokens = (rest or "").strip().split()
        if not tokens:
            return ("usage: /wisdom read <work> [chapter] [page]\n"
                    "e.g. /wisdom read gospel-of-thomas\n"
                    "     /wisdom read gospel-of-thomas 2  (page 2)\n"
                    "     /wisdom read gospel-of-thomas 2 1  (chapter 2, page 1)")
        # Longest leading prefix that names a work; the rest is chapter/page.
        got = None
        err: Exception | None = None
        for split in range(len(tokens), 0, -1):
            try:
                chapter, page = parse_read_args(tokens[split:])
                got = keeper.corpus.read_work(" ".join(tokens[:split]),
                                              chapter=chapter, page=page)
                break
            except Exception as exc:  # noqa: BLE001 - try shorter prefix
                err = exc
                continue
        if got is None:
            return f"⚠️ read failed: {type(err).__name__}: {err}"
        head = got["title"]
        if got["chapter"]:
            head += f" — chapter {got['chapter']}"
            if got["chapter_title"]:
                head += f": {got['chapter_title']}"
        lines = [f"📖 {head} (page {got['page']}/{got['total_pages']}, "
                 f"{got['total_chars']:,} chars total)",
                 f"   source: {got['source_url']} · seed: {got['seed']}",
                 "─" * 24, got["text"], "─" * 24]
        if got["has_prev"]:
            lines.append(f"← prev: {got['prev_command']}")
        if got["has_next"]:
            lines.append(f"next → {got['next_command']}")
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines))

    def _control_wisdom_traditions(self, keeper, rest: str, chat) -> str:
        try:
            detail = keeper.corpus.traditions_detail((rest or "").strip())
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ traditions failed: {type(exc).__name__}: {exc}"
        if not detail:
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                "no traditions registered yet.",
                [[("🌱 Seed the library", "wisdom seed")]])
        lines = ["A \"tradition\" = a family of texts + teachings that grew "
                 "together around shared scriptures, teachers, and practices."]
        for trad in detail:
            lines.append(f"\n{trad['label']} ({trad['tradition']}): "
                         f"{trad['ingested']}/{trad['total']} ingested")
            lines.append(f"  {trad['description']}")
            for w in trad["works"]:
                state = (f"ingested, {w['chars']:,} chars, {w['pages']} pp"
                         if w["ingested"] else "PENDING")
                lines.append(f"  {'✓' if w['ingested'] else '○'} {w['title']} "
                             f"({w['slug']}) — seed: {w['seed']} — {state}")
                lines.append(f"      {w['read_command']} · {w['source_url']}")
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines))

    def _control_wisdom_seeds(self, keeper, chat) -> str:
        try:
            seeds = keeper.corpus.seeds()
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ seeds failed: {type(exc).__name__}: {exc}"
        lines = ["Seeds — every source WisdomKeeper knowledge can come from. "
                 "A seed is a starting source list / discovery channel, "
                 "not the texts themselves."]
        for s in seeds:
            lines.append(f"\n{s['id']} — {s['name']} ({s['works']} work(s))")
            lines.append(f"  {s['what']}")
            lines.append(f"  Add: {s['how_to_add']}")
            if s["work_slugs"]:
                lines.append(f"  works: {', '.join(s['work_slugs'][:12])}"
                             + (" …" if len(s["work_slugs"]) > 12 else ""))
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines))

    def _control_wisdom_seed(self, keeper, chat) -> str:
        """Seed the manifest and ingest the starter library in background."""
        try:
            n = keeper.corpus.seed()
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ seed failed: {type(exc).__name__}: {exc}"
        st = keeper.status()["corpus"]
        total, ingested = st["texts"], st["ingested"]
        if ingested >= total and total > 0:
            return f"📚 corpus already complete: {ingested}/{total} texts ingested."
        # Ingest in background — 47 books take a while on a slow link.
        import threading

        def _bg():
            try:
                from ...wisdom import ArchiveIngestor
                ing = ArchiveIngestor(self.context)
                for e in keeper.corpus.list():
                    try:
                        ing.ingest_entry(e)
                    except Exception as exc:  # noqa: BLE001 - one bad book, not a dead run
                        record_failure("runtime_wisdom._bg:221", exc)
                        continue
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_wisdom._bg:223", exc)

        threading.Thread(target=_bg, daemon=True,
                         name="wisdom-seed").start()
        from ...social.chat.tgbot_buttons import actionable

        return actionable(
            f"📚 seeded {n} new texts ({total} total). Ingesting "
            f"the library in the background — ask away as it fills.",
            [[("📊 Ingest status", "wisdom status")]])

    def _control_wisdom_ask(self, keeper, query: str, chat) -> str:
        if not query:
            return "ask me anything — e.g. /wisdom when was the earth created?"
        try:
            st = keeper.status()["corpus"]
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_wisdom._control_wisdom_ask", exc)
            st = {}
        try:
            ans = keeper.ask(query, top=5)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ ask failed: {type(exc).__name__}: {exc}"
        lines = [ans.synthesis, ""]
        for p in ans.passages[:5]:
            lines.append(f"[{p.work} — {p.section}]")
            lines.append(f"  {p.snippet.strip()}")
            src = p.url or "(no url)"
            lines.append(f"  source: {src} [{p.canon_status}]")
            if p.slug:
                ch = f" {p.chapter_number}" if p.chapter_number else ""
                lines.append(f"  📖 Read full: /wisdom read {p.slug}{ch}")
            lines.append("")
        if not ans.passages and not st.get("ingested"):
            # Truly empty corpus (not just no match): keep the seed
            # pointer the old early-return carried, but route the
            # reply through the checker like every other ask answer —
            # no reply class bypasses _send_long_checked.
            lines.append(
                "📚 the corpus is empty — run /wisdom seed and I'll fetch "
                "the 47-text starter library (public-domain scriptures, "
                "hermetic, gnostic, and wisdom texts), then ask again.")
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines).strip())

    def _control_wisdom_practice(self, keeper, rest: str, *,
                                 chat_key: str, chat) -> str:
        sub = (rest or "").strip().split(None, 1)
        subverb = sub[0].lower() if sub else "list"
        manager = self._wisdom_chat_manager()
        if subverb == "list":
            try:
                sessions = keeper.practice.list_sessions()
            except Exception as exc:  # noqa: BLE001
                return f"⚠️ practice list failed: {type(exc).__name__}: {exc}"
            lines = ["🧘 practice sessions:"]
            for s in sessions:
                tag = "beginner" if s["beginner"] else "advanced"
                total = int(s["total_seconds"])
                lines.append(
                    f"  /wisdom practice {s['id']} — {s['name']} "
                    f"({total // 60}m{total % 60:02d}s, {tag})")
            lines.append("start one in this chat: /wisdom practice <session-id>")
            return "\n".join(lines)
        if subverb in ("stop", "cancel"):
            return manager.stop_session(chat_key)
        if subverb == "status":
            st = manager.status(chat_key)
            if st["active"]:
                a = st["active"]
                return (f"🧘 running: {a['name']} ({a['session_id']}) "
                        f"— state {a['state']}")
            if st["journal_await"]:
                return (f"no active session — journal reply awaited for "
                        f"{st['journal_await']}.")
            return "no practice session is running here."
        session_id = sub[0]
        try:
            return manager.start_session(
                chat_key, session_id, platform=chat.platform)
        except PracticeError as exc:
            # unknown session id — the message already lists what's available
            return f"⚠️ {exc}"

    def _control_wisdom_timeline(self, keeper, rest: str, chat) -> str:
        from ...wisdom.reading import parse_timeline_args
        # `timeline <tradition> [n]` — trailing int is the count, never
        # part of the tradition name; matching is case-insensitive.
        tradition, limit = parse_timeline_args((rest or "").split())
        try:
            events = keeper.history.timeline(tradition, -3000, 2100,
                                             limit=limit)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ timeline failed: {type(exc).__name__}: {exc}"
        if not events:
            return "no timeline events found."
        head = (f"📜 timeline — {tradition}"
                if tradition else "📜 timeline")
        show = limit or 10
        shown = events[:show]
        lines = [f"{head} ({len(events)} events, showing {len(shown)}):"]
        for e in shown:
            yr = (f"{e['start']}" if e["start"] == e["end"]
                  else f"{e['start']}…{e['end']}")
            lines.append(f"\n  {yr}: {e['title']} [{e['tradition']}] — "
                         f"{e['region']}")
            lines.append(f"    {e['summary']}")
            lines.append(f"    sources: {', '.join(e['sources'])}")
        if len(events) > len(shown):
            more = (f"/wisdom timeline {tradition} {len(events)}"
                    if tradition else
                    f"/wisdom timeline {len(events)}")
            lines.append(f"\n  …and {len(events) - len(shown)} more — {more}")
        return self._send_long_checked(chat.platform, chat, "\n".join(lines))

    def _control_wisdom_compare(self, keeper, topic: str, chat) -> str:
        if not topic:
            return "usage: /wisdom compare <topic>"
        try:
            out = keeper.history.compare(topic, corpus=keeper)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ compare failed: {type(exc).__name__}: {exc}"
        lines = [f"⚖️ topic: {out['topic']}", ""]
        for p in out["passages"][:5]:
            lines.append(f"[{p.get('work', '?')} — {p.get('canon_status', '')}]")
            lines.append(f"  {p.get('snippet', '')}")
            if p.get("slug"):
                ch = f" {p['chapter_number']}" if p.get("chapter_number") else ""
                lines.append(f"  📖 Read full: /wisdom read {p['slug']}{ch}")
            lines.append("")
        lines.append("— timeline context —")
        for e in out["timeline_context"][:10]:
            lines.append(f"  {e['start']}: {e['title']} — {e.get('summary', '')}")
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines).strip())

    # ── /wisdom oral ────────────────────────────────────────────────
    def _control_wisdom_oral(self, keeper, rest: str, chat) -> str:
        """Dispatch ``/wisdom oral <subverb>``.

        Submit takes its transcript from the lines AFTER the first
        (the command line carries the id + ``--flags``); verify /
        ingest / list / read take only the first line.
        """
        import shlex

        from ...wisdom.oral import parse_oral_flags

        lines = (rest or "").splitlines()
        try:
            head = shlex.split(lines[0]) if lines else []
        except ValueError as exc:
            return f"⚠️ oral: cannot parse the command line: {exc}"
        if not head:
            return ("usage:\n"
                    "  /wisdom oral submit <id> [--lang L] [--pack P] "
                    "[--restriction R]\n"
                    "      [--granted-by NAME --scope S] — then the "
                    "transcript on the next lines\n"
                    "  /wisdom oral verify <id> [--notes ...] [--reject]\n"
                    "  /wisdom oral ingest <id> | list [--status S] | "
                    "read <id>")
        sub = head[0].lower()
        try:
            flags, pos = parse_oral_flags(head[1:])
        except ValueError as exc:
            return f"⚠️ oral: {exc}"
        if sub == "submit":
            return self._oral_chat_submit(keeper, flags, pos,
                                          "\n".join(lines[1:]).strip())
        if sub == "verify":
            return self._oral_chat_verify(keeper, flags, pos)
        if sub == "ingest":
            return self._oral_chat_ingest(keeper, pos)
        if sub == "list":
            return self._oral_chat_list(keeper, flags)
        if sub == "read":
            return self._oral_chat_read(keeper, pos, chat)
        return f"unknown /wisdom oral verb {sub!r}\n\n{_WISDOM_USAGE}"

    def _oral_chat_permission(self, flags):
        granted_by = str(flags.get("granted-by", "") or "")
        scope = str(flags.get("scope", "") or "")
        contact_ref = str(flags.get("contact-ref", "") or "")
        if not granted_by and (scope or contact_ref):
            raise ValueError(
                "--scope / --contact-ref need --granted-by (who granted it)")
        if not granted_by:
            return None
        from ...wisdom import PermissionRecord
        return PermissionRecord(granted_by=granted_by, scope=scope,
                                contact_ref=contact_ref)

    def _oral_chat_submit(self, keeper, flags, pos, transcript: str) -> str:
        if not pos:
            return ("usage: /wisdom oral submit <id> [--lang L] [--pack P] "
                    "[--restriction R] [--granted-by NAME --scope S]\n"
                    "then the transcript on the next lines "
                    "(\"SPEAKER: text\" lines)")
        if not transcript:
            return ("⚠️ oral submit needs the transcript after the first "
                    "line — paste it on the following lines")
        try:
            permission = self._oral_chat_permission(flags)
        except ValueError as exc:
            return f"⚠️ oral: {exc}"
        try:
            draft = keeper.oral.submit_draft(
                pos[0],
                audio_ref=str(flags.get("audio-ref", "") or ""),
                transcript=transcript,
                language=str(flags.get("language", "")
                             or flags.get("lang", "") or ""),
                tradition_pack_id=str(flags.get("pack", "") or ""),
                restriction=str(flags.get("restriction", "open") or "open"),
                permission=permission,
                contributor=str(flags.get("contributor", "") or ""))
        except Exception as exc:  # noqa: BLE001 - fail fast, honestly
            return f"⚠️ oral submit failed: {type(exc).__name__}: {exc}"
        return (f"🎙️ draft {draft.draft_id} submitted: "
                f"{len(draft.turns)} turn(s), "
                f"{draft.language or '-'} / "
                f"{draft.tradition_pack_id or '-'}, "
                f"{draft.restriction} — verify with: "
                f"/wisdom oral verify {draft.draft_id}")

    def _oral_chat_verify(self, keeper, flags, pos) -> str:
        if not pos:
            return "usage: /wisdom oral verify <id> [--notes ...] [--reject]"
        try:
            draft = keeper.oral.verify(
                pos[0], owner="owner",
                approve=not bool(flags.get("reject", False)),
                notes=str(flags.get("notes", "") or ""))
        except Exception as exc:  # noqa: BLE001 - fail fast, honestly
            return f"⚠️ oral verify failed: {type(exc).__name__}: {exc}"
        if draft.status == "verified":
            return (f"🎙️ draft {draft.draft_id} verified by owner — "
                    f"ingest with: /wisdom oral ingest {draft.draft_id}")
        return (f"🎙️ draft {draft.draft_id} rejected — it can never "
                f"ingest; submit a fresh draft to try again")

    def _oral_chat_ingest(self, keeper, pos) -> str:
        if not pos:
            return "usage: /wisdom oral ingest <id>"
        try:
            out = keeper.oral.ingest(pos[0])
        except Exception as exc:  # noqa: BLE001 - fail fast, honestly
            return f"⚠️ oral ingest failed: {type(exc).__name__}: {exc}"
        return (f"🎙️ ingested: {out['draft_id']} ({out['turns']} turn(s), "
                f"{out['words']} words) — ask about it with "
                f"/wisdom ask, or read it with "
                f"/wisdom oral read {out['draft_id']}")

    def _oral_chat_list(self, keeper, flags) -> str:
        status = str(flags.get("status", "") or "")
        drafts = keeper.oral.list_drafts(status=status)
        ingested = keeper.oral.list_ingested()
        if not drafts and not ingested:
            return ("no oral drafts yet — submit one with:\n"
                    "/wisdom oral submit <id> [--lang L] [--pack P]\n"
                    "then the transcript on the next lines")
        lines = ["🎙️ oral drafts:"]
        for d in drafts:
            perm = ("permission on file" if d.get("permission")
                    else "no permission")
            lines.append(f"  {d['draft_id']}: {d['status']} — "
                         f"{len(d.get('turns', []))} turn(s), "
                         f"{d.get('language') or '-'} / "
                         f"{d.get('tradition_pack_id') or '-'}, "
                         f"{d.get('restriction')}, {perm}")
        for e in ingested:
            lines.append(f"  {e['draft_id']}: ingested — {e['turns']} "
                         f"turn(s), {e['words']} words, "
                         f"{e.get('language') or '-'} / "
                         f"{e.get('tradition_pack_id') or '-'}, "
                         f"{e['restriction']}")
        return "\n".join(lines)

    def _oral_chat_read(self, keeper, pos, chat) -> str:
        if not pos:
            return "usage: /wisdom oral read <id>"
        try:
            rec = keeper.read_oral(pos[0])
        except Exception as exc:  # noqa: BLE001 - fail fast, honestly
            return f"⚠️ oral read failed: {type(exc).__name__}: {exc}"
        prov = rec.get("provenance", {}) or {}
        perm = rec.get("permission") or {}
        lines = [f"🎙️ oral draft {rec['draft_id']} — "
                 f"{rec.get('language') or '-'} / "
                 f"{rec.get('tradition_pack_id') or '-'} "
                 f"({rec.get('restriction', 'open')})",
                 f"contributor: {prov.get('contributor') or '-'} · "
                 f"permission: {perm.get('granted_by') or '-'} "
                 f"({perm.get('scope') or '-'})",
                 "─" * 40]
        for t in rec.get("turns", []):
            lines.append(f"{t.get('speaker', 'unknown')}: "
                         f"{t.get('text', '')}")
        return self._send_long_checked(chat.platform, chat,
                                       "\n".join(lines))
