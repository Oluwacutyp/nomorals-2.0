"""RuntimeLiveMixin: PartnerRuntime command group (live)."""

from __future__ import annotations


class RuntimeLiveMixin:
    """RuntimeLiveMixin for :class:`PartnerRuntime`."""


    # ── weather awareness + USA situations + timezones ───────────────────
    def _control_weather(self, tail: str) -> str:
        """/weather [place] | usa | forecast <place> | alerts <place> — keyless live weather."""
        from ..weather import Weather, USASituations, owner_tz, tz_note

        w = Weather()
        sub = (tail or "").strip()
        if not sub or sub.lower() == "now":
            return w.now()["text"]
        low = sub.lower()
        if low == "usa":
            return USASituations(w).overview()["text"]
        if low.startswith("forecast"):
            place = sub[8:].strip() or None
            return w.forecast(place, days=3)["text"]
        if low.startswith("alerts"):
            place = sub[6:].strip() or None
            res = w.now(place)
            alerts = res.get("alerts") or []
            if not alerts:
                return f"no active alerts — {res['text']}"
            return "\n".join(f"• [{a.get('severity','?')}] {a.get('title','')}"
                             for a in alerts[:8])
        if low.startswith("tz"):
            return tz_note(owner_tz())
        return w.now(sub)["text"]

    def _control_tz(self, tail: str) -> str:
        """/tz [YYYY-MM-DD HH:MM [from-zone] [to-zone]] — convert time zones."""
        from ..weather import convert_time, owner_tz, tz_note

        toks = (tail or "").strip().split()
        if not toks:
            return tz_note(owner_tz())
        # optional "YYYY-MM-DD HH:MM" glued at the front
        raw = " ".join(toks[:2]) if len(toks) > 1 and toks[1].count(":") else toks[0]
        rest = toks[2:] if len(toks) > 1 and toks[1].count(":") else toks[1:]
        from_zone = rest[0] if len(rest) > 0 else owner_tz()
        to_zone = rest[1] if len(rest) > 1 else owner_tz()
        res = convert_time(raw, from_zone, to_zone)
        return res.get("text") or f"couldn't parse that: {res.get('error','')}"

    # ── news ─────────────────────────────────────────────────────────────────
    def _control_news(self, tail: str) -> str:
        from ..features import feature_enabled
        from ..news import NewsAgent, publisher_of

        if not feature_enabled(self.context, "news"):
            return "news is off. /features news on"
        # The command reply IS the delivery — NewsAgent is constructed
        # with NO notifier on purpose. NewsAgent.run() publishes through
        # a notifier when given one, and attaching it here pushed the
        # full digest to every owner channel on EVERY /news run: a
        # duplicate inside the owner's own chat, and worse, an
        # owner-channel push triggerable by any non-owner (/news is a
        # PUBLIC command).
        agent = NewsAgent(self.context)
        parts = tail.split()
        verb = parts[0].lower() if parts else "run"
        if verb == "status":
            rows = agent.recent(8)
            if not rows:
                return "no news stored yet — /news run to fetch the feeds."
            lines = ["recent news:"]
            for row in rows:
                # credit the publisher when the row carries attribution
                # (pass #83); legacy rows show their feed name as before
                lines.append(f"  [{publisher_of(row)}] {row['title'][:70]}")
            # Delivery-state census + interest health (pass #85): the
            # degraded no-brain state is VISIBLE here, never silent.
            try:
                st = agent.status()
                lines.append(
                    f"delivery: {st['undelivered']} undelivered, "
                    f"{st['pushed_today']} pushed today "
                    f"(cap {st['push_cap_per_day']}/day), "
                    f"{st['archived']} archived")
                lines.append(
                    f"interest: {st['interest']}"
                    + (" (quiet hours — pushes held)"
                       if st.get("quiet_now") else ""))
                lines.append(
                    "beats: " + ", ".join(st["beats"])
                    + (f" | pins: {', '.join(st['pins'])}"
                       if st["pins"] else ""))
            except Exception:  # noqa: BLE001 - status extras are best-effort
                pass
            return "\n".join(lines)
        if verb == "poll":
            # No notifier on this surface (see the run path's note):
            # a public command must not push to owner channels.  The
            # poll still fetches/stores/scores; high-tier stories are
            # reported held, and the scheduled news-poll job (which
            # owns a notifier) is what pushes them.
            report = agent.poll()
            return (f"news poll — {report['polled']} new stories stored "
                    f"({report['held']} undelivered, "
                    f"{report['archived']} archived low-interest); "
                    f"interest: {report['interest']}; "
                    f"sources: {len(report.get('sources_ok') or [])}/"
                    f"{report.get('sources_tried', 0)} feeds answered")
        if verb == "brief":
            report = agent.brief()
            if not report["ok"]:
                return "nothing undelivered — the digest is empty."
            return (f"news digest — {report['items']} items\n\n"
                    f"{report['digest']}")
        if verb in ("pin", "unpin"):
            topic = tail.split(None, 1)[1].strip() if len(parts) > 1 else ""
            if not topic:
                return f"usage: /news {verb} <topic>"
            profile = (agent.pin(topic) if verb == "pin"
                       else agent.unpin(topic))
            pins = profile["pins"]
            return (f"interest pins: {', '.join(pins) if pins else '(none)'}")
        if verb == "interest":
            profile = agent.interest_profile()
            return ("interest profile\nbeats: " + profile["beats_text"]
                    + "\npins: "
                    + (", ".join(profile["pins"]) or "(none)"))
        if verb == "run" or verb == "briefing":
            cap = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            report = agent.run(cap)
            if not report["ok"]:
                errs = "; ".join(report["errors"][:3])
                return f"news run got nothing readable ({report['items']} items)" + (f" — {errs}" if errs else "")
            # Mark what this run delivered in the pulse seen-store, so the
            # next briefing slot doesn't repeat it as "fresh news"
            # (its _filter_unseen only knows slot-marked keys).
            try:
                from ..morning_pulse import _mark_seen
                urls = report.get("delivered_urls")
                if urls is None:
                    delivered = agent.recent(report["items"])
                    urls = [str(r.get("url") or r.get("title", ""))
                            for r in delivered]
                _mark_seen(self.context, [str(u) for u in urls], "news")
            except Exception:  # noqa: BLE001 - seen-marking is best-effort
                pass
            tried = report.get("sources_tried", 0)
            ok_srcs = report.get("sources_ok") or []
            src_line = (f"sources: {len(ok_srcs)}/{tried} feeds answered "
                        f"({', '.join(ok_srcs[:12])}"
                        f"{'…' if len(ok_srcs) > 12 else ''})")
            stood = report.get("stood_down") or []
            stood_line = ("\n(standing down unhealthy feeds, skipped this "
                          f"run: {', '.join(stood)})" if stood else "")
            # FULL digest — no [:3000] cut; the gateway chunks it losslessly.
            return (f"news digest — {report['items']} new items (of {report['fresh']} fresh)\n{src_line}{stood_line}\n\n{report['digest']}")
        return ("usage: /news [run [n]|briefing|poll|brief|status|"
                "interest|pin <topic>|unpin <topic>]")

    # ── pulse / briefings ──────────────────────────────────────────────
    def _control_pulse(self, tail: str) -> str:
        """Briefing schedule + on-demand runs. Never raises.

        /pulse                → status (all four slots)
/pulse status [slot]   → schedules + last runs
        /pulse config [slot]  → times / enabled / voice
        /pulse run [slot|all] → deliver the briefing now
        /pulse catchup [slot] → run undelivered slots
        Slots: morning · afternoon · evening · midnight.
        """
        try:
            from ..morning_pulse import (
                BRIEFING_SLOTS,
                _read_slot_marker,
                _slot_prefs,
                check_slot_catchup,
                last_pulse_run,
                run_slot,
                slot_enabled,
                slot_time,
                slot_voice,
            )
            parts = (tail or "").split()
            verb = parts[0].lower() if parts else "status"
            slot = parts[1].lower() if len(parts) > 1 else "all"

            def _last_run_text(s: str) -> str:
                try:
                    m = (last_pulse_run(self.context) if s == "morning"
                         else _read_slot_marker(self.context, s))
                    ts = float(m.get("ts") or 0)
                    if not ts:
                        return "never"
                    from datetime import datetime
                    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                except Exception:  # noqa: BLE001
                    return "?"

            def _status_line(s: str) -> str:
                spec = BRIEFING_SLOTS[s]
                tz = str(_slot_prefs(self.context, s).get("timezone") or "")
                return (f"• {s} — {slot_time(self.context, s)} {tz}, "
                        f"{'on' if slot_enabled(self.context, s) else 'off'}, "
                        f"voice {'on' if slot_voice(self.context, s) else 'off'}, "
                        f"last run: {_last_run_text(s)} "
                        f"({spec['title']})")

            if verb == "run":
                if slot in ("all", ""):
                    lines = []
                    for s in BRIEFING_SLOTS:
                        r = run_slot(self.context, s)
                        lines.append(
                            f"• {s}: {'delivered' if r.get('delivered_text') else 'not delivered'}"
                            + (f" (text+audio)" if r.get("delivered_audio") else ""))
                    return "briefings run:\n" + "\n".join(lines)
                if slot not in BRIEFING_SLOTS:
                    return (f"unknown slot {slot!r} — "
                            f"slots: {', '.join(BRIEFING_SLOTS)}")
                r = run_slot(self.context, slot)
                if r.get("ok"):
                    return (f"{slot} briefing: "
                            f"{'delivered' if r.get('delivered_text') else 'not delivered'}"
                            + (f" (text+audio)" if r.get("delivered_audio") else "")
                            + f" — {r.get('elapsed_s', 0)}s")
                return f"{slot} briefing failed: {r.get('error', '?')}"
            if verb == "status":
                if slot in BRIEFING_SLOTS:
                    return _status_line(slot)
                if slot in ("all", ""):
                    return ("briefing schedule:\n" + "\n".join(
                        _status_line(s) for s in BRIEFING_SLOTS))
                return (f"unknown slot {slot!r} — "
                        f"slots: {', '.join(BRIEFING_SLOTS)}")
            if verb == "config":
                if slot in BRIEFING_SLOTS:
                    return _status_line(slot)
                if slot in ("all", ""):
                    return ("briefing config:\n" + "\n".join(
                        _status_line(s) for s in BRIEFING_SLOTS))
                return (f"unknown slot {slot!r} — "
                        f"slots: {', '.join(BRIEFING_SLOTS)}")
            if verb == "catchup":
                if slot in BRIEFING_SLOTS:
                    r = check_slot_catchup(self.context, slot)
                    return f"{slot} catch-up: {r}"
                if slot in ("all", ""):
                    from ..morning_pulse import check_all_catchup
                    return f"catch-up: {check_all_catchup(self.context)}"
                return (f"unknown slot {slot!r} — "
                        f"slots: {', '.join(BRIEFING_SLOTS)}")
            return ("usage: /pulse [status|config|run|catchup] "
                    "[morning|afternoon|evening|midnight|all]")
        except Exception as exc:  # noqa: BLE001 — command never raises
            return f"pulse error: {exc}"

    # ── always-on research ───────────────────────────────────────────────────
    def _control_research(self, tail: str, chat_key: str = "",
                          message: Any = None) -> str:
        from ..features import feature_enabled
        from ..news import NewsAgent  # noqa: F401 - keeps imports local & symmetric
        from ..notifier import Notifier
        from ..researcher import DOMAINS, ResearchAgent

        if not feature_enabled(self.context, "research"):
            return "research is off. /features research on"
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb in {"from", "ask", "docs", "done", "clear"}:
            return self._research_grounded(verb, parts[1:], chat_key, message)
        agent = ResearchAgent(self.context, notifier=Notifier(self.context, self.gateway))
        if verb == "status":
            counts: dict[str, int] = {}
            try:
                for row in self.context.db.query("SELECT domain, COUNT(*) AS n FROM research_log GROUP BY domain"):
                    counts[str(row.get("domain"))] = int(row.get("n", 0))
            except Exception:  # noqa: BLE001
                pass
            lines = [f"research feature: on — domains: {', '.join(sorted(DOMAINS))}"]
            lines.append(f"stored digests: {sum(counts.values())}" +
                         (f" ({', '.join(f'{d} {n}' for d, n in sorted(counts.items()))})" if counts else ""))
            lines.append(f"background loop: {'running' if agent.loop_running() else 'idle'}")
            return "\n".join(lines)
        if verb == "run":
            domain = parts[1] if len(parts) > 1 and parts[1] in DOMAINS else None
            result = agent.run_cycle(domain)
            if not result.get("ok"):
                return f"research failed: {result.get('error')}"
            return (
                f"researched ({result['domain']} / {result.get('researcher', '?')}): {result['topic']}\n"
                f"suggestion: {result['suggestion']}\n"
                f"score: {result.get('score', 0):.2f} → {result.get('status', '?')}\n"
                f"({result.get('seconds', 0)}s — {self._status_note(result.get('status'))})"
            )
        if verb in {"ideas", "queue", "pending"}:
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            rows = agent.list_proposals(limit=n)
            if not rows:
                return "no proposals yet — /research run"
            lines = [f"top proposals ({len(rows)}):"]
            for row in rows:
                lines.append(
                    f"· {row['id'][:8]}  {row['score']:.2f}  [{row['status']}] "
                    f"{row['domain']} — {str(row['topic'])[:52]}\n"
                    f"  {str(row['suggestion'])[:120]}"
                )
            lines.append("approve: /research approve <id|latest>   deny: /research deny <id|latest>")
            return "\n".join(lines)
        if verb == "approve" and len(parts) > 1:
            return agent.approve(parts[1])
        if verb == "deny" and len(parts) > 1:
            return agent.deny(parts[1])
        if verb == "history":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            if self.context.db is None:
                return "no database attached — research history only lives in the full runtime"
            rows = self.context.db.query(
                "SELECT id, domain, topic, score, status, created_at FROM research_log "
                "ORDER BY created_at DESC LIMIT ?", (max(1, min(n, 20)),))
            if not rows:
                return "no research history yet"
            lines = ["recent research:"]
            for row in rows:
                lines.append(
                    f"· {row['id'][:8]}  {float(row.get('score') or 0):.2f}  "
                    f"[{row.get('status', 'pending')}] {row['domain']} — "
                    f"{str(row.get('topic'))[:56]}"
                )
            return "\n".join(lines)
        return ("usage: /research [run [lifestyle|tech|cyber]|status|ideas [n]|approve <id|latest>|deny <id|latest>|history [n]]\n"
                "grounded: /research from [paths] | /research ask <question> | /research docs | /research done")

    def _control_web_research(self, tail: str, *, chat_key: str = "",
                              chat: Any = None) -> str:
        """``/research`` — ad-hoc web research + loop management.

        Bare topic (or ``run <topic>``) → one-shot swarm research on the
        topic, brief back in chat. Loop-management verbs (status, ideas,
        run <domain>, approve, …) delegate to the research loop handler.
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "research"):
            return "research is off. /features research on"
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        loop_verbs = {"status", "ideas", "queue", "pending", "approve",
                      "deny", "history", "from", "ask", "docs", "done",
                      "clear"}
        if verb in loop_verbs:
            return self._control_research(tail, chat_key=chat_key)
        if verb == "run":
            # "run <domain>" → loop cycle; "run <free topic>" → ad-hoc.
            from ..researcher import DOMAINS
            if not rest or rest.lower() in DOMAINS:
                return self._control_research(tail, chat_key=chat_key)
            topic = rest
        elif verb:
            topic = (tail or "").strip()
        else:
            return ("research what? /research <topic> — e.g. "
                    "/research agent coding")
        try:
            from ..research_digest import ResearchPipeline
            result = ResearchPipeline.run(topic, self.context)
        except Exception as exc:  # noqa: BLE001 - honest failure, not silence
            return f"research failed: {type(exc).__name__}: {exc}"
        brief = (result.get("brief") or "").strip()
        if brief:
            return self._send_long_checked(
                chat.platform, chat, brief[:6000]) if chat is not None else brief
        n_find = len((result.get("report") or {}).get("findings") or [])
        if n_find:
            return f"researched {topic!r}: {n_find} findings, brief empty."
        return (f"no solid findings on {topic!r} — the swarm came back "
                f"empty. Try rephrasing or a narrower topic.")

    # ── source-grounded research (per-chat document sessions) ──────────────

    #: mime types treated as groundable documents (besides kind == "document"
    #: attachments and text/*)
    _GROUNDED_DOC_MIMES = frozenset({
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "application/epub+zip",
        "application/json",
        "application/rtf",
        "text/csv",
    })

    #: extensions treated as groundable documents when the mime is missing
    _GROUNDED_DOC_EXTS = frozenset({
        ".pdf", ".txt", ".md", ".markdown", ".rst", ".rtf",
        ".doc", ".docx", ".odt", ".epub",
        ".html", ".htm", ".csv", ".tsv", ".json",
    })

    def _grounded_store(self):
        """Chat-bound grounded session store, created lazily per runtime."""
        store = getattr(self, "_grounded_sessions", None)
        if store is None:
            from pathlib import Path

            from ...research.grounded_store import GroundedSessionStore

            settings = getattr(self.context, "settings", None)
            resolve = getattr(settings, "resolve", None)
            root = resolve("grounded") if callable(resolve) else Path("data/grounded")
            store = GroundedSessionStore(root)
            self._grounded_sessions = store
        return store

    @staticmethod
    def _is_groundable_media(m: Any) -> bool:
        path = getattr(m, "path", None)
        if not path:
            return False
        if getattr(m, "kind", "") == "document":
            return True
        mime = (getattr(m, "mime", "") or "").lower()
        if mime.startswith("text/") or mime in RuntimeLiveMixin._GROUNDED_DOC_MIMES:
            return True
        from pathlib import Path

        return Path(str(path)).suffix.lower() in RuntimeLiveMixin._GROUNDED_DOC_EXTS

    def _research_grounded(self, verb: str, args: list[str],
                           chat_key: str, message: Any) -> str:
        """/research from|ask|docs|done — source-grounded Q&A over chat documents."""
        from ...research.grounded import GroundedError

        store = self._grounded_store()
        if verb == "from":
            return self._grounded_from(store, args, chat_key, message)
        if verb in ("done", "clear"):
            store.drop(chat_key)
            return "grounded session cleared — documents removed."
        session = store.get(chat_key)
        if session is None:
            return ("no grounded session for this chat yet — "
                    "attach documents with /research from first.")
        if verb == "docs":
            docs = store.list_docs(chat_key)
            if not docs:
                return "grounded session is empty — /research from to add documents."
            lines = [f"grounded documents ({len(docs)}):"]
            lines.extend(f"· {d['doc_id']} — {d['title']}" for d in docs)
            return "\n".join(lines)
        # ask
        question = " ".join(args).strip()
        if not question:
            return "usage: /research ask <question>"
        try:
            answer = session.ask(question, context=self.context)
        except GroundedError as exc:
            return f"grounded research failed: {exc}"
        return answer.render()

    def _grounded_from(self, store: Any, args: list[str],
                       chat_key: str, message: Any) -> str:
        """Ingest the triggering message's document attachments and/or tail
        file paths into the chat-bound grounded session."""
        from pathlib import Path

        from ...research.grounded import GroundedError

        media_docs = []
        if message is not None:
            for m in (getattr(message, "media", None) or []):
                if self._is_groundable_media(m):
                    media_docs.append(m)
        path_args = [a for a in args if Path(a).is_file()]
        if not media_docs and not path_args:
            return ("nothing to ground on — attach a document to the message "
                    "or pass file paths: /research from <path> ...")

        titles: list[str] = []
        errors: list[str] = []
        jobs: list[tuple[str, str, str]] = []  # (filename, source path, mime)
        for m in media_docs:
            jobs.append((getattr(m, "name", "") or Path(str(m.path)).name,
                         str(m.path), getattr(m, "mime", "") or ""))
        for p in path_args:
            jobs.append((Path(p).name, p, ""))

        for filename, source, mime in jobs:
            try:
                data = Path(source).read_bytes()
            except OSError as exc:
                errors.append(f"{filename}: unreadable ({exc})")
                continue
            try:
                store.add_upload(chat_key, data, filename, mime=mime)
            except (GroundedError, ValueError) as exc:
                errors.append(f"{filename}: {exc}")
                continue
            titles.append(filename)

        if not titles:
            detail = "; ".join(errors[:3])
            return "couldn't ingest any documents" + (f": {detail}" if detail else ".")
        reply = (f"grounded on {len(titles)} document(s): {', '.join(titles)}. "
                 f"Ask with /research ask <question>")
        if errors:
            reply += f" ({len(errors)} failed: " + "; ".join(errors[:2]) + ")"
        return reply

    # ── finance: the native-TA FinancialExpert ─────────────────────────────
    def _control_finance(self, tail: str, chat_key: str) -> str:
        """Conversational finance over free market data.

        /finance quote <symbol> [market]
        /finance analyze <symbol> [market] [timeframe]
        /finance signal <symbol> [market]
        /finance idea <symbol> [market] [--profile default|aggressive|conservative]
        /finance backtest <symbol> [market] [--profile P] [--strategy NAME]
        /finance strategies — the native strategy zoo
        /finance compare <sym1,sym2,..> [market]
        /finance watch <symbol> <above|below> <price> [market]
        /finance doctor
        """
        from ..financial_expert import FinancialExpert
        from ...integrations import sentinel_bridge as bridge

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        def _usage() -> str:
            return (
                "/finance quote <symbol> [market] — spot price, no engine\n"
                "/finance analyze <symbol> [market] [timeframe] — regime + bias\n"
                "/finance signal <symbol> [market] — directional call\n"
                "/finance idea <symbol> [market] [--profile P] — full trade plan\n"
                "/finance backtest <symbol> [market] [--profile P] [--strategy NAME]\n"
                "/finance strategies — list the strategy zoo\n"
                "/finance compare <s1,s2,..> [market]\n"
                "/finance watch <symbol> <above|below> <price> [market] — price alert\n"
                "/finance doctor — integration health\n"
                "markets: crypto (default) | forex | stocks")

        if verb in ("", "help"):
            return _usage()

        def _market(args: list[str], default: str = "crypto") -> tuple[str, list[str]]:
            if args and args[-1].lower() in ("crypto", "forex", "stocks"):
                return args[-1].lower(), args[:-1]
            return default, args

        try:
            if verb == "doctor":
                return bridge.doctor().summary_text()

            if verb == "strategies":
                rows = FinancialExpert(self.context).strategies()
                lines = ["strategy zoo (native, no submodule needed):"]
                for r in rows:
                    params = ", ".join(f"{k}={v}"
                                       for k, v in r["params"].items())
                    lines.append(f"  {r['name']:15s} [{r['kind']:9s}] "
                                 f"{r['blurb']} ({params})")
                return "\n".join(lines)

            if verb == "quote":
                toks = rest.split()
                if not toks:
                    return "usage: /finance quote <symbol> [market]"
                market, toks = _market(toks)
                q = FinancialExpert(self.context).quote(toks[0], market)
                if not q:
                    return f"no quote for {toks[0]} [{market}]"
                chg = q.get("change_pct_24h")
                chg_s = f" ({chg:+.2f}% 24h)" if isinstance(chg, (int, float)) else ""
                return (f"{q['symbol']} {q['price']:,.2f} "
                        f"{q.get('currency', '')}{chg_s} — via {q['source']}")

            if verb in ("analyze", "signal", "backtest"):
                toks = rest.split()
                if not toks:
                    return f"usage: /finance {verb} <symbol> [market] [timeframe]"
                market, toks = _market(toks)
                symbol = toks[0]
                expert = FinancialExpert(self.context)
                if verb == "analyze":
                    tf = toks[1] if len(toks) > 1 else "1h"
                    return expert.analyze(symbol, market, timeframe=tf).summary_text()
                if verb == "signal":
                    return expert.signal(symbol, market).summary_text()
                profile = "default"
                strategy = None
                for flag, slot in (("--profile", "profile"),
                                   ("--strategy", "strategy")):
                    if flag in toks:
                        try:
                            val = toks[toks.index(flag) + 1]
                        except IndexError:
                            val = None
                        if slot == "profile":
                            profile = val or profile
                        else:
                            strategy = val
                return expert.backtest(symbol, market, strategy=strategy,
                                       profile=profile).summary_text()

            if verb == "idea":
                toks = rest.split()
                if not toks:
                    return "usage: /finance idea <symbol> [market] [--profile P]"
                profile = "default"
                if "--profile" in toks:
                    i = toks.index("--profile")
                    try:
                        profile = toks[i + 1]
                    except IndexError:  # noqa: E103 - missing --profile value keeps default
                        pass
                    toks = toks[:i] + toks[i + 2:]
                market, toks = _market(toks)
                if not toks:
                    return "usage: /finance idea <symbol> [market] [--profile P]"
                return FinancialExpert(self.context).trade_idea(
                    toks[0], market, profile=profile).summary_text()

            if verb == "compare":
                toks = rest.split()
                if not toks:
                    return "usage: /finance compare <s1,s2,..> [market]"
                market, toks = _market(toks)
                symbols = [s.strip() for s in " ".join(toks).split(",") if s.strip()]
                if not symbols:
                    return "usage: /finance compare <s1,s2,..> [market]"
                return FinancialExpert(self.context).compare(symbols, market).summary_text()

            if verb == "watch":
                toks = rest.split()
                # /finance watch BTC below 60000 [crypto]
                if len(toks) < 3:
                    return "usage: /finance watch <symbol> <above|below> <price> [market]"
                market, toks = _market(toks)
                symbol, direction, price_s = toks[0], toks[1].lower(), toks[2]
                if direction not in ("above", "below"):
                    return "usage: /finance watch <symbol> <above|below> <price> [market]"
                try:
                    price = float(price_s.replace(",", ""))
                except ValueError:
                    return f"not a price: {price_s!r}"
                op = "gt" if direction == "above" else "lt"
                res = FinancialExpert(self.context).watch_price(
                    symbol, market,
                    condition={"op": op, "field": "value", "value": price},
                    name=f"{symbol.upper()} {direction} {price:,.4g}")
                if not res.get("ok"):
                    return f"couldn't create the alert: {res}"
                w = res["watcher"]
                return (f"watching {symbol.upper()} [{market}] — alert when "
                        f"price goes {direction} {price:,.4g} "
                        f"(watcher {w.get('id')}). {res.get('echo', '')}".strip())
        except bridge.SentinelUnavailable as exc:
            return str(exc)
        except bridge.SentinelError as exc:
            return f"finance error: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat must never traceback
            return f"finance error: {exc}"
        return f"unknown /finance verb {verb!r}\n{_usage()}"

    @staticmethod
    def _status_note(status: str) -> str:
        return {
            "notified": "delivered",
            "pending": "valuable but over the daily cap — queued (see /research ideas)",
            "skipped": "below the quality bar — logged, not delivered",
        }.get(status, "stored")
