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
        from ..news import NewsAgent
        from ..notifier import Notifier

        if not feature_enabled(self.context, "news"):
            return "news is off. /features news on"
        agent = NewsAgent(self.context, notifier=Notifier(self.context, self.gateway))
        parts = tail.split()
        verb = parts[0].lower() if parts else "run"
        if verb == "status":
            rows = agent.recent(8)
            if not rows:
                return "no news stored yet — /news run to fetch the feeds."
            lines = ["recent news:"]
            for row in rows:
                lines.append(f"  [{row['source']}] {row['title'][:70]}")
            return "\n".join(lines)
        if verb == "run":
            cap = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            report = agent.run(cap)
            if not report["ok"]:
                errs = "; ".join(report["errors"][:3])
                return f"news run got nothing readable ({report['items']} items)" + (f" — {errs}" if errs else "")
            return (f"news digest — {report['items']} new items (of {report['fresh']} fresh) "
                    f"— delivered where a channel is live.\n\n{report['digest'][:3000]}")
        return "usage: /news [run [n]|status]"

    # ── always-on research ───────────────────────────────────────────────────
    def _control_research(self, tail: str) -> str:
        from ..features import feature_enabled
        from ..news import NewsAgent  # noqa: F401 - keeps imports local & symmetric
        from ..notifier import Notifier
        from ..researcher import DOMAINS, ResearchAgent

        if not feature_enabled(self.context, "research"):
            return "research is off. /features research on"
        agent = ResearchAgent(self.context, notifier=Notifier(self.context, self.gateway))
        parts = tail.split()
        verb = parts[0].lower() if parts else "status"
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
        return "usage: /research [run [lifestyle|tech|cyber]|status|ideas [n]|approve <id|latest>|deny <id|latest>|history [n]]"

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
