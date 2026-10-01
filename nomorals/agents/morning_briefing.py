"""Prompt 04 — morning briefing: the overnight digest.

One concise briefing every morning: what happened overnight, what needs
attention today, what's worth knowing.  Composed from pluggable sections,
personalized by explicit topics (no implicit profiling in v1), delivered
through the Notifier.

Composition (survey-first, per the standing rule):
- ``NewsAgent`` (agents/news.py) — RSS digest, consumed not replaced.
- ``Notifier`` (agents/notifier.py) — durable multi-channel delivery.
- Prompt 03 watchers — held ``info`` alerts via ``WatcherStore.held_alerts()``
  / ``mark_digested()``; repo activity via held repo-kind alerts.
- ``Scheduler`` (agents/scheduler.py) — one durable ``briefing`` job,
  ``daily HH:MM`` in the owner's timezone.
- Prompt 05 rooms — dirty/stale room flags in their own section.
- Markets — ``MarketDataProvider`` protocol; the shipped provider reuses
  the CoinGecko keyless pattern from the watchers ``price`` kind.

The briefing is READ-ONLY over the owner's data: it summarizes, it never
acts.  ``nomorals/agents/brief.py`` is the *mission* briefing agent and is
unrelated — this module is deliberately named ``morning_briefing.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: hard cap on briefing length (spec: ~600 words / 40 lines)
MAX_WORDS = 600
#: anti-monopoly: max items from a single source per section
MAX_PER_SOURCE = 3
#: scheduler job name (idempotent by name)
BRIEFING_JOB_NAME = "briefing"


# ── data ─────────────────────────────────────────────────────────────────

@dataclass
class BriefingSection:
    """One rendered section of the briefing."""
    name: str
    title: str
    priority: int
    source: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def render_text(self) -> str:
        head = f"— {self.title} —"
        body = self.lines or [
            f"• {i.get('title', '')}" for i in self.items[:8]]
        return "\n".join([head, *body])

    def render_items(self) -> list[dict[str, Any]]:
        return self.items

    def word_count(self) -> int:
        return len(self.render_text().split())


@dataclass
class Briefing:
    id: str
    date: str
    sections: list[BriefingSection]
    generated_at: float
    generation_ms: float
    late: bool = False
    truncated_note: str = ""

    def word_count(self) -> int:
        n = sum(s.word_count() for s in self.sections)
        return n + (len(self.truncated_note.split()) if self.truncated_note
                    else 0)

    def render_text(self) -> str:
        parts = [f"☀️ Morning briefing — {self.date}"
                 + (" (late — Devon was down at briefing time)"
                    if self.late else "")]
        for s in self.sections:
            parts.append("")
            parts.append(s.render_text())
        if self.truncated_note:
            parts.append("")
            parts.append(self.truncated_note)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "date": self.date, "late": self.late,
            "generated_at": self.generated_at,
            "generation_ms": self.generation_ms,
            "word_count": self.word_count(),
            "truncated_note": self.truncated_note,
            "sections": [
                {"name": s.name, "title": s.title, "priority": s.priority,
                 "source": s.source, "items": s.items, "lines": s.lines}
                for s in self.sections],
        }


# ── settings helpers ─────────────────────────────────────────────────────

def _briefing_settings(context: Any) -> Any:
    return getattr(getattr(context, "settings", None), "briefing", None)


def briefing_time(context: Any) -> str:
    """Daily briefing time HH:MM from settings (default 07:00)."""
    return str(_prefs(context).get("time") or "07:00")


def briefing_topics(context: Any) -> list[str]:
    topics = _prefs(context).get("topics", [])
    return [str(t) for t in (topics or []) if str(t).strip()]


def _prefs(context: Any) -> dict[str, Any]:
    """Unified briefing prefs: settings.briefing namespace first, then a
    JSON sidecar next to the workspace dir.  Counts only, no content."""
    out: dict[str, Any] = {"topics": [], "symbols": [],
                           "pinned_sections": [], "time": "07:00"}
    bs = _briefing_settings(context)
    if bs is not None and hasattr(bs, "__dict__"):
        for k in out:
            v = getattr(bs, k, None)
            if v:
                out[k] = v if k == "time" else list(v)
        return out
    try:
        from pathlib import Path
        sidecar = (Path(context.settings.workspace_dir)
                   / "briefing_prefs.json")
        if sidecar.is_file():
            import json as _json
            out.update(_json.loads(sidecar.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return out


def _save_prefs(context: Any, prefs: dict[str, Any]) -> None:
    bs = _briefing_settings(context)
    if bs is not None and hasattr(bs, "__dict__"):
        for k, v in prefs.items():
            setattr(bs, k, v)
        return
    try:
        from pathlib import Path
        import json as _json
        sidecar = (Path(context.settings.workspace_dir)
                   / "briefing_prefs.json")
        sidecar.write_text(_json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _owner_tz(context: Any) -> str:
    settings = getattr(context, "settings", None)
    for attr in ("timezone", "tz", "owner_timezone"):
        tz = getattr(settings, attr, None) if settings else None
        if tz:
            return str(tz)
    partner = getattr(settings, "partner", None) if settings else None
    for attr in ("timezone", "tz"):
        tz = getattr(partner, attr, None) if partner else None
        if tz:
            return str(tz)
    import os
    return os.environ.get("TZ", "UTC")

# ── market data provider ─────────────────────────────────────────────────
# The ``MarketDataProvider`` protocol lives in
# ``nomorals/integrations/sentinel_bridge.py`` — it was defined there
# explicitly for the briefing to depend on (Prompt 07 / FinancialExpert
# plugs a richer Sentinel-backed provider in later).  Reused here per the
# standing survey-first rule; NOT redefined.

try:
    from ..integrations.sentinel_bridge import (
        MarketDataProvider as _MarketDataProvider)
except Exception:  # noqa: BLE001 — integrations optional in some builds
    _MarketDataProvider = None  # type: ignore[assignment]

#: the protocol the briefing's markets section programs against
MarketDataProvider = _MarketDataProvider


class CoinGeckoMarketProvider:
    """Keyless CoinGecko ``simple/price`` quotes — the same pattern the
    watchers ``price`` kind uses in-repo.  Read-only, no key needed.

    Implements the shared ``MarketDataProvider`` protocol (structural:
    ``quote(symbol, market="crypto")`` + ``overnight_movers``)."""

    _SYMBOL_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "TON": "the-open-network", "TRX": "tron",
        "USDT": "tether", "USDC": "usd-coin",
    }

    def __init__(self, context: Any) -> None:
        self.context = context

    def _client(self) -> Any:
        from ..core.http import HttpClient
        settings = getattr(self.context, "settings", None)
        tools = getattr(settings, "tools", None)
        return HttpClient(
            timeout=getattr(tools, "http_timeout", 15.0),
            user_agent=getattr(tools, "user_agent", "NoMoralsCore/0.1"),
            proxy_url=getattr(tools, "proxy_url", ""),
        )

    def quote(self, symbol: str,
              market: str = "crypto") -> dict[str, Any] | None:
        sym = (symbol or "").strip().upper()
        cg_id = self._SYMBOL_MAP.get(sym)
        if not cg_id:
            return None
        try:
            resp = self._client().get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd",
                        "include_24hr_change": "true"},
            )
            data = resp.json().get(cg_id, {})
            price = data.get("usd")
            if price is None:
                return None
            return {"symbol": sym, "price": float(price),
                    "change_pct_24h": data.get("usd_24h_change")}
        except Exception as exc:  # noqa: BLE001 — one bad quote ≠ no briefing
            _log.debug("market quote failed for %s: %s", sym, exc)
            return None

    def overnight_movers(self, symbols: list[str],
                         market: str = "crypto") -> list[dict[str, Any]]:
        """Quotes sorted by absolute overnight change, descending."""
        quotes = []
        for sym in symbols:
            try:
                q = self.quote(sym, market=market)
            except Exception:  # noqa: BLE001
                q = None
            if q:
                quotes.append(q)
        def _abs_chg(q: dict[str, Any]) -> float:
            chg = q.get("change_pct_24h")
            return abs(chg) if isinstance(chg, (int, float)) else 0.0
        quotes.sort(key=_abs_chg, reverse=True)
        return quotes


# ── section providers ────────────────────────────────────────────────────

class _Provider:
    """Base: ``collect(ctx, since) -> BriefingSection | None``."""
    name = "base"
    title = "Base"
    priority = 999

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        raise NotImplementedError


def _watcher_store(ctx: Any) -> Any | None:
    try:
        from .watchers import WatcherStore
        return WatcherStore(getattr(ctx, "db", None))
    except Exception:  # noqa: BLE001 — watchers optional
        return None


class OvernightAlertsProvider(_Provider):
    """Held ``info`` watcher digest items since the last briefing."""
    name = "alerts"
    title = "Overnight alerts"
    priority = 10

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        store = _watcher_store(ctx)
        if store is None:
            return None
        try:
            held = store.held_alerts()
        except Exception:  # noqa: BLE001
            return None
        # repo-kind alerts belong to the repos section — don't double-claim
        try:
            repo_ids = {w.id for w in store.list()
                        if (w.kind or "") == "repo"}
        except Exception:  # noqa: BLE001
            repo_ids = set()
        fresh = [a for a in held
                 if float(a.get("created_at") or 0) >= since
                 and (a.get("severity") or "info") == "info"
                 and a.get("watcher_id") not in repo_ids]
        if not fresh:
            return None
        # anti-monopoly: max 3 per watcher
        by_w: dict[str, list] = {}
        for a in fresh:
            by_w.setdefault(a.get("watcher_id", "?"), []).append(a)
        items = []
        for wid, alerts in by_w.items():
            for a in alerts[:MAX_PER_SOURCE]:
                items.append({
                    "id": a.get("id"), "title": a.get("title", ""),
                    "body": (a.get("body", "") or "")[:300],
                    "watcher_id": wid,
                    "created_at": a.get("created_at"),
                })
        if not items:
            return None
        # mark consumed — the briefing owns these now
        try:
            store.mark_digested([i["id"] for i in items if i.get("id")])
        except Exception:  # noqa: BLE001
            pass
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="watchers",
            items=items,
            lines=[f"• {i['title']}" for i in items[:8]])


class CalendarProvider(_Provider):
    """Today's agenda.  Best-effort: renders nothing when no calendar
    is configured — never an error, never 'not connected' noise."""
    name = "calendar"
    title = "Today"
    priority = 20

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        events = self._today(ctx)
        if not events:
            return None
        items = [{"id": f"cal-{n}", "title": e.get("title", ""),
                  "body": e.get("when", "")} for n, e in enumerate(events[:8])]
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="calendar", items=items,
            lines=[f"• {e.get('when', '')} — {e.get('title', '')}"
                   for e in events[:8]])

    def _today(self, ctx: Any) -> list[dict[str, Any]]:
        # try known calendar surfaces; all optional
        tools = getattr(ctx, "tools", None)
        if tools is not None:
            for tool_name, action in (("calendar", "agenda"),
                                      ("google_calendar", "today")):
                try:
                    out = tools.call(tool_name, action=action)
                    if getattr(out, "ok", False):
                        val = out.value
                        if isinstance(val, dict):
                            evs = val.get("events") or val.get("items")
                            if evs:
                                return list(evs)[:10]
                        if isinstance(val, list) and val:
                            return list(val)[:10]
                except Exception:  # noqa: BLE001 — optional surface
                    continue
        cal = getattr(ctx, "calendar", None)
        if cal is not None:
            try:
                evs = cal.list_events() if hasattr(cal, "list_events") else []
                return list(evs or [])[:10]
            except Exception:  # noqa: BLE001
                pass
        return []


class MarketsProvider(_Provider):
    """Market snapshot for the owner's followed symbols (settings list;
    empty list = section hidden)."""
    name = "markets"
    title = "Markets"
    priority = 30

    def __init__(self, provider: Any | None = None) -> None:
        self._provider = provider

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        symbols = [s for s in (str(x).strip().upper()
                               for x in (_prefs(ctx).get("symbols", []) or []))
                   if s]
        if not symbols:
            return None
        prov = self._provider or CoinGeckoMarketProvider(ctx)
        items = []
        for sym in symbols[:10]:
            try:
                q = prov.quote(sym)
            except Exception:  # noqa: BLE001
                q = None
            if q:
                items.append({"id": f"mkt-{sym}", "title": sym,
                              "body": json.dumps(q, default=str), **q})
        if not items:
            return None
        lines = []
        for q in items:
            chg = q.get("change_pct_24h")
            chg_s = f" ({chg:+.1f}% 24h)" if isinstance(chg, (int, float)) else ""
            lines.append(f"• {q['symbol']}: ${q['price']:,.2f}{chg_s}")
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="markets", items=items, lines=lines)


class RepoProvider(_Provider):
    """Overnight repo activity from held repo-kind watcher alerts."""
    name = "repos"
    title = "Repos"
    priority = 40

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        store = _watcher_store(ctx)
        if store is None:
            return None
        try:
            held = store.held_alerts()
        except Exception:  # noqa: BLE001
            return None
        # repo-kind alerts: match watcher kind via the store's listing
        try:
            watchers = store.list()
            repo_ids = {w.id for w in watchers
                        if (w.kind or "") == "repo"}
        except Exception:  # noqa: BLE001
            repo_ids = set()
        fresh = [a for a in held
                 if float(a.get("created_at") or 0) >= since
                 and a.get("watcher_id") in repo_ids]
        if not fresh:
            return None
        by_w: dict[str, list] = {}
        for a in fresh:
            by_w.setdefault(a.get("watcher_id", "?"), []).append(a)
        items = []
        for wid, alerts in by_w.items():
            for a in alerts[:MAX_PER_SOURCE]:
                items.append({
                    "id": a.get("id"), "title": a.get("title", ""),
                    "body": (a.get("body", "") or "")[:300],
                    "watcher_id": wid})
        try:
            store.mark_digested([i["id"] for i in items if i.get("id")])
        except Exception:  # noqa: BLE001
            pass
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="watchers:repo",
            items=items,
            lines=[f"• {i['title']}" for i in items[:8]])

class NewsProvider(_Provider):
    """Top news FILTERED by the owner's explicit topics (settings list).
    Consumes NewsAgent's stored digest — never replaces its behavior."""
    name = "news"
    title = "Worth knowing"
    priority = 50

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        topics = briefing_topics(ctx)
        try:
            from .news import NewsAgent
            agent = NewsAgent(ctx)
            items = agent.recent(limit=40)
        except Exception:  # noqa: BLE001
            return None
        if topics:
            lowered = [t.lower() for t in topics]
            items = [i for i in items
                     if any(t in f"{i.get('title', '')} "
                                    f"{i.get('summary', '')}".lower()
                            for t in lowered)]
        # anti-monopoly: max 3 per source
        by_src: dict[str, list] = {}
        for i in items:
            by_src.setdefault(i.get("source", "?"), []).append(i)
        picked = []
        for src, src_items in by_src.items():
            for i in src_items[:MAX_PER_SOURCE]:
                picked.append({
                    "id": f"news-{i.get('id', '')}",
                    "title": i.get("title", ""),
                    "body": (i.get("summary", "") or "")[:300],
                    "url": i.get("url", ""),
                    "source": src})
        if not picked:
            return None
        picked = picked[:9]
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="news",
            items=picked,
            lines=[f"• {i['title']} ({i['source']})" for i in picked])


class RoomsProvider(_Provider):
    """Project-room flags: dirty rooms (crashed sessions needing review)
    and stale rooms (idle > N days).  Prompt 05 integration."""
    name = "rooms"
    title = "Project rooms"
    priority = 60

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        try:
            from pathlib import Path
            from ..workspace.rooms import RoomManager
            root = Path(ctx.settings.workspace_dir)
            mgr = RoomManager(root, db=ctx.db)
            rooms = mgr.list()
            stale_slugs = {r.slug for r in mgr.stale_rooms()}
        except Exception:  # noqa: BLE001 — rooms optional
            return None
        dirty = [r for r in rooms if r.dirty and r.status == "active"]
        stale = [r for r in rooms
                 if r.slug in stale_slugs and r.status == "active"]
        if not dirty and not stale:
            return None
        items, lines = [], []
        for r in dirty[:5]:
            items.append({"id": f"room-{r.slug}", "title": r.title,
                          "body": f"dirty — crashed session, needs review "
                                  f"(blockers: {', '.join(r.blockers[:2])})",
                          "slug": r.slug})
            lines.append(f"• ⚠️ {r.title}: crashed session needs review")
        for r in stale[:5]:
            items.append({"id": f"room-{r.slug}", "title": r.title,
                          "body": "idle > 30 days — archive?",
                          "slug": r.slug})
            lines.append(f"• 💤 {r.title}: idle — archive?")
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="rooms", items=items, lines=lines)


class DevonSelfProvider(_Provider):
    """Devon's own overnight status, one line each max: improvement-loop
    edits, watcher stats, scheduler health, failed jobs."""
    name = "devon"
    title = "Devon self-check"
    priority = 100

    def collect(self, ctx: Any, since: float) -> BriefingSection | None:
        lines, items = [], []
        db = getattr(ctx, "db", None)

        # improvement loop (Prompt 01)
        try:
            row = db.query_one(
                "SELECT COUNT(*) AS n FROM skill_edits "
                "WHERE created_at >= ? AND status='applied'", (since,)) \
                if db else None
            n = (row["n"] if row else 0) or 0
            if n:
                lines.append(f"• 🔧 {n} skill self-improvement edit(s) applied")
        except Exception:  # noqa: BLE001 — table may not exist
            pass

        # watcher stats (Prompt 03) — only when there's something to say
        try:
            store = _watcher_store(ctx)
            if store is not None:
                held = len(store.held_alerts())
                watchers = store.list_watchers()
                active = sum(1 for w in watchers
                             if w.get("status") == "active")
                if active or held:
                    lines.append(
                        f"• 👁 {active} watcher(s) active, "
                        f"{held} held for digest")
        except Exception:  # noqa: BLE001
            pass

        # scheduler health: failed jobs since last briefing (only)
        try:
            from .scheduler import Scheduler
            sched = Scheduler(ctx)
            jobs = sched.list_jobs()
            failed = [j for j in jobs
                      if str(j.get("last_status") or "") == "failed"
                      and float(j.get("last_run_at") or 0) >= since]
            for j in failed[:3]:
                lines.append(f"• 🚨 job failed: {j.get('name', '?')}")
                items.append({"id": f"job-{j.get('id')}",
                              "title": j.get("name", ""),
                              "body": str(j.get("last_error") or "")[:200]})
        except Exception:  # noqa: BLE001
            pass

        if not lines:
            return None
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="devon", items=items, lines=lines)


# ── composer ───────────────────────────────────────────────────────────────

class BriefingComposer:
    """Runs section providers, drops empties, orders by priority
    (engagement-adjusted), truncates to the word cap."""

    def __init__(self) -> None:
        self.providers: list[_Provider] = [
            OvernightAlertsProvider(),
            CalendarProvider(),
            MarketsProvider(),
            RepoProvider(),
            NewsProvider(),
            RoomsProvider(),
            DevonSelfProvider(),
        ]
        # Weather + USA situations (agents/weather.py): appended lazily so
        # there is no import cycle (weather.py imports _Provider from here).
        # A failed import/construct never sinks the briefing.
        try:
            from .weather import USASituationsProvider, WeatherProvider
            # weather slots between calendar and markets; USA after news.
            self.providers.insert(2, WeatherProvider())
            self.providers.append(USASituationsProvider())
        except Exception:  # noqa: BLE001
            pass

    def register(self, provider: _Provider) -> None:
        self.providers.append(provider)

    def compose(self, ctx: Any, date: str,
                since: float | None = None) -> Briefing:
        t0 = time.time()
        since = since if since is not None else _last_briefing_at(ctx) or 0.0
        sections: list[BriefingSection] = []
        for p in self.providers:
            try:
                sec = p.collect(ctx, since)
            except Exception as exc:  # noqa: BLE001 — one bad provider ≠ no briefing
                _log.warning("briefing provider %s failed: %s", p.name, exc)
                continue
            if sec is not None:
                sections.append(sec)
        sections = self._apply_engagement(ctx, sections)
        sections.sort(key=lambda s: s.priority)
        briefing = Briefing(
            id=new_id("briefing"), date=date, sections=sections,
            generated_at=time.time(), generation_ms=0.0)
        note = self._enforce_cap(briefing)
        briefing.truncated_note = note
        briefing.generation_ms = (time.time() - t0) * 1000.0
        return briefing

    # -- engagement: demote sustained-ignored sections, honor pins --------
    def _apply_engagement(self, ctx: Any,
                          sections: list[BriefingSection]
                          ) -> list[BriefingSection]:
        store = _engagement_store(ctx)
        if store is None:
            return sections
        pinned = set(_prefs(ctx).get("pinned_sections", []) or [])
        for s in sections:
            if s.name in pinned:
                continue
            st = store.get(s.name)
            views = st.get("views", 0)
            followups = st.get("followups", 0)
            if views >= 7 and followups / max(views, 1) < 0.2:
                s.priority += 50  # sustained low engagement → demoted
        return sections

    # -- length cap: drop lowest-priority sections first ------------------
    def _enforce_cap(self, briefing: Briefing) -> str:
        if briefing.word_count() <= MAX_WORDS:
            return ""
        dropped = 0
        # never drop alerts (priority 10) — drop from the bottom
        droppable = [s for s in briefing.sections if s.priority > 10]
        droppable.sort(key=lambda s: -s.priority)
        while briefing.word_count() > MAX_WORDS and droppable:
            victim = droppable.pop(0)
            briefing.sections.remove(victim)
            dropped += 1
        if not droppable:  # still over: truncate lines in place
            for s in briefing.sections:
                while briefing.word_count() > MAX_WORDS and len(s.lines) > 1:
                    s.lines.pop()
                    dropped += 1
        return (f"+{dropped} more in the full digest — "
                f"run `nm briefing today` for everything." if dropped else "")

# ── persistence ────────────────────────────────────────────────────────────

class _EngagementStore:
    def __init__(self, db: Any) -> None:
        self.db = db

    def get(self, section: str) -> dict[str, Any]:
        try:
            row = self.db.query_one(
                "SELECT views, followups, pinned FROM briefing_engagement "
                "WHERE section=?", (section,))
        except Exception:  # noqa: BLE001
            row = None
        if row:
            return {"views": row["views"] or 0,
                    "followups": row["followups"] or 0,
                    "pinned": bool(row["pinned"])}
        return {"views": 0, "followups": 0, "pinned": False}

    def record_view(self, section: str) -> None:
        self._bump(section, "views")

    def record_followup(self, section: str) -> None:
        self._bump(section, "followups")

    def set_pinned(self, section: str, pinned: bool) -> None:
        try:
            self.db.execute(
                "INSERT INTO briefing_engagement (section, views, followups, "
                "pinned) VALUES (?,?,?,?) ON CONFLICT(section) DO UPDATE "
                "SET pinned=excluded.pinned",
                (section, 0, 0, 1 if pinned else 0))
        except Exception:  # noqa: BLE001
            pass

    def _bump(self, section: str, col: str) -> None:
        # first touch inserts a 1 in the bumped column (no conflict yet,
        # so the DO UPDATE wouldn't fire); later touches increment.
        init = {"views": 0, "followups": 0}
        init[col] = 1
        try:
            self.db.execute(
                "INSERT INTO briefing_engagement "
                "(section, views, followups, pinned) VALUES (?,?,?,0) "
                "ON CONFLICT(section) DO UPDATE SET "
                f"{col}=briefing_engagement.{col}+1",
                (section, init["views"], init["followups"]))
        except Exception:  # noqa: BLE001
            pass


def _engagement_store(ctx: Any) -> _EngagementStore | None:
    db = getattr(ctx, "db", None)
    return _EngagementStore(db) if db is not None else None


def store_briefing(ctx: Any, briefing: Briefing) -> None:
    db = getattr(ctx, "db", None)
    if db is None:
        return
    db.execute(
        "INSERT OR REPLACE INTO briefings "
        "(id, date, sections_json, generated_at, generation_ms, late) "
        "VALUES (?,?,?,?,?,?)",
        (briefing.id, briefing.date,
         json.dumps([{"name": s.name, "title": s.title,
                      "priority": s.priority, "source": s.source,
                      "items": s.items, "lines": s.lines}
                     for s in briefing.sections], default=str),
         briefing.generated_at, briefing.generation_ms,
         1 if briefing.late else 0))


def latest_briefing(ctx: Any, date: str = "") -> dict[str, Any] | None:
    """Newest stored briefing (optionally for one date)."""
    db = getattr(ctx, "db", None)
    if db is None:
        return None
    try:
        if date:
            row = db.query_one(
                "SELECT * FROM briefings WHERE date=? "
                "ORDER BY generated_at DESC LIMIT 1", (date,))
        else:
            row = db.query_one(
                "SELECT * FROM briefings ORDER BY generated_at DESC LIMIT 1")
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    return {"id": row["id"], "date": row["date"],
            "sections": json.loads(row["sections_json"] or "[]"),
            "generated_at": row["generated_at"],
            "late": bool(row["late"])}


def _last_briefing_at(ctx: Any) -> float:
    b = latest_briefing(ctx)
    return float(b["generated_at"]) if b else 0.0


def followup_item(ctx: Any, n: int) -> dict[str, Any] | None:
    """Resolve 'tell me more about item N' against the stored briefing.

    Items are numbered in render order across sections (1-based).
    Records engagement for the owning section."""
    b = latest_briefing(ctx)
    if not b:
        return None
    flat: list[tuple[str, dict]] = []
    for s in b["sections"]:
        for it in s.get("items", []):
            flat.append((s.get("name", ""), it))
    if n < 1 or n > len(flat):
        return None
    section, item = flat[n - 1]
    store = _engagement_store(ctx)
    if store is not None:
        store.record_followup(section)
    return {"n": n, "section": section, "item": item,
            "briefing_date": b["date"]}


def record_briefing_views(ctx: Any, briefing: Briefing) -> None:
    store = _engagement_store(ctx)
    if store is None:
        return
    for s in briefing.sections:
        store.record_view(s.name)


# ── scheduling ─────────────────────────────────────────────────────────────

def ensure_briefing_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``briefing`` job (idempotent by name).

    Runs ``daily HH:MM`` at ``settings.briefing.time`` (default 07:00) in
    the owner's timezone.  Safe to call on every boot.  If the owner
    changed the time, the job is replaced (scheduler stores the daily
    detail as HH:MM in ``spec``).
    """
    from .scheduler import Scheduler

    sched = Scheduler(context)
    want = briefing_time(context)
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == BRIEFING_JOB_NAME]
    except Exception:  # noqa: BLE001 — scheduler table may not exist yet
        have = []
    if have and have[0].get("spec") == want:
        return {"name": BRIEFING_JOB_NAME, "already_scheduled": True,
                "job_id": have[0].get("id")}
    for j in have:  # stale time → replace
        try:
            sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
    job = sched.add(BRIEFING_JOB_NAME, f"daily {want}", "tool",
                    {"tool": "briefing", "args": {"action": "run"}})
    _log.info("scheduled briefing job: %s (daily %s)", BRIEFING_JOB_NAME,
              want)
    return {"name": BRIEFING_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


def _today_str(context: Any) -> str:
    from zoneinfo import ZoneInfo
    from datetime import datetime
    try:
        tz = ZoneInfo(_owner_tz(context))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    return datetime.now(tz).strftime("%Y-%m-%d")


def check_catchup(context: Any) -> dict[str, Any]:
    """Boot-time catch-up: if Devon was down at briefing time, deliver
    exactly ONE late briefing on next startup — never a backlog."""
    from zoneinfo import ZoneInfo
    from datetime import datetime

    today = _today_str(context)
    if latest_briefing(context, today):
        return {"catchup": False, "reason": "already delivered"}
    # only catch up if briefing time has already passed today
    try:
        tz = ZoneInfo(_owner_tz(context))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    hh, mm = briefing_time(context).split(":")[:2]
    if (now.hour, now.minute) < (int(hh), int(mm)):
        return {"catchup": False, "reason": "briefing time not reached"}
    result = run_briefing(context, late=True)
    return {"catchup": True, **result}


# ── run ────────────────────────────────────────────────────────────────────

def run_briefing(context: Any, late: bool = False) -> dict[str, Any]:
    """Compose, store, and deliver the briefing.  Never raises: total
    provider failure → short fallback message, never silence."""
    date = _today_str(context)
    composer = BriefingComposer()
    try:
        briefing = composer.compose(context, date)
    except Exception as exc:  # noqa: BLE001 — total failure → fallback
        return _deliver_fallback(context, date, str(exc), late=late)
    briefing.late = late
    try:
        store_briefing(context, briefing)
        record_briefing_views(context, briefing)
    except Exception as exc:  # noqa: BLE001 — storage must not kill delivery
        _log.warning("briefing store failed: %s", exc)
    text = briefing.render_text()
    if not briefing.sections:
        text += ("\n\n(quiet night — nothing to report. "
                 "Your watchers, rooms and feeds are all nominal.)")
    return _deliver(context, briefing, text)


def _deliver(context: Any, briefing: Briefing,
             text: str) -> dict[str, Any]:
    from .notifier import Notifier
    title = (f"☀️ morning briefing — {briefing.date}"
             + (" (late)" if briefing.late else ""))
    try:
        # Notifier resolves the live gateway from the context itself
        # (context.extras["gateway"] in the runtime) — the briefing is
        # pushed to the owner's DMs, never just stored.
        notifier = Notifier(context)
        res = notifier.publish("briefing", title, text,
                               force=briefing.late)
        delivered = bool(res.get("delivered"))
        delivery_state = res.get("delivery_state") or (
            "sent" if delivered else "pending")
    except Exception as exc:  # noqa: BLE001 — notifier must not raise
        _log.warning("briefing delivery failed: %s", exc)
        delivered = False
        delivery_state = "failed"
    return {"ok": True, "briefing_id": briefing.id, "date": briefing.date,
            "sections": len(briefing.sections),
            "word_count": briefing.word_count(),
            "delivered": delivered, "delivery_state": delivery_state,
            "late": briefing.late, "text": text}


def _deliver_fallback(context: Any, date: str, reason: str,
                      late: bool = False) -> dict[str, Any]:
    from .notifier import Notifier
    title = f"☀️ morning briefing — {date}" + (" (late)" if late else "")
    body = (f"Briefing failed: {reason[:160]}. "
            f"Run `nm briefing retry`.")
    try:
        notifier = Notifier(context)
        notifier.publish("briefing", title, body, force=True)
    except Exception:  # noqa: BLE001
        pass
    _log.error("briefing total failure: %s", reason[:300])
    return {"ok": False, "date": date, "fallback": True, "reason": reason[:160],
            "delivered": True, "late": late}

def proactive_status(context: Any, limit: int = 10) -> dict[str, Any]:
    """Proactive delivery status: current switches + recent proactive
    sends (briefing + watcher alerts) with their delivery states.

    Powers ``nm briefing status`` and the chat ``/notify`` view.
    Delivery states: sent / failed / pending (no live channel yet) /
    held-quiet-hours / disabled (proactive switch off) / muted /
    deduped.
    """
    from .notifier import Notifier
    partner = getattr(getattr(context, "settings", None), "partner", None)

    def _b(name: str, default: bool = True) -> bool:
        return bool(getattr(partner, name, default)) if partner else default

    settings = {
        "proactive_enabled": _b("proactive_enabled"),
        "proactive_briefing": _b("proactive_briefing"),
        "proactive_watchers": _b("proactive_watchers"),
        "quiet_hours": (f"{getattr(partner, 'quiet_start', 22):02d}:00-"
                        f"{getattr(partner, 'quiet_end', 8):02d}:00"
                        if partner else "22:00-08:00"),
        "timezone": _owner_tz(context),
        "briefing_time": briefing_time(context),
    }
    try:
        recent = Notifier(context).delivery_summary(
            limit, kinds=("briefing", "watcher"))
    except Exception:  # noqa: BLE001 — status must not raise
        recent = []
    return {"settings": settings, "recent": recent}


# ── tool registration ──────────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the ``briefing`` tool (agent-callable morning briefing)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "briefing",
        description=(
            "Morning briefing: the overnight digest. action=run (compose, "
            "store, deliver now) | today (print the last stored briefing, "
            "no regeneration) | retry (regenerate + deliver) | config "
            "(show time/timezone/topics/sections) | status (proactive "
            "delivery switches + recent sends with delivery states) | "
            "followup <n> (expand item n from the stored briefing)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — run|today|retry|config|status|followup",
            "n": "int (optional) — item number for followup",
        },
    )
    def briefing(*, action: str = "", n: str = "") -> dict[str, Any]:
        act = (action or "").strip().lower()
        if act == "run":
            return run_briefing(context)
        if act == "retry":
            return run_briefing(context)
        if act == "status":
            return proactive_status(context)
        if act == "today":
            b = latest_briefing(context)
            if not b:
                return {"briefing": None,
                        "hint": "no briefing stored yet — run `nm briefing now`"}
            return {"briefing": b}
        if act == "config":
            prefs = _prefs(context)
            return {"time": briefing_time(context),
                    "timezone": _owner_tz(context),
                    "topics": prefs.get("topics", []),
                    "symbols": prefs.get("symbols", []) or [],
                    "pinned_sections": prefs.get("pinned_sections", []) or [],
                    "max_words": MAX_WORDS}
        if act == "followup":
            try:
                num = int(str(n).strip())
            except (TypeError, ValueError):
                return {"error": "followup needs an item number"}
            item = followup_item(context, num)
            if item is None:
                return {"error": f"no item {num} in the latest briefing"}
            return item
        return {"error": f"unsupported briefing action {act!r}"}
