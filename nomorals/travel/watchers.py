"""Price watchers — "track this for me" as persistent monitoring.

"Track Lagos→London flights for me" → a persistent watcher → a Duffel
price check on the scheduler → an owner-DM alert when the fare drops.
The alert payload format follows the telegram-flight-ai-assistant
pattern: route, previous→current, savings %, scarcity signal, and
[Book] [Dismiss] action buttons.

Rules:
- Scheduler-backed and persistent (SQLite). Survives restarts.
- Never spams: one alert per drop level, not one per check.
- Owner-only. Prices are advisory — booking still goes through
  #70's confirmation gate + #69's mandate.

Scheduler seam: ``ensure_schedule(scheduler)`` registers a daily cron
with action ``PRICE_WATCH_ACTION``; the host calls ``check_all()``
when that action fires.
"""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PriceWatcher",
    "PriceWatch",
    "PriceAlert",
    "PricePoint",
    "PRICE_WATCH_ACTION",
    "ensure_schedule",
    "check_all",
    "parse_watch_request",
    "format_alert",
    "format_watches",
    "trend_stats",
    "forecast",
    "verdict_banner",
    "VERDICT_BUY",
    "VERDICT_WAIT",
    "VERDICT_WATCH",
    "NOTIFY_DROP",
    "NOTIFY_RISE",
    "NOTIFY_ANY",
]

#: Scheduler action name the host dispatches to :func:`check_all`.
PRICE_WATCH_ACTION = "price_watch_check"

#: Cron: daily price check at 07:00.
WATCH_CRON = "0 7 * * *"

#: Verdicts from :func:`forecast` (Hopper-style buy-now/wait guidance).
VERDICT_BUY = "buy"
VERDICT_WAIT = "wait"
VERDICT_WATCH = "watch"

#: Alert trigger modes.
NOTIFY_DROP = "drop"
NOTIFY_RISE = "rise"
NOTIFY_ANY = "any"

_DEFAULT_DB = "~/.nomorals/travel/watches.db"


def _naira(kobo: int) -> str:
    return f"₦{kobo / 100:,.0f}"


def _short(amount_kobo: int) -> str:
    """₦450,000 → '₦450k'."""
    n = amount_kobo / 100
    if n >= 1000:
        return f"₦{n / 1000:,.1f}k".replace(".0k", "k")
    return f"₦{n:,.0f}"


def _to_minor(amount: str, currency: str) -> int:
    """Decimal amount string → minor units (kobo/cents)."""
    try:
        return int(round(float(amount) * 100))
    except (TypeError, ValueError):
        return 0


@dataclass
class PriceWatch:
    id: str
    origin: str
    destination: str
    departure_date: str  # YYYY-MM-DD
    target_kobo: int = 0  # 0 = no target, alert on any drop ≥ threshold
    threshold_pct: float = 10.0
    last_kobo: int = 0
    last_currency: str = ""
    alerted_kobo: int = 0  # price level already alerted at
    active: bool = True
    created_at: float = 0.0
    last_checked: float = 0.0
    notify_on: str = NOTIFY_DROP  # drop | rise | any
    cooldown_h: float = 24.0  # min hours between repeat alerts

    @property
    def route(self) -> str:
        return f"{self.origin}→{self.destination}"


@dataclass
class PriceAlert:
    watch_id: str
    route: str
    previous_kobo: int
    current_kobo: int
    savings_pct: float
    scarcity: str = ""
    text: str = ""
    buttons: list = field(default_factory=list)
    kind: str = NOTIFY_DROP  # drop | rise
    trend_line: str = ""
    verdict: str = ""  # "🟢 BUY NOW …" banner, filled from forecast()

    def __post_init__(self) -> None:
        if not self.text:
            self.text = format_alert(self)
        if not self.buttons:
            self.buttons = [
                [("✈️ Book", f"book_flight:{self.watch_id}"),
                 ("Dismiss", f"dismiss_watch:{self.watch_id}")],
            ]


@dataclass
class PricePoint:
    date: str
    amount_kobo: int
    currency: str = ""
    cheapest: bool = False


def format_alert(alert: PriceAlert, *,
                 budget_kobo: int | None = None,
                 programs: list | None = None) -> str:
    # #75: budget line + points-vs-cash when provided.
    prev = _short(alert.previous_kobo)
    curr = _short(alert.current_kobo)
    if alert.kind == NOTIFY_RISE:
        lines = [
            f"📈 {alert.route} climbing: {prev} → {curr} "
            f"(+{alert.savings_pct:.0f}%).",
        ]
        close = "Fares are moving up — book soon if this trip matters."
    else:
        lines = [
            f"✈️ {alert.route} dropped: {prev} → {curr} "
            f"({alert.savings_pct:.0f}% off).",
        ]
        close = "Want it? Book before the fare moves again."
    if alert.trend_line:
        lines.append(alert.trend_line)
    if alert.verdict:
        lines.append(alert.verdict)
    if alert.scarcity:
        lines.append(alert.scarcity)
    if budget_kobo:
        if alert.current_kobo <= budget_kobo:
            lines.append(f"within your {_short(budget_kobo)} budget ✅")
        else:
            lines.append(
                f"over budget by "
                f"₦{(alert.current_kobo - budget_kobo) / 100:,.0f} ⚠️")
    if programs:
        from .display import points_vs_cash
        pvc = points_vs_cash(alert.current_kobo, programs)
        if pvc:
            lines.append(f"🎖️ {pvc}")
    lines.append(close)
    return "\n".join(lines)


# ── price history + Hopper-style forecast ─────────────────────────────────

def trend_stats(points: list[PricePoint]) -> dict[str, Any]:
    """Min/avg/max + 7-day moving average from price history. Pure."""
    vals = [p.amount_kobo for p in points if p.amount_kobo > 0]
    if not vals:
        return {}
    stats: dict[str, Any] = {
        "n": len(vals),
        "low": min(vals),
        "high": max(vals),
        "avg": sum(vals) / len(vals),
        "latest": vals[-1],
    }
    # 7-day moving average over the trailing points (each check ≈ daily)
    tail = vals[-7:]
    stats["avg7"] = sum(tail) / len(tail)
    if len(vals) >= 2:
        prev = vals[-2]
        stats["day_change_pct"] = (
            (vals[-1] - prev) / prev * 100 if prev else 0.0)
    return stats


def forecast(points: list[PricePoint]) -> dict[str, Any]:
    """Hopper-style buy-now/wait guidance from price history. Pure.

    Heuristic, honest about limits: airline revenue management is dynamic,
    so this is a signal, not a guarantee. Needs ≥3 points for a verdict
    beyond "watch".
    """
    stats = trend_stats(points)
    if not stats or stats["n"] < 3:
        return {"verdict": VERDICT_WATCH, "confidence": 0.0,
                "reasons": ["not enough price history yet — check back "
                            "after a few days of tracking"]}
    latest = stats["latest"]
    reasons: list[str] = []
    score = 0.0  # positive → buy, negative → wait
    # 30-day (available) low
    low = stats["low"]
    if latest <= low * 1.02:
        score += 2.0
        reasons.append(f"at/near the {_short(low)} low we've seen")
    elif latest <= stats["avg"] * 0.95:
        score += 1.0
        reasons.append(f"{(1 - latest / stats['avg']) * 100:.0f}% below "
                       f"the average ({_short(int(stats['avg']))})")
    # vs 7-day moving average
    avg7 = stats["avg7"]
    if latest >= avg7 * 1.05:
        score -= 1.5
        reasons.append(f"{(latest / avg7 - 1) * 100:.0f}% above the 7-day "
                       f"average — trending up")
    elif latest <= avg7 * 0.97:
        score += 1.0
        reasons.append("below the 7-day average — trending down")
    # day-over-day momentum
    day = stats.get("day_change_pct", 0.0)
    if day >= 5:
        score -= 0.5
        reasons.append(f"up {day:.0f}% since yesterday")
    elif day <= -5:
        score += 0.5
        reasons.append(f"down {abs(day):.0f}% since yesterday")
    if score >= 1.5:
        verdict = VERDICT_BUY
    elif score <= -1.0:
        verdict = VERDICT_WAIT
    else:
        verdict = VERDICT_WATCH
    confidence = min(0.95, 0.35 + 0.15 * abs(score))
    return {"verdict": verdict, "confidence": round(confidence, 2),
            "reasons": reasons or ["prices are flat — keep watching"],
            "stats": stats}


def verdict_banner(fc: dict[str, Any]) -> str:
    """'🟢 BUY NOW — at the 30-day low (confidence 80%)'"""
    v = fc.get("verdict", VERDICT_WATCH)
    conf = int(fc.get("confidence", 0) * 100)
    reasons = fc.get("reasons") or []
    if v == VERDICT_BUY:
        head = "🟢 BUY NOW"
    elif v == VERDICT_WAIT:
        head = "🟡 WAIT"
    else:
        head = "⚪ KEEP WATCHING"
    tail = f" ({reasons[0]})" if reasons else ""
    return f"{head}{tail} — confidence {conf}%"


_IATA_RE = re.compile(r"\b([A-Z]{3})\b")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_TARGET_RE = re.compile(r"(?:under|below|max)\s*[₦N]?\s*([\d,.]+)\s*k?\b", re.I)


def parse_watch_request(text: str) -> dict[str, Any] | None:
    """Parse 'track LOS LHR 2026-12-01 under 400k' → dict.

    Optional trigger mode: 'rise' watches for fare *increases*
    ("track LOS LHR 2026-12-01 rise 15%"), 'any' alerts on either move.
    Returns None when no route (two IATA codes) is found.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    codes = _IATA_RE.findall(raw.upper())
    if len(codes) < 2:
        return None
    date_m = _DATE_RE.search(raw)
    target_kobo = 0
    tm = _TARGET_RE.search(raw)
    if tm:
        has_k = "k" in tm.group(0).lower()
        try:
            target_kobo = int(
                float(tm.group(1).replace(",", ""))
                * (1000 if has_k else 1) * 100)
        except ValueError:
            target_kobo = 0
    low = raw.lower()
    notify_on = NOTIFY_DROP
    if re.search(r"\brise\b|\bincreas", low):
        notify_on = NOTIFY_RISE
    elif re.search(r"\bany\s+(move|change|direction)\b", low):
        notify_on = NOTIFY_ANY
    return {
        "origin": codes[0],
        "destination": codes[1],
        "departure_date": date_m.group(1) if date_m else "",
        "target_kobo": target_kobo,
        "notify_on": notify_on,
    }


class PriceWatcher:
    """Persistent flight-price watchers backed by Duffel + the scheduler."""

    def __init__(
        self,
        db_path: str = "",
        *,
        duffel: Any = None,
        sender: Callable[..., bool] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        import os
        path = db_path or os.path.expanduser(_DEFAULT_DB)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._duffel = duffel
        self._sender = sender
        self._now = now or time.time
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS watches (
                   id TEXT PRIMARY KEY, origin TEXT, destination TEXT,
                   departure_date TEXT, target_kobo INTEGER DEFAULT 0,
                   threshold_pct REAL DEFAULT 10.0,
                   last_kobo INTEGER DEFAULT 0, last_currency TEXT DEFAULT '',
                   alerted_kobo INTEGER DEFAULT 0,
                   active INTEGER DEFAULT 1,
                   created_at REAL, last_checked REAL DEFAULT 0)"""
        )
        # additive migration for older DBs
        for _ddl in (
            "ALTER TABLE watches ADD COLUMN notify_on TEXT DEFAULT 'drop'",
            "ALTER TABLE watches ADD COLUMN cooldown_h REAL DEFAULT 24.0",
        ):
            try:
                self._db.execute(_ddl)
            except Exception:  # noqa: BLE001 — column already there
                pass
        # price history: every check writes a point → trends + forecast
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS price_history (
                   watch_id TEXT NOT NULL, checked_at REAL NOT NULL,
                   kobo INTEGER NOT NULL, currency TEXT DEFAULT '')"""
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ph_watch "
            "ON price_history (watch_id, checked_at)"
        )
        # alert log: cooldown enforcement per trigger
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS alert_log (
                   watch_id TEXT NOT NULL, at REAL NOT NULL,
                   kind TEXT NOT NULL, kobo INTEGER NOT NULL)"""
        )
        self._db.commit()

    # ── CRUD ──────────────────────────────────────────────────────────

    def watch(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        *,
        target_kobo: int = 0,
        threshold_pct: float = 10.0,
        notify_on: str = NOTIFY_DROP,
        cooldown_hours: float = 24.0,
    ) -> PriceWatch:
        origin = (origin or "").upper().strip()
        destination = (destination or "").upper().strip()
        if not (re.fullmatch(r"[A-Z]{3}", origin)
                and re.fullmatch(r"[A-Z]{3}", destination)):
            raise ValueError("origin/destination must be 3-letter IATA codes")
        if not _DATE_RE.fullmatch(departure_date or ""):
            raise ValueError("departure_date must be YYYY-MM-DD")
        notify_on = (notify_on or NOTIFY_DROP).lower()
        if notify_on not in (NOTIFY_DROP, NOTIFY_RISE, NOTIFY_ANY):
            raise ValueError("notify_on must be drop | rise | any")
        w = PriceWatch(
            id="watch_" + uuid.uuid4().hex[:8],
            origin=origin, destination=destination,
            departure_date=departure_date,
            target_kobo=max(0, int(target_kobo or 0)),
            threshold_pct=float(threshold_pct or 10.0),
            notify_on=notify_on,
            cooldown_h=float(cooldown_hours or 24.0),
            created_at=self._now(),
        )
        self._db.execute(
            """INSERT INTO watches (id, origin, destination, departure_date,
               target_kobo, threshold_pct, notify_on, cooldown_h, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (w.id, w.origin, w.destination, w.departure_date,
             w.target_kobo, w.threshold_pct, w.notify_on, w.cooldown_h,
             w.created_at),
        )
        self._db.commit()
        return w

    def unwatch(self, watch_id: str) -> bool:
        cur = self._db.execute(
            "UPDATE watches SET active = 0 WHERE id = ?", (watch_id,))
        self._db.commit()
        return cur.rowcount > 0

    def list_watches(self, *, active_only: bool = True) -> list[PriceWatch]:
        q = "SELECT * FROM watches"
        if active_only:
            q += " WHERE active = 1"
        q += " ORDER BY created_at"
        return [self._row_to_watch(r) for r in self._db.execute(q).fetchall()]

    def get(self, watch_id: str) -> PriceWatch | None:
        r = self._db.execute(
            "SELECT * FROM watches WHERE id = ?", (watch_id,)).fetchone()
        return self._row_to_watch(r) if r else None

    @staticmethod
    def _row_to_watch(r: sqlite3.Row) -> PriceWatch:
        cols = set(r.keys())
        return PriceWatch(
            id=r["id"], origin=r["origin"], destination=r["destination"],
            departure_date=r["departure_date"], target_kobo=r["target_kobo"],
            threshold_pct=r["threshold_pct"], last_kobo=r["last_kobo"],
            last_currency=r["last_currency"] or "",
            alerted_kobo=r["alerted_kobo"], active=bool(r["active"]),
            created_at=r["created_at"], last_checked=r["last_checked"],
            notify_on=r["notify_on"] if "notify_on" in cols else NOTIFY_DROP,
            cooldown_h=(r["cooldown_h"] if "cooldown_h" in cols
                        else 24.0),
        )

    # ── checking ──────────────────────────────────────────────────────

    def _duffel_client(self) -> Any:
        if self._duffel is not None:
            return self._duffel
        try:
            from ..connectors.duffel import DuffelConnector
            from ..connectors.registry import get_connector
            return get_connector("duffel")
        except Exception:  # noqa: BLE001
            return None

    def _cheapest(self, origin: str, destination: str,
                  date: str) -> tuple[int, str, Any] | None:
        """Cheapest current fare → (minor, currency, offer) or None."""
        client = self._duffel_client()
        if client is None:
            return None
        try:
            offers = client.search_offers(origin, destination, date)
        except Exception:  # noqa: BLE001
            _log.debug("watcher search failed", exc_info=True)
            return None
        if not offers:
            return None
        best = min(
            (o for o in offers
             if _to_minor(o.total_amount, o.total_currency) > 0),
            key=lambda o: _to_minor(o.total_amount, o.total_currency),
            default=None,
        )
        if best is None:
            return None
        return (_to_minor(best.total_amount, best.total_currency),
                best.total_currency, best)

    @staticmethod
    def _scarcity(offer: Any) -> str:
        try:
            raw = offer.raw or {}
            svcs = raw.get("available_services") or []
            if isinstance(svcs, list) and 0 < len(svcs) <= 5:
                return (f"Only {len(svcs)} seat{'s' if len(svcs) != 1 else ''} "
                        f"left at this fare.")
        except Exception:  # noqa: BLE001
            pass
        return ""

    # ── price history (trends + forecast) ─────────────────────────────

    def _record_point(self, watch_id: str, kobo: int, currency: str,
                      now: float) -> None:
        self._db.execute(
            "INSERT INTO price_history (watch_id, checked_at, kobo, currency)"
            " VALUES (?, ?, ?, ?)",
            (watch_id, now, int(kobo), currency or ""))
        # prune: keep 120 days so the table stays small
        self._db.execute(
            "DELETE FROM price_history WHERE watch_id = ?"
            " AND checked_at < ?", (watch_id, now - 120 * 86400))

    def history(self, watch_id: str, *, days: int = 60) -> list[PricePoint]:
        """Price points for a watch, oldest first. Never raises."""
        try:
            cutoff = self._now() - days * 86400
            rows = self._db.execute(
                "SELECT checked_at, kobo, currency FROM price_history"
                " WHERE watch_id = ? AND checked_at >= ?"
                " ORDER BY checked_at",
                (watch_id, cutoff)).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [PricePoint(
            date=time.strftime("%Y-%m-%d", time.gmtime(r["checked_at"])),
            amount_kobo=r["kobo"], currency=r["currency"] or "")
            for r in rows]

    def watch_forecast(self, watch_id: str) -> dict[str, Any]:
        """Buy-now/wait verdict for a watch from its price history."""
        return forecast(self.history(watch_id))

    def _recent_alert(self, watch_id: str, kind: str, now: float,
                      cooldown_h: float) -> bool:
        if cooldown_h <= 0:
            return False
        try:
            row = self._db.execute(
                "SELECT MAX(at) AS at FROM alert_log"
                " WHERE watch_id = ? AND kind = ?",
                (watch_id, kind)).fetchone()
        except Exception:  # noqa: BLE001
            return False
        return bool(row and row["at"]
                    and now - row["at"] < cooldown_h * 3600)

    def _log_alert(self, watch_id: str, kind: str, kobo: int,
                   now: float) -> None:
        try:
            self._db.execute(
                "INSERT INTO alert_log (watch_id, at, kind, kobo)"
                " VALUES (?, ?, ?, ?)", (watch_id, now, kind, int(kobo)))
        except Exception:  # noqa: BLE001
            pass

    def check(self, watch: PriceWatch) -> PriceAlert | None:
        """Run one price check. Returns an alert on a qualifying move.

        Every successful check writes a price-history point (drives the
        buy-now/wait forecast). Triggers: drop (default), rise, or any —
        plus cooldowns so a bouncing fare doesn't spam.
        """
        now = self._now()
        # refresh: the caller may hold a stale object across checks
        fresh = self.get(watch.id)
        if fresh is not None:
            watch = fresh
        got = self._cheapest(watch.origin, watch.destination,
                             watch.departure_date)
        self._db.execute(
            "UPDATE watches SET last_checked = ? WHERE id = ?",
            (now, watch.id))
        if got is None:
            self._db.commit()
            return None
        current_kobo, currency, offer = got
        prev_kobo = watch.last_kobo or current_kobo
        self._record_point(watch.id, current_kobo, currency, now)
        self._db.execute(
            """UPDATE watches SET last_kobo = ?, last_currency = ?
               WHERE id = ?""",
            (current_kobo, currency, watch.id))
        self._db.commit()

        # trend line + forecast banner for whichever alert fires
        points = self.history(watch.id)
        fc = forecast(points)
        trend_line = ""
        stats = fc.get("stats") or {}
        if stats:
            trend_line = (
                f"📉 low {_short(stats['low'])} · 7d avg "
                f"{_short(int(stats['avg7']))} · now "
                f"{_short(current_kobo)}")
        verdict = verdict_banner(fc) if stats.get("n", 0) >= 3 else ""

        alert = None
        if watch.last_kobo:
            if current_kobo < watch.last_kobo and watch.notify_on in (
                    NOTIFY_DROP, NOTIFY_ANY):
                alert = self._drop_alert(watch, current_kobo, now, offer,
                                         trend_line, verdict)
            elif current_kobo > watch.last_kobo and watch.notify_on in (
                    NOTIFY_RISE, NOTIFY_ANY):
                alert = self._rise_alert(watch, current_kobo, now, offer,
                                         trend_line, verdict)
        if alert is not None and self._sender is not None:
            try:
                self._sender(alert.text, buttons=alert.buttons)
            except Exception:  # noqa: BLE001
                _log.debug("watcher alert send failed", exc_info=True)
        return alert

    def _drop_alert(self, watch: PriceWatch, current_kobo: int, now: float,
                    offer: Any, trend_line: str,
                    verdict: str) -> PriceAlert | None:
        drop_pct = (watch.last_kobo - current_kobo) / watch.last_kobo * 100
        new_low = (not watch.alerted_kobo
                   or current_kobo < watch.alerted_kobo)
        threshold_ok = drop_pct >= watch.threshold_pct
        target_crossed = (watch.target_kobo
                          and current_kobo <= watch.target_kobo
                          and watch.last_kobo > watch.target_kobo)
        target_ok = (not watch.target_kobo
                     or current_kobo <= watch.target_kobo)
        if not ((threshold_ok or target_crossed) and new_low and target_ok):
            return None
        # new lows are the alert currency in drop mode (one alert per drop
        # level); cooldowns apply to rise alerts, not fresh lows.
        alert = PriceAlert(
            watch_id=watch.id, route=watch.route,
            previous_kobo=watch.last_kobo,
            current_kobo=current_kobo,
            savings_pct=drop_pct,
            scarcity=self._scarcity(offer),
            kind=NOTIFY_DROP,
            trend_line=trend_line,
            verdict=verdict,
        )
        self._db.execute(
            "UPDATE watches SET alerted_kobo = ? WHERE id = ?",
            (current_kobo, watch.id))
        self._log_alert(watch.id, NOTIFY_DROP, current_kobo, now)
        self._db.commit()
        return alert

    def _rise_alert(self, watch: PriceWatch, current_kobo: int, now: float,
                    offer: Any, trend_line: str,
                    verdict: str) -> PriceAlert | None:
        rise_pct = (current_kobo - watch.last_kobo) / watch.last_kobo * 100
        new_high = (not watch.alerted_kobo
                    or current_kobo > watch.alerted_kobo)
        if not (rise_pct >= watch.threshold_pct and new_high):
            return None
        if self._recent_alert(watch.id, NOTIFY_RISE, now, watch.cooldown_h):
            return None
        alert = PriceAlert(
            watch_id=watch.id, route=watch.route,
            previous_kobo=watch.last_kobo,
            current_kobo=current_kobo,
            savings_pct=rise_pct,
            scarcity=self._scarcity(offer),
            kind=NOTIFY_RISE,
            trend_line=trend_line,
            verdict=verdict,
        )
        self._db.execute(
            "UPDATE watches SET alerted_kobo = ? WHERE id = ?",
            (current_kobo, watch.id))
        self._log_alert(watch.id, NOTIFY_RISE, current_kobo, now)
        self._db.commit()
        return alert

    def check_all(self) -> list[PriceAlert]:
        alerts: list[PriceAlert] = []
        for w in self.list_watches():
            try:
                a = self.check(w)
            except Exception:  # noqa: BLE001
                _log.debug("watcher check failed", exc_info=True)
                continue
            if a is not None:
                alerts.append(a)
        return alerts

    # ── date grid ─────────────────────────────────────────────────────

    def price_grid(
        self,
        origin: str,
        destination: str,
        dates: list[str],
    ) -> list[PricePoint]:
        """Cheapest fare per date; the minimum is flagged cheapest."""
        points: list[PricePoint] = []
        for d in dates:
            got = self._cheapest(origin.upper(), destination.upper(), d)
            if got is None:
                continue
            minor, currency, _offer = got
            points.append(PricePoint(date=d, amount_kobo=minor,
                                     currency=currency))
        if points:
            best = min(points, key=lambda p: p.amount_kobo)
            best.cheapest = True
        return points

    def format_grid(self, origin: str, destination: str,
                    points: list[PricePoint]) -> str:
        if not points:
            return f"no fares found for {origin}→{destination}."
        lines = [f"📅 {origin}→{destination} fares:"]
        best = next((p for p in points if p.cheapest), None)
        for p in points:
            mark = " ⭐ cheapest" if p.cheapest else ""
            lines.append(f"• {p.date}: {_short(p.amount_kobo)}{mark}")
        if best:
            lines.append(f"Leave {best.date}, save "
                         f"{_naira(max(p.amount_kobo for p in points) - best.amount_kobo)} "
                         f"vs the priciest day.")
        return "\n".join(lines)

    # ── watch list (/showwatches) ─────────────────────────────────────

    def format_watches(self, watches: list[PriceWatch] | None = None
                       ) -> str:
        """One line per watch: route, date, last fare, trend, verdict."""
        watches = self.list_watches() if watches is None else watches
        if not watches:
            return ("No active price watches. "
                    "Try `track LOS LHR 2026-12-01 under 400k`.")
        lines = ["👀 Active price watches:"]
        for w in watches:
            bit = f"• {w.route} {w.departure_date or 'flexible'}"
            if w.last_kobo:
                bit += f" — {_short(w.last_kobo)}"
                fc = forecast(self.history(w.id))
                v = fc.get("verdict")
                if v == VERDICT_BUY:
                    bit += " 🟢 buy now"
                elif v == VERDICT_WAIT:
                    bit += " 🟡 wait"
                elif v == VERDICT_WATCH:
                    bit += " ⚪ watching"
            else:
                bit += " — not checked yet"
            if w.target_kobo:
                bit += f" (target ≤ {_short(w.target_kobo)})"
            if w.notify_on != NOTIFY_DROP:
                bit += f" [{w.notify_on}]"
            lines.append(bit)
        return "\n".join(lines)


def check_all(watcher: PriceWatcher | None = None, **kwargs: Any) -> list[PriceAlert]:
    """Host entry point: run every active watcher once."""
    w = watcher or PriceWatcher()
    return w.check_all()


def ensure_schedule(scheduler: Any) -> bool:
    """Register the daily price-watch cron. Idempotent-ish.

    Returns True when a job was (or is already) registered.
    """
    try:
        import asyncio

        async def _ensure() -> bool:
            jobs = []
            try:
                jobs = scheduler.list_jobs()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            for j in jobs or []:
                params = getattr(j, "parameters", {}) or {}
                if getattr(j, "action", "") == PRICE_WATCH_ACTION:
                    return True
            await scheduler.schedule_cron(
                task_id="price-watch-daily",
                cron_expr=WATCH_CRON,
                action=PRICE_WATCH_ACTION,
                parameters={},
            )
            return True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # called from async context — schedule as a task and assume ok
            loop.create_task(_ensure())
            return True
        return asyncio.run(_ensure())
    except Exception:  # noqa: BLE001
        _log.debug("ensure_schedule failed", exc_info=True)
        return False
