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
]

#: Scheduler action name the host dispatches to :func:`check_all`.
PRICE_WATCH_ACTION = "price_watch_check"

#: Cron: daily price check at 07:00.
WATCH_CRON = "0 7 * * *"

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


def format_alert(alert: PriceAlert) -> str:
    prev = _short(alert.previous_kobo)
    curr = _short(alert.current_kobo)
    lines = [
        f"✈️ {alert.route} dropped: {prev} → {curr} "
        f"({alert.savings_pct:.0f}% off).",
    ]
    if alert.scarcity:
        lines.append(alert.scarcity)
    lines.append("Want it? Book before the fare moves again.")
    return "\n".join(lines)


_IATA_RE = re.compile(r"\b([A-Z]{3})\b")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_TARGET_RE = re.compile(r"(?:under|below|max)\s*[₦N]?\s*([\d,.]+)\s*k?\b", re.I)


def parse_watch_request(text: str) -> dict[str, Any] | None:
    """Parse 'track LOS LHR 2026-12-01 under 400k' → dict.

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
    return {
        "origin": codes[0],
        "destination": codes[1],
        "departure_date": date_m.group(1) if date_m else "",
        "target_kobo": target_kobo,
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
    ) -> PriceWatch:
        origin = (origin or "").upper().strip()
        destination = (destination or "").upper().strip()
        if not (re.fullmatch(r"[A-Z]{3}", origin)
                and re.fullmatch(r"[A-Z]{3}", destination)):
            raise ValueError("origin/destination must be 3-letter IATA codes")
        if not _DATE_RE.fullmatch(departure_date or ""):
            raise ValueError("departure_date must be YYYY-MM-DD")
        w = PriceWatch(
            id="watch_" + uuid.uuid4().hex[:8],
            origin=origin, destination=destination,
            departure_date=departure_date,
            target_kobo=max(0, int(target_kobo or 0)),
            threshold_pct=float(threshold_pct or 10.0),
            created_at=self._now(),
        )
        self._db.execute(
            """INSERT INTO watches (id, origin, destination, departure_date,
               target_kobo, threshold_pct, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (w.id, w.origin, w.destination, w.departure_date,
             w.target_kobo, w.threshold_pct, w.created_at),
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
        return PriceWatch(
            id=r["id"], origin=r["origin"], destination=r["destination"],
            departure_date=r["departure_date"], target_kobo=r["target_kobo"],
            threshold_pct=r["threshold_pct"], last_kobo=r["last_kobo"],
            last_currency=r["last_currency"] or "",
            alerted_kobo=r["alerted_kobo"], active=bool(r["active"]),
            created_at=r["created_at"], last_checked=r["last_checked"],
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

    def check(self, watch: PriceWatch) -> PriceAlert | None:
        """Run one price check. Returns an alert on a qualifying drop."""
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
        self._db.execute(
            """UPDATE watches SET last_kobo = ?, last_currency = ?
               WHERE id = ?""",
            (current_kobo, currency, watch.id))
        self._db.commit()

        alert = None
        if watch.last_kobo and current_kobo < watch.last_kobo:
            drop_pct = (watch.last_kobo - current_kobo) / watch.last_kobo * 100
            new_low = (not watch.alerted_kobo
                       or current_kobo < watch.alerted_kobo)
            threshold_ok = drop_pct >= watch.threshold_pct
            target_crossed = (watch.target_kobo
                              and current_kobo <= watch.target_kobo
                              and watch.last_kobo > watch.target_kobo)
            target_ok = (not watch.target_kobo
                         or current_kobo <= watch.target_kobo)
            if (threshold_ok or target_crossed) and new_low and target_ok:
                alert = PriceAlert(
                    watch_id=watch.id, route=watch.route,
                    previous_kobo=watch.last_kobo,
                    current_kobo=current_kobo,
                    savings_pct=drop_pct,
                    scarcity=self._scarcity(offer),
                )
                self._db.execute(
                    "UPDATE watches SET alerted_kobo = ? WHERE id = ?",
                    (current_kobo, watch.id))
                self._db.commit()
        if alert is not None and self._sender is not None:
            try:
                self._sender(alert.text, buttons=alert.buttons)
            except Exception:  # noqa: BLE001
                _log.debug("watcher alert send failed", exc_info=True)
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
