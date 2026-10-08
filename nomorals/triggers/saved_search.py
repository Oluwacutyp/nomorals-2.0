"""Persistent match-alerts: saved searches with instant push.

"Tell me the moment a 2-bed under ₦1.5M/year hits Yaba."

In fast informal markets the alert IS the product (Lamudi's lesson:
search-as-destination loses to alert-as-product).  A saved search is
``{domain, query, filters, cadence}`` + an expiry; a scheduler job runs
the matcher, dedups against what was already seen (photo-hash + address
normalization, shared with #93's ``property.scam``), and pushes new
matches to the owner DM.

First domain: ``property``.  The primitive is deliberately generic —
``gig`` (#1) and ``flight`` (#71) inherit it via the matcher registry.

Scheduler seam: ``ensure_schedule(scheduler)`` registers an hourly cron
with action ``SAVED_SEARCH_ACTION``; the host calls ``check_all()`` when
that action fires.  Each search carries its own ``cadence_s`` and
``check_all`` only runs the ones that are due.

Stale-listing hygiene (BuyRentKenya pattern): every search expires
(7/14/28 days, default 14) — stale demand is how scams recycle.
``purge_expired()`` removes them; ``check_all`` skips them.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..property.scam import _normalize_address, _photo_hash

_log = get_logger(__name__)

#: action name the host dispatches to :func:`check_all`
SAVED_SEARCH_ACTION = "saved_search_check"

#: the scheduler cron runs hourly; each search's own cadence gates it
CHECK_CRON = "0 * * * *"

_DEFAULT_DB = os.path.expanduser("~/.nomorals/triggers/saved_search.db")

#: domains the primitive supports; property first, gig + flight inherit
DOMAINS = ("property", "gig", "flight")

#: NL aliases for domains
_DOMAIN_ALIASES = {
    "job": "gig", "jobs": "gig", "freelance": "gig",
    "flat": "property", "rent": "property", "rental": "property",
    "house": "property", "apartment": "property",
}

#: expiry choices in days (BuyRentKenya stale-demand hygiene)
EXPIRY_DAYS = (7, 14, 28)
DEFAULT_TTL_DAYS = 14

#: default cadence between checks for one search (6 hours)
DEFAULT_CADENCE = "6h"

__all__ = [
    "SAVED_SEARCH_ACTION",
    "DOMAINS",
    "EXPIRY_DAYS",
    "SavedSearch",
    "SavedSearchStore",
    "Match",
    "register_matcher",
    "get_matcher",
    "check_search",
    "check_all",
    "ensure_schedule",
    "format_match",
    "format_search",
    "parse_watch",
    "control_watch",
    "register",
]


# ── cadence parsing ──────────────────────────────────────────────────

_CADENCE_RE = re.compile(r"^\s*(\d+)\s*([mhd])\s*$", re.IGNORECASE)


def parse_cadence(text: str) -> int | None:
    """``'30m'``/``'6h'``/``'1d'`` → seconds, or None when unparsable."""
    m = _CADENCE_RE.match(text or "")
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    if n <= 0:
        return None
    return {"m": 60, "h": 3600, "d": 86400}[unit] * n


# ── model ────────────────────────────────────────────────────────────

@dataclass
class SavedSearch:
    """One persistent match-alert: what to watch, how often, until when."""

    id: str = ""
    domain: str = "property"          # property | gig | flight
    query: str = ""                   # human text, e.g. "2bed Yaba under 1.5m"
    filters: dict[str, Any] = field(default_factory=dict)
    cadence_s: int = 6 * 3600
    ttl_days: int = DEFAULT_TTL_DAYS
    created_at: float = 0.0
    expires_at: float = 0.0
    last_run: float = 0.0
    match_count: int = 0
    active: bool = True

    @property
    def expired(self) -> bool:
        return bool(self.expires_at) and time.time() >= self.expires_at

    def due(self, now: float | None = None) -> bool:
        """Active, unexpired, and the cadence interval has elapsed."""
        now = now if now is not None else time.time()
        if not self.active or (self.expires_at and now >= self.expires_at):
            return False
        return (now - (self.last_run or 0)) >= max(60, self.cadence_s)


@dataclass
class Match:
    """One new listing that matched a saved search."""

    search_id: str = ""
    domain: str = "property"
    title: str = ""
    price_kobo: int = 0
    area: str = ""
    address: str = ""
    beds: str = ""
    url: str = ""
    dedup_key: str = ""
    found_at: float = 0.0


# ── store ────────────────────────────────────────────────────────────

class SavedSearchStore:
    """SQLite store for saved searches + seen-match dedup keys."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _DEFAULT_DB
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS saved_searches (
                       id TEXT PRIMARY KEY, domain TEXT, query TEXT,
                       filters_json TEXT, cadence_s INTEGER,
                       ttl_days INTEGER, created_at REAL, expires_at REAL,
                       last_run REAL, match_count INTEGER, active INTEGER)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS seen_matches (
                       search_id TEXT, dedup_key TEXT,
                       found_at REAL,
                       PRIMARY KEY (search_id, dedup_key))""")
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB is an empty store
            _log.warning("saved_search: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    # — CRUD —

    def create(self, domain: str, query: str,
               filters: dict[str, Any] | None = None,
               *, cadence: str = DEFAULT_CADENCE,
               ttl_days: int = DEFAULT_TTL_DAYS,
               now: float | None = None) -> SavedSearch | None:
        """Create a saved search. Never raises; None on bad input."""
        try:
            domain = _canon_domain(domain)
            if domain is None:
                return None
            query = (query or "").strip()[:300]
            if not query:
                return None
            cadence_s = parse_cadence(cadence) or parse_cadence(DEFAULT_CADENCE)
            ttl = int(ttl_days) if int(ttl_days) in EXPIRY_DAYS else DEFAULT_TTL_DAYS
            now = now if now is not None else time.time()
            s = SavedSearch(
                id="watch_" + uuid.uuid4().hex[:8], domain=domain,
                query=query, filters=dict(filters or {}),
                cadence_s=cadence_s, ttl_days=ttl,
                created_at=now, expires_at=now + ttl * 86400)
            if self._db is None:
                return s
            import json as _json
            self._db.execute(
                "INSERT INTO saved_searches VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (s.id, s.domain, s.query, _json.dumps(s.filters),
                 s.cadence_s, s.ttl_days, s.created_at, s.expires_at,
                 s.last_run, s.match_count, 1))
            self._db.commit()
            return s
        except Exception:  # noqa: BLE001
            _log.debug("saved_search create failed", exc_info=True)
            return None

    def get(self, search_id: str) -> SavedSearch | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM saved_searches WHERE id = ?",
                (search_id,)).fetchone()
            return _row_to_search(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def list(self, *, active_only: bool = True,
             include_expired: bool = False) -> list[SavedSearch]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM saved_searches ORDER BY created_at DESC"
            ).fetchall()
            out = [_row_to_search(r) for r in rows]
            now = time.time()
            if active_only:
                out = [s for s in out if s.active]
            if not include_expired:
                out = [s for s in out
                       if not (s.expires_at and now >= s.expires_at)]
            return out
        except Exception:  # noqa: BLE001
            return []

    def remove(self, search_id: str) -> bool:
        """Delete a search and its seen keys. True when it existed."""
        try:
            if self._db is None:
                return False
            cur = self._db.execute(
                "DELETE FROM saved_searches WHERE id = ?", (search_id,))
            self._db.execute(
                "DELETE FROM seen_matches WHERE search_id = ?", (search_id,))
            self._db.commit()
            return (cur.rowcount or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    def purge_expired(self, now: float | None = None) -> int:
        """Delete expired searches (stale-demand hygiene). Returns count."""
        try:
            if self._db is None:
                return 0
            now = now if now is not None else time.time()
            ids = [r["id"] for r in self._db.execute(
                "SELECT id FROM saved_searches WHERE expires_at <= ?",
                (now,)).fetchall()]
            for sid in ids:
                self.remove(sid)
            return len(ids)
        except Exception:  # noqa: BLE001
            return 0

    def mark_run(self, search_id: str, new_matches: int,
                 now: float | None = None) -> None:
        try:
            if self._db is None:
                return
            now = now if now is not None else time.time()
            self._db.execute(
                "UPDATE saved_searches SET last_run = ?, "
                "match_count = match_count + ? WHERE id = ?",
                (now, max(0, int(new_matches)), search_id))
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("saved_search mark_run failed", exc_info=True)

    # — dedup —

    def seen(self, search_id: str, dedup_key: str) -> bool:
        try:
            if self._db is None or not dedup_key:
                return False
            return self._db.execute(
                "SELECT 1 FROM seen_matches WHERE search_id = ? "
                "AND dedup_key = ?", (search_id, dedup_key)).fetchone() \
                is not None
        except Exception:  # noqa: BLE001
            return False

    def mark_seen(self, search_id: str, dedup_key: str,
                  now: float | None = None) -> None:
        try:
            if self._db is None or not dedup_key:
                return
            now = now if now is not None else time.time()
            self._db.execute(
                "INSERT OR IGNORE INTO seen_matches VALUES (?, ?, ?)",
                (search_id, dedup_key, now))
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("saved_search mark_seen failed", exc_info=True)


def _canon_domain(domain: str) -> str | None:
    d = (domain or "").strip().lower()
    if d in DOMAINS:
        return d
    return _DOMAIN_ALIASES.get(d)


def _row_to_search(row: sqlite3.Row) -> SavedSearch:
    import json as _json
    try:
        filters = _json.loads(row["filters_json"] or "{}")
    except Exception:  # noqa: BLE001
        filters = {}
    return SavedSearch(
        id=row["id"], domain=row["domain"] or "property",
        query=row["query"] or "", filters=filters,
        cadence_s=int(row["cadence_s"] or 21600),
        ttl_days=int(row["ttl_days"] or DEFAULT_TTL_DAYS),
        created_at=float(row["created_at"] or 0),
        expires_at=float(row["expires_at"] or 0),
        last_run=float(row["last_run"] or 0),
        match_count=int(row["match_count"] or 0),
        active=bool(row["active"]))


# ── matcher registry ────────────────────────────────────────────────

#: matcher(domain, query, filters) -> [listing dicts].  Listing dict keys:
#: title, price_kobo, area, address, beds, url, photo_bytes (optional).
MatcherFn = Callable[[str, str, dict[str, Any]], list[dict[str, Any]]]

_matchers: dict[str, MatcherFn] = {}


def register_matcher(domain: str, fn: MatcherFn) -> bool:
    """Register a listing-source matcher for a domain. Never raises."""
    try:
        d = _canon_domain(domain)
        if d is None or not callable(fn):
            return False
        _matchers[d] = fn
        return True
    except Exception:  # noqa: BLE001
        return False


def get_matcher(domain: str) -> MatcherFn | None:
    return _matchers.get(_canon_domain(domain) or "")


# ── filtering + dedup ───────────────────────────────────────────────

def _matches_filters(listing: dict[str, Any],
                     filters: dict[str, Any],
                     domain: str = "property") -> bool:
    """Apply beds / max-price / area filters. Best-effort, never raises.

    Property uses verbatim area substring; gig/flight treat the leftover
    as keywords (any word hits).
    """
    try:
        want_beds = filters.get("beds")
        if want_beds:
            have = str(listing.get("beds") or "")
            if str(want_beds) not in have and have not in str(want_beds):
                # numeric compare when both parse
                try:
                    if int(have) != int(want_beds):
                        return False
                except (TypeError, ValueError):
                    return False
        max_price = filters.get("max_price_kobo")
        if max_price:
            try:
                if int(listing.get("price_kobo") or 0) > int(max_price):
                    return False
            except (TypeError, ValueError):
                pass
        area = str(filters.get("area") or "").strip().lower()
        if area:
            hay = " ".join([str(listing.get("area") or ""),
                            str(listing.get("address") or ""),
                            str(listing.get("title") or "")]).lower()
            if domain == "property":
                if area not in hay:
                    return False
            else:
                words = [w for w in area.split() if len(w) > 2]
                if words and not any(w in hay for w in words):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return True


def dedup_key(listing: dict[str, Any]) -> str:
    """Stable identity for a listing.

    Shared with #93: photo-hash when bytes exist, otherwise the
    normalized address (+ price) — the same photo or the same address
    is the same listing across sources.
    """
    try:
        photo = listing.get("photo_bytes") or b""
        if photo:
            return "ph:" + _photo_hash(photo)
        addr = _normalize_address(
            str(listing.get("address") or listing.get("title") or ""))
        price = int(listing.get("price_kobo") or 0)
        return f"ad:{addr}|{price}"
    except Exception:  # noqa: BLE001
        return ""


# ── checking ─────────────────────────────────────────────────────────

def check_search(store: SavedSearchStore, search: SavedSearch, *,
                 matcher: MatcherFn | None = None,
                 now: float | None = None) -> list[Match]:
    """Run one saved search: match → filter → dedup → remember.

    Never raises; returns [] when no matcher is configured, the matcher
    fails, or nothing new matched.
    """
    now = now if now is not None else time.time()
    try:
        if search is None or search.expired or not search.active:
            return []
        fn = matcher or get_matcher(search.domain)
        if fn is None:
            _log.debug("saved_search %s: no matcher for %s",
                       search.id, search.domain)
            return []
        try:
            raw = fn(search.domain, search.query, dict(search.filters)) or []
        except Exception:  # noqa: BLE001 — a dead source != a dead watch
            _log.warning("saved_search %s matcher failed", search.id,
                         exc_info=True)
            return []
        fresh: list[Match] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if not _matches_filters(item, search.filters, search.domain):
                continue
            key = dedup_key(item)
            if not key or store.seen(search.id, key):
                continue
            store.mark_seen(search.id, key, now=now)
            fresh.append(Match(
                search_id=search.id, domain=search.domain,
                title=str(item.get("title") or "")[:160],
                price_kobo=int(item.get("price_kobo") or 0),
                area=str(item.get("area") or "")[:80],
                address=str(item.get("address") or "")[:160],
                beds=str(item.get("beds") or "")[:20],
                url=str(item.get("url") or "")[:300],
                dedup_key=key, found_at=now))
        store.mark_run(search.id, len(fresh), now=now)
        return fresh
    except Exception:  # noqa: BLE001
        _log.debug("saved_search check failed", exc_info=True)
        return []


def check_all(store: SavedSearchStore, *,
              matchers: dict[str, MatcherFn] | None = None,
              sender: Callable[[str], Any] | None = None,
              now: float | None = None) -> list[Match]:
    """Host entry point: purge expired, run due searches, push matches.

    ``sender(text)`` is the owner-DM push seam (mirrors #71's watcher
    seam).  Never raises.
    """
    now = now if now is not None else time.time()
    found: list[Match] = []
    try:
        purged = store.purge_expired(now=now)
        if purged:
            _log.info("saved_search purged %d expired", purged)
        for search in store.list(active_only=True):
            if not search.due(now=now):
                continue
            matcher = (matchers or {}).get(search.domain) \
                or get_matcher(search.domain)
            for m in check_search(store, search, matcher=matcher, now=now):
                found.append(m)
                if sender is not None:
                    try:
                        sender(format_match(m))
                    except Exception:  # noqa: BLE001 — one bad send != dead run
                        _log.warning("saved_search alert send failed",
                                     exc_info=True)
        return found
    except Exception:  # noqa: BLE001
        _log.debug("saved_search check_all failed", exc_info=True)
        return found


def ensure_schedule(scheduler: Any) -> bool:
    """Register the hourly saved-search cron. Idempotent-ish.

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
                if getattr(j, "action", "") == SAVED_SEARCH_ACTION:
                    return True
            await scheduler.schedule_cron(
                task_id="saved-search-hourly",
                cron_expr=CHECK_CRON,
                action=SAVED_SEARCH_ACTION,
                parameters={},
            )
            return True

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_ensure())
        _log.warning("saved_search ensure_schedule inside a running loop")
        return False
    except Exception:  # noqa: BLE001
        _log.debug("saved_search ensure_schedule failed", exc_info=True)
        return False


# ── formatting ───────────────────────────────────────────────────────

def _short_kobo(kobo: int) -> str:
    try:
        n = int(kobo or 0)
    except (TypeError, ValueError):
        return "?"
    if n <= 0:
        return "?"
    if n % 100 == 0:
        amt = n // 100
        if amt >= 1_000_000:
            m = amt / 1_000_000
            return f"₦{m:g}m"
        if amt >= 1_000:
            k = amt / 1_000
            return f"₦{k:g}k"
        return f"₦{amt:,}"
    return f"₦{n / 100:,.0f}"


def format_match(m: Match) -> str:
    """Owner-DM push text for one new match."""
    bits = []
    if m.beds:
        bits.append(f"{m.beds}-bed")
    where = m.area or (m.address[:40] if m.address else "")
    if where:
        bits.append(f"in {where}")
    head = " ".join(bits) or m.title or "new listing"
    price = _short_kobo(m.price_kobo)
    line = f"🏠 New match: {head} — {price}"
    if m.title and m.title not in head:
        line += f"\n{m.title}"
    if m.url:
        line += f"\n{m.url}"
    return line


def format_search(s: SavedSearch) -> str:
    """One-line summary of a saved search for /watch list."""
    f = s.filters
    bits = [s.domain]
    if f.get("beds"):
        bits.append(f"{f['beds']}bed")
    if f.get("area"):
        bits.append(str(f["area"]))
    if f.get("max_price_kobo"):
        bits.append(f"under {_short_kobo(f['max_price_kobo'])}")
    days_left = max(0, int((s.expires_at - time.time()) / 86400)) \
        if s.expires_at else 0
    return (f"• `{s.id}` {' '.join(bits)} — every "
            f"{_fmt_cadence(s.cadence_s)}, {s.match_count} match(es), "
            f"expires in {days_left}d")


def _fmt_cadence(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


# ── NL parsing ───────────────────────────────────────────────────────

_PRICE_RE = re.compile(
    r"(?:under|max|below)\s*[₦]?\s*([\d.,]+)\s*([mk])?\b", re.IGNORECASE)
_BEDS_RE = re.compile(r"\b(\d+)\s*-?\s*bed(?:room)?s?\b", re.IGNORECASE)
_TTL_RE = re.compile(r"\bfor\s+(\d+)\s*days?\b", re.IGNORECASE)
_FILLER = {"in", "a", "the", "for", "flat", "apartment", "house", "rent",
           "rental", "to", "me", "please", "pls"}


def _parse_price_kobo(text: str) -> int:
    """'under 1.5m' → 150000000 kobo. 0 when unparsable."""
    m = _PRICE_RE.search(text or "")
    if not m:
        return 0
    try:
        amt = float(m.group(1).replace(",", ""))
    except ValueError:
        return 0
    unit = (m.group(2) or "").lower()
    if unit == "m":
        amt *= 1_000_000
    elif unit == "k":
        amt *= 1_000
    return int(amt * 100)


def parse_watch(text: str) -> dict[str, Any] | None:
    """Parse '/watch 2bed Yaba under 1.5m for 14 days' → search spec.

    Returns ``{domain, query, filters, ttl_days}`` or None when there is
    nothing watchable.  Never raises.
    """
    try:
        raw = (text or "").strip()
        if not raw:
            return None
        words = raw.split()
        domain = "property"
        first = words[0].lower()
        canon = _canon_domain(first)
        if canon is not None and len(words) > 1:
            domain, words = canon, words[1:]
        body = " ".join(words)

        filters: dict[str, Any] = {}
        m = _BEDS_RE.search(body)
        if m:
            filters["beds"] = m.group(1)
        price_kobo = _parse_price_kobo(body)
        if price_kobo > 0:
            filters["max_price_kobo"] = price_kobo
        ttl = DEFAULT_TTL_DAYS
        m = _TTL_RE.search(body)
        if m:
            want = int(m.group(1))
            ttl = min(EXPIRY_DAYS, key=lambda d: abs(d - want))

        # area = leftover words after stripping matched phrases + filler
        clean = _BEDS_RE.sub(" ", body)
        clean = _PRICE_RE.sub(" ", clean)
        clean = _TTL_RE.sub(" ", clean)
        area_words: list[str] = []
        for w in clean.split():
            wl = w.lower().strip(",.")
            if not wl or wl in _FILLER:
                continue
            if re.fullmatch(r"[\d.,₦mk]+", wl):
                continue
            area_words.append(w.strip(",."))
        area = re.sub(r"\s+", " ", " ".join(area_words)).strip()
        if area:
            filters["area"] = area[:60]

        if not filters and not body:
            return None
        return {"domain": domain, "query": raw[:300],
                "filters": filters, "ttl_days": ttl}
    except Exception:  # noqa: BLE001
        return None


# ── chat ─────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("🔎 /watch — persistent match alerts.\n"
            "  /watch 2bed Yaba under 1.5m [for 7|14|28 days] — watch property\n"
            "  /watch gig <query> — watch gigs · /watch flight <route>\n"
            "  /watch list — active watches\n"
            "  /watch stop <id> — remove a watch\n"
            "Owner only. Watches auto-expire (default 14d) — stale demand "
            "is how scams recycle.")


def _get_store(context: Any = None) -> SavedSearchStore:
    store = getattr(context, "saved_search_store", None) \
        if context is not None else None
    return store if isinstance(store, SavedSearchStore) \
        else SavedSearchStore()


def control_watch(tail: str, context: Any = None, chat: Any = None,
                  **kwargs: Any) -> str:
    """Chat entry: /watch. Owner-only (enforced at dispatch)."""
    try:
        tail = (tail or "").strip()
        store = _get_store(context)
        if not tail or tail.lower() in ("help", "?"):
            return _usage()
        low = tail.lower()
        if low == "list":
            watches = store.list(active_only=True)
            if not watches:
                return "no active watches. /watch 2bed Yaba under 1.5m"
            return "🔎 active watches:\n" + "\n".join(
                format_search(s) for s in watches)
        if low.startswith("stop "):
            sid = tail[5:].strip()
            if store.remove(sid):
                return f"stopped watch `{sid}`."
            return f"no watch `{sid}` — /watch list"
        if low.startswith("check"):
            # manual sweep (owner): run due watches now
            sender = getattr(context, "owner_sender", None) \
                if context is not None else None
            found = check_all(store, sender=sender)
            if not found:
                return "swept due watches — no new matches."
            return f"swept — {len(found)} new match(es) pushed."
        spec = parse_watch(tail)
        if spec is None:
            return _usage()
        cadence = "6h"
        s = store.create(spec["domain"], spec["query"],
                         spec["filters"], cadence=cadence,
                         ttl_days=spec["ttl_days"])
        if s is None:
            return "couldn't create that watch — try /watch help"
        f = s.filters
        desc = []
        if f.get("beds"):
            desc.append(f"{f['beds']}-bed")
        if f.get("area"):
            desc.append(str(f["area"]))
        if f.get("max_price_kobo"):
            desc.append(f"under {_short_kobo(f['max_price_kobo'])}")
        what = " ".join(desc) or s.query
        return (f"🔎 watching: {what} ({s.domain}) — I'll ping you the "
                f"moment a match lands. `{s.id}`, expires in "
                f"{s.ttl_days}d. /watch stop {s.id}")
    except Exception:  # noqa: BLE001
        _log.warning("watch failed", exc_info=True)
        return "watch failed — try /watch help"


def register(registry: Any) -> None:
    """Tool-registry hook."""
    try:
        registry.register(
            "watch", control_watch,
            "Persistent match alerts — saved searches that push the moment "
            "a listing matches (property first; gig/flight inherit).")
    except Exception:  # noqa: BLE001
        _log.debug("watch register failed", exc_info=True)
