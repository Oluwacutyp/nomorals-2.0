"""Source health: every source verified-live, or stood down.

Owner rule (2026-10-18): a source that doesn't work is WORSE than no
source — it looks like coverage but isn't. This module is the
enforcement: each registered source carries a live-verified status,
re-checked on a schedule by the ``sourcehealth`` expansion system, and
only ``verified-live`` sources belong in an active chain.

Design (mined, not guessed — see module docstring notes):

* Feed-health tracking (RSS aggregator practice): every check records
  ``last_checked`` / ``last_ok`` / ``fail_count`` / ``last_error``.
* Circuit breaker: N consecutive failures stand a source down
  (excluded from active sets, still re-checked); a higher threshold
  disables it. A later success resets the streak and re-activates.
* Conditional requests: ETag / Last-Modified are stored and re-sent,
  so an unchanged feed answers 304 and costs almost nothing.
* Classification is evidence-based, not vibes: HTTP status + payload
  shape decide. A bot-wall page (Cloudflare / CAPTCHA / "Just a
  moment") is ``bot-blocked`` — reachable but unusable from this
  network, NOT dead. A refused/reset connection is ``unverifiable``
  — the evidence is about the verifier's network, not the source.
  404/410, or HTTP 200 with zero usable items, is ``dead``.
* Verification means reachable AND usable: an RSS feed must parse to
  >=1 item, a JSON API must return >=1 result, an HTML search page
  must contain >=1 real result link. A 200 with an empty shell (the
  JS-SPA placeholder pattern) is dead-as-a-source.

The checker is transport-agnostic: callers inject ``fetcher(url,
headers) -> FetchResult``. The default fetcher is plain urllib with a
browser UA. Tests inject fakes; production passes the real one. The
checker never raises.
"""

from __future__ import annotations

import re
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "SourceSpec",
    "SourceRecord",
    "FetchResult",
    "SourceHealth",
    "check_one",
    "RSS", "JSON", "JSON_WEATHER", "HTML_LINKS", "HTML_POSTS",
    "HTML_SONGLINKS", "HTML_BANDCAMP", "HTML_ARCHIVE", "TEXT",
    "JSON_GUTENBERG", "EXTERNAL_ADAPTER",
    "STAND_DOWN_AFTER", "DISABLE_AFTER",
    "STATUSES",
]

#: Source kinds — each maps to a "usable payload" parser.
RSS = "rss"
JSON = "json"
JSON_WEATHER = "json-weather"
HTML_LINKS = "html-links"
HTML_POSTS = "html-posts"
HTML_SONGLINKS = "html-songlinks"
HTML_BANDCAMP = "html-bandcamp"
HTML_ARCHIVE = "html-archive"
TEXT = "text"
JSON_GUTENBERG = "json-gutenberg"
#: Adapter-gated sources (yt-dlp / OAuth / API adapters): not
#: plain-HTTP verifiable, so health never pretends to verify them.
EXTERNAL_ADAPTER = "external-adapter"

#: Consecutive failures before a source is stood down / disabled.
STAND_DOWN_AFTER = 3
DISABLE_AFTER = 10

STATUSES = (
    "unknown", "verified-live", "dead", "bot-blocked", "geo-blocked",
    "unverifiable", "stood-down", "disabled",
)

_BOT_SIGNALS = (
    "captcha", "verify you are human", "are you a robot",
    "unusual traffic", "access denied", "attention required",
    "cloudflare", "just a moment", "please verify", "bot detected",
    "automated requests", "enable javascript", "js challenge",
    "client challenge", "access restricted", "request was blocked",
    "activity is prohibited",
)
_GEO_SIGNALS = (
    "not available in your country", "not available in your region",
    "geo-block", "geoblocked", "restricted in your country",
    "not available in your location", "unavailable for legal reasons",
)

_BROWSER_UA = (
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
)
_FETCH_TIMEOUT = 20.0


@dataclass
class SourceSpec:
    """One verifiable source, as collected by sources_registry."""

    key: str
    category: str
    name: str
    url: str
    kind: str = RSS
    notes: str = ""


@dataclass
class SourceRecord:
    """Health state for one source (persisted in SQLite)."""

    key: str
    category: str = ""
    name: str = ""
    url: str = ""
    kind: str = RSS
    status: str = "unknown"
    http: int = 0
    items: int = 0
    ms: int = 0
    fail_count: int = 0
    consecutive_failures: int = 0
    last_checked: float = 0.0
    last_ok: float = 0.0
    last_error: str = ""
    etag: str = ""
    last_modified: str = ""
    notes: str = ""


@dataclass
class FetchResult:
    """One HTTP attempt. ``status`` 0 = transport failure (see error)."""

    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)
    error: str = ""


def default_fetcher(url: str, headers: dict | None = None) -> FetchResult:
    """Plain-urllib GET with a browser UA. Never raises."""
    merged = {"User-Agent": _BROWSER_UA,
              "Accept": "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,application/json;q=0.8,"
                        "*/*;q=0.7"}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, headers=merged)
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            return FetchResult(resp.status, resp.read()[:2_000_000],
                               {k.lower(): v for k, v in resp.headers.items()})
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()[:200_000]
        except Exception:  # noqa: BLE001
            body = b""
        return FetchResult(exc.code, body,
                           {k.lower(): v for k, v in (exc.headers or {}).items()})
    except Exception as exc:  # noqa: BLE001 — transport failure
        return FetchResult(0, b"", {}, error=str(exc)[:300])


# ── payload usability: "usable" means >=1 real item ──────────────────

def _rss_items(body: bytes) -> int:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return 0
    return sum(1 for n in root.iter()
               if n.tag.rsplit("}", 1)[-1].lower() in ("item", "entry"))


def _json_items(body: bytes) -> int:
    import json as _json

    try:
        data = _json.loads(body.decode("utf-8", errors="replace") or "{}")
    except Exception:  # noqa: BLE001
        return 0
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return 0
    best = 0
    for key in ("results", "data", "docs", "items", "recordings",
                "audio", "photos", "features"):
        val = data.get(key)
        if isinstance(val, list):
            best = max(best, len(val))
    query = data.get("query")
    if isinstance(query, dict):
        found = query.get("search")
        if isinstance(found, list):
            best = max(best, len(found))
    for key in ("result_count", "numFound", "count", "total"):
        val = data.get(key)
        if isinstance(val, int) and val > 0:
            best = max(best, val)
    if best == 0 and "response" in data and isinstance(data["response"], dict):
        docs = data["response"].get("docs")
        if isinstance(docs, list):
            best = len(docs)
    return best


def _count_usable(kind: str, body: bytes) -> int:
    text = body.decode("utf-8", errors="replace")
    if kind == RSS:
        return _rss_items(body)
    if kind == JSON:
        return _json_items(body)
    if kind == JSON_WEATHER:
        import json as _json

        try:
            data = _json.loads(text or "{}")
        except Exception:  # noqa: BLE001
            return 0
        if not isinstance(data, dict):
            return 0
        keys = ("hourly", "current", "results", "properties", "features",
                "current_weather", "daily")
        return 1 if any(k in data for k in keys) else 0
    if kind == HTML_LINKS:
        links = set(re.findall(r'href="(https?://[^"]+)"', text))
        return len(links)
    if kind == HTML_POSTS:
        return len(set(re.findall(
            r'<a[^>]+href="([^"]+)"[^>]*>([^<]{8,120})</a>', text)))
    if kind == HTML_SONGLINKS:
        return len(re.findall(r"/songs/\d+", text))
    if kind == HTML_BANDCAMP:
        return len(re.findall(r"bandcamp\.com/(?:album|track)/", text))
    if kind == HTML_ARCHIVE:
        return _json_items(body)
    if kind == TEXT:
        return 1 if len(text.strip()) > 100 else 0
    if kind == JSON_GUTENBERG:
        return 1 if len(text) > 1000 else 0
    return 0


def _classify(status: int, body: bytes, items: int) -> tuple[str, str]:
    """(status, detail) from one fetch + its usable-item count."""
    text = body.decode("utf-8", errors="replace")
    low = text[:4000].lower()
    if status == 0:
        return "unverifiable", "transport failure (evidence is about " \
                                "this network, not the source)"
    if status in (404, 410):
        return "dead", f"HTTP {status}"
    if status == 451:
        return "geo-blocked", "HTTP 451 unavailable for legal reasons"
    if any(sig in low for sig in _GEO_SIGNALS):
        return "geo-blocked", f"HTTP {status} region-blocked page"
    if any(sig in low for sig in _BOT_SIGNALS):
        return "bot-blocked", f"HTTP {status} bot-wall/challenge page"
    if status in (403, 429, 202) and items == 0:
        return "bot-blocked", f"HTTP {status} no usable payload (anti-bot likely)"
    if status >= 500:
        return "unverifiable", f"HTTP {status} server error (transient)"
    if status != 200:
        return "unverifiable", f"HTTP {status}"
    if items <= 0:
        return "dead", "HTTP 200 but zero usable items (empty/changed/unparseable)"
    return "verified-live", f"{items} usable items"


def check_one(spec: SourceSpec,
              fetcher: Callable[..., FetchResult] | None = None,
              *, now: float | None = None,
              prior: SourceRecord | None = None) -> SourceRecord:
    """Verify one source. Never raises. ``prior`` carries ETag/state."""
    now = time.time() if now is None else now
    fetch = fetcher or default_fetcher
    rec = SourceRecord(
        key=spec.key, category=spec.category, name=spec.name,
        url=spec.url, kind=spec.kind, notes=spec.notes)
    if prior is not None:
        rec.fail_count = prior.fail_count
        rec.consecutive_failures = prior.consecutive_failures
        rec.last_ok = prior.last_ok
        rec.items = prior.items
        rec.etag = prior.etag
        rec.last_modified = prior.last_modified
    if spec.kind == EXTERNAL_ADAPTER:
        rec.status = "unknown"
        rec.last_checked = now
        rec.last_error = "adapter-gated: not plain-HTTP verifiable"
        return rec
    headers: dict[str, str] = {}
    if prior is not None:
        if prior.etag:
            headers["If-None-Match"] = prior.etag
        if prior.last_modified:
            headers["If-Modified-Since"] = prior.last_modified
    started = time.time()
    try:
        result = fetch(spec.url, headers or None)
    except Exception as exc:  # noqa: BLE001 — a fetcher must never kill health
        result = FetchResult(0, b"", {}, error=str(exc)[:300])
    rec.last_checked = now
    rec.http = result.status
    rec.ms = int((time.time() - started) * 1000)
    if result.error and result.status == 0:
        rec.last_error = result.error
    new_etag = result.headers.get("etag", "")
    if new_etag:
        rec.etag = new_etag
    new_lm = result.headers.get("last-modified", "")
    if new_lm:
        rec.last_modified = new_lm
    if result.status == 304 and prior is not None:
        # Unchanged since last check: still live, previous items stand.
        rec.status = "verified-live"
        rec.consecutive_failures = 0
        rec.last_ok = now
        rec.last_error = ""
        return rec
    items = _count_usable(spec.kind, result.body)
    status, detail = _classify(result.status, result.body, items)
    if result.error and result.status == 0:
        detail = result.error
    rec.last_error = "" if status == "verified-live" else detail
    if status == "verified-live":
        rec.items = items
        rec.consecutive_failures = 0
        rec.last_ok = now
        rec.status = "verified-live"
        return rec
    # Failure path: circuit breaker counts consecutive failures.
    rec.consecutive_failures = (prior.consecutive_failures + 1) if prior else 1
    rec.fail_count = (prior.fail_count + 1) if prior else 1
    if status == "unverifiable":
        rec.status = "unverifiable"
    elif rec.consecutive_failures >= DISABLE_AFTER:
        rec.status = "disabled"
    elif rec.consecutive_failures >= STAND_DOWN_AFTER:
        rec.status = "stood-down"
    else:
        rec.status = status  # dead / bot-blocked / geo-blocked
    return rec


class SourceHealth:
    """Durable per-source health over the injected DB handle."""

    def __init__(self, db: Any,
                 fetcher: Callable[..., FetchResult] | None = None) -> None:
        self.db = db
        self._fetcher = fetcher
        db.execute("""
            CREATE TABLE IF NOT EXISTS agency_source_health (
                key TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL DEFAULT '', url TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'rss', status TEXT NOT NULL DEFAULT 'unknown',
                http INTEGER NOT NULL DEFAULT 0, items INTEGER NOT NULL DEFAULT 0,
                ms INTEGER NOT NULL DEFAULT 0, fail_count INTEGER NOT NULL DEFAULT 0,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                last_checked REAL NOT NULL DEFAULT 0, last_ok REAL NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '', etag TEXT NOT NULL DEFAULT '',
                last_modified TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT ''
            )""")

    # ── persistence ──────────────────────────────────────────────
    def _save(self, rec: SourceRecord) -> None:
        self.db.execute(
            "INSERT INTO agency_source_health VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET category=excluded.category, name=excluded.name, "
            "url=excluded.url, kind=excluded.kind, status=excluded.status, http=excluded.http, "
            "items=excluded.items, ms=excluded.ms, fail_count=excluded.fail_count, "
            "consecutive_failures=excluded.consecutive_failures, last_checked=excluded.last_checked, "
            "last_ok=excluded.last_ok, last_error=excluded.last_error, etag=excluded.etag, "
            "last_modified=excluded.last_modified, notes=excluded.notes",
            (rec.key, rec.category, rec.name, rec.url, rec.kind, rec.status,
             rec.http, rec.items, rec.ms, rec.fail_count,
             rec.consecutive_failures, rec.last_checked, rec.last_ok,
             rec.last_error, rec.etag, rec.last_modified, rec.notes))

    def record(self, key: str) -> SourceRecord | None:
        row = self.db.query_one(
            "SELECT * FROM agency_source_health WHERE key = ?", (key,))
        if not row:
            return None
        return SourceRecord(
            key=row["key"], category=row["category"], name=row["name"],
            url=row["url"], kind=row["kind"], status=row["status"],
            http=int(row["http"]), items=int(row["items"]),
            ms=int(row["ms"]), fail_count=int(row["fail_count"]),
            consecutive_failures=int(row["consecutive_failures"]),
            last_checked=float(row["last_checked"]),
            last_ok=float(row["last_ok"]), last_error=row["last_error"],
            etag=row["etag"], last_modified=row["last_modified"],
            notes=row["notes"])

    def all_records(self) -> list[SourceRecord]:
        rows = self.db.query(
            "SELECT * FROM agency_source_health ORDER BY category, name") or []
        out = []
        for row in rows:
            rec = self.record(row["key"])
            if rec is not None:
                out.append(rec)
        return out

    def active_sources(self) -> list[SourceRecord]:
        """Only verified-live sources belong in an active chain."""
        return [r for r in self.all_records() if r.status == "verified-live"]

    # ── verification ─────────────────────────────────────────────
    def verify(self, spec: SourceSpec,
               fetcher: Callable[..., FetchResult] | None = None, *,
               now: float | None = None) -> SourceRecord:
        rec = check_one(spec, fetcher or self._fetcher, now=now,
                        prior=self.record(spec.key))
        self._save(rec)
        return rec

    def note_result(self, spec: SourceSpec, *, ok: bool,
                    error: str = "", items: int = 0, http: int = 0,
                    now: float | None = None) -> SourceRecord:
        """Record one observed fetch outcome WITHOUT re-fetching.

        For callers (like ``NewsAgent.run``) that already fetched the
        source themselves: reports the outcome into the same circuit
        breaker ``check_one`` maintains, so run-path failures count
        toward stand-down/disable and successes re-activate.

        Transitions mirror ``check_one``: a success marks the source
        ``verified-live`` and resets the consecutive-failure streak; a
        failure bumps the streak — a lone failure is ``unverifiable``
        (evidence about this attempt, not a verdict), ``STAND_DOWN_AFTER``
        consecutive failures stand the source down (excluded from active
        sets until a success re-activates it), ``DISABLE_AFTER`` disables
        it. Never raises.
        """
        now = time.time() if now is None else now
        prior = self.record(spec.key)
        rec = SourceRecord(
            key=spec.key, category=spec.category, name=spec.name,
            url=spec.url, kind=spec.kind, notes=spec.notes)
        if prior is not None:
            rec.fail_count = prior.fail_count
            rec.consecutive_failures = prior.consecutive_failures
            rec.last_ok = prior.last_ok
            rec.items = prior.items
            rec.etag = prior.etag
            rec.last_modified = prior.last_modified
        rec.last_checked = now
        rec.http = http
        if ok:
            rec.status = "verified-live"
            rec.consecutive_failures = 0
            rec.last_ok = now
            rec.last_error = ""
            rec.items = items
        else:
            rec.consecutive_failures = (prior.consecutive_failures + 1) \
                if prior else 1
            rec.fail_count = (prior.fail_count + 1) if prior else 1
            rec.last_error = (error or "fetch failed")[:300]
            if rec.consecutive_failures >= DISABLE_AFTER:
                rec.status = "disabled"
            elif rec.consecutive_failures >= STAND_DOWN_AFTER:
                rec.status = "stood-down"
            else:
                rec.status = "unverifiable"
        try:
            self._save(rec)
        except Exception:  # noqa: BLE001 - health must never break a run
            pass
        return rec

    def verify_all(self, specs: list[SourceSpec],
                   fetcher: Callable[..., FetchResult] | None = None, *,
                   now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        summary: dict[str, Any] = {"checked": 0, "verified_live": 0,
                                   "dead": 0, "bot_blocked": 0,
                                   "geo_blocked": 0, "unverifiable": 0,
                                   "stood_down": 0, "disabled": 0,
                                   "changed": []}
        for spec in specs:
            before = self.record(spec.key)
            rec = self.verify(spec, fetcher=fetcher, now=now)
            summary["checked"] += 1
            bucket = {"verified-live": "verified_live", "dead": "dead",
                      "bot-blocked": "bot_blocked",
                      "geo-blocked": "geo_blocked",
                      "unverifiable": "unverifiable",
                      "stood-down": "stood_down", "disabled": "disabled"}.get(rec.status)
            if bucket:
                summary[bucket] += 1
            if before is None or before.status != rec.status:
                summary["changed"].append(
                    {"key": rec.key,
                     "from": before.status if before else None,
                     "to": rec.status})
        return summary
