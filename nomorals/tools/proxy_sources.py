"""PROXY_SOURCES: the 2026 source catalog + per-source yield discipline.

This module is the *new* PROXY_SOURCES build.  It does NOT rebuild the
existing organs — it wires into them:

* :mod:`nomorals.tools.proxysources` — the persistent ``SourceRegistry``
  (built-in catalog + health + 24h cooldown).  Entries here that are
  already built-in are *references*, not duplicates.
* :mod:`nomorals.tools.proxylab` — ``ProxyScraper`` (fetch + parse),
  ``ProxyTester`` (liveness / latency / egress-IP / anonymity), ``Proxy``
  (now carrying ``source`` / ``also_from`` provenance).
* :mod:`nomorals.connectors.proxypool` — storage + rotation + the tier
  gate (``core.proxy_tiers``).

What this module adds:

* ``SOURCES_2026`` — the 2026 feed catalog: JSON APIs (ProxyScrape v4,
  GeoNode, Proxifly POST, proxmint), GitHub raw lists, HTML list sites.
  Every entry carries its measured live rate (where a sweep measured
  one), its probe budget, and a ``verified`` flag.  Entries whose
  endpoint could NOT be verified (HideMy.name — now a VPN product with
  no list page; scrappey — no verifiable list endpoint found 2026-10-16)
  are listed with an empty URL and ``verified=False`` so they are
  *visible* but never fetched until a real endpoint is confirmed.
  0%-yield sources (Proxifly REST, vakhov) stay in the catalog with a
  near-zero probe budget — kept for refresh checks, never burned on.
* ``SourceYieldTracker`` — per-source live rate + EWMA, auto-quarantine
  of a whole source when its yield drops, per-source probe budgets.
* ``dedupe_proxies`` — dedupe by ``(host, port, scheme)`` *before*
  testing, recording which source each proxy came from (``source`` +
  ``also_from``) instead of dropping provenance.
* ``verify_content_integrity`` — content-integrity check alongside the
  egress-IP proof: hash a known-stable page fetched through the proxy
  and compare — catches proxies that rewrite content.
* ``refresh_sources`` — merge the 2026 catalog into a
  ``SourceRegistry``, re-check staleness (the brief's staleness check
  found 8/27 sources dead — this re-checks every source instead of
  trusting that number), surface quarantined sources.
* ``fetch_proxifly`` — the Proxifly POST feed (the only POST feed in
  the catalog; the scraper's GET fetcher cannot speak it).

All proxies in this catalog are ``free`` tier.  Nothing here is ever
eligible for credential-carrying traffic — see ``core.proxy_tiers``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .proxysources import BUILT_IN_SOURCES, SourceRegistry

__all__ = [
    "SOURCES_2026",
    "SOURCE_BY_NAME",
    "DEFAULT_PROBE_BUDGET",
    "ZERO_YIELD_PROBE_BUDGET",
    "YIELD_QUARANTINE_HOURS",
    "STALE_AFTER_DAYS",
    "SourceYieldTracker",
    "dedupe_proxies",
    "verify_content_integrity",
    "sha256_hex",
    "check_staleness",
    "refresh_sources",
    "fetch_proxifly",
    "fetch_proxifly_text",
]

_log = get_logger(__name__)

#: Default probe budget per source per test round (matches the lab's
#: test_many default scale; the tracker enforces it per source).
DEFAULT_PROBE_BUDGET = 150
#: Near-zero budget for 0%-yield sources: enough to notice a revival,
#: never enough to burn a round on a dead feed.
ZERO_YIELD_PROBE_BUDGET = 2
#: Whole-source quarantine when yield collapses (hours).
YIELD_QUARANTINE_HOURS = 24.0
#: A source with no successful fetch in this long is stale.
STALE_AFTER_DAYS = 30.0
#: EWMA smoothing for per-source live rate (same alpha as pool latency).
YIELD_EWMA_ALPHA = 0.3

#: ``kind`` values match ``proxysources``: list / protocol / html / json.
#: ``method`` is GET except Proxifly (POST).  ``measured_live_rate`` is
#: the fraction of scraped entries that passed a live relay test in the
#: sweep that measured it (``None`` = never measured, not zero).
SOURCES_2026: tuple[dict[str, Any], ...] = (
    # ---- JSON APIs ---------------------------------------------------
    {"name": "proxyscrape-v4", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Already in BUILT_IN_SOURCES (per-protocol v4 endpoints). Reference, not a duplicate."},
    {"name": "geonode-api", "url": "", "kind": "json", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Already in BUILT_IN_SOURCES (proxylist.geonode.com API). Reference, not a duplicate."},
    {"name": "proxifly-post", "url": "https://api.proxifly.dev/get-proxy",
     "kind": "list", "method": "POST",
     "tier": "free", "measured_live_rate": 0.0, "probe_budget": ZERO_YIELD_PROBE_BUDGET,
     "verified": True,
     "notes": "POST feed (see fetch_proxifly). 0% measured — keep the feed, budget near-zero probes. "
              "GET mirror lives under proxifly-github-*."},
    {"name": "proxifly-github-all",
     "url": "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/all/data.txt",
     "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Proxifly's GitHub mirror, updated every 5 min. Prefer over the POST feed."},
    {"name": "proxifly-github-http",
     "url": "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt",
     "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Proxifly GitHub mirror, HTTP."},
    {"name": "proxifly-github-socks4",
     "url": "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/socks4/data.txt",
     "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Proxifly GitHub mirror, SOCKS4."},
    {"name": "proxifly-github-socks5",
     "url": "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/socks5/data.txt",
     "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Proxifly GitHub mirror, SOCKS5."},
    {"name": "proxmint-api",
     "url": "https://proxmint.com/api/free-proxies?protocol=socks5&format=txt",
     "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "proxmint live API, re-validated every 30 min by a real request through each proxy. "
              "60 req/min, no key. Swap protocol= for http/https/socks4."},
    {"name": "proxmint-github-all",
     "url": "https://raw.githubusercontent.com/proxmint/free-proxy-list/main/proxies/all.txt",
     "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "proxmint GitHub mirror: every entry passed a relay test within ~5.5h (median 30 min)."},
    {"name": "proxmint-github-socks5",
     "url": "https://raw.githubusercontent.com/proxmint/free-proxy-list/main/proxies/socks5.txt",
     "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "proxmint GitHub mirror, SOCKS5."},
    {"name": "proxmint-github-elite",
     "url": "https://raw.githubusercontent.com/proxmint/free-proxy-list/main/proxies/elite.txt",
     "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "proxmint elite subset (no leak found in their header check). Still free tier — "
              "a source's anonymity claim never overrides the lab's own test or the tier gate."},
    {"name": "proxmint-github-json",
     "url": "https://raw.githubusercontent.com/proxmint/free-proxy-list/main/proxies/all.json",
     "kind": "json", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "proxmint structured feed: country, anonymity, latency, uptime, score."},
    # ---- GitHub raw (already-built-in ones are references) ------------
    {"name": "iplocate-all", "url": "", "kind": "protocol", "method": "GET",
     "tier": "free", "measured_live_rate": 0.117, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Already in BUILT_IN_SOURCES. Measured live rate 11.7% (prior sweep)."},
    {"name": "thespeedx", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": 0.083, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Already in BUILT_IN_SOURCES (per-protocol files). Measured live rate 8.3% (prior sweep)."},
    {"name": "monosans", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": 0.008, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Already in BUILT_IN_SOURCES. Measured live rate 0.8% (prior sweep) — low yield but "
              "honest metadata (every entry passed a full fetch on their side); keep."},
    {"name": "hookzof-socks5", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES."},
    {"name": "roosterkid", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (HTTPS/SOCKS4/SOCKS5 + RAW)."},
    {"name": "zloi-user", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (zloi-http)."},
    {"name": "sunny9577-socks5",
     "url": "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/generated/socks5_proxies.txt",
     "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "Re-added at its current path (an earlier pass cut it as moved). "
              "First fetch re-verifies the endpoint; a 404 quarantines it via the registry."},
    {"name": "vakhov-socks5", "url": "", "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": 0.0, "probe_budget": ZERO_YIELD_PROBE_BUDGET,
     "verified": True,
     "notes": "Already in BUILT_IN_SOURCES. Measured 0% — deprioritised, near-zero probes."},
    # ---- HTML list sites ----------------------------------------------
    {"name": "free-proxy-list.net", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (fpl-http / fpl-ssl)."},
    {"name": "us-proxy.org", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (usproxy-http)."},
    {"name": "socks-proxy.net", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (socksproxy-socks5)."},
    {"name": "sslproxies.org", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (sslproxies-http)."},
    {"name": "proxynova",
     "url": "https://www.proxynova.com/proxy-server-list/",
     "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "HTML table site. ProxyNova obfuscates IP/port cells in-page — the generic HTML "
              "parser may return 0 rows; a dedicated parse is a follow-up, not a silent skip."},
    {"name": "hidemy-name", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": 0,
     "verified": False,
     "notes": "UNVERIFIED — no proxy-list endpoint found 2026-10-16; hidemy.name is now a VPN "
              "product. Listed so the gap is visible; NEVER fetched until a real list endpoint "
              "is confirmed."},
    {"name": "spys-one", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True, "notes": "Already in BUILT_IN_SOURCES (spys-http / spys-mixed / spys-socks)."},
    {"name": "proxy-list.download",
     "url": "https://www.proxy-list.download/api/v1/get?type=socks5",
     "kind": "list", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "API form of the site (type=http|https|socks4|socks5). An earlier pass saw a 502 "
              "from the site — the API endpoint is kept with a normal budget and the registry "
              "quarantine handles a repeat failure."},
    {"name": "openproxy.space",
     "url": "https://openproxy.space/list",
     "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": ZERO_YIELD_PROBE_BUDGET,
     "verified": False,
     "notes": "UNVERIFIED/STALE — 521 in the 2026-10-01 pass. Kept with near-zero probes; "
              "a successful fetch re-verifies it, repeated failure quarantines it."},
    {"name": "scrappey", "url": "", "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": 0,
     "verified": False,
     "notes": "UNVERIFIED — no verifiable list endpoint found 2026-10-16. Listed so the gap is "
              "visible; NEVER fetched until a real endpoint is confirmed."},
    {"name": "geonode-site",
     "url": "https://geonode.com/free-proxy-list",
     "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "GeoNode's human-readable list page (the API lives under geonode-api)."},
    {"name": "proxyscrape-site",
     "url": "https://proxyscrape.com/free-proxy-list",
     "kind": "html", "method": "GET",
     "tier": "free", "measured_live_rate": None, "probe_budget": DEFAULT_PROBE_BUDGET,
     "verified": True,
     "notes": "ProxyScrape's human-readable list page (the API lives under proxyscrape-v4)."},
)

SOURCE_BY_NAME: dict[str, dict[str, Any]] = {
    entry["name"]: entry for entry in SOURCES_2026
}


def _builtin_names() -> set[str]:
    return {name for name, _url, _kind in BUILT_IN_SOURCES}


# ── per-source yield tracking ──────────────────────────────────────────────


@dataclass
class SourceYieldTracker:
    """Per-source live rate, EWMA, auto-quarantine, probe budgets.

    Persisted as JSON next to the pool (same directory pattern as the
    registry).  One record per source::

        {scraped, alive, live_rate, live_rate_ewma, zero_yield_streak,
         probes_spent, quarantined_until, last_ok, last_seen, error}

    A whole source auto-quarantines when its yield collapses: EWMA live
    rate under 1% after at least 3 measured rounds, or 3 consecutive
    zero-yield scrapes.  Quarantine is time-boxed (24h) — a source can
    revive; the registry's own cooldown works the same way.
    """

    path: str | Path | None = None
    quarantine_hours: float = YIELD_QUARANTINE_HOURS
    alpha: float = YIELD_EWMA_ALPHA
    _data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        self._data = {}
        if self.path is None:
            return
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
        except Exception:  # noqa: BLE001 - corrupt yield file means: rebuild
            self._data = {}

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, indent=1, sort_keys=True),
                encoding="utf-8")
        except OSError:
            pass

    def _entry(self, name: str) -> dict[str, Any]:
        entry = self._data.get(name)
        if entry is None:
            entry = {"scraped": 0, "alive": 0, "live_rate": None,
                     "live_rate_ewma": None, "zero_yield_streak": 0,
                     "rounds": 0,
                     "probes_spent": 0, "quarantined_until": 0.0,
                     "last_ok": 0.0, "last_seen": 0.0, "error": ""}
            self._data[name] = entry
        return entry

    # -- recording ---------------------------------------------------------
    def record_scrape(self, name: str, *, scraped: int, alive: int,
                      error: str = "") -> dict[str, Any]:
        """One scrape + test round for ``name`` → updated yield record."""
        entry = self._entry(name)
        now = time.time()
        scraped = max(0, int(scraped))
        alive = max(0, min(int(alive), scraped)) if scraped else 0
        entry["scraped"] = int(entry.get("scraped") or 0) + scraped
        entry["alive"] = int(entry.get("alive") or 0) + alive
        entry["last_seen"] = now
        if scraped:
            entry["rounds"] = int(entry.get("rounds") or 0) + 1
            entry["live_rate"] = round(alive / scraped, 4)
            prev = entry.get("live_rate_ewma")
            entry["live_rate_ewma"] = round(
                alive / scraped if prev is None
                else self.alpha * (alive / scraped) + (1 - self.alpha) * prev,
                4)
            if alive == 0:
                entry["zero_yield_streak"] = int(entry.get("zero_yield_streak") or 0) + 1
            else:
                entry["zero_yield_streak"] = 0
                entry["last_ok"] = now
        if error:
            entry["error"] = str(error)[:200]
        # auto-quarantine the WHOLE source when its yield collapses
        ewma = entry.get("live_rate_ewma")
        if ((ewma is not None and ewma < 0.01
             and int(entry.get("rounds") or 0) >= 3)
                or int(entry.get("zero_yield_streak") or 0) >= 3):
            entry["quarantined_until"] = now + self.quarantine_hours * 3600.0
            _log.warning("proxy source quarantined on yield: %s ewma=%s streak=%s",
                         name, ewma, entry.get("zero_yield_streak"))
        self._save()
        return dict(entry)

    def record_probes(self, name: str, count: int = 1) -> int:
        """Spend ``count`` of ``name``'s probe budget. Returns total spent."""
        entry = self._entry(name)
        entry["probes_spent"] = int(entry.get("probes_spent") or 0) + max(0, int(count))
        self._save()
        return entry["probes_spent"]

    # -- gates ---------------------------------------------------------------
    def probe_budget_for(self, name: str) -> int:
        """Probe cap for ``name``: catalog budget, 0%-yield → near-zero."""
        entry = SOURCE_BY_NAME.get(name)
        if entry is not None:
            return int(entry.get("probe_budget", DEFAULT_PROBE_BUDGET))
        return DEFAULT_PROBE_BUDGET

    def should_probe(self, name: str, *, count: int = 1,
                     now: float | None = None) -> bool:
        """True when ``name`` may spend ``count`` more probes now."""
        if self.quarantined(name, now=now):
            return False
        entry = self._data.get(name) or {}
        spent = int(entry.get("probes_spent") or 0)
        return spent + max(1, int(count)) <= self.probe_budget_for(name)

    def quarantined(self, name: str, *, now: float | None = None) -> bool:
        entry = self._data.get(name)
        if not entry:
            return False
        return float(entry.get("quarantined_until") or 0) > (now or time.time())

    def reset_budget(self, name: str | None = None) -> None:
        """New round: zero the probe spend (per-source budgets are per round)."""
        if name is None:
            for entry in self._data.values():
                entry["probes_spent"] = 0
        elif name in self._data:
            self._data[name]["probes_spent"] = 0
        self._save()

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        now = time.time()
        for name, entry in self._data.items():
            out[name] = {
                **dict(entry),
                "quarantined": float(entry.get("quarantined_until") or 0) > now,
                "probe_budget": self.probe_budget_for(name),
            }
        return out


# ── dedupe ─────────────────────────────────────────────────────────────────


def dedupe_proxies(proxies: list[Any]) -> tuple[list[Any], list[Any]]:
    """Dedupe by ``(host, port, scheme/protocol)`` BEFORE testing.

    Returns ``(unique, duplicates)``.  The first copy wins; the
    duplicate's source is appended to the winner's ``also_from`` and the
    winner keeps its original ``source`` — provenance survives dedupe,
    so per-source yield stays attributable.  Works on ``proxylab.Proxy``
    objects and on plain ``{host, port, scheme|protocol, source}`` dicts.
    """
    seen: dict[tuple[str, int, str], Any] = {}
    duplicates: list[Any] = []

    def _key(item: Any) -> tuple[str, int, str]:
        if isinstance(item, dict):
            host = str(item.get("host") or "").strip().lower()
            port = int(item.get("port") or 0)
            scheme = str(item.get("scheme") or item.get("protocol") or "http").strip().lower()
        else:
            host = str(getattr(item, "host", "") or "").strip().lower()
            port = int(getattr(item, "port", 0) or 0)
            scheme = str(getattr(item, "scheme", "http") or "http").strip().lower()
        return (host, port, scheme)

    def _source(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("source") or "")
        return str(getattr(item, "source", "") or "")

    def _note_also(winner: Any, source: str) -> None:
        if not source or _source(winner) == source:
            return
        if isinstance(winner, dict):
            also = winner.setdefault("also_from", [])
            if source not in also:
                also.append(source)
        else:
            also = getattr(winner, "also_from", None)
            if also is not None and source not in also:
                also.append(source)

    for item in proxies or []:
        key = _key(item)
        if key in seen:
            _note_also(seen[key], _source(item))
            duplicates.append(item)
            continue
        seen[key] = item
    return list(seen.values()), duplicates


# ── content integrity ────────────────────────────────────────────────────────


def sha256_hex(body: bytes) -> str:
    """SHA-256 hex digest of a fetched body."""
    return hashlib.sha256(body or b"").hexdigest()


def verify_content_integrity(
    get_body: Callable[[], bytes],
    expected_sha256: str,
    *,
    url: str = "http://example.com/",
) -> dict[str, Any]:
    """Content-integrity check alongside the egress-IP proof.

    ``get_body`` is a zero-arg callable that fetches ``url`` *through
    the proxy under test* and returns the raw bytes (the caller wires
    the routing — e.g. ``proxylab.ProxyTester._probe``).  The digest is
    compared against ``expected_sha256`` (the hash of the same page
    fetched direct, or of a known-stable page).  A mismatch means the
    proxy rewrote the content (ad injection, script injection, MITM) —
    the proxy is unusable even if its egress IP proved relay.

    Returns ``{ok, digest, expected, tampered, url}``; a fetch failure
    is ``ok=False`` with the error, never a silent pass.
    """
    try:
        body = get_body()
    except Exception as exc:  # noqa: BLE001 - fetch failure is a result
        return {"ok": False, "digest": "", "expected": expected_sha256,
                "tampered": False, "url": url, "error": str(exc)[:200]}
    digest = sha256_hex(body)
    tampered = digest != (expected_sha256 or "").strip().lower()
    return {"ok": not tampered, "digest": digest,
            "expected": expected_sha256, "tampered": tampered, "url": url,
            "error": "" if not tampered else "content hash mismatch — proxy rewrote the page"}


# ── staleness + refresh ─────────────────────────────────────────────────────


def check_staleness(entries: list[dict[str, Any]], *,
                    max_age_days: float = STALE_AFTER_DAYS,
                    now: float | None = None) -> dict[str, Any]:
    """Split ``entries`` into fresh / stale / never-OK sources.

    ``entries`` are ``{name, last_ok}`` with ``last_ok`` a unix timestamp
    (``0`` / missing = never had a successful fetch).  A source is stale
    when it has never succeeded or its last success is older than
    ``max_age_days``.  Pure function — no I/O, trivially testable.
    """
    now = now if now is not None else time.time()
    cutoff = now - float(max_age_days) * 86400.0
    fresh, stale, never = [], [], []
    for entry in entries or []:
        last_ok = float(entry.get("last_ok") or 0)
        if not last_ok:
            never.append(entry.get("name", ""))
        elif last_ok < cutoff:
            stale.append(entry.get("name", ""))
        else:
            fresh.append(entry.get("name", ""))
    return {"fresh": fresh, "stale": stale, "never_ok": never,
            "stale_count": len(stale) + len(never),
            "total": len(entries or [])}


def refresh_sources(registry: SourceRegistry, *,
                    tracker: SourceYieldTracker | None = None,
                    max_age_days: float = STALE_AFTER_DAYS) -> dict[str, Any]:
    """Merge the 2026 catalog into ``registry`` + re-check staleness.

    * New catalog entries with a verified URL join the registry as
      ``discovered`` sources (built-ins are already there — skipped).
    * Unverified entries (no URL) are reported, never added.
    * Staleness is re-checked from the registry's own health state:
      sources never-OK or not-OK for ``max_age_days`` are listed so a
      caller can quarantine or drop them — the catalog never silently
      keeps serving dead feeds.
    """
    present = {name for name, _url, _kind in registry.sources()}
    added, skipped_unverified, already = [], [], []
    for entry in SOURCES_2026:
        name = entry["name"]
        if name in present or name in _builtin_names():
            already.append(name)
            continue
        if not entry.get("url"):
            # No URL: a verified entry is a *reference* to built-in
            # endpoints (already present by definition); an unverified
            # one is a gap — visible, never added, never fetched.
            if entry.get("verified"):
                already.append(name)
            else:
                skipped_unverified.append(name)
            continue
        if not entry.get("verified"):
            skipped_unverified.append(name)
            continue
        registry.add_discovered(
            name, entry["url"], entry["kind"],
            seed="proxy_sources.SOURCES_2026", found=0)
        added.append(name)
    # staleness from the registry's health view: health()["last_ok"] is
    # seconds-since-success (None = never OK)
    health = registry.health()
    now = time.time()
    stale_entries = []
    for row in health:
        last_ok_age = row.get("last_ok")
        stale_entries.append({
            "name": row["name"],
            "last_ok": (0.0 if last_ok_age is None
                        else now - float(last_ok_age)),
        })
    staleness = check_staleness(stale_entries, max_age_days=max_age_days, now=now)
    quarantined = []
    if tracker is not None:
        for name in staleness["stale"] + staleness["never_ok"]:
            if tracker.quarantined(name):
                quarantined.append(name)
    return {"added": added, "already_present": already,
            "skipped_unverified": skipped_unverified,
            "staleness": staleness,
            "yield_quarantined": quarantined,
            "registry_stats": registry.stats()}


# ── Proxifly POST feed ───────────────────────────────────────────────────────


def fetch_proxifly(*, protocol: str = "http", quantity: int = 1,
                   format: str = "text", api_key: str = "",
                   post: Callable[..., bytes] | None = None,
                   timeout: float = 15.0) -> list[Any]:
    """Fetch proxies from the Proxifly POST API.

    ``POST https://api.proxifly.dev/get-proxy`` with a form body
    (``format``, ``protocol``, ``quantity``).  The scraper's GET fetcher
    cannot speak this feed, so it lives here.  ``quantity`` is clamped
    to Proxifly's 1–20 range and callers should keep it at the
    near-zero probe budget — this feed measured 0% live.  ``post`` is
    an injectable ``post(url, data, headers, timeout) -> bytes`` for
    tests; the default uses stdlib urllib.
    """
    from .proxylab import ProxyScraper

    quantity = max(1, min(20, int(quantity)))
    protocol = (protocol or "http").strip().lower()
    fmt = (format or "text").strip().lower()
    payload = {"format": fmt, "protocol": protocol, "quantity": quantity}
    if api_key:
        payload["apiKey"] = api_key
    url = SOURCE_BY_NAME["proxifly-post"]["url"]
    if post is not None:
        raw = post(url, payload, {"Content-Type": "application/x-www-form-urlencoded"},
                   timeout)
    else:
        data = urllib.parse.urlencode(payload).encode("ascii")
        request = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "nomorals-proxy-lab/1.0"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read()
    text = raw.decode("utf-8", "replace")
    if fmt == "json":
        proxies = ProxyScraper.parse_json(text)
    else:
        proxies = ProxyScraper.parse_list(text, protocol)
    for proxy in proxies:
        proxy.source = "proxifly-post"
    return proxies


def fetch_proxifly_text(*args: Any, **kwargs: Any) -> list[str]:
    """Proxifly POST feed as ``protocol://host:port`` URL strings."""
    kwargs.setdefault("format", "text")
    return [proxy.url for proxy in fetch_proxifly(*args, **kwargs)]
