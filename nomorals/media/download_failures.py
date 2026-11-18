"""Durable per-URL / per-source download-failure memory (lane 26).

The owner's report (2026-11-17): /play kept hammering the same dead
SoundCloud link — the same signed URL, retried minutes apart — instead
of moving to the next source. Two holes: the sweep threw away
per-source copies of the same song (dedupe by title+artist), and
NOTHING remembered a failure — no per-URL record, no per-source
record — so a fresh attempt re-picked the same dead link and re-ran
the same chain.

This module is the memory. It reuses the durable failure organ
(:mod:`nomorals.core.failure_store` — atomic JSON, NM_HOME-resolved
home, the same organ :mod:`nomorals.media.failures` binds): one
namespace, ``"media-download-failures"``, whose entries are translated
at the boundary — the store's shape is ``{ts, site, exc_type,
message}``; here ``site`` is the failed URL, ``exc_type`` is the
failure kind, ``message`` carries ``source=<source> scope=<scope>``.
No new persistence code, no L1 change.

Two scopes:
* ``"download"`` — a candidate URL failed to download (kind + time,
  TTL 6h). The sweep demotes dead URLs to the end; the fall-through
  skips them outright and names the skip in the journal trail.
* ``"search"`` — a source's search fetch failed (timeout / 403 / ...,
  TTL 10 min). The sweep reads it back as evidence so a failed search
  reports ``"failed: <kind>"`` instead of ``"no results"``.

Kind taxonomy (single-sourced here — the resolver, the scrape
sources and the fall-through all classify through it):
``timeout`` / ``http_403`` / ``http_404`` / ``http_5xx`` /
``site_error`` / ``download_error`` / ``wrong_file`` / ``no_media``.
``SOURCE_DEAD_KINDS`` are the classes that condemn a whole source for
one song inside a fall-through attempt: a 403, a timeout, a 5xx or a
site error on one copy means the source is hostile right now, so its
other copies for the same song are skipped too. A generic download
error or a wrong file condemns only its own URL.

Stdlib only besides the L1 store. Every public function never raises
— a memory that can crash its callers would defeat the purpose.
"""

from __future__ import annotations

import re
import socket
import threading
import time
import urllib.error
import urllib.parse

from nomorals.core import failure_store as _failure_store

__all__ = [
    "SOURCE_DEAD_KINDS",
    "DEFAULT_URL_TTL_S",
    "SEARCH_FAILURE_TTL_S",
    "classify_failure",
    "normalize_url",
    "record_url_failure",
    "record_search_failure",
    "is_url_dead",
    "dead_kind_for_url",
    "last_search_failure",
    "clear_failures",
]

#: A download-failed URL stays dead for six hours: long enough that a
#: fresh /play never re-picks the link that just died, short enough
#: that a transient outage recovers on its own.
DEFAULT_URL_TTL_S = 6 * 3600
#: A search failure is evidence for the sweep that just ran it (and
#: near-term retries), not a standing verdict — the registry already
#: keeps the long-term per-source stats.
SEARCH_FAILURE_TTL_S = 600

#: Failure kinds that condemn a whole source for one song inside a
#: fall-through attempt (see the module docstring).
SOURCE_DEAD_KINDS = frozenset(
    {"timeout", "http_403", "http_5xx", "site_error"})

#: Ring size, mirroring media/failures.py — enough to see a failing
#: URL's history, small enough to never grow.
_MAX_ENTRIES = 200

_STORE_NAMESPACE = "media-download-failures"

_store = _failure_store.Store(_STORE_NAMESPACE)
_ledger: list[dict] = []
_lock = threading.Lock()
_loaded = False
# The data home the in-memory ledger was loaded from. The store
# resolves its home (NM_HOME) on every call and tests redirect
# NM_HOME per test — without this, a load-once ledger would keep
# answering from the first home it ever saw.
_loaded_home: str | None = None
_total = 0


# ── failure-kind taxonomy ───────────────────────────────────────────

def classify_failure(exc_or_reason: object) -> str:
    """One failure kind for an exception or a download-failure reason.

    Never raises; anything unrecognised is ``"download_error"``.
    """
    try:
        if isinstance(exc_or_reason, BaseException):
            return _classify_exception(exc_or_reason)
        text = str(exc_or_reason or "").lower()
        if not text:
            return "download_error"
        if "timed out" in text or "timeout" in text \
                or "timed-out" in text:
            return "timeout"
        if "403" in text or "forbidden" in text:
            return "http_403"
        if "404" in text or "not found" in text:
            return "http_404"
        if any(code in text for code in
               ("500", "502", "503", "504", "bad gateway",
                "service unavailable", "server error")):
            return "http_5xx"
        if "site error" in text or "site_error" in text:
            return "site_error"
        if "no downloadable" in text or "no media" in text:
            return "no_media"
        return "download_error"
    except Exception:  # noqa: BLE001 — classification never raises
        return "download_error"


def _classify_exception(exc: BaseException) -> str:
    """Kind for a real exception (timeouts and HTTP errors first)."""
    if isinstance(exc, urllib.error.HTTPError):
        code = int(getattr(exc, "code", 0) or 0)
        if code == 403:
            return "http_403"
        if code == 404:
            return "http_404"
        if 500 <= code <= 599:
            return "http_5xx"
        if 400 <= code <= 499:
            return "download_error"
        return "site_error"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "site_error"
    if isinstance(exc, (ConnectionError, OSError)):
        return "site_error"
    return "download_error"


def normalize_url(url: object) -> str:
    """Canonical form of a URL for dedupe + memory keys.

    Scheme and host lowercased, fragment dropped, a trailing slash
    stripped from the path; the query is kept (signed URLs differ by
    it — they are distinct download targets). A bare path (a local
    file) is returned as-is apart from the trailing slash: paths are
    case-sensitive. Never raises.
    """
    try:
        u = str(url or "").strip()
        if not u:
            return ""
        if "://" not in u:
            return u.rstrip("/")
        parts = urllib.parse.urlsplit(u)
        path = parts.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit((
            parts.scheme.lower(), parts.netloc.lower(),
            path, parts.query, ""))
    except Exception:  # noqa: BLE001 — normalisation never raises
        return str(url or "").strip()


# ── durable ledger (failure_store organ, translated at the boundary) ─

def _to_store_row(entry: dict) -> dict:
    """One memory entry as a failure_store row."""
    return {
        "ts": float(entry["ts"]),
        "site": str(entry["url"]),
        "exc_type": str(entry["kind"]),
        "message": "source={} scope={}".format(
            str(entry.get("source") or ""),
            str(entry.get("scope") or "download")),
    }


_MESSAGE_RE = re.compile(r"^source=(.*) scope=(\S+)$")


def _from_store_row(row: dict) -> dict | None:
    """One failure_store row back as a memory entry, or None."""
    try:
        url = str(row["site"])
        kind = str(row["exc_type"])
        ts = float(row["ts"])
        source, scope = "", "download"
        match = _MESSAGE_RE.match(str(row.get("message") or ""))
        if match:
            source, scope = match.group(1), match.group(2)
        if scope not in ("download", "search"):
            scope = "download"
        return {"url": url, "kind": kind, "source": source,
                "ts": ts, "scope": scope}
    except Exception:  # noqa: BLE001 — a bad row is skipped, not fatal
        return None


def _ensure_loaded() -> None:
    """Load persisted entries; reload when the data home moved.

    Caller holds ``_lock``. The store resolves its home (NM_HOME)
    per call and tests redirect NM_HOME per test — a plain
    load-once ledger would keep answering later tests (or a
    re-homed process) from the first home's entries, and would
    write new records through to the wrong home's file. In
    production the home never moves, so this reloads exactly
    never there; in tests it makes per-test NM_HOME isolation
    actually isolate.
    """
    global _loaded, _loaded_home, _total
    try:
        home = str(_failure_store.home_dir())
    except Exception:  # noqa: BLE001 — home resolution never breaks us
        home = _loaded_home
    if _loaded and _loaded_home == home:
        return
    _loaded = True
    _loaded_home = home
    _ledger.clear()
    entries, total = _store.load()
    for row in entries:
        entry = _from_store_row(row)
        if entry is not None:
            _ledger.append(entry)
    _prune_locked()
    _total = max(_total, total, len(_ledger))


def _prune_locked() -> None:
    """Drop entries older than the longest TTL. Caller holds ``_lock``."""
    cutoff = time.time() - DEFAULT_URL_TTL_S
    _ledger[:] = [e for e in _ledger if float(e["ts"]) >= cutoff]


def _save_locked() -> None:
    """Write the ring through. Caller holds ``_lock``."""
    global _total
    _prune_locked()
    kept = _ledger[-_MAX_ENTRIES:]
    _store.save([_to_store_row(e) for e in kept], _total)


def _record_locked(url: str, kind: str, source: str, scope: str) -> None:
    """Append one entry (already under ``_lock``)."""
    global _total
    _ensure_loaded()
    _ledger.append({"url": normalize_url(url), "kind": kind,
                    "source": source or "", "ts": time.time(),
                    "scope": scope})
    _total += 1
    _save_locked()


# ── public API ──────────────────────────────────────────────────────

def record_url_failure(url: object, kind: str, source: str = "") -> None:
    """Remember that *url* failed to download as *kind*. Never raises."""
    try:
        u = normalize_url(url)
        if not u or not kind:
            return
        with _lock:
            _record_locked(u, str(kind), source or "", "download")
    except Exception:  # noqa: BLE001 — the memory never breaks a play
        pass


def record_search_failure(source: str, kind: str, url: object = "") -> None:
    """Remember that *source*'s search fetch failed as *kind*.

    The fetched URL rides along for diagnosis; the lookup is by
    source name. Never raises.
    """
    try:
        if not source or not kind:
            return
        with _lock:
            _record_locked(str(url or ""), str(kind),
                           str(source), "search")
    except Exception:  # noqa: BLE001
        pass


def is_url_dead(url: object, within_s: float = DEFAULT_URL_TTL_S) -> bool:
    """True when *url* recorded a download failure within ``within_s``."""
    return dead_kind_for_url(url, within_s=within_s) is not None


def dead_kind_for_url(url: object,
                      within_s: float = DEFAULT_URL_TTL_S) -> str | None:
    """The recorded download-failure kind for *url*, or None.

    The newest record inside the window wins. Never raises.
    """
    try:
        key = normalize_url(url)
        if not key:
            return None
        cutoff = time.time() - float(within_s)
        with _lock:
            _ensure_loaded()
            for entry in reversed(_ledger):
                if entry["scope"] != "download":
                    continue
                if entry["url"] == key and float(entry["ts"]) >= cutoff:
                    return str(entry["kind"])
        return None
    except Exception:  # noqa: BLE001
        return None


def last_search_failure(source: str,
                        within_s: float = SEARCH_FAILURE_TTL_S
                        ) -> str | None:
    """The newest search-failure kind for *source* in the window.

    ``None`` when the source's search never failed (or not recently)
    — a clean search returning [] is NOT a failure. Never raises.
    """
    try:
        name = (source or "").strip()
        if not name:
            return None
        cutoff = time.time() - float(within_s)
        with _lock:
            _ensure_loaded()
            for entry in reversed(_ledger):
                if entry["scope"] != "search":
                    continue
                if entry["source"] == name \
                        and float(entry["ts"]) >= cutoff:
                    return str(entry["kind"])
        return None
    except Exception:  # noqa: BLE001
        return None


def clear_failures() -> None:
    """Empty the memory and its store file (tests, mostly)."""
    global _total, _loaded, _loaded_home
    try:
        home = str(_failure_store.home_dir())
    except Exception:  # noqa: BLE001 — the clear itself stands
        home = _loaded_home
    with _lock:
        _ledger.clear()
        _total = 0
        _loaded = True
        _loaded_home = home
        _store.clear()
