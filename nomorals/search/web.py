"""Web-search backends for federated search: the missing half of the index.

The federated layer used to know only local subsystems (memory, books,
docs, code, timeline, wisdom). This module adds the live web behind the
same :class:`SourceAdapter` interface — six backends, ordered by the
standing rule "best FREE first, popular != best":

1. ``web_searxng`` — **primary free backend.** SearXNG is a self-hosted /
   public metasearch engine aggregating 70+ upstream engines with one
   keyless JSON API (``GET {instance}/search?q=..&format=json``). No API
   key, no quota, no account — the only free option with Google/Bing-tier
   coverage. Instance URL(s) via ``NM_SEARXNG_URL`` (comma-separated for
   failover). Honest gotcha, kept in the probe note: most *public*
   instances disable ``format=json`` (HTTP 403) and rate-limit
   aggressively — self-hosting (one Docker container) is the reliable
   path; the adapter fails over across every configured instance.
2. ``web_ddgs`` — **keyless fallback.** The ``ddgs`` package (formerly
   ``duckduckgo_search``) drives DuckDuckGo's endpoints plus a
   multi-engine backend ladder (bing, brave, duckduckgo, google, mojeek,
   startpage, yandex, yahoo, wikipedia) with zero config and no API key.
   Optional dependency: if ``ddgs`` is not installed the source reports
   itself unavailable with the pip command instead of crashing.
3. ``web_tavily`` — best free *API-key* backend: 1,000 free credits/month,
   no credit card required, agent-oriented JSON (title/url/content/score).
   Key via ``NM_TAVILY_API_KEY`` (or ``TAVILY_API_KEY``).
4. ``web_serper`` — 2,500 free queries, one-time, no card; Google SERP
   JSON. Key via ``NM_SERPER_API_KEY`` (or ``SERPER_API_KEY``).
5. ``web_exa`` — $10/month free credits (~2,500 instant searches);
   neural/semantic search with highlights. Key via ``NM_EXA_API_KEY``
   (or ``EXA_API_KEY``).
6. ``web_brave`` — Brave Search API on Brave's own 30B+ page index.
   NOTE (verified 2026-02): the free tier was withdrawn for new accounts —
   new users get $5/month in metered credits *with a card on file*, so
   this is a paid/legacy-key backend, not a free primary. Key via
   ``NM_BRAVE_SEARCH_API_KEY`` (or ``BRAVE_SEARCH_API_KEY``).

Deliberately NOT included (researched 2026-10, all rejected on facts):
Mojeek (API paid-only, HTML scraping hits a CAPTCHA wall), Marginalia
(public JSON API down), Google Custom Search (closed to new customers,
API off 2027-01-01), Bing Web Search API (retired 2025-08-11).

Transport uses only the stdlib (``urllib``): the core package keeps its
"zero mandatory third-party dependencies" promise. Every backend maps
into :class:`SearchResult` with ``type="web"``, the result URL in
provenance, and the backend's native score stashed at
``provenance["backend_score"]``. Hits are BM25 re-ranked
(:mod:`nomorals.search.rerank`) by default — disable per-query cost-free
with ``NM_WEB_RERANK=0``.

Pass #90 (open-sources law): the fixed-host backends are sources in
the ONE media source registry (kinds ``search``/``web-search`` — see
:data:`WEB_BACKEND_DOMAINS`), and every attempt on this path records
its outcome there: served results credit the backend, an empty
result set is weak no-result evidence (never a breaker trip), an
operational failure records with an error class, and a 429 / quota
signal records as rate-limited — its own class, a hard stop, never
retried around. An auth failure (401/403 invalid key) records
NOTHING: a wrong key is configuration, not evidence — and a backend
dark for want of a key never reaches ``search`` at all (probe()
skips it in the federated layer). SearXNG has no fixed host: its
outcomes are recorded per configured instance, against a descriptor
registered for that instance's host. ``web_ddgs`` has no single host
at all (it aggregates other engines), so it gets NO descriptor and
no credit — its component engines are recorded on the
``tools/web.py`` chain, where they are hosts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..media.source_registry import WEB_BACKEND_DOMAINS
from .base import SourceAdapter
from .errors import SearchError
from .model import SearchResult
from .rerank import bm25_rerank

__all__ = [
    "WebBackendError",
    "WebSearchSource",
    "SearXNGWebSource",
    "DdgsWebSource",
    "TavilyWebSource",
    "SerperWebSource",
    "ExaWebSource",
    "BraveWebSource",
    "WEB_SPECS",
    "WEB_BACKEND_DOMAINS",
    "web_source_names",
    "web_backends_configured",
    "web_registry_for_context",
    "registry_desc_for_backend",
    "ordered_web_backend_names",
]

#: Identify ourselves to shared infrastructure (SearXNG instances,
#: DDG endpoints) instead of spoofing a browser.
_USER_AGENT = "DevonSearch/1.0 (nomorals federated web search; +https://github.com/Oluwacutyp/No-morals-ai)"


class WebBackendError(SearchError):
    """A web-search backend failed (transport, auth, quota, bad payload).

    ``federated_search`` records this per-source in
    ``sources_skipped`` and keeps going — one bad source never kills
    the whole search.
    """


def _env(*names: str, default: str = "") -> str:
    """First set env var wins; ``NM_``-prefixed names take precedence."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _timeout(default: float = 10.0) -> float:
    raw = _env("NM_WEB_TIMEOUT", "WEB_TIMEOUT")
    try:
        return max(1.0, float(raw)) if raw else default
    except ValueError:
        return default


def _rerank_enabled() -> bool:
    return _env("NM_WEB_RERANK", "WEB_RERANK", default="1") != "0"


def _http(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, bytes]:
    """One HTTP round-trip. Returns ``(status, body)``.

    HTTP error statuses (403/429/...) are *returned*, not raised, so the
    caller can map them to precise backend errors. Transport failures
    (DNS, refused, TLS, timeout) raise :class:`WebBackendError`.
    """
    full = url
    if params:
        query = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}
        )
        full = f"{url}{'&' if '?' in url else '?'}{query}"
    data = None
    req_headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(full, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 - best-effort body on error paths
            body = b""
        return exc.code, body
    except Exception as exc:  # noqa: BLE001 - transport failure, mapped below
        raise WebBackendError(f"HTTP {method} {url} failed: {exc}") from exc


def _json(status: int, body: bytes, backend: str) -> Any:
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise WebBackendError(
            f"{backend}: non-JSON response (HTTP {status}): "
            f"{body[:120].decode('utf-8', errors='replace')!r}"
        ) from exc


def _parse_ts(value: Any) -> float | None:
    """Best-effort ISO-8601 / epoch → epoch seconds; never raises."""
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        return number
    try:
        iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ── the web-search class in the ONE source registry (Pass #90) ───────
#
# :data:`WEB_BACKEND_DOMAINS` — the host each federated backend
# actually fetches from, i.e. the backend's source in the ONE source
# registry — is defined in ``media/source_registry.py`` (Pass #95:
# the table is shared with the registry's own status views, so it
# lives at the registry's layer and is imported back here, downward;
# the name, contents and order are unchanged for every consumer of
# this module, ``search.federated`` included).


def web_registry_for_context(context: Any = None,
                             injected: Any = None) -> Any:
    """The source registry web-search outcomes land in, or None.

    Mirrors the #88/#89 resolution exactly: an injected registry
    wins; otherwise the shared ``DEFAULT_REGISTRY`` bound to the
    context's KV store — and only when the context carries a
    database, so db-less callers keep the pre-#90 behaviour exactly.
    Never raises."""
    if injected is not None:
        return injected
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return None
        from ..media.source_registry import DEFAULT_REGISTRY
        reg = DEFAULT_REGISTRY
        try:
            from ..storage.kv import KVStore
            reg.bind_store(KVStore(db))
        except Exception:  # noqa: BLE001 - persistence is an
            # enhancement; its absence never sinks a search
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _ensure_searx_instance(reg: Any, instance_url: str) -> Any:
    """The registry descriptor for one configured SearXNG instance
    host, registering it (origin ``registered``) when new: an
    owner-configured instance is a configured source, not a
    discovery — it is trusted exactly as far as the owner's config
    trusts it, carries the instance's real search template (the
    health loop probes it like any keyless engine), and persists
    through the registry like any registered source.  Never
    raises."""
    try:
        host = urllib.parse.urlparse(instance_url or "").hostname or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return None
        desc = reg.for_domain(host)
        if desc is not None:
            return desc if "web-search" in desc.kinds else None
        from ..media.source_registry import (
            Licence, SourceDescriptor, TrustTier)

        base = str(instance_url or "").strip().rstrip("/")
        desc = SourceDescriptor(
            nation="GLOBAL", language="en", domain=host,
            name=f"SearXNG instance ({host})",
            kinds=("search", "web-search"), licence=Licence.UNKNOWN,
            trust_tier=TrustTier.UNVERIFIED,
            notes="owner-configured SearXNG instance (search/web.py)",
            search_url_template=f"{base}/search?q={{query}}&format=json")
        if reg.register(desc):
            reg._persist()  # a registered source survives restarts too
            return desc
        return reg.for_domain(host)
    except Exception:  # noqa: BLE001
        return None


def registry_desc_for_backend(reg: Any, source: Any) -> Any:
    """The registry descriptor that should be credited for one
    backend's attempt, or None when the backend has no creditable
    host (ddgs) or is unconfigured (SearXNG with no instance).
    Never raises."""
    try:
        if reg is None or source is None:
            return None
        name = str(getattr(source, "name", "") or "")
        if name == "web_searxng":
            instances = source._instances() \
                if hasattr(source, "_instances") else []
            if not instances:
                return None
            return _ensure_searx_instance(reg, instances[0])
        domain = WEB_BACKEND_DOMAINS.get(name)
        if not domain:
            return None
        return reg.for_domain(domain)
    except Exception:  # noqa: BLE001
        return None


def _backend_error_evidence(exc: BaseException) -> tuple[str, str]:
    """Classify one backend exception into registry evidence:
    ``(kind, error_class)`` where kind is ``rate_limited`` |
    ``failure`` | ``none``.

    * 429 / rate-limit / quota-or-credits-exhausted signals →
      ``rate_limited`` (its own class, Pass #90 — the attempt is
      recorded once and never retried);
    * 401/403 invalid-key signals → ``none``: a wrong or missing
      key is a fact about this install's configuration, not about
      the engine's health;
    * a non-JSON payload → failure, ``malformed_response``;
    * a timeout → failure, ``timeout``; an ``HTTP <code>`` signal →
      failure, ``http_<code>``; anything else → failure,
      ``search_failed`` (transport failures land here — the stdlib
      transport wraps them without a status).
    """
    msg = str(exc or "").lower()
    if "429" in msg or "rate limit" in msg or "rate-limited" in msg \
            or "432" in msg or "quota" in msg \
            or "credits exhausted" in msg or "credit quota" in msg:
        return "rate_limited", "rate_limited"
    if "401" in msg or "403" in msg or "invalid api key" in msg \
            or "invalid subscription" in msg:
        return "none", ""
    if "non-json" in msg:
        return "failure", "malformed_response"
    if "timed out" in msg or "timeout" in msg:
        return "failure", "timeout"
    match = re.search(r"http (\d{3})", msg)
    if match:
        return "failure", f"http_{match.group(1)}"
    return "failure", "search_failed"


def ordered_web_backend_names(reg: Any, names: list[str]) -> list[str]:
    """Outcome-aware order for the federated web backends (Pass
    #90), through the registry's score/health semantics, mirroring
    the #88/#89 ordering:

    * NO learned state anywhere among the named backends → the
      list is returned untouched (day-one pin: ``WEB_SPECS`` order,
      byte-for-byte);
    * otherwise active backends sort by registry score (stable)
      and stood-down / disabled / dead backends are demoted to the
      end — demoted, never dropped.  Backends with no descriptor
      (SearXNG without an instance, ddgs) keep their relative
      place among the active set and earn no score.
    Never raises."""
    try:
        if reg is None:
            return list(names)
        domains = WEB_BACKEND_DOMAINS
        items: list[tuple[str, Any]] = []
        evidence = False
        for name in names:
            domain = domains.get(name)
            desc = reg.for_domain(domain) if domain else None
            items.append((name, desc))
            if desc is None:
                continue
            st = reg.stats_for(desc)
            if st["ok"] or st["fail"] or st["no_result"] \
                    or st.get("blocked") or st.get("rate_limited") \
                    or not reg.is_active(desc):
                evidence = True
        if not evidence:
            return list(names)
        active = [it for it in items
                  if it[1] is None or reg.is_active(it[1])]
        inactive = [it for it in items
                    if it[1] is not None and not reg.is_active(it[1])]
        active.sort(key=lambda it: -reg.score(it[1])
                    if it[1] is not None else 0.0)
        return [n for n, _d in active] + [n for n, _d in inactive]
    except Exception:  # noqa: BLE001
        return list(names)


# ── base ───────────────────────────────────────────────────────────────

class WebSearchSource(SourceAdapter):
    """One live web-search backend behind the federated source interface.

    Subclasses implement :meth:`_fetch` returning raw hit dicts with at
    least ``title``/``url``/``snippet`` keys (optional: ``score``,
    ``published``, ``engine``). ``search`` maps them into
    :class:`SearchResult`, then BM25 re-ranks unless disabled.
    """

    name = "web"
    result_type = "web"
    description = "live web search"

    def __init__(self, context: Any = None, *,
                 source_registry: Any = None) -> None:
        # Web backends are configured via environment, not the context —
        # the parameter keeps build_adapters() uniform across sources.
        self._context = context
        # Source registry (Pass #90): the ONE registry this backend's
        # outcomes land in.  Tests may inject; otherwise resolved
        # lazily from the context (see _registry) — and only when the
        # context carries a database, so db-less callers keep the
        # pre-#90 behaviour exactly.
        self._source_registry_injected = source_registry
        self._source_registry_cache: Any = None
        self._source_registry_resolved = False

    # -- source registry (Pass #90) ------------------------------------
    def _registry(self) -> Any:
        """The source registry for this backend's outcomes, or None
        when it has none to speak to.  Never raises."""
        if self._source_registry_resolved:
            return self._source_registry_cache
        self._source_registry_resolved = True
        self._source_registry_cache = web_registry_for_context(
            self._context, injected=self._source_registry_injected)
        return self._source_registry_cache

    def _record_search_outcome(self, raw_hits: list, latency_s: float) -> None:
        """Credit one served attempt: results → served; an empty
        result set → weak no-result evidence (#88 precedent — an
        obscure query must never stand an engine down).
        Best-effort; never raises."""
        try:
            reg = self._registry()
            if reg is None:
                return
            desc = registry_desc_for_backend(reg, self)
            if desc is None:
                return
            if raw_hits:
                reg.record_outcome(desc, True, latency_s=latency_s)
            else:
                reg.record_no_result(desc)
        except Exception:  # noqa: BLE001 - bookkeeping never breaks search
            pass

    def _record_search_failure(self, exc: BaseException,
                               latency_s: float) -> None:
        """Blame one failed attempt, classified by
        :func:`_backend_error_evidence`.  Best-effort; never raises."""
        try:
            kind, error_class = _backend_error_evidence(exc)
            if kind == "none":
                return
            reg = self._registry()
            if reg is None:
                return
            desc = registry_desc_for_backend(reg, self)
            if desc is None:
                return
            if kind == "rate_limited":
                # recorded once — a 429 is a hard stop for this
                # backend, never a cue to retry around it
                reg.record_rate_limited(desc, latency_s=latency_s)
            else:
                reg.record_outcome(desc, False, latency_s=latency_s,
                                   error_class=error_class)
        except Exception:  # noqa: BLE001
            pass

    # -- configuration -------------------------------------------------
    def _api_key(self) -> str:
        return ""

    def _key_names(self) -> tuple[str, ...]:
        return ()

    def _missing_key_note(self) -> str:
        names = ", ".join(self._key_names())
        return f"no API key configured (set {names})"

    # -- plumbing ------------------------------------------------------
    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def probe(self) -> str | None:
        if self._key_names() and not self._api_key():
            return self._missing_key_note()
        return None

    def _to_result(
        self, query: str, rank: int, raw: dict[str, Any]
    ) -> SearchResult:
        url = str(raw.get("url") or raw.get("link") or raw.get("href") or "")
        title = str(raw.get("title") or url or "(untitled)")
        snippet = str(raw.get("snippet") or raw.get("content") or raw.get("body") or "")
        native = raw.get("score")
        try:
            native_score = float(native) if native is not None else 1.0 / (rank + 1)
        except (TypeError, ValueError):
            native_score = 1.0 / (rank + 1)
        url_hash = hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        provenance: dict[str, Any] = {
            "url": url,
            "provider": self.name,
            "backend_score": native_score,
        }
        if raw.get("engine"):
            provenance["engine"] = raw["engine"]
        if raw.get("source"):
            provenance["result_source"] = raw["source"]
        return SearchResult(
            query=query,
            title=title[:300],
            snippet=snippet[:1200],
            source=self.name,
            type=self.result_type,
            raw_score=native_score,
            provenance=provenance,
            timestamp=_parse_ts(raw.get("published")),
            source_id=f"{self.name}:{url_hash}",
        )

    def search(
        self, query: str, *, limit: int, since: float | None = None,
        before: float | None = None,
    ) -> list[SearchResult]:
        # Date windows are applied by federated_search post-hoc (undated
        # web hits are kept); backends don't get lossy time_range mapping.
        # Pass #90: the attempt's outcome lands in the source registry.
        # SearXNG is the one exception at this level — its _fetch
        # records per configured instance (its real unit of credit),
        # so a search-level record would double-count it.
        t0 = time.time()
        try:
            raw_hits = self._fetch(query, limit)
        except WebBackendError as exc:
            if self.name != "web_searxng":
                self._record_search_failure(exc, time.time() - t0)
            raise
        except Exception as exc:  # noqa: BLE001 - wrapped with backend name
            wrapped = WebBackendError(
                f"{self.name}: unexpected error: {exc}")
            if self.name != "web_searxng":
                self._record_search_failure(wrapped, time.time() - t0)
            raise wrapped from exc
        if self.name != "web_searxng":
            self._record_search_outcome(raw_hits, time.time() - t0)
        results = [
            self._to_result(query, i, raw)
            for i, raw in enumerate(raw_hits[:limit])
            if raw.get("url") or raw.get("link") or raw.get("href") or raw.get("title")
        ]
        if _rerank_enabled():
            results = bm25_rerank(results, query)
        return results


# ── 1. SearXNG (primary free backend) ──────────────────────────────────

class SearXNGWebSource(WebSearchSource):
    """SearXNG metasearch: keyless JSON over 70+ upstream engines.

    ``NM_SEARXNG_URL`` — one URL or comma-separated list (failover in
    order), e.g. ``https://searx.example.com,http://localhost:8888``.

    Per-query vertical routing (the part every programmatic SearXNG
    integration uses, and we were missing):

    - ``NM_SEARXNG_CATEGORY`` — ``general`` (default), ``news``,
      ``science``, ``it``, ``videos``, ``images``, ``map``, ``music``,
      ``files``, ``"social media"``;
    - ``NM_SEARXNG_TIME_RANGE`` — ``day``, ``week``, ``month``, ``year``
      (recency filter; most useful with the ``news`` category);
    - ``NM_SEARXNG_ENGINES`` — comma-separated engine pinning, e.g.
      ``google,bing``;
    - ``NM_SEARXNG_LANGUAGE`` — e.g. ``en``, ``fr`` (default ``auto``).
    """

    name = "web_searxng"
    description = (
        "SearXNG metasearch (keyless, 70+ upstream engines, "
        "NM_SEARXNG_URL; self-hosted recommended)"
    )

    #: valid categories per the SearXNG query API
    CATEGORIES = frozenset({
        "general", "images", "videos", "news", "map", "music",
        "science", "social media", "files", "it",
    })
    #: valid time ranges per the SearXNG query API
    TIME_RANGES = frozenset({"day", "week", "month", "year"})

    def _instances(self) -> list[str]:
        raw = _env("NM_SEARXNG_URL", "SEARXNG_URL")
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]

    def _category(self) -> str:
        cat = _env("NM_SEARXNG_CATEGORY", "SEARXNG_CATEGORY",
                   default="general").strip().lower()
        return cat if cat in self.CATEGORIES else "general"

    def _time_range(self) -> str:
        tr = _env("NM_SEARXNG_TIME_RANGE", "SEARXNG_TIME_RANGE",
                  default="").strip().lower()
        return tr if tr in self.TIME_RANGES else ""

    def _engines(self) -> str:
        return _env("NM_SEARXNG_ENGINES", "SEARXNG_ENGINES", default="").strip()

    def _language(self) -> str:
        return _env("NM_SEARXNG_LANGUAGE", "SEARXNG_LANGUAGE",
                    default="auto").strip() or "auto"

    def probe(self) -> str | None:
        if not self._instances():
            return (
                "no SearXNG instance configured "
                "(set NM_SEARXNG_URL to your instance URL, e.g. "
                "http://localhost:8888 for a self-hosted instance)"
            )
        return None

    def _record_instance(self, instance: str, kind: str,
                         latency_s: float = 0.0,
                         error_class: str = "") -> None:
        """Record one instance attempt against that instance's own
        registry descriptor (Pass #90 — the instance is this
        backend's real unit of credit).  Best-effort; never raises."""
        try:
            reg = self._registry()
            if reg is None:
                return
            desc = _ensure_searx_instance(reg, instance)
            if desc is None:
                return
            if kind == "served":
                reg.record_outcome(desc, True, latency_s=latency_s)
            elif kind == "no_result":
                reg.record_no_result(desc)
            elif kind == "blocked":
                reg.record_blocked(desc, latency_s=latency_s)
            elif kind == "rate_limited":
                reg.record_rate_limited(desc, latency_s=latency_s)
            else:
                reg.record_outcome(desc, False, latency_s=latency_s,
                                   error_class=error_class or "site_error")
        except Exception:  # noqa: BLE001
            pass

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        timeout = _timeout(12.0)
        params: dict[str, Any] = {
            "q": query, "format": "json", "language": self._language(),
            "safesearch": "1", "pageno": "1",
            "categories": self._category(),
        }
        if self._time_range():
            params["time_range"] = self._time_range()
        if self._engines():
            params["engines"] = self._engines()
        errors: list[str] = []
        for instance in self._instances():
            t0 = time.time()
            try:
                status, body = _http(
                    "GET", f"{instance}/search",
                    params=params,
                    timeout=timeout,
                )
            except WebBackendError as exc:
                # transport failure on this instance (the pre-#90 code
                # aborted the loop here, and still does — the outcome
                # is recorded, then the error propagates unchanged)
                self._record_instance(
                    instance, "failure", time.time() - t0,
                    error_class=_backend_error_evidence(exc)[1]
                    or "search_failed")
                raise
            if status == 403:
                # this instance refuses format=json — a wall on this
                # instance, recorded blocked (never dead), then the
                # pre-existing failover to the next instance continues
                self._record_instance(instance, "blocked",
                                      time.time() - t0)
                errors.append(
                    f"{instance}: 403 — this instance has format=json "
                    "disabled in settings.yml; try another instance or "
                    "self-host"
                )
                continue
            if status == 429:
                # recorded rate-limited ONCE for this instance — never
                # re-queried; failover to a *different* configured
                # instance is the pre-existing behaviour, not a retry
                self._record_instance(instance, "rate_limited",
                                      time.time() - t0)
                errors.append(f"{instance}: 429 — instance rate-limited us")
                continue
            if status != 200:
                self._record_instance(
                    instance, "failure", time.time() - t0,
                    error_class=f"http_{status}")
                errors.append(f"{instance}: HTTP {status}")
                continue
            try:
                payload = _json(status, body, "searxng")
            except WebBackendError:
                self._record_instance(
                    instance, "failure", time.time() - t0,
                    error_class="malformed_response")
                raise
            out = []
            for r in payload.get("results", []) or []:
                out.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("content"),
                    "engine": r.get("engine"),
                    "score": r.get("score"),
                    "published": r.get("publishedDate"),
                })
            self._record_instance(
                instance, "served" if out else "no_result",
                time.time() - t0)
            return out
        raise WebBackendError(
            "searxng: all instances failed: " + "; ".join(errors)
        )


# ── 2. ddgs / DuckDuckGo multi-engine (keyless fallback) ───────────────

class DdgsWebSource(WebSearchSource):
    """``ddgs`` package (ex-duckduckgo_search): keyless multi-engine text
    search — backends include bing, brave, duckduckgo, google, mojeek,
    startpage, yandex, yahoo, wikipedia. Optional dependency.

    ``NM_DDGS_KIND`` routes the vertical: ``text`` (default), ``news``,
    ``images``, ``videos`` — the ``ddgs`` package's own methods, so a
    news or media query gets engines tuned for it instead of generic
    text results.
    """

    name = "web_ddgs"
    description = (
        "ddgs multi-engine web search (keyless; pip install ddgs; "
        "NM_DDGS_BACKEND to pin engines; NM_DDGS_KIND=text|news|images|videos)"
    )

    _ddgs_missing: bool | None = None

    #: the ddgs vertical methods we route to
    KINDS = frozenset({"text", "news", "images", "videos"})

    def probe(self) -> str | None:
        if DdgsWebSource._ddgs_missing is None:
            try:
                __import__("ddgs")
                DdgsWebSource._ddgs_missing = False
            except ImportError:
                DdgsWebSource._ddgs_missing = True
        if DdgsWebSource._ddgs_missing:
            return "ddgs package not installed (pip install ddgs for keyless web search)"
        return None

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise WebBackendError(
                "ddgs package not installed (pip install ddgs)"
            ) from exc
        backend = _env("NM_DDGS_BACKEND", "DDGS_BACKEND", default="auto") or "auto"
        # dual-scope transport: NM_DDGS_REGION="ng-ng" pins Nigeria-flavoured
        # results, "us-en" the US; default "wt-wt" (no region) keeps it global
        region = _env("NM_DDGS_REGION", "DDGS_REGION", default="wt-wt") or "wt-wt"
        kind = _env("NM_DDGS_KIND", "DDGS_KIND", default="text").strip().lower()
        if kind not in self.KINDS:
            kind = "text"
        timeout = _timeout(15.0)
        try:
            client = DDGS(timeout=int(timeout))
            if kind == "text":
                try:
                    rows = client.text(
                        query, region=region, safesearch="moderate",
                        backend=backend, max_results=limit,
                    )
                except TypeError:
                    # older ddgs without the backend kwarg
                    rows = client.text(
                        query, region=region, safesearch="moderate",
                        max_results=limit,
                    )
            elif kind == "news":
                rows = client.news(
                    query, region=region, safesearch="moderate",
                    max_results=limit,
                )
            elif kind == "images":
                rows = client.images(
                    query, region=region, safesearch="moderate",
                    max_results=limit,
                )
            else:  # videos
                rows = client.videos(
                    query, region=region, safesearch="moderate",
                    max_results=limit,
                )
            out = []
            for r in rows or []:
                # news/images/videos rows use the same title/body keys;
                # image/video rows carry the media URL in "image"/"content"
                url = r.get("href") or r.get("image") or r.get("content")
                if kind in ("images", "videos"):
                    snippet = r.get("title") or ""
                else:
                    snippet = r.get("body")
                out.append({
                    "title": r.get("title"),
                    "url": url,
                    "snippet": snippet,
                    "engine": r.get("backend") or backend,
                    "kind": kind,
                })
            return out
        except WebBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - ddgs raises its own errors
            raise WebBackendError(f"ddgs backend {backend!r} failed: {exc}") from exc


# ── keyed API backends ─────────────────────────────────────────────────

class _KeyedWebSource(WebSearchSource):
    """Base for API-key web backends: probe() skips when the key is absent."""

    def _key_names(self) -> tuple[str, ...]:
        return ()

    def _api_key(self) -> str:
        return _env(*self._key_names())

    def _auth_headers(self) -> dict[str, str]:
        return {}


class TavilyWebSource(_KeyedWebSource):
    """Tavily search API — 1,000 free credits/month, no card required."""

    name = "web_tavily"
    description = "Tavily web search API (1k free credits/mo, NM_TAVILY_API_KEY)"

    def _key_names(self) -> tuple[str, ...]:
        return ("NM_TAVILY_API_KEY", "TAVILY_API_KEY")

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key()}"}

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        status, body = _http(
            "POST", "https://api.tavily.com/search",
            json_body={
                "query": query, "max_results": min(limit, 10),
                "search_depth": "basic", "include_answer": False,
                "include_raw_content": False,
            },
            headers=self._auth_headers(),
            timeout=_timeout(),
        )
        if status == 401:
            raise WebBackendError("tavily: 401 — invalid API key")
        if status == 432:
            raise WebBackendError("tavily: 432 — monthly credit quota exhausted")
        if status == 429:
            raise WebBackendError("tavily: 429 — rate limited, back off and retry")
        if status != 200:
            raise WebBackendError(f"tavily: HTTP {status}")
        payload = _json(status, body, "tavily")
        return [{
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content"),
            "score": r.get("score"),
            "published": r.get("published_date"),
        } for r in payload.get("results", []) or []]


class SerperWebSource(_KeyedWebSource):
    """Serper.dev — Google SERP JSON; 2,500 free queries, one-time, no card."""

    name = "web_serper"
    description = "Serper Google SERP API (2.5k free queries, NM_SERPER_API_KEY)"

    def _key_names(self) -> tuple[str, ...]:
        return ("NM_SERPER_API_KEY", "SERPER_API_KEY")

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        status, body = _http(
            "POST", "https://google.serper.dev/search",
            json_body={"q": query, "num": min(limit, 20)},
            headers={"X-API-KEY": self._api_key()},
            timeout=_timeout(),
        )
        if status in (401, 403):
            raise WebBackendError(f"serper: HTTP {status} — invalid API key")
        if status == 429:
            raise WebBackendError("serper: 429 — rate limited / quota exhausted")
        if status != 200:
            raise WebBackendError(f"serper: HTTP {status}")
        payload = _json(status, body, "serper")
        return [{
            "title": r.get("title"),
            "url": r.get("link"),
            "snippet": r.get("snippet"),
            "published": r.get("date"),
        } for r in payload.get("organic", []) or []]


class ExaWebSource(_KeyedWebSource):
    """Exa neural web search — $10/month free credits, highlights included."""

    name = "web_exa"
    description = "Exa neural web search ($10/mo free credits, NM_EXA_API_KEY)"

    def _key_names(self) -> tuple[str, ...]:
        return ("NM_EXA_API_KEY", "EXA_API_KEY")

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        status, body = _http(
            "POST", "https://api.exa.ai/search",
            json_body={
                "query": query,
                "numResults": min(limit, 10),
                "contents": {"highlights": True},
            },
            headers={"x-api-key": self._api_key()},
            timeout=_timeout(15.0),
        )
        if status == 401:
            raise WebBackendError("exa: 401 — invalid API key")
        if status == 429:
            raise WebBackendError("exa: 429 — rate limited / credits exhausted")
        if status == 422:
            raise WebBackendError("exa: 422 — bad request parameters")
        if status != 200:
            raise WebBackendError(f"exa: HTTP {status}")
        payload = _json(status, body, "exa")
        out = []
        for r in payload.get("results", []) or []:
            highlights = r.get("highlights") or []
            snippet = " … ".join(highlights) if highlights else (r.get("text") or "")
            out.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": snippet,
                "published": r.get("publishedDate"),
            })
        return out


class BraveWebSource(_KeyedWebSource):
    """Brave Search API on Brave's own index.

    Pricing reality (2026-02+): no free tier for new accounts — $5/month
    metered credits with a card on file. Kept as a keyed backend for
    users with keys / grandfathered free plans, never the default.
    """

    name = "web_brave"
    description = (
        "Brave Search API, own index (metered since 2026-02; "
        "NM_BRAVE_SEARCH_API_KEY)"
    )

    def _key_names(self) -> tuple[str, ...]:
        return ("NM_BRAVE_SEARCH_API_KEY", "BRAVE_SEARCH_API_KEY")

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        status, body = _http(
            "GET", "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(limit, 20), "safesearch": "moderate"},
            headers={"X-Subscription-Token": self._api_key()},
            timeout=_timeout(),
        )
        if status in (401, 403):
            raise WebBackendError(f"brave: HTTP {status} — invalid subscription token")
        if status == 429:
            raise WebBackendError("brave: 429 — rate limited / plan quota hit")
        if status != 200:
            raise WebBackendError(f"brave: HTTP {status}")
        payload = _json(status, body, "brave")
        web = payload.get("web") or {}
        return [{
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("description"),
        } for r in web.get("results", []) or []]


#: Canonical web-backend order: best free first (keyless), then keyed
#: free-tier APIs, metered Brave last. Doubles as the ranking tie-break.
WEB_SPECS: list[tuple[str, type[WebSearchSource]]] = [
    ("web_searxng", SearXNGWebSource),
    ("web_ddgs", DdgsWebSource),
    ("web_tavily", TavilyWebSource),
    ("web_serper", SerperWebSource),
    ("web_exa", ExaWebSource),
    ("web_brave", BraveWebSource),
]


def web_source_names() -> list[str]:
    """Names of every registered web-search backend, in priority order."""
    return [name for name, _ in WEB_SPECS]


def web_backends_configured() -> dict[str, bool]:
    """Which web backends could serve a query right now (cheap checks,
    no network): keyed backends need their key, SearXNG needs an instance
    URL, ddgs needs its package installed."""
    out: dict[str, bool] = {}
    for name, cls in WEB_SPECS:
        try:
            out[name] = cls().probe() is None
        except Exception:  # noqa: BLE001 - probe reports, never raises
            out[name] = False
    return out
