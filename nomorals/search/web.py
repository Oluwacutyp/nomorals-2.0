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
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

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
    "web_source_names",
    "web_backends_configured",
]

#: Identify ourselves to shared infrastructure (SearXNG instances,
#: DDG endpoints) instead of spoofing a browser.
_USER_AGENT = "DevonSearch/1.0 (nomorals federated web search; +https://github.com/Oluwacutyp/No-morals-ai)"


class WebBackendError(SearchError):
    """A web-search backend failed (transport, auth, quota, bad payload).

    ``federated_search`` wraps this as ``SearchError("source %r failed:
    ...")`` — fail fast, naming the source, with the real cause chained.
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

    def __init__(self, context: Any = None) -> None:
        # Web backends are configured via environment, not the context —
        # the parameter keeps build_adapters() uniform across sources.
        self._context = context

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
        try:
            raw_hits = self._fetch(query, limit)
        except WebBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrapped with backend name
            raise WebBackendError(f"{self.name}: unexpected error: {exc}") from exc
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
    """

    name = "web_searxng"
    description = (
        "SearXNG metasearch (keyless, 70+ upstream engines, "
        "NM_SEARXNG_URL; self-hosted recommended)"
    )

    def _instances(self) -> list[str]:
        raw = _env("NM_SEARXNG_URL", "SEARXNG_URL")
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]

    def probe(self) -> str | None:
        if not self._instances():
            return (
                "no SearXNG instance configured "
                "(set NM_SEARXNG_URL to your instance URL, e.g. "
                "http://localhost:8888 for a self-hosted instance)"
            )
        return None

    def _fetch(self, query: str, limit: int) -> list[dict[str, Any]]:
        timeout = _timeout(12.0)
        errors: list[str] = []
        for instance in self._instances():
            status, body = _http(
                "GET", f"{instance}/search",
                params={
                    "q": query, "format": "json", "language": "auto",
                    "safesearch": "1", "pageno": "1",
                },
                timeout=timeout,
            )
            if status == 403:
                errors.append(
                    f"{instance}: 403 — this instance has format=json "
                    "disabled in settings.yml; try another instance or "
                    "self-host"
                )
                continue
            if status == 429:
                errors.append(f"{instance}: 429 — instance rate-limited us")
                continue
            if status != 200:
                errors.append(f"{instance}: HTTP {status}")
                continue
            payload = _json(status, body, "searxng")
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
            return out
        raise WebBackendError(
            "searxng: all instances failed: " + "; ".join(errors)
        )


# ── 2. ddgs / DuckDuckGo multi-engine (keyless fallback) ───────────────

class DdgsWebSource(WebSearchSource):
    """``ddgs`` package (ex-duckduckgo_search): keyless multi-engine text
    search — backends include bing, brave, duckduckgo, google, mojeek,
    startpage, yandex, yahoo, wikipedia. Optional dependency."""

    name = "web_ddgs"
    description = (
        "ddgs multi-engine web search (keyless; pip install ddgs; "
        "NM_DDGS_BACKEND to pin engines)"
    )

    _ddgs_missing: bool | None = None

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
        timeout = _timeout(15.0)
        try:
            client = DDGS(timeout=int(timeout))
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
            out = []
            for r in rows or []:
                out.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body"),
                    "engine": r.get("backend") or backend,
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
