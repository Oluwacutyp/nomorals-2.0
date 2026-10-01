"""Dedicated search engine: search → read → crawl → curate → summarize.

Built on the existing keyless web tools (DuckDuckGo HTML, robots.txt-aware
fetch, html→text), so a personal AI can research without a single API key.

Two modes:
    quick — one search, top pages read, summarized. Always available.
    deep  — query decomposition, sub-query fan-out, optional crawl, sourced
            synthesis. A power-mode capability on purpose: it spends real
            network + model budget, and the owner asked for that dial.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from typing import Any

from ...core.errors import ToolError
from ...core.http import HttpClient
from ...core.ids import new_short_id
from ...core.logging_setup import get_logger
from ...core.policy import Capability
from ...tools.web import RobotsCache, html_to_text
from ..power import power_mode_for
from . import curate, summarize as summarize_mod
from .trust import SourceTrust

__all__ = ["SearchEngine"]

_log = get_logger(__name__)

_DEEP_WALL_SECONDS = 180.0
_PAGE_TTL_SECONDS = 3600.0
_CRAWL_LINK = re.compile(r'(?i)<a[^>]+href=["\']([^"#]+)["\']')
_LINK_ANCHOR = re.compile(
    r'(?is)<a[^>]+href=["\']([^"#]+)["\'][^>]*>(.*?)</a>'
)
_JUNK_LINK_ENDINGS = (
    ".zip", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mp3",
    ".apk", ".exe", ".dmg", ".css", ".js", ".ico", ".svg", ".woff", ".woff2",
)


def _extract_external_links(markup: str, page_domain: str, cap: int = 300) -> list[tuple[str, str]]:
    """(url, anchor) pairs leaving the page's own domain — the fuel for
    cross-source digging.  Junk targets (media, assets, mailto) dropped."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _LINK_ANCHOR.finditer(markup or ""):
        href = (match.group(1) or "").strip()
        if href.startswith("//"):
            href = "https:" + href
        if not href.startswith(("http://", "https://")):
            continue
        low = href.lower()
        if any(low.split("?")[0].endswith(e) for e in _JUNK_LINK_ENDINGS):
            continue
        if curate.domain(href) == page_domain:
            continue
        anchor = html_to_text(match.group(2) or "")
        if href in seen:
            continue
        seen.add(href)
        out.append((href, anchor[:120]))
        if len(out) >= cap:
            break
    return out

#: Legit platforms where people get paid for small tasks — research list only.
#: The engine reports these; it never signs anyone up anywhere.
#: Query angles for the /searchleads research pass — broadened so one
#: dead angle doesn't sink the list. Nigeria-relevant angles included.
_LEAD_QUERIES = (
    "get paid for user testing websites",
    "legitimate microtask platforms pay per task",
    "paid online transcription work real companies",
    "freelance platforms for beginners first gig",
    "paid survey sites that actually pay",
    "data annotation labeling jobs remote beginners",
    "AI training data gigs get paid to train AI",
    "website usability testing jobs work from home",
    "paid translation gigs online no degree",
    "virtual assistant jobs beginners remote",
    "sell stock photos videos online passive income",
    "video game testing jobs remote playtesters",
    "bug bounty platforms beginners first payout",
    "remote microtask sites that pay to Nigeria Africa",
    "get paid to test websites Nigeria work from home",
)


class SearchEngine:
    def __init__(self, context: Any) -> None:
        self.context = context
        settings = getattr(context, "settings", None)
        tools = getattr(settings, "tools", None)
        self._user_agent = getattr(tools, "user_agent", "NoMoralsCore/0.1")
        self._timeout = getattr(tools, "http_timeout", 30.0)
        self._respect_robots = getattr(tools, "robots_txt", True)
        self._proxy = getattr(tools, "proxy_url", "")
        self._client = HttpClient(timeout=self._timeout, user_agent=self._user_agent,
                                  proxy_url=self._proxy)
        self._robots = RobotsCache()

    # ── primitives ──────────────────────────────────────────────────────────
    def search(self, query: str, max_results: int = 8, freshness: str = "") -> list[dict[str, str]]:
        """Rank results from the keyless engine chain (web_search tool)."""
        if self.context is None or getattr(self.context, "tools", None) is None:
            raise ToolError("search needs the tool registry (build the context with tools)")
        outcome = self.context.tools.call(
            "web_search", query=query, max_results=max_results, freshness=freshness,
        )
        if not outcome.ok:
            raise ToolError(f"search failed: {outcome.error.message if outcome.error else 'unknown'}")
        return list(outcome.unwrap().get("results") or [])

    def read(self, url: str, max_chars: int = 40000) -> dict[str, Any] | None:
        """Fetch one page (robots-aware), cached in page_cache for an hour.

        PDFs are first-class sources: their text comes from the pure-Python
        PDF reader, so deep research can pull the actual report, not just
        the landing page that links to it.  HTML pages also carry their
        external ``links`` (href, anchor) — the fuel for the dig loop.
        """
        url = (url or "").strip()
        if not url or not url.startswith(("http://", "https://")):
            return None
        cached = self._cache_get(url)
        if cached:
            return cached
        if self._respect_robots and not self._robots.allowed(url, self._user_agent, self._client):
            _log.debug("robots.txt disallows %s", url)
            return None
        try:
            response = self._client.get(url)
        except Exception as exc:  # noqa: BLE001 - one dead page must not kill the run
            _log.debug("fetch failed for %s: %s", url, exc)
            return None
        if not response.ok:
            return None
        is_pdf = url.lower().endswith(".pdf") or "pdf" in response.content_type
        title = ""
        if is_pdf:
            from ...core.pdf import read_pdf_text

            try:
                text = read_pdf_text(response.body)[:max_chars]
            except Exception:  # noqa: BLE001 - a corrupt PDF is skipped, not fatal
                text = ""
            title = url.rsplit("/", 1)[-1] or "pdf"
        else:
            markup = response.text
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", markup)
            if title_match:
                title = re.sub(r"(?s)<[^>]*>", "", title_match.group(1)).strip()
            text = html_to_text(markup)[:max_chars]
        page = {
            "url": response.url or url,
            "title": title,
            "text": text,
            "domain": curate.domain(url),
            "chars": len(text),
            "pdf": is_pdf,
            "links": _extract_external_links(response.text if not is_pdf else "",
                                              curate.domain(url))[:300],
        }
        self._cache_put(page)
        return page

    def crawl(self, root_url: str, max_pages: int = 8, max_chars_per_page: int = 12000) -> list[dict[str, Any]]:
        """Breadth-first read of a site, same-domain only, robots-aware,
        hard-capped — a reader, not a scraper at scale."""
        root_domain = curate.domain(root_url)
        seen: set[str] = set()
        frontier = [root_url]
        pages: list[dict[str, Any]] = []
        while frontier and len(pages) < max_pages:
            url = frontier.pop(0)
            norm = url.rstrip("/").lower()
            if norm in seen:
                continue
            seen.add(norm)
            if curate.domain(url) != root_domain:
                continue
            try:
                markup = self._client.get(url).text
            except Exception:  # noqa: BLE001
                continue
            for link in _CRAWL_LINK.findall(markup)[: 8 * max_pages]:
                target = link.strip()
                if target.startswith("//"):
                    target = "https:" + target
                if target.startswith("/"):
                    target = f"{urllib.parse.urlparse(url).scheme}://{urllib.parse.urlparse(url).netloc}{target}"
                if target not in seen and curate.domain(target) == root_domain:
                    frontier.append(target)
            page = self.read(url, max_chars=max_chars_per_page)
            if page:
                pages.append(page)
        return pages

    # ── model awareness ─────────────────────────────────────────────────────
    def _model_available(self) -> bool:
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return False
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001
            return False
        active = str(snap.get("active") or "")
        return bool(active) and active not in {"mock", "offline", "test"}

    def _decompose(self, query: str) -> list[str]:
        """Split a research question into 2-5 sub-queries. Model when a real
        model is answering; otherwise just the original query."""
        if not self._model_available():
            return [query]
        try:
            from ...llm.base import Message, SamplingParams

            response = self.context.router.chat(
                [
                    Message.system(
                        "Decompose a research question into 2-5 complementary search "
                        "sub-queries. Reply with ONLY a JSON array of strings."
                    ),
                    Message.user(query),
                ],
                SamplingParams(temperature=0.2, max_tokens=300),
            )
            text = (response.text or "").strip()
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end > start:
                import json

                sub = [str(x).strip() for x in json.loads(text[start : end + 1]) if str(x).strip()]
                if sub:
                    return sub[:5]
        except Exception as exc:  # noqa: BLE001 - decomposition is a bonus, not a gate
            _log.debug("query decomposition failed: %s", exc)
        return [query]

    # ── the runs ────────────────────────────────────────────────────────────
    def run(self, query: str, mode: str = "quick", pages: int = 3, crawl: bool = False,
            dig: bool = True, freshness: str = "") -> dict[str, Any]:
        started = time.time()
        if mode not in {"quick", "deep"}:
            raise ToolError(f"unknown mode {mode!r}: quick | deep")
        if mode == "deep":
            if not power_mode_for(self.context).active:
                raise ToolError("deep research is a power-mode capability — enable power mode first")
            from .deep import DeepResearcher

            researcher = DeepResearcher(self.context, engine=self,
                                        max_pages=pages,
                                        pages_per_query=max(2, pages // 3),
                                        dig=dig)
            try:
                return researcher.run(query)
            except TimeoutError as exc:
                raise ToolError(str(exc)) from exc

        sub_queries = [query]
        all_results: list[dict[str, str]] = []
        sub_reports: list[dict[str, Any]] = []
        for sub in sub_queries:
            if time.time() - started > _DEEP_WALL_SECONDS:
                _log.info("search wall clock hit; stopping sub-queries")
                break
            raw = self.search(sub, freshness=freshness) if freshness else self.search(sub)
            results = curate.curate(raw, sub, top_n=pages + 2)
            all_results.extend(results)
            sub_reports.append({"query": sub, "results": results})

        unique = curate.dedupe(all_results)
        unique.sort(key=lambda r: -float(r.get("score", 0)))
        top = unique[: max(1, pages)]

        # wave 85: source trust — every result carries a trust score, and
        # each read attempt feeds back (a source that keeps failing to
        # read earns a persistent penalty)
        trust: SourceTrust | None = None
        try:
            trust = SourceTrust(self.context)
            trust.annotate(top)
        except Exception:  # noqa: BLE001 — trust is a display bonus
            _log.debug("source trust unavailable", exc_info=True)
            trust = None

        read_pages: list[dict[str, Any]] = []
        for r in top:
            page = self.read(r["url"])
            if trust is not None:
                try:
                    trust.feedback(r["url"], page is not None)
                except Exception:  # noqa: BLE001
                    pass
            if page:
                read_pages.append(page)
            if time.time() - started > _DEEP_WALL_SECONDS:
                break

        if crawl and read_pages:
            crawl_root = read_pages[0]["url"]
            try:
                read_pages = read_pages + self.crawl(crawl_root, max_pages=max(2, pages - 1))[: max(2, pages - 1)]
            except Exception as exc:  # noqa: BLE001
                _log.debug("crawl failed: %s", exc)

        model_on = self._model_available()
        error = ""
        try:
            summary = summarize_mod.model_summarize(self.context.router, query, read_pages) if model_on \
                else summarize_mod.extractive_summarize(query, read_pages)
        except Exception as exc:  # noqa: BLE001 - a dead model must not kill the research
            error = str(exc)
            summary = summarize_mod.extractive_summarize(query, read_pages)

        report = {
            "id": new_short_id("search"),
            "query": query,
            "mode": mode,
            "sub_queries": sub_queries,
            "results": top,
            "pages_read": [p["url"] for p in read_pages],
            "pages": read_pages,
            "summary": summary,
            "model_summary": model_on and not error,
            "error": error,
            "seconds": round(time.time() - started, 2),
        }
        self._journal(report)
        return report

    def leads(self, save: bool = True) -> list[dict[str, Any]]:
        """Research pass over legitimate 'get paid for small tasks' platforms:
        search, curate, de-duplicate by domain. A report — the engine does
        not create accounts anywhere on anyone's behalf."""
        found: dict[str, dict[str, Any]] = {}
        for q in _LEAD_QUERIES:
            try:
                for r in curate.curate(self.search(q, max_results=6), q, top_n=4):
                    dom = r["domain"]
                    if dom and dom not in found:
                        found[dom] = {"url": r["url"], "title": r.get("title", ""), "domain": dom}
            except Exception as exc:  # noqa: BLE001 - one dead endpoint must not sink the list
                _log.debug("leads search failed for %r: %s", q, exc)
        out = list(found.values())
        if save and out:
            try:
                with self.context.db.transaction():
                    for lead in out:
                        self.context.db.execute(
                            "INSERT OR IGNORE INTO search_leads (id, url, title, note, created_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (new_short_id("lead"), lead["url"], lead["title"], "paid-task platform", time.time()),
                        )
            except Exception as exc:  # noqa: BLE001
                _log.warning("could not save leads: %s", exc)
        return out

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            rows = self.context.db.query(
                "SELECT * FROM search_log ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        except Exception:  # noqa: BLE001
            return []
        return list(rows)

    # ── persistence ─────────────────────────────────────────────────────────
    def _journal(self, report: dict[str, Any]) -> None:
        try:
            import json

            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO search_log (id, query, mode, results, summary, seconds, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        report["id"],
                        report["query"],
                        report["mode"],
                        json.dumps(report["results"]),
                        report["summary"][:8000],
                        report["seconds"],
                        time.time(),
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - journaling is best-effort
            _log.warning("search journal write failed: %s", exc)

    def _cache_get(self, url: str) -> dict[str, Any] | None:
        try:
            row = self.context.db.query_one("SELECT * FROM page_cache WHERE url = ?", (url,))
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        if time.time() - float(row.get("fetched_at") or 0) > _PAGE_TTL_SECONDS:
            return None
        return {
            "url": row["url"],
            "title": row.get("title") or "",
            "text": row.get("text") or "",
            "domain": curate.domain(url),
            "chars": len(row.get("text") or ""),
        }

    def _cache_put(self, page: dict[str, Any]) -> None:
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO page_cache (url, title, text, fetched_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(url) DO UPDATE SET title=excluded.title, text=excluded.text, "
                    "fetched_at=excluded.fetched_at",
                    (page["url"], page["title"], page["text"][:60000], time.time()),
                )
        except Exception as exc:  # noqa: BLE001
            _log.debug("page_cache write failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY TOOL
# ═══════════════════════════════════════════════════════════════════════════


def register(registry: Any) -> None:
    """Attach web_research to a registry."""
    context = registry.context

    @registry.register(
        "web_research",
        description=(
            "Research a question on the open web: search (multi-engine), read the top "
            "pages (PDFs included), return a summarized, sourced answer. deep=true "
            "(power mode only) decomposes the question, reads more, and digs one level "
            "deeper (mined follow-up queries + external links); crawl=true also reads "
            "through the best site; freshness=d|w|m|y biases recency. Keyless, robots-aware."
        ),
        capability=Capability.NET_OUT,
    )
    def web_research(query: str, *, deep: bool = False, pages: int = 3,
                     crawl: bool = False, dig: bool = True, freshness: str = "") -> dict[str, Any]:
        return SearchEngine(context).run(query, mode="deep" if deep else "quick",
                                         pages=pages, crawl=crawl, dig=dig,
                                         freshness=freshness)

    @registry.register(
        "deep_search",
        description=(
            "full deep research: parallel sub-queries, diverse fresh sources, "
            "section-level extraction, cited synthesis, then a dig loop "
            "(follow-up queries mined from what was read + strongest external "
            "links followed; PDFs read as sources). Power mode only."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "query": "str — the research question",
            "pages": "int (optional, default 8, max 12) — how many sources to read",
            "dig": "bool (optional, true) — the deeper-dig loop (follow-ups + external links)",
        },
    )
    def deep_search(query: str, *, pages: int = 8, dig: bool = True) -> dict[str, Any]:
        from ..power import power_mode_for
        from .deep import DeepResearcher

        if not power_mode_for(context).active:
            raise ToolError("deep research is a power-mode capability — enable power mode first")
        researcher = DeepResearcher(context, engine=SearchEngine(context),
                                    max_pages=pages, pages_per_query=max(2, pages // 3),
                                    dig=dig)
        try:
            return researcher.run(query)
        except TimeoutError as exc:
            raise ToolError(str(exc)) from exc
