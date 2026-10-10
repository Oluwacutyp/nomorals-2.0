"""Web tools: fetch, search, extract, and download.

Search is multi-engine: DuckDuckGo's HTML endpoints stay primary and Bing's
HTML search is an automatic fallback whenever DDG returns nothing or errors —
DDG's scrapes break often enough that a lone-engine search is a coin flip.

Fetch detects the page charset (Content-Type header first, then ``<meta>``)
instead of blindly assuming UTF-8. ``web_extract`` adds readability-style
main-content extraction for article pages.

robots.txt is honoured by default. It costs nothing and it is the difference
between a tool and a nuisance.
"""

from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.robotparser
from typing import Any

from ..core.errors import ToolError, classify
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..core.text import approx_token_count, normalize_text

__all__ = ["html_to_text", "readability_extract", "web_extract", "register"]

_log = get_logger(__name__)

_SCRIPT_STYLE = re.compile(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]*>")
_ENTITIES = re.compile(r"&(?!#?\w+;)")

#: Per-engine search timeout. Search engines should answer fast; if one is
#: slow it is usually about to be down, so fail soft and move on.
SEARCH_TIMEOUT = 8.0

#: Boilerplate elements stripped before readability scoring. Never the
#: article itself.
_BOILERPLATE = re.compile(
    r"(?is)<(nav|header|footer|aside|form|menu|script|style|noscript|svg|iframe|canvas|object|embed)[^>]*>.*?</\1>"
)


def html_to_text(markup: str, *, keep_links: bool = False) -> str:
    """Strip HTML to readable text. No BeautifulSoup required."""
    if not markup:
        return ""
    text = _SCRIPT_STYLE.sub(" ", markup)
    if keep_links:
        text = re.sub(
            r'(?i)<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            lambda m: f"{_TAGS.sub('', m.group(2))} [{m.group(1)}]",
            text,
            flags=re.S,
        )
    text = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>", "\n", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(_ENTITIES.sub("&amp;", text))
    return normalize_text(text)


class RobotsCache:
    """Per-host robots.txt, cached for ten minutes."""

    def __init__(self, ttl: float = 600.0) -> None:
        self._cache: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}
        self.ttl = ttl

    def allowed(self, url: str, user_agent: str = "*", client: HttpClient | None = None) -> bool:
        parts = urllib.parse.urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        entry = self._cache.get(origin)
        now = time.time()
        if entry and now - entry[0] < self.ttl:
            parser = entry[1]
        else:
            parser = urllib.robotparser.RobotFileParser()
            try:
                response = (client or HttpClient(timeout=10.0)).get(f"{origin}/robots.txt")
                parser.parse(response.text.splitlines() if response.ok else [])
            except Exception:  # noqa: BLE001 - unreachable robots.txt means "no restrictions known"
                parser = None
            self._cache[origin] = (now, parser)
        if parser is None:
            return True
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:  # noqa: BLE001 - malformed robots.txt must not block
            return True


_robots = RobotsCache()


def web_extract(
    url: str,
    *,
    max_chars: int = 20000,
    client: HttpClient | None = None,
    user_agent: str = "NoMoralsCore/0.1",
    respect_robots: bool = True,
) -> dict[str, Any]:
    """Readability-style extraction: main content, not the chrome.

    Returns title + best content + word count. Raises ToolError when
    robots.txt disallows the URL."""
    client = client or HttpClient(timeout=30.0)
    if respect_robots and not _robots.allowed(url, user_agent, client):
        raise ToolError(f"robots.txt disallows {url}")
    response = client.get(url)
    markup = _decode_body(response.body, response.content_type)
    article = readability_extract(markup)
    text = article["text"]
    return {
        "url": response.url or url,
        "status": response.status,
        "content_type": response.content_type,
        "title": article["title"],
        "words": article["words"],
        "chars": len(text),
        "tokens": approx_token_count(text),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def register(registry: Any) -> None:
    """Attach the web tools to a registry."""
    context = registry.context
    settings = getattr(context, "settings", None) if context is not None else None
    tools_settings = getattr(settings, "tools", None) if settings else None
    user_agent = getattr(tools_settings, "user_agent", "NoMoralsCore/0.1") if tools_settings else "NoMoralsCore/0.1"
    timeout = getattr(tools_settings, "http_timeout", 30.0) if tools_settings else 30.0
    respect_robots = getattr(tools_settings, "robots_txt", True) if tools_settings else True
    client = HttpClient(timeout=timeout, user_agent=user_agent)
    # Search gets its own short-timeout client: one sluggish engine must not
    # hold the whole search hostage.
    search_client = HttpClient(timeout=SEARCH_TIMEOUT, user_agent=user_agent)

    @registry.register(
        "web_fetch",
        description="Fetch a URL and return its text content. Pass extract=True for the readable article body instead of the whole page.",
        capability=Capability.NET_OUT,
    )
    def web_fetch(url: str, *, max_chars: int = 40000, raw: bool = False, extract: bool = False) -> dict[str, Any]:
        """Fetch a URL; charset-detected, robots-checked. ``extract=True``
        returns the readability-style article body instead of full-page text."""
        if respect_robots and not _robots.allowed(url, user_agent, client):
            raise ToolError(f"robots.txt disallows {url}")
        response = client.get(url)
        markup = _decode_body(response.body, response.content_type)
        title = _extract_title(markup)
        if extract:
            article = readability_extract(markup)
            text = article["text"]
            title = article["title"] or title
        else:
            text = markup if raw else html_to_text(markup)
        return {
            "url": response.url or url,
            "status": response.status,
            "content_type": response.content_type,
            "title": title,
            "chars": len(text),
            "tokens": approx_token_count(text),
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    @registry.register(
        "web_extract",
        description="Extract the readable article body from a URL: strips nav/header/footer/aside and returns the main content with title and word count.",
        capability=Capability.NET_OUT,
    )
    def _web_extract_tool(url: str, *, max_chars: int = 20000) -> dict[str, Any]:
        """Readability-style extraction: main content, not the chrome."""
        return web_extract(
            url, max_chars=max_chars, client=client,
            user_agent=user_agent, respect_robots=respect_robots,
        )

    @registry.register(
        "web_search",
        description="Search the web (DuckDuckGo primary, Bing automatic fallback; no API key) and return ranked links.",
        capability=Capability.NET_OUT,
    )
    def web_search(query: str, *, max_results: int = 8, site: str = "", freshness: str = "") -> dict[str, Any]:
        needle = f"{query} site:{site}" if site else query
        encoded = urllib.parse.quote_plus(needle)
        results: list[dict[str, str]] = []
        engines_used: list[str] = []
        for engine, endpoint, parser in (
            ("duckduckgo", f"https://html.duckduckgo.com/html/?q={encoded}", _parse_results),
            ("duckduckgo-lite", f"https://lite.duckduckgo.com/lite/?q={encoded}", _parse_results),
            ("bing", f"https://www.bing.com/search?q={encoded}&count=10", _parse_bing),
        ):
            try:
                response = search_client.get(endpoint)
            except Exception as exc:  # noqa: BLE001 - try the next engine
                _log.debug("search engine %s failed: %s", engine, classify(exc).message)
                continue
            got = parser(response.text)
            if got:
                engines_used.append(engine)
                results.extend(got)
                break  # primary engine hit: no need to consult the fallback
            _log.debug("search engine %s returned no results; falling back", engine)
        results = _dedupe_results(results)
        return {
            "query": needle,
            "count": len(results[:max_results]),
            "engines": engines_used,
            "results": results[:max_results],
        }

    @registry.register(
        "web_download",
        description="Download a URL to a file in the workspace.",
        capability=Capability.NET_DOWNLOAD,
    )
    def web_download(url: str, *, filename: str = "") -> dict[str, Any]:
        from .filesystem import safe_path

        name = filename or urllib.parse.urlparse(url).path.rsplit("/", 1)[-1] or "download.bin"
        target = safe_path(context, f"downloads/{name}")
        client.download(url, target)
        return {"url": url, "path": str(target), "bytes": target.stat().st_size}


# ── charset ──────────────────────────────────────────────────────────────────

_CHARSET_PARAM = re.compile(r"(?i)charset\s*=\s*[\"']?([\w\-\.]+)")
_META_CHARSET = re.compile(r'(?is)<meta[^>]+charset\s*=\s*["\']?([\w\-\.]+)')


def _decode_body(body: bytes, content_type: str) -> str:
    """Decode page bytes using the Content-Type charset, then ``<meta
    charset>``, then UTF-8 with errors replaced. Never raises."""
    charset: str | None = None
    if content_type:
        match = _CHARSET_PARAM.search(content_type)
        if match:
            charset = match.group(1)
    if not charset and body:
        head = body[:4096].decode("latin-1", errors="replace")
        match = _META_CHARSET.search(head)
        if match:
            charset = match.group(1)
    for candidate in (charset, "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def _extract_title(markup: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", markup)
    if not match:
        return ""
    return html.unescape(_TAGS.sub("", match.group(1))).strip()


# ── readability-style extraction ─────────────────────────────────────────────

_BLOCK_RE = re.compile(r"(?is)<(article|main|section|div|td)\b[^>]*>(.*?)</\1>")
_LINK_TEXT_RE = re.compile(r"(?is)<a\b[^>]*>(.*?)</a>")
_HEADING_RE = re.compile(r"(?is)<h[1-6]\b[^>]*>(.*?)</h[1-6]>")

#: A block this long is worth keeping even when link-heavy.
_MIN_BLOCK_CHARS = 280


def _trafilatura_extract(markup: str, url: str = "",
                     favor_recall: bool = False,
                     output_format: str = "txt") -> dict[str, Any] | None:
    """Try trafilatura (best general-purpose extractor, F1 ~0.94).

    Returns None when trafilatura is not installed or extraction yields
    nothing - the caller falls back to the built-in scorer. Metadata
    (title/author/date) comes from trafilatura's JSON output.
    """
    try:
        import trafilatura  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        if output_format == "json":
            raw = trafilatura.extract(
                markup, output_format="json", include_tables=True,
                favor_recall=favor_recall, url=url or None)
            if not raw:
                return None
            data = json.loads(raw)
            text = data.get("text") or ""
            if not text.strip():
                return None
            return {
                "title": data.get("title") or "",
                "author": data.get("author") or "",
                "date": data.get("date") or "",
                "text": text,
                "words": len(text.split()),
                "engine": "trafilatura",
                "confidence": "high",
            }
        fmt = "markdown" if output_format == "markdown" else "txt"
        text = trafilatura.extract(
            markup, output_format=fmt, include_tables=True,
            include_links=(fmt == "markdown"),
            favor_recall=favor_recall, url=url or None)
        if not text or not text.strip():
            return None
        return {
            "title": _extract_title(markup),
            "text": text,
            "words": len(text.split()),
            "engine": "trafilatura",
            "confidence": "high",
        }
    except Exception:  # noqa: BLE001 - fall back to the built-in scorer
        return None


def readability_extract(markup: str, url: str = "",
                        favor_recall: bool = False,
                        output_format: str = "txt") -> dict[str, Any]:
    """Pull the main article out of a page.

    Extraction ladder (mined best practice): trafilatura first (best
    general-purpose extractor), then the built-in scorer (boilerplate
    strip + block scoring by length penalized by link density;
    ``<article>`` wins outright when present and non-trivial).

    ``output_format``: "txt" (default), "markdown" (keeps headings,
    lists, tables, links - best for LLM ingestion), or "json" (adds
    author/date metadata when the engine provides it).
    ``favor_recall`` extracts more (less precision) for difficult pages.
    The result always names its ``engine`` and a ``confidence`` note so
    callers know when the page defeated the extractor.
    """
    if not markup:
        return {"title": "", "text": "", "words": 0,
                "engine": "none", "confidence": "empty"}
    first = _trafilatura_extract(markup, url, favor_recall, output_format)
    if first is not None:
        return first
    title = _extract_title(markup)
    cleaned = _BOILERPLATE.sub(" ", markup)

    # An explicit <article> is the page telling us where the content is.
    best_html: str | None = None
    best_text: str | None = None
    for match in re.finditer(r"(?is)<article\b[^>]*>(.*?)</article>", cleaned):
        text = html_to_text(match.group(1))
        if len(text) >= _MIN_BLOCK_CHARS and (best_text is None or len(text) > len(best_text)):
            best_text, best_html = text, match.group(1)
    if best_text is not None:
        text = (_html_to_markdown(best_html)
                if output_format == "markdown" and best_html else best_text)
        return {"title": title, "text": text, "words": len(text.split()),
                "engine": "builtin", "confidence": "medium"}

    scored: list[tuple[float, str, str]] = []
    for match in _BLOCK_RE.finditer(cleaned):
        inner = match.group(2)
        if "<article" in inner.lower():
            continue  # already considered
        text = html_to_text(inner)
        if len(text) < _MIN_BLOCK_CHARS:
            continue
        link_text = "".join(
            _TAGS.sub(" ", m.group(1)) for m in _LINK_TEXT_RE.finditer(inner)
        )
        density = len(link_text) / max(len(text), 1)
        heading_bonus = len(_HEADING_RE.findall(inner)) * 60.0
        score = len(text) * (1.0 - min(density, 0.95)) + heading_bonus
        scored.append((score, text, inner))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        _score, text, inner = scored[0]
        if output_format == "markdown":
            text = _html_to_markdown(inner)
        return {"title": title, "text": text, "words": len(text.split()),
                "engine": "builtin", "confidence": "medium"}

    # Nothing scored: the page is flat (or the regexes missed it). Whole page.
    # Low confidence - the caller should know this may be boilerplate.
    text = html_to_text(cleaned)
    if output_format == "markdown":
        text = _html_to_markdown(cleaned)
    return {"title": title, "text": text, "words": len(text.split()),
            "engine": "builtin", "confidence": "low"}


def _html_to_markdown(html: str) -> str:
    """Best-effort HTML -> markdown for LLM ingestion.

    markdownify when installed (better nested lists/tables), else the
    stdlib fallback below. Never raises.
    """
    try:
        import markdownify  # type: ignore
        return markdownify.markdownify(html or "", heading_style="ATX").strip()
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    text = html or ""
    text = re.sub(r"(?is)<h([1-6])[^>]*>(.*?)</h\1>",
                  lambda m: "\n" + "#" * int(m.group(1)) + " " +
                  _TAGS.sub("", m.group(2)).strip() + "\n", text)
    text = re.sub(r"(?is)<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", text)
    text = re.sub(r"(?is)<(em|i)[^>]*>(.*?)</\1>", r"*\2*", text)
    text = re.sub(r'(?is)<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                  lambda m: "[" + _TAGS.sub("", m.group(2)).strip() + "](" +
                  m.group(1) + ")", text)
    text = re.sub(r"(?is)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?is)<(br|p|tr)[^>]*>", "\n", text)
    return html_to_text(text).strip()


# ── search ───────────────────────────────────────────────────────────────────


def _normalize_url(url: str) -> str:
    """Canonical form for dedupe: scheme + lowercased host + bare path."""
    parsed = urllib.parse.urlparse(url.strip())
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


def _dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop duplicate URLs (engine overlap), keeping first-seen order."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in results:
        key = _normalize_url(item["url"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _parse_results(markup: str) -> list[dict[str, str]]:
    """Extract result links and snippets from DuckDuckGo's HTML."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(
        r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', markup
    ):
        href, title = match.group(1), html.unescape(_TAGS.sub("", match.group(2))).strip()
        url = _unwrap_ddg(href)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"url": url, "title": title, "snippet": ""})
    if out:
        return out
    # Fallback shape: plain links outside the site's own navigation.
    for match in re.finditer(r'(?is)<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', markup):
        url = _unwrap_ddg(match.group(1))
        title = html.unescape(_TAGS.sub("", match.group(2))).strip()
        if url and url not in seen and title:
            seen.add(url)
            out.append({"url": url, "title": title, "snippet": ""})
    return out


def _parse_bing(markup: str) -> list[dict[str, str]]:
    """Extract results from Bing's HTML search page (``li.b_algo`` blocks)."""
    out: list[dict[str, str]] = []
    for block in re.finditer(r'(?is)<li\b[^>]*class="b_algo"[^>]*>(.*?)</li>', markup):
        inner = block.group(1)
        link = re.search(r'(?is)<h2>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', inner)
        if not link:
            continue
        href = html.unescape(link.group(1)).strip()
        title = html.unescape(_TAGS.sub("", link.group(2))).strip()
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        snippet = ""
        caption = re.search(r'(?is)<div\b[^>]*class="b_caption"[^>]*>(.*?)</div>', inner)
        if caption:
            snippet = html_to_text(caption.group(1))[:400]
        out.append({"url": href, "title": title, "snippet": snippet})
    return out


def _parse_ddg(markup: str) -> list[dict[str, str]]:
    """DuckDuckGo HTML endpoint parser (html.duckduckgo.com)."""
    return _parse_results(markup)


def _parse_lite(markup: str) -> list[dict[str, str]]:
    """DuckDuckGo lite endpoint parser (lite.duckduckgo.com).

    The lite page uses the ``rel="nofollow"`` plain-link shape, which the
    shared DDG parser already handles as its fallback."""
    return _parse_results(markup)


def _parse_mojeek(markup: str) -> list[dict[str, str]]:
    """Extract results from Mojeek's HTML search page."""
    out: list[dict[str, str]] = []
    for match in re.finditer(
        r'(?is)<a[^>]+class="title"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', markup
    ):
        href = html.unescape(match.group(1)).strip()
        title = html.unescape(_TAGS.sub("", match.group(2))).strip()
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not title:
            continue
        out.append({"url": href, "title": title, "snippet": ""})
    return out


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps outbound links in a redirect; pull the real URL out."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return target
    return href if parsed.scheme in {"http", "https"} else ""
