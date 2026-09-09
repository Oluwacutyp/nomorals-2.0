"""Web tools: fetch, search, and HTML→text.

Search uses DuckDuckGo's HTML endpoint and lite.duckduckgo.com, which need no API
key. That is a deliberate choice: a personal AI should not require you to register
five accounts before it can look something up.

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

__all__ = ["html_to_text", "register"]

_log = get_logger(__name__)

_SCRIPT_STYLE = re.compile(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]*>")
_ENTITIES = re.compile(r"&(?!#?\w+;)")


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


def register(registry: Any) -> None:
    """Attach the web tools to a registry."""
    context = registry.context
    settings = getattr(context, "settings", None) if context is not None else None
    tools_settings = getattr(settings, "tools", None) if settings else None
    user_agent = getattr(tools_settings, "user_agent", "NoMoralsCore/0.1") if tools_settings else "NoMoralsCore/0.1"
    timeout = getattr(tools_settings, "http_timeout", 30.0) if tools_settings else 30.0
    respect_robots = getattr(tools_settings, "robots_txt", True) if tools_settings else True
    client = HttpClient(timeout=timeout, user_agent=user_agent)

    @registry.register(
        "web_fetch",
        description="Fetch a URL and return its text content.",
        capability=Capability.NET_OUT,
    )
    def web_fetch(url: str, *, max_chars: int = 40000, raw: bool = False) -> dict[str, Any]:
        if respect_robots and not _robots.allowed(url, user_agent, client):
            raise ToolError(f"robots.txt disallows {url}")
        response = client.get(url)
        body = response.text
        title = ""
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
        if match:
            title = html.unescape(_TAGS.sub("", match.group(1))).strip()
        text = body if raw else html_to_text(body)
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
        "web_search",
        description="Search the web (DuckDuckGo HTML, no API key) and return ranked links.",
        capability=Capability.NET_OUT,
    )
    def web_search(query: str, *, max_results: int = 8, site: str = "") -> dict[str, Any]:
        needle = f"{query} site:{site}" if site else query
        encoded = urllib.parse.quote_plus(needle)
        results: list[dict[str, str]] = []
        for endpoint in (
            f"https://html.duckduckgo.com/html/?q={encoded}",
            f"https://lite.duckduckgo.com/lite/?q={encoded}",
        ):
            try:
                response = client.get(endpoint)
            except Exception as exc:  # noqa: BLE001 - try the next endpoint
                _log.debug("search endpoint %s failed: %s", endpoint, classify(exc).message)
                continue
            results = _parse_results(response.text)
            if results:
                break
        return {"query": needle, "count": len(results[:max_results]), "results": results[:max_results]}

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
