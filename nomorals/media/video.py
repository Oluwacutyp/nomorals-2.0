"""VideoFinder — locate videos across the open web, with real metadata.

Pipeline:
1. **search** — runs the existing multi-engine web search (DuckDuckGo →
   Bing → Mojeek) with video-oriented query variants, optionally pinned to
   a platform (``platform="youtube"`` adds ``site:youtube.com``).
2. **rank** — results are scored: video-URL patterns (watch?v=, youtu.be,
   vimeo, dailymotion, tiktok, …) beat page URLs; title/snippet overlap
   with the query beats loose matches.
3. **enrich** — YouTube results are enriched via oEmbed (author,
   thumbnail) and, when the ``media_probe`` tool is available, via yt-dlp
   metadata (duration, formats).  Enrichment is best-effort: every failure
   leaves the result intact, just thinner.

    from nomorals.media.video import VideoFinder
    finder = VideoFinder(context)
    hits = finder.find("lofi beats for studying", max_results=8)
    hits[0]  # {title, url, source, snippet, author, thumbnail, duration…}

    finder.download(hits[0]["url"])   # real file via yt-dlp/direct HTTP

Registered as the ``video_finder`` tool.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

from ..core.http import HttpClient
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["VideoFinder", "register"]

#: (source, compiled URL pattern) — order = rank weight
_VIDEO_SOURCES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("youtube", re.compile(
        r"(?:youtube\.com/(?:watch\?[^ ]*v=|shorts/|embed/)|youtu\.be/)[A-Za-z0-9_\-]{4,}",
        re.I)),
    ("vimeo", re.compile(r"vimeo\.com/(\d+|[A-Za-z0-9./\-]+)", re.I)),
    ("dailymotion", re.compile(r"dailymotion\.com/(video|embed/video)/[A-Za-z0-9]+",
                               re.I)),
    ("tiktok", re.compile(r"tiktok\.com/(@[A-Za-z0-9._]+/video/)?\d+", re.I)),
    ("twitch", re.compile(r"twitch\.tv/videos/\d+", re.I)),
    ("rumble", re.compile(r"rumble\.com/(?:embed/|v)?\d+", re.I)),
    ("pinterest", re.compile(r"pin\.it/\d+", re.I)),
)

#: sites where a bare page URL is still very likely a video
_VIDEO_PAGES = ("youtube.com/watch", "youtu.be", "vimeo.com",
                "dailymotion.com/video", "tiktok.com")

_PLATFORM_SITES = {
    "youtube": "youtube.com",
    "vimeo": "vimeo.com",
    "tiktok": "tiktok.com",
    "dailymotion": "dailymotion.com",
    "twitch": "twitch.tv",
    "rumble": "rumble.com",
}


def _video_score(url: str) -> tuple[float, str]:
    """(score, source) for a URL — higher is more clearly a video."""
    for i, (source, pat) in enumerate(_VIDEO_SOURCES):
        if pat.search(url):
            return (10.0 + i * 0.1), source
    for page in _VIDEO_PAGES:
        if page in url:
            return (6.0, page.split(".")[0])
    return (0.0, "")


def _relevance(text: str, query: str) -> float:
    """0..1 — fraction of meaningful query words present in text."""
    q_words = [w for w in re.findall(r"[a-z0-9']+", query.lower())
               if len(w) > 2]
    if not q_words:
        return 0.5
    t = text.lower()
    hits = sum(1 for w in q_words if w in t)
    return hits / len(q_words)


class VideoFinder:
    """Search, rank, enrich, and download videos."""

    role = "video"

    def __init__(self, context: Any) -> None:
        self.context = context

    # ── search + rank ─────────────────────────────────────────────────────
    def find(self, query: str, *, max_results: int = 10, platform: str = "",
             freshness: str = "") -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ToolError("video_finder needs a query")
        max_results = max(1, min(int(max_results), 30))
        platform = (platform or "").strip().lower()
        site = _PLATFORM_SITES.get(platform, "")

        raw = self._search(query, site=site, freshness=freshness)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in raw:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            vscore, source = _video_score(url)
            rel = _relevance(f"{r.get('title', '')} {r.get('snippet', '')}",
                             query)
            entry: dict[str, Any] = {
                "title": (r.get("title") or "").strip(),
                "url": url,
                "snippet": (r.get("snippet") or "").strip()[:400],
                "source": source or self._domain(url),
                "score": round(vscore + rel * 4.0, 3),
                "is_video_url": vscore > 0,
            }
            if platform and site not in url and not entry["is_video_url"]:
                continue  # platform pin: drop non-matching page results
            results.append(entry)
        results.sort(key=lambda e: -e["score"])
        top = results[:max_results]
        self._enrich(top)
        return {"query": query, "count": len(top), "results": top}

    def _search(self, query: str, *, site: str = "",
                freshness: str = "") -> list[dict[str, str]]:
        tools = getattr(self.context, "tools", None)
        call = getattr(tools, "call", None) if tools else None
        if call is not None:
            out = call("web_search", query=f"{query} video"
                       if not site else query,
                       site=site, freshness=freshness, max_results=12)
            if out.ok:
                return list(out.unwrap().get("results", []))
        # direct fallback (same engines, same parsers, no registry needed)
        return web_search_impl(f"{query} video" if not site else query,
                               max_results=12, site=site,
                               freshness=freshness)

    @staticmethod
    def _domain(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            return host[4:] if host.startswith("www.") else host
        except ValueError:
            return ""

    # ── enrichment ────────────────────────────────────────────────────────
    def _enrich(self, results: list[dict[str, Any]]) -> None:
        for entry in results:
            url = entry["url"]
            if "youtube" in entry["source"] or "youtu.be" in url:
                self._enrich_youtube(entry)
            self._enrich_probe(entry)

    def _enrich_youtube(self, entry: dict[str, Any]) -> None:
        vid = self._youtube_id(entry["url"])
        if not vid:
            return
        try:
            oembed = (f"https://www.youtube.com/oembed?url="
                      f"https://www.youtube.com/watch?v={vid}&format=json")
            resp = HttpClient().get(oembed, timeout=8.0)
            if getattr(resp, "ok", False):
                import json as _json

                data = _json.loads(resp.text or resp.body.decode("utf-8", "ignore"))
                if data.get("title"):
                    entry["title"] = data["title"]
                if data.get("author_name"):
                    entry["author"] = data["author_name"]
                if data.get("thumbnail_url"):
                    entry["thumbnail"] = data["thumbnail_url"]
        except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
            _log.debug("oembed failed: %s", exc)

    def _enrich_probe(self, entry: dict[str, Any]) -> None:
        tools = getattr(self.context, "tools", None)
        call = getattr(tools, "call", None) if tools else None
        if call is None:
            return
        try:
            out = call("media_probe", url=entry["url"])
            if out.ok:
                v = out.unwrap()
                for k in ("duration", "title", "formats", "uploader"):
                    if v.get(k) not in (None, "", []):
                        entry.setdefault(k, v[k])
        except Exception as exc:  # noqa: BLE001
            _log.debug("media_probe failed: %s", exc)

    @staticmethod
    def _youtube_id(url: str) -> str:
        m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_\-]{11})",
                      url)
        return m.group(1) if m else ""

    # ── download ──────────────────────────────────────────────────────────
    def download(self, url: str, *, audio_only: bool = False) -> dict[str, Any]:
        tools = getattr(self.context, "tools", None)
        call = getattr(tools, "call", None) if tools else None
        if call is None:
            raise ToolError("video download needs the tool registry")
        out = call("media_download", url=url, audio_only=audio_only)
        if not out.ok:
            raise ToolError(f"download failed: {out.error}")
        return out.unwrap()


def web_search_impl(query: str, *, max_results: int = 8, site: str = "",
                    freshness: str = "") -> list[dict[str, str]]:
    """Keyless multi-engine search without a registry (direct HttpClient)."""
    from ..tools.web import (_parse_bing, _parse_ddg, _parse_lite,
                             _parse_mojeek)

    needle = f"{query} site:{site}" if site else query
    encoded = quote_plus(needle)
    df = f"&df={freshness}" if freshness in {"d", "w", "m", "y"} else ""
    client = HttpClient()
    engines = (
        ("ddg", f"https://html.duckduckgo.com/html/?q={encoded}{df}",
         _parse_ddg),
        ("lite", f"https://lite.duckduckgo.com/lite/?q={encoded}{df}",
         _parse_lite),
        ("bing", f"https://www.bing.com/search?q={encoded}", _parse_bing),
        ("mojeek", f"https://www.mojeek.com/search?q={encoded}",
         _parse_mojeek),
    )
    for name, endpoint, parser in engines:
        try:
            response = client.get(endpoint)
        except Exception as exc:  # noqa: BLE001
            _log.debug("search endpoint %s failed: %s", name, exc)
            continue
        if not getattr(response, "ok", False):
            continue
        results = parser(response.text or "")
        if results:
            for r in results:
                r["engine"] = name
            return results[:max_results]
    return []


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "video_finder",
        description=(
            "Find videos across the open web: multi-engine search, ranked by "
            "video-URL confidence + query relevance, enriched with oEmbed "
            "(author/thumbnail) and yt-dlp metadata (duration) when "
            "available. action=find (query, max_results, platform, "
            "freshness) | download (url, as_audio) | platforms."
        ),
        capability=Capability.NET_OUT,
    )
    def video_finder(action: str = "find", query: str = "", url: str = "",
                     max_results: int = 10, platform: str = "",
                     freshness: str = "", audio_only: bool = False) -> dict[str, Any]:
        finder = VideoFinder(context)
        if action == "platforms":
            return {"platforms": sorted(_PLATFORM_SITES)}
        if action == "download":
            if not url:
                raise ToolError("video_finder download needs url=")
            return finder.download(url, audio_only=audio_only)
        if action == "find":
            return finder.find(query, max_results=max_results,
                               platform=platform, freshness=freshness)
        raise ToolError(f"unknown video_finder action {action!r}")
