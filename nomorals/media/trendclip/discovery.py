"""Trend discovery — ported from the source's ``trendclip/discovery.py``.

Two discovery paths, both free and keyless:

1. **Feed discovery**: an allow-listed set of HTTPS hosts (the source's
   allow-list, kept) fetched with plain ``urllib`` — a feed item is a video
   whose page yields a title, a URL and a published time.
2. **YouTube trend search**: ``yt-dlp -J`` over a topic query, ranked by
   views-per-hour and the deterministic ``score_feed_item`` — the same
   shape the source uses, with yt-dlp provisioned through Devon's own
   ``media/resolver.ensure_ytdlp`` instead of the source's ad-hoc lookup.

Both paths take injectable collaborators (``downloader=``, ``runner=``) so
tests run offline; production wires the real network.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .scoring import score_feed_item
from .topics import TopicConfig

__all__ = [
    "ALLOWED_HOSTS",
    "discover_feeds",
    "discover_youtube_trends",
    "inspect_youtube_url",
    "rank_feed_items",
]

#: The source's allow-list: only these hosts are fetched for feed
#: discovery. Anything else is refused, never fetched.
ALLOWED_HOSTS = frozenset((
    "www.youtube.com",
    "youtube.com",
    "youtu.be",
    "www.tiktok.com",
    "tiktok.com",
    "www.instagram.com",
    "instagram.com",
    "www.facebook.com",
    "facebook.com",
))

_YTDLP_SEARCH = "ytsearch{limit}:{query}"


def _is_allowed(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in ALLOWED_HOSTS
    except Exception:  # noqa: BLE001 - a malformed URL is simply disallowed
        return False


def _token_match(text: str, keywords: list[str]) -> float:
    """Fraction of keywords present in the text (0..1)."""
    if not keywords:
        return 0.0
    haystack = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in haystack)
    return hits / len(keywords)


def _freshness(published_at: str | None, *, now: datetime | None = None,
               horizon_hours: float = 72.0) -> float:
    """Recency 0..1: 1.0 at publish time, 0.0 past the horizon."""
    if not published_at:
        return 0.0
    try:
        published = datetime.fromisoformat(
            published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    age_hours = max(0.0, (now - published).total_seconds() / 3600.0)
    return max(0.0, 1.0 - age_hours / horizon_hours)


def rank_feed_items(items: list[dict[str, Any]], topic: TopicConfig,
                    *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Rank discovered items by views-per-hour first, then the
    deterministic feed score (relevance + freshness + reliability)."""
    ranked = []
    for item in items:
        title = str(item.get("title") or "")
        relevance = _token_match(title, topic.keywords)
        freshness = _freshness(str(item.get("published_at") or ""),
                               now=now)
        reliability = 1.0 if str(item.get("channel") or "") in topic.channels \
            else 0.5
        views_per_hour = float(item.get("views_per_hour") or 0.0)
        ranked.append({
            **item,
            "relevance": round(relevance, 3),
            "freshness": round(freshness, 3),
            "feed_score": score_feed_item(
                relevance=relevance, freshness=freshness,
                reliability=reliability),
            "views_per_hour": views_per_hour,
        })
    ranked.sort(key=lambda i: (i["views_per_hour"], i["feed_score"]),
                reverse=True)
    return ranked


def discover_feeds(topic: TopicConfig, *,
                   urls: list[str] | None = None,
                   downloader: Callable[[str], str] | None = None,
                   limit: int = 20) -> list[dict[str, Any]]:
    """Fetch allow-listed feed URLs and extract video items.

    ``downloader`` is ``url -> html``; the default fetches with urllib and
    a plain user agent. A URL outside the allow-list is skipped, never
    fetched. The extractor is deliberately light (title + link + published
    time from the page's meta tags); YouTube search is the richer path.
    """
    urls = urls or []
    fetch = downloader or _default_fetch
    items: list[dict[str, Any]] = []
    for url in urls:
        if not _is_allowed(url):
            continue
        try:
            html = fetch(url)
        except Exception:  # noqa: BLE001 - one dead feed never kills discovery
            continue
        for match in re.finditer(
                r'<a[^>]+href="([^"]+)"[^>]*>([^<]{5,200})</a>', html):
            href, text = match.group(1), match.group(2).strip()
            if not _is_allowed(href):
                continue
            if _token_match(text, topic.keywords) <= 0:
                continue
            items.append({
                "title": text,
                "url": href,
                "source": "feed",
                "channel": urlparse(url).netloc,
                "published_at": _meta_published(html),
                "views_per_hour": 0.0,
            })
            if len(items) >= limit:
                return rank_feed_items(items, topic)
    return rank_feed_items(items, topic)


def _default_fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Devon-trendclip/1.0"})
    with urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _meta_title(html: str) -> str:
    match = re.search(r"<title>([^<]{1,300})</title>", html,
                      re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _meta_published(html: str) -> str:
    match = re.search(
        r'(?:article:published_time|datePublished)"?\s*content="([^"]+)"',
        html, re.IGNORECASE)
    return match.group(1) if match else ""


def discover_youtube_trends(topic: TopicConfig, *,
                            limit: int = 10,
                            runner: Callable[..., Any] | None = None,
                            ytdlp: str = "yt-dlp") -> list[dict[str, Any]]:
    """Search YouTube for a topic's trending videos via ``yt-dlp -J``.

    The query is the topic's keywords joined with spaces. Results are
    ranked by views-per-hour (views divided by age in hours, minimum 1h)
    and then by the deterministic feed score. Exclusions filter titles.
    """
    query = " ".join(topic.keywords) or topic.topic_name
    run = runner or subprocess.run
    cmd = [ytdlp, "-J", "--no-warnings", "--flat-playlist",
           _YTDLP_SEARCH.format(limit=limit, query=query)]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001 - yt-dlp missing/slow: empty, named
        return [{"error": f"yt-dlp search failed: {exc}"}]
    if getattr(proc, "returncode", 1) != 0:
        return [{"error": "yt-dlp search failed: "
                 + str(getattr(proc, "stderr", "") or "")[:300]}]
    try:
        data = json.loads(getattr(proc, "stdout", "") or "{}")
    except ValueError:
        return [{"error": "yt-dlp returned unparseable JSON"}]
    items: list[dict[str, Any]] = []
    for entry in data.get("entries") or []:
        title = str(entry.get("title") or "")
        if not title:
            continue
        if any(exc.lower() in title.lower() for exc in topic.exclusions):
            continue
        url = entry.get("url") or entry.get("webpage_url") or ""
        if entry.get("id") and not url.startswith("http"):
            url = "https://www.youtube.com/watch?v=" + str(entry["id"])
        views = float(entry.get("view_count") or 0)
        published = str(entry.get("upload_date") or "")
        age_hours = 1.0
        if len(published) == 8 and published.isdigit():
            try:
                published_dt = datetime(
                    int(published[:4]), int(published[4:6]),
                    int(published[6:8]), tzinfo=UTC)
                age_hours = max(
                    1.0,
                    (datetime.now(UTC) - published_dt)
                    .total_seconds() / 3600.0)
            except ValueError:
                pass
        items.append({
            "title": title,
            "url": url,
            "source": "youtube",
            "channel": str(entry.get("channel") or entry.get("uploader")
                           or ""),
            "published_at": published,
            "view_count": views,
            "views_per_hour": round(views / age_hours, 1),
        })
    return rank_feed_items(items[:limit], topic)


def inspect_youtube_url(url: str, *,
                        runner: Callable[..., Any] | None = None,
                        ytdlp: str = "yt-dlp") -> dict[str, Any]:
    """Metadata for one YouTube URL via ``yt-dlp -J`` (no download)."""
    if not _is_allowed(url):
        return {"ok": False, "error": "URL is not an allowed video host"}
    run = runner or subprocess.run
    try:
        proc = run([ytdlp, "-J", "--no-warnings", url],
                   capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"yt-dlp failed: {exc}"}
    if getattr(proc, "returncode", 1) != 0:
        return {"ok": False,
                "error": "yt-dlp failed: "
                + str(getattr(proc, "stderr", "") or "")[:300]}
    try:
        data = json.loads(getattr(proc, "stdout", "") or "{}")
    except ValueError:
        return {"ok": False, "error": "yt-dlp returned unparseable JSON"}
    return {
        "ok": True,
        "title": str(data.get("title") or ""),
        "uploader": str(data.get("uploader") or ""),
        "duration": float(data.get("duration") or 0.0),
        "view_count": int(data.get("view_count") or 0),
        "url": str(data.get("webpage_url") or url),
    }
