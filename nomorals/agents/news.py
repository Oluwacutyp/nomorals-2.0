"""News system: a news sub-agent + a summarizer sub-agent.

* **Fetcher** — keyless RSS feeds (29 across world / tech / nigeria /
  business / science / sports — see ``FEED_CATEGORIES``) over the
  proxy-aware search client, robots-aware, deduped by URL in
  ``news_items``.
* **Summarizer** — model-sourced when a live model is answering, extractive
  one-liners otherwise. Never invents a headline it didn't read.
* **Digest** — top N items per source, delivered through the notifier.
"""

from __future__ import annotations

import html as _html
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

from ..core.ids import new_id
from ..core.http import HttpClient
from .search.engine import SearchEngine

__all__ = ["DEFAULT_FEEDS", "FEED_CATEGORIES", "feeds_for", "NewsAgent",
           "parse_feed"]

#: Sane keyless defaults when settings.news.feeds is empty.
#: Categorized so callers can subscribe by interest (``feeds_for``);
#: ``DEFAULT_FEEDS`` is the general-interest flattening.
FEED_CATEGORIES: dict[str, tuple[tuple[str, str], ...]] = {
    "world": (
        ("BBC News", "https://feeds.bbci.co.uk/news/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
        ("DW", "https://rss.dw.com/rdf/rss-en-all"),
        ("France 24", "https://www.france24.com/en/rss"),
    ),
    "tech": (
        ("BBC Tech", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("Hacker News", "https://hnrss.org/frontpage"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://arstechnica.com/feed/"),
        ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ),
    "nigeria": (
        ("Punch", "https://punchng.com/feed/"),
        ("Vanguard", "https://www.vanguardngr.com/feed/"),
        ("Premium Times", "https://www.premiumtimesng.com/feed"),
        ("Channels TV", "https://www.channelstv.com/feed/"),
        ("TheCable", "https://www.thecable.ng/feed"),
        ("TechCabal", "https://techcabal.com/feed/"),
    ),
    "business": (
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("Business Insider", "https://www.businessinsider.com/rss"),
        ("Nairametrics", "https://nairametrics.com/feed/"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ),
    "science": (
        ("BBC Science", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
        ("ScienceDaily", "https://www.sciencedaily.com/rss/all.xml"),
        ("New Scientist", "https://www.newscientist.com/feed/home"),
        ("NASA", "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
    ),
    "sports": (
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("ESPN", "https://www.espn.com/espn/rss/news"),
        ("Sky Sports", "https://www.skysports.com/rss/12040"),
        ("Complete Sports", "https://www.completesports.com/feed/"),
    ),
}

def feeds_for(*categories: str) -> list[tuple[str, str]]:
    """Union of feeds for the named categories (deduped, order kept).

    Unknown category names are ignored. No args → every category.
    """
    wanted = [c.lower() for c in categories] or list(FEED_CATEGORIES)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for cat in wanted:
        for name, url in FEED_CATEGORIES.get(cat, ()):
            if url not in seen:
                seen.add(url)
                out.append((name, url))
    return out


#: world + tech + nigeria — the general default digest.
DEFAULT_FEEDS: tuple[tuple[str, str], ...] = tuple(
    feeds_for("world", "tech", "nigeria"))

_TAG = re.compile(r"<[^>]+>")


def _strip(markup: str, limit: int = 500) -> str:
    text = _TAG.sub(" ", markup or "")
    text = _html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_feed(markup: str) -> list[dict[str, str]]:
    """Parse RSS/Atom-ish markup into [{title, url, summary, source_ts}]."""
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(markup)
    except ET.ParseError:
        return items
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag not in {"item", "entry"}:
            continue
        item: dict[str, str] = {"title": "", "url": "", "summary": "", "source_ts": 0.0}
        for child in node:
            ctag = child.tag.rsplit("}", 1)[-1].lower()
            text = (child.text or "").strip()
            if ctag == "title":
                item["title"] = _strip(text)
            elif ctag == "link":
                item["url"] = text or (child.get("href") or "")
            elif ctag in {"description", "summary", "content"} and not item["summary"]:
                item["summary"] = _strip(text)
            elif ctag in {"pubdate", "published", "updated", "dc:date"}:
                try:
                    item["source_ts"] = time.mktime(
                        time.strptime(text[:25].replace("Z", ""))
                    )
                except Exception:  # noqa: BLE001
                    pass
        if item["title"] and item["url"]:
            items.append(item)
    return items


class NewsAgent:
    """Fetches feeds, dedupes, summarizes, and delivers a digest."""

    def __init__(self, context: Any, notifier: Any = None) -> None:
        self.context = context
        self.settings = getattr(context, "settings", None)
        self.db = getattr(context, "db", None)
        self.notifier = notifier
        tools = getattr(self.settings, "tools", None)
        self._client = HttpClient(
            timeout=getattr(tools, "http_timeout", 30.0),
            user_agent=getattr(tools, "user_agent", "NoMoralsCore/0.1"),
            proxy_url=getattr(tools, "proxy_url", ""),
        )

    def _feeds(self) -> list[tuple[str, str]]:
        raw = (getattr(getattr(self.settings, "news", None), "feeds", "") or "").strip()
        if raw:
            out = []
            for part in raw.split(","):
                part = part.strip()
                if part.startswith("http"):
                    out.append((part.split("/")[-2] or part, part))
            return out or list(DEFAULT_FEEDS)
        return list(DEFAULT_FEEDS)

    def _model_summaries(self, items: list[dict[str, str]]) -> dict[str, str]:
        router = getattr(self.context, "router", None)
        if router is None:
            return {}
        from ..llm.base import Message

        lines = [f"{i}. {it['title']} — {it['summary'][:160]}" for i, it in enumerate(items[:12])]
        prompt = (
            "Summarize each news item in ONE sentence (no invented facts), "
            "numbered 1..N in the same order.\n" + "\n".join(lines)
        )
        try:
            response = router.chat([Message(role="user", content=prompt)])
            text = (getattr(response, "text", "") or "").strip()
        except Exception:  # noqa: BLE001
            return {}
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line[0] not in "0123456789":
                continue
            num, _, sentence = line.partition(".")
            try:
                idx = int(num.strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < len(items) and sentence.strip():
                out[items[idx]["url"]] = sentence.strip()
        return out

    # ── one digest cycle ─────────────────────────────────────────────────────
    def run(self, cap: int | None = None) -> dict[str, Any]:
        settings_news = getattr(self.settings, "news", None)
        cap = int(cap or getattr(settings_news, "cap", 12))
        cap = max(1, min(cap, 40))
        fetched: list[dict[str, str]] = []
        errors: list[str] = []
        for source, url in self._feeds():
            try:
                response = self._client.get(url)
                items = parse_feed(response.text)
            except Exception as exc:  # noqa: BLE001 - one dead feed must not kill the digest
                errors.append(f"{source}: {exc}")
                continue
            for it in items:
                it["source"] = source
                fetched.append(it)

        fresh: list[dict[str, str]] = []
        for it in fetched:
            if self.db is not None:
                try:
                    known = self.db.query_one("SELECT id FROM news_items WHERE url = ?", (it["url"],))
                except Exception:  # noqa: BLE001
                    known = None
                if known is not None:
                    continue
            fresh.append(it)
        fresh.sort(key=lambda i: float(i.get("source_ts") or 0), reverse=True)
        top = fresh[:cap]
        summaries = self._model_summaries(top)
        if self.db is not None:
            try:
                with self.db.transaction():
                    for it in top:
                        self.db.execute(
                            "INSERT OR IGNORE INTO news_items (id, source, title, url, summary, published, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (new_id(), it["source"], it["title"], it["url"],
                             it.get("summary", "")[:400], float(it.get("source_ts") or 0), time.time()),
                        )
            except Exception:  # noqa: BLE001
                pass
        lines: list[str] = []
        by_source: dict[str, list[dict[str, str]]] = {}
        for it in top:
            by_source.setdefault(it["source"], []).append(it)
        for source, items in by_source.items():
            lines.append(f"■ {source}")
            for it in items[:4]:
                summary = summaries.get(it["url"]) or it.get("summary") or ""
                lines.append(f"  • {it['title']}" + (f" — {summary[:200]}" if summary else ""))
        digest = "\n".join(lines)
        result = {
            "ok": bool(top),
            "items": len(top),
            "fresh": len(fresh),
            "errors": errors,
            "digest": digest,
        }
        if self.notifier is not None and top:
            self.notifier.publish("news", f"news digest — {len(top)} new items", digest[:3500])
        return result

    def recent(self, limit: int = 15) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM news_items ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        except Exception:  # noqa: BLE001
            return []
