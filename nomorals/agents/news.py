"""News system: a news sub-agent + a summarizer sub-agent.

* **Fetcher** — keyless RSS feeds (60+ across world / tech / nigeria /
  africa / business / science / sports / usa / europe / asia / wires /
  entertainment — see ``FEED_CATEGORIES``) over the proxy-aware search
  client, robots-aware, deduped by URL + normalized title in
  ``news_items``.  MANY sources, never 1–2: every run fetches ALL
  default feeds, one dead feed never kills the digest, and the result
  reports ``sources_tried`` / ``sources_ok`` / per-source counts so the
  user sees the breadth.
* **Summarizer** — model-sourced when a live model is answering, extractive
  one-liners otherwise. Never invents a headline it didn't read.
* **Digest** — top N items per source, delivered through the notifier.

Every feed URL was live-fetched and classified 2026-10-18 (see
``docs/source-verification-2026-10-18.md``): 9 dead feeds removed
(404 / 200-but-HTML), 7 verified-live replacements added. Feeds that
answer bot-wall / 5xx / connection-reset from the verifier network are
kept — they work from other networks — and ``agency.sourcehealth``
re-checks them at runtime with a circuit breaker.
The legacy wire services (Reuters/AP/AFP/…) no longer publish public RSS;
``wires`` carries Reuters via its site-scoped aggregator query
(live-verified 2026-10-22 — every item attributed to reuters.com via its
``<source>`` element) plus the working global aggregators.

Coverage is GLOBAL, not Nigeria-limited: Nigeria/Africa is ONE region
among many. Every region — Americas, Europe, Asia, Africa, Middle East,
Oceania — has its own outlets in the default digest (see FEED_REGIONS).
"""

from __future__ import annotations

import html as _html
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

from ..llm.brain import brain_for
from ..core.ids import new_id
from ..core.http import HttpClient
from .search.engine import SearchEngine

__all__ = ["DEFAULT_FEEDS", "DEFAULT_CATEGORIES", "FEED_CATEGORIES",
           "FEED_REGIONS", "NEWS_GLOBAL_REGIONS", "feeds_for",
           "news_covered_regions", "feed_health_key",
           "NewsAgent", "parse_feed"]

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
        ("CBC World", "https://www.cbc.ca/cmlink/rss-world"),
        ("ABC News", "https://abcnews.go.com/abcnews/topstories"),
        ("CBS News", "https://www.cbsnews.com/latest/rss/main"),
        ("NBC News", "https://feeds.nbcnews.com/nbcnews/public/news"),
    ),
    "tech": (
        ("BBC Tech", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("Hacker News", "https://hnrss.org/frontpage"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://arstechnica.com/feed/"),
        ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
        ("Wired", "https://www.wired.com/feed/rss"),
        ("Engadget", "https://www.engadget.com/rss.xml"),
        ("Gizmodo", "https://gizmodo.com/rss"),
        ("9to5Google", "https://9to5google.com/feed/"),
    ),
    "nigeria": (
        ("Punch", "https://rss.punchng.com/v1/category/latest_news"),
        ("Vanguard", "https://www.vanguardngr.com/feed/"),
        ("Premium Times", "https://www.premiumtimesng.com/feed"),
        ("Channels TV", "https://www.channelstv.com/feed/"),
        ("Sahara Reporters", "https://saharareporters.com/rss.xml"),
        ("TechCabal", "https://techcabal.com/feed/"),
        ("Guardian NG", "https://guardian.ng/feed/"),
        ("Daily Trust", "https://dailytrust.com/feed/"),
        ("TheCable", "https://www.thecable.ng/feed"),
        ("Leadership", "https://leadership.ng/feed/"),
        ("BusinessDay", "https://businessday.ng/feed/"),
    ),
    "africa": (
        ("BBC Africa", "https://feeds.bbci.co.uk/news/world/africa/rss.xml"),
        ("Joy Online", "https://joyonline.org/feed/"),
        ("Nation Africa", "https://nation.africa/feed"),
        ("Nairaland", "https://www.nairaland.com/feed"),
        ("AfricaNews", "https://www.africanews.com/feed/rss"),
    ),
    "business": (
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("Business Insider", "https://feeds.businessinsider.com/custom/all"),
        ("Nairametrics", "https://nairametrics.com/feed/"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("The Hindu Business", "https://www.thehindu.com/business/feeder/default.rss"),
        ("FT Markets", "https://www.ft.com/rss/home"),
    ),
    "science": (
        ("BBC Science", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
        ("ScienceDaily", "https://www.sciencedaily.com/rss/all.xml"),
        ("New Scientist", "https://www.newscientist.com/feed/"),
        ("NASA", "https://www.nasa.gov/news-release/feed/"),
        ("Nature", "https://www.nature.com/nature.rss"),
        ("Quanta", "https://www.quantamagazine.org/feed/"),
    ),
    "sports": (
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("CBS Sports", "https://www.cbssports.com/rss/headlines/"),
        ("Sky Sports", "https://www.skysports.com/rss/12040"),
        ("Complete Sports", "https://www.completesports.com/feed/"),
        ("ESPN", "https://www.espn.com/espn/rss/news"),
        ("ESPN FC", "https://www.espn.com/espn/rss/soccer/news"),
    ),
    "usa": (
        ("CNN", "http://rss.cnn.com/rss/edition.rss"),
        ("NYT", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
        ("PBS NewsHour", "https://www.pbs.org/newshour/feeds/rss/headlines"),
        ("Washington Post", "https://feeds.washingtonpost.com/rss/national"),
        ("Fox News", "http://feeds.foxnews.com/foxnews/latest"),
        ("USA Today", "https://rssfeeds.usatoday.com/usatoday-NewsTopStories"),
    ),
    "europe": (
        ("The Guardian", "https://www.theguardian.com/world/rss"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/home.xml"),
        ("Euronews", "https://www.euronews.com/rss"),
        ("Le Monde", "https://www.lemonde.fr/rss/une.xml"),
        ("Der Spiegel", "https://www.spiegel.de/international/index.rss"),
        ("RTÉ", "https://www.rte.ie/rss/news.xml"),
        ("NOS", "https://feeds.nos.nl/nosnieuwsalgemeen"),
    ),
    "asia": (
        ("Straits Times", "https://www.straitstimes.com/news/world/rss.xml"),
        ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
        ("Hong Kong FP", "https://hongkongfp.com/feed/"),
        ("The Diplomat", "https://thediplomat.com/feed/"),
        ("Channel News Asia",
         "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml"),
        ("Japan Times", "https://www.japantimes.co.jp/news/feed/"),
        ("SCMP", "https://www.scmp.com/rss/91/feed"),
        ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
        ("Geo News", "https://www.geo.tv/rss/1/0"),
    ),
    "latam": (
        ("El País", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"),
        ("G1", "https://g1.globo.com/rss/g1/"),
        ("Clarín", "https://www.clarin.com/rss/lo-ultimo/"),
    ),
    "middle_east": (
        ("Arab News", "https://www.arabnews.com/rss.xml"),
        ("Times of Israel", "https://www.timesofisrael.com/feed/"),
        ("Middle East Eye", "https://www.middleeasteye.net/rss"),
        ("Al Monitor", "https://www.al-monitor.com/rss"),
    ),
    "oceania": (
        ("ABC Australia", "https://www.abc.net.au/news/feed/1"),
        ("Sydney Morning Herald", "https://www.smh.com.au/rss/feed.xml"),
        ("Stuff", "https://www.stuff.co.nz/rss"),
    ),
    "americas": (
        ("CBC Top Stories", "https://www.cbc.ca/cmlink/rss-topstories"),
        ("La Nación", "https://www.lanacion.com.ar/arc/outboundfeeds/rss/"),
        ("Folha de S.Paulo", "https://feeds.folha.uol.com.br/emcimadahora/rss091.xml"),
    ),
    "entertainment": (
        ("BBC Entertainment", "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"),
        ("Variety", "https://variety.com/feed/"),
        ("Hollywood Reporter", "https://www.hollywoodreporter.com/feed/"),
        ("Deadline", "https://deadline.com/feed/"),
    ),
    # The legacy wires (Reuters/AP/AFP/…) publish no DIRECT public RSS
    # any more; verified dead/walled 2026-10-01. Reuters is carried via
    # its site-scoped Google News query — live-verified 2026-10-22
    # (~100 items, newest <1h old, every item's <source> is
    # reuters.com). Item links are Google News redirects (unwrapped at
    # open time by core/http.unwrap_redirect_url) and items attribute
    # to the publisher via parse_feed's <source> reading. The rest are
    # the working global wire-style aggregators.
    "wires": (
        ("Reuters",
         "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US&gl=US&ceid=US:en"),
        ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
        ("Google News World",
         "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"),
        ("Google News Nigeria",
         "https://news.google.com/rss?hl=en-NG&gl=NG&ceid=NG:en"),
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


#: world + tech + business + regional spread (usa/europe/asia/nigeria/
#: africa) + entertainment — the general default digest.
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "world", "tech", "business", "usa", "europe", "asia", "nigeria",
    "africa", "latam", "middle_east", "oceania", "americas",
    "entertainment", "wires",
)

#: Sane keyless defaults when settings.news.feeds is empty.
#: Categorized so callers can subscribe by interest (``feeds_for``);
#: ``DEFAULT_FEEDS`` is the general-interest flattening.
DEFAULT_FEEDS: tuple[tuple[str, str], ...] = tuple(
    feeds_for(*DEFAULT_CATEGORIES))

#: The six world regions coverage is measured against.  Nigeria/Africa
#: is ONE region among many — the owner rule is GLOBAL coverage.
NEWS_GLOBAL_REGIONS: tuple[str, ...] = (
    "americas", "europe", "asia", "africa", "middle_east", "oceania",
)

#: Region each feed category covers.  "global" = worldwide outlets
#: (covers every region in NEWS_GLOBAL_REGIONS).
FEED_REGIONS: dict[str, str] = {
    "world": "global",
    "tech": "global",
    "nigeria": "africa",
    "africa": "africa",
    "business": "global",
    "science": "global",
    "sports": "global",
    "usa": "americas",
    "europe": "europe",
    "asia": "asia",
    "entertainment": "global",
    "wires": "global",
    "latam": "americas",
    "middle_east": "middle_east",
    "oceania": "oceania",
    "americas": "americas",
}


def news_covered_regions(categories: tuple[str, ...] | None = None) -> set[str]:
    """World regions the default digest covers: "global" expands to all."""
    out: set[str] = set()
    for cat in (categories if categories is not None else DEFAULT_CATEGORIES):
        region = FEED_REGIONS.get(cat)
        if region == "global":
            out.update(NEWS_GLOBAL_REGIONS)
        elif region in NEWS_GLOBAL_REGIONS:
            out.add(region)
    return out


#: Reverse index feed-URL → registry health key. Built lazily; the key
#: scheme MUST match ``agency.sources_registry._collect_news``
#: (``news:{category}:{name}``) so the run path and the scheduled
#: SourceHealth re-verification read and write the SAME records —
#: otherwise stood-down feeds would never be re-activated.
_FEED_URL_TO_HEALTH_KEY: dict[str, str] | None = None


def feed_health_key(source: str, url: str) -> str:
    """Health-record key for one feed, matching the sources registry.

    Custom (settings-configured) feeds not in FEED_CATEGORIES get a
    URL-based key under ``news:custom:`` — still tracked, just never
    re-verified by the registry collector.
    """
    global _FEED_URL_TO_HEALTH_KEY
    if _FEED_URL_TO_HEALTH_KEY is None:
        index: dict[str, str] = {}
        for cat, feeds in FEED_CATEGORIES.items():
            for name, feed_url in feeds:
                index[feed_url] = f"news:{cat}:{name}"
        _FEED_URL_TO_HEALTH_KEY = index
    hit = (_FEED_URL_TO_HEALTH_KEY or {}).get(url)
    if hit:
        return hit
    return f"news:custom:{url}"

_TAG = re.compile(r"<[^>]+>")


def _strip(markup: str, limit: int = 500) -> str:
    text = _TAG.sub(" ", markup or "")
    text = _html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _short_summary(summary: str, limit: int = 200) -> str:
    """Shorten ``summary`` to ``limit`` chars without cutting mid-sentence.

    Keeps whole sentences that fit; if the first sentence alone exceeds
    the limit, falls back to the last word boundary. No mid-word cuts,
    no mid-sentence cuts when avoidable.
    """
    text = (summary or "").strip()
    if len(text) <= limit:
        return text
    # whole sentences that fit entirely
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    kept = ""
    for sent in sentences:
        candidate = f"{kept} {sent}".strip() if kept else sent
        if len(candidate) <= limit:
            kept = candidate
        else:
            break
    if kept:
        return kept
    # single over-long sentence: cut at the last space before the limit
    cut = text.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip() + "…"


def parse_feed(markup: str) -> list[dict[str, str]]:
    """Parse RSS/Atom-ish markup into item dicts.

    Keys: title, url, summary, source_ts — plus ``source_name`` /
    ``source_url`` from the item's ``<source>`` element when present:
    the PUBLISHER of an aggregator-carried item. Google News item links
    are redirects (news.google.com); without <source> the item would
    attribute to the aggregator instead of the outlet that wrote it
    (reuters.com, trust tier 0.80). Feeds without <source> yield ""
    for both, exactly as before.
    """
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(markup)
    except ET.ParseError:
        return items
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag not in {"item", "entry"}:
            continue
        item: dict[str, str] = {"title": "", "url": "", "summary": "",
                                "source_ts": 0.0,
                                "source_name": "", "source_url": ""}
        for child in node:
            ctag = child.tag.rsplit("}", 1)[-1].lower()
            text = (child.text or "").strip()
            if ctag == "title":
                item["title"] = _strip(text)
            elif ctag == "link":
                item["url"] = text or (child.get("href") or "")
            elif ctag == "source" and not item["source_url"] \
                    and not item["source_name"]:
                # RSS: <source url="https://www.reuters.com">Reuters</source>
                item["source_url"] = (child.get("url") or "").strip()
                item["source_name"] = _strip(text)
                if not item["source_name"] or not item["source_url"]:
                    # Atom <source> nests <title> / <link> children.
                    for sub in child:
                        stag = sub.tag.rsplit("}", 1)[-1].lower()
                        stext = (sub.text or "").strip()
                        if stag == "title" and not item["source_name"]:
                            item["source_name"] = _strip(stext)
                        elif stag == "link" and not item["source_url"]:
                            item["source_url"] = stext or (
                                sub.get("href") or "")
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
            response = brain_for(self.context).chat([Message(role="user", content=prompt)], task_kind="summarize")
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
        from ..agency.sourcehealth import SourceHealth, SourceSpec, RSS

        settings_news = getattr(self.settings, "news", None)
        cap = int(cap or getattr(settings_news, "cap", 12))
        cap = max(1, min(cap, 40))
        fetched: list[dict[str, str]] = []
        errors: list[str] = []
        per_source: dict[str, int] = {}
        sources_ok: list[str] = []
        stood_down: list[str] = []
        feeds = self._feeds()
        # Circuit breaker: a feed the health table has stood down (or
        # disabled) is skipped, not retried forever — the scheduled
        # SourceHealth re-verification re-activates it on recovery.
        # Every attempt here is ALSO recorded, so run-path failures
        # count toward stand-down even between scheduled checks.
        health = SourceHealth(self.db) if self.db is not None else None
        for source, url in feeds:
            spec = SourceSpec(key=feed_health_key(source, url),
                              category="news", name=source,
                              url=url, kind=RSS)
            if health is not None:
                rec = health.record(spec.key)
                if rec is not None and rec.status in ("stood-down",
                                                      "disabled"):
                    stood_down.append(f"{source} ({rec.status})")
                    continue
            try:
                # RSS Accept headers: several feeds (CBS, ESPN, PBS,
                # Engadget — verified 2026-10-18) reject the generic
                # Accept with 406/202/403 but serve 200 + items when
                # asked for RSS explicitly.
                response = self._client.get(url, headers={
                    "Accept": "application/rss+xml, "
                              "application/atom+xml, "
                              "application/xml;q=0.9, text/xml;q=0.8",
                })
                items = parse_feed(response.text)
            except Exception as exc:  # noqa: BLE001 - one dead feed must not kill the digest
                errors.append(f"{source}: {exc}")
                if health is not None:
                    health.note_result(spec, ok=False,
                                       error=str(exc)[:200])
                continue
            if not items:
                # HTTP 200 but zero usable items: unusable as a source —
                # a health failure (counts toward stand-down), though the
                # digest itself still degrades gracefully below.
                if health is not None:
                    health.note_result(spec, ok=False,
                                       error="zero usable items", http=200)
            elif health is not None:
                health.note_result(spec, ok=True, items=len(items),
                                   http=200)
            sources_ok.append(source)
            per_source[source] = len(items)
            for it in items:
                it["source"] = source
                fetched.append(it)

        # Dedupe by URL AND by normalized title — the same story carried
        # by 6 outlets must not flood the digest as 6 "different" items.
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        deduped: list[dict[str, str]] = []
        for it in fetched:
            url_key = (it.get("url") or "").strip().lower().rstrip("/")
            title_key = re.sub(r"\W+", " ", (it.get("title") or "").lower()).strip()
            if url_key and url_key in seen_urls:
                continue
            if title_key and title_key in seen_titles:
                continue
            if url_key:
                seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)
            deduped.append(it)
        fetched = deduped

        # Cross-run memory: news_items is the organ's seen-store, and
        # its contract (module docstring) is dedupe by URL + normalized
        # title. URL alone is NOT a stable identity for
        # aggregator-carried stories: the same Reuters story arrives
        # via the generic Google News feed and via the Reuters
        # site-scoped feed under DIFFERENT redirect URLs, so a URL-only
        # check delivers it twice when the two routes surface it in
        # different runs. A story whose normalized title was delivered
        # recently is known under either route. (For aggregator items
        # the normalized title embeds the publisher — "… - Reuters" —
        # so this is effectively source+title.) The window is bounded:
        # a genuinely new story reusing an old headline after a
        # fortnight is news again.
        known_titles: set[str] = set()
        if self.db is not None:
            try:
                rows = self.db.query(
                    "SELECT title FROM news_items WHERE created_at >= ?",
                    (time.time() - 14 * 86400,))
                for row in rows or []:
                    raw = (row.get("title") if hasattr(row, "get")
                           else row[0]) or ""
                    key = re.sub(r"\W+", " ", str(raw).lower()).strip()
                    if key:
                        known_titles.add(key)
            except Exception:  # noqa: BLE001
                known_titles = set()

        fresh: list[dict[str, str]] = []
        for it in fetched:
            if self.db is not None:
                try:
                    known = self.db.query_one("SELECT id FROM news_items WHERE url = ?", (it["url"],))
                except Exception:  # noqa: BLE001
                    known = None
                if known is not None:
                    continue
                title_key = re.sub(
                    r"\W+", " ", (it.get("title") or "").lower()).strip()
                if title_key and title_key in known_titles:
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
            # every selected item is shown — the old [:4]-per-source slice
            # silently dropped stories that made it into `top`.
            for it in items:
                summary = summaries.get(it["url"]) or it.get("summary") or ""
                lines.append(f"  • {it['title']}" + (
                    f" — {_short_summary(summary)}" if summary else ""))
        digest = "\n".join(lines)
        result = {
            "ok": bool(top),
            "items": len(top),
            "fresh": len(fresh),
            "errors": errors,
            "digest": digest,
            # Visible breadth: the user sees HOW MANY sources were tried
            # and which ones answered — never a silent 1–2. Stood-down
            # feeds are reported, not silently dropped.
            "sources_tried": len(feeds),
            "sources_ok": sources_ok,
            "per_source": per_source,
            "stood_down": stood_down,
        }
        if self.notifier is not None and top:
            # full digest — the gateway chunks it; no [:3500] cut.
            self.notifier.publish("news", f"news digest — {len(top)} new items", digest)
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
