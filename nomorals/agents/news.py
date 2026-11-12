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
* **Digest** — top N items grouped by PUBLISHER when the item carries
  one (``<source>`` attribution, pass #80/#83), by carrying feed
  otherwise, delivered through the notifier.  Stored rows persist the
  attribution (``news_items.publisher`` / ``publisher_url``) alongside
  the feed name in ``source`` (provenance), so stored-path consumers
  credit the same publisher the live digest did.

* **Continuous poll + interest tiers (pass #85)** — fetch and delivery
  are separate acts.  ``poll()`` fetches and stores EVERY fresh story
  (no cap) as ``undelivered``; the brain then scores each story 0–1
  against the owner's interest profile (seed beats + pins, persisted
  in ``news_interest``).  High interest is pushed anytime — through
  the notifier, held during quiet hours and under a per-day cap;
  mid interest waits for the digest; low interest is archived
  (stored, queryable, never delivered).  ``brief()`` composes the
  digest from ALL undelivered stories — nothing is silently dropped —
  and marks exactly those delivered.  Every stored row carries an
  honest ``delivery_state`` (``undelivered`` / ``pushed`` /
  ``digested`` / ``archived``); rows stored before this pass have a
  NULL state and count as already delivered (the old run() published
  what it stored).  With no brain available, scoring degrades
  honestly: nothing is pushed, scores stay NULL, and ``status()``
  reports interest ``unavailable`` — scores are never fabricated.

AA-55 (assembly audit, open-sources law): the feed list grows.
Publisher hosts encountered in fetched items (an item's ``<source>``
publisher URL, else the item's own host — aggregator pages surface
their publishers exactly this way) intake as untrusted candidates
in the ONE media source registry (kinds ``("news",)``); the shared
health loop promotes live hosts, and promoted feeds join
:meth:`NewsAgent._feeds` alongside the seeds through the generic
``https://<host>/feed/`` shape, their poll outcomes recorded back
into the registry so dead growth is stood down and shed by the
same loop.

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

import calendar
import email.utils
import html as _html
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

from ..llm.brain import brain_for
from ..core.ids import new_id
from ..core.http import HttpClient
from .search.engine import SearchEngine
from .failures import record_failure

__all__ = ["DEFAULT_FEEDS", "DEFAULT_CATEGORIES", "FEED_CATEGORIES",
           "FEED_REGIONS", "NEWS_GLOBAL_REGIONS", "feeds_for",
           "news_covered_regions", "feed_health_key",
           "NewsAgent", "parse_feed", "publisher_of"]

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


def _parse_feed_date(text: str) -> float:
    """Parse a feed-item date to an epoch, or 0.0 when unparseable.

    Real feeds publish RFC-822 (``Sat, 24 Oct 2026 10:00:00 GMT``) or
    ISO-8601 (``2026-10-24T10:00:00Z``) dates; the old code ran
    ``time.strptime(text[:25])`` with no format, so only ctime-shaped
    text parsed and every other date silently became 0.0 — the
    newest-first ordering in :meth:`NewsAgent.poll` was lost. Order:
    the legacy ctime fallback first (byte-identical to the old path),
    then RFC-822, then ISO-8601; naive datetimes read as UTC.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        return time.mktime(time.strptime(text[:25].replace("Z", "")))
    except Exception:  # noqa: BLE001
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            return calendar.timegm(parsed.timetuple())
        return parsed.timestamp()
    try:
        iso = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if iso.tzinfo is None:
        return calendar.timegm(iso.timetuple())
    return iso.timestamp()


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
                item["source_ts"] = _parse_feed_date(text)
        if item["title"] and item["url"]:
            items.append(item)
    return items


# ── stored attribution (pass #83) ─────────────────────────────────────
#
# ``news_items.source`` has always held the CARRYING FEED's display name
# and still does — it is provenance ("carried by the Google News wires
# feed"), and feed-health records key on it.  What it never held is the
# PUBLISHER of an aggregator-carried item, which pass #80 taught
# ``parse_feed`` to read from the item's ``<source>`` element.  These
# two additive, nullable columns persist that attribution at ingest;
# rows written before they existed read NULL and every consumer falls
# back to the legacy feed-name behaviour, byte-identically.

_ATTRIBUTION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("publisher", "TEXT"),
    ("publisher_url", "TEXT"),
)


def _ensure_attribution_columns(db: Any) -> None:
    """Lazily add the attribution columns (pass #71's pattern).

    Best-effort and idempotent: ``ALTER TABLE`` per column, swallowing
    the duplicate-column error once the column exists.  A failure here
    must never stop news — an un-migrated table simply stores rows the
    legacy way and attribution resumes on the next construction.
    """
    if db is None:
        return
    for name, decl in _ATTRIBUTION_COLUMNS:
        try:
            db.execute(
                f"ALTER TABLE news_items ADD COLUMN {name} {decl}")
        except Exception as exc:  # noqa: BLE001
            if "duplicate column" not in str(exc).lower():
                # table absent (pre-migration db) or a transient lock:
                # nothing more to do — storage fails open as before.
                return


# ── delivery state + interest profile (pass #85) ──────────────────────
#
# ``news_items`` rows gain a delivery state machine.  Before this pass
# storage WAS the seen-marker: run() stored exactly the capped digest
# selection, so "stored" and "delivered" were one fact and stories
# beyond the cap vanished unaudited.  Now poll() stores everything as
# STATE_UNDELIVERED and delivery is a recorded transition.

STATE_UNDELIVERED = "undelivered"
STATE_PUSHED = "pushed"
STATE_DIGESTED = "digested"
STATE_ARCHIVED = "archived"

_DELIVERY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("delivery_state", "TEXT"),
    ("interest", "REAL"),
    ("delivered_at", "REAL"),
    ("delivery_channel", "TEXT"),
)

#: The owner's seed beats — interest-profile DATA, shipped as defaults
#: and owner-correctable (persisted profile in ``news_interest`` wins
#: once saved; ``settings.briefing.topics`` also count as stated
#: interests at scoring time).  His beats weigh more by construction.
INTEREST_SEED_BEATS: tuple[tuple[str, float], ...] = (
    ("XAUUSD and gold trading", 1.0),
    ("AI and model work", 0.9),
    ("Nigerian tech and Lagos", 0.9),
    ("Devon herself — the owner's own agent and her systems", 1.0),
    ("USA money opportunities, including loans", 0.8),
)

#: Tier thresholds (settings.news.interest_high / interest_low win).
INTEREST_HIGH_DEFAULT = 0.75
INTEREST_LOW_DEFAULT = 0.35
#: Anytime-push budget per day (settings.news.push_cap_per_day wins).
PUSH_CAP_DEFAULT = 6
#: Bounds on per-poll work: stories scored / backlog rows tiered.
_SCORE_BATCH = 20
_SCORE_MAX_PER_POLL = 80
_TIER_BACKLOG_MAX = 200


def _ensure_delivery_columns(db: Any) -> None:
    """Lazily add the pass-#85 columns + profile table.

    Same best-effort, idempotent pattern as the attribution columns:
    a failure here must never stop news — storage falls back to the
    legacy column list and delivery tracking resumes next construct.
    """
    if db is None:
        return
    for name, decl in _DELIVERY_COLUMNS:
        try:
            db.execute(
                f"ALTER TABLE news_items ADD COLUMN {name} {decl}")
        except Exception as exc:  # noqa: BLE001
            if "duplicate column" not in str(exc).lower():
                return
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS news_interest "
            "(key TEXT PRIMARY KEY, value TEXT)")
    except Exception:  # noqa: BLE001
        pass


def _ensure_translations_table(db: Any) -> None:
    """The news_translations record (docs 19+21, N1): one durable row
    per (news item, target language) holding the translated spans the
    digest renders, the T1 receipt ids that minted them, and the
    gate's quality mark.  Runtime-created (Batch B) — migrations stay
    frozen.  Best-effort like the other ensure helpers: a failure here
    degrades re-digests to re-translation, never to broken news."""
    if db is None:
        return
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS news_translations ("
            "item_ref TEXT NOT NULL, "
            "url TEXT, "
            "content_hash TEXT, "
            "source_lang TEXT, "
            "source_confidence REAL, "
            "target_lang TEXT NOT NULL, "
            "receipt_id TEXT, "
            "title_receipt_id TEXT, "
            "quality_mark TEXT, "
            "rung TEXT, "
            "translated INTEGER, "
            "translated_title TEXT, "
            "translated_summary TEXT, "
            "created_at REAL, "
            "PRIMARY KEY (item_ref, target_lang))")
    except Exception as exc:  # noqa: BLE001
        record_failure("news.ensure_translations_table", exc)


def news_registry_for_context(context: Any = None,
                              injected: Any = None) -> Any:
    """The source registry news growth lands in, or None.

    Mirrors the #88–#94 resolver exactly: an injected registry wins;
    otherwise the shared ``DEFAULT_REGISTRY`` bound to the context's
    KV store — and only when the context carries a database, so
    db-less callers keep the pre-AA-55 behaviour exactly.  Never
    raises."""
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
            # enhancement; its absence never sinks a poll
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _grown_news_feeds(reg: Any) -> list[tuple[str, str]]:
    """Feeds the registry health loop grew into the news class
    (AA-55): promoted candidates (origin ``grown``, kind ``news``)
    served through the generic feed shape — ``https://<host>/feed/``,
    the WordPress/RSS shape most publisher hosts answer (the
    DiscoveredBlogSource bargain: a host that does not serve it
    records failures on the real path and is stood down / shed by
    the same machinery that promoted it).  Never raises."""
    try:
        if reg is None:
            return []
        out: list[tuple[str, str]] = []
        for desc in reg.by_kind("news"):
            if reg.origin_for(desc) != "grown":
                continue
            out.append((str(desc.name or desc.domain),
                        f"https://{desc.domain}/feed/"))
        return sorted(out, key=lambda t: t[1])
    except Exception:  # noqa: BLE001
        return []


def _note_news_hosts(reg: Any, items: list[dict[str, Any]],
                     seen: set[str]) -> int:
    """Discovery intake for one poll (AA-55): publisher hosts from
    fetched items (the item's ``<source>`` publisher URL first, the
    item's own host second — aggregator pages surface their
    publishers exactly this way) become UNTRUSTED registry
    candidates, kinds ``("news",)``.  ``seen`` dedupes within the
    poll so each host is noted at most once per run.  Returns the
    number of NEW candidates noted.  Best-effort; never raises."""
    noted = 0
    try:
        if reg is None:
            return 0
        for item in items:
            # ONE intake per item: the publisher host (the item's
            # <source> URL) wins; the item's own host is only the
            # fallback when no publisher is named.  A host already
            # seen this poll ends the item — its other URL is the
            # aggregator's, not a new publisher.
            host = ""
            for raw in (item.get("source_url"), item.get("url")):
                host = reg._host_of(str(raw or ""))
                if host:
                    break
            if not host or "." not in host or host in seen:
                continue
            seen.add(host)
            if reg.note_candidate(
                    host, via="news-item", kinds=("news",)):
                noted += 1
        return noted
    except Exception:  # noqa: BLE001
        return noted


def publisher_of(item: dict[str, Any]) -> str:
    """The truthful display attribution for a news item or stored row.

    Publisher when known — a live item's ``source_name`` (from the RSS
    ``<source>`` element) or a stored row's ``publisher`` column — else
    the carrying feed's name in ``source``.  Legacy rows (NULL
    attribution) therefore resolve to exactly the label they always
    rendered under.
    """
    for key in ("publisher", "source_name"):
        val = str(item.get(key) or "").strip()
        if val:
            return val
    return str(item.get("source") or "").strip()


class NewsAgent:
    """Fetches feeds, dedupes, summarizes, and delivers a digest."""

    def __init__(self, context: Any, notifier: Any = None, *,
                 registry: Any = None) -> None:
        self.context = context
        self.settings = getattr(context, "settings", None)
        self.db = getattr(context, "db", None)
        # AA-55: the ONE source registry — news discovery intake and
        # grown feeds resolve through it (None = db-less context =
        # seed behaviour exactly).
        self._registry = news_registry_for_context(context, registry)
        _ensure_attribution_columns(self.db)
        _ensure_delivery_columns(self.db)
        _ensure_translations_table(self.db)
        self.notifier = notifier
        #: db-less contexts keep translation records in memory (the
        #: _pending precedent): one instance's digests still reuse.
        self._translations_mem: dict[tuple[str, str], dict[str, Any]] = {}
        #: db-less contexts keep the last poll's rows in memory so a
        #: poll() → brief() pair on one instance still composes.
        self._pending: list[dict[str, Any]] = []
        #: last scoring outcome ("ok" / "unavailable" / "error: …") —
        #: surfaced by status() so degradation is never silent.
        self._interest_state: str | None = None
        tools = getattr(self.settings, "tools", None)
        self._client = HttpClient(
            timeout=getattr(tools, "http_timeout", 30.0),
            user_agent=getattr(tools, "user_agent", "NoMoralsCore/0.1"),
            proxy_url=getattr(tools, "proxy_url", ""),
        )

    # ── seam tables (None on the SQLite default) ─────────────────────────
    # Mongo P3: with a document store attached, news_items / news_interest
    # route through storage.rowstore (documents are carry-v1 rows; the
    # runtime-ALTER columns are declared to the mapping). The SQLite
    # default runs the native statements below unchanged.
    _ITEM_EXTRA = ("publisher", "publisher_url", "delivery_state",
                   "interest", "delivered_at", "delivery_channel")

    def _items(self):
        from ..storage.rowstore import attached_rows
        return attached_rows(self.db, "news_items",
                             extra_columns=self._ITEM_EXTRA)

    def _profile_t(self):
        from ..storage.rowstore import attached_rows
        return attached_rows(self.db, "news_interest")

    def _upd_item(self, item_id: str, fields: dict[str, Any],
                  native_sql: str, native_params: tuple) -> None:
        items = self._items()
        if items is not None:
            items.update(item_id, fields)
            return
        self.db.execute(native_sql, native_params)

    def _feeds(self) -> list[tuple[str, str]]:
        raw = (getattr(getattr(self.settings, "news", None), "feeds", "") or "").strip()
        if raw:
            out = []
            for part in raw.split(","):
                part = part.strip()
                if part.startswith("http"):
                    out.append((part.split("/")[-2] or part, part))
            base = out or list(DEFAULT_FEEDS)
        else:
            base = list(DEFAULT_FEEDS)
        # AA-55: health-promoted grown feeds join alongside the
        # seeds (and alongside an owner-configured list) — growth
        # is additive, never a replacement.
        have = {url for _name, url in base}
        for name, url in _grown_news_feeds(self._registry):
            if url not in have:
                base.append((name, url))
                have.add(url)
        return base

    def _grown_feed_urls(self) -> set[str]:
        """URLs in the current feed list that came from registry
        growth (their poll outcomes land back in the registry)."""
        return {url for _name, url in _grown_news_feeds(self._registry)}

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

    # ── poll: fetch + store + score + tier (pass #85) ──────────────────
    def poll(self) -> dict[str, Any]:
        """Fetch every feed, store EVERY fresh story as undelivered,
        score with the brain, and apply the interest tiers.

        The digest is brief()'s separate act; poll() itself only
        delivers the high-interest tier (the anytime push path).
        Report keys: the fetch breadth (sources_tried / sources_ok /
        per_source / stood_down / errors) plus the tier outcome
        (polled / scored / pushed / archived / held, the interest
        state, and the discipline flags quiet_hold / push_cap_reached).
        """
        from ..agency.sourcehealth import SourceHealth, SourceSpec, RSS

        fetched: list[dict[str, str]] = []
        errors: list[str] = []
        per_source: dict[str, int] = {}
        sources_ok: list[str] = []
        stood_down: list[str] = []
        feeds = self._feeds()
        # AA-55: grown feeds' outcomes land back in the source
        # registry (the loop that promoted them also sheds them),
        # and publisher hosts in fetched items intake as candidates.
        grown_urls = self._grown_feed_urls()
        seen_hosts: set[str] = set()
        candidates_noted = 0
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
                if url in grown_urls:
                    self._record_grown_outcome(url, "failure")
                continue
            if not items:
                # HTTP 200 but zero usable items: unusable as a source —
                # a health failure (counts toward stand-down), though the
                # digest itself still degrades gracefully below.
                if health is not None:
                    health.note_result(spec, ok=False,
                                       error="zero usable items", http=200)
                if url in grown_urls:
                    self._record_grown_outcome(url, "no_result")
            elif health is not None:
                health.note_result(spec, ok=True, items=len(items),
                                   http=200)
            if items and url in grown_urls:
                self._record_grown_outcome(url, "served")
            sources_ok.append(source)
            per_source[source] = len(items)
            for it in items:
                it["source"] = source
                fetched.append(it)
            candidates_noted += _note_news_hosts(
                self._registry, items, seen_hosts)

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
                items = self._items()
                if items is not None:
                    rows = items.find(
                        {"created_at": {"$gte": time.time() - 14 * 86400}})
                else:
                    rows = self.db.query(
                        "SELECT title FROM news_items WHERE created_at >= ?",
                        (time.time() - 14 * 86400,))
                for row in rows or []:
                    raw = (row.get("title") if hasattr(row, "get")
                           else row[0]) or ""
                    key = re.sub(r"\W+", " ", str(raw).lower()).strip()
                    if key:
                        known_titles.add(key)
            except Exception as exc:  # noqa: BLE001
                record_failure("news.known_titles", exc)
                known_titles = set()

        fresh: list[dict[str, str]] = []
        for it in fetched:
            if self.db is not None:
                try:
                    items = self._items()
                    if items is not None:
                        known = items.find_one({"url": it["url"]})
                    else:
                        known = self.db.query_one("SELECT id FROM news_items WHERE url = ?", (it["url"],))
                except Exception as exc:  # noqa: BLE001
                    record_failure("news.url_known", exc)
                    known = None
                if known is not None:
                    continue
                title_key = re.sub(
                    r"\W+", " ", (it.get("title") or "").lower()).strip()
                if title_key and title_key in known_titles:
                    continue
            fresh.append(it)
        fresh.sort(key=lambda i: float(i.get("source_ts") or 0), reverse=True)
        # Store EVERY fresh story — the old fresh[:cap] selection is
        # gone; the digest cap no longer decides what exists.
        new_rows = self._store_fresh(fresh)
        tiers = self._score_and_tier(new_rows)
        held = 0
        if self.db is not None:
            try:
                items = self._items()
                if items is not None:
                    held = items.count(
                        {"delivery_state": STATE_UNDELIVERED})
                else:
                    row = self.db.query_one(
                        "SELECT COUNT(*) AS n FROM news_items "
                        "WHERE delivery_state = ?", (STATE_UNDELIVERED,))
                    held = int((row or {}).get("n") or 0)
            except Exception as exc:  # noqa: BLE001
                record_failure("news.held_count", exc)
                held = 0
        else:
            held = sum(1 for r in self._pending
                       if r.get("delivery_state") == STATE_UNDELIVERED)
        return {
            "ok": bool(sources_ok or new_rows),
            "polled": len(new_rows),
            "fresh": len(new_rows),
            "errors": errors,
            # Visible breadth: the user sees HOW MANY sources were tried
            # and which ones answered — never a silent 1–2. Stood-down
            # feeds are reported, not silently dropped.
            "sources_tried": len(feeds),
            "sources_ok": sources_ok,
            "per_source": per_source,
            "stood_down": stood_down,
            # AA-55: growth on the real path, reported not hidden.
            "grown_feeds": len(grown_urls),
            "candidates_noted": candidates_noted,
            **tiers,
            "held": held,
        }

    def _record_grown_outcome(self, feed_url: str, kind: str) -> None:
        """Land one grown feed's poll outcome in the source
        registry (AA-55): served / no_result / failure, mirroring
        the vocabulary the registry's health loop sheds by.
        Best-effort; never raises."""
        try:
            reg = self._registry
            if reg is None:
                return
            desc = reg.for_domain(reg._host_of(feed_url))
            if desc is None:
                return
            if kind == "served":
                reg.record_outcome(desc, True)
            elif kind == "no_result":
                reg.record_no_result(desc)
            else:
                reg.record_outcome(desc, False,
                                   error_class="feed_failed")
        except Exception as exc:  # noqa: BLE001
            record_failure("news.grown_outcome", exc)

    def _store_fresh(self, fresh: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Persist fresh items as UNDELIVERED; return their row dicts.

        Attribution persists alongside the carrying feed (pass #83):
        ``source`` stays the feed (provenance), ``publisher`` /
        ``publisher_url`` carry the <source> attribution (NULL when
        the item has none).  If the pass-#85 columns are somehow
        unavailable (the lazy ensure failed), fall back to the legacy
        column list so storage keeps working exactly as it did before
        delivery state existed.
        """
        now = time.time()
        rows: list[dict[str, Any]] = []
        for it in fresh:
            rows.append({
                "id": new_id(),
                "source": it["source"],
                "publisher": (it.get("source_name") or "").strip() or None,
                "publisher_url":
                    (it.get("source_url") or "").strip() or None,
                "source_name": it.get("source_name") or "",
                "source_url": it.get("source_url") or "",
                "title": it["title"],
                "url": it["url"],
                "summary": (it.get("summary") or "")[:400],
                "published": float(it.get("source_ts") or 0),
                "created_at": now,
                "delivery_state": STATE_UNDELIVERED,
                "interest": None,
                "delivered_at": None,
                "delivery_channel": None,
            })
        if self.db is None:
            self._pending.extend(rows)
            return rows
        items = self._items()
        if items is not None:
            # INSERT OR IGNORE semantics, including the UNIQUE(url)
            # constraint: a row whose id or url is already stored is
            # skipped, exactly as the native statement skips it.
            for r in rows:
                if items.get(r["id"]) is not None:
                    continue
                if items.find_one({"url": r["url"]}) is not None:
                    continue
                items.put({
                    "id": r["id"], "source": r["source"],
                    "publisher": r["publisher"],
                    "publisher_url": r["publisher_url"],
                    "title": r["title"], "url": r["url"],
                    "summary": r["summary"], "published": r["published"],
                    "created_at": r["created_at"],
                    "delivery_state": STATE_UNDELIVERED})
            return rows
        stored = False
        try:
            with self.db.transaction():
                for r in rows:
                    self.db.execute(
                        "INSERT OR IGNORE INTO news_items (id, source, publisher, publisher_url, title, url, summary, published, created_at, delivery_state) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (r["id"], r["source"], r["publisher"],
                         r["publisher_url"], r["title"], r["url"],
                         r["summary"], r["published"], r["created_at"],
                         STATE_UNDELIVERED),
                    )
            stored = True
        except Exception:  # noqa: BLE001
            stored = False
        if not stored:
            try:
                with self.db.transaction():
                    for r in rows:
                        self.db.execute(
                            "INSERT OR IGNORE INTO news_items (id, source, publisher, publisher_url, title, url, summary, published, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (r["id"], r["source"], r["publisher"],
                             r["publisher_url"], r["title"], r["url"],
                             r["summary"], r["published"], r["created_at"]),
                        )
            except Exception as exc:  # noqa: BLE001
                record_failure("news.store_fresh_legacy", exc)
            # rows written without a state read back NULL = legacy
            # (already delivered) — keep this instance's copies so the
            # poll → brief flow on THIS agent still composes them.
            self._pending.extend(rows)
        return rows

    # ── interest: profile, scoring, tiers ────────────────────────────────

    def _news_setting(self, name: str, default: Any) -> Any:
        val = getattr(getattr(self.settings, "news", None), name, None)
        return default if val is None else val

    def _thresholds(self) -> tuple[float, float]:
        try:
            high = float(self._news_setting(
                "interest_high", INTEREST_HIGH_DEFAULT))
        except (TypeError, ValueError):
            high = INTEREST_HIGH_DEFAULT
        try:
            low = float(self._news_setting(
                "interest_low", INTEREST_LOW_DEFAULT))
        except (TypeError, ValueError):
            low = INTEREST_LOW_DEFAULT
        return (max(0.0, min(high, 1.0)), max(0.0, min(low, 1.0)))

    def _push_cap(self) -> int:
        try:
            return max(0, int(self._news_setting(
                "push_cap_per_day", PUSH_CAP_DEFAULT)))
        except (TypeError, ValueError):
            return PUSH_CAP_DEFAULT

    def _load_profile(self) -> dict[str, Any]:
        """The stored profile, or the seeds when none was saved yet."""
        profile: dict[str, Any] = {
            "beats": [{"topic": t, "weight": w}
                      for t, w in INTEREST_SEED_BEATS],
            "pins": [],
        }
        saved: Any = None
        if self.db is not None:
            try:
                profile_t = self._profile_t()
                if profile_t is not None:
                    row = profile_t.get("profile")
                else:
                    row = self.db.query_one(
                        "SELECT value FROM news_interest WHERE key = ?",
                        ("profile",))
                raw = (row or {}).get("value")
                if raw:
                    saved = json.loads(raw)
            except Exception as exc:  # noqa: BLE001 - seeds are the fallback
                record_failure("news.load_profile", exc)
                saved = None
        elif isinstance(getattr(self, "_profile_mem", None), dict):
            saved = self._profile_mem  # type: ignore[attr-defined]
        if isinstance(saved, dict):
            beats = saved.get("beats")
            if isinstance(beats, list) and beats:
                profile["beats"] = [
                    {"topic": str(b.get("topic") or ""),
                     "weight": float(b.get("weight") or 0.5)}
                    for b in beats
                    if isinstance(b, dict) and b.get("topic")]
            pins = saved.get("pins")
            if isinstance(pins, list):
                profile["pins"] = [str(p) for p in pins if str(p).strip()]
        return profile

    def _save_profile(self, profile: dict[str, Any]) -> None:
        if self.db is None:
            self._profile_mem = profile  # type: ignore[attr-defined]
            return
        try:
            profile_t = self._profile_t()
            if profile_t is not None:
                profile_t.put({"key": "profile",
                               "value": json.dumps(profile)})
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO news_interest (key, value) "
                    "VALUES (?, ?)",
                    ("profile", json.dumps(profile)))
        except Exception as exc:  # noqa: BLE001
            record_failure("news.save_profile", exc)

    def interest_profile(self) -> dict[str, Any]:
        """The effective interest profile: stored beats + pins, with
        the owner's already-configured briefing topics folded in as
        stated interests (settings.briefing.topics — the same list the
        morning briefing's NewsProvider filters by)."""
        profile = self._load_profile()
        beats = [dict(b) for b in profile["beats"]]
        known = {b["topic"].lower() for b in beats}
        briefing = getattr(self.settings, "briefing", None)
        for topic in (getattr(briefing, "topics", None) or []):
            topic = str(topic).strip()
            if topic and topic.lower() not in known:
                beats.append({"topic": topic, "weight": 1.0})
                known.add(topic.lower())
        pins = list(profile["pins"])
        beats_text = "; ".join(
            f"{b['topic']} (weight {b['weight']:g})" for b in beats)
        if pins:
            beats_text += " | pinned: " + ", ".join(pins)
        return {"beats": beats, "pins": pins, "beats_text": beats_text}

    def pin(self, topic: str) -> dict[str, Any]:
        """Pin a topic: it renders as always-high interest to the brain."""
        topic = (topic or "").strip()
        profile = self._load_profile()
        if topic and topic.lower() not in {
                p.lower() for p in profile["pins"]}:
            profile["pins"].append(topic)
            self._save_profile(profile)
        return self.interest_profile()

    def unpin(self, topic: str) -> dict[str, Any]:
        topic = (topic or "").strip().lower()
        profile = self._load_profile()
        profile["pins"] = [p for p in profile["pins"]
                           if p.lower() != topic]
        self._save_profile(profile)
        return self.interest_profile()

    def _brain_available(self) -> bool:
        # The module's own convention (see _model_summaries): no
        # router on the context means no live model is answering.
        return getattr(self.context, "router", None) is not None

    def _score_interest(
            self, rows: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
        """Brain-score rows 0–1 against the interest profile.

        Returns ({row_id: score}, state).  State is "unavailable"
        when no brain is answering (scores stay NULL — never
        fabricated) and "error: …" when the brain answered garbage;
        both degrade to digest-only delivery, visibly.
        """
        if not rows:
            return {}, (self._interest_state or "ok")
        if not self._brain_available():
            self._interest_state = "unavailable"
            return {}, "unavailable"
        from ..llm.base import Message

        profile = self.interest_profile()
        scores: dict[str, float] = {}
        try:
            brain = brain_for(self.context)
            for start in range(0, len(rows), _SCORE_BATCH):
                batch = rows[start:start + _SCORE_BATCH]
                lines = []
                for i, r in enumerate(batch):
                    who = publisher_of(r)
                    lines.append(
                        f"{i + 1}. {r.get('title', '')} — "
                        f"{(r.get('summary') or '')[:160]}"
                        + (f" [{who}]" if who else ""))
                prompt = (
                    "Score each news story for INTEREST to one specific "
                    "reader, 0.0 to 1.0: novelty x consequence x "
                    "likelihood this reader cares. The reader's beats "
                    "(higher weight = cares more): "
                    f"{profile['beats_text']}. Stories on the reader's "
                    "beats score higher; pinned topics are always high "
                    "interest. Reply with one line per story, exactly: "
                    "\"N. <score>\".\n" + "\n".join(lines))
                response = brain.chat(
                    [Message(role="user", content=prompt)],
                    task_kind="summarize")
                text = (getattr(response, "text", "") or "").strip()
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line[0] not in "0123456789":
                        continue
                    num, _, rest = line.partition(".")
                    try:
                        idx = int(num.strip()) - 1
                        val = float(rest.strip().split()[0])
                    except (ValueError, IndexError):
                        continue
                    if 0 <= idx < len(batch):
                        scores[batch[idx]["id"]] = max(0.0, min(val, 1.0))
        except Exception as exc:  # noqa: BLE001
            self._interest_state = f"error: {type(exc).__name__}: {exc}"[:120]
            return {}, self._interest_state
        if not scores:
            self._interest_state = "error: brain returned no usable scores"
            return {}, self._interest_state
        self._interest_state = "ok"
        return scores, "ok"

    def _set_state(self, row: dict[str, Any], state: str,
                   channel: str | None = None) -> None:
        row["delivery_state"] = state
        row["delivered_at"] = time.time()
        if channel is not None:
            row["delivery_channel"] = channel
        if self.db is not None:
            try:
                self._upd_item(
                    row["id"],
                    {"delivery_state": state,
                     "delivered_at": row["delivered_at"],
                     "delivery_channel": row.get("delivery_channel")},
                    "UPDATE news_items SET delivery_state = ?, "
                    "delivered_at = ?, delivery_channel = ? WHERE id = ?",
                    (state, row["delivered_at"],
                     row.get("delivery_channel"), row["id"]))
            except Exception as exc:  # noqa: BLE001
                record_failure("news.set_state", exc)

    def _pushes_today(self) -> int:
        lt = time.localtime()
        start = time.mktime(
            (lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        if self.db is None:
            return sum(1 for r in self._pending
                       if r.get("delivery_state") == STATE_PUSHED
                       and (r.get("delivered_at") or 0) >= start)
        try:
            items = self._items()
            if items is not None:
                return items.count({
                    "delivery_state": STATE_PUSHED,
                    "delivered_at": {"$gte": start}})
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM news_items "
                "WHERE delivery_state = ? AND delivered_at >= ?",
                (STATE_PUSHED, start))
            return int((row or {}).get("n") or 0)
        except Exception as exc:  # noqa: BLE001
            record_failure("news.pushes_today", exc)
            return 0

    def _quiet_now(self) -> bool:
        """Quiet hours per the codebase's one discipline: the
        notifier's partner quiet window (quiet_start/quiet_end)."""
        try:
            from .notifier import _in_quiet_hours_now
            return bool(_in_quiet_hours_now(self.context))
        except Exception as exc:  # noqa: BLE001
            record_failure("news.quiet_now", exc)
            return False

    def _score_and_tier(self, new_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Score the undelivered backlog and apply the tiers.

        Candidates = the fresh rows + stored undelivered rows (a story
        held by quiet hours or a dead brain is re-tiered by the next
        poll).  High → anytime push (quiet hours / per-day cap hold
        it, still undelivered); low → archived; mid → stays for the
        digest.  Unscored rows are never tiered by a guess.
        """
        out: dict[str, Any] = {
            "scored": 0, "pushed": 0, "archived": 0,
            "interest": self._interest_state or (
                "ok" if self._brain_available() else "unavailable"),
            "quiet_hold": False, "push_cap_reached": False,
        }
        candidates: dict[str, dict[str, Any]] = {
            r["id"]: r for r in new_rows}
        if self.db is not None:
            try:
                items = self._items()
                if items is not None:
                    rows = items.find(
                        {"delivery_state": STATE_UNDELIVERED})
                    rows.sort(key=lambda r: r.get("created_at") or 0)
                    rows = rows[:_TIER_BACKLOG_MAX]
                else:
                    rows = self.db.query(
                        "SELECT * FROM news_items WHERE delivery_state = ? "
                        "ORDER BY created_at ASC LIMIT ?",
                        (STATE_UNDELIVERED, _TIER_BACKLOG_MAX))
                for r in rows or []:
                    candidates.setdefault(r["id"], dict(r))
            except Exception as exc:  # noqa: BLE001
                record_failure("news.score_tier_backlog", exc)
        else:
            for r in self._pending:
                if r.get("delivery_state") == STATE_UNDELIVERED:
                    candidates.setdefault(r["id"], r)
        if not candidates:
            return out
        rows = list(candidates.values())
        unscored = [r for r in rows if r.get("interest") is None]
        # fresh first: newest stories get the bounded scoring budget
        unscored.sort(key=lambda r: float(r.get("created_at") or 0),
                      reverse=True)
        scores, state = self._score_interest(
            unscored[:_SCORE_MAX_PER_POLL])
        out["interest"] = state
        out["scored"] = len(scores)
        for rid, val in scores.items():
            row = candidates[rid]
            row["interest"] = val
            if self.db is not None:
                try:
                    self._upd_item(
                        rid, {"interest": val},
                        "UPDATE news_items SET interest = ? WHERE id = ?",
                        (val, rid))
                except Exception as exc:  # noqa: BLE001
                    record_failure("news.score_tier_update", exc)
        high, low = self._thresholds()
        push_candidates: list[dict[str, Any]] = []
        for row in rows:
            score = row.get("interest")
            if score is None:
                continue  # unscored: never tiered by a guess
            if float(score) < low:
                if row.get("delivery_state") == STATE_UNDELIVERED:
                    self._set_state(row, STATE_ARCHIVED, channel="interest")
                    out["archived"] += 1
            elif float(score) >= high:
                push_candidates.append(row)
        if push_candidates:
            push_candidates.sort(
                key=lambda r: float(r.get("interest") or 0), reverse=True)
            if self.notifier is None:
                out["push_held_no_notifier"] = len(push_candidates)
            elif self._quiet_now():
                out["quiet_hold"] = True
            else:
                remaining = self._push_cap() - self._pushes_today()
                if remaining <= 0:
                    out["push_cap_reached"] = True
                else:
                    if len(push_candidates) > remaining:
                        out["push_cap_reached"] = True
                    for row in push_candidates[:remaining]:
                        # Owner-language delivery (translation organ):
                        # the alert's headline + gist arrive translated,
                        # the original headline kept as a small line,
                        # the source link untouched.
                        disp = self._localized_item(
                            str(row.get("title") or ""),
                            _short_summary(row.get("summary") or ""))
                        body = disp.summary or ""
                        if disp.translated and disp.original_title:
                            body = (body + "\n" if body else "") \
                                + f"original: {disp.original_title}"
                        body = (body + "\n" if body else "") \
                            + str(row.get("url") or "")
                        try:
                            res = self.notifier.publish(
                                "news", disp.title, body)
                        except Exception as exc:  # noqa: BLE001
                            record_failure("news.score_tier_publish", exc)
                            res = {}
                        if isinstance(res, dict) and res.get("delivered"):
                            self._set_state(row, STATE_PUSHED,
                                            channel="push")
                            out["pushed"] += 1
        return out

    # ── brief: the digest of everything undelivered ──────────────────────

    @staticmethod
    def _normalize_row(row: Any) -> Any:
        """Stored row → live-item shape (pass #83): the persisted
        ``publisher`` / ``publisher_url`` surface as ``source_name`` /
        ``source_url``.  Legacy rows get "" — the fallback every
        consumer already applies."""
        if isinstance(row, dict):
            if not row.get("source_name"):
                row["source_name"] = row.get("publisher") or ""
            if not row.get("source_url"):
                row["source_url"] = row.get("publisher_url") or ""
        return row

    def undelivered(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Stored stories not yet delivered, newest first."""
        rows: list[dict[str, Any]] = []
        if self.db is not None:
            try:
                items = self._items()
                if items is not None:
                    queried = items.find(
                        {"delivery_state": STATE_UNDELIVERED})
                else:
                    queried = self.db.query(
                        "SELECT * FROM news_items WHERE delivery_state = ? "
                        "ORDER BY published DESC",
                        (STATE_UNDELIVERED,)) or []
                rows = [r for r in queried if isinstance(r, dict)]
            except Exception as exc:  # noqa: BLE001
                record_failure("news.undelivered", exc)
                rows = []
        seen_ids = {r.get("id") for r in rows}
        # this instance's in-memory copies (db-less contexts, and the
        # legacy-column fallback where stored rows carry no state)
        for r in self._pending:
            if r.get("delivery_state") == STATE_UNDELIVERED \
                    and r.get("id") not in seen_ids:
                rows.append(r)
        rows = [self._normalize_row(r) for r in rows]
        rows.sort(key=lambda r: float(r.get("published") or 0),
                  reverse=True)
        return rows if limit is None else rows[:limit]

    # ── news translation (docs 19+21, N1) ─────────────────────────────
    # Translate-before-summarize: when the owner's news-translation
    # setting is ON, digest items are translated FIRST through the
    # T1/T2 ticket path (requester "news") and _model_summaries reads
    # the TRANSLATED text.  The span translated is exactly the span
    # the summarizer consumes — the title plus summary[:160] (the
    # slice _model_summaries itself reads) — never whole articles.
    # Every translation lands in the news_translations record; renders
    # and re-digests READ the record instead of re-translating (the
    # T2 durable cache is the second layer, not a substitute).
    # news_items stays canonical: nothing here ever writes to it.

    @staticmethod
    def _item_ref(row: dict[str, Any]) -> str:
        """The store's item identity: the news_items id when the row
        carries one, else its URL."""
        rid = row.get("id")
        if rid is not None and str(rid) != "":
            return str(rid)
        return str(row.get("url") or "")

    def _interest_kv_get(self, key: str) -> str | None:
        """One news_interest value (the profile store's dual path:
        rowstore seam when attached, native SQL otherwise, memory
        when db-less)."""
        if self.db is not None:
            try:
                profile_t = self._profile_t()
                if profile_t is not None:
                    row = profile_t.get(key)
                else:
                    row = self.db.query_one(
                        "SELECT value FROM news_interest WHERE key = ?",
                        (key,))
                value = (row or {}).get("value")
                return None if value is None else str(value)
            except Exception as exc:  # noqa: BLE001
                record_failure("news.kv_get", exc)
                return None
        mem = getattr(self, "_kv_mem", None)
        return mem.get(key) if isinstance(mem, dict) else None

    def _interest_kv_set(self, key: str, value: str) -> None:
        if self.db is None:
            if not isinstance(getattr(self, "_kv_mem", None), dict):
                self._kv_mem = {}  # type: ignore[attr-defined]
            self._kv_mem[key] = value  # type: ignore[attr-defined]
            return
        try:
            profile_t = self._profile_t()
            if profile_t is not None:
                profile_t.put({"key": key, "value": value})
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO news_interest (key, value) "
                    "VALUES (?, ?)", (key, value))
        except Exception as exc:  # noqa: BLE001
            record_failure("news.kv_set", exc)

    def news_translation(self) -> dict[str, Any]:
        """The effective news-translation posture: the stored runtime
        pref (set by ``/news target``) wins over the settings seed;
        an unset target falls back to the translation organ's owner
        language.  ``effective_target`` resolves only when enabled,
        so an OFF digest never touches the translator."""
        stored_on = self._interest_kv_get("translation_enabled")
        stored_target = self._interest_kv_get("translation_target")
        news_cfg = getattr(self.settings, "news", None)
        if stored_on is not None:
            enabled = stored_on == "1"
            origin = "stored"
        else:
            enabled = bool(
                getattr(news_cfg, "translation_enabled", False))
            origin = "settings" if enabled else "default"
        target = (stored_target
                  or getattr(news_cfg, "translation_target", "")
                  or "").strip().lower()
        effective = ""
        if enabled:
            effective = target
            if not effective:
                try:
                    from ..llm.translate import translator_for

                    effective = translator_for(
                        self.context).owner_language()
                except Exception:  # noqa: BLE001
                    effective = "en"
        return {"enabled": enabled, "target": target,
                "effective_target": effective, "source": origin}

    def set_news_translation(self, *, enabled: bool | None = None,
                             target: str | None = None
                             ) -> dict[str, Any]:
        """Persist the owner's news-translation posture (the runtime
        pref /news target writes); returns the effective posture."""
        if enabled is not None:
            self._interest_kv_set(
                "translation_enabled", "1" if enabled else "0")
        if target is not None:
            self._interest_kv_set("translation_target", target)
        return self.news_translation()

    def _translation_target(self) -> str:
        """The digest's target language, or "" when translation is
        OFF (the byte-identical legacy pipeline)."""
        posture = self.news_translation()
        return posture["effective_target"] if posture["enabled"] else ""

    def _translation_record(self, item_ref: str,
                            target: str) -> dict[str, Any] | None:
        if self.db is not None:
            try:
                row = self.db.query_one(
                    "SELECT * FROM news_translations "
                    "WHERE item_ref = ? AND target_lang = ?",
                    (item_ref, target))
                return dict(row) if row else None
            except Exception as exc:  # noqa: BLE001
                record_failure("news.translation_record_read", exc)
                return None
        return self._translations_mem.get((item_ref, target))

    def _save_translation_record(self, rec: dict[str, Any]) -> None:
        if self.db is not None:
            try:
                self.db.execute(
                    "INSERT OR REPLACE INTO news_translations ("
                    "item_ref, url, content_hash, source_lang, "
                    "source_confidence, target_lang, receipt_id, "
                    "title_receipt_id, quality_mark, rung, translated, "
                    "translated_title, translated_summary, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rec["item_ref"], rec["url"], rec["content_hash"],
                     rec["source_lang"], rec["source_confidence"],
                     rec["target_lang"], rec["receipt_id"],
                     rec["title_receipt_id"], rec["quality_mark"],
                     rec["rung"], 1 if rec["translated"] else 0,
                     rec["translated_title"], rec["translated_summary"],
                     rec["created_at"]))
                return
            except Exception as exc:  # noqa: BLE001
                record_failure("news.translation_record_write", exc)
        self._translations_mem[(rec["item_ref"],
                                rec["target_lang"])] = rec

    @staticmethod
    def _span_quality(outcome: Any) -> tuple[str, str, str]:
        """(quality mark, receipt id, rung) for one ticket outcome —
        the T1 mark vocabulary read off the gate verdict: a failed
        gate marks the span unverified; it never swaps or blocks."""
        inner = outcome.outcome
        rung = str(getattr(inner, "rung", "") or "")
        if not getattr(inner, "translated", False):
            return ("same-language" if rung == "same-language"
                    else "identity"), "", rung
        status = str((outcome.gate or {}).get("status") or "")
        mark = {"passed": "verified",
                "failed": "unverified"}.get(status, "not_checked")
        receipt = outcome.receipt_id or (
            outcome.receipt_ids[0] if outcome.receipt_ids else "")
        return mark, receipt, rung

    def _translate_span(self, translator: Any, text: str, source: str,
                        target: str) -> dict[str, Any]:
        """One span through the T1/T2 ticket path at fidelity
        exactness — fidelity is what runs the quality gate, and the
        gate's Yoruba posture is warn: the verdict lands as the
        span's mark, never as a block."""
        from ..llm.ticket import TranslationTicket, run_ticket

        ticket = TranslationTicket(
            text=text, target_lang=target, source_lang=source,
            exactness="fidelity", requester="news")
        outcome = run_ticket(translator, ticket)
        mark, receipt, rung = self._span_quality(outcome)
        return {"text": outcome.outcome.text,
                "source_lang": outcome.outcome.source_lang or "",
                "mark": mark, "receipt_id": receipt, "rung": rung,
                "translated": bool(outcome.outcome.translated)}

    def _item_translation(self, row: dict[str, Any],
                          target: str) -> dict[str, Any] | None:
        """The item's translation record for ``target``: read from
        news_translations when the item's content is unchanged (the
        reuse key is item identity + target, guarded by the content
        hash so a re-fetched story with new text re-translates), else
        translated now and recorded.  The source language is the
        detector's word — undetected stays an empty source_lang on
        the row, never a guessed language.  None = the record path
        failed; the caller falls back to the legacy render for this
        item.  Never raises."""
        import hashlib

        title = str(row.get("title") or "")
        span = str(row.get("summary") or "")[:160]
        item_ref = self._item_ref(row)
        digest = hashlib.sha256(
            (title + "\x00" + span).encode("utf-8")).hexdigest()
        try:
            existing = self._translation_record(item_ref, target)
            if (existing is not None
                    and existing.get("content_hash") == digest):
                existing["translated"] = bool(existing.get("translated"))
                return existing
            from ..llm.languages import detect_language
            from ..llm.translate import translator_for

            translator = translator_for(self.context)
            detected, confidence = detect_language(
                (title + " " + span).strip())
            rec: dict[str, Any] = {
                "item_ref": item_ref, "url": str(row.get("url") or ""),
                "content_hash": digest,
                "source_lang": "" if detected == "und" else detected,
                "source_confidence": float(confidence or 0.0),
                "target_lang": target, "receipt_id": "",
                "title_receipt_id": "", "quality_mark": "identity",
                "rung": "", "translated": False,
                "translated_title": title, "translated_summary": span,
                "created_at": time.time(),
            }
            if detected == target:
                rec["quality_mark"] = "same-language"
                rec["rung"] = "same-language"
                self._save_translation_record(rec)
                return rec
            source = "" if detected == "und" else detected
            title_res = (self._translate_span(
                translator, title, source, target) if title else None)
            span_res = (self._translate_span(
                translator, span, source, target) if span else None)
            marks: list[str] = []
            rungs: list[str] = []
            for res, key in ((title_res, "translated_title"),
                             (span_res, "translated_summary")):
                if res is None:
                    continue
                rungs.append(res["rung"])
                if res["source_lang"] and not rec["source_lang"]:
                    rec["source_lang"] = res["source_lang"]
                if res["translated"]:
                    rec[key] = res["text"]
                    marks.append(res["mark"])
            rec["rung"] = rungs[0] if rungs else ""
            if title_res is not None:
                rec["title_receipt_id"] = title_res["receipt_id"]
            if span_res is not None:
                rec["receipt_id"] = span_res["receipt_id"]
            if not rec["receipt_id"]:
                rec["receipt_id"] = rec["title_receipt_id"]
            if marks:
                rec["translated"] = True
                if "unverified" in marks:
                    rec["quality_mark"] = "unverified"
                elif all(m == "verified" for m in marks):
                    rec["quality_mark"] = "verified"
                else:
                    rec["quality_mark"] = "not_checked"
            elif "same-language" in rungs:
                rec["quality_mark"] = "same-language"
            self._save_translation_record(rec)
            return rec
        except Exception as exc:  # noqa: BLE001
            record_failure("news.item_translation", exc)
            return None

    @staticmethod
    def _translation_badge(rec: dict[str, Any]) -> str:
        """The digest badge for a translated item: source→target plus
        the gate's mark — verified wears a check, unverified is
        spelled out so a gate-failed translation never dresses as a
        passed one, not_checked stays plain (no quality claim).
        Identity / same-language items carry NO badge."""
        if not rec.get("translated") or not rec.get("source_lang"):
            return ""
        badge = f"🌐 {rec['source_lang']}→{rec['target_lang']}"
        mark = str(rec.get("quality_mark") or "")
        if mark == "verified":
            badge += " ✓"
        elif mark == "unverified":
            badge += " ⚠ unverified"
        return badge

    def _compose_digest(
            self, rows: list[dict[str, Any]]) -> str:
        # Docs 19+21 (N1): with news translation ON, items are
        # translated FIRST (into news_translations records) and the
        # summarizer reads the translated text; OFF, this method is
        # the pre-N1 pipeline unchanged — summarize the source,
        # localize at render.
        target = self._translation_target()
        records: dict[str, dict[str, Any]] = {}
        if target:
            for it in rows:
                rec = self._item_translation(it, target)
                if rec is not None:
                    records[self._item_ref(it)] = rec
            sum_rows: list[dict[str, Any]] = []
            for it in rows:
                rec = records.get(self._item_ref(it))
                if rec is not None and rec["translated"]:
                    shown = dict(it)
                    shown["title"] = rec["translated_title"]
                    shown["summary"] = rec["translated_summary"]
                    sum_rows.append(shown)
                else:
                    sum_rows.append(it)
            summaries = self._model_summaries(sum_rows)  # type: ignore[arg-type]
        else:
            summaries = self._model_summaries(rows)  # type: ignore[arg-type]
        lines: list[str] = []
        # Group by PUBLISHER when the item carries attribution, else by
        # carrying feed (pass #83): a Reuters story reads under
        # "■ Reuters" whichever feed carried it.
        by_source: dict[str, list[dict[str, Any]]] = {}
        for it in rows:
            by_source.setdefault(publisher_of(it), []).append(it)
        for source, items in by_source.items():
            lines.append(f"■ {source}")
            # every item is shown — no per-source slice; the digest
            # carries everything undelivered (pass #85).
            for it in items:
                rec = records.get(self._item_ref(it))
                if rec is not None and rec["translated"]:
                    # Translated BEFORE summarizing: the model summary
                    # was made from the translated text and is already
                    # in the target language — never re-translated.
                    # The record's span is the no-brain fallback; the
                    # badge carries language + quality mark only.
                    summary = (summaries.get(it["url"])
                               or rec["translated_summary"] or "")
                    line = f"  • {rec['translated_title']}"
                    badge = self._translation_badge(rec)
                    if badge:
                        line += f" {badge}"
                    if summary:
                        line += f" — {_short_summary(summary)}"
                    lines.append(line)
                    if it["title"]:
                        lines.append(f"    ◦ original: {it['title']}")
                    continue
                summary = summaries.get(it["url"]) or it.get("summary") or ""
                if rec is not None:
                    # Translation ON but this item was not translated
                    # (already in the target language, undetected, or
                    # every rung failed): it renders in its source
                    # text, plainly — no badge, no dressing.
                    lines.append(f"  • {it['title']}" + (
                        f" — {_short_summary(summary)}"
                        if summary else ""))
                    continue
                # Owner-language delivery (translation organ): the
                # headline + gist arrive in the owner's language; the
                # original headline rides as a small secondary line.
                disp = self._localized_item(it["title"], summary)
                lines.append(f"  • {disp.title}" + (
                    f" — {_short_summary(disp.summary)}"
                    if disp.summary else ""))
                if disp.translated and disp.original_title:
                    lines.append(f"    ◦ original: {disp.original_title}")
        return "\n".join(lines)

    def _localized_item(self, title: str, summary: str) -> Any:
        """One headline + gist in the owner's language (the
        translation organ's item shape).  Degrades to the organ's
        marked identity — the original text — when no rung can
        translate; never raises into a delivery path."""
        try:
            from ..llm.translate import translator_for

            return translator_for(self.context).translate_item(
                str(title or ""), str(summary or ""))
        except Exception as exc:  # noqa: BLE001
            record_failure("news.localized_item", exc)
            from ..llm.translate import ItemTranslation

            return ItemTranslation(title=str(title or ""),
                                   summary=str(summary or ""))

    def brief(self, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Compose and deliver the digest of ALL undelivered stories.

        No cap: what poll() stored and no push took, the digest
        carries.  The briefed rows are marked ``digested`` — when a
        notifier is attached the digest is published through it; with
        no notifier the caller's own surface IS the delivery (the
        /news command reply, the pulse's briefing), the convention
        those callers already document.
        """
        if rows is None:
            rows = self.undelivered()
        else:
            rows = [r for r in rows
                    if r.get("delivery_state") == STATE_UNDELIVERED]
        if not rows:
            return {"ok": False, "items": 0, "digest": "",
                    "delivered_urls": []}
        digest = self._compose_digest(rows)
        if self.notifier is not None:
            # full digest — the gateway chunks it; no [:3500] cut.
            self.notifier.publish(
                "news", f"news digest — {len(rows)} new items", digest)
        for r in rows:
            self._set_state(r, STATE_DIGESTED, channel="digest")
        return {"ok": True, "items": len(rows), "digest": digest,
                "delivered_urls": [str(r.get("url") or "") for r in rows]}

    def mark_delivered(self, keys: list[str], *,
                       channel: str = "digest") -> int:
        """Record delivery of undelivered rows by URL or id.

        The shared state machine's door for the OTHER delivery
        surfaces (the pulse slots deliver stories inside their own
        briefings): what they delivered must not be re-digested here.
        Returns how many rows transitioned.
        """
        wanted = {str(k) for k in (keys or []) if str(k).strip()}
        if not wanted:
            return 0
        now = time.time()
        if self.db is None:
            n = 0
            for r in self._pending:
                if r.get("delivery_state") == STATE_UNDELIVERED and (
                        str(r.get("url")) in wanted
                        or str(r.get("id")) in wanted):
                    r["delivery_state"] = STATE_DIGESTED
                    r["delivered_at"] = now
                    r["delivery_channel"] = channel
                    n += 1
            return n
        items = self._items()
        try:
            if items is not None:
                rows = [r for r in items.all()
                        if r.get("delivery_state") == STATE_UNDELIVERED
                        or r.get("delivery_state") is None]
            else:
                rows = self.db.query(
                    "SELECT id, url FROM news_items WHERE delivery_state = ? "
                    "OR delivery_state IS NULL", (STATE_UNDELIVERED,))
        except Exception as exc:  # noqa: BLE001
            record_failure("news.mark_delivered_select", exc)
            return 0
        n = 0
        for r in rows or []:
            if str(r.get("url")) in wanted or str(r.get("id")) in wanted:
                try:
                    self._upd_item(
                        r["id"],
                        {"delivery_state": STATE_DIGESTED,
                         "delivered_at": now, "delivery_channel": channel},
                        "UPDATE news_items SET delivery_state = ?, "
                        "delivered_at = ?, delivery_channel = ? "
                        "WHERE id = ?",
                        (STATE_DIGESTED, now, channel, r["id"]))
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    record_failure("news.mark_delivered_update", exc)
        return n

    def status(self) -> dict[str, Any]:
        """Delivery-state census + interest health, for /news status."""
        counts: dict[str, int] = {}
        if self.db is not None:
            try:
                items = self._items()
                if items is not None:
                    for r in items.all():
                        key = str(r.get("delivery_state") or "legacy")
                        counts[key] = counts.get(key, 0) + 1
                else:
                    for r in self.db.query(
                            "SELECT delivery_state, COUNT(*) AS n "
                            "FROM news_items GROUP BY delivery_state") or []:
                        key = r.get("delivery_state") or "legacy"
                        counts[str(key)] = int(r.get("n") or 0)
            except Exception as exc:  # noqa: BLE001
                record_failure("news.status", exc)
                counts = {}
        else:
            for r in self._pending:
                key = str(r.get("delivery_state") or "legacy")
                counts[key] = counts.get(key, 0) + 1
        profile = self.interest_profile()
        interest = self._interest_state or (
            "ok" if self._brain_available() else "unavailable")
        out = {
            "stored": sum(counts.values()),
            "by_state": counts,
            "undelivered": counts.get(STATE_UNDELIVERED, 0),
            "archived": counts.get(STATE_ARCHIVED, 0),
            "pushed_today": self._pushes_today(),
            "push_cap_per_day": self._push_cap(),
            "interest": interest,
            "quiet_now": self._quiet_now(),
            "beats": [b["topic"] for b in profile["beats"]],
            "pins": profile["pins"],
            "translation": self.news_translation(),
        }
        # AA-55: the growth surface — grown feeds in rotation and
        # candidates awaiting the health loop's verdict.
        if self._registry is not None:
            try:
                out["grown_feeds"] = len(
                    _grown_news_feeds(self._registry))
                out["news_candidates_pending"] = sum(
                    1 for d in self._registry.candidates()
                    if "news" in d.kinds)
            except Exception:  # noqa: BLE001 - status never breaks
                pass
        return out

    # ── one digest cycle (compatibility composition) ─────────────────────
    def run(self, cap: int | None = None) -> dict[str, Any]:
        """poll() + brief(): fetch, tier, and digest what remains.

        The historical single-call contract for /news, the pulse
        slots, and the agency adapter.  ``cap`` is accepted for
        signature compatibility and no longer selects what gets
        delivered — the digest carries every undelivered story
        (pass #85); it survives only as the settings default the
        callers pass through.
        """
        report = self.poll()
        brief = self.brief()
        report.update({
            "ok": bool(brief["items"] or report.get("pushed")),
            "items": brief["items"],
            "digest": brief["digest"],
            "delivered_urls": brief["delivered_urls"],
        })
        return report

    def recent(self, limit: int = 15) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            items = self._items()
            if items is not None:
                rows = items.all()
                rows.sort(key=lambda r: r.get("created_at") or 0,
                          reverse=True)
                rows = rows[:limit]
            else:
                rows = self.db.query(
                    "SELECT * FROM news_items ORDER BY created_at DESC LIMIT ?", (limit,)
                )
        except Exception as exc:  # noqa: BLE001
            record_failure("news.recent", exc)
            return []
        # Normalize stored rows to the live-item shape (pass #83), so
        # stored-path consumers see the same attribution fields a
        # freshly parsed item has.
        for row in rows or []:
            self._normalize_row(row)
        return rows or []


# ── agent tool (pass #85) ─────────────────────────────────────────────
#
# Until this pass the news organ had NO registry tool: the only
# scheduled deliveries were the four pulse slots, and nothing polled
# between them.  The tool closes over ``registry.context`` (the
# codebase convention for scheduler-bound tools — the registry never
# passes a context argument) and is what the "news poll" job binds.

def register(registry: Any) -> None:
    """Register the ``news`` tool (poll / brief / run / status / recent)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "news",
        description=(
            "News organ: action=poll (fetch all feeds, store every new "
            "story, brain-score interest, push the high tier anytime) | "
            "brief (digest of ALL undelivered stories) | run (poll + "
            "brief) | status (delivery-state census + interest health + "
            "profile) | recent (latest stored stories)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — poll|brief|run|status|recent",
        },
    )
    def news(*, action: str = "run") -> dict[str, Any]:
        act = (action or "run").strip().lower()
        if act == "status":
            return NewsAgent(context).status()
        if act == "recent":
            agent = NewsAgent(context)
            return {"items": [
                {"title": r.get("title", ""),
                 "source": publisher_of(r),
                 "url": r.get("url", ""),
                 "delivery_state": r.get("delivery_state")}
                for r in agent.recent(limit=15)]}
        from .notifier import Notifier

        agent = NewsAgent(context, notifier=Notifier(context))
        if act == "poll":
            return agent.poll()
        if act == "brief":
            return agent.brief()
        if act == "run":
            return agent.run()
        return {"error": f"unknown action {act!r} — "
                         "poll|brief|run|status|recent"}
