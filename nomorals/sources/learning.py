"""Learning collection (OpenCulture — 52-websites site #12).

OpenCulture (openculture.com, since 2006) is an editorial directory
of free cultural and educational media: hand-maintained lists of free
online courses, movies, audiobooks, ebooks, textbooks, language
lessons and MOOCs — each entry a vetted link with a short human
description. No hosting, no accounts: the system is sustained expert
curation over the open web's free resources, organised by learner
intent.

Devon's native organ is the same shape, built as her own collection:

* :class:`CollectionEntry` — one free learning resource: subject,
  title, provider, url, format, level, description, provenance,
  seeded date, and live health state (status + checked date).
* :class:`LearningCollection` — the durable collection, seeded from
  the open directories and grown by live search; stored as JSON
  under ``<home>/learning_collection.json``.
* :func:`check_entry` / :func:`refresh_health` — per-entry health
  probes following the ``agency.sourcehealth`` idiom (same status
  words, same bot/geo signals, same "usable payload" rule), done
  locally so this L4 organ never imports the L5 agency package.
* :class:`LiveLearningSearch` — the live-search rung: OpenCulture
  site search, Archive.org advanced search, Open Library search —
  all keyless, all verified live 2026-11-18.
* :func:`learn_free` — the query: vetted entries first (checked
  live), unchecked next (labelled "not yet checked", never wearing
  a fresh date), flagged entries shown with their reason and never
  recommended, live search last and clearly separated.

Honesty rules (from the mining report): every served entry shows
provenance and a check date; a stale entry never wears a fresh
recommendation's authority. A dead entry is flagged, not served as
good — and never silently dropped either: the flag names the reason.
Seed titles + providers were read off the live open-directory lists
on 2026-11-18 (OpenCulture's 1,700-course list, its free-textbook
and free-language-lesson lists, the providers' own open
catalogues); every seed URL was verified live (HTTP 200) from this
box the same day, except the two Project Gutenberg seeds —
gutenberg.org answered 503 to this box's egress (its home page
too), so they ship with health "unknown" and the first health tick
verifies them. A MOOC that charges for a certificate is still free
to audit — the reply says so where the provider says so.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "CollectionEntry",
    "LearningCollection",
    "LiveLearningSearch",
    "DIRECTORY_SPECS",
    "FORMATS",
    "LEVELS",
    "SEED_ENTRIES",
    "check_entry",
    "format_learning_reply",
    "learn_free",
    "refresh_health",
]

_COLLECTION_FILENAME = "learning_collection.json"

#: Download blogs and SPAs routinely 403 curl/python-urllib defaults.
_BROWSER_UA = (
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
)

#: Hard ceiling for any single source HTTP round-trip.
_HTTP_TIMEOUT = 20.0

#: Entries older than this without a re-check are re-probed by
#: refresh_health (a week — the collection is curated, not live data).
HEALTH_MAX_AGE_S = 7 * 24 * 3600

#: The day the seeds were read off the live directories.
SEED_DATE = "2026-11-18"

FORMATS = ("course", "textbook", "audiobook", "ebook", "movie", "lesson")
LEVELS = ("beginner", "intermediate", "advanced", "all")

#: Health words, shared with agency.sourcehealth by design (same
#: vocabulary, same meanings — the organ just probes locally).
HEALTH_STATUSES = (
    "unknown", "verified-live", "dead", "bot-blocked", "geo-blocked",
    "unverifiable",
)

#: The seed directories themselves, as the agency sources registry
#: collects them: (key, name, url, kind). The registry imports these
#: from this module — it owns no URL list of its own.
DIRECTORY_SPECS: list[tuple[str, str, str, str]] = [
    ("learning:openculture-courses", "openculture-courses",
     "https://www.openculture.com/freeonlinecourses", "html-links"),
    ("learning:openculture-textbooks", "openculture-textbooks",
     "https://www.openculture.com/free_textbooks", "html-links"),
    ("learning:openculture-languages", "openculture-languages",
     "https://www.openculture.com/freelanguagelessons", "html-links"),
    ("learning:openlibrary-search", "openlibrary-search",
     "https://openlibrary.org/search.json"
     "?q=pride+and+prejudice&limit=2", "json"),
    ("learning:archive-search", "archive-search",
     "https://archive.org/advancedsearch.php"
     "?q=title%3A%28pride+and+prejudice%29+AND+mediatype%3A%28texts%29"
     "&fl%5B%5D=identifier&rows=1&output=json", "json"),
]

#: Bot-wall / geo-block signals — the same idiom as
#: agency.sourcehealth, kept local so this L4 organ never imports L5.
_BOT_SIGNALS = (
    "captcha", "verify you are human", "are you a robot",
    "unusual traffic", "access denied", "attention required",
    "cloudflare", "just a moment", "please verify", "bot detected",
    "automated requests", "enable javascript", "js challenge",
    "client challenge", "access restricted", "request was blocked",
    "activity is prohibited",
)
_GEO_SIGNALS = (
    "not available in your country", "not available in your region",
    "geo-block", "geoblocked", "restricted in your country",
    "not available in your location", "unavailable for legal reasons",
)

_LINK_RE = re.compile(r'href="(https?://[^"]+)"', re.IGNORECASE)
_QUERY_STRIP_RE = re.compile(
    r"^(i want to learn|i'd like to learn|i wanna learn|teach me|"
    r"learn|study|show me)\s+", re.IGNORECASE)
_FREE_SUFFIX_RE = re.compile(
    r"\s+(for free|free|online free|free online)\s*$", re.IGNORECASE)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def _iso_date(epoch: float | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()


# ── records ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CollectionEntry:
    """One free learning resource in the collection."""

    subject: str = ""
    title: str = ""
    provider: str = ""
    url: str = ""
    format: str = ""               # course | textbook | audiobook |
                                   # ebook | movie | lesson
    level: str = "all"             # beginner | intermediate | advanced | all
    description: str = ""
    provenance: str = ""           # the open directory it was curated from
    seeded_at: str = ""            # YYYY-MM-DD the entry entered the collection
    health_status: str = "unknown"
    checked_at: float | None = None
    last_error: str = ""

    def __post_init__(self) -> None:
        if self.format and self.format not in FORMATS:
            raise ValueError(f"unknown format: {self.format!r}")
        if self.level and self.level not in LEVELS:
            raise ValueError(f"unknown level: {self.level!r}")
        if self.health_status not in HEALTH_STATUSES:
            raise ValueError(f"unknown health: {self.health_status!r}")

    @property
    def slug(self) -> str:
        return _slugify(f"{self.provider}-{self.title}")

    @property
    def checked_date(self) -> str | None:
        return _iso_date(self.checked_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "title": self.title,
            "provider": self.provider, "url": self.url,
            "format": self.format, "level": self.level,
            "description": self.description,
            "provenance": self.provenance, "seeded_at": self.seeded_at,
            "health_status": self.health_status,
            "checked_at": self.checked_at,
            "checked_date": self.checked_date,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionEntry:
        return cls(
            subject=data.get("subject", ""), title=data.get("title", ""),
            provider=data.get("provider", ""), url=data.get("url", ""),
            format=data.get("format", ""), level=data.get("level", "all"),
            description=data.get("description", ""),
            provenance=data.get("provenance", ""),
            seeded_at=data.get("seeded_at", ""),
            health_status=data.get("health_status", "unknown"),
            checked_at=data.get("checked_at"),
            last_error=data.get("last_error", ""),
        )


def _entry(subject: str, title: str, provider: str, url: str,
           format: str, level: str, description: str,
           provenance: str) -> CollectionEntry:
    return CollectionEntry(
        subject=subject, title=title, provider=provider, url=url,
        format=format, level=level, description=description,
        provenance=provenance, seeded_at=SEED_DATE)


#: Seed entries. Title + provider were read off the live open-directory
#: lists on 2026-11-18; every URL below answered HTTP 200 from this box
#: the same day, except the two Gutenberg seeds (gutenberg.org answered
#: 503 to this box's egress — home page too — so they ship unchecked).
SEED_ENTRIES: list[CollectionEntry] = [
    _entry("economics", "Financial Markets", "Yale Open Courses",
           "https://oyc.yale.edu/economics/econ-252", "course", "all",
           "How financial markets work, from Yale's Open Yale Courses.",
           "OpenCulture free-courses list"),
    _entry("economics", "Game Theory", "Yale Open Courses",
           "https://oyc.yale.edu/economics/econ-159", "course", "all",
           "Strategic thinking, games and incentives.",
           "OpenCulture free-courses list"),
    _entry("history", "Ancient Greek History", "Yale Open Courses",
           "https://oyc.yale.edu/classics/clcv-205", "course", "all",
           "Greek history from the Bronze Age to Alexander.",
           "OpenCulture free-courses list"),
    _entry("history", "The Civil War and Reconstruction Era, 1845–1877",
           "Yale Open Courses",
           "https://oyc.yale.edu/history/hist-119", "course", "all",
           "Slavery, war and reconstruction in America.",
           "OpenCulture free-courses list"),
    _entry("history", "European Civilization, 1648–1945",
           "Yale Open Courses",
           "https://oyc.yale.edu/history/hist-202", "course", "all",
           "Europe from the wars of religion to World War II.",
           "OpenCulture free-courses list"),
    _entry("history", "Epidemics in Western Society Since 1600",
           "Yale Open Courses",
           "https://oyc.yale.edu/history/hist-234", "course", "all",
           "Disease and society across four centuries.",
           "OpenCulture free-courses list"),
    _entry("economics", "Principles of Microeconomics",
           "MIT OpenCourseWare",
           "https://ocw.mit.edu/courses/14-01-principles-of-microeconomics-fall-2023/",
           "course", "beginner",
           "Supply, demand and markets, with lecture videos.",
           "OpenCulture free-courses list"),
    _entry("computer science",
           "Introduction to Computer Science and Programming in Python",
           "MIT OpenCourseWare",
           "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/",
           "course", "beginner",
           "Python from zero, with problem sets and exams.",
           "MIT OpenCourseWare"),
    _entry("computer science", "Introduction to Algorithms",
           "MIT OpenCourseWare",
           "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/",
           "course", "intermediate",
           "Algorithms and data structures, with video lectures.",
           "MIT OpenCourseWare"),
    _entry("computer science", "Machine Learning",
           "Coursera (Andrew Ng, Stanford)",
           "https://www.coursera.org/learn/machine-learning",
           "course", "intermediate",
           "The classic ML course; audit free, certificate paid.",
           "Coursera"),
    _entry("computer science", "CS50: Introduction to Computer Science",
           "Harvard",
           "https://cs50.harvard.edu/x/", "course", "beginner",
           "Harvard's intro CS, free with lectures and problem sets.",
           "Harvard CS50"),
    _entry("psychology", "The Science of Well-Being",
           "Coursera (Laurie Santos, Yale)",
           "https://www.coursera.org/learn/the-science-of-well-being",
           "course", "beginner",
           "The psychology of happiness; audit free, certificate paid.",
           "Coursera"),
    _entry("literature", "Introduction to Theory of Literature",
           "Yale Open Courses",
           "https://oyc.yale.edu/english/engl-300", "course", "all",
           "How to read literature as a theorist.",
           "OpenCulture free-courses list"),
    _entry("physics", "College Physics", "OpenStax",
           "https://openstax.org/details/books/college-physics",
           "textbook", "beginner",
           "Algebra-based physics, peer-reviewed and openly licensed.",
           "OpenCulture free-textbooks list"),
    _entry("mathematics", "Calculus Volume 1", "OpenStax",
           "https://openstax.org/details/books/calculus-volume-1",
           "textbook", "beginner",
           "Functions, limits, derivatives and integration.",
           "OpenStax"),
    _entry("physics", "University Physics Volume 1", "OpenStax",
           "https://openstax.org/details/books/university-physics-volume-1",
           "textbook", "intermediate",
           "Calculus-based physics: mechanics, sound, waves.",
           "OpenStax"),
    _entry("spanish", "Learn Spanish", "Duolingo",
           "https://www.duolingo.com/course/es/en/Learn-Spanish",
           "lesson", "beginner",
           "Bite-size Spanish lessons, free with ads.",
           "OpenCulture free-language-lessons list"),
    _entry("french", "Learn French", "Duolingo",
           "https://www.duolingo.com/course/fr/en/Learn-French",
           "lesson", "beginner",
           "Bite-size French lessons, free with ads.",
           "OpenCulture free-language-lessons list"),
    _entry("german", "Learn German", "Duolingo",
           "https://www.duolingo.com/course/de/en/Learn-German",
           "lesson", "beginner",
           "Bite-size German lessons, free with ads.",
           "OpenCulture free-language-lessons list"),
    _entry("literature", "Pride and Prejudice (audiobook)", "LibriVox",
           "https://librivox.org/pride-and-prejudice-by-jane-austen/",
           "audiobook", "all",
           "Jane Austen's novel, read by volunteers.",
           "LibriVox"),
    _entry("literature", "The Adventures of Sherlock Holmes (audiobook)",
           "LibriVox",
           "https://librivox.org/the-adventures-of-sherlock-holmes-by-sir-arthur-conan-doyle/",
           "audiobook", "all",
           "Doyle's stories, read by volunteers.",
           "LibriVox"),
    _entry("literature", "Pride and Prejudice (ebook)",
           "Project Gutenberg",
           "https://www.gutenberg.org/ebooks/1342", "ebook", "all",
           "Jane Austen's novel, public domain.",
           "Project Gutenberg"),
    _entry("literature", "Frankenstein (ebook)", "Project Gutenberg",
           "https://www.gutenberg.org/ebooks/84", "ebook", "all",
           "Mary Shelley's novel, public domain.",
           "Project Gutenberg"),
    _entry("film", "Feature Films collection", "Internet Archive",
           "https://archive.org/details/feature_films", "movie", "all",
           "Thousands of free feature films, public domain and "
           "Creative Commons.",
           "Internet Archive"),
]


# ── the durable collection ─────────────────────────────────────────────────

class LearningCollection:
    """The learning collection: entries + seeds, durable JSON.

    ``home`` is a directory; the store lives at
    ``<home>/learning_collection.json``. With ``home=None`` the
    collection is in-memory only (seeds, no disk writes) — the shape
    the tests and the offline path use.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        self.path: Path | None = (
            Path(home) / _COLLECTION_FILENAME if home is not None else None)
        self.entries: dict[str, CollectionEntry] = {}
        if self.path is not None and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for raw in data.get("entries", {}).values():
                    entry = CollectionEntry.from_dict(raw)
                    self.entries[entry.slug] = entry
            except Exception:  # noqa: BLE001 — corrupt store reseeds
                _log.warning("learning collection unreadable; reseeding",
                             exc_info=True)
                self.entries = {}
        if not self.entries:
            for entry in SEED_ENTRIES:
                self.entries[entry.slug] = entry
            self._save()

    # -- persistence ------------------------------------------------------

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "entries": {s: e.to_dict()
                            for s, e in self.entries.items()},
            }, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001 — the collection stays in memory
            _log.warning("learning collection save failed", exc_info=True)

    # -- mutation ---------------------------------------------------------

    def upsert(self, entry: CollectionEntry) -> None:
        """Add or refresh an entry (live search merges here)."""
        old = self.entries.get(entry.slug)
        if old is not None:
            # A live entry refreshes the seed but never erases a field
            # the live source didn't carry — and never overwrites a
            # checked health state with an unchecked one.
            merged = CollectionEntry.from_dict(old.to_dict())
            for key in ("description", "provider", "format", "level",
                        "provenance"):
                value = getattr(entry, key)
                if value not in ("", None):
                    setattr(merged, key, value)
            if entry.url:
                merged = replace(merged, url=entry.url)
            if entry.health_status != "unknown":
                merged = replace(
                    merged, health_status=entry.health_status,
                    checked_at=entry.checked_at,
                    last_error=entry.last_error)
            self.entries[entry.slug] = merged
        else:
            self.entries[entry.slug] = entry
        self._save()

    def remove(self, slug: str) -> bool:
        """Drop an entry by slug. Returns True when one was removed."""
        if slug not in self.entries:
            return False
        del self.entries[slug]
        self._save()
        return True

    # -- query ------------------------------------------------------------

    def all_entries(self) -> list[CollectionEntry]:
        return sorted(self.entries.values(),
                      key=lambda e: (e.subject, e.title.lower()))

    def subjects(self) -> list[str]:
        return sorted({e.subject for e in self.entries.values() if e.subject})

    def entry(self, slug: str) -> CollectionEntry | None:
        return self.entries.get(slug)


# ── health: the sourcehealth idiom, probed locally ─────────────────────────

@dataclass
class _ProbeResult:
    """One HTTP attempt. ``status`` 0 = transport failure (see error)."""

    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)
    error: str = ""


def _default_probe(url: str) -> _ProbeResult:
    req = urllib.request.Request(url, headers={
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,"
                  "application/xml;q=0.9,application/json;q=0.8,"
                  "*/*;q=0.7"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return _ProbeResult(
                resp.status, resp.read()[:2_000_000],
                {k.lower(): v for k, v in resp.headers.items()})
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()[:200_000]
        except Exception:  # noqa: BLE001
            body = b""
        return _ProbeResult(exc.code, body,
                            {k.lower(): v for k, v in
                             (exc.headers or {}).items()})
    except Exception as exc:  # noqa: BLE001 — transport failure
        return _ProbeResult(0, b"", {}, error=str(exc)[:300])


Fetcher = Callable[[str], _ProbeResult]


def _count_usable_links(body: bytes) -> int:
    text = body.decode("utf-8", errors="replace")
    return len(set(_LINK_RE.findall(text)))


def _classify(status: int, body: bytes, links: int) -> tuple[str, str]:
    """(status, detail) from one fetch + its usable-link count."""
    text = body.decode("utf-8", errors="replace")
    low = text[:4000].lower()
    if status == 0:
        return "unverifiable", "transport failure (evidence is about "\
                                "this network, not the resource)"
    if status in (404, 410):
        return "dead", f"HTTP {status}"
    if status == 451:
        return "geo-blocked", "HTTP 451 unavailable for legal reasons"
    if any(sig in low for sig in _GEO_SIGNALS):
        return "geo-blocked", f"HTTP {status} region-blocked page"
    if any(sig in low for sig in _BOT_SIGNALS):
        return "bot-blocked", f"HTTP {status} bot-wall/challenge page"
    if status in (403, 429, 202) and links == 0:
        return "bot-blocked", f"HTTP {status} no usable payload "\
                              "(anti-bot likely)"
    if status >= 500:
        return "unverifiable", f"HTTP {status} server error (transient)"
    if status != 200:
        return "unverifiable", f"HTTP {status}"
    if links <= 0:
        return "dead", "HTTP 200 but zero usable links "\
                        "(empty/changed/unparseable)"
    return "verified-live", f"{links} usable links"


def check_entry(entry: CollectionEntry, *,
                fetcher: Fetcher | None = None,
                now: float | None = None) -> CollectionEntry:
    """Probe one entry's URL; returns a new entry carrying the verdict.

    Never raises. A fresh probe never deletes the entry — it only
    updates its health state, so a dead link stays visible as dead.
    """
    now = time.time() if now is None else now
    fetch = fetcher or _default_probe
    try:
        result = fetch(entry.url)
    except Exception as exc:  # noqa: BLE001 — a fetcher must never kill health
        result = _ProbeResult(0, b"", {}, error=str(exc)[:300])
    links = _count_usable_links(result.body)
    status, detail = _classify(result.status, result.body, links)
    error = "" if status == "verified-live" else detail
    if result.error and result.status == 0:
        error = result.error
    return replace(entry, health_status=status, checked_at=now,
                   last_error=error)


def refresh_health(collection: LearningCollection, *,
                   fetcher: Fetcher | None = None,
                   now: float | None = None,
                   max_age_s: float = HEALTH_MAX_AGE_S) -> dict[str, Any]:
    """Re-probe every entry whose check is missing or stale.

    Never raises per entry. Returns the count checked and the slugs
    that came back dead or blocked — the caller decides what to do
    with them; this function never deletes.
    """
    now = time.time() if now is None else now
    checked = 0
    live = 0
    flagged: list[str] = []
    for entry in list(collection.entries.values()):
        if (entry.checked_at is not None
                and now - entry.checked_at < max_age_s):
            continue
        fresh = check_entry(entry, fetcher=fetcher, now=now)
        collection.entries[fresh.slug] = fresh
        checked += 1
        if fresh.health_status == "verified-live":
            live += 1
        else:
            flagged.append(fresh.slug)
    if checked:
        collection._save()
    return {"checked": checked, "live": live, "flagged": flagged}


# ── live search: the second rung ───────────────────────────────────────────

class _SearchLinksParser(HTMLParser):
    """Tolerant parser for OpenCulture's WordPress search page.

    A result is an anchor whose href is an openculture.com post/page
    and whose text is long enough to be a title; nav/chrome links
    (short text, or the site's own section links) are skipped by the
    same rule. Records what the page carries and nothing more.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._in_anchor = False
        self._href = ""
        self._text = ""

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._in_anchor = True
            self._href = dict(attrs).get("href", "") or ""
            self._text = ""

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in_anchor:
            return
        self._in_anchor = False
        text = self._text.strip()
        href = self._href
        self._href = ""
        self._text = ""
        if not href.startswith("https://www.openculture.com/"):
            return
        if len(text) < 12 or len(text) > 180:
            return
        if href.rstrip("/") in (
                "https://www.openculture.com",
                "https://www.openculture.com/freeonlinecourses",
                "https://www.openculture.com/free_textbooks",
                "https://www.openculture.com/freelanguagelessons"):
            return
        if (href, text) not in self.results:
            self.results.append((href, text))

    def handle_data(self, data: str) -> None:
        if self._in_anchor:
            self._text += data


def _fetch_text(url: str, fetcher: Fetcher | None) -> str:
    fetch = fetcher or _default_probe
    try:
        result = fetch(url)
    except Exception:  # noqa: BLE001
        return ""
    if result.status != 200:
        return ""
    return result.body.decode("utf-8", errors="replace")


def _fetch_json(url: str, fetcher: Fetcher | None) -> dict[str, Any]:
    text = _fetch_text(url, fetcher)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


class LiveLearningSearch:
    """Live search over the open directories: OpenCulture site search,
    Archive.org advanced search, Open Library search. All keyless.
    Every public method never raises."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher = fetcher

    # -- OpenCulture site search ------------------------------------------

    def search_openculture(self, query: str,
                           limit: int = 5) -> list[CollectionEntry]:
        """WordPress search over openculture.com's own posts."""
        query = (query or "").strip()
        if not query:
            return []
        url = ("https://www.openculture.com/?s="
               + urllib.parse.quote_plus(query))
        html = _fetch_text(url, self._fetcher)
        if not html:
            return []
        parser = _SearchLinksParser()
        try:
            parser.feed(html)
            parser.close()
        except Exception:  # noqa: BLE001 — a broken page parses to nothing
            _log.debug("openculture search parse failed", exc_info=True)
            return []
        # Relevance gate: the search page also carries the site's own
        # chrome (section/list links like "1,150 Free Movies"), which
        # never carries the query. A real result names the query in
        # its title or its URL slug.
        tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower())
                  if len(t) >= 3]
        out: list[CollectionEntry] = []
        for href, text in parser.results[: max(limit, 0) * 3]:
            hay = f"{text.lower()} {href.lower()}"
            if tokens and not any(tok in hay for tok in tokens):
                continue
            out.append(CollectionEntry(
                subject=query.lower(), title=text,
                provider="OpenCulture",
                url=href, format="", level="all",
                description="Found by OpenCulture site search.",
                provenance="OpenCulture site search",
                seeded_at=SEED_DATE))
            if len(out) >= max(limit, 0):
                break
        return out

    # -- Archive.org advanced search --------------------------------------

    def search_archive(self, query: str, *, mediatype: str = "texts",
                       limit: int = 5) -> list[CollectionEntry]:
        """Archive.org advancedsearch: identifier + title, keyless JSON."""
        query = (query or "").strip()
        if not query:
            return []
        url = ("https://archive.org/advancedsearch.php?q="
               + urllib.parse.quote_plus(
                   f"title:({query}) AND mediatype:({mediatype})")
               + "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=creator"
                 "&rows=" + str(max(limit, 0)) + "&output=json")
        data = _fetch_json(url, self._fetcher)
        docs = (data.get("response") or {}).get("docs") or []
        out: list[CollectionEntry] = []
        for doc in docs:
            ident = str(doc.get("identifier", "")).strip()
            title = str(doc.get("title", "")).strip()
            if not ident or not title:
                continue
            creator = str(doc.get("creator", "")).strip()
            out.append(CollectionEntry(
                subject=query.lower(), title=title,
                provider="Internet Archive" + (f" ({creator})"
                                               if creator else ""),
                url=f"https://archive.org/details/{ident}",
                format="movie" if mediatype == "movies" else "ebook",
                level="all",
                description="Found by Archive.org advanced search.",
                provenance="Archive.org advanced search",
                seeded_at=SEED_DATE))
        return out

    # -- Open Library search ----------------------------------------------

    def search_openlibrary(self, query: str,
                           limit: int = 5) -> list[CollectionEntry]:
        """Open Library search API: public-domain + open ebooks, keyless."""
        query = (query or "").strip()
        if not query:
            return []
        url = ("https://openlibrary.org/search.json?q="
               + urllib.parse.quote_plus(query)
               + "&limit=" + str(max(limit, 0)))
        data = _fetch_json(url, self._fetcher)
        docs = data.get("docs") or []
        out: list[CollectionEntry] = []
        for doc in docs:
            title = str(doc.get("title", "")).strip()
            key = str(doc.get("key", "")).strip()
            if not title or not key.startswith("/works/"):
                continue
            authors = doc.get("author_name") or []
            author = str(authors[0]) if authors else ""
            out.append(CollectionEntry(
                subject=query.lower(), title=title,
                provider="Open Library" + (f" ({author})" if author else ""),
                url=f"https://openlibrary.org{key}",
                format="ebook", level="all",
                description="Found by Open Library search.",
                provenance="Open Library search",
                seeded_at=SEED_DATE))
        return out

    def search_all(self, query: str, *,
                   limit: int = 5) -> list[CollectionEntry]:
        """All live rungs, de-duplicated by URL, in rung order."""
        seen: set[str] = set()
        out: list[CollectionEntry] = []
        for rung in (self.search_openculture, self.search_archive,
                     self.search_openlibrary):
            try:
                found = rung(query, limit=limit)
            except Exception:  # noqa: BLE001 — one dead rung never kills search
                continue
            for entry in found:
                if entry.url in seen:
                    continue
                seen.add(entry.url)
                out.append(entry)
                if len(out) >= max(limit, 0):
                    return out
        return out


# ── the query: vetted first, live search second ────────────────────────────

def _normalize_subject(raw: str) -> str:
    text = (raw or "").strip()
    text = _QUERY_STRIP_RE.sub("", text)
    text = _FREE_SUFFIX_RE.sub("", text)
    text = text.strip().strip(",.!?:;")
    return re.sub(r"\s+", " ", text).strip()


def _matches(entry: CollectionEntry, subject: str,
             format: str | None, level: str | None) -> bool:
    if format and entry.format != format:
        return False
    if level and entry.level not in ("all", level):
        return False
    if not subject:
        return True
    hay = " ".join((entry.subject, entry.title, entry.provider,
                    entry.description)).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", subject.lower()) if t]
    if not tokens:
        return True
    content = [tok for tok in tokens if len(tok) >= 3]
    # A subject word matches the entry's own subject field, or the
    # query's content words all appear somewhere in the entry.
    if any(tok in entry.subject.lower() for tok in content):
        return True
    if not content:
        return all(tok in hay for tok in tokens)
    return all(tok in hay for tok in content)


def learn_free(collection: LearningCollection, subject: str, *,
               format: str | None = None, level: str | None = None,
               limit: int = 5,
               live_search: LiveLearningSearch | None = None,
               fetcher: Fetcher | None = None) -> dict[str, Any]:
    """Answer "I want to learn X, free" from the collection first.

    Bands, in order: ``vetted`` (checked live, with the check date),
    ``unchecked`` (never checked — labelled as such, never wearing a
    fresh date), ``flagged`` (dead / blocked / unverifiable — shown
    with the reason, never recommended), ``live`` (the live-search
    rung, clearly separated). Never raises; an empty collection still
    answers, honestly.
    """
    subject = _normalize_subject(subject)
    limit = max(limit, 0)
    matched = [e for e in collection.all_entries()
               if _matches(e, subject, format, level)]
    vetted = [e for e in matched if e.health_status == "verified-live"]
    unchecked = [e for e in matched if e.health_status == "unknown"]
    flagged = [e for e in matched
               if e.health_status not in ("verified-live", "unknown")]
    searcher = live_search or LiveLearningSearch(fetcher=fetcher)
    known_urls = {e.url for e in matched}
    live: list[CollectionEntry] = []
    try:
        for entry in searcher.search_all(subject, limit=limit):
            if entry.url in known_urls:
                continue
            known_urls.add(entry.url)
            live.append(entry)
    except Exception:  # noqa: BLE001 — live search is best-effort
        live = []
    return {
        "ok": True,
        "subject": subject,
        "vetted": [e.to_dict() for e in vetted[:limit]],
        "unchecked": [e.to_dict() for e in unchecked[:limit]],
        "flagged": [e.to_dict() for e in flagged[:limit]],
        "live": [e.to_dict() for e in live[:limit]],
    }


# ── chat rendering ─────────────────────────────────────────────────────────

def _entry_line(entry: dict[str, Any]) -> str:
    head = f"- {entry['title']} ({entry['provider']})"
    bits = []
    if entry.get("format"):
        bits.append(entry["format"])
    if entry.get("level") and entry["level"] != "all":
        bits.append(entry["level"])
    if entry.get("provenance"):
        bits.append(f"via {entry['provenance']}")
    if entry.get("checked_date"):
        bits.append(f"checked {entry['checked_date']}")
    elif entry.get("health_status") == "unknown":
        bits.append("not yet checked")
    line = head + (" — " + " · ".join(bits) if bits else "")
    if entry.get("description"):
        line += f"\n  {entry['description']}"
    if entry.get("url"):
        line += f"\n  {entry['url']}"
    if entry.get("last_error"):
        line += f"\n  ⚠ {entry['last_error']}"
    return line


def format_learning_reply(subject: str, result: dict[str, Any]) -> str:
    """Plain-words reply for the brain tool (and any chat surface)."""
    subject = subject or "that subject"
    lines = [f"Free ways to learn {subject}:"]
    vetted = result.get("vetted") or []
    unchecked = result.get("unchecked") or []
    flagged = result.get("flagged") or []
    live = result.get("live") or []
    if vetted:
        lines.append("\nVetted — checked live:")
        lines.extend(_entry_line(e) for e in vetted)
    if unchecked:
        lines.append("\nIn the collection, not yet checked:")
        lines.extend(_entry_line(e) for e in unchecked)
    if flagged:
        lines.append("\nFlagged — not recommended, kept for honesty:")
        lines.extend(_entry_line(e) for e in flagged)
    if live:
        lines.append("\nLive search (not vetted yet):")
        lines.extend(_entry_line(e) for e in live)
    if not (vetted or unchecked or flagged or live):
        lines.append("Nothing found — try a broader subject, or a "
                     "different format.")
    lines.append("\nVetted entries were checked live on the date shown; "
                 "unchecked ones carry no check date yet.")
    return "\n".join(lines)
