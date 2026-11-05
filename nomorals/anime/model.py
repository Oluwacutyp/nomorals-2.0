"""Normalized anime records + shape-validating normalizers.

Source payloads (AniList GraphQL, Jikan REST) are validated field
by field as they are normalized: a payload missing a field the
record cannot exist without raises :class:`ShapeError` naming the
field — the caller reports a named source failure instead of
rendering a half-invented record.  Optional fields degrade to
``None``/empty and the presentation layer simply omits them.

Internal source ids live on the records (they key favorites and
follow-up lookups) but are NEVER rendered to the user — members
navigate by title and pick numbers.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AiringEntry",
    "CharacterRecord",
    "EpisodeEntry",
    "MediaRecord",
    "PersonRecord",
    "QuoteRecord",
    "ShapeError",
    "StudioRecord",
    "media_from_anilist",
    "media_from_jikan",
]


class ShapeError(Exception):
    """A source payload did not carry the shape the code reads."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source} shape changed: {detail}")
        self.source = source
        self.detail = detail


_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(raw: Any, *, limit: int = 0) -> str:
    """Strip AniList's HTML-ish description markup to plain words."""
    text = html.unescape(_TAG_RE.sub(" ", str(raw or "")))
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _title_of(node: dict[str, Any]) -> tuple[str, str, str]:
    """(display, romaji, native) from an AniList title block."""
    title = node.get("title")
    if not isinstance(title, dict):
        raise ShapeError("AniList", "media.title missing")
    romaji = str(title.get("romaji") or "").strip()
    english = str(title.get("english") or "").strip()
    native = str(title.get("native") or "").strip()
    display = english or romaji or native
    if not display:
        raise ShapeError("AniList", "media.title carries no name")
    return display, romaji or display, native


@dataclass
class MediaRecord:
    """One anime or manga record, source-agnostic."""

    source: str                       # "anilist" | "jikan"
    source_id: int                    # internal only — never printed
    kind: str                         # "anime" | "manga"
    title: str                        # display title
    title_romaji: str = ""
    title_native: str = ""
    format: str = ""                  # TV | MOVIE | MANGA | …
    status: str = ""                  # RELEASING | FINISHED | …
    score: float | None = None        # average score, 0–100 scale
    popularity: int | None = None
    favourites: int | None = None
    episodes: int | None = None
    chapters: int | None = None
    duration_min: int | None = None
    season: str = ""
    year: int | None = None
    genres: list[str] = field(default_factory=list)
    description: str = ""
    cover_url: str = ""
    studios: list[str] = field(default_factory=list)
    external_links: list[tuple[str, str, str]] = field(
        default_factory=list)         # (site, url, type)
    trailer_url: str = ""
    mal_id: int | None = None
    mal_score: float | None = None    # Jikan cross-check, 0–10 scale
    next_episode: int | None = None
    next_airing_at: int | None = None  # epoch seconds
    characters: list[CharacterRecord] = field(default_factory=list)
    staff: list[tuple[str, str]] = field(default_factory=list)
    recommendations: list[MediaRecord] = field(default_factory=list)
    reviews: list[tuple[str, int | None]] = field(default_factory=list)
    streaming_episodes: list[EpisodeEntry] = field(default_factory=list)

    @property
    def ref(self) -> str:
        """Opaque follow-up key (internal; encodes source + id)."""
        return f"{self.source}:{self.source_id}"


@dataclass
class CharacterRecord:
    source: str
    source_id: int
    name: str
    name_native: str = ""
    image_url: str = ""
    description: str = ""
    favourites: int | None = None
    role: str = ""                    # MAIN | SUPPORTING (per-title)
    media_titles: list[str] = field(default_factory=list)


@dataclass
class PersonRecord:
    """A staff member (voice actor, director, …)."""

    source: str
    source_id: int
    name: str
    name_native: str = ""
    image_url: str = ""
    description: str = ""
    favourites: int | None = None
    occupations: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    media_titles: list[str] = field(default_factory=list)


@dataclass
class StudioRecord:
    source: str
    source_id: int
    name: str
    is_animation_studio: bool = False
    favourites: int | None = None
    works: list[MediaRecord] = field(default_factory=list)


@dataclass
class EpisodeEntry:
    number: int | None = None
    title: str = ""
    url: str = ""
    site: str = ""
    aired_at: int | None = None       # epoch seconds


@dataclass
class AiringEntry:
    media: MediaRecord
    episode: int
    airing_at: int                    # epoch seconds


@dataclass
class QuoteRecord:
    text: str
    anime_title: str = ""
    character_name: str = ""


# ── AniList normalizers ─────────────────────────────────────────────

def media_from_anilist(node: Any, *, detail: bool = False) -> MediaRecord:
    if not isinstance(node, dict):
        raise ShapeError("AniList", "media node is not an object")
    mid = node.get("id")
    if not isinstance(mid, int):
        raise ShapeError("AniList", "media.id missing")
    display, romaji, native = _title_of(node)
    type_word = str(node.get("type") or "").upper()
    cover = node.get("coverImage") or {}
    start = node.get("startDate") or {}
    rec = MediaRecord(
        source="anilist",
        source_id=mid,
        kind="manga" if type_word == "MANGA" else "anime",
        title=display,
        title_romaji=romaji,
        title_native=native,
        format=str(node.get("format") or ""),
        status=str(node.get("status") or ""),
        score=_as_float(node.get("averageScore")),
        popularity=_as_int(node.get("popularity")),
        favourites=_as_int(node.get("favourites")),
        episodes=_as_int(node.get("episodes")),
        chapters=_as_int(node.get("chapters")),
        duration_min=_as_int(node.get("duration")),
        season=str(node.get("season") or ""),
        year=_as_int(start.get("year")) if isinstance(start, dict) else None,
        genres=[str(g) for g in (node.get("genres") or [])
                if isinstance(g, str)],
        description=clean_text(node.get("description"), limit=600),
        cover_url=str((cover or {}).get("extraLarge")
                      or (cover or {}).get("large") or ""),
        mal_id=_as_int(node.get("idMal")),
    )
    studios = node.get("studios") or {}
    if isinstance(studios, dict):
        rec.studios = [str(n.get("name") or "")
                       for n in (studios.get("nodes") or [])
                       if isinstance(n, dict) and n.get("name")]
    for link in node.get("externalLinks") or []:
        if isinstance(link, dict) and link.get("site") and link.get("url"):
            rec.external_links.append((str(link["site"]), str(link["url"]),
                                       str(link.get("type") or "")))
    trailer = node.get("trailer") or {}
    if isinstance(trailer, dict) and trailer.get("id"):
        site = str(trailer.get("site") or "youtube").lower()
        if site == "youtube":
            rec.trailer_url = (
                f"https://www.youtube.com/watch?v={trailer['id']}")
        elif site == "dailymotion":
            rec.trailer_url = f"https://www.dailymotion.com/video/{trailer['id']}"
    nxt = node.get("nextAiringEpisode") or {}
    if isinstance(nxt, dict):
        rec.next_episode = _as_int(nxt.get("episode"))
        rec.next_airing_at = _as_int(nxt.get("airingAt"))
    if detail:
        _fill_detail(rec, node)
    return rec


def _fill_detail(rec: MediaRecord, node: dict[str, Any]) -> None:
    chars = node.get("characters") or {}
    if isinstance(chars, dict):
        for edge in chars.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            cnode = edge.get("node") or {}
            try:
                crec = character_from_anilist(cnode)
            except ShapeError:
                continue
            crec.role = str(edge.get("role") or "")
            rec.characters.append(crec)
    staff = node.get("staff") or {}
    if isinstance(staff, dict):
        for edge in staff.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            snode = edge.get("node") or {}
            name = ((snode.get("name") or {}).get("full")
                    if isinstance(snode, dict) else "")
            if name:
                rec.staff.append((str(name), str(edge.get("role") or "")))
    recs = node.get("recommendations") or {}
    if isinstance(recs, dict):
        for rnode in recs.get("nodes") or []:
            target = (rnode or {}).get("mediaRecommendation") \
                if isinstance(rnode, dict) else None
            if isinstance(target, dict):
                try:
                    rec.recommendations.append(media_from_anilist(target))
                except ShapeError:
                    continue
    reviews = node.get("reviews") or {}
    if isinstance(reviews, dict):
        for rnode in reviews.get("nodes") or []:
            if isinstance(rnode, dict):
                summary = clean_text(rnode.get("summary"), limit=220)
                if summary:
                    rec.reviews.append(
                        (summary, _as_int(rnode.get("score"))))
    for ep in node.get("streamingEpisodes") or []:
        if isinstance(ep, dict) and ep.get("title"):
            rec.streaming_episodes.append(EpisodeEntry(
                title=clean_text(ep.get("title"), limit=120),
                url=str(ep.get("url") or ""),
                site=str(ep.get("site") or "")))


def character_from_anilist(node: Any) -> CharacterRecord:
    if not isinstance(node, dict):
        raise ShapeError("AniList", "character node is not an object")
    cid = node.get("id")
    if not isinstance(cid, int):
        raise ShapeError("AniList", "character.id missing")
    name = node.get("name") or {}
    full = str((name or {}).get("full") or "").strip() \
        if isinstance(name, dict) else ""
    if not full:
        raise ShapeError("AniList", "character.name.full missing")
    image = node.get("image") or {}
    rec = CharacterRecord(
        source="anilist", source_id=cid, name=full,
        name_native=str((name or {}).get("native") or ""),
        image_url=str((image or {}).get("large") or "")
        if isinstance(image, dict) else "",
        description=clean_text(node.get("description"), limit=500),
        favourites=_as_int(node.get("favourites")),
    )
    media = node.get("media") or {}
    if isinstance(media, dict):
        for mnode in media.get("nodes") or []:
            if isinstance(mnode, dict):
                try:
                    display, _, _ = _title_of(mnode)
                    rec.media_titles.append(display)
                except ShapeError:
                    continue
    return rec


def person_from_anilist(node: Any) -> PersonRecord:
    if not isinstance(node, dict):
        raise ShapeError("AniList", "staff node is not an object")
    sid = node.get("id")
    if not isinstance(sid, int):
        raise ShapeError("AniList", "staff.id missing")
    name = node.get("name") or {}
    full = str((name or {}).get("full") or "").strip() \
        if isinstance(name, dict) else ""
    if not full:
        raise ShapeError("AniList", "staff.name.full missing")
    image = node.get("image") or {}
    rec = PersonRecord(
        source="anilist", source_id=sid, name=full,
        name_native=str((name or {}).get("native") or ""),
        image_url=str((image or {}).get("large") or "")
        if isinstance(image, dict) else "",
        description=clean_text(node.get("description"), limit=400),
        favourites=_as_int(node.get("favourites")),
        occupations=[str(o) for o in (node.get("primaryOccupations") or [])
                     if isinstance(o, str)],
    )
    chars = node.get("characters") or {}
    if isinstance(chars, dict):
        for cnode in chars.get("nodes") or []:
            if isinstance(cnode, dict):
                cname = ((cnode.get("name") or {}).get("full")
                         if isinstance(cnode.get("name"), dict) else "")
                if cname:
                    rec.characters.append(str(cname))
    media = node.get("media") or {}
    if isinstance(media, dict):
        for mnode in media.get("nodes") or []:
            if isinstance(mnode, dict):
                try:
                    display, _, _ = _title_of(mnode)
                    rec.media_titles.append(display)
                except ShapeError:
                    continue
    return rec


def studio_from_anilist(node: Any) -> StudioRecord:
    if not isinstance(node, dict):
        raise ShapeError("AniList", "studio node is not an object")
    sid = node.get("id")
    if not isinstance(sid, int):
        raise ShapeError("AniList", "studio.id missing")
    name = str(node.get("name") or "").strip()
    if not name:
        raise ShapeError("AniList", "studio.name missing")
    rec = StudioRecord(
        source="anilist", source_id=sid, name=name,
        is_animation_studio=bool(node.get("isAnimationStudio")),
        favourites=_as_int(node.get("favourites")),
    )
    media = node.get("media") or {}
    if isinstance(media, dict):
        for mnode in media.get("nodes") or []:
            try:
                rec.works.append(media_from_anilist(mnode))
            except ShapeError:
                continue
    return rec


# ── Jikan normalizers ───────────────────────────────────────────────

def media_from_jikan(node: Any, *, kind: str = "anime") -> MediaRecord:
    if not isinstance(node, dict):
        raise ShapeError("Jikan", "entry is not an object")
    mid = node.get("mal_id")
    if not isinstance(mid, int):
        raise ShapeError("Jikan", "entry.mal_id missing")
    title = str(node.get("title_english") or node.get("title")
                or "").strip()
    if not title:
        raise ShapeError("Jikan", "entry carries no title")
    images = node.get("images") or {}
    jpg = (images.get("jpg") or {}) if isinstance(images, dict) else {}
    aired = node.get("aired") or node.get("published") or {}
    year = None
    if isinstance(aired, dict):
        prop = (aired.get("prop") or {}).get("from") or {}
        year = _as_int(prop.get("year")) if isinstance(prop, dict) else None
    rec = MediaRecord(
        source="jikan", source_id=mid, kind=kind, title=title,
        title_romaji=str(node.get("title") or ""),
        title_native=str(node.get("title_japanese") or ""),
        format=str(node.get("type") or ""),
        status=str(node.get("status") or ""),
        score=None,
        popularity=_as_int(node.get("popularity")),
        favourites=_as_int(node.get("favorites")),
        episodes=_as_int(node.get("episodes")),
        chapters=_as_int(node.get("chapters")),
        year=year,
        genres=[str((g or {}).get("name") or "")
                for g in (node.get("genres") or [])
                if isinstance(g, dict) and (g or {}).get("name")],
        description=clean_text(node.get("synopsis"), limit=600),
        cover_url=str(jpg.get("large_image_url")
                      or jpg.get("image_url") or ""),
        mal_id=mid,
    )
    raw_score = _as_float(node.get("score"))
    if raw_score is not None:
        rec.mal_score = raw_score
    studios = node.get("studios") or node.get("authors") or []
    if isinstance(studios, list):
        rec.studios = [str((s or {}).get("name") or "")
                       for s in studios
                       if isinstance(s, dict) and (s or {}).get("name")]
    trailer = node.get("trailer") or {}
    if isinstance(trailer, dict) and trailer.get("youtube_id"):
        rec.trailer_url = (
            f"https://www.youtube.com/watch?v={trailer['youtube_id']}")
    return rec


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
