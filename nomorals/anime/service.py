"""Anime service — one method per capability, records in, outcomes out.

The service owns the source discipline (seeds walked in order,
attempts accounted, cross-checks by title) and returns structured
:class:`Outcome` values; presentation (typography cards, buttons,
image delivery) belongs to the caller.  Nothing here invents a
record: every title shown came out of a source response, and every
failure says which sources were tried and why they failed.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .favorites import FavoritesStore
from .model import (
    AiringEntry,
    CharacterRecord,
    MediaRecord,
    PersonRecord,
    ShapeError,
    StudioRecord,
)
from .sources import (
    AniListClient,
    ImagePoolClient,
    JikanClient,
    QuoteClient,
    SourceFailure,
    TraceMoeClient,
)

_log = logging.getLogger(__name__)

__all__ = ["AnimeService", "Outcome", "ResultCache", "WAT"]

#: West Africa Time — the owner's timezone; UTC+1 all year, no DST,
#: so a fixed offset is exact (and immune to Windows tzdata gaps).
WAT = timezone(timedelta(hours=1), "WAT")

#: trace.moe matches below this similarity are guesses, not answers.
SIMILARITY_FLOOR = 0.85

_LIST_CAP = 10


@dataclass
class Outcome:
    """One capability's structured answer, ready to present."""

    ok: bool
    title: str = ""
    icon: str = "🎌"
    lines: list[str] = field(default_factory=list)
    stats: list[tuple[str, Any]] = field(default_factory=list)
    image_url: str = ""
    image_caption: str = ""
    picks: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""
    error: str = ""

    @classmethod
    def fail(cls, error: str, *, title: str = "") -> Outcome:
        return cls(ok=False, title=title, error=error)


class ResultCache:
    """Last shown list per (chat, kind), for "/<kind> open <n>"."""

    def __init__(self, ttl: float = 600.0) -> None:
        self._ttl = ttl
        self._store: dict[tuple[str, str], tuple[float, list[Any]]] = {}

    def put(self, chat_key: str, kind: str, items: list[Any]) -> None:
        self._store[(chat_key, kind)] = (time.time(), list(items))

    def get(self, chat_key: str, kind: str) -> list[Any] | None:
        entry = self._store.get((chat_key, kind))
        if entry is None:
            return None
        stamped, items = entry
        if time.time() - stamped > self._ttl:
            self._store.pop((chat_key, kind), None)
            return None
        return items


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _countdown(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _wat_label(epoch: int) -> str:
    dt = datetime.fromtimestamp(epoch, tz=WAT)
    return dt.strftime("%a %d %b, %H:%M") + " WAT"


def _score_text(rec: MediaRecord) -> str:
    if rec.score is not None:
        return f"★ {rec.score:.0f}"
    if rec.mal_score is not None:
        return f"★ {rec.mal_score * 10:.0f}"
    return "unscored"


class AnimeService:
    """The anime organ's brain: sources in, outcomes out."""

    def __init__(self, *, anilist: AniListClient, jikan: JikanClient,
                 quotes: QuoteClient, images: ImagePoolClient,
                 trace: TraceMoeClient, favorites: FavoritesStore,
                 rng: random.Random | None = None,
                 clock: Callable[[], float] = time.time,
                 cache: ResultCache | None = None) -> None:
        self._anilist = anilist
        self._jikan = jikan
        self._quotes = quotes
        self._images = images
        self._trace = trace
        self._favorites = favorites
        self._rng = rng if rng is not None else random.Random()
        self._clock = clock
        self._cache = cache if cache is not None else ResultCache()

    # ── shared resolution ─────────────────────────────────────────
    def _search(self, kind: str, query: str) \
            -> tuple[list[MediaRecord], list[str]]:
        """AniList first, Jikan as the second seed; attempts kept."""
        attempts: list[str] = []
        try:
            found = self._anilist.search_media(query, kind=kind)
            if found:
                return found, attempts
            attempts.append("AniList: no match")
        except (SourceFailure, ShapeError) as exc:
            attempts.append(f"AniList: {exc}")
        try:
            found = self._jikan.search_media(query, kind=kind)
            if found:
                return found, attempts
            attempts.append("Jikan: no match")
        except (SourceFailure, ShapeError) as exc:
            attempts.append(f"Jikan: {exc}")
        return [], attempts

    def _resolve(self, kind: str, query: str) \
            -> tuple[MediaRecord | None, list[str]]:
        found, attempts = self._search(kind, query)
        return (found[0] if found else None), attempts

    def _not_found(self, kind: str, query: str,
                   attempts: list[str]) -> Outcome:
        tried = "; ".join(attempts) if attempts else "no sources answered"
        label = "manga" if kind == "manga" else "anime"
        return Outcome.fail(
            f"couldn't find {label} “{query}” — {tried}. "
            "Try the exact title, or the original Japanese name.")

    def _detail(self, rec: MediaRecord) -> tuple[MediaRecord, str]:
        """Enrich a search hit with the full record + MAL cross-check.

        Returns (record, note).  Enrichment failing never kills the
        answer: the search record stands, and the note says so."""
        note = ""
        if rec.source == "anilist":
            try:
                rec = self._anilist.media_detail(rec.source_id)
            except (SourceFailure, ShapeError):
                note = "full record didn't load — showing the search hit"
        if rec.source == "anilist" and rec.mal_score is None:
            try:
                hits = self._jikan.search_media(rec.title_romaji
                                                or rec.title,
                                                kind=rec.kind, limit=5)
                want = {_norm_title(rec.title), _norm_title(rec.title_romaji)}
                for hit in hits:
                    names = {_norm_title(hit.title),
                             _norm_title(hit.title_romaji)}
                    if want & names and hit.mal_score is not None:
                        rec.mal_score = hit.mal_score
                        break
            except (SourceFailure, ShapeError) as exc:
                _log.debug("service._detail failed", exc_info=True)
        return rec, note

    # ── title lookup: /anime /manga ───────────────────────────────
    def search_title(self, kind: str, query: str, *,
                     chat_key: str) -> Outcome:
        found, attempts = self._search(kind, query)
        if not found:
            return self._not_found(kind, query, attempts)
        self._cache.put(chat_key, kind, found)
        top, note = self._detail(found[0])
        out = self._media_card(top, note=note)
        out.picks = [
            (f"open: {rec.title[:24]}", f"/{kind} open {i}")
            for i, rec in enumerate(found[1:5], start=2)]
        if len(found) > 1:
            others = " · ".join(
                f"{i}. {rec.title}"
                for i, rec in enumerate(found[1:5], start=2))
            out.lines.append(f"also matched: {others} — "
                             "open one with its number")
        return out

    def open_result(self, kind: str, index: int, *,
                    chat_key: str) -> Outcome:
        items = self._cache.get(chat_key, kind)
        if not items:
            return Outcome.fail(
                f"that list expired — run /{kind} again first.")
        if index < 1 or index > len(items):
            return Outcome.fail(
                f"pick a number 1–{len(items)} from the last list.")
        item = items[index - 1]
        if isinstance(item, MediaRecord):
            top, note = self._detail(item)
            return self._media_card(top, note=note)
        if isinstance(item, CharacterRecord):
            return self._character_card(item)
        if isinstance(item, PersonRecord):
            return self._person_card(item)
        if isinstance(item, StudioRecord):
            return self._studio_card(item)
        return Outcome.fail("that entry can't be opened.")

    def _media_card(self, rec: MediaRecord, *, note: str = "") -> Outcome:
        out = Outcome(ok=True, title=rec.title, icon="🎌")
        out.image_url = rec.cover_url
        out.image_caption = rec.title
        rows: list[tuple[str, Any]] = []
        if rec.format:
            rows.append(("Format", rec.format.replace("_", " ").title()))
        if rec.status:
            rows.append(("Status", rec.status.replace("_", " ").title()))
        if rec.score is not None:
            rows.append(("Score", f"{rec.score:.0f}/100 (AniList)"))
        if rec.mal_score is not None:
            rows.append(("MAL score", f"{rec.mal_score:.2f}/10"))
        if rec.episodes:
            rows.append(("Episodes", rec.episodes))
        if rec.chapters:
            rows.append(("Chapters", rec.chapters))
        if rec.year:
            season = f"{rec.season.title()} " if rec.season else ""
            rows.append(("Released", f"{season}{rec.year}"))
        if rec.studios:
            rows.append(("Studio" if rec.kind == "anime" else "Author",
                         ", ".join(rec.studios[:3])))
        if rec.genres:
            rows.append(("Genres", ", ".join(rec.genres[:6])))
        out.stats = rows
        if rec.description:
            out.lines.append(rec.description)
        if rec.next_airing_at:
            out.lines.append(
                f"episode {rec.next_episode} airs "
                f"{_wat_label(rec.next_airing_at)} "
                f"(in {_countdown(rec.next_airing_at - self._clock())})")
        src = "AniList" if rec.source == "anilist" else "MyAnimeList"
        out.note = note or f"record via {src}"
        if rec.source == "anilist" and rec.mal_score is not None:
            out.note = (note + " · " if note else "") + \
                "MAL score cross-checked on MyAnimeList"
        return out

    # ── people & studios ──────────────────────────────────────────
    def search_character(self, query: str, *,
                         chat_key: str) -> Outcome:
        try:
            found = self._anilist.search_characters(query)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"character search failed — {exc}")
        if not found:
            return Outcome.fail(
                f"no character named “{query}” on AniList. "
                "Try the full name, family name last.")
        self._cache.put(chat_key, "anichar", found)
        out = self._character_card(found[0])
        out.picks = [(f"open: {c.name[:24]}", f"/anichar open {i}")
                     for i, c in enumerate(found[1:5], start=2)]
        return out

    def _character_card(self, rec: CharacterRecord) -> Outcome:
        out = Outcome(ok=True, title=rec.name, icon="🧑")
        out.image_url = rec.image_url
        out.image_caption = rec.name
        if rec.name_native:
            out.lines.append(f"native name: {rec.name_native}")
        if rec.favourites:
            out.stats = [("Favourites", rec.favourites)]
        if rec.media_titles:
            out.lines.append("known from: "
                             + ", ".join(rec.media_titles[:5]))
        if rec.description:
            out.lines.append(rec.description)
        out.note = "record via AniList"
        return out

    def search_staff(self, query: str, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.search_staff(query)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"staff search failed — {exc}")
        if not found:
            return Outcome.fail(
                f"no staff member named “{query}” on AniList.")
        self._cache.put(chat_key, "staff", found)
        out = self._person_card(found[0])
        out.picks = [(f"open: {p.name[:24]}", f"/staff open {i}")
                     for i, p in enumerate(found[1:5], start=2)]
        return out

    def _person_card(self, rec: PersonRecord) -> Outcome:
        out = Outcome(ok=True, title=rec.name, icon="🎙️")
        out.image_url = rec.image_url
        out.image_caption = rec.name
        rows: list[tuple[str, Any]] = []
        if rec.occupations:
            rows.append(("Work", ", ".join(rec.occupations[:3])))
        if rec.favourites:
            rows.append(("Favourites", rec.favourites))
        out.stats = rows
        if rec.characters:
            out.lines.append("voices / plays: "
                             + ", ".join(rec.characters[:5]))
        if rec.media_titles:
            out.lines.append("worked on: "
                             + ", ".join(rec.media_titles[:5]))
        if rec.description:
            out.lines.append(rec.description)
        out.note = "record via AniList"
        return out

    def search_studio(self, query: str, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.search_studios(query)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"studio search failed — {exc}")
        if not found:
            return Outcome.fail(f"no studio named “{query}” on AniList.")
        self._cache.put(chat_key, "studio", found)
        out = self._studio_card(found[0])
        out.picks = [(f"open: {s.name[:24]}", f"/studio open {i}")
                     for i, s in enumerate(found[1:5], start=2)]
        return out

    def _studio_card(self, rec: StudioRecord) -> Outcome:
        out = Outcome(ok=True, title=rec.name, icon="🏢")
        rows: list[tuple[str, Any]] = []
        rows.append(("Animation studio",
                     "yes" if rec.is_animation_studio else "no"))
        if rec.favourites:
            rows.append(("Favourites", rec.favourites))
        out.stats = rows
        if rec.works:
            out.lines.append("best-known works:")
            out.lines.extend(
                f"• {w.title} — {_score_text(w)}" for w in rec.works[:6])
        if rec.works and rec.works[0].cover_url:
            out.image_url = rec.works[0].cover_url
            out.image_caption = rec.name
        out.note = "record via AniList"
        return out

    # ── per-title capabilities ────────────────────────────────────
    def cast_of(self, query: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            return self._not_found("anime", query, attempts)
        full, note = self._detail(rec)
        if not full.characters:
            return Outcome.fail(
                f"no cast on the record for “{full.title}” yet.",
                title=full.title)
        out = Outcome(ok=True, title=f"{full.title} — cast", icon="🎭")
        out.image_url = full.cover_url
        out.image_caption = full.title
        out.lines = [
            f"• {c.name}" + (f" — {c.role.lower()}"
                              if c.role else "")
            for c in full.characters[:12]]
        out.note = note or "cast via AniList"
        return out

    def episodes(self, query: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            return self._not_found("anime", query, attempts)
        full, note = self._detail(rec)
        out = Outcome(ok=True, title=f"{full.title} — episodes",
                      icon="📺")
        out.image_url = full.cover_url
        out.image_caption = full.title
        if full.episodes:
            out.stats = [("Episodes", full.episodes)]
        shown = 0
        if full.streaming_episodes:
            out.lines.append("streaming episode list:")
            for ep in full.streaming_episodes[:_LIST_CAP]:
                out.lines.append(f"• {ep.title}"
                                 + (f" ({ep.site})" if ep.site else ""))
                shown += 1
        if full.mal_id:
            try:
                eps = self._jikan.episodes(full.mal_id)
            except (SourceFailure, ShapeError):
                eps = []
            if eps:
                out.lines.append("episode guide (MyAnimeList):")
                for num, title, aired in eps[:_LIST_CAP]:
                    suffix = f" — {aired}" if aired else ""
                    out.lines.append(f"• {num}. {title}{suffix}")
                shown += len(eps[:_LIST_CAP])
        if not shown:
            if full.episodes:
                out.lines.append(
                    f"{full.episodes} episode(s) on the record, but no "
                    "per-episode list is published for it yet.")
            else:
                out.lines.append(
                    "no episode information on the record yet.")
        out.note = note or "via AniList + MyAnimeList"
        return out

    def trailer(self, query: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            return self._not_found("anime", query, attempts)
        full, note = self._detail(rec)
        if not full.trailer_url:
            return Outcome.fail(
                f"no trailer on the record for “{full.title}”.",
                title=full.title)
        out = Outcome(ok=True, title=f"{full.title} — trailer", icon="🎬")
        out.image_url = full.cover_url
        out.image_caption = full.title
        out.lines = [full.trailer_url,
                     "tap below (or /video the link) to watch it here."]
        out.picks = [("▶ watch trailer", f"/video {full.trailer_url}")]
        out.note = note or "trailer via the record's own link"
        return out

    def streaming(self, query: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            return self._not_found("anime", query, attempts)
        full, note = self._detail(rec)
        links = [(site, url) for site, url, typ in full.external_links
                 if typ.upper() == "STREAMING"]
        out = Outcome(ok=True, title=f"{full.title} — where to watch",
                      icon="📡")
        out.image_url = full.cover_url
        out.image_caption = full.title
        if links:
            out.lines = [f"• {site}: {url}" for site, url in links[:8]]
        else:
            out.lines = ["no listed streaming source on the record — "
                         "that means no legal watch link is filed "
                         "for it, not that none exists anywhere."]
        out.note = note or "links via AniList"
        return out

    def review(self, query: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            rec, attempts2 = self._resolve("manga", query)
            if rec is None:
                return self._not_found("anime", query,
                                       attempts + attempts2)
        full, note = self._detail(rec)
        out = Outcome(ok=True, title=f"{full.title} — reviews",
                      icon="📝")
        out.image_url = full.cover_url
        out.image_caption = full.title
        rows: list[tuple[str, Any]] = []
        if full.score is not None:
            rows.append(("Community score", f"{full.score:.0f}/100"))
        if full.mal_score is not None:
            rows.append(("MAL score", f"{full.mal_score:.2f}/10"))
        if full.popularity:
            rows.append(("Popularity rank pool", full.popularity))
        out.stats = rows
        if full.reviews:
            out.lines.append("top community reviews (AniList):")
            for summary, score in full.reviews:
                prefix = f"[{score}/100] " if score else ""
                out.lines.append(f"• {prefix}“{summary}”")
        else:
            out.lines.append(
                "no written community reviews on the record — "
                "the scores above are what the community filed.")
        out.note = note or "scores + reviews via AniList"
        return out

    def next_episode(self, query: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            return self._not_found("anime", query, attempts)
        full, note = self._detail(rec)
        out = Outcome(ok=True, title=full.title, icon="⏭️")
        out.image_url = full.cover_url
        out.image_caption = full.title
        if full.next_airing_at:
            out.lines = [
                f"episode {full.next_episode} airs "
                f"{_wat_label(full.next_airing_at)} — in "
                f"{_countdown(full.next_airing_at - self._clock())}."]
            out.note = note or "schedule via AniList · times read in WAT"
        elif full.status == "RELEASING":
            out.lines = ["it's still releasing, but the next episode "
                         "isn't on the published schedule yet."]
        elif full.status == "FINISHED":
            total = f" — {full.episodes} episodes" if full.episodes \
                else ""
            out.lines = [f"finished{total}, all aired. Nothing "
                         "scheduled."]
        elif full.status == "NOT_YET_RELEASED":
            when = f" in {full.year}" if full.year else ""
            out.lines = [f"not out yet{when} — no episode schedule "
                         "published."]
        else:
            out.lines = [f"status on the record: "
                         f"{full.status or 'unknown'} — no next-episode "
                         "information filed."]
        return out

    def recommend(self, query: str, *, chat_key: str) -> Outcome:
        rec, attempts = self._resolve("anime", query)
        if rec is None:
            rec, attempts2 = self._resolve("manga", query)
            if rec is None:
                return self._not_found("anime", query,
                                       attempts + attempts2)
        full, note = self._detail(rec)
        out = Outcome(ok=True, title=f"if you liked {full.title}",
                      icon="💡")
        out.image_url = full.cover_url
        out.image_caption = full.title
        picks_from: list[MediaRecord] = []
        if full.recommendations:
            picks_from = full.recommendations[:_LIST_CAP]
            out.lines = ["AniList readers also filed:"]
        elif full.genres:
            try:
                picks_from = self._anilist.browse_media(
                    kind=full.kind, sort="SCORE_DESC",
                    genre=full.genres[0], per_page=_LIST_CAP + 1)
                picks_from = [r for r in picks_from
                              if r.source_id != full.source_id]
                picks_from = picks_from[:_LIST_CAP]
                out.lines = [f"no filed recommendations — top "
                             f"{full.genres[0]} by score instead:"]
            except (SourceFailure, ShapeError):
                picks_from = []
        if not picks_from:
            return Outcome.fail(
                f"no recommendations on the record for "
                f"“{full.title}”, and no genre fallback answered.",
                title=full.title)
        out.lines.extend(
            f"{i}. {r.title} — {_score_text(r)}"
            for i, r in enumerate(picks_from, start=1))
        self._cache.put(chat_key, full.kind, picks_from)
        out.note = note or "recommendations via AniList"
        out.picks = [(f"open: {r.title[:24]}",
                      f"/{full.kind} open {i}")
                     for i, r in enumerate(picks_from, start=1)]
        return out

    # ── genres ────────────────────────────────────────────────────
    def genres(self) -> Outcome:
        try:
            genres = self._anilist.genres()
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"genre list failed — {exc}")
        out = Outcome(ok=True, title="Anime genres on AniList",
                      icon="🏷️")
        out.lines = [", ".join(genres)]
        out.note = "browse one with /genre <name>"
        return out

    def genre(self, name: str, *, chat_key: str) -> Outcome:
        try:
            known = self._anilist.genres()
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"genre check failed — {exc}")
        match = next((g for g in known
                      if g.lower() == name.strip().lower()), None)
        if match is None:
            return Outcome.fail(
                f"“{name}” isn't an AniList genre. /genres lists "
                f"them all ({len(known)} on file).")
        try:
            found = self._anilist.browse_media(
                kind="anime", sort="SCORE_DESC", genre=match,
                per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"genre browse failed — {exc}")
        return self._list_outcome(
            f"Top {match} anime", found, kind="anime",
            chat_key=chat_key, icon="🏷️")

    # ── browse lists ──────────────────────────────────────────────
    def _list_outcome(self, title: str, records: list[MediaRecord], *,
                      kind: str, chat_key: str, icon: str = "🎌",
                      extra: Callable[[MediaRecord], str] | None = None,
                      ) -> Outcome:
        if not records:
            return Outcome.fail(f"nothing on the record for {title}.")
        out = Outcome(ok=True, title=title, icon=icon)
        for i, rec in enumerate(records[:_LIST_CAP], start=1):
            bits = [_score_text(rec)]
            if rec.format:
                bits.append(rec.format.replace("_", " ").title())
            if rec.year:
                bits.append(str(rec.year))
            if extra:
                tail = extra(rec)
                if tail:
                    bits.append(tail)
            out.lines.append(f"{i}. {rec.title} — " + " · ".join(bits))
        self._cache.put(chat_key, kind, records[:_LIST_CAP])
        out.picks = [(f"open: {r.title[:24]}", f"/{kind} open {i}")
                     for i, r in enumerate(records[:_LIST_CAP], start=1)]
        out.note = "via AniList"
        return out

    def top(self, kind: str, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.browse_media(
                kind=kind, sort="SCORE_DESC", per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"top list failed — {exc}")
        label = "manga" if kind == "manga" else "anime"
        return self._list_outcome(f"Top {label} of all time", found,
                                  kind=kind, chat_key=chat_key, icon="🏆")

    def popular(self, kind: str, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.browse_media(
                kind=kind, sort="POPULARITY_DESC", per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"popular list failed — {exc}")
        label = "manga" if kind == "manga" else "anime"
        return self._list_outcome(f"Most popular {label}", found,
                                  kind=kind, chat_key=chat_key, icon="🔥")

    def trending(self, kind: str, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.browse_media(
                kind=kind, sort="TRENDING_DESC", per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"trending list failed — {exc}")
        label = "manga" if kind == "manga" else "anime"
        return self._list_outcome(f"Trending {label} now", found,
                                  kind=kind, chat_key=chat_key, icon="📈")

    def airing(self, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.browse_media(
                kind="anime", sort="POPULARITY_DESC",
                status="RELEASING", per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"airing list failed — {exc}")
        now = self._clock()

        def _next(rec: MediaRecord) -> str:
            if rec.next_airing_at:
                return (f"ep {rec.next_episode} in "
                        f"{_countdown(rec.next_airing_at - now)}")
            return ""

        return self._list_outcome("Airing now", found, kind="anime",
                                  chat_key=chat_key, icon="📺",
                                  extra=_next)

    def upcoming(self, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.browse_media(
                kind="anime", sort="POPULARITY_DESC",
                status="NOT_YET_RELEASED", per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"upcoming list failed — {exc}")
        return self._list_outcome("Upcoming anime", found, kind="anime",
                                  chat_key=chat_key, icon="🔜")

    def season(self, season_name: str, year: int | None, *,
               chat_key: str) -> Outcome:
        season_name = (season_name or "").upper()
        if season_name not in ("WINTER", "SPRING", "SUMMER", "FALL"):
            return Outcome.fail(
                "which season? winter, spring, summer or fall — "
                "e.g. /season fall 2025 (bare /season = this season).")
        year = year or datetime.fromtimestamp(
            self._clock(), tz=WAT).year
        try:
            found = self._anilist.browse_media(
                kind="anime", sort="POPULARITY_DESC", season=season_name,
                season_year=year, per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"season list failed — {exc}")
        return self._list_outcome(
            f"{season_name.title()} {year} anime", found, kind="anime",
            chat_key=chat_key, icon="🗓️")

    def current_season(self) -> tuple[str, int]:
        dt = datetime.fromtimestamp(self._clock(), tz=WAT)
        month, year = dt.month, dt.year
        if month == 12:
            return "WINTER", year + 1
        if month in (1, 2):
            return "WINTER", year
        if month in (3, 4, 5):
            return "SPRING", year
        if month in (6, 7, 8):
            return "SUMMER", year
        return "FALL", year

    def top_characters(self, *, chat_key: str) -> Outcome:
        try:
            found = self._anilist.top_characters(per_page=_LIST_CAP)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"top characters failed — {exc}")
        if not found:
            return Outcome.fail("no characters on the record.")
        out = Outcome(ok=True, title="Most-loved characters", icon="💖")
        for i, c in enumerate(found, start=1):
            from_text = (f" ({c.media_titles[0]})" if c.media_titles
                         else "")
            favs = f" — ♥ {c.favourites:,}" if c.favourites else ""
            out.lines.append(f"{i}. {c.name}{from_text}{favs}")
        self._cache.put(chat_key, "anichar", found)
        out.picks = [(f"open: {c.name[:24]}", f"/anichar open {i}")
                     for i, c in enumerate(found, start=1)]
        out.note = "ranked by AniList favourites"
        return out

    # ── weekly schedule ───────────────────────────────────────────
    def schedule(self) -> Outcome:
        now = int(self._clock())
        try:
            entries = self._anilist.airing_between(now, now + 7 * 86400)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"schedule failed — {exc}")
        if not entries:
            return Outcome.fail(
                "nothing on the airing schedule for the next 7 days "
                "— AniList returned an empty grid.")
        by_day: dict[str, list[AiringEntry]] = {}
        for entry in entries:
            day = datetime.fromtimestamp(entry.airing_at, tz=WAT) \
                .strftime("%A %d %b")
            by_day.setdefault(day, []).append(entry)
        out = Outcome(ok=True, title="Airing this week", icon="🗓️")
        shown = 0
        for day, day_entries in by_day.items():
            out.lines.append(f"— {day} —")
            for entry in day_entries[:12]:
                tod = datetime.fromtimestamp(entry.airing_at, tz=WAT) \
                    .strftime("%H:%M")
                out.lines.append(
                    f"{tod} · {entry.media.title} — ep {entry.episode}")
                shown += 1
            if len(day_entries) > 12:
                out.lines.append(f"… and {len(day_entries) - 12} more")
        out.note = (f"{len(entries)} episodes across the week · "
                    "times read in WAT (UTC+1)")
        _ = shown
        return out

    # ── random & quotes & pictures ────────────────────────────────
    def random_media(self, kind: str) -> Outcome:
        try:
            rec = self._anilist.random_media(kind=kind, rng=self._rng)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"random pick failed — {exc}")
        full, note = self._detail(rec)
        out = self._media_card(full, note=note)
        out.title = f"🎲 {full.title}"
        return out

    def random_character(self) -> Outcome:
        try:
            rec = self._anilist.random_character(rng=self._rng)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"random pick failed — {exc}")
        out = self._character_card(rec)
        out.title = f"🎲 {rec.name}"
        return out

    def quote(self, anime_title: str = "") -> Outcome:
        try:
            rec = self._quotes.random_quote(anime_title)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"quote failed — {exc}")
        out = Outcome(ok=True, title="Anime quote", icon="💬")
        out.lines = [f"“{rec.text}”"]
        who = rec.character_name or "unknown"
        from_text = f" · {rec.anime_title}" if rec.anime_title else ""
        out.note = f"— {who}{from_text} (via animechan)"
        return out

    def picture(self, pool: str) -> Outcome:
        try:
            url, source = self._images.fetch_url(pool, rng=self._rng)
        except SourceFailure as exc:
            return Outcome.fail(f"picture failed — {exc}")
        out = Outcome(ok=True, title="", icon="🖼️")
        out.image_url = url
        out.image_caption = f"via {source}"
        out.note = f"picture via {source}"
        return out

    # ── reverse identification (/whatanime) ───────────────────────
    def identify_url(self, image_url: str) -> Outcome:
        try:
            results = self._trace.search_url(image_url)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"identification failed — {exc}")
        return self._identify_outcome(results)

    def identify_file(self, path: str, mime: str = "image/jpeg") \
            -> Outcome:
        try:
            results = self._trace.search_file(path, mime)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"identification failed — {exc}")
        return self._identify_outcome(results)

    def _identify_outcome(self, results: list) -> Outcome:
        if not results:
            return Outcome.fail(
                "trace.moe found nothing for that frame.")
        best = max(results, key=lambda r: r.similarity)
        if best.similarity < SIMILARITY_FLOOR:
            return Outcome.fail(
                f"no confident match — the closest guess scored "
                f"{best.similarity * 100:.0f}% similarity, under the "
                f"{int(SIMILARITY_FLOOR * 100)}% floor I trust. "
                "A clearer frame (no borders, no subtitles baked in) "
                "identifies better.")
        title = best.title
        if not title and best.anilist_id:
            try:
                rec = self._anilist.media_detail(best.anilist_id)
                title = rec.title
            except (SourceFailure, ShapeError):
                title = ""
        if not title:
            return Outcome.fail(
                "trace.moe matched a frame but the record behind it "
                "has no readable title — no match I can name.")
        minutes, seconds = divmod(int(best.at_seconds), 60)
        out = Outcome(ok=True, title=title, icon="🔍")
        rows: list[tuple[str, Any]] = [
            ("Match", f"{best.similarity * 100:.1f}% similarity")]
        if best.episode not in (None, ""):
            rows.append(("Episode", best.episode))
        rows.append(("Timestamp", f"{minutes:02d}:{seconds:02d}"))
        out.stats = rows
        out.lines = ["that's the frame's home — /anime "
                     f"“{title}” opens the full record."]
        out.picks = [("open the record", f"/anime {title}")]
        out.note = "identified via trace.moe"
        return out

    # ── favorites (Devon's own list) ──────────────────────────────
    def favorites_list(self, user_key: str) -> Outcome:
        favs = self._favorites.list(user_key)
        out = Outcome(ok=True, title="Your anime favorites", icon="⭐")
        if not favs:
            out.lines = ["nothing saved yet — /favorite add <title> "
                         "saves one."]
            return out
        for i, fav in enumerate(favs, start=1):
            kind = "manga" if fav.kind == "manga" else "anime"
            out.lines.append(f"{i}. {fav.title} ({kind})")
        out.picks = [
            (f"open: {fav.title[:24]}", f"/favorite open {i}")
            for i, fav in enumerate(favs, start=1)]
        out.note = "your list — only you see it"
        return out

    def favorites_add(self, user_key: str, query: str) -> Outcome:
        kind = "anime"
        rec, attempts = self._resolve(kind, query)
        if rec is None:
            rec, attempts2 = self._resolve("manga", query)
            if rec is None:
                return self._not_found("anime", query,
                                       attempts + attempts2)
            kind = "manga"
        try:
            added = self._favorites.add(
                user_key, source=rec.source, source_id=rec.source_id,
                kind=kind, title=rec.title)
        except RuntimeError:
            return Outcome.fail(
                "favorites storage is unavailable in this session.")
        if not added:
            return Outcome(ok=True, title="Already saved", icon="⭐",
                           lines=[f"“{rec.title}” is already in your "
                                  "favorites."])
        return Outcome(ok=True, title="Saved ⭐", icon="⭐",
                       lines=[f"“{rec.title}” added to your favorites."],
                       image_url=rec.cover_url,
                       image_caption=rec.title)

    def favorites_remove(self, user_key: str, arg: str) -> Outcome:
        favs = self._favorites.list(user_key)
        if not favs:
            return Outcome.fail("your favorites list is empty.")
        target = None
        text = (arg or "").strip()
        if text.isdigit():
            n = int(text)
            if 1 <= n <= len(favs):
                target = favs[n - 1]
        if target is None and text:
            target = next((f for f in favs
                           if f.title.lower() == text.lower()), None)
        if target is None:
            return Outcome.fail(
                f"“{arg}” isn't in your favorites — /favorite lists "
                "them with numbers.")
        removed = self._favorites.remove(
            user_key, source=target.source,
            source_id=target.source_id)
        if removed is None:
            return Outcome.fail("couldn't remove that one — try again.")
        return Outcome(ok=True, title="Removed", icon="⭐",
                       lines=[f"“{removed.title}” removed from your "
                              "favorites."])

    def favorites_open(self, user_key: str, index: int) -> Outcome:
        favs = self._favorites.list(user_key)
        if not favs:
            return Outcome.fail("your favorites list is empty.")
        if index < 1 or index > len(favs):
            return Outcome.fail(
                f"pick a number 1–{len(favs)} from your favorites.")
        fav = favs[index - 1]
        if fav.source == "anilist":
            try:
                rec = self._anilist.media_detail(fav.source_id)
            except (SourceFailure, ShapeError) as exc:
                return Outcome.fail(f"couldn't reopen that record — {exc}")
            return self._media_card(rec)
        try:
            hits = self._jikan.search_media(fav.title, kind=fav.kind,
                                            limit=1)
        except (SourceFailure, ShapeError) as exc:
            return Outcome.fail(f"couldn't reopen that record — {exc}")
        if not hits:
            return Outcome.fail(
                f"“{fav.title}” no longer answers on MyAnimeList.")
        return self._media_card(hits[0])
