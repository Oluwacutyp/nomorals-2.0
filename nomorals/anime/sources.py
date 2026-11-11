"""Source clients for the anime organ — seeds, never a closed list.

Each client validates the envelope it reads and raises
:class:`SourceFailure` with a named reason (timeout, HTTP status,
rate limit, shape changed) instead of returning partial fiction.
The service layer walks the seeds in order and keeps an attempts
trail, exactly the /play resolver discipline.
"""

from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from .model import (
    AiringEntry,
    CharacterRecord,
    MediaRecord,
    PersonRecord,
    QuoteRecord,
    ShapeError,
    StudioRecord,
    character_from_anilist,
    media_from_anilist,
    media_from_jikan,
    person_from_anilist,
    studio_from_anilist,
)
from .transport import DEFAULT_TIMEOUT, Transport, TransportError

__all__ = [
    "AniListClient",
    "IMAGE_POOLS",
    "ImagePoolClient",
    "JikanClient",
    "PoolHit",
    "QuoteClient",
    "SourceFailure",
    "TraceMoeClient",
    "TraceResult",
    "direct_source_url",
    "pool_classes",
]

ANILIST_URL = "https://graphql.anilist.co"
JIKAN_URL = "https://api.jikan.moe/v4"
TRACEMOE_URL = "https://api.trace.moe/search"
ANIMECHAN_URL = "https://animechan.io/api/v1/quotes"


class SourceFailure(Exception):
    """One source failed, named: which source, and why."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


def _check_http(source: str, result: Any) -> None:
    if result.status == 429:
        retry = result.headers.get("retry-after", "")
        hint = f" (retry after {retry}s)" if retry else ""
        raise SourceFailure(source, f"rate limited{hint}")
    if not result.ok:
        raise SourceFailure(source, f"HTTP {result.status}")


# ── AniList ─────────────────────────────────────────────────────────

_MEDIA_FIELDS = """
    id idMal type format status episodes chapters duration
    season seasonYear averageScore popularity favourites genres
    description(asHtml: false)
    title { romaji english native }
    coverImage { extraLarge large }
    startDate { year month day }
    studios(isMain: true) { nodes { name } }
    externalLinks { site url type }
    trailer { id site }
    nextAiringEpisode { episode airingAt timeUntilAiring }
"""

_MEDIA_DETAIL_FIELDS = _MEDIA_FIELDS + """
    characters(sort: [ROLE, RELEVANCE], perPage: 12) {
      edges { role node { id name { full native }
        image { large } favourites } } }
    staff(perPage: 8) {
      edges { role node { id name { full } } } }
    recommendations(perPage: 6, sort: [ID]) {
      nodes { mediaRecommendation {
        id type format averageScore popularity
        title { romaji english native }
        coverImage { extraLarge large } } } }
    reviews(perPage: 3, sort: [RATING_DESC]) {
      nodes { summary score } }
    streamingEpisodes { title url site }
"""

_PAGE_INFO = "pageInfo { total currentPage lastPage hasNextPage }"


class AniListClient:
    """AniList GraphQL — the primary record seed (public, no key)."""

    name = "AniList"

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    # -- plumbing ---------------------------------------------------
    def graphql(self, query: str, variables: dict[str, Any]) -> dict:
        try:
            result = self._t.request(
                "POST", ANILIST_URL,
                json_body={"query": query, "variables": variables},
                timeout=DEFAULT_TIMEOUT)
        except TransportError as exc:
            raise SourceFailure(self.name, f"unreachable ({exc})") from exc
        _check_http(self.name, result)
        payload = result.data
        if not isinstance(payload, dict):
            raise SourceFailure(self.name, "response was not JSON")
        errors = payload.get("errors")
        if errors:
            first = errors[0] if isinstance(errors, list) and errors \
                else errors
            msg = (first or {}).get("message") if isinstance(first, dict) \
                else str(first)
            raise SourceFailure(self.name, f"query refused ({msg})")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ShapeError(self.name, "response.data missing")
        return data

    def _page(self, query: str, variables: dict[str, Any],
              key: str) -> tuple[list[Any], dict[str, Any]]:
        data = self.graphql(query, variables)
        page = data.get("Page")
        if not isinstance(page, dict):
            raise ShapeError(self.name, "Page missing from response")
        items = page.get(key)
        if not isinstance(items, list):
            raise ShapeError(self.name, f"Page.{key} missing")
        info = page.get("pageInfo")
        return items, info if isinstance(info, dict) else {}

    # -- media --------------------------------------------------------
    def search_media(self, query: str, *, kind: str,
                     per_page: int = 5) -> list[MediaRecord]:
        q = f"""query ($search: String, $type: MediaType) {{
          Page(page: 1, perPage: {per_page}) {{ {_PAGE_INFO}
            media(search: $search, type: $type,
                  sort: [SEARCH_MATCH]) {{ {_MEDIA_FIELDS} }} }} }}"""
        items, _ = self._page(q, {"search": query,
                                  "type": "MANGA" if kind == "manga"
                                  else "ANIME"}, "media")
        return [media_from_anilist(n) for n in items]

    def media_detail(self, source_id: int) -> MediaRecord:
        q = f"""query ($id: Int) {{
          Media(id: $id) {{ {_MEDIA_DETAIL_FIELDS} }} }}"""
        data = self.graphql(q, {"id": source_id})
        node = data.get("Media")
        if not isinstance(node, dict):
            raise ShapeError(self.name, "Media missing from response")
        return media_from_anilist(node, detail=True)

    def browse_media(self, *, kind: str, sort: str, page: int = 1,
                     per_page: int = 10, season: str = "",
                     season_year: int | None = None, genre: str = "",
                     status: str = "") -> list[MediaRecord]:
        variables: dict[str, Any] = {
            "type": "MANGA" if kind == "manga" else "ANIME",
            "sort": [sort], "page": page, "perPage": per_page}
        clauses = ["type: $type", "sort: $sort"]
        if season:
            variables["season"] = season
            clauses.append("season: $season")
        if season_year:
            variables["seasonYear"] = season_year
            clauses.append("seasonYear: $seasonYear")
        if genre:
            variables["genre"] = genre
            clauses.append("genre_in: [$genre]")
        if status:
            variables["status"] = status
            clauses.append("status: $status")
        q = f"""query ($type: MediaType, $sort: [MediaSort],
                        $page: Int, $perPage: Int,
                        $season: MediaSeason, $seasonYear: Int,
                        $genre: String, $status: MediaStatus) {{
          Page(page: $page, perPage: $perPage) {{ {_PAGE_INFO}
            media({", ".join(clauses)}) {{ {_MEDIA_FIELDS} }} }} }}"""
        items, _info = self._page(q, variables, "media")
        return [media_from_anilist(n) for n in items]

    def genres(self) -> list[str]:
        data = self.graphql("query { GenreCollection }", {})
        genres = data.get("GenreCollection")
        if not isinstance(genres, list) or \
                not all(isinstance(g, str) for g in genres):
            raise ShapeError(self.name, "GenreCollection shape changed")
        return list(genres)

    # -- people & studios ---------------------------------------------
    def search_characters(self, query: str, *,
                          per_page: int = 5) -> list[CharacterRecord]:
        q = f"""query ($search: String) {{
          Page(page: 1, perPage: {per_page}) {{ {_PAGE_INFO}
            characters(search: $search, sort: [SEARCH_MATCH]) {{
              id name {{ full native }} image {{ large }}
              description favourites
              media(perPage: 6, sort: [POPULARITY_DESC]) {{
                nodes {{ title {{ romaji english native }} }} }} }} }} }}"""
        items, _ = self._page(q, {"search": query}, "characters")
        return [character_from_anilist(n) for n in items]

    def top_characters(self, *, per_page: int = 10) -> list[CharacterRecord]:
        q = f"""query {{
          Page(page: 1, perPage: {per_page}) {{ {_PAGE_INFO}
            characters(sort: [FAVOURITES_DESC]) {{
              id name {{ full native }} image {{ large }}
              favourites
              media(perPage: 3, sort: [POPULARITY_DESC]) {{
                nodes {{ title {{ romaji english native }} }} }} }} }} }}"""
        items, _ = self._page(q, {}, "characters")
        return [character_from_anilist(n) for n in items]

    def search_staff(self, query: str, *,
                     per_page: int = 5) -> list[PersonRecord]:
        q = f"""query ($search: String) {{
          Page(page: 1, perPage: {per_page}) {{ {_PAGE_INFO}
            staff(search: $search, sort: [SEARCH_MATCH]) {{
              id name {{ full native }} image {{ large }}
              description favourites primaryOccupations
              characters(perPage: 6, sort: [FAVOURITES_DESC]) {{
                nodes {{ name {{ full }} }} }}
              media(perPage: 6, sort: [POPULARITY_DESC]) {{
                nodes {{ title {{ romaji english native }} }} }} }} }} }}"""
        items, _ = self._page(q, {"search": query}, "staff")
        return [person_from_anilist(n) for n in items]

    def search_studios(self, query: str, *,
                       per_page: int = 5) -> list[StudioRecord]:
        q = f"""query ($search: String) {{
          Page(page: 1, perPage: {per_page}) {{ {_PAGE_INFO}
            studios(search: $search, sort: [SEARCH_MATCH]) {{
              id name isAnimationStudio favourites
              media(perPage: 8, sort: [POPULARITY_DESC]) {{
                nodes {{ {_MEDIA_FIELDS} }} }} }} }} }}"""
        items, _ = self._page(q, {"search": query}, "studios")
        return [studio_from_anilist(n) for n in items]

    # -- schedule -----------------------------------------------------
    def airing_between(self, start_epoch: int, end_epoch: int, *,
                       max_pages: int = 4) -> list[AiringEntry]:
        q = f"""query ($from: Int, $to: Int, $page: Int) {{
          Page(page: $page, perPage: 50) {{ {_PAGE_INFO}
            airingSchedules(airingAt_greater: $from,
                            airingAt_lesser: $to, sort: [TIME]) {{
              episode airingAt
              media {{ {_MEDIA_FIELDS} }} }} }} }}"""
        entries: list[AiringEntry] = []
        page = 1
        while page <= max_pages:
            items, info = self._page(
                q, {"from": start_epoch, "to": end_epoch, "page": page},
                "airingSchedules")
            for node in items:
                if not isinstance(node, dict):
                    continue
                media_node = node.get("media")
                if not isinstance(media_node, dict):
                    continue
                try:
                    media = media_from_anilist(media_node)
                except ShapeError:
                    continue
                ep = node.get("episode")
                at = node.get("airingAt")
                if isinstance(ep, int) and isinstance(at, int):
                    entries.append(AiringEntry(media=media, episode=ep,
                                               airing_at=at))
            if not info.get("hasNextPage"):
                break
            page += 1
        return entries

    # -- random ---------------------------------------------------------
    def random_media(self, *, kind: str, rng: Any) -> MediaRecord:
        count_q = f"""query ($type: MediaType) {{
          Page(page: 1, perPage: 1) {{ {_PAGE_INFO}
            media(type: $type) {{ id }} }} }}"""
        _items, info = self._page(
            count_q, {"type": "MANGA" if kind == "manga" else "ANIME"},
            "media")
        total = info.get("total")
        if not isinstance(total, int) or total < 1:
            raise ShapeError(self.name, "pageInfo.total unusable")
        page = rng.randint(1, total)
        q = f"""query ($type: MediaType, $page: Int) {{
          Page(page: $page, perPage: 1) {{ {_PAGE_INFO}
            media(type: $type, sort: [POPULARITY_DESC]) {{
              {_MEDIA_FIELDS} }} }} }}"""
        items, _ = self._page(q, {"type": "MANGA" if kind == "manga"
                                  else "ANIME", "page": page}, "media")
        if not items:
            raise SourceFailure(self.name, "random page came back empty")
        return media_from_anilist(items[0])

    def random_character(self, *, rng: Any) -> CharacterRecord:
        count_q = f"""query {{
          Page(page: 1, perPage: 1) {{ {_PAGE_INFO}
            characters {{ id }} }} }}"""
        _items, info = self._page(count_q, {}, "characters")
        total = info.get("total")
        if not isinstance(total, int) or total < 1:
            raise ShapeError(self.name, "pageInfo.total unusable")
        page = rng.randint(1, total)
        q = f"""query ($page: Int) {{
          Page(page: $page, perPage: 1) {{ {_PAGE_INFO}
            characters(sort: [FAVOURITES_DESC]) {{
              id name {{ full native }} image {{ large }}
              description favourites
              media(perPage: 4, sort: [POPULARITY_DESC]) {{
                nodes {{ title {{ romaji english native }} }} }} }} }} }}"""
        items, _ = self._page(q, {"page": page}, "characters")
        if not items:
            raise SourceFailure(self.name, "random page came back empty")
        return character_from_anilist(items[0])


# ── Jikan (MyAnimeList) ─────────────────────────────────────────────

class JikanClient:
    """Jikan — the second seed: fallback search + episode detail."""

    name = "Jikan"

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            result = self._t.request("GET", f"{JIKAN_URL}{path}",
                                     params=params,
                                     timeout=DEFAULT_TIMEOUT)
        except TransportError as exc:
            raise SourceFailure(self.name, f"unreachable ({exc})") from exc
        _check_http(self.name, result)
        payload = result.data
        if not isinstance(payload, dict) or "data" not in payload:
            raise ShapeError(self.name, "response.data missing")
        return payload["data"]

    def search_media(self, query: str, *, kind: str,
                     limit: int = 5) -> list[MediaRecord]:
        path = "/manga" if kind == "manga" else "/anime"
        data = self.get(path, {"q": query, "limit": limit,
                               "order_by": "members", "sort": "desc"})
        if not isinstance(data, list):
            raise ShapeError(self.name, "search data is not a list")
        return [media_from_jikan(n, kind=kind) for n in data]

    def episodes(self, mal_id: int, *,
                 limit: int = 30) -> list[tuple[int, str, str]]:
        """(number, title, aired-date) for one anime on MAL."""
        data = self.get(f"/anime/{mal_id}/episodes",
                        {"page": 1})
        if not isinstance(data, list):
            raise ShapeError(self.name, "episodes data is not a list")
        out: list[tuple[int, str, str]] = []
        for node in data[:limit]:
            if not isinstance(node, dict):
                continue
            num = node.get("mal_id")
            title = str(node.get("title") or "")
            aired = str(node.get("aired") or "")
            if isinstance(num, int) and title:
                out.append((num, title, aired[:10]))
        return out


# ── Quotes (animechan) ──────────────────────────────────────────────

class QuoteClient:
    name = "animechan"

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def random_quote(self, anime_title: str = "") -> QuoteRecord:
        url = f"{ANIMECHAN_URL}/random"
        params: dict[str, Any] | None = None
        if anime_title:
            url = f"{ANIMECHAN_URL}/random/anime"
            params = {"title": anime_title}
        try:
            result = self._t.request("GET", url, params=params,
                                     timeout=DEFAULT_TIMEOUT)
        except TransportError as exc:
            raise SourceFailure(self.name, f"unreachable ({exc})") from exc
        _check_http(self.name, result)
        payload = result.data
        if not isinstance(payload, dict):
            raise ShapeError(self.name, "response was not JSON")
        data = payload.get("data")
        if not isinstance(data, dict) or not data.get("content"):
            if anime_title:
                raise SourceFailure(
                    self.name, f"no quotes filed for “{anime_title}”")
            raise ShapeError(self.name, "quote data missing")
        anime = data.get("anime") or {}
        character = data.get("character") or {}
        return QuoteRecord(
            text=str(data["content"]).strip(),
            anime_title=str((anime or {}).get("name") or ""),
            character_name=str((character or {}).get("name") or ""))


# ── The unified picture pool ────────────────────────────────────
#
# ONE pool organ for the whole picture-command family: the
# anime pool kinds (/waifu /neko /kitsune /husbando /charpic,
# via AnimeService) and — by delegation from
# social/chat/makers — the /couplepp chain.  Every endpoint's
# URL, response shape and safety check lives in this table and
# nowhere else; the makers providers of the same names are
# thin adapters over this client (the duplicate-endpoint split
# is why the 2026-11-07 diversification reached couplepp only).
#
# Diversity is a property of the table (owner, 2026-11-07
# "MORE SOURCES, NOT JUST ANIME", re-reported 2026-11-10):
# each kind's chain spans source CLASSES — anime art, real
# photography, real animals, general GIF search (keyed), and
# Devon's own studio.  Anime stays a kind's FIRST class (the
# top weight in its chain) but never its only one.  Sources
# are seeds, never a closed list: a new source is one more
# entry, a new class one more ``source_class`` value.

#: Source classes — the pool's diversity axis.
CLASS_ANIME = "anime"
CLASS_PHOTOGRAPHY = "photography"
CLASS_ANIMALS = "animals"
CLASS_GENERAL = "general"
CLASS_STUDIO = "studio"


@dataclass
class PoolHit:
    """One served picture: who served it, its source class,
    and where it lives — ``url`` for a network source,
    ``local_path`` when her own studio drew it (a render on
    disk, not a link)."""

    url: str = ""
    source: str = ""
    source_class: str = ""
    local_path: str = ""


class _PoolSource:
    def __init__(self, name: str, url: str, shape: str, *,
                 source_class: str = CLASS_ANIME, weight: int = 1,
                 animated: bool = False, key: str = "",
                 query: str = "") -> None:
        self.name = name
        self.url = url
        self.shape = shape
        self.source_class = source_class
        self.weight = max(1, int(weight))
        self.animated = animated
        self.key = key        # settings key that must exist ("" = keyless)
        self.query = query    # search text for the search-shaped sources


_WAIFU_PICS = "https://api.waifu.pics/sfw/"
_NEKOS_BEST = "https://nekos.best/api/v2/"
_NEKOSAPI = ("https://api.nekosapi.com/v4/images/random"
             "?rating=safe")


def _anime_seed(name: str, url: str, shape: str) -> _PoolSource:
    return _PoolSource(name, url, shape, weight=3)


def _picsum() -> _PoolSource:
    # picsum.photos — real photography, keyless: a seeded URL
    # answers a redirect to one real photograph.
    return _PoolSource(
        "picsum", "https://picsum.photos/seed/{seed}/1200/900",
        "picsum", source_class=CLASS_PHOTOGRAPHY, weight=1)


def _studio() -> _PoolSource:
    # Her own generation: resolved by the generator callable
    # injected into the client (the runtime wires the studio);
    # with no generator the entry never enters the walk.
    return _PoolSource("devon-studio", "", "studio",
                       source_class=CLASS_STUDIO, weight=1)


def _keyed(query: str) -> list[_PoolSource]:
    # Tenor / Giphy GIF search — a configured key is required
    # (neither has a keyless tier); unconfigured, these entries
    # simply never enter the walk, like expression's
    # default_providers.  The pool is rich without them.
    return [
        _PoolSource("tenor", "https://tenor.googleapis.com/v2/search",
                    "tenor", source_class=CLASS_GENERAL, weight=2,
                    animated=True, key="tenor", query=query),
        _PoolSource("giphy", "https://api.giphy.com/v1/gifs/search",
                    "giphy", source_class=CLASS_GENERAL, weight=2,
                    animated=True, key="giphy", query=query),
    ]


#: Pool chains per picture kind.  nekosapi enters only through
#: its safe-rated door: the request asks for ``rating=safe``
#: and the extractor re-checks each row's rating in code,
#: because the unfiltered pool serves suggestive/explicit
#: rows.  kitsune's second source is a real fox (randomfox —
#: a kitsune IS a fox; the 2026-11-07 "no second verified SFW
#: source" remainder closes with a verified animal source,
#: not another anime pool).  "couplepp" is the stills pool
#: the makers chain delegates to (the pair path itself
#: refuses animated bytes and re-draws — a diptych needs a
#: still).
IMAGE_POOLS: dict[str, list[_PoolSource]] = {
    "waifu": [
        _anime_seed("waifu.pics", _WAIFU_PICS + "waifu", "waifu.pics"),
        _anime_seed("nekos.best", _NEKOS_BEST + "waifu", "nekos.best"),
        _anime_seed("nekosapi", _NEKOSAPI, "nekosapi"),
        _picsum(),
        *_keyed("anime waifu"),
        _studio(),
    ],
    "neko": [
        _anime_seed("waifu.pics", _WAIFU_PICS + "neko", "waifu.pics"),
        _anime_seed("nekos.best", _NEKOS_BEST + "neko", "nekos.best"),
        _anime_seed("nekosapi", _NEKOSAPI, "nekosapi"),
        _PoolSource("cataas", "https://cataas.com/cat?json=true",
                    "cataas", source_class=CLASS_ANIMALS, weight=2),
        _PoolSource("cataas-gif",
                    "https://cataas.com/cat/gif?json=true",
                    "cataas", source_class=CLASS_ANIMALS, weight=2,
                    animated=True),
        *_keyed("cute cat"),
        _studio(),
    ],
    "kitsune": [
        _anime_seed("nekos.best", _NEKOS_BEST + "kitsune",
                    "nekos.best"),
        _PoolSource("randomfox", "https://randomfox.ca/floof/",
                    "randomfox", source_class=CLASS_ANIMALS,
                    weight=2),
        *_keyed("fox"),
        _studio(),
    ],
    "husbando": [
        _anime_seed("nekos.best", _NEKOS_BEST + "husbando",
                    "nekos.best"),
        _picsum(),
        *_keyed("anime husbando"),
        _studio(),
    ],
    "charpic": [
        _anime_seed("waifu.pics", _WAIFU_PICS + "waifu", "waifu.pics"),
        _anime_seed("waifu.pics", _WAIFU_PICS + "shinobu",
                    "waifu.pics"),
        _anime_seed("nekos.best", _NEKOS_BEST + "neko", "nekos.best"),
        _anime_seed("nekos.best", _NEKOS_BEST + "waifu", "nekos.best"),
        _anime_seed("nekosapi", _NEKOSAPI, "nekosapi"),
        _picsum(),
        *_keyed("anime character"),
        _studio(),
    ],
    "couplepp": [
        _anime_seed("waifu.pics", _WAIFU_PICS + "waifu", "waifu.pics"),
        _anime_seed("nekos.best", _NEKOS_BEST + "waifu", "nekos.best"),
        _anime_seed("nekosapi", _NEKOSAPI, "nekosapi"),
        _picsum(),
        # Registered so the pool's class span is honest; the
        # makers walk is URL-only (fetch_url), so this rung
        # sits out there — MakerService keeps its own
        # pair-shaped studio floor below the chain.
        _studio(),
    ],
    # The GIF kind (2026-11-11, expression repair F2): /gif
    # used to live entirely outside this organ, replaying a
    # library that started empty.  Its chain is the general
    # GIF-search class (keyed Tenor/Giphy), one verified
    # keyless animated source (cataas GIF cats — real-life
    # class, the same diversifier the neko chain carries),
    # whatever the source registry adds at runtime, and her
    # studio as the floor.
    "gif": [
        *_keyed("gif"),
        _PoolSource("cataas-gif",
                    "https://cataas.com/cat/gif?json=true",
                    "cataas", source_class=CLASS_ANIMALS, weight=2,
                    animated=True),
        _studio(),
    ],
}


def pool_classes(pool: str) -> set[str]:
    """The source classes a pool's chain spans, as tabled
    (keyed and studio rungs included — what the chain CAN
    serve when fully configured)."""
    return {src.source_class for src in IMAGE_POOLS.get(pool) or []}


def _pool_source_from_spec(spec: dict[str, Any]) -> _PoolSource | None:
    """One registered-source spec (see registered_sources.py)
    as a pool source, or None when its shape is not one this
    walk can read.  The returned source carries its registry
    name so the walk can journal the outcome back."""
    shape = str(spec.get("shape") or "")
    if shape not in ("waifu.pics", "nekos.best", "results-url",
                     "nekosapi", "cataas", "randomfox", "tenor",
                     "giphy", "url-list"):
        return None
    src = _PoolSource(
        str(spec.get("name") or ""), str(spec.get("url") or ""),
        shape,
        source_class=str(spec.get("source_class") or CLASS_GENERAL),
        weight=int(spec.get("weight") or 1),
        animated=bool(spec.get("animated", True)),
        key=str(spec.get("key") or ""),
        query=str(spec.get("query") or ""))
    src._registry_name = str(spec.get("name") or "")  # type: ignore[attr-defined]
    src._key_param = str(spec.get("key_param") or "")  # type: ignore[attr-defined]
    return src


def direct_source_url(name: str, *, rng: Any) -> str:
    """The composed URL of a named direct source (one whose
    seeded URL IS the picture — picsum) from the shared table.
    The makers picsum adapter delegates here so the template
    lives only in this table."""
    for entries in IMAGE_POOLS.values():
        for src in entries:
            if src.name == name and "{seed}" in src.url:
                return src.url.format(
                    seed=rng.randrange(1, 10 ** 9))
    return ""


class ImagePoolClient:
    """Random pictures from the unified pool table.

    The walk is a WEIGHTED shuffle: a kind's home class
    carries the top weight so an anime-flavoured kind stays
    anime-first across calls, while the other classes
    genuinely rotate in (repeated calls visibly vary — the
    owner's monotony complaint).  A dead or shapeless source
    is noted in the attempts trail and the walk falls
    through in order.  Her own studio is a weighted rung of
    the rotation AND the floor: its weight buys it a share of
    ordinary draws, and when every network source in the walk
    has failed, the studio entry is tried last before the
    pool gives up — generation is never only an error path,
    and never ahead of the network walk it floors.
    """

    def __init__(self, transport: Transport, *,
                 pools: dict[str, list[_PoolSource]] | None = None,
                 generator: Any = None,
                 tenor_key: str = "", giphy_key: str = "",
                 registry: Any = None,
                 extra_keys: dict[str, str] | None = None) -> None:
        self._t = transport
        self._pools = pools if pools is not None else IMAGE_POOLS
        # generator: callable(pool_kind) -> local file path of a
        # picture she rendered ("" / raise = her hand failed).
        self._generator = generator
        self._keys = {"tenor": str(tenor_key or ""),
                      "giphy": str(giphy_key or "")}
        for slot, value in (extra_keys or {}).items():
            self._keys[str(slot)] = str(value or "")
        # registry: her runtime source registry
        # (registered_sources.SourceRegistry) — its specs join
        # every walk beside the tabled seeds, and the walk
        # journals each registered source's outcome back into
        # it.  Read fresh per walk: a source registered at
        # runtime joins the NEXT walk, no restart.
        self._registry = registry

    def _registered_sources(self, pool: str) -> list[_PoolSource]:
        if self._registry is None:
            return []
        out: list[_PoolSource] = []
        for spec in self._registry.specs(pool):
            src = _pool_source_from_spec(spec)
            if src is not None:
                out.append(src)
        return out

    # -- introspection ------------------------------------------
    def available_classes(self, pool: str) -> set[str]:
        """The classes this client can actually serve for
        ``pool`` under its own configuration: keyed rungs
        count only with their key configured, the studio only
        with a generator wired."""
        out: set[str] = set()
        for src in list(self._pools.get(pool) or []) \
                + self._registered_sources(pool):
            if src.key and not self._keys.get(src.key):
                continue
            if src.shape == "studio" and self._generator is None:
                continue
            out.add(src.source_class)
        return out

    # -- the walk -------------------------------------------------
    def fetch_media(self, pool: str, *, rng: Any,
                    allow_local: bool = True) -> PoolHit:
        """One picture from ``pool`` — walks the chain and
        returns the first source that serves, with its class.
        Raises SourceFailure with the full attempts trail."""
        sources = list(self._pools.get(pool) or []) \
            + self._registered_sources(pool)
        if not sources:
            raise SourceFailure("image pools", f"unknown pool {pool!r}")
        network: list[_PoolSource] = []
        studio: list[_PoolSource] = []
        for src in sources:
            if src.key and not self._keys.get(src.key):
                continue  # an unconfigured keyed rung never enters
            if src.shape == "studio":
                studio.append(src)
            else:
                network.append(src)
        plan = self._weighted_order(network, rng)
        if studio and self._generator is not None and allow_local:
            total = sum(s.weight for s in network + studio)
            share = sum(s.weight for s in studio) / max(1, total)
            # Her hand takes this draw, or floors the walk.
            plan = studio + plan if rng.random() < share \
                else plan + studio
        attempts: list[str] = []
        for src in plan:
            before = len(attempts)
            hit = self._try_source(src, pool, rng, attempts)
            reg_name = str(getattr(src, "_registry_name", "") or "")
            if reg_name and self._registry is not None:
                # A registered source's outcome is JOURNALED
                # back into the registry: failures accumulate
                # toward stand-down/disable there, so a dead
                # registration is skipped by later walks
                # instead of retried forever.
                if hit is not None:
                    self._registry.note_result(reg_name, True)
                else:
                    self._registry.note_result(
                        reg_name, False,
                        error=attempts[-1] if len(attempts) > before
                        else "no answer")
            if hit is not None:
                return hit
        raise SourceFailure(
            "image pools", "; ".join(attempts) or "no sources")

    def fetch_url(self, pool: str, *, rng: Any) -> tuple[str, str]:
        """(image_url, source_name) — the URL-only walk (the
        studio rung sits out: a render is a file, not a URL)."""
        hit = self.fetch_media(pool, rng=rng, allow_local=False)
        return hit.url, hit.source

    @staticmethod
    def _weighted_order(sources: list[_PoolSource],
                        rng: Any) -> list[_PoolSource]:
        """Weighted sampling without replacement (Efraimidis–
        Spirakis): each source draws ``random ** (1/weight)``;
        the heaviest class tops the walk most often, every
        source can top it sometimes."""
        keyed = [(rng.random() ** (1.0 / src.weight), src)
                 for src in sources]
        keyed.sort(key=lambda pair: pair[0], reverse=True)
        return [src for _key, src in keyed]

    def _try_source(self, src: _PoolSource, pool: str, rng: Any,
                    attempts: list[str]) -> PoolHit | None:
        if src.shape == "studio":
            try:
                path = str(self._generator(pool) or "")
            except Exception as exc:  # noqa: BLE001 - her hand failed
                attempts.append(f"{src.name}: render failed ({exc})")
                return None
            if path:
                return PoolHit(source="her own studio",
                               source_class=src.source_class,
                               local_path=path)
            attempts.append(f"{src.name}: render came back empty")
            return None
        url = src.url
        params: dict[str, Any] | None = None
        if src.shape == "picsum":
            # The seeded URL IS the picture: compose it, then
            # prove it live (the request follows the redirect
            # to the real photograph) before handing it out.
            url = src.url.format(seed=rng.randrange(1, 10 ** 9))
            try:
                result = self._t.request("GET", url,
                                         timeout=DEFAULT_TIMEOUT)
            except TransportError as exc:
                attempts.append(f"{src.name}: unreachable ({exc})")
                return None
            if result.ok and result.content:
                return PoolHit(url=str(result.url or url),
                               source=src.name,
                               source_class=src.source_class)
            attempts.append(f"{src.name}: HTTP {result.status}")
            return None
        if src.shape == "tenor":
            params = {"q": src.query,
                      "key": self._keys.get(src.key or "tenor", ""),
                      "limit": 8, "media_filter": "gif",
                      "contentfilter": "medium"}
        elif src.shape == "giphy":
            params = {"api_key": self._keys.get(
                          src.key or "giphy", ""),
                      "q": src.query,
                      "limit": 8, "rating": "pg-13"}
        elif getattr(src, "_registry_name", ""):
            # A registered source: its URL template may carry
            # {query}; its key (when the slot is filled) rides
            # as a query param under its declared name; a
            # search-shaped source takes its text as ``q``.
            if "{query}" in url:
                url = url.replace(
                    "{query}", urllib.parse.quote(src.query or ""))
            params = {}
            if src.key and self._keys.get(src.key):
                params[str(getattr(src, "_key_param", "")
                           or "key")] = self._keys[src.key]
            if src.query and src.shape in ("results-url", "url-list"):
                params["q"] = src.query
            if not params:
                params = None
        try:
            result = self._t.request("GET", url, params=params,
                                     timeout=DEFAULT_TIMEOUT)
        except TransportError as exc:
            attempts.append(f"{src.name}: unreachable ({exc})")
            return None
        if not result.ok:
            attempts.append(f"{src.name}: HTTP {result.status}")
            return None
        found = self._extract(src, result.data, rng)
        if found:
            return PoolHit(url=found, source=src.name,
                           source_class=src.source_class)
        attempts.append(f"{src.name}: shape changed (no image url)")
        return None

    @staticmethod
    def _extract(src: _PoolSource, data: Any, rng: Any) -> str:
        if src.shape == "waifu.pics":
            if isinstance(data, dict) and isinstance(data.get("url"), str):
                return data["url"]
            return ""
        if src.shape == "nekosapi":
            # A bare list of rows; ONLY a row the pool itself
            # rates "safe" may pass — the URL's rating filter
            # is the source's claim, this check is ours.
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict) \
                            and str(row.get("rating") or "") == "safe" \
                            and isinstance(row.get("url"), str) \
                            and row["url"]:
                        return row["url"]
            return ""
        if src.shape == "cataas":
            # Two real answers: the single-cat {"url": …}
            # (relative on older answers, absolute on newer)
            # and the /api/cats row list a registered search
            # spec reads — GIF rows only, addressed by id.
            if isinstance(data, list):
                urls = [f"https://cataas.com/cat/{row['id']}"
                        for row in data
                        if isinstance(row, dict) and row.get("id")
                        and row.get("mimetype") == "image/gif"]
                return rng.choice(urls) if urls else ""
            if isinstance(data, dict):
                url = str(data.get("url") or "")
                if url.startswith("/"):
                    url = "https://cataas.com" + url
                return url
            return ""
        if src.shape == "url-list":
            # A bare JSON list: URL strings, or {"url": …}
            # rows — the simplest shape a source can serve.
            if isinstance(data, list):
                urls = [str(x) for x in data
                        if isinstance(x, str)
                        and x.startswith("http")]
                urls += [str(x["url"]) for x in data
                         if isinstance(x, dict)
                         and isinstance(x.get("url"), str)
                         and x["url"]]
                return rng.choice(urls) if urls else ""
            return ""
        if src.shape == "randomfox":
            if isinstance(data, dict) \
                    and isinstance(data.get("image"), str):
                return data["image"]
            return ""
        if src.shape == "tenor":
            rows = data.get("results") if isinstance(data, dict) \
                else None
            urls = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                formats = row.get("media_formats") or {}
                pick = formats.get("gif") or formats.get("tinygif") \
                    or {}
                if pick.get("url"):
                    urls.append(str(pick["url"]))
            return rng.choice(urls) if urls else ""
        if src.shape == "giphy":
            rows = data.get("data") if isinstance(data, dict) \
                else None
            urls = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                images = row.get("images") or {}
                pick = images.get("fixed_height") or {}
                if pick.get("url"):
                    urls.append(str(pick["url"]))
            return rng.choice(urls) if urls else ""
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list) and results and \
                    isinstance(results[0], dict) and \
                    isinstance(results[0].get("url"), str):
                return results[0]["url"]
        return ""


# ── trace.moe (reverse identification) ──────────────────────────────

class TraceResult:
    def __init__(self, *, anilist_id: int | None, title: str,
                 episode: Any, at_seconds: float, similarity: float,
                 filename: str = "") -> None:
        self.anilist_id = anilist_id
        self.title = title
        self.episode = episode
        self.at_seconds = at_seconds
        self.similarity = similarity
        self.filename = filename


class TraceMoeClient:
    """trace.moe — which anime is this frame from?"""

    name = "trace.moe"

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def search_url(self, image_url: str) -> list[TraceResult]:
        try:
            result = self._t.request(
                "GET", TRACEMOE_URL,
                params={"anilistInfo": "", "url": image_url},
                timeout=DEFAULT_TIMEOUT + 8.0)
        except TransportError as exc:
            raise SourceFailure(self.name, f"unreachable ({exc})") from exc
        return self._parse(result)

    def search_file(self, path: str, mime: str = "image/jpeg") \
            -> list[TraceResult]:
        try:
            result = self._t.request(
                "POST", f"{TRACEMOE_URL}?anilistInfo",
                files=[("image", path, mime or "image/jpeg")],
                timeout=DEFAULT_TIMEOUT + 8.0)
        except TransportError as exc:
            raise SourceFailure(self.name, f"unreachable ({exc})") from exc
        return self._parse(result)

    def _parse(self, result: Any) -> list[TraceResult]:
        _check_http(self.name, result)
        payload = result.data
        if not isinstance(payload, dict):
            raise ShapeError(self.name, "response was not JSON")
        if payload.get("error"):
            raise SourceFailure(self.name, str(payload["error"]))
        rows = payload.get("result")
        if not isinstance(rows, list):
            raise ShapeError(self.name, "result list missing")
        out: list[TraceResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            info = row.get("anilist")
            anilist_id: int | None = None
            title = ""
            if isinstance(info, dict):
                raw_id = info.get("id")
                anilist_id = raw_id if isinstance(raw_id, int) else None
                t = info.get("title") or {}
                if isinstance(t, dict):
                    title = str(t.get("english") or t.get("romaji")
                                or t.get("native") or "")
            elif isinstance(info, int):
                anilist_id = info
            sim = row.get("similarity")
            at = row.get("from")
            out.append(TraceResult(
                anilist_id=anilist_id, title=title,
                episode=row.get("episode"),
                at_seconds=float(at) if isinstance(at, (int, float))
                else 0.0,
                similarity=float(sim) if isinstance(sim, (int, float))
                else 0.0,
                filename=str(row.get("filename") or "")))
        return out


def now_epoch() -> int:
    return int(time.time())
