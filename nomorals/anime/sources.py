"""Source clients for the anime organ — seeds, never a closed list.

Each client validates the envelope it reads and raises
:class:`SourceFailure` with a named reason (timeout, HTTP status,
rate limit, shape changed) instead of returning partial fiction.
The service layer walks the seeds in order and keeps an attempts
trail, exactly the /play resolver discipline.
"""

from __future__ import annotations

import time
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
    "ImagePoolClient",
    "JikanClient",
    "QuoteClient",
    "SourceFailure",
    "TraceMoeClient",
    "TraceResult",
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


# ── Image pools (waifu.pics / nekos.best) ───────────────────────────

@dataclass
class _PoolSource:
    def __init__(self, name: str, url: str, shape: str) -> None:
        self.name = name
        self.url = url
        self.shape = shape  # "waifu.pics" | "nekos.best"


#: Pool chains per picture command: seeds in try order.
IMAGE_POOLS: dict[str, list[_PoolSource]] = {
    "waifu": [
        _PoolSource("waifu.pics", "https://api.waifu.pics/sfw/waifu",
                    "waifu.pics"),
        _PoolSource("nekos.best", "https://nekos.best/api/v2/waifu",
                    "nekos.best"),
    ],
    "neko": [
        _PoolSource("waifu.pics", "https://api.waifu.pics/sfw/neko",
                    "waifu.pics"),
        _PoolSource("nekos.best", "https://nekos.best/api/v2/neko",
                    "nekos.best"),
    ],
    "kitsune": [
        _PoolSource("nekos.best", "https://nekos.best/api/v2/kitsune",
                    "nekos.best"),
    ],
    "husbando": [
        _PoolSource("nekos.best", "https://nekos.best/api/v2/husbando",
                    "nekos.best"),
    ],
    "charpic": [
        _PoolSource("waifu.pics", "https://api.waifu.pics/sfw/waifu",
                    "waifu.pics"),
        _PoolSource("waifu.pics", "https://api.waifu.pics/sfw/shinobu",
                    "waifu.pics"),
        _PoolSource("nekos.best", "https://nekos.best/api/v2/neko",
                    "nekos.best"),
        _PoolSource("nekos.best", "https://nekos.best/api/v2/waifu",
                    "nekos.best"),
    ],
}


class ImagePoolClient:
    """Random anime pictures from the public pool APIs."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def fetch_url(self, pool: str, *, rng: Any) -> tuple[str, str]:
        """(image_url, source_name) — walks the pool's seed chain."""
        sources = list(IMAGE_POOLS.get(pool) or [])
        if not sources:
            raise SourceFailure("image pools", f"unknown pool {pool!r}")
        rng.shuffle(sources)
        attempts: list[str] = []
        for src in sources:
            try:
                result = self._t.request("GET", src.url,
                                         timeout=DEFAULT_TIMEOUT)
            except TransportError as exc:
                attempts.append(f"{src.name}: unreachable ({exc})")
                continue
            if not result.ok:
                attempts.append(f"{src.name}: HTTP {result.status}")
                continue
            url = self._extract(src, result.data)
            if url:
                return url, src.name
            attempts.append(f"{src.name}: shape changed (no image url)")
        raise SourceFailure("image pools", "; ".join(attempts))

    @staticmethod
    def _extract(src: _PoolSource, data: Any) -> str:
        if src.shape == "waifu.pics":
            if isinstance(data, dict) and isinstance(data.get("url"), str):
                return data["url"]
            return ""
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
