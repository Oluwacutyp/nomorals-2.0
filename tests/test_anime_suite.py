"""Anime organ (FEI gap wave B) — network-free, fixture-shaped tests.

Fixtures carry the REAL response shapes of AniList GraphQL, Jikan,
trace.moe, animechan, waifu.pics and nekos.best (the fields the code
reads, in the envelopes the APIs return).  A routing fake transport
answers by query/variables/path, so the service's seed order,
cross-checks and variable discipline are all exercised.  Runtime
tests drive the real PartnerRuntime with the transport injected
through context.extras, ws2-style.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.anime.favorites import FavoritesStore
from nomorals.anime.present import present_outcome
from nomorals.anime.service import AnimeService
from nomorals.anime.sources import (
    AniListClient,
    ImagePoolClient,
    JikanClient,
    QuoteClient,
    TraceMoeClient,
)
from nomorals.anime.transport import HttpResult, TransportError
from nomorals.storage.db import Database

NOW = datetime(2025, 11, 5, 12, 0, tzinfo=UTC).timestamp()
NEXT_AT = int(NOW) + 90000  # +25h → Thu 06 Nov, 14:00 WAT

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


# ── fixture payloads (real API shapes) ──────────────────────────────

def _media_node(**over: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": 154587, "idMal": 52991, "type": "ANIME", "format": "TV",
        "status": "FINISHED", "episodes": 28, "chapters": None,
        "duration": 24, "season": "FALL", "seasonYear": 2023,
        "averageScore": 92, "popularity": 310000, "favourites": 21000,
        "genres": ["Adventure", "Drama", "Fantasy"],
        "description": "The adventure is over but life goes on for an "
                       "immortal elf mage learning about humans.",
        "title": {"romaji": "Sousou no Frieren",
                  "english": "Frieren: Beyond Journey's End",
                  "native": "葬送のフリーレン"},
        "coverImage": {
            "extraLarge": "https://s4.anilist.co/file/anilistcdn/media/"
                          "anime/cover/large/frieren.jpg",
            "large": "https://s4.anilist.co/file/anilistcdn/media/"
                     "anime/cover/medium/frieren.jpg"},
        "startDate": {"year": 2023, "month": 9, "day": 29},
        "studios": {"nodes": [{"name": "Madhouse"}]},
        "externalLinks": [
            {"site": "Crunchyroll",
             "url": "https://www.crunchyroll.com/series/frieren",
             "type": "STREAMING"},
            {"site": "Official Site", "url": "https://frieren-anime.jp",
             "type": "INFO"}],
        "trailer": {"id": "ytfrieren1", "site": "youtube"},
        "nextAiringEpisode": None,
    }
    node.update(over)
    return node


def _airing_node() -> dict[str, Any]:
    return _media_node(
        id=21, idMal=21, status="RELEASING", episodes=None,
        averageScore=89,
        title={"romaji": "One Piece", "english": "One Piece",
               "native": "ワンピース"},
        genres=["Action", "Adventure"],
        nextAiringEpisode={"episode": 1145, "airingAt": NEXT_AT,
                           "timeUntilAiring": 90000})


def _manga_node() -> dict[str, Any]:
    return _media_node(
        id=30002, idMal=2, type="MANGA", format="MANGA",
        status="RELEASING", episodes=None, chapters=364, duration=None,
        season=None, seasonYear=None, averageScore=93,
        title={"romaji": "Berserk", "english": "Berserk",
               "native": "ベルセルク"},
        genres=["Action", "Drama", "Horror"],
        studios={"nodes": []},
        externalLinks=[], trailer=None)


def _page(items: list, key: str = "media", total: int | None = None,
          has_next: bool = False) -> dict[str, Any]:
    return {"data": {"Page": {
        "pageInfo": {"total": total if total is not None else len(items),
                     "currentPage": 1, "lastPage": 1,
                     "hasNextPage": has_next},
        key: items}}}


def _detail_node() -> dict[str, Any]:
    node = _media_node()
    node["characters"] = {"edges": [
        {"role": "MAIN", "node": {
            "id": 176754,
            "name": {"full": "Frieren", "native": "フリーレン"},
            "image": {"large": "https://s4.anilist.co/char/frieren.jpg"},
            "favourites": 15000}},
        {"role": "MAIN", "node": {
            "id": 176755, "name": {"full": "Fern", "native": "フェルン"},
            "image": {"large": "https://s4.anilist.co/char/fern.jpg"},
            "favourites": 6000}},
        {"role": "SUPPORTING", "node": {
            "id": 176756, "name": {"full": "Stark", "native": "シュタルク"},
            "image": {"large": "https://s4.anilist.co/char/stark.jpg"},
            "favourites": 4000}},
    ]}
    node["staff"] = {"edges": [
        {"role": "Director", "node": {"id": 106579,
                                      "name": {"full": "Keiichirou Saitou"}}},
        {"role": "Music", "node": {"id": 116001,
                                   "name": {"full": "Evan Call"}}},
    ]}
    node["recommendations"] = {"nodes": [
        {"mediaRecommendation": _media_node(
            id=457, idMal=457, averageScore=90,
            title={"romaji": "Mushishi", "english": "Mushishi",
                   "native": "蟲師"})},
        {"mediaRecommendation": _media_node(
            id=101922, idMal=41025, averageScore=88,
            title={"romaji": "Violet Evergarden",
                   "english": "Violet Evergarden",
                   "native": "ヴァイオレット・エヴァーガーデン"})},
    ]}
    node["reviews"] = {"nodes": [
        {"summary": "A quiet masterpiece about time and grief.",
         "score": 100},
        {"summary": "Slow, warm, and devastating.", "score": 95},
    ]}
    node["streamingEpisodes"] = [
        {"title": "Episode 1 - The Journey's End",
         "url": "https://www.crunchyroll.com/watch/frieren-ep1",
         "site": "Crunchyroll"},
        {"title": "Episode 2 - It Didn't Have to Be Magic",
         "url": "https://www.crunchyroll.com/watch/frieren-ep2",
         "site": "Crunchyroll"},
    ]
    return node


def _char_node(**over: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": 176754,
        "name": {"full": "Frieren", "native": "フリーレン"},
        "image": {"large": "https://s4.anilist.co/char/frieren.jpg"},
        "description": "An elf mage who outlived her hero party.",
        "favourites": 15000,
        "media": {"nodes": [
            {"title": {"romaji": "Sousou no Frieren",
                       "english": "Frieren: Beyond Journey's End",
                       "native": "葬送のフリーレン"}}]},
    }
    node.update(over)
    return node


def _jikan_entry(**over: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "mal_id": 52991, "title": "Sousou no Frieren",
        "title_english": "Frieren: Beyond Journey's End",
        "title_japanese": "葬送のフリーレン", "type": "TV",
        "status": "Finished Airing", "score": 9.31, "popularity": 90,
        "favorites": 12000, "episodes": 28,
        "genres": [{"name": "Adventure"}, {"name": "Drama"}],
        "synopsis": "During their decade-long quest, the hero party "
                    "defeated the Demon King.",
        "images": {"jpg": {
            "image_url": "https://cdn.myanimelist.net/frieren.jpg",
            "large_image_url": "https://cdn.myanimelist.net/frieren-l.jpg"}},
        "aired": {"prop": {"from": {"year": 2023}}},
        "studios": [{"name": "Madhouse"}],
        "trailer": {"youtube_id": "malfrieren1"},
    }
    entry.update(over)
    return entry


class FakeTransport:
    """Routes by URL + GraphQL shape + variables, like the real APIs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.trace_similarity = 0.97
        self.waifu_pics_down = False

    # -- helpers ------------------------------------------------------
    def _graphql(self, body: dict[str, Any]) -> HttpResult:
        query = str((body or {}).get("query") or "")
        var = dict((body or {}).get("variables") or {})
        if "GenreCollection" in query:
            return HttpResult(200, data={"data": {"GenreCollection": [
                "Action", "Adventure", "Comedy", "Drama", "Fantasy",
                "Horror", "Mecha", "Romance", "Sci-Fi", "Slice of Life"]}})
        if "airingSchedules(" in query:
            entries = [
                {"episode": 5, "airingAt": int(NOW) + 3600,
                 "media": _airing_node()},
                {"episode": 1145, "airingAt": NEXT_AT,
                 "media": _airing_node()},
                {"episode": 9, "airingAt": int(NOW) + 2 * 86400,
                 "media": _media_node(id=777, title={
                     "romaji": "Friday Show", "english": "Friday Show",
                     "native": ""})},
            ]
            return HttpResult(200, data=_page(entries,
                                              key="airingSchedules"))
        if "Media(id:" in query or '"id"' in str(var) and "Media" in query:
            media_id = var.get("id")
            if media_id == 21:
                node = _airing_node()
                node["characters"] = {"edges": []}
                node["staff"] = {"edges": []}
                node["recommendations"] = {"nodes": []}
                node["reviews"] = {"nodes": []}
                node["streamingEpisodes"] = []
                return HttpResult(200, data={"data": {"Media": node}})
            if media_id == 999001:
                node = _media_node(id=999001, title={
                    "romaji": "Frieren Second Hit", "english": "",
                    "native": ""})
                node["characters"] = {"edges": []}
                node["staff"] = {"edges": []}
                node["recommendations"] = {"nodes": []}
                node["reviews"] = {"nodes": []}
                node["streamingEpisodes"] = []
                return HttpResult(200, data={"data": {"Media": node}})
            if media_id == 30002:
                node = _manga_node()
                node["characters"] = {"edges": []}
                node["staff"] = {"edges": []}
                node["recommendations"] = {"nodes": []}
                node["reviews"] = {"nodes": []}
                node["streamingEpisodes"] = []
                return HttpResult(200, data={"data": {"Media": node}})
            return HttpResult(200, data={"data": {"Media": _detail_node()}})
        if "characters(search:" in query:
            chars = [_char_node()] if "frieren" in str(
                var.get("search") or "").lower() else []
            return HttpResult(200, data=_page(chars, key="characters"))
        if "characters(sort: [FAVOURITES_DESC])" in query:
            if "page" in var:  # the random pick
                return HttpResult(200, data=_page(
                    [_char_node(id=555, name={"full": "Random Char",
                                             "native": ""})],
                    key="characters", total=5000))
            chars = [_char_node(),
                     _char_node(id=88, name={"full": "Lelouch Lamperouge",
                                             "native": ""},
                                favourites=30000)]
            return HttpResult(200, data=_page(chars, key="characters"))
        if "characters { id }" in query:
            return HttpResult(200, data=_page([{"id": 1}],
                                              key="characters",
                                              total=5000))
        if "staff(search:" in query:
            staff = [{
                "id": 106580, "name": {"full": "Atsumi Tanezaki",
                                       "native": "種﨑敦美"},
                "image": {"large": "https://s4.anilist.co/staff/at.jpg"},
                "description": "Japanese voice actress.",
                "favourites": 3200,
                "primaryOccupations": ["Voice Actor"],
                "characters": {"nodes": [{"name": {"full": "Frieren"}},
                                         {"name": {"full": "Anya Forger"}}]},
                "media": {"nodes": [
                    {"title": {"romaji": "Sousou no Frieren",
                               "english": "Frieren", "native": ""}}]},
            }] if "tanezaki" in str(var.get("search") or "").lower() \
                else []
            return HttpResult(200, data=_page(staff, key="staff"))
        if "studios(search:" in query:
            studios = [{
                "id": 11, "name": "Madhouse", "isAnimationStudio": True,
                "favourites": 9000,
                "media": {"nodes": [_media_node(),
                                    _airing_node()]},
            }] if "madhouse" in str(var.get("search") or "").lower() \
                else []
            return HttpResult(200, data=_page(studios, key="studios"))
        if "media(search:" in query:
            search = str(var.get("search") or "").lower()
            if "frieren" in search:
                items = [_media_node(), _media_node(
                    id=999001, title={"romaji": "Frieren Second Hit",
                                      "english": "", "native": ""})]
                return HttpResult(200, data=_page(items))
            if "one piece" in search:
                return HttpResult(200, data=_page([_airing_node()]))
            if "berserk" in search:
                return HttpResult(200, data=_page([_manga_node()]))
            return HttpResult(200, data=_page([]))
        if "media(type:" in query:
            if "{ id }" in query:  # random count probe (ids only)
                return HttpResult(200, data=_page([{"id": 1}],
                                                  total=1000))
            if "perPage: 1" in query and "page" in var:
                # the random pick itself (browse passes page too,
                # but its perPage rides as a $variable, not a literal)
                return HttpResult(200, data=_page([_media_node(id=4242)],
                                                  total=1000))
            sort = (var.get("sort") or [""])[0]
            status = var.get("status") or ""
            items = [_media_node(), _airing_node()]
            if var.get("type") == "MANGA" or sort == "SCORE_DESC" \
                    and var.get("type") == "MANGA":
                items = [_manga_node()]
            if status == "NOT_YET_RELEASED":
                items = [_media_node(id=313131, status="NOT_YET_RELEASED",
                                     title={"romaji": "Future Show",
                                            "english": "Future Show",
                                            "native": ""})]
            return HttpResult(200, data=_page(items))
        return HttpResult(200, data=_page([]))

    # -- the transport seam --------------------------------------------
    def request(self, method: str, url: str, *, json_body: Any = None,
                params: dict[str, Any] | None = None,
                files: Any = None,
                timeout: float = 12.0) -> HttpResult:
        self.calls.append({"method": method, "url": url,
                           "json_body": json_body, "params": params,
                           "files": files})
        if method == "GET" and url.split("?")[0].endswith(
                (".jpg", ".png")):
            # a file download from any host — bytes, not an API body
            return HttpResult(200, data=None, content=JPEG)
        if "graphql.anilist.co" in url:
            return self._graphql(json_body or {})
        if "api.jikan.moe" in url:
            if "/episodes" in url:
                return HttpResult(200, data={"data": [
                    {"mal_id": 1, "title": "The Journey's End",
                     "aired": "2023-09-29T00:00:00+00:00"},
                    {"mal_id": 2, "title": "It Didn't Have to Be Magic",
                     "aired": "2023-09-29T00:00:00+00:00"}],
                    "pagination": {"last_visible_page": 1}})
            q = str((params or {}).get("q") or "").lower()
            if "frieren" in q or "sousou" in q or "jikanonly" in q:
                return HttpResult(200, data={"data": [_jikan_entry()],
                                             "pagination": {}})
            if "berserk" in q:
                return HttpResult(200, data={"data": [
                    _jikan_entry(mal_id=2, title="Berserk",
                                 title_english="Berserk", score=9.47)],
                    "pagination": {}})
            return HttpResult(200, data={"data": [], "pagination": {}})
        if "api.trace.moe" in url:
            return HttpResult(200, data={
                "frameCount": 100, "error": "",
                "result": [{
                    "anilist": {"id": 154587, "title": {
                        "romaji": "Sousou no Frieren",
                        "english": "Frieren: Beyond Journey's End",
                        "native": "葬送のフリーレン"}},
                    "filename": "frieren-ep3.mp4", "episode": 3,
                    "from": 754.5, "to": 760.0,
                    "similarity": self.trace_similarity,
                    "video": "https://media.trace.moe/video/x.mp4",
                    "image": "https://media.trace.moe/image/x.jpg"}]})
        if "animechan.io" in url:
            title = str((params or {}).get("title") or "")
            return HttpResult(200, data={"status": "success", "data": {
                "content": "People die when they are killed.",
                "anime": {"id": 1, "name": title or "Fate/stay night"},
                "character": {"id": 2, "name": "Shirou Emiya"}}})
        if "api.waifu.pics" in url:
            if self.waifu_pics_down:
                return HttpResult(500, data=None)
            return HttpResult(200, data={
                "url": "https://i.waifu.pics/abc123.jpg"})
        if "nekos.best" in url:
            return HttpResult(200, data={"results": [
                {"url": "https://nekos.best/xyz.png",
                 "artist_name": "someone"}]})
        raise TransportError(f"no fixture route for {url}")


def _service(transport: FakeTransport, db: Database | None = None,
             rng: random.Random | None = None) -> AnimeService:
    if db is None:
        db = Database(":memory:")
    return AnimeService(
        anilist=AniListClient(transport), jikan=JikanClient(transport),
        quotes=QuoteClient(transport), images=ImagePoolClient(transport),
        trace=TraceMoeClient(transport), favorites=FavoritesStore(db),
        rng=rng if rng is not None else random.Random(7),
        clock=lambda: NOW)


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = FakeTransport()
        self.svc = _service(self.t)

    def test_anime_lookup_exact_with_mal_crosscheck(self) -> None:
        out = self.svc.search_title("anime", "frieren", chat_key="c1")
        self.assertTrue(out.ok)
        self.assertEqual(out.title, "Frieren: Beyond Journey's End")
        stats = dict(out.stats)
        self.assertEqual(stats["Score"], "92/100 (AniList)")
        self.assertEqual(stats["MAL score"], "9.31/10")
        self.assertIn("cross-checked", out.note)
        self.assertTrue(out.image_url.endswith("frieren.jpg"))
        # a second hit rides as an openable pick, never an id
        self.assertTrue(any(cmd == "/anime open 2"
                            for _label, cmd in out.picks))
        self.assertNotIn("154587", " ".join(out.lines))

    def test_open_second_hit_from_cache(self) -> None:
        self.svc.search_title("anime", "frieren", chat_key="c1")
        out = self.svc.open_result("anime", 2, chat_key="c1")
        self.assertTrue(out.ok)
        self.assertEqual(out.title, "Frieren Second Hit")

    def test_lookup_miss_is_honest_and_names_sources(self) -> None:
        out = self.svc.search_title("anime", "zzzz-not-real",
                                    chat_key="c1")
        self.assertFalse(out.ok)
        self.assertIn("AniList", out.error)
        self.assertIn("Jikan", out.error)

    def test_jikan_fallback_when_anilist_empty(self) -> None:
        out = self.svc.search_title("anime", "jikanonly", chat_key="c1")
        self.assertTrue(out.ok)
        self.assertIn("MyAnimeList", out.note)
        self.assertEqual(out.title, "Frieren: Beyond Journey's End")

    def test_manga_lookup(self) -> None:
        out = self.svc.search_title("manga", "berserk", chat_key="c1")
        self.assertTrue(out.ok)
        self.assertEqual(dict(out.stats)["Chapters"], 364)

    def test_character_staff_studio(self) -> None:
        char = self.svc.search_character("frieren", chat_key="c1")
        self.assertTrue(char.ok)
        self.assertEqual(char.title, "Frieren")
        self.assertIn("Frieren: Beyond Journey's End",
                      " ".join(char.lines))
        staff = self.svc.search_staff("tanezaki", chat_key="c1")
        self.assertTrue(staff.ok)
        self.assertIn("Frieren", " ".join(staff.lines))
        studio = self.svc.search_studio("madhouse", chat_key="c1")
        self.assertTrue(studio.ok)
        self.assertEqual(studio.title, "Madhouse")
        self.assertIn("One Piece", " ".join(studio.lines))

    def test_cast_and_episodes(self) -> None:
        cast = self.svc.cast_of("frieren")
        self.assertTrue(cast.ok)
        self.assertIn("Fern", " ".join(cast.lines))
        eps = self.svc.episodes("frieren")
        self.assertTrue(eps.ok)
        body = " ".join(eps.lines)
        self.assertIn("The Journey's End", body)     # streaming list
        self.assertIn("episode guide", body)          # Jikan guide

    def test_trailer_streaming_review(self) -> None:
        trailer = self.svc.trailer("frieren")
        self.assertTrue(trailer.ok)
        self.assertIn("youtube.com/watch?v=ytfrieren1",
                      " ".join(trailer.lines))
        self.assertEqual(trailer.picks[0][1],
                         "/video https://www.youtube.com/watch?"
                         "v=ytfrieren1")
        streaming = self.svc.streaming("frieren")
        self.assertIn("Crunchyroll", " ".join(streaming.lines))
        review = self.svc.review("frieren")
        self.assertIn("quiet masterpiece", " ".join(review.lines))
        self.assertEqual(dict(review.stats)["MAL score"], "9.31/10")

    def test_nextep_countdown_fixed_clock(self) -> None:
        out = self.svc.next_episode("one piece")
        self.assertTrue(out.ok)
        line = " ".join(out.lines)
        self.assertIn("episode 1145", line)
        self.assertIn("1d 1h", line)
        self.assertIn("Thu 06 Nov, 14:00 WAT", line)

    def test_nextep_finished_show_says_so(self) -> None:
        out = self.svc.next_episode("frieren")
        self.assertTrue(out.ok)
        self.assertIn("finished", " ".join(out.lines))

    def test_genres_and_genre_variables(self) -> None:
        out = self.svc.genres()
        self.assertTrue(out.ok)
        self.assertIn("Mecha", out.lines[0])
        top = self.svc.genre("Mecha", chat_key="c1")
        self.assertTrue(top.ok)
        genre_calls = [c for c in self.t.calls
                       if (c["json_body"] or {}).get("variables", {})
                       .get("genre") == "Mecha"]
        self.assertTrue(genre_calls)
        bad = self.svc.genre("Mechha", chat_key="c1")
        self.assertFalse(bad.ok)
        self.assertIn("isn't an AniList genre", bad.error)

    def test_top_and_trending_carry_sort_variables(self) -> None:
        self.svc.top("anime", chat_key="c1")
        self.svc.trending("anime", chat_key="c1")
        sorts = [tuple((c["json_body"] or {}).get("variables", {})
                       .get("sort") or ())
                 for c in self.t.calls]
        self.assertIn(("SCORE_DESC",), sorts)
        self.assertIn(("TRENDING_DESC",), sorts)

    def test_season_carries_variables(self) -> None:
        out = self.svc.season("FALL", 2025, chat_key="c1")
        self.assertTrue(out.ok)
        calls = [(c["json_body"] or {}).get("variables", {})
                 for c in self.t.calls]
        self.assertTrue(any(v.get("season") == "FALL"
                            and v.get("seasonYear") == 2025
                            for v in calls))
        season_name, year = self.svc.current_season()
        self.assertEqual((season_name, year), ("FALL", 2025))

    def test_schedule_groups_by_wat_day(self) -> None:
        out = self.svc.schedule()
        self.assertTrue(out.ok)
        body = "\n".join(out.lines)
        self.assertIn("Wednesday 05 Nov", body)
        self.assertIn("Thursday 06 Nov", body)
        self.assertIn("Friday 07 Nov", body)
        self.assertIn("ep 1145", body)
        self.assertIn("WAT", out.note)

    def test_airing_and_upcoming(self) -> None:
        airing = self.svc.airing(chat_key="c1")
        self.assertTrue(airing.ok)
        self.assertIn("ep 1145 in 1d 1h", " ".join(airing.lines))
        upcoming = self.svc.upcoming(chat_key="c1")
        self.assertTrue(upcoming.ok)
        self.assertIn("Future Show", " ".join(upcoming.lines))

    def test_recommend_returns_real_records(self) -> None:
        out = self.svc.recommend("frieren", chat_key="c1")
        self.assertTrue(out.ok)
        body = " ".join(out.lines)
        self.assertIn("Mushishi", body)
        self.assertIn("Violet Evergarden", body)
        self.assertTrue(any(cmd == "/anime open 1"
                            for _label, cmd in out.picks))

    def test_random_is_seeded_and_real(self) -> None:
        expected_page = random.Random(7).randint(1, 1000)
        out = self.svc.random_media("anime")
        self.assertTrue(out.ok)
        pages = [(c["json_body"] or {}).get("variables", {}).get("page")
                 for c in self.t.calls]
        self.assertIn(expected_page, pages)
        char = self.svc.random_character()
        self.assertTrue(char.ok)

    def test_top_characters(self) -> None:
        out = self.svc.top_characters(chat_key="c1")
        self.assertTrue(out.ok)
        self.assertIn("Lelouch Lamperouge", " ".join(out.lines))

    def test_quote_attributed(self) -> None:
        out = self.svc.quote()
        self.assertTrue(out.ok)
        self.assertIn("People die when they are killed.", out.lines[0])
        self.assertIn("Shirou Emiya", out.note)
        named = self.svc.quote("naruto")
        self.assertIn("naruto", named.note)

    def test_picture_pool_walks_the_chain(self) -> None:
        out = self.svc.picture("waifu")
        self.assertTrue(out.ok)
        self.assertTrue(out.image_url)
        self.t.waifu_pics_down = True
        out2 = self.svc.picture("waifu")
        self.assertTrue(out2.ok)
        self.assertIn("nekos.best", out2.image_url)

    def test_whatanime_confident_match(self) -> None:
        out = self.svc.identify_url("https://example.com/frame.jpg")
        self.assertTrue(out.ok)
        self.assertEqual(out.title, "Frieren: Beyond Journey's End")
        stats = dict(out.stats)
        self.assertEqual(stats["Episode"], 3)
        self.assertEqual(stats["Timestamp"], "12:34")

    def test_whatanime_under_floor_is_no_match(self) -> None:
        self.t.trace_similarity = 0.62
        out = self.svc.identify_url("https://example.com/frame.jpg")
        self.assertFalse(out.ok)
        self.assertIn("no confident match", out.error)

    def test_presentation_renders_card_and_buttons(self) -> None:
        out = self.svc.search_title("anime", "frieren", chat_key="c1")
        reply = present_outcome(out)
        self.assertIn("Frieren", reply.text)
        flat = [cmd for row in reply.buttons for _label, cmd in row]
        # actionable() carries the bare command; the platform layer
        # re-adds the slash when the button fires.
        self.assertIn("anime open 2", flat)
        self.assertNotIn("154587", reply.text)


class FavoritesTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / "workspace" / "anime-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-anime-fav-", dir=str(scratch))
        self.db = Database(str(Path(self.tmp.name) / "fav.db"))
        self.t = FakeTransport()
        self.svc = _service(self.t, db=self.db)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_add_list_remove_and_isolation(self) -> None:
        alice, bob = "telegram:111", "telegram:222"
        added = self.svc.favorites_add(alice, "frieren")
        self.assertTrue(added.ok)
        again = self.svc.favorites_add(alice, "frieren")
        self.assertIn("already", " ".join(again.lines))
        mine = self.svc.favorites_list(alice)
        self.assertIn("Frieren", " ".join(mine.lines))
        theirs = self.svc.favorites_list(bob)
        self.assertNotIn("Frieren", " ".join(theirs.lines))
        removed = self.svc.favorites_remove(alice, "1")
        self.assertTrue(removed.ok)
        self.assertIn("nothing saved yet", " ".join(
            self.svc.favorites_list(alice).lines))
        bob_removed = self.svc.favorites_remove(bob, "frieren")
        self.assertFalse(bob_removed.ok)

    def test_add_unknown_title_saves_nothing(self) -> None:
        out = self.svc.favorites_add("telegram:111", "zzzz-not-real")
        self.assertFalse(out.ok)
        self.assertIn("nothing saved yet", " ".join(
            self.svc.favorites_list("telegram:111").lines))

    def test_open_favorite_reopens_record(self) -> None:
        self.svc.favorites_add("telegram:111", "frieren")
        out = self.svc.favorites_open("telegram:111", 1)
        self.assertTrue(out.ok)
        self.assertEqual(out.title, "Frieren: Beyond Journey's End")


if __name__ == "__main__":
    unittest.main()


# ── runtime level: real PartnerRuntime, injected transport ──────────

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner.runtime_anime import ANIME_KINDS  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social.chat.base import (  # noqa: E402
    ChatKind,
    ChatMessage,
    ChatRef,
    MediaRef,
)
from nomorals.social.chat.control import (  # noqa: E402
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    help_menu,
    help_text,
    list_catalog,
    parse_control,
)

OWNER_ID = "5478650254"


class FakeGateway:
    def __init__(self) -> None:
        self.files: list[dict[str, Any]] = []

    def send_file(self, platform: str, chat: Any, path: str,
                  **kwargs: Any) -> Any:
        self.files.append({"platform": platform, "chat": chat,
                           "path": path,
                           "caption": kwargs.get("caption", "")})
        # the real gateway answers a SendResult with .ok
        return SimpleNamespace(ok=True)


def _make_runtime(transport: FakeTransport):
    scratch = Path.home() / "workspace" / ".test-tmp-anime"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-anime-",
                                      dir=str(scratch))
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    context.extras["anime_transport"] = transport
    gateway = FakeGateway()
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, gateway, tmp


def _msg(chat_key: str, *, kind: ChatKind = ChatKind.DM,
         sender_id: str = OWNER_ID,
         media: list | None = None) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "1", kind=kind),
        incoming=True, text="", sender="owner", sender_id=sender_id,
        meta={"is_owner": True}, media=media or [])


class RegistryTests(unittest.TestCase):
    def test_every_kind_fully_registered(self) -> None:
        for kind in ANIME_KINDS:
            self.assertIn(kind, CONTROL_COMMANDS, kind)
            self.assertIn(kind, COMMAND_DETAILS, kind)
            self.assertIn(kind, LIST_ONELINERS, kind)
            self.assertIn(kind, PUBLIC_COMMANDS, kind)
            slots = sum(kind in kinds for _name, kinds in LIST_GROUPS)
            self.assertEqual(slots, 1, kind)
            parsed = parse_control(f"/{kind} x")
            self.assertEqual(parsed.kind, kind)

    def test_surfaces_carry_the_suite(self) -> None:
        self.assertIn("/anischedule", help_text())
        self.assertIn("/whatanime", help_text())
        self.assertIn("anischedule", list_catalog())
        menu = help_menu("whatsapp")
        self.assertIn("anime", menu.lower())
        self.assertIn("anischedule", menu)


def _text(reply: Any) -> str:
    """handle_control returns str or ActionableMessage — read both."""
    return str(getattr(reply, "text", reply) or "")


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = FakeTransport()
        self.rt, self.gw, self.tmp = _make_runtime(self.t)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dm_lookup_sends_card_and_cover(self) -> None:
        reply = _text(self.rt.handle_control(
            "/anime frieren", "telegram:1", message=_msg("telegram:1")))
        self.assertIn("Frieren: Beyond Journey's End", reply)
        self.assertIn("92/100", reply)
        self.assertTrue(self.gw.files, "cover image should be sent")
        sent = Path(self.gw.files[0]["path"])
        self.assertTrue(sent.exists())
        self.assertEqual(sent.read_bytes(), JPEG)

    def test_open_pick_via_buttons_route(self) -> None:
        self.rt.handle_control("/anime frieren", "telegram:1",
                               message=_msg("telegram:1"))
        reply = _text(self.rt.handle_control(
            "/anime open 2", "telegram:1", message=_msg("telegram:1")))
        self.assertIn("Frieren Second Hit", reply)

    def test_group_unclaimed_is_gated(self) -> None:
        reply = _text(self.rt.handle_control(
            "/anime frieren", "telegram:-100123",
            message=_msg("telegram:-100123", kind=ChatKind.GROUP)))
        self.assertIn("isn't switched on", reply)
        self.assertNotIn("Frieren: Beyond", reply)

    def test_favorite_flow_end_to_end_isolated(self) -> None:
        alice = _msg("telegram:1")
        bob = _msg("telegram:2", sender_id="777")
        r1 = _text(self.rt.handle_control("/favorite add frieren",
                                          "telegram:1", message=alice))
        self.assertIn("added to your favorites", r1)
        r2 = _text(self.rt.handle_control("/favorite", "telegram:1",
                                          message=alice))
        self.assertIn("Frieren", r2)
        r3 = _text(self.rt.handle_control("/favorite", "telegram:2",
                                          message=bob))
        self.assertNotIn("Frieren", r3)
        r4 = _text(self.rt.handle_control("/favorite remove 1",
                                          "telegram:1", message=alice))
        self.assertIn("removed", r4)

    def test_whatanime_from_attached_frame(self) -> None:
        frame = Path(self.tmp.name) / "frame.jpg"
        frame.write_bytes(JPEG)
        msg = _msg("telegram:1", media=[MediaRef(
            path=str(frame), kind="image", mime="image/jpeg",
            name="frame.jpg")])
        reply = _text(self.rt.handle_control("/whatanime", "telegram:1",
                                             message=msg))
        self.assertIn("Frieren: Beyond Journey's End", reply)
        self.assertIn("12:34", reply)

    def test_waifu_delivers_attachment(self) -> None:
        reply = _text(self.rt.handle_control(
            "/waifu", "telegram:1", message=_msg("telegram:1")))
        self.assertIn("sent it here", reply)
        self.assertTrue(self.gw.files)
        self.assertEqual(Path(self.gw.files[0]["path"]).read_bytes(),
                         JPEG)

    def test_nextep_via_runtime(self) -> None:
        # service clock is real time here; the fixture's next episode
        # is fixed in 2025 — the reply must still be a computed answer,
        # never an error or a fabrication.
        reply = _text(self.rt.handle_control(
            "/nextep one piece", "telegram:1",
            message=_msg("telegram:1")))
        self.assertIn("One Piece", reply)
