"""Tests for the /play pick-list flow (old-school WhatsApp-bot style).

Covers: the specificity heuristic, title/artist parsing, the pick
cache (TTL, single-use), candidate dedupe, pick-list formatting, and
the never-raises contract.
"""

import time

import pytest

from nomorals.media.picklist import (
    PICK_LIMIT,
    PickCache,
    format_duration,
    format_picklist,
    is_specific,
    parse_title_artist,
    search_query_for,
)
from nomorals.media.resolver import ResolvedAudio


@pytest.fixture(autouse=True)
def _clean_cache():
    PickCache.clear()
    yield
    PickCache.clear()


def _cand(title, artist="", duration=180.0, kind="youtube"):
    return ResolvedAudio(ok=True, path_or_url=f"https://x/{title}",
                         title=title, artist=artist, duration=duration,
                         kind=kind, source_name="test")


# ── specificity heuristic ─────────────────────────────────────────────


class TestIsSpecific:
    def test_single_word_is_vague(self):
        assert is_specific("lifestyle") is False

    def test_artist_plus_title_is_specific(self):
        assert is_specific("lifestyle ya man") is True

    def test_two_words_is_specific(self):
        assert is_specific("burna boy") is True

    def test_dash_separator_is_specific(self):
        assert is_specific("burna boy - last last") is True

    def test_by_separator_is_specific(self):
        assert is_specific("lifestyle by ayo maff") is True
        assert is_specific("Lifestyle BY Ayo Maff") is True

    def test_quoted_is_specific(self):
        assert is_specific('"lifestyle"') is True

    def test_empty_is_not_specific(self):
        assert is_specific("") is False
        assert is_specific("   ") is False
        assert is_specific(None) is False

    def test_stopwords_dont_count(self):
        # "the" alone carries no identifying weight
        assert is_specific("the") is False

    def test_never_raises(self):
        assert is_specific(None) is False
        assert isinstance(is_specific(123), bool)


# ── title/artist parsing ──────────────────────────────────────────────


class TestParseTitleArtist:
    def test_by_separator(self):
        title, artist = parse_title_artist("lifestyle (YA MAN) by ayo maff")
        assert title == "lifestyle (YA MAN)"
        assert artist == "ayo maff"

    def test_by_separator_case_insensitive(self):
        title, artist = parse_title_artist("Lifestyle BY Ayo Maff")
        assert title == "Lifestyle"
        assert artist == "Ayo Maff"

    def test_dash_separator_artist_first(self):
        title, artist = parse_title_artist("burna boy - last last")
        assert title == "last last"
        assert artist == "burna boy"

    def test_no_separator(self):
        title, artist = parse_title_artist("lifestyle ya man")
        assert title == "lifestyle ya man"
        assert artist == ""

    def test_by_without_artist_falls_through(self):
        # "by" with nothing after it isn't a separator
        title, artist = parse_title_artist("stand by")
        assert artist == "" or title == "stand by"

    def test_empty(self):
        assert parse_title_artist("") == ("", "")
        assert parse_title_artist(None) == ("", "")

    def test_never_raises(self):
        assert parse_title_artist(None) == ("", "")


class TestSearchQueryFor:
    def test_artist_first_ordering(self):
        assert search_query_for("lifestyle (YA MAN)", "ayo maff") == \
            "ayo maff lifestyle (YA MAN)"

    def test_title_only(self):
        assert search_query_for("lifestyle", "") == "lifestyle"

    def test_artist_only(self):
        assert search_query_for("", "ayo maff") == "ayo maff"

    def test_never_raises(self):
        assert isinstance(search_query_for(None, None), str)


# ── pick cache ────────────────────────────────────────────────────────


class TestPickCache:
    def test_store_and_get(self):
        cands = [_cand("A"), _cand("B")]
        token = PickCache.store(cands, "tg:123")
        assert token
        entry = PickCache.get(token)
        assert entry is not None
        assert [c.title for c in entry.candidates] == ["A", "B"]

    def test_get_for_chat(self):
        cands = [_cand("A")]
        token = PickCache.store(cands, "wa:456")
        found = PickCache.get_for_chat("wa:456")
        assert found is not None
        assert found[0] == token

    def test_missing_token_returns_none(self):
        assert PickCache.get("deadbeef") is None
        assert PickCache.get("") is None
        assert PickCache.get(None) is None

    def test_missing_chat_returns_none(self):
        assert PickCache.get_for_chat("nope:0") is None

    def test_expiry(self):
        token = PickCache.store([_cand("A")], "tg:1", ttl=0.05)
        assert PickCache.get(token) is not None
        time.sleep(0.08)
        assert PickCache.get(token) is None

    def test_consume_is_single_use(self):
        token = PickCache.store([_cand("A")], "tg:1")
        first = PickCache.consume(token)
        assert first is not None
        assert PickCache.get(token) is None
        assert PickCache.consume(token) is None

    def test_latest_token_wins_per_chat(self):
        t1 = PickCache.store([_cand("A")], "tg:9")
        t2 = PickCache.store([_cand("B")], "tg:9")
        assert t1 != t2
        found = PickCache.get_for_chat("tg:9")
        assert found[0] == t2

    def test_never_raises(self):
        assert PickCache.store(None, None) is not None or True
        assert PickCache.get(None) is None


# ── formatting ────────────────────────────────────────────────────────


class TestFormatDuration:
    def test_minutes_seconds(self):
        assert format_duration(204) == "3:24"

    def test_zero(self):
        assert format_duration(0) == "0:00"

    def test_never_raises(self):
        assert isinstance(format_duration(None), str)
        assert isinstance(format_duration("x"), str)


class TestFormatPicklist:
    def test_compact_lines(self):
        cands = [_cand("Lifestyle", "Ya Man", 204),
                 _cand("Lifestyle", "Other", 180)]
        msg = format_picklist(cands, "lifestyle", "a1b2c3d4")
        assert "1. Lifestyle — Ya Man (3:24)" in msg
        assert "2. Lifestyle — Other (3:00)" in msg
        assert "pick:a1b2c3d4" in msg
        # compact: no wall of text
        assert len(msg.splitlines()) <= len(cands) + 3

    def test_no_artist_no_duration(self):
        msg = format_picklist([_cand("X", "", 0)], "x", "a1b2c3d4")
        assert "1. X" in msg

    def test_never_raises(self):
        assert isinstance(format_picklist(None, None, None), str)
        assert isinstance(format_picklist([], "q", ""), str)


# ── resolver.search_candidates ────────────────────────────────────────


class TestSearchCandidates:
    def test_never_raises_without_adapters(self):
        from nomorals.media.resolver import SourceResolver
        r = SourceResolver(None)
        # no adapters wired, no yt-dlp in CI → empty list, no raise
        out = r.search_candidates("lifestyle")
        assert isinstance(out, list)

    def test_empty_query(self):
        from nomorals.media.resolver import SourceResolver
        r = SourceResolver(None)
        assert r.search_candidates("") == []
        assert r.search_candidates(None) == []

    def test_dedupe(self):
        from unittest.mock import MagicMock
        from nomorals.media.resolver import SourceResolver
        sc = MagicMock()
        sc.search_tracks.return_value = [
            {"permalink_url": "https://soundcloud.com/a/1",
             "title": "Lifestyle", "artist": "Ya Man",
             "duration_ms": 200000},
            {"permalink_url": "https://soundcloud.com/a/2",
             "title": "Lifestyle", "artist": "Ya Man",
             "duration_ms": 200000},  # dup
            {"permalink_url": "https://soundcloud.com/a/3",
             "title": "Lifestyle", "artist": "Other",
             "duration_ms": 180000},
        ]
        r = SourceResolver(None, soundcloud=sc)
        out = r.search_candidates("lifestyle")
        keys = [(c.title, c.artist) for c in out]
        assert ("Lifestyle", "Ya Man") in keys
        assert ("Lifestyle", "Other") in keys
        assert len(keys) == len(set(keys))  # no dupes

    def test_limit_respected(self):
        from unittest.mock import MagicMock
        from nomorals.media.resolver import SourceResolver
        sc = MagicMock()
        sc.search_tracks.return_value = [
            {"permalink_url": f"https://soundcloud.com/a/{i}",
             "title": f"Track {i}", "duration_ms": 1000}
            for i in range(30)
        ]
        r = SourceResolver(None, soundcloud=sc)
        out = r.search_candidates("x", limit=5)
        assert len(out) <= 5


# ── telegram button derivation ────────────────────────────────────────


class TestTelegramButtons:
    def test_picklist_buttons(self):
        from nomorals.social.chat.tgbot_buttons import buttons_for_text
        msg = ("🎵 “lifestyle” — pick one (reply with the number):\n"
               "1. Lifestyle — Ya Man (3:24)\n"
               "2. Lifestyle — Other (3:00)\n"
               "tap a number below 👇\n"
               "pick:a1b2c3d4")
        kb = buttons_for_text(msg)
        assert kb is not None
        flat = [b for row in kb for b in row]
        assert ("1", "play pick a1b2c3d4 1") in flat
        assert ("2", "play pick a1b2c3d4 2") in flat

    def test_no_marker_no_buttons(self):
        from nomorals.social.chat.tgbot_buttons import buttons_for_text
        assert buttons_for_text("just a normal message") is None

    def test_callback_budget(self):
        from nomorals.social.chat.tgbot_buttons import check_callback_data
        # "play pick <8hex> <n>" must fit the 47-byte budget
        check_callback_data("play pick a1b2c3d4 8")
