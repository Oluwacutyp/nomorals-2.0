"""Pass tail-4 pins: media/picklist failures degrade honestly, ledger intact.

Target: nomorals/media/picklist.py — 0 converts / 11 exclusions.
Every site below maps a failure to the function's documented miss /
unknown / safe-closed value, and the module proceeds honestly — it
never fabricates a track, never auto-downloads on a failure, never
claims a list it does not have. None of them records, because none of
them hides a user-visible outcome.
Ledger REUSED: nomorals/media/failures.py (same package — read-only
for pins; no record call, no import added to the target, which
contains no record_failure call at all).

Each test pins the degraded behavior AND the NOT-recording property
(ledger empty after the trigger). No except blocks at all in this file
— the stubs raise, the code under test catches.
"""

from unittest import mock

from nomorals.media import failures as mf
from nomorals.media import picklist as pl
from nomorals.media.picklist import PickCache, PickEntry


def _silent():
    assert mf.recent_failures() == []


def setup_function():
    PickCache.clear()


def teardown_function():
    PickCache.clear()


class BadStrip(str):
    """str whose strip() raises on the FIRST call, then works."""

    def __new__(cls, value):
        obj = super().__new__(cls, value)
        obj._calls = 0
        return obj

    def strip(self, *a, **k):
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("strip boom")
        return super().strip(*a, **k)


class BoomTitle:
    @property
    def title(self):
        raise RuntimeError("title boom")


def test_109_get_failure_is_none():
    token = PickCache.store(["a", "b"], "chat1")
    assert token
    with mock.patch.object(PickEntry, "expired",
                           side_effect=RuntimeError("boom")):
        assert PickCache.get(token) is None
    # The failed fetch popped nothing: the entry is still retrievable.
    assert PickCache.get(token) is not None
    _silent()


def test_124_get_for_chat_failure_is_none():
    token = PickCache.store(["a"], "chat2")
    assert token
    with mock.patch.object(PickCache, "get",
                           side_effect=RuntimeError("boom")):
        assert PickCache.get_for_chat("chat2") is None
    _silent()


def test_148_prune_failure_leaves_entries():
    PickCache.store(["a"], "chat3")
    before = dict(PickCache._by_token)
    with mock.patch("time.time", side_effect=RuntimeError("boom")):
        assert PickCache.prune() is None
    # Stale entries linger, but get() still enforces expiry on read,
    # so a failed prune can never serve an expired pick.
    assert PickCache._by_token == before
    _silent()


def test_180_is_specific_failure_is_false():
    # Failure fails CLOSED to the pick-list: downloading the wrong
    # song is worse than one extra tap (module docstring).
    assert pl.is_specific(123) is False
    assert pl.is_specific("lifestyle ya man") is True
    _silent()


def test_210_parse_title_artist_failure_keeps_query():
    q = BadStrip("burna boy - last last")
    assert pl.parse_title_artist(q) == ("burna boy - last last", "")
    _silent()


def test_235_match_tokens_failure_is_empty():
    with mock.patch("re.findall", side_effect=RuntimeError("boom")):
        assert pl._match_tokens("lifestyle ya man") == []
    _silent()


def test_262_match_confidence_failure_is_zero():
    with mock.patch.object(pl, "_match_tokens",
                           side_effect=RuntimeError("boom")):
        assert pl.match_confidence("x", "y", "x", "y") == 0.0
    _silent()


def test_300_is_confident_match_failure_is_false():
    # Failure fails CLOSED to the pick-list, never to auto-delivery.
    with mock.patch.object(pl, "_match_tokens",
                           side_effect=RuntimeError("boom")):
        assert pl.is_confident_match("x", "y", "x", "y") is False
    _silent()


def test_313_search_query_for_failure_keeps_title():
    t = BadStrip("last last")
    assert pl.search_query_for(t, "") == "last last"
    _silent()


def test_322_format_duration_failure_is_empty():
    assert pl.format_duration("abc") == ""
    assert pl.format_duration(204) == "3:24"
    _silent()


def test_385_format_picklist_failure_is_header_only():
    # No entries, no token line, no fabricated track: the fallback
    # claims no list it does not have and triggers no download.
    out = pl.format_picklist([BoomTitle()], "lifestyle", "deadbeef")
    assert out == "🎵 “lifestyle” — pick one by number."
    assert "pick:" not in out
    _silent()
