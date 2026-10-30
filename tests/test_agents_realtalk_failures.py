"""Tail-58 pins for nomorals/agents/realtalk.py —
0 converts / 7 exclusions (160, 221, 242, 268, 275, 324, 336).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

realtalk.py calls no failures-ledger record function anywhere
(grep-proven; its sibling handlers log via _log instead), so
NOT-recording is pinned against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-58 SURVEY.md):
- 160: _talk_history served_items failure -> []; the genuine
  no-history value; history feeds mix PREFERENCES only
  ("a preference, never a gate"), nothing is rendered from it.
- 221: _trending_item talked-refs read failure -> empty refs;
  dedupe weakens for one slot (a taught story may repeat) but
  the talk stays grounded with true provenance — no surface
  is told the story is new.
- 242: publisher_of failure -> the row's own source_name or
  "news"; the feed's own publisher field, literally true,
  rendered next to the story URL.
- 268: _wheel_item malformed history row -> ordering ignored;
  only the least-recently-talked preference is lost (the
  underlying _talk_history already guards the read itself).
- 275: _wheel_item theme_in_cooldown failure -> domain kept;
  fail-open on a mix-discipline preference, the same
  direction the real ledger's cooldown itself fails.
- 324: _library_item candidates() failure -> None; the
  genuine no-material value — the pulse simply carries no
  talk that slot, and the next slot retries.
- 336: _library_item theme_in_cooldown failure -> candidate
  kept; mirror of 275 for the Library pool.
"""

import random
from unittest import mock

import nomorals.agents.news as news_mod
import nomorals.agents.realtalk as rt
from nomorals.agents.failures import recent_failures


def _snap():
    return list(recent_failures(200))


_LONG = "x " * 300
_ROW = {"url": "https://example.com/story", "title": "Story",
        "source_name": "FeedName", "interest": 5, "published": 10}
_CAND = {"theme": "discipline", "weight": 1.0,
         "source": {"type": "wisdom", "ref": "r", "label": "L",
                    "passage": "p"}}


class _Ledger:
    """served_items double with a call log and optional failure."""

    def __init__(self, fail=False, rows=None):
        self.fail = fail
        self.rows = rows or []
        self.calls = []

    def served_items(self, since=None, surface_prefix=""):
        self.calls.append((since, surface_prefix))
        if self.fail:
            raise RuntimeError("served boom")
        return list(self.rows)


class _FakeNewsAgent:
    rows = []
    undelivered_calls = 0

    def __init__(self, ctx):
        pass

    def undelivered(self, limit=30):
        _FakeNewsAgent.undelivered_calls += 1
        return list(_FakeNewsAgent.rows)


def _trending(ledger, publisher):
    _FakeNewsAgent.rows = [dict(_ROW)]
    _FakeNewsAgent.undelivered_calls = 0
    with mock.patch.object(news_mod, "NewsAgent", _FakeNewsAgent), \
            mock.patch.object(news_mod, "publisher_of", publisher), \
            mock.patch.object(rt, "_extract", lambda url: _LONG):
        return rt._trending_item(object(), ledger, 1000.0)


class _WheelLedger:
    def __init__(self, rows=None, cooldown=None):
        self.rows = rows or []
        self._cooldown = cooldown or (lambda d, now=None: False)
        self.cooldown_calls = []

    def served_items(self, since=None, surface_prefix=""):
        return list(self.rows)

    def theme_in_cooldown(self, theme, now=None):
        self.cooldown_calls.append(theme)
        return self._cooldown(theme, now=now)


def _wheel(ledger):
    hunt_calls = []

    def fake_hunt(ctx, query, limit=5):
        hunt_calls.append(query)
        return []

    with mock.patch.object(rt, "_hunt", fake_hunt):
        out = rt._wheel_item(object(), ledger, 1000.0)
    return out, hunt_calls


class _LibLedger:
    def __init__(self, pool=None, fail=False, cooldown=None):
        self.pool = pool or []
        self.fail = fail
        self._cooldown = cooldown or (lambda t, now=None: False)
        self.cand_calls = 0
        self.cd_calls = []

    def candidates(self, now=None):
        self.cand_calls += 1
        if self.fail:
            raise RuntimeError("cand boom")
        return list(self.pool)

    def theme_in_cooldown(self, theme, now=None):
        self.cd_calls.append(theme)
        return self._cooldown(theme, now=now)


# ── 160: _talk_history served_items failure -> [] ──────────────

def test_excluded_160_talk_history_failure_returns_empty():
    before = _snap()
    led = _Ledger(fail=True)
    assert rt._talk_history(led, 1000.0) == []
    assert len(led.calls) == 1 and led.calls[0][1] == "pulse-"
    led2 = _Ledger(rows=[{"kind": "realtalk", "id": 1},
                         {"kind": "teaching", "id": 2}])
    assert [r["id"] for r in rt._talk_history(led2, 1000.0)] == [1]
    assert _snap() == before


# ── 221: talked-refs failure -> dedupe off, story serves ───────

def test_excluded_221_talked_refs_failure_still_serves():
    before = _snap()
    led = _Ledger(fail=True)
    out = _trending(led, lambda row: "CanonPub")
    assert out is not None and out[1] == "trending"
    assert len(led.calls) == 1  # served_items was attempted
    assert _FakeNewsAgent.undelivered_calls == 1
    led_ok = _Ledger(rows=[{
        "kind": "realtalk",
        "source": {"type": "news", "ref": _ROW["url"]}, "served": []}])
    # control: with the ref read working, the same story dedupes
    assert _trending(led_ok, lambda row: "CanonPub") is None
    assert _snap() == before


# ── 242: publisher_of failure -> source_name ───────────────────

def test_excluded_242_publisher_failure_uses_source_name():
    before = _snap()
    pub_calls = []

    def bad_pub(row):
        pub_calls.append(row)
        raise RuntimeError("pub boom")

    out = _trending(_Ledger(), bad_pub)
    assert out is not None and out[0]["source"]["label"] == "FeedName"
    assert len(pub_calls) == 1  # publisher_of was attempted
    out = _trending(_Ledger(), lambda row: "CanonPub")
    assert out is not None and out[0]["source"]["label"] == "CanonPub"
    assert _snap() == before


# ── 268: malformed history row -> ordering ignored ─────────────

def test_excluded_268_malformed_history_ignored():
    before = _snap()
    mal = [{"kind": "realtalk", "theme": "science",
            "served": [{"at": "not-a-float"}]}]
    out, hunts = _wheel(_WheelLedger(rows=mal))
    assert out is None
    assert hunts and "science" in hunts[0]  # first DOMAINS entry
    rows_all = [{"kind": "realtalk", "theme": d,
                 "served": [{"at": 900.0}]}
                for d in rt.DOMAINS if d != "craft"]
    out, hunts = _wheel(_WheelLedger(rows=rows_all))
    # control: valid history is used — the untalked domain leads
    assert hunts and "craft" in hunts[0]
    assert _snap() == before


# ── 275: cooldown failure -> domain kept (fail-open) ───────────

def test_excluded_275_cooldown_failure_keeps_domains():
    before = _snap()

    def raising_cd(theme, now=None):
        raise RuntimeError("cd boom")

    wl = _WheelLedger(cooldown=raising_cd)
    out, hunts = _wheel(wl)
    assert len(wl.cooldown_calls) == len(rt.DOMAINS)  # all attempted
    assert hunts and "science" in hunts[0]  # fail-open: first domain
    wl2 = _WheelLedger(cooldown=lambda d, now=None: d != "craft")
    out, hunts = _wheel(wl2)
    assert hunts and "craft" in hunts[0]  # control: veto works
    assert _snap() == before


# ── 324: candidates failure -> None ────────────────────────────

def test_excluded_324_candidates_failure_returns_none():
    before = _snap()
    led = _LibLedger(fail=True)
    assert rt._library_item(
        object(), led, 1000.0, random.Random(0)) is None
    assert led.cand_calls == 1  # candidates() was attempted
    out = rt._library_item(
        object(), _LibLedger(pool=[dict(_CAND)]), 1000.0,
        random.Random(0))
    assert out is not None and out[1] == "library"  # control
    assert _snap() == before


# ── 336: candidate cooldown failure -> candidate kept ──────────

def test_excluded_336_candidate_cooldown_failure_keeps_candidate():
    before = _snap()

    def raising_cd(theme, now=None):
        raise RuntimeError("cd boom")

    led = _LibLedger(pool=[dict(_CAND)], cooldown=raising_cd)
    out = rt._library_item(object(), led, 1000.0, random.Random(0))
    assert out is not None and out[1] == "library"  # fail-open
    assert led.cd_calls == ["discipline"]  # check was attempted
    led2 = _LibLedger(pool=[dict(_CAND)],
                      cooldown=lambda t, now=None: True)
    assert rt._library_item(
        object(), led2, 1000.0, random.Random(0)) is None  # control
    assert _snap() == before
