"""Tail-42 pins for nomorals/agents/teachings.py — 0 converts /
7 exclusions / 1 site flagged (552).

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the except path was
taken (trigger == fallback AND control != fallback), asserting via
the double's own call log that the failing operation itself was
attempted. The file contains no exception handlers, so it adds zero
census sites.

teachings.py calls no failures-ledger record function anywhere
(grep-proven: no record_failure import or call), so NOT-recording is
pinned with the tail-8 snapshot method: agents recent_failures()
before/after each trigger must be unchanged.

Site 552's pin records OBSERVED behavior only, not endorsed: it is
FLAGGED for parent adjudication (see the tail-42 SURVEY.md) — on an
items-table failure, served_items returns [] (the documented
fail-soft value), which also zeroes the stats counts and switches
both cooldown views off, indistinguishable from a genuinely unserved
history. Do not read this pin as blessing that indistinguishability.

Verdict notes (full grounds in SURVEY.md):
- 216/243: producer per-theme failure -> continue; documented
  "a dead source yields fewer candidates" contract; stored counts
  stay literally true.
- 478: candidate-cache query failure -> []; the pool is a <=24h
  cache, re-stocked next call; no guarantee reads this table.
- 588/592/596: stock()'s producer-call guards -> count 0 (literally
  true); near-unreachable, producers are fail-soft end-to-end.
- 769: corrupt stored served_json -> merge skipped, only the new
  serve persisted; no shipped writer can produce corrupt json
  (defensive catch-all, tail-30/33 family).
"""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "/home/hatch/workspace/nm-tail42-tree")

from nomorals.agents import failures as ag_fail
from nomorals.agents import teachings as T


def _snap():
    return list(ag_fail.recent_failures())


def _ctx(db=None):
    return SimpleNamespace(db=db)


LONG = "a passage long enough to clear the forty character floor easily"


class FakeDB:
    """context.db double with a call log and programmable handlers."""

    def __init__(self, query=None, query_one=None):
        self.calls = []
        self._query = query
        self._query_one = query_one

    def query(self, sql, params=()):
        self.calls.append(("query", sql))
        if self._query:
            return self._query(sql)
        return []

    def query_one(self, sql, params=()):
        self.calls.append(("query_one", sql))
        if self._query_one:
            return self._query_one(sql)
        return None

    def execute(self, sql, params=()):
        self.calls.append(("execute", sql, params))
        return None

    def sqls(self, needle):
        return [c for c in self.calls if needle in str(c[1])]


class FakeOrgan:
    def __init__(self, c):
        pass

    def recent_digests(self, limit=6):
        return []


class FakeKeeperEmpty:
    def __init__(self, c):
        pass

    def ask(self, theme, top=3):
        return SimpleNamespace(passages=[])


class FakeLibraryEmpty:
    def __init__(self, c):
        pass

    def search(self, theme, top=3):
        return []


def _keeper(fail_theme):
    class FakeKeeper:
        calls = []

        def __init__(self, c):
            pass

        def ask(self, theme, top=3):
            FakeKeeper.calls.append(theme)
            if theme == fail_theme:
                raise RuntimeError("keeper down for theme")
            p = SimpleNamespace(snippet=LONG, work="Work", section="",
                                slug="slug-" + theme, canon_status="")
            return SimpleNamespace(passages=[p])

    return FakeKeeper


def _library(fail_theme):
    class FakeLib:
        calls = []

        def __init__(self, c):
            pass

        def search(self, theme, top=3):
            FakeLib.calls.append(theme)
            if theme == fail_theme:
                raise RuntimeError("search down for theme")
            hit = SimpleNamespace(context=LONG, text="", snippet="",
                                  passage="", title="Book " + theme,
                                  slug="book-" + theme)
            return [hit]

    return FakeLib


# ── 216: keeper.ask per-theme failure -> continue
def test_216_keeper_theme_failure_skips_only_that_theme():
    before = _snap()
    fk = _keeper("discipline")
    with patch("nomorals.wisdom.keeper.WisdomKeeper", fk), \
         patch("nomorals.wisdom.autonomy.WisdomOrgan", FakeOrgan), \
         patch("nomorals.books.library.Library", FakeLibraryEmpty):
        out = T.library_candidates(_ctx(), ["discipline", "money"])
    assert fk.calls == ["discipline", "money"]  # both attempted
    assert {it["theme"] for it in out} == {"money"}
    fk2 = _keeper(None)
    with patch("nomorals.wisdom.keeper.WisdomKeeper", fk2), \
         patch("nomorals.wisdom.autonomy.WisdomOrgan", FakeOrgan), \
         patch("nomorals.books.library.Library", FakeLibraryEmpty):
        out2 = T.library_candidates(_ctx(), ["discipline", "money"])
    assert {it["theme"] for it in out2} == {"discipline", "money"}
    assert _snap() == before


# ── 243: library.search per-theme failure -> continue
def test_243_library_theme_failure_skips_only_that_theme():
    before = _snap()
    fl = _library("discipline")
    with patch("nomorals.books.library.Library", fl), \
         patch("nomorals.wisdom.keeper.WisdomKeeper", FakeKeeperEmpty), \
         patch("nomorals.wisdom.autonomy.WisdomOrgan", FakeOrgan):
        out = T.library_candidates(_ctx(), ["discipline", "money"])
    assert fl.calls == ["discipline", "money"]
    assert [it["theme"] for it in out] == ["money"]
    fl2 = _library(None)
    with patch("nomorals.books.library.Library", fl2), \
         patch("nomorals.wisdom.keeper.WisdomKeeper", FakeKeeperEmpty), \
         patch("nomorals.wisdom.autonomy.WisdomOrgan", FakeOrgan):
        out2 = T.library_candidates(_ctx(), ["discipline", "money"])
    assert {it["theme"] for it in out2} == {"discipline", "money"}
    assert _snap() == before


# ── 478: _candidate_rows query failure -> []
def test_478_candidate_rows_failure_returns_empty():
    before = _snap()

    def boom(sql):
        raise RuntimeError("db locked")

    db = FakeDB(query=boom)
    led = T.TeachingsLedger(_ctx(db))
    assert led._candidate_rows() == []
    assert len(db.sqls("FROM teachings_candidates")) >= 1
    item = {"id": "x", "theme": "money",
            "source": {"type": "web", "ref": "u", "label": "l",
                       "passage": LONG},
            "title": "t"}
    db2 = FakeDB(query=lambda sql: [
        {"item_json": json.dumps(item), "fetched_at": 500.0}])
    led2 = T.TeachingsLedger(_ctx(db2))
    rows = led2._candidate_rows()
    assert len(rows) == 1 and rows[0][0]["id"] == "x" \
        and rows[0][1] == 500.0
    assert _snap() == before


# ── 552 (FLAGGED, observed only): served_items failure -> []
SERVED_ROW = {
    "id": "i1", "kind": "teaching", "title": "T", "body": "",
    "theme": "discipline", "source_type": "wisdom", "source_ref": "r",
    "source_label": "L", "passage": LONG, "created_at": 900.0,
    "served_json": json.dumps([{"surface": "motivation-morning",
                                 "at": 950.0}]),
    "item_key": "k1",
}


def test_552_served_items_failure_returns_empty_observed():
    before = _snap()

    def boom(sql):
        raise RuntimeError("db locked")

    db = FakeDB(query=boom)
    led = T.TeachingsLedger(_ctx(db))
    assert led.served_items() == []
    assert led.theme_last_served("discipline") == 0.0
    assert len(db.sqls("FROM teachings_items")) >= 1
    db2 = FakeDB(query=lambda sql: [dict(SERVED_ROW)])
    led2 = T.TeachingsLedger(_ctx(db2))
    assert len(led2.served_items()) == 1
    assert led2.theme_last_served("discipline") == 950.0
    assert _snap() == before


# ── 588/592/596: stock() producer guards -> (name, [])
def _stock_counts(raise_which):
    def raiser(*a, **k):
        raise RuntimeError("producer exploded")

    def empty(*a, **k):
        return []

    calls = {"library": 0, "net": 0, "memory": 0}

    def wrap(name, fn):
        def w(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return w

    fns = {"library": empty, "net": empty, "memory": empty}
    if raise_which:
        fns[raise_which] = raiser
    led = T.TeachingsLedger(_ctx(None))
    with patch.object(T, "library_candidates",
                      wrap("library", fns["library"])), \
         patch.object(T, "net_candidates", wrap("net", fns["net"])), \
         patch.object(T, "memory_candidates",
                      wrap("memory", fns["memory"])):
        counts = led.stock(now=1000.0, themes=["money"])
    return counts, calls


def test_588_stock_library_producer_failure_counts_zero():
    before = _snap()
    counts, calls = _stock_counts("library")
    assert calls["library"] == 1
    assert counts["library"] == 0

    def one(*a, **k):
        return [T._base_item(kind="teaching", title="t", theme="money",
                             source_type="web", source_ref="u",
                             source_label="l", passage=LONG)]

    def empty(*a, **k):
        return []

    led = T.TeachingsLedger(_ctx(None))
    with patch.object(T, "library_candidates", one), \
         patch.object(T, "net_candidates", empty), \
         patch.object(T, "memory_candidates", empty):
        counts2 = led.stock(now=1000.0, themes=["money"])
    assert counts2["library"] == 1
    assert _snap() == before


def test_592_stock_net_producer_failure_counts_zero():
    before = _snap()
    counts, calls = _stock_counts("net")
    assert calls["net"] == 1
    assert counts["net"] == 0
    counts2, _ = _stock_counts(None)
    assert counts2["net"] == 0  # empty producer: same 0, honestly
    assert _snap() == before


def test_596_stock_memory_producer_failure_counts_zero():
    before = _snap()
    counts, calls = _stock_counts("memory")
    assert calls["memory"] == 1
    assert counts["memory"] == 0
    assert _snap() == before


# ── 769: mark_served corrupt served_json -> merge skipped
ITEM = {"id": "teach1", "kind": "teaching", "title": "T",
        "theme": "money",
        "source": {"type": "web", "ref": "u", "label": "l",
                   "passage": LONG}}


def test_769_corrupt_served_json_drops_prior_history():
    before = _snap()
    db = FakeDB(query_one=lambda sql: {"served_json": "{not json"})
    led = T.TeachingsLedger(_ctx(db))
    row = led.mark_served(dict(ITEM), "motivation-morning", now=1234.0)
    assert len(db.sqls("SELECT served_json")) >= 1
    assert row["served"] == [{"surface": "motivation-morning",
                               "at": 1234.0}]
    upd = [c for c in db.calls if c[0] == "execute" and "UPDATE" in c[1]]
    assert len(upd) == 1 and json.loads(upd[0][2][1]) == row["served"]
    db2 = FakeDB(query_one=lambda sql: {
        "served_json": json.dumps([{"surface": "trivia",
                                     "at": 1000.0}])})
    led2 = T.TeachingsLedger(_ctx(db2))
    row2 = led2.mark_served(dict(ITEM), "motivation-morning",
                            now=1234.0)
    assert len(row2["served"]) == 2 \
        and row2["served"][0]["surface"] == "trivia"
    assert _snap() == before
