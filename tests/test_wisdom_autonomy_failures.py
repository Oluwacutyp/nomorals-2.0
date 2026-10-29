"""Tail-54 pins for nomorals/wisdom/autonomy.py —
0 converts / 8 exclusions (56, 249, 452, 464, 526, 533, 691, 702).

8 tests for 8 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

autonomy.py calls no failures-ledger record function anywhere
(grep-proven: no `record_` call in the file) and the wisdom
package has no failures ledger of its own, so NOT-recording is
pinned against the module's own observability surface: the
organ DB state (organ_events bus table + the wisdom_ingest_
queue / wisdom_digests / wisdom_links tables) snapshot
before/after each trigger — no event emitted, no failure row
written, no table mutated beyond the site's intended operation.

Verdict notes (full grounds in the tail-54 SURVEY.md):
- 56: _brain_suggest brain failure -> None, the documented
  genuine no-brain value ("no brain == maths-only refill");
  TriviaForge is built to run suggest=None, and failures of an
  actual suggest() call are recorded by the forge itself.
- 249: queue_ingest lastrowid readout failure -> 0; the INSERT
  itself succeeded (row visible), 0 is the same value a None
  lastrowid yields, and no consumer branches on the id.
- 452: one seed-term keeper.ask failure -> continue; the
  digest is built from the surviving terms' real passages, or
  digest_work raises downstream — never a fabricated digest.
- 464: fallback keeper.ask failure -> [] transiently, then
  digest_work raises ValueError; no digest is persisted or
  rendered from the empty list.
- 526: PRAGMA failure in _ensure_digest_schema -> return;
  the caller's next SELECT decides — if the schema is truly
  unusable it raises visibly there.
- 533: ALTER failure -> pass; the anticipated failure is the
  duplicate-column race, and the staleness verdict after the
  pass is provably still correct.
- 691: HistoryEngine failure -> timeline_thin_traditions [];
  trigger is the fail-fast HistoryError on a corrupt timeline
  dataset; no consumer renders or decides from that field,
  and candidates derive from the separate manifest leg.
- 702: one tradition's ingestor.search failure -> continue;
  search records every backend failure in the source registry
  before raising, and the candidate list is a best-effort
  hunt proposal, not an absence claim.
"""

import sqlite3
from types import SimpleNamespace
from unittest import mock

import pytest

import nomorals.llm.brain as brain_mod
import nomorals.wisdom.history as hist_mod
import nomorals.wisdom.ingestor as ing_mod
from nomorals.wisdom import autonomy as wa


class FakeDB:
    """Minimal in-memory stand-in for nomorals.storage.db.Database."""

    def __init__(self):
        self._c = sqlite3.connect(":memory:")
        self._c.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        cur = self._c.execute(sql, params)
        self._c.commit()
        return cur

    def query(self, sql, params=()):
        return [dict(r) for r in self._c.execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        row = self._c.execute(sql, params).fetchone()
        return dict(row) if row is not None else None


def _organ(db):
    organ = wa.WisdomOrgan.__new__(wa.WisdomOrgan)
    organ.context = SimpleNamespace(db=db)
    organ.db = db
    wa.ensure_schema(db)
    return organ


def _snap(db):
    return {
        t: db.query(f"SELECT * FROM {t} ORDER BY 1")
        for t in ("organ_events", "wisdom_ingest_queue",
                  "wisdom_digests", "wisdom_links")
    }


def _entry(slug, title, tradition, sha="abc"):
    return SimpleNamespace(
        slug=slug, title=title, tradition=tradition,
        canon_status="canon", source_url="http://x", license="pd",
        translator="", sha256=sha, ingested_at=1.0, notes="")


SNIP = ("In the beginning God created the heaven and the earth. "
        "And the earth was without form, and void, and darkness "
        "was upon the face of the deep.")


def _passage(slug):
    return SimpleNamespace(work="King James Bible", section="Gen 1",
                           snippet=SNIP, url="", slug=slug)


class FakeKeeper:
    def __init__(self, entries, ask_fn):
        self.corpus = SimpleNamespace(_manifest=entries)
        self._ask_fn = ask_fn
        self.ask_log = []

    def ask(self, query, top=5, mode="keyword"):
        self.ask_log.append(query)
        return self._ask_fn(query)


# ── 56 ────────────────────────────────────────────────────────────

def test_56_brain_suggest_failure_returns_none_documented_no_brain():
    db = FakeDB()
    wa.ensure_schema(db)
    ctx = SimpleNamespace(db=db)
    calls = []

    def raising_brain_for(c):
        calls.append(c)
        raise RuntimeError("brain wiring boom")

    before = _snap(db)
    with mock.patch.object(brain_mod, "brain_for", raising_brain_for):
        assert wa._brain_suggest(ctx) is None
    assert calls == [ctx]  # brain_for was actually attempted
    assert _snap(db) == before  # nothing recorded on the organ DB
    # genuine value control: no brain at all -> also None
    with mock.patch.object(brain_mod, "brain_for", lambda c: None):
        assert wa._brain_suggest(ctx) is None
    # healthy control: a real brain yields a working suggest callable
    chat_log = []

    class FakeBrain:
        def chat(self, msgs):
            chat_log.append(msgs)
            return SimpleNamespace(text="suggestion-text")

    with mock.patch.object(brain_mod, "brain_for", lambda c: FakeBrain()):
        suggest = wa._brain_suggest(ctx)
    assert suggest is not None
    assert suggest("prompt-1") == "suggestion-text"
    assert chat_log[0][0]["content"] == "prompt-1"


# ── 249 ───────────────────────────────────────────────────────────

class _BadCursor:
    def __init__(self, real):
        self._real = real

    @property
    def lastrowid(self):
        raise RuntimeError("lastrowid boom")


class _WrapDB:
    def __init__(self, real):
        self._real = real
        self.sql_log = []

    def execute(self, sql, params=()):
        self.sql_log.append(sql)
        cur = self._real.execute(sql, params)
        if sql.lstrip().upper().startswith(
                "INSERT INTO WISDOM_INGEST_QUEUE"):
            return _BadCursor(cur)
        return cur

    def query(self, sql, params=()):
        return self._real.query(sql, params)

    def query_one(self, sql, params=()):
        return self._real.query_one(sql, params)


def test_249_queue_ingest_lastrowid_failure_returns_zero_row_saved():
    db = FakeDB()
    organ = _organ(db)
    wrap = _WrapDB(db)
    organ.db = wrap
    before = _snap(db)
    qid = organ.queue_ingest("http://x/y", "Some Text", "sufism")
    assert qid == 0
    # the INSERT was attempted and SUCCEEDED — only the id readout failed
    assert any("INSERT INTO wisdom_ingest_queue" in s
               for s in wrap.sql_log)
    row = db.query_one(
        "SELECT id, url FROM wisdom_ingest_queue WHERE url = 'http://x/y'")
    assert row is not None and row["id"] >= 1
    after = _snap(db)
    # exactly the intended queue row was added; nothing else recorded
    assert len(after["wisdom_ingest_queue"]) == \
        len(before["wisdom_ingest_queue"]) + 1
    assert after["organ_events"] == before["organ_events"]
    assert after["wisdom_links"] == before["wisdom_links"]
    # healthy control: real id returned, dedupe returns the same id
    organ.db = db
    qid2 = organ.queue_ingest("http://x/z", "Other Text", "sufism")
    assert qid2 == db.query_one(
        "SELECT id FROM wisdom_ingest_queue WHERE url='http://x/z'")["id"]
    assert organ.queue_ingest("http://x/z", "Other Text", "sufism") == qid2


# ── 452 ───────────────────────────────────────────────────────────

def test_452_digest_seed_term_ask_failure_continues_to_other_terms():
    db = FakeDB()
    organ = _organ(db)
    entry = _entry("bible-kjv", "King James Bible", "christianity")

    def ask_fn(q):
        if q == "king":
            raise RuntimeError("ask boom for king")
        return SimpleNamespace(passages=[_passage("bible-kjv")])

    keeper = FakeKeeper({"bible-kjv": entry}, ask_fn)
    organ._keeper = lambda: keeper
    before = _snap(db)
    digest = organ.digest_work("bible-kjv")
    assert keeper.ask_log[0] == "king"  # failing term attempted first
    assert "james" in keeper.ask_log  # loop continued past the failure
    assert digest["slug"] == "bible-kjv" and digest["passages"]
    assert organ.get_digest("bible-kjv")["title"] == "King James Bible"
    after = _snap(db)
    # exactly the intended digest row; no event / queue / link recorded
    assert len(after["wisdom_digests"]) == len(before["wisdom_digests"]) + 1
    assert after["organ_events"] == before["organ_events"]
    assert after["wisdom_ingest_queue"] == before["wisdom_ingest_queue"]
    assert after["wisdom_links"] == before["wisdom_links"]


# ── 464 ───────────────────────────────────────────────────────────

def test_464_digest_fallback_ask_failure_raises_no_digest_fabricated():
    db = FakeDB()
    organ = _organ(db)
    entry = _entry("bible-kjv", "King James Bible", "christianity")

    def ask_all_boom(q):
        raise RuntimeError("ask boom everywhere")

    keeper = FakeKeeper({"bible-kjv": entry}, ask_all_boom)
    organ._keeper = lambda: keeper
    before = _snap(db)
    with pytest.raises(ValueError, match="no passages found"):
        organ.digest_work("bible-kjv")
    # the fallback ask (slug-derived query) was attempted last
    assert keeper.ask_log[-1] == "bible kjv"
    assert keeper.ask_log[0] == "king"  # seed terms attempted first
    assert db.query("SELECT slug FROM wisdom_digests") == []
    assert _snap(db) == before  # nothing recorded, nothing persisted
    # control: seed asks fail but the fallback ask succeeds -> real digest
    db2 = FakeDB()
    organ2 = _organ(db2)

    def ask_fallback_ok(q):
        if q == "bible kjv":
            return SimpleNamespace(passages=[_passage("bible-kjv")])
        raise RuntimeError("seed ask boom")

    keeper2 = FakeKeeper({"bible-kjv": entry}, ask_fallback_ok)
    organ2._keeper = lambda: keeper2
    digest2 = organ2.digest_work("bible-kjv")
    assert digest2["slug"] == "bible-kjv" and digest2["passages"]


# ── 526 ───────────────────────────────────────────────────────────

class _PragmaBoomDB:
    def __init__(self, real):
        self._real = real
        self.sql_log = []

    def execute(self, sql, params=()):
        self.sql_log.append(sql)
        if "PRAGMA" in sql.upper():
            raise RuntimeError("pragma boom")
        return self._real.execute(sql, params)

    def query(self, sql, params=()):
        return self._real.query(sql, params)

    def query_one(self, sql, params=()):
        return self._real.query_one(sql, params)


def test_526_ensure_digest_schema_pragma_failure_returns():
    db = FakeDB()
    organ = _organ(db)
    boom = _PragmaBoomDB(db)
    organ.db = boom
    before = _snap(db)
    assert organ._ensure_digest_schema() is None  # returns, never raises
    assert any("PRAGMA" in s.upper() for s in boom.sql_log)  # attempted
    assert _snap(db) == before
    # healthy control: the ALTER path adds source_sha to a fresh table
    organ.db = db
    cols_before = {r[1] for r in db.execute(
        "PRAGMA table_info(wisdom_digests)").fetchall()}
    assert "source_sha" not in cols_before
    organ._ensure_digest_schema()
    cols_after = {r[1] for r in db.execute(
        "PRAGMA table_info(wisdom_digests)").fetchall()}
    assert "source_sha" in cols_after


# ── 533 ───────────────────────────────────────────────────────────

class _AlterBoomDB:
    """PRAGMA reports source_sha missing; ALTER raises duplicate-column
    (the column actually exists underneath — the migration race)."""

    def __init__(self, real):
        self._real = real
        self.sql_log = []

    def execute(self, sql, params=()):
        self.sql_log.append(sql)
        up = sql.upper()
        if "PRAGMA" in up:
            real_cur = self._real.execute(sql, params)

            class FakeCur:
                def fetchall(self):
                    return [r for r in real_cur.fetchall()
                            if r[1] != "source_sha"]
            return FakeCur()
        if "ALTER TABLE" in up:
            raise RuntimeError("duplicate column name: source_sha")
        return self._real.execute(sql, params)

    def query(self, sql, params=()):
        return self._real.query(sql, params)

    def query_one(self, sql, params=()):
        return self._real.query_one(sql, params)


def test_533_ensure_digest_schema_alter_failure_passes_verdict_intact():
    db = FakeDB()
    organ = _organ(db)
    organ._ensure_digest_schema()  # column really added (race setup)
    db.execute(
        "INSERT INTO wisdom_digests (slug, title, tradition, key_terms,"
        " passages, digested_at, source_sha) VALUES"
        " ('bible-kjv', 'King James Bible', 'christianity',"
        " '[]', '[]', 1, 'abc')")
    entry = _entry("bible-kjv", "King James Bible", "christianity",
                   sha="abc")
    keeper = FakeKeeper({"bible-kjv": entry},
                        lambda q: SimpleNamespace(passages=[]))
    organ._keeper = lambda: keeper
    wrap = _AlterBoomDB(db)
    organ.db = wrap
    before = _snap(db)
    assert organ._ensure_digest_schema() is None  # pass, no raise
    assert any("ALTER TABLE" in s.upper() for s in wrap.sql_log)
    # the staleness verdict after the swallowed ALTER is still CORRECT:
    # stored sha == entry sha -> not stale
    assert organ.digest_stale("bible-kjv") is False
    assert _snap(db) == before


# ── 691 ───────────────────────────────────────────────────────────

def test_691_gap_analysis_history_failure_timeline_empty_only():
    db = FakeDB()
    organ = _organ(db)
    entry = _entry("w1", "Some Work", "sufism")
    keeper = FakeKeeper({"w1": entry},
                        lambda q: SimpleNamespace(passages=[]))
    organ._keeper = lambda: keeper
    hist_calls = []

    class BoomHistory:
        def __init__(self, ctx):
            hist_calls.append(ctx)
            raise RuntimeError("timeline dataset boom")

    class FakeIngestor:
        def __init__(self, ctx):
            pass

        def search(self, q, tradition="", max_results=3, sources=None):
            return []

    before = _snap(db)
    with mock.patch.object(hist_mod, "HistoryEngine", BoomHistory), \
            mock.patch.object(ing_mod, "ArchiveIngestor", FakeIngestor):
        gaps = organ.gap_analysis(min_works_per_tradition=3)
    assert hist_calls  # HistoryEngine construction was attempted
    assert gaps["timeline_thin_traditions"] == []
    # the manifest leg of the same report stays truthful
    assert gaps["thin_traditions"] == ["sufism"]
    assert gaps["ingested_by_tradition"] == {"sufism": 1}
    assert gaps["candidates"] == []
    assert _snap(db) == before
    # healthy control: a working timeline leg lands in the report

    class GoodHistory:
        def __init__(self, ctx):
            pass

        def gaps(self):
            return {"thin_traditions": ["vedic"]}

    with mock.patch.object(hist_mod, "HistoryEngine", GoodHistory), \
            mock.patch.object(ing_mod, "ArchiveIngestor", FakeIngestor):
        gaps2 = organ.gap_analysis(min_works_per_tradition=3)
    assert gaps2["timeline_thin_traditions"] == ["vedic"]


# ── 702 ───────────────────────────────────────────────────────────

def test_702_gap_analysis_search_failure_skips_one_tradition():
    db = FakeDB()
    organ = _organ(db)
    e1 = _entry("w1", "Alpha Work", "alpha-trad")
    e2 = _entry("w2", "Beta Work", "beta-trad")
    keeper = FakeKeeper({"w1": e1, "w2": e2},
                        lambda q: SimpleNamespace(passages=[]))
    organ._keeper = lambda: keeper

    class GoodHistory:
        def __init__(self, ctx):
            pass

        def gaps(self):
            return {"thin_traditions": []}

    search_log = []

    class FlakyIngestor:
        def __init__(self, ctx):
            pass

        def search(self, q, tradition="", max_results=3, sources=None):
            search_log.append(tradition)
            if tradition == "alpha-trad":
                raise RuntimeError("every search backend failed")
            return [{"title": "Beta Candidate", "url": "http://x/beta"}]

    before = _snap(db)
    with mock.patch.object(hist_mod, "HistoryEngine", GoodHistory), \
            mock.patch.object(ing_mod, "ArchiveIngestor", FlakyIngestor):
        gaps = organ.gap_analysis(min_works_per_tradition=3)
    # both traditions were actually hunted, in census (sorted) order
    assert search_log == ["alpha-trad", "beta-trad"]
    assert [c["for_tradition"] for c in gaps["candidates"]] == ["beta-trad"]
    assert gaps["thin_traditions"] == ["alpha-trad", "beta-trad"]
    assert _snap(db) == before
    # healthy control: both searches succeed -> both candidates present

    class GoodIngestor:
        def __init__(self, ctx):
            pass

        def search(self, q, tradition="", max_results=3, sources=None):
            return [{"title": f"Cand {tradition}", "url": "http://x/c"}]

    with mock.patch.object(hist_mod, "HistoryEngine", GoodHistory), \
            mock.patch.object(ing_mod, "ArchiveIngestor", GoodIngestor):
        gaps2 = organ.gap_analysis(min_works_per_tradition=3)
    assert len(gaps2["candidates"]) == 2
