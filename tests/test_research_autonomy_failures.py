"""Tail-81 failure pins — nomorals/research/autonomy.py.

All six sites are EXCLUSIONS, probed live first
(nm-tail81-verify/probes.py, 24/24 behavior confirmations):

- 92: the ensure_schema migration loop swallows the
  "column already there" failure of idempotent ALTERs — the
  documented steady state on any pre-existing DB; a genuinely
  broken DB still fails loudly at the unguarded CREATE TABLEs
  that follow.
- 245: note_gap's lastrowid read is guarded to the no-id
  value 0. The production Database.execute() returns a real
  sqlite3.Cursor whose lastrowid never raises, so the guard
  is unreachable under the production contract; the consumer
  (note_knowledge_gap tool) returns ok:True unconditionally
  and _drain_directives ignores the value entirely. The pin
  records the shim-cursor behavior honestly: 0 returned, the
  gap row still inserted (dedupe finds it on re-note).
- 271: gap_stats is best-effort observability with NO
  production consumer (tick reads open_gaps directly); its
  in-code contract is "stats never break the organ". A broken
  DB reads as the zeroed stats dict.
- 281: _domain's "" is the genuine no-domain value; its only
  consumer (record_source_quality) early-returns on "" —
  nothing is recorded either way, which is correct: no domain
  could be extracted.
- 493: malformed stored unresolved/contradictions JSON
  degrades to the empty lists. The organ writes those columns
  itself via json.dumps (always valid), so malformed implies
  external corruption; the only production consumer
  (agency/systems.py) reads gap_id/status/synthesis only.
  The degradation is read-only — the stored row is untouched.
- 570: the MemoryKind import fallback "fact" is
  value-identical to MemoryKind.FACT (a plain string
  constant), so the integrated memory is stored under the
  same kind on both rungs.
"""
from types import SimpleNamespace
from unittest import mock

import nomorals.memory.base as base_mod
from nomorals.research import autonomy as ra
from nomorals.storage.db import Database


def _organ(tmp_path, name="organ.db"):
    db = Database(tmp_path / name)
    ctx = SimpleNamespace(db=db, tools=None, memory=None,
                          gateway=None, router=None)
    return ra.ResearchOrgan(ctx), db


class _BadCursor:
    """Cursor shim whose lastrowid read raises (never happens
    with the production sqlite3.Cursor)."""

    def __init__(self, real):
        self._real = real

    @property
    def lastrowid(self):
        raise RuntimeError("lastrowid boom")

    @property
    def rowcount(self):
        return self._real.rowcount


class _ShimDB:
    def __init__(self, real):
        self._real = real

    def execute(self, sql, params=()):
        cur = self._real.execute(sql, params)
        if sql.lstrip().upper().startswith("INSERT INTO KNOWLEDGE_GAPS"):
            return _BadCursor(cur)
        return cur

    def query(self, sql, params=()):
        return self._real.query(sql, params)

    def query_one(self, sql, params=()):
        return self._real.query_one(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_92_ensure_schema_alter_swallow_is_idempotent(tmp_path):
    db = Database(tmp_path / "schema.db")
    # First call creates; later calls hit "column already
    # there" on every migration ALTER — swallowed by design.
    ra.ensure_schema(db)
    ra.ensure_schema(db)
    ra.ensure_schema(db)
    tables = {r["name"] for r in db.query(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"research_watches", "knowledge_gaps",
            "source_quality"} <= tables


def test_245_note_gap_lastrowid_failure_returns_zero_but_row_persists(tmp_path):
    organ, db = _organ(tmp_path)
    gid = organ.note_gap("why is the sky blue?", "pin")
    assert gid > 0
    assert organ.note_gap("why is the sky blue?", "pin") == gid

    shim = ra.ResearchOrgan(SimpleNamespace(
        db=_ShimDB(db), tools=None, memory=None,
        gateway=None, router=None))
    gid2 = shim.note_gap("a wholly different question?", "")
    assert gid2 == 0
    # The insert persisted despite the no-id return: dedupe on
    # a re-note finds the real row and returns its real id.
    assert organ.note_gap("a wholly different question?", "") > 0


def test_271_gap_stats_broken_db_returns_zeroed_stats(tmp_path):
    organ, _db = _organ(tmp_path)
    organ.note_gap("counted gap?", "")
    stats = organ.gap_stats()
    assert stats["open"] == 1 and stats["resolved"] == 0

    class BoomDB:
        def query(self, *a, **k):
            raise RuntimeError("db gone")

    broken = ra.ResearchOrgan.__new__(ra.ResearchOrgan)
    broken.db = BoomDB()
    broken.context = SimpleNamespace()
    # "stats never break the organ": the zeroed snapshot is
    # the documented degradation, byte-identical to an empty
    # gaps table.
    assert broken.gap_stats() == {
        "open": 0, "resolved": 0, "stalled": 0, "top_open": []}


def test_281_domain_unparseable_returns_empty_and_records_nothing(tmp_path):
    organ, db = _organ(tmp_path)
    assert ra.ResearchOrgan._domain(
        "https://www.Example.com/x") == "example.com"
    assert ra.ResearchOrgan._domain("notaurl") == ""
    # urlparse raises ValueError on an invalid IPv6 literal;
    # the guard yields the same "" as a domain-less URL.
    assert ra.ResearchOrgan._domain("http://[::1/bad") == ""
    # NOT-recording: an unparseable URL teaches the organ
    # nothing — the source_quality table stays empty.
    organ.record_source_quality("http://[::1/bad", worth=True)
    assert db.query("SELECT * FROM source_quality") == []
    organ.record_source_quality("https://example.com/a", worth=True)
    assert len(db.query("SELECT * FROM source_quality")) == 1


def test_493_investigate_gap_malformed_stored_json_returns_empty_lists(tmp_path):
    organ, db = _organ(tmp_path)
    gid = organ.note_gap("malformed storage gap?", "")
    db.execute("UPDATE knowledge_gaps SET unresolved = 'not-json',"
               " contradictions = '{bad' WHERE id = ?", (gid,))
    with mock.patch.object(ra.ResearchOrgan, "_work_one_gap",
                           lambda self, g, r: None):
        res = organ.investigate_gap(gid)
    assert res["unresolved"] == [] and res["contradictions"] == []
    assert res["status"] == "open"
    # Read-only degradation: the stored row is NOT rewritten.
    row = db.query_one(
        "SELECT unresolved FROM knowledge_gaps WHERE id = ?", (gid,))
    assert row["unresolved"] == "not-json"


def test_570_integrate_gap_answer_kind_fallback_is_the_fact_string(tmp_path):
    organ, _db = _organ(tmp_path)
    captured = {}

    class FakeMemory:
        def remember(self, content, **kw):
            captured.update(kw)

    organ.context = SimpleNamespace(db=organ.db, memory=FakeMemory())
    gap = {"id": 7, "question": "q?"}
    deep = SimpleNamespace(synthesis="synth", learnings=["l1"])
    organ._integrate_gap_answer(gap, deep)
    assert captured["kind"] == base_mod.MemoryKind.FACT == "fact"

    saved = base_mod.MemoryKind
    del base_mod.MemoryKind
    captured.clear()
    try:
        organ._integrate_gap_answer(gap, deep)
    finally:
        base_mod.MemoryKind = saved
    # The fallback rung stores under the identical kind value.
    assert captured["kind"] == "fact"
