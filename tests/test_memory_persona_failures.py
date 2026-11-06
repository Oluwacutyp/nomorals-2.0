"""Tail-78 failure pins — nomorals/memory/persona.py.

NOTE: persona.py is a guardrail file; the batch as first
surveyed changed no production byte in it.  Five sites are
exclusions (predicate default, fail-safe dedup comparison,
fail-closed contradiction/restore refusals, tolerant metadata
parse).  Site 821 was FLAGGED, then OVERRULED to CONVERT by
parent adjudication (doctor-77 false-success shape: the inner
query guard in _forget_with_archive defeated scheduled_pass's
designed error channel, so a crashed hygiene query reported a
clean "forgotten: 0" with no error key).  The amendment removed
that one guard; the 821 pin asserts the NEW behavior.

Probed live first (nm-tail78-verify/probes.py).
"""
import time
from types import SimpleNamespace

from nomorals.memory import persona as P


# ── 60 (E104) _is_private: unreadable metadata → False ──────────

def test_60_is_private_unreadable_metadata_defaults_false():
    # failure value is byte-identical to a genuinely public record
    assert P._is_private(SimpleNamespace(metadata="corrupt")) is False
    assert P._is_private(SimpleNamespace(metadata={})) is False
    # controls: the predicate itself is exact on readable metadata
    assert P._is_private(
        SimpleNamespace(metadata={"private": True})) is True
    assert P._is_private(
        SimpleNamespace(metadata={"private": False})) is False


# ── 725 (E104) dedup: is_duplicate crash → pair never merged ────

def test_725_duplicate_check_failure_keeps_both_records(monkeypatch):
    from nomorals.memory import extract as extract_mod

    def boom(a, b, r):
        raise RuntimeError("similarity boom")

    monkeypatch.setattr(extract_mod, "is_duplicate", boom)

    class Rec:
        def __init__(self, rid, content):
            self.id = rid
            self.content = content
            self.kind = "episode"
            self.importance = 0.5
            self.metadata = {}

        def reinforcement(self):
            return 0.0

    class StubManager:
        def __init__(self):
            self.forgotten = []
            self.updated = []

        def recall(self, q, limit=0, kind=""):
            return SimpleNamespace(records=[Rec("a", "same text"),
                                            Rec("b", "same text")])

        def update(self, rid, metadata=None):
            self.updated.append(rid)

        def forget(self, rid):
            self.forgotten.append(rid)

    mgr = StubManager()
    out = P.MemoryCurator(mgr).deduplicate()
    # fail-safe direction: a crashed comparison merges nothing —
    # no update, no archive, no forget is recorded
    assert out == {"ok": True, "merged": 0, "kept": 2}
    assert mgr.forgotten == [] and mgr.updated == []


# ── 766 (E104) check_contradiction: recall crash → {} ───────────

def test_766_contradiction_recall_failure_returns_empty():
    class SickManager:
        def __init__(self):
            self.updated = []

        def get(self, rid):
            return SimpleNamespace(id=rid, kind=P.MemoryKind.FACT,
                                   content="likes tea", metadata={})

        def recall(self, *a, **k):
            raise RuntimeError("recall boom")

        def update(self, rid, metadata=None):
            self.updated.append(rid)

    mgr = SickManager()
    out = P.MemoryCurator(mgr).check_contradiction("n1")
    # {} is the documented no-contradiction return; fail-closed:
    # nothing is marked superseded on a crashed lookup
    assert out == {}
    assert mgr.updated == []


def test_766_contradiction_recall_failure_is_recorded():
    # AA-76: the {} above must not be the whole story — a crashed
    # recall leaves a trace in the memory failure ledger, distinct
    # from a genuine no-contradiction result.
    from nomorals.memory import failures as F

    class SickManager:
        def get(self, rid):
            return SimpleNamespace(id=rid, kind=P.MemoryKind.FACT,
                                   content="likes tea", metadata={})

        def recall(self, *a, **k):
            raise RuntimeError("recall boom")

        def update(self, rid, metadata=None):
            raise AssertionError("must not update on a crashed recall")

    F.clear_failures()
    try:
        out = P.MemoryCurator(SickManager()).check_contradiction("n1")
        assert out == {}
        sites = [f.site for f in F.recent_failures(10)]
        assert "memory.check_contradiction_recall" in sites, sites
    finally:
        F.clear_failures()


# ── 821 (E104) CONVERTED (parent adjudication): a crashed ──────
# hygiene query must surface through scheduled_pass's error
# channel, never as a clean-looking "forgotten: 0".

def test_821_forget_query_failure_raises_and_reports_error():
    import pytest

    class SickDB:
        def query(self, sql, params=()):
            raise RuntimeError("query boom")

    class CurManager:
        def __init__(self):
            self.db = SickDB()

        def recall(self, q, limit=0, kind=""):
            return SimpleNamespace(records=[])

    cur = P.MemoryCurator(CurManager())
    # the inner guard is gone: the query crash propagates out of
    # _forget_with_archive instead of returning a silent 0
    with pytest.raises(RuntimeError, match="query boom"):
        cur._forget_with_archive(0.08)
    # and scheduled_pass's designed handler records it: the
    # report carries the error, forgotten stays 0 — a crashed
    # pass is no longer byte-identical to a clean one
    report = cur.scheduled_pass()
    assert report["forgotten"] == 0
    assert report["error"] == "query boom"


# ── 869 (E104) restore: archive query crash → False ─────────────

def test_869_restore_query_failure_returns_false():
    class SickDB:
        def query(self, sql, params=()):
            raise RuntimeError("query boom")

    cur = P.MemoryCurator(SimpleNamespace(db=SickDB()))
    assert cur.restore("x1") is False
    # byte-identical to the genuine no-such-record refusal
    class EmptyDB:
        def query(self, sql, params=()):
            return []

    cur2 = P.MemoryCurator(SimpleNamespace(db=EmptyDB()))
    assert cur2.restore("x1") is False


# ── 878 (E104) restore: corrupt metadata JSON → {} ──────────────

def test_878_restore_corrupt_metadata_restores_with_empty():
    class ArchiveDB:
        def query(self, sql, params=()):
            return [{"id": "x1", "content": "old fact", "kind": "fact",
                     "importance": 0.7, "metadata": "{corrupt json",
                     "archived_at": time.time()}]

    class RestoreManager:
        def __init__(self):
            self.db = ArchiveDB()
            self.calls = []

        def remember(self, content, **kw):
            self.calls.append((content, kw))
            return "new-id"

    mgr = RestoreManager()
    cur = P.MemoryCurator(mgr)
    assert cur.restore("x1") is True
    # the substance (content) is restored; only the corrupt
    # metadata degrades to the genuine empty-metadata value
    assert mgr.calls[0][0] == "old fact"
    assert mgr.calls[0][1].get("metadata") == {}
