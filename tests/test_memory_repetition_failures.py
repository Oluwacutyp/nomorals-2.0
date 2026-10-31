"""Tail-79 failure pins — nomorals/memory/repetition.py.

All six sites are EXCLUSIONS.  The module's contract is
printed on its face: "Every public method never raises — a
broken scheduler must never break chat or the briefing."
Each guarded failure returns the module's own documented
broken/unknown/empty value (None / False / 0), the direction
is fail-closed or truthful, and no production decision
consumes any of these surfaces (production calls due(),
schedule() and review() — none of them census sites).

Probed live first (nm-tail79-verify/probes.py).
"""
from nomorals.memory.repetition import GRADE_GOOD, RepetitionScheduler


def _sched(tmp_path, name="rep.db"):
    return RepetitionScheduler(db=str(tmp_path / name))


def _boom(*a, **k):
    raise RuntimeError("query boom")


# ── 328 (E103) _init_schema: ALTER on an existing column ───────

def test_328_schema_reinit_tolerates_existing_columns(tmp_path):
    first = _sched(tmp_path)
    assert first.schedule("m1", "ada lovelace") is True
    # second init runs the FSRS ALTERs against columns that
    # already exist; the guard absorbs exactly that, and the
    # scheduler comes up fully working (failure state == the
    # success state: the column is there)
    second = _sched(tmp_path)
    assert second.get("m1") is not None
    assert second.schedule("m2", "grace hopper") is True
    assert second.count() == 2


# ── 446 (E104) retrievability_of: read failure → None ──────────

def test_446_retrievability_failure_returns_none(tmp_path):
    s = _sched(tmp_path)
    s.schedule("m1", "grace hopper")
    s.review("m1", GRADE_GOOD)
    healthy = s.retrievability_of("m1")
    assert isinstance(healthy, float) and 0.0 < healthy <= 1.0
    # a crashed read yields None — the documented unknown /
    # broken value, distinguishable from every real probability
    s.get = _boom
    assert s.retrievability_of("m1") is None
    # byte-identical to the genuine unknown-card value
    s2 = _sched(tmp_path, "other.db")
    assert s2.retrievability_of("nope") is None


# ── 456 (E104) fs_card: read failure → None ────────────────────

def test_456_fs_card_failure_returns_none(tmp_path):
    s = _sched(tmp_path)
    s.schedule("m1", "grace hopper")
    s.review("m1", GRADE_GOOD)
    assert s.fs_card("m1") is not None
    s.get = _boom
    assert s.fs_card("m1") is None
    # byte-identical to the genuine unknown-card value
    assert _sched(tmp_path, "other.db").fs_card("nope") is None


# ── 483 (E104) remove: delete failure → False ──────────────────

def test_483_remove_failure_returns_false_and_keeps_card(tmp_path):
    s = _sched(tmp_path)
    s.schedule("m1", "stored card")
    s._db.close()  # every execute now raises ProgrammingError
    # False is the truthful answer: the card was NOT removed
    assert s.remove("m1") is False
    reopened = _sched(tmp_path)
    assert reopened.get("m1") is not None
    assert reopened.count() == 1
    # healthy control: a landed delete returns True
    assert reopened.remove("m1") is True
    assert reopened.count() == 0


# ── 497 (E104) get: read failure → None ────────────────────────

def test_497_get_failure_returns_none(tmp_path):
    s = _sched(tmp_path)
    s.schedule("m1", "stored card")
    assert s.get("m1") is not None
    s._db.close()
    # None is the documented "no card / broken store" value;
    # every internal consumer (review, bury, retrievability)
    # treats it as unknown and does nothing — fail-closed
    assert s.get("m1") is None
    # byte-identical to the genuine missing-card value
    assert _sched(tmp_path, "other.db").get("nope") is None


# ── 551 (E104) count: count failure → 0 ────────────────────────

def test_551_count_failure_returns_zero(tmp_path):
    s = _sched(tmp_path)
    s.schedule("m1", "stored card")
    assert s.count() == 1
    s._db.close()
    # the crashed count reads 0 — byte-identical to a genuinely
    # empty store AND to the module's designed broken-store
    # value (the _broken() path returns the same 0); no
    # production surface consumes count()
    assert s.count() == 0
    assert _sched(tmp_path).count() == 1  # the card is really there
    assert _sched(tmp_path, "empty.db").count() == 0
