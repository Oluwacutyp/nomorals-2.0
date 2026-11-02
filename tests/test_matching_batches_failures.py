"""Pass tail-115 pins for nomorals/matching/batches.py — 0 converts / 3 exclusions.

157 (curate_daily) and 218 (explain_pick): a preferences dict whose
values fail float() degrades to {} — the module's documented
unweighted path (in-code note: "bad prefs degrade to unweighted"),
byte-identical to passing no preferences at all. Nothing is
fabricated: the batch is still curated from the real pool, and the
explanation truthfully reports that no preferences matched.

349 (DailyBatchStore.candidates): one row's corrupt ``attributes``
JSON degrades that row's attributes to {} while the candidate itself
survives with its real id/title/tags/quality. Per-record parse guard;
siblings with valid JSON round-trip untouched. The guard records
nothing (the module's outer handlers log warnings — this inner one
does not; pinned via caplog).

No except handlers appear in this file (AST-checked).
"""
import logging

from nomorals.matching.batches import (Candidate, DailyBatchStore,
                                       curate_daily, explain_pick)


def _cands():
    return [Candidate(candidate_id=f"c{i}", title=f"T{i}",
                      tags=("afrobeats",) if i % 2 == 0 else ("jazz",),
                      quality=0.5 + i * 0.1) for i in range(4)]


def test_157_bad_preference_values_degrade_to_unweighted():
    bad = curate_daily(_cands(), n=2, preferences={"afrobeats": "loud"},
                       seed=42)
    plain = curate_daily(_cands(), n=2, preferences=None, seed=42)
    assert [c.candidate_id for c in bad] == [c.candidate_id for c in plain]
    assert len(bad) == 2


def test_157_healthy_preferences_control():
    picked = curate_daily(_cands(), n=2, preferences={"afrobeats": 3.0},
                          seed=42)
    assert len(picked) == 2


def test_218_bad_preference_values_explain_as_no_match():
    txt = explain_pick(_cands()[0], preferences={"afrobeats": "loud"})
    assert "no stated preferences matched" in txt


def test_218_healthy_preferences_control():
    txt = explain_pick(_cands()[0], preferences={"afrobeats": 2.0})
    assert "prefs" in txt and "no stated" not in txt


def test_349_corrupt_attributes_row_survives_with_empty_attrs(tmp_path,
                                                              caplog):
    store = DailyBatchStore(str(tmp_path / "b.db"))
    assert store.add_candidate(Candidate(candidate_id="x1", title="Gig One",
                                         attributes={"pay": 500}))
    store._db.execute(
        "UPDATE candidates SET attributes=? WHERE candidate_id=?",
        ("{not json", "x1"))
    store._db.commit()
    with caplog.at_level(logging.WARNING, logger="nomorals.matching.batches"):
        got = store.candidates()
    assert len(got) == 1
    assert got[0].candidate_id == "x1"
    assert got[0].title == "Gig One"
    assert got[0].attributes == {}
    # the inner guard records nothing — no "candidates failed" warning
    assert "candidates failed" not in caplog.text


def test_349_healthy_attributes_round_trip(tmp_path):
    store = DailyBatchStore(str(tmp_path / "b2.db"))
    assert store.add_candidate(Candidate(candidate_id="y1",
                                         attributes={"pay": 700}))
    assert store.candidates()[0].attributes == {"pay": 700}
