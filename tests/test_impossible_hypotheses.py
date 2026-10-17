"""Tests for multi-hypothesis Impossible stage 8 + attempt progression.

One hypothesis is an anchor, not a diagnosis: a problem must produce
>=3 competing hypotheses (mechanism, constraint, framing), each with a
falsifier + cheapest killing experiment, ranked by falsifiability x
cheapness x explanatory power. A failed first attempt moves to the next
hypothesis, not back to guessing.
"""

from __future__ import annotations

import pytest

from nomorals.impossible import ImpossibleMadePossible
from nomorals.impossible import formulate as formulate_mod
from nomorals.impossible import hypothesis as hypothesis_mod
from nomorals.impossible.attempt import AttemptQueue
from nomorals.impossible.ledger import HonestyLedger


@pytest.fixture()
def imp(tmp_path):
    return ImpossibleMadePossible(ledger_path=tmp_path / "ledger.json",
                                  queue_path=tmp_path / "queue.json",
                                  spec_dir=tmp_path / "specs")


def _formulation(problem="a house in Lagos that stays cool without grid power, under $500"):
    return formulate_mod.formulate(problem)


def test_multi_generates_three_ranked_hypotheses():
    f = _formulation()
    multi = hypothesis_mod.hypothesize_multi(f, {"verdict": "achievable_with_work", "failed": []})
    hyps = multi["hypotheses"]
    assert len(hyps) >= 3
    angles = {h["angle"] for h in hyps}
    assert {"mechanism", "constraint", "framing"} <= angles
    for h in hyps:
        assert h["falsifier"], f"{h['angle']} has no falsifier"
        assert h["cheapest_killing_experiment"], f"{h['angle']} has no experiment"
        assert 0.0 < h["score"] <= 1.0
        assert h["rank"] >= 1
    # Ranked best-first by score.
    scores = [h["score"] for h in hyps]
    assert scores == sorted(scores, reverse=True)
    assert multi["primary"]["angle"] == hyps[0]["angle"]


def test_score_is_falsifiability_times_cheapness_times_power():
    h = {"falsifiability": 0.9, "cheapness": 0.9, "explanatory_power": 0.7}
    assert hypothesis_mod.score_hypothesis(h) == round(0.9 * 0.9 * 0.7, 4)
    # A hypothesis that cannot be killed cheaply ranks low even if it explains a lot.
    weak = {"falsifiability": 0.5, "cheapness": 0.5, "explanatory_power": 0.95}
    strong = {"falsifiability": 0.9, "cheapness": 0.9, "explanatory_power": 0.7}
    assert hypothesis_mod.score_hypothesis(weak) < hypothesis_mod.score_hypothesis(strong)


def test_hypothesize_backward_compatible_shape():
    f = _formulation()
    single = hypothesis_mod.hypothesize(f, {"verdict": "achievable_with_work", "failed": []})
    # Old keys still present.
    assert single["claim"] and single["falsifier"] and single["cheapest_killing_experiment"]
    assert single["status"] == "hypothesis"
    # New: full ranked list rides along.
    assert len(single["hypotheses"]) >= 3
    assert single["primary_angle"] == single["hypotheses"][0]["angle"]


def test_submit_dossier_carries_competing_hypotheses(imp):
    d = imp.submit("a house in Lagos that stays cool without grid power, under $500")
    assert len(d["hypothesis"]["hypotheses"]) >= 3
    assert "Competing hypotheses:" in d["report"]


def test_attempt_tries_best_first_then_next(tmp_path):
    imp = ImpossibleMadePossible(ledger_path=tmp_path / "ledger.json",
                                queue_path=tmp_path / "queue.json",
                                spec_dir=tmp_path / "specs")
    d = imp.submit("a house in Lagos that stays cool without grid power, under $500")
    claim_id = d["claim_id"]
    ranked = d["hypothesis"]["hypotheses"]
    first = imp.attempt(claim_id)
    assert first["ok"] is True
    assert first["hypothesis"] == ranked[0]["angle"]
    assert first["remaining_hypotheses"] == [h["angle"] for h in ranked[1:]]
    # Second attempt moves to the NEXT hypothesis, not a repeat of the first.
    second = imp.attempt(claim_id)
    assert second["ok"] is True
    assert second["hypothesis"] == ranked[1]["angle"]
    assert second["hypothesis"] != first["hypothesis"]
    # Ledger records which hypothesis each attempt used.
    entry = HonestyLedger(tmp_path / "ledger.json").get(claim_id)
    lessons = [a["lesson"] for a in entry["attempts"]]
    assert any(ranked[0]["angle"] in l for l in lessons)
    assert any(ranked[1]["angle"] in l for l in lessons)


def test_attempt_queue_standalone_rank_progression(tmp_path):
    queue = AttemptQueue(path=tmp_path / "q.json", spec_dir=tmp_path / "specs")
    ledger = HonestyLedger(tmp_path / "ledger.json")
    f = _formulation()
    hyp = hypothesis_mod.hypothesize(f, {"verdict": "achievable_with_work", "failed": []})
    dossier = {"formulation": f.to_dict(), "gate": {"verdict": "achievable_with_work"},
               "hypothesis": hyp, "reframed": {"reframed_goal": "cool house"},
               "transfer": {"triz": [], "inversion": []},
               "first_principles": {}, "domain": "hardware"}
    claim_id = "testclaim01"
    queue.enqueue(claim_id, "cool house", dossier)
    r1 = queue.attempt(claim_id, dossier, ledger=ledger)
    r2 = queue.attempt(claim_id, dossier, ledger=ledger)
    assert r1["hypothesis"] != r2["hypothesis"]
    assert r2["hypothesis"] == hyp["hypotheses"][1]["angle"]
