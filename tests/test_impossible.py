"""Tests for nomorals/impossible — offline suite stays offline (no live fetch)."""

from __future__ import annotations

import pytest

from nomorals.impossible import ImpossibleMadePossible
from nomorals.impossible import charter as charter_mod
from nomorals.impossible import classify as classify_mod
from nomorals.impossible import feasible as feasible_mod
from nomorals.impossible import first_principles as fp_mod
from nomorals.impossible import formulate as formulate_mod
from nomorals.impossible import human as human_mod
from nomorals.impossible import lattice as lattice_mod
from nomorals.impossible import transfer as transfer_mod
from nomorals.impossible.attempt import AttemptQueue
from nomorals.impossible.digest import digest_tick
from nomorals.impossible.ledger import HonestyLedger


@pytest.fixture()
def imp(tmp_path):
    return ImpossibleMadePossible(ledger_path=tmp_path / "ledger.json",
                                  queue_path=tmp_path / "queue.json",
                                  spec_dir=tmp_path / "specs")


# --- formulate: underspecified -> clarification, not a guess ---------------
def test_underspecified_triggers_clarification():
    f = formulate_mod.formulate("make it better")
    assert f.underspecified is True
    assert f.clarifications


def test_formulate_frames_present():
    f = formulate_mod.formulate("a house in Lagos that stays cool without grid power, under $500")
    assert set(f.frames.keys()) == {"function_not_form", "first_principles", "inversion",
                                    "trivial_move", "analogy", "scale", "user_ache"}


# --- classify ----------------------------------------------------------------
def test_classify_perpetual_motion_physical():
    f = formulate_mod.formulate("a perpetual motion machine that makes free energy")
    c = classify_mod.classify(f)
    assert c["primary"] == "physical"


def test_classify_default_engineering():
    f = formulate_mod.formulate("a cheap roof that lasts 20 years in Lagos humidity")
    c = classify_mod.classify(f)
    assert "engineering" in [k["kind"] for k in c["kinds"]]


# --- transfer ----------------------------------------------------------------
def test_triz_40_complete():
    assert len(transfer_mod.TRIZ_40) == 40
    assert transfer_mod.TRIZ_40[1][0] == "Segmentation"
    assert transfer_mod.TRIZ_40[13][0] == "The Other Way Round"
    assert transfer_mod.TRIZ_40[40][0] == "Composite Materials"


def test_contradiction_matrix_strength_weight():
    contra = transfer_mod.detect_contradiction("make it stronger but lighter and cheaper")
    assert contra is not None
    assert contra["principles"] == [40, 26, 27, 1]


def test_scamper_seven():
    f = formulate_mod.formulate("a cooler that keeps drinks cold")
    moves = transfer_mod.scamper_moves(f)
    assert len(moves) == 7
    assert {m["move"] for m in moves} == set(transfer_mod.SCAMPER.keys())


def test_morphological_combos():
    combos = transfer_mod.morphological_search({"material": ["steel", "aluminium"], "power": ["solar", "grid"]}, limit=3)
    assert len(combos) == 3
    assert combos[0]["combination"] == {"material": "steel", "power": "solar"}


def test_relaxation_convention_vs_physics():
    out = transfer_mod.relax_constraints(["must follow the usual industry standard",
                                          "must obey conservation of energy"])
    by_kind = {o["constraint"]: o for o in out}
    assert by_kind["must follow the usual industry standard"]["relaxable"] is True
    assert by_kind["must obey conservation of energy"]["relaxable"] is False


# --- feasibility gate --------------------------------------------------------
def test_dimensional_balanced_and_unbalanced():
    ok = feasible_mod.dimensional_check("J = kg * m^2 / s^2")
    assert ok["pass"] is True
    bad = feasible_mod.dimensional_check("J = kg * m / s")
    assert bad["pass"] is False


def test_conservation_catches_free_energy():
    r = feasible_mod.conservation_check("free energy machine", energy_in=100, energy_out=150)
    assert r["pass"] is False


def test_carnot_bound():
    r = feasible_mod.thermo_check(hot_k=600, cold_k=300, claimed_efficiency=0.6)
    assert r["pass"] is False  # Carnot = 0.5
    r2 = feasible_mod.thermo_check(hot_k=600, cold_k=300, claimed_efficiency=0.4)
    assert r2["pass"] is True


def test_materials_melt_and_food_safe():
    r = feasible_mod.materials_check("ABS plastic", max_temp_c=200)
    assert r["pass"] is False
    r2 = feasible_mod.materials_check("steel mild", food_contact=True)
    assert r2["pass"] is False  # mild steel not food-safe


def test_cost_over_budget_bounded():
    r = feasible_mod.cost_check(build_usd=1000, yearly_operating_usd=100, lifetime_years=5, budget_usd=500)
    assert r["pass"] is False
    assert r["lifetime_cost"] == 1500


def test_scale_square_cube():
    r = feasible_mod.scale_check(10, "size")
    assert r["area_factor"] == 100
    assert r["volume_factor"] == 1000


def test_gate_excluded_for_perpetual_motion(imp):
    d = imp.submit("a perpetual motion machine that makes free energy",
                   energy_in=100, energy_out=150)
    assert d["gate"]["verdict"] == "excluded"
    # Never refused: still gets a reframed achievable version + falsifier.
    assert d["reframed"]["reframed_goal"]
    assert d["hypothesis"]["falsifier"]


# --- first principles (benchmark) -------------------------------------------
def test_first_principles_order_and_domain():
    fp = fp_mod.decompose("make the search endpoint 10x faster", domain="software")
    assert "correctness" in fp.fundamentals
    assert fp.deletions and fp.simplifications and fp.optimizations
    assert fp.accelerations and fp.automations


def test_first_principles_question_physics_vs_convention():
    q = fp_mod.question_requirement("must follow the usual industry standard")
    assert q["negotiable"] is True
    q2 = fp_mod.question_requirement("must obey the second law of thermodynamics")
    assert q2["kind"] == "physics" and q2["negotiable"] is False


def test_worked_examples_outreason_analogy():
    car = fp_mod.worked_example_car()
    assert "3-4x" in car["first_principles_answer"] or "3-4x" in car["first_principles_answer"].replace("3–4x", "3-4x") or "energy" in car["first_principles_answer"].lower()
    sw = fp_mod.worked_example_software()
    assert "N+1" in sw["first_principles_answer"]
    assert sw["first_principles_answer"] != sw["analogy_answer"]


# --- human-like reasoning ----------------------------------------------------
def test_common_sense_heat_and_liquids():
    rules = human_mod.common_sense("a water tank that gets hot in the sun")
    assert {"heat", "liquids"} <= {r["rule"] for r in rules}


def test_mental_simulation_flags_risks():
    sim = human_mod.mental_simulation("heat the sealed tank")
    assert sim["risks"]
    assert len(sim["chain"]) == 4


def test_plan_backtracking():
    plan = human_mod.plan_with_backtracking("build a cooler", ["cut foam", "seal joints", "test ice"])
    assert len(plan["steps"]) == 3
    assert "Backtrack" in plan["steps"][1]["if_fails"]


# --- ledger / attempt / digest ----------------------------------------------
def test_failed_attempt_changes_next_attempt(imp):
    d = imp.submit("a cheap roof that lasts 20 years in Lagos humidity")
    cid = d["claim_id"]
    imp.ledger.record_attempt(cid, outcome="failed", evidence="coupon test",
                              lesson="Bitumen softens at 60C in direct sun")
    hint = imp.ledger.next_attempt_hint(cid)
    assert "Bitumen" in hint


def test_attempt_hardware_writes_spec(imp, tmp_path):
    d = imp.submit("a house in Lagos that stays cool without grid power")
    result = imp.attempt(d["claim_id"])
    assert result["ok"] is True
    assert result["kind"] == "hardware"
    assert "spec_file" in result
    assert "awaiting delegation" in result["delegation"]


def test_digest_tick_moves_or_cools(imp):
    d = imp.submit("a cooler that keeps drinks cold for 48 hours without power")
    out = digest_tick(queue=imp.queue, ledger=imp.ledger, max_problems=1,
                      prior_art_fn=lambda q: {"results": {}, "summary": "offline stub"})
    assert out["problems_checked"] == 1


def test_lattice_and_charter():
    lat = lattice_mod.lattice()
    assert set(lat["zones"].keys()) == {"physical", "biological", "digital", "social", "economic"}
    ch = charter_mod.charter()
    assert "Nothing is refused" in ch["stance"]


def test_submit_underspecified_returns_clarification(imp):
    d = imp.submit("make it better")
    assert d["status"] == "needs_clarification"
    assert d["clarifications"]


def test_chat_examples_and_charter():
    from nomorals.impossible.chat import control_impossible
    assert "N+1" in control_impossible("/impossible examples")
    assert "Nothing is refused" in control_impossible("/impossible charter")
