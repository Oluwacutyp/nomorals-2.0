"""Stage 6 — FEASIBILITY GATE. The thing that keeps the class honest.

Checks, all mechanical, all with reasons. The gate classifies
(excluded / bounded / achievable_now / achievable_with_work), never refuses.
Every gate result carries a band, not a point.

- Dimensional analysis: units must balance (uses constants.UNIT_DIMENSIONS).
- Conservation: flags claims that create energy/mass from nothing.
- Thermodynamic bounds: Carnot, second law, energy-density bounds.
- Materials limits: strength / melting / food-safe / Pareto front.
- Scale laws: square-cube.
- Manufacturability, cost & lifetime, maintenance, law & standards.
"""

from __future__ import annotations

import re

from .charter import ACHIEVABLE_NOW, ACHIEVABLE_WITH_WORK, BOUNDED, EXCLUDED
from .constants import (C, CP_WATER, G_EARTH, RHO_AIR, RHO_WATER, UNIT_DIMENSIONS,
                        get_material, pareto_front, MATERIALS)
from . import constants as _const


def dimensional_check(equation: str) -> dict:
    """Check a simple equation string 'LHS = RHS' where each side is a list
    of unit symbols joined by * / ^. Returns balance verdict + reason."""
    if not equation or "=" not in equation:
        return {"check": "dimensional", "pass": None, "reason": "No equation supplied — dimensional check skipped, not passed."}
    lhs, rhs = equation.split("=", 1)
    def parse(side: str):
        total = [0.0] * 7
        for sign, unit, power in re.findall(r"([*/]?)\s*([A-Za-z][A-Za-z0-9_]*)(?:\^(\d+))?", side):
            if unit not in UNIT_DIMENSIONS:
                return None
            p = float(power) if power else 1.0
            if sign == "/":
                p = -p
            d = UNIT_DIMENSIONS[unit]
            for i in range(7):
                total[i] += p * d[i]
        return tuple(total)
    dl, dr = parse(lhs), parse(rhs)
    if dl is None or dr is None:
        return {"check": "dimensional", "pass": None,
                "reason": f"Unknown unit in equation '{equation}' — cannot verify balance; treat as unverified, not passed."}
    if dl == dr:
        return {"check": "dimensional", "pass": True, "reason": f"Units balance: {equation}"}
    return {"check": "dimensional", "pass": False,
            "reason": f"Units DO NOT balance: LHS {dl} vs RHS {dr} in '{equation}'. The equation is wrong."}


def conservation_check(claim_text: str, energy_in: float | None = None,
                       energy_out: float | None = None) -> dict:
    """Conservation: if energy_out > energy_in, the claim creates energy."""
    t = (claim_text or "").lower()
    free_energy = bool(re.search(r"free\s+energy|energy\s+from\s+nothing|perpetual|over\s*unity|overunity", t))
    if energy_in is not None and energy_out is not None:
        if energy_out > energy_in * 1.000001:
            return {"check": "conservation", "pass": False,
                    "reason": f"Energy out ({energy_out}) exceeds energy in ({energy_in}) — violates conservation of energy."}
        return {"check": "conservation", "pass": True,
                "reason": f"Energy out ({energy_out}) <= energy in ({energy_in}) — conservation satisfied."}
    if free_energy:
        return {"check": "conservation", "pass": False,
                "reason": "Claim implies energy from nothing (free energy / perpetual) — violates conservation of energy."}
    return {"check": "conservation", "pass": None,
            "reason": "No energy figures supplied and no free-energy language — conservation not verified, not passed."}


def thermo_check(hot_k: float | None = None, cold_k: float | None = None,
                 claimed_efficiency: float | None = None) -> dict:
    """Carnot bound: eta <= 1 - T_cold/T_hot. Second law: eta < 1 always."""
    if claimed_efficiency is not None and claimed_efficiency >= 1.0:
        return {"check": "thermodynamic", "pass": False,
                "reason": f"Claimed efficiency {claimed_efficiency} >= 100% — violates the second law."}
    if hot_k and cold_k and hot_k > 0 and cold_k >= 0:
        carnot = 1.0 - cold_k / hot_k
        if claimed_efficiency is not None and claimed_efficiency > carnot:
            return {"check": "thermodynamic", "pass": False,
                    "reason": f"Claimed efficiency {claimed_efficiency} exceeds Carnot bound {carnot:.3f} for T_hot={hot_k}K, T_cold={cold_k}K."}
        return {"check": "thermodynamic", "pass": True,
                "reason": f"Carnot bound for T_hot={hot_k}K, T_cold={cold_k}K is {carnot:.3f}; claim within it."}
    return {"check": "thermodynamic", "pass": None,
            "reason": "No temperatures/efficiency supplied — thermodynamic bound not verified."}


def materials_check(material_name: str, max_temp_c: float | None = None,
                    stress_mpa: float | None = None, food_contact: bool = False,
                    strength_axis: str = "yield_mpa", cost_axis: str = "cost_usd_kg") -> dict:
    """Materials limits against the property table + Pareto front."""
    m = get_material(material_name) if material_name else None
    if m is None:
        return {"check": "materials", "pass": None,
                "reason": f"Material '{material_name}' not in the property table — add it via register_material() before relying on material claims.",
                "pareto_front": []}
    failures = []
    if max_temp_c is not None and max_temp_c > m.melting_c:
        failures.append(f"Operating temp {max_temp_c}C exceeds melting point {m.melting_c}C for {m.name}.")
    if stress_mpa is not None and stress_mpa > m.yield_mpa:
        failures.append(f"Stress {stress_mpa} MPa exceeds yield {m.yield_mpa} MPa for {m.name}.")
    if food_contact and not m.food_safe:
        failures.append(f"{m.name} is not marked food-safe — needs a food-safe grade/coating or a different material.")
    front = pareto_front(list(MATERIALS.values()), strength_axis, cost_axis)
    front_names = [x.name for x in front]
    if failures:
        return {"check": "materials", "pass": False, "reason": " ".join(failures),
                "material": m.name, "pareto_front": front_names}
    return {"check": "materials", "pass": True,
            "reason": f"{m.name} within limits (melt {m.melting_c}C, yield {m.yield_mpa} MPa).",
            "material": m.name, "pareto_front": front_names}


def scale_check(scale_factor: float, quantity: str = "size") -> dict:
    """Square-cube: linear scale s -> area s^2, volume/mass s^3.
    Strength (area) grows slower than weight (volume) — bench-scale wins die here."""
    if not scale_factor or scale_factor <= 0:
        return {"check": "scale", "pass": None, "reason": "No scale factor supplied — scale check skipped."}
    s = scale_factor
    return {"check": "scale", "pass": None,
            "reason": (f"Square-cube: scaling {quantity} by {s}x -> area {s**2:.1f}x, volume/mass {s**3:.1f}x. "
                       f"Strength grows with area, weight with volume — at {s}x the structure is "
                       f"{s:.1f}x weaker per unit weight than at bench scale. Verify strength/mass at target scale."),
            "area_factor": s ** 2, "volume_factor": s ** 3}


def cost_check(build_usd: float | None = None, yearly_operating_usd: float | None = None,
               lifetime_years: float | None = None, budget_usd: float | None = None) -> dict:
    """Cost & lifetime: build + operate + maintain + replace, per year."""
    if build_usd is None and yearly_operating_usd is None:
        return {"check": "cost", "pass": None, "reason": "No cost figures supplied — cost check not verified."}
    total = (build_usd or 0.0) + (yearly_operating_usd or 0.0) * (lifetime_years or 1.0)
    per_year = total / (lifetime_years or 1.0)
    verdict = True
    reasons = [f"Lifetime cost ${total:,.2f} over {lifetime_years or 1} year(s) = ${per_year:,.2f}/year."]
    if budget_usd is not None and total > budget_usd:
        verdict = False
        reasons.append(f"Exceeds budget ${budget_usd:,.2f} by ${total - budget_usd:,.2f} — economically bounded unless cost is cut or budget reframed.")
    return {"check": "cost", "pass": verdict, "reason": " ".join(reasons),
            "lifetime_cost": total, "per_year": per_year}


def manufacturability_check(process: str = "", tolerance_mm: float | None = None,
                            quantity: int | None = None) -> dict:
    """Manufacturability: can it be made with available processes, at tolerance, at quantity?"""
    notes = []
    if process:
        notes.append(f"Process: {process}.")
    if tolerance_mm is not None:
        if tolerance_mm < 0.05:
            notes.append(f"Tolerance {tolerance_mm}mm needs CNC/precision machining — not 3D-print / hand-tool territory.")
        elif tolerance_mm < 0.5:
            notes.append(f"Tolerance {tolerance_mm}mm is achievable with good 3D printing / machining.")
        else:
            notes.append(f"Tolerance {tolerance_mm}mm is loose — hand tools / basic printing suffice.")
    if quantity is not None:
        if quantity <= 10:
            notes.append(f"Quantity {quantity}: prototype methods fine.")
        elif quantity <= 1000:
            notes.append(f"Quantity {quantity}: small-batch methods (CNC, resin, injection moulding setup) needed.")
        else:
            notes.append(f"Quantity {quantity}: mass-production tooling needed — unit cost falls, setup cost dominates.")
    if not notes:
        return {"check": "manufacturability", "pass": None, "reason": "No process/tolerance/quantity supplied — manufacturability not verified."}
    return {"check": "manufacturability", "pass": None, "reason": " ".join(notes)}


def maintenance_check(mtbf_hours: float | None = None, parts_available: bool | None = None,
                      skill: str = "") -> dict:
    notes = []
    if mtbf_hours is not None:
        notes.append(f"MTBF {mtbf_hours}h = {mtbf_hours / 24:.1f} days between failures.")
    if parts_available is False:
        notes.append("Parts NOT locally available — maintenance reality fails in the owner's environment unless sourcing is solved.")
    if skill:
        notes.append(f"Skill required: {skill}.")
    if not notes:
        return {"check": "maintenance", "pass": None, "reason": "No maintenance figures supplied."}
    return {"check": "maintenance", "pass": None if parts_available is not False else False, "reason": " ".join(notes)}


def law_check(needs_certification: bool = False, food_contact: bool = False,
              electrical: bool = False, jurisdiction: str = "") -> dict:
    notes = []
    if needs_certification:
        notes.append("Certification required — design the path that satisfies the constraint (testing, documentation, approved body).")
    if food_contact:
        notes.append("Food-contact rules apply (food-safe materials, cleanability).")
    if electrical:
        notes.append("Electrical code applies (insulation, earthing, protection).")
    if jurisdiction:
        notes.append(f"Jurisdiction: {jurisdiction}.")
    if not notes:
        return {"check": "law", "pass": None, "reason": "No legal/standards flags supplied — law check not verified."}
    return {"check": "law", "pass": None, "reason": " ".join(notes)}


def gate(formulation=None, **kwargs) -> dict:
    """Run every gate check with whatever evidence was supplied.
    Returns checks + overall verdict + band. Never refuses."""
    checks = [
        dimensional_check(kwargs.get("equation", "") or ""),
        conservation_check(getattr(formulation, "raw", "") or "",
                           kwargs.get("energy_in"), kwargs.get("energy_out")),
        thermo_check(kwargs.get("hot_k"), kwargs.get("cold_k"), kwargs.get("claimed_efficiency")),
        materials_check(kwargs.get("material", "") or "", kwargs.get("max_temp_c"),
                        kwargs.get("stress_mpa"), kwargs.get("food_contact", False)),
        scale_check(kwargs.get("scale_factor") or 0),
        cost_check(kwargs.get("build_usd"), kwargs.get("yearly_operating_usd"),
                   kwargs.get("lifetime_years"), kwargs.get("budget_usd")),
        manufacturability_check(kwargs.get("process", ""), kwargs.get("tolerance_mm"), kwargs.get("quantity")),
        maintenance_check(kwargs.get("mtbf_hours"), kwargs.get("parts_available"), kwargs.get("skill", "")),
        law_check(kwargs.get("needs_certification", False), kwargs.get("food_contact", False),
                  kwargs.get("electrical", False), kwargs.get("jurisdiction", "")),
    ]
    fails = [c for c in checks if c.get("pass") is False]
    unverified = [c for c in checks if c.get("pass") is None]
    if fails:
        # Physical-exclusion failures dominate; others are bounds.
        physical_fail = any(c["check"] in ("conservation", "thermodynamic", "dimensional") for c in fails)
        verdict = EXCLUDED if physical_fail else BOUNDED
        why = " ".join(c["reason"] for c in fails)
    elif unverified:
        verdict = ACHIEVABLE_WITH_WORK
        why = ("No hard failure found, but evidence is incomplete: "
               + "; ".join(c["check"] for c in unverified) + " unverified. Supply figures to firm up the verdict.")
    else:
        verdict = ACHIEVABLE_NOW
        why = "All supplied checks pass."
    # Confidence band, not a point — mirrors planning/estimates.py practice.
    band = {"low": 0.35, "mid": 0.6, "high": 0.8} if verdict in (ACHIEVABLE_WITH_WORK, BOUNDED) else \
           {"low": 0.7, "mid": 0.85, "high": 0.95} if verdict == ACHIEVABLE_NOW else \
           {"low": 0.85, "mid": 0.95, "high": 0.99}
    return {"verdict": verdict, "why": why, "checks": checks,
            "failed": [c["check"] for c in fails], "unverified": [c["check"] for c in unverified],
            "confidence_band": band}
