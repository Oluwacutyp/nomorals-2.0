"""The team composer (L5) — docs 22+23, Stage C P8.

Stage A's graph, role contracts and pack form describe the
cooperative; Stage B's pilots run it where a driver was
hand-written per domain.  This module is the composer the
evaluation names: given a goal, it assembles a **plan** — an
ordered set of plan steps in the exact duck-typed shape the
shipped planner seam consumes — from a named **preset** (doc 22
§3's composition patterns: solo, pipeline, studio, research,
incident) and, where one governs, a craft pack.

The composer owns composition only.  The contract seam is doc
18's G9 — :mod:`nomorals.agents.plan_contracts` — and it is
consumed **as-is**: whether a step earns a contract is decided by
the shipped ``consequential`` rule (self-declared · declared
writes · debate flag · heavy class), a contract is opened through
``open_step_contract`` BEFORE any handler runs, and a finished
step settles through ``settle_step_contract``.  The composer
re-implements none of it, and it routes around none of it:

* a step below the rule gets no contract and no ceremony — the
  skip is recorded with the rule's own shape ("meets none of the
  shipped signals"), exactly the parity the steward's cells keep;
* a step the seam refuses stays refused **in the seam's own
  words**: the opening record carries the refusal verbatim, the
  plan is marked refused, and no later step is opened — the
  composer never falls back to running a step uncontracted and
  never rewords a refusal into a softer one.

A plan composed over a draft pack is labelled unproven (the
filebook's own label): work proceeds over drafts — that is what
drafts are for — and the *claim* discipline lives with the levels
and the temporary-competence machinery, not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..agents import plan_contracts
from ..agents.delegation import DelegationStore
from ..agents.manifests import class_for_role
from ..skills.packs import PackView, validate_pack

__all__ = [
    "ComposerError",
    "PresetStep",
    "Preset",
    "PRESETS",
    "preset_names",
    "get_preset",
    "ComposedPlan",
    "compose_plan",
    "StepOpening",
    "PlanOpening",
    "open_plan",
    "StepSettlement",
    "PlanSettlement",
    "settle_plan",
]


class ComposerError(ValueError):
    """A composition cannot be made: the preset is unknown, the
    pack does not validate, or a step is mis-shaped.  The message
    names the problem — composition never proceeds on a guess."""


# ── presets: doc 22 §3's composition patterns, as data ───────────


@dataclass(frozen=True)
class PresetStep:
    """One step of a preset pattern.

    ``role`` is a planner role name — it resolves to a specialist
    class through the shipped manifests resolver, exactly as a
    planner step's does.  ``produces`` marks the step that carries
    the goal's declared writes (the preset's primary producing
    step — the first one so marked); ``debate`` sets the planner's
    high-stakes flag (meaningful self-verification).
    """

    name: str
    role: str
    produces: bool = False
    debate: bool = False


@dataclass(frozen=True)
class Preset:
    """A named, reusable composition: a shape and its steps."""

    name: str
    title: str
    shape: str
    steps: tuple[PresetStep, ...]


PRESETS: dict[str, Preset] = {
    "solo": Preset(
        name="solo",
        title="Solo",
        shape="one worker carries the goal end-to-end",
        steps=(PresetStep("work", "execution", produces=True),)),
    "pipeline": Preset(
        name="pipeline",
        title="Pipeline",
        shape="research → build → deliver, in order (the content "
              "cell's shape)",
        steps=(PresetStep("research", "research"),
               PresetStep("build", "execution", produces=True),
               PresetStep("deliver", "memory"))),
    "studio": Preset(
        name="studio",
        title="Studio",
        shape="draft → refine → review, for made artifacts",
        steps=(PresetStep("draft", "execution", produces=True),
               PresetStep("refine", "vision"),
               PresetStep("review", "critic", debate=True))),
    "research": Preset(
        name="research",
        title="Research cell",
        shape="scout → synthesize → verdict (the idea cell's "
              "shape)",
        steps=(PresetStep("scout", "research"),
               PresetStep("synthesize", "research", produces=True),
               PresetStep("verdict", "critic", debate=True))),
    "incident": Preset(
        name="incident",
        title="Incident team",
        shape="triage → fix → verify → close-out (the steward's "
              "shape)",
        steps=(PresetStep("triage", "research"),
               PresetStep("fix", "execution", produces=True),
               PresetStep("verify", "critic", debate=True),
               PresetStep("close-out", "memory"))),
}


def preset_names() -> tuple[str, ...]:
    """Every preset the composer knows, in catalogue order."""
    return tuple(PRESETS)


def get_preset(name: str) -> Preset:
    """One preset by name — refused, never guessed, when unknown."""
    preset = PRESETS.get(str(name or "").strip().lower())
    if preset is None:
        raise ComposerError(
            f"unknown preset {name!r} (known: "
            f"{', '.join(preset_names())})")
    return preset


# ── the composed plan ────────────────────────────────────────────


@dataclass
class ComposedPlan:
    """One composed cell: the goal, the pattern it was composed
    from, and its steps in the planner seam's duck-typed shape
    (``name`` / ``goal`` / ``role`` / ``writes`` / ``payload`` —
    the shape ``agents/orchestrator`` builds per task and
    ``agents/plan_contracts`` consumes)."""

    goal: str
    preset: str
    domain: str = ""
    pack_id: str = ""
    pack_draft: bool = False
    steps: tuple[dict[str, Any], ...] = ()

    def step(self, name: str) -> dict[str, Any] | None:
        for step in self.steps:
            if step.get("name") == name:
                return step
        return None

    def consequentiality(self) -> tuple[tuple[str, bool, tuple[str, ...]], ...]:
        """Per step: does the shipped consequential rule fire, and
        its named reasons — the seam's own verdicts, computed by
        the seam's own function."""
        out = []
        for step in self.steps:
            fires, reasons = plan_contracts.step_consequentiality(step)
            out.append((str(step.get("name") or ""), fires,
                        tuple(reasons)))
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal,
                "preset": self.preset,
                "domain": self.domain,
                "pack_id": self.pack_id,
                "pack_draft": self.pack_draft,
                "steps": [dict(step) for step in self.steps]}


def _step_mapping(name: str, role: str, *, goal: str, preset: str,
                  writes: tuple[str, ...], debate: bool,
                  pack_id: str, domain: str,
                  pack_draft: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "goal": goal, "preset": preset, "writes": list(writes),
    }
    if debate:
        payload["debate"] = True
    if pack_id:
        payload["pack"] = pack_id
        payload["pack_status"] = ("draft (unproven)" if pack_draft
                                  else "live")
    if domain:
        payload["domain"] = domain
    return {"name": name, "goal": goal, "role": role,
            "writes": list(writes), "payload": payload}


def _normalize_custom_step(raw: Any, *, goal: str,
                           pack_id: str, domain: str,
                           pack_draft: bool) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ComposerError(
            f"a composed step must be a mapping, got {raw!r}")
    name = str(raw.get("name") or "").strip()
    role = str(raw.get("role") or "").strip()
    if not name or not role:
        raise ComposerError(
            f"a composed step names both a step and a role: {raw!r}")
    payload_in = raw.get("payload")
    payload_in = payload_in if isinstance(payload_in, Mapping) else {}
    writes = raw.get("writes") or payload_in.get("writes") or ()
    step = _step_mapping(
        name, role, goal=str(raw.get("goal") or goal),
        preset="custom",
        writes=tuple(str(w) for w in writes),
        debate=bool(payload_in.get("debate")),
        pack_id=pack_id, domain=domain, pack_draft=pack_draft)
    for key, value in payload_in.items():
        step["payload"].setdefault(key, value)
    return step


def compose_plan(goal: str, *, preset: str = "pipeline",
                 pack: PackView | None = None,
                 domain: str = "",
                 writes: tuple[str, ...] | list[str] = (),
                 steps: Any = None) -> ComposedPlan:
    """Compose a plan for one goal.

    ``preset`` names the composition pattern (or ``steps`` supplies
    explicit step mappings and the preset is recorded as
    ``custom``).  ``pack``, when given, must validate under the
    pack schema — its problems refuse the composition in the
    schema's own words; a *draft* pack composes fine and labels
    the plan unproven.  ``writes`` declares the goal's side
    effects; they ride the preset's primary producing step, where
    the shipped rule reads them.
    """
    goal = str(goal or "").strip()
    if not goal:
        raise ComposerError("a composition needs a goal")
    pack_id = ""
    pack_draft = False
    if pack is not None:
        problems = validate_pack(pack.doc)
        if problems:
            raise ComposerError(
                "the governing pack does not validate: "
                + "; ".join(problems))
        pack_id = pack.doc.id
        pack_draft = bool(pack.doc.is_draft)
        domain = domain or str(pack.doc.domain or "")
    declared = tuple(str(w) for w in writes)
    if steps is not None:
        built = tuple(
            _normalize_custom_step(raw, goal=goal, pack_id=pack_id,
                                   domain=domain,
                                   pack_draft=pack_draft)
            for raw in steps)
        return ComposedPlan(goal=goal, preset="custom",
                            domain=domain, pack_id=pack_id,
                            pack_draft=pack_draft, steps=built)
    chosen = get_preset(preset)
    built_steps: list[dict[str, Any]] = []
    writes_placed = False
    for template in chosen.steps:
        step_writes: tuple[str, ...] = ()
        if template.produces and not writes_placed:
            step_writes = declared
            writes_placed = True
        built_steps.append(_step_mapping(
            template.name, template.role, goal=goal,
            preset=chosen.name, writes=step_writes,
            debate=template.debate, pack_id=pack_id,
            domain=domain, pack_draft=pack_draft))
    return ComposedPlan(goal=goal, preset=chosen.name, domain=domain,
                        pack_id=pack_id, pack_draft=pack_draft,
                        steps=tuple(built_steps))


# ── opening + settling through the shipped seam ──────────────────


@dataclass
class StepOpening:
    """One step's opening record: the contract it earned (by id —
    the record lives in the delegation store), the skip the rule
    decided, or the refusal the seam returned, verbatim."""

    name: str
    contract_id: str = ""
    skipped: str = ""
    refusal: str = ""


@dataclass
class PlanOpening:
    """A plan's opening: every step's record, in plan order."""

    mission_id: str
    steps: tuple[StepOpening, ...] = ()

    @property
    def refused(self) -> bool:
        return any(step.refusal for step in self.steps)

    def opening_for(self, name: str) -> StepOpening | None:
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def contract_ids(self) -> tuple[str, ...]:
        return tuple(step.contract_id for step in self.steps
                     if step.contract_id)


def open_plan(store: DelegationStore, plan: ComposedPlan, *,
              mission_id: str) -> PlanOpening:
    """Open a composed plan's contracts through the G9 seam.

    Every step is judged by the shipped consequential rule; a step
    that meets it gets its contract BEFORE any handler runs, a
    step below it is recorded as skipped under the same rule.  If
    the seam refuses a step, the refusal is recorded **verbatim**,
    the opening is marked refused, and no later step is opened —
    the composer does not route around the seam, soften its words,
    or substitute an uncontracted run.
    """
    openings: list[StepOpening] = []
    refused = False
    for step in plan.steps:
        name = str(step.get("name") or "")
        if refused:
            openings.append(StepOpening(
                name=name,
                skipped="not opened — an earlier step's contract "
                        "was refused by the seam, and the composer "
                        "does not route around a refusal"))
            continue
        fires, _reasons = plan_contracts.step_consequentiality(step)
        if not fires:
            openings.append(StepOpening(
                name=name,
                skipped="below the consequential rule — the step "
                        "meets none of the shipped signals "
                        "(self-declared, declared writes, debate "
                        "flag, heavy class), so it earns no "
                        "contract and no ceremony"))
            continue
        try:
            contract = plan_contracts.open_step_contract(
                store, mission_id=mission_id, step=step)
        except Exception as exc:  # noqa: BLE001 — the seam's refusal
            # is data for the opening record, in its own words
            openings.append(StepOpening(name=name, refusal=str(exc)))
            refused = True
            continue
        openings.append(StepOpening(name=name,
                                    contract_id=contract.id))
    return PlanOpening(mission_id=mission_id, steps=tuple(openings))


@dataclass
class StepSettlement:
    """One step's settlement: the seam's outcome dict, verbatim,
    for a contracted step; ``None`` for a step that never held a
    contract (parity — nothing stamps a step the rule skipped)."""

    name: str
    contract_id: str = ""
    outcome: dict[str, Any] | None = None


@dataclass
class PlanSettlement:
    """A plan's settlement: every step's record, in plan order."""

    mission_id: str
    steps: tuple[StepSettlement, ...] = field(default_factory=tuple)

    @property
    def completed(self) -> bool:
        """True when every contracted step's gate accepted its
        recorded result (skipped steps carry no verdict)."""
        return all(step.outcome and step.outcome.get("completed")
                   for step in self.steps if step.contract_id)


def settle_plan(store: DelegationStore, plan: ComposedPlan,
                opening: PlanOpening, results: Mapping[str, Any], *,
                mission_id: str) -> PlanSettlement:
    """Settle a composed plan's finished steps through the seam.

    ``results`` maps step name → the step's recorded result (the
    work product, as the orchestrator's dispatch receives it).  A
    result that is an exception instance, or a contracted step
    with no recorded result, is recorded on its contract as a
    failure through the seam's ``fail_step_contract``.  Every
    other contracted step flows through ``settle_step_contract``
    and its outcome is carried verbatim — a gate refusal names
    itself in the seam's reasons, and the work product stands.
    """
    settlements: list[StepSettlement] = []
    for step_opening in opening.steps:
        if not step_opening.contract_id:
            settlements.append(StepSettlement(name=step_opening.name))
            continue
        contract = store.get_contract(step_opening.contract_id)
        step = plan.step(step_opening.name) or {}
        if contract is None:
            settlements.append(StepSettlement(
                name=step_opening.name,
                contract_id=step_opening.contract_id,
                outcome={"contract_id": step_opening.contract_id,
                         "completed": False,
                         "reasons": ["the contract record is gone "
                                     "from the delegation store"],
                         "evidence_ref": ""}))
            continue
        if step_opening.name not in results:
            detail = (f"step {step_opening.name!r} recorded no "
                      "result for the settlement")
            plan_contracts.fail_step_contract(
                store, contract, detail=detail)
            settlements.append(StepSettlement(
                name=step_opening.name,
                contract_id=step_opening.contract_id,
                outcome={"contract_id": step_opening.contract_id,
                         "completed": False, "reasons": [detail],
                         "evidence_ref": ""}))
            continue
        result = results[step_opening.name]
        if isinstance(result, BaseException):
            detail = f"step {step_opening.name!r} raised: {result}"
            plan_contracts.fail_step_contract(
                store, contract, detail=detail)
            settlements.append(StepSettlement(
                name=step_opening.name,
                contract_id=step_opening.contract_id,
                outcome={"contract_id": step_opening.contract_id,
                         "completed": False, "reasons": [detail],
                         "evidence_ref": ""}))
            continue
        outcome = plan_contracts.settle_step_contract(
            store, contract, step=step, result=result,
            mission_id=mission_id)
        settlements.append(StepSettlement(
            name=step_opening.name,
            contract_id=step_opening.contract_id, outcome=outcome))
    return PlanSettlement(mission_id=mission_id,
                          steps=tuple(settlements))


def step_class(step: Mapping[str, Any]) -> str:
    """The specialist class a composed step's role resolves to —
    the shipped resolver, named for intake renders."""
    return class_for_role(str(step.get("role") or ""))
