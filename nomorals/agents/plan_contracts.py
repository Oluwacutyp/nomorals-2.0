"""Planner-created delegation contracts (doc 18 G9).

The seam between a planner's plan steps and the shipped delegation
substrate. The rule for WHEN a step earns a contract is the shipped
``consequential`` doctrine, applied — not reinvented:

* the delegation contract itself carries a ``consequential`` field,
  and the WS3 cells set it by hand per call site (idea_development
  True, outcome_steward True, dayrecord False, content_cell by
  parameter);
* the source doc's §2C states the rule in words: "Planner creates
  contracts only for work with side effects, integration risk, data
  reliance/unknown confidence or meaningful self-verification. Tiny
  read-only helper calls don't need bloated six-role ceremony."

This module composes that rule from signals the plan step already
carries, each producing a NAMED reason (see step_consequentiality):

1. the step DECLARES itself consequential (the contract field,
   declared at plan level);
2. the step declares non-empty ``writes`` — side effects (the same
   declaration contract_for_step maps onto the contract's scope);
3. the step's payload carries ``debate`` — the planner's own
   high-stakes flag: meaningful self-verification;
4. the step's role resolves through the specialist manifests to a
   HEAVY class — integration risk (the manifests' heavy flag).

A step meeting none of these executes exactly as it always has: no
contract, no stamp, no overhead. A step meeting any of them gets a
REAL contract — created BEFORE the step executes — and its recorded
result flows through the contract's evidence → skeptic → verifier →
sign-off → completion gate, mirroring the WS3 cells' completion
flows.

Fail-open discipline: contract CREATION failing never stops the
planner. The caller records the failure (core.failures) and executes
the step uncontracted, with the gap stamped onto the step's result
so it is visible on the run record and the ticket — never silent.
Settling never raises into the step's own outcome either: a gate
refusal is reported in the stamp, and the work product stands (the
dayrecord precedent).

This module is deliberately orchestrator-free: it takes duck-typed
step mappings, so the docs 22+23 composer (and any future planner)
reuses the same seam instead of rebuilding it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .delegation import DelegationStore, WorkContract, contract_for_step
from .failures import record_failure

__all__ = [
    "VERIFIER",
    "SKEPTIC",
    "STEWARD",
    "SUPERVISOR",
    "step_consequentiality",
    "open_step_contract",
    "settle_step_contract",
    "fail_step_contract",
    "install_world_sink",
    "clear_world_sink",
]

# -- the world-model feed (2026-11-10) ----------------------------------
# The seam's outcomes are exactly the work-state facts the world model
# (agents/worldmodel.py, Doc 5 §4) was built to hold — its own test
# suite asserts facts beside delegation contracts.  The feed is an
# INJECTED sink, installed by the partner runtime with a persistent
# world: uninjected, this module behaves byte-for-byte as it always
# has (every existing caller and test untouched).  Feeding is
# fail-open by the module's own discipline: a sink that raises is
# recorded and the settlement's outcome is exactly what it would have
# been without the feed — the world model observes the seam, it never
# gates it.

_world_sink: Any = None


def install_world_sink(sink: Any) -> None:
    """Install the world-model sink (a callable taking one event
    mapping).  Called once by the production runtime; tests install
    their own or none."""
    global _world_sink
    _world_sink = sink


def clear_world_sink() -> None:
    """Remove the installed sink (test hygiene / shutdown)."""
    global _world_sink
    _world_sink = None


def _feed_world(event: dict[str, Any]) -> None:
    sink = _world_sink
    if sink is None:
        return
    try:
        sink(event)
    except Exception as exc:  # noqa: BLE001 - the feed never gates the seam
        record_failure("plan_contracts.world_feed", exc)

#: Role holders for planner-created contracts. The worker is derived
#: by contract_for_step from the step's role + specialist class
#: ("<class>:<role>"), so these never collide with it — the substrate
#: refuses a contract whose verifier IS the worker.
VERIFIER = "plan:verifier"
SKEPTIC = "plan:skeptic"
#: The integration owner: files the completion, as the cells' steward
#: completes theirs.
STEWARD = "plan:steward"
#: The recovery supervisor named on failure reports.
SUPERVISOR = "plan:supervisor"


def _payload(step: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = step.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _declared(step: Mapping[str, Any], key: str) -> Any:
    """A step declaration, read from the step or its payload."""
    if key in step:
        return step.get(key)
    return _payload(step).get(key)


def step_consequentiality(step: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Does this plan step meet the shipped consequential rule?

    Returns ``(consequential, reasons)`` — every signal that fired,
    named, so the decision is auditable on the contract and the
    ticket. Composed only from shipped signals; see the module
    docstring for the rule's sources.
    """
    reasons: list[str] = []
    if _declared(step, "consequential"):
        reasons.append("step declares itself consequential")
    writes = _declared(step, "writes") or ()
    if isinstance(writes, (list, tuple)) and len(writes) > 0:
        reasons.append(
            f"step declares writes (side effects): "
            f"{', '.join(str(w) for w in writes)}")
    if _payload(step).get("debate"):
        reasons.append(
            "step is debate-flagged (meaningful self-verification)")
    role = str(step.get("role") or "")
    if role:
        from .manifests import class_for_role, manifest

        spec = manifest(class_for_role(role))
        if spec is not None and spec.heavy:
            reasons.append(
                f"role {role!r} resolves to heavy class "
                f"{spec.class_id!r} (integration risk)")
    return bool(reasons), reasons


def open_step_contract(
    store: DelegationStore, *, mission_id: str, step: Mapping[str, Any],
) -> WorkContract:
    """Create the contract for one consequential plan step.

    Thin over the shipped contract_for_step dispatcher seam with the
    planner's role holders filled in; the contract is consequential
    by construction (it only exists because the rule fired).
    Declarations a producer nested in the step's payload (``writes``)
    are hoisted to the top level contract_for_step reads.
    """
    normalized = dict(step)
    writes = _declared(step, "writes")
    if writes is not None:
        normalized["writes"] = writes
    return contract_for_step(
        store, mission_id=mission_id, step=normalized,
        verifier=VERIFIER, skeptic=SKEPTIC, integration_owner=STEWARD,
        recovery_supervisor=SUPERVISOR, arbiter="owner",
        consequential=True)


def _result_defect(step: Mapping[str, Any], result: Any) -> str:
    """A concrete defect in the step's recorded result, or "".

    Mirrors the cells' skeptic discipline: challenge on a CONCRETE
    defect only — a missing result, or declared writes that produced
    an empty result (the writes did not land).
    """
    if result is None:
        return "the step recorded no result"
    writes = _declared(step, "writes") or ()
    if (isinstance(writes, (list, tuple)) and writes
            and type(result) in (str, dict, list) and len(result) == 0):
        return (f"step declared writes ({', '.join(str(w) for w in writes)})"
                " but its recorded result is empty")
    return ""


def settle_step_contract(
    store: DelegationStore, contract: WorkContract, *,
    step: Mapping[str, Any], result: Any, mission_id: str,
) -> dict[str, Any]:
    """Flow a finished step's recorded result through its contract.

    The WS3 cells' completion flow, over the planner's roles: one
    evidence claim per required-evidence kind (the claim IS the
    step's execution record, ref ``plan-step:<run>/<step>``), the
    verifier verifies each against the recorded result, ONE standing
    skeptic challenge is raised and resolved on the evidence (a
    concrete defect resolves it contradicted), the verifier signs
    off when the checks pass, and the steward completes. A gate
    refusal is returned in the outcome — never raised, never hidden:
    the work product stands and the contract stays open with its
    blockers named.
    """
    step_name = str(step.get("name") or "step")
    ref = f"plan-step:{mission_id}/{step_name}"
    defect = _result_defect(step, result)
    kinds = tuple(contract.required_evidence) or ("execution record",)
    claims = [
        store.submit_evidence(
            contract.id, kind=kind,
            claim=(f"plan step {step_name!r} executed under run "
                   f"{mission_id}; recorded result: "
                   f"{repr(result)[:1000]}"),
            producer=contract.worker, evidence_ref=ref)
        for kind in kinds
    ]
    verdict = "contradicted" if defect else "supported"
    for claim in claims:
        store.verify_claim(claim.id, validator=VERIFIER, verdict=verdict)
    if contract.skeptic:
        challenge = store.challenge(
            contract.id, challenger=SKEPTIC,
            claim=("standing review: does the recorded result carry "
                   "the step's declared work?"),
            target_claim_id=claims[0].id)
        store.resolve_challenge(
            challenge.id, resolver=VERIFIER, verdict=verdict,
            note=defect or "recorded result carries the step's work")
    if not defect:
        store.sign_off(contract.id, verifier=VERIFIER,
                       note=f"plan step {step_name!r} verified")
    decision = store.complete(contract.id, by=STEWARD)
    _feed_world({
        "mission_id": mission_id,
        "step": step_name,
        "goal": contract.goal,
        "worker": contract.worker,
        "outcome": "completed" if decision.ok else "refused",
        "reasons": list(decision.reasons),
    })
    return {
        "contract_id": contract.id,
        "completed": bool(decision.ok),
        "reasons": list(decision.reasons),
        "evidence_ref": ref,
    }


def fail_step_contract(
    store: DelegationStore, contract: WorkContract, *, detail: str,
) -> None:
    """Record a step failure on its contract (the shipped
    report_failure: the contract stays open, the failure is on the
    ledger, a retry cuts a fresh contract)."""
    store.report_failure(contract.id, by=SUPERVISOR, detail=detail)
    _feed_world({
        "fact_key": f"plan-failure:{contract.id}",
        "goal": contract.goal,
        "worker": contract.worker,
        "outcome": "failed",
        "detail": detail,
    })
