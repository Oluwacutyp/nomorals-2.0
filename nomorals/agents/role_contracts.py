"""Role contracts (L5) — docs 22+23, Stage A P2.

Doc 22 §2: *"Role is a contract, not a persona."*  The fields are
the doc's own list — ``role_id, domain, competencies, authority,
inputs, tools, output_schema, quality_rubric, verifier, escalation,
budget, rate/cost model``.

A :class:`RoleContract` is **derived, never authored as a third
definition of a role**:

* :func:`contract_for_manifest` builds one from a specialist
  manifest (``agents/manifests.py`` — completed by this pass with
  the contract terms: authority, output schema, quality rubric,
  escalation, budget) plus the governed role specs it binds
  (``agents/role_specs.py`` — the tools and numeric budgets are the
  specs' own, enforced by ``RoleEnforcingRegistry``);
* :func:`contract_for_role_spec` builds one from a bare role spec.
  A spec declares no competencies, so its contract is honestly
  *partial*: :meth:`RoleContract.validate` names exactly that gap
  instead of the derivation inventing competencies.

Validation is the contract's teeth, and it borrows the shipped
independence law (``agents/delegation.py`` raises
``IndependenceError`` when a verifier would judge their own work):
a contract whose verifier is itself, whose authority word is
unknown, or whose read-only authority is paired with mutating
tools is refused, with every problem named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .manifests import MANIFESTS, SpecialistManifest, manifest
from .role_specs import BUILTIN_ROLES, RoleSpec, is_mutating_tool

__all__ = [
    "AUTHORITIES",
    "ContractBudget",
    "RoleContract",
    "RoleContractError",
    "contract_for_manifest",
    "contract_for_role_spec",
    "contract_for_class",
    "all_contracts",
    "contracts_index",
]


class RoleContractError(ValueError):
    """A role contract failed validation; the message names every
    problem found."""


#: The authority vocabulary.  A contract's authority is exactly one
#: of these words — anything else is refused at validation.
#: ``read_only`` may hold no mutating tool; ``workspace_write`` may
#: change files inside its guarded scope; ``execute`` may run
#: processes and jobs; ``operate`` directs other roles' work.
AUTHORITIES: tuple[str, ...] = (
    "read_only", "workspace_write", "execute", "operate")

#: The domain every shipped manifest class serves today (the
#: manifests are the coding-agent OS taxonomy; other domains gain
#: manifests — and therefore contracts — as their packs land).
MANIFEST_DOMAIN = "software"

#: Who independently checks a class's work, by class.  The default
#: is the critic spec — the shipped adversarial reviewer.  The lead
#: and the quality class itself answer to the contract **arbiter**
#: (in the shipped cells, the owner): the lead's integration verdict
#: is accepted or refused there, and the critic's own verdicts are
#: challenged on the claim ledger, not by another critic.
_VERIFIER_BY_CLASS: dict[str, str] = {
    "lead_engineering": "arbiter",
    "quality_resilience": "arbiter",
}
_DEFAULT_VERIFIER = "critic"


@dataclass(frozen=True)
class ContractBudget:
    """A contract's numeric budget, from the bound role spec where
    one binds.  ``cost_model`` carries the manifest's budget
    statement (the rate/cost model in prose) — internal work is
    unpriced today, so the prose says what bounds it instead."""

    wall_seconds: float = 0.0
    tokens: int = 0
    cost_model: str = ""

    def describe(self) -> str:
        parts = []
        if self.wall_seconds:
            parts.append(f"{int(self.wall_seconds)}s wall")
        if self.tokens:
            parts.append(f"{self.tokens} tokens")
        if self.cost_model:
            parts.append(self.cost_model)
        return "; ".join(parts) or "no budget declared"


@dataclass(frozen=True)
class RoleContract:
    """One role, as a contract (doc 22 §2's field list)."""

    role_id: str
    domain: str = ""
    competencies: tuple[str, ...] = ()
    authority: str = ""
    inputs: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    output_schema: tuple[str, ...] = ()
    quality_rubric: str = ""
    verifier: str = ""
    escalation: str = ""
    budget: ContractBudget = field(default_factory=ContractBudget)
    cost_model: str = ""

    def validate(self) -> list[str]:
        """Every problem keeping this contract from being accepted,
        named.  An empty list accepts the contract as stated."""
        problems: list[str] = []
        if not str(self.role_id or "").strip():
            problems.append("role_id is empty")
        if not str(self.domain or "").strip():
            problems.append("domain is empty")
        if not self.competencies:
            problems.append(
                "competencies are empty — a contract names what "
                "the role can do")
        if self.authority not in AUTHORITIES:
            problems.append(
                f"authority {self.authority!r} is not one of "
                f"{', '.join(AUTHORITIES)}")
        if not self.output_schema:
            problems.append(
                "output_schema is empty — a contract names the "
                "shape of its deliverable")
        if not str(self.quality_rubric or "").strip():
            problems.append("quality_rubric is empty")
        if not str(self.verifier or "").strip():
            problems.append(
                "verifier is empty — every contract names who "
                "checks the work")
        elif self.verifier == self.role_id:
            problems.append(
                "verifier is the role itself — verification must "
                "be independent (delegation's independence law, "
                "as contract law)")
        if not str(self.escalation or "").strip():
            problems.append("escalation is empty")
        if self.authority == "read_only":
            mutating = [name for name in self.tools
                        if is_mutating_tool(name)]
            if mutating:
                problems.append(
                    "authority is read_only but tools include "
                    "mutating tools: " + ", ".join(mutating))
        return problems

    def require_valid(self) -> RoleContract:
        """The contract, or a refusal naming every problem."""
        problems = self.validate()
        if problems:
            raise RoleContractError(
                f"role contract {self.role_id!r} is not acceptable: "
                + "; ".join(problems))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "domain": self.domain,
            "competencies": list(self.competencies),
            "authority": self.authority,
            "inputs": list(self.inputs),
            "tools": list(self.tools),
            "output_schema": list(self.output_schema),
            "quality_rubric": self.quality_rubric,
            "verifier": self.verifier,
            "escalation": self.escalation,
            "budget": {"wall_seconds": self.budget.wall_seconds,
                       "tokens": self.budget.tokens,
                       "cost_model": self.budget.cost_model},
            "cost_model": self.cost_model,
        }


def _bound_specs(m: SpecialistManifest) -> list[RoleSpec]:
    return [BUILTIN_ROLES[name] for name in m.role_specs
            if name in BUILTIN_ROLES]


def contract_for_manifest(m: SpecialistManifest) -> RoleContract:
    """The role contract a specialist class carries.

    Terms come from the manifest's own contract fields; tools and
    the numeric budget come from the governed role specs the
    manifest binds (the enforced surface, not a restatement).
    """
    specs = _bound_specs(m)
    tools: list[str] = []
    for spec in specs:
        for name in spec.tool_allowlist:
            if name not in tools:
                tools.append(name)
    if specs:
        budget = ContractBudget(
            wall_seconds=max(s.budget.wall_seconds for s in specs),
            tokens=max(s.budget.tokens for s in specs),
            cost_model=m.budget)
    else:
        budget = ContractBudget(cost_model=m.budget)
    return RoleContract(
        role_id=m.class_id,
        domain=MANIFEST_DOMAIN,
        competencies=tuple(m.keywords),
        authority=m.authority,
        inputs=("task brief",),
        tools=tuple(tools),
        output_schema=tuple(m.output_schema),
        quality_rubric=m.quality_rubric,
        verifier=_VERIFIER_BY_CLASS.get(m.class_id, _DEFAULT_VERIFIER),
        escalation=m.escalation,
        budget=budget,
        cost_model="",
    )


def contract_for_role_spec(spec: RoleSpec) -> RoleContract:
    """The role contract a bare governed role spec carries.

    Honest partial: a spec declares tools, an output contract and a
    budget, but no competencies — :meth:`RoleContract.validate`
    names that single gap rather than the derivation inventing one.
    """
    return RoleContract(
        role_id=spec.name,
        domain=MANIFEST_DOMAIN,
        competencies=(),
        authority="read_only" if spec.read_only else "workspace_write",
        inputs=("task brief",),
        tools=tuple(spec.tool_allowlist),
        output_schema=tuple(spec.output_contract),
        quality_rubric=(
            "Deliver the output contract exactly; "
            + (spec.description or "the spec's terms hold.")),
        verifier="arbiter" if spec.name == "critic" else "critic",
        escalation="raise to the contract arbiter; never self-approve",
        budget=ContractBudget(
            wall_seconds=spec.budget.wall_seconds,
            tokens=spec.budget.tokens,
            cost_model=""),
        cost_model="",
    )


def contract_for_class(class_id: str) -> RoleContract | None:
    """The contract for one manifest class id, or ``None``."""
    found = manifest(class_id)
    return contract_for_manifest(found) if found is not None else None


def all_contracts() -> list[RoleContract]:
    """Every manifest class's contract, in manifest order."""
    return [contract_for_manifest(m) for m in MANIFESTS]


def contracts_index() -> dict[str, Any]:
    """All contracts as data — the composition root's view."""
    return {contract.role_id: contract.to_dict()
            for contract in all_contracts()}
