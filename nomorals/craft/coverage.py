"""Intake & coverage (L5) — docs 22+23, Stage C P10.

Renders over the cooperative's records, and nothing else:

* **Coverage** — for every domain the system knows (the goals
  registry, the taxonomy seed, the packs' declared domains, the
  role contracts' domains), what the records say: which packs
  exist and in what state, which contracts name the domain and
  what they did, which deliveries are on record — and the domain's
  capability **level, derived** from those records by
  :func:`nomorals.craft.levels.derive_level`.  Absence is named,
  never zeroed or hidden: a domain the seed alone knows reads as
  greenfield, in those words.  The board reads measured state
  only — it stores nothing, and a level changes only when an
  underlying record changes.

* **Intake** — the chat-first hire render: a finished outcome in,
  and out come its decomposition (the composed plan's steps), its
  team (each step's role, governing class and role contract), its
  gate (the shipped consequential rule's verdict per step, with
  reasons), its risks (the governing pack's recorded failure
  modes + the domain's coverage gaps), and its cost (each role
  contract's budget, in the contract's own words).  Every element
  is a render over Stage A–C records; the intake opens nothing
  and claims nothing the records do not carry.

The chat surface's board (``render_board`` — the at-a-glance
page over the same report) and one-field claim page
(``render_capability_status``, over
:func:`nomorals.craft.temporary.capability_status`) live here
too: same records, same derivation, thinner pages.

Contract-to-domain links are read from the records themselves,
three honest ties and no invented ones: a contract's declared
scope names the domain (the CBT pilot's shape), the contract's
specialist class carries a role contract in the domain (the
software substrate's shape), or the class is an occupation of the
seed domain.  A contract with none of these ties names no domain,
and coverage says so by omission rather than by guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..agents.delegation import DelegationStore
from ..agents.role_contracts import all_contracts, contract_for_class
from ..goals.classes import list_classes
from ..skills.filebook import FileSkillLibrary
from ..skills.packs import pack_docs, validate_pack
from .composer import ComposedPlan, step_class
from .levels import LEVEL_INFO, LevelFacts, derive_level
from .seed import SEED_DOMAINS
from .temporary import CompetenceGrantStore, capability_status

__all__ = [
    "DomainCoverage",
    "CoverageReport",
    "coverage_report",
    "render_board",
    "render_capability_status",
    "render_coverage",
    "render_intake",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", str(text or "").lower()).strip("-")


@dataclass
class DomainCoverage:
    """One domain's coverage, read from records: its derived
    level with the earning facts, its packs and contracts and
    deliveries as counts and states (never internal ids), the
    capability claim the records carry, and its named absences."""

    domain: str
    title: str
    known_from: tuple[str, ...] = ()
    level: str | None = None
    basis: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    packs: tuple[dict[str, Any], ...] = ()
    contracts: dict[str, int] = field(default_factory=dict)
    deliveries: int = 0
    claim: dict[str, Any] = field(default_factory=dict)
    absent: dict[str, str] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """The whole board: one row per known domain, plus notes
    about sources that could not be read (named, never hidden)."""

    domains: tuple[DomainCoverage, ...] = ()
    source_notes: tuple[str, ...] = ()

    def row(self, domain: str) -> DomainCoverage | None:
        wanted = _slug(domain)
        for row in self.domains:
            if row.domain == wanted:
                return row
        return None


def _seed_occupation_domains() -> dict[str, set[str]]:
    """Specialist class id → the seed domains whose occupations
    bear that id (the software seed's occupations ARE the
    manifest class ids)."""
    out: dict[str, set[str]] = {}
    for domain in SEED_DOMAINS:
        for occupation in domain["occupations"]:
            out.setdefault(str(occupation["id"]), set()).add(
                str(domain["id"]))
    return out


def _contract_domains(contract: Any,
                      occupation_domains: dict[str, set[str]]
                      ) -> set[str]:
    """The domains one contract's own record names."""
    domains: set[str] = set()
    for entry in contract.scope or ():
        domains.add(_slug(str(entry)))
    class_id = str(contract.specialist_class or "")
    if class_id:
        terms = contract_for_class(class_id)
        if terms is not None and terms.domain:
            domains.add(_slug(terms.domain))
        domains |= {_slug(d) for d in
                    occupation_domains.get(class_id, set())}
    return domains


def coverage_report(db: Any = None, *,
                    library: FileSkillLibrary | None = None,
                    grants: CompetenceGrantStore | None = None,
                    limit: int = 200,
                    now: float | None = None) -> CoverageReport:
    """Compute the coverage board from the records, now.

    ``db`` is the shared Database the record stores live on; with
    no db the board carries the static sources only (registries,
    seed, filebook) and contract/delivery absences are named as
    unread rather than implied.
    """
    library = library if library is not None else FileSkillLibrary()
    notes: list[str] = []
    titles: dict[str, str] = {}
    known_from: dict[str, list[str]] = {}
    order: list[str] = []

    def know(domain: str, title: str, source: str) -> None:
        if domain not in titles:
            titles[domain] = title or domain
            order.append(domain)
        if source not in known_from.setdefault(domain, []):
            known_from[domain].append(source)

    for cls in list_classes():
        know(_slug(cls.id), cls.title, "goals registry")
    for domain in SEED_DOMAINS:
        know(_slug(str(domain["id"])), str(domain["title"]),
             "taxonomy seed")
    for terms in all_contracts():
        if terms.domain:
            know(_slug(terms.domain), "", "role contracts")

    packs_by_domain: dict[str, list[dict[str, Any]]] = {}
    live_knowledge: set[str] = set()
    draft_knowledge: set[str] = set()
    for doc in pack_docs(library):
        domain = _slug(str(doc.domain or ""))
        if not domain:
            continue
        problems = validate_pack(doc)
        packs_by_domain.setdefault(domain, []).append({
            "id": doc.id,
            "status": "draft" if doc.is_draft else "live",
            "valid": not problems,
            "problems": len(problems)})
        know(domain, "", "pack")
        if doc.is_draft:
            draft_knowledge.add(domain)
        else:
            live_knowledge.add(domain)
    for doc in library.docs():
        if doc.is_draft or str(doc.kind or "").strip():
            continue
        domain = _slug(str(doc.domain or ""))
        if domain:
            live_knowledge.add(domain)
            know(domain, "", "skill")

    contracts_by_domain: dict[str, dict[str, int]] = {}
    deliveries_by_domain: dict[str, int] = {}
    produced: set[str] = set()
    tool_executed: set[str] = set()
    operated: set[str] = set()
    if db is not None:
        try:
            store = DelegationStore(db)
            occupation_domains = _seed_occupation_domains()
            delivery_counts: dict[str, int] = {}
            try:
                rows = db.query(
                    "SELECT contract_id, post_ref FROM "
                    "content_receipts")
                for row in rows or []:
                    if str(row["post_ref"] or ""):
                        key = str(row["contract_id"] or "")
                        delivery_counts[key] = \
                            delivery_counts.get(key, 0) + 1
            except Exception as exc:  # noqa: BLE001 — an unreadable
                # source contributes nothing, and says so.  A
                # missing table is not unreadable: the receipts
                # table is runtime-created by its store, so its
                # absence is the plain fact that no delivery was
                # ever recorded on this database.
                if "no such table" not in str(exc):
                    notes.append(
                        f"delivery receipts: unavailable "
                        f"({type(exc).__name__}: {exc})")
            for contract in store.recent_contracts(limit=limit):
                domains = _contract_domains(
                    contract, occupation_domains)
                for domain in domains:
                    know(domain, "", "contract records")
                    counts = contracts_by_domain.setdefault(
                        domain, {"linked": 0, "completed": 0,
                                 "open": 0, "failed": 0})
                    counts["linked"] += 1
                    status = str(contract.status or "")
                    if status in counts:
                        counts[status] += 1
                    claims = store.claims(contract.id, limit=1000)
                    if claims:
                        produced.add(domain)
                    if status == "completed" and any(
                            claim.kind == "evidence"
                            and claim.verdict == "supported"
                            and claim.evidence_ref
                            for claim in claims):
                        tool_executed.add(domain)
                    tied = delivery_counts.get(contract.id, 0)
                    if tied:
                        operated.add(domain)
                        deliveries_by_domain[domain] = \
                            deliveries_by_domain.get(domain, 0) + tied
        except Exception as exc:  # noqa: BLE001 — same discipline
            notes.append(
                f"contracts: unavailable "
                f"({type(exc).__name__}: {exc})")

    rows: list[DomainCoverage] = []
    for domain in order:
        facts = LevelFacts(
            live_knowledge=domain in live_knowledge,
            draft_knowledge=domain in draft_knowledge,
            produced=domain in produced,
            tool_executed=domain in tool_executed,
            operated=domain in operated)
        derived = derive_level(facts)
        absent: dict[str, str] = {}
        packs = tuple(packs_by_domain.get(domain, ()))
        if not packs and domain not in live_knowledge:
            absent["pack"] = ("no pack recorded for this domain — "
                              "its workflows and rubrics are not "
                              "written down anywhere the router "
                              "can load")
        counts = contracts_by_domain.get(
            domain, {"linked": 0, "completed": 0, "open": 0,
                     "failed": 0})
        if db is None:
            absent["contracts"] = ("not read — no record store "
                                   "was given to this board")
        elif not counts["linked"]:
            absent["contracts"] = ("no delegation contract names "
                                   "this domain — no cell has run "
                                   "under it")
        deliveries = deliveries_by_domain.get(domain, 0)
        if not deliveries and db is not None and counts["linked"]:
            absent["deliveries"] = ("no delivery receipt names "
                                    "this domain — work may be "
                                    "done, delivery is not on "
                                    "record")
        row_notes = list(derived.notes)
        if known_from.get(domain) == ["taxonomy seed"]:
            row_notes.append(
                "known to the taxonomy seed only — a place in "
                "the ontology, not a capability: greenfield, no "
                "substrate records stand under it")
        rows.append(DomainCoverage(
            domain=domain, title=titles[domain],
            known_from=tuple(known_from.get(domain, ())),
            level=derived.level, basis=derived.basis,
            notes=tuple(row_notes), packs=packs, contracts=counts,
            deliveries=deliveries,
            claim=capability_status(library, domain, grants=grants,
                                    now=now),
            absent=absent))
    return CoverageReport(domains=tuple(rows),
                          source_notes=tuple(notes))


# ── renders: ordinals and subjects, no internal ids ──────────────


def render_board(report: CoverageReport) -> str:
    """The cooperative board as chat text — the /craft page: every
    known domain with the level its records derive and the facts
    that earn it, its pack states, its open items, and its
    absences named in words.  The full audit detail (source
    notes, the whole contract ledger) is render_coverage's page;
    this is the page the owner reads first."""
    lines = ["craft — the cooperative board (every level below is "
             "derived from records at read; nothing on this page "
             "is assigned)"]
    if report.source_notes:
        for note in report.source_notes:
            lines.append(f"note: {note}")
    if not report.domains:
        lines.append("the board is empty: no domain is known to "
                     "any record — no goals class, no taxonomy "
                     "seed entry, no pack and no role contract "
                     "names one")
        return "\n".join(lines)
    open_total = 0
    for idx, row in enumerate(report.domains, 1):
        if row.level:
            level_line = (f"level {row.level} "
                          f"({LEVEL_INFO[row.level]['name']})")
        else:
            level_line = "no level claimed"
        lines.append(f"{idx}. {row.title} — {level_line}")
        for basis in row.basis:
            lines.append(f"   · {basis}")
        for note in row.notes:
            lines.append(f"   · {note}")
        if row.packs:
            rendered = ", ".join(
                f"{pack['id']} ({pack['status']}"
                + (", valid" if pack["valid"] else
                   f", {pack['problems']} schema problem(s)")
                + ")"
                for pack in row.packs)
            lines.append(f"   packs: {rendered}")
        counts = row.contracts
        if counts.get("linked"):
            open_total += int(counts.get("open", 0))
            lines.append(
                f"   open items: {counts['open']} contract(s) "
                f"still open · {counts['completed']} completed · "
                f"{counts['failed']} failed")
        if row.deliveries:
            lines.append(
                f"   deliveries: {row.deliveries} on record under "
                "grant")
        for name, why in row.absent.items():
            lines.append(f"   absent — {name}: {why}")
    lines.append(
        f"open items across the board: {open_total} contract(s)")
    lines.append("/craft coverage — the full measured detail · "
                 "/craft intake <goal> — a read-only plan preview "
                 "· /craft status <field> — one field's claim")
    return "\n".join(lines)


def render_capability_status(status: dict[str, Any]) -> str:
    """One field's capability claim as chat text — the /craft
    status render over capability_status's derived read: a live
    pack's claim, a temporary grant's claim with its expiry and
    the benchmark citation its basis carries, or the claim unmade
    with its reason.  The dict this renders was itself derived at
    read; no stored standing token is consulted here either."""
    domain = str(status.get("domain") or "")
    lines = [f"craft status — {domain}"]
    if not status.get("claimed"):
        lines.append("no claim stands for this field")
        reason = str(status.get("reason") or "").strip()
        if reason:
            lines.append(f"why: {reason}")
        return "\n".join(lines)
    level = str(status.get("level") or "")
    level_name = LEVEL_INFO.get(level, {}).get("name", "")
    grade = str(status.get("grade") or "")
    pack_id = str(status.get("pack_id") or "")
    if status.get("temporary"):
        expires = status.get("grant_expires_at")
        when = (time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(float(expires)))
                if expires else "an unrecorded time")
        lines.append(
            f"claimed temporarily — level {level} "
            f"({level_name}), grade {grade} (the grade the "
            "level's evidence caps the claim at)")
        lines.append(
            f"the claim stands on a recorded competence grant "
            f"over the draft pack {pack_id}; it lapses at {when} "
            "— after that the field reads unmade unless the pack "
            "has proved itself durable")
    else:
        lines.append(
            f"claimed from a live pack — level {level} "
            f"({level_name}), grade {grade} (the grade the "
            "level's evidence caps the claim at)")
        lines.append(
            f"pack {pack_id} is live in the filebook and carries "
            "this field's routed knowledge")
    for basis in status.get("basis") or ():
        lines.append(f"· {basis}")
    return "\n".join(lines)


def render_coverage(report: CoverageReport) -> str:
    """The coverage board as chat text: one block per domain —
    its derived level and the records that earn it, its packs and
    contract counts, and its absences named in words."""
    lines = ["craft coverage — measured state only"]
    if report.source_notes:
        for note in report.source_notes:
            lines.append(f"note: {note}")
    if not report.domains:
        lines.append("no domains are known to any record")
        return "\n".join(lines)
    for idx, row in enumerate(report.domains, 1):
        if row.level:
            level_line = (f"level {row.level} "
                          f"({LEVEL_INFO[row.level]['name']})")
        else:
            level_line = "no level claimed"
        lines.append(f"{idx}. {row.title} — {level_line}")
        for basis in row.basis:
            lines.append(f"   · {basis}")
        for note in row.notes:
            lines.append(f"   · {note}")
        if row.packs:
            rendered = ", ".join(
                f"{pack['id']} ({pack['status']}"
                + (", valid" if pack["valid"] else
                   f", {pack['problems']} schema problem(s)")
                + ")"
                for pack in row.packs)
            lines.append(f"   packs: {rendered}")
        if row.contracts.get("linked"):
            counts = row.contracts
            lines.append(
                f"   contracts: {counts['linked']} on record · "
                f"{counts['completed']} completed · "
                f"{counts['open']} open · {counts['failed']} "
                "failed")
        if row.deliveries:
            lines.append(
                f"   deliveries: {row.deliveries} on record under "
                "grant")
        for name, why in row.absent.items():
            lines.append(f"   absent — {name}: {why}")
    return "\n".join(lines)


def render_intake(plan: ComposedPlan,
                  row: DomainCoverage | None = None, *,
                  pack: Any = None) -> str:
    """The hire intake for one composed plan: decomposition,
    team, gate, risks and cost — every line a render over the
    plan's and the domain's records."""
    lines = [f"intake — {plan.goal}"]
    preset_line = f"pattern: {plan.preset}"
    if plan.pack_id:
        preset_line += f" · governed by pack {plan.pack_id}"
        if plan.pack_draft:
            preset_line += " (draft — unproven)"
    lines.append(preset_line)
    if row is not None:
        if row.level:
            lines.append(
                f"domain standing: {row.title} at level "
                f"{row.level} ({LEVEL_INFO[row.level]['name']})")
        else:
            lines.append(
                f"domain standing: {row.title} — no level "
                "claimed by its records")
    lines.append("decomposition:")
    for idx, step in enumerate(plan.steps, 1):
        role = str(step.get("role") or "")
        writes = step.get("writes") or []
        line = (f"  {idx}. {step.get('name')} — role {role} → "
                f"class {step_class(step)}")
        if writes:
            line += f" · declares writes: {', '.join(writes)}"
        lines.append(line)
    lines.append("team (role contracts):")
    for step in plan.steps:
        terms = contract_for_class(step_class(step))
        name = str(step.get("name") or "")
        if terms is None:
            lines.append(
                f"  {name}: no role contract recorded for its "
                "class")
            continue
        lines.append(
            f"  {name}: authority {terms.authority} · verifier "
            f"{terms.verifier} · escalation: {terms.escalation}")
    lines.append("gate (the shipped consequential rule):")
    for name, fires, reasons in plan.consequentiality():
        if fires:
            lines.append(
                f"  {name}: a delegation contract opens before "
                f"it runs — {'; '.join(reasons)}")
        else:
            lines.append(
                f"  {name}: no contract — the step meets none of "
                "the shipped signals (self-declared, declared "
                "writes, debate flag, heavy class)")
    lines.append("risks:")
    risks: list[str] = []
    if pack is not None:
        failure_modes = pack.section("failure_modes")
        for raw in failure_modes.splitlines():
            text = raw.strip().lstrip("-* ").strip()
            if text:
                risks.append(f"pack failure mode: {text[:140]}")
            if len(risks) >= 3:
                break
    if plan.pack_draft:
        risks.append(
            "the governing pack is an unproven draft — work "
            "proceeds, and the capability claim waits for its "
            "benchmark or promotion records")
    if row is not None:
        for name, why in row.absent.items():
            risks.append(f"coverage gap ({name}): {why}")
    if not risks:
        risks.append("none recorded — no pack failure modes and "
                     "no coverage gaps on record for this plan")
    for risk in risks:
        lines.append(f"  - {risk}")
    lines.append("cost (role-contract budgets):")
    for step in plan.steps:
        terms = contract_for_class(step_class(step))
        name = str(step.get("name") or "")
        if terms is None:
            lines.append(f"  {name}: no budget recorded")
        else:
            lines.append(
                f"  {name}: {terms.budget.describe()}")
    return "\n".join(lines)
