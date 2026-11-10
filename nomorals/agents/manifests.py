"""Specialist capability manifests (L5) — the coding-agent OS taxonomy.

Doc 5's seventeen specialist classes (Lead + sixteen) as **manifests
over the roster that already exists**: the typed subagent roster
(``agents.subagents.ROSTER``), the mission-step agent roles
(``agents.roles.ROLES``) and the governed role specs
(``agents.role_specs``).  A manifest describes a class — what it owns,
which real workers can fill it, what evidence its work must carry —
without rewriting a single agent.

Two consumers:

* :func:`select_specialists` — the dispatcher seam: given a task text
  (a mission step, a decomposition), rank the classes and resolve each
  to its real bindings plus an **execution location** for the profile
  (heavy classes declare honest remote/deferred placement on small
  devices instead of pretending to run locally).
* :func:`plan_for_steps` — maps mission plan steps (by role) to the
  manifest class that governs them, which the accountable-delegation
  layer uses when it cuts work contracts for a mission.

A consistency test asserts every binding names a worker that actually
exists in the roster it points at — a manifest can never drift into
describing an imaginary workforce.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SpecialistManifest",
    "MANIFESTS",
    "manifest",
    "manifest_index",
    "select_specialists",
    "plan_for_steps",
    "class_for_role",
    "HEAVY_CLASSES",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stemmed(text: str) -> set[str]:
    """Token set with light plural normalization (tests → test), so a
    task written in natural plurals still matches keyword stems."""
    out: set[str] = set()
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        out.add(token)
        if len(token) > 3 and token.endswith("s") \
                and not token.endswith("ss"):
            out.add(token[:-1])
    return out


@dataclass(frozen=True)
class SpecialistManifest:
    """One specialist class, described over the existing roster.

    The last five fields are the class's **role-contract terms**
    (docs 22+23 Stage A P2; doc 22 §2: a role is a contract, not a
    persona).  :mod:`nomorals.agents.role_contracts` derives the
    enforceable :class:`RoleContract` from them; the persona strings
    in ``role_specs``/``subagents`` render *from* contract terms,
    they do not define the class independently.
    """

    class_id: str
    title: str
    responsibilities: str
    example_subagents: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    subagents: tuple[str, ...] = ()      # agents.subagents.ROSTER keys
    agent_roles: tuple[str, ...] = ()    # agents.roles.ROLES keys
    role_specs: tuple[str, ...] = ()     # agents.role_specs names
    evidence_duties: tuple[str, ...] = ()
    heavy: bool = False                  # needs a real workstation/GPU
    # Role-contract terms (agents.role_contracts).  ``authority`` is
    # one of role_contracts.AUTHORITIES' words (mirrored here as data;
    # the Stage A suite pins every value used below to that
    # vocabulary).  ``output_schema`` defaults to the evidence duties
    # — the class's deliverable must carry exactly that evidence.
    authority: str = ""
    output_schema: tuple[str, ...] = ()
    quality_rubric: str = ""
    escalation: str = ""
    budget: str = ""

    def bindings(self) -> dict[str, list[str]]:
        return {"subagents": list(self.subagents),
                "agent_roles": list(self.agent_roles),
                "role_specs": list(self.role_specs)}


#: Classes whose honest execution location is a workstation-class
#: worker; on a constrained profile they resolve to a remote worker or
#: an explicit deferral — never a silent local attempt.
HEAVY_CLASSES = frozenset({"graphics_3d", "platform_delivery"})


def _m(class_id: str, title: str, responsibilities: str,
       examples: Sequence[str], keywords: Sequence[str], *,
       subagents: Sequence[str] = (), roles: Sequence[str] = (),
       specs: Sequence[str] = (),
       evidence: Sequence[str] = (),
       heavy: bool = False,
       authority: str = "", output_schema: Sequence[str] = (),
       rubric: str = "", escalation: str = "",
       budget: str = "") -> SpecialistManifest:
    duties = tuple(evidence)
    return SpecialistManifest(
        class_id=class_id, title=title, responsibilities=responsibilities,
        example_subagents=tuple(examples), keywords=tuple(keywords),
        subagents=tuple(subagents), agent_roles=tuple(roles),
        role_specs=tuple(specs), evidence_duties=duties,
        heavy=heavy, authority=authority,
        output_schema=tuple(output_schema) or duties,
        quality_rubric=rubric or (
            "Accepted only when the work carries its evidence "
            "duties: " + ", ".join(duties) + "."),
        escalation=escalation, budget=budget)


MANIFESTS: tuple[SpecialistManifest, ...] = (
    _m("lead_engineering", "Lead engineering",
       "Decomposition, sequencing, integration, acceptance and recovery.",
       ("chief architect/planner", "technical product manager",
        "release captain", "incident commander"),
       ("plan", "decompose", "coordinate", "integrate", "release",
        "recover", "lead"),
       subagents=("planner",), specs=("architect",),
       evidence=("plan record", "integration verdict"),
       authority="operate",
       escalation="to the owner as arbiter when a step names work "
                  "no bound role can perform",
       budget="the mission's step budget, allocated per step"),
    _m("product_requirements", "Product & requirements",
       "User intent, domain research, acceptance criteria and scope control.",
       ("research analyst", "specification writer", "workflow mapper"),
       ("requirements", "research", "spec", "scope", "acceptance",
        "user intent"),
       roles=("research",), specs=("researcher",), subagents=("planner",),
       evidence=("source citations", "acceptance criteria"),
       authority="read_only",
       output_schema=("findings", "sources", "confidence",
                      "open_questions"),
       escalation="to lead engineering when acceptance criteria "
                  "cannot be grounded in a citable source",
       budget="600s wall / 200000 tokens (researcher spec)"),
    _m("architecture", "Architecture",
       "Boundaries, data flow, interfaces, technology choice and "
       "long-term change paths.",
       ("API architect", "data architect", "distributed-systems designer"),
       ("architecture", "api", "interface", "boundary", "design",
        "data flow", "schema"),
       subagents=("api_designer", "planner"), specs=("architect",),
       evidence=("interface contract", "decision record"),
       authority="read_only",
       output_schema=("steps", "risks", "role_assignments"),
       escalation="to lead engineering when a decomposition names a "
                  "step no role can perform",
       budget="600s wall / 200000 tokens (architect spec)"),
    _m("ux_interaction", "UX & interaction",
       "Flows, states, commands, errors, accessibility and usability evidence.",
       ("dialogue designer", "interaction prototyper", "accessibility "
        "auditor"),
       ("ux", "flow", "interaction", "usability", "accessibility",
        "dialogue", "states"),
       roles=("vision",), subagents=("reviewer",),
       evidence=("flow walkthrough", "state coverage"),
       authority="read_only",
       escalation="to lead engineering with the failing flow named",
       budget="per-task brief; no standing budget"),
    _m("visual_ui_design", "Visual & UI design",
       "Layout, typography, visual systems, design tokens and interface polish.",
       ("design-system builder", "typography specialist", "motion designer"),
       ("visual", "layout", "typography", "design", "ui", "theme",
        "polish"),
       roles=("vision",), subagents=("reviewer",),
       evidence=("rendered screenshot", "token diff"),
       authority="read_only",
       escalation="to lead engineering when a design cannot be "
                  "rendered for evidence",
       budget="per-task brief; no standing budget"),
    _m("frontend", "Frontend",
       "Web/native interfaces, client state, rendering, forms and media display.",
       ("React specialist", "canvas/WebGL UI engineer", "form and "
        "validation engineer"),
       ("frontend", "ui code", "client", "render", "form", "page",
        "component"),
       subagents=("implementer",), specs=("coder",), roles=("coding",),
       evidence=("build output", "rendered screenshot"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when the build stays red "
                  "after two repair attempts",
       budget="1200s wall / 500000 tokens (coder spec)"),
    _m("backend", "Backend",
       "Services, business rules, jobs, APIs, realtime systems and integrations.",
       ("service engineer", "queue/worker engineer", "integration engineer"),
       ("backend", "service", "api", "server", "queue", "worker",
        "integration", "endpoint"),
       subagents=("implementer", "api_designer"), specs=("coder",),
       roles=("coding", "execution"),
       evidence=("test run", "api transcript"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when the test run stays red "
                  "after two repair attempts",
       budget="1200s wall / 500000 tokens (coder spec)"),
    _m("data_storage", "Data & storage",
       "Schemas, migrations, indexing, caching, backups and data integrity.",
       ("database engineer", "search/index engineer", "backup and "
        "recovery engineer"),
       ("database", "schema", "storage", "migration", "index", "cache",
        "backup", "data"),
       roles=("data_collection", "memory"), specs=("coder",),
       evidence=("schema diff", "integrity check"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when an integrity check "
                  "fails — never forward over a corrupt store",
       budget="1200s wall / 500000 tokens (coder spec)"),
    _m("platform_delivery", "Platform & delivery",
       "Builds, packaging, CI/CD, environments, observability and rollback.",
       ("build engineer", "release automation", "SRE and observability"),
       ("build", "deploy", "ci", "release", "packaging", "rollback",
        "environment"),
       subagents=("dep_hunter",), roles=("execution",), heavy=True,
       evidence=("build log", "deploy receipt"),
       authority="execute",
       escalation="to lead engineering on a failed deploy; roll "
                  "back before any retry",
       budget="workstation job budget; heavy class — remote worker "
              "or explicit deferral on constrained profiles"),
    _m("desktop_mobile_embedded", "Desktop, mobile & embedded",
       "Native apps, device APIs, permissions, offline behaviour and "
       "hardware interfaces.",
       ("Android specialist", "desktop app engineer", "firmware engineer"),
       ("android", "mobile", "desktop", "native", "device", "offline",
        "embedded"),
       roles=("execution",), subagents=("implementer",),
       evidence=("device run log", "install receipt"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when no bound device will "
                  "run the build",
       budget="per-task brief; device runs bounded by the device "
              "profile"),
    _m("graphics_3d", "Graphics, simulation & 3D",
       "Geometry, materials, rendering, animation, physics and generated assets.",
       ("Three.js/WebGPU engineer", "Blender automation", "shader "
        "specialist", "game systems engineer"),
       ("3d", "graphics", "render", "blender", "shader", "animation",
        "game", "asset"),
       roles=("vision",), subagents=("implementer",), heavy=True,
       evidence=("rendered frame", "asset manifest"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when a frame cannot be "
                  "rendered for evidence",
       budget="workstation job budget; heavy class — remote worker "
              "or explicit deferral on constrained profiles"),
    _m("media_audio", "Media & audio",
       "Image, video and audio generation/editing, transcription and "
       "provenance.",
       ("image pipeline", "video editor", "speech and music pipeline"),
       ("media", "image", "video", "audio", "music", "transcription",
        "voice"),
       roles=("social",), subagents=("implementer",),
       evidence=("artifact file", "provenance record"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when a render cannot be "
                  "produced by any bound worker",
       budget="per-task brief; renders bounded by the media "
              "profile"),
    _m("ai_automation", "AI & automation",
       "Model/tool orchestration, retrieval, evaluations, workflow "
       "automation and agent systems.",
       ("LLM workflow engineer", "retrieval engineer", "evaluation "
        "engineer", "browser automation"),
       ("ai", "model", "llm", "agent", "automation", "retrieval",
        "evaluation", "prompt"),
       roles=("training", "research"), specs=("researcher",),
       evidence=("evaluation run", "trace sample"),
       authority="read_only",
       output_schema=("findings", "sources", "confidence",
                      "open_questions"),
       escalation="implementation lands through the coding "
                  "classes; this class evaluates and specifies",
       budget="600s wall / 200000 tokens (researcher spec)"),
    _m("quality_resilience", "Quality & resilience",
       "Testing, fuzzing, performance, failure injection and acceptance "
       "verification.",
       ("test engineer", "performance engineer", "chaos/recovery "
        "tester", "visual QA"),
       ("test", "qa", "quality", "performance", "fuzz", "verify",
        "acceptance", "regression"),
       subagents=("tester", "reviewer"), specs=("tester", "critic"),
       roles=("critic",),
       evidence=("test run", "failure reproduction"),
       authority="execute",
       escalation="a failing acceptance with no reproduction "
                  "escalates to lead engineering as unverified, "
                  "never as passed",
       budget="900s wall / 300000 tokens (tester spec)"),
    _m("security_privacy", "Security & privacy",
       "Threat models, auth, secrets, permissions, abuse resistance and "
       "data handling.",
       ("application security", "identity and access", "privacy "
        "engineer", "dependency auditor"),
       ("security", "auth", "secret", "permission", "privacy", "abuse",
        "vulnerability", "audit"),
       subagents=("reviewer",), specs=("critic",), roles=("critic",),
       evidence=("threat note", "audit result"),
       authority="read_only",
       output_schema=("verdict", "issues", "score"),
       escalation="a critical finding escalates to the owner "
                  "immediately, not at task end",
       budget="600s wall / 200000 tokens (critic spec)"),
    _m("documentation_support", "Documentation & support",
       "User guides, developer docs, release notes, diagnostics and "
       "knowledge transfer.",
       ("technical writer", "docs information architect", "support "
        "diagnostician"),
       ("docs", "documentation", "guide", "release notes", "support",
        "explain"),
       specs=("writer",), roles=("memory", "reflection"),
       evidence=("published doc", "changelog entry"),
       authority="workspace_write",
       output_schema=("files_changed", "notes"),
       escalation="to lead engineering when the source material "
                  "contradicts itself",
       budget="600s wall / 200000 tokens (writer spec)"),
    _m("integration_maintenance", "Integration & maintenance",
       "Cross-specialist assembly, dependency upgrades, refactoring and "
       "compatibility.",
       ("integration owner", "refactoring specialist", "dependency "
        "steward"),
       ("integrate", "integration", "refactor", "upgrade", "dependency",
        "compatibility", "maintenance"),
       subagents=("refactorer",), specs=("architect",),
       roles=("reflection",),
       evidence=("integration test run", "compatibility note"),
       authority="workspace_write",
       output_schema=("files_changed", "tests_run", "tests_passed",
                      "notes"),
       escalation="to lead engineering when an upgrade breaks a "
                  "consumer outside the change set",
       budget="per-task brief; bounded by the mission step "
              "budget"),
)

_BY_ID = {m.class_id: m for m in MANIFESTS}

# Mission-step agent role → governing specialist class.  Roles are the
# agents.roles.ROLES keys; every role in the catalog resolves.
_ROLE_TO_CLASS = {
    "research": "product_requirements",
    "coding": "backend",
    "vision": "visual_ui_design",
    "data_collection": "data_storage",
    "training": "ai_automation",
    "social": "media_audio",
    "execution": "backend",
    "memory": "documentation_support",
    "critic": "quality_resilience",
    "reflection": "integration_maintenance",
}


def manifest(class_id: str) -> SpecialistManifest | None:
    return _BY_ID.get(class_id)


def _contract_terms(m: SpecialistManifest) -> dict[str, Any]:
    """The class's role-contract terms, as data (Stage A P2)."""
    return {"authority": m.authority,
            "output_schema": list(m.output_schema),
            "quality_rubric": m.quality_rubric,
            "escalation": m.escalation,
            "budget": m.budget}


def manifest_index() -> dict[str, Any]:
    """The full index, JSON-shaped for the composition root."""
    return {m.class_id: {"title": m.title,
                         "responsibilities": m.responsibilities,
                         "bindings": m.bindings(),
                         "evidence_duties": list(m.evidence_duties),
                         "heavy": m.heavy,
                         "contract": _contract_terms(m)}
            for m in MANIFESTS}


def class_for_role(role: str) -> str:
    """The specialist class governing a mission-step role."""
    return _ROLE_TO_CLASS.get(str(role or "").strip().lower(),
                              "lead_engineering")


def execution_location(m: SpecialistManifest, *, profile: str = "",
                       remote_available: bool = False) -> str:
    """Where this class honestly runs under a device profile.

    Doc 5 project F: the plan must declare the actual execution
    location.  Heavy classes on a constrained profile go to a remote
    worker when one is available and are an explicit deferral when not
    — never a silent local attempt.
    """
    constrained = str(profile or "").lower() in ("termux", "phone",
                                                 "constrained")
    if m.heavy and constrained:
        return "remote-worker" if remote_available else "deferred"
    return "local"


def select_specialists(text: str, *, limit: int = 4, profile: str = "",
                       remote_available: bool = False) -> list[dict]:
    """Rank specialist classes for a task text (the dispatcher seam).

    Deterministic keyword scoring over each manifest's title, keywords
    and responsibilities; ties break by class id.  Each result carries
    the real roster bindings and the declared execution location.
    """
    tokens = _stemmed(text)
    scored: list[tuple[float, SpecialistManifest]] = []
    for m in MANIFESTS:
        score = 0.0
        for keyword in m.keywords:
            parts = _stemmed(keyword)
            if parts and parts <= tokens:
                score += 3.0 if len(parts) > 1 else 2.0
        score += 1.0 * len(tokens & _stemmed(m.title))
        score += 0.5 * len(tokens & _stemmed(m.responsibilities))
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda pair: (-pair[0], pair[1].class_id))
    return [{"class_id": m.class_id, "title": m.title, "score": score,
             "bindings": m.bindings(),
             "evidence_duties": list(m.evidence_duties),
             "contract": _contract_terms(m),
             "execution_location": execution_location(
                 m, profile=profile, remote_available=remote_available)}
            for score, m in scored[:max(0, limit)]]


def plan_for_steps(steps: Iterable[Mapping[str, Any]], *,
                   profile: str = "",
                   remote_available: bool = False) -> list[dict]:
    """Map mission plan steps to their governing specialist classes."""
    plan = []
    for step in steps:
        role = str(step.get("role") or "")
        class_id = class_for_role(role)
        m = _BY_ID[class_id]
        plan.append({
            "step": str(step.get("name") or "step"), "role": role,
            "class_id": class_id, "title": m.title,
            "bindings": m.bindings(),
            "evidence_duties": list(m.evidence_duties),
            "contract": _contract_terms(m),
            "execution_location": execution_location(
                m, profile=profile, remote_available=remote_available),
        })
    return plan
