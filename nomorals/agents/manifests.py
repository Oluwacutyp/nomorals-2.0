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
    """One specialist class, described over the existing roster."""

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
       heavy: bool = False) -> SpecialistManifest:
    return SpecialistManifest(
        class_id=class_id, title=title, responsibilities=responsibilities,
        example_subagents=tuple(examples), keywords=tuple(keywords),
        subagents=tuple(subagents), agent_roles=tuple(roles),
        role_specs=tuple(specs), evidence_duties=tuple(evidence),
        heavy=heavy)


MANIFESTS: tuple[SpecialistManifest, ...] = (
    _m("lead_engineering", "Lead engineering",
       "Decomposition, sequencing, integration, acceptance and recovery.",
       ("chief architect/planner", "technical product manager",
        "release captain", "incident commander"),
       ("plan", "decompose", "coordinate", "integrate", "release",
        "recover", "lead"),
       subagents=("planner",), specs=("architect",),
       evidence=("plan record", "integration verdict")),
    _m("product_requirements", "Product & requirements",
       "User intent, domain research, acceptance criteria and scope control.",
       ("research analyst", "specification writer", "workflow mapper"),
       ("requirements", "research", "spec", "scope", "acceptance",
        "user intent"),
       roles=("research",), specs=("researcher",), subagents=("planner",),
       evidence=("source citations", "acceptance criteria")),
    _m("architecture", "Architecture",
       "Boundaries, data flow, interfaces, technology choice and "
       "long-term change paths.",
       ("API architect", "data architect", "distributed-systems designer"),
       ("architecture", "api", "interface", "boundary", "design",
        "data flow", "schema"),
       subagents=("api_designer", "planner"), specs=("architect",),
       evidence=("interface contract", "decision record")),
    _m("ux_interaction", "UX & interaction",
       "Flows, states, commands, errors, accessibility and usability evidence.",
       ("dialogue designer", "interaction prototyper", "accessibility "
        "auditor"),
       ("ux", "flow", "interaction", "usability", "accessibility",
        "dialogue", "states"),
       roles=("vision",), subagents=("reviewer",),
       evidence=("flow walkthrough", "state coverage")),
    _m("visual_ui_design", "Visual & UI design",
       "Layout, typography, visual systems, design tokens and interface polish.",
       ("design-system builder", "typography specialist", "motion designer"),
       ("visual", "layout", "typography", "design", "ui", "theme",
        "polish"),
       roles=("vision",), subagents=("reviewer",),
       evidence=("rendered screenshot", "token diff")),
    _m("frontend", "Frontend",
       "Web/native interfaces, client state, rendering, forms and media display.",
       ("React specialist", "canvas/WebGL UI engineer", "form and "
        "validation engineer"),
       ("frontend", "ui code", "client", "render", "form", "page",
        "component"),
       subagents=("implementer",), specs=("coder",), roles=("coding",),
       evidence=("build output", "rendered screenshot")),
    _m("backend", "Backend",
       "Services, business rules, jobs, APIs, realtime systems and integrations.",
       ("service engineer", "queue/worker engineer", "integration engineer"),
       ("backend", "service", "api", "server", "queue", "worker",
        "integration", "endpoint"),
       subagents=("implementer", "api_designer"), specs=("coder",),
       roles=("coding", "execution"),
       evidence=("test run", "api transcript")),
    _m("data_storage", "Data & storage",
       "Schemas, migrations, indexing, caching, backups and data integrity.",
       ("database engineer", "search/index engineer", "backup and "
        "recovery engineer"),
       ("database", "schema", "storage", "migration", "index", "cache",
        "backup", "data"),
       roles=("data_collection", "memory"), specs=("coder",),
       evidence=("schema diff", "integrity check")),
    _m("platform_delivery", "Platform & delivery",
       "Builds, packaging, CI/CD, environments, observability and rollback.",
       ("build engineer", "release automation", "SRE and observability"),
       ("build", "deploy", "ci", "release", "packaging", "rollback",
        "environment"),
       subagents=("dep_hunter",), roles=("execution",), heavy=True,
       evidence=("build log", "deploy receipt")),
    _m("desktop_mobile_embedded", "Desktop, mobile & embedded",
       "Native apps, device APIs, permissions, offline behaviour and "
       "hardware interfaces.",
       ("Android specialist", "desktop app engineer", "firmware engineer"),
       ("android", "mobile", "desktop", "native", "device", "offline",
        "embedded"),
       roles=("execution",), subagents=("implementer",),
       evidence=("device run log", "install receipt")),
    _m("graphics_3d", "Graphics, simulation & 3D",
       "Geometry, materials, rendering, animation, physics and generated assets.",
       ("Three.js/WebGPU engineer", "Blender automation", "shader "
        "specialist", "game systems engineer"),
       ("3d", "graphics", "render", "blender", "shader", "animation",
        "game", "asset"),
       roles=("vision",), subagents=("implementer",), heavy=True,
       evidence=("rendered frame", "asset manifest")),
    _m("media_audio", "Media & audio",
       "Image, video and audio generation/editing, transcription and "
       "provenance.",
       ("image pipeline", "video editor", "speech and music pipeline"),
       ("media", "image", "video", "audio", "music", "transcription",
        "voice"),
       roles=("social",), subagents=("implementer",),
       evidence=("artifact file", "provenance record")),
    _m("ai_automation", "AI & automation",
       "Model/tool orchestration, retrieval, evaluations, workflow "
       "automation and agent systems.",
       ("LLM workflow engineer", "retrieval engineer", "evaluation "
        "engineer", "browser automation"),
       ("ai", "model", "llm", "agent", "automation", "retrieval",
        "evaluation", "prompt"),
       roles=("training", "research"), specs=("researcher",),
       evidence=("evaluation run", "trace sample")),
    _m("quality_resilience", "Quality & resilience",
       "Testing, fuzzing, performance, failure injection and acceptance "
       "verification.",
       ("test engineer", "performance engineer", "chaos/recovery "
        "tester", "visual QA"),
       ("test", "qa", "quality", "performance", "fuzz", "verify",
        "acceptance", "regression"),
       subagents=("tester", "reviewer"), specs=("tester", "critic"),
       roles=("critic",),
       evidence=("test run", "failure reproduction")),
    _m("security_privacy", "Security & privacy",
       "Threat models, auth, secrets, permissions, abuse resistance and "
       "data handling.",
       ("application security", "identity and access", "privacy "
        "engineer", "dependency auditor"),
       ("security", "auth", "secret", "permission", "privacy", "abuse",
        "vulnerability", "audit"),
       subagents=("reviewer",), specs=("critic",), roles=("critic",),
       evidence=("threat note", "audit result")),
    _m("documentation_support", "Documentation & support",
       "User guides, developer docs, release notes, diagnostics and "
       "knowledge transfer.",
       ("technical writer", "docs information architect", "support "
        "diagnostician"),
       ("docs", "documentation", "guide", "release notes", "support",
        "explain"),
       specs=("writer",), roles=("memory", "reflection"),
       evidence=("published doc", "changelog entry")),
    _m("integration_maintenance", "Integration & maintenance",
       "Cross-specialist assembly, dependency upgrades, refactoring and "
       "compatibility.",
       ("integration owner", "refactoring specialist", "dependency "
        "steward"),
       ("integrate", "integration", "refactor", "upgrade", "dependency",
        "compatibility", "maintenance"),
       subagents=("refactorer",), specs=("architect",),
       roles=("reflection",),
       evidence=("integration test run", "compatibility note")),
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


def manifest_index() -> dict[str, Any]:
    """The full index, JSON-shaped for the composition root."""
    return {m.class_id: {"title": m.title,
                         "responsibilities": m.responsibilities,
                         "bindings": m.bindings(),
                         "evidence_duties": list(m.evidence_duties),
                         "heavy": m.heavy}
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
            "execution_location": execution_location(
                m, profile=profile, remote_available=remote_available),
        })
    return plan
