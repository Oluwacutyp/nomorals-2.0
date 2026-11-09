"""Per-layer instruction files (L4) — the readable law beside the test.

``tests/test_layering.py`` enforces the import graph.  This module is
its human/agent-readable twin: for each architectural layer, one
instruction file under ``nomorals/skills/layers/`` declaring what the
layer owns and what it may not do.

Drift cannot hide: :data:`LAYER_GROUPS` mirrors the test's ``LAYERS``
map by hand, a consistency test parses the test file's AST and asserts
the two groupings are identical, and each checked-in layer file must
equal :func:`render_layer_file` for its group — change the law here
and regenerate, or the pin fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["LAYER_GROUPS", "render_layer_file", "layers_dir",
           "load_layer_file", "layer_of"]

# Mirror of tests/test_layering.py: LAYERS (by package) + SHARED (layer
# 0) + TOP_LEVEL_LAYER (layer 7 modules).  The consistency test in
# tests/test_skillbook.py parses that file and pins this grouping.
LAYER_GROUPS: dict[int, dict[str, Any]] = {
    0: {
        "name": "shared",
        "title": "Shared surface — importable by everyone, imports nothing",
        "packages": ["compat", "version"],
        "owns": ("Compatibility shims and version identity: the tiny "
                 "surface every layer may lean on without creating a "
                 "dependency."),
        "may_not": ("Import any other nomorals package. If shared code "
                    "needs a real dependency it belongs in core, not "
                    "here."),
    },
    1: {
        "name": "kernel",
        "title": "Kernel — errors, events, tasks, text, time",
        "packages": ["core"],
        "owns": ("The primitives everything else is built from: error "
                 "types, the event bus, task kinds, text/time helpers, "
                 "configuration shape."),
        "may_not": ("Import any other nomorals package — core has zero "
                    "intra-project imports, enforced by test. No organ "
                    "logic lives here; core stays a kernel, not a "
                    "junk drawer."),
    },
    2: {
        "name": "persistence-and-device",
        "title": "Persistence & device — storage, accounts, native bridges",
        "packages": ["accounts", "native", "storage"],
        "owns": ("Durable state (the Database and its schema), account "
                 "and credential custody, and the native bridge "
                 "processes that talk to platforms."),
        "may_not": ("Know about missions, agents or chat. Storage "
                    "serves queries; it never decides policy. Applied "
                    "migrations are frozen — new persistence is a "
                    "runtime-created table or a JSON store."),
    },
    3: {
        "name": "engines",
        "title": "Engines — memory, models, cognition, matching, security",
        "packages": ["cognition", "i18n", "llm", "matching", "memory",
                     "planning", "security", "ta", "training"],
        "owns": ("The deep engines: long-term memory, the model ladder "
                 "and providers, cognition and planning, matching, "
                 "technical analysis, training, and the security "
                 "substrate (vault, capabilities, audit)."),
        "may_not": ("Reach up into chat, missions or agents. An engine "
                    "exposes capability; it never orchestrates a user "
                    "surface."),
    },
    4: {
        "name": "services",
        "title": "Services — tools, social, media, skills, domains",
        "packages": ["audio", "browser", "characters", "commerce",
                     "community", "datasci", "documents", "finance",
                     "goals", "learn", "legal", "marketing", "media",
                     "media_edit", "plugins", "property", "research",
                     "scheduler", "skills", "social", "tools", "vision",
                     "voice"],
        "owns": ("User-facing capability organs: the tool registry, "
                 "chat/social surfaces and living cards, media and "
                 "voice, the browser, documents, the skills system, "
                 "and the domain services (finance, legal, property, "
                 "marketing, learning)."),
        "may_not": ("Import layer-5 application organs (agents, "
                    "missions, partner, context). A service that needs "
                    "orchestration receives it as an injected callable."),
    },
    5: {
        "name": "application",
        "title": "Application — agents, missions, partner, context",
        "packages": ["agents", "autonomy", "backup", "books", "codews",
                     "connectors", "console", "context", "games",
                     "health", "hub", "integrations", "mesh", "missions",
                     "partner", "power", "search", "stream", "sync",
                     "travel", "triggers", "wisdom", "workspace"],
        "owns": ("The acting system: the agent roster and delegation, "
                 "missions and their lifecycle, the partner runtime "
                 "that speaks in chat, context assembly, autonomy, "
                 "games, search and the integration organs."),
        "may_not": ("Import the control plane (os/tui) or entry points. "
                    "The partner runtime receives OS hooks as injected "
                    "callables; it never reaches up for them."),
    },
    6: {
        "name": "control-plane",
        "title": "Control plane — OS services and the terminal UI",
        "packages": ["os", "tui"],
        "owns": ("Machine-level services: process and mission state "
                 "machines, update/restart machinery, and the terminal "
                 "interface."),
        "may_not": ("Be imported by the layers it controls. Control "
                    "flows down through explicit seams, never up "
                    "through imports."),
    },
    7: {
        "name": "entry-points",
        "title": "Entry points — API, builders, CLI",
        "packages": ["api", "builders", "cmdline"],
        "modules": ["__main__", "archives", "bench", "builders_proxy",
                    "cli", "execbox", "exporter", "self_improvement"],
        "owns": ("How the system is entered: the HTTP API, the builder "
                 "harness, the `nm` command line, and the top-level "
                 "single-file entry modules."),
        "may_not": ("Contain organ logic. An entry point parses, "
                    "dispatches into the layers below, and renders the "
                    "result — nothing more."),
    },
}


def layers_dir() -> Path:
    return Path(__file__).resolve().parent / "layers"


def layer_of(package: str) -> int | None:
    for number, group in LAYER_GROUPS.items():
        names = list(group.get("packages", ())) + list(
            group.get("modules", ()))
        if package in names:
            return number
    return None


def render_layer_file(number: int) -> str:
    """The canonical text of one layer instruction file."""
    group = LAYER_GROUPS[number]
    packages = ", ".join(group.get("packages", ()))
    modules = ", ".join(group.get("modules", ()))
    lines = [
        "---",
        f"id: layer-{number}-{group['name']}",
        f"title: Layer {number} — {group['title']}",
        f"summary: Instruction file for architectural layer {number} "
        f"({group['name']}).",
        f"topics: layers, architecture, layer {number}",
        "triggers: layering, import rules, which layer",
        "layer: architecture",
        "version: 1",
        "---",
        "",
        f"# Layer {number} — {group['title']}",
        "",
        f"**Packages:** {packages}" if packages else "",
    ]
    if modules:
        lines.append(f"**Top-level modules:** {modules}")
    lines += [
        "",
        "## What this layer owns",
        "",
        group["owns"],
        "",
        "## What this layer may not do",
        "",
        group["may_not"],
        "",
        "## The rule that enforces this",
        "",
        "Imports may only point at the same layer or a lower one; "
        "`tests/test_layering.py` scans every import in the repo "
        "(including lazy, function-level imports) and fails on an "
        "upward edge. This file is generated from "
        "`nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency "
        "test pins to the test's own layer map — the two cannot drift "
        "silently.",
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def load_layer_file(number: int) -> str:
    path = layers_dir() / f"layer-{number}-{LAYER_GROUPS[number]['name']}.md"
    return path.read_text(encoding="utf-8")
