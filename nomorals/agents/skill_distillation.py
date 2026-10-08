"""Post-task skill distillation (Hermes loop) — success-driven learning.

After a task with 5+ tool calls succeeds, extract the reusable approach
as a NEW skill draft.  This is the "what did I learn?" step: the
complement to ``skill_evolution.py`` (which only fixes FAILING skills).

The distilled draft installs as a new skill version with ``active=0`` —
it never auto-activates.  Promotion goes through the existing pin
mechanism after canary validation.

Profile gating: distillation costs an LLM call.  ``workstation`` and
``laptop`` distill automatically; ``termux`` only when opted in
(``NM_DISTILL=1``).

Usage::

    from nomorals.agents.skill_distillation import maybe_distill
    # called from the agentic loop post-task:
    maybe_distill(result, memory, context)
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "SkillDraft",
    "distill",
    "maybe_distill",
    "should_distill",
]

#: minimum tool calls before a trace is worth distilling
MIN_TOOL_CALLS = 5

#: cap the draft body so a runaway trace can't bloat the registry
MAX_DRAFT_CHARS = 4000


@dataclass
class SkillDraft:
    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    workflow: str = ""          # the reusable approach, as text
    trace_hash: str = ""        # provenance: hash of the originating trace
    task_kind: str = ""

    def to_manifest(self) -> dict[str, Any]:
        # Cache the version on the instance: the same draft must map to
        # one version across calls (install -> prove -> pin), otherwise a
        # second ticking over between calls breaks the version match.
        version = getattr(self, "_manifest_version", "")
        if not version:
            version = f"0.1.{int(time.time()) % 100000}"
            self._manifest_version = version
        return {
            "name": self.name,
            "version": version,
            "tools": self.tools,
            "description": self.description,
            "wiring": [{"step": "distilled",
                        "detail": self.workflow[:MAX_DRAFT_CHARS]}],
            "owner": "distillation",
        }


_DISTILL_PROMPT = """You watched an AI agent complete a task successfully.
Extract the REUSABLE approach as a skill draft.

TASK: {task}

TOOL SEQUENCE (what worked):
{sequence}

Extract:
1. SKILL NAME — short, snake_case, e.g. "research_then_summarize"
2. DESCRIPTION — one line: when to use this skill
3. TOOLS — the tool names in order, comma-separated
4. WORKFLOW — the reusable pattern in 3-6 numbered steps.  Be specific
   about what each step does and what to pass between steps.  No
   task-specific details (no URLs, no names, no dates).

Reply in exactly this format:
NAME: <name>
DESCRIPTION: <description>
TOOLS: <tool1>, <tool2>
WORKFLOW:
1. ...
2. ...
"""


def _trace_hash(steps: list[Any]) -> str:
    h = hashlib.sha256()
    for s in steps:
        h.update(str(getattr(s, "tool_name", "")).encode())
        h.update(str(getattr(s, "thought", "")).encode())
    return h.hexdigest()[:16]


def should_distill(result: Any, context: Any = None) -> bool:
    """True when a result is worth distilling."""
    if result is None or not getattr(result, "success", False):
        return False
    tools_called = getattr(result, "tools_called", []) or []
    if len(tools_called) < MIN_TOOL_CALLS:
        return False
    # profile gate: workstation/laptop auto, termux opt-in
    try:
        from ..core.profiles import get_profile_kind
        profile = get_profile_kind()
    except Exception:  # noqa: BLE001
        profile = "workstation"
    if profile == "termux" and os.environ.get("NM_DISTILL") != "1":
        return False
    return True


def distill(task: str, steps: list[Any], tools_called: list[str],
            llm_fn: Any = None, context: Any = None) -> Optional[SkillDraft]:
    """Distill a successful trace into a skill draft.  Returns None on failure."""
    sequence = []
    for s in steps:
        tool = getattr(s, "tool_name", "")
        thought = getattr(s, "thought", "")
        if tool:
            sequence.append(f"- {tool}: {thought[:120]}")
    if not sequence:
        return None
    prompt = _DISTILL_PROMPT.format(
        task=task[:500],
        sequence="\n".join(sequence[:20]),
    )
    try:
        if llm_fn is not None:
            text = llm_fn(prompt)
        else:
            router = getattr(context, "router", None) if context else None
            if router is None:
                _log.warning("distillation skipped: no LLM available")
                return None
            resp = router.complete(prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as exc:  # noqa: BLE001
        _log.warning("distillation LLM call failed: %s", exc)
        return None

    draft = _parse_draft(text)
    if draft is None:
        return None
    draft.trace_hash = _trace_hash(steps)
    draft.tools = [t for t in tools_called if t in draft.tools] or tools_called[:8]
    return draft


def _parse_draft(text: str) -> Optional[SkillDraft]:
    """Parse the LLM's structured reply into a SkillDraft."""
    name = description = ""
    tools: list[str] = []
    workflow_lines: list[str] = []
    in_workflow = False
    for line in (text or "").splitlines():
        line = line.strip()
        if line.upper().startswith("NAME:"):
            name = line[5:].strip().lower().replace(" ", "_")
            in_workflow = False
        elif line.upper().startswith("DESCRIPTION:"):
            description = line[12:].strip()
            in_workflow = False
        elif line.upper().startswith("TOOLS:"):
            tools = [t.strip() for t in line[6:].split(",") if t.strip()]
            in_workflow = False
        elif line.upper().startswith("WORKFLOW:"):
            in_workflow = True
        elif in_workflow and line:
            workflow_lines.append(line)
    if not name or not workflow_lines:
        return None
    # sanitize the name
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)[:48]
    if not name:
        return None
    return SkillDraft(
        name=f"distilled_{name}",
        description=description[:200] or "distilled skill",
        tools=tools,
        workflow="\n".join(workflow_lines)[:MAX_DRAFT_CHARS],
    )


def maybe_distill(result: Any, memory: Any, context: Any = None,
                  llm_fn: Any = None, db: Any = None,
                  prove: bool = False) -> Optional[SkillDraft]:
    """Post-task hook: distill and install a skill draft when warranted.

    Called from the agentic loop after a run.  Installs the draft with
    ``active=0`` — never auto-promotes.  Returns the draft or None.

    ``llm_fn`` is a ``prompt -> text`` callable; when omitted the router
    on ``context`` is used.  ``db`` is the skill database; when omitted
    it is taken from ``context.db`` or opened from the app's default
    storage path.  Without a database the draft is still returned but
    installation is skipped (logged, never raises).

    ``prove`` enables the H.O.T-Jarvis test-proof path (see
    ``skill_proving``): the draft gets an auto-generated proof test and
    only activates on a passing test; a failing test flags and disables
    the skill instead.  Default False preserves the manual
    canary-validation + explicit-pin flow.
    """
    if not should_distill(result, context):
        return None
    steps = []
    if memory is not None:
        steps = getattr(memory, "steps", []) or []
    tools_called = list(getattr(result, "tools_called", []) or [])
    task = ""
    if memory is not None:
        task = str(getattr(memory, "user_message", "") or "")

    draft = distill(task, steps, tools_called, llm_fn=llm_fn, context=context)
    if draft is None:
        return None
    try:
        database = db if db is not None else getattr(context, "db", None)
        if database is None:
            from ..core.config import StorageSettings
            from ..storage.db import open_database
            database, _, _ = open_database(StorageSettings().path)
        from ..skills.registry import SkillRegistry
        registry = SkillRegistry(database)
        registry.install(draft.to_manifest())
        # Drafts never auto-activate: the first installed version becomes
        # the active pin by registry design, so clear it immediately. The
        # draft waits for canary validation + an explicit pin.
        registry.deactivate(draft.name)
        _log.info("distilled skill installed (inactive): %s", draft.name)
        if prove:
            from .skill_proving import promote_on_proof, prove_skill
            proof = prove_skill(draft, database, llm_fn=llm_fn)
            outcome = promote_on_proof(draft, proof, registry, database)
            _log.info("distilled skill %s proof outcome: %s",
                      draft.name, outcome)
    except Exception as exc:  # noqa: BLE001
        _log.warning("distilled skill install failed: %s", exc)
    return draft
