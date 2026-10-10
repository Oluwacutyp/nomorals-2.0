"""Native prompt engineering: task-kind system prompts, owned in-repo.

Ad-hoc system strings scattered across forty call sites drift, contradict
each other, and can't be tested.  This module is the single home for the
prompts Devon's own machinery uses when it talks to a model *as machinery*
(intent classification, judging, summarization, JSON extraction, planning):
one template per task-kind, slot-rendered, unit-testable.

Callers that need persona-flavored prompts (the partner responder, the
coding agent's developer voice) keep their own builders — this is the
machine-task layer, not the character layer.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

__all__ = [
    "TASK_SYSTEM_PROMPTS",
    "PromptLibrary",
    "get_library",
    "render_prompt",
    "rubric_prompt",
    "system_prompt_for",
]

#: task-kind → system prompt template.  ``{slots}`` are filled by
#: :func:`render_prompt`; unknown slots are left as-is so a template can
#: be rendered partially.
TASK_SYSTEM_PROMPTS: dict[str, str] = {
    "intent": (
        "You are the intent router for an agent OS. Classify the user's "
        "message. Reply with ONLY one word from this list:\n"
        "schedule (reminders, alerts, 'in X minutes')\n"
        "mission (tasks, background work)\n"
        "research (look up, find information)\n"
        "build (create code, build something)\n"
        "chat (general conversation, no specific action)\n"
        "status (system status check)\n"
        "If unsure, reply 'chat'."
    ),
    "intent_detail": (
        "You are the intent router for an agent OS. Classify the user "
        "message. Reply with JSON only: "
        '{"kind": "chat|research|build|browse|download|mission|game|status", '
        '"target": "...", "confidence": 0.0, "why": "max 12 words"}. '
        "Kinds: research (web investigation), build (create software), "
        "browse (open/read a page), download (fetch a file), "
        "mission (queue durable work), game (start/continue one of: {games}), "
        "status (system state), chat (everything else)."
    ),
    "judge": (
        "You are an impartial judge. Compare the candidate answers below "
        "for the given question. Score each on correctness (no invented "
        "facts), completeness, and clarity (no filler). Think step by step "
        "FIRST — work through the dimensions for each candidate and note "
        "the evidence — THEN give the verdict. Penalize verbosity that "
        "adds no information. Reply with JSON only: "
        '{"scores": {"<1-based candidate>": {"correctness": 0-5, '
        '"completeness": 0-5, "clarity": 0-5}}, '
        '"winner": <1-based index of the best answer>, '
        '"ranking": [<indices best to worst>], '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<one or two sentences>"}. '
        "If all candidates are poor, still rank them — say so in rationale."
    ),
    "summarize": (
        "You summarize conversation history for an agent's working memory. "
        "Compress the turns below into a compact summary: keep decisions, "
        "facts about the user, open threads, and anything the agent "
        "promised to do. Drop greetings, filler, and repeated small talk. "
        "Reply with the summary only, no preamble."
    ),
    "extract": (
        "You extract structured data. Read the input and reply with ONLY "
        "the requested JSON — no prose, no code fences, no commentary. "
        "If a field is unknown, use null rather than inventing it."
    ),
    "plan": (
        "You are a planning assistant for an agent OS. Given the goal, "
        "produce a short ordered plan: concrete steps, each verifiable. "
        "Reply with JSON only: "
        '{"steps": ["..."], "risks": ["..."], "done_when": "..."}. '
        "Keep it tight — no filler steps."
    ),
    "research": (
        "You are a research synthesizer. Answer the query using ONLY the "
        "provided sources. Cite which sources you used. If the sources do "
        "not answer it, say so plainly — do not invent."
    ),
    "creative": (
        "You are a creative writing assistant. Write vivid, specific prose "
        "— concrete details over adjectives, no filler openers."
    ),
    "chat": "",
}


def system_prompt_for(task_kind: str) -> str:
    """The machine-task system prompt for a task-kind ("" when none)."""
    return TASK_SYSTEM_PROMPTS.get((task_kind or "").lower(), "")


def render_prompt(task_kind: str, **slots: Any) -> str:
    """Render a task-kind prompt with ``{slot}`` values.

    Unknown slots are left in place (partial rendering is allowed);
    unknown task-kinds render to "".
    """
    template = system_prompt_for(task_kind)
    if not template:
        return ""
    out = template
    for key, value in slots.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def rubric_prompt(
    dimensions: dict[str, str] | None = None,
    *,
    reference: str = "",
    scale: int = 5,
) -> str:
    """Reference-guided judge rubric, rendered as a system prompt.

    ``dimensions`` maps dimension name → what a top score looks like
    (a rubric written like a spec, not a vibe — vague criteria produce
    vague judgments).  ``reference`` is the gold answer the judge grades
    against when one exists.  The judge reasons step-by-step *before* the
    verdict (CoT-first measurably improves consistency) and replies with
    JSON only.
    """
    dims = dimensions or {
        "correctness": "factually right; no invented facts or unsupported claims",
        "completeness": "answers every part of the question, nothing important omitted",
        "clarity": "direct and well-structured; no filler or hedging",
    }
    lines = [
        "You are an impartial judge. Grade each candidate answer against the "
        "rubric below.",
    ]
    if reference.strip():
        lines.append(
            "Reference answer (the gold standard — grade closeness to it):\n"
            f"<<<\n{reference.strip()[:4000]}\n>>>"
        )
    lines.append(f"Rubric — score each dimension 0–{scale}:")
    for name, desc in dims.items():
        lines.append(f"- {name}: {desc}")
    lines.append(
        "Think step by step FIRST: for each candidate, work through the "
        "dimensions one by one and note the evidence. THEN give the verdict. "
        "Reply with JSON only: {\"scores\": {\"<candidate-1-based>\": "
        f"{{\"<dimension>\": 0-{scale}, ...}}}}, \"winner\": <1-based index>, "
        "\"ranking\": [<indices best to worst>], "
        "\"confidence\": <0.0-1.0>, "
        "\"rationale\": \"<two sentences max>\"}. "
        "Penalize verbosity that adds no information. If all candidates are "
        "poor, still rank them — say so in rationale."
    )
    return "\n".join(lines)


class PromptLibrary:
    """Versioned in-repo prompt registry (Langfuse-style, local).

    Machine-task prompts drift when they live as string literals across
    forty call sites.  The library stores every prompt with a version
    history: rolling out a prompt change is a registry operation, not a
    deploy — pin ``version="3"`` for stability or ``"latest"`` for the
    newest.  Thread-safe; persists to a JSON file when a path is given.
    The built-in :data:`TASK_SYSTEM_PROMPTS` are seeded as version 1.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._prompts: dict[str, list[dict[str, Any]]] = {}
        self._path = Path(path) if path else None
        for name, template in TASK_SYSTEM_PROMPTS.items():
            self._prompts[name] = [{
                "version": 1,
                "template": template,
                "created_at": time.time(),
                "note": "built-in seed",
            }]
        if self._path is not None:
            self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return
        except Exception:
            return
        if isinstance(data, dict):
            with self._lock:
                for name, versions in data.items():
                    if isinstance(versions, list) and versions:
                        self._prompts[str(name)] = versions

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                snapshot = {k: list(v) for k, v in self._prompts.items()}
            self._path.write_text(json.dumps(snapshot, indent=2),
                                  encoding="utf-8")
        except Exception:
            pass

    # ── registry ─────────────────────────────────────────────────────────
    def register(self, name: str, template: str, *,
                 note: str = "") -> int:
        """Store a new version of ``name``.  Returns the version number."""
        name = (name or "").strip().lower()
        if not name or not template:
            raise ValueError("prompt name and template are required")
        with self._lock:
            versions = self._prompts.setdefault(name, [])
            version = (versions[-1]["version"] + 1) if versions else 1
            versions.append({
                "version": version,
                "template": template,
                "created_at": time.time(),
                "note": note,
            })
        self._save()
        return version

    def get(self, name: str, version: int | str = "latest") -> str:
        """The template for ``name`` at ``version`` ("" when unknown)."""
        name = (name or "").strip().lower()
        with self._lock:
            versions = self._prompts.get(name) or []
            if not versions:
                return ""
            if version == "latest":
                return str(versions[-1]["template"])
            for v in reversed(versions):
                if v["version"] == version:
                    return str(v["template"])
            return ""

    def history(self, name: str) -> list[dict[str, Any]]:
        """Version history for ``name`` (newest last)."""
        name = (name or "").strip().lower()
        with self._lock:
            return [dict(v) for v in self._prompts.get(name, [])]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._prompts)

    def render(self, prompt_name: str, version: int | str = "latest",
               **slots: Any) -> str:
        """Get + slot-render in one call (partial rendering allowed)."""
        out = self.get(prompt_name, version)
        for key, value in slots.items():
            out = out.replace("{" + key + "}", str(value))
        return out


_library: PromptLibrary | None = None
_library_lock = threading.Lock()


def get_library(path: str | Path | None = None) -> PromptLibrary:
    """Shared process-wide prompt library (path-pinned on first call)."""
    global _library
    with _library_lock:
        if _library is None:
            _library = PromptLibrary(path=path)
        return _library
