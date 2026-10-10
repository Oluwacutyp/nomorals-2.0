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

from typing import Any

__all__ = [
    "TASK_SYSTEM_PROMPTS",
    "render_prompt",
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
        "for the given question. Score each on correctness, completeness, "
        "and honesty (no invented facts). Reply with JSON only: "
        '{"winner": <1-based index of the best answer>, '
        '"ranking": [<indices best to worst>], '
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
