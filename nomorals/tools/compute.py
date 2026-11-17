"""Compute tool: ``compute`` — the brain's computational answers.

Devon asks a computable question and gets a derived answer with its
rung and its steps — never a guessed number. The tool is a thin
surface over :mod:`nomorals.codews.compute`; the organ owns the
ladder (WolframAlpha when keyed, sympy CAS, exact stdlib) and the
honest refusal when no rung can answer.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _appid_from_context(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return ""
    wolfram = getattr(settings, "wolfram", None)
    if wolfram is None:
        return ""
    return str(getattr(wolfram, "appid", "") or "")


def register(registry: Any) -> None:
    """Attach the ``compute`` tool to a registry."""
    context = registry.context

    @registry.register(
        "compute",
        description=(
            "Compute an answer to a computable question — arithmetic, "
            "units, dates, symbolic math (integrate, differentiate, "
            "solve, factor), and curated facts through WolframAlpha "
            "when it is keyed. The answer always carries its rung and "
            "its derivation, or an honest refusal when no rung can "
            "answer it. question=<the question in plain words>; "
            "rung=<auto|wolfram|sympy|stdlib> picks one rung instead "
            "of the ladder. Returns ok, result, rung, steps, source, "
            "pods and assumptions (WolframAlpha), or error."
        ),
        capability=Capability.NET_OUT,
    )
    def compute_tool(question: str = "", rung: str = "auto") -> dict[str, Any]:
        from ..codews.compute import compute_answer

        if not question or not str(question).strip():
            raise ValueError("compute needs a question")
        comp = compute_answer(
            str(question).strip(),
            appid=_appid_from_context(context) or None,
            rung=(rung or "auto").strip().lower())
        return {
            "ok": comp.ok,
            "question": comp.query,
            "result": comp.result,
            "rung": comp.rung,
            "steps": list(comp.steps),
            "source": comp.source,
            "pods": comp.pods,
            "assumptions": list(comp.assumptions),
            "error": comp.error,
        }
