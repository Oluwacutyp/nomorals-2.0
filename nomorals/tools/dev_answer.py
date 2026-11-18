"""Brain tool: the developer answer engine (Phind core, 52-websites #40).

Ask a technical question — or paste a full stack trace — and get a
direct answer: explanation + code + citations to real sources, with
every self-contained code block EXECUTED before it is presented.
A block that did not run green is labelled "not verified", never
presented as working.

The tool is a thin shell over ``nomorals.research.dev_answer``:
retrieval is the organ's own (dev source pack over the federated
engine), synthesis is the organ's (extractive, or the model rung
when the context carries a router), and execution is the
``run_code`` tool reached through the registry — the sanctioned
tool-to-tool composition path, no import of the L7 runner at all.
When ``run_code`` is absent or refuses, the organ still answers,
and every block carries "no code runner available" instead of a
false verification.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from .paper_reader import _home_from_context, _llm_from_context


def _run_fn_from_registry(registry: Any):
    """A run callable over the ``run_code`` tool, or None.

    ``registry.call`` returns an Outcome: ``.ok`` False means the
    runner is unavailable (the organ marks blocks unverified);
    ``.ok`` True carries the ``CodeRunner.run`` dict as ``.value``.
    """
    if registry is None:
        return None

    def _run(code: str, lang: str):
        try:
            outcome = registry.call(
                "run_code", action="run", code=code, lang=lang,
                timeout=30.0)
        except Exception:  # noqa: BLE001 — a dead runner is unverified
            return None
        if not getattr(outcome, "ok", False):
            return None
        value = getattr(outcome, "value", None)
        return value if isinstance(value, dict) else None

    return _run


def register(registry: Any) -> None:
    """Attach the ``dev_answer`` tool to a registry."""
    context = registry.context

    @registry.register(
        "dev_answer",
        description=(
            "Developer answer engine (Phind core) — ask a technical "
            "question or paste a stack trace and get a direct answer: "
            "explanation + working code + citations to real sources "
            "(official docs, Stack Overflow, GitHub issues and "
            "discussions, changelogs, and Devon's own code index "
            "first). Every self-contained code block is EXECUTED "
            "before it is presented; a block that did not run green "
            "is labelled not verified, never presented as working. "
            "action=<ask|list|show>; ask takes question=<question or "
            "pasted traceback> and answer_id=<id from a previous "
            "answer> for a follow-up that keeps context; list shows "
            "past answers; show takes answer_id=<id>. An unanswered "
            "question is refused honestly — no answer, code, or "
            "citation is ever invented. Returns ok, answer_id, "
            "answer, code_blocks (each with verified_by_execution), "
            "citations, reply, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def dev_answer_tool(
        question: str = "",
        action: str = "ask",
        answer_id: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        from ..research.dev_answer import (
            answer_dev_question,
            format_dev_answer_reply,
            open_dev_answer_store,
        )

        home = _home_from_context(context)
        store = open_dev_answer_store(home)
        act = (action or "ask").strip().lower()
        if act == "list":
            return {"ok": True, "answers": store.list()}
        if act == "show":
            row = store.get((answer_id or question or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored dev answer "
                                 f"'{answer_id or question}'"}
            shown = dict(row)
            shown["ok"] = True
            shown["reply"] = format_dev_answer_reply(
                {**row, "ok": True})
            return shown
        text = (question or "").strip()
        if not text:
            return {"ok": False,
                    "error": "ask needs a question (or a pasted "
                             "stack trace)"}
        return answer_dev_question(
            home, text, store=store, limit=limit,
            llm_fn=_llm_from_context(context),
            run_fn=_run_fn_from_registry(registry),
            answer_id=answer_id.strip() or None)
