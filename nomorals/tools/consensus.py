"""Consensus tool: ``consensus`` — the brain's stance meter.

Devon asks what the literature agrees on and gets a meter: each
paper's finding classified as yes / no / possibly / mixed, weighed
by quality (citations, recency, study design — never count alone),
with the dissent named by name and what would change it computed
from the meter's own weights. The tool is a thin surface over
:mod:`nomorals.research.consensus`, which itself builds on the
evidence organ (:mod:`nomorals.research.extraction`) — one
retrieval path, no second corpus. A question that is not yes/no
gets prose instead of a meter; a question with no papers is
refused honestly, never invented.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``consensus`` tool to a registry."""
    context = registry.context

    @registry.register(
        "consensus",
        description=(
            "Research consensus — what the literature agrees on. Ask "
            "a yes/no research question and get a meter over real "
            "papers from open scholarly sources (OpenAlex, Semantic "
            "Scholar, arXiv): each paper's finding classified yes / "
            "no / possibly / mixed, weighed by quality (citations, "
            "recency, study design — never count alone), with the "
            "dissent named by name and what would change it computed "
            "from the meter's own weights. A question that is not "
            "yes/no gets prose instead of a meter; a question with no "
            "papers is refused honestly, never invented. "
            "action=<ask|list|show|meter>; ask takes question=<the "
            "research question> and depth=<quick|deep, default "
            "quick>; show/meter take review_id=<id from a previous "
            "ask>. Returns ok, review_id, evidence_review_id, meter, "
            "verdicts, synthesis, sources, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def consensus_tool(
        question: str = "",
        action: str = "ask",
        review_id: str = "",
        depth: str = "quick",
    ) -> dict[str, Any]:
        from ..research.consensus import (
            answer_consensus,
            format_consensus_reply,
            open_consensus_store,
        )

        home = _home_from_context(context)
        store = open_consensus_store(home)
        act = (action or "ask").strip().lower()
        if act == "list":
            return {"ok": True, "reviews": store.list()}
        if act in {"show", "meter"}:
            if not (review_id or "").strip():
                raise ValueError("consensus show/meter needs review_id=<id>")
            review = store.get(review_id.strip())
            if review is None:
                return {"ok": False, "review_id": review_id.strip(),
                        "error": f"no consensus review “{review_id.strip()}”"}
            shown = dict(review)
            shown["ok"] = True
            shown["reply"] = format_consensus_reply(
                {**review, "ok": True})
            return shown
        if act != "ask":
            raise ValueError(
                f"unknown consensus action “{act}” — ask|list|show|meter")
        if not (question or "").strip():
            raise ValueError(
                "consensus ask needs question=<research question>")
        return answer_consensus(home, question.strip(), store=store,
                                depth=(depth or "quick").strip().lower())
