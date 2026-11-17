"""Evidence tool: ``evidence`` — the brain's research assistant.

Devon asks a research question and gets an answer built from real
papers: each paper named with the claim it makes, the method behind
it and the result it found, plus a synthesis that names what agrees,
what disagrees and what is still unanswered. The tool is a thin
surface over :mod:`nomorals.research.extraction` and
:mod:`nomorals.research.screening`; the organs own retrieval,
screening, the cited cell matrix and the honest refusal when no
papers exist. Nothing is ever invented: a question with no papers
is answered "no papers found".
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
    """Attach the ``evidence`` tool to a registry."""
    context = registry.context

    @registry.register(
        "evidence",
        description=(
            "Research answers from real papers — the Elicit core. Ask "
            "a research question and get a paper set from open scholarly "
            "sources (OpenAlex, Semantic Scholar, arXiv), screened "
            "against inclusion criteria, with a cited extraction table "
            "per paper (claim, method, result, stance — every cell "
            "quotes its paper verbatim) and a synthesis naming what "
            "agrees, what disagrees and what is still unanswered. "
            "action=<ask|list|show|table>; ask takes question=<the "
            "research question> and limit=<max papers, default 10>; "
            "show/table take review_id=<id from a previous ask>. A "
            "question with no papers is refused honestly, never "
            "invented. Returns ok, review_id, papers_screened, "
            "papers_included, table, synthesis, sources, csv_path, "
            "or error."
        ),
        capability=Capability.NET_OUT,
    )
    def evidence_tool(
        question: str = "",
        action: str = "ask",
        review_id: str = "",
        limit: Any = 10,
    ) -> dict[str, Any]:
        from ..research.extraction import (
            EvidenceTable,
            answer_question,
            export_csv,
            format_evidence_reply,
            open_review_store,
        )

        home = _home_from_context(context)
        store = open_review_store(home)
        act = (action or "ask").strip().lower()
        if act == "list":
            return {"ok": True, "reviews": store.list()}
        if act in {"show", "table"}:
            if not (review_id or "").strip():
                raise ValueError("evidence show/table needs review_id=<id>")
            review = store.get(review_id.strip())
            if review is None:
                return {"ok": False, "review_id": review_id.strip(),
                        "error": f"no evidence review “{review_id.strip()}”"}
            if act == "show":
                shown = dict(review)
                shown["ok"] = True
                shown["reply"] = format_evidence_reply(
                    {**review, "ok": True})
                return shown
            table = EvidenceTable.from_dict(review.get("table") or {})
            path = review.get("csv_path") or ""
            if not path:
                from ..research.extraction import (
                    TABLE_DIRNAME,
                    _default_home,
                )

                path = export_csv(
                    table,
                    _default_home(home) / TABLE_DIRNAME
                    / f"{review_id.strip()}.csv")
                review["csv_path"] = path
                store.save(review)
            return {"ok": True, "review_id": review_id.strip(),
                    "csv_path": path,
                    "cells": len(table.cells),
                    "papers": len(table.paper_set.papers)}
        if act != "ask":
            raise ValueError(
                f"unknown evidence action “{act}” — ask|list|show|table")
        if not (question or "").strip():
            raise ValueError("evidence ask needs question=<research question>")
        try:
            limit_n = int(limit) if limit is not None else 10
        except (TypeError, ValueError):
            limit_n = 10
        return answer_question(home, question.strip(), store=store,
                               limit=max(1, limit_n))
