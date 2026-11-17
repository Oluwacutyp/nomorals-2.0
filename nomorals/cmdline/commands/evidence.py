"""``nm evidence`` — research answers from real papers.

    nm evidence does vitamin D reduce respiratory infections? [--json]
    nm evidence list [--json]
    nm evidence show <id> [--json]
    nm evidence table <id> [--json]

The same evidence organ chat /evidence uses: the question is asked
of the open scholarly corpus (OpenAlex, Semantic Scholar, arXiv),
the paper set is screened, a cited extraction table is built (claim
/ method / result / stance per paper — every cell quotes its paper
verbatim), and the synthesis names what agrees, what disagrees and
what is still unanswered. A question with no papers prints the
honest reason and exits 1 — never an invented answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_evidence"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _print(answer: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer, indent=2, default=str))
    else:
        print(answer.get("reply") or answer.get("error") or "")


def _cmd_evidence(args: argparse.Namespace, context: Any) -> int:
    from ...research.extraction import (
        EvidenceTable,
        answer_question,
        export_csv,
        format_evidence_reply,
        open_review_store,
    )

    home = _home(context)
    store = open_review_store(home)
    words = list(getattr(args, "evidence_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    limit = getattr(args, "limit", None) or 10
    try:
        limit_n = max(1, int(limit))
    except (TypeError, ValueError):
        limit_n = 10
    verb = (words[0].lower() if words else "")
    if verb == "list":
        reviews = store.list()
        if as_json:
            print(json.dumps({"ok": True, "reviews": reviews}, indent=2,
                             default=str))
        elif not reviews:
            print("no evidence reviews yet — ask a research question "
                  "with: nm evidence <question>")
        else:
            for review in reviews:
                print(f"{review['id']} — {review['question']} "
                      f"({review['papers_included']} papers)")
        return 0
    if verb in {"show", "table"}:
        if len(words) < 2:
            print(f"evidence: give me a review id — "
                  f"nm evidence {verb} <id>", file=sys.stderr)
            return 2
        review = store.get(words[1])
        if review is None:
            print(f"no evidence review “{words[1]}”", file=sys.stderr)
            return 1
        if verb == "show":
            _print({**review, "ok": True,
                    "reply": format_evidence_reply(
                        {**review, "ok": True})}, as_json)
            return 0
        table = EvidenceTable.from_dict(review.get("table") or {})
        path = review.get("csv_path") or ""
        if not path:
            from ...research.extraction import TABLE_DIRNAME, _default_home

            path = export_csv(
                table, _default_home(home) / TABLE_DIRNAME
                / f"{words[1]}.csv")
            review["csv_path"] = path
            store.save(review)
        print(f"extraction table for “{review['question']}”: "
              f"{len(table.cells)} cells over "
              f"{len(table.paper_set.papers)} papers — {path}")
        return 0
    question = " ".join(words).strip()
    if not question:
        print("evidence: give me a research question — "
              "nm evidence <question>", file=sys.stderr)
        return 2
    answer = answer_question(home, question, store=store, limit=limit_n)
    if not answer.get("ok"):
        print(answer.get("error") or "no papers found",
              file=sys.stderr)
        return 1
    _print(answer, as_json)
    return 0
