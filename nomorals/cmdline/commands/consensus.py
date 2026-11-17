"""``nm consensus`` — what the literature agrees on.

    nm consensus does vitamin D reduce respiratory infections? [--deep] [--json]
    nm consensus list [--json]
    nm consensus show <id> [--json]
    nm consensus meter <id> [--json]

The same consensus organ chat /consensus uses: the question is
asked of the open scholarly corpus (OpenAlex, Semantic Scholar,
arXiv), each paper's finding is classified yes / no / possibly /
mixed and weighed by quality (citations, recency, study design),
and the answer is a meter with the dissent named by name and what
would change it computed from the meter's own weights. A question
that is not yes/no gets prose instead of a meter. A question with
no papers prints the honest reason and exits 1 — never an invented
consensus.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_consensus"]


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


def _cmd_consensus(args: argparse.Namespace, context: Any) -> int:
    from ...research.consensus import (
        answer_consensus,
        format_consensus_reply,
        open_consensus_store,
    )

    home = _home(context)
    store = open_consensus_store(home)
    words = list(getattr(args, "consensus_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    deep = bool(getattr(args, "deep", False))
    depth = "deep" if deep else "quick"
    verb = (words[0].lower() if words else "")
    if verb == "list":
        reviews = store.list()
        if as_json:
            print(json.dumps({"ok": True, "reviews": reviews}, indent=2,
                             default=str))
        elif not reviews:
            print("no consensus reviews yet — ask a research question "
                  "with: nm consensus <question>")
        else:
            for review in reviews:
                meter = review.get("meter") or {}
                print(f"{review['id']} — {review['question']} "
                      f"({meter.get('strength', '')}, "
                      f"{meter.get('n', 0)} papers)")
        return 0
    if verb in {"show", "meter"}:
        if len(words) < 2:
            print(f"consensus: give me a review id — "
                  f"nm consensus {verb} <id>", file=sys.stderr)
            return 2
        review = store.get(words[1])
        if review is None:
            print(f"no consensus review “{words[1]}”", file=sys.stderr)
            return 1
        _print({**review, "ok": True,
                "reply": format_consensus_reply(
                    {**review, "ok": True})}, as_json)
        return 0
    question = " ".join(words).strip()
    if not question:
        print("consensus: give me a research question — "
              "nm consensus <question>", file=sys.stderr)
        return 2
    kwargs: dict[str, Any] = {"depth": depth}
    answer = answer_consensus(home, question, store=store, **kwargs)
    if not answer.get("ok"):
        print(answer.get("error") or "no papers found",
              file=sys.stderr)
        return 1
    _print(answer, as_json)
    return 0
