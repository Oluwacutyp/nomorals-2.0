"""``nm devask`` — developer answers with cited sources and executed code.

    nm devask why does my Go bridge get 'index out of range'? [--json]
    nm devask <pasted stack trace> [--json]
    nm devask list [--json]
    nm devask show <id> [--json]

The same dev-answer organ chat /devask uses: the question (or pasted
traceback) is asked of the dev source pack (Devon's own code index
first, then the live web backends), the answer is synthesised with
real citations, and every self-contained code block is EXECUTED
through CodeRunner before it is presented — a block that did not
run green is labelled not verified, never presented as working.
A question no source answers prints the honest reason and exits 1
— never an invented answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_devask", "_default_run_fn"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _default_run_fn(context: Any):
    """The code-execution callable for this CLI (CodeRunner).

    Module-level seam: tests monkeypatch this name to inject a fake
    runner. cmdline is L7, so the static import of the L7 runner is
    the sanctioned direction.
    """
    from ...execbox import CodeRunner

    box = CodeRunner(context)

    def _run(code: str, lang: str):
        return box.run(code, lang=lang, timeout=30.0)

    return _run


def _print(answer: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer, indent=2, default=str))
    else:
        print(answer.get("reply") or answer.get("error") or "")


def _cmd_devask(args: argparse.Namespace, context: Any) -> int:
    from ...research.dev_answer import (
        answer_dev_question,
        format_dev_answer_reply,
        open_dev_answer_store,
    )

    home = _home(context)
    store = open_dev_answer_store(home)
    words = list(getattr(args, "devask_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    limit = getattr(args, "limit", None) or 5
    try:
        limit_n = max(1, int(limit))
    except (TypeError, ValueError):
        limit_n = 5
    verb = (words[0].lower() if words else "")
    if verb == "list":
        answers = store.list()
        if as_json:
            print(json.dumps({"ok": True, "answers": answers}, indent=2,
                             default=str))
        elif not answers:
            print("no dev answers yet — ask a developer question "
                  "with: nm devask <question>")
        else:
            for row in answers:
                print(f"{row['answer_id']} — {row['question']}")
        return 0
    if verb == "show":
        if len(words) < 2:
            print("devask: give me an answer id — nm devask show <id>",
                  file=sys.stderr)
            return 2
        row = store.get(words[1])
        if row is None:
            print(f"no dev answer “{words[1]}”", file=sys.stderr)
            return 1
        _print({**row, "ok": True,
                "reply": format_dev_answer_reply({**row, "ok": True})},
               as_json)
        return 0
    question = " ".join(words).strip()
    if not question:
        print("devask: give me a developer question (or a pasted "
              "stack trace) — nm devask <question>", file=sys.stderr)
        return 2
    answer = answer_dev_question(
        home, question, store=store, limit=limit_n,
        run_fn=_default_run_fn(context))
    if not answer.get("ok"):
        print(answer.get("error") or "no dev sources answered",
              file=sys.stderr)
        return 1
    _print(answer, as_json)
    return 0
