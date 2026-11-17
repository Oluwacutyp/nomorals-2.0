"""``nm readpaper`` — read a paper and explain it in plain language.

    nm readpaper <DOI or title> [--json]
    nm readpaper text <pasted paper text...> [--json]
    nm readpaper list [--json]
    nm readpaper show <id> [--json]
    nm readpaper ask <id> <question...> [--json]
    nm readpaper explain <id> <equation 1|table 1> [--json]
    nm readpaper cite <id> [apa|mla|chicago|bibtex] [--json]
    nm readpaper paraphrase <text...> [--json]

The same paper-reader organ chat /readpaper uses: what the paper
asks, what it did, what it found — every finding quoted verbatim
from the paper — why it matters, and the hard parts (equations,
tables) unpacked. A paper with no record, or a question the paper
does not cover, prints the honest reason and exits 1 — never an
invented explanation.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...research.paper_reader import (
    PaperText,
    ask_paper,
    explain_span,
    format_citation,
    format_reading_reply,
    open_reading_store,
    paraphrase_span,
    read_paper,
)

__all__ = ["_cmd_readpaper"]


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


def _cmd_readpaper(args: argparse.Namespace, context: Any) -> int:
    home = _home(context)
    store = open_reading_store(home)
    words = list(getattr(args, "readpaper_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    verb = (words[0].lower() if words else "")
    rest = " ".join(words[1:]).strip() if len(words) > 1 else ""
    if not words:
        print("usage: nm readpaper <DOI or title> | text <pasted text> | "
              "list | show <id> | ask <id> <question> | explain <id> "
              "<equation 1|table 1> | cite <id> [style] | paraphrase "
              "<text>")
        return 2
    if verb == "list":
        rows = store.list()
        if as_json:
            print(json.dumps({"ok": True, "readings": rows}, indent=2))
        elif not rows:
            print("no paper readings yet — read one with "
                  "nm readpaper <DOI or title>")
        else:
            print("paper readings:")
            for row in rows[:10]:
                print(f"  {row['reading_id']} — {row['title']} "
                      f"({row.get('year') or '?'})")
        return 0
    if verb == "show":
        if not rest:
            print("usage: nm readpaper show <id>")
            return 2
        row = store.get(rest)
        if row is None:
            print(f"no stored reading '{rest}'")
            return 1
        _print({"ok": True, "reading_id": row["paper"]["paper_id"],
                "paper": row["paper"],
                "explanation": row["explanation"],
                "reply": format_reading_reply(
                    {"ok": True,
                     "reading_id": row["paper"]["paper_id"],
                     "paper": row["paper"],
                     "explanation": row["explanation"]})}, as_json)
        return 0
    if verb == "ask":
        parts = rest.split(None, 1)
        if len(parts) < 2:
            print("usage: nm readpaper ask <id> <question>")
            return 2
        reading_id, question = parts
        row = store.get(reading_id)
        if row is None:
            print(f"no stored reading '{reading_id}' — read the paper "
                  "first")
            return 1
        result = ask_paper(PaperText.from_dict(row["paper"]), question)
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        elif not result.get("ok"):
            print(result.get("error") or "the paper does not cover that")
        else:
            print(result["answer"])
            for quote in result.get("quotes") or []:
                print(f"  “{quote}”")
        if result.get("ok"):
            store.log_question(reading_id, question, result["answer"])
        return 0 if result.get("ok") else 1
    if verb == "explain":
        parts = rest.split(None, 1)
        if len(parts) < 2:
            print("usage: nm readpaper explain <id> <equation 1|table 1>")
            return 2
        reading_id, span_ref = parts
        row = store.get(reading_id)
        if row is None:
            print(f"no stored reading '{reading_id}' — read the paper "
                  "first")
            return 1
        result = explain_span(PaperText.from_dict(row["paper"]), span_ref)
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        elif not result.get("ok"):
            print(result.get("error") or "no such span in this paper")
        else:
            part = result["explanation"]
            print(f"{part.span_ref} ({part.kind}): {part.plain_text}")
            for quote in part.grounded_quotes:
                print(f"  “{quote}”")
        return 0 if result.get("ok") else 1
    if verb == "cite":
        parts = rest.split(None, 1)
        if not parts:
            print("usage: nm readpaper cite <id> [apa|mla|chicago|bibtex]")
            return 2
        reading_id = parts[0]
        style = (parts[1].strip().lower() if len(parts) > 1 else "apa")
        row = store.get(reading_id)
        if row is None:
            print(f"no stored reading '{reading_id}' — read the paper "
                  "first")
            return 1
        from ...research.paper_records import PaperRecord

        record = PaperRecord.from_dict(row["paper"])
        try:
            citation = format_citation(record, style)
        except ValueError as exc:
            print(str(exc))
            return 2
        if as_json:
            print(json.dumps({"ok": True, "citation": citation},
                             indent=2))
        else:
            print(citation)
        return 0
    if verb == "paraphrase":
        if not rest:
            print("usage: nm readpaper paraphrase <text>")
            return 2
        result = paraphrase_span(rest)
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        elif not result.get("ok"):
            print(result.get("error") or "nothing to paraphrase")
        else:
            print(result["paraphrase"])
            if result.get("method") == "quoted":
                print("(kept verbatim — a simpler version would lose a "
                      "number or a name)")
        return 0 if result.get("ok") else 1
    if verb == "text":
        if not rest:
            print("usage: nm readpaper text <pasted paper text>")
            return 2
        result = read_paper(rest, home=home)
        _print(result, as_json)
        return 0 if result.get("ok") else 1
    result = read_paper(" ".join(words).strip(), home=home)
    _print(result, as_json)
    return 0 if result.get("ok") else 1
