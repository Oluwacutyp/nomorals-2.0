"""``nm regex`` — the regex workbench on the command line.

    nm regex '\\d{3}-\\d{4}' --text 'call 555-1234' [--json]
    nm regex gen 'an email address' [--json]
    nm regex debug '(a+)+$' --text 'aaaaaaaaaab' [--json]
    nm regex sub '\\d+' --replacement '[N]' --text 'a1b22' [--json]
    nm regex code <id> <lang> [--json]
    nm regex list [--json]
    nm regex show <id> [--json]

The same regex-lab organ chat /regex uses: the pattern is parsed by
the real pattern parser (token tree + plain-English explanation),
matched with groups, named groups and positions, traced with a step
budget that catches catastrophic backtracking, and code is generated
per language and EXECUTED in codews to prove it reproduces the
session's matches — a snippet that does not reproduce is labelled,
never presented as verified. A generated pattern is proved against
positive and negative cases before it is printed. A pattern a
flavour does not support is refused by name; a flavour with no
engine here is labelled not verifiable, never simulated.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_regex", "_default_run_fn"]


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


def _print(record: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(record, indent=2, default=str))
    else:
        print(record.get("reply") or record.get("error") or "")


def _cmd_regex(args: argparse.Namespace, context: Any) -> int:
    from ...tools.regex_lab import (
        format_regex_reply,
        generate_code,
        open_regex_store,
        run_regex_session,
        verify_code,
    )

    home = _home(context)
    store = open_regex_store(home)
    words = list(getattr(args, "regex_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    texts = [t for t in (getattr(args, "text", None) or [])
             if t.strip()]
    flavour = getattr(args, "flavour", None) or "python"
    flags = getattr(args, "flags", None) or ""
    verb = (words[0].lower() if words else "")
    if verb == "list":
        rows = store.list()
        if as_json:
            print(json.dumps({"ok": True, "sessions": rows},
                             indent=2, default=str))
        elif not rows:
            print("no regex sessions yet — test a pattern with: "
                  "nm regex '<pattern>' --text '<text>'")
        else:
            for row in rows:
                print(f"{row['session_id']} — {row['pattern']} "
                      f"({row['flavour']})")
        return 0
    if verb == "show":
        if len(words) < 2:
            print("regex: give me a session id — nm regex show <id>",
                  file=sys.stderr)
            return 2
        row = store.get(words[1])
        if row is None:
            print(f"no regex session “{words[1]}”", file=sys.stderr)
            return 1
        _print({**row, "ok": True,
                "reply": format_regex_reply({**row, "ok": True})},
               as_json)
        return 0
    if verb == "gen":
        description = " ".join(words[1:]).strip()
        if not description:
            print("regex: give me a description — nm regex gen "
                  "'an email address'", file=sys.stderr)
            return 2
        record = run_regex_session(
            home, description=description,
            run_fn=_default_run_fn(context), store=store)
        if not record.get("ok"):
            print(record.get("error") or "generation failed",
                  file=sys.stderr)
            return 1
        record["reply"] = format_regex_reply(record)
        _print(record, as_json)
        return 0
    if verb == "debug":
        if len(words) < 2 or not texts:
            print("regex: give me a pattern and text — nm regex "
                  "debug '<pattern>' --text '<text>'",
                  file=sys.stderr)
            return 2
        record = run_regex_session(
            home, pattern=words[1], flavour=flavour, flags=flags,
            trace_text=texts[0], run_fn=_default_run_fn(context),
            store=store)
        if not record.get("ok"):
            print(record.get("error") or "debug failed",
                  file=sys.stderr)
            return 1
        record["reply"] = format_regex_reply(record)
        _print(record, as_json)
        return 0
    if verb == "sub":
        replacement = getattr(args, "replacement", None) or ""
        if len(words) < 2 or not texts:
            print("regex: give me a pattern, a replacement and "
                  "text — nm regex sub '<pattern>' --replacement "
                  "'<replacement>' --text '<text>'",
                  file=sys.stderr)
            return 2
        record = run_regex_session(
            home, pattern=words[1], flavour=flavour, flags=flags,
            substitution=replacement, texts=texts,
            run_fn=_default_run_fn(context), store=store)
        if not record.get("ok"):
            print(record.get("error") or "substitution failed",
                  file=sys.stderr)
            return 1
        record["reply"] = format_regex_reply(record)
        _print(record, as_json)
        return 0
    if verb == "code":
        if len(words) < 3:
            print("regex: give me a session id and a language — "
                  "nm regex code <id> <lang>", file=sys.stderr)
            return 2
        row = store.get(words[1])
        if row is None:
            print(f"no regex session “{words[1]}”", file=sys.stderr)
            return 1
        snippet = generate_code(
            str(row.get("pattern", "")),
            str(row.get("flavour", "python")), words[2],
            texts=[str(r.get("text", "")) for r in
                   (row.get("matches") or {}).get("results", [])])
        if not snippet.get("ok"):
            print(snippet.get("error") or "code generation failed",
                  file=sys.stderr)
            return 1
        record = dict(row)
        record["code"] = snippet
        record["code_verification"] = verify_code(
            snippet["code"], words[2],
            (row.get("matches") or {}).get("results", []),
            run_fn=_default_run_fn(context))
        record["reply"] = format_regex_reply(record)
        _print(record, as_json)
        return 0
    pattern = " ".join(words).strip()
    if not pattern:
        print("regex: give me a pattern — nm regex '<pattern>' "
              "--text '<text>'", file=sys.stderr)
        return 2
    record = run_regex_session(
        home, pattern=pattern, flavour=flavour, flags=flags,
        texts=texts, run_fn=_default_run_fn(context), store=store)
    if not record.get("ok"):
        print(record.get("error") or "regex failed", file=sys.stderr)
        return 1
    record["reply"] = format_regex_reply(record)
    _print(record, as_json)
    return 0
