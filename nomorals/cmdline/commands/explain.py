"""``nm explain`` — explain a shell command fragment by fragment.

    nm explain tar -czf archive.tgz dir [--json]
    nm explain list [--json]
    nm explain show <id> [--json]
    nm explain ingest [--man-root <dir>] [--json]
    nm explain status [--json]

The same explainer organ chat /explain uses: each fragment of the
command is annotated with the exact help text for that fragment,
grounded in the man pages of the binaries actually installed on
this host, in ``--help`` output harvested in the sandbox, or in
tldr pages (with --online). A fragment with no grounded doc is
marked unknown, never filled in silently. The explainer itself
never executes the command; ``nm explain run`` is the explicit
explain-before-run workflow (explain first, then run in the
sandbox through ``run_sandboxed``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_explain"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _print(explanation: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(explanation, indent=2, default=str))
    else:
        print(explanation.get("reply") or explanation.get("error") or "")


def _default_help_fetcher():
    """``<cmd> --help`` harvested through the sandboxed runner."""
    from ...tools.shell import run_sandboxed

    def _fetch(command: str):
        name = (command or "").strip().split("/")[-1]
        if not name:
            return None
        try:
            result = run_sandboxed(
                f"{name} --help", cwd=".", timeout=15.0, network=False)
        except Exception:  # noqa: BLE001 — harvest is a miss
            return None
        if result.get("exit_code") != 0:
            return None
        text = (result.get("stdout") or "").strip()
        return text or None

    return _fetch


def _cmd_explain(args: argparse.Namespace, context: Any) -> int:
    from ...codews.explain import (
        explain_command,
        format_explanation_reply,
        open_explanation_store,
    )
    from ...codews.man_corpus import fetch_tldr_page, open_man_corpus

    home = _home(context)
    store = open_explanation_store(home)
    words = list(getattr(args, "explain_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    man_root = getattr(args, "man_root", None)
    online = bool(getattr(args, "online", False))
    verb = (words[0].lower() if words else "")
    if verb == "list":
        rows = store.list()
        if as_json:
            print(json.dumps({"ok": True, "explanations": rows},
                             indent=2, default=str))
        elif not rows:
            print("no explanations yet — explain a command with: "
                  "nm explain <shell command>")
        else:
            for row in rows:
                print(f"{row['explanation_id']} — {row['command']}")
        return 0
    if verb == "show":
        if len(words) < 2:
            print("explain: give me an explanation id — "
                  "nm explain show <id>", file=sys.stderr)
            return 2
        row = store.get(words[1])
        if row is None:
            print(f"no explanation “{words[1]}”", file=sys.stderr)
            return 1
        _print({**row, "ok": True,
                "reply": format_explanation_reply({**row, "ok": True})},
               as_json)
        return 0
    if verb == "ingest":
        roots = [man_root] if man_root else None
        corpus = open_man_corpus(home)
        try:
            stats = corpus.ingest(roots)
        finally:
            corpus.close()
        if as_json:
            print(json.dumps({"ok": True, **stats}, indent=2,
                             default=str))
        else:
            print(f"man corpus ingested: {stats['pages']} pages, "
                  f"{stats['options']} options "
                  f"({stats['skipped']} unchanged)")
        return 0
    if verb == "status":
        corpus = open_man_corpus(home)
        try:
            status = corpus.status()
        finally:
            corpus.close()
        status["explanations"] = len(store.list())
        if as_json:
            print(json.dumps({"ok": True, **status}, indent=2,
                             default=str))
        else:
            print(f"man corpus: {status['pages']} pages, "
                  f"{status['options']} options")
            print(f"stored at {status['path']}")
            print(f"explanations stored: {status['explanations']}")
        return 0
    if verb == "run":
        command = " ".join(words[1:]).strip()
        if not command:
            print("explain: give me a command to explain and run — "
                  "nm explain run <shell command>", file=sys.stderr)
            return 2
        explanation = explain_command(
            home, command, store=store,
            help_fetcher=_default_help_fetcher(),
            tldr_fetcher=fetch_tldr_page if online else None)
        _print(explanation, as_json)
        if not explanation.get("ok"):
            return 1
        from ...tools.shell import run_sandboxed

        try:
            result = run_sandboxed(command, cwd=".", timeout=120.0,
                                   network=False)
        except Exception as exc:  # noqa: BLE001 — run failure is rc 1
            print(f"run failed: {exc}", file=sys.stderr)
            return 1
        if not as_json:
            print("--- run ---")
            print(result.get("stdout") or "")
            if result.get("stderr"):
                print(result["stderr"], file=sys.stderr)
        return 0 if result.get("exit_code") == 0 else 1
    command = " ".join(words).strip()
    if not command:
        print("explain: give me a shell command — "
              "nm explain <shell command>", file=sys.stderr)
        return 2
    explanation = explain_command(
        home, command, store=store,
        help_fetcher=_default_help_fetcher(),
        tldr_fetcher=fetch_tldr_page if online else None)
    _print(explanation, as_json)
    return 0 if explanation.get("ok") else 1
