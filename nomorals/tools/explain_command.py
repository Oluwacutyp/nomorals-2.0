"""Brain tool: the shell-command explainer (ExplainShell core,
52-websites #43).

"what does this command do" / "explain `tar -czf …`" — paste a
shell command and get each fragment annotated with the exact help
text for that fragment, grounded in the man pages of the binaries
actually installed on this host, in ``--help`` output harvested
in the sandbox, or in tldr community pages. A fragment with no
grounded doc is marked unknown, never filled in silently.

The tool is a thin shell over ``nomorals.codews.explain``: the
organ owns the parser, the corpus and the matching, and never
executes anything. Two rungs are wired here, both injected into
the organ as fetchers: ``--help`` harvesting through the
sanctioned ``shell_run`` tool (tool-to-tool composition, the same
shape as ``dev_answer``'s ``run_code`` wiring) and tldr pages
through the organ's own urllib fetcher. ``explain_and_run`` is
the explain-before-run workflow: the explanation is produced
first, and only then is the command run in the sandbox through
``shell_run`` — the run result rides beside the explanation.
"""

from __future__ import annotations

from typing import Any

from ..codews.man_corpus import fetch_tldr_page
from ..core.policy import Capability
from .paper_reader import _home_from_context, _llm_from_context


def _shell_run_fn_from_registry(registry: Any):
    """A sandboxed-run callable over the ``shell_run`` tool, or None.

    ``registry.call`` returns an Outcome: ``.ok`` False means the
    runner is unavailable; ``.ok`` True carries the ``run_sandboxed``
    dict as ``.value``.
    """
    if registry is None:
        return None

    def _run(command: str, timeout: float = 30.0):
        try:
            outcome = registry.call(
                "shell_run", command=command, timeout=timeout)
        except Exception:  # noqa: BLE001 — a dead runner is a miss
            return None
        if not getattr(outcome, "ok", False):
            return None
        value = getattr(outcome, "value", None)
        return value if isinstance(value, dict) else None

    return _run


def _help_fetcher_from_registry(registry: Any):
    """A ``<cmd> --help`` harvester over ``shell_run``, or None.

    Runs ONLY ``<cmd> --help`` in the sandbox — never the command
    being explained — and returns its stdout when the run exits 0.
    """
    run = _shell_run_fn_from_registry(registry)
    if run is None:
        return None

    def _fetch(command: str):
        name = (command or "").strip().split("/")[-1]
        if not name:
            return None
        result = run(f"{name} --help", timeout=15.0)
        if not result or result.get("exit_code") != 0:
            return None
        text = (result.get("stdout") or "").strip()
        return text or None

    return _fetch


def register(registry: Any) -> None:
    """Attach the ``explain_command`` tool to a registry."""
    context = registry.context

    @registry.register(
        "explain_command",
        description=(
            "Shell-command explainer (ExplainShell core) — paste a "
            "shell command and get each fragment annotated with the "
            "exact help text for that fragment: the command, every "
            "option, arguments, pipes and redirects, grounded in the "
            "man pages of the binaries actually installed on this "
            "host, in --help output, or in tldr pages. A fragment "
            "with no grounded doc is marked unknown, never filled "
            "in silently. Ask 'what does this command do' or "
            "'explain `tar -czf …`'. action=<explain|ingest|status|"
            "list|show|explain_and_run>; explain takes command=<the "
            "shell command>; ingest rebuilds the man corpus "
            "(man_roots=<comma-separated man roots>, force=<true>); "
            "explain_and_run explains first and then runs the "
            "command in the sandbox, returning the explanation "
            "beside the run result. The explainer itself never "
            "executes anything. Returns ok, explanation_id, "
            "command, fragments (each with doc/unknown/suggestion), "
            "summary, reply, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def explain_command_tool(
        command: str = "",
        action: str = "explain",
        man_roots: str = "",
        force: bool = False,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        from ..codews.explain import (
            explain_command,
            format_explanation_reply,
            open_explanation_store,
        )
        from ..codews.man_corpus import open_man_corpus

        home = _home_from_context(context)
        store = open_explanation_store(home)
        act = (action or "explain").strip().lower()
        if act == "list":
            return {"ok": True, "explanations": store.list()}
        if act == "show":
            row = store.get((command or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored explanation "
                                 f"'{command}'"}
            shown = dict(row)
            shown["ok"] = True
            shown["reply"] = format_explanation_reply(
                {**row, "ok": True})
            return shown
        if act == "status":
            corpus = open_man_corpus(home)
            try:
                status = corpus.status()
            finally:
                corpus.close()
            status["ok"] = True
            status["explanations"] = len(store.list())
            return status
        if act == "ingest":
            roots = [r.strip() for r in (man_roots or "").split(",")
                     if r.strip()] or None
            corpus = open_man_corpus(home)
            try:
                stats = corpus.ingest(roots, force=bool(force))
            finally:
                corpus.close()
            stats["ok"] = True
            return stats
        text = (command or "").strip()
        if not text:
            return {"ok": False,
                    "error": "explain needs a command"}
        help_fetcher = _help_fetcher_from_registry(registry)
        explanation = explain_command(
            home, text, store=store,
            llm_fn=_llm_from_context(context),
            help_fetcher=help_fetcher,
            tldr_fetcher=fetch_tldr_page)
        if act == "explain_and_run":
            if not explanation.get("ok"):
                return explanation
            run = _shell_run_fn_from_registry(registry)
            if run is None:
                explanation["run"] = None
                explanation["run_error"] = (
                    "shell_run is not available on this registry — "
                    "the explanation stands, nothing was run")
                return explanation
            result = run(text, timeout=float(timeout or 120.0))
            if result is None:
                explanation["run"] = None
                explanation["run_error"] = (
                    "shell_run refused the command — the explanation "
                    "stands, nothing was run")
                return explanation
            explanation["run"] = result
            return explanation
        return explanation
