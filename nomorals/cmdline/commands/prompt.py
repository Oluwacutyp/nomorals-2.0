"""``nm prompt`` — the DSPy-style prompt compiler from the CLI.

Compiles a task's prompt from its graded journal (best-graded outputs
as few-shot demos, worst as counter-examples, the instruction refined
from the grades) into a versioned prompt in the prompt store. See
nomorals/llm/prompt_compiler.py.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

from ...tools.prompt_compiler import TASKS
from ..emit import _emit


def _settings(context: Any) -> Any:
    settings = getattr(context, "settings", None)
    if settings is not None:
        value = getattr(settings, "prompt_compiler", None)
        if value is not None:
            return value
    from ...core.config import load_settings

    return load_settings().prompt_compiler


def _brain_prompt_runner(context: Any, timeout_s: float):
    """The live model as a compile-validation runner (temperature 0)."""
    from ...llm.base import Message, SamplingParams
    from ...llm.brain import brain_for

    brain = brain_for(context)

    def run(prompt: str) -> str:
        response = brain.chat(
            [Message.user(prompt)],
            SamplingParams(temperature=0.0, max_tokens=512),
            task_kind="prompt-compile",
            timeout_s=float(timeout_s),
        )
        if not response.ok or not (response.text or "").strip():
            raise RuntimeError(
                f"brain compile run failed "
                f"({response.failure_class or 'unknown'}): "
                f"{response.error or 'no error detail'}")
        return str(response.text)

    return run


def _cmd_prompt(args: argparse.Namespace, context: Any) -> int:
    """Route `nm prompt` to compile / status / list / show."""
    action = getattr(args, "prompt_action", "") or ""
    try:
        if action == "compile":
            return _prompt_compile(args, context)
        if action == "status":
            return _prompt_status(args, context)
        if action == "list":
            return _prompt_list(args, context)
        if action == "show":
            return _prompt_show(args, context)
    except ValueError as exc:
        print(f"prompt: {exc}", file=sys.stderr)
        return 2
    print("usage: nm prompt {compile|status|list|show} ...", file=sys.stderr)
    return 2


def _prompt_compile(args: argparse.Namespace, context: Any) -> int:
    task = str(getattr(args, "task", "") or "")
    entry = TASKS.get(task)
    if entry is None:
        known = ", ".join(sorted(TASKS))
        print(f"prompt: unknown task {task!r} (known: {known})",
              file=sys.stderr)
        return 2
    module_name, func_name, required = entry
    ids: dict[str, str] = {}
    for key in required:
        value = str(getattr(args, key, "") or "")
        if not value.strip():
            print(f"prompt: {task} needs --{key.replace('_', '-')}",
                  file=sys.stderr)
            return 2
        ids[key] = value
    suggest = None
    metric = None
    if bool(getattr(args, "validate", False)):
        from ...llm.prompt_compiler import imitation_metric

        suggest = _brain_prompt_runner(
            context, float(getattr(args, "timeout", 30.0) or 30.0))
        metric = imitation_metric
    module = importlib.import_module(module_name)
    supplier = getattr(module, func_name)
    outcome = supplier(suggest=suggest, metric=metric,
                       settings=_settings(context), **ids)
    lines = [f"prompt {task}: {outcome.status} — {outcome.reason}"]
    if outcome.compiled is not None:
        lines.append(f"version {outcome.compiled.version} "
                     f"({len(outcome.compiled.prompt_text)} chars, "
                     f"{len(outcome.compiled.demos)} demos, "
                     f"{len(outcome.compiled.counter_examples)} "
                     f"counter-examples)")
    _emit(args, {"task": task, "status": outcome.status,
                 "reason": outcome.reason,
                 "n_examples": outcome.n_examples,
                 "report": (outcome.report.to_dict()
                            if outcome.report else {}),
                 "version": (outcome.compiled.version
                             if outcome.compiled else None)},
          "\n".join(lines))
    return 0 if outcome.status == "compiled" else 1


def _prompt_status(args: argparse.Namespace, context: Any) -> int:
    from ...llm.prompt_compiler import PromptStore, control_prompt

    task = str(getattr(args, "task", "") or "")
    if task and task not in TASKS:
        print(f"prompt: unknown task {task!r} "
              f"(known: {', '.join(sorted(TASKS))})", file=sys.stderr)
        return 2
    store = PromptStore()
    text = control_prompt(f"status {task}" if task else "status",
                          store=store)
    data: dict[str, Any] = {"task": task or "all",
                            "tasks": store.list_tasks()}
    if task:
        compiled = store.load(task)
        if compiled is not None and compiled.report is not None:
            data["report"] = compiled.report.to_dict()
            data["version"] = compiled.version
    _emit(args, data, text)
    return 0


def _prompt_list(args: argparse.Namespace, context: Any) -> int:
    from ...llm.prompt_compiler import PromptStore, control_prompt

    store = PromptStore()
    text = control_prompt("list", store=store)
    _emit(args, {"tasks": store.list_tasks()}, text)
    return 0


def _prompt_show(args: argparse.Namespace, context: Any) -> int:
    from ...llm.prompt_compiler import PromptStore

    task = str(getattr(args, "task", "") or "")
    if task not in TASKS:
        print(f"prompt: unknown task {task!r} "
              f"(known: {', '.join(sorted(TASKS))})", file=sys.stderr)
        return 2
    store = PromptStore()
    version = int(getattr(args, "version", 0) or 0)
    compiled = (store.load_version(task, version) if version
                else store.load(task))
    if compiled is None:
        print(f"prompt: no compiled prompt for {task!r} yet — "
              f"run `nm prompt compile {task}` first", file=sys.stderr)
        return 1
    lines = [f"prompt {task} — v{compiled.version}"]
    if compiled.report is not None:
        rep = compiled.report
        lines.append(f"base grade {rep.baseline_grade:+.2f} -> demo grade "
                     f"{rep.demo_grade:+.2f} "
                     f"({'validated' if rep.validated else 'unvalidated'})")
    lines.append("")
    lines.append(compiled.prompt_text)
    _emit(args, {"task": task, "version": compiled.version,
                 "prompt_text": compiled.prompt_text,
                 "report": (compiled.report.to_dict()
                            if compiled.report else {})},
          "\n".join(lines))
    return 0
