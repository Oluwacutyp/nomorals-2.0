"""Prompt-compiler tools for the registry (DSPy shape, Devon-built).

``prompt_compile`` compiles a task's prompt from its graded journal
(the same rows the task's learner trains on) into a versioned prompt
in the prompt store; ``prompt_status`` reports the latest compiled
version and its before/after numbers. See
:mod:`nomorals.llm.prompt_compiler` for the compiler itself.

This module is a tool SURFACE over the L5 organs that own the
journals (``games.npc_learn``, ``characters.learn``) — it composes
them the way ``tools/games.py`` composes ``games.*`` — so it is
pinned at L5 in ``tests/test_layering.py`` (MODULE_PINS). The
registry loads it dynamically by dotted name (the same contract as
every builtin), never by static import.

Validation (``validate=True``) runs the base and compiled prompts on
held-out inputs through the model and scores them — the model runner
is attached with :func:`set_model_runner` (the trace_view seam: the
CLI attaches the live brain, tests attach a fake). With no runner
attached, a validation request compiles unvalidated and says so.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["register", "set_model_runner", "get_model_runner",
           "TASKS"]

#: Task name -> (supplier import path, supplier function, required ids).
#: The supplier functions live in the organs that own the journals.
TASKS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "npc_dialogue": ("nomorals.games.npc_learn", "compile_npc_prompt",
                     ("game_id", "npc_id")),
    "character_dialogue": ("nomorals.characters.learn",
                           "compile_character_prompt", ("char_id",)),
}

_MODEL_RUNNER: Callable[[str], str] | None = None


def set_model_runner(runner: Callable[[str], str] | None) -> None:
    """Attach the model runner used for ``validate=True`` compiles."""
    global _MODEL_RUNNER
    _MODEL_RUNNER = runner


def get_model_runner() -> Callable[[str], str] | None:
    """The attached model runner, or None."""
    return _MODEL_RUNNER


def _settings(context: Any) -> Any:
    settings = getattr(context, "settings", None)
    if settings is not None:
        value = getattr(settings, "prompt_compiler", None)
        if value is not None:
            return value
    from ..core.config import load_settings

    return load_settings().prompt_compiler


def _compile_task(task: str, ids: dict[str, str], *, validate: bool,
                  context: Any) -> dict[str, Any]:
    import importlib

    from ..core.errors import ToolError
    from ..llm.prompt_compiler import imitation_metric

    entry = TASKS.get(task)
    if entry is None:
        known = ", ".join(sorted(TASKS))
        raise ToolError(f"unknown prompt task {task!r} "
                        f"(known: {known})")
    module_name, func_name, required = entry
    missing = [key for key in required if not (ids.get(key) or "").strip()]
    if missing:
        raise ToolError(f"prompt_compile {task}: missing "
                        f"{', '.join(missing)}")
    try:
        module = importlib.import_module(module_name)
        supplier = getattr(module, func_name)
    except (ImportError, AttributeError) as exc:
        raise ToolError(f"prompt_compile {task}: supplier unavailable: "
                        f"{exc}") from exc
    suggest = None
    metric = None
    no_runner = False
    if validate:
        suggest = get_model_runner()
        if suggest is None:
            no_runner = True
        else:
            metric = imitation_metric
    kwargs: dict[str, Any] = {"suggest": suggest, "metric": metric,
                              "settings": _settings(context)}
    for key in required:
        kwargs[key] = ids[key]
    outcome = supplier(**kwargs)
    result: dict[str, Any] = {
        "task": task, "status": outcome.status,
        "reason": outcome.reason,
        "n_examples": outcome.n_examples,
        "report": outcome.report.to_dict() if outcome.report else {},
    }
    if no_runner:
        result["validation_requested_no_runner"] = True
        result["reason"] += (" Validation was requested but no model "
                             "runner is attached, so this compile is "
                             "unvalidated.")
    if outcome.compiled is not None:
        result["version"] = outcome.compiled.version
        text = outcome.compiled.prompt_text or ""
        result["prompt_excerpt"] = (
            text[:2000] + ("…" if len(text) > 2000 else ""))
        result["prompt_chars"] = len(text)
    return result


def _status_task(task: str, context: Any) -> dict[str, Any]:
    from ..core.errors import ToolError
    from ..llm.prompt_compiler import PromptStore, control_prompt

    if task and task not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise ToolError(f"unknown prompt task {task!r} "
                        f"(known: {known})")
    store = PromptStore()
    text = control_prompt(f"status {task}" if task else "status",
                          store=store)
    latest = store.latest_version(task) if task else 0
    return {"task": task or "all", "status_text": text,
            "latest_version": latest,
            "tasks": store.list_tasks()}


def register(registry: Any) -> None:
    """Attach the prompt-compiler tools to a registry."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "prompt_compile",
        description="Compile a task's prompt from its graded journal "
                    "(DSPy-style: best-graded outputs as few-shot demos, "
                    "worst as counter-examples, instruction refined from "
                    "the grades) into a versioned prompt. Tasks: "
                    "npc_dialogue (needs game_id + npc_id), "
                    "character_dialogue (needs char_id). validate=true "
                    "re-runs base vs compiled on held-out inputs through "
                    "the model (needs an attached runner).",
        capability=Capability.MODEL_CALL,
    )
    def prompt_compile(task: str, game_id: str = "", npc_id: str = "",
                       char_id: str = "", validate: bool = False
                       ) -> dict[str, Any]:
        ids = {"game_id": game_id, "npc_id": npc_id, "char_id": char_id}
        return _compile_task(task, ids, validate=bool(validate),
                             context=context)

    @registry.register(
        "prompt_status",
        description="Status of the compiled prompts: the latest version "
                    "per task with its before/after grades (base grade "
                    "-> demo grade, validated or not). Pass a task for "
                    "one task, or nothing for all of them.",
        capability=Capability.FS_READ,
    )
    def prompt_status(task: str = "") -> dict[str, Any]:
        return _status_task((task or "").strip(), context)
