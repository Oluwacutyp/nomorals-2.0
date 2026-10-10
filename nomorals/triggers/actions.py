"""Trigger action executors.

Each action is ``execute(trigger, engine) -> dict``.  All reuse existing
subsystems — nothing here reinvents a send path:

* ``notify`` — the owner-only proactive channel (``agents.notifier``),
  ``critical=True`` so trigger alerts bypass quiet hours like an alarm.
* ``message`` — a chat send via an injected sender.  The live runtime
  binds ``PartnerRuntime.say``; with no sender bound the action fails
  fast with a clear error instead of pretending to send.
* ``command`` — runs an ``nm`` command in a subprocess.  ``cmdline`` is
  L7 and triggers is L5, so this deliberately does *not* import the CLI
  — it spawns ``sys.executable -m nomorals …`` (not an import, so the
  layering test stays green).
* ``mission`` — starts a mission through ``missions.runner.MissionRunner``.

Evidence templates: ``title``/``body``/``text``/``goal`` may contain
``{{dotted.path}}`` placeholders resolved against the fire evidence
(Huginn Liquid parity) — e.g. ``"BTC is now {{price}}"``.  Missing keys
render as ``""``; never raises.

Digest mode: ``notify``/``message`` params accept ``digest: true`` —
instead of sending immediately, the rendered alert is buffered in the
store and ``TriggerEngine.flush_digests()`` sends one combined message
(Huginn digest-agent parity).  ``digest_every_s`` (default 3600) and
``digest_max`` (default 25) control the flush.

Every executor is dependency-injected on the engine (``notify_fn``,
``send_message``, ``run_command``, ``start_mission``) so tests can use
fakes and production binds the real paths.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .models import (
    ACTION_COMMAND,
    ACTION_MESSAGE,
    ACTION_MISSION,
    ACTION_NOTIFY,
    Trigger,
    TriggerError,
)

_log = get_logger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}")


def _resolve_path(evidence: dict[str, Any], path: str) -> Any:
    node: Any = evidence
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return ""
    if node is None:
        return ""
    if isinstance(node, (dict, list)):
        import json as _json
        try:
            return _json.dumps(node)[:500]
        except Exception:  # noqa: BLE001
            return str(node)[:500]
    return node


def render_template(text: str, evidence: dict[str, Any] | None) -> str:
    """Render ``{{dotted.path}}`` placeholders against fire evidence.

    Missing paths render as ``""``.  Never raises — a broken template
    must not break a firing action.
    """
    if not text or "{{" not in text:
        return text or ""
    evidence = evidence or {}

    def _sub(m: "re.Match[str]") -> str:
        try:
            return str(_resolve_path(evidence, m.group(1)))
        except Exception:  # noqa: BLE001 - templates never break firing
            return ""

    try:
        return _TEMPLATE_RE.sub(_sub, text)
    except Exception:  # noqa: BLE001
        return text


def _evidence_of(engine: Any) -> dict[str, Any]:
    """The evidence of the fire currently being executed (set by the
    engine around each action call)."""
    return dict(getattr(engine, "_current_evidence", None) or {})


# ── notify ─────────────────────────────────────────────────────────────────

def default_notify(trigger: Trigger, title: str, body: str,
                   engine: Any) -> dict[str, Any]:
    """Send via the existing owner-only notifier (critical, like an alarm)."""
    from ..agents.notifier import notify as _notify

    context = engine.context
    if context is None:
        raise TriggerError(
            "notify action needs an engine context (none bound)")
    # force=True: the trigger's own cooldown governs repetition — the
    # notifier's dedupe window must not swallow a re-fire the owner asked
    # to be told about.
    return _notify(context, "trigger", title, body or "",
                   critical=True, force=True)


def execute_notify(trigger: Trigger, engine: Any) -> dict[str, Any]:
    params = trigger.action_params
    evidence = _evidence_of(engine)
    title = render_template(str(params.get("title") or trigger.name
                                or trigger.id), evidence)
    body = render_template(str(params.get("body") or ""), evidence)
    if params.get("digest"):
        buffered = engine.store.digest_append(trigger.id, title, body)
        _log.info("trigger %s digested notify (%d buffered)",
                  trigger.id, buffered)
        return {"digested": True, "buffered": buffered,
                "digest_every_s": float(params.get("digest_every_s") or 3600),
                "digest_max": int(params.get("digest_max") or 25)}
    fn = engine.notify_fn or default_notify
    return dict(fn(trigger, title, body, engine) or {})


# ── message ────────────────────────────────────────────────────────────────

def execute_message(trigger: Trigger, engine: Any) -> dict[str, Any]:
    params = trigger.action_params
    evidence = _evidence_of(engine)
    chat = str(params["chat"])
    text = render_template(str(params["text"]), evidence)
    if params.get("digest"):
        buffered = engine.store.digest_append(trigger.id, "", text)
        _log.info("trigger %s digested message (%d buffered)",
                  trigger.id, buffered)
        return {"digested": True, "buffered": buffered, "chat": chat,
                "digest_every_s": float(params.get("digest_every_s") or 3600),
                "digest_max": int(params.get("digest_max") or 25)}
    sender: Callable[[str, str], Any] | None = engine.send_message
    if sender is None:
        raise TriggerError(
            "message action has no sender bound — start the engine with "
            "TriggerEngine(..., send_message=...) (the live runtime binds "
            "PartnerRuntime.say)")
    result = sender(chat, text)
    return {"chat": chat, "sent": True,
            "result": None if result is None else str(result)[:200]}


def format_digest(trigger: Trigger,
                  items: list[dict[str, Any]]) -> tuple[str, str]:
    """Combine buffered digest lines into one (title, body) pair."""
    name = trigger.name or trigger.id
    title = f"📦 {name}: {len(items)} alert(s)"
    lines = []
    for it in items:
        head = f"• {it['title']}: " if it.get("title") else "• "
        lines.append(f"{head}{it['text']}".rstrip())
    return title, "\n".join(lines)[:4000]


# ── command ────────────────────────────────────────────────────────────────

def build_command_argv(params: dict[str, Any]) -> list[str]:
    """Render action params into ``[python, -m, nomorals, ...argv]``."""
    argv = params.get("argv")
    if argv is not None:
        rest = list(argv)
    else:
        rest = shlex.split(str(params.get("command") or "").strip())
    if not rest:
        raise TriggerError("command action resolved to an empty argv")
    return [sys.executable, "-m", "nomorals", *rest]


def default_run_command(argv: list[str], timeout_s: float) -> dict[str, Any]:
    """Run the argv as a subprocess; non-zero exit is a TriggerError."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TriggerError(
            f"command timed out after {timeout_s:g}s: "
            f"{' '.join(argv[3:])}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise TriggerError(
            f"command exited {proc.returncode} "
            f"({' '.join(argv[3:])}): {tail}")
    return {"exit_code": 0,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:]}


def execute_command(trigger: Trigger, engine: Any) -> dict[str, Any]:
    params = trigger.action_params
    argv = build_command_argv(params)
    timeout_s = float(params.get("timeout_s", 120) or 120)
    fn = engine.run_command or default_run_command
    out = dict(fn(argv, timeout_s) or {})
    out.setdefault("argv", argv[3:])
    return out


# ── mission ────────────────────────────────────────────────────────────────

def default_start_mission(goal: str, max_iterations: int,
                          engine: Any) -> dict[str, Any]:
    """Start a mission through the wired MissionRunner (os hook attached)."""
    from ..missions.wiring import wired_runner

    context = engine.context
    if context is None:
        raise TriggerError(
            "mission action needs an engine context (none bound)")
    runner = wired_runner(context, milestones=False)
    result = runner.start(goal, max_iterations=max_iterations)
    return {"mission_id": getattr(result, "mission_id", ""),
            "status": getattr(result, "status", ""),
            "success": bool(getattr(result, "success", False))}


def execute_mission(trigger: Trigger, engine: Any) -> dict[str, Any]:
    params = trigger.action_params
    goal = render_template(str(params["goal"]), _evidence_of(engine))
    max_iterations = int(params.get("max_iterations") or 8)
    fn = engine.start_mission or default_start_mission
    return dict(fn(goal, max_iterations, engine) or {})


ACTION_HANDLERS: dict[str, Callable[[Trigger, Any], dict[str, Any]]] = {
    ACTION_NOTIFY: execute_notify,
    ACTION_MESSAGE: execute_message,
    ACTION_COMMAND: execute_command,
    ACTION_MISSION: execute_mission,
}
