"""``nm schedule`` — CLI mirror of the ``/schedule`` chat command.

Manages the same in-process cron-style jobs (the ``schedule_jobs`` table
the chat command writes to): at / every / daily, with
``message | tool | command`` actions.

    nm schedule list [--json]
    nm schedule add <name> <when> message <text...>
    nm schedule add <name> <when> tool <tool-name> [<json-args>]
    nm schedule add <name> <when> command <shell-command...>
    nm schedule rm <name-or-id>
    nm schedule enable|disable <name-or-id>
    nm schedule run <name-or-id>

``when``: 'at 2026-12-25 09:00' | 'every 30m' | 'daily 02:00' | '22:00' | '30m'

The ``add`` argument parsing is identical to ``/schedule add``: every word
after ``<name>`` up to the first ``message|tool|command`` keyword is the
``when`` spec, and the rest is the payload.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def _scheduler(context: Any):
    from ...agents.scheduler import Scheduler

    return Scheduler(context)


def _usage() -> int:
    print("usage: nm schedule list [--json]\n"
          "       nm schedule add <name> <when> message <text...> |\n"
          "                              tool <tool-name> [<json-args>] |\n"
          "                              command <shell-command...>\n"
          "       nm schedule rm <name-or-id>\n"
          "       nm schedule enable|disable <name-or-id>\n"
          "       nm schedule run <name-or-id>\n"
          "when: 'at 2026-12-25 09:00' | 'every 30m' | 'daily 02:00' | "
          "'22:00' | '30m'",
          file=sys.stderr)
    return 2


def _cmd_schedule(args: Any, context: Any) -> int:
    """Route ``nm schedule <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        return _usage()
    verb = words[0].lower()
    try:
        if verb in ("list", "status"):
            return _schedule_list(args, context)
        if verb == "add":
            return _schedule_add(context, words[1:])
        if verb in ("rm", "remove"):
            return _schedule_remove(context, words[1:])
        if verb in ("enable", "disable"):
            return _schedule_enable(context, words[1:], enabled=(verb == "enable"))
        if verb == "run":
            return _schedule_run(context, words[1:])
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown schedule verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _schedule_list(args: Any, context: Any) -> int:
    sched = _scheduler(context)
    jobs = sched.list_jobs()
    if _as_json(args):
        print(json.dumps(jobs, indent=2, default=str))
        return 0
    if not jobs:
        print("no scheduled jobs — nm schedule add <name> <when> message <text>")
        return 0
    print("scheduled jobs:")
    for job in jobs[:15]:
        state = "on " if job["enabled"] else "off"
        nxt = job.get("next_run_iso") or ("past" if job["kind"] == "at" else "—")
        print(f"  [{state}] {job['name']} — {job['kind']} {job['spec']} (next: {nxt})")
        if job.get("last_result"):
            print(f"        last: {job['last_result'][:80]}")
    return 0


def _schedule_add(context: Any, parts: list[str]) -> int:
    """parts = the tail after 'add': [name, spec..., message|tool|command, ...]."""
    if len(parts) < 3:
        print("usage: nm schedule add <name> <when> message <text...> |\n"
              "                              tool <tool-name> [<json-args>] |\n"
              "                              command <shell-command...>\n"
              "  e.g. nm schedule add goodnight 22:00 message goodnight",
              file=sys.stderr)
        return 2
    name = parts[0]
    rest = parts[1:]
    payload_kind = ""
    split_at = -1
    for i, tok in enumerate(rest):
        if tok.lower() in {"message", "tool", "command"}:
            payload_kind = tok.lower()
            split_at = i
            break
    if split_at < 0:
        print("add needs an action: message <text> | tool <name> [<json-args>] | "
              "command <cmd>", file=sys.stderr)
        return 2
    spec = " ".join(rest[:split_at])
    payload_parts = rest[split_at + 1:]
    payload: dict[str, Any]
    if payload_kind == "message":
        payload = {"text": " ".join(payload_parts)}
    elif payload_kind == "tool":
        if not payload_parts:
            print("tool action needs: <tool name> [<json-args>]", file=sys.stderr)
            return 2
        args: dict[str, Any] = {}
        if len(payload_parts) > 1:
            try:
                parsed = json.loads(" ".join(payload_parts[1:]))
                if isinstance(parsed, dict):
                    args = parsed
            except (ValueError, TypeError):
                print('tool args must be a JSON object, e.g. {"query":"ai news"}',
                      file=sys.stderr)
                return 2
        payload = {"tool": payload_parts[0], "args": args}
    else:
        if not payload_parts:
            print("command action needs the command text — e.g.\n"
                  "  nm schedule add backup every 1h command python3 backup.py",
                  file=sys.stderr)
            return 2
        payload = {"command": " ".join(payload_parts)}
    sched = _scheduler(context)
    try:
        job = sched.add(name, spec, payload_kind, payload)
    except (ValueError, RuntimeError) as exc:
        print(f"scheduling failed: {exc}", file=sys.stderr)
        return 1
    when = time.strftime("%m-%d %H:%M", time.localtime(job["next_run"]))
    print(f"scheduled {job['name']} — {job['kind']} ({spec}) — next {when}")
    return 0


def _schedule_remove(context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm schedule rm <name-or-id>", file=sys.stderr)
        return 2
    from ...core.errors import AmbiguousRef

    ref = " ".join(rest)
    sched = _scheduler(context)
    try:
        removed = sched.remove(ref)
    except AmbiguousRef as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not removed:
        print(f"no job named {ref!r} — nm schedule list to see the jobs",
              file=sys.stderr)
        return 1
    print("removed.")
    return 0


def _schedule_enable(context: Any, rest: list[str], enabled: bool) -> int:
    verb = "enable" if enabled else "disable"
    if not rest:
        print(f"usage: nm schedule {verb} <name-or-id>", file=sys.stderr)
        return 2
    from ...core.errors import AmbiguousRef

    ref = " ".join(rest)
    sched = _scheduler(context)
    try:
        job = sched.set_enabled(ref, enabled=enabled)
    except (LookupError, AmbiguousRef) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{job['name']} {'enabled' if job['enabled'] else 'disabled'} "
          f"(next {job.get('next_run_iso') or '—'}).")
    return 0


def _schedule_run(context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm schedule run <name-or-id>", file=sys.stderr)
        return 2
    from ...core.errors import AmbiguousRef

    ref = " ".join(rest)
    sched = _scheduler(context)
    try:
        outcome = sched.run_now(ref)
    except (LookupError, AmbiguousRef) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"ran {outcome['name']}: {outcome['result'][:400]}")
    return 0
