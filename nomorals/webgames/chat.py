"""Chat surface. Four verbs; nothing else is configurable by design."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .runner import RunConfig, get_runner

_log = get_logger(__name__)
USAGE = """/game — autonomous web-game farming (Lagos Life default)

  /game target <handle>     who the bonus goes to (default death4all)
  /game amount <n>          bonus size (default 1,050,000)
  /game slots <n>           parallel slots (default 2)
  /game start | stop | status | provision
"""


def handle_game(args: str, context: Any) -> str:
    home = Path(getattr(context, "home", ".") or ".")
    runner = get_runner(home)
    parts = (args or "").split()
    verb = parts[0].lower() if parts else "status"
    rest = parts[1:]

    if verb in ("help", "?"):
        return USAGE
    if verb == "target":
        if not rest:
            return f"target is {runner.config.target}"
        runner.config.target = rest[0].lstrip("@")
        runner._stats["target_id"] = ""
        return f"target set to @{runner.config.target}"
    if verb == "amount":
        try:
            runner.config.amount = max(1, int(rest[0]))
        except (IndexError, ValueError):
            return "amount needs a number"
        return f"amount set to {runner.config.amount:,}"
    if verb == "slots":
        try:
            runner.config.slots = max(1, min(8, int(rest[0])))
        except (IndexError, ValueError):
            return "slots needs a number 1-8"
        return f"slots set to {runner.config.slots}"
    if verb == "provision":
        from .driver import ensure_browser
        ensure_browser()
        return "browser provisioned"
    if verb == "start":
        result = runner.start()
        return (f"already running ({result['slots']} slots)"
                if result.get("already") else
                f"started {result['slots']} slot(s) → @{result['target']}")
    if verb == "stop":
        return f"stopped {runner.stop()['slots']} slot(s)"
    if verb == "status":
        s = runner.status()
        return (f"{'running' if s['running'] else 'idle'} · target "
                f"@{s['target']} · {s['amount']:,} each · slots {s['slots']}\n"
                f"created {s['created']} · gifted {s['gifted']} · "
                f"failed {s['failed']} · walls {s['walls']} · "
                f"moved {s['value']:,}")
    return USAGE
