"""The ``computer`` tool: Devon's own desktop, brain-callable.

Actions: status | open | close | sessions | perceive | act | run |
takeover | handback | stop | resume.

The tool is a thin face on nomorals/computer: persistent desktop
sessions on the host she runs on, driven through her own bridge
(raw X11 on Linux, raw Win32 on Windows — no third-party
automation). ``act`` performs ONE gated action; ``run`` hands the
desktop to the brain for a goal under the verified control loop.
The personal-surface gate binds here exactly as in chat: a sticky
owner stop, an owner takeover, or a degraded brain (this context's
model power not fully healthy) makes acting refuse — watching
(perceive/status) always works.
"""

from __future__ import annotations

import json
from typing import Any


def _service(registry: Any) -> Any:
    from ..computer.service import default_service

    context = getattr(registry, "context", None)
    return default_service(context=context)


def register(registry: Any) -> None:
    from ..core.policy import Capability

    @registry.register(
        "computer",
        description=(
            "Devon's own computer: a persistent desktop on the host "
            "she runs on, driven end-to-end (screenshot, windows, "
            "mouse, keyboard, app launch/focus). Sessions survive "
            "between tasks. Actions: status (availability + "
            "sessions + gate), open (new session), close, sessions, "
            "perceive (screenshot + window list + pointer — always "
            "allowed), act (ONE action: kind click|move|drag|scroll|"
            "key|type|launch|focus|close_window|kill with args), "
            "run (the brain drives toward a goal under the verified "
            "control loop), takeover/handback (owner control), "
            "stop/resume (the owner's sticky stop). Acting is "
            "refused while the owner has stopped or taken over the "
            "desktop, or while the model power in this context is "
            "degraded — observe-only then."
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "action": "str — status|open|close|sessions|perceive|act|run|takeover|handback|stop|resume",
            "session": "str — session id (default: latest open session)",
            "kind": "str — for act: click|move|drag|scroll|key|type|launch|focus|close_window|kill",
            "args": "str — for act: JSON object of action args, e.g. {\"x\": 100, \"y\": 50} or {\"text\": \"hello\"} or {\"keys\": \"ctrl+l\"} or {\"argv\": [\"xterm\"]}",
            "goal": "str — for run: what the desktop should achieve",
            "mission": "str — for open: the owning mission label",
            "max_steps": "int — for run: step budget (default 25)",
        },
    )
    def computer(action: str = "status", session: str = "",
                 kind: str = "", args: str = "", goal: str = "",
                 mission: str = "", max_steps: int = 25,
                 **_: Any) -> dict[str, Any]:
        svc = _service(registry)
        verb = (action or "status").strip().lower()
        if verb == "status":
            return {"ok": True, "text": svc.status_text(),
                    **svc.status()}
        if verb == "sessions":
            rows = [rec.describe() for rec in svc.manager.sessions()]
            return {"ok": True, "sessions": rows}
        if verb == "open":
            return svc.open(mission=mission)
        if verb == "close":
            return svc.close(session)
        if verb == "perceive":
            return svc.perceive(session)
        if verb == "act":
            parsed: dict[str, Any] = {}
            if args:
                try:
                    decoded = json.loads(args)
                except ValueError as exc:
                    return {"ok": False,
                            "error": f"args is not valid JSON: {exc}"}
                if not isinstance(decoded, dict):
                    return {"ok": False,
                            "error": "args must be a JSON object"}
                parsed = decoded
            if not kind:
                return {"ok": False,
                        "error": "act needs a kind (click|type|key|...)"}
            return svc.act(kind, parsed, session)
        if verb == "run":
            if not goal:
                return {"ok": False, "error": "run needs a goal"}
            return svc.run(goal, session,
                           max_steps=max(1, int(max_steps or 25)))
        if verb == "takeover":
            return svc.takeover(session, by="owner via tool")
        if verb == "handback":
            return svc.handback(session)
        if verb == "stop":
            return svc.stop(session)
        if verb == "resume":
            return svc.resume(session)
        return {"ok": False, "error": f"unknown computer action {verb!r}"}
