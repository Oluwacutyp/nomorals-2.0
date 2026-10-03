"""``nm session`` — inspect and manage OS sessions across surfaces."""

from __future__ import annotations

import argparse
from typing import Any

from ..emit import _emit


def _cmd_session(args: argparse.Namespace, context: Any) -> int:
    """``nm session list|show|end`` — the unified session layer.

    Every surface (Telegram, WhatsApp, Discord, CLI, TUI, API) attaches to
    one OS Session per chat. These commands inspect that layer.
    """
    from ...os.session_bridge import SessionBridge

    task = " ".join(args.task).strip() if args.task else "list"
    verb = task.split()[0] if task else "list"
    rest = task.split()[1:] if task else []

    db = getattr(context, "db", None)
    bridge = SessionBridge(db=db)
    store = bridge.store

    if verb == "list":
        sessions = store.list_active()
        _emit(args, {"sessions": [s.to_dict() for s in sessions]},
              "\n".join(
                  f"{s.id}  {s.frontend:10s} {s.principal:8s} "
                  f"{s.conversation_id:30s} mode={s.state.get('gating_mode', '?')}"
                  for s in sessions
              ) or "no active sessions")
        return 0

    if verb == "show":
        if not rest:
            print("usage: nm session show <session-id>", flush=True)
            return 2
        session = store.get(rest[0])
        if session is None:
            print(f"no such session: {rest[0]}", flush=True)
            return 1
        d = session.to_dict()
        text = "\n".join(
            f"{key:16s} {d[key]}"
            for key in ("id", "principal", "frontend", "project_id",
                        "conversation_id", "created_at", "updated_at",
                        "ended_at")
        )
        text += (f"\n{'gating_mode':16s} {d['state'].get('gating_mode', '?')}"
                 f"\n{'platform':16s} {d['state'].get('platform', '?')}")
        _emit(args, d, text)
        return 0

    if verb == "end":
        if not rest:
            print("usage: nm session end <session-id>", flush=True)
            return 2
        if bridge.end_session(rest[0]):
            print(f"ended {rest[0]}")
            return 0
        print(f"no such session: {rest[0]}", flush=True)
        return 1

    print(f"unknown session verb: {verb} (list|show|end)", flush=True)
    return 2
