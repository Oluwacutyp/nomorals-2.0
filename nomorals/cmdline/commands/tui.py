"""``nm tui`` — terminal UI launcher."""

from __future__ import annotations

import argparse
import sys
from typing import Any



def _cmd_tui(args: argparse.Namespace, context: Any) -> int:
    """Interactive terminal UI. Commands are dispatched against the context."""
    from ...tui import TuiState, run as run_tui
    from ...tui.app import TuiApp

    state = TuiState(status=f"profile {context.settings.profile}")

    def on_submit(text: str) -> None:
        _dispatch_tui_command(context, state, text)

    app = TuiApp(context, state=state, on_submit=on_submit)
    import curses

    try:
        curses.wrapper(app.run)
    except curses.error as exc:
        print(f"could not start the TUI: {exc}", file=sys.stderr)
        return 1
    return 0


def _dispatch_tui_command(context: Any, state: Any, text: str) -> None:
    """Handle one line of TUI input. Slash-commands first, then chat."""
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        command = parts[0][1:].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if command in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if command == "help":
            state.say(
                "plain text chats with the active model\n"
                "/mem <text> remember   /recall <query> search memory\n"
                "/tools list tools   /models list models   /missions list missions\n"
                "/doctor environment   /clear clear   /quit exit\n"
                "press ? (or F1) for the full key-binding overlay",
                kind="info",
            )
        elif command == "tools":
            for name in context.tools.names():
                state.tool(name)
        elif command == "mem":
            state.say(f"remembered: {context.memory.remember(argument, source='tui')}", kind="tool")
        elif command == "recall":
            for record in context.memory.recall(argument, limit=5).records:
                state.say(f"{record.score:.3f} {record.content[:120]}", kind="assistant")
        elif command == "missions":
            from ...missions import MissionStore

            for mission in MissionStore(context.db).list(limit=10):
                state.say(f"{mission.id}  [{mission.status}]  {mission.goal[:60]}", kind="info")
        elif command == "models":
            from ...llm.registry import ModelRegistry

            for record in ModelRegistry(context.db).list(limit=10):
                mark = "*" if record.active else " "
                state.say(f" {mark} {record.name} ({record.kind})", kind="info")
        elif command == "doctor":
            state.say(f"profile={context.settings.profile} schema={context.db.scalar('SELECT COALESCE(MAX(version),0) FROM schema_migrations')}", kind="info")
        elif command == "clear":
            state.clear()
        else:
            state.error(f"unknown command /{command} — try /help")
        return

    from ...llm.base import Message, SamplingParams

    response = context.router.chat(
        [Message.user(text)], SamplingParams(temperature=0.7, max_tokens=1024)
    )
    if response.ok:
        state.say(response.text, kind="assistant")
    else:
        state.error(response.error or "no response")
