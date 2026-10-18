"""Adaptive inline buttons for outputs — derived from the output, never stale.

The old system attached the SAME static footer (Retry / Expand / Next)
to every tool result, keyed off the last inbound ``/command`` in the
chat. A scheduled push (e.g. memory-consolidation) therefore got
Retry = re-run whatever the owner last typed, Expand = help for that
unrelated command, Next = the catalog — stale filler on every message.

The rule now: buttons come from the OUTPUT — its kind + its content —
never from the last inbound command. The caller that builds an output
passes ``output_kind`` + ``payload``; this module returns the actions
relevant to THAT output, each a real routable command (verified by
test against ``CONTROL_COMMANDS``). When no action is relevant it
returns None — never stale filler buttons.

Callback budget: Telegram caps ``callback_data`` at 64 bytes and the
adapter's HMAC signature adds 17, so payloads stay at or under 47
bytes. A button whose payload exceeds the budget is OMITTED, never
sent truncated (a truncated query would run the wrong query).

This module never raises.
"""

from __future__ import annotations

from typing import Any

from .tgbot_buttons import buttons_for_text, check_callback_data

__all__ = ["adaptive_buttons_for"]

# Button = (label, callback_data). Rows are lists of buttons.
Button = tuple[str, str]
Keyboard = list[list[Button]]


def _btn(label: str, data: str) -> Button | None:
    """One button, or None when its payload exceeds the callback budget."""
    try:
        check_callback_data(data)
    except ValueError:
        return None
    return (label, data)


def _row(*pairs: tuple[str, str]) -> list[Button]:
    """One row, dropping over-budget entries."""
    row: list[Button] = []
    for label, data in pairs:
        btn = _btn(label, data)
        if btn is not None:
            row.append(btn)
    return row


def _scheduled_buttons(payload: dict[str, Any] | None) -> Keyboard | None:
    """Buttons for a scheduled-job push: actions about THAT job."""
    job = ""
    if isinstance(payload, dict):
        job = str(payload.get("job") or "").strip()
    if job == "memory-consolidation":
        rows = [
            _row(("🧠 Recall", "recall"), ("📋 Memories", "memories")),
            _row(("📋 Jobs", "schedule list"),
                 ("🔁 Re-run", "schedule run memory-consolidation")),
        ]
        return [r for r in rows if r] or None
    if job:
        rows = [
            _row(("📋 Jobs", "schedule list"), ("📊 Status", "schedule status")),
            _row(("🔁 Re-run", f"schedule run {job}")),
        ]
        return [r for r in rows if r] or None
    row = _row(("📋 Jobs", "schedule list"), ("📊 Status", "schedule status"))
    return [row] if row else None


def _search_buttons(payload: dict[str, Any] | None) -> Keyboard | None:
    """Buttons for a search result: save / deepen / re-run THAT query."""
    query = ""
    if isinstance(payload, dict):
        query = str(payload.get("query") or "").strip()
    if not query:
        return None
    rows = [
        _row(("💾 Save query", f"remember {query}"),
             ("🔬 Deep research", f"searchdeep {query}")),
        _row(("🔁 Search again", f"search {query}")),
    ]
    return [r for r in rows if r] or None


def _game_buttons(payload: dict[str, Any] | None) -> Keyboard | None:
    """Buttons for a game output: actions for THAT game.

    The label names the game (``Rematch trivia``), so the keyboard
    visibly adapts to the output — the same Rematch/Next/Leaderboard
    shape, but for this game, not a generic set.
    """
    game = ""
    if isinstance(payload, dict):
        game = str(payload.get("game") or payload.get("name") or "").strip()
    rematch = f"🔁 Rematch {game}" if game else "🔁 Rematch"
    rows = [
        _row((rematch, "game rematch"), ("⏭️ Next", "game list")),
        _row(("🏆 Leaderboard", "game leaderboard")),
    ]
    return [r for r in rows if r] or None


def _memory_buttons() -> Keyboard | None:
    """Buttons for a memory output: the memory-system commands.

    NOTE: ``/memory`` is a GAME — memory-system actions are
    ``/recall`` / ``/memories`` / ``/forget``, never ``/memory``.
    """
    row = _row(("🧠 Recall", "recall"), ("📋 Memories", "memories"))
    return [row] if row else None


def _retry_only(origin_command: str) -> Keyboard | None:
    """Retry for a command result — the origin, nothing else.

    No Expand (help for the command) and no Next (catalog) filler:
    those appeared on every result regardless of content. A command
    result earns exactly one generic action: re-run the command that
    produced it.
    """
    origin = (origin_command or "").strip()
    if not origin:
        return None
    row = _row(("🔁 Retry", origin))
    return [row] if row else None


def adaptive_buttons_for(
    *,
    output_kind: str | None = None,
    payload: dict[str, Any] | None = None,
    origin_command: str = "",
    text: str = "",
) -> Keyboard | None:
    """Inline keyboard for an output, or None when nothing is relevant.

    ``output_kind`` is the output's kind (``"scheduled"`` / ``"search"``
    / ``"game"`` / ``"memory"`` / ``"command"`` / ``"tool"``) with
    ``payload`` its structured content. ``origin_command`` is the
    command that produced this output, when the caller knows it
    (a command result). ``text`` is the output text, used for the
    content-pattern fallback (game results, play pick-lists, confirm
    prompts, …) when no kind is given.

    The kind handlers win over text patterns: a scheduled push whose
    text happens to mention a game gets job buttons, not game buttons.
    """
    kind = (output_kind or "").strip().lower()
    if kind == "scheduled":
        return _scheduled_buttons(payload)
    if kind == "search":
        return _search_buttons(payload)
    if kind == "game":
        return _game_buttons(payload)
    if kind == "memory":
        return _memory_buttons()
    if kind in ("command", "tool"):
        return _retry_only(origin_command)

    # No kind: content-pattern fallback (game results, play pick-lists,
    # confirm prompts, …) — those patterns ARE content-derived, not
    # stale. Only then the retry-only fallback for a command result.
    try:
        kb = buttons_for_text(text or "")
    except Exception:  # noqa: BLE001 - button derivation never breaks a send
        kb = None
    if kb:
        return kb
    if origin_command:
        from .action_footers import is_tool_result

        if is_tool_result(text or ""):
            return _retry_only(origin_command)
    return None
