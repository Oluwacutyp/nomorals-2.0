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

from .expression import button_label
from .tgbot_buttons import (buttons_for_text, check_callback_data,
                            grid_layout)
from .failures import record_failure

__all__ = ["adaptive_buttons_for", "OUTPUT_KINDS"]

#: The output kinds the dispatcher below routes — the enumerable
#: registry the surface manifest's drift test compares against
#: (Living Chat P1): a new kind handled here without a manifest
#: update fails by name.
OUTPUT_KINDS = frozenset({
    "pulse_stories", "scheduled", "search", "game", "memory",
    "music-ask", "command", "tool",
})

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
            _row((button_label("recall", "Recall"), "recall"), (button_label("memories", "Memories"), "memories")),
            _row((button_label("jobs", "Jobs"), "schedule list"),
                 (button_label("rerun", "Re-run"), "schedule run memory-consolidation")),
        ]
        return grid_layout([r for r in rows if r])
    if job:
        rows = [
            _row((button_label("jobs", "Jobs"), "schedule list"), (button_label("status", "Status"), "schedule status")),
            _row((button_label("rerun", "Re-run"), f"schedule run {job}")),
        ]
        return grid_layout([r for r in rows if r])
    row = _row((button_label("jobs", "Jobs"), "schedule list"), (button_label("status", "Status"), "schedule status"))
    return grid_layout([row] if row else None)


def _pulse_stories_buttons(payload: dict[str, Any] | None) -> Keyboard | None:
    """Pager buttons for a briefing's story list (the paging law):
    Next/Prev over the delivered list's pages, riding the routable
    ``pulse stories <batch> <n>`` command — derived from THIS
    delivery's batch, never inherited from an earlier output."""
    if not isinstance(payload, dict):
        return None
    batch = str(payload.get("batch") or "").strip()
    try:
        page = int(payload.get("page") or 1)
        pages = int(payload.get("pages") or 1)
    except (TypeError, ValueError):
        return None
    if not batch or pages <= 1:
        return None
    pairs: list[tuple[str, str]] = []
    if page > 1:
        pairs.append(("◀ Prev", f"pulse stories {batch} {page - 1}"))
    if page < pages:
        pairs.append(("Next ▶", f"pulse stories {batch} {page + 1}"))
    row = _row(*pairs)
    return grid_layout([row] if row else None)


def _search_buttons(payload: dict[str, Any] | None) -> Keyboard | None:
    """Buttons for a search result: save / deepen / re-run THAT query."""
    query = ""
    if isinstance(payload, dict):
        query = str(payload.get("query") or "").strip()
    if not query:
        return None
    rows = [
        _row((button_label("save", "Save query"), f"remember {query}"),
             (button_label("research", "Deep research"), f"searchdeep {query}")),
        _row((button_label("rerun", "Search again"), f"search {query}")),
    ]
    return grid_layout([r for r in rows if r])


def _game_buttons(payload: dict[str, Any] | None) -> Keyboard | None:
    """Buttons for a game output: actions for THAT game.

    The label names the game (``Rematch trivia``), so the keyboard
    visibly adapts to the output — the same Rematch/Next/Leaderboard
    shape, but for this game, not a generic set.
    """
    game = ""
    if isinstance(payload, dict):
        game = str(payload.get("game") or payload.get("name") or "").strip()
    rematch = button_label("rematch", f"Rematch {game}" if game else "Rematch")
    rows = [
        _row((rematch, "game rematch"), (button_label("next", "Next"), "game list")),
        _row((button_label("leaderboard", "Leaderboard"), "game leaderboard")),
    ]
    return grid_layout([r for r in rows if r])


def _music_ask_buttons() -> Keyboard | None:
    """Buttons for the music voice-note ask: Find vs Create.

    Both callbacks are real routable commands (``/musicnote find`` /
    ``/musicnote create``) — the pending clip lives in the per-chat
    store, so the callbacks need no payload and stay well inside the
    47-byte budget.
    """
    row = _row((button_label("search", "Find this music"), "musicnote find"),
               (button_label("music", "Create a song"), "musicnote create"))
    return grid_layout([row] if row else None)


def _memory_buttons() -> Keyboard | None:
    """Buttons for a memory output: the memory-system commands.

    NOTE: ``/memory`` is a GAME — memory-system actions are
    ``/recall`` / ``/memories`` / ``/forget``, never ``/memory``.
    """
    row = _row((button_label("recall", "Recall"), "recall"), (button_label("memories", "Memories"), "memories"))
    return grid_layout([row] if row else None)


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
    row = _row((button_label("rerun", "Retry"), origin))
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
    / ``"game"`` / ``"memory"`` / ``"music-ask"`` / ``"command"`` /
    ``"tool"``) with ``payload`` its structured content. ``origin_command`` is the
    command that produced this output, when the caller knows it
    (a command result). ``text`` is the output text, used for the
    content-pattern fallback (game results, play pick-lists, confirm
    prompts, …) when no kind is given.

    The kind handlers win over text patterns: a scheduled push whose
    text happens to mention a game gets job buttons, not game buttons.
    """
    kind = (output_kind or "").strip().lower()
    if kind == "pulse_stories":
        return _pulse_stories_buttons(payload)
    if kind == "scheduled":
        return _scheduled_buttons(payload)
    if kind == "search":
        return _search_buttons(payload)
    if kind == "game":
        return _game_buttons(payload)
    if kind == "memory":
        return _memory_buttons()
    if kind == "music-ask":
        return _music_ask_buttons()
    if kind in ("command", "tool"):
        return _retry_only(origin_command)

    # No kind: content-pattern fallback (game results, play pick-lists,
    # confirm prompts, …) — those patterns ARE content-derived, not
    # stale. Only then the retry-only fallback for a command result.
    try:
        kb = buttons_for_text(text or "")
    except Exception as exc:  # noqa: BLE001 - button derivation never breaks a send
        record_failure('adaptive_buttons.adaptive_buttons_for', exc)
        kb = None
    if kb:
        return kb
    if origin_command:
        from .action_footers import is_tool_result

        if is_tool_result(text or ""):
            return _retry_only(origin_command)
    return None
