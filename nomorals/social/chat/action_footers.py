"""Generalized action footers for tool results.

Every tool result in chat gets the same five actions — Retry, Expand,
Save, Send, Next — instead of each handler inventing its own buttons.
This module is the single place that builds them; handlers that know
their origin command pass it in, and the adapter's ``send_with_footer``
attaches the keyboard.

Semantics (all callbacks are real, routable commands — never dead
buttons):

* Retry — re-runs the exact origin command that produced the result.
* Expand — ``/help <kind>``: the full detail page for that command.
* Save — caller-supplied save command (e.g. ``/remember …`` or
  ``/checkpoint …``); omitted when the caller has nothing to save.
* Send — caller-supplied send command (e.g. ``/say …`` / ``/file …``);
  omitted when the caller has no destination.
* Next — ``/list``: the catalog, i.e. "what can I do next".

Callback budget: Telegram caps ``callback_data`` at 64 bytes and the
adapter's HMAC signature adds 17, so payloads stay at or under 47 bytes
— enforced here via :func:`check_callback_data`, same as the rest of
the button system.

``footer_for_text`` is the auto-attach heuristic the adapter uses when
it knows the origin command (tracked per chat from the last inbound
``/command``): only texts that look like tool results get a footer —
plain conversation never does.
"""

from __future__ import annotations

from .tgbot_buttons import check_callback_data

__all__ = [
    "STANDARD_ACTIONS",
    "action_footer",
    "tool_result_footer",
    "is_tool_result",
    "footer_for_text",
    "parse_action_callback",
]

#: The five standard actions, in order.
STANDARD_ACTIONS: tuple[str, ...] = ("retry", "expand", "save", "send", "next")

_ACTION_LABELS: dict[str, str] = {
    "retry": "🔁 Retry",
    "expand": "🔍 Expand",
    "save": "💾 Save",
    "send": "📤 Send",
    "next": "⏭️ Next",
}

#: Substrings that mark a message as a tool result rather than chat.
_TOOL_RESULT_MARKERS: tuple[str, ...] = (
    "✅", "📊", "📋", "🎮", "🏆", "Result:", "result:", "completed",
    "points and coins credited", "The House wins", "commands probed",
    "— pick one", "modes & mastery",
)

# Button = (label, callback_data); Keyboard = rows of buttons.
Button = tuple[str, str]
Keyboard = list[list[Button]]


def action_footer(buttons: list[tuple[str, str]]) -> Keyboard:
    """Build a one-row action footer from ``[(label, command), …]``.

    Commands are stored without the leading slash (the adapter re-adds
    it); every payload is budget-checked. This generalizes
    :func:`tgbot_buttons.actionable` — same shape, but the *caller*
    chooses the actions instead of the text pattern choosing them.
    """
    row: list[Button] = []
    for label, command in buttons:
        data = (command or "").strip()
        if data.startswith("/"):
            data = data[1:]
        check_callback_data(data)
        row.append((label, data))
    return [row] if row else []


def tool_result_footer(
    origin_command: str,
    *,
    save_command: str = "",
    send_command: str = "",
) -> Keyboard:
    """The standard five-action footer for a tool result.

    ``origin_command`` is the full command that produced the result
    (``"/game trivia"``). Retry re-runs it; Expand opens its help page;
    Next opens the catalog. Save/Send appear only when the caller
    supplies a real command for them — a footer never carries a button
    that does nothing.
    """
    origin = (origin_command or "").strip()
    kind = origin.lstrip("/").split(None, 1)[0].lower() if origin else ""
    pairs: list[tuple[str, str]] = []
    if origin:
        pairs.append((_ACTION_LABELS["retry"], origin))
    if kind:
        pairs.append((_ACTION_LABELS["expand"], f"help {kind}"))
    if save_command.strip():
        pairs.append((_ACTION_LABELS["save"], save_command.strip()))
    if send_command.strip():
        pairs.append((_ACTION_LABELS["send"], send_command.strip()))
    pairs.append((_ACTION_LABELS["next"], "list"))
    return action_footer(pairs)


def is_tool_result(text: str) -> bool:
    """Heuristic: does this reply look like a tool result, not chat?"""
    t = text or ""
    if not t.strip():
        return False
    return any(m in t for m in _TOOL_RESULT_MARKERS)


def footer_for_text(text: str, origin_command: str = "") -> Keyboard | None:
    """Footer for an outgoing reply, or None when it earns none.

    Returns None for plain conversation and when no origin command is
    known (Retry without an origin would be a dead button). Callers
    that know a save/send destination should use
    :func:`tool_result_footer` directly instead.
    """
    if not is_tool_result(text):
        return None
    if not (origin_command or "").strip():
        return None
    return tool_result_footer(origin_command)


def parse_action_callback(data: str) -> tuple[str, str] | None:
    """Split a footer callback ``"<command>"`` — kept for symmetry.

    Footer callbacks *are* the commands (Retry = the origin command),
    so there is nothing namespaced to parse: this returns
    ``("retry", command)`` when the data re-runs a command, else None.
    It exists so future namespaced actions have one parse entry point.
    """
    data = (data or "").strip()
    if not data:
        return None
    if data.startswith("act:"):
        rest = data[4:]
        action, _, payload = rest.partition(":")
        if action in STANDARD_ACTIONS:
            return (action, payload)
        return None
    return ("retry", data) if data else None
