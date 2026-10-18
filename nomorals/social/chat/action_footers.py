"""Tool-result detection for the adaptive button system.

History: this module used to build the SAME static footer
(Retry / Expand / Save / Send / Next) for every tool result, keyed off
the last inbound ``/command`` in the chat — so a scheduled push got
Retry = re-run whatever the owner last typed. That stale set is gone:
:mod:`nomorals.social.chat.adaptive_buttons` derives buttons from the
output (kind + content), and a push with no relevant action gets no
buttons at all.

What remains here is the ``is_tool_result`` heuristic — "does this
reply look like a tool result, not chat" — which the adaptive system
uses for its retry-only fallback (a command result earns exactly one
generic action: re-run the command that produced it).
"""

from __future__ import annotations

__all__ = ["is_tool_result"]

#: Substrings that mark a message as a tool result rather than chat.
_TOOL_RESULT_MARKERS: tuple[str, ...] = (
    "✅", "📊", "📋", "🎮", "🏆", "Result:", "result:", "completed",
    "points and coins credited", "The House wins", "commands probed",
    "— pick one", "modes & mastery",
)


def is_tool_result(text: str) -> bool:
    """Heuristic: does this reply look like a tool result, not chat?"""
    t = text or ""
    if not t.strip():
        return False
    return any(m in t for m in _TOOL_RESULT_MARKERS)
