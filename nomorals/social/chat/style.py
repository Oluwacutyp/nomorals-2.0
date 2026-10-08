"""Septorch-style output design for Devon's chat messages.

The patterns: emoji section headers, monospace commands, progress bars,
and an explicit next-action CTA.  Outputs are HTML-formatted for
Telegram (``<b>``, ``<code>``) — the send path renders them with
``parse_mode="HTML"`` only when the caller opts in, so plain-text
messages elsewhere are untouched.

Usage::

    from nomorals.social.chat.style import section, cmd, cta, bar

    lines = [
        section("🎮", "ARENA"),
        f"  {cmd('/game arena')} — start a bout",
        "",
        cta("try /game arena to start"),
    ]
    send(..., "\\n".join(lines), parse_mode="HTML")
"""

from __future__ import annotations

import html

__all__ = ["section", "cmd", "cta", "bar", "bold", "escape"]


def escape(text: str) -> str:
    """HTML-escape user/model content (never escape the helpers' output)."""
    return html.escape(str(text or ""), quote=False)


def bold(text: str) -> str:
    return f"<b>{text}</b>"


def section(emoji: str, title: str) -> str:
    """An emoji section header: `🎮 ARENA`."""
    return f"{emoji} <b>{escape(title.upper())}</b>"


def cmd(command: str) -> str:
    """A tappable monospace command: `/game arena`."""
    c = command if command.startswith("/") else f"/{command}"
    return f"<code>{escape(c)}</code>"


def cta(text: str) -> str:
    """An explicit next-action call-to-action."""
    return f"🎯 {escape(text)}"


def bar(frac: float, width: int = 10) -> str:
    """A progress bar: `██████░░░░ 60%`."""
    frac = max(0.0, min(1.0, float(frac)))
    filled = int(round(frac * width))
    return f"<code>{'█' * filled}{'░' * (width - filled)}</code> {int(frac * 100)}%"
