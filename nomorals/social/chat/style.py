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


# ── platform-aware menus ─────────────────────────────────────────────────────
# Sections with emoji icons and a scannable command list, rendered for the
# target platform. Telegram gets HTML; WhatsApp gets *bold* / ```mono```;
# the same structure reads well on both.


def _plat(platform: str) -> str:
    try:
        from .platforms import TELEGRAM, detect_platform
        p = (platform or "").strip().lower()
        if p in ("telegram", "whatsapp", "sms", "web", "discord", "local"):
            return p
        return TELEGRAM
    except Exception:  # noqa: BLE001
        return "telegram"


def menu_section(emoji: str, title: str, platform: str = "telegram") -> str:
    """A menu section header: ``🎮 GAMES``. Never raises."""
    try:
        p = _plat(platform)
        t = str(title or "").upper()
        if p == "whatsapp":
            return f"{emoji} *{t}*"
        return f"{emoji} <b>{escape(t)}</b>"
    except Exception:  # noqa: BLE001
        return f"{emoji} {title}"


def menu_item(command: str, description: str, platform: str = "telegram") -> str:
    """One menu row: command + description. Never raises."""
    try:
        p = _plat(platform)
        c = str(command or "")
        d = str(description or "")
        if p == "whatsapp":
            return f"• `{c}` — {d}"
        if p in ("sms",):
            return f"- {c} — {d}"
        return f"• <code>{escape(c)}</code> — {escape(d)}"
    except Exception:  # noqa: BLE001
        return f"{command} — {description}"


def menu_divider(platform: str = "telegram") -> str:
    """A visual separator between menu sections. Never raises."""
    try:
        return "━━━━━━━━━━━━━━━"
    except Exception:  # noqa: BLE001
        return "---"


def render_menu(sections: list, platform: str = "telegram",
                header: str = "", footer: str = "") -> str:
    """Render a full menu.

    ``sections``: ``[(emoji, title, [(command, description), ...]), ...]``.
    Never raises.
    """
    try:
        p = _plat(platform)
        lines: list[str] = []
        if header:
            lines.append(menu_section("✨", header, p))
            lines.append("")
        for i, sec in enumerate(sections or []):
            emoji, title, items = sec[0], sec[1], sec[2]
            if i:
                lines.append(menu_divider(p))
            lines.append(menu_section(emoji, title, p))
            for command, desc in items or []:
                lines.append(menu_item(command, desc, p))
            lines.append("")
        if footer:
            lines.append(f"_{footer}_" if p == "whatsapp" else f"<i>{escape(footer)}</i>")
        return "\n".join(lines).rstrip()
    except Exception:  # noqa: BLE001
        return ""
