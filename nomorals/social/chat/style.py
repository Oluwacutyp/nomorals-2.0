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

__all__ = [
    "section", "cmd", "cta", "bar", "bold", "escape",
    "THEMES", "set_theme", "current_theme", "header", "quote", "kv",
    "stat_line", "card", "table", "sparkline", "tag",
    "render_menu", "menu_section", "menu_item", "menu_divider",
]


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


# ── output themes ───────────────────────────────────────────────────────────
# How loud the output is. "rich" (default) is the full Devon voice; "minimal"
# drops emoji for dense dashboards; "terminal" is pure monospace/ASCII for
# logs and SMS; "plain" strips everything for screen readers / a11y.

THEMES: dict[str, dict[str, object]] = {
    "rich": {"emoji": True, "box": True, "markup": True},
    "minimal": {"emoji": False, "box": False, "markup": True},
    "terminal": {"emoji": False, "box": False, "markup": False},
    "plain": {"emoji": False, "box": False, "markup": False},
}

_theme: str = "rich"


def set_theme(name: str) -> str:
    """Switch the output theme. Returns the previous theme."""
    global _theme
    prev = _theme
    _theme = name if name in THEMES else "rich"
    return prev


def current_theme() -> str:
    return _theme


def _theme_cfg() -> dict[str, object]:
    return THEMES.get(_theme, THEMES["rich"])


def _markup(text: str, tag_name: str) -> str:
    cfg = _theme_cfg()
    if not cfg.get("markup"):
        return text
    if _theme in ("terminal",):
        return text
    if tag_name == "b":
        return f"<b>{text}</b>"
    if tag_name == "i":
        return f"<i>{text}</i>"
    if tag_name == "code":
        return f"<code>{text}</code>"
    return text


def tag(text: str, label: str) -> str:
    """A labeled pill: `● LIVE` / theme-aware."""
    cfg = _theme_cfg()
    dot = "●" if cfg.get("emoji") else "["
    close = "" if cfg.get("emoji") else "]"
    return f"{_markup(f'{dot} {escape(label.upper())}{close}', 'b')}"


def header(title: str, subtitle: str = "") -> str:
    """A big title card: one job per screen, first line states the point."""
    cfg = _theme_cfg()
    lines = [f"{'✨ ' if cfg.get('emoji') else ''}{_markup(escape(title.upper()), 'b')}"]
    if subtitle:
        lines.append(_markup(escape(subtitle), "i"))
    if cfg.get("box"):
        lines.append("━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def quote(text: str) -> str:
    """A blockquote. Never raises."""
    try:
        lines = [l for l in str(text or "").splitlines()]
        return "\n".join(f"▍ {l}" if l.strip() else "▍" for l in lines)
    except Exception:  # noqa: BLE001
        return str(text or "")


def kv(pairs: list[tuple[str, str]], *, indent: int = 2) -> str:
    """Aligned key/value block: `Likes      1.2k`. Never raises."""
    try:
        rows = [(str(k), str(v)) for k, v in (pairs or [])]
        if not rows:
            return ""
        width = max(len(k) for k, _ in rows)
        pad = " " * indent
        return "\n".join(f"{pad}{_markup(k.ljust(width), 'b')}  {v}" for k, v in rows)
    except Exception:  # noqa: BLE001
        return ""


def stat_line(label: str, value: str, delta: str = "") -> str:
    """One metric line: `📈 Engagement  12.4%  ▲2.1`. Never raises."""
    cfg = _theme_cfg()
    icon = "📈 " if cfg.get("emoji") else ""
    parts = [f"{icon}{_markup(escape(label), 'b')}  {escape(value)}"]
    if delta:
        parts.append(f"({escape(delta)})")
    return " ".join(parts)


_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """Unicode trend sparkline: `[3, 7, 2, 9] → ▂▆▁█`. Never raises."""
    try:
        vals = [float(v) for v in (values or [])]
        if not vals:
            return ""
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return _SPARK[3] * len(vals)
        return "".join(
            _SPARK[min(7, int((v - lo) / (hi - lo) * 8))] for v in vals
        )
    except Exception:  # noqa: BLE001
        return ""


def card(title: str, fields: list[tuple[str, str]], *,
         footer: str = "") -> str:
    """A structured info card: title + aligned fields + optional footer.

    Platform-agnostic structure — run through ``format_for_platform()``
    for WhatsApp/SMS, or send as-is for Telegram HTML.
    """
    lines = [header(title), ""]
    body = kv(fields)
    if body:
        lines.append(body)
    if footer:
        cfg = _theme_cfg()
        lines += ["", f"{'💡 ' if cfg.get('emoji') else ''}{_markup(escape(footer), 'i')}"]
    return "\n".join(lines).rstrip()


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Aligned monospace table. Tables flatten to `Header: value` lines
    when the content is too wide for a chat bubble — this keeps the
    readable middle ground. Never raises."""
    try:
        heads = [str(h) for h in (headers or [])]
        data = [[str(c) for c in r] for r in (rows or [])]
        if not heads:
            return ""
        widths = [len(h) for h in heads]
        for row in data:
            for i, cell in enumerate(row[: len(widths)]):
                widths[i] = max(widths[i], len(cell))
        lines = ["<code>"]
        lines.append("  ".join(h.ljust(w) for h, w in zip(heads, widths)))
        lines.append("  ".join("─" * w for w in widths))
        for row in data:
            cells = list(row) + [""] * (len(widths) - len(row))
            lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        lines.append("</code>")
        if not _theme_cfg().get("markup"):
            return "\n".join(
                l.replace("<code>", "").replace("</code>", "") for l in lines
            )
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


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
