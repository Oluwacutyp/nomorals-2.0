"""Shared presentation primitives for the agents module.

Every agent output that reaches a human — briefs, debate verdicts, swarm
reports, red-team findings, benchmark cards — goes through here so Devon
speaks one visual language: section banners, icon-tagged status lines,
aligned key/value tables, and markdown tables. Plain text stays readable
in Telegram/terminal; no ANSI codes (chat clients strip them), no
dependencies.

Style guide (god-tier, not functional):
- Lead with the verdict/answer. Details fold under it.
- One icon per status: ✅ ❌ ⚠️ 🔁 ⏳ 🧠 ⚔️ 📊. Icons are signposts, not confetti.
- Tables for anything with ≥2 rows of structured data; prose for narrative.
- Numbers always paired with units and context ("3/12 legs done", not "3").
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

__all__ = [
    "ICONS", "banner", "section", "kv", "table", "status_line", "bar",
    "numbered", "bullets", "truncate",
]

#: Status icons — the single visual vocabulary across agent outputs.
ICONS = {
    "ok": "✅",
    "fail": "❌",
    "warn": "⚠️",
    "retry": "🔁",
    "pending": "⏳",
    "thinking": "🧠",
    "debate": "⚔️",
    "stats": "📊",
    "info": "ℹ️",
    "plan": "🗺️",
    "tool": "🔧",
    "time": "⏱️",
    "shield": "🛡️",
    "star": "⭐",
    "arrow": "→",
}


def truncate(text: str, limit: int = 200) -> str:
    """Cut text to ``limit`` chars with an ellipsis marker. Never raises."""
    try:
        text = str(text or "")
    except Exception:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def banner(title: str, icon: str = "") -> str:
    """A bold section banner: ``🧠 **Reasoning**``."""
    prefix = f"{icon} " if icon else ""
    return f"{prefix}**{title}**"


def section(title: str, body: str, icon: str = "") -> str:
    """Banner + body, separated cleanly for chat readability."""
    body = (body or "").strip()
    return f"{banner(title, icon)}\n{body}" if body else banner(title, icon)


def status_line(icon_key: str, text: str) -> str:
    """One icon-tagged status line."""
    icon = ICONS.get(icon_key, ICONS["info"])
    return f"{icon} {text}"


def kv(pairs: Iterable[tuple[str, Any]], *, colon: str = ":") -> str:
    """Aligned key/value block, monospace keys so it lines up in chat.

    >>> print(kv([("model", "groq"), ("latency", "1.2s")]))
    `model`   : groq
    `latency` : 1.2s
    """
    rows = [(str(k), str(v)) for k, v in pairs]
    if not rows:
        return ""
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"`{k.ljust(width)}` {colon} {v}" for k, v in rows)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
          *, max_width: int = 28) -> str:
    """GitHub-flavored markdown table. Cells truncated for chat width."""
    headers = [str(h) for h in headers]
    body = [[truncate(str(c), max_width) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in body:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
    def _row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(
            c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    lines = [_row(headers),
             "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(_row(r) for r in body)
    return "\n".join(lines)


def bar(frac: float, width: int = 12) -> str:
    """Text progress bar: ``██████░░░░░░ 50%``."""
    try:
        frac = max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        frac = 0.0
    filled = int(round(frac * width))
    return f"{'█' * filled}{'░' * (width - filled)} {int(frac * 100)}%"


def numbered(items: Sequence[str]) -> str:
    """1. 2. 3. list."""
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def bullets(items: Sequence[str], icon: str = "•") -> str:
    """Bulleted list."""
    return "\n".join(f"{icon} {item}" for item in items)
