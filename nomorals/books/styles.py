"""BookForge output styles — presentation that feels god-tier, not functional.

Every reader-facing surface in the books module can render through these
helpers. Three themes:

* ``rich``   — emoji + typographic flourishes, the default for chat
* ``plain``  — ASCII-safe, for terminals/log files that can't do unicode
* ``minimal`` — tight one-liners for dense feeds/digests

Mined from Royal Road/Tapas chapter cards and KOReader's progress display:
readers respond to progress bars, per-chapter retention, and cards that
show status at a glance — not raw dicts.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "THEMES",
    "theme_ok",
    "progress_bar",
    "render_book_card",
    "render_chapter_card",
    "render_stats_table",
    "render_beat_sheet",
    "render_tree",
    "render_retention",
]

THEMES = ("rich", "plain", "minimal")


def theme_ok(theme: str) -> str:
    return theme if theme in THEMES else "rich"


def _glyphs(theme: str) -> dict[str, str]:
    if theme == "plain":
        return {"book": "[book]", "chap": "[ch]", "ok": "[ok]", "warn": "[!]",
                "fire": "[hot]", "heart": "<3", "star": "*",
                "full": "#", "empty": "-", "arrow": "->"}
    if theme == "minimal":
        return {"book": "", "chap": "", "ok": "", "warn": "",
                "fire": "", "heart": "", "star": "*",
                "full": "#", "empty": "-", "arrow": "->"}
    return {"book": "📕", "chap": "📄", "ok": "✅", "warn": "⚠️",
            "fire": "🔥", "heart": "❤️", "star": "⭐",
            "full": "█", "empty": "░", "arrow": "→"}


def progress_bar(pct: float, width: int = 20, theme: str = "rich") -> str:
    """Render a percentage as a bar: ``██████░░░░ 62%``."""
    theme = theme_ok(theme)
    g = _glyphs(theme)
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(round(width * pct / 100.0))
    bar = g["full"] * filled + g["empty"] * (width - filled)
    return f"{bar} {pct:.0f}%"


def render_book_card(book: dict[str, Any], theme: str = "rich") -> str:
    """One rich card for a book dict (from ``book_status`` / ``book_list``)."""
    theme = theme_ok(theme)
    g = _glyphs(theme)
    title = book.get("display_title") or book.get("title") or book.get("slug", "?")
    author = book.get("author") or ""
    genre = book.get("genre") or ""
    status = book.get("status", "?")
    written = book.get("chapters_written", 0)
    total = book.get("total_chapters", 0)
    words = book.get("words", book.get("total_words", 0))
    pct = book.get("progress_pct")
    if pct is None and total:
        pct = round(100.0 * written / total, 1)

    if theme == "minimal":
        bits = [str(title), f"{written}/{total} ch", f"{words}w", str(status)]
        return " · ".join(b for b in bits if b)

    lines = [f"{g['book']} *{title}*"]
    if author:
        lines[0] += f" — {author}"
    meta = " · ".join(x for x in [genre, status] if x)
    if meta:
        lines.append(meta)
    lines.append(f"{g['chap']} {written}/{total} chapters · "
                 f"{words:,} words".replace(",", ","))
    if pct is not None:
        lines.append(progress_bar(float(pct), theme=theme))
    return "\n".join(lines)


def render_chapter_card(ch: dict[str, Any], theme: str = "rich") -> str:
    """One card for a chapter dict."""
    theme = theme_ok(theme)
    g = _glyphs(theme)
    n = ch.get("number", ch.get("chapter", "?"))
    title = ch.get("title") or f"Chapter {n}"
    words = ch.get("words", 0)
    if theme == "minimal":
        return f"ch.{n}: {title} ({words}w)"
    lines = [f"{g['chap']} *{title}*"]
    bits = []
    if words:
        bits.append(f"{words:,} words")
    if ch.get("released"):
        bits.append("released")
    if ch.get("status"):
        bits.append(str(ch["status"]))
    if bits:
        lines.append(" · ".join(bits))
    if ch.get("hook"):
        lines.append(f"{g['fire']} {ch['hook']}")
    return "\n".join(lines)


def render_stats_table(stats: dict[str, Any], theme: str = "rich") -> str:
    """Render a reading/writing stats dict as an aligned table."""
    theme = theme_ok(theme)
    rows = []
    for key, val in stats.items():
        if isinstance(val, dict) or key.startswith("_"):
            continue
        label = key.replace("_", " ").title()
        rows.append((label, str(val)))
    if not rows:
        return "(no stats yet)"
    width = max(len(r[0]) for r in rows)
    sep = " │ " if theme == "rich" else " | "
    return "\n".join(f"{label:<{width}}{sep}{val}" for label, val in rows)


def render_beat_sheet(beats: list[dict[str, Any]], theme: str = "rich") -> str:
    """Render a Save-the-Cat-style beat sheet."""
    theme = theme_ok(theme)
    g = _glyphs(theme)
    lines = []
    for b in beats:
        ch = b.get("chapter", "?")
        name = b.get("beat", "?")
        pos = b.get("position_pct", "")
        note = b.get("note", "")
        if theme == "minimal":
            lines.append(f"ch.{ch}: {name}")
        else:
            head = f"ch.{ch} — *{name}*"
            if pos != "":
                head += f" ({pos}%)"
            lines.append(head)
            if note:
                lines.append(f"  {g['arrow']} {note}")
    return "\n".join(lines)


def render_tree(tree: dict[str, Any], theme: str = "rich") -> str:
    """Render an ASCII branch tree (from ``branches.tree()``)."""
    theme = theme_ok(theme)
    lines: list[str] = []

    def walk(node: dict[str, Any], prefix: str = "", last: bool = True) -> None:
        if theme == "rich":
            elbow, pipe, blank = "└─ ", "│  ", "   "
        else:
            elbow, pipe, blank = "+- ", "|  ", "   "
        name = node.get("name", "?")
        info = node.get("info", "")
        head = f"{prefix}{elbow if prefix else ''}{name}"
        if info:
            head += f"  ({info})"
        lines.append(head)
        kids = node.get("children", [])
        for i, kid in enumerate(kids):
            walk(kid, prefix + (blank if last else pipe), i == len(kids) - 1)

    walk(tree)
    return "\n".join(lines)


def render_retention(curve: list[dict[str, Any]], theme: str = "rich") -> str:
    """Render a per-chapter retention curve (from publish retention)."""
    theme = theme_ok(theme)
    g = _glyphs(theme)
    lines = []
    for row in curve:
        ch = row.get("chapter", "?")
        rate = float(row.get("retention_pct", 0) or 0)
        bar = progress_bar(rate, width=12, theme=theme)
        flag = f" {g['warn']} drop" if row.get("drop") else ""
        lines.append(f"ch.{ch}: {bar}{flag}")
    return "\n".join(lines) if lines else "(no retention data)"
