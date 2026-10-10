"""Presentation themes for browser output (layer 4).

Every surface the browser module shows a human — error cards, snapshot
headers, download lines, pacing summaries — renders through a theme so
the output feels deliberate, not dumped. ``rich`` (default) uses icons
and compact structure for chat/CLI; ``plain`` is ASCII-only for logs,
pipes, and screen readers.

Nothing here touches the browser: these are pure formatters over the
dicts the service already returns.
"""

from __future__ import annotations

from typing import Any

from .errors import BrowserError, summarize as _summarize_error

__all__ = [
    "THEMES",
    "error_card",
    "snapshot_block",
    "download_line",
    "tab_line",
    "pacing_line",
    "stats_block",
]

#: The available output themes.
THEMES = ("rich", "plain")


def _theme(style: str) -> str:
    return "plain" if str(style or "").lower() == "plain" else "rich"


def error_card(error: BrowserError, style: str = "rich") -> str:
    """A failure card: symptom → diagnosis → prescription, with honest
    retry guidance. Never a bare traceback."""
    return _summarize_error(error, style=_theme(style))


def snapshot_block(result: dict[str, Any], style: str = "rich") -> str:
    """Header + accessibility tree for a :meth:`RenderedTab.snapshot`
    result, ready to paste into chat/CLI output."""
    theme = _theme(style)
    snap = str(result.get("snapshot") or "")
    refs = result.get("refs") or {}
    source = result.get("source") or "?"
    url = result.get("url") or ""
    icon = "\U0001F5FA\uFE0F " if theme == "rich" else ""
    head = f"{icon}page snapshot ({source}, {len(refs)} refs)"
    if url:
        head += f" — {url}"
    if result.get("truncated"):
        head += " [truncated]"
    tip = ("tip: act with ref:<id> (e.g. ref:e7)" if theme == "rich"
           else "tip: act with ref:<id>")
    return f"{head}\n{tip}\n{snap}"


def download_line(record: dict[str, Any], style: str = "rich") -> str:
    """One-line download summary for lists and notifications."""
    theme = _theme(style)
    name = str(record.get("filename") or record.get("path") or "?")
    size = record.get("size")
    size_s = f" ({_human_size(size)})" if isinstance(size, int) else ""
    status = str(record.get("status") or "done")
    icon = {"done": "\u2705 ", "failed": "\u274C ",
            "pending": "\u23F3 "}.get(status, "")
    if theme == "plain":
        icon = ""
    return f"{icon}{name}{size_s} — {status}"


def tab_line(tab: dict[str, Any], style: str = "rich") -> str:
    """One-line tab summary for tab lists."""
    theme = _theme(style)
    tab_id = str(tab.get("tab_id") or "?")
    title = str(tab.get("title") or "").strip() or "(untitled)"
    url = str(tab.get("url") or "")
    icon = "\U0001F4D1 " if theme == "rich" else ""
    line = f"{icon}[{tab_id}] {title}"
    if url:
        line += f" — {url}"
    return line


def pacing_line(desc: dict[str, Any], style: str = "rich") -> str:
    """One-line pacing summary: config + observed usage."""
    theme = _theme(style)
    if not desc.get("enabled"):
        return "pacing: off" if theme == "plain" else "\U0001F4A8 pacing: off"
    base = (f"pacing: {desc.get('delay_ms')}ms + up to "
            f"{desc.get('jitter_ms')}ms jitter")
    per_action = desc.get("per_action") or {}
    if per_action:
        overrides = ", ".join(
            f"{a}={d}+{j}" for a, (d, j) in sorted(per_action.items()))
        base += f" (per-action: {overrides})"
    pauses = desc.get("pauses") or 0
    slept = desc.get("slept_s") or 0
    if pauses:
        base += f" — {pauses} pauses, {slept:.1f}s slept"
    icon = "\U0001F422 " if theme == "rich" else ""
    return f"{icon}{base}"


def stats_block(stats: dict[str, Any], style: str = "rich") -> str:
    """Multi-line service overview from :meth:`BrowserService.stats`."""
    theme = _theme(style)
    icon = "\U0001F310 " if theme == "rich" else ""
    lines = [f"{icon}browser service"]
    sessions = stats.get("sessions") or []
    lines.append(f"sessions: {len(sessions)}"
                 + (f" ({', '.join(sessions)})" if sessions else ""))
    lines.append(f"tabs: {stats.get('plain_tabs', 0)} plain, "
                 f"{stats.get('rendered_tabs', 0)} rendered")
    lines.append(f"downloads tracked: {stats.get('downloads', 0)}")
    lines.append(pacing_line(stats.get("pacing") or {}, style=theme))
    pool = "attached" if stats.get("proxy_pool_attached") else "none"
    lines.append(f"proxy pool: {pool}")
    return "\n".join(lines)


def _human_size(size: int) -> str:
    size = max(0, int(size))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size //= 1024
    return f"{size}B"
