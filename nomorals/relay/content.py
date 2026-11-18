"""The ticket→issue contract, ported from
storytold/discord-github-ticket-relay (MIT, Copyright (c) 2026
juansebsol) — ``src/github/issue-content.ts``.

The shape is the port; the code is an original Python
reimplementation of the pattern, no TypeScript copied:

* the issue title carries the app (here the repo's short name) in
  brackets, an optional ``[Type]`` prefix, and a ``[Test]`` prefix
  for test filings, capped at 256 characters on one line;
* the issue body is a metadata table (app, type, reporter + Discord
  user id, submitted time, Discord server, channel) plus the ticket
  text, the ticket type's answers as sections, and a hidden
  submission marker — ``<!-- ticket-relay:submission=<ref> -->`` —
  so the issue can always be traced back to its ticket;
* user-written text is neutralised before it reaches GitHub: an
  ``@`` outside fenced/inline code gets a zero-width space after it,
  so a ticket can never @-mention a GitHub user or team by accident.
  (Parity with upstream: closing keywords like ``fixes #123`` are
  left alone — only @-mentions are neutralised.)
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

__all__ = [
    "MARKER_PREFIX",
    "build_issue_body",
    "build_issue_title",
    "neutralize_github_mentions",
    "submission_marker",
]

MARKER_PREFIX = "ticket-relay:submission="

#: GitHub titles over this length are cut (upstream's cap).
GITHUB_TITLE_MAX = 256

#: A zero-width space — breaks an @-mention without changing the
#: visible text.
_ZWSP = "\u200b"


def submission_marker(ref: str) -> str:
    """The hidden marker tying an issue back to its ticket."""
    return f"<!-- {MARKER_PREFIX}{ref} -->"


def neutralize_github_mentions(text: str) -> str:
    """Stop user text from @-mentioning GitHub users or teams.

    An ``@`` followed by a word character outside fenced and inline
    code gets a zero-width space after it; code blocks and inline
    code pass through untouched, exactly as upstream does it.
    """
    out_lines: list[str] = []
    in_fence = False
    for line in (text or "").split("\n"):
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        rebuilt = "".join(
            part if i % 2 == 1 else re.sub(
                r"(^|[^\w`])@(?=[A-Za-z0-9])", r"\1@" + _ZWSP, part)
            for i, part in enumerate(parts))
        out_lines.append(rebuilt)
    return "\n".join(out_lines)


def _cell(value: str) -> str:
    """One metadata-table cell: no pipes, no newlines."""
    return value.replace("|", "\\|").replace("\r", " ").replace(
        "\n", " ").strip()


def _heading_text(question: str) -> str:
    text = re.sub(r"[\r\n#]+", " ", question or "")
    text = re.sub(r"\s+", " ", text).strip()
    return neutralize_github_mentions(text) or "Answer"


def build_issue_title(app_name: str, title: str, *,
                      is_test: bool = False,
                      type_prefix: str | None = None) -> str:
    """``[app] [type] title`` on one line, capped at 256 characters."""
    prefix = ""
    if is_test:
        prefix += "[Test] "
    prefix += f"[{_cell(app_name) or 'ticket'}] "
    if type_prefix:
        clean = re.sub(r"[\[\]\r\n]", "", type_prefix).strip()
        if clean:
            prefix += f"[{clean}] "
    full = prefix + re.sub(r"[\r\n]+", " ", title or "").strip()
    if len(full) > GITHUB_TITLE_MAX:
        return full[:GITHUB_TITLE_MAX - 1] + "…"
    return full


def build_issue_body(
    *,
    ref: str,
    app_name: str,
    title: str,
    description: str,
    reporter_display_name: str,
    reporter_username: str | None = None,
    reporter_discord_id: str = "",
    submitted_at: datetime | str | None = None,
    guild_name: str | None = None,
    guild_id: str | None = None,
    channel_id: str | None = None,
    is_test: bool = False,
    ticket_type: str | None = None,
    answers: list[dict[str, Any]] | None = None,
    files_block: str | None = None,
) -> str:
    """The issue body for one ticket: metadata table, text, marker."""
    who = reporter_display_name or "unknown reporter"
    if reporter_username and reporter_username != reporter_display_name:
        who += f" (Discord username: {reporter_username})"
    if submitted_at is None:
        submitted = ""
    elif isinstance(submitted_at, str):
        submitted = submitted_at
    else:
        submitted = submitted_at.isoformat().replace(
            "T", " ").replace("+00:00", " UTC")
    rows: list[tuple[str, str]] = [
        ("App", _cell(app_name)),
    ]
    if ticket_type:
        rows.append(("Type", _cell(ticket_type)))
    reported = _cell(neutralize_github_mentions(who))
    if reporter_discord_id:
        reported += f" — Discord user ID `{reporter_discord_id}`"
    rows.append(("Reported by", reported))
    if submitted:
        rows.append(("Submitted", _cell(submitted)))
    if guild_name or guild_id:
        server = _cell(guild_name or "Unknown")
        if guild_id:
            server += f" (`{guild_id}`)"
        rows.append(("Discord server", server))
    if channel_id:
        rows.append(("Submitted from channel", f"`{channel_id}`"))
    lines = [f"## {title.strip() or 'Ticket'}", ""]
    lines.append("| | |")
    lines.append("| --- | --- |")
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    lines.append("")
    text = neutralize_github_mentions(description or "")
    if text.strip():
        lines.append(text.strip())
        lines.append("")
    for answer in answers or []:
        question = _heading_text(str(answer.get("question", "")))
        value = neutralize_github_mentions(
            str(answer.get("value", ""))).strip()
        lines.append(f"### {question}")
        lines.append("")
        lines.append(value)
        lines.append("")
    if files_block:
        lines.append("### Attachments")
        lines.append("")
        lines.append(files_block.strip())
        lines.append("")
    if is_test:
        lines.append("*Test filing — no real ticket behind this.*")
        lines.append("")
    lines.append(submission_marker(ref))
    return "\n".join(lines).rstrip() + "\n"
