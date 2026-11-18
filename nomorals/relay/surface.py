"""Renders for the ticket relay board: plain words, one line per
ticket, every line naming a real record."""
from __future__ import annotations

from .store import TicketLink

__all__ = [
    "render_open_result",
    "render_ticket",
    "render_ticket_board",
]


def render_ticket_board(links: list[TicketLink]) -> str:
    """Every ticket, one line each: ref, repo#issue, state, reporter."""
    if not links:
        return "No tickets on record yet — /ticket open <owner/repo> | <title> files one."
    lines = ["Tickets — Discord → GitHub:"]
    for link in links:
        issue = (f"{link.repo}#{link.issue_number}"
                 if link.issue_number else f"{link.repo} (not filed)")
        pending = f" · {len(link.pending_posts)} pending" \
            if link.pending_posts else ""
        lines.append(
            f"• {link.ref} · {issue} · {link.state} · "
            f"{link.title or '(no title)'} · by "
            f"{link.reporter or 'Discord member'}{pending}")
    return "\n".join(lines)


def render_ticket(link: TicketLink) -> str:
    """One ticket whole: link, origin, state, sync state, failures."""
    issue = (link.issue_url or
             (f"{link.repo}#{link.issue_number}"
              if link.issue_number else "not filed"))
    lines = [
        f"Ticket {link.ref} — {link.title or '(no title)'}",
        f"Issue: {issue}",
        f"State: {link.state}"
        + (f" (GitHub: {link.last_issue_state})"
           if link.last_issue_state else ""),
        f"From: {link.origin_platform}:{link.origin_chat or '?'}"
        + (f" · reported by {link.reporter}" if link.reporter else ""),
        f"Comments seen: {link.last_comment_count}",
    ]
    if link.pending_posts:
        lines.append(
            f"Pending posts to Discord: {len(link.pending_posts)}")
    if link.failures:
        lines.append(f"Last failure: {link.failures[-1]['reason']}")
    return "\n".join(lines)


def render_open_result(link: TicketLink, issue: dict) -> str:
    """What the filer hears back: the issue link, never a bare number."""
    url = issue.get("url") or link.issue_url or ""
    return (
        f"Ticket {link.ref} filed: {link.repo}#{link.issue_number}\n"
        f"{url}\n"
        f"Replies here go to the issue; issue comments come back here."
    )
