"""Chat renders for the triage board, queue and decisions."""
from __future__ import annotations

from typing import Any

from .store import TriageStore

__all__ = ["render_board", "render_queue", "render_decisions",
           "render_item"]


def _type_tag(item: dict[str, Any]) -> str:
    type_ = item.get("type")
    return str(type_) if type_ else "untyped"


def render_board(store: TriageStore, repo: str | None = None) -> str:
    """The board: every tracked repo with open counts + queue heads."""
    repos = [repo] if repo else store.tracked()
    if not repos:
        return ("no repos tracked — /triage track <owner/name> to start "
                "triaging a backlog.")
    lines = ["Issue triage — autocraft's mirror + classify + rank, "
             "in Devon:"]
    for name in repos:
        items = store.items_for(name, state="open")
        decisions = store.decisions(status="open", repo=name)
        head = ""
        ranked = sorted(
            (row for row in items if row.get("work_rank") is not None),
            key=lambda row: -int(row.get("work_rank") or 0))[:3]
        if ranked:
            head = " — top: " + ", ".join(
                f"#{row.get('number')} [{row.get('priority') or 'P2'}]"
                for row in ranked)
        lines.append(f"• {name}: {len(items)} open, "
                     f"{len(decisions)} decision(s) waiting{head}")
    return "\n".join(lines)


def render_queue(items: list[dict[str, Any]], repo: str = "") -> str:
    """The work queue: open items in work-rank order."""
    if not items:
        return f"nothing open in {repo or 'this repo'} — the queue is clear."
    lines = [f"Work queue — {repo} (highest rank first):"
             if repo else "Work queue (highest rank first):"]
    for item in items:
        number = item.get("number")
        priority = item.get("priority") or "P2"
        flags = ""
        if item.get("security"):
            flags += " 🔒"
        if item.get("needs_decision"):
            flags += " ?"
        lines.append(f"#{number} [{priority}] {_type_tag(item)} — "
                     f"{item.get('title', '')}{flags}")
    return "\n".join(lines)


def render_decisions(decisions: list[dict[str, Any]]) -> str:
    """Open decisions with their question and options."""
    if not decisions:
        return "no open decisions — nothing waiting on you."
    lines = ["Decisions waiting on you:"]
    for dec in decisions:
        lines.append(f"• {dec.get('id')} [{dec.get('kind')}] "
                     f"{dec.get('question', '')}")
        options = ", ".join(str(o) for o in dec.get("options", []))
        lines.append(f"  options: {options}")
        if dec.get("summary"):
            lines.append(f"  {dec.get('summary')}")
    return "\n".join(lines)


def render_item(item: dict[str, Any],
                decisions: list[dict[str, Any]] | None = None) -> str:
    """One item whole: state, type, flags, priority, rank, decisions."""
    lines = [
        f"#{item.get('number')} {item.get('title', '')}",
        f"repo: {item.get('repo')} — {item.get('state')} — "
        f"{item.get('url', '')}",
        f"type: {_type_tag(item)} "
        f"(source: {item.get('type_source') or 'unsettled'}) — "
        f"priority: {item.get('priority') or 'P2'} — "
        f"reach: {item.get('reach') or 0} — "
        f"rank: {item.get('work_rank') or 0}",
        f"security: {'yes' if item.get('security') else 'no'} — "
        f"needs decision: {'yes' if item.get('needs_decision') else 'no'}",
        f"author: {item.get('author', '')} — "
        f"comments: {item.get('comments', 0)}",
    ]
    body = (item.get("body") or "").strip()
    if body:
        lines.append(body[:1000])
    for dec in decisions or []:
        lines.append(f"decision {dec.get('id')}: {dec.get('question', '')}")
    return "\n".join(lines)
