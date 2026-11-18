"""The grill: an honest, evidence-grounded roast of the owner.

``/grill-me`` reads the records Devon already keeps — saved plans,
the commitment ledger, live delegation runs — and says plainly
where the owner stands: what is still running, what failed, what
was promised and not kept. Every line names a real record (id,
task, age); an empty record answers "nothing to grill yet" instead
of inventing a roast.

The phrasing is deterministic — the house pattern for control
verbs — because the content is the evidence, not the prose.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

__all__ = [
    "SECTIONS",
    "gather_grill_evidence",
    "grill_me",
    "render_grill",
]

SECTIONS = ("plans", "commitments", "runs")

_LIVE_RUN_STATES = ("running", "steering", "finalizing")


def _age_str(seconds: float) -> str:
    """``12 days`` / ``5 hours`` / ``20 minutes`` / ``just now``."""
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute" + ("s" if minutes != 1 else "")
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour" + ("s" if hours != 1 else "")
    days = int(seconds // 86400)
    return f"{days} day" + ("s" if days != 1 else "")


def _home_from_context(context: Any) -> Path | None:
    settings = getattr(context, "settings", None) if context is not None \
        else None
    if settings is None:
        return None
    for attr in ("home_path", "home"):
        value = getattr(settings, attr, None)
        if value:
            return Path(value)
    return None


def _default_plans(home: Path) -> Any:
    from .plan.store import PlanStore, default_plans_dir, default_plans_path

    return PlanStore(path=default_plans_path(home),
                     plans_dir=default_plans_dir(home))


def _default_commitments(home: Path) -> Any:
    from ..commitments.ledger import CommitmentLedger

    return CommitmentLedger(str(home / "data" / "commitments.json"))


def gather_grill_evidence(*, plans: Any = None,
                           commitments: Any = None,
                           manager: Any = None,
                           home: Path | str | None = None,
                           now: float | None = None) -> dict[str, Any]:
    """The raw material for a grill, from the real stores.

    ``plans`` / ``commitments`` / ``manager`` are injected seams
    (tests pass in-memory stores); each defaults to the live object
    under ``home`` when omitted. A store left as None is skipped
    and reported as unknown — never read as empty.
    """
    now = time.time() if now is None else float(now)
    home = Path(home) if home is not None else None
    if plans is None and home is not None:
        plans = _default_plans(home)
    if commitments is None and home is not None:
        commitments = _default_commitments(home)

    evidence: dict[str, Any] = {"now": now}
    if plans is None:
        evidence["plans"] = None
    else:
        rows = plans.list_plans()
        by_state: dict[str, int] = {}
        for row in rows:
            by_state[row.get("state", "?")] = \
                by_state.get(row.get("state", "?"), 0) + 1
        running = [r for r in rows if r.get("state") == "running"]
        oldest = None
        if running:
            pick = min(running, key=lambda r: r.get("created_at", now))
            oldest = {
                "id": pick["id"], "task": pick.get("task", ""),
                "age_s": now - float(pick.get("created_at", now))}
        evidence["plans"] = {
            "total": len(rows),
            "running": by_state.get("running", 0),
            "done": by_state.get("done", 0),
            "failed": by_state.get("failed", 0),
            "cancelled": by_state.get("cancelled", 0),
            "degraded": getattr(plans, "degraded_reason", "") or "",
            "oldest_running": oldest,
            "failed_rows": [
                {"id": r["id"], "task": r.get("task", ""),
                 "error": r.get("plan_error", "")}
                for r in rows if r.get("state") == "failed"],
        }
    if commitments is None:
        evidence["commitments"] = None
    else:
        all_rows = list(commitments.commitments.values())
        by_status: dict[str, int] = {}
        for row in all_rows:
            by_status[row.status] = by_status.get(row.status, 0) + 1
        overdue = [
            {"id": c.id, "what": c.what, "promisee": c.promisee,
             "overdue_s": now - float(c.deadline)}
            for c in commitments.overdue(now)
            if c.deadline is not None]
        evidence["commitments"] = {
            "total": len(all_rows),
            "open": by_status.get("open", 0),
            "kept": by_status.get("kept", 0),
            "broken": by_status.get("broken", 0),
            "cancelled": by_status.get("cancelled", 0),
            "superseded": by_status.get("superseded", 0),
            "overdue": overdue,
            "due_soon": len(commitments.due_soon(86400, now)),
            "reliability": commitments.reliability(),
        }
    if manager is None:
        evidence["runs"] = None
    else:
        live = [r for r in manager.list_runs(limit=500)
                if getattr(r, "status", "") in _LIVE_RUN_STATES]
        oldest_run = None
        if live:
            pick = min(live, key=lambda r: getattr(
                r, "created_at", now) or now)
            oldest_run = {
                "id": pick.id, "role": pick.role,
                "age_s": now - float(getattr(
                    pick, "created_at", now) or now)}
        evidence["runs"] = {"live": len(live), "oldest": oldest_run}
    return evidence


def _render_plans(plans: dict[str, Any] | None) -> list[str]:
    if plans is None:
        return ["plans: not checked (no plan store)."]
    lines = [
        f"plans: {plans['total']} on the board — "
        f"{plans['running']} running, {plans['done']} done, "
        f"{plans['failed']} failed, {plans['cancelled']} cancelled."]
    if plans.get("degraded"):
        lines.append(f"plans store degraded: {plans['degraded']}")
    oldest = plans.get("oldest_running")
    if oldest is not None:
        lines.append(
            f"oldest still running: '{oldest['task']}' "
            f"({oldest['id']}) — {_age_str(oldest['age_s'])} old "
            "and counting.")
    for row in plans.get("failed_rows") or []:
        reason = row.get("error") or "no reason recorded"
        lines.append(
            f"failed and left there: '{row['task']}' ({row['id']}) — "
            f"{reason}.")
    if plans["cancelled"]:
        lines.append(
            f"cancelled: {plans['cancelled']} — at least those you "
            "called off yourself.")
    return lines


def _render_commitments(
        commitments: dict[str, Any] | None) -> list[str]:
    if commitments is None:
        return ["promises: not checked (no commitment ledger)."]
    lines = []
    stats = commitments.get("reliability") or {}
    resolved = stats.get("resolved", 0)
    if resolved:
        lines.append(
            f"promises: kept {stats['kept']} of {resolved} "
            f"({stats['rate']:.0%}).")
    else:
        lines.append(
            "promises: none resolved yet — nothing kept, "
            "nothing broken.")
    overdue = commitments.get("overdue") or []
    if overdue:
        lines.append(f"overdue: {len(overdue)} —")
        for row in overdue[:5]:
            who = f" to {row['promisee']}" if row.get("promisee") \
                else ""
            lines.append(
                f"• '{row['what']}'{who} — "
                f"{_age_str(row['overdue_s'])} late.")
        if len(overdue) > 5:
            lines.append(f"• and {len(overdue) - 5} more overdue.")
    due_soon = commitments.get("due_soon", 0)
    if due_soon:
        lines.append(f"due in the next day: {due_soon}.")
    lines.append(f"open promises: {commitments['open']}.")
    return lines


def _render_runs(runs: dict[str, Any] | None) -> list[str]:
    if runs is None:
        return ["live runs: not checked (no delegation manager)."]
    lines = [f"live delegation runs right now: {runs['live']}."]
    oldest = runs.get("oldest")
    if oldest is not None:
        lines.append(
            f"oldest live run: {oldest['role']} ({oldest['id']}) — "
            f"{_age_str(oldest['age_s'])} in.")
    return lines


def render_grill(evidence: dict[str, Any],
                 section: str | None = None) -> str:
    """The grill itself: pointed lines, every one naming a record.

    ``section`` renders one part (``plans`` / ``commitments`` /
    ``runs``); anything else raises ValueError. With nothing on
    record at all, the answer is honest emptiness, not a roast.
    """
    if section is not None and section not in SECTIONS:
        raise ValueError(
            f"no grill section {section!r} — pick one of "
            + ", ".join(SECTIONS))
    plans = evidence.get("plans")
    commitments = evidence.get("commitments")
    runs = evidence.get("runs")
    nothing = (
        (plans is None or plans["total"] == 0)
        and (commitments is None or commitments["total"] == 0)
        and (runs is None or runs["live"] == 0)
        and not (plans and plans.get("degraded")))
    if nothing and section is None:
        return ("nothing to grill yet — no plans, no commitments, "
                "no live runs on record. Give me something to hold "
                "you to first.")
    lines = ["grill-me:"]
    if section in (None, "plans"):
        lines.extend(_render_plans(plans))
    if section in (None, "commitments"):
        lines.extend(_render_commitments(commitments))
    if section in (None, "runs"):
        lines.extend(_render_runs(runs))
    return "\n".join(lines)


def grill_me(*, context: Any = None,
             home: Path | str | None = None,
             section: str | None = None,
             manager: Any = None,
             now: float | None = None) -> str:
    """Gather the evidence and render the grill in one call.

    ``home`` defaults to the context's home path; ``manager`` is
    the runtime's delegation manager when one is live.
    """
    if home is None:
        home = _home_from_context(context)
    evidence = gather_grill_evidence(
        manager=manager, home=home, now=now)
    return render_grill(evidence, section=section)
