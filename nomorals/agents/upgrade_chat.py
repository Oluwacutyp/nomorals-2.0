"""Chat rendering for the research → approve → evolve loop.

Pure text renderers over upgrade-queue proposals (research digest
tickets that cleared the confidence/vagueness gate and were filed by
:class:`nomorals.agents.upgrade_queue.UpgradeQueue`).  Dispatch lives in
``nomorals.agents.partner_runtime._control_upgrade``; these functions do
the formatting so they stay unit-testable without a runtime.

All upgrade commands are owner-only by construction: ``handle_control``
is only reachable from operator chats (the ``_is_operator`` gate in
``on_message`` — a slash from anyone else falls through to ordinary
conversation), so no per-command owner check is needed here.
"""
from __future__ import annotations

import difflib
from typing import Any

__all__ = [
    "UPGRADE_USAGE",
    "resolve_proposal",
    "render_upgrade_list",
    "render_upgrade_show",
    "render_upgrade_diff",
    "render_applied_digest",
]

UPGRADE_USAGE = (
    "usage: /upgrade list | /upgrade show <id> | /upgrade diff <id> | "
    "/upgrade approve <id> | /upgrade deny <id> <reason> | /upgrade applied"
)

#: statuses searched by resolve_proposal, in review order.
_STATUSES = ("proposed", "approved", "implemented", "failed", "denied")


def _plan(proposal: dict[str, Any]) -> dict[str, Any]:
    plan = proposal.get("patch_plan")
    return plan if isinstance(plan, dict) else {}


def resolve_proposal(queue: Any, ref: str) -> tuple[dict | None, str]:
    """Resolve an id, id prefix, or title substring to a proposal.

    Returns ``(proposal, "")`` on success or ``(None, message)`` when the
    reference is missing, unknown, or ambiguous — the message is
    chat-ready.
    """
    filt = (ref or "").strip()
    if not filt:
        return None, ("give me an id — /upgrade list shows the pending "
                      "ones, or /upgrade applied for the landed ones.")
    direct = queue.get(filt)
    if direct is not None:
        return direct, ""
    low = filt.lower()
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for status in _STATUSES:
        for p in queue.list(status=status, limit=200):
            pid = str(p.get("id") or "")
            if pid in seen:
                continue
            if (pid.lower().startswith(low)
                    or low in str(p.get("title") or "").lower()):
                seen.add(pid)
                matches.append(p)
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, (f"no upgrade proposal matching {filt!r} — /upgrade "
                      "list to see the pending ones.")
    lines = [f"{filt!r} is ambiguous — matches:"]
    for p in matches[:8]:
        lines.append(f"  {p.get('id')} [{p.get('status')}] "
                     f"{str(p.get('title') or '')[:60]}")
    return None, "\n".join(lines)


def render_upgrade_list(proposals: list[dict[str, Any]]) -> str:
    """One line per pending proposal: id, title, source."""
    if not proposals:
        return ("no pending upgrade proposals — research findings that "
                "clear the ticket gate land here for your review.")
    lines = [f"pending upgrades ({len(proposals)}):"]
    for p in proposals[:25]:
        src = p.get("source") or "?"
        lines.append(f"  · {p.get('id')} [{src}] {str(p.get('title') or '')[:80]}")
    if len(proposals) > 25:
        lines.append(f"  … +{len(proposals) - 25} more")
    lines.append("review: /upgrade show <id> · preview edits: /upgrade diff <id>")
    return "\n".join(lines)


def render_upgrade_show(proposal: dict[str, Any]) -> str:
    """The full ticket: problem, patch plan, files, tests, risk."""
    plan = _plan(proposal)
    pid = proposal.get("id") or "?"
    lines = [
        f"⬆️ {pid} [{proposal.get('status') or 'proposed'}]",
        str(proposal.get("title") or ""),
        "",
        "problem:",
        f"  {str(proposal.get('rationale') or '—')}",
        "",
        "patch plan:",
    ]
    files = [f for f in (proposal.get("files") or []) if f]
    lines.append(f"  files: {', '.join(files) if files else '—'}")
    tests = [t for t in (proposal.get("tests") or []) if t]
    lines.append(f"  tests: {', '.join(tests) if tests else '—'}")
    steps = plan.get("steps")
    if isinstance(steps, list) and steps:
        lines.append("  steps:")
        for i, step in enumerate(steps[:12], 1):
            lines.append(f"    {i}. {step}")
        if len(steps) > 12:
            lines.append(f"    … +{len(steps) - 12} more")
    evo_id = plan.get("evolution_proposal_id")
    if evo_id:
        lines.append(f"  evolution proposal: {evo_id}")
    skill_edit_id = plan.get("skill_edit_id")
    if skill_edit_id:
        lines.append(f"  staged skill edit: {skill_edit_id}")
    for key in ("budget", "branch"):
        if plan.get(key) not in (None, ""):
            lines.append(f"  {key}: {plan[key]}")
    # anything else the plan carries that we don't have a slot for
    known = {"source", "steps", "evolution_proposal_id", "skill_edit_id",
             "risk", "risks", "budget", "branch"}
    extra = {k: v for k, v in plan.items() if k not in known}
    for key, val in list(extra.items())[:6]:
        lines.append(f"  {key}: {_short(val)}")
    risk = plan.get("risk") or plan.get("risks")
    if risk:
        risk_txt = ", ".join(str(r) for r in risk) if isinstance(risk, list) \
            else str(risk)
    else:
        risk_txt = f"unstated in the ticket — touches {len(files)} file(s)"
    lines.append("")
    lines.append(f"risk: {risk_txt}")
    src = proposal.get("source")
    if src:
        lines.append(f"source: {src}")
    lines.append("")
    lines.append(f"preview the edits: /upgrade diff {pid} · "
                 f"run it: /upgrade approve {pid}")
    return "\n".join(lines)


def render_upgrade_diff(proposal: dict[str, Any],
                        evo_proposal: Any = None) -> str:
    """Preview the actual patch BEFORE approve.

    When the ticket references a planned evolution proposal and that
    proposal's edits were loaded, render the real unified hunks.  (The
    evolution agent plans edits against the tree, so what you see is
    what would be written — stale plans are rejected at apply time.)
    Otherwise render the ticket's patch plan concretely: steps, files,
    and every plan field.
    """
    plan = _plan(proposal)
    pid = proposal.get("id") or "?"
    lines = [f"🔍 diff preview — {pid}",
             str(proposal.get("title") or ""), ""]
    edits = getattr(evo_proposal, "edits", None)
    if edits:
        evo_id = getattr(evo_proposal, "id", "?")
        lines.append(f"evolution plan {evo_id} — {len(edits)} edit(s):")
        for edit in edits[:8]:
            lines.extend(_render_hunk(edit if isinstance(edit, dict) else {}))
        if len(edits) > 8:
            lines.append(f"  … +{len(edits) - 8} more edits")
        lines.append("")
        lines.append(f"looks right? /upgrade approve {pid} runs it through "
                     "the test-gated apply.")
        return "\n".join(lines)

    evo_id = plan.get("evolution_proposal_id")
    if evo_id:
        lines.append(f"references evolution proposal {evo_id}, but its "
                     "planned edits are not on disk — the preview falls "
                     "back to the ticket text:")
        lines.append("")
    skill_edit_id = plan.get("skill_edit_id")
    if skill_edit_id:
        lines.append(f"staged skill edit {skill_edit_id}: the skill loop "
                     "holds the concrete diff; approving commits it after "
                     "its own verification.")
        lines.append("")
    steps = plan.get("steps")
    if isinstance(steps, list) and steps:
        lines.append("steps the apply will run:")
        for i, step in enumerate(steps[:15], 1):
            lines.append(f"  {i}. {step}")
        if len(steps) > 15:
            lines.append(f"  … +{len(steps) - 15} more")
    else:
        known = {"source", "steps", "evolution_proposal_id", "skill_edit_id",
                 "risk", "risks"}
        shown = False
        for key, val in plan.items():
            if key in known:
                continue
            lines.append(f"  {key}: {_short(val)}")
            shown = True
        if not shown:
            lines.append("  (the ticket carries no structured patch detail "
                         "beyond files/tests — /upgrade show for the full "
                         "ticket)")
    files = [f for f in (proposal.get("files") or []) if f]
    if files:
        lines.append("")
        lines.append(f"files in scope: {', '.join(files)}")
    return "\n".join(lines)


def _render_hunk(edit: dict[str, Any]) -> list[str]:
    """One edit as chat-sized unified diff lines."""
    path = edit.get("path") or "?"
    old = str(edit.get("old") or "")
    new = str(edit.get("new") or "")
    lines = [f"  📄 {path}"]
    if not old and new:
        new_lines = new.splitlines()
        lines.append("    (new file)")
        for dl in new_lines[:25]:
            lines.append(f"    + {dl}")
        if len(new_lines) > 25:
            lines.append(f"    + … ({len(new_lines)} lines total)")
        return lines
    diff = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="", n=3))
    if not diff:
        lines.append("    (no visible change)")
        return lines
    for dl in diff[:40]:
        lines.append(f"    {dl}")
    if len(diff) > 40:
        lines.append(f"    … +{len(diff) - 40} more diff lines")
    return lines


def render_applied_digest(proposal: dict[str, Any]) -> str:
    """Short what-changed summary from ``record_implemented``'s result.

    ``applied_result`` is whatever the apply path returned:
    * evolution path — ``{"applied": bool, "edits": [...],
      "verified": bool, "commit": ..., "reason"/"report": ...}``
    * staged skill-edit path — ``{"ok": bool, "edit": id, ...}``
    * exception path — ``{"ok": False, "error": ...}``
    """
    res = proposal.get("applied_result")
    res = res if isinstance(res, dict) else {}
    title = str(proposal.get("title") or proposal.get("id") or "?")
    status = str(proposal.get("status") or "")

    ok = res.get("applied")
    if ok is None:
        ok = res.get("ok")
    if ok is None:
        ok = (status == "implemented")

    header = ("✅ upgrade applied" if ok else "❌ upgrade failed") + f": {title}"
    lines = [header]

    if res.get("applied") is True or (res.get("ok") is True and "edit" in res):
        # success — evolution path and staged-skill-edit path
        edits = res.get("edits") or []
        edit_id = res.get("edit")
        if edits:
            lines.append(f"  files touched ({len(edits)}): "
                         + ", ".join(str(e) for e in edits[:12]))
            if len(edits) > 12:
                lines.append(f"  … +{len(edits) - 12} more")
        elif edit_id:
            lines.append(f"  staged skill edit {edit_id} committed")
        if "verified" in res:
            gate = ("passed" if res.get("verified")
                    else "FAILED (reverted)")
            lines.append(f"  test gate: {gate}")
        commit = res.get("commit")
        if commit:
            branch = res.get("branch") or ""
            lines.append(f"  commit: {str(commit)[:12]}"
                         + (f" (branch {branch})" if branch else ""))
        elif "applied" in res:
            lines.append("  commit: on disk (not committed)")
        tag = res.get("tag")
        if tag:
            lines.append(f"  tag: {tag}")
        return "\n".join(lines)

    # failure paths
    err = res.get("error") or res.get("reason") or ""
    if err:
        lines.append(f"  error: {_short(err, 300)}")
    report = res.get("report") or ""
    if report:
        tail = [l for l in str(report).splitlines() if l.strip()]
        excerpt = " / ".join(tail[-3:])
        if excerpt:
            lines.append(f"  detail: {_short(excerpt, 400)}")
    if status == "failed":
        lines.append("  recorded as failed — the tree was left untouched")
    return "\n".join(lines)


def _short(val: Any, limit: int = 160) -> str:
    txt = val if isinstance(val, str) else repr(val)
    txt = " ".join(txt.split())
    return txt if len(txt) <= limit else txt[:limit] + "…"
