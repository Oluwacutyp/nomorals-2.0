"""Post-task distillation into the flat-file skillbook (L4).

When a mission finishes, what it learned should not evaporate with its
transcript.  :func:`distill_task` turns a finished task record into a
**draft** skill file in the library's drafts area — deterministic
template extraction (steps, roles, tools, lessons), no model call, so
it can run on every terminal mission without cost or flakiness.  The
brain may amend a draft afterwards through the same file format.

Promotion rule (also stated in ``skill-writing.md`` and enforced by
``FileSkillLibrary``): a draft stays unproven — routable but labelled —
until it has been loaded in :data:`PROMOTE_AFTER_SUCCESSES` distinct
missions that succeeded, or the owner promotes it explicitly with
``/skills promote <id>``.  A draft whose slug matches a live skill is a
*revision* of that skill (``revises:``), never a silent overwrite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..core.text import slugify
from .filebook import PROMOTE_AFTER_SUCCESSES, FileSkillLibrary, SkillDoc

__all__ = ["distill_task", "should_distill", "PROMOTE_AFTER_SUCCESSES"]


def should_distill(task: Mapping[str, Any]) -> bool:
    """Distillation is for work with something to teach: a mission that
    ran at least three steps, or one that recorded lessons, or one that
    failed (failures teach the most)."""
    steps = task.get("steps") or []
    lessons = task.get("lessons") or []
    return len(steps) >= 3 or bool(lessons) \
        or str(task.get("outcome") or "") == "failed"


def _title_for(goal: str) -> str:
    text = " ".join(str(goal or "untitled task").split())
    return (text[:72].rstrip() + "…") if len(text) > 73 else text


def distill_task(library: FileSkillLibrary,
                 task: Mapping[str, Any]) -> dict[str, Any]:
    """Write a draft skill from a finished task record.

    ``task`` keys: ``goal``, ``outcome`` ("done"/"failed"), ``summary``,
    ``mission_id``, ``steps`` (list of ``{name, role, tools}``),
    ``lessons`` (list of strings), ``tags`` (list of strings).
    Returns ``{"ok", "draft_id", "path", "revises"}``.
    """
    goal = str(task.get("goal") or "").strip()
    if not goal:
        return {"ok": False, "reason": "task has no goal to distill"}
    slug = slugify(goal, limit=48, fallback="task")
    steps: Sequence[Mapping[str, Any]] = task.get("steps") or []
    lessons: Sequence[str] = [str(x) for x in (task.get("lessons") or [])]

    topics: list[str] = []
    for tag in task.get("tags") or []:
        tag = str(tag).strip().lower()
        if tag and tag not in topics:
            topics.append(tag)
    for step in steps:
        role = str(step.get("role") or "").strip().lower()
        if role and role not in topics:
            topics.append(role)
        for tool in step.get("tools") or []:
            tool = str(tool).strip().lower()
            if tool and tool not in topics:
                topics.append(tool)
    topics = topics[:8]

    lines = [f"# {_title_for(goal)}", "",
             "## When this applies", "", goal, ""]
    if steps:
        lines += ["## Approach that worked", ""]
        for step in steps:
            name = str(step.get("name") or "step")
            role = str(step.get("role") or "")
            tools = ", ".join(str(t) for t in (step.get("tools") or []))
            detail = " — ".join(p for p in (role, tools) if p)
            lines.append(f"- {name}" + (f" ({detail})" if detail else ""))
        lines.append("")
    summary = str(task.get("summary") or "").strip()
    outcome = str(task.get("outcome") or "done")
    lines += ["## Evidence", "",
              f"Distilled from a mission that ended **{outcome}**"
              + (f": {summary}" if summary else "."),
              ""]
    if lessons:
        lines += ["## Lessons", ""]
        lines += [f"- {lesson}" for lesson in lessons]
        lines.append("")

    live = library.get(slug)
    doc = SkillDoc(
        id=slug,
        title=f"Distilled: {_title_for(goal)}",
        summary=(f"Playbook distilled from a completed task: "
                 f"{_title_for(goal)}"),
        topics=tuple(topics),
        triggers=(goal.lower()[:60],),
        layer="distilled",
        version=1,
        body="\n".join(lines).strip(),
        origin="draft",
        revises=slug if live is not None else "",
    )
    path = library.write_draft(doc)
    return {"ok": True, "draft_id": slug, "path": str(path),
            "revises": doc.revises,
            "mission_id": str(task.get("mission_id") or "")}
