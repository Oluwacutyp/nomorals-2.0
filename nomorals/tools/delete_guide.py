"""Delete-guide tool: ``delete_guide`` — the AccountKiller (#28)
runbook registry as a brain tool.

One tool, action-dispatched, over :mod:`nomorals.accounts.deletion`;
the organ owns the registry, the rendering and the walk state — this
module is only the surface.

``action=guide`` renders the full runbook for a service
(``service=...``): difficulty and grace window up front, the
export-first warning, every step, the delete URL. A stale guide is
labelled stale, never served as current. ``action=walk`` walks the
runbook with the owner (``service=...``, ``done=done|next`` advances
a step, ``done=reset`` restarts) and stops at every step that needs
the owner's hand. ``action=list`` lists every service with its
difficulty and grace window. ``action=search`` finds guides by
keyword (``query=...``). ``action=add`` grows the registry by
discovery (``service=...``, ``steps=...`` one per line, prefix a step
with ``HAND: `` when it needs the owner's hand, ``difficulty=...``,
``grace_days=...``, ``delete_url=...``, ``export_first=...``).
``action=status`` reports the guide count, how many are stale, and
the store path.

An unknown service is answered honestly: the tool says what it does
know (nearby services, if any) instead of inventing a guide.
"""

from __future__ import annotations

from typing import Any

from ..accounts.deletion import (
    DIFFICULTIES,
    STALE_AFTER_DAYS,
    DeletionGuide,
    DeletionStep,
    DeletionStore,
    is_stale,
    render_guide,
    walk_advance,
    walk_current,
    walk_reset,
)
from ..core.policy import Capability

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_steps(raw: str) -> tuple[DeletionStep, ...]:
    steps: list[DeletionStep] = []
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        hand = text.lower().startswith("hand:")
        if hand:
            text = text[5:].strip()
        if text:
            steps.append(DeletionStep(text=text, owner_hand=hand))
    return tuple(steps)


def register(registry: Any) -> None:
    """Attach the ``delete_guide`` tool."""
    context = registry.context

    @registry.register(
        "delete_guide",
        description=(
            "How to delete or close an account, for any service: a "
            "maintained deletion-runbook registry, not a deleter. "
            "action=guide renders the full runbook for a service "
            "(service=<name>): difficulty and grace window up front, "
            "the export-first warning, every step, the delete URL — a "
            "stale guide is labelled stale, never served as current. "
            "action=walk walks the runbook with the owner "
            "(service=<name>, done=done|next advances a step, "
            "done=reset restarts) and stops at every step that needs "
            "the owner's hand (login, CAPTCHA, final confirm). "
            "action=list lists every service with difficulty and "
            "grace window. action=search finds guides by keyword "
            "(query=<text>). action=add grows the registry by "
            "discovery (service=<name>, steps=<one per line, prefix "
            "a step with 'HAND: ' when it needs the owner's hand>, "
            "difficulty=<easy|medium|hard|impossible>, "
            "grace_days=<days to change your mind>, "
            "delete_url=<link or empty>, export_first=<true|false>). "
            "action=status reports the guide count, how many are "
            "stale, and the store path. An unknown service is "
            "answered honestly instead of inventing a guide — Devon "
            "guides and assists, the account holder's own hand "
            "finishes."
        ),
        capability=Capability.MEM_WRITE,
    )
    def delete_guide_tool(action: str = "guide", service: str = "",
                          query: str = "", steps: str = "",
                          difficulty: str = "medium", grace_days: str = "",
                          delete_url: str = "", export_first: str = "",
                          done: str = "") -> dict[str, Any]:
        verb = (action or "guide").strip().lower()
        home = _home_of(context)
        if home is None:
            raise ValueError("delete_guide needs a home (settings.home)")
        store = DeletionStore(home)
        if verb == "guide":
            name = (service or "").strip()
            if not name:
                raise ValueError("delete_guide guide needs a service")
            guide = store.get(name)
            if guide is None:
                nearby = [g.service for g in store.search(name)][:5]
                return {
                    "ok": False,
                    "action": "guide",
                    "service": name,
                    "error": (
                        f"I don't have a deletion guide for {name} "
                        "yet — deletion flows churn, so I won't "
                        "invent one."
                    ),
                    "known_nearby": nearby,
                }
            return {
                "ok": True,
                "action": "guide",
                "service": guide.service,
                "difficulty": guide.difficulty,
                "grace_days": guide.grace_days,
                "stale": is_stale(guide),
                "checked_at": guide.checked_at,
                "text": render_guide(guide),
            }
        if verb == "walk":
            name = (service or "").strip()
            if not name:
                raise ValueError("delete_guide walk needs a service")
            guide = store.get(name)
            if guide is None:
                return {
                    "ok": False,
                    "action": "walk",
                    "service": name,
                    "error": (
                        f"I don't have a deletion guide for {name} "
                        "yet — I can't walk a runbook I don't have."
                    ),
                }
            command = (done or "").strip().lower()
            if command in ("done", "next"):
                text = walk_advance(store, guide)
            elif command == "reset":
                text = walk_reset(store, guide)
            else:
                text = walk_current(store, guide)
            position = store.walk_position(guide.service)
            finished = position >= len(guide.steps)
            return {
                "ok": True,
                "action": "walk",
                "service": guide.service,
                "step": position,
                "of": len(guide.steps),
                "finished": finished,
                "text": text,
            }
        if verb == "list":
            guides = store.list_all()
            return {
                "ok": True,
                "action": "list",
                "count": len(guides),
                "guides": [
                    {
                        "service": guide.service,
                        "difficulty": guide.difficulty,
                        "grace_days": guide.grace_days,
                        "stale": is_stale(guide),
                    }
                    for guide in guides
                ],
            }
        if verb == "search":
            needle = (query or "").strip()
            guides = store.search(needle)
            return {
                "ok": True,
                "action": "search",
                "query": needle,
                "count": len(guides),
                "guides": [
                    {
                        "service": guide.service,
                        "difficulty": guide.difficulty,
                        "grace_days": guide.grace_days,
                        "stale": is_stale(guide),
                    }
                    for guide in guides
                ],
            }
        if verb == "add":
            name = (service or "").strip()
            if not name:
                raise ValueError("delete_guide add needs a service")
            parsed_steps = _parse_steps(steps)
            if not parsed_steps:
                raise ValueError(
                    "delete_guide add needs steps (one per line)"
                )
            level = (difficulty or "medium").strip().lower()
            if level not in DIFFICULTIES:
                raise ValueError(
                    f"bad difficulty {level!r}: use one of "
                    f"{DIFFICULTIES}"
                )
            try:
                grace = int((grace_days or "0").strip())
            except ValueError as exc:
                raise ValueError(
                    f"bad grace_days {grace_days!r}: use a number"
                ) from exc
            if grace < 0:
                raise ValueError("grace_days cannot be negative")
            guide = DeletionGuide(
                service=name,
                delete_url=(delete_url or "").strip(),
                steps=parsed_steps,
                difficulty=level,
                grace_days=grace,
                export_first=_parse_bool(export_first, True),
            )
            store.add(guide)
            return {
                "ok": True,
                "action": "add",
                "service": guide.service,
                "difficulty": guide.difficulty,
                "grace_days": guide.grace_days,
                "steps": len(guide.steps),
                "text": render_guide(guide),
            }
        if verb == "status":
            guides = store.list_all()
            stale = [g.service for g in guides if is_stale(g)]
            return {
                "ok": True,
                "action": "status",
                "count": len(guides),
                "stale_count": len(stale),
                "stale_after_days": STALE_AFTER_DAYS,
                "stale": stale,
                "store": store.path,
            }
        raise ValueError(f"unknown delete_guide action {verb!r}")
