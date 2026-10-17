"""Autonomy tools — weakness management, presence, patterns.

Spine tools so the brain can inspect and manage the autonomous
nervous system through plain language:

* ``weakness_list`` — open weaknesses and proposals
* ``weakness_approve`` — owner approves a fix proposal
* ``weakness_dismiss`` — owner dismisses a weakness
* ``presence_sense`` — what the system knows right now
* ``interests`` — current interest model
* ``__presence_heartbeat__`` — scheduler-internal heartbeat (notice,
  prepare, surface). Registered via closure in ``register()``, NOT via
  the ``TOOLS`` dict: the registry calls fns with no ``context`` arg,
  and a ``context``-taking fn would fail with TypeError (the pre-existing
  ``presence_sense`` failure mode). See ``pulse``/``briefing``.
"""

from __future__ import annotations

from typing import Any


def _db(context: Any) -> Any:
    from ..storage.db import Database
    ws = getattr(getattr(context, "settings", None), "workspace_dir", ".")
    return Database(ws or ".")


def weakness_list(context: Any) -> dict[str, Any]:
    """List open weaknesses and fix proposals."""
    from ..autonomy.weakness import open_weaknesses
    db = _db(context)
    items = open_weaknesses(db)
    if not items:
        return {"ok": True, "weaknesses": [],
                "note": "no open weaknesses — the system is healthy"}
    return {"ok": True, "weaknesses": items}


def weakness_approve(context: Any, weakness_id: str) -> dict[str, Any]:
    """Approve a weakness fix proposal (owner only)."""
    from ..autonomy.weakness import approve_weakness
    db = _db(context)
    if approve_weakness(db, weakness_id):
        return {"ok": True,
                "note": f"proposal {weakness_id} approved — "
                        "ready for sandbox build/test"}
    return {"ok": False,
            "error": "not found or not in 'proposed' state"}


def weakness_dismiss(context: Any, weakness_id: str) -> dict[str, Any]:
    """Dismiss a weakness (owner only)."""
    from ..autonomy.weakness import ensure_schema
    db = _db(context)
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'dismissed' WHERE id = ?",
        (weakness_id,))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": bool(cur.rowcount)}


def presence_sense(context: Any) -> dict[str, Any]:
    """What the system knows right now: time, windows, interests, health."""
    from ..autonomy.presence import sense
    db = _db(context)
    return {"ok": True, "presence": sense(db)}


def interests(context: Any, limit: int = 10) -> dict[str, Any]:
    """Current interest model — what the owner cares about, with decay."""
    from ..autonomy.patterns import current_interests, rising_interests
    db = _db(context)
    return {
        "ok": True,
        "interests": current_interests(db, limit=limit),
        "rising": rising_interests(db),
    }


TOOLS = {
    "weakness_list": {
        "fn": weakness_list,
        "description": "List open system weaknesses and fix proposals.",
        "capabilities": {"owner"},
    },
    "weakness_approve": {
        "fn": weakness_approve,
        "description": "Approve a weakness fix proposal by id. Owner only.",
        "capabilities": {"owner"},
        "args": {"weakness_id": "the weakness id"},
    },
    "weakness_dismiss": {
        "fn": weakness_dismiss,
        "description": "Dismiss a weakness by id. Owner only.",
        "capabilities": {"owner"},
        "args": {"weakness_id": "the weakness id"},
    },
    "presence_sense": {
        "fn": presence_sense,
        "description": "What the system knows right now: time, predicted "
                       "active windows, interests, open weaknesses.",
        "capabilities": {"owner"},
    },
    "interests": {
        "fn": interests,
        "description": "The owner's interest model — topics with "
                       "time-decayed scores, plus rising interests.",
        "capabilities": {"owner"},
    },
}


def register(registry: Any) -> None:
    for name, spec in TOOLS.items():
        registry.register(
            name,
            spec["fn"],
            description=spec["description"],
            capability="owner",
        )
    # Scheduler-internal heartbeat tool. The ``presence-heartbeat`` job is
    # stored with payload_kind="tool" and payload
    # {"tool": "__presence_heartbeat__"}, so the tool MUST exist in the
    # tool registry — the scheduler's ``register_action`` path (see
    # ``ensure_heartbeat_job``) targets a different scheduler class and
    # never runs here, which is why the job failed with "unknown tool"
    # every 30 min. Closure over ``registry.context``: ``registry.call``
    # passes no ``context`` arg, so a ``context``-taking fn (the ``TOOLS``
    # dict pattern) would fail with TypeError. Same pattern as
    # ``pulse``/``briefing``. Uses the canonical ``workspace_db`` (not
    # ``Database(workspace_dir)``) so the heartbeat shares one state
    # store with the rest of autonomy.
    _context = registry.context

    @registry.register(
        "__presence_heartbeat__",
        description="Run the slow presence heartbeat: notice rising "
        "interests, prepare surfaces, surface one due item. "
        "Scheduler-internal; runs every ~30 min.",
        capability="owner",
    )
    def _presence_heartbeat_tool() -> dict[str, Any]:
        from ..autonomy.idle import workspace_db
        from ..autonomy.presence import heartbeat

        ws = getattr(getattr(_context, "settings", None),
                     "workspace_dir", None) or "."
        db = workspace_db(ws)
        return {"ok": True, "heartbeat": heartbeat(db)}
