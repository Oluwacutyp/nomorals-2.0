"""Fleet tool: the brain's surface over the fleet organ.

One tool, action-dispatched, over :mod:`nomorals.agents.fleet`:
the organ owns the model, the store, the worktrees and the merge;
this module is only the surface.

``action=create`` makes a fleet over a repo (``repo`` + ``name``).
``dispatch`` runs a coding brief as an isolated task (``fleet`` +
``brief``, optional ``key``; ``wait`` seconds blocks until the task
leaves queued/running). ``board`` renders the Kanban board,
``list`` lists fleets, ``review`` shows a task's diff,
``approve`` merges a reviewed task into the repo, ``cleanup``
removes finished tasks' worktrees. The fleet home is
``<home>/fleet``; a fleet or task is addressed by id, id prefix,
or (fleets) name.

Capability is EXEC_CODE: a dispatch runs the real coding agent
inside a worktree (model calls, file writes, test runs), and an
approve applies a patch to a repo. Board/list/review are reads,
but the tool's strongest action sets the gate — the same shape as
``code_executor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store(context: Any):  # -> FleetStore
    from ..agents.fleet import FleetStore

    home = _home_of(context)
    if not home:
        raise ValueError("fleet needs a home directory (no settings.home)")
    return FleetStore(Path(home) / "fleet", context=context)


def register(registry: Any) -> None:
    """Attach the ``fleet`` tool to a registry."""
    from ..agents.fleet import FleetError, render_board

    context = registry.context

    @registry.register(
        "fleet",
        description="Run parallel coding agents as a fleet: create a "
                    "fleet over a repo, dispatch coding briefs (each "
                    "runs isolated in its own git worktree), watch "
                    "the Kanban board, review a task's diff, approve "
                    "to merge it into the repo. "
                    "action=create|dispatch|board|list|review|approve|"
                    "cleanup.",
        capability=Capability.EXEC_CODE,
    )
    def fleet(action: str = "board", fleet: str = "", repo: str = "",
              name: str = "", brief: str = "", key: str = "",
              task: str = "", wait: float = 0) -> dict[str, Any]:
        store = _store(context)
        act = (action or "").strip().lower()
        try:
            if act == "create":
                if not repo:
                    raise FleetError("create needs repo=<path>")
                return store.create_fleet(name or "fleet", repo)
            if act == "dispatch":
                if not fleet:
                    raise FleetError("dispatch needs fleet=<id or name>")
                record = store.dispatch(
                    fleet, brief, key=key or None, run_sync=False)
                if wait and float(wait) > 0:
                    record = store.wait(record["task_id"],
                                        timeout=float(wait))
                return record
            if act == "board":
                if not fleet:
                    fleets = store.list_fleets()
                    return {"fleets": fleets}
                board = store.board(fleet)
                payload = board.to_dict()
                payload["text"] = render_board(board)
                return payload
            if act == "list":
                return {"fleets": store.list_fleets()}
            if act == "review":
                if not task:
                    raise FleetError("review needs task=<id>")
                return store.review(task)
            if act == "approve":
                if not task:
                    raise FleetError("approve needs task=<id>")
                return store.approve(task)
            if act == "cleanup":
                if not fleet:
                    raise FleetError("cleanup needs fleet=<id or name>")
                return store.cleanup(fleet)
            raise FleetError(
                f"unknown fleet action {action!r} (create|dispatch|"
                "board|list|review|approve|cleanup)")
        except FleetError as exc:
            return {"ok": False, "error": str(exc)}
