"""Triage tool: ``triage`` — the brain's handle on the autocraft
issue-triage organ, plus the ``__triage_sync__`` scheduler tick.

One tool, action-dispatched, over :mod:`nomorals.triage`; the organ
owns the mirror, the duplicate/classify/priority rules and the
decisions — this module is only the surface.

``action=track`` tracks a repo (``repo=owner/name``);
``action=untrack`` drops one; ``action=repos`` lists tracked repos;
``action=sync`` mirrors a repo's issues (``repo=``);
``action=triage`` runs one full pass over a repo (sync + duplicates
+ classify + prioritize); ``action=queue`` lists the repo's open
items in work-rank order; ``action=item`` shows one item
(``ref=owner/repo#n``); ``action=duplicates`` finds duplicate groups
(``repo=``); ``action=decisions`` lists open decisions (``repo=``);
``action=answer`` answers one (``id=``, ``option=``, ``note=``);
``action=apply`` applies an answered decision (``id=``).

Without a GitHub credential every action that needs GitHub answers
honestly as not connected — run ``nm connectors connect --name
github`` first — never a fabricated link. With no model the code
rung runs and model verdicts are recorded as unavailable, never
invented.
"""
from __future__ import annotations

from typing import Any

from ..triage import github as _github_mod
from ..triage.engine import TriageEngine
from ..triage.jobs import TRIAGE_SYNC_TOOL
from ..triage.store import TriageStore, triage_path

__all__ = ["register"]


def _store(context: Any) -> TriageStore:
    settings = getattr(context, "settings", None)
    return TriageStore(triage_path(settings))


def _suggest(context: Any):
    """A ``(prompt) -> text`` model callable, or None.

    A suggest passed in on the context (``context._triage_suggest``)
    wins — the test seam. Otherwise the brain, wrapped the way the
    ladder wraps it: a failed model call returns "" and the caller
    falls through to the code rung.
    """
    injected = getattr(context, "_triage_suggest", None)
    if injected is not None:
        return injected
    try:
        from ..llm.base import Message
        from ..llm.brain import brain_for
    except Exception:  # noqa: BLE001 - no brain, no model rung
        return None
    try:
        brain = brain_for(context)
    except Exception:  # noqa: BLE001 - no brain, no model rung
        return None

    def _call(prompt: str) -> str:
        try:
            resp = brain.chat(
                [Message(role="user", content=prompt)],
                task_kind="chat", timeout_s=30.0)
            return getattr(resp, "text", "") or ""
        except Exception:  # noqa: BLE001 - the model must never kill a job
            return ""

    return _call


def _engine(context: Any) -> TriageEngine | None:
    """The triage engine, or None when GitHub is not connected."""
    client = _github_mod.github_client(context)
    if client is None:
        return None
    return TriageEngine(client, _store(context),
                        suggest=_suggest(context))


def _read_engine(context: Any) -> TriageEngine:
    """An engine for store-only reads (no GitHub client needed)."""
    return TriageEngine(None, _store(context), suggest=_suggest(context))


def register(registry: Any) -> None:
    """Attach the ``triage`` tool and the sync tick tool."""
    context = registry.context

    @registry.register(
        "triage",
        description=(
            "Issue triage over tracked GitHub repos (autocraft port): "
            "mirror issues, find duplicate groups, classify type + "
            "flags, assign P0-P4 priority, rank the work, ask the "
            "owner the scope questions. action=track tracks a repo "
            "(repo=owner/name); action=untrack drops one; "
            "action=repos lists tracked repos; action=sync mirrors a "
            "repo (repo=); action=triage runs one full pass (repo=); "
            "action=queue lists open items in work-rank order "
            "(repo=); action=item shows one item "
            "(ref=owner/repo#n); action=duplicates finds duplicate "
            "groups (repo=); action=decisions lists open decisions "
            "(repo=); action=answer answers one (id=, option=, "
            "note=); action=apply applies an answered decision "
            "(id=). Without a GitHub credential every action that "
            "needs GitHub answers honestly as not connected, never a "
            "fabricated link."
        ),
        capability="owner",
    )
    def triage_tool(
        action: str = "queue",
        repo: str = "",
        ref: str = "",
        id: str = "",
        option: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        store = _store(context)
        action = (action or "queue").strip().lower()
        if action == "track":
            if not (repo or "").strip():
                return {"ok": False,
                        "error": "give a repo: action=track "
                                 "repo=owner/name"}
            store.track(repo.strip())
            return {"ok": True, "repo": repo.strip(),
                    "tracked": store.tracked()}
        if action == "untrack":
            if not (repo or "").strip():
                return {"ok": False,
                        "error": "give a repo: action=untrack "
                                 "repo=owner/name"}
            store.untrack(repo.strip())
            return {"ok": True, "repo": repo.strip(),
                    "tracked": store.tracked()}
        if action == "repos":
            return {"ok": True, "repos": store.tracked()}
        if action in ("queue", "decisions", "item"):
            engine = _read_engine(context)
            if action == "queue":
                if not (repo or "").strip():
                    return {"ok": False,
                            "error": "give a repo: action=queue "
                                     "repo=owner/name"}
                items = engine.queue(repo.strip())
                return {"ok": True, "repo": repo.strip(), "items": [
                    {"key": row.get("key"), "number": row.get("number"),
                     "title": row.get("title"),
                     "type": row.get("type"),
                     "priority": row.get("priority") or "P2",
                     "work_rank": row.get("work_rank") or 0,
                     "security": bool(row.get("security")),
                     "needs_decision": bool(row.get("needs_decision")),
                     "url": row.get("url", "")}
                    for row in items]}
            if action == "decisions":
                return {"ok": True,
                        "decisions": engine.decisions(
                            (repo or "").strip() or None)}
            item = store.get_item((ref or "").strip())
            if item is None:
                return {"ok": False,
                        "error": f"no item {ref} mirrored — sync the "
                                 "repo first"}
            return {"ok": True, "item": item}
        engine = _engine(context)
        if engine is None:
            return {"ok": False,
                    "error": _github_mod.github_not_connected_reason()}
        if action == "sync":
            if not (repo or "").strip():
                return {"ok": False,
                        "error": "give a repo: action=sync "
                                 "repo=owner/name"}
            out = engine.sync(repo.strip())
            if out.get("error"):
                return {"ok": False, "repo": repo.strip(),
                        "error": out["error"]}
            return {"ok": True, **out}
        if action == "triage":
            if not (repo or "").strip():
                return {"ok": False,
                        "error": "give a repo: action=triage "
                                 "repo=owner/name"}
            out = engine.triage(repo.strip())
            return {"ok": True, **out}
        if action == "duplicates":
            if not (repo or "").strip():
                return {"ok": False,
                        "error": "give a repo: action=duplicates "
                                 "repo=owner/name"}
            groups = engine.find_duplicates(repo.strip())
            return {"ok": True, "repo": repo.strip(),
                    "groups": groups}
        if action == "answer":
            try:
                decision = engine.answer_decision(
                    (id or "").strip(), (option or "").strip(), note=note)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "decision": decision}
        if action == "apply":
            try:
                out = engine.apply_decision((id or "").strip())
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, **out}
        return {"ok": False,
                "error": f"unknown action {action!r} — track, untrack, "
                         "repos, sync, triage, queue, item, duplicates, "
                         "decisions, answer, apply"}

    @registry.register(
        TRIAGE_SYNC_TOOL,
        description="Run one triage sync tick: sync every tracked "
                    "repo's issue mirror and run the code-rung triage "
                    "pass (the model rung only when the brain is up). "
                    "Scheduler-internal; runs hourly. Never raises — "
                    "a bad tick is recorded, never fatal to the job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _triage_sync_tool() -> dict[str, Any]:
        store = _store(context)
        engine = _engine(context)
        if engine is None:
            return {"ok": False,
                    "error": _github_mod.github_not_connected_reason()}
        total_groups = 0
        total_classified = 0
        ran: list[str] = []
        for repo in store.tracked():
            out = engine.triage(repo)
            total_groups += int(out.get("groups", 0))
            total_classified += int(out.get("classified", 0))
            ran.append(repo)
        return {"ok": True, "repos": ran, "groups": total_groups,
                "classified": total_classified,
                "model_calls": engine.model_calls,
                "model_budget_exhausted": engine.model_budget_exhausted}
