"""Spine-native memory tools — first-class, granular, model-driven.

The old path exposed memory as one mega-tool (``memory`` with an action
enum) plus bolted-on prompt injection. These tools are granular so the
brain picks them naturally from plain language:

- ``memory_remember`` — decide what's worth keeping (she decides, not just
  what she's told)
- ``memory_recall`` — search memories
- ``memory_forget`` — GUARDED: only on explicit owner command
- ``memory_update`` — correct a record
- ``memory_consolidate`` — additive distillation (never deletes)
- ``memory_timeline`` — episodic walk through time
- ``memory_anticipate`` — surface what might matter soon

The user's standing rule: never delete/forget except on explicit command.
Consolidation is additive — nothing is ever deleted by it.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)


def _manager(registry: Any):
    context = registry.context
    memory = getattr(context, "memory", None)
    if memory is None:
        raise RuntimeError("memory is off in this session")
    return memory


def register(registry: Any) -> None:
    """Register granular memory tools on the spine."""
    from ..core.policy import Capability

    @registry.register(
        "memory_remember",
        description=(
            "Save something worth remembering. Use when the owner shares a "
            "fact, preference, decision, or event that should persist beyond "
            "this conversation — or when you judge something from the "
            "exchange is worth keeping. kind: fact|preference|episode|"
            "decision|skill. importance 0-1. scope optionally namespaces it, "
            "e.g. project:devon-arena."
        ),
        capability=Capability.MEM_WRITE,
        parameters={
            "text": "str — what to remember",
            "kind": "str (optional) — fact|preference|episode|decision|skill",
            "importance": "float (optional) — 0-1, default 0.7",
            "scope": "str (optional) — memory space namespace",
        },
    )
    def _remember(text: str, kind: str = "fact",
                  importance: float = 0.7, scope: str = "") -> dict[str, Any]:
        memory = _manager(registry)
        record_id = memory.remember(
            text, kind=kind or "fact",
            importance=max(0.0, min(1.0, float(importance or 0.7))),
            scope=scope or None,
            source="spine:memory_remember",
        )
        return {"ok": True, "record_id": record_id}

    @registry.register(
        "memory_recall",
        description=(
            "Search long-term memory for anything relevant — facts, "
            "preferences, past decisions, episodes. Use proactively when a "
            "conversation touches on something she might know from before. "
            "Returns matching records with ids, kinds, and scores."
        ),
        capability=Capability.MEM_READ,
        parameters={
            "query": "str — what to search for",
            "limit": "int (optional) — max results, default 5",
            "kind": "str (optional) — filter by memory kind",
            "scope": "str (optional) — restrict to a memory space",
        },
    )
    def _recall(query: str, limit: int = 5, kind: str = "",
                scope: str = "") -> dict[str, Any]:
        memory = _manager(registry)
        result = memory.recall(query, limit=int(limit or 5),
                               kind=kind or None, scope=scope or None)
        records = getattr(result, "records", []) or []
        return {
            "ok": True,
            "matches": [
                {
                    "id": r.id,
                    "kind": getattr(r, "kind", ""),
                    "text": getattr(r, "text", ""),
                    "score": round(float(getattr(r, "score", 0.0) or 0.0), 3),
                    "created": getattr(r, "created_at", ""),
                }
                for r in records
            ],
        }

    @registry.register(
        "memory_forget",
        description=(
            "Delete a memory record by id. GUARDED: only use when the owner "
            "explicitly asks to forget or delete something — never on your "
            "own initiative, never as cleanup, never during consolidation. "
            "Pass confirmed=true only when the request is an explicit owner "
            "command."
        ),
        capability=Capability.MEM_WRITE,
        parameters={
            "record_id": "str — the record to delete",
            "confirmed": "bool — must be true; confirms explicit owner command",
        },
    )
    def _forget(record_id: str, confirmed: bool = False) -> dict[str, Any]:
        if not confirmed:
            return {
                "ok": False,
                "error": "forget requires confirmed=true — only on explicit "
                         "owner command, never on your own initiative",
            }
        memory = _manager(registry)
        removed = memory.forget(record_id)
        return {"ok": True, "removed": removed}

    @registry.register(
        "memory_update",
        description=(
            "Correct or refine an existing memory record. Use when the owner "
            "says 'that's wrong' or new information supersedes a stored "
            "record. Additive correction — the old text is replaced, the "
            "record keeps its id and history."
        ),
        capability=Capability.MEM_WRITE,
        parameters={
            "record_id": "str — the record to correct",
            "text": "str — the corrected content",
        },
    )
    def _update(record_id: str, text: str) -> dict[str, Any]:
        memory = _manager(registry)
        record = memory.find_one(record_id)
        if record is None:
            return {"ok": False, "error": f"no memory with id {record_id}"}
        # Additive correction: keep the record, update its text.
        update_fn = getattr(memory, "update", None)
        if update_fn is not None:
            update_fn(record_id, text)
        else:
            # Fallback: remember the correction linked to the original.
            memory.remember(
                text, kind=getattr(record, "kind", "fact"),
                importance=0.8, source=f"spine:correction-of:{record_id}",
            )
        return {"ok": True, "record_id": record_id}

    @registry.register(
        "memory_consolidate",
        description=(
            "Review recent episodic memories and distill them into durable "
            "semantic facts. ADDITIVE ONLY — consolidation never deletes "
            "anything. Run periodically or when memory feels cluttered. "
            "Returns what was distilled."
        ),
        capability=Capability.MEM_WRITE,
        parameters={},
    )
    def _consolidate() -> dict[str, Any]:
        memory = _manager(registry)
        consolidate_fn = getattr(memory, "consolidate_additive", None)
        if consolidate_fn is not None:
            result = consolidate_fn()
        else:
            result = memory.consolidate(forget=False)
        if isinstance(result, dict):
            return {"ok": True, **result}
        return {"ok": True, "result": str(result)}

    @registry.register(
        "memory_timeline",
        description=(
            "Walk through episodic memories in time order around a query — "
            "what happened, when, in sequence. Use for 'what did we do last "
            "week' or reconstructing how something unfolded."
        ),
        capability=Capability.MEM_READ,
        parameters={
            "query": "str — what to build a timeline around",
            "limit": "int (optional) — max episodes, default 10",
        },
    )
    def _timeline(query: str, limit: int = 10) -> dict[str, Any]:
        memory = _manager(registry)
        result = memory.recall(query, limit=int(limit or 10), kind="episode")
        records = getattr(result, "records", []) or []
        episodes = sorted(
            records, key=lambda r: getattr(r, "created_at", "") or "")
        return {
            "ok": True,
            "episodes": [
                {
                    "id": r.id,
                    "text": getattr(r, "text", ""),
                    "created": getattr(r, "created_at", ""),
                }
                for r in episodes
            ],
        }

    @registry.register(
        "memory_anticipate",
        description=(
            "Proactive recall: given what's happening now, surface memories "
            "that might matter soon — upcoming commitments, related past "
            "decisions, people involved. Use when planning or when the "
            "conversation hints at future needs."
        ),
        capability=Capability.MEM_READ,
        parameters={
            "query": "str — the current situation or plan",
            "limit": "int (optional) — max results, default 5",
        },
    )
    def _anticipate(query: str, limit: int = 5) -> dict[str, Any]:
        memory = _manager(registry)
        result = memory.recall(query, limit=int(limit or 5))
        records = getattr(result, "records", []) or []
        # Prefer commitments, decisions, and preferences for anticipation.
        ranked = sorted(
            records,
            key=lambda r: (
                getattr(r, "kind", "") in ("decision", "preference", "commitment"),
                float(getattr(r, "score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return {
            "ok": True,
            "relevant": [
                {
                    "id": r.id,
                    "kind": getattr(r, "kind", ""),
                    "text": getattr(r, "text", ""),
                }
                for r in ranked
            ],
        }
