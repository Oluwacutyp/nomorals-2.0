"""side_chats — agent tool for persistent conversation threads.

Wraps :class:`nomorals.social.chat.side_chats.SideChatManager` so agents
can create and manage side conversation threads with their own
memory/context/topic.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register", "init_side_chats"]

_manager: Any = None


def init_side_chats(manager: Any) -> None:
    """Initialize with a SideChatManager instance."""
    global _manager
    _manager = manager


def _get_manager() -> Any:
    if _manager is None:
        raise RuntimeError("SideChatManager not initialized. Call init_side_chats() first.")
    return _manager


def register(registry: Any) -> None:
    @registry.register(
        "side_chat",
        description=(
            "Manage persistent side conversation threads. Actions: create, "
            "list, history, archive, delete, search."
        ),
        capability="memory",
        parameters={
            "action": "str — create|list|history|archive|delete|search",
            "thread_id": "str — thread id (history/archive/delete)",
            "topic": "str — thread topic (create)",
            "title": "str — thread title (create)",
            "query": "str — search query (search)",
            "limit": "int — max results",
        },
    )
    def side_chat(context: Any, action: str, thread_id: str = "",
                  topic: str = "", title: str = "", query: str = "",
                  limit: int = 20) -> dict[str, Any]:
        import asyncio

        mgr = _get_manager()
        action = (action or "list").lower()

        async def _run() -> dict[str, Any]:
            if action == "create":
                thread = await mgr.create(topic=topic, title=title or topic)
                return {"action": "create", "thread": thread.to_dict()}
            if action == "list":
                threads = await mgr.list_threads(limit=limit)
                return {"action": "list", "threads": [t.to_dict() for t in threads]}
            if action == "history":
                msgs = await mgr.get_history(thread_id, limit=limit)
                return {"action": "history", "messages": [m.to_dict() for m in msgs]}
            if action == "archive":
                ok = await mgr.archive(thread_id)
                return {"action": "archive", "ok": ok}
            if action == "delete":
                ok = await mgr.delete(thread_id)
                return {"action": "delete", "ok": ok}
            if action == "search":
                results = await mgr.search(query, limit=limit)
                return {"action": "search", "results": [r.to_dict() for r in results]}
            return {"error": f"Unknown action: {action}"}

        return asyncio.get_event_loop().run_until_complete(_run())
