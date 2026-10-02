"""Proposal approval helpers shared by the runtime and the CLI path."""

from __future__ import annotations

import time
from typing import Any, Callable

def _key_set(raw: str) -> set[str]:
    from ...social.chat.gateway import parse_chat_keys

    return parse_chat_keys(raw or "")


def _direct_approve(context: Any, gateway: Any, proposal_id: str) -> dict[str, Any]:
    """Approve a held proposal without a running autonomy agent (CLI path)."""
    row = context.db.query_one("SELECT * FROM proactive_log WHERE id = ?", (proposal_id,))
    if row is None:
        return {"ok": False, "error": f"no proposal {proposal_id!r}"}
    if row["status"] != "pending":
        return {"ok": False, "error": f"proposal is {row['status']}, not pending"}
    from ...social.chat.base import ChatRef

    chat = ChatRef(
        platform=row["platform"], chat_id=row["chat_id"],
        kind="group" if row["kind"] == "group" else "dm",
    )
    result = gateway.send(chat.platform, chat, row["content"])
    status = "sent" if result.ok else "failed"
    try:
        context.db.execute(
            "UPDATE proactive_log SET status = ?, acted_at = ? WHERE id = ?",
            (status, time.time(), proposal_id),
        )
    except Exception:  # noqa: BLE001
        pass
    return {"ok": result.ok, "status": status, "error": result.error}


def _direct_deny(context: Any, proposal_id: str) -> dict[str, Any]:
    row = context.db.query_one("SELECT status FROM proactive_log WHERE id = ?", (proposal_id,))
    if row is None:
        return {"ok": False, "error": f"no proposal {proposal_id!r}"}
    if row["status"] != "pending":
        return {"ok": False, "error": f"proposal is {row['status']}, not pending"}
    context.db.execute(
        "UPDATE proactive_log SET status = 'denied', acted_at = ? WHERE id = ?",
        (time.time(), proposal_id),
    )
    return {"ok": True, "status": "denied"}
