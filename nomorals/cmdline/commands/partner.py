"""``nm reason`` / ``workspace`` / partner ask surfaces."""

from __future__ import annotations

import sys
from ...llm.brain import brain_for
from ..emit import _emit

from ...core.logging_setup import get_logger

_log = get_logger(__name__)



def _cmd_reason(args, context):
    """Stub: reason command."""
    _emit(args, {"status": "ok"}, "Reasoning engine ready")
    return 0


def _cmd_workspace(args, context):
    """Stub: workspace command."""
    _emit(args, {"status": "ok"}, "Workspace ready")
    return 0


def _cmd_partner_ask(args, context) -> int:
    """One-shot partner question straight from the terminal.

    No gateway, no adapters, no platforms started: just her brain answering
    once. The exchange is journalled like any other DM so the memory and
    training pipelines see it too.
    """
    import time as _time

    from ...core.ids import ulid_now
    from ...llm.base import Message

    ask = (getattr(args, "ask", "") or "").strip()
    if not ask:
        print('usage: nm partner ask "<message>"', file=sys.stderr)
        return 2
    chat_key = getattr(args, "chat", "") or "local:console"
    try:
        from ...partner.persona import default_persona

        persona = default_persona()
        override = getattr(getattr(context.settings, "partner", None),
                           "persona_name", "") or ""
        name = override or persona.name
        pronouns = persona.pronouns
    except Exception:  # noqa: BLE001 — the CLI works without the persona pack
        name, pronouns = "partner", "she/her"

    reply, model = "", ""
    router = getattr(context, "router", None)
    if router is not None:
        response = brain_for(context).chat([
            Message.system(f"You are {name} ({pronouns}), answering your "
                           "person directly. Warm, brief, honest."),
            Message.user(ask),
        ], task_kind="chat")
        if not response.ok:
            print(f"partner: {response.error}", file=sys.stderr)
            return 1
        reply, model = response.text, response.model or ""

    db = context.db
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO conversations (id, title, agent, channel, "
                "created_at, updated_at) VALUES (?, ?, 'partner', 'local', "
                "0, ?) ON CONFLICT(id) DO UPDATE SET updated_at = "
                "excluded.updated_at",
                (chat_key, "you", _time.time()))
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, "
                "name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid_now(), chat_key, "user", ask, "you", "", _time.time()))
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, "
                "name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid_now(), chat_key, "assistant", reply, name, model,
                 _time.time()))
    except Exception as exc:  # noqa: BLE001 — journalling must not swallow the reply
        _log.warning("partner ask journal failed: %s", exc)

    _emit(args, {"reply": reply, "model": model, "chat": chat_key},
          f"{name} — {pronouns} [{model}]\n{reply}")
    return 0
