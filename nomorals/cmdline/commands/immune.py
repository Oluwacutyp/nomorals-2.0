"""``nm immune`` — CLI front door over nomorals.immune.

Delegates to :func:`nomorals.immune.chat.control_immune` — the same
control the chat /immune command uses — so CLI and chat stay in sync.
Quarantine / disarm / arrest are owner-only in chat; on the console
the operator is the owner.
"""

from __future__ import annotations

import sys
from typing import Any


def _cmd_immune(args: Any, context: Any) -> int:
    """Route ``nm immune <subcommand> ...``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm immune status | scan <text> | antibodies list | add <pattern> <reason> | retire <id> <reason> | test <pattern>\n"
              "       nm immune quarantine <artifact> | disarm <grant-id> | arrest <process-id> [system]\n"
              "       nm immune fever status | start <reason> <seconds> | end\n"
              "       nm immune drift add <tokens> <spawned> <grants> <errors> | report\n"
              "       nm immune casefile list | add <text> | seal <id-prefix>\n"
              "       nm immune corpus | corpus harvest\n"
              "       nm immune feeds | mark-fp <antibody-id> [reason]\n"
              "       nm immune gate [spawned growth]",
              file=sys.stderr)
        return 2
    from ...immune.chat import control_immune

    # Same organ-bus event sink the chat surface injects (AA-42):
    # immune events from CLI scans persist alongside chat ones.
    sink = None
    db = getattr(context, "db", None)
    if db is not None:
        from ... import organs

        def sink(kind: str, payload: dict) -> None:
            organs.emit(db, "immune", "brain", kind, payload)

    # Same police teeth the chat surface injects (doc 20 P7):
    # revoke-by-id from the grant registry, the fenced
    # own-process kill. Fail-open — an unwired console keeps its
    # honest receipts rather than breaking.
    revoke_fn = kill_fn = None
    try:
        from ...core.grant_registry import police_revoke_fn
        from ...core.own_processes import police_kill_fn

        revoke_fn, kill_fn = police_revoke_fn(), police_kill_fn()
    except Exception:  # noqa: BLE001 - unwired beats a broken console
        revoke_fn = kill_fn = None

    print(control_immune(" ".join(words), context=context, chat_key="cli",
                         event_sink=sink,
                         revoke_fn=revoke_fn, kill_fn=kill_fn))
    return 0
