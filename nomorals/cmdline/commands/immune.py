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
              "       nm immune casefile list | add <text> | seal <id-prefix>",
              file=sys.stderr)
        return 2
    from ...immune.chat import control_immune
    print(control_immune(" ".join(words), context=context, chat_key="cli"))
    return 0
