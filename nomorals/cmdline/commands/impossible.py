"""``nm impossible`` — CLI front door over nomorals.impossible.

Delegates to :func:`nomorals.impossible.chat.control_impossible` — the same
control the chat /impossible command uses — so CLI and chat stay in sync.
"""

from __future__ import annotations

import sys
from typing import Any


def _cmd_impossible(args: Any, context: Any) -> int:
    """Route ``nm impossible ...``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm impossible solve <problem> [--software] [--fetch-art] | status <id> | attempt <id> | dossier <id> | queue | charter | lattice | examples | sources [domain] | cross <problem>",
              file=sys.stderr)
        return 2
    from ...impossible.chat import control_impossible
    print(control_impossible("/impossible " + " ".join(words), context=context, chat_key="cli"))
    return 0
