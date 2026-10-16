"""``nm learn`` — CLI front door over nomorals.learn (Tier 0 wiring).

Delegates to :func:`nomorals.learn.chat.control_learn` — the same
control the chat /learn command uses — so CLI and chat stay in sync.
"""

from __future__ import annotations

import sys
from typing import Any


def _cmd_learn(args: Any, context: Any) -> int:
    """Route ``nm learn <subcommand> ...``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm learn tutor [guide me] <topic> | answer <text> | hint | status | stop\n"
              "       nm learn quiz formats | quiz start <exam> <subject>\n"
              "       nm learn flashcards due [n] | stats | list | review <id> <1-4> | export <path>\n"
              "       nm learn mastery [weakest]\n"
              "       nm learn waec <subject> | waec grade <pct>\n"
              "       nm learn jamb <subject> | jamb formats",
              file=sys.stderr)
        return 2
    from ...learn.chat import control_learn
    print(control_learn(" ".join(words), context=context, chat_key="cli"))
    return 0
