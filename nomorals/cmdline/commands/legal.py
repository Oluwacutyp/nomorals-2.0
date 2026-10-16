"""``nm legal`` — CLI front door over nomorals.legal (Tier 0 wiring).

Delegates to :func:`nomorals.legal.chat.control_legal_full` — the same
control the chat /legal command uses — so CLI and chat stay in sync.
Every legal output carries the "not legal advice" disclaimer (each
legal module attaches it; the router adds it to its own outputs).
"""

from __future__ import annotations

import sys
from typing import Any


def _cmd_legal(args: Any, context: Any) -> int:
    """Route ``nm legal <subcommand> ...``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm legal contract review [tenancy|employment|freelance] <paste contract text>\n"
              "       nm legal portfolio — obligation tracker (renewals, notice deadlines, SLA)\n"
              "       nm legal regulatory regulators | add <REG> [topics] | list | check | calendar\n"
              "       nm legal aid [en|pcm|yo|ha|ig] <question> — plain-language aid\n"
              "       nm legal research <question> — grounded research with citations\n"
              "       nm legal citations — why information-only (precedent citations)\n"
              "       nm legal [en|pcm|yo|ha|ig] <question> — ask directly (aid)",
              file=sys.stderr)
        return 2
    from ...legal.chat import control_legal_full
    print(control_legal_full(" ".join(words), context=context))
    return 0
