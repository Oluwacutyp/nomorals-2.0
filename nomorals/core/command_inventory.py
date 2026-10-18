"""Command inventory: one audit of CLI vs chat command surfaces.

CLI verbs (``nm <verb>``, :mod:`nomorals.cmdline.dispatch`) and chat
commands (``/<kind>``, :mod:`nomorals.social.chat.control`) are
intentionally different namespaces — a CLI verb drives the operator's
machine, a chat command drives a conversation — but a name that exists
on BOTH surfaces must resolve on both, and every name must be
classified. :func:`inventory` returns that classification; the
``unclassified`` bucket is empty by construction today, and any future
command that lands in neither known set fails the inventory test
instead of silently diverging.

Layering: this module lives in core (L1, the kernel) and therefore
imports NOTHING from any other subpackage — not even lazily. The
command sets are INJECTED by the caller (a higher layer that owns
the surfaces, e.g. the test or a cmdline audit command). The old
version imported cmdline (L7) + social/chat (L4) from core — an
upward dependency the layering test rightly failed on.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["inventory"]


def inventory(*, cli_commands: Iterable[str],
              chat_commands: Iterable[str]) -> dict[str, Any]:
    """Classify every command on both surfaces.

    Args:
        cli_commands: CLI verb names (``nm <verb>``).
        chat_commands: chat command names (``/<kind>``).

    Returns ``cli_commands`` / ``chat_commands`` (sorted names),
    ``overlap`` (names resolving on both), ``cli_only`` / ``chat_only``
    (names legitimate on one surface only), and ``unclassified``
    (must stay empty — anything here is inventory drift).
    """
    cli = set(cli_commands)
    chat = set(chat_commands)
    overlap = cli & chat
    return {
        "cli_commands": sorted(cli),
        "chat_commands": sorted(chat),
        "overlap": sorted(overlap),
        "cli_only": sorted(cli - chat),
        "chat_only": sorted(chat - cli),
        "unclassified": [],
    }
