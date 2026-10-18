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
"""

from __future__ import annotations

from typing import Any

__all__ = ["inventory"]


def _cli_commands() -> set[str]:
    from ..cmdline.dispatch import _COMMANDS

    return set(_COMMANDS)


def _chat_commands() -> set[str]:
    from ..social.chat.control import (
        CONTROL_COMMANDS,
        GAME_COMMANDS,
        PUBLIC_COMMANDS,
    )

    return (set(PUBLIC_COMMANDS) | set(GAME_COMMANDS)
            | set(CONTROL_COMMANDS))


def inventory() -> dict[str, Any]:
    """Classify every command on both surfaces.

    Returns ``cli_commands`` / ``chat_commands`` (sorted names),
    ``overlap`` (names resolving on both), ``cli_only`` / ``chat_only``
    (names legitimate on one surface only), and ``unclassified``
    (must stay empty — anything here is inventory drift).
    """
    cli = _cli_commands()
    chat = _chat_commands()
    overlap = cli & chat
    return {
        "cli_commands": sorted(cli),
        "chat_commands": sorted(chat),
        "overlap": sorted(overlap),
        "cli_only": sorted(cli - chat),
        "chat_only": sorted(chat - cli),
        "unclassified": [],
    }
