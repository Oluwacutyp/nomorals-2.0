"""Game autopilot tool: play browser games via the agent.

Exposes :func:`autopilot_play` to the tool registry so the brain can invoke
browser game playing from natural language ("play lagos life for me").
Defaults to dry-run: looking is free, touching needs an explicit flag.

Capability: ``net.browser`` — game pages are interactive web origins.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..games.player.session import run_session

__all__ = ["register"]

_log = get_logger(__name__)


def register(registry: Any) -> None:
    """Attach the game autopilot tool to a registry."""

    @registry.register(
        "autopilot_play",
        description=(
            "Play a browser game: capture state, decide moves, execute them. "
            "game is a strategy name (e.g. lagos_life, generic). "
            "Defaults to dry-run (no state-changing actions). "
            "Set live=true to really act, max_moves caps the session."
        ),
        capability=Capability.NET_BROWSER,
    )
    def autopilot_play(
        game: str,
        url: str,
        *,
        max_moves: int = 20,
        live: bool = False,
        auto_confirm: bool = False,
    ) -> dict[str, Any]:
        """Run an autopilot session and return the report.

        Safety: dry-run unless ``live=true``; irreversible moves are declined
        unless ``auto_confirm=true``; real-money and destructive moves are
        always blocked.
        """
        max_moves = max(1, min(100, int(max_moves)))
        _log.info(
            "autopilot_play called: game=%s url=%s max_moves=%d live=%s",
            game,
            url,
            max_moves,
            live,
        )
        report = run_session(
            game,
            url,
            max_moves=max_moves,
            dry_run=not live,
            confirm=(lambda _m, _s: True) if auto_confirm else None,
        )
        result = report.as_dict()
        result["summary"] = report.as_text()
        return result
