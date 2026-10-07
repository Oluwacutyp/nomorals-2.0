"""Play session: the ``capture -> decide -> execute`` loop with guardrails.

A session runs at most ``max_moves`` moves, stops when the decider has
nothing safe to do, when a move is blocked or declined, or when the driver
errors. Irreversible moves need confirmation; without a confirmation
callback the session fails closed and stops.

:func:`handle_autopilot_command` parses the chat form
``/autopilot <game> <url> [moves=N] [--dry-run] [--yes]`` and returns a
plain-text report — that is the chat entry point.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.errors import ToolError
from ...core.logging_setup import get_logger
from .decider import Decider, Move
from .driver import BrowserDriver, HttpDriver, Page
from .executor import Executor, SafetyBlocked
from .state import GameState, state_from_page

_log = get_logger(__name__)

__all__ = [
    "PlaySession",
    "SessionReport",
    "handle_autopilot_command",
    "run_session",
]


@dataclass
class SessionReport:
    game: str
    start_url: str
    moves: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    dry_run: bool = True
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "start_url": self.start_url,
            "moves": self.moves,
            "executed": sum(1 for m in self.moves if m.get("executed")),
            "blocked": sum(1 for m in self.moves if m.get("blocked_reason")),
            "stop_reason": self.stop_reason,
            "dry_run": self.dry_run,
            "seconds": round(self.ended_at - self.started_at, 1)
            if self.ended_at
            else 0.0,
        }

    def as_text(self) -> str:
        lines = [
            f"autopilot: {self.game} ({self.start_url})",
            f"mode: {'dry-run' if self.dry_run else 'LIVE'}",
            f"moves: {len(self.moves)}",
        ]
        for i, m in enumerate(self.moves, 1):
            action = m.get("action", {})
            status = "executed" if m.get("executed") else "dry-run"
            if m.get("blocked_reason"):
                status = f"BLOCKED: {m['blocked_reason']}"
            lines.append(
                f"  {i}. [{status}] {action.get('kind')} {action.get('label')!r}"
                f" — {m.get('reason', '')}"
            )
        lines.append(f"stopped: {self.stop_reason}")
        return "\n".join(lines)


class PlaySession:
    """One bounded autopilot run."""

    def __init__(
        self,
        driver: BrowserDriver,
        *,
        decider: Decider | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.driver = driver
        self.decider = decider
        self.executor = executor

    def run(
        self,
        game: str,
        url: str,
        *,
        max_moves: int = 20,
        dry_run: bool = True,
        confirm: Callable[[Move, GameState], bool] | None = None,
    ) -> SessionReport:
        """Run the loop. Never raises for game-level outcomes; raises only on
        programmer errors (bad args). Driver failures end the session."""
        if max_moves < 1:
            raise ToolError("max_moves must be >= 1")
        if not url.startswith(("http://", "https://")):
            raise ToolError(f"refusing non-http(s) game url: {url!r}")

        decider = self.decider or Decider(game)
        executor = self.executor or Executor(
            self.driver, dry_run=dry_run, confirm=confirm
        )
        report = SessionReport(game=game, start_url=url, dry_run=dry_run)
        _log.info(
            "autopilot start: game=%s url=%s max_moves=%d dry_run=%s",
            game,
            url,
            max_moves,
            dry_run,
        )

        current_url = url
        for n in range(max_moves):
            try:
                page: Page = self.driver.fetch(current_url)
            except ToolError as exc:
                report.stop_reason = f"fetch failed: {exc}"
                _log.warning("autopilot stop: %s", report.stop_reason)
                break
            state = state_from_page(page)

            move = decider.decide(state)
            if move is None:
                report.stop_reason = "no safe move found"
                _log.info("autopilot stop: no safe move found")
                break

            try:
                result = executor.execute(move, state, page)
            except SafetyBlocked as exc:
                report.moves.append(
                    {
                        "action": {
                            "kind": move.action.kind,
                            "target": move.action.target,
                            "label": move.action.label,
                        },
                        "reason": move.reason,
                        "executed": False,
                        "blocked_reason": str(exc),
                    }
                )
                report.stop_reason = f"safety: {exc}"
                _log.warning("autopilot stop on safety: %s", exc)
                break

            report.moves.append(
                {
                    "action": {
                        "kind": move.action.kind,
                        "target": move.action.target,
                        "label": move.action.label,
                    },
                    "reason": move.reason,
                    "executed": result.ok and not result.dry_run,
                    "blocked_reason": result.blocked_reason,
                    "dry_run": result.dry_run,
                    "url": result.page.url if result.page else None,
                }
            )
            if result.page is not None:
                current_url = result.page.url
        else:
            report.stop_reason = f"move limit reached ({max_moves})"

        report.ended_at = time.time()
        _log.info(
            "autopilot end: game=%s moves=%d reason=%s",
            game,
            len(report.moves),
            report.stop_reason,
        )
        return report


def run_session(
    game: str,
    url: str,
    *,
    max_moves: int = 20,
    dry_run: bool = True,
    confirm: Callable[[Move, GameState], bool] | None = None,
    driver: BrowserDriver | None = None,
) -> SessionReport:
    """One-shot session with a default HTTP driver."""
    session = PlaySession(driver or HttpDriver())
    return session.run(
        game, url, max_moves=max_moves, dry_run=dry_run, confirm=confirm
    )


def handle_autopilot_command(tail: str) -> str:
    """Chat entry point: ``/autopilot <game> <url> [moves=N] [--dry-run] [--yes]``.

    ``--yes`` auto-confirms irreversible moves (default: they are declined).
    Without ``--dry-run`` the session really acts; default is dry-run.
    """
    try:
        parts = shlex.split(tail or "")
    except ValueError as exc:
        return f"autopilot: could not parse arguments: {exc}"
    if len(parts) < 2:
        return (
            "usage: /autopilot <game> <url> [moves=N] [--dry-run] [--live] [--yes]\n"
            "example: /autopilot lagos_life https://example.com/game moves=10"
        )
    game, url = parts[0], parts[1]
    max_moves = 20
    dry_run = True
    auto_yes = False
    for part in parts[2:]:
        if part.startswith("moves="):
            try:
                max_moves = max(1, min(100, int(part.split("=", 1)[1])))
            except ValueError:
                return f"autopilot: bad moves value: {part!r}"
        elif part == "--dry-run":
            dry_run = True
        elif part == "--live":
            dry_run = False
        elif part == "--yes":
            auto_yes = True
        else:
            return f"autopilot: unknown argument: {part!r}"

    confirm: Callable[[Move, GameState], bool] | None = (
        (lambda _m, _s: True) if auto_yes else None
    )
    try:
        report = run_session(
            game, url, max_moves=max_moves, dry_run=dry_run, confirm=confirm
        )
    except ToolError as exc:
        return f"autopilot failed: {exc}"
    return report.as_text()
