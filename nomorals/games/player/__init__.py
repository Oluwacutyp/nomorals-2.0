"""Browser game autopilot: read game state, decide moves, execute them.

The loop is ``capture -> decide -> execute``:

* :mod:`driver` — how we talk to the game page (HTTP driver included;
  anything implementing :class:`BrowserDriver` works).
* :mod:`state` — turn a page into a structured :class:`GameState`.
* :mod:`decider` — pick the next :class:`Move` from a :class:`GameState`.
  Strategies are pluggable per game via :func:`register_strategy`.
* :mod:`executor` — run a move through the driver behind a safety gate.
  Real money is never spent, nothing is ever deleted, irreversible moves
  need confirmation. Every move is logged.
* :mod:`session` — the play loop with move limits and stop conditions.

Safety posture: dry-run is the default everywhere a move could change
server-side state. Nothing here is a stub — the HTTP driver really fetches
and really submits, the strategies really decide, the gate really blocks.
"""

from __future__ import annotations

from .decider import (
    Decider,
    Move,
    Strategy,
    decide,
    heuristic_strategy,
    lagos_life_strategy,
    register_strategy,
    strategies,
)
from .driver import (
    Action,
    BrowserDriver,
    HttpDriver,
    InteractiveElement,
    Page,
    extract_elements,
)
from .executor import (
    ExecutionResult,
    Executor,
    SafetyBlocked,
    is_forbidden,
)
from .session import (
    PlaySession,
    SessionReport,
    handle_autopilot_command,
    run_session,
)
from .state import (
    GameState,
    capture_state,
    state_from_page,
)

__all__ = [
    "Action",
    "BrowserDriver",
    "Decider",
    "ExecutionResult",
    "Executor",
    "GameState",
    "HttpDriver",
    "InteractiveElement",
    "Move",
    "Page",
    "PlaySession",
    "SafetyBlocked",
    "SessionReport",
    "Strategy",
    "capture_state",
    "decide",
    "extract_elements",
    "handle_autopilot_command",
    "heuristic_strategy",
    "is_forbidden",
    "lagos_life_strategy",
    "register_strategy",
    "run_session",
    "state_from_page",
    "strategies",
]
