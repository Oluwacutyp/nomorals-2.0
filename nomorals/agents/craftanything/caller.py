"""Who invoked a tool, and the cooperative cancellation + progress shapes.

Port of upstream ``Caller`` / ``CancelToken`` / ``Progress``
(``crates/engine/src/tool.rs``): who invoked a tool matters — programmatic
callers never get dialogs, so consent is a setting for them, not an
interactive prompt — and long-running tools report progress and honour
cancellation. This module is standalone (stdlib only) so both the
consent gate and the engine can use it without an import cycle.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Caller", "CancelToken", "Progress", "ProgressFn"]


class Caller(StrEnum):
    """Who invoked a tool (upstream, verbatim).

    Programmatic callers never get dialogs: for them consent is a
    setting (a grant, a standing rule, ``--allow-run``), and a
    consent-gated call returns ``consent_required`` for the client's
    own approval flow instead of blocking.
    """

    UI = "ui"
    CLI = "cli"
    MCP = "mcp"
    AGENT = "agent"


class CancelToken:
    """Cooperative cancellation, shared with long-running work.

    The engine checks the token before dispatch and hands it to the
    call; a tool that runs long checks it and stops. Cancelling never
    kills a thread — it asks, and the engine reports ``cancelled``
    instead of a result.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


@dataclass(frozen=True)
class Progress:
    """A progress report from a running tool (upstream).

    ``fraction`` is 0..=1 when known, ``None`` when the work cannot say
    how far along it is; ``message`` says what it is doing.
    """

    fraction: float | None
    message: str


#: A progress callback: called with start (0.0), finish (1.0), and any
#: reports the tool itself makes in between.
ProgressFn = Callable[[Progress], None]
