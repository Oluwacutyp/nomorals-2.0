"""One TUI facade over the two console surfaces.

Skip-list consolidation: Devon has TWO terminal UIs — the fullscreen
:class:`~nomorals.console.godconsole.GodConsole` dashboard and the
scrollback-native ``nm`` command line (:mod:`nomorals.cmdline`).
Both stay; this facade unifies them behind ONE API so callers stop
picking a surface:

* ``list_commands()`` — the union of both surfaces' commands.
* ``run_command(cmd)`` — route one command string through the same
  dispatch both surfaces use (cmdline dispatch first; GodConsole
  ``on_command`` for console-native commands).
* ``render_frame()`` / ``push_event()`` — GodConsole passthrough.
* ``failures()`` — the console failure ledger, newest first.
* ``cmdline_commands()`` / ``create_godconsole()`` — direct access to
  each surface's real API, unchanged.

Facade rule: the facade PRESERVES APIs — it never re-implements a
surface. Handlers are injected (default: the real modules, lazily
imported), so tests can inject fakes and assert passthrough equality:
``facade.run_command(x) == direct_handler(x)``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence
from .failures import record_failure

__all__ = ["TUIFacade", "cmdline_commands", "create_godconsole"]


def cmdline_commands(commands: Any | None = None) -> list[str]:
    """``nm`` command names from an injected command table.

    Layering: this facade must not import L7 cmdline dispatch.
    Pass ``commands`` (mapping or iterable) from the caller that owns it.
    """
    if commands is None:
        return []
    if isinstance(commands, dict):
        return sorted(commands)
    return sorted(commands)


def create_godconsole(**kwargs: Any) -> Any:
    """Construct a real :class:`GodConsole` with the caller's kwargs."""
    from ..console.godconsole import GodConsole
    return GodConsole(**kwargs)


class TUIFacade:
    """Unified facade: one command surface over godconsole + cmdline."""

    def __init__(self, *,
                 godconsole: Any | None = None,
                 cmdline_dispatch: Callable[[str], Any] | None = None,
                 console_commands: Sequence[str] | None = None) -> None:
        self._godconsole = godconsole
        self._cmdline_dispatch = cmdline_dispatch
        self._console_commands = list(console_commands or
                                      ("help", "status", "events", "clear", "view"))

    # ── GodConsole passthrough ─────────────────────────────────────────────
    @property
    def godconsole(self) -> Any:
        if self._godconsole is None:
            self._godconsole = create_godconsole()
        return self._godconsole

    def push_event(self, event: str) -> None:
        self.godconsole.push_event(event)

    def failures(self, limit: int = 20) -> str:
        """Console failure ledger, newest first — the read-back for
        every best-effort fallback the console recorded instead of
        raising (see :mod:`nomorals.console.failures`). Empty when
        nothing has failed; the facade never invents failures."""
        from .failures import format_failures

        return format_failures(limit)

    def render_frame(self) -> str:
        return self.godconsole._render_frame()

    # ── unified commands ───────────────────────────────────────────────────
    def list_commands(self) -> list[str]:
        """Union of cmdline commands + console-native commands, sorted."""
        try:
            cli = set(cmdline_commands())
        except Exception as e:
            record_failure('tui_facade.TUIFacade.list_commands', e)
            cli = set()
        return sorted(cli | set(self._console_commands))

    def run_command(self, cmd: str) -> Any:
        """Route ``cmd``: cmdline dispatch when it knows the command,
        else the GodConsole ``on_command`` handler. Same result as
        calling the underlying handler directly (passthrough)."""
        name = (cmd or "").strip().split()[0] if (cmd or "").strip() else ""
        if self._cmdline_dispatch is not None:
            try:
                known = set(cmdline_commands())
            except Exception as e:
                record_failure('tui_facade.TUIFacade.run_command', e)
                known = set()
            if name in known:
                return self._cmdline_dispatch(cmd)
        handler = getattr(self.godconsole, "_on_command", None)
        if handler is not None:
            return handler(cmd)
        return f"echo: {cmd}"

    def run(self) -> str:
        """Run the fullscreen console (GodConsole passthrough)."""
        return self.godconsole.run()
