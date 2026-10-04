"""Graceful shutdown helpers: make SIGTERM behave like Ctrl-C.

Long-running entry points (``nm serve``, the partner bot runtime) always
caught KeyboardInterrupt but died instantly on SIGTERM: no ``stop()`` ran,
no final status beacon was written, and the CLI's ``with
build_context(...)`` teardown was skipped. Mapping SIGTERM onto
KeyboardInterrupt reuses every existing shutdown path instead of inventing
a second one.

Stdlib only, no package imports at module level beyond logging — this is
L1 core and must stay importable from anywhere.
"""

from __future__ import annotations

import signal

from .logging_setup import get_logger

__all__ = ["install_sigterm_as_interrupt"]

_log = get_logger(__name__)


def install_sigterm_as_interrupt() -> bool:
    """Map SIGTERM to KeyboardInterrupt in the main thread.

    Returns True when the handler was installed. Safe to call from any
    thread — signal handlers can only be installed on the main thread, so
    elsewhere this is a no-op returning False — and on platforms without
    SIGTERM. Idempotent: calling twice keeps a single handler.

    Every long-running entry point already shuts down cleanly on
    KeyboardInterrupt (the CLI's top-level ``except KeyboardInterrupt``,
    ``APIServer.serve()``'s finally block, ``PartnerRuntime.run()``'s
    finally block), so this one mapping gives SIGTERM the same graceful
    treatment: context teardown runs, the event bus drains, the gateway
    stops, and the final status beacon is written with ``stopped: true``.
    """
    try:
        sigterm = signal.SIGTERM
    except AttributeError:  # pragma: no cover - Windows has no SIGTERM
        return False
    previous = signal.getsignal(sigterm)

    def _sigterm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        raise KeyboardInterrupt()

    # Mark our own closure so a repeat install stays silent.
    _sigterm_handler._devon_sigterm_handler = True  # type: ignore[attr-defined]
    if getattr(previous, "_devon_sigterm_handler", False):
        return True
    try:
        signal.signal(sigterm, _sigterm_handler)
    except (OSError, ValueError, RuntimeError) as exc:
        # ValueError: not the main thread. OSError/RuntimeError: the
        # interpreter refuses (e.g. embedded). Either way, degrade to the
        # default SIGTERM behavior rather than breaking the boot.
        _log.debug("SIGTERM handler not installed: %s", exc)
        return False
    if previous not in (signal.SIG_DFL, None):
        _log.debug("replaced existing SIGTERM handler %r", previous)
    return True
