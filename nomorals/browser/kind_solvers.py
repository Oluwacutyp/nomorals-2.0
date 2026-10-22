"""Per-kind local solver registry — the captcha estate's spine.

Every captcha kind gets solved by *something local first* when a
local solver exists for it: the routing layers (the browser
:class:`~nomorals.browser.challenge.ChallengeSolver` ladder and the
:mod:`nomorals.tools.captcha_flow` pipeline) never hardcode which
kinds are locally solvable — they consult this registry.

A kind solver is a callable::

    fn(tab, challenge: dict) -> {"ok": bool, "detail": str, ...}

``tab`` is the live browser tab (duck-typed; may be ``None`` for
offline flows, in which case the solver must decline honestly).
``challenge`` carries at least ``kind`` and ``page_url``, plus
whatever the routing layer knows (metadata, sitekey, image bytes).

Registration is how future backends plug in without touching the
spine: an image-captcha OCR solver registers for ``image_captcha``,
a local-ASR solver for ``audio_captcha``, a Cloudflare cookie-flow
solver for ``cloudflare`` — one ``register_kind_solver`` call each,
and both routing layers pick the new kind up automatically. The
slider solver (:mod:`nomorals.browser.slider_solver`) is the first
registrant: kinds ``slider`` and ``geetest``.

This module is deliberately a leaf (stdlib only): routing layers
import it without dragging in any solver's dependencies. Solvers
register themselves when their module is imported; routing layers
import the solver modules lazily at dispatch time, which is what
makes registration happen.
"""
from __future__ import annotations

from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "KindSolverFn",
    "register_kind_solver",
    "kind_solver_for",
    "registered_kinds",
]

#: (tab, challenge-dict) -> result-dict with at least {"ok", "detail"}
KindSolverFn = Callable[[Any, dict], dict]

_KIND_SOLVERS: dict[str, KindSolverFn] = {}


def register_kind_solver(kind: str, fn: KindSolverFn) -> bool:
    """Register ``fn`` as the local solver for ``kind``.

    Idempotent for the same callable; a different callable replaces
    the previous one (last registration wins, and says so in the log).
    Returns False on a malformed registration instead of raising —
    registration is additive and must never sink an import.
    """
    try:
        key = str(kind or "").strip().lower()
        if not key or not callable(fn):
            return False
        prev = _KIND_SOLVERS.get(key)
        if prev is not None and prev is not fn:
            _log.info("kind solver for %r replaced", key)
        _KIND_SOLVERS[key] = fn
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("kind solver registration failed: %s", exc)
        return False


def kind_solver_for(kind: str) -> KindSolverFn | None:
    """The registered local solver for ``kind``, or None."""
    return _KIND_SOLVERS.get(str(kind or "").strip().lower())


def registered_kinds() -> list[str]:
    """Kinds with a local solver registered in this process."""
    return sorted(_KIND_SOLVERS)
