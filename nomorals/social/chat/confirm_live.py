"""Visible expiry for staged confirmations (design absorption #4,
2026-11-09 — the Fuse Button's burning fuse, in chat grammar).

The confirm gate (``PartnerRuntime._confirm_gate``) stages a
destructive action behind a Confirm/Cancel prompt that lives for
``CONFIRM_TTL_S``. Until now the window existed only in the
store: the prompt said nothing about how long it lived, and when
it died the message just sat there with live buttons.

This module is the countdown's text grammar — pure functions over
the pending's REAL ``expires_at`` (the same value the gate
enforces; there is no second timer anywhere in this design):

* :func:`prompt_text` — the staged prompt, countdown line
  included, rendered from ``expires_at`` at stage time.
* :func:`with_countdown` — re-render the countdown line inside an
  already-sent prompt as the window burns (the runtime's
  countdown ticker edits the live message with this).
* :func:`expired_text` / :func:`settled_text` — the terminal
  renders: an expired prompt says the action did NOT run; a
  resolved one says how it resolved. Both are rendered WITHOUT
  buttons by the caller — a dead prompt keeps no live keyboard.

The runtime half (registration of the sent prompt, the ticker
thread, the edits) lives in ``agents/partner/runtime.py`` next to
the gate; everything here is deterministic text math so the
grammar is pinnable without a runtime.
"""

from __future__ import annotations

import math
import re

__all__ = [
    "COUNTDOWN_GLYPH",
    "EXPIRED_GLYPH",
    "DEFAULT_TICK_S",
    "format_remaining",
    "prompt_text",
    "with_countdown",
    "expired_text",
    "settled_text",
    "is_confirm_prompt",
]

#: The countdown line's glyph (the fuse). A prompt line starting
#: with it IS the countdown line — with_countdown replaces it.
COUNTDOWN_GLYPH = "⏳"

#: The expired render's glyph (the fuse burned out).
EXPIRED_GLYPH = "⌛"

#: Countdown ticker cadence: the displayed remaining time is the
#: true remaining time at the moment of the edit; a 5s cadence
#: bounds a 300s window to ~60 edits of one message.
DEFAULT_TICK_S = 5.0

_PROMPT_RE = re.compile(r"⚠️ confirm: /([a-z0-9_]+)")


def format_remaining(seconds: float) -> str:
    """Remaining-window readout: ``m:ss`` (``5:00``, ``0:47``)
    down to ten seconds, ``{n}s`` below that. Ceiling-rounded to
    whole seconds, so the card may round the true remainder up
    by less than a second — it is the rounded face of the one
    real ``expires_at``, never a separate allowance."""
    n = max(0, math.ceil(float(seconds)))
    if n < 10:
        return f"{n}s"
    return f"{n // 60}:{n % 60:02d}"


def _countdown_line(kind: str, expires_at: float, now: float) -> str:
    remaining = format_remaining(expires_at - now)
    return f"{COUNTDOWN_GLYPH} {remaining} left to confirm"


def prompt_text(kind: str, what: str, expires_at: float,
                now: float) -> str:
    """The staged prompt: the confirm line (the shape
    ``buttons_for_text`` derives the Confirm/Cancel keyboard
    from) plus the countdown line, both from the one real
    ``expires_at`` the gate stored."""
    return (f"⚠️ confirm: /{kind} — this will {what}.\n"
            + _countdown_line(kind, expires_at, now))


def is_confirm_prompt(text: str, kind: str = "") -> str:
    """The kind a confirm prompt text belongs to, or "" when the
    text is not a confirm prompt (or names another kind). The
    same marker ``buttons_for_text`` keys on."""
    m = _PROMPT_RE.search(str(text or ""))
    if not m:
        return ""
    found = m.group(1)
    if kind and found != kind:
        return ""
    return found


def with_countdown(text: str, kind: str, expires_at: float,
                   now: float) -> str:
    """``text`` with its countdown line re-rendered for ``now``.

    The countdown line (the one starting with the fuse glyph) is
    replaced in place — the rest of the sent text (including the
    presentation stamp the send channel appended) is untouched.
    A text without a countdown line gains one at the end, so an
    older prompt still joins the grammar."""
    line = _countdown_line(kind, expires_at, now)
    lines = str(text or "").split("\n")
    for i, existing in enumerate(lines):
        if existing.startswith(COUNTDOWN_GLYPH):
            lines[i] = line
            return "\n".join(lines)
    lines.append(line)
    return "\n".join(lines)


def expired_text(kind: str) -> str:
    """The terminal render when the window closes unused: the
    fuse is out, the action did NOT run, and the way back is
    named. (The gate's own late-confirm reply says the same in
    its words; this is the prompt message's own settled face.)"""
    return (f"{EXPIRED_GLYPH} that confirmation expired — "
            f"/{kind} did not run. Send /{kind} again "
            "if you still want it.")


def settled_text(kind: str, resolved: str) -> str:
    """The terminal render when the pending resolved by an
    explicit act: ``confirmed`` (the action went ahead) or
    ``cancelled`` (it did not run). Mirrors the gate's own reply
    wording for the same outcomes."""
    if resolved == "confirmed":
        return f"✓ confirmed — /{kind} went ahead."
    return f"✗ cancelled — /{kind} did not run."
