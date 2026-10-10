"""Phone-native biometric approval (fingerprint) for sensitive actions.

Termux-only: uses the ``termux-fingerprint`` binary from Termux:API, which
shows the system fingerprint dialog on the device screen and exits 0 when
the user authenticates. Everywhere else the module reports unavailable and
callers fall back to the text-confirm flow — biometric is an *additional*
approval path, never the only one.

Attempt policy mirrors Android's own rules (CDD 7.3.10: rate-limit attempts
for at least 30 seconds after 5 false trials): 5 consecutive prompt denials
inside a 10-minute window arms a 30-second cooldown during which
:func:`biometric_available` reports locked-out instead of letting callers
spam the dialog (the OS would lock the *sensor* anyway; we surface the
signal instead of hammering it).

Nothing here ever prompts implicitly: :func:`request_biometric` is only
called from explicit approval helpers (``policy.approve_with_biometric``,
``ToolRegistry.request_approval``).  A non-blocking in-flight guard means a
second concurrent call gets ``busy`` instead of stacking a second dialog —
background threads must never hang on, or pile up, fingerprint dialogs.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "BiometricResult",
    "BiometricStatus",
    "biometric_available",
    "request_biometric",
    "request_biometric_ex",
]

#: Consecutive prompt denials inside _DENIAL_WINDOW_S that arm the cooldown.
_MAX_DENIALS = 5
#: Denials older than this stop counting toward the lockout.
_DENIAL_WINDOW_S = 600.0
#: Cooldown after _MAX_DENIALS consecutive denials (Android CDD: >= 30s).
_LOCKOUT_S = 30.0

_state_lock = threading.Lock()
_prompt_lock = threading.Lock()  # in-flight dialog guard (non-blocking)
_denial_times: list[float] = []  # monotonic timestamps, consecutive denials
_lockout_until: float = 0.0  # monotonic timestamp


class BiometricStatus(Enum):
    """Outcome of a biometric prompt request."""

    APPROVED = "approved"
    DENIED = "denied"  # dialog shown, user failed/cancelled (or infra error)
    TIMEOUT = "timeout"  # no answer within timeout_s
    BUSY = "busy"  # another prompt is already showing; not stacked
    LOCKED_OUT = "locked_out"  # cooldown after repeated denials
    UNAVAILABLE = "unavailable"  # no Termux profile / no binary


@dataclass
class BiometricResult:
    """Rich outcome of :func:`request_biometric_ex`."""

    status: BiometricStatus
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.status is BiometricStatus.APPROVED

    def __bool__(self) -> bool:
        return self.approved


def _reset_attempt_ledger() -> None:
    """Clear denial history and any active cooldown (tests only)."""
    global _lockout_until
    with _state_lock:
        _denial_times.clear()
        _lockout_until = 0.0


def _lockout_remaining(now: float | None = None) -> float:
    now = time.monotonic() if now is None else now
    with _state_lock:
        return max(0.0, _lockout_until - now)


def _record_denial() -> None:
    """A shown prompt ended without approval.  Arms the cooldown at
    _MAX_DENIALS consecutive denials inside the window."""
    global _lockout_until
    now = time.monotonic()
    with _state_lock:
        cutoff = now - _DENIAL_WINDOW_S
        _denial_times[:] = [t for t in _denial_times if t >= cutoff]
        _denial_times.append(now)
        if len(_denial_times) >= _MAX_DENIALS:
            _lockout_until = now + _LOCKOUT_S
            _denial_times.clear()
            _log.info("biometric locked out for %.0fs after %d consecutive "
                      "denials", _LOCKOUT_S, _MAX_DENIALS)


def _record_approval() -> None:
    with _state_lock:
        _denial_times.clear()


def biometric_available() -> tuple[bool, str]:
    """``(True, "")`` when a fingerprint prompt can be shown right now.

    Requires the ``termux`` runtime profile AND the Termux:API
    ``termux-fingerprint`` binary AND no active cooldown. Never raises.
    """
    try:
        from ..core.profiles import get_profile_kind

        if get_profile_kind() != "termux":
            return False, "biometric approval needs the Termux profile"
    except Exception:  # noqa: BLE001 - fail closed when the profile is unknown
        return False, "could not determine runtime profile"
    if shutil.which("termux-fingerprint") is None:
        return False, "termux-fingerprint not found (install the Termux:API app)"
    remaining = _lockout_remaining()
    if remaining > 0:
        return False, (f"biometric locked out for {remaining:.0f}s after "
                       f"{_MAX_DENIALS} consecutive denials")
    return True, ""


def request_biometric_ex(title: str, *,
                         timeout_s: float = 60.0) -> BiometricResult:
    """Show the fingerprint dialog and return the rich outcome.

    Never raises: any failure is a non-approved :class:`BiometricResult`.
    At most one dialog is ever in flight — a concurrent call returns
    ``busy`` immediately.  Logs at debug only so auth prompts don't spam
    the log (the lockout transition logs at info; it is security-relevant).
    """
    _log.debug("biometric prompt requested: %s", title)
    if _lockout_remaining() > 0:
        remaining = _lockout_remaining()
        return BiometricResult(
            BiometricStatus.LOCKED_OUT,
            f"cooldown active for {remaining:.0f}s after repeated denials")
    if not _prompt_lock.acquire(blocking=False):
        return BiometricResult(BiometricStatus.BUSY,
                               "another biometric prompt is already showing")
    try:
        # NOTE: no biometric_available() probe here on purpose — callers
        # (policy.approve_with_biometric) probe first, and going straight to
        # the exec keeps the failure modes honest: a missing binary raises
        # FileNotFoundError and surfaces as UNAVAILABLE below.
        try:
            proc = subprocess.run(
                ["termux-fingerprint"],
                timeout=timeout_s,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            _log.debug("biometric prompt timed out: %s", title)
            _record_denial()
            return BiometricResult(BiometricStatus.TIMEOUT,
                                   f"no answer within {timeout_s:g}s")
        except FileNotFoundError:
            # Binary vanished between the availability probe and the exec —
            # not a user denial, so it does not count toward lockout.
            _log.debug("biometric binary missing at prompt time: %s", title)
            return BiometricResult(BiometricStatus.UNAVAILABLE,
                                   "termux-fingerprint not found")
        except Exception as exc:  # noqa: BLE001 - any failure is a denial
            _log.debug("biometric prompt failed: %s", exc)
            _record_denial()
            return BiometricResult(BiometricStatus.DENIED,
                                   f"prompt error: {exc}")
        if proc.returncode == 0:
            _log.debug("biometric prompt approved")
            _record_approval()
            return BiometricResult(BiometricStatus.APPROVED)
        _log.debug("biometric prompt denied (exit %d)", proc.returncode)
        _record_denial()
        return BiometricResult(BiometricStatus.DENIED,
                               f"termux-fingerprint exited {proc.returncode}")
    finally:
        _prompt_lock.release()


def request_biometric(title: str, *, timeout_s: float = 60.0) -> bool:
    """Show the fingerprint dialog; True iff the user authenticated.

    ``title`` is caller context for logs only — ``termux-fingerprint``
    shows the system dialog as-is. Never raises: any failure (missing
    binary, timeout, non-zero exit, lockout, busy) is a denial. Logs at
    debug only so auth prompts don't spam the log.
    """
    return request_biometric_ex(title, timeout_s=timeout_s).approved
