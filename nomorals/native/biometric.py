"""Phone-native biometric approval (fingerprint) for sensitive actions.

Termux-only: uses the ``termux-fingerprint`` binary from Termux:API, which
shows the system fingerprint dialog on the device screen and exits 0 when
the user authenticates. Everywhere else the module reports unavailable and
callers fall back to the text-confirm flow — biometric is an *additional*
approval path, never the only one.

Nothing here ever prompts implicitly: :func:`request_biometric` is only
called from explicit approval helpers (``policy.approve_with_biometric``,
``ToolRegistry.request_approval``). Background threads must never hang on
a fingerprint dialog.
"""

from __future__ import annotations

import shutil
import subprocess

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["biometric_available", "request_biometric"]


def biometric_available() -> tuple[bool, str]:
    """``(True, "")`` when a fingerprint prompt can be shown right now.

    Requires the ``termux`` runtime profile AND the Termux:API
    ``termux-fingerprint`` binary. Never raises.
    """
    try:
        from ..core.profiles import get_profile_kind

        if get_profile_kind() != "termux":
            return False, "biometric approval needs the Termux profile"
    except Exception:  # noqa: BLE001 - fail closed when the profile is unknown
        return False, "could not determine runtime profile"
    if shutil.which("termux-fingerprint") is None:
        return False, "termux-fingerprint not found (install the Termux:API app)"
    return True, ""


def request_biometric(title: str, *, timeout_s: float = 60.0) -> bool:
    """Show the fingerprint dialog; True iff the user authenticated.

    ``title`` is caller context for logs only — ``termux-fingerprint``
    shows the system dialog as-is. Never raises: any failure (missing
    binary, timeout, non-zero exit) is a denial. Logs at debug only so
    auth prompts don't spam the log.
    """
    _log.debug("biometric prompt requested: %s", title)
    try:
        proc = subprocess.run(
            ["termux-fingerprint"],
            timeout=timeout_s,
            capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a denial
        _log.debug("biometric prompt failed: %s", exc)
        return False
    ok = proc.returncode == 0
    _log.debug("biometric prompt %s", "approved" if ok else "denied")
    return ok
