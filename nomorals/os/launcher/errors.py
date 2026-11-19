"""Typed failures for the Craft-Launcher self-update path.

Every refusal names its cause: a verification failure names the file
or check, a trust refusal names the rule, an install failure names
the step. Nothing here is a silent skip — a caller that catches one
of these knows exactly what was refused and why.
"""

from __future__ import annotations


class LauncherError(Exception):
    """Base failure for the launcher organ."""


class VerificationError(LauncherError):
    """The package failed verification and was not installed."""


class TrustError(LauncherError):
    """The package's source or hash failed a catalog trust rule."""


class InstallError(LauncherError):
    """The install itself failed (staging, swap, rollback, health)."""
