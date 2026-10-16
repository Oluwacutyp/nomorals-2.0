"""Typed failures for :mod:`nomorals.backup`.

Nothing in the backup pipeline fails silently: every failure mode the
coordinator knows about has a distinct type so callers (and the owner,
via :mod:`nomorals.backup.report`) can tell *which* step broke instead of
getting a bare ``Exception`` or, worse, a skipped run that looked like
success.
"""

from __future__ import annotations

from nomorals.core.errors import NoMoralsError

__all__ = [
    "BackupError",
    "BackupKeyError",
    "TokenScopeError",
    "TokenExpiredError",
    "RateLimitedBackupError",
    "ChecksumMismatchError",
    "CorruptBackupError",
    "DiskFullError",
    "KEY_LOSS_WARNING",
]


#: Said out loud at every key touchpoint: the backup key is the backup.
#: There is no reset flow, no "forgot key" — the ciphertext is opaque
#: without it. Keep an offline copy (paper / password manager) separate
#: from every machine that holds the backup itself.
KEY_LOSS_WARNING = (
    "BACKUP KEY WARNING: lose this key and the backup is unrecoverable — "
    "there is no reset, no recovery flow. Keep an offline copy (paper or "
    "password manager) separate from every machine holding the backup. "
    "The key lives in the vault, never in the repo, never in .env."
)


class BackupError(NoMoralsError):
    """Base class for backup-coordinator failures."""


class BackupKeyError(BackupError):
    """No backup encryption key available (or the stored key is unusable)."""

    def __init__(self, message: str = "") -> None:
        super().__init__(
            (message + " " if message else "") + KEY_LOSS_WARNING
        )


class TokenScopeError(BackupError):
    """The GitHub token lacks the scope/permission the call needs.

    ``required`` names the exact scope or permission so the owner can fix
    the token instead of guessing.
    """

    def __init__(self, message: str, *, required: str = "") -> None:
        super().__init__(message)
        self.required = required


class TokenExpiredError(BackupError):
    """The GitHub token was rejected (401): invalid, expired, or revoked."""


class RateLimitedBackupError(BackupError):
    """GitHub rate limit hit. ``retry_after`` is seconds to wait."""

    def __init__(self, message: str, *, retry_after: float = 60.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ChecksumMismatchError(BackupError):
    """Bytes on the far side do not match the manifest — failed, not partial."""


class CorruptBackupError(BackupError):
    """A snapshot failed integrity_check / extraction — never shipped."""


class DiskFullError(BackupError):
    """Local disk full mid-snapshot. The previous generation stays intact."""
