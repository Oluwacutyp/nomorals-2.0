"""Errors (Craft-Libs port of ai-core ``error.rs``).

One error type for everything a provider or the model layer can
report. The ``kind`` lets an app choose behaviour (retry, ask the
user to sign in, show "this model cannot take a mask"); the
``message`` is for people and is safe to show. Build provider
errors with :meth:`AiError.redacted` so an echoed key never
reaches a log.
"""

from __future__ import annotations

from enum import Enum

from .secret import redact

__all__ = ["AiError", "ErrorKind"]


class ErrorKind(Enum):
    """What went wrong, in terms an app can act on."""

    UNSUPPORTED = "unsupported"
    INVALID_REQUEST = "invalid_request"
    AUTH = "auth"
    RATE_LIMITED = "rate_limited"
    QUOTA = "quota"
    NETWORK = "network"
    PROVIDER = "provider"
    CANCELLED = "cancelled"
    CONSENT_REQUIRED = "consent_required"
    INTERNAL = "internal"

    def as_str(self) -> str:
        return {
            ErrorKind.UNSUPPORTED: "unsupported",
            ErrorKind.INVALID_REQUEST: "invalid request",
            ErrorKind.AUTH: "authentication",
            ErrorKind.RATE_LIMITED: "rate limited",
            ErrorKind.QUOTA: "quota",
            ErrorKind.NETWORK: "network",
            ErrorKind.PROVIDER: "provider",
            ErrorKind.CANCELLED: "cancelled",
            ErrorKind.CONSENT_REQUIRED: "consent required",
            ErrorKind.INTERNAL: "internal",
        }[self]


class AiError(Exception):
    """The error type of the whole ai layer."""

    def __init__(self, kind: ErrorKind, message: str,
                 retry_after_secs: int | None = None) -> None:
        super().__init__(f"{kind.as_str()}: {message}")
        self.kind = kind
        self.message = str(message)
        self.retry_after_secs = retry_after_secs

    @classmethod
    def new(cls, kind: ErrorKind, message: str) -> AiError:
        return cls(kind, message)

    @classmethod
    def unsupported(cls, message: str) -> AiError:
        return cls(ErrorKind.UNSUPPORTED, message)

    @classmethod
    def invalid(cls, message: str) -> AiError:
        return cls(ErrorKind.INVALID_REQUEST, message)

    @classmethod
    def cancelled(cls) -> AiError:
        return cls(ErrorKind.CANCELLED, "cancelled")

    @classmethod
    def redacted(cls, kind: ErrorKind, message: str,
                  secrets) -> AiError:
        """Build an error whose message has every known secret
        scrubbed out of it."""
        return cls(kind, redact(message, secrets))

    def is_retryable(self) -> bool:
        return self.kind in (ErrorKind.RATE_LIMITED, ErrorKind.NETWORK)

    def to_dict(self) -> dict:
        data = {"kind": self.kind.value, "message": self.message}
        if self.retry_after_secs is not None:
            data["retry_after_secs"] = self.retry_after_secs
        return data

    @classmethod
    def from_dict(cls, data: dict) -> AiError:
        return cls(ErrorKind(data["kind"]), str(data["message"]),
                   data.get("retry_after_secs"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, AiError)
                and self.kind == other.kind
                and self.message == other.message
                and self.retry_after_secs == other.retry_after_secs)

    def __hash__(self):
        return hash((self.kind, self.message, self.retry_after_secs))
