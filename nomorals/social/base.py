"""Platform adapters and the account model.

One rule governs this whole layer: **official APIs only**. Scraping a platform to
automate it violates its terms, breaks whenever markup changes, and is the fastest
way to get an account banned. Every adapter here talks to a documented endpoint.

The second rule: credentials are never stored. ``Account.credentials`` holds a
*reference* into the secret store or an environment variable name. A database that
can be backed up to a git repo must not contain bearer tokens, and this layer is
built on the assumption that it will be.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import NoMoralsError, ValidationError

__all__ = ["SocialError", "PostResult", "Account", "PlatformAdapter", "PostStatus"]


class SocialError(NoMoralsError):
    """A platform rejected us, or we refused to ask it."""


class PostStatus:
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    POSTING = "posting"
    POSTED = "posted"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = frozenset({POSTED, FAILED, CANCELLED})


@dataclass
class PostResult:
    """What one platform said about one post."""

    platform: str
    ok: bool
    external_id: str = ""
    url: str = ""
    error: str = ""
    status_code: int = 0
    seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    rate_limit_remaining: int | None = None
    rate_limit_reset: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform, "ok": self.ok, "external_id": self.external_id,
            "url": self.url, "error": self.error, "status_code": self.status_code,
            "seconds": round(self.seconds, 3), "metrics": self.metrics,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset,
        }


@dataclass
class Account:
    """A connected platform account. Holds a credential *reference*, never a secret."""

    platform: str
    handle: str
    id: str = ""
    active: bool = True
    credentials: str = ""
    limits: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.platform:
            raise ValidationError("an account needs a platform", field="platform")
        if not self.handle:
            raise ValidationError("an account needs a handle", field="handle")

    @property
    def posts_per_day(self) -> int:
        """Platform posting ceiling. Defaults conservatively when unset."""
        return int(self.limits.get("posts_per_day", 50))

    @property
    def min_interval(self) -> float:
        """Seconds required between posts. Stops a loop hammering one account."""
        return float(self.limits.get("min_interval_seconds", 30.0))

    @property
    def max_chars(self) -> int:
        return int(self.limits.get("max_chars", 500))

    def resolve_token(self) -> str:
        """Turn the credential reference into a token, without persisting it.

        Accepts ``env:NAME`` or a literal. A literal is supported because a
        single-user local install may reasonably keep the token in its own
        config file, which is not the database and not this repo.
        """
        if not self.credentials:
            return ""
        if self.credentials.startswith("env:"):
            return os.environ.get(self.credentials[4:], "")
        return self.credentials

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "platform": self.platform, "handle": self.handle,
            "active": int(self.active), "credentials": self.credentials,
            "limits": self.limits, "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Account":
        import json

        def _decode(raw: Any, default: Any) -> Any:
            if isinstance(raw, (dict, list)) or raw in (None, ""):
                return raw if isinstance(raw, (dict, list)) else default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        return cls(
            id=row["id"], platform=row["platform"], handle=row.get("handle") or "",
            active=bool(row.get("active", 1)), credentials=row.get("credentials") or "",
            limits=_decode(row.get("limits"), {}),
            created_at=float(row.get("created_at") or 0.0),
            metadata=_decode(row.get("metadata"), {}),
        )


class PlatformAdapter(ABC):
    """What a platform integration must provide.

    ``post`` returns a :class:`PostResult` rather than raising for ordinary
    rejections: a 429 or a 403 is an expected outcome that the scheduler must
    record and react to, not an exceptional one.
    """

    name: str = "unknown"
    max_chars: int = 500
    supports_media: bool = False
    supports_thread: bool = False

    @abstractmethod
    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        """Publish one post."""

    def delete(self, account: Account, external_id: str) -> PostResult:
        return PostResult(platform=self.name, ok=False, error="delete not supported")

    def metrics(self, account: Account, external_id: str) -> dict[str, Any]:
        """Engagement counts. Optional: not every platform exposes them."""
        return {}

    def validate(self, content: str) -> str:
        """Reject content this platform cannot accept, before spending a request."""
        if not content or not content.strip():
            raise ValidationError("post content is empty", field="content")
        if len(content) > self.max_chars:
            raise ValidationError(
                f"{self.name} allows {self.max_chars} characters, got {len(content)}",
                field="content",
            )
        return content.strip()

    def health(self, account: Account) -> bool:
        """Cheap credential check. Defaults to 'we have a token'."""
        return bool(account.resolve_token())
