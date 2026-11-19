"""Plugin contract for authenticated session forging.

Sync on purpose: the repo's HttpClient is sync and a plugin that needs
async brings asyncio into every caller for no gain. A plugin gets a
client, returns a LoginResult. It never raises for a credential outcome.

`enumerate` answers "does this account exist" only where the site leaks
it anonymously; UNKNOWN is a valid answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

__all__ = ["AuthResult", "LoginResult", "SessionData", "EnumerateResult",
           "WebPlugin"]


class AuthResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    MFA = "mfa"
    LOCKOUT = "lockout"
    CAPTCHA = "captcha"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class SessionData:
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    proxy_used: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"cookies": self.cookies, "headers": self.headers,
                "tokens": self.tokens, "user_agent": self.user_agent,
                "proxy_used": self.proxy_used}


@dataclass
class LoginResult:
    result: AuthResult
    username: str = ""
    password: str = ""
    message: str = ""
    response_time_ms: float = 0.0
    session: Optional[SessionData] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result is AuthResult.SUCCESS

    def as_dict(self, *, with_password: bool = True) -> dict[str, Any]:
        return {"result": self.result.value, "username": self.username,
                "password": self.password if with_password else "",
                "message": self.message,
                "response_time_ms": round(self.response_time_ms, 1),
                "session": self.session.as_dict() if self.session else None,
                "metadata": self.metadata}


@dataclass
class EnumerateResult:
    username: str
    exists: Optional[bool] = None
    confidence: float = 0.0
    indicators: list[str] = field(default_factory=list)


class WebPlugin:
    name: str = ""
    domain: str = ""
    #: cookie names that prove authentication (used by the verifier)
    auth_cookies: tuple[str, ...] = ()

    def login(self, username: str, password: str, client: Any) -> LoginResult:
        raise NotImplementedError

    def enumerate(self, username: str, client: Any) -> EnumerateResult:
        return EnumerateResult(username=username)
