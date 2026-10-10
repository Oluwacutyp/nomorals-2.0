"""Connect UX helpers shared by every connector.

Secrets arrive through exactly three doors, in this order:

1. an explicit argument (``connect(token=...)``),
2. an environment variable (non-interactive / automation),
3. a secure interactive prompt (TTY only).

There is no fourth door. In particular a non-TTY session without the env
var fails fast with a clear message instead of hanging on stdin or
silently continuing without credentials.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import os
import secrets
import sys
import time
from collections.abc import Callable
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .base import ConnectorError

__all__ = [
    "device_flow_token",
    "new_state",
    "pick_scopes",
    "pkce_pair",
    "prompt_secret",
]

_log = get_logger(__name__)


def pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code verifier + S256 challenge (RFC 7636).

    Returns ``(verifier, challenge)``. The verifier is 64 chars of
    high-entropy urlsafe text (within the 43–128 range); the challenge is
    ``BASE64URL(SHA256(verifier))`` with padding stripped. Recommended for
    *all* authorization-code flows (RFC 9700 baseline) — public clients
    must use it, confidential clients should.
    """
    verifier = secrets.token_urlsafe(48)[:64]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def new_state(nbytes: int = 32) -> str:
    """High-entropy ``state`` for an authorization request (CSRF binding).

    Validate it *before* accepting anything else in the callback — a
    mismatch gets rejected and the flow keeps waiting.
    """
    return secrets.token_urlsafe(nbytes)


def prompt_secret(prompt: str, *, env_var: str | None = None) -> str:
    """Read a secret from env, else a secure TTY prompt. Fail fast otherwise.

    The secret is never echoed, never logged, and never appears in error
    messages — only its presence/absence is ever reported.
    """
    if env_var:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    if not sys.stdin.isatty():
        need = f" (set the {env_var} environment variable)" if env_var else ""
        raise ConnectorError(
            f"cannot prompt for {prompt!r}: not an interactive terminal{need}"
        )
    # KeyboardInterrupt propagates: the CLI dispatch handles it as a
    # deliberate top-level shutdown (exit 130).
    try:
        return getpass.getpass(f"{prompt}: ")
    except EOFError as exc:
        raise ConnectorError(f"secret entry cancelled for {prompt!r}") from exc


def pick_scopes(
    available: list[str],
    defaults: list[str],
    *,
    prompt: str = "scopes",
    input_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """Let the owner choose scopes interactively; defaults when not a TTY.

    Shows a numbered list; the owner answers with numbers (``1,3``), ``all``,
    or empty for the defaults. ``input_fn`` exists so tests can drive the
    picker without a terminal.
    """
    read = input_fn or (input if sys.stdin.isatty() else None)
    if read is None:
        return list(defaults)
    print(f"Available {prompt} (default: {', '.join(defaults) or 'none'}):")
    for i, scope in enumerate(available, 1):
        mark = "*" if scope in defaults else " "
        print(f"  {mark} {i}. {scope}")
    try:
        answer = read(
            "Enter numbers, 'all', or empty for defaults: "
        ).strip().lower()
    except EOFError as exc:
        raise ConnectorError(f"{prompt} selection cancelled") from exc
    if not answer:
        return list(defaults)
    if answer == "all":
        return list(available)
    picked: list[str] = []
    for part in answer.replace(",", " ").split():
        if part.isdigit() and 1 <= int(part) <= len(available):
            scope = available[int(part) - 1]
            if scope not in picked:
                picked.append(scope)
    return picked or list(defaults)


def device_flow_token(
    *,
    client_id: str,
    device_code_url: str,
    token_url: str,
    scope: str = "",
    http: HttpClient | None = None,
    poll_timeout: float = 600.0,
    on_code: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """OAuth 2.0 device authorization grant (RFC 8628), fully implemented.

    Returns the token response (``access_token``, ...). The caller shows the
    user code + verification URI — via ``on_code`` or its own UX — while this
    polls. Raises :class:`ConnectorError` on denial, expiry, or timeout.
    """
    client = http or HttpClient()
    form: dict[str, Any] = {"client_id": client_id}
    if scope:
        form["scope"] = scope
    resp = client.post_form(device_code_url, form).raise_for_status().json()
    device_code = resp.get("device_code", "")
    user_code = resp.get("user_code", "")
    verification_uri = resp.get(
        "verification_uri_complete", resp.get("verification_uri", "")
    )
    expires_in = float(resp.get("expires_in", 900))
    interval = max(float(resp.get("interval", 5)), 1.0)
    if not device_code or not user_code:
        raise ConnectorError(
            "device flow failed: authorization server did not return "
            "a device_code/user_code pair"
        )
    info = {
        "user_code": user_code,
        "verification_uri": verification_uri,
        "expires_in": expires_in,
    }
    if on_code is not None:
        on_code(info)
    else:
        print(f"Open {verification_uri} and enter code: {user_code}")
    deadline = time.time() + min(poll_timeout, expires_in)
    wait = interval
    while time.time() < deadline:
        time.sleep(wait)
        token_resp = client.post_form(
            token_url,
            {
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        ).json()
        if token_resp.get("access_token"):
            return token_resp
        error = str(token_resp.get("error", ""))
        if error == "authorization_pending":
            wait = interval
            continue
        if error == "slow_down":
            wait = interval + 5.0
            continue
        if error in ("access_denied", "expired_token"):
            raise ConnectorError(
                f"device flow {error.replace('_', ' ')}: the owner did not "
                "approve the connection"
            )
        raise ConnectorError(
            f"device flow failed: {error or 'unexpected token response'}"
        )
    raise ConnectorError("device flow timed out waiting for approval")
