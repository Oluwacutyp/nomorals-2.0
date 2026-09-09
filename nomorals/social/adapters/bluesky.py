"""Bluesky adapter — the AT Protocol HTTP API.

Bluesky needs a session JWT rather than a long-lived token: you authenticate with
a handle and an app password, and the access token expires. So the adapter logs in
on demand and caches the session per handle until it stops working.

The app password is the credential, never the account password. That distinction is
the whole reason Bluesky has app passwords.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ...core.http import HttpClient
from ...core.logging_setup import get_logger
from ..base import Account, PlatformAdapter, PostResult

__all__ = ["Adapter"]

_log = get_logger(__name__)

DEFAULT_PDS = "https://bsky.social"


class Adapter(PlatformAdapter):
    name = "bluesky"
    max_chars = 300
    supports_media = True
    supports_thread = False

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._sessions: dict[str, dict[str, Any]] = {}

    def _pds(self, account: Account) -> str:
        return str(account.limits.get("pds") or DEFAULT_PDS).rstrip("/")

    def _login(self, account: Account) -> dict[str, Any]:
        """Create or reuse a session. App password in, access token out."""
        cached = self._sessions.get(account.handle)
        if cached and cached.get("accessJwt"):
            return cached
        password = account.resolve_token()
        if not password:
            raise ValueError(f"no app password configured for {account.handle}")
        client = HttpClient(timeout=self.timeout)
        response = client.post_json(
            f"{self._pds(account)}/xrpc/com.atproto.server.createSession",
            json_body={"identifier": account.handle, "password": password},
        )
        if not response.ok:
            raise ValueError(f"bluesky login failed ({response.status}): {response.text[:200]}")
        session = json.loads(response.text or "{}")
        self._sessions[account.handle] = session
        return session

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        try:
            session = self._login(account)
        except Exception as exc:  # noqa: BLE001 - auth failure is a result
            return PostResult(platform=self.name, ok=False, error=str(exc),
                              seconds=time.perf_counter() - started)

        did = session.get("did", "")
        record = {
            "$type": "app.bsky.feed.post",
            "text": content,
            "createdAt": _now_iso(),
            "langs": account.limits.get("langs") or ["en"],
        }
        reply_to = kwargs.get("reply_to") or ""
        if reply_to:
            record["reply"] = reply_to  # caller supplies the full reply ref

        client = HttpClient(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {session.get('accessJwt', '')}"},
        )
        body = {
            "repo": did,
            "collection": "app.bsky.feed.post",
            "record": record,
        }
        try:
            response = client.post_json(
                f"{self._pds(account)}/xrpc/com.atproto.repo.createRecord", body
            )
        except Exception as exc:  # noqa: BLE001
            return PostResult(platform=self.name, ok=False, error=str(exc),
                              seconds=time.perf_counter() - started)

        if not response.ok:
            # An expired session is retryable exactly once; drop it and re-login.
            if response.status in {400, 401} and "auth" in response.text.lower():
                self._sessions.pop(account.handle, None)
            return PostResult(
                platform=self.name, ok=False, error=response.text[:300],
                status_code=response.status, seconds=time.perf_counter() - started,
            )
        try:
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            data = {}
        uri = str(data.get("uri", ""))
        return PostResult(
            platform=self.name, ok=True, external_id=uri.rsplit("/", 1)[-1],
            url=_post_url(account.handle, uri), status_code=response.status,
            seconds=time.perf_counter() - started,
        )

    def health(self, account: Account) -> bool:
        try:
            self._login(account)
            return True
        except Exception:  # noqa: BLE001
            return False


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _post_url(handle: str, uri: str) -> str:
    """Build a bsky.app permalink from an at:// URI."""
    if not uri:
        return ""
    parts = uri.split("/")
    rkey = parts[-1] if parts else ""
    name = handle.split("@")[-1] if "@" in handle else handle
    return f"https://bsky.app/profile/{name}/post/{rkey}"
