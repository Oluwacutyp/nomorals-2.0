"""Mastodon adapter — the official REST API, no scraping.

Mastodon is the reference adapter here: it is federated, self-hostable, and its
API is stable and well documented. Most other ActivityPub servers accept the same
endpoints, so this doubles as the template for them.
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


class Adapter(PlatformAdapter):
    name = "mastodon"
    max_chars = 500
    supports_media = True
    supports_thread = True

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def _client(self, account: Account) -> HttpClient:
        token = account.resolve_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return HttpClient(timeout=self.timeout, headers=headers)

    def _base(self, account: Account) -> str:
        """Instance URL, from limits or derived from the handle's domain."""
        base = account.limits.get("base_url") or ""
        if base:
            return str(base).rstrip("/")
        domain = account.handle.split("@")[-1] if "@" in account.handle else ""
        if not domain:
            raise ValueError(f"cannot derive a Mastodon instance for {account.handle!r}")
        return f"https://{domain}"

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        base = self._base(account)
        client = self._client(account)
        payload: dict[str, Any] = {"status": content, "visibility": account.limits.get("visibility", "public")}
        reply_to = kwargs.get("reply_to") or ""
        if reply_to:
            payload["in_reply_to_id"] = reply_to
        media_ids = kwargs.get("media_ids") or []
        if media_ids:
            payload["media_ids[]"] = media_ids

        try:
            response = client.post_json(f"{base}/api/v1/statuses", payload)
        except Exception as exc:  # noqa: BLE001 - network errors are results
            return PostResult(platform=self.name, ok=False, error=str(exc),
                              seconds=time.perf_counter() - started)

        if not response.ok:
            return PostResult(
                platform=self.name, ok=False, error=response.text[:300],
                status_code=response.status, seconds=time.perf_counter() - started,
                rate_limit_remaining=_int_header(response.headers, "x-ratelimit-remaining"),
                rate_limit_reset=_parse_reset(response.headers.get("x-ratelimit-reset", "")),
            )
        try:
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            data = {}
        return PostResult(
            platform=self.name, ok=True, external_id=str(data.get("id", "")),
            url=str(data.get("url", "")), status_code=response.status,
            seconds=time.perf_counter() - started,
            rate_limit_remaining=_int_header(response.headers, "x-ratelimit-remaining"),
        )

    def delete(self, account: Account, external_id: str) -> PostResult:
        started = time.perf_counter()
        try:
            response = self._client(account).request(
                "DELETE", f"{self._base(account)}/api/v1/statuses/{external_id}"
            )
        except Exception as exc:  # noqa: BLE001
            return PostResult(platform=self.name, ok=False, error=str(exc))
        return PostResult(
            platform=self.name, ok=response.ok, external_id=external_id,
            status_code=response.status, error="" if response.ok else response.text[:300],
            seconds=time.perf_counter() - started,
        )

    def metrics(self, account: Account, external_id: str) -> dict[str, Any]:
        try:
            response = self._client(account).get(
                f"{self._base(account)}/api/v1/statuses/{external_id}"
            )
            data = json.loads(response.text or "{}") if response.ok else {}
        except Exception:  # noqa: BLE001 - metrics are best-effort
            return {}
        return {
            "reblogs": int(data.get("reblogs_count") or 0),
            "favourites": int(data.get("favourites_count") or 0),
            "replies": int(data.get("replies_count") or 0),
        }

    def health(self, account: Account) -> bool:
        if not account.resolve_token():
            return False
        try:
            response = self._client(account).get(f"{self._base(account)}/api/v1/accounts/verify_credentials")
            return response.ok
        except Exception:  # noqa: BLE001 - unreachable is not healthy
            return False


def _int_header(headers: dict[str, str], name: str) -> int | None:
    raw = headers.get(name) or headers.get(name.upper())
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_reset(raw: str) -> float | None:
    """Mastodon sends an ISO-8601 reset time; turn it into an epoch float."""
    if not raw:
        return None
    try:
        import datetime

        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
