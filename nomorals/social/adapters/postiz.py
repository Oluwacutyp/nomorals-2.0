"""Postiz adapter — self-hosted publishing backend for 30+ networks.

Postiz (gitroomhq/postiz-app) is the "don't rebuild OAuth hell" answer: you
connect your X / LinkedIn / Instagram / TikTok / YouTube / Discord / Telegram
/ Bluesky / ... accounts once in Postiz's UI, and Devon publishes through one
documented REST API. All per-platform OAuth pain lives in Postiz, not here.

Self-host (Docker, one command)::

    docker run -d --name postiz -p 5000:5000 \\
        -e MAIN_URL="http://localhost:5000" \\
        -e FRONTEND_URL="http://localhost:5000" \\
        -e DATABASE_URL="postgresql://postiz:postizpw@127.0.0.1:5432/postiz-db-local" \\
        -e REDIS_URL="redis://127.0.0.1:6379" \\
        -e JWT_SECRET="replace-me" \\
        ghcr.io/gitroomhq/postiz-app:latest

Then: Settings > Developers > Public API → create an API key, and connect
your social accounts in the Postiz UI. Devon needs::

    POSTIZ_URL=https://postiz.example.com        # or http://localhost:5000
    POSTIZ_API_KEY=...                           # env:POSTIZ_API_KEY

Public API contract (docs.postiz.com/public-api, verified against the
postiz-app issue tracker and community clients; ``api_prefix`` is
configurable because Postiz versions differ on ``/api/public/v1`` vs
``/public/v1``):

- Auth: ``Authorization: <API_KEY>`` header.
- Create: ``POST {prefix}/posts`` with::

      {"type": "now"|"schedule"|"draft", "date": "ISO8601",
       "shortLink": false, "tags": [],
       "posts": [{"integration": {"id": "<integration-id>"},
                  "value": [{"content": "...",
                             "image": [{"id": "<upload-id>", "path": "<cdn-url>"}]}],
                  "group": "<uuid-shared-by-this-batch>",
                  "settings": {<platform-specific>}}]}

- Upload: ``POST {prefix}/upload`` (multipart) → ``{"id": ..., "path": ...}``.
  Media is a two-step dance: upload first, then reference the returned
  ``id`` (required — Postiz rejects ``{path}`` alone).
- Integrations: ``GET {prefix}/integrations`` → the channel ids.
- Delete: ``DELETE {prefix}/posts/<id>``.

Platform ``settings`` gotchas (omit → Postiz 400s): YouTube needs
``{"title", "type": "public"|"private"|"unlisted"}``; Discord needs
``{"channel": "#name"}``; Instagram needs ``{"post_type": "post"|"story"}``;
Pinterest needs ``{"title", "board"}``. Pass them per-integration via the
``settings`` kwarg (``{integration_id: {...}}``).

One account on this adapter = one Postiz instance. ``integration_ids``
(kwarg, or ``account.limits["integration_id"]``) selects which connected
channels receive the post.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from ...core.http import HttpClient, HttpResponse
from ...core.logging_setup import get_logger
from ..base import Account, PlatformAdapter, PostResult, SocialError

__all__ = ["Adapter"]

_log = get_logger(__name__)

#: Fields Postiz versions are known to disagree on. Verified against
#: docs.postiz.com/public-api and postiz-app issue #717; ``api_prefix`` is
#: configurable for instances that serve ``/public/v1`` instead.
DEFAULT_API_PREFIX = "/api/public/v1"

#: Postiz rate limits: ~90 req/hour self-hosted, ~100/hour cloud.
RATE_LIMIT_PER_HOUR = 90

RETRYABLE = {429, 500, 502, 503, 504}


class Adapter(PlatformAdapter):
    name = "postiz"
    # Postiz enforces per-network limits itself; keep ours generous.
    max_chars = 3000
    supports_media = True
    supports_thread = False

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        api_prefix: str = DEFAULT_API_PREFIX,
        timeout: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self.base_url = (base_url or os.environ.get("POSTIZ_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("POSTIZ_API_KEY", "")
        self.api_prefix = (api_prefix or DEFAULT_API_PREFIX).rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))

    # ── config ──────────────────────────────────────────────────────────

    def _config(self, account: Account) -> tuple[str, str]:
        """Resolve (base_url, api_key). Fail closed with a clear error."""
        base = (account.limits.get("base_url") or self.base_url).rstrip("/")
        key = account.resolve_token() or self.api_key
        missing = []
        if not base:
            missing.append("POSTIZ_URL (or account.limits['base_url'])")
        if not key:
            missing.append("POSTIZ_API_KEY (or credentials='env:POSTIZ_API_KEY')")
        if missing:
            raise SocialError(
                "Postiz is not configured — set " + " and ".join(missing) + ". "
                "Self-host with Docker (see this module's docstring), connect "
                "your accounts in the Postiz UI, then retry. Nothing was posted."
            )
        return base, key

    def _client(self, api_key: str) -> HttpClient:
        # A self-hosted Postiz usually lives on localhost or a LAN IP; the
        # base URL is operator-configured, so private IPs are explicitly OK.
        return HttpClient(
            timeout=self.timeout,
            headers={"Authorization": api_key},
            allow_private_ips=True,
        )

    def _url(self, base: str, path: str) -> str:
        return f"{base}{self.api_prefix}{path}"

    # ── integrations ────────────────────────────────────────────────────

    def _integration_ids(self, account: Account, kwargs: dict[str, Any]) -> list[str]:
        ids = kwargs.get("integration_ids") or kwargs.get("platforms") or []
        if isinstance(ids, str):
            ids = [ids]
        ids = [str(i) for i in ids]
        if not ids:
            single = account.limits.get("integration_id") or ""
            if single:
                ids = [str(single)]
        if not ids:
            raise SocialError(
                "no Postiz integration selected — pass integration_ids=[...] "
                "(from GET /integrations) or set "
                "account.limits['integration_id']. Nothing was posted."
            )
        return ids

    def list_integrations(self, account: Account) -> list[dict[str, Any]]:
        """Connected channels on this Postiz instance (id, provider, name)."""
        base, key = self._config(account)
        response = self._client(key).request(
            "GET", self._url(base, "/integrations"))
        if not response.ok:
            raise SocialError(
                f"Postiz integrations lookup failed ({response.status}): "
                f"{response.text[:200]}")
        try:
            data = json.loads(response.text or "[]")
        except json.JSONDecodeError:
            data = []
        items = data if isinstance(data, list) else data.get("integrations", [])
        out = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                out.append({
                    "id": str(item.get("id", "")),
                    "provider": str(item.get("identifier") or item.get("provider") or ""),
                    "name": str(item.get("name") or item.get("username") or ""),
                })
        return out

    # ── media ───────────────────────────────────────────────────────────

    def _upload_media(
        self, client: HttpClient, base: str, paths: Sequence[str]
    ) -> list[dict[str, str]]:
        """Two-step upload: file → Postiz → {id, path} reference. Required
        before the post call; Postiz rejects bare paths."""
        uploaded = []
        for path in paths:
            response = self._request_with_retry(
                client, "POST", self._url(base, "/upload"), multipart=[path])
            if not response.ok:
                raise SocialError(
                    f"Postiz media upload failed ({response.status}): "
                    f"{response.text[:200]}")
            try:
                data = json.loads(response.text or "{}")
            except json.JSONDecodeError as exc:
                raise SocialError(f"Postiz upload returned bad JSON: {exc}") from exc
            # Shape varies by version: {"id","path"} or {"file":{"id","path"}}.
            node = data.get("file", data) if isinstance(data, dict) else {}
            file_id = str(node.get("id", ""))
            file_path = str(node.get("path", ""))
            if not file_id:
                raise SocialError(
                    f"Postiz upload gave no file id: {str(data)[:200]}")
            uploaded.append({"id": file_id, "path": file_path})
        return uploaded

    # ── HTTP with retry ─────────────────────────────────────────────────

    def _request_with_retry(
        self,
        client: HttpClient,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        multipart: list[str] | None = None,
    ) -> HttpResponse:
        last: HttpResponse | None = None
        for attempt in range(1, self.max_attempts + 1):
            if multipart:
                files = [("file", p, "") for p in multipart]
                response = client.post_multipart(url, files=files)
            else:
                response = client.post_json(url, payload or {})
            last = response
            if response.status not in RETRYABLE or attempt == self.max_attempts:
                return response
            wait = min(2.0 ** attempt, 15.0)
            _log.warning("postiz %s %s → retrying in %.0fs (attempt %d/%d)",
                         response.status, url, wait, attempt, self.max_attempts)
            time.sleep(wait)
        return last  # type: ignore[return-value]

    # ── post ────────────────────────────────────────────────────────────

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        try:
            self.validate(content)
            base, key = self._config(account)
            integration_ids = self._integration_ids(account, kwargs)
            client = self._client(key)

            media_paths = list(kwargs.get("media_paths") or [])
            images = self._upload_media(client, base, media_paths) if media_paths else []

            schedule_at = kwargs.get("schedule_at")
            post_type, date = self._schedule(schedule_at)

            settings = kwargs.get("settings") or {}
            variants: dict[str, str] = kwargs.get("variants") or {}
            group = uuid.uuid4().hex
            posts = []
            for iid in integration_ids:
                text = str(variants.get(iid, content))
                posts.append({
                    "integration": {"id": iid},
                    "value": [{"content": text, "image": images}],
                    "group": group,
                    "settings": dict(settings.get(iid, {}) or {}),
                })
            payload = {
                "type": post_type,
                "date": date,
                "shortLink": bool(kwargs.get("short_link", False)),
                "tags": list(kwargs.get("tags") or []),
                "posts": posts,
            }
            response = self._request_with_retry(
                client, "POST", self._url(base, "/posts"), payload=payload)
        except Exception as exc:  # noqa: BLE001 - platform errors are results
            return PostResult(
                platform=self.name, ok=False, error=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started)

        seconds = time.perf_counter() - started
        if not response.ok:
            return PostResult(
                platform=self.name, ok=False,
                error=f"Postiz rejected the post ({response.status}): "
                      f"{response.text[:300]}",
                status_code=response.status, seconds=seconds,
                metrics={"channels": [
                    {"integration_id": iid, "ok": False,
                     "error": f"HTTP {response.status}"}
                    for iid in integration_ids]},
            )
        data = self._json(response)
        channels = self._channel_results(data, integration_ids)
        failed = [c for c in channels if not c["ok"]]
        return PostResult(
            platform=self.name,
            ok=not failed,
            external_id=str(data.get("id", "")),
            url=str(data.get("url", "")),
            error="; ".join(c["error"] for c in failed if c.get("error")),
            status_code=response.status,
            seconds=seconds,
            metrics={"channels": channels,
                     "post_type": post_type,
                     "scheduled_for": date if post_type == "schedule" else ""},
        )

    def _schedule(self, schedule_at: Any) -> tuple[str, str]:
        """→ ("now"|"schedule", ISO date). Accepts datetime or ISO string."""
        if schedule_at is None:
            return "now", self._iso_now()
        if isinstance(schedule_at, datetime):
            dt = schedule_at
        else:
            try:
                dt = datetime.fromisoformat(str(schedule_at))
            except ValueError as exc:
                raise SocialError(
                    f"schedule_at must be a datetime or ISO string, got "
                    f"{schedule_at!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return "schedule", dt.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _json(response: HttpResponse) -> dict[str, Any]:
        try:
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _channel_results(
        self, data: dict[str, Any], integration_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Per-channel outcomes. Postiz's response shape is version-dependent;
        parse defensively and never claim per-channel success we can't see."""
        channels: list[dict[str, Any]] = []
        # Some versions return per-post entries under "posts"/"items".
        entries = data.get("posts") or data.get("items") or []
        by_integration: dict[str, dict[str, Any]] = {}
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                iid = str((entry.get("integration") or {}).get("id")
                          or entry.get("integration_id") or "")
                if iid:
                    by_integration[iid] = entry
        for iid in integration_ids:
            entry = by_integration.get(iid, {})
            err = str(entry.get("error", ""))
            ok = not err and data.get("id") not in (None, "")
            channels.append({
                "integration_id": iid,
                "ok": ok,
                "error": err,
                "external_id": str(entry.get("id") or data.get("id") or ""),
            })
        return channels

    # ── delete / health ─────────────────────────────────────────────────

    def delete(self, account: Account, external_id: str) -> PostResult:
        started = time.perf_counter()
        try:
            base, key = self._config(account)
            response = self._client(key).request(
                "DELETE", self._url(base, f"/posts/{external_id}"))
        except Exception as exc:  # noqa: BLE001
            return PostResult(platform=self.name, ok=False,
                              error=f"{type(exc).__name__}: {exc}",
                              seconds=time.perf_counter() - started)
        if not response.ok:
            return PostResult(platform=self.name, ok=False,
                              error=f"Postiz delete failed ({response.status}): "
                                    f"{response.text[:200]}",
                              status_code=response.status,
                              seconds=time.perf_counter() - started)
        return PostResult(platform=self.name, ok=True, external_id=external_id,
                          seconds=time.perf_counter() - started)

    def health(self, account: Account) -> bool:
        base = (account.limits.get("base_url") or self.base_url).strip()
        return bool(base and (account.resolve_token() or self.api_key))
