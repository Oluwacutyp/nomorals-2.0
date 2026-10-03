"""Slack Web API connector.

Drives a Slack workspace over the Web API (https://api.slack.com) with
the framework HttpClient — no slack-sdk dependency.

Auth: a bot token (``API_KEY``, ``xoxb-…``) in the
``Authorization: Bearer`` header. Create the app at
https://api.slack.com/apps, add the scopes you need (``chat:write``,
``channels:read``, ``channels:history``, ...), and install it to the
workspace.

Slack answers every call with HTTP 200 and an ``ok`` field — a false
``ok`` carries a machine-readable ``error`` string (e.g.
``channel_not_found``). ``_api`` checks ``ok`` and raises on ``error``;
transport-level 429s surface the ``Retry-After`` header.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.errors import NoMoralsError, RateLimited
from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["SlackConnector", "SlackError"]

_log = get_logger(__name__)

API_BASE = "https://slack.com/api"
TOKEN_ENV = "SLACK_BOT_TOKEN"
DOCS_URL = "https://api.slack.com"
MAX_MESSAGE_CHARS = 40000


class SlackError(ConnectorError):
    """A Slack Web API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        slack_error: str = "",
        retry_after: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.slack_error = slack_error
        self.retry_after = retry_after


@register_connector
class SlackConnector(Connector):
    """Devon's Slack Web API adapter."""

    id = "slack"
    name = "Slack"
    description = (
        "Slack Web API: verify the bot, list channels, send messages, "
        "and read channel history. Authenticates with a bot token "
        "(xoxb-…) in the Authorization header."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a bot token against auth.test and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "slack is already connected — one account per service. "
                "Disconnect first to switch tokens."
            )
        secret = (token or "").strip() or prompt_secret(
            "Slack bot token (xoxb-…)", env_var=TOKEN_ENV
        )
        if not secret:
            raise ConnectorError("empty bot token: nothing to connect with")
        info = self._api("POST", "/auth.test", token=secret)
        team = str(info.get("team", ""))
        user = str(info.get("user", ""))
        label = f"@{user}@{team}" if team else f"@{user}"
        self._store_credential(
            label,
            secret,
            credential_type="api_key",
            scopes=["chat:write", "channels:read", "channels:history"],
            metadata={
                "team_id": info.get("team_id"),
                "team": team,
                "user_id": info.get("user_id"),
                "bot_user": user,
            },
        )
        _log.info("slack connected as %s", label)
        return ConnectResult(
            ok=True,
            account=label,
            scopes=["chat:write", "channels:read", "channels:history"],
            message=(
                f"connected as Slack bot @{user} on team {team}. The token "
                "is in the encrypted vault. The bot must be invited to a "
                "channel before it can read or post there."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name slack`",
            )
        try:
            info = self._api("POST", "/auth.test", token=cred.password)
        except SlackError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reinstall the app or "
                       "rotate the token and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=f"@{info.get('user', cred.username)}",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"team {info.get('team')} responding",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("POST", "/auth.test", token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── Web API ──────────────────────────────────────────────────

    def test_auth(self) -> dict[str, Any]:
        """Bot identity (``POST /auth.test``) — also the verifier."""
        data = self._api("POST", "/auth.test")
        return data if isinstance(data, dict) else {}

    def list_channels(
        self,
        *,
        limit: int = 200,
        cursor: str = "",
        types: str = "public_channel,private_channel,im,mpim",
    ) -> dict[str, Any]:
        """Channels the bot can see (``GET /conversations.list``).

        Returns ``{"channels": [...], "next_cursor": "..."}`` — pass
        ``next_cursor`` back as ``cursor`` to page.
        """
        params: dict[str, Any] = {
            "types": types,
            "limit": max(1, min(limit, 1000)),
        }
        if cursor:
            params["cursor"] = cursor
        data = self._api("GET", "/conversations.list", params=params)
        channels = data.get("channels", [])
        meta = data.get("response_metadata") or {}
        return {
            "channels": channels if isinstance(channels, list) else [],
            "next_cursor": str(meta.get("next_cursor", "")),
        }

    def send_message(
        self,
        channel: str,
        text: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Send a message (``POST /chat.postMessage``).

        ``channel`` is a channel id (``C…``/``G…``/``D…``) or ``#name``.
        Slack caps messages at 40,000 characters. Consequential: gated
        behind explicit owner confirmation.
        """
        channel = (channel or "").strip()
        if not channel:
            raise ConnectorError("channel is required")
        if not (text or ""):
            raise ConnectorError("refusing to send an empty message")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ConnectorError(
                f"message is {len(text)} chars; slack caps messages at "
                f"{MAX_MESSAGE_CHARS} — split it first"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="send_message",
            title=f"Send Slack message to {channel}",
            instructions="\n".join([
                "Devon wants to send this message on your Slack.",
                "Review it — sending is final.",
                f"Channel: {channel}",
                "",
                text if len(text) <= 500 else text[:500] + "…",
            ]),
            resume_state={"channel": channel, "text": text},
        )
        data = self._api(
            "POST", "/chat.postMessage",
            payload={"channel": channel, "text": text},
        )
        return data if isinstance(data, dict) else {}

    def read_history(
        self,
        channel: str,
        *,
        limit: int = 50,
        cursor: str = "",
        oldest: str = "",
        latest: str = "",
    ) -> dict[str, Any]:
        """Channel history (``GET /conversations.history``).

        Returns ``{"messages": [...], "has_more": bool,
        "next_cursor": "..."}``.
        """
        channel = (channel or "").strip()
        if not channel:
            raise ConnectorError("channel is required")
        params: dict[str, Any] = {
            "channel": channel,
            "limit": max(1, min(limit, 1000)),
        }
        if cursor:
            params["cursor"] = cursor
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest
        data = self._api("GET", "/conversations.history", params=params)
        messages = data.get("messages", [])
        meta = data.get("response_metadata") or {}
        return {
            "messages": messages if isinstance(messages, list) else [],
            "has_more": bool(data.get("has_more", False)),
            "next_cursor": str(meta.get("next_cursor", "")),
        }

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "slack is not connected — run "
                "`nm connectors connect --name slack` first"
            )
        return cred

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> Any:
        """One Slack Web API call; failures become SlackError.

        Slack reports API errors with HTTP 200 + ``{"ok": false,
        "error": "…"}`` — the ``ok`` field is authoritative, not the
        status code.
        """
        secret = token or self._require_credential().password
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {secret}"}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise SlackError(
                "slack rate limited (429): tier-1 methods allow ~1 "
                f"call/min — back off ~{exc.retry_after:.0f}s",
                status_code=429,
                slack_error="ratelimited",
                retry_after=exc.retry_after,
            ) from exc
        except NoMoralsError as exc:
            raise SlackError(f"slack request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SlackError(f"slack request failed: {exc}") from exc
        if resp.status == 429:
            raise SlackError(
                "slack rate limited (429): slow down and retry after "
                f"{self._retry_after(resp):.1f}s",
                status_code=429,
                slack_error="ratelimited",
                retry_after=self._retry_after(resp),
            )
        if not resp.ok:
            raise SlackError(
                f"slack {method} {path} failed (HTTP {resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise SlackError(
                f"slack {method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(body, dict) or not body.get("ok"):
            err = (
                str(body.get("error", "unknown error"))
                if isinstance(body, dict)
                else "non-dict response"
            )
            raise SlackError(
                f"slack {method} {path} failed: {self._friendly(err)}",
                status_code=resp.status,
                slack_error=err,
            )
        return body

    @staticmethod
    def _retry_after(resp: Any) -> float:
        try:
            return float(resp.headers.get("retry-after", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _friendly(error: str) -> str:
        """Turn Slack's terse error codes into actionable messages."""
        hints = {
            "invalid_auth": "the token is invalid or revoked — "
                            "reinstall the app and reconnect",
            "account_inactive": "the token belongs to a deactivated user",
            "token_revoked": "the token was revoked — reinstall the app",
            "channel_not_found": "no such channel, or the bot was never "
                                 "invited to it",
            "not_in_channel": "the bot is not a member of that channel — "
                              "invite it first",
            "is_archived": "the channel is archived",
            "msg_too_long": "the message exceeds Slack's length limit",
            "no_text": "the message had no text",
            "ratelimited": "too many calls — back off before retrying",
            "missing_scope": "the app lacks a scope for this method — add "
                             "it at api.slack.com/apps and reinstall",
        }
        hint = hints.get(error)
        return f"{error} ({hint})" if hint else error
