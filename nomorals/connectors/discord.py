"""Discord Bot API connector.

Drives a Discord bot over the REST API (https://discord.com/developers/docs)
with stdlib + the framework HttpClient — no discord.py dependency.

Auth: one bot token (``API_KEY``) in the ``Authorization: Bot <token>``
header. Create the application at https://discord.com/developers/applications,
add a bot, enable the privileged intents you need, and invite it with the
OAuth2 URL generator (``bot`` scope; Administrator only if you mean it).

Bot limits: 2000 chars per message, 5s+ per-message pacing on busy
channels, and gateway (websocket) events are out of scope — this connector
is REST-only, so it polls; it does not receive real-time events.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["DiscordConnector", "DiscordError"]

_log = get_logger(__name__)

API_BASE = "https://discord.com/api/v10"
TOKEN_ENV = "DISCORD_BOT_TOKEN"
DOCS_URL = "https://discord.com/developers/docs"


class DiscordError(ConnectorError):
    """A Discord REST call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        discord_code: int = 0,
        retry_after: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.discord_code = discord_code
        self.retry_after = retry_after


@register_connector
class DiscordConnector(Connector):
    """Devon's Discord REST adapter."""

    id = "discord"
    name = "Discord"
    description = (
        "Discord Bot API over plain REST: identify the bot, list guilds "
        "and channels, read users, and send messages to channels. "
        "Authenticates with a bot token (Authorization: Bot header)."
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
        """Validate a bot token against /users/@me and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "discord is already connected — one account per service. "
                "Disconnect first to switch bot tokens."
            )
        secret = (token or "").strip() or prompt_secret(
            "Discord bot token", env_var=TOKEN_ENV
        )
        if not secret:
            raise ConnectorError("empty bot token: nothing to connect with")
        me = self._api("GET", "/users/@me", token=secret)
        username = str(me.get("username", ""))
        discrim = str(me.get("discriminator", ""))
        label = f"@{username}#{discrim}" if discrim != "0" else f"@{username}"
        if not username:
            label = f"bot {me.get('id', '?')}"
        self._store_credential(
            label,
            secret,
            credential_type="api_key",
            scopes=["bot", "messages.write", "guilds.read"],
            metadata={"bot_id": me.get("id"), "username": username},
        )
        _log.info("discord connected as %s", label)
        return ConnectResult(
            ok=True,
            account=label,
            scopes=["bot", "messages.write", "guilds.read"],
            message=(
                f"connected as Discord bot {label}. The token is in the "
                "encrypted vault. The bot must already be a member of a "
                "guild to see its channels."
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
                       "--name discord`",
            )
        try:
            me = self._api("GET", "/users/@me", token=cred.password)
        except DiscordError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reset it in the developer "
                       "portal and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=f"@{me.get('username', cred.username)}",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"bot id {me.get('id')} responding",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/users/@me", token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── REST API ─────────────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """Bot identity (``GET /users/@me``) — also the verifier."""
        data = self._api("GET", "/users/@me")
        return data if isinstance(data, dict) else {}

    def get_user(self, user_id: str | int) -> dict[str, Any]:
        """A user's public profile (``GET /users/{id}``)."""
        data = self._api("GET", f"/users/{user_id}")
        return data if isinstance(data, dict) else {}

    def list_guilds(
        self, *, limit: int = 100, before: str = "", after: str = ""
    ) -> list[dict[str, Any]]:
        """Guilds the bot is in (``GET /users/@me/guilds``)."""
        params: dict[str, Any] = {"limit": max(1, min(limit, 200))}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        data = self._api("GET", "/users/@me/guilds", params=params)
        return data if isinstance(data, list) else []

    def list_channels(self, guild_id: str | int) -> list[dict[str, Any]]:
        """Channels of a guild (``GET /guilds/{id}/channels``).

        Includes text, voice, and category channels — filter on ``type``
        (0 = text, 2 = voice, 4 = category) as needed.
        """
        data = self._api("GET", f"/guilds/{guild_id}/channels")
        return data if isinstance(data, list) else []

    def get_channel(self, channel_id: str | int) -> dict[str, Any]:
        """One channel's metadata (``GET /channels/{id}``)."""
        data = self._api("GET", f"/channels/{channel_id}")
        return data if isinstance(data, dict) else {}

    def send_message(
        self,
        channel_id: str | int,
        content: str,
        *,
        tts: bool = False,
    ) -> dict[str, Any]:
        """Send a message to a channel (``POST /channels/{id}/messages``).

        Discord caps messages at 2000 characters — longer input is
        rejected, not silently truncated. The bot needs "Send Messages"
        permission in the channel.
        """
        if not str(content):
            raise ConnectorError("refusing to send an empty message")
        if len(content) > 2000:
            raise ConnectorError(
                f"message is {len(content)} chars; discord caps messages "
                "at 2000 — split it first"
            )
        data = self._api(
            "POST",
            f"/channels/{channel_id}/messages",
            payload={"content": content, "tts": bool(tts)},
        )
        return data if isinstance(data, dict) else {}

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "discord is not connected — run "
                "`nm connectors connect --name discord` first"
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
        """One Discord REST call; failures become DiscordError."""
        secret = token or self._require_credential().password
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bot {secret}"}
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=headers, params=params
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            elif method == "DELETE":
                resp = self.http.request(
                    "DELETE", url, headers=headers, params=params
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DiscordError(f"discord request failed: {exc}") from exc
        if resp.status == 401:
            raise DiscordError(
                "discord rejected the bot token (401): it is invalid or "
                "was reset — reset it in the developer portal and reconnect",
                status_code=401,
            )
        if resp.status == 403:
            detail, code = self._error_detail(resp)
            raise DiscordError(
                "discord refused (403): the bot lacks permission for this "
                f"— check its role and channel overrides ({detail})",
                status_code=403,
                discord_code=code,
            )
        if resp.status == 429:
            retry_after = self._retry_after(resp)
            raise DiscordError(
                "discord rate limited (429): slow down and retry after "
                f"{retry_after:.1f}s",
                status_code=429,
                retry_after=retry_after,
            )
        if not resp.ok:
            detail, code = self._error_detail(resp)
            raise DiscordError(
                f"discord {method} {path} failed ({resp.status}): {detail}",
                status_code=resp.status,
                discord_code=code,
            )
        if resp.status == 204:
            return None
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise DiscordError(
                f"discord {method} {path} returned invalid JSON"
            ) from exc

    @staticmethod
    def _retry_after(resp: Any) -> float:
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("retry_after"):
                return float(body["retry_after"])
        except Exception:  # noqa: BLE001 - best effort only
            pass
        try:
            return float(resp.headers.get("retry-after", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, int]:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return resp.text[:200], 0
        if isinstance(body, dict):
            message = str(body.get("message", body))[:200]
            try:
                code = int(body.get("code", 0))
            except (TypeError, ValueError):
                code = 0
            return message, code
        return resp.text[:200], 0
