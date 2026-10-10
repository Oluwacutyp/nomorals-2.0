"""Telegram Bot API connector.

This is the Bot API (https://core.telegram.org/bots/api) as a service
connector — it drives a bot created via @BotFather. It is NOT the userbot
or the social chat adapter; those live in ``nomorals/social/``.

Auth: one bot token from @BotFather (``API_KEY``). The token travels in the
request URL path (``/bot<token>/<method>``) per Telegram's API design — it
is never logged, and error paths scrub it before raising.

Bot limitations worth knowing: a bot can only message chats where it was
added or that messaged it first (no cold outreach to arbitrary users), and
bots cannot join groups on their own.
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
    ConnectorRateLimitError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["TelegramConnector", "TelegramError"]

_log = get_logger(__name__)

API_BASE = "https://api.telegram.org"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
DOCS_URL = "https://core.telegram.org/bots/api"


class TelegramError(ConnectorError):
    """A Telegram Bot API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        error_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@register_connector
class TelegramConnector(Connector):
    """Devon's Telegram Bot API adapter."""

    id = "telegram"
    name = "Telegram"
    description = (
        "Telegram Bot API: verify the bot, send messages and media (photo, "
        "video, document, albums), edit/delete messages, answer inline "
        "keyboard callbacks, manage webhooks, poll updates, inspect chats, "
        "and leave chats. Authenticates with a bot token from @BotFather."
    )
    auth_methods = (AuthMethod.API_KEY,)
    CATEGORY = "messaging"

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a bot token against getMe and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "telegram is already connected — one account per service. "
                "Disconnect first to switch bot tokens."
            )
        secret = (token or "").strip() or prompt_secret(
            "Telegram bot token", env_var=TOKEN_ENV
        )
        if not secret:
            raise ConnectorError("empty bot token: nothing to connect with")
        me = self._api("getMe", token=secret)
        username = str(me.get("username", ""))
        label = f"@{username}" if username else f"bot {me.get('id', '?')}"
        can_groups = me.get("can_join_groups", False)
        can_read = me.get("can_read_all_group_messages", False)
        self._store_credential(
            label,
            secret,
            credential_type="api_key",
            scopes=["send", "updates", "chats"],
            metadata={"bot_id": me.get("id"), "username": username},
        )
        _log.info("telegram connected as %s", label)
        return ConnectResult(
            ok=True,
            account=label,
            scopes=["send", "updates", "chats"],
            message=(
                f"connected as Telegram bot {label} (id {me.get('id')}). "
                f"Can join groups: {can_groups}; reads all group messages: "
                f"{can_read}. The token is in the encrypted vault. Talk to "
                f"{label} or add it to a chat, then use send_message()."
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
                       "--name telegram`",
            )
        try:
            me = self._api("getMe", token=cred.password)
        except TelegramError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): create a fresh one with "
                       "@BotFather and reconnect",
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
            self._api("getMe", token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── bot API ──────────────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """Bot identity (``getMe``) — also the connection verifier."""
        return self._api("getMe")

    def send_message(
        self,
        chat_id: str | int,
        text: str,
        *,
        parse_mode: str = "",
        disable_notification: bool = False,
        reply_to_message_id: int = 0,
    ) -> dict[str, Any]:
        """Send a text message (``sendMessage``).

        ``chat_id`` is a numeric id, ``@channelusername``, or (for a bot
        that the user messaged first) the user's id. Text over 4096
        characters is rejected by Telegram — split it before calling.
        """
        if not str(text):
            raise ConnectorError("refusing to send an empty message")
        if len(text) > 4096:
            raise ConnectorError(
                f"message is {len(text)} chars; telegram caps sendMessage "
                "at 4096 — split it first"
            )
        params: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            if parse_mode not in ("HTML", "MarkdownV2"):
                raise ConnectorError(
                    f"invalid parse_mode {parse_mode!r}: use 'HTML' or "
                    "'MarkdownV2'"
                )
            params["parse_mode"] = parse_mode
        if disable_notification:
            params["disable_notification"] = True
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        return self._api("sendMessage", params=params)

    def get_updates(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        timeout: int = 0,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Poll pending updates (``getUpdates``).

        Long-poll with ``timeout`` up to 50. This is polling, not a
        webhook — do not call it in a tight loop; back off when empty.
        """
        params: dict[str, Any] = {
            "offset": offset,
            "limit": max(1, min(limit, 100)),
            "timeout": max(0, min(timeout, 50)),
        }
        if allowed_updates:
            params["allowed_updates"] = allowed_updates
        result = self._api("getUpdates", params=params)
        return result if isinstance(result, list) else []

    def get_chat(self, chat_id: str | int) -> dict[str, Any]:
        """Chat info (``getChat``) — title, type, member count hints."""
        return self._api("getChat", params={"chat_id": chat_id})

    def leave_chat(self, chat_id: str | int) -> bool:
        """Leave a group/supergroup/channel (``leaveChat``)."""
        return bool(self._api("leaveChat", params={"chat_id": chat_id}))

    # ── media ────────────────────────────────────────────────────

    @staticmethod
    def _media_params(
        chat_id: str | int, caption: str, parse_mode: str
    ) -> dict[str, Any]:
        if len(caption) > 1024:
            raise ConnectorError(
                f"caption is {len(caption)} chars; telegram caps captions "
                "at 1024 — shorten it first"
            )
        if parse_mode and parse_mode not in ("HTML", "MarkdownV2"):
            raise ConnectorError(
                f"invalid parse_mode {parse_mode!r}: use 'HTML' or 'MarkdownV2'"
            )
        params: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            params["caption"] = caption
        if parse_mode:
            params["parse_mode"] = parse_mode
        return params

    @staticmethod
    def _is_local_file(value: str) -> bool:
        from pathlib import Path
        return not value.startswith(("http://", "https://", "file_id:")) and bool(
            value) and Path(value).is_file()

    def send_photo(
        self, chat_id: str | int, photo: str,
        *, caption: str = "", parse_mode: str = "",
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        """Send a photo (``sendPhoto``).

        ``photo`` is a local file path (uploaded), an ``https://`` URL, or
        a Telegram ``file_id``. Photos cap at 10 MB server-side.
        """
        params = self._media_params(chat_id, caption, parse_mode)
        if disable_notification:
            params["disable_notification"] = True
        if self._is_local_file(photo):
            return self._api_multipart(
                "sendPhoto", params, {"photo": photo})
        params["photo"] = photo
        return self._api("sendPhoto", params=params)

    def send_document(
        self, chat_id: str | int, document: str,
        *, caption: str = "", parse_mode: str = "",
        filename: str = "",
    ) -> dict[str, Any]:
        """Send a file (``sendDocument``) — up to 50 MB via Bot API."""
        params = self._media_params(chat_id, caption, parse_mode)
        if self._is_local_file(document):
            return self._api_multipart(
                "sendDocument", params, {"document": document})
        params["document"] = document
        if filename:
            params["filename"] = filename
        return self._api("sendDocument", params=params)

    def send_video(
        self, chat_id: str | int, video: str,
        *, caption: str = "", parse_mode: str = "",
        supports_streaming: bool = True,
    ) -> dict[str, Any]:
        """Send a video (``sendVideo``) — up to 50 MB via Bot API."""
        params = self._media_params(chat_id, caption, parse_mode)
        params["supports_streaming"] = supports_streaming
        if self._is_local_file(video):
            return self._api_multipart(
                "sendVideo", params, {"video": video})
        params["video"] = video
        return self._api("sendVideo", params=params)

    def send_media_group(
        self, chat_id: str | int, media: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Send an album of 2–10 photos/videos (``sendMediaGroup``).

        Each item: ``{"type": "photo"|"video", "media": <path|url|file_id>,
        "caption": "..."}``. Local files are uploaded with ``attach://``
        references.
        """
        import json as _json

        if not 2 <= len(media) <= 10:
            raise ConnectorError(
                f"media groups need 2–10 items, got {len(media)}")
        upload_fields: dict[str, str] = {}
        wire: list[dict[str, Any]] = []
        for i, item in enumerate(media):
            kind = str(item.get("type", "photo"))
            if kind not in ("photo", "video"):
                raise ConnectorError(
                    f"media group item {i}: bad type {kind!r}")
            src = str(item.get("media", ""))
            entry: dict[str, Any] = {"type": kind}
            if self._is_local_file(src):
                ref = f"attach://upload{i}"
                entry["media"] = ref
                upload_fields[f"upload{i}"] = src
            else:
                entry["media"] = src
            if item.get("caption"):
                entry["caption"] = item["caption"][:1024]
            wire.append(entry)
        fields = {
            "chat_id": str(chat_id),
            "media": _json.dumps(wire),
        }
        if upload_fields:
            return self._api_multipart("sendMediaGroup", fields, upload_fields)
        result = self._api("sendMediaGroup", params=fields)
        return result if isinstance(result, list) else []

    # ── message management ───────────────────────────────────────

    def edit_message_text(
        self, chat_id: str | int, message_id: int, text: str,
        *, parse_mode: str = "",
    ) -> dict[str, Any]:
        """Edit a sent message (``editMessageText``)."""
        if not str(text):
            raise ConnectorError("refusing to set an empty message text")
        if len(text) > 4096:
            raise ConnectorError(
                f"text is {len(text)} chars; telegram caps messages at 4096")
        params: dict[str, Any] = {
            "chat_id": chat_id, "message_id": message_id, "text": text}
        if parse_mode:
            if parse_mode not in ("HTML", "MarkdownV2"):
                raise ConnectorError(
                    f"invalid parse_mode {parse_mode!r}")
            params["parse_mode"] = parse_mode
        return self._api("editMessageText", params=params)

    def delete_message(self, chat_id: str | int, message_id: int) -> bool:
        """Delete a message (``deleteMessage``)."""
        return bool(self._api("deleteMessage", params={
            "chat_id": chat_id, "message_id": message_id}))

    def answer_callback_query(
        self, callback_query_id: str, *, text: str = "",
        show_alert: bool = False,
    ) -> bool:
        """Acknowledge an inline-keyboard button press.

        Always answer callbacks — an unanswered button spins forever on
        the user's client. ``text`` ≤ 200 chars.
        """
        params: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text[:200]
        if show_alert:
            params["show_alert"] = True
        return bool(self._api("answerCallbackQuery", params=params))

    # ── webhooks ─────────────────────────────────────────────────

    def set_webhook(
        self, url: str, *, secret_token: str = "",
        allowed_updates: list[str] | None = None,
    ) -> bool:
        """Register a webhook URL (``setWebhook``).

        ``secret_token`` (1–256 chars) makes Telegram send
        ``X-Telegram-Bot-Api-Secret-Token`` with every update — validate
        it on the endpoint. Empty url switches back to polling.
        """
        params: dict[str, Any] = {"url": url}
        if secret_token:
            if not 1 <= len(secret_token) <= 256:
                raise ConnectorError(
                    "secret_token must be 1–256 characters")
            params["secret_token"] = secret_token
        if allowed_updates:
            params["allowed_updates"] = allowed_updates
        return bool(self._api("setWebhook", params=params))

    def delete_webhook(self) -> bool:
        """Remove the webhook (``deleteWebhook``) — back to polling."""
        return bool(self._api("deleteWebhook"))

    def get_webhook_info(self) -> dict[str, Any]:
        """Current webhook status (``getWebhookInfo``)."""
        result = self._api("getWebhookInfo")
        return result if isinstance(result, dict) else {}

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "telegram is not connected — run "
                "`nm connectors connect --name telegram` first"
            )
        return cred

    def _api(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        token: str | None = None,
    ) -> Any:
        """One Bot API call. API-level failures become TelegramError."""
        secret = token or self._require_credential().password
        url = f"{API_BASE}/bot{secret}/{method}"
        try:
            resp = self.http.post_json(url, params or {})
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise TelegramError(
                f"telegram request failed: {exc}"
            ) from exc
        # Scrub the token out of anything that could reach a log or error.
        safe_url = url.replace(secret, "<token>")
        if resp.status == 401:
            raise TelegramError(
                "telegram rejected the bot token (401): it is invalid or "
                "revoked — create a fresh one with @BotFather and reconnect",
                status_code=401,
            )
        if resp.status == 429:
            # Telegram returns 429 with {"parameters": {"retry_after": N}} —
            # the taxonomy carries the wait so callers back off properly.
            retry_after = 1.0
            try:
                body429 = resp.json()
                params = (body429.get("parameters") or {}) if isinstance(
                    body429, dict) else {}
                retry_after = float(params.get("retry_after", 1) or 1)
            except Exception:  # noqa: BLE001 - advisory only
                pass
            raise ConnectorRateLimitError(
                "telegram rate limit hit (429) — back off before retrying",
                retry_after=retry_after,
            )
        if not resp.ok:
            raise TelegramError(
                f"telegram {method} failed (HTTP {resp.status}, "
                f"{safe_url}): {resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise TelegramError(
                f"telegram {method} returned invalid JSON"
            ) from exc
        if not isinstance(body, dict) or not body.get("ok"):
            description = (
                body.get("description", "unknown error")
                if isinstance(body, dict)
                else "non-dict response"
            )
            code = (
                int(body.get("error_code", 0))
                if isinstance(body, dict)
                else 0
            )
            raise TelegramError(
                f"telegram {method} failed: {description}",
                status_code=resp.status,
                error_code=code,
            )
        return body.get("result")

    def _api_multipart(
        self,
        method: str,
        fields: dict[str, Any],
        uploads: dict[str, str],
    ) -> Any:
        """One Bot API call with file uploads (multipart/form-data).

        ``uploads`` maps form field name → local file path.
        """
        from pathlib import Path

        secret = self._require_credential().password
        url = f"{API_BASE}/bot{secret}/{method}"
        str_fields = {k: str(v) for k, v in fields.items()}
        files = [
            (name, Path(path), "")
            for name, path in uploads.items()
        ]
        try:
            resp = self.http.post_multipart(
                url, fields=str_fields, files=files)
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise TelegramError(
                f"telegram upload failed: {exc}") from exc
        if resp.status == 401:
            raise TelegramError(
                "telegram rejected the bot token (401)",
                status_code=401,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise TelegramError(
                f"telegram {method} returned invalid JSON") from exc
        if not isinstance(body, dict) or not body.get("ok"):
            description = (
                body.get("description", "unknown error")
                if isinstance(body, dict) else "non-dict response")
            raise TelegramError(
                f"telegram {method} failed: {description}",
                status_code=resp.status,
            )
        return body.get("result")
