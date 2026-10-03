"""Gmail connector — the owner's mailbox over the Gmail API.

Docs: https://developers.google.com/gmail/api/reference/rest

Auth: OAuth 2.0 (``AuthMethod.OAUTH2``) via the shared Google
authorization-code flow in ``_google_oauth`` — the owner grants the
``gmail.readonly`` + ``gmail.send`` scopes in their own browser and pastes
back the code. The refresh token is vaulted; access tokens auto-refresh.

Sending is a consequential action: ``send_message`` refuses to run without
explicit owner confirmation (``confirmed=True`` after the owner has seen
the exact recipient/subject/body, or a human checkpoint when ``db`` is
given). Reads never need confirmation.
"""

from __future__ import annotations

import base64
import email.utils
import time
from typing import Any

from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from ._google_oauth import GoogleOAuth
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .checkpoints import (
    CheckpointKind,
    CheckpointState,
    HumanCheckpointPending,
)
from .registry import register_connector

__all__ = ["GmailConnector", "GmailError"]

_log = get_logger(__name__)

API_BASE = "https://gmail.googleapis.com/gmail/v1"

_SCOPE_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
_SCOPE_SEND = "https://www.googleapis.com/auth/gmail.send"


class GmailError(ConnectorError):
    """A Gmail API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, reason: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


@register_connector
class GmailConnector(Connector, GoogleOAuth):
    """Devon's Gmail adapter: search, read, and send mail."""

    id = "gmail"
    name = "Gmail"
    description = (
        "Gmail API: list/search messages, read full messages, and send "
        "mail. OAuth 2.0 (readonly + send scopes); sending requires "
        "explicit owner confirmation."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def _google_default_scopes(self) -> list[str]:
        return [_SCOPE_READONLY, _SCOPE_SEND]

    def connect(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        code: str | None = None,
        scopes: list[str] | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Complete the Google OAuth flow and vault the tokens.

        Without ``code``: prints the grant guide + authorization URL. With
        ``db`` the flow pauses at a human checkpoint (the owner grants
        access in their browser, then resolves the checkpoint with the
        code); without it, the owner hands the code back and calls
        ``connect(..., code=<code>)`` again.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "gmail is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        cid = self._google_client_id(client_id)
        wanted = list(scopes or self._google_default_scopes())
        if code:
            secret = self._google_client_secret(client_secret)
            tokens = self._google_exchange_code(cid, secret, code)
            account = self._profile_email(tokens.get("access_token", ""))
            self._store_google_tokens(
                account or "gmail", cid, tokens,
                account=account, scopes=wanted,
            )
            return ConnectResult(
                ok=True,
                account=account or "gmail",
                scopes=wanted,
                message=(
                    f"connected to Gmail as {account or 'the granted account'}. "
                    "Tokens are in the encrypted vault."
                ),
            )
        guide = self._google_connect_guide(cid, wanted)
        print(guide)
        if db is None:
            return ConnectResult(
                ok=False,
                account="",
                scopes=wanted,
                message=(
                    "grant access in your browser, then call "
                    "connect(..., code=<code>) with the code Google shows"
                ),
            )
        try:
            self.request_human(
                CheckpointKind.MANUAL_STEP,
                "Grant Gmail access",
                guide
                + "\n\nWhen Google shows the authorization code, resolve "
                "this checkpoint with the code, e.g. note "
                "'code=<authorization code>'.",
                db=db,
                context=context,
                resume_state={
                    "stage": "oauth_code",
                    "client_id": cid,
                    "scopes": wanted,
                },
            )
        except HumanCheckpointPending as pending:
            return ConnectResult(
                ok=False,
                account="",
                scopes=wanted,
                message=(
                    "grant access in your browser, then resolve "
                    f"checkpoint {pending.checkpoint.id} with the code"
                ),
            )
        # Interactive TTY: resolved already — but the code arrives via the
        # checkpoint note, so finish through resume_checkpoint.
        raise ConnectorError(
            "access granted interactively but no code was captured — call "
            "connect(..., code=<code>) with the code Google showed"
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name gmail`",
            )
        try:
            email = self._profile_email(self._google_access_token())
        except GmailError as exc:
            return ConnectorStatus(
                connected=False,
                account=str((cred.metadata or {}).get("account", "gmail")),
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reconnect with a fresh code",
            )
        return ConnectorStatus(
            connected=True,
            account=email,
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="oauth tokens valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._profile_email(self._google_access_token())
            return True
        except ConnectorError:
            return False

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
        client_secret: str | None = None,
    ) -> dict[str, Any]:
        """Continue after a human checkpoint resolved."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — finish the human step first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage == "oauth_code":
            code = self._code_from_note(checkpoint.result_note or "")
            if not code:
                raise ConnectorError(
                    "the resolved checkpoint has no authorization code — "
                    "resolve it again with note 'code=<authorization code>'"
                )
            cid = str((checkpoint.resume_state or {}).get("client_id", ""))
            scopes = list(
                (checkpoint.resume_state or {}).get(
                    "scopes", self._google_default_scopes()
                )
            )
            secret = self._google_client_secret(client_secret)
            tokens = self._google_exchange_code(cid, secret, code)
            account = self._profile_email(tokens["access_token"])
            self._store_google_tokens(
                account or "gmail", cid, tokens,
                account=account, scopes=scopes,
            )
            return {"connected": True, "account": account}
        if stage == "send_email":
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("to"):
                raise ConnectorError(
                    "the resolved checkpoint has no message payload — "
                    "it cannot send"
                )
            return self._send_now(payload)
        raise ConnectorError(
            f"gmail cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def list_messages(
        self,
        *,
        query: str = "",
        max_results: int = 50,
        page_token: str = "",
        label_ids: list[str] | None = None,
        include_spam_trash: bool = False,
    ) -> dict[str, Any]:
        """List messages (``users.messages.list``).

        ``query`` is Gmail search syntax (``from:``, ``subject:``,
        ``after:``, ``has:attachment``, ``is:unread``, ...).
        """
        params: dict[str, Any] = {
            "maxResults": max(1, min(max_results, 500)),
            "includeSpamTrash": str(include_spam_trash).lower(),
        }
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token
        if label_ids:
            params["labelIds"] = label_ids
        data = self._api("GET", "/users/me/messages", params=params)
        return {
            "messages": data.get("messages", []),
            "next_page_token": data.get("nextPageToken", ""),
            "result_size_estimate": data.get("resultSizeEstimate", 0),
        }

    def get_message(
        self, message_id: str, *, format: str = "full"  # noqa: A002
    ) -> dict[str, Any]:
        """One message (``users.messages.get``).

        ``format``: ``full`` (default), ``metadata``, ``minimal``, or
        ``raw``. Snippet, labels, and payload are returned as-is.
        """
        if format not in ("full", "metadata", "minimal", "raw"):
            raise ConnectorError(
                f"invalid message format {format!r}: use full, metadata, "
                "minimal, or raw"
            )
        if not (message_id or "").strip():
            raise ConnectorError("empty message id")
        return self._api(
            "GET",
            f"/users/me/messages/{message_id}",
            params={"format": format},
        )

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        """A whole conversation (``users.threads.get``)."""
        if not (thread_id or "").strip():
            raise ConnectorError("empty thread id")
        return self._api("GET", f"/users/me/threads/{thread_id}")

    def list_labels(self) -> list[dict[str, Any]]:
        """Mailbox labels (``users.labels.list``)."""
        data = self._api("GET", "/users/me/labels")
        return data.get("labels", [])

    # ── sending (confirmation-gated) ─────────────────────────────

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        *,
        cc: str = "",
        bcc: str = "",
        html: bool = False,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Send a message (``users.messages.send``).

        Sending moves information out of the owner's mailbox, so it never
        runs on implied consent: pass ``confirmed=True`` only after the
        owner has seen the exact recipient, subject, and body — or pass
        ``db`` to park the exact draft on a human checkpoint instead.
        """
        payload = {
            "to": (to or "").strip(),
            "subject": subject or "",
            "body": body or "",
            "cc": (cc or "").strip(),
            "bcc": (bcc or "").strip(),
            "html": bool(html),
        }
        if not payload["to"]:
            raise ConnectorError("refusing to send: no recipient")
        if not payload["body"]:
            raise ConnectorError("refusing to send an empty body")
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="send_email",
            title=f"Send email to {payload['to']}",
            instructions="\n".join([
                "Devon wants to send this email from your Gmail.",
                "Review it — sending is final.",
                f"To: {payload['to']}",
                (f"Cc: {payload['cc']}" if payload["cc"] else ""),
                f"Subject: {payload['subject']}",
                "",
                payload["body"],
            ]),
            resume_state={"payload": payload},
        )
        return self._send_now(payload)

    def _send_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = self._build_rfc822(payload)
        data = self._api(
            "POST", "/users/me/messages/send", payload={"raw": raw}
        )
        _log.info(
            "gmail sent message %s to %s",
            data.get("id", "?"), payload.get("to", "?"),
        )
        return {
            "id": data.get("id", ""),
            "thread_id": data.get("threadId", ""),
            "label_ids": data.get("labelIds", []),
        }

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "gmail is not connected — run "
                "`nm connectors connect --name gmail` first"
            )
        return cred

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._google_access_token()}"}

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One Gmail API call; errors become GmailError."""
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=self._headers(), params=params
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=self._headers()
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GmailError(f"gmail request failed: {exc}") from exc
        if resp.status == 401:
            raise GmailError(
                "gmail rejected the access token (401) — the grant was "
                "revoked or expired; reconnect with a fresh code",
                status_code=401,
            )
        if resp.status == 403:
            reason = self._api_reason(resp)
            raise GmailError(
                f"gmail refused (403{', ' + reason if reason else ''}): "
                "the grant lacks this scope — reconnect granting the "
                "missing scope",
                status_code=403,
                reason=reason,
            )
        if resp.status == 429:
            raise GmailError(
                "gmail rate limit exceeded (429) — back off and retry",
                status_code=429,
            )
        if not resp.ok:
            raise GmailError(
                f"gmail {method} {path} failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise GmailError(
                f"gmail {method} {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _api_reason(resp: Any) -> str:
        try:
            body = resp.json()
            errors = (body.get("error") or {}).get("errors", [])
            if errors:
                return str(errors[0].get("reason", ""))
        except Exception:  # noqa: BLE001 - best effort only
            pass
        return ""

    def _profile_email(self, access_token: str) -> str:
        """The connected address (``users.getProfile``)."""
        try:
            resp = self.http.get(
                f"{API_BASE}/users/me/profile",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GmailError(f"gmail profile lookup failed: {exc}") from exc
        if not resp.ok:
            if resp.status == 401:
                raise GmailError(
                    "gmail rejected the access token (401) — the grant was "
                    "revoked or expired; reconnect with a fresh code",
                    status_code=401,
                )
            raise GmailError(
                f"gmail profile lookup failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return str(resp.json().get("emailAddress", ""))
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise GmailError(
                "gmail profile lookup returned invalid JSON"
            ) from exc

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_rfc822(payload: dict[str, Any]) -> str:
        """RFC 822 message, base64url-encoded for messages.send."""
        lines = [
            f"To: {payload['to']}",
            f"Subject: {payload['subject']}",
            f"Date: {email.utils.formatdate(localtime=True)}",
            "MIME-Version: 1.0",
        ]
        if payload.get("cc"):
            lines.append(f"Cc: {payload['cc']}")
        if payload.get("bcc"):
            lines.append(f"Bcc: {payload['bcc']}")
        content_type = (
            'text/html; charset="utf-8"'
            if payload.get("html")
            else 'text/plain; charset="utf-8"'
        )
        lines.append(f"Content-Type: {content_type}")
        lines.append("Content-Transfer-Encoding: 8bit")
        lines.append("")
        lines.append(payload["body"])
        raw = "\r\n".join(lines).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    @staticmethod
    def _code_from_note(note: str) -> str:
        """Pull the authorization code out of a checkpoint note."""
        text = (note or "").strip()
        if not text:
            return ""
        import re

        match = re.search(
            r"code\s*(?:=|:|\bis\b)?\s*[\"']?([A-Za-z0-9_.\-/]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        tokens = text.split()
        if len(tokens) == 1 and tokens[0].lower() != "code":
            return tokens[0].strip("\"'")
        return ""
