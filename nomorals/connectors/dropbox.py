"""Dropbox connector — files over the Dropbox HTTP API.

Docs: https://www.dropbox.com/developers/documentation/http/documentation

Auth: ``AuthMethod.API_KEY`` (a long-lived access token the owner mints in
the Dropbox App Console) or ``AuthMethod.OAUTH2`` (authorization-code flow
with ``token_access_type=offline`` so a refresh token is issued; short-lived
access tokens auto-refresh). ``connect()`` accepts either and validates
via ``users/get_current_account``.

Endpoints used:
* ``POST https://api.dropboxapi.com/2/...`` — RPC (list_folder, delete_v2,
  users/get_current_account)
* ``POST https://content.dropboxapi.com/2/...`` — content (upload, download)
  with the ``Dropbox-API-Arg`` header carrying the JSON argument.

Uploading and deleting are consequential: they run only with
``confirmed=True`` (owner approved the exact path) or on a human
checkpoint when ``db`` is given.

OAuth app setup (human step, guided by ``connect_instructions()``):
1. https://www.dropbox.com/developers/apps/create — "Scoped access",
   choose Full Dropbox or App folder, name the app.
2. Permissions tab: enable ``files.content.read`` and
   ``files.content.write`` (and ``account_info.read`` for the account
   check), then generate a long-lived access token — or use OAuth with
   redirect URI ``https://localhost`` and grab the code.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from ._confirm import confirm_or_checkpoint
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["DropboxConnector", "DropboxError"]

_log = get_logger(__name__)

RPC_BASE = "https://api.dropboxapi.com"
CONTENT_BASE = "https://content.dropboxapi.com"
OAUTH_AUTHORIZE_URL = "https://www.dropbox.com/oauth2/authorize"
OAUTH_TOKEN_URL = f"{RPC_BASE}/oauth2/token"
REDIRECT_URI = "https://localhost"

DROPBOX_CLIENT_ID_ENV = "DROPBOX_CLIENT_ID"
DROPBOX_CLIENT_SECRET_ENV = "DROPBOX_CLIENT_SECRET"

#: Refresh the access token this far ahead of expiry.
_REFRESH_LEEWAY = 60.0


class DropboxError(ConnectorError):
    """A Dropbox API call failed."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class DropboxConnector(Connector):
    """Devon's Dropbox adapter: list, upload, download, delete files."""

    id = "dropbox"
    name = "Dropbox"
    description = (
        "Dropbox files API: list folders, upload/download files, delete "
        "files. Authenticates with a long-lived access token or OAuth2 "
        "(refresh token, auto-refreshed)."
    )
    auth_methods = (AuthMethod.API_KEY, AuthMethod.OAUTH2)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        code: str | None = None,
        redirect_uri: str = "",
    ) -> ConnectResult:
        """Connect with a long-lived token, or finish the OAuth flow.

        * ``token=...`` — validate a long-lived access token and store it.
        * ``code=...`` (+ ``client_id``/``client_secret``) — exchange an
          authorization code for tokens; the refresh token is vaulted.
        * neither — prints the app-setup guide and (for OAuth) the
          authorization URL; the owner completes the grant and calls back
          with the code.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "dropbox is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        if (token or "").strip():
            return self._connect_token((token or "").strip())
        if (code or "").strip():
            return self._connect_code(
                client_id, client_secret, (code or "").strip(),
                (redirect_uri or "").strip() or REDIRECT_URI,
            )
        print(self.connect_instructions())
        cid = self._client_id(client_id)
        url = self.authorize_url(cid, (redirect_uri or "").strip()
                                 or REDIRECT_URI)
        return ConnectResult(
            ok=False,
            message=(
                "grant access in your browser, then call "
                "connect(..., code=<code>) with the code Dropbox shows.\n"
                f"Authorization URL: {url}\n"
                "Already have a long-lived token? connect(token=<token>)."
            ),
        )

    def _connect_token(self, token: str) -> ConnectResult:
        account = self._rpc("/2/users/get_current_account", {},
                            token=token)
        account_id = str(account.get("account_id", ""))
        email = str(account.get("email", ""))
        cred = self._store_credential(
            account_id or "dropbox",
            token,
            credential_type="api_key",
            metadata={"auth": "api_key", "account": email},
        )
        _log.info("dropbox connected (token) as %s", email or account_id)
        return ConnectResult(
            ok=True,
            account=email or account_id,
            message=(
                f"connected to Dropbox as {email or account_id}. The token "
                "is in the encrypted vault."
            ),
            credential_id=cred.id,
        )

    def _connect_code(
        self,
        client_id: str | None,
        client_secret: str | None,
        code: str,
        redirect_uri: str,
    ) -> ConnectResult:
        cid = self._client_id(client_id)
        secret = self._client_secret(client_secret)
        tokens = self._exchange_code(cid, secret, code, redirect_uri)
        account = self._rpc("/2/users/get_current_account", {},
                            token=str(tokens.get("access_token", "")))
        account_id = str(account.get("account_id", ""))
        email = str(account.get("email", ""))
        self._store_oauth_tokens(account_id or "dropbox", cid, tokens,
                                 account=email)
        _log.info("dropbox connected (oauth) as %s", email or account_id)
        return ConnectResult(
            ok=True,
            account=email or account_id,
            scopes=["files.content.read", "files.content.write"],
            message=(
                f"connected to Dropbox as {email or account_id}. The "
                "refresh token is in the encrypted vault; access tokens "
                "auto-refresh."
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
                       "--name dropbox`",
            )
        try:
            account = self._rpc("/2/users/get_current_account", {})
        except DropboxError as exc:
            return ConnectorStatus(
                connected=False,
                account=str((cred.metadata or {}).get("account", "")),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reconnect with a fresh one",
            )
        return ConnectorStatus(
            connected=True,
            account=str(account.get("email", "")
                        or (cred.metadata or {}).get("account", "")),
            last_checked=time.time(),
            detail="token valid",
        )

    def test_connection(self) -> bool:
        if self._load_credential() is None:
            return False
        try:
            self._rpc("/2/users/get_current_account", {})
            return True
        except ConnectorError:
            return False

    def connect_instructions(self) -> str:
        return "\n".join([
            "Dropbox setup (one human step — only you can do this):",
            "1. Create an app: https://www.dropbox.com/developers/apps/create",
            "   - Choose 'Scoped access', then 'Full Dropbox' or 'App folder'.",
            "   - Name it (e.g. 'Devon').",
            "2. Permissions tab: enable files.content.read,",
            "   files.content.write, and account_info.read.",
            "3. Either: generate a long-lived access token and call",
            "   connect(token=<token>) — simplest, or: add",
            f"   '{REDIRECT_URI}' as an OAuth2 redirect URI and use the",
            "   OAuth flow (refresh token, auto-refreshed).",
        ])

    # ── reads ────────────────────────────────────────────────────

    def get_current_account(self) -> dict[str, Any]:
        """The connected account (``users/get_current_account``)."""
        return self._rpc("/2/users/get_current_account", {})

    def list_folder(
        self,
        path: str = "",
        *,
        recursive: bool = False,
        limit: int = 2000,
        cursor: str = "",
    ) -> dict[str, Any]:
        """List a folder (``files/list_folder``).

        ``path`` is ``""`` for the root or a ``/folder/sub`` path. Returns
        ``entries`` (each with ``.tag`` file/folder, name, path_lower,
        size for files), ``cursor``, and ``has_more`` — pass the cursor
        back for the next page.
        """
        if cursor:
            data = self._rpc("/2/files/list_folder/continue",
                             {"cursor": cursor})
        else:
            arg: dict[str, Any] = {
                "path": path or "",
                "recursive": bool(recursive),
                "limit": max(1, min(limit, 2000)),
            }
            data = self._rpc("/2/files/list_folder", arg)
        return {
            "entries": [
                self._summarize_entry(e)
                for e in data.get("entries", [])
            ],
            "cursor": data.get("cursor", ""),
            "has_more": bool(data.get("has_more", False)),
        }

    @staticmethod
    def _summarize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "tag": entry.get(".tag", ""),
            "name": entry.get("name", ""),
            "path": entry.get("path_lower", ""),
            "size": entry.get("size", 0),
            "modified": entry.get("client_modified", ""),
            "id": entry.get("id", ""),
        }

    def get_metadata(self, path: str) -> dict[str, Any]:
        """Metadata for a file or folder (``files/get_metadata``)."""
        if not (path or "").strip():
            raise ConnectorError("empty path")
        return self._rpc("/2/files/get_metadata", {"path": path})

    def download(
        self,
        path: str,
        dest: str | Path | None = None,
    ) -> dict[str, Any]:
        """Download a file (``files/download``).

        Returns the bytes in ``data`` (plus ``metadata`` from the
        ``Dropbox-API-Result`` header) when ``dest`` is omitted; otherwise
        writes to ``dest``.
        """
        if not (path or "").strip():
            raise ConnectorError("empty path")
        resp = self._content("/2/files/download", {"path": path})
        meta: dict[str, Any] = {}
        try:
            meta = json.loads(resp.headers.get("dropbox-api-result", "{}"))
        except Exception as exc:  # noqa: BLE001 - header is best-effort
            _log.debug("dropbox could not parse metadata header: %s", exc)
        data = resp.body
        if dest is not None:
            out = Path(dest)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            _log.info("dropbox download ok: %s -> %s (%d bytes)",
                      path, out, len(data))
            return {"path": path, "dest": str(out), "size": len(data),
                    "metadata": meta}
        return {"path": path, "size": len(data), "data": data,
                "metadata": meta}

    # ── writes (confirmation-gated) ──────────────────────────────

    def upload(
        self,
        local_path: str | Path,
        dropbox_path: str,
        *,
        mode: str = "add",
        autorename: bool = True,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Upload a file (``files/upload``).

        ``mode``: ``add`` (fail if a file exists there — use with
        ``autorename=True``), ``overwrite``, or ``update`` (needs rev —
        not supported here, use overwrite). The whole file is read into
        memory; for files over ~150 MB use the Dropbox desktop client or
        an upload session (documented gap: ``upload_session`` is not
        implemented).

        Uploading writes into the owner's Dropbox, so it never runs on
        implied consent: ``confirmed=True`` after the owner approved the
        exact paths, or ``db`` to park the payload on a human checkpoint.
        """
        src = Path(local_path)
        if not src.is_file():
            raise ConnectorError(f"no such file: {src}")
        if mode not in ("add", "overwrite"):
            raise ConnectorError(
                f"invalid upload mode {mode!r}: use 'add' or 'overwrite'"
            )
        dropbox_path = (dropbox_path or "").strip()
        if not dropbox_path.startswith("/"):
            raise ConnectorError(
                f"dropbox path must start with '/': {dropbox_path!r}"
            )
        data = src.read_bytes()
        if len(data) > 150 * 1024 * 1024:
            raise ConnectorError(
                f"{src} is {len(data)} bytes — over the ~150 MB single-shot "
                "limit; upload sessions are not implemented, use the "
                "Dropbox desktop client for this file"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="upload_file",
            title=f"Upload to Dropbox {dropbox_path}",
            instructions="\n".join([
                "Devon wants to upload this file to your Dropbox.",
                f"Local: {src} ({len(data)} bytes)",
                f"Dropbox: {dropbox_path} (mode={mode}, "
                f"autorename={autorename})",
            ]),
            resume_state={
                "local": str(src), "path": dropbox_path, "mode": mode,
                "size": len(data),
            },
        )
        arg = {
            "path": dropbox_path,
            "mode": mode,
            "autorename": bool(autorename),
            "mute": False,
            "strict_conflict": False,
        }
        resp = self._content("/2/files/upload", arg, data=data)
        try:
            result = json.loads(resp.headers.get("dropbox-api-result", "{}"))
        except Exception as exc:  # noqa: BLE001 - header is best-effort
            raise DropboxError(
                "dropbox upload succeeded but the metadata header was "
                f"unreadable: {exc}"
            ) from exc
        _log.info("dropbox upload ok: %s -> %s", src, dropbox_path)
        return {
            "name": result.get("name", ""),
            "path": result.get("path_lower", dropbox_path),
            "size": result.get("size", len(data)),
            "id": result.get("id", ""),
        }

    def delete(
        self,
        path: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Delete a file or folder (``files/delete_v2``).

        Deletion is final in Dropbox (recoverable from the website's
        deleted-files view for 30 days, not via this API), so it needs
        ``confirmed=True`` or a human checkpoint via ``db``.
        """
        path = (path or "").strip()
        if not path:
            raise ConnectorError("empty path")
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="delete_path",
            title=f"Delete from Dropbox {path}",
            instructions="\n".join([
                "Devon wants to delete this from your Dropbox.",
                f"Path: {path}",
                "Deletion is final via the API (the Dropbox website keeps",
                "deleted files for 30 days).",
            ]),
            resume_state={"path": path},
        )
        data = self._rpc("/2/files/delete_v2", {"path": path})
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        _log.info("dropbox delete ok: %s", path)
        return {"deleted": path, "metadata": metadata}

    # ── OAuth plumbing ───────────────────────────────────────────

    def _client_id(self, client_id: str | None) -> str:
        cid = ((client_id or "").strip()
               or os.environ.get(DROPBOX_CLIENT_ID_ENV, "").strip())
        if not cid:
            raise ConnectorError(
                "no Dropbox app key — create an app at "
                "https://www.dropbox.com/developers/apps/create and pass "
                f"client_id=... or set {DROPBOX_CLIENT_ID_ENV}"
            )
        return cid

    def _client_secret(self, client_secret: str | None) -> str:
        return ((client_secret or "").strip() or prompt_secret(
            "Dropbox app secret", env_var=DROPBOX_CLIENT_SECRET_ENV
        ))

    def authorize_url(self, client_id: str, redirect_uri: str) -> str:
        """The URL the owner opens to grant access."""
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "token_access_type": "offline",  # ask for a refresh token
        }
        return f"{OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    def _exchange_code(
        self,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Swap an authorization code for access + refresh tokens."""
        try:
            resp = self.http.post_form(
                OAUTH_TOKEN_URL,
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DropboxError(
                f"dropbox token exchange failed: {exc}"
            ) from exc
        if not resp.ok:
            raise DropboxError(
                f"dropbox rejected the authorization code ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise DropboxError(
                "dropbox token exchange returned an unexpected response"
            )
        return data

    def _refresh(
        self, client_id: str, client_secret: str, refresh_token: str
    ) -> dict[str, Any]:
        try:
            resp = self.http.post_form(
                OAUTH_TOKEN_URL,
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DropboxError(
                f"dropbox token refresh failed: {exc}"
            ) from exc
        if not resp.ok:
            raise DropboxError(
                f"dropbox rejected the refresh token ({resp.status}): "
                f"{resp.text[:200]} — reconnect with a fresh code",
                status_code=resp.status,
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise DropboxError(
                "dropbox token refresh returned an unexpected response"
            )
        return data

    def _store_oauth_tokens(
        self,
        username: str,
        client_id: str,
        tokens: dict[str, Any],
        *,
        account: str = "",
    ) -> None:
        refresh = str(tokens.get("refresh_token", ""))
        access = str(tokens.get("access_token", ""))
        if not refresh:
            raise ConnectorError(
                "dropbox did not return a refresh token — authorize with "
                "token_access_type=offline (already set) and use a fresh "
                "code"
            )
        if not access:
            raise ConnectorError(
                "dropbox did not return an access token on exchange"
            )
        expires_in = float(tokens.get("expires_in", 14400) or 14400)
        self._store_credential(
            username,
            refresh,
            credential_type="oauth_token",
            scopes=["files.content.read", "files.content.write"],
            metadata={
                "auth": "oauth2",
                "client_id": client_id,
                "account": account,
                "access_token": access,
                "access_expires_at": time.time() + expires_in,
            },
        )

    def _access_token(self) -> str:
        """A fresh access token: direct for API keys, auto-refresh for OAuth."""
        cred = self._require_credential()
        meta = cred.metadata or {}
        if str(meta.get("auth", "")) == "api_key":
            return cred.password
        client_id = str(meta.get("client_id", ""))
        access = str(meta.get("access_token", ""))
        expires_at = float(meta.get("access_expires_at", 0) or 0)
        if access and client_id and expires_at - time.time() > _REFRESH_LEEWAY:
            return access
        secret = self._client_secret(None)
        tokens = self._refresh(client_id, secret, cred.password)
        new_access = str(tokens.get("access_token", ""))
        if not new_access:
            raise ConnectorError(
                "dropbox did not return an access token on refresh"
            )
        merged = dict(tokens)
        merged.setdefault("refresh_token", cred.password)
        self._store_oauth_tokens(
            cred.username, client_id, merged,
            account=str(meta.get("account", "")),
        )
        _log.info("dropbox: access token refreshed")
        return new_access

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "dropbox is not connected — run "
                "`nm connectors connect --name dropbox` first"
            )
        return cred

    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def _fail(self, op: str, resp: Any) -> DropboxError:
        status = resp.status
        summary = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                summary = str(body.get("error_summary", ""))
        except Exception:  # noqa: BLE001 - best effort only
            pass
        if status == 401:
            return DropboxError(
                "dropbox rejected the token (401) — it is invalid, expired, "
                "or the app permissions changed; reconnect",
                status_code=401,
            )
        if status == 429:
            retry = resp.headers.get("retry-after", "")
            return DropboxError(
                "dropbox rate limit exceeded (429)"
                + (f" — retry after {retry}s" if retry else "")
                + " — back off and retry",
                status_code=429,
            )
        if status == 409:
            return DropboxError(
                f"dropbox {op} failed (409): {summary or resp.text[:200]}",
                status_code=409,
            )
        return DropboxError(
            f"dropbox {op} failed ({status}): "
            f"{summary or resp.text[:200]}",
            status_code=status,
        )

    def _rpc(
        self,
        path: str,
        arg: dict[str, Any],
        *,
        token: str | None = None,
    ) -> dict[str, Any]:
        """One RPC call to api.dropboxapi.com."""
        bearer = token or self._access_token()
        try:
            resp = self.http.post_json(
                f"{RPC_BASE}{path}", arg, headers=self._headers(bearer)
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DropboxError(f"dropbox request failed: {exc}") from exc
        if not resp.ok:
            raise self._fail(f"POST {path}", resp)
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise DropboxError(
                f"dropbox POST {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}

    def _content(
        self,
        path: str,
        arg: dict[str, Any],
        *,
        data: bytes | None = None,
    ) -> Any:
        """One content call to content.dropboxapi.com.

        ``data`` set → upload body; unset → download (empty body). Returns
        the raw response so callers can read the ``Dropbox-API-Result``
        header and the raw bytes.
        """
        bearer = self._access_token()
        headers = {
            **self._headers(bearer),
            "Dropbox-API-Arg": json.dumps(arg),
        }
        try:
            if data is None:
                resp = self.http.request(
                    "POST", f"{CONTENT_BASE}{path}",
                    data=b"", headers=headers,
                )
            else:
                headers["Content-Type"] = "application/octet-stream"
                resp = self.http.request(
                    "POST", f"{CONTENT_BASE}{path}",
                    data=data, headers=headers,
                )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DropboxError(f"dropbox request failed: {exc}") from exc
        if not resp.ok:
            raise self._fail(f"POST {path}", resp)
        return resp
