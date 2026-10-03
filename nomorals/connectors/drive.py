"""Google Drive connector — the owner's files over the Drive API v3.

Docs: https://developers.google.com/drive/api/reference/rest/v3

Auth: OAuth 2.0 (``AuthMethod.OAUTH2``) via the shared Google
authorization-code flow in ``_google_oauth`` — default scope
``drive.file`` (only files Devon creates or the owner explicitly opens
with the app), widening to full ``drive`` only on the owner's explicit
choice. The refresh token is vaulted; access tokens auto-refresh.

Uploads use ``uploadType=multipart`` (metadata + content in one request);
downloads stream ``files/{id}?alt=media`` straight to disk.
"""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
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

__all__ = ["DriveConnector", "DriveError"]

_log = get_logger(__name__)

API_BASE = "https://www.googleapis.com/drive/v3"
UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

_SCOPE_FILE = "https://www.googleapis.com/auth/drive.file"
_SCOPE_FULL = "https://www.googleapis.com/auth/drive"
_SCOPE_READONLY = "https://www.googleapis.com/auth/drive.readonly"

_DEFAULT_FIELDS = (
    "id,name,mimeType,size,modifiedTime,parents,trashed,webViewLink"
)


class DriveError(ConnectorError):
    """A Drive API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, reason: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


@register_connector
class DriveConnector(Connector, GoogleOAuth):
    """Devon's Drive adapter: list, upload, download, delete files."""

    id = "gdrive"
    name = "Google Drive"
    description = (
        "Drive API v3: list/search files, upload, download, and delete. "
        "OAuth 2.0; default scope drive.file (least privilege), full "
        "drive only on explicit owner choice."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def _google_default_scopes(self) -> list[str]:
        return [_SCOPE_FILE]

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
        ``db`` the flow pauses at a human checkpoint; without it, the owner
        hands the code back and calls ``connect(..., code=<code>)`` again.
        Pass ``scopes=[...]`` to widen from the default ``drive.file``
        (e.g. full ``drive``) — the owner sees exactly what they grant.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "drive is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        cid = self._google_client_id(client_id)
        wanted = list(scopes or self._google_default_scopes())
        if code:
            secret = self._google_client_secret(client_secret)
            tokens = self._google_exchange_code(cid, secret, code)
            account = self._about_email(tokens.get("access_token", ""))
            self._store_google_tokens(
                account or "gdrive", cid, tokens,
                account=account, scopes=wanted,
            )
            return ConnectResult(
                ok=True,
                account=account or "gdrive",
                scopes=wanted,
                message=(
                    f"connected to Drive as {account or 'the granted account'} "
                    f"(scopes: {', '.join(wanted)}). Tokens are in the "
                    "encrypted vault."
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
                "Grant Drive access",
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
                       "--name gdrive`",
            )
        try:
            email = self._about_email(self._google_access_token())
        except DriveError as exc:
            return ConnectorStatus(
                connected=False,
                account=str((cred.metadata or {}).get("account", "gdrive")),
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
            self._about_email(self._google_access_token())
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
        """Continue after the owner resolved the grant checkpoint."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — finish the human step first"
            )
        if (checkpoint.resume_state or {}).get("stage") != "oauth_code":
            raise ConnectorError(
                "drive cannot resume checkpoint stage "
                f"{(checkpoint.resume_state or {}).get('stage')!r}"
            )
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
        account = self._about_email(tokens["access_token"])
        self._store_google_tokens(
            account or "gdrive", cid, tokens,
            account=account, scopes=scopes,
        )
        return {"connected": True, "account": account}

    # ── files ────────────────────────────────────────────────────

    def list_files(
        self,
        *,
        query: str = "",
        page_size: int = 50,
        page_token: str = "",
        order_by: str = "modifiedTime desc",
        fields: str = _DEFAULT_FIELDS,
    ) -> dict[str, Any]:
        """List files (``files.list``).

        ``query`` is Drive search syntax, e.g.
        ``name contains 'report' and trashed = false``.
        """
        params: dict[str, Any] = {
            "pageSize": max(1, min(page_size, 1000)),
            "orderBy": order_by,
            "fields": f"nextPageToken, files({fields})",
        }
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token
        data = self._api("GET", "/files", params=params)
        return {
            "files": data.get("files", []),
            "next_page_token": data.get("nextPageToken", ""),
        }

    def get_file(self, file_id: str) -> dict[str, Any]:
        """File metadata (``files.get``)."""
        if not (file_id or "").strip():
            raise ConnectorError("empty file id")
        return self._api(
            "GET", f"/files/{file_id}",
            params={"fields": _DEFAULT_FIELDS},
        )

    def upload(
        self,
        file_path: str,
        *,
        name: str = "",
        mime_type: str = "",
        folder_id: str = "",
        description: str = "",
    ) -> dict[str, Any]:
        """Upload a file (multipart ``files.create``).

        ``file_path`` is local; ``name`` defaults to the local filename.
        Returns the created file's metadata.
        """
        src = Path(file_path).expanduser()
        if not src.is_file():
            raise ConnectorError(
                f"cannot upload {file_path!r}: not a file"
            )
        metadata: dict[str, Any] = {"name": name or src.name}
        if folder_id:
            metadata["parents"] = [folder_id]
        if description:
            metadata["description"] = description
        mime = (
            mime_type
            or mimetypes.guess_type(src.name)[0]
            or "application/octet-stream"
        )
        body = self._multipart_body(metadata, src, mime)
        url = f"{UPLOAD_BASE}/files?uploadType=multipart"
        try:
            resp = self.http.request(
                "POST",
                url,
                data=body[0],
                headers={
                    "Authorization": self._bearer(),
                    "Content-Type": body[1],
                },
            )
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DriveError(f"drive upload failed: {exc}") from exc
        data = self._api_result("POST", "/files", resp)
        _log.info(
            "drive uploaded %s as %s (%s)",
            src.name, data.get("id", "?"), data.get("name", "?"),
        )
        return data

    def download(self, file_id: str, dest_path: str) -> dict[str, Any]:
        """Download file content (``files.get?alt=media``) to ``dest_path``.

        Streams to disk (no full read into memory); resumes partial files.
        Google-native files (Docs/Sheets/...) cannot be downloaded this way
        — use :meth:`export_file` for those.
        """
        if not (file_id or "").strip():
            raise ConnectorError("empty file id")
        dest = Path(dest_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{API_BASE}/files/{file_id}"
        try:
            self.http.get(
                url,
                headers={"Authorization": self._bearer()},
                params={"alt": "media"},
                stream_to=dest,
            )
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DriveError(f"drive download failed: {exc}") from exc
        size = dest.stat().st_size if dest.exists() else 0
        _log.info("drive downloaded %s -> %s (%d bytes)",
                  file_id, dest, size)
        return {"file_id": file_id, "path": str(dest), "bytes": size}

    def export_file(
        self, file_id: str, dest_path: str, *, mime_type: str
    ) -> dict[str, Any]:
        """Export a Google-native file (``files.export``).

        e.g. a Google Doc to ``application/pdf`` or
        ``application/vnd.openxmlformats-officedocument.wordprocessingml.document``.
        """
        if not (file_id or "").strip():
            raise ConnectorError("empty file id")
        if not mime_type:
            raise ConnectorError(
                "export needs a target mime_type (e.g. application/pdf)"
            )
        dest = Path(dest_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.http.get(
                f"{API_BASE}/files/{file_id}/export",
                headers={"Authorization": self._bearer()},
                params={"mimeType": mime_type},
                stream_to=dest,
            )
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DriveError(f"drive export failed: {exc}") from exc
        size = dest.stat().st_size if dest.exists() else 0
        return {"file_id": file_id, "path": str(dest), "bytes": size}

    def delete(self, file_id: str) -> bool:
        """Delete a file (``files.delete``) — permanent, not trash."""
        if not (file_id or "").strip():
            raise ConnectorError("empty file id")
        self._api("DELETE", f"/files/{file_id}")
        _log.info("drive deleted %s", file_id)
        return True

    def trash(self, file_id: str) -> dict[str, Any]:
        """Move a file to trash (recoverable ``files.update``)."""
        if not (file_id or "").strip():
            raise ConnectorError("empty file id")
        return self._api(
            "PATCH", f"/files/{file_id}", payload={"trashed": True}
        )

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "drive is not connected — run "
                "`nm connectors connect --name gdrive` first"
            )
        return cred

    def _bearer(self) -> str:
        return f"Bearer {self._google_access_token()}"

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One Drive API call; errors become DriveError."""
        url = f"{API_BASE}{path}"
        headers = {"Authorization": self._bearer()}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            elif method == "PATCH":
                resp = self.http.request(
                    "PATCH",
                    url,
                    headers={**headers,
                             "Content-Type": "application/json"},
                    params=params,
                    data=_json_bytes(payload or {}),
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
            raise DriveError(f"drive request failed: {exc}") from exc
        return self._api_result(method, path, resp)

    def _api_result(
        self, method: str, path: str, resp: Any
    ) -> dict[str, Any]:
        if resp.status == 401:
            raise DriveError(
                "drive rejected the access token (401) — the grant was "
                "revoked or expired; reconnect with a fresh code",
                status_code=401,
            )
        if resp.status == 403:
            reason = self._api_reason(resp)
            raise DriveError(
                f"drive refused (403{', ' + reason if reason else ''}): "
                "the grant lacks this scope — reconnect granting it "
                "(full 'drive' scope for files Devon didn't create)",
                status_code=403,
                reason=reason,
            )
        if resp.status == 404:
            raise DriveError(
                f"drive {method} {path}: not found (404) — bad id, or the "
                "file isn't shared with the connected account",
                status_code=404,
            )
        if resp.status == 429:
            raise DriveError(
                "drive rate limit exceeded (429) — back off and retry",
                status_code=429,
            )
        if not resp.ok:
            raise DriveError(
                f"drive {method} {path} failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        if resp.status == 204:
            return {}
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise DriveError(
                f"drive {method} {path} returned invalid JSON"
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

    def _about_email(self, access_token: str) -> str:
        """The connected address (``about.get``)."""
        try:
            resp = self.http.get(
                f"{API_BASE}/about",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"fields": "user(emailAddress)"},
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DriveError(f"drive about lookup failed: {exc}") from exc
        if not resp.ok:
            raise DriveError(
                f"drive about lookup failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return str(resp.json().get("user", {}).get("emailAddress", ""))
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise DriveError(
                "drive about lookup returned invalid JSON"
            ) from exc

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _multipart_body(
        metadata: dict[str, Any], src: Path, mime: str
    ) -> tuple[bytes, str]:
        """Multipart/related body with a JSON metadata part + file part."""
        import json
        import secrets

        boundary = f"==============={secrets.token_hex(16)}=="
        meta_json = json.dumps(metadata).encode("utf-8")
        chunks = [
            f"--{boundary}\r\n".encode(),
            b'Content-Type: application/json; charset="UTF-8"\r\n\r\n',
            meta_json + b"\r\n",
            f"--{boundary}\r\n".encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            src.read_bytes() + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        return b"".join(chunks), f'multipart/related; boundary="{boundary}"'

    @staticmethod
    def _code_from_note(note: str) -> str:
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


def _json_bytes(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload).encode("utf-8")
