"""YouTube connector — channel info, video listing, uploads.

Docs: https://developers.google.com/youtube/v3/docs

Auth: OAuth 2.0 (``AuthMethod.OAUTH2``) via the shared Google
authorization-code flow in ``_google_oauth`` — the same pattern as
``gmail.py``. Scopes: ``youtube.readonly`` + ``youtube.upload``. The
refresh token is vaulted; access tokens auto-refresh.

Quota note: the YouTube Data API bills quota units per call (a ``search``
costs 100 units out of the default 10,000/day; ``playlistItems.list`` and
``channels.list`` cost 1). ``list_videos`` therefore reads the channel's
uploads playlist (1 unit) instead of searching.

Uploads use the direct multipart path (``uploadType=multipart``): the file
is read into memory and sent in one request. Honest limits of this path:

* files over 256 MB are refused — use the resumable upload session flow
  for anything bigger (documented gap: not implemented here);
* the whole file sits in memory during the upload;
* quota cost of ``videos.insert`` is ~1600 units, so roughly six uploads
  fit in a fresh default daily quota.

Uploading publishes to the owner's channel, so it is
confirmation-gated: ``confirmed=True`` after the owner approved the exact
file/title/description, or ``db`` to park the payload on a human
checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
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

__all__ = ["YouTubeConnector", "YouTubeError"]

_log = get_logger(__name__)

API_BASE = "https://www.googleapis.com"
UPLOAD_BASE = "https://www.googleapis.com/upload/youtube/v3"

_SCOPE_READONLY = "https://www.googleapis.com/auth/youtube.readonly"
_SCOPE_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"

#: Direct multipart uploads are refused past this — resumable sessions are
#: the right tool for bigger files (not implemented; documented gap).
_MULTIPART_MAX_BYTES = 256 * 1024 * 1024

_PRIVACY = ("private", "unlisted", "public")


class YouTubeError(ConnectorError):
    """A YouTube Data API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, reason: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


@register_connector
class YouTubeConnector(Connector, GoogleOAuth):
    """Devon's YouTube adapter: channel, videos, uploads."""

    id = "youtube"
    name = "YouTube"
    description = (
        "YouTube Data API: channel info, video listings, and video "
        "uploads (direct multipart, 256 MB cap). OAuth 2.0 "
        "(readonly + upload scopes); uploads require explicit owner "
        "confirmation."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def _google_default_scopes(self) -> list[str]:
        return [_SCOPE_READONLY, _SCOPE_UPLOAD]

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

        Same flow as the Gmail connector: without ``code`` it prints the
        grant guide + authorization URL; with ``db`` the grant pauses at a
        human checkpoint; with ``code`` the tokens are exchanged and
        stored. The YouTube Data API must be enabled on the Google Cloud
        project first.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "youtube is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        cid = self._google_client_id(client_id)
        wanted = list(scopes or self._google_default_scopes())
        if code:
            secret = self._google_client_secret(client_secret)
            tokens = self._google_exchange_code(cid, secret, code)
            channel = self._my_channel(tokens.get("access_token", ""))
            title = str(
                ((channel.get("snippet") or {}).get("title")) or "youtube"
            )
            self._store_google_tokens(
                title, cid, tokens, account=title, scopes=wanted,
            )
            return ConnectResult(
                ok=True,
                account=title,
                scopes=wanted,
                message=(
                    f"connected to YouTube as {title}. Tokens are in the "
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
                    "enable the YouTube Data API v3 on the Cloud project, "
                    "grant access in your browser, then call "
                    "connect(..., code=<code>) with the code Google shows"
                ),
            )
        try:
            self.request_human(
                CheckpointKind.MANUAL_STEP,
                "Grant YouTube access",
                guide
                + "\n\nEnable the YouTube Data API v3 on the Cloud project "
                "first. When Google shows the authorization code, resolve "
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
                       "--name youtube`",
            )
        try:
            channel = self._my_channel(self._google_access_token())
        except YouTubeError as exc:
            return ConnectorStatus(
                connected=False,
                account=str((cred.metadata or {}).get("account", "youtube")),
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reconnect with a fresh code",
            )
        return ConnectorStatus(
            connected=True,
            account=str((channel.get("snippet") or {}).get("title", "")
                        or (cred.metadata or {}).get("account", "")),
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="oauth tokens valid",
        )

    def test_connection(self) -> bool:
        if self._load_credential() is None:
            return False
        try:
            self._my_channel(self._google_access_token())
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
            channel = self._my_channel(tokens["access_token"])
            title = str(
                ((channel.get("snippet") or {}).get("title")) or "youtube"
            )
            self._store_google_tokens(
                title, cid, tokens, account=title, scopes=scopes,
            )
            return {"connected": True, "account": title}
        if stage == "upload_video":
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("file"):
                raise ConnectorError(
                    "the resolved checkpoint has no upload payload — "
                    "it cannot upload"
                )
            return self._upload_now(payload)
        raise ConnectorError(
            f"youtube cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def get_channel(self) -> dict[str, Any]:
        """The connected channel (``channels.list?mine=true``)."""
        return self._my_channel(self._google_access_token())

    def list_videos(
        self,
        *,
        max_results: int = 25,
        page_token: str = "",
    ) -> dict[str, Any]:
        """The channel's uploaded videos (uploads playlist, 1 quota unit).

        Reads the channel's uploads playlist via ``playlistItems.list``
        instead of ``search.list`` (100 quota units) — same videos,
        far cheaper.
        """
        channel = self._my_channel(self._google_access_token())
        uploads = (
            ((channel.get("contentDetails") or {})
             .get("relatedPlaylists") or {}).get("uploads", "")
        )
        if not uploads:
            raise YouTubeError(
                "the channel has no uploads playlist — it may not be a "
                "full YouTube channel"
            )
        data = self._api(
            "GET",
            "/youtube/v3/playlistItems",
            params={
                "part": "snippet,contentDetails",
                "playlistId": uploads,
                "maxResults": max(1, min(max_results, 50)),
                **({"pageToken": page_token} if page_token else {}),
            },
        )
        return {
            "videos": [
                self._summarize_playlist_item(item)
                for item in data.get("items", [])
            ],
            "next_page_token": data.get("nextPageToken", ""),
            "total": data.get("pageInfo", {}).get("totalResults", 0),
        }

    @staticmethod
    def _summarize_playlist_item(item: dict[str, Any]) -> dict[str, Any]:
        snippet = item.get("snippet", {}) or {}
        content = item.get("contentDetails", {}) or {}
        return {
            "video_id": content.get("videoId", ""),
            "title": snippet.get("title", ""),
            "description": (snippet.get("description", "") or "")[:200],
            "published_at": snippet.get("publishedAt", ""),
            "thumbnail": (
                ((snippet.get("thumbnails") or {}).get("default") or {})
                .get("url", "")
            ),
        }

    def search_videos(
        self,
        query: str,
        *,
        max_results: int = 10,
        order: str = "relevance",
    ) -> list[dict[str, Any]]:
        """Search YouTube (``search.list`` — costs 100 quota units/call).

        Expensive on quota; prefer ``list_videos`` for the owner's own
        uploads.
        """
        if not (query or "").strip():
            raise ConnectorError("empty search query")
        if order not in ("relevance", "date", "rating", "title", "viewCount"):
            raise ConnectorError(
                f"invalid order {order!r}: relevance, date, rating, title, "
                "viewCount"
            )
        data = self._api(
            "GET",
            "/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": order,
                "maxResults": max(1, min(max_results, 50)),
            },
        )
        out = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {}) or {}
            out.append({
                "video_id": ((item.get("id") or {}).get("videoId", "")),
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
            })
        return out

    # ── upload (confirmation-gated) ──────────────────────────────

    def upload_video(
        self,
        file_path: str | Path,
        title: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        privacy: str = "private",
        category_id: str = "22",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Upload a video (direct multipart ``videos.insert``).

        Publishing to the owner's channel is consequential: pass
        ``confirmed=True`` only after the owner approved the exact file,
        title, description, and privacy — or ``db`` to park the payload on
        a human checkpoint.

        Limits (honest): files over 256 MB are refused — resumable upload
        sessions are not implemented yet. ``videos.insert`` costs ~1600
        quota units.
        """
        path = Path(file_path)
        if not path.is_file():
            raise ConnectorError(f"no such file: {path}")
        title = (title or "").strip()
        if not title:
            raise ConnectorError("a video title is required")
        if privacy not in _PRIVACY:
            raise ConnectorError(
                f"invalid privacy {privacy!r}: use "
                + ", ".join(_PRIVACY)
            )
        size = path.stat().st_size
        if size > _MULTIPART_MAX_BYTES:
            raise ConnectorError(
                f"{path} is {size} bytes — over the 256 MB direct-upload "
                "limit; resumable upload sessions are not implemented, "
                "upload this file from YouTube Studio instead"
            )
        payload = {
            "file": str(path),
            "title": title,
            "description": description or "",
            "tags": list(tags or []),
            "privacy": privacy,
            "category_id": category_id,
            "size": size,
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="upload_video",
            title=f"Upload video '{title}' to YouTube",
            instructions="\n".join([
                "Devon wants to publish this video to your YouTube channel.",
                f"File: {path} ({size} bytes)",
                f"Title: {title}",
                f"Privacy: {privacy}",
                "Description:",
                (description or "(none)")[:500],
            ]),
            resume_state={"payload": payload},
        )
        return self._upload_now(payload)

    def _upload_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(payload["file"]))
        data = path.read_bytes()
        metadata = {
            "snippet": {
                "title": payload["title"],
                "description": payload.get("description", ""),
                "tags": payload.get("tags", []),
                "categoryId": payload.get("category_id", "22"),
            },
            "status": {"privacyStatus": payload.get("privacy", "private")},
        }
        boundary = f"----nm-yt-{int(time.time() * 1000)}"
        body = b"\r\n".join([
            f"--{boundary}".encode(),
            b"Content-Type: application/json; charset=UTF-8",
            b"",
            json.dumps(metadata).encode("utf-8"),
            f"--{boundary}".encode(),
            b"Content-Type: application/octet-stream",
            b"",
            data,
            f"--{boundary}--".encode(),
            b"",
        ])
        url = (f"{UPLOAD_BASE}/videos?uploadType=multipart"
               "&part=snippet,status")
        try:
            resp = self.http.request(
                "POST",
                url,
                data=body,
                headers={
                    "Authorization":
                        f"Bearer {self._google_access_token()}",
                    "Content-Type":
                        f"multipart/related; boundary={boundary}",
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise YouTubeError(f"youtube upload failed: {exc}") from exc
        if not resp.ok:
            reason = self._api_reason(resp)
            raise YouTubeError(
                f"youtube upload failed ({resp.status}"
                f"{', ' + reason if reason else ''}): {resp.text[:300]}",
                status_code=resp.status,
                reason=reason,
            )
        try:
            result = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise YouTubeError(
                "youtube upload returned invalid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise YouTubeError("youtube upload returned an unexpected response")
        _log.info("youtube upload ok: %s (%s)",
                  result.get("id", "?"), payload.get("title", "?"))
        return {
            "video_id": result.get("id", ""),
            "title": ((result.get("snippet") or {}).get("title", "")),
            "privacy": ((result.get("status") or {})
                        .get("privacyStatus", "")),
        }

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "youtube is not connected — run "
                "`nm connectors connect --name youtube` first"
            )
        return cred

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._google_access_token()}"}

    def _api(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One YouTube Data API call; errors become YouTubeError."""
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=self._headers(), params=params
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise YouTubeError(f"youtube request failed: {exc}") from exc
        if resp.status == 401:
            raise YouTubeError(
                "youtube rejected the access token (401) — the grant was "
                "revoked or expired; reconnect with a fresh code",
                status_code=401,
            )
        if resp.status == 403:
            reason = self._api_reason(resp)
            raise YouTubeError(
                f"youtube refused (403{', ' + reason if reason else ''}): "
                "quota exhausted or the grant lacks the scope — wait for "
                "the quota reset or reconnect granting the missing scope",
                status_code=403,
                reason=reason,
            )
        if not resp.ok:
            raise YouTubeError(
                f"youtube {method} {path} failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise YouTubeError(
                f"youtube {method} {path} returned invalid JSON"
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

    def _my_channel(self, access_token: str) -> dict[str, Any]:
        """The connected channel (``channels.list?mine=true``)."""
        try:
            resp = self.http.get(
                f"{API_BASE}/youtube/v3/channels",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"part": "snippet,contentDetails", "mine": "true"},
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise YouTubeError(
                f"youtube channel lookup failed: {exc}"
            ) from exc
        if resp.status == 401:
            raise YouTubeError(
                "youtube rejected the access token (401) — the grant was "
                "revoked or expired; reconnect with a fresh code",
                status_code=401,
            )
        if not resp.ok:
            raise YouTubeError(
                f"youtube channel lookup failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise YouTubeError(
                "youtube channel lookup returned invalid JSON"
            ) from exc
        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            raise YouTubeError(
                "no YouTube channel on this Google account — create one "
                "at https://www.youtube.com first"
            )
        return items[0]

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
