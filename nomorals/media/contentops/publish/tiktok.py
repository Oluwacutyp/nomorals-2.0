"""TikTok publisher — Direct Post via the official Content Posting API.

What works headless TODAY (verified 2026-10-09 against TikTok's
developer docs and production integrations):

* Full OAuth 2.0 flow (``authorize_url`` → owner grants → ``exchange_code``)
  with scopes ``video.upload`` + ``video.publish``; refresh tokens for
  the 24h access tokens.
* ``POST /v2/post/publish/creator_info/query/`` — the account's allowed
  privacy levels, max durations, comment/duet/stitch toggles. Called
  first, always; the privacy level is picked from what TikTok returns.
* Direct Post: ``/v2/post/publish/video/init/`` (``FILE_UPLOAD`` source)
  → chunked PUTs to the returned ``upload_url`` → ``/v2/post/publish/
  video/publish/``. Implemented here, tested against mocks.

The honest limits (this is where other tools lie by omission):

* UNAUDITED APPS POST PRIVATE-ONLY. Until your TikTok developer app
  passes TikTok's Content Posting API audit, every Direct Post is
  forced to ``SELF_ONLY`` regardless of what you request, and the app
  is limited to 5 posting users per 24h. The API answers with
  ``unaudited_client_can_only_post_to_private_accounts`` — this module
  surfaces that as :class:`CapabilityUnavailable` with the audit step,
  never as a fake public post.
* ``PULL_FROM_URL`` needs the video's domain verified in the dev portal;
  this module uses ``FILE_UPLOAD`` (chunked binary) so no public URL is
  needed.
* The app itself needs TikTok's approval (1–3 days) before OAuth works
  at all — a manual step, documented in :meth:`auth_guide`.

Docs: https://developers.tiktok.com/doc/content-posting-api-get-started
"""

from __future__ import annotations

import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from ....core.http import HttpClient
from ....core.logging_setup import get_logger
from . import CapabilityUnavailable, adapt_description
from .ledger import PublishLedger

__all__ = ["TikTokPublisher", "TIKTOK_SCOPES"]

_log = get_logger(__name__)

API_BASE = "https://open.tiktokapis.com/v2"
AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = f"{API_BASE}/oauth/token/"

TIKTOK_SCOPES = ("video.upload", "video.publish")

CLIENT_KEY_ENV = "TIKTOK_CLIENT_KEY"
CLIENT_SECRET_ENV = "TIKTOK_CLIENT_SECRET"
ACCESS_TOKEN_ENV = "TIKTOK_ACCESS_TOKEN"
REFRESH_TOKEN_ENV = "TIKTOK_REFRESH_TOKEN"

#: Chunk size for FILE_UPLOAD. TikTok accepts the size you declare in
#: ``init``; 10 MiB chunks are the common production choice.
_CHUNK_SIZE = 10 * 1024 * 1024

_AUDIT_MANUAL_STEP = (
    "Pass TikTok's Content Posting API audit so posts can go public: "
    "in the TikTok developer portal (https://developers.tiktok.com → "
    "Manage Apps → your app), open the Content Posting API product and "
    "submit the Direct Post audit — you must upload screen recordings of "
    "(1) the TikTok OAuth consent screen, (2) navigating to your app's "
    "Post-to-TikTok page, and (3) what happens after posting, all matching "
    "TikTok's Content Sharing Guidelines "
    "(https://developers.tiktok.com/doc/content-sharing-guidelines). "
    "Until the audit passes, every API post is forced private (SELF_ONLY)."
)


class TikTokPublisher:
    """Direct Post publisher for one TikTok account.

    Credentials arrive via constructor args or env vars
    (``TIKTOK_CLIENT_KEY`` / ``TIKTOK_CLIENT_SECRET`` /
    ``TIKTOK_ACCESS_TOKEN`` / ``TIKTOK_REFRESH_TOKEN``). Without an
    access token every posting call raises :class:`CapabilityUnavailable`
    with the exact manual step — nothing is faked.
    """

    def __init__(
        self,
        *,
        client_key: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        http: Any | None = None,
    ) -> None:
        self.client_key = (client_key or os.environ.get(CLIENT_KEY_ENV, "")
                           ).strip()
        self.client_secret = (client_secret or
                              os.environ.get(CLIENT_SECRET_ENV, "")).strip()
        self.access_token = (access_token or
                             os.environ.get(ACCESS_TOKEN_ENV, "")).strip()
        self.refresh_token = (refresh_token or
                              os.environ.get(REFRESH_TOKEN_ENV, "")).strip()
        self.http = http or HttpClient()

    # ── auth ─────────────────────────────────────────────────────

    @staticmethod
    def auth_guide() -> str:
        """The EXACT manual steps to get TikTok posting credentials."""
        return "\n".join([
            "TikTok Direct Post setup (only you can do this — one time):",
            "1. Create an app at https://developers.tiktok.com → Manage Apps.",
            "2. Add the 'Content Posting API' product to the app.",
            "3. Wait for TikTok's app approval (typically 1–3 days) — OAuth",
            "   fails until the app itself is approved.",
            "4. Set an HTTPS redirect URI on the app (TikTok requires HTTPS).",
            "5. Open the authorize URL (see authorize_url()), grant scopes",
            f"   {', '.join(TIKTOK_SCOPES)}, and paste the code back via",
            "   exchange_code().",
            "6. To post PUBLICLY (not just private), pass TikTok's Content",
            "   Posting API audit — see the audit step in the error message.",
        ])

    def authorize_url(self, redirect_uri: str,
                      scopes: tuple[str, ...] = TIKTOK_SCOPES) -> str:
        """The URL the owner opens to grant posting access."""
        if not self.client_key:
            raise CapabilityUnavailable(
                "tiktok",
                "no TikTok client key — create the app first",
                manual_step=self.auth_guide(),
            )
        params = {
            "client_key": self.client_key,
            "response_type": "code",
            "scope": ",".join(scopes),
            "redirect_uri": redirect_uri,
            "state": f"nm-{int(time.time())}",
        }
        return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code: str, redirect_uri: str) -> dict[str, str]:
        """Swap an authorization code for access + refresh tokens."""
        if not (self.client_key and self.client_secret):
            raise CapabilityUnavailable(
                "tiktok",
                "no TikTok client key/secret — create the app first",
                manual_step=self.auth_guide(),
            )
        code = (code or "").strip()
        if not code:
            raise CapabilityUnavailable(
                "tiktok",
                "empty authorization code",
                manual_step=("Open authorize_url() in your browser, grant "
                            "access, and paste the code TikTok redirects "
                            "with."),
            )
        resp = self.http.post_json(TOKEN_URL, {
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        })
        data = self._tiktok_data(resp, "token exchange")
        self.access_token = str(data.get("access_token", ""))
        self.refresh_token = str(data.get("refresh_token", ""))
        if not self.access_token:
            raise CapabilityUnavailable(
                "tiktok",
                "TikTok returned no access token on exchange",
                manual_step=self.auth_guide(),
            )
        _log.info("tiktok: tokens obtained via authorization code")
        return {"access_token": self.access_token,
                "refresh_token": self.refresh_token}

    def refresh_access_token(self) -> str:
        """Mint a fresh 24h access token from the refresh token."""
        if not (self.client_key and self.client_secret
                and self.refresh_token):
            raise CapabilityUnavailable(
                "tiktok",
                "no TikTok refresh token to refresh from",
                manual_step=self.auth_guide(),
            )
        resp = self.http.post_json(TOKEN_URL, {
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        })
        data = self._tiktok_data(resp, "token refresh")
        token = str(data.get("access_token", ""))
        if not token:
            raise CapabilityUnavailable(
                "tiktok",
                "TikTok refused the refresh token — re-authorize",
                manual_step=self.auth_guide(),
            )
        self.access_token = token
        _log.info("tiktok: access token refreshed")
        return token

    def _require_token(self) -> str:
        if not self.access_token:
            raise CapabilityUnavailable(
                "tiktok",
                "no TikTok access token — the account isn't connected",
                manual_step=self.auth_guide(),
            )
        return self.access_token

    # ── API plumbing ─────────────────────────────────────────────

    def _tiktok_data(self, resp: Any, what: str) -> dict[str, Any]:
        if not resp.ok:
            raise CapabilityUnavailable(
                "tiktok",
                f"TikTok {what} failed ({resp.status}): {resp.text[:200]}",
                manual_step=self.auth_guide(),
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise CapabilityUnavailable(
                "tiktok",
                f"TikTok {what} returned invalid JSON",
                manual_step=self.auth_guide(),
            ) from exc
        if not isinstance(body, dict):
            raise CapabilityUnavailable(
                "tiktok", f"TikTok {what} returned an unexpected response",
                manual_step=self.auth_guide(),
            )
        err = body.get("error") or {}
        code = str(err.get("code", ""))
        if code and code != "ok":
            message = str(err.get("message", ""))
            if "unaudited" in code or "audit" in message.lower():
                raise CapabilityUnavailable(
                    "tiktok",
                    f"TikTok refused ({code}): {message or 'app not audited'}",
                    manual_step=_AUDIT_MANUAL_STEP,
                    details={"tiktok_code": code},
                )
            raise CapabilityUnavailable(
                "tiktok",
                f"TikTok {what} failed ({code}): {message}",
                manual_step=self.auth_guide(),
                details={"tiktok_code": code},
            )
        data = body.get("data")
        return data if isinstance(data, dict) else {}

    def _post(self, path: str, payload: dict[str, Any],
              what: str) -> dict[str, Any]:
        resp = self.http.post_json(
            f"{API_BASE}{path}",
            payload,
            headers={"Authorization": f"Bearer {self._require_token()}"},
        )
        return self._tiktok_data(resp, what)

    # ── posting ──────────────────────────────────────────────────

    def creator_info(self) -> dict[str, Any]:
        """The account's posting capabilities (privacy levels, toggles)."""
        return self._post("/post/publish/creator_info/query/", {},
                          "creator_info query")

    def post_video(
        self,
        file_path: str | Path,
        *,
        caption: str = "",
        tags: list[str] | None = None,
        privacy_level: str = "",
        disable_duet: bool = False,
        disable_comment: bool = False,
        disable_stitch: bool = False,
        brand_organic_toggle: bool = False,
        branded_content_toggle: bool = False,
        is_aigc: bool = True,
        chunk_size: int = _CHUNK_SIZE,
        ledger: PublishLedger | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Direct Post a video (init → chunked FILE_UPLOAD → publish).

        ``privacy_level`` is picked from ``creator_info`` when omitted
        (``PUBLIC_TO_EVERYONE`` when allowed, else the first allowed
        level). On unaudited apps TikTok forces ``SELF_ONLY`` itself —
        that refusal surfaces as :class:`CapabilityUnavailable` with
        the audit step, never as a fake public post.
        """
        path = Path(file_path)
        if not path.is_file():
            raise CapabilityUnavailable(
                "tiktok", f"no such file: {path}",
                manual_step="Render the video first, then post it.",
            )
        info = self.creator_info()
        allowed = [str(p) for p in info.get("privacy_level_options", [])]
        if not allowed:
            raise CapabilityUnavailable(
                "tiktok",
                "TikTok returned no allowed privacy levels for this account",
                manual_step=self.auth_guide(),
            )
        level = (privacy_level or "").strip()
        if not level:
            level = ("PUBLIC_TO_EVERYONE" if "PUBLIC_TO_EVERYONE" in allowed
                     else allowed[0])
        elif level not in allowed:
            raise CapabilityUnavailable(
                "tiktok",
                f"privacy_level {level!r} not allowed for this account "
                f"(allowed: {', '.join(allowed)})",
                manual_step=("Pick one of the allowed levels, or pass "
                             "TikTok's audit to unlock public posting: "
                             + _AUDIT_MANUAL_STEP),
            )
        caption = adapt_description("tiktok", caption, tags)
        total = path.stat().st_size
        total_chunks = max(1, (total + chunk_size - 1) // chunk_size)
        init = self._post("/post/publish/video/init/", {
            "post_info": {
                "title": caption,
                "privacy_level": level,
                "disable_duet": bool(disable_duet),
                "disable_comment": bool(disable_comment),
                "disable_stitch": bool(disable_stitch),
                "video_cover_timestamp_ms": 1000,
                "brand_organic_toggle": bool(brand_organic_toggle),
                "brand_content_toggle": bool(branded_content_toggle),
                # Devon's shorts are AI-generated — disclose it honestly.
                "is_aigc": bool(is_aigc),
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": total,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunks,
            },
        }, "video init")
        post_id = str(init.get("post_id", ""))
        upload_url = str(init.get("upload_url", ""))
        if not post_id or not upload_url:
            raise CapabilityUnavailable(
                "tiktok",
                "TikTok init returned no post_id/upload_url",
                manual_step=self.auth_guide(),
            )
        self._upload_chunks(upload_url, path, total, chunk_size,
                            on_progress)
        published = self._post("/post/publish/video/publish/",
                               {"post_id": post_id}, "video publish")
        result = {
            "post_id": post_id,
            "publish_id": str(published.get("publish_id", "")),
            "privacy_level": level,
            "caption": caption,
        }
        if ledger is not None:
            entry = ledger.record(
                platform="tiktok", file=str(path),
                title=caption[:80],
                platform_video_id=result["publish_id"] or post_id,
                status="processing",  # TikTok processes async after publish
                privacy=level,
            )
            result["ledger_id"] = entry["id"]
        _log.info("tiktok direct post ok: post_id=%s privacy=%s",
                  post_id, level)
        return result

    def _upload_chunks(
        self,
        upload_url: str,
        path: Path,
        total: int,
        chunk_size: int,
        on_progress: Callable[[int, int], None] | None,
    ) -> None:
        """PUT each chunk to TikTok's upload URL with Content-Range."""
        sent = 0
        with path.open("rb") as fh:
            while sent < total:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                end = sent + len(chunk) - 1
                try:
                    resp = self.http.request(
                        "PUT", upload_url, data=chunk,
                        headers={
                            "Authorization":
                                f"Bearer {self._require_token()}",
                            "Content-Type": "video/mp4",
                            "Content-Length": str(len(chunk)),
                            "Content-Range":
                                f"bytes {sent}-{end}/{total}",
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    raise CapabilityUnavailable(
                        "tiktok",
                        f"TikTok chunk upload failed at byte {sent}: {exc}",
                        manual_step=self.auth_guide(),
                    ) from exc
                if not resp.ok:
                    raise CapabilityUnavailable(
                        "tiktok",
                        f"TikTok chunk upload failed ({resp.status}): "
                        f"{resp.text[:200]}",
                        manual_step=self.auth_guide(),
                    )
                sent = end + 1
                if on_progress:
                    on_progress(sent, total)
