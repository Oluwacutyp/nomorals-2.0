"""Instagram Graph API connector (Meta).

Drives Instagram business/creator accounts through the Meta Graph API
(https://developers.facebook.com/docs/instagram-platform) with the
framework HttpClient — no private-API scraping.

Auth: a Facebook user access token (``OAUTH2``) with
``instagram_basic`` + ``instagram_content_publish`` (and
``pages_read_engagement``) scopes. ``connect(token=...)`` takes a
pre-obtained token; without one it prints the grant guide and fails fast —
Meta offers no device flow for this.

Hard boundary, stated honestly: PERSONAL Instagram accounts cannot post
via the API at all — only business/creator accounts linked to a Facebook
Page can publish. ``connect()`` verifies the linkage and refuses otherwise.

Publishing is a two-step flow, per the API's design:
``create_media_container`` (``POST /{ig-user-id}/media`` with a public
``image_url``) returns a creation id, then ``publish_media``
(``POST /{ig-user-id}/media_publish``) makes it live. Unpublished
containers expire, so ``publish_image`` wraps both steps atomically behind
the confirmation gate. Videos need ``media_type=VIDEO`` and a
``video_url``; publishing before Meta finishes processing fails with a
clear "not ready" message — poll ``container_status`` yourself.
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

__all__ = ["InstagramConnector", "InstagramError"]

_log = get_logger(__name__)

API_BASE = "https://graph.facebook.com/v18.0"
TOKEN_ENV = "INSTAGRAM_ACCESS_TOKEN"
DOCS_URL = "https://developers.facebook.com/docs/instagram-platform"

REQUIRED_SCOPES = (
    "instagram_basic",
    "instagram_content_publish",
    "pages_read_engagement",
)


class InstagramError(ConnectorError):
    """An Instagram Graph API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        ig_error_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.ig_error_code = ig_error_code


@register_connector
class InstagramConnector(Connector):
    """Devon's Instagram Graph API adapter (business/creator accounts)."""

    id = "instagram"
    name = "Instagram"
    description = (
        "Instagram Graph API for business/creator accounts: identify the "
        "account, list media, and publish photos via the two-step "
        "container flow. Personal accounts cannot post via the API."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a user access token and resolve the IG business account.

        Takes a pre-obtained Facebook user access token (the Meta OAuth
        dance happens in the owner's browser — Meta has no device flow).
        Without ``token`` the grant guide prints and the flow fails fast.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "instagram is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        try:
            secret = (token or "").strip() or prompt_secret(
                "Instagram (Meta) user access token", env_var=TOKEN_ENV
            )
        except ConnectorError as exc:
            raise ConnectorError(
                "no access token given. Grant one in the owner's browser: "
                "create a Meta app at https://developers.facebook.com/apps, "
                "add the Instagram Graph API product, and authorize with "
                f"scopes {', '.join(REQUIRED_SCOPES)} — then pass "
                "token=<access token>."
            ) from exc
        # 1. the token is valid at all
        me = self._api("GET", "/me", params={"fields": "id,name"},
                       token=secret)
        fb_name = str(me.get("name", me.get("id", "?")))
        # 2. find the linked Instagram business/creator account
        ig_id, ig_username = self._resolve_ig_account(secret)
        self._store_credential(
            f"@{ig_username}",
            secret,
            credential_type="oauth2",
            scopes=list(REQUIRED_SCOPES),
            metadata={
                "fb_user_id": me.get("id"),
                "fb_name": fb_name,
                "ig_user_id": ig_id,
                "username": ig_username,
            },
        )
        _log.info("instagram connected as @%s", ig_username)
        return ConnectResult(
            ok=True,
            account=f"@{ig_username}",
            scopes=list(REQUIRED_SCOPES),
            message=(
                f"connected as Instagram business account @{ig_username} "
                f"(id {ig_id}, Facebook user {fb_name}). The access token "
                "is in the encrypted vault. Long-lived tokens expire after "
                "~60 days — reconnect then."
            ),
        )

    def _resolve_ig_account(self, secret: str) -> tuple[str, str]:
        """The IG business account id + username behind a user token."""
        pages = self._api(
            "GET", "/me/accounts",
            params={"fields": "id,name,instagram_business_account{id,username}"},
            token=secret,
        )
        accounts = pages.get("data") if isinstance(pages, dict) else None
        if isinstance(accounts, list):
            for page in accounts:
                ig = page.get("instagram_business_account") or {}
                if ig.get("id"):
                    return str(ig["id"]), str(ig.get("username", ""))
        raise ConnectorError(
            "no Instagram business/creator account linked to this token — "
            "personal accounts cannot use the API. Link a business or "
            "creator account to one of your Facebook Pages in the "
            "account's settings (Settings → Account type and tools → "
            "Switch to professional account), then reconnect."
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name instagram`",
            )
        meta = cred.metadata or {}
        try:
            profile = self._api(
                "GET", f"/{meta.get('ig_user_id')}",
                params={"fields": "id,username"},
                token=cred.password,
            )
        except InstagramError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list(meta.get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): it expired or was "
                       "revoked — reconnect with a fresh token",
            )
        return ConnectorStatus(
            connected=True,
            account=f"@{profile.get('username', cred.username)}",
            scopes=list(meta.get("scopes", [])),
            last_checked=time.time(),
            detail=f"ig user id {profile.get('id')} responding",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/me", params={"fields": "id"},
                      token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── Graph API ────────────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """The Facebook user identity (``GET /me?fields=id,username``).

        This is the Meta-side user behind the token; use
        ``get_profile()`` for the Instagram business account itself.
        """
        data = self._api("GET", "/me", params={"fields": "id,name"})
        return data if isinstance(data, dict) else {}

    def get_profile(self) -> dict[str, Any]:
        """The Instagram business account profile."""
        meta = self._require_credential().metadata or {}
        data = self._api(
            "GET", f"/{meta['ig_user_id']}",
            params={"fields": "id,username,name,account_type,media_count,"
                              "followers_count"},
        )
        return data if isinstance(data, dict) else {}

    def list_media(
        self,
        *,
        limit: int = 25,
        after: str = "",
    ) -> list[dict[str, Any]]:
        """Published media (``GET /{ig-user-id}/media``)."""
        meta = self._require_credential().metadata or {}
        params: dict[str, Any] = {
            "fields": "id,caption,media_type,media_url,thumbnail_url,"
                      "permalink,timestamp,like_count,comments_count",
            "limit": max(1, min(limit, 100)),
        }
        if after:
            params["after"] = after
        data = self._api("GET", f"/{meta['ig_user_id']}/media",
                         params=params)
        items = data.get("data") if isinstance(data, dict) else None
        return items if isinstance(items, list) else []

    def create_media_container(
        self,
        image_url: str = "",
        *,
        video_url: str = "",
        caption: str = "",
    ) -> str:
        """Create a media container (``POST /{ig-user-id}/media``).

        Returns the creation id. Image posts take ``image_url``; video
        posts take ``video_url`` + ``media_type=VIDEO`` (reels need the
        video at a public HTTPS URL). The container is a draft — nothing
        goes live until ``publish_media``.
        """
        image_url = (image_url or "").strip()
        video_url = (video_url or "").strip()
        if not image_url and not video_url:
            raise ConnectorError(
                "create_media_container needs image_url or video_url — "
                "nothing to publish from"
            )
        if image_url and video_url:
            raise ConnectorError(
                "pass image_url OR video_url, not both"
            )
        for label, url in (("image_url", image_url),
                           ("video_url", video_url)):
            if url and not url.lower().startswith("https://"):
                raise ConnectorError(
                    f"{label} must be a public https URL — Meta fetches "
                    "the media itself"
                )
        meta = self._require_credential().metadata or {}
        payload: dict[str, Any] = {"caption": caption or ""}
        if video_url:
            payload["media_type"] = "VIDEO"
            payload["video_url"] = video_url
        else:
            payload["image_url"] = image_url
        data = self._api("POST", f"/{meta['ig_user_id']}/media",
                         payload=payload)
        creation_id = data.get("id") if isinstance(data, dict) else ""
        if not creation_id:
            raise InstagramError(
                "instagram returned no creation id — the container was "
                "not created"
            )
        return str(creation_id)

    def container_status(self, creation_id: str) -> str:
        """Processing status of a media container (``status_code``)."""
        if not (creation_id or "").strip():
            raise ConnectorError("creation_id is required")
        data = self._api(
            "GET", f"/{creation_id}", params={"fields": "status_code"}
        )
        return str(data.get("status_code", "UNKNOWN")) if isinstance(
            data, dict) else "UNKNOWN"

    def publish_media(self, creation_id: str) -> dict[str, Any]:
        """Publish a container (``POST /{ig-user-id}/media_publish``).

        Refuses to publish a video container that Meta is still processing
        — call ``container_status`` until it reports ``FINISHED``.
        """
        if not (creation_id or "").strip():
            raise ConnectorError("creation_id is required")
        meta = self._require_credential().metadata or {}
        data = self._api(
            "POST", f"/{meta['ig_user_id']}/media_publish",
            payload={"creation_id": creation_id},
        )
        return data if isinstance(data, dict) else {}

    def publish_image(
        self,
        image_url: str,
        *,
        caption: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Publish a photo in one call (container + publish).

        Consequential: posting goes live on the account — gated behind
        explicit owner confirmation.
        """
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="publish_image",
            title=f"Publish Instagram photo ({image_url[:60]}…)",
            instructions="\n".join([
                "Devon wants to publish this photo to your Instagram "
                "business account.",
                "Review it — publishing is public and final.",
                f"Image: {image_url}",
                f"Caption: {caption or '(none)'}",
            ]),
            resume_state={"image_url": image_url, "caption": caption},
        )
        creation_id = self.create_media_container(image_url,
                                                  caption=caption)
        return self.publish_media(creation_id)

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "instagram is not connected — run "
                "`nm connectors connect --name instagram` first"
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
        """One Graph API call; failures become InstagramError.

        The access token rides the ``access_token`` query parameter, per
        Meta's documented pattern — it never goes in headers or logs, and
        error messages scrub it out of URLs.
        """
        secret = token or self._require_credential().password
        url = f"{API_BASE}{path}"
        merged_params = {"access_token": secret, **(params or {})}
        safe_params = dict(params or {})
        try:
            if method == "GET":
                resp = self.http.get(url, params=merged_params)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, params=merged_params
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise InstagramError(
                "instagram rate limit hit (429): the Graph API throttles "
                f"aggressively — back off ~{exc.retry_after:.0f}s",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise InstagramError(f"instagram request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise InstagramError(f"instagram request failed: {exc}") from exc
        # Keep the token out of anything that could reach a log.
        safe_url = f"{url}?{'&'.join(f'{k}=…' for k in safe_params)}"
        if resp.status == 401:
            raise InstagramError(
                "instagram rejected the access token (401): it expired or "
                "was revoked — reconnect with a fresh token",
                status_code=401,
            )
        if resp.status == 429:
            raise InstagramError(
                "instagram rate limit hit (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            detail, code = self._error_detail(resp)
            raise InstagramError(
                f"instagram {method} {path} failed ({resp.status}, "
                f"{safe_url}): {detail}",
                status_code=resp.status,
                ig_error_code=code,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise InstagramError(
                f"instagram {method} {path} returned invalid JSON"
            ) from exc
        if isinstance(body, dict) and body.get("error"):
            detail, code = self._error_detail(resp)
            if resp.status == 400 and "not finished" in detail.lower():
                raise InstagramError(
                    "instagram: the video container is still processing — "
                    "wait and call container_status() until it reports "
                    "FINISHED, then publish",
                    status_code=400,
                    ig_error_code=code,
                )
            raise InstagramError(
                f"instagram {method} {path} failed: {detail}",
                status_code=resp.status,
                ig_error_code=code,
            )
        return body

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, int]:
        """Unwrap Graph API's ``{"error": {"message", "code"}}`` shape."""
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return resp.text[:200], 0
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                message = str(err.get("message", ""))[:200]
                try:
                    code = int(err.get("code", 0))
                except (TypeError, ValueError):
                    code = 0
                return message or resp.text[:200], code
        return resp.text[:200], 0
