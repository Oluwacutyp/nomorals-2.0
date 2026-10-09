"""Meta publisher — Instagram Reels + Facebook Page video.

Subclasses :class:`nomorals.connectors.instagram.InstagramConnector`,
reusing its OAuth/token plumbing, the two-step container flow, and the
confirmation gate.

Instagram Reels (verified 2026-10-09 against Meta's docs and production
integrations):

* Two-step: ``POST /{ig-user-id}/media`` with ``media_type=REELS``,
  ``video_url``, ``caption``, optional ``cover_url`` / ``share_to_feed``
  → container id → poll ``status_code`` until ``FINISHED`` → ``POST
  /{ig-user-id}/media_publish``. (The old ``VIDEO`` media type is
  deprecated — REELS is the current one.)
* HARD REQUIREMENT: ``video_url`` must be a PUBLIC HTTPS URL — Meta's
  servers fetch the media themselves. A local file path cannot be
  posted; without a public URL this raises :class:`CapabilityUnavailable`
  with the exact fix (object storage / CDN with a public URL).
* The account must be Business or Creator. Caption limit: 2200 chars.
* Scopes (Facebook-login path, as the parent connector uses):
  ``instagram_basic`` + ``instagram_content_publish`` (+
  ``pages_read_engagement``). The newer Instagram-login path uses
  ``instagram_business_content_publish`` on graph.instagram.com — the
  container flow is identical.
* App Review is NOT needed to publish to your own connected accounts in
  development mode — only to post on behalf of other people's accounts.
* Rate limit: ~100 API-published posts per rolling 24h per account
  (Meta raised this from 50 — verify in the current docs before
  building a schedule around it).

Facebook Page video:

* ``POST /{page-id}/videos`` with ``file_url`` (public HTTPS, Meta
  fetches it) or a direct multipart upload of the bytes.
* Needs a Page access token (``pages_manage_posts`` /
  ``pages_read_engagement``) — get one via ``GET /me/accounts`` with a
  user token. Without it: :class:`CapabilityUnavailable` with the step.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ....connectors._confirm import confirm_or_checkpoint
from ....connectors.instagram import API_BASE, InstagramConnector
from ....core.logging_setup import get_logger
from . import CapabilityUnavailable, adapt_description
from .ledger import PublishLedger

__all__ = ["MetaPublisher"]

_log = get_logger(__name__)

_VIDEO_URL_STEP = (
    "Give Meta a public HTTPS URL for the video: upload the file to "
    "object storage or a CDN with public read (e.g. Cloudflare R2, S3, "
    "or any static host) and pass video_url='https://…'. Meta's servers "
    "fetch the media themselves — a local file path can never work, "
    "and localhost/private URLs fail."
)


class MetaPublisher(InstagramConnector):
    """Instagram Reels + Facebook Page video publishing.

    Auth: identical to ``InstagramConnector`` — ``connect(token=...)``
    with a Meta user access token, then the IG business account id is
    resolved and vaulted.
    """

    # ── Instagram Reels ──────────────────────────────────────────

    def publish_reel(
        self,
        video_url: str,
        *,
        caption: str = "",
        tags: list[str] | None = None,
        cover_url: str = "",
        share_to_feed: bool = True,
        thumb_offset_ms: int = 0,
        poll_timeout_s: float = 300.0,
        poll_interval_s: float = 10.0,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
        ledger: PublishLedger | None = None,
    ) -> dict[str, Any]:
        """Publish a Reel in one call (container → poll → publish).

        ``video_url`` MUST be public HTTPS — Meta fetches it. Polls the
        container until Meta finishes processing (bounded; raises on
        ``ERROR`` with Meta's reason).
        """
        video_url = (video_url or "").strip()
        if not video_url.lower().startswith("https://"):
            raise CapabilityUnavailable(
                "instagram",
                "Instagram Reels need the video at a public HTTPS URL — "
                f"got {video_url!r}",
                manual_step=_VIDEO_URL_STEP,
            )
        cover_url = (cover_url or "").strip()
        if cover_url and not cover_url.lower().startswith("https://"):
            raise CapabilityUnavailable(
                "instagram",
                f"cover_url must be public HTTPS, got {cover_url!r}",
                manual_step=_VIDEO_URL_STEP,
            )
        caption = adapt_description("instagram", caption, tags)
        meta = self._require_credential().metadata or {}
        ig_user_id = str(meta.get("ig_user_id", ""))
        if not ig_user_id:
            raise CapabilityUnavailable(
                "instagram",
                "no Instagram business account id on the credential — "
                "reconnect",
                manual_step=("Run connect(token=<Meta user access token>): "
                             "create a Meta app, add the Instagram Graph API "
                             "product, authorize with instagram_basic + "
                             "instagram_content_publish, then connect with "
                             "the token. The IG account must be Business or "
                             "Creator."),
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="publish_reel",
            title="Publish Instagram Reel",
            instructions="\n".join([
                "Devon wants to publish this Reel to your Instagram "
                "business account.",
                f"Video: {video_url}",
                f"Caption: {caption[:300] or '(none)'}",
                "Publishing is public and final — review it first.",
            ]),
            resume_state={
                "video_url": video_url,
                "caption": caption,
                "cover_url": cover_url,
                "share_to_feed": share_to_feed,
            },
        )
        payload: dict[str, Any] = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": bool(share_to_feed),
        }
        if cover_url:
            payload["cover_url"] = cover_url
        if thumb_offset_ms > 0:
            payload["thumb_offset"] = int(thumb_offset_ms)
        data = self._api("POST", f"/{ig_user_id}/media", payload=payload)
        creation_id = str(data.get("id", "")) if isinstance(data, dict) else ""
        if not creation_id:
            raise CapabilityUnavailable(
                "instagram",
                "Meta returned no container id — the Reel was not created",
                manual_step=self._meta_error_step(data),
            )
        _log.info("instagram reel container %s created, polling",
                  creation_id)
        self._wait_for_container(creation_id, poll_timeout_s,
                                 poll_interval_s)
        published = self.publish_media(creation_id)
        media_id = str(published.get("id", "")) if isinstance(
            published, dict) else ""
        result: dict[str, Any] = {
            "media_id": media_id,
            "creation_id": creation_id,
        }
        if media_id:
            info = self._api(
                "GET", f"/{media_id}", params={"fields": "permalink"})
            if isinstance(info, dict) and info.get("permalink"):
                result["permalink"] = info["permalink"]
        if ledger is not None:
            entry = ledger.record(
                platform="instagram", file=video_url,
                title=caption[:80] or "(reel)",
                platform_video_id=media_id, status="uploaded",
            )
            result["ledger_id"] = entry["id"]
        _log.info("instagram reel published: %s", media_id)
        return result

    def _wait_for_container(
        self,
        creation_id: str,
        timeout_s: float,
        interval_s: float,
    ) -> None:
        """Poll ``status_code`` until FINISHED; raise on ERROR/timeout."""
        deadline = time.time() + max(30.0, timeout_s)
        last = ""
        while time.time() < deadline:
            status = self.container_status(creation_id)
            last = status
            if status == "FINISHED":
                return
            if status == "ERROR":
                detail = self._api(
                    "GET", f"/{creation_id}",
                    params={"fields": "status_code,error_message"})
                msg = ""
                if isinstance(detail, dict):
                    msg = str(detail.get("error_message", ""))
                raise CapabilityUnavailable(
                    "instagram",
                    f"Meta rejected the Reel container (ERROR)"
                    f"{': ' + msg if msg else ''}",
                    manual_step=("Fix the media to Meta's spec and retry: "
                                 "MP4 (H.264), ≤4GB, ≤60min, 9:16 vertical "
                                 "for Reels, public HTTPS URL. " +
                                 _VIDEO_URL_STEP),
                )
            time.sleep(max(2.0, interval_s))
        raise CapabilityUnavailable(
            "instagram",
            f"Reel container {creation_id} still {last!r} after "
            f"{timeout_s:.0f}s — Meta never finished processing it",
            manual_step=("Check the video in Meta's Media Library / Graph "
                         "API Explorer, or retry with a fresh container."),
        )

    @staticmethod
    def _meta_error_step(data: Any) -> str:
        detail = ""
        if isinstance(data, dict):
            err = data.get("error") or {}
            detail = str(err.get("message", ""))
        return ("Inspect the Graph API error and retry. "
                f"{'Meta said: ' + detail if detail else ''}Docs: "
                "https://developers.facebook.com/docs/instagram-platform/"
                "content-publishing")

    # ── Facebook Page video ──────────────────────────────────────

    def publish_facebook_video(
        self,
        page_id: str,
        *,
        video_url: str = "",
        file_path: str | Path | None = None,
        description: str = "",
        tags: list[str] | None = None,
        page_access_token: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
        ledger: PublishLedger | None = None,
    ) -> dict[str, Any]:
        """Post a video to a Facebook Page.

        ``video_url`` (public HTTPS — Meta fetches it) or ``file_path``
        (direct multipart upload). Needs a Page access token.
        """
        page_id = (page_id or "").strip()
        if not page_id:
            raise CapabilityUnavailable(
                "facebook", "page_id is required",
                manual_step="Pass the numeric Facebook Page id to post to.",
            )
        page_token = (page_access_token or "").strip()
        if not page_token:
            raise CapabilityUnavailable(
                "facebook",
                "no Page access token — a user token can't post to a Page",
                manual_step=("Mint one: GET "
                             "https://graph.facebook.com/v18.0/me/accounts"
                             "?access_token=<your user token> with "
                             "pages_show_list, then use the target Page's "
                             "'access_token'. It needs pages_manage_posts to "
                             "publish video."),
            )
        video_url = (video_url or "").strip()
        if video_url and not video_url.lower().startswith("https://"):
            raise CapabilityUnavailable(
                "facebook",
                f"video_url must be public HTTPS, got {video_url!r}",
                manual_step=_VIDEO_URL_STEP,
            )
        path = Path(file_path) if file_path else None
        if path is not None and not path.is_file():
            raise CapabilityUnavailable(
                "facebook", f"no such file: {path}",
                manual_step="Render the video first, then post it.",
            )
        if not video_url and path is None:
            raise CapabilityUnavailable(
                "facebook",
                "nothing to post — pass video_url or file_path",
                manual_step=_VIDEO_URL_STEP,
            )
        description = adapt_description("facebook", description, tags)
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="publish_facebook_video",
            title=f"Publish video to Facebook Page {page_id}",
            instructions="\n".join([
                "Devon wants to publish this video to your Facebook Page.",
                f"Source: {video_url or path}",
                f"Description: {description[:300] or '(none)'}",
                "Publishing is public and final — review it first.",
            ]),
            resume_state={
                "page_id": page_id,
                "video_url": video_url,
                "file": str(path) if path else "",
                "description": description,
            },
        )
        params = {"access_token": page_token}
        if video_url:
            resp = self.http.post_json(
                f"{API_BASE}/{page_id}/videos",
                {"file_url": video_url, "description": description},
                params=params,
            )
        else:
            assert path is not None
            resp = self.http.request(
                "POST",
                f"{API_BASE}/{page_id}/videos",
                data=self._multipart_file(path),
                headers={"Content-Type": self._multipart_ct(path)},
                params=params,
            )
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not resp.ok or not isinstance(body, dict) or not body.get("id"):
            detail = ""
            if isinstance(body, dict):
                err = body.get("error") or {}
                detail = str(err.get("message", ""))
            raise CapabilityUnavailable(
                "facebook",
                f"Facebook video post failed ({resp.status})"
                f"{': ' + detail if detail else ''}",
                manual_step=self._meta_error_step(body),
            )
        video_id = str(body["id"])
        result = {"video_id": video_id, "page_id": page_id}
        if ledger is not None:
            entry = ledger.record(
                platform="facebook", file=video_url or str(path),
                title=description[:80] or "(video)",
                platform_video_id=video_id, status="uploaded",
            )
            result["ledger_id"] = entry["id"]
        _log.info("facebook page video posted: %s", video_id)
        return result

    @staticmethod
    def _multipart_file(path: Path) -> bytes:
        boundary = MetaPublisher._boundary
        data = path.read_bytes()
        return b"\r\n".join([
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="source"; '
            + f'filename="{path.name}"'.encode(),
            b"Content-Type: video/mp4",
            b"",
            data,
            f"--{boundary}--".encode(),
            b"",
        ])

    _boundary = "----nm-fb-video"

    @classmethod
    def _multipart_ct(cls, path: Path) -> str:  # noqa: ARG003
        return f"multipart/form-data; boundary={cls._boundary}"
