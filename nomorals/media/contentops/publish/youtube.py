"""YouTube publisher — full headless uploads via YouTube Data API v3.

Subclasses :class:`nomorals.connectors.youtube.YouTubeConnector`, so all
OAuth machinery (grant flow, vaulted refresh token, auto-refresh) is
reused, not reinvented. This module adds what the connector's direct
multipart path deliberately left out:

* resumable upload sessions — files of ANY size, chunked PUTs with
  ``Content-Range``, ``308 Resume Incomplete`` handling, and resume
  from the server's confirmed offset after a failure;
* ``thumbnails.set`` (custom thumbnail);
* ``playlistItems.insert`` (add the upload to a playlist);
* scheduled publishing via ``status.publishAt`` (requires
  ``privacyStatus: "private"`` — ``publishAt`` is silently ignored
  otherwise, per the API's own behavior);
* per-call quota accounting — every API call's quota cost is logged
  and accumulated into the publish result.

Quota costs (from Google's quota-cost table,
https://developers.google.com/youtube/v3/determine_quota_cost —
verified 2026-10-09):

* ``videos.insert`` — 1600 units (~6 uploads/day on the default
  10,000-unit daily quota; request an increase via the Cloud console
  for more)
* ``thumbnails.set`` — 50 units
* ``playlistItems.insert`` — 50 units
* ``videos.update`` — 50 units
* ``search.list`` — 100 units; every ``*.list`` — 1 unit

Shorts: YouTube detects Shorts automatically — there is no API flag.
A video ≤60s in square/vertical aspect becomes a Short on its own;
``is_shorts_candidate()`` encodes that heuristic so the caller can
label intent, but the platform decides.

Scopes needed: ``https://www.googleapis.com/auth/youtube.upload``
(sensitive scope — for personal use keep the Cloud project in Testing
mode with yourself as a test user; no Google verification needed).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ....connectors._confirm import confirm_or_checkpoint
from ....connectors.youtube import (
    UPLOAD_BASE,
    YouTubeConnector,
    YouTubeError,
)
from ....core.logging_setup import get_logger
from . import PLATFORM_LIMITS, adapt_description, adapt_title
from .ledger import PublishLedger

__all__ = ["YouTubePublisher", "QUOTA_COSTS"]

_log = get_logger(__name__)

#: Quota cost per YouTube Data API v3 method (units). From Google's
#: quota-cost table — see module docstring.
QUOTA_COSTS: dict[str, int] = {
    "videos.insert": 1600,
    "thumbnails.set": 50,
    "playlistItems.insert": 50,
    "videos.update": 50,
    "search.list": 100,
    "videos.list": 1,
    "channels.list": 1,
    "playlists.list": 1,
    "playlistItems.list": 1,
    "commentThreads.list": 1,
    "captions.insert": 400,
}

#: Chunk size for resumable uploads. Google recommends multiples of
#: 256 KiB; 8 MiB is the sweet spot for throughput vs retry cost.
_CHUNK_SIZE = 8 * 1024 * 1024

_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}

_RANGE_RE = re.compile(r"bytes=0-(\d+)")


def _mime_type(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "video/mp4")


class YouTubePublisher(YouTubeConnector):
    """Full YouTube upload publisher. Not registered as a connector —
    it extends the ``youtube`` connector's auth/HTTP plumbing with the
    complete posting surface.

    Auth: identical to ``YouTubeConnector`` — run its ``connect()`` once
    (OAuth code flow), tokens live in the vault afterwards.
    """

    # ── quota ────────────────────────────────────────────────────

    def _charge_quota(self, method: str, quota_spent: dict[str, int]) -> int:
        """Log + accumulate one call's quota cost. Returns the cost."""
        cost = QUOTA_COSTS.get(method, 0)
        quota_spent["total"] = quota_spent.get("total", 0) + cost
        quota_spent[method] = quota_spent.get(method, 0) + cost
        _log.info("youtube quota: %s cost %d units (session total %d)",
                  method, cost, quota_spent["total"])
        return cost

    # ── resumable upload ─────────────────────────────────────────

    def start_resumable_session(
        self,
        metadata: dict[str, Any],
        total_bytes: int,
        mime_type: str = "video/mp4",
    ) -> str:
        """Open a resumable upload session; returns the session URI.

        Step 1 of Google's resumable protocol: POST the video metadata
        with ``X-Upload-Content-Length``; the ``Location`` response
        header is the session URI that receives the bytes.
        """
        url = (f"{UPLOAD_BASE}/videos"
               "?uploadType=resumable&part=snippet,status")
        body = json.dumps(metadata).encode("utf-8")
        try:
            resp = self.http.request(
                "POST",
                url,
                data=body,
                headers={
                    "Authorization":
                        f"Bearer {self._google_access_token()}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Length": str(total_bytes),
                    "X-Upload-Content-Type": mime_type,
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise YouTubeError(
                f"youtube resumable session init failed: {exc}"
            ) from exc
        if resp.status != 200:
            raise YouTubeError(
                f"youtube resumable session init failed ({resp.status}"
                f"{', ' + self._api_reason(resp) if self._api_reason(resp) else ''}): "
                f"{resp.text[:300]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        session_uri = ""
        try:
            session_uri = str(resp.headers.get("Location", "")
                              or resp.headers.get("location", ""))
        except Exception:  # noqa: BLE001 - headers may be a plain dict
            pass
        if not session_uri:
            raise YouTubeError(
                "youtube returned no resumable session URI (no Location "
                "header) — the session was not created"
            )
        _log.info("youtube resumable session opened (%d bytes)", total_bytes)
        return session_uri

    def query_upload_offset(self, session_uri: str, total_bytes: int) -> int:
        """Ask the server how many bytes it already has (resume support).

        Sends a zero-length PUT with ``Content-Range: bytes */{total}``;
        a ``308`` with a ``Range: bytes=0-{n}`` header means the next
        byte to send is ``n+1``.
        """
        try:
            resp = self.http.request(
                "PUT",
                session_uri,
                data=b"",
                headers={
                    "Authorization":
                        f"Bearer {self._google_access_token()}",
                    "Content-Length": "0",
                    "Content-Range": f"bytes */{total_bytes}",
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise YouTubeError(
                f"youtube upload-status query failed: {exc}"
            ) from exc
        if resp.status == 308:
            try:
                headers = resp.headers or {}
            except Exception:  # noqa: BLE001 - defensive
                headers = {}
            kept = str(headers.get("Range", "") or headers.get("range", ""))
            match = _RANGE_RE.search(kept)
            return int(match.group(1)) + 1 if match else 0
        if 200 <= resp.status < 300:
            # Already complete — the video resource comes back.
            return total_bytes
        raise YouTubeError(
            f"youtube upload-status query failed ({resp.status}): "
            f"{resp.text[:200]}",
            status_code=resp.status,
        )

    def upload_chunks(
        self,
        session_uri: str,
        path: str | Path,
        total_bytes: int,
        mime_type: str = "video/mp4",
        *,
        chunk_size: int = _CHUNK_SIZE,
        start_at: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """PUT the file in chunks; returns the created video resource.

        Handles ``308 Resume Incomplete`` per chunk. On a transport
        failure it queries the server's confirmed offset and resumes
        from there instead of restarting.
        """
        path = Path(path)
        token = self._google_access_token()
        sent = max(0, start_at)
        with path.open("rb") as fh:
            fh.seek(sent)
            while sent < total_bytes:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                end = sent + len(chunk) - 1
                try:
                    resp = self.http.request(
                        "PUT",
                        session_uri,
                        data=chunk,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Length": str(len(chunk)),
                            "Content-Type": mime_type,
                            "Content-Range":
                                f"bytes {sent}-{end}/{total_bytes}",
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - retry via resume
                    _log.warning(
                        "youtube chunk upload failed at byte %d (%s) — "
                        "querying server offset and resuming", sent, exc)
                    sent = self.query_upload_offset(session_uri, total_bytes)
                    if sent >= total_bytes:
                        break
                    fh.seek(sent)
                    continue
                if resp.status == 308:
                    sent = end + 1
                    if on_progress:
                        on_progress(sent, total_bytes)
                    continue
                if 200 <= resp.status < 300:
                    try:
                        result = resp.json()
                    except Exception as exc:  # noqa: BLE001
                        raise YouTubeError(
                            "youtube upload completed but returned invalid "
                            "JSON"
                        ) from exc
                    if on_progress:
                        on_progress(total_bytes, total_bytes)
                    if not isinstance(result, dict) or not result.get("id"):
                        raise YouTubeError(
                            "youtube upload finished without a video id — "
                            f"response: {resp.text[:200]}"
                        )
                    _log.info("youtube upload complete: %s", result.get("id"))
                    return result
                # A hard error on a chunk: try to resume from the kept
                # offset once before giving up.
                _log.warning(
                    "youtube chunk PUT failed (%d) — attempting resume",
                    resp.status)
                try:
                    kept = self.query_upload_offset(session_uri, total_bytes)
                except YouTubeError:
                    raise YouTubeError(
                        f"youtube chunk upload failed ({resp.status}"
                        f"{', ' + self._api_reason(resp) if self._api_reason(resp) else ''}): "
                        f"{resp.text[:300]}",
                        status_code=resp.status,
                        reason=self._api_reason(resp),
                    ) from None
                if kept <= sent:
                    raise YouTubeError(
                        f"youtube chunk upload failed ({resp.status}): "
                        f"{resp.text[:300]} (server kept no further bytes)",
                        status_code=resp.status,
                        reason=self._api_reason(resp),
                    )
                sent = kept
                fh.seek(sent)
        raise YouTubeError(
            "youtube upload ended without a completed video resource — "
            "the file may be partially uploaded; re-run to resume"
        )

    # ── one-call publish ─────────────────────────────────────────

    @staticmethod
    def is_shorts_candidate(
        duration_s: float, width: int, height: int
    ) -> bool:
        """Heuristic: would YouTube classify this as a Short?

        YouTube detects Shorts automatically (no API flag): ≤60 seconds
        and square or vertical aspect. This mirrors that rule so the
        caller can label intent; the platform has the final say.
        """
        if duration_s <= 0 or duration_s > 60:
            return False
        if width <= 0 or height <= 0:
            return False
        return height >= width  # vertical or square

    def publish(
        self,
        file_path: str | Path,
        title: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        category_id: str = "22",
        privacy: str = "private",
        publish_at: datetime | None = None,
        thumbnail_path: str | Path | None = None,
        playlist_id: str = "",
        made_for_kids: bool = False,
        duration_s: float = 0,
        width: int = 0,
        height: int = 0,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
        ledger: PublishLedger | None = None,
        chunk_size: int = _CHUNK_SIZE,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Upload a video end-to-end (resumable, any size).

        Steps: confirm-gate → resumable session → chunked upload →
        optional thumbnail → optional playlist add → ledger record.

        ``publish_at`` schedules the video: it forces ``privacy`` to
        ``private`` and sets ``status.publishAt``. Passing ``publish_at``
        with any other privacy is refused loudly — the API would
        silently ignore the schedule.

        Returns ``video_id``, ``quota_spent`` (dict + total), and the
        ledger entry id when a ledger was given.
        """
        path = Path(file_path)
        if not path.is_file():
            raise YouTubeError(f"no such file: {path}")
        title = adapt_title("youtube", title)
        if not title:
            raise YouTubeError("a video title is required")
        description = adapt_description("youtube", description, tags)
        quota: dict[str, int] = {}

        privacy_status = (privacy or "private").strip().lower()
        publish_at_iso = ""
        if publish_at is not None:
            if publish_at.tzinfo is None:
                publish_at = publish_at.replace(tzinfo=timezone.utc)
            if publish_at <= datetime.now(timezone.utc):
                raise YouTubeError(
                    "publish_at must be in the future — "
                    f"got {publish_at.isoformat()}"
                )
            if privacy_status != "private":
                raise YouTubeError(
                    "publish_at requires privacy='private' — YouTube "
                    "silently ignores publishAt on any other privacy "
                    "status, so this is refused instead of half-scheduled"
                )
            publish_at_iso = publish_at.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        payload = {
            "file": str(path),
            "title": title,
            "description": description,
            "tags": list(tags or []),
            "category_id": category_id,
            "privacy": privacy_status,
            "publish_at": publish_at_iso,
            "made_for_kids": bool(made_for_kids),
            "thumbnail": str(thumbnail_path) if thumbnail_path else "",
            "playlist_id": playlist_id or "",
            "duration_s": duration_s,
            "width": width,
            "height": height,
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="publish_resumable",
            title=f"Publish '{title}' to YouTube",
            instructions="\n".join([
                "Devon wants to publish this video to your YouTube channel.",
                f"File: {path} ({path.stat().st_size} bytes, resumable)",
                f"Title: {title}",
                f"Privacy: {privacy_status}"
                + (f", scheduled for {publish_at_iso}" if publish_at_iso
                   else ""),
                ("Thumbnail: " + payload["thumbnail"]
                 if payload["thumbnail"] else "Thumbnail: (none)"),
                ("Playlist: " + playlist_id if playlist_id
                 else "Playlist: (none)"),
                "Quota: 1600 units for the upload"
                + (" + 50 thumbnail" if payload["thumbnail"] else "")
                + (" + 50 playlist add" if playlist_id else ""),
            ]),
            resume_state={"payload": payload},
        )
        return self.publish_now(payload, ledger=ledger,
                                 chunk_size=chunk_size,
                                 on_progress=on_progress, quota=quota)

    def publish_now(
        self,
        payload: dict[str, Any],
        *,
        ledger: PublishLedger | None = None,
        chunk_size: int = _CHUNK_SIZE,
        on_progress: Callable[[int, int], None] | None = None,
        quota: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Run the publish payload now (post-confirmation / checkpoint)."""
        quota = quota if quota is not None else {}
        path = Path(str(payload["file"]))
        if not path.is_file():
            raise YouTubeError(f"no such file: {path}")
        total = path.stat().st_size
        mime = _mime_type(path)
        status: dict[str, Any] = {
            "privacyStatus": payload.get("privacy", "private"),
            "selfDeclaredMadeForKids": bool(
                payload.get("made_for_kids", False)),
        }
        if payload.get("publish_at"):
            status["publishAt"] = payload["publish_at"]
        metadata = {
            "snippet": {
                "title": payload["title"],
                "description": payload.get("description", ""),
                "tags": payload.get("tags", []),
                "categoryId": str(payload.get("category_id", "22")),
            },
            "status": status,
        }
        self._charge_quota("videos.insert", quota)
        session_uri = self.start_resumable_session(metadata, total, mime)
        video = self.upload_chunks(
            session_uri, path, total, mime,
            chunk_size=chunk_size, on_progress=on_progress,
        )
        video_id = str(video.get("id", ""))
        result: dict[str, Any] = {
            "video_id": video_id,
            "title": payload["title"],
            "privacy": status["privacyStatus"],
            "publish_at": payload.get("publish_at", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "quota_spent": dict(quota),
        }
        if (float(payload.get("duration_s", 0) or 0) > 0
                and int(payload.get("width", 0) or 0) > 0
                and int(payload.get("height", 0) or 0) > 0):
            result["shorts_candidate"] = self.is_shorts_candidate(
                float(payload["duration_s"]),
                int(payload["width"]),
                int(payload["height"]),
            )
        if payload.get("thumbnail"):
            self.set_thumbnail(video_id, payload["thumbnail"], quota=quota)
            result["thumbnail_set"] = True
        if payload.get("playlist_id"):
            self.add_to_playlist(video_id, payload["playlist_id"],
                                 quota=quota)
            result["playlist_id"] = payload["playlist_id"]
        result["quota_spent"] = dict(quota)
        result["quota_total"] = quota.get("total", 0)
        entry = None
        if ledger is not None:
            entry = ledger.record(
                platform="youtube",
                file=str(path),
                title=payload["title"],
                platform_video_id=video_id,
                status=("scheduled" if payload.get("publish_at")
                        else "uploaded"),
                quota_spent=result["quota_total"],
                privacy=status["privacyStatus"],
                publish_at=payload.get("publish_at", ""),
            )
            result["ledger_id"] = entry["id"]
        _log.info("youtube published %s (quota total %d)",
                  video_id, result["quota_total"])
        return result

    # ── extras ───────────────────────────────────────────────────

    def set_thumbnail(
        self,
        video_id: str,
        image_path: str | Path,
        *,
        quota: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Set a custom thumbnail (``thumbnails.set`` — 50 quota units)."""
        if not (video_id or "").strip():
            raise YouTubeError("video_id is required")
        img = Path(image_path)
        if not img.is_file():
            raise YouTubeError(f"no such thumbnail file: {img}")
        ctype = ("image/jpeg" if img.suffix.lower() in (".jpg", ".jpeg")
                 else "image/png")
        boundary = f"----nm-thumb-{int(time.time() * 1000)}"
        data = img.read_bytes()
        body = b"\r\n".join([
            f"--{boundary}".encode(),
            f"Content-Type: {ctype}".encode(),
            b"",
            data,
            f"--{boundary}--".encode(),
            b"",
        ])
        url = (f"{UPLOAD_BASE}/thumbnails/set?videoId={video_id}")
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
            raise YouTubeError(
                f"youtube thumbnails.set failed: {exc}"
            ) from exc
        if not resp.ok:
            raise YouTubeError(
                f"youtube thumbnails.set failed ({resp.status}"
                f"{', ' + self._api_reason(resp) if self._api_reason(resp) else ''}): "
                f"{resp.text[:300]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        if quota is not None:
            self._charge_quota("thumbnails.set", quota)
        _log.info("youtube thumbnail set for %s", video_id)
        return {"video_id": video_id, "thumbnail_set": True}

    def add_to_playlist(
        self,
        video_id: str,
        playlist_id: str,
        *,
        quota: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Add the video to a playlist (``playlistItems.insert`` — 50)."""
        if not (video_id or "").strip():
            raise YouTubeError("video_id is required")
        if not (playlist_id or "").strip():
            raise YouTubeError("playlist_id is required")
        try:
            resp = self.http.post_json(
                f"https://www.googleapis.com/youtube/v3/playlistItems"
                "?part=snippet",
                {
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id,
                        },
                    },
                },
                headers=self._headers(),
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise YouTubeError(
                f"youtube playlistItems.insert failed: {exc}"
            ) from exc
        if not resp.ok:
            raise YouTubeError(
                f"youtube playlistItems.insert failed ({resp.status}"
                f"{', ' + self._api_reason(resp) if self._api_reason(resp) else ''}): "
                f"{resp.text[:300]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        if quota is not None:
            self._charge_quota("playlistItems.insert", quota)
        _log.info("youtube: %s added to playlist %s", video_id, playlist_id)
        return {"video_id": video_id, "playlist_id": playlist_id}

    # ── checkpoint resume ────────────────────────────────────────

    def resume_checkpoint(  # type: ignore[override]
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
        client_secret: str | None = None,
    ) -> dict[str, Any]:
        """Resume after a human checkpoint (adds the resumable stage)."""
        from ....connectors.checkpoints import CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            from ....connectors.base import ConnectorError

            raise ConnectorError(
                f"checkpoint {checkpoint.id} is "
                f"{checkpoint.state.value}, not resolved"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage == "publish_resumable":
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("file"):
                from ....connectors.base import ConnectorError

                raise ConnectorError(
                    "the resolved checkpoint has no upload payload"
                )
            return self.publish_now(payload)
        return super().resume_checkpoint(
            checkpoint, db=db, context=context,
            client_secret=client_secret,
        )


#: Title/description limits re-exported for callers.
TITLE_LIMIT = PLATFORM_LIMITS["youtube"]["title"]
DESCRIPTION_LIMIT = PLATFORM_LIMITS["youtube"]["description"]
