"""X publisher — video upload (chunked) + tweet via X API v2.

Flow (verified 2026-10-09; X migrated chunked uploads to dedicated v2
endpoints in 2025, the old v1.1 ``upload.json`` command-param flow is
gone):

1. ``POST https://api.x.com/2/media/upload/initialize`` — JSON
   ``{total_bytes, media_type, media_category}`` → ``data.id``
2. ``POST https://api.x.com/2/media/upload/{id}/append`` — multipart
   ``media`` chunk + ``segment_index``
3. ``POST https://api.x.com/2/media/upload/{id}/finalize``
4. ``GET https://api.x.com/2/media/upload?media_id={id}&command=STATUS``
   — poll ``processing_info.state`` until ``succeeded`` (respecting
   ``check_after_secs``)
5. ``POST https://api.x.com/2/tweets`` — ``{text, media: {media_ids}}``

Auth: OAuth 2.0 USER context with ``tweet.write`` + ``media.write``
scopes (or OAuth 1.0a user context). App-only bearer tokens CANNOT
upload or post — this raises :class:`CapabilityUnavailable` instead of
failing cryptically.

The honest blocker: X write APIs require a PAID API tier (Basic/Pro on
https://developer.x.com). Free-tier keys are read-only. Without a user
access token every call raises :class:`CapabilityUnavailable` with the
exact manual step — never a silent no-op, never fake success.

Video limits for ``tweet_video``: ≤140 seconds, ≤512 MB. Longer videos
need the ``amplify_video`` category (not wired here — refused loudly
with the step, not half-posted).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from ....core.http import HttpClient
from ....core.logging_setup import get_logger
from . import PLATFORM_LIMITS, CapabilityUnavailable, adapt_description
from .ledger import PublishLedger

__all__ = ["XPublisher"]

_log = get_logger(__name__)

API_BASE = "https://api.x.com/2"

ACCESS_TOKEN_ENV = "X_ACCESS_TOKEN"

#: Chunk size for APPEND. 5 MiB raw per chunk.
_CHUNK_SIZE = 5 * 1024 * 1024

#: tweet_video limits (X's documented media limits).
_MAX_TWEET_VIDEO_SECONDS = 140
_MAX_VIDEO_BYTES = 512 * 1024 * 1024

_X_MANUAL_STEP = "\n".join([
    "X video posting setup (only you can do this — one time):",
    "1. Create an app at https://developer.x.com → Projects & Apps.",
    "2. Enable OAuth 2.0 with type 'Confidential' or use OAuth 1.0a;",
    "   request scopes tweet.write + media.write (user context).",
    "3. NOTE: X write APIs need a PAID API tier (Basic or Pro) — free",
    "   keys are read-only and uploads will 403.",
    "4. Authorize the app as YOUR X account (user context, not app-only",
    "   bearer token — app-only tokens cannot upload or post).",
    "5. Pass the user access token as X_ACCESS_TOKEN (or access_token=).",
])


class XPublisher:
    """Video tweets for one X account (user-context token)."""

    def __init__(
        self,
        *,
        access_token: str | None = None,
        http: Any | None = None,
    ) -> None:
        self.access_token = (access_token or
                             os.environ.get(ACCESS_TOKEN_ENV, "")).strip()
        self.http = http or HttpClient()

    # ── plumbing ─────────────────────────────────────────────────

    def _require_token(self) -> str:
        if not self.access_token:
            raise CapabilityUnavailable(
                "x",
                "no X user access token — the account isn't connected",
                manual_step=_X_MANUAL_STEP,
            )
        return self.access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._require_token()}"}

    def _data(self, resp: Any, what: str) -> dict[str, Any]:
        if not resp.ok:
            hint = ""
            if resp.status == 403:
                hint = (" — 403 on upload/post usually means the app is on "
                        "X's free tier (write needs Basic/Pro) or the token "
                        "is app-only instead of user context")
            raise CapabilityUnavailable(
                "x",
                f"X {what} failed ({resp.status}): {resp.text[:200]}{hint}",
                manual_step=_X_MANUAL_STEP,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise CapabilityUnavailable(
                "x", f"X {what} returned invalid JSON",
                manual_step=_X_MANUAL_STEP,
            ) from exc
        if not isinstance(body, dict):
            raise CapabilityUnavailable(
                "x", f"X {what} returned an unexpected response",
                manual_step=_X_MANUAL_STEP,
            )
        data = body.get("data")
        return data if isinstance(data, dict) else body

    # ── chunked upload ───────────────────────────────────────────

    def upload_video(
        self,
        file_path: str | Path,
        *,
        duration_s: float = 0,
        chunk_size: int = _CHUNK_SIZE,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        """Chunked video upload → media_id (INIT/APPEND/FINALIZE/STATUS).

        ``duration_s``: pass the probed duration when known; >140s is
        refused loudly (``tweet_video`` caps at 140s — ``amplify_video``
        is the longer-video route and isn't wired here).
        """
        path = Path(file_path)
        if not path.is_file():
            raise CapabilityUnavailable(
                "x", f"no such file: {path}",
                manual_step="Render the video first, then post it.",
            )
        total = path.stat().st_size
        if total > _MAX_VIDEO_BYTES:
            raise CapabilityUnavailable(
                "x",
                f"video is {total} bytes — X caps tweet_video at 512 MB",
                manual_step="Re-encode smaller (or use the amplify_video "
                            "route, which isn't wired here yet).",
            )
        if duration_s > _MAX_TWEET_VIDEO_SECONDS:
            raise CapabilityUnavailable(
                "x",
                f"video is {duration_s:.0f}s — X caps tweet_video at 140s",
                manual_step=("Trim to ≤140s, or post it as amplify_video "
                             "(longer-video category, not wired here yet)."),
            )
        init = self._data(
            self.http.post_json(
                f"{API_BASE}/media/upload/initialize",
                {
                    "total_bytes": total,
                    "media_type": "video/mp4",
                    "media_category": "tweet_video",
                },
                headers=self._headers(),
            ),
            "media initialize",
        )
        media_id = str(init.get("id", ""))
        if not media_id:
            raise CapabilityUnavailable(
                "x", "X initialize returned no media id",
                manual_step=_X_MANUAL_STEP,
            )
        _log.info("x media initialize ok: %s (%d bytes)", media_id, total)
        sent = 0
        segment = 0
        with path.open("rb") as fh:
            while sent < total:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                resp = self.http.request(
                    "POST",
                    f"{API_BASE}/media/upload/{media_id}/append",
                    data=self._multipart_chunk(chunk, segment),
                    headers={
                        **self._headers(),
                        "Content-Type": self._multipart_ct(),
                    },
                )
                if not resp.ok:
                    raise CapabilityUnavailable(
                        "x",
                        f"X media append failed (segment {segment}, "
                        f"{resp.status}): {resp.text[:200]}",
                        manual_step=_X_MANUAL_STEP,
                    )
                sent += len(chunk)
                segment += 1
                if on_progress:
                    on_progress(sent, total)
        final = self._data(
            self.http.request(
                "POST",
                f"{API_BASE}/media/upload/{media_id}/finalize",
                headers=self._headers(),
            ),
            "media finalize",
        )
        self._wait_for_processing(media_id, final)
        return media_id

    _boundary = "----nm-x-media"

    @classmethod
    def _multipart_ct(cls) -> str:
        return f"multipart/form-data; boundary={cls._boundary}"

    @classmethod
    def _multipart_chunk(cls, chunk: bytes, segment: int) -> bytes:
        b = cls._boundary
        return b"\r\n".join([
            f"--{b}".encode(),
            b'Content-Disposition: form-data; name="media"; '
            b'filename="chunk.mp4"',
            b"Content-Type: video/mp4",
            b"",
            chunk,
            f"--{b}".encode(),
            b'Content-Disposition: form-data; name="segment_index"',
            b"",
            str(segment).encode(),
            f"--{b}--".encode(),
            b"",
        ])

    def _wait_for_processing(
        self, media_id: str, final: dict[str, Any],
        timeout_s: float = 300.0,
    ) -> None:
        """Poll STATUS until processing_info.state == 'succeeded'."""
        info = final.get("processing_info") or {}
        deadline = time.time() + timeout_s
        while isinstance(info, dict) and info.get("state") not in (
                "", "succeeded"):
            state = str(info.get("state", ""))
            if state == "failed":
                err = info.get("error") or {}
                raise CapabilityUnavailable(
                    "x",
                    f"X failed to process the video: "
                    f"{err.get('message', state) if isinstance(err, dict) else state}",
                    manual_step=("Re-encode to X's spec (MP4/H.264, ≤140s, "
                                 "≤512MB) and retry."),
                )
            wait = info.get("check_after_secs", 5)
            try:
                wait_s = max(2.0, float(wait))
            except (TypeError, ValueError):
                wait_s = 5.0
            if time.time() + wait_s > deadline:
                raise CapabilityUnavailable(
                    "x",
                    f"X video processing still {state!r} after "
                    f"{timeout_s:.0f}s",
                    manual_step="Check the media in X's API dashboard, or "
                                "retry the upload.",
                )
            time.sleep(wait_s)
            status = self._data(
                self.http.request(
                    "GET",
                    f"{API_BASE}/media/upload",
                    params={"media_id": media_id, "command": "STATUS"},
                    headers=self._headers(),
                ),
                "media status",
            )
            info = status.get("processing_info") or {}
        _log.info("x media %s processing succeeded", media_id)

    # ── tweet ────────────────────────────────────────────────────

    def post_video(
        self,
        file_path: str | Path,
        text: str,
        *,
        tags: list[str] | None = None,
        duration_s: float = 0,
        ledger: PublishLedger | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Upload a video and post it as a tweet (280-char text)."""
        text = adapt_description("x", text, tags)
        media_id = self.upload_video(
            file_path, duration_s=duration_s, on_progress=on_progress)
        tweet = self._data(
            self.http.post_json(
                f"{API_BASE}/tweets",
                {"text": text, "media": {"media_ids": [media_id]}},
                headers=self._headers(),
            ),
            "tweet create",
        )
        tweet_id = str(tweet.get("id", ""))
        if not tweet_id:
            raise CapabilityUnavailable(
                "x", "X accepted the upload but returned no tweet id",
                manual_step=_X_MANUAL_STEP,
            )
        result = {"tweet_id": tweet_id, "media_id": media_id, "text": text}
        if ledger is not None:
            entry = ledger.record(
                platform="x", file=str(file_path),
                title=text[:80] or "(video)",
                platform_video_id=tweet_id, status="uploaded",
            )
            result["ledger_id"] = entry["id"]
        _log.info("x tweet posted: %s", tweet_id)
        return result


#: Text limit re-exported for callers.
TEXT_LIMIT = PLATFORM_LIMITS["x"]["text"]
