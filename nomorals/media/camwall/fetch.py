"""Fetching camera frames — snapshots and MJPEG, over stdlib urllib.

The URLs come ONLY from the owner's cameras config — never from
chat text — so the fetch is the owner's own endpoints, LAN/private
included, by design. Three hard rules the fetch never breaks:

- NO redirects: a 3xx is ``bad_status``. A camera that redirects
  has its final URL put in the config; the wall never chases one.
- Byte cap: a frame past ``MAX_FRAME_BYTES`` is ``frame_too_large``.
  A camera serving an endless stream cannot grow the process.
- A frame must be a real JPEG (SOI + EOI markers) or it is
  ``bad_image`` — garbage is never stored as a frame.

Basic auth comes from the camera config (username/password).
Timeouts are per request, from the camera config.
"""

from __future__ import annotations

import base64
import urllib.error
import urllib.request
from typing import Any

from .errors import CamWallError

__all__ = [
    "MAX_FRAME_BYTES",
    "MJPEGReader",
    "fetch_mjpeg_frame",
    "fetch_snapshot",
    "is_jpeg",
]

#: Largest single frame the wall accepts — 8 MiB. An IP camera
#: snapshot is tens to hundreds of KiB; past this something is
#: wrong with the endpoint, not the frame.
MAX_FRAME_BYTES = 8 * 1024 * 1024

#: Largest MJPEG read budget for one frame attempt — 16 MiB of
#: stream before giving up on that camera for this tick.
MAX_MJPEG_READ_BYTES = 16 * 1024 * 1024

#: Read chunk for MJPEG streams.
_CHUNK = 64 * 1024

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def is_jpeg(data: bytes) -> bool:
    """True when ``data`` is a complete JPEG: SOI first, EOI last.

    The EOI check searches the tail (some cameras append a
    trailing CRLF after the marker); the SOI check is exact.
    """
    if len(data) < 4 or not data.startswith(_JPEG_SOI):
        return False
    tail = data[-16:]
    return _JPEG_EOI in tail


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects outright — a 3xx is a wall error, not a hop."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise CamWallError(
            f"camera redirected (HTTP {code}) to {newurl} — the wall "
            f"never follows redirects; put the final URL in the "
            f"cameras config",
            kind="bad_status")


def _auth_headers(username: str, password: str) -> dict[str, str]:
    if not username:
        return {}
    token = base64.b64encode(
        f"{username}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect())


def fetch_snapshot(url: str, *, timeout_s: float = 10.0,
                   username: str = "", password: str = "") -> bytes:
    """Fetch one snapshot JPEG. Returns the validated frame bytes.

    Refusals, by name: ``camera_unreachable`` (nothing listening),
    ``fetch_timeout``, ``bad_status`` (non-200 or a redirect),
    ``frame_too_large``, ``bad_image`` (not a JPEG).
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "Devon-camwall/1.0",
                      **_auth_headers(username, password)},
        method="GET")
    try:
        with _opener().open(request, timeout=timeout_s) as raw:
            if raw.status != 200:
                raise CamWallError(
                    f"camera answered HTTP {raw.status} for {url}",
                    kind="bad_status")
            data = raw.read(MAX_FRAME_BYTES + 1)
    except CamWallError:
        raise
    except urllib.error.HTTPError as exc:
        raise CamWallError(
            f"camera answered HTTP {exc.code} for {url}",
            kind="bad_status") from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise CamWallError(
                f"camera {url} timed out: {reason}",
                kind="fetch_timeout") from exc
        raise CamWallError(
            f"camera {url} unreachable: {reason}",
            kind="camera_unreachable") from exc
    except TimeoutError as exc:
        raise CamWallError(
            f"camera {url} timed out",
            kind="fetch_timeout") from exc
    except OSError as exc:
        raise CamWallError(
            f"camera {url} unreachable: {exc}",
            kind="camera_unreachable") from exc
    if len(data) > MAX_FRAME_BYTES:
        raise CamWallError(
            f"camera frame past the {MAX_FRAME_BYTES}-byte cap",
            kind="frame_too_large")
    if not is_jpeg(data):
        raise CamWallError(
            f"camera {url} did not return a JPEG "
            f"({len(data)} bytes, no SOI/EOI markers)",
            kind="bad_image")
    return data


class MJPEGReader:
    """Incremental MJPEG frame extractor.

    Feed the raw multipart stream in; complete JPEG frames come
    out. A frame is SOI..EOI in the byte stream — the boundary
    markers are framing, the JPEG markers are the truth, so a
    camera with sloppy boundaries still yields frames and a
    camera with garbage between frames never yields garbage.
    The buffer is bounded: past ``MAX_MJPEG_READ_BYTES`` of
    unparsed stream the oldest bytes are dropped, so a stream
    that never yields a frame cannot grow the process.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """Append stream bytes; return every complete frame found."""
        self._buf += data
        if len(self._buf) > MAX_MJPEG_READ_BYTES:
            del self._buf[:len(self._buf) - MAX_MJPEG_READ_BYTES]
        frames: list[bytes] = []
        while True:
            start = self._buf.find(_JPEG_SOI)
            if start < 0:
                self._buf.clear()
                break
            if start > 0:
                del self._buf[:start]
            end = self._buf.find(_JPEG_EOI, 2)
            if end < 0:
                break
            frame = bytes(self._buf[:end + 2])
            del self._buf[:end + 2]
            if is_jpeg(frame):
                frames.append(frame)
        return frames


def fetch_mjpeg_frame(url: str, *, timeout_s: float = 10.0,
                      username: str = "", password: str = "",
                      max_read_bytes: int = MAX_MJPEG_READ_BYTES
                      ) -> bytes:
    """First complete JPEG frame off an MJPEG stream.

    Opens the stream and reads chunks until one frame completes
    or the read budget runs out (``fetch_timeout``). Same refusal
    names as :func:`fetch_snapshot`.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "Devon-camwall/1.0",
                      **_auth_headers(username, password)},
        method="GET")
    reader = MJPEGReader()
    try:
        with _opener().open(request, timeout=timeout_s) as raw:
            if raw.status != 200:
                raise CamWallError(
                    f"camera answered HTTP {raw.status} for {url}",
                    kind="bad_status")
            read = 0
            while read < max_read_bytes:
                try:
                    chunk = raw.read(_CHUNK)
                except TimeoutError as exc:
                    raise CamWallError(
                        f"MJPEG stream {url} timed out",
                        kind="fetch_timeout") from exc
                if not chunk:
                    break
                read += len(chunk)
                frames = reader.feed(chunk)
                if frames:
                    return frames[0]
    except CamWallError:
        raise
    except urllib.error.HTTPError as exc:
        raise CamWallError(
            f"camera answered HTTP {exc.code} for {url}",
            kind="bad_status") from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise CamWallError(
                f"MJPEG stream {url} timed out: {reason}",
                kind="fetch_timeout") from exc
        raise CamWallError(
            f"MJPEG stream {url} unreachable: {reason}",
            kind="camera_unreachable") from exc
    except TimeoutError as exc:
        raise CamWallError(
            f"MJPEG stream {url} timed out",
            kind="fetch_timeout") from exc
    except OSError as exc:
        raise CamWallError(
            f"MJPEG stream {url} unreachable: {exc}",
            kind="camera_unreachable") from exc
    raise CamWallError(
        f"MJPEG stream {url} yielded no complete frame within "
        f"{max_read_bytes} bytes",
        kind="fetch_timeout")
