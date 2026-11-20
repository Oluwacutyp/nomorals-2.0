"""Cameras wall errors — every refusal named, never a silent failure.

The port (echelon/ip-security-camera-viewer, MIT) fails the same way:
a camera that cannot be reached is a named refusal with the reason,
not an empty frame and not a crash of the wall.
"""

from __future__ import annotations

__all__ = ["CamWallError"]


class CamWallError(Exception):
    """A cameras-wall failure with a machine-readable kind.

    Kinds:
    - ``bad_config`` — the cameras config is not a dict/list of
      cameras, or a camera entry is missing its URL form.
    - ``no_cameras`` — the config loaded but names no cameras.
    - ``bad_url`` — a camera URL is not http(s), has no host, or
      has both a full-url and a host+path form.
    - ``unknown_camera`` — no camera by that name/id in this wall.
    - ``camera_unreachable`` — the fetch could not connect at all.
    - ``fetch_timeout`` — the camera did not answer in time.
    - ``bad_status`` — the camera answered with a non-200 status
      (or a redirect — the wall never follows one).
    - ``frame_too_large`` — the frame passed the byte cap.
    - ``bad_image`` — the bytes are not a JPEG.
    - ``no_frame`` — the camera has staged no frame yet.
    - ``no_wall`` — no wall by that name is loaded.
    - ``out_not_empty`` — a web export refuses a non-empty dir.
    - ``bad_run`` — a bad tick count / refresh interval.
    - ``bad_size`` — a bad export size.
    - ``bad_demo`` — serve needs an exported wall directory.
    - ``bad_port`` — a bad serve port.
    - ``not_serving`` — no server serves that directory.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind
