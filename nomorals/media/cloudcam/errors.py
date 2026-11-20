"""CloudCam errors — one exception, named kinds, never silent."""

from __future__ import annotations

__all__ = ["CloudCamError"]


class CloudCamError(Exception):
    """A streamed point-cloud failure that names what went wrong.

    ``kind`` is the machine-readable cause (``missing_pyzmq``,
    ``no_frame``, ``bad_packet``, ``unknown_session`` …); the
    message carries the human-readable detail. Callers refuse by
    name; nothing in this organ fails silently.
    """

    def __init__(self, message: str, *, kind: str = "cloudcam_error") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
