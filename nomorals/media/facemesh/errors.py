"""Face mesh errors — every failure names itself.

Kinds:
  missing_mediapipe   mediapipe is not importable and the install
                      attempt failed (or installs were refused)
  provision_failed    the pip install of mediapipe failed
  missing_model       the face-landmarker model file is absent and
                      the download failed (or downloads refused)
  model_unavailable   the model file is present but unreadable
  missing_ffmpeg      no ffmpeg binary to decode video with
  missing_video       no video file at the given path
  bad_video           ffmpeg could not decode the video
  no_face             no face found in a frame / the whole video
  bad_landmarks       a landmark set is not 468 (or 478) points
  bad_frame           a frame is not an RGB image of the stated size
  camera_unavailable  no live camera path exists on this machine
                      (live capture is the next rung, not built)
"""

from __future__ import annotations

__all__ = ["FaceMeshError"]


class FaceMeshError(Exception):
    """A facemesh failure with a named kind."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind
