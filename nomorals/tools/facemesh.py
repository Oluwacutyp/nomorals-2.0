"""FaceMesh: MediaPipe Face Mesh capture — a camera recording
becomes a face take.

A thin surface over :mod:`nomorals.media.facemesh` — the
organ owns the detection, the landmark-to-blendshape mapping,
the take and the provision; the tool only routes arguments
and shapes the reply.

One take is one recording, direct, no hub: a video file (any
webcam or phone camera — this is the correction to B.23's
iPhone-only LiveLink capture) runs through Face Landmarker,
its 468 landmarks become the rig's 52 ARKit blendshape
weights plus the head bone, and the frames land as a face
take in the shared take registry — so ``facemcap drive``,
``render``, ``export`` and ``attach`` work on it with zero
duplication. Writes the model file on first use, so FS_WRITE.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.facemcap.registry import register_take
from ..media.facemesh import capture_image, capture_video, mediapipe_status

__all__ = ["register"]

_ACTIONS = ("capture", "landmarks", "status")


def register(registry: Any) -> None:
    """Attach the facemesh family to a registry."""
    @registry.register(
        "facemesh",
        description=(
            "MediaPipe Face Mesh capture — a camera recording "
            "becomes a face take. A video file (any webcam or "
            "phone camera, free, open, CPU) runs through Face "
            "Landmarker: its 468 landmarks become the rig's 52 "
            "ARKit blendshape weights plus the head bone, and "
            "the frames land as a face take in the shared take "
            "registry, so facemcap drive/render/export/attach "
            "work on it. Facemesh actions: capture=<run a video "
            "file through Face Mesh and register the take, "
            "sources=[path.mp4], node=<name>, params=<{"
            "max_frames}>>; landmarks=<extract landmarks and "
            "blendshapes from an image or the first frames of a "
            "video, sources=[path], params=<{max_frames}>>; "
            "status=<mediapipe/model/ffmpeg availability, no "
            "install>. action=<capture|landmarks|status>; "
            "sources=<list of paths>; node=<take name>."
        ),
        capability=Capability.FS_WRITE,
    )
    def facemesh_tool(action: str = "status",
                      sources: list[str] | None = None,
                      output: str = "", kind: str = "",
                      node: str = "", value: Any = None,
                      command: str = "",
                      params: dict | None = None,
                      title: str = "",
                      ) -> dict[str, Any]:
        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown facemesh action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        args = dict(params or {})
        sources = list(sources or [])
        name = (node or "").strip() or "default"
        if action == "status":
            return mediapipe_status()
        if not sources:
            raise ValueError(
                f"{action} needs sources=[path.mp4|path.jpg]")
        if action == "capture":
            max_frames = args.get("max_frames")
            result = capture_video(
                sources[0], name=name,
                max_frames=(int(max_frames)
                            if max_frames is not None else None))
            register_take(name, result.take)
            return dict(result.take.to_dict(), ok=True,
                        frames_read=result.frames_read,
                        frames_with_face=result.frames_with_face,
                        frames_skipped=result.frames_skipped)
        if action == "landmarks":
            path = sources[0]
            suffix = path.rsplit(".", 1)[-1].lower()
            if suffix in ("mp4", "mov", "mkv", "avi", "webm"):
                max_frames = int(args.get("max_frames") or 1)
                result = capture_video(
                    path, name=name, max_frames=max_frames)
                frames = []
                for frame in result.take.frames:
                    frames.append({
                        "time_s": frame.time_s,
                        "blendshapes": frame.blendshapes,
                        "bones": frame.bones})
                return {"source": path,
                        "landmarks": result.first_landmarks,
                        "frames": frames,
                        "frames_read": result.frames_read,
                        "frames_with_face":
                            result.frames_with_face,
                        "frames_skipped": result.frames_skipped}
            result = capture_image(path, name=name)
            frame = result.take.frames[0]
            return {"source": path,
                    "landmarks": result.first_landmarks,
                    "blendshapes": frame.blendshapes,
                    "bones": frame.bones,
                    "frames": 1}
        raise ValueError(f"unhandled facemesh action {action!r}")
