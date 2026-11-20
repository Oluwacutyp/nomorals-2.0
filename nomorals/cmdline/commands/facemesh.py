"""``nm facemesh`` — MediaPipe Face Mesh capture, as face
takes, live from the terminal.

Thin CLI over the facemesh organ: the same takes the
``facemesh`` brain tool captures, from the terminal. A video
file (any webcam or phone camera) runs through Face
Landmarker; its 468 landmarks become the rig's 52 ARKit
blendshape weights plus the head bone, and the frames land
as a face take in the shared take registry — so
``nm facemcap drive`` works on them.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.facemcap.registry import register_take
from ...media.facemesh import FaceMeshError, capture_image, capture_video, mediapipe_status


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    else:
        print(obj)
    return 0


def _cmd_facemesh(args: argparse.Namespace, context: Any) -> int:
    """Route `nm facemesh` subcommands."""
    action = args.facemesh_action
    name = (getattr(args, "session", "") or "").strip() or "default"
    try:
        if action == "status":
            return _out(args, mediapipe_status())
        if action == "capture":
            if not args.path:
                print("capture needs --path PATH.mp4",
                      file=sys.stderr)
                return 2
            result = capture_video(
                args.path, name=name,
                max_frames=(args.max_frames
                            if args.max_frames is not None
                            else None))
            register_take(name, result.take)
            return _out(args, dict(
                result.take.to_dict(), ok=True,
                frames_read=result.frames_read,
                frames_with_face=result.frames_with_face,
                frames_skipped=result.frames_skipped))
        if action == "landmarks":
            if not args.path:
                print("landmarks needs --path PATH.mp4|PATH.jpg",
                      file=sys.stderr)
                return 2
            suffix = args.path.rsplit(".", 1)[-1].lower()
            if suffix in ("mp4", "mov", "mkv", "avi", "webm"):
                result = capture_video(
                    args.path, name=name,
                    max_frames=int(args.max_frames or 1))
                return _out(args, {
                    "source": args.path,
                    "landmarks": result.first_landmarks,
                    "frames": [{
                        "time_s": frame.time_s,
                        "blendshapes": frame.blendshapes,
                        "bones": frame.bones}
                        for frame in result.take.frames],
                    "frames_read": result.frames_read,
                    "frames_with_face": result.frames_with_face,
                    "frames_skipped": result.frames_skipped})
            result = capture_image(args.path, name=name)
            frame = result.take.frames[0]
            return _out(args, {
                "source": args.path,
                "landmarks": result.first_landmarks,
                "blendshapes": frame.blendshapes,
                "bones": frame.bones,
                "frames": 1})
        print(f"unknown facemesh action {action!r}",
              file=sys.stderr)
        return 2
    except FaceMeshError as exc:
        print(f"facemesh: {exc} [{exc.kind}]", file=sys.stderr)
        return 1
