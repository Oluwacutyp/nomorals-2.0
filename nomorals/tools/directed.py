"""Directed video animation tools — plain-language access to the spine.

- animate_photo: "make the person raise two fingers, selfie style" just works.
- camera_look: apply a camera look to any footage.
- list_actions: the directed action vocabulary.
"""

from __future__ import annotations

from typing import Any


def register(registry: Any) -> None:
    from ..core.policy import Capability

    @registry.register(
        "animate_photo",
        description=(
            "Animate a still photo with a DIRECTED action. ('make the person "
            "in this photo raise two fingers', 'animate her waving, phone "
            "selfie style'). The action is directed — the person performs "
            "that specific motion, not random movement. Optional camera "
            "look: phone_selfie, phone, handheld, cctv, dashcam, cinema. "
            "Returns the video path."
        ),
        capability=Capability.MEDIA,
    )
    def animate_photo(image: str, action: str, *, camera: str = "",
                      duration_s: float = 4.0,
                      workdir: str | None = None) -> dict[str, Any]:
        from ..media.directed.pose_rig import (
            build_track, resolve_action, list_actions)
        from ..media.directed.animator import direct_animate, ModelUnavailable
        key = resolve_action(action)
        if key is None:
            return {"ok": False,
                    "error": f"don't know how to direct {action!r}",
                    "known_actions": list_actions()}
        n_frames = max(8, int(duration_s * 10))
        track = build_track(key, n_frames=n_frames, fps=10.0)
        try:
            res = direct_animate(image, track, workdir=workdir)
        except ModelUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        path, backend, note = res.path, res.backend, res.note
        if camera:
            from ..media.directed.camera import camera_look_video, LOOKS
            if camera not in LOOKS:
                return {"ok": False,
                        "error": f"unknown camera look {camera!r}",
                        "looks": LOOKS}
            import os
            styled = os.path.splitext(path)[0] + f"_{camera}.mp4"
            path = camera_look_video(path, camera, styled, fps=10)
            note += f" + {camera} look"
        return {"ok": True, "path": path, "backend": backend,
                "action": key, "note": note}

    @registry.register(
        "camera_look",
        description=(
            "Apply a camera look to any video file. ('make this look shot "
            "on a phone', 'cctv footage style', 'cinema look'). Looks: "
            "phone_selfie (front-cam selfie video), phone, handheld, cctv, "
            "dashcam, cinema, tripod. Real post-process, works on generated "
            "or real footage."
        ),
        capability=Capability.MEDIA,
    )
    def camera_look(video: str, look: str, *,
                    out_path: str = "") -> dict[str, Any]:
        from ..media.directed.camera import camera_look_video, LOOKS
        if look not in LOOKS:
            return {"ok": False, "error": f"unknown look {look!r}",
                    "looks": LOOKS}
        import os
        out = out_path or os.path.splitext(video)[0] + f"_{look}.mp4"
        try:
            path = camera_look_video(video, look, out)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": path, "look": look}

    @registry.register(
        "list_actions",
        description=("List the directed animation action vocabulary "
                     "('what can you make the person do')."),
        capability=Capability.MEDIA,
    )
    def list_actions() -> dict[str, Any]:
        from ..media.directed.pose_rig import list_actions as _la
        return {"ok": True, "actions": _la()}

    @registry.register(
        "direct_to_timeline",
        description=("Drop a directed-animation clip into an edit timeline "
                     "file (grading/cutting like any footage)."),
        capability=Capability.MEDIA,
    )
    def direct_to_timeline(shot_path: str, timeline_path: str, *,
                           at: int = -1) -> dict[str, Any]:
        from ..media.genedit import shot_to_timeline, GenShot
        shot = GenShot(path=shot_path, mode="directed", backend="directed")
        try:
            path = shot_to_timeline(shot, timeline_path, at=at)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "timeline": path}

    # ── lip sync ─────────────────────────────────────────────────────

    @registry.register(
        "lip_sync",
        description=(
            "Lip-sync a video to new audio — the subject's lips move in "
            "sync with the audio. ('sync this video to the new voiceover', "
            "'make her lips match this audio'). Neural when a GPU backend "
            "is installed (Wav2Lip/LatentSync), honest CPU envelope warp "
            "otherwise. face_box is (x0,y0,x1,y1) normalized; omit it and "
            "the tool estimates from the pose track when animating, or "
            "asks."
        ),
        capability=Capability.MEDIA,
    )
    def lip_sync(video: str, audio: str, *,
                 face_box: list[float] | None = None,
                 prefer: str = "auto") -> dict[str, Any]:
        from ..media.directed.lipsync import (
            lip_sync as _ls, ModelUnavailable)
        try:
            res = _ls(video, audio,
                      face_box=tuple(face_box) if face_box else None,
                      prefer=prefer)
        except ModelUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": res.path, "backend": res.backend,
                "note": res.note}

    @registry.register(
        "talk_photo",
        description=(
            "Animate a still photo into a talking head from audio. "
            "('make this photo speak this audio'). SadTalker backend: full "
            "head motion + lip movement. Needs GPU + SadTalker install; "
            "honest error otherwise."
        ),
        capability=Capability.MEDIA,
    )
    def talk_photo(image: str, audio: str) -> dict[str, Any]:
        from ..media.directed.lipsync import (
            sadtaker_animate, ModelUnavailable)
        try:
            res = sadtaker_animate(image, audio)
        except ModelUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": res.path, "backend": res.backend,
                "note": res.note}

    @registry.register(
        "dub_video",
        description=(
            "Dub a video: translated text is spoken in a catalogue voice, "
            "then lip-synced to the footage. ('dub this video into Yoruba "
            "with my voice'). Pass the already-translated text — the brain "
            "translates, this tool voices + syncs + muxes."
        ),
        capability=Capability.MEDIA,
    )
    def dub_video(video: str, translated_text: str, voice: str, *,
                  face_box: list[float] | None = None,
                  prefer: str = "auto") -> dict[str, Any]:
        from ..media.directed.lipsync import (
            dub_video as _dub, ModelUnavailable)
        try:
            res = _dub(video, translated_text, voice,
                       face_box=tuple(face_box) if face_box else None,
                       prefer=prefer)
        except ModelUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": res.path, "backend": res.backend,
                "note": res.note}
