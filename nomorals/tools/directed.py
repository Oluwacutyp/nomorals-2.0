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
        "structure_prompt",
        description=(
            "Run a plain description through the prompt engine -> "
            "model-optimized structured prompt (action-first, time-stamped "
            "beats, camera specs, lighting, physical details, negatives). "
            "('make this prompt hit harder for LTX'). backend: ltx | wan "
            "| motion | image."
        ),
        capability=Capability.MEDIA,
    )
    def structure_prompt(plain: str, *, backend: str = "ltx",
                         duration_s: float = 5.0) -> dict[str, Any]:
        from ..media.directed.prompt_engine import structure
        try:
            sp = structure(plain, backend=backend, duration_s=duration_s)
            r = sp.render(backend)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "prompt": r["prompt"],
                "negative_prompt": r["negative_prompt"],
                "beats": sp.beats, "backend": r["backend"]}

    @registry.register(
        "edit_image",
        description=(
            "AI image editing arsenal, one op at a time. op: inpaint | "
            "outpaint | style | faceswap | bgreplace | remove | add | "
            "relight | expression. ('remove the trash can from this photo', "
            "'swap this face onto that photo', 'relight this portrait "
            "rembrandt style'). mask_path marks the region (white = edit) "
            "for inpaint/remove/add. Honest backend labels; neural ops "
            "raise a clear error when the model isn't installed."
        ),
        capability=Capability.MEDIA,
    )
    def edit_image(image: str, op: str, *,
                   mask_path: str = "",
                   prompt: str = "",
                   source_path: str = "",
                   bg_path: str = "",
                   direction: str = "left",
                   warmth: float = 0.0,
                   width: int = 0, height: int = 0) -> dict[str, Any]:
        from PIL import Image as _I
        from ..media.directed import ai_edit as _ae
        try:
            img = _I.open(image).convert("RGB")
            mask = _I.open(mask_path).convert("L") if mask_path else None
            o = (op or "").lower().strip()
            if o == "inpaint":
                if mask is None:
                    return {"ok": False, "error":
                            "inpaint needs mask_path (white = fill region)"}
                out, backend = _ae.inpaint(img, mask, prompt)
            elif o == "outpaint":
                out, backend = _ae.outpaint(img, width or img.width * 2,
                                            height or img.height,
                                            prompt=prompt)
            elif o == "style":
                out, backend = _ae.style_transfer(img, prompt)
            elif o == "faceswap":
                if not source_path:
                    return {"ok": False, "error":
                            "faceswap needs source_path (the face to use)"}
                out, backend = _ae.face_swap(_I.open(source_path), img)
            elif o == "bgreplace":
                if not bg_path or mask is None:
                    return {"ok": False, "error":
                            "bgreplace needs bg_path + mask_path"}
                out, backend = _ae.background_replace(
                    img, _I.open(bg_path), mask)
            elif o == "remove":
                if mask is None:
                    return {"ok": False, "error":
                            "remove needs mask_path (white = object)"}
                out, backend = _ae.object_removal(img, mask, prompt=prompt)
            elif o == "add":
                if mask is None or not prompt:
                    return {"ok": False, "error":
                            "add needs mask_path + prompt (what to add)"}
                out, backend = _ae.object_addition(img, mask, prompt)
            elif o == "relight":
                out, backend = _ae.relight_photo(img, direction, warmth)
            elif o == "expression":
                if not source_path:
                    return {"ok": False, "error":
                            "expression needs source_path (expression ref)"}
                out, backend = _ae.expression_edit(
                    img, _I.open(source_path))
            else:
                return {"ok": False, "error":
                        f"unknown op {op!r}"}
            from pathlib import Path as _P
            import tempfile as _t
            out_path = str(_P(_t.mkdtemp(prefix="edit_")) / f"{o}.png")
            out.save(out_path)
        except _ae.ModelUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": out_path, "op": o,
                "backend": backend}

    @registry.register(
        "fill_media",
        description=(
            "img2img filler: synthesize in-between frames (fill_frames), "
            "extend an image's background (extend_bg), or repair a dropped "
            "video segment (fill_gap). ('fill the gap between 2s and 4s', "
            "'make this photo wider')."
        ),
        capability=Capability.MEDIA,
    )
    def fill_media(kind: str, *, image: str = "", video: str = "",
                   t0: float = 0.0, t1: float = 0.0,
                   width: int = 0, height: int = 0,
                   prompt: str = "") -> dict[str, Any]:
        from ..media.directed import filler as _f
        try:
            k = (kind or "").lower().strip()
            if k == "extend_bg":
                from PIL import Image as _I
                out, backend = _f.extend_background(
                    _I.open(image).convert("RGB"), width, height,
                    prompt=prompt)
                import tempfile as _t
                from pathlib import Path as _P
                p = str(_P(_t.mkdtemp(prefix="fill_")) / "extended.png")
                out.save(p)
                return {"ok": True, "path": p, "backend": backend}
            if k == "fill_gap":
                p = _f.fill_video_gap(video, t0, t1)
                return {"ok": True, "path": p, "backend": "morph-bridge"}
            return {"ok": False, "error": f"unknown kind {kind!r}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

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
