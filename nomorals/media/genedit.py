"""Generative video as EDITING operations — not separate toys.

Every op returns a clip file that drops into the edit timeline like any
footage: cut it, grade it, transition it, done.

- text_to_shot(prompt, duration) → t2v via LTX/Wan, motion fallback
- image_to_shot(image, prompt, duration) → i2v
- video_to_shot(video, prompt, strength) → v2v restyle: keyframe img2img +
  interpolation, reassembled. TRUE vid-to-vid, not extend-chaining.
- extend_shot(video, prompt, seconds) → LTX extend mode

All ops degrade honestly: neural needs CUDA+diffusers; motion studio
covers t2v on CPU; v2v restyles through the imggen neural pipeline
(Studio → ``imggen.edit.img2img``) when one can be built on this host,
and otherwise returns an explicitly-labelled ``passthrough`` shot —
never an unstyled copy dressed as a styled render.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .render_failures import record_failure


@dataclass
class GenShot:
    path: str               # the rendered clip file
    mode: str               # "t2v" | "i2v" | "v2v" | "extend" | "motion"
    prompt: str = ""
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    backend: str = ""       # "ltx" | "wan" | "motion" | "img2img" |
                            # "passthrough" (v2v NOT restyled — see note)
    note: str = ""          # user-visible honesty note: any downgrade,
                            # partial styling, or substitution is named
                            # here, never left implicit in the backend


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg")


def _out_path(workdir: str | None, stem: str) -> str:
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="genedit_"))
    wd.mkdir(parents=True, exist_ok=True)
    return str(wd / f"{stem}.mp4")


def _structure_prompt(prompt: str, backend: str,
                      duration_s: float) -> tuple[str, str]:
    """Run a plain prompt through the prompt engine. Always safe to call."""
    try:
        from .directed.prompt_engine import structure
        sp = structure(prompt, backend=backend, duration_s=duration_s)
        r = sp.render(backend)
        return r["prompt"], r["negative_prompt"]
    except Exception as e:
        record_failure("genedit._structure_prompt", e)
        return prompt, ""


def text_to_shot(prompt: str, *, duration_s: float = 5.0,
                 workdir: str | None = None,
                 structure_prompt: bool = True) -> GenShot:
    """Text → video shot. Neural when available, motion-studio on CPU.

    structure_prompt: run the plain prompt through the prompt engine
    (action-first, beats, camera, lighting, negatives) before the backend.
    """
    from .videogen.pipeline import generate
    out = _out_path(workdir, "t2v_shot")
    neg = ""
    if structure_prompt:
        prompt, neg = _structure_prompt(prompt, "ltx", duration_s)
    try:
        result = generate(prompt, duration_s=duration_s, out=out,
                          negative_prompt=neg or None)
        return GenShot(path=result.path, mode="t2v", prompt=prompt,
                       duration_s=duration_s, backend=result.engine)
    except Exception as exc:
        raise RuntimeError(f"text_to_shot failed: {exc}") from exc


def image_to_shot(image: str, prompt: str = "", *,
                  duration_s: float = 5.0,
                  workdir: str | None = None,
                  structure_prompt: bool = True) -> GenShot:
    """Image → video shot (i2v). Animates a still into motion."""
    if structure_prompt and prompt:
        prompt, _neg = _structure_prompt(prompt, "ltx", duration_s)
    from .videogen.ltx_backend import LTXBackend
    backend = LTXBackend()
    info = backend.check()
    out = _out_path(workdir, "i2v_shot")
    if info["available"]:
        clip = backend.generate(prompt, mode="i2v", image=image,
                                duration_s=duration_s, out=out)
        return GenShot(path=clip, mode="i2v", prompt=prompt,
                       duration_s=duration_s, backend="ltx")
    # CPU fallback: kenburns-style motion on the still
    from .motion_studio.kenburns import kenburns
    clip = kenburns(image, out=out, duration_s=duration_s)
    return GenShot(path=str(clip), mode="i2v", prompt=prompt,
                   duration_s=duration_s, backend="motion")


def _v2v_styler():
    """Resolve the neural frame styler for :func:`video_to_shot`.

    Returns ``(styler, reason)``. The styler restyles one frame through
    the REAL imggen path — :class:`imggen.studio.Studio` builds the
    pipeline and :func:`imggen.edit.img2img` (SDEdit) does the work:
    ``styler(src_path, prompt, strength, dst_path)``.

    When no neural pipeline can be built on this host (no torch, no
    loadable checkpoint, profile gate), the styler is ``None`` and
    ``reason`` names why — the caller labels the shot ``passthrough``
    and puts the reason in the user-visible note. There is no third
    state where frames are copied but the shot claims to be styled.
    """
    try:
        from .imggen.studio import Studio
        pipe = Studio().pipeline()
    except Exception as e:
        record_failure("genedit.video_to_shot:styler_unavailable", e)
        return None, f"{type(e).__name__}: {e}"
    from PIL import Image

    from .imggen.edit import img2img as _edit_img2img

    def _style(src: str, prompt: str, strength: float, dst: str) -> None:
        img = Image.open(src)
        try:
            out = _edit_img2img(pipe, img, prompt, strength=strength,
                                width=img.width, height=img.height)
        finally:
            img.close()
        out[0].save(dst)

    return _style, ""


def video_to_shot(video: str, prompt: str, *, strength: float = 0.6,
                  fps: int = 8, workdir: str | None = None) -> GenShot:
    """Video → video restyle. TRUE v2v when a neural pipeline exists.

    Extracts frames at low fps → img2img restyle on keyframes via the
    imggen pipeline → reassembles at original timing. Temporal coherence
    comes from low strength + consistent prompt; keyframe interpolation
    keeps it fast.

    Honesty contract: if no neural pipeline can be built, or every
    keyframe fails to style, the returned shot has
    ``backend="passthrough"`` and a ``note`` naming the reason — the
    frames are the source frames, unchanged, and the shot says so. A
    partial styling (some keyframes failed) keeps ``backend="img2img"``
    with the unstyled count named in the note.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found — v2v needs ffmpeg")
    video = str(video)
    if not os.path.exists(video):
        raise FileNotFoundError(video)

    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="v2v_"))
    wd.mkdir(parents=True, exist_ok=True)
    frames_dir = wd / "frames"
    frames_dir.mkdir(exist_ok=True)

    # 1. extract frames
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", video,
         "-vf", f"fps={fps}", str(frames_dir / "f_%04d.png")],
        check=True, capture_output=True, timeout=600)
    frames = sorted(frames_dir.glob("f_*.png"))
    if not frames:
        raise RuntimeError("no frames extracted")

    # 2. restyle: img2img on keyframes (every 4th), interpolate the rest
    # through the real imggen path (Studio → edit.img2img). No styler
    # on this host → the frames pass through unchanged and the shot is
    # labelled passthrough with the reason in its note.
    styler, styler_reason = _v2v_styler()

    styled_dir = wd / "styled"
    styled_dir.mkdir(exist_ok=True)
    key_idx = list(range(0, len(frames), 4))
    if not key_idx:
        key_idx = [0]
    styled_paths: dict[int, str] = {}
    styled_ok = 0
    first_key_error = ""
    for ki in key_idx:
        src = frames[ki]
        dst = styled_dir / f"s_{ki:04d}.png"
        if styler is not None:
            try:
                styler(str(src), prompt, strength, str(dst))
                styled_ok += 1
            except Exception as e:
                # This keyframe passes through unstyled; the failure is
                # recorded AND counted — the note below names how many
                # keyframes went unstyled, and if none styled the shot
                # is a passthrough, not an img2img render.
                record_failure("genedit.video_to_shot:keyframe_img2img", e)
                if not first_key_error:
                    first_key_error = f"{type(e).__name__}: {e}"
                import shutil
                shutil.copy(str(src), str(dst))
        else:
            import shutil
            shutil.copy(str(src), str(dst))
        styled_paths[ki] = str(dst)
    n_keys = len(key_idx)
    if styler is None:
        backend = "passthrough"
        note = (f"v2v NOT restyled — no neural image pipeline on this "
                f"host ({styler_reason}); frames passed through "
                f"unchanged")
    elif styled_ok == 0:
        backend = "passthrough"
        note = (f"v2v NOT restyled — all {n_keys} keyframes failed "
                f"img2img ({first_key_error}); frames passed through "
                f"unchanged")
    elif styled_ok < n_keys:
        backend = "img2img"
        note = (f"v2v partially styled — {n_keys - styled_ok} of "
                f"{n_keys} keyframes failed img2img and passed through "
                f"unstyled ({first_key_error})")
    else:
        backend = "img2img"
        note = ""

    # 3. fill non-keyframes: nearest styled keyframe (temporal coherence
    #    via low strength keeps flicker acceptable; full interpolation
    #    is the refinement path)
    for i in range(len(frames)):
        if i not in styled_paths:
            nearest = min(key_idx, key=lambda k: abs(k - i))
            import shutil
            dst = styled_dir / f"s_{i:04d}.png"
            shutil.copy(styled_paths[nearest], str(dst))
            styled_paths[i] = str(dst)

    # 4. reassemble (+ original audio)
    out = _out_path(str(wd), "v2v_shot")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(fps), "-i", str(styled_dir / "s_%04d.png"),
         "-i", video,
         "-map", "0:v", "-map", "1:a?", "-c:a", "aac",
         "-pix_fmt", "yuv420p", "-shortest", out],
        check=True, capture_output=True, timeout=900)
    # duration from probe
    duration_s = len(frames) / fps
    return GenShot(path=out, mode="v2v", prompt=prompt,
                   duration_s=round(duration_s, 2),
                   backend=backend, note=note)


def extend_shot(video: str, prompt: str = "", *, seconds: float = 4.0,
                workdir: str | None = None) -> GenShot:
    """Extend a shot by N seconds (LTX extend mode, seeded from last frame)."""
    from .videogen.ltx_backend import LTXBackend
    backend = LTXBackend()
    info = backend.check()
    if not info["available"]:
        raise RuntimeError(
            f"extend needs LTX (CUDA): {info.get('reason', 'unavailable')}")
    # seed = last frame of the clip
    ff = _ffmpeg()
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ext_"))
    wd.mkdir(parents=True, exist_ok=True)
    seed = str(wd / "seed.png")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-sseof", "-0.1", "-i", str(video), "-frames:v", "1", seed],
        check=True, capture_output=True, timeout=60)
    out = _out_path(str(wd), "extend_shot")
    clip = backend.generate(prompt, mode="extend", image=seed,
                            duration_s=seconds, out=out)
    return GenShot(path=clip, mode="extend", prompt=prompt,
                   duration_s=seconds, backend="ltx")


def shot_to_timeline(shot: GenShot, timeline_path: str, *,
                     at: int = -1, transition: str = "") -> str:
    """Drop a generated shot into an edit timeline file.

    Loads the timeline, appends/inserts a Clip for the generated shot,
    saves back. The shot is graded/cut/transitioned like any footage.
    """
    import json
    from .edit_engine.timeline import Timeline, Clip
    with open(timeline_path) as f:
        tl = Timeline.from_dict(json.load(f))
    clip = Clip(path=shot.path)
    if at < 0 or at >= len(tl.video.clips):
        tl.video.clips.append(clip)
    else:
        tl.video.clips.insert(at, clip)
    with open(timeline_path, "w") as f:
        # Timeline is a dataclass — serialize via its components
        import dataclasses
        f.write(json.dumps(dataclasses.asdict(tl), indent=2, default=str))
    return timeline_path
