"""Builtin tools for Devon Studio — the real editing studio.

Wires automation (silence/scene/reframe/batch), keyframes, pro color
(LUTs/curves/wheels/auto-grade/shot-match), and the edit engine into
the spine so the brain reaches them from plain language:
"cut the silences out of this video", "reframe this for TikTok",
"grade this like that reference", "add a punch-zoom here".

Native-first: ffmpeg/OpenCV do the work. No fake edits, ever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

STUDIO_DIR = "studio"


def _out_dir(context: Any, kind: str) -> Path:
    base = Path(getattr(context, "workspace", None) or ".")
    d = base / STUDIO_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(registry: Any) -> None:
    """Attach the studio tools to a registry."""
    context = registry.context

    # ── automation ──────────────────────────────────────────────

    @registry.register(
        "studio_cut_silences",
        description=("Remove silent regions from a video/audio file, keeping "
                     "a padding around speech. ('cut the dead air out of this', "
                     "'remove silences'). Returns output path and seconds removed."),
        capability=Capability.FS_WRITE,
    )
    def studio_cut_silences(path: str, *, noise_db: float = -30.0,
                            min_duration: float = 0.5,
                            padding: float = 0.1) -> dict[str, Any]:
        """Auto-cut silences from media."""
        from ..media.studio_automation import cut_silences
        out = _out_dir(context, "automation") / (Path(path).stem + "_nosilence.mp4")
        return cut_silences(path, out=str(out), noise_db=noise_db,
                            min_duration=min_duration, padding=padding)

    @registry.register(
        "studio_detect_scenes",
        description=("Detect scene cuts in a video, returning a list of scenes "
                     "with start/end/duration. ('find the scenes', 'split this "
                     "into shots')."),
        capability=Capability.FS_READ,
    )
    def studio_detect_scenes(path: str, *, threshold: float = 0.4,
                             min_length: float = 1.0) -> dict[str, Any]:
        """Scene-cut detection."""
        from ..media.studio_automation import detect_scenes
        scenes = detect_scenes(path, threshold=threshold, min_length=min_length)
        return {"ok": True, "scenes": [
            {"index": s.index, "start": s.start, "end": s.end,
             "duration": s.duration} for s in scenes]}

    @registry.register(
        "studio_split_scenes",
        description=("Split a video into per-scene files. Returns the list of "
                     "scene files. ('split this video by scene')."),
        capability=Capability.FS_WRITE,
    )
    def studio_split_scenes(path: str, *, threshold: float = 0.4,
                            min_length: float = 1.0) -> dict[str, Any]:
        """Split video on scene cuts."""
        from ..media.studio_automation import split_video_on_scenes
        out = _out_dir(context, "scenes")
        return split_video_on_scenes(path, out_dir=str(out),
                                     threshold=threshold, min_length=min_length)

    @registry.register(
        "studio_reframe",
        description=("Reframe a video to a target aspect ratio (9:16, 1:1, "
                     "16:9), tracking faces/subjects to keep them centered. "
                     "('reframe this for TikTok', 'make this vertical')."),
        capability=Capability.FS_WRITE,
    )
    def studio_reframe(path: str, *, aspect: str = "9:16") -> dict[str, Any]:
        """Subject-tracked auto-reframe."""
        from ..media.studio_automation import auto_reframe
        out = _out_dir(context, "reframed") / (
            Path(path).stem + f"_{aspect.replace(':', 'x')}.mp4")
        return auto_reframe(path, out=str(out), aspect=aspect)

    @registry.register(
        "studio_rough_cut",
        description=("Full auto-edit pass: cut silences then list scenes. "
                     "The starting point for an edit. ('rough cut this video', "
                     "'auto-edit this')."),
        capability=Capability.FS_WRITE,
    )
    def studio_rough_cut(path: str) -> dict[str, Any]:
        """Silence cut + scene list in one pass."""
        from ..media.studio_automation import rough_cut
        out = _out_dir(context, "roughcuts") / (Path(path).stem + "_rough.mp4")
        return rough_cut(path, out=str(out))

    @registry.register(
        "studio_automation_batch",
        description=("Apply one automation op (cut_silences, auto_reframe) to "
                     "many files at once. ('cut silences from all of these')."),
        capability=Capability.FS_WRITE,
    )
    def studio_batch(paths: list[str], *, op: str,
                     aspect: str = "9:16") -> dict[str, Any]:
        """Batch automation."""
        from ..media.studio_automation import batch_process
        out = _out_dir(context, "batch")
        kw: dict[str, Any] = {"aspect": aspect} if op == "auto_reframe" else {}
        return batch_process(paths, op=op, out_dir=str(out), **kw)

    # ── color ───────────────────────────────────────────────────

    @registry.register(
        "studio_grade_lut",
        description=("Apply a 3D .cube LUT to a video. ('grade this with the "
                     "cinematic LUT', 'apply this look'). Options: LUT path, "
                     "strength 0-1."),
        capability=Capability.FS_WRITE,
    )
    def studio_grade_lut(path: str, lut: str, *,
                         strength: float = 1.0) -> dict[str, Any]:
        """3D LUT color grade."""
        from ..media.studio_color import apply_lut
        out = _out_dir(context, "graded") / (Path(path).stem + "_graded.mp4")
        return apply_lut(path, lut, out=str(out), strength=strength)

    @registry.register(
        "studio_grade_curves",
        description=("RGB curves grade. Points like '0/0 0.5/0.6 1/1' for "
                     "master/red/green/blue. ('lift the shadows', 'S-curve "
                     "this')."),
        capability=Capability.FS_WRITE,
    )
    def studio_grade_curves(path: str, *, master: str = "",
                            red: str = "", green: str = "",
                            blue: str = "") -> dict[str, Any]:
        """Curves color grade."""
        from ..media.studio_color import apply_curves
        out = _out_dir(context, "graded") / (Path(path).stem + "_curves.mp4")
        return apply_curves(path, master=master or None, red=red or None,
                            green=green or None, blue=blue or None,
                            out=str(out))

    @registry.register(
        "studio_grade_wheels",
        description=("Lift/gamma/gain color wheels per channel. lift shifts "
                     "shadows, gamma mids, gain highlights. ('warm up the "
                     "highlights', 'cool the shadows')."),
        capability=Capability.FS_WRITE,
    )
    def studio_grade_wheels(path: str, *,
                            lift_r: float = 0.0, lift_g: float = 0.0,
                            lift_b: float = 0.0,
                            gamma_r: float = 1.0, gamma_g: float = 1.0,
                            gamma_b: float = 1.0,
                            gain_r: float = 1.0, gain_g: float = 1.0,
                            gain_b: float = 1.0) -> dict[str, Any]:
        """Color wheels grade."""
        from ..media.studio_color import color_wheels
        out = _out_dir(context, "graded") / (Path(path).stem + "_wheels.mp4")
        return color_wheels(path,
                            lift=(lift_r, lift_g, lift_b),
                            gamma=(gamma_r, gamma_g, gamma_b),
                            gain=(gain_r, gain_g, gain_b),
                            out=str(out))

    @registry.register(
        "studio_auto_grade",
        description=("One-click balance: auto white-balance + exposure "
                     "normalize, measured from the frame. ('fix the colors', "
                     "'auto grade this')."),
        capability=Capability.FS_WRITE,
    )
    def studio_auto_grade(path: str) -> dict[str, Any]:
        """Automatic color balance."""
        from ..media.studio_color import auto_grade
        out = _out_dir(context, "graded") / (Path(path).stem + "_auto.mp4")
        return auto_grade(path, out=str(out))

    @registry.register(
        "studio_match_shot",
        description=("Match a video's color to a reference video/frame via "
                     "histogram transfer. Two angles, one look. ('match this "
                     "to that look', 'grade like the reference')."),
        capability=Capability.FS_WRITE,
    )
    def studio_match_shot(path: str, reference: str) -> dict[str, Any]:
        """Shot matching."""
        from ..media.studio_color import match_shot
        out = _out_dir(context, "graded") / (Path(path).stem + "_matched.mp4")
        return match_shot(path, reference, out=str(out))

    @registry.register(
        "studio_look",
        description=("Apply a named look preset (cinematic, teal_orange, noir, "
                     "vibrant, phonk). Styles live in presets; engines stay "
                     "agnostic. ('make this cinematic')."),
        capability=Capability.FS_WRITE,
    )
    def studio_look(path: str, look: str) -> dict[str, Any]:
        """Named look preset."""
        from ..media.studio_color import apply_look, list_looks
        if look not in list_looks():
            return {"ok": False,
                    "reason": f"unknown look; pick from {list_looks()}"}
        out = _out_dir(context, "graded") / (Path(path).stem + f"_{look}.mp4")
        return apply_look(path, look, out=str(out))

    # ── keyframes ───────────────────────────────────────────────

    @registry.register(
        "studio_keyframe",
        description=("Build a keyframe animation track: animate any numeric "
                     "effect param over time (zoom ramps, opacity fades). "
                     "Keys as 'time:value' pairs, e.g. ['0:1.0','2:1.5'] "
                     "with easing. Returns the track as JSON + ffmpeg filter."),
        capability=Capability.FS_READ,
    )
    def studio_keyframe(clip_id: str, effect: str, param: str,
                        keys: list[str], *,
                        easing: str = "smooth") -> dict[str, Any]:
        """Keyframe track builder."""
        from ..media.edit_engine.keyframes import KeyframeTrack, track_to_filter
        track = KeyframeTrack(clip_id=clip_id, effect=effect, param=param)
        for k in keys:
            try:
                t, v = k.split(":")
                track.add(float(t), float(v), easing=easing)
            except ValueError:
                return {"ok": False,
                        "reason": f"bad key {k!r}; use 'time:value'"}
        return {"ok": True, "track": track.to_dict(),
                "sample_filter": track_to_filter(track, f"eq=brightness={{v}}"),
                "note": "use sample_filter as a template; {v} is the value"}

    @registry.register(
        "studio_fade",
        description=("Build a fade in/out keyframe track for a clip. "
                     "('fade this in over 2 seconds')."),
        capability=Capability.FS_READ,
    )
    def studio_fade(clip_id: str, *, direction: str = "in",
                    start: float = 0.0,
                    duration: float = 1.0) -> dict[str, Any]:
        """Fade keyframe builder."""
        from ..media.edit_engine.keyframes import fade_in, fade_out
        track = fade_in(clip_id, duration) if direction == "in" \
            else fade_out(clip_id, start, duration)
        return {"ok": True, "track": track.to_dict()}

    @registry.register(
        "studio_zoom_ramp",
        description=("Build a punch-zoom keyframe track (1.0 → 1.5 by "
                     "default) between two times. ('zoom into this')."),
        capability=Capability.FS_READ,
    )
    def studio_zoom_ramp(clip_id: str, start: float, end: float, *,
                         z0: float = 1.0, z1: float = 1.5) -> dict[str, Any]:
        """Zoom ramp keyframe builder."""
        from ..media.edit_engine.keyframes import zoom_ramp
        track = zoom_ramp(clip_id, start, end, z0=z0, z1=z1)
        return {"ok": True, "track": track.to_dict()}

    # ── timeline ────────────────────────────────────────────────

    @registry.register(
        "studio_timeline_new",
        description=("Create a new edit timeline (multi-track: video, audio, "
                     "text). Returns the timeline as JSON. The foundation "
                     "every edit builds on."),
        capability=Capability.FS_READ,
    )
    def studio_timeline_new(name: str = "untitled") -> dict[str, Any]:
        """New timeline."""
        from ..media.edit_engine.timeline import Timeline, Track
        tl = Timeline(video=Track(kind="video"), text_layers=[],
                      audio_mix=None)
        d = tl.to_dict()
        d["name"] = name
        return {"ok": True, "timeline": d}

    @registry.register(
        "studio_render_timeline",
        description=("Render a timeline JSON to a video file. The timeline "
                     "comes from studio_timeline_new plus clip additions."),
        capability=Capability.FS_WRITE,
    )
    def studio_render_timeline(timeline: dict, *,
                               out_name: str = "timeline.mp4") -> dict[str, Any]:
        """Render a timeline."""
        from ..media.edit_engine.timeline import Timeline
        from ..media.edit_engine.render import render_timeline
        tl = Timeline.from_dict(timeline)
        out = _out_dir(context, "renders") / out_name
        return render_timeline(tl, out=str(out))
