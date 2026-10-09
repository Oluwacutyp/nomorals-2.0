"""Style-agnostic render: Timeline/EditSpec → ffmpeg filter graph → mp4.

Migrated proven machinery (frame-quantized cuts, windowed effects,
velocity ramps, still materialization) with the style-specific beat
logic lifted out — cut grids arrive as data (``cut_points``), never as
assumptions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...core.profile import detect_profile
from ...media_edit.videos import (
    MediaEditError,
    run_ffmpeg,
    video_probe,
)
from .audio import AudioLayer, AudioMix, mix_layers
from .effects import _ctx, _effect_filter, EFFECTS
from .text import TextLayer, burn_text_layers
from .timeline import Clip, Effect, Timeline, Track, Transition, canonical_effect
from .transitions import (
    edge_fade_filters,
    glitch_burst_params,
    is_overlap,
    validate_transition,
    xfade_args,
)

_log = get_logger(__name__)

WIDTH = 1080
HEIGHT = 1920

#: profile kind → encode knobs (speed only — never features)
_PROFILE_PERF: dict[str, dict[str, Any]] = {
    "termux": {"preset": "ultrafast", "crf": 23, "threads": 2},
    "mobile": {"preset": "ultrafast", "crf": 23, "threads": 2},
    "embedded": {"preset": "ultrafast", "crf": 24, "threads": 1},
    "pc": {"preset": "veryfast", "crf": 21, "threads": 0},
    "vps": {"preset": "veryfast", "crf": 21, "threads": 0},
    "workstation": {"preset": "medium", "crf": 19, "threads": 0},
}


def _perf_for(profile: str | None) -> dict[str, Any]:
    kind = (profile or "").strip().lower()
    if not kind:
        try:
            kind = detect_profile().kind
        except Exception:  # noqa: BLE001 — detection must never break a render
            kind = "pc"
    perf = _PROFILE_PERF.get(kind, _PROFILE_PERF["pc"])
    return {"profile": kind, **perf}


# ---------------------------------------------------------------------------
# chain machinery — apply effects (optionally windowed) to a labelled stream
# ---------------------------------------------------------------------------

def _chain_effect(parts: list[str], cur: str, fx: Effect,
                  ctx: dict[str, Any], tag: str) -> str:
    """Append filters applying ``fx`` to stream ``cur``; return new label.

    Windowed effects (``fx.at``) split the stream into head/window/tail,
    filter the window, and concat — works for every filter, including the
    ones without timeline support (noise, vignette, ...).
    """
    if fx.name == "velocity_ramp":
        return _chain_velocity(parts, cur, fx, ctx, tag)
    if fx.name == "rgb_split" and (fx.params or {}).get("animate"):
        return _chain_rgb_animated(parts, cur, fx, ctx, tag)
    filt = _effect_filter(fx.name, fx.params, ctx)
    at = fx.at
    if not at:
        parts.append(f"[{cur}]{filt}[{tag}]")
        return tag
    a, b = float(at[0]), float(at[1])
    dur = ctx["duration"]
    a, b = max(0.0, a), min(dur, b)
    if not a < b:
        raise MediaEditError(
            f"effect window at={(at[0], at[1])} is empty for a {dur:.2f}s clip")
    eps = 0.001
    segs: list[tuple[str, str]] = []  # (trim-args, piece-tag)
    if a > eps:
        segs.append((f"start=0:end={a:.4f}", f"{tag}p0"))
    segs.append((f"start={a:.4f}:end={b:.4f}", f"{tag}p1"))
    if b < dur - eps:
        segs.append((f"start={b:.4f}", f"{tag}p2"))
    if len(segs) == 1:
        parts.append(f"[{cur}]trim={segs[0][0]},setpts=PTS-STARTPTS,"
                     f"{filt}[{tag}]")
        return tag
    outs = "".join(f"[{tag}s{i}]" for i in range(len(segs)))
    parts.append(f"[{cur}]split={len(segs)}{outs}")
    srcs = [f"{tag}s{i}" for i in range(len(segs))]
    concat_in = ""
    for (trim_args, ptag), s in zip(segs, srcs):
        extra = f",{filt}" if ptag == f"{tag}p1" else ""
        parts.append(f"[{s}]trim={trim_args},setpts=PTS-STARTPTS{extra}"
                     f"[{ptag}]")
        concat_in += f"[{ptag}]"
    parts.append(f"{concat_in}concat=n={len(segs)}:v=1:a=0[{tag}]")
    return tag


def _chain_rgb_animated(parts: list[str], cur: str, fx: Effect,
                        ctx: dict[str, Any], tag: str) -> str:
    """Animated rgb_split via discretized constant shifts.

    This ffmpeg build's rgbashift rejects per-frame expressions, so the
    sine envelope is sampled into windowed constant-shift sub-effects —
    all through the proven windowed chain machinery. Deterministic per
    seed, visually a smooth wobble at ≥8 steps/period.
    """
    import math
    from .effects import _seed_phases
    params = fx.params or {}
    shift = float(params.get("shift", 10.0))
    period = float(params.get("period", 0.5))
    dur = ctx["duration"]
    if dur <= 0:
        raise MediaEditError("rgb_split animate needs a positive duration")
    s1, _ = _seed_phases(int(ctx["seed"]) + 7)
    steps = max(4, min(32, int(dur / max(period, 1e-3) * 8) or 4))
    base_at = fx.at
    a0 = float(base_at[0]) if base_at else 0.0
    b0 = float(base_at[1]) if base_at else dur
    span = b0 - a0
    for i in range(steps):
        a = a0 + span * i / steps
        b = a0 + span * (i + 1) / steps
        # sample the envelope at the sub-window midpoint
        tmid = (a + b) / 2.0 - a0
        env = shift * math.sin(2 * math.pi * tmid / period + s1)
        sub = Effect(name="rgb_split", params={"shift": round(env, 2)},
                     at=(a, b))
        cur = _chain_effect(parts, cur, sub, ctx, f"{tag}q{i}")
    return cur


def _chain_velocity(parts: list[str], cur: str, fx: Effect,
                    ctx: dict[str, Any], tag: str) -> str:
    """velocity_ramp: piecewise setpts — the one structural effect."""
    dur = ctx["duration"]
    params = fx.params or {}
    if "ramps" in params:
        ramps = [(float(t0), float(t1), float(s))
                 for t0, t1, s in params["ramps"]]
    else:
        speed = float(params.get("speed", 2.0))
        at = fx.at or (0.0, dur)
        ramps = [(float(at[0]), float(at[1]), speed)]
    if not ramps:
        raise MediaEditError("velocity_ramp needs speed= or ramps=")
    # full-coverage timeline: identity outside ramp windows
    bounds = sorted({0.0, dur} | {t for r in ramps for t in (r[0], r[1])})
    pieces: list[tuple[float, float, float]] = []
    for x0, x1 in zip(bounds, bounds[1:]):
        if x1 - x0 < 0.001:
            continue
        speed = 1.0
        for t0, t1, s in ramps:
            if t0 <= x0 + 1e-6 and x1 <= t1 + 1e-6:
                speed = s
                break
        if speed <= 0:
            raise MediaEditError("velocity_ramp speed must be > 0")
        pieces.append((x0, x1, speed))
    if len(pieces) == 1 and pieces[0][2] == 1.0:
        parts.append(f"[{cur}]null[{tag}]")
        return tag
    outs = "".join(f"[{tag}v{i}]" for i in range(len(pieces)))
    if len(pieces) > 1:
        parts.append(f"[{cur}]split={len(pieces)}{outs}")
        srcs = [f"{tag}v{i}" for i in range(len(pieces))]
    else:
        srcs = [cur]
    concat_in = ""
    for i, ((x0, x1, s), s_lbl) in enumerate(zip(pieces, srcs)):
        parts.append(f"[{s_lbl}]trim=start={x0:.4f}:end={x1:.4f},"
                     f"setpts=(PTS-STARTPTS)/{s:.4f},settb=AVTB[{tag}w{i}]")
        concat_in += f"[{tag}w{i}]"
    parts.append(f"{concat_in}concat=n={len(pieces)}:v=1:a=0[{tag}]")
    return tag


def _atempo_chain(speed: float) -> str:
    """atempo filter chain for ``speed`` (atempo only does 0.5–2.0)."""
    if speed <= 0:
        raise MediaEditError("speed must be > 0")
    chain: list[str] = []
    s = speed
    while s > 2.0:
        chain.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        chain.append("atempo=0.5")
        s /= 0.5
    chain.append(f"atempo={s:.4f}")
    return ",".join(chain)


def _is_still(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` is a still image (not a video)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 — not an image, or PIL disagrees
        return False


# ---------------------------------------------------------------------------
# single-clip helpers
# ---------------------------------------------------------------------------

def _clip_window(clip: Clip) -> tuple[float, float]:
    """Source window (start, end) in seconds; probes duration when needed."""
    p = Path(clip.path)
    if not p.exists():
        raise MediaEditError(f"no such clip: {clip.path}")
    info = video_probe(p)
    dur = info.get("duration") or 0.0
    start = max(0.0, float(clip.start))
    end = float(clip.end) if clip.end is not None else dur
    if end <= start:
        raise MediaEditError(
            f"clip {clip.path}: empty source window [{start}, {end}]")
    if start >= dur and dur > 0:
        raise MediaEditError(
            f"clip {clip.path}: start {start}s past duration {dur:.1f}s")
    return start, min(end, dur) if dur > 0 else end


def render_vertical(src: str | os.PathLike[str], *,
                    out: str | os.PathLike[str] | None = None,
                    fps: int = 30,
                    width: int = WIDTH, height: int = HEIGHT,
                    crf: int | None = None,
                    preset: str | None = None,
                    profile: str | None = None,
                    still_duration: float | None = None,
                    suffix: str = "vertical",
                    progress_cb: Any = None) -> dict[str, Any]:
    """Any video (or still image with ``still_duration``) → H.264 + AAC.

    Centre-crop fill to ``width``×``height``, CFR, yuv420p.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    perf = _perf_for(profile)
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset
    info = video_probe(p)
    dur = info.get("duration") or 0.0
    has_a = any(s.get("type") == "audio" for s in info.get("streams", []))
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p")
    out_p = Path(out) if out else p.with_name(f"{p.stem}.{suffix}.mp4")
    cmd: list[str] = []
    if still_duration is not None:
        cmd += ["-loop", "1", "-framerate", str(fps),
                "-t", f"{still_duration:.3f}"]
        dur = still_duration
    cmd += ["-i", str(p), "-vf", vf, "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if has_a and still_duration is None:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd.append(str(out_p))
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=dur or None)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "fps": fps, "width": width, "height": height,
            "profile": perf["profile"]}


def apply_effect(src: str | os.PathLike[str], effect: str, *,
                 out: str | os.PathLike[str] | None = None,
                 seed: int = 0,
                 at: tuple[float, float] | None = None,
                 still_duration: float | None = None,
                 fps: int = 30,
                 suffix: str | None = None,
                 progress_cb: Any = None,
                 **params: Any) -> dict[str, Any]:
    """Apply one named effect to ``src`` → new video file.

    ``at`` windows the effect to (start, end) seconds. ``still_duration``
    treats an image input as a still of that many seconds.

    Raises MediaEditError on unknown effects — never silently ignores.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    fx = Effect(name=effect, params=dict(params), at=at)
    if fx.name not in EFFECTS and fx.name != "velocity_ramp":
        raise MediaEditError(
            f"unknown effect {effect!r}; use: {sorted(EFFECTS)}")
    info = video_probe(p)
    w, h = info.get("width") or 0, info.get("height") or 0
    dur = info.get("duration") or 0.0
    is_still = still_duration is not None
    if is_still:
        if not (w and h):
            w, h = w or 1080, h or 1080
        dur = float(still_duration)
    if dur <= 0:
        raise MediaEditError(f"could not determine duration of {src}")
    ctx = _ctx(w, h, fps, dur, seed)
    parts: list[str] = []
    final = _chain_effect(parts, "0:v", fx, ctx, "fx")
    out_p = Path(out) if out else p.with_name(
        f"{p.stem}.{suffix or effect}.mp4")
    cmd: list[str] = []
    if is_still:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}"]
    cmd += ["-i", str(p), "-filter_complex", ";".join(parts),
            "-map", f"[{final}]", "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(out_p)]
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=dur)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "effect": effect}


# ---------------------------------------------------------------------------
# segment planning — explicit plans or a cut grid (beats are one source)
# ---------------------------------------------------------------------------

def plan_from_cut_points(n_clips: int, cut_points: list[float],
                         fps: int = 30) -> list[tuple[int, float]]:
    """Round-robin (clip_idx, play_duration) plan over a cut grid.

    Boundaries are quantized to the frame grid (frame-accurate cuts).
    ``cut_points`` is *any* time grid — beats, silence gaps, chapter
    marks. Needs ≥2 points.
    """
    if n_clips < 1:
        raise MediaEditError("need at least one clip to plan")
    qb = sorted({round(float(b) * fps) / fps
                 for b in cut_points if b >= 0})
    if len(qb) < 2:
        raise MediaEditError("need at least 2 cut points for a grid plan")
    seg_durs = [b - a for a, b in zip(qb, qb[1:]) if b - a >= 1.0 / fps]
    if not seg_durs:
        raise MediaEditError("cut grid produced no usable segments")
    return [(j % n_clips, d) for j, d in enumerate(seg_durs)]


def _materialize_stills(clips: list[Clip], *, fps: int, width: int,
                        height: int, crf: int, preset: str,
                        profile: str, tmp: Path) -> list[Clip]:
    """Still images → video segments (motion effects need real frames)."""
    out: list[Clip] = []
    for i, clip in enumerate(clips):
        if _is_still(clip.path):
            dur = ((clip.end or 0.0) - clip.start) if clip.end else 0.0
            still = render_vertical(
                clip.path, out=tmp / f"still{i}.mp4", fps=fps,
                width=width, height=height,
                still_duration=max(dur, 3.0), crf=crf, preset=preset,
                profile=profile)
            out.append(Clip(path=still["output"], start=0.0,
                            end=max(dur, 3.0) if dur else None,
                            effects=clip.effects, speed=clip.speed))
        else:
            out.append(clip)
    return out


# ---------------------------------------------------------------------------
# assemble_segments — the general multi-clip assembly, one ffmpeg run
# ---------------------------------------------------------------------------

def _normalize_transitions(
        transitions: Any, n_boundaries: int) -> list[Transition]:
    """str | Transition | list → per-boundary Transition list."""
    if isinstance(transitions, (str, Transition)):
        transitions = [transitions] * n_boundaries
    out: list[Transition] = []
    for t in list(transitions)[:n_boundaries]:
        out.append(t if isinstance(t, Transition)
                   else Transition.from_dict(t))
    while len(out) < n_boundaries:
        out.append(Transition(kind="cut"))
    return out


def assemble_segments(clips: list[Clip], *,
                      segments: list[tuple[int, float]] | None = None,
                      cut_points: list[float] | None = None,
                      transitions: Any = "cut",
                      out: str | os.PathLike[str] | None = None,
                      fps: int = 30,
                      width: int = WIDTH, height: int = HEIGHT,
                      seed: int = 0,
                      crf: int | None = None,
                      preset: str | None = None,
                      profile: str | None = None,
                      keep_audio: bool = False,
                      suffix: str = "edit",
                      workdir: str | os.PathLike[str] | None = None,
                      progress_cb: Any = None) -> dict[str, Any]:
    """Assemble ``clips`` per ``segments`` (or a ``cut_points`` grid).

    ``segments`` = explicit ``[(clip_idx, play_duration), …]``.
    ``cut_points`` = time grid → round-robin plan (beats, silence gaps,
    chapter marks — any grid). ``transitions``: one kind/Transition for
    every boundary, or a per-boundary list.

    One ffmpeg invocation. Returns the run report.
    """
    if not clips:
        raise MediaEditError("assemble_segments needs at least one clip")
    perf = _perf_for(profile)
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset
    tmp = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="asm-"))
    tmp.mkdir(parents=True, exist_ok=True)

    clips = _materialize_stills(clips, fps=fps, width=width, height=height,
                                crf=crf, preset=preset,
                                profile=perf["profile"], tmp=tmp)

    if segments is None:
        if cut_points is not None:
            segments = plan_from_cut_points(len(clips), cut_points, fps)
        else:
            segments = None  # sequential: one segment per clip
    plan: list[tuple[int, float]]
    if segments is None:
        windows0 = [_clip_window(c) for c in clips]
        plan = []
        for i, c in enumerate(clips):
            s0, s1 = windows0[i]
            plan.append((i, (s1 - s0) / max(c.speed, 1e-6)))
    else:
        plan = [(int(ci), float(d)) for ci, d in segments]
        if not plan:
            raise MediaEditError("empty segment plan")
        for ci, d in plan:
            if not (0 <= ci < len(clips)):
                raise MediaEditError(f"segment clip_idx {ci} out of range")
            if d <= 0:
                raise MediaEditError("segment durations must be > 0")

    tlist = _normalize_transitions(transitions, max(len(plan) - 1, 0))

    windows = [_clip_window(c) for c in clips]
    infos = [video_probe(c.path) for c in clips]
    has_audio = [any(s.get("type") == "audio" for s in i.get("streams", []))
                 for i in infos]

    parts: list[str] = []
    v_labels: list[str] = []
    v_durs: list[float] = []
    a_labels: list[str] = []
    cursors = [w0 for w0, _ in windows]  # continuous walk per clip
    total_dur = 0.0
    for n, (ci, play_d) in enumerate(plan):
        clip = clips[ci]
        s0, s1 = windows[ci]
        src_dur = s1 - s0
        speed = max(float(clip.speed), 1e-6)
        need = play_d * speed  # source seconds to cover play_d
        # take source footage at the cursor, wrapping (honest loop)
        takes: list[tuple[float, float]] = []
        remaining, cur = need, cursors[ci]
        guard = 0
        while remaining > 1e-6 and guard < 64:
            take = min(remaining, s1 - cur)
            takes.append((cur, cur + take))
            remaining -= take
            cur = s0 if cur + take >= s1 - 1e-9 else cur + take
            guard += 1
        cursors[ci] = cur
        tag = f"sg{n}"
        if len(takes) == 1:
            a, b = takes[0]
            parts.append(f"[{ci}:v]trim=start={a:.4f}:end={b:.4f},"
                         f"setpts=PTS-STARTPTS[{tag}raw]")
            vcur = f"{tag}raw"
        else:
            outs = "".join(f"[{tag}r{i}]" for i in range(len(takes)))
            parts.append(f"[{ci}:v]split={len(takes)}{outs}")
            cat = ""
            for i, (a, b) in enumerate(takes):
                parts.append(f"[{tag}r{i}]trim=start={a:.4f}:end={b:.4f},"
                             f"setpts=PTS-STARTPTS[{tag}w{i}]")
                cat += f"[{tag}w{i}]"
            parts.append(f"{cat}concat=n={len(takes)}:v=1:a=0[{tag}raw]")
            vcur = f"{tag}raw"
        if abs(speed - 1.0) > 1e-6:
            parts.append(f"[{vcur}]setpts=PTS/{speed:.4f},settb=AVTB[{tag}spd]")
            vcur = f"{tag}spd"
        # normalize to the canvas BEFORE effects (they assume w/h)
        parts.append(
            f"[{vcur}]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}[{tag}norm]")
        vcur = f"{tag}norm"
        ctx = _ctx(width, height, fps, play_d, seed + n * 131,
                   beats=[0.0, play_d])
        for k, fx in enumerate(clip.effects):
            vcur = _chain_effect(parts, vcur, fx, ctx, f"{tag}e{k}")
        # blink-kind transitions bake edge fades into the segment
        if n < len(plan) - 1:
            tr = tlist[n]
            if tr.kind in ("flash", "dip"):
                head_f, tail_f = edge_fade_filters(play_d, tr.kind,
                                                   tr.params)
                # tail fade on this segment; head fade on the next
                if tail_f:
                    parts.append(f"[{vcur}]{','.join(tail_f)}[{tag}tr]")
                    vcur = f"{tag}tr"
            elif tr.kind == "glitch_cut":
                gp = glitch_burst_params(tr.params)
                win = gp.pop("window")
                gfx = Effect(name="rgb_split", params=gp,
                             at=(max(play_d - win, 0.0), play_d))
                vcur = _chain_effect(parts, vcur, gfx, ctx, f"{tag}gl")
        # settb=AVTB: xfade requires identical timebases on both inputs
        parts.append(f"[{vcur}]format=yuv420p,settb=AVTB[{tag}v]")
        v_labels.append(f"[{tag}v]")
        v_durs.append(play_d)
        total_dur += play_d
        # audio twin (optional)
        if keep_audio:
            if has_audio[ci]:
                atakes = takes  # same source takes, speed-matched
                if len(atakes) == 1:
                    a, b = atakes[0]
                    parts.append(
                        f"[{ci}:a]atrim=start={a:.4f}:end={b:.4f},"
                        f"asetpts=PTS-STARTPTS[{tag}araw]")
                    acur = f"{tag}araw"
                else:
                    outs = "".join(f"[{tag}ar{i}]" for i in range(len(atakes)))
                    parts.append(f"[{ci}:a]asplit={len(atakes)}{outs}")
                    cat = ""
                    for i, (a, b) in enumerate(atakes):
                        parts.append(
                            f"[{tag}ar{i}]atrim=start={a:.4f}:end={b:.4f},"
                            f"asetpts=PTS-STARTPTS[{tag}aw{i}]")
                        cat += f"[{tag}aw{i}]"
                    parts.append(f"{cat}concat=n={len(atakes)}:v=0:a=1"
                                 f"[{tag}araw]")
                    acur = f"{tag}araw"
                if abs(speed - 1.0) > 1e-6:
                    parts.append(f"[{acur}]{_atempo_chain(speed)}[{tag}aspd]")
                    acur = f"{tag}aspd"
                if abs(float(clip.gain) - 1.0) > 1e-6:
                    parts.append(f"[{acur}]volume={clip.gain:.4f}[{tag}ag]")
                    acur = f"{tag}ag"
            else:
                parts.append(f"anullsrc=r=48000:cl=stereo:d={play_d:.4f}"
                             f"[{tag}araw]")
                acur = f"{tag}araw"
            parts.append(f"[{acur}]aresample=48000,aformat=channel_layouts=stereo"
                         f"[{tag}a]")
            a_labels.append(f"[{tag}a]")

    # --- boundary fold: xfade runs vs concat runs --------------------
    vfinal, vcat_dur = _fold_boundaries(parts, v_labels, v_durs, tlist)
    if keep_audio:
        acat = "".join(a_labels)
        parts.append(f"{acat}concat=n={len(plan)}:v=0:a=1[acat]")
        maps = ["-map", vfinal, "-map", "[acat]"]
        # NOTE: xfade overlaps shrink video vs audio; keep_audio +
        # overlap transitions is caller error territory — documented.
        if any(is_overlap(t.kind) for t in tlist):
            _log.warning("keep_audio with overlap transitions: audio not "
                         "time-shifted to match xfades")
    else:
        maps = ["-map", vfinal]

    out_p = Path(out) if out else Path(
        f"{Path(clips[0].path).stem}.{suffix}.mp4")
    cmd = []
    for c in clips:
        cmd += ["-i", c.path]
    cmd += ["-filter_complex", ";".join(parts), *maps,
            "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    cmd.append(str(out_p))
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=total_dur or None)
    return {"input_clips": [c.path for c in clips], "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "segments": len(plan), "duration": round(vcat_dur, 3),
            "fps": fps, "width": width, "height": height,
            "profile": perf["profile"]}


def _fold_boundaries(parts: list[str], labels: list[str],
                     durs: list[float],
                     transitions: list[Transition]) -> tuple[str, float]:
    """Fold segment boundaries → (final_label, final_duration).

    Overlap kinds (fade/dissolve/whip_pan) fold pairwise via xfade;
    everything else concatenates. Returns a single label.
    """
    if not labels:
        raise MediaEditError("nothing to fold")
    if len(labels) == 1:
        return labels[0], durs[0]
    # group into runs: maximal runs of xfade boundaries fold together
    pieces: list[str] = []   # labels to concat at the end
    piece_durs: list[float] = []
    cur_lbl, cur_dur = labels[0], durs[0]
    for i, tr in enumerate(transitions):
        nxt_lbl, nxt_dur = labels[i + 1], durs[i + 1]
        if is_overlap(tr.kind):
            name, td = xfade_args(tr.kind, tr.params, tr.duration)
            td = min(td, cur_dur, nxt_dur)
            if td < 0.05:
                raise MediaEditError(
                    f"xfade needs ≥0.05 s overlap; segments {i}/{i + 1} "
                    f"are {cur_dur:.2f}s / {nxt_dur:.2f}s")
            parts.append(
                f"{cur_lbl}{nxt_lbl}xfade=transition={name}:"
                f"duration={td:.4f}:offset={cur_dur - td:.4f}[xf{i}]")
            cur_lbl, cur_dur = f"[xf{i}]", cur_dur + nxt_dur - td
        else:
            pieces.append(cur_lbl)
            piece_durs.append(cur_dur)
            cur_lbl, cur_dur = nxt_lbl, nxt_dur
    pieces.append(cur_lbl)
    piece_durs.append(cur_dur)
    if len(pieces) == 1:
        return pieces[0], piece_durs[0]
    cat = "".join(pieces)
    parts.append(f"{cat}concat=n={len(pieces)}:v=1:a=0[vfold]")
    return "[vfold]", sum(piece_durs)


# ---------------------------------------------------------------------------
# render_timeline — Timeline + text + audio → finished mp4
# ---------------------------------------------------------------------------

def render_timeline(timeline: Timeline, *,
                    segments: list[tuple[int, float]] | None = None,
                    cut_points: list[float] | None = None,
                    transitions: Any = "cut",
                    text_layers: list[TextLayer] | None = None,
                    audio_mix: AudioMix | None = None,
                    out: str | os.PathLike[str] | None = None,
                    crf: int | None = None,
                    preset: str | None = None,
                    profile: str | None = None,
                    keep_clip_audio: bool = False,
                    suffix: str = "timeline",
                    workdir: str | os.PathLike[str] | None = None,
                    progress_cb: Any = None) -> dict[str, Any]:
    """Render a :class:`Timeline` → finished mp4.

    1. video track → assemble_segments (effects, transitions)
    2. audio_mix → mix_layers (or clip audio when keep_clip_audio)
    3. text_layers → burn_text_layers
    Returns the run report.
    """
    if not timeline.video.clips:
        raise MediaEditError("Timeline needs at least one video clip")
    tmp = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="tl-"))
    tmp.mkdir(parents=True, exist_ok=True)

    video = assemble_segments(
        timeline.video.clips, segments=segments, cut_points=cut_points,
        transitions=transitions,
        out=tmp / "video.mp4", fps=timeline.fps,
        width=timeline.width, height=timeline.height, seed=timeline.seed,
        crf=crf, preset=preset, profile=profile,
        keep_audio=keep_clip_audio, workdir=tmp, progress_cb=progress_cb)
    final = Path(video["output"])
    video_dur = video["duration"]
    audio_out: dict[str, Any] | None = None

    mix = (audio_mix if audio_mix is not None else timeline.audio_mix)
    if mix is not None and mix.layers:
        audio_out = mix_layers(mix, video_dur, tmp / "mix.m4a")
        muxed = tmp / "muxed.mp4"
        run_ffmpeg(["-i", str(final), "-i", audio_out["output"],
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(muxed)], duration=video_dur)
        final = muxed
    elif keep_clip_audio:
        pass  # already in the video stream

    layers = [l if isinstance(l, TextLayer) else TextLayer.from_dict(dict(l))
              for l in (text_layers if text_layers is not None
                        else timeline.text_layers)]
    text_out: dict[str, Any] | None = None
    if layers:
        text_out = burn_text_layers(final, layers, width=timeline.width,
                                    height=timeline.height,
                                    out=tmp / "captioned.mp4",
                                    workdir=tmp)
        final = Path(text_out["output"])

    if out:
        out_p = Path(out)
    else:
        out_p = Path(timeline.video.clips[0].path).parent / (
            f"{Path(timeline.video.clips[0].path).stem}.{suffix}.mp4")
    if final.resolve() != out_p.resolve():
        run_ffmpeg(["-i", str(final), "-c", "copy", str(out_p)],
                   duration=video_dur)
    return {"output": str(out_p), "bytes": out_p.stat().st_size,
            "duration": video_dur, "profile": video["profile"],
            "video": video, "audio": audio_out, "text": text_out,
            "workdir": str(tmp)}
