"""Style-agnostic render: Timeline/EditSpec → ffmpeg filter graph → mp4.

Migrated proven machinery (frame-quantized cuts, windowed effects,
velocity ramps, still materialization) with the style-specific beat
logic lifted out — cut grids arrive as data (``cut_points``), never as
assumptions.
"""

from __future__ import annotations

import os
import re
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from ...core.profile import detect_profile
from ...media_edit.videos import (
    MediaEditError,
    run_ffmpeg,
    video_probe,
)
from .audio import AudioLayer, AudioMix, mix_layers
from .effects import (
    _ctx,
    _effect_filter,
    EFFECTS,
    deferred_effect_reason,
    effect_rung,
)
from .frametime import FrameGrid
from .keyframes import (
    KeyframeTrack,
    classify_track,
    discretize,
    ffmpeg_expr,
    keyframe_filter,
    shifted,
)
from .text import TextLayer, burn_text_layers
from .timeline import (
    Clip,
    Effect,
    OverlayTrack,
    Timeline,
    Track,
    Transition,
    canonical_effect,
)
from .transitions import (
    edge_fade_filters,
    glitch_burst_params,
    is_overlap,
    validate_transition,
    xfade_args,
)

from .failures import record_failure

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

#: CRF quality presets (fftools guide: each +6 roughly halves file size).
#: ``quality="social"`` → crf 28, etc. Explicit `crf=` always wins.
CRF_PRESETS: dict[str, dict[str, Any]] = {
    "archival": {"crf": 18, "note": "visually lossless — masters, archives"},
    "high": {"crf": 21, "note": "high quality — client delivery"},
    "balanced": {"crf": 23, "note": "default — great quality, sane size"},
    "social": {"crf": 28, "note": "small files — feeds, stories, previews"},
    "draft": {"crf": 32, "note": "tiny — review drafts only"},
}


def crf_for(quality: str | None, explicit: int | None = None) -> int | None:
    """Resolve a quality preset name → CRF value. Never raises."""
    if explicit is not None:
        try:
            return max(0, min(51, int(explicit)))
        except (TypeError, ValueError):
            return None
    if not quality:
        return None
    preset = CRF_PRESETS.get(str(quality).lower())
    return preset["crf"] if preset else None


# ── render report + self-review ───────────────────────────────────────────────

from dataclasses import dataclass, field  # noqa: E402


@dataclass
class RenderReport:
    """Structured, JSON-serializable render report (fftools-style honesty).

    Built by :func:`render_timeline` / :func:`assemble_segments` and
    enriched by :meth:`self_review`, which runs ffprobe assertions
    against the finished file (jianying portable-build pattern):
    duration, dimensions, fps, frame count, codec.
    """
    output: str = ""
    bytes: int = 0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    crf: int | None = None
    quality: str = ""
    sha256: str = ""
    filter_graph_path: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    passed: bool | None = None
    #: five-pass frame-QC contact sheet record (§1.5c), attached by
    #: :func:`~.qc.qc_render` — sampled frames + per-axis verdicts.
    contact_sheet: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output, "bytes": self.bytes,
                "duration": round(self.duration, 2),
                "width": self.width, "height": self.height,
                "fps": round(self.fps, 2), "crf": self.crf,
                "quality": self.quality, "sha256": self.sha256,
                "filter_graph_path": self.filter_graph_path,
                "checks": self.checks, "passed": self.passed,
                "contact_sheet": self.contact_sheet}

    def self_review(self, *, expect_duration: float = 0.0,
                    tolerance_s: float = 0.5) -> "RenderReport":
        """ffprobe the output and assert it matches the plan.

        Checks: file exists + non-empty, H.264/yuv420p, dimensions,
        fps, duration within tolerance, frame count consistent.
        Records per-check pass/fail; ``passed`` is the AND of all.
        Never raises — a missing ffprobe is itself a recorded check.
        """
        checks: list[dict[str, Any]] = []

        def _check(name: str, ok: bool, detail: str = "") -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        p = Path(self.output)
        _check("exists", p.is_file(),
               f"{self.bytes} bytes" if p.is_file() else "missing")
        if not p.is_file():
            self.checks = checks
            self.passed = False
            return self
        try:
            probe = video_probe(str(p))
        except Exception as exc:  # noqa: BLE001
            _check("ffprobe", False, f"unavailable: {exc}")
            self.checks = checks
            self.passed = False
            return self
        _check("ffprobe", True, "ok")
        streams = probe.get("streams", []) if isinstance(probe, dict) else []
        v = next((s for s in streams if s.get("codec_type") == "video"),
                 {}) if streams else {}
        codec = str(v.get("codec_name", ""))
        pix = str(v.get("pix_fmt", ""))
        _check("codec", codec in ("h264", "hevc", "av1", "vp9"),
               f"{codec}/{pix}")
        w = int(v.get("width", 0) or 0)
        h = int(v.get("height", 0) or 0)
        if self.width and self.height:
            _check("dimensions", w == self.width and h == self.height,
                   f"{w}x{h} vs planned {self.width}x{self.height}")
        else:
            _check("dimensions", w > 0 and h > 0, f"{w}x{h}")
            self.width, self.height = w, h
        fps_s = str(v.get("avg_frame_rate", "0/1"))
        try:
            num, den = fps_s.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except Exception as e:
            record_failure("render.self_review:fps", e)
            fps = 0.0
        if self.fps:
            _check("fps", abs(fps - self.fps) < 0.5,
                   f"{fps:.2f} vs planned {self.fps:.2f}")
        else:
            self.fps = round(fps, 2)
            _check("fps", fps > 0, f"{fps:.2f}")
        dur = float(v.get("duration", 0) or 0)
        want = expect_duration or self.duration
        if want:
            _check("duration", abs(dur - want) <= tolerance_s,
                   f"{dur:.2f}s vs planned {want:.2f}s (±{tolerance_s}s)")
            self.duration = dur
        nframes = int(v.get("nb_frames", 0) or 0)
        if nframes and fps:
            expect_n = int(round(dur * fps))
            _check("frame_count", abs(nframes - expect_n) <= max(2, fps),
                   f"{nframes} frames ≈ {dur:.1f}s × {fps:.1f}fps")
        self.checks = checks
        self.passed = all(c["ok"] for c in checks)
        return self

    def report_text(self) -> str:
        """God-tier render summary."""
        from ..style import theme as _theme
        th = _theme()
        title = "Render Report"
        sub = self.quality or (f"crf {self.crf}" if self.crf else "default")
        lines = [th.banner(title, sub)]
        lines.append(th.kv([
            ("output", self.output),
            ("size", f"{self.bytes / 1048576:.1f} MiB"),
            ("duration", f"{self.duration:.2f}s"),
            ("dims", f"{self.width}x{self.height} @ {self.fps:.0f}fps"
             if self.width else ""),
            ("sha256", (self.sha256[:16] + "…") if self.sha256 else ""),
        ]))
        if self.checks:
            lines.append(th.section("Self-review"))
            for c in self.checks:
                lines.append("  " + th.status_line(
                    c["ok"], c["name"], c.get("detail", "")))
            lines.append(th.status_line(
                self.passed, "overall",
                "all checks passed — safe to present"
                if self.passed else "REVIEW FAILED — do not present"))
        return "\n".join(lines)


def _perf_for(profile: str | None) -> dict[str, Any]:
    kind = (profile or "").strip().lower()
    if not kind:
        try:
            kind = detect_profile().kind
        except Exception as e:  # detection must never break a render
            record_failure("render._perf_for", e)
            kind = "pc"
    perf = _PROFILE_PERF.get(kind, _PROFILE_PERF["pc"])
    return {"profile": kind, **perf}


# ---------------------------------------------------------------------------
# chain machinery — apply effects (optionally windowed) to a labelled stream
# ---------------------------------------------------------------------------

def _fx_record(fx: Effect, *, kind: str = "builder") -> dict[str, Any]:
    """One honesty record for one effect application (Wave 5).

    Names the producing rung, the window, the record's source (the
    clip's own effect list, a Document look_lock, a keyframe track)
    and — for a regional effect — the mask rung that anchored it.
    """
    params = fx.params or {}
    rec: dict[str, Any] = {
        "effect": fx.name, "rung": effect_rung(fx.name), "kind": kind,
        "window": ([float(fx.at[0]), float(fx.at[1])] if fx.at else None),
        "source": str(params.get("_source", "clip"))}
    if params.get("_mask_png"):
        mask_rec: dict[str, Any] = {
            "rung": str(params.get("_mask_rung", "")),
            "detail": str(params.get("_mask_detail", "")),
            "anchor": "source frame at the clip window start "
                      "(mask holds; tracking not requested)"}
        tracking = str(params.get("_mask_tracking", ""))
        if tracking:
            t_detail = str(params.get("_mask_track_detail", ""))
            mask_rec["tracking"] = {"status": tracking,
                                    "detail": t_detail}
            if tracking == "held":
                mask_rec["anchor"] = (
                    "source frame at the clip window start "
                    f"(tracking requested but held: {t_detail})")
            else:
                mask_rec["anchor"] = (
                    "mask tracked from the source frame at the "
                    f"clip window start ({t_detail})")
        rec["mask"] = mask_rec
    return rec


def _composite_masked(parts: list[str], base_label: str,
                      effected_label: str, mask_label: str,
                      tag: str) -> str:
    """Composite an effected branch over the base through a mask.

    The effected branch takes (its own alpha × the mask) as its
    alpha and composites over the untouched base — the overlay
    machinery the composite already trusts, in-chain. NOT
    maskedmerge: on this ffmpeg maskedmerge blends per-plane against
    the mask's constant chroma planes, washing the unmasked region
    with the effected stream (proven on fixtures, see PARITY.md).
    Branch and base pair by timestamp (overlay framesync), so the
    branch must live on the base's timeline — the contract the
    regional velocity resampler is built to (:func:`_chain_masked`
    gets it for free, both sides being splits of one stream).
    """
    parts.append(f"[{effected_label}]format=yuva420p[{tag}md]")
    parts.append(f"[{tag}md]split=2[{tag}mc][{tag}mx]")
    parts.append(f"[{tag}mx]alphaextract[{tag}ea]")
    parts.append(f"[{tag}ea][{mask_label}]blend=all_mode=multiply"
                 f"[{tag}am]")
    parts.append(f"[{tag}mc][{tag}am]alphamerge[{tag}ma]")
    # the composite lands exactly on the requested tag — the chain
    # contract every caller (plain, windowed, segment) relies on.
    parts.append(f"[{base_label}][{tag}ma]overlay=0:0[{tag}]")
    return tag


def _chain_masked(parts: list[str], cur: str, filt: str,
                  mask_label: str, tag: str) -> str:
    """Apply ``filt`` to ``cur`` through a mask stream; return label."""
    parts.append(f"[{cur}]split=2[{tag}mb][{tag}mf]")
    parts.append(f"[{tag}mf]{filt}[{tag}me]")
    return _composite_masked(parts, f"{tag}mb", f"{tag}me",
                             mask_label, tag)


def _chain_effect(parts: list[str], cur: str, fx: Effect,
                  ctx: dict[str, Any], tag: str,
                  rec: list[dict[str, Any]] | None = None) -> str:
    """Append filters applying ``fx`` to stream ``cur``; return new label.

    Windowed effects (``fx.at``) split the stream into head/window/tail,
    filter the window, and concat — works for every filter, including the
    ones without timeline support (noise, vignette, ...).

    A regional effect (the renderer resolved its ``mask`` param to
    ``_mask_png``) applies through the mask composite
    (:func:`_chain_masked`) — plain or windowed alike. ``rec``, when
    given, collects the per-effect honesty record.
    """
    params = fx.params or {}
    masked = bool(params.get("_mask_png"))
    if fx.name == "velocity_ramp":
        if masked:
            mask_label = (ctx.get("mask_labels") or {}).get(
                params["_mask_png"])
            if mask_label is None:
                raise MediaEditError(
                    f"effect {fx.name!r} carries a resolved mask but "
                    f"no mask stream is registered in this render — "
                    f"masks are resolved by the renderer before "
                    f"chaining")
            out = _chain_velocity_resampled(parts, cur, fx, ctx, tag,
                                            mask_label)
            if rec is not None:
                rec.append(_fx_record(fx,
                                      kind="structural-resampled"))
            return out
        out = _chain_velocity(parts, cur, fx, ctx, tag)
        if rec is not None:
            rec.append(_fx_record(fx, kind="structural"))
        return out
    if fx.name == "rgb_split" and params.get("animate"):
        out = _chain_rgb_animated(parts, cur, fx, ctx, tag)
        if rec is not None:
            rec.append(_fx_record(fx, kind="animated-discretized"))
        return out
    mask_label: str | None = None
    if masked:
        mask_label = (ctx.get("mask_labels") or {}).get(
            params["_mask_png"])
        if mask_label is None:
            raise MediaEditError(
                f"effect {fx.name!r} carries a resolved mask but no "
                f"mask stream is registered in this render — masks "
                f"are resolved by the renderer before chaining")

    def apply(src: str, out_tag: str) -> str:
        if mask_label is None:
            parts.append(f"[{src}]{filt}[{out_tag}]")
            return out_tag
        return _chain_masked(parts, src, filt, mask_label, out_tag)

    filt = _effect_filter(fx.name, fx.params, ctx)
    at = fx.at
    if not at:
        out = apply(cur, tag)
        if rec is not None:
            rec.append(_fx_record(fx))
        return out
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
        parts.append(f"[{cur}]trim={segs[0][0]},setpts=PTS-STARTPTS"
                     f"[{tag}w]")
        out = apply(f"{tag}w", tag)
        if rec is not None:
            rec.append(_fx_record(fx))
        return out
    outs = "".join(f"[{tag}s{i}]" for i in range(len(segs)))
    parts.append(f"[{cur}]split={len(segs)}{outs}")
    srcs = [f"{tag}s{i}" for i in range(len(segs))]
    concat_in = ""
    for (trim_args, ptag), s in zip(segs, srcs):
        if ptag == f"{tag}p1":
            parts.append(f"[{s}]trim={trim_args},setpts=PTS-STARTPTS"
                         f"[{ptag}w]")
            apply(f"{ptag}w", ptag)
        else:
            parts.append(f"[{s}]trim={trim_args},setpts=PTS-STARTPTS"
                         f"[{ptag}]")
        concat_in += f"[{ptag}]"
    parts.append(f"{concat_in}concat=n={len(segs)}:v=1:a=0[{tag}]")
    if rec is not None:
        rec.append(_fx_record(fx))
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
    # a resolved mask (and the record source) rides into every
    # sub-window — the discretization must never drop the anchor.
    carried = {k: params[k] for k in
               ("_mask_png", "_mask_rung", "_mask_detail", "_source",
                "_mask_tracking", "_mask_track_detail")
               if k in params}
    for i in range(steps):
        a = a0 + span * i / steps
        b = a0 + span * (i + 1) / steps
        # sample the envelope at the sub-window midpoint
        tmid = (a + b) / 2.0 - a0
        env = shift * math.sin(2 * math.pi * tmid / period + s1)
        sub = Effect(name="rgb_split",
                     params={"shift": round(env, 2), **carried},
                     at=(a, b))
        cur = _chain_effect(parts, cur, sub, ctx, f"{tag}q{i}")
    return cur


def _velocity_pieces(dur: float,
                     ramps: list[tuple[float, float, float]]
                     ) -> list[tuple[float, float, float]]:
    """Full-coverage (start, end, speed) pieces for a ramp set.

    Identity (speed 1.0) outside ramp windows. The one piece of ramp
    math: the in-chain video builder and the timeline-level ramp
    pass (video + audio) both consume it, so the two can never
    disagree about where a ramp starts or how long it lasts.
    """
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
    return pieces


def _velocity_ramps(fx: Effect,
                    dur: float) -> list[tuple[float, float, float]]:
    """The authored ramp set of a velocity_ramp effect.

    ``ramps=`` wins; otherwise one window from the effect's
    ``at`` (or the whole stream) at ``speed=``. The one parse,
    shared by the structural chain and the regional resampler,
    so the two can never disagree about what was authored.
    """
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
    return ramps


def _chain_velocity(parts: list[str], cur: str, fx: Effect,
                    ctx: dict[str, Any], tag: str) -> str:
    """velocity_ramp: piecewise setpts — the one structural effect."""
    dur = ctx["duration"]
    ramps = _velocity_ramps(fx, dur)
    pieces = _velocity_pieces(dur, ramps)
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


def _chain_velocity_resampled(parts: list[str], cur: str, fx: Effect,
                              ctx: dict[str, Any], tag: str,
                              mask_label: str) -> str:
    """Regional velocity_ramp — the region time-remap resampler.

    The unmasked chain rewrites the stream's timeline (trim/setpts
    pieces + concat), which a masked composite can never pair with:
    it framesyncs branch and base on ONE timeline. The resampler
    instead builds the branch ON the base timeline. Every surviving
    source frame τ is stamped with the output instant that shows
    it — τ itself outside ramp windows, ``x0 + (τ − x0)/s`` inside
    a window [x0, x1) at speed s — and the composite's framesync
    does the resampling: at output time t the region shows the
    source frame whose stamp interval covers t, i.e. the source
    time σ(t) = x0 + s·(t − x0) inside the window. The window is
    half-open, exactly the structural chain's trim semantics, so
    the frame at x1 is the first rejoined frame, never a window
    frame. The two edge absorptions live inside the region only:

    * s > 1 (the region races): σ reaches x1 at t* = x0 + (x1−x0)/s
      and the window's last frame then HOLDS — it stays the active
      branch frame until the identity-stamped frame at x1 takes
      over at the window edge.
    * s < 1 (the region lags): frames whose stamp would land at or
      after x1 are the unplayed tail of the window's material;
      ``select`` cuts them, and at x1 the region snaps back to the
      base clock.

    Outside the windows the branch is the untouched stream, the
    composite's total duration is the base's, and no other stream
    (audio included) is ever routed here. The semantics above are
    the effect's contract, recorded in PARITY.md and measured in
    tests/test_effects_remainders.py.
    """
    dur = ctx["duration"]
    pieces = _velocity_pieces(dur, _velocity_ramps(fx, dur))
    active = [(x0, x1, s) for x0, x1, s in pieces
              if abs(s - 1.0) > 1e-9]
    parts.append(f"[{cur}]split=2[{tag}mb][{tag}mf]")
    if not active:
        parts.append(f"[{tag}mf]null[{tag}me]")
        return _composite_masked(parts, f"{tag}mb", f"{tag}me",
                                 mask_label, tag)
    stamp = "PTS*TB"
    for x0, x1, s in reversed(active):
        stamp = (f"if(gte(PTS*TB,{x0:.6f})*lt(PTS*TB,{x1:.6f}),"
                 f"{x0:.6f}+(PTS*TB-{x0:.6f})/{s:.6f},{stamp})")
    chain: list[str] = []
    drops = [f"gte(t,{x0:.6f})*lt(t,{x1:.6f})"
             f"*gte(t,{x0 + s * (x1 - x0):.6f})"
             for x0, x1, s in active if s < 1.0]
    if drops:
        chain.append("select='not(" +
                     "+".join(f"({d})" for d in drops) + ")'")
    chain.append(f"setpts='({stamp})/TB'")
    parts.append(f"[{tag}mf]{','.join(chain)}[{tag}me]")
    return _composite_masked(parts, f"{tag}mb", f"{tag}me",
                             mask_label, tag)


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


def _group_sampled_tracks(
        pairs: list[tuple[KeyframeTrack, KeyframeTrack]], *,
        fps: int) -> list[dict[str, Any]]:
    """Group sampled-family tracks by effect for one clip window.

    ``pairs`` = (track, window-shifted track) in author order.
    Each returned group is ``{"tracks": [track, ...], "windows":
    [(a, b, {param: value, ...}), ...]}`` in first-appearance
    order. A one-track group's windows are exactly
    ``discretize(window, fps=fps)`` — the Wave-5 binding, unchanged.

    A multi-track group (several params of ONE effect — the
    animated-CDL case: slope/saturation components bound by
    separate tracks) is discretized JOINTLY: the union of every
    track's frame-grid boundaries, each window carrying every
    track's value sampled at the window midpoint. The components
    land in ONE sub-effect per window, so they compose into a
    single application of the effect (one CDL bake) instead of
    sequential re-applications of the whole effect per component.
    Two tracks on the same (effect, param) are ambiguous and raise.
    """
    groups: list[dict[str, Any]] = []
    by_effect: dict[str, dict[str, Any]] = {}
    for track, window in pairs:
        g = by_effect.get(track.effect)
        if g is None:
            g = {"tracks": [], "windows_in": []}
            by_effect[track.effect] = g
            groups.append(g)
        if any(t.param == track.param for t in g["tracks"]):
            raise MediaEditError(
                f"two keyframe tracks target "
                f"{track.effect}.{track.param} on clip "
                f"{track.clip_id!r} — ambiguous animation")
        g["tracks"].append(track)
        g["windows_in"].append(window)
    for g in groups:
        tracks: list[KeyframeTrack] = g["tracks"]
        wins_in: list[KeyframeTrack] = g.pop("windows_in")
        if len(tracks) == 1:
            t = tracks[0]
            g["windows"] = [
                (a, b, {t.param: round(v, 4)})
                for a, b, v in discretize(wins_in[0], fps=fps)]
            continue
        bounds: set[float] = set()
        for w in wins_in:
            for a, b, _v in discretize(w, fps=fps):
                bounds.add(a)
                bounds.add(b)
        ordered = sorted(bounds)
        windows: list[tuple[float, float, dict[str, float]]] = []
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b - a < 1e-9:
                continue
            mid = (a + b) / 2.0
            windows.append((a, b, {
                t.param: round(w.sample(mid), 4)
                for t, w in zip(tracks, wins_in, strict=True)}))
        g["windows"] = windows
    return groups


class _MaskStreams:
    """Per-render registry of resolved effect masks.

    Each resolved mask becomes ONE extra ffmpeg input and ONE graph
    label; every masked effect in the render references the label.
    A mask is either a *still* (one PNG, looped for the render's
    duration — the anchor-and-hold shape) or a *video* (a tracked
    mask sequence, encoded losslessly by the tracking rung; it
    plays from its own first frame in lockstep with the consuming
    chain). Input order = registration order, so the caller can
    compute stream indices deterministically.
    """

    def __init__(self) -> None:
        self.labels: dict[str, str] = {}   # media path → graph label
        self.inputs: list[str] = []        # media paths, input order
        self.kinds: dict[str, str] = {}    # media path → still|video

    def register(self, png: str, kind: str = "still") -> str:
        if png not in self.labels:
            self.labels[png] = f"msk{len(self.inputs)}"
            self.inputs.append(png)
            self.kinds[png] = kind
        return self.labels[png]

    def preamble(self, parts: list[str], *, first_index: int,
                 width: int, height: int) -> None:
        """Graph preambles turning each mask input into its label."""
        for j, png in enumerate(self.inputs):
            parts.append(
                f"[{first_index + j}:v]scale={width}:{height},"
                f"format=gray[{self.labels[png]}]")

    def input_args(self, *, fps: int, duration: float) -> list[str]:
        """cmd args adding the mask inputs (looped stills, bounded;
        tracked videos play their own frames, unbounded by -t)."""
        args: list[str] = []
        for png in self.inputs:
            if self.kinds.get(png) == "video":
                args += ["-i", png]
            else:
                args += ["-loop", "1", "-framerate", str(fps),
                         "-t", f"{max(duration, 0.1):.3f}", "-i", png]
        return args


def _grab_anchor_image(source_path: str | os.PathLike[str], at_s: float,
                       width: int, height: int, tmp: Path,
                       tag: str) -> Any:
    """The frame a mask anchors to: the source frame at ``at_s``,
    scaled to the canvas (mask coordinates are canvas pixels)."""
    from PIL import Image
    p = Path(source_path)
    if _is_still(p):
        with Image.open(p) as im:
            return im.convert("RGB").resize((width, height))
    png = tmp / f"anchor{tag}.png"
    run_ffmpeg(["-ss", f"{max(0.0, at_s):.3f}", "-i", str(p),
                "-frames:v", "1", "-y", str(png)])
    with Image.open(png) as im:
        img = im.convert("RGB")
    return img.resize((width, height)) if img.size != (width, height) \
        else img


def _extract_track_frames(source_path: str | os.PathLike[str], *,
                          start_s: float, step_s: float, count: int,
                          width: int, height: int, tmp: Path,
                          tag: str) -> list[Path]:
    """Source frames at the chain's consumption cadence.

    Frame j is the source at ``start_s + j * step_s`` (the fps
    filter resamples the decoded stream to that grid — the same
    grid the consuming chain plays at), scaled to the canvas so
    mask geometry is canvas geometry. Fewer files than ``count``
    come back when the source ends first; the caller holds the
    mask at its last offset for the remainder.
    """
    seq = tmp / f"trackframes{tag}"
    seq.mkdir(parents=True, exist_ok=True)
    rate = 1.0 / max(step_s, 1e-6)
    run_ffmpeg(["-ss", f"{max(0.0, start_s):.3f}", "-i",
                str(source_path),
                "-vf", f"fps={rate:.6f},scale={width}:{height}",
                "-frames:v", str(count),
                "-y", str(seq / "f%05d.png")])
    return sorted(seq.glob("f*.png"))


def _track_mask_attempt(source_path: str | os.PathLike[str], *,
                        source_start: float, speed: float, fps: int,
                        play_frames: int, width: int, height: int,
                        mask_img: Any, tmp: Path,
                        tag: str) -> tuple[str, str, str | None]:
    """Try to turn a resolved anchor mask into a tracked sequence.

    Returns ``(status, detail, video_path|None)`` — status/detail
    land verbatim in the effect record. NEVER raises: tracking is
    an enhancement of the anchor-and-hold shape, so every failure
    mode (still source, no timeline supplied, frame cap, no
    texture, unmatchable content, a tracker fault) comes back as
    ``held`` with the reason named, and the caller keeps the
    static mask. A held outcome is a recorded fact in
    ``effects_applied``, never a silent fallback.
    """
    from . import mask_track
    try:
        if _is_still(source_path):
            return ("held", "the source is a still image — there "
                            "are no frames to track", None)
        if fps <= 0 or play_frames < 2:
            return ("held", "the renderer supplied no frame "
                            "timeline to track over", None)
        if play_frames > mask_track.MAX_TRACK_FRAMES:
            return ("held",
                    f"the clip plays {play_frames} frames — beyond "
                    f"the {mask_track.MAX_TRACK_FRAMES}-frame "
                    f"tracking cap", None)
        import numpy as np
        from PIL import Image
        step_s = max(float(speed), 1e-6) / float(fps)
        paths = _extract_track_frames(
            source_path, start_s=source_start, step_s=step_s,
            count=play_frames, width=width, height=height,
            tmp=tmp, tag=tag)
        if len(paths) < 2:
            return ("held",
                    f"only {len(paths)} frame(s) could be "
                    f"extracted from the source window", None)

        def _gray() -> Any:
            for p in paths:
                with Image.open(p) as im:
                    yield np.asarray(im.convert("L"),
                                     dtype=np.float64)

        base = np.asarray(mask_img.convert("L")) > 127
        res = mask_track.track_sequence(_gray(), base)
        if res.status == "held":
            return ("held", res.reason, None)
        offsets = list(res.offsets)
        detail = res.detail()
        if len(offsets) < play_frames:
            detail += (f"; the source yielded {len(paths)} of "
                       f"{play_frames} played frames — the mask "
                       f"holds at its last offset beyond")
        while len(offsets) < play_frames:
            offsets.append(offsets[-1])
        # one second of held tail past the played span: segment
        # trims round at frame edges, and a mask stream that ends
        # early would starve the composite — the still path's
        # render-length loop is the same insurance.
        offsets.extend([offsets[-1]] * fps)
        seq_dir = tmp / f"trackseq{tag}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for j, (dx, dy) in enumerate(offsets):
            Image.fromarray(
                mask_track.shift_mask(base, dx, dy)).save(
                    str(seq_dir / f"m{j:05d}.png"))
        out = tmp / f"masktrack{tag}.mkv"
        run_ffmpeg(["-framerate", str(fps), "-i",
                    str(seq_dir / "m%05d.png"),
                    "-c:v", "ffv1", "-pix_fmt", "gray",
                    "-y", str(out)])
        return (res.status, detail, str(out))
    except Exception as exc:  # noqa: BLE001 — held, named, recorded
        return ("held", f"the tracking attempt failed: {exc}", None)


def _prepare_effects(effects: Any, *, source_path: str,
                     source_start: float, width: int, height: int,
                     tmp: Path, masks: _MaskStreams,
                     tag_prefix: str, fps: int = 0,
                     speed: float = 1.0,
                     play_frames: int = 0) -> list[Effect]:
    """Resolve ``mask`` params on effects → renderer-internal copies.

    An effect may carry ``params["mask"]`` = a MaskRef dict (the
    imggen carrier — the same shape ImageDocument ops and the motion
    brush use). Resolution runs the MaskRef ladder against the
    source frame at the clip's window start, writes the mask PNG
    into ``tmp`` and registers its stream; the prepared copy carries
    ``_mask_png``/``_mask_rung``/``_mask_detail`` instead of the ref.
    Authored effects are never mutated. A mask that cannot resolve
    raises MediaEditError with the ladder's reason — before any
    ffmpeg run, never a silent unmasked render.

    ``params["mask_track"]`` (truthy) asks for the mask to follow
    its content across the clip: the renderer tracks it with
    :mod:`.mask_track` over the ``play_frames`` the clip will play
    (at ``fps``, source advancing ``speed``×) and registers a
    tracked mask *video* instead of the looped still. The outcome —
    tracked / partial / held, with the reason when held — is
    stamped ``_mask_tracking``/``_mask_track_detail`` and surfaces
    in the effect record; a held outcome keeps the still.
    """
    out: list[Effect] = []
    for i, fx in enumerate(effects):
        params = fx.params or {}
        ref_d = params.get("mask")
        if ref_d is None:
            out.append(fx)
            continue
        from ..imggen.masks import MaskRef
        try:
            ref = ref_d if isinstance(ref_d, MaskRef) \
                else MaskRef.from_dict(dict(ref_d))
        except Exception as exc:
            raise MediaEditError(
                f"effect {fx.name!r}: bad mask ref: {exc}") from exc
        anchor = _grab_anchor_image(source_path, source_start,
                                    width, height, tmp,
                                    f"{tag_prefix}{i}")
        try:
            result = ref.resolve(anchor)
        except Exception as exc:
            raise MediaEditError(
                f"effect {fx.name!r}: mask did not resolve: {exc}"
            ) from exc
        png = tmp / f"mask{tag_prefix}{i}.png"
        m = result.mask
        if m.size != (width, height):
            m = m.resize((width, height))
        m.save(str(png))
        new_params = {k: v for k, v in params.items()
                      if k not in ("mask", "mask_track")}
        new_params["_mask_png"] = str(png)
        new_params["_mask_rung"] = result.tier
        new_params["_mask_detail"] = result.detail
        if params.get("mask_track"):
            status, detail, video = _track_mask_attempt(
                source_path, source_start=source_start, speed=speed,
                fps=fps, play_frames=play_frames, width=width,
                height=height, mask_img=m, tmp=tmp,
                tag=f"{tag_prefix}{i}")
            new_params["_mask_tracking"] = status
            new_params["_mask_track_detail"] = detail
            if video is not None:
                masks.register(video, kind="video")
                new_params["_mask_png"] = video
            else:
                masks.register(str(png))
        else:
            masks.register(str(png))
        out.append(Effect(name=fx.name, params=new_params, at=fx.at))
    return out


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
                 profile: str | None = None,
                 progress_cb: Any = None,
                 **params: Any) -> dict[str, Any]:
    """Apply one named effect to ``src`` → new video file.

    ``at`` windows the effect to (start, end) seconds. ``still_duration``
    treats an image input as a still of that many seconds. ``profile``
    selects the encode knobs (the surface has always accepted it —
    Wave 5 wires it to _perf_for instead of letting it fall into the
    effect params, where the property registry now rightly rejects
    undeclared keys).

    Raises MediaEditError on unknown effects — never silently ignores.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    fx = Effect(name=effect, params=dict(params), at=at)
    reason = deferred_effect_reason(fx.name)
    if reason is not None:
        raise MediaEditError(
            f"effect {effect!r} is deferred, not unknown: {reason}")
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
    masks = _MaskStreams()
    fx_rec: list[dict[str, Any]] = []
    if (fx.params or {}).get("mask") is not None:
        mtmp = Path(tempfile.mkdtemp(prefix="fxmask-"))
        fx = _prepare_effects(
            [fx], source_path=str(p),
            source_start=float(at[0]) if at else 0.0,
            width=w, height=h, tmp=mtmp, masks=masks,
            tag_prefix="single", fps=fps, speed=1.0,
            play_frames=int(dur * fps) + 1)[0]
        ctx["mask_labels"] = masks.labels
        masks.preamble(parts, first_index=1, width=w, height=h)
    final = _chain_effect(parts, "0:v", fx, ctx, "fx", rec=fx_rec)
    out_p = Path(out) if out else p.with_name(
        f"{p.stem}.{suffix or effect}.mp4")
    cmd: list[str] = []
    if is_still:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}"]
    enc_preset, enc_crf = "veryfast", 20
    if profile is not None:
        perf = _perf_for(profile)
        enc_preset, enc_crf = perf["preset"], perf["crf"]
    cmd += ["-i", str(p)]
    cmd += masks.input_args(fps=fps, duration=dur)
    cmd += ["-filter_complex", ";".join(parts),
            "-map", f"[{final}]", "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", enc_preset,
            "-crf", str(enc_crf),
            "-pix_fmt", "yuv420p", str(out_p)]
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=dur)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "effect": effect, "effects_applied": fx_rec}


# ---------------------------------------------------------------------------
# segment planning — explicit plans or a cut grid (beats are one source)
# ---------------------------------------------------------------------------

def plan_from_cut_points(n_clips: int, cut_points: list[float],
                         fps: int = 30) -> list[tuple[int, float]]:
    """Round-robin (clip_idx, play_duration) plan over a cut grid.

    Boundaries are quantized to the frame grid (frame-accurate cuts).
    ``cut_points`` is *any* time grid — beats, silence gaps, chapter
    marks. Needs ≥2 points.

    Wave 1.2a: the grid math is integer frame counts (FrameGrid) —
    segment durations are frame differences, so the plan's total is
    the frame sum *exactly*, at any grid length.
    """
    if n_clips < 1:
        raise MediaEditError("need at least one clip to plan")
    grid = FrameGrid(fps)
    qf = sorted({grid.frames(b) for b in cut_points if b >= 0})
    if len(qf) < 2:
        raise MediaEditError("need at least 2 cut points for a grid plan")
    seg_frames = [b - a for a, b in zip(qf, qf[1:]) if b - a >= 1]
    if not seg_frames:
        raise MediaEditError("cut grid produced no usable segments")
    return [(j % n_clips, float(grid.seconds(f)))
            for j, f in enumerate(seg_frames)]


def _materialize_stills(clips: list[Clip], *, fps: int, width: int,
                        height: int, crf: int, preset: str,
                        profile: str, tmp: Path) -> list[Clip]:
    """Still images → video segments (motion effects need real frames)."""
    out: list[Clip] = []
    for i, clip in enumerate(clips):
        if _is_still(clip.path):
            # the clip's authored duration wins; end-start next; the
            # 3 s default applies only when neither is authored.
            if clip.duration:
                dur = float(clip.duration)
            elif clip.end:
                dur = float(clip.end) - float(clip.start)
            else:
                dur = 0.0
            still = render_vertical(
                clip.path, out=tmp / f"still{i}.mp4", fps=fps,
                width=width, height=height,
                still_duration=dur if dur > 0 else 3.0, crf=crf,
                preset=preset, profile=profile)
            out.append(Clip(path=still["output"], start=0.0,
                            end=dur if dur > 0 else None,
                            effects=clip.effects, speed=clip.speed,
                            at=clip.at, gain=clip.gain, id=clip.id))
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


def _match_keyframes(clips: list[Clip],
                     tracks: list[KeyframeTrack] | None
                     ) -> dict[int, list[KeyframeTrack]]:
    """Keyframe tracks → {clip index: [tracks]} by clip identity.

    Resolution order per track: the clip's explicit ``id``, then the
    per-track index spelling (``"0"``, ``"1"``, …), then the source
    path. A track matching no clip raises — an animation aimed at
    nothing is a build error, never a silent no-op.
    """
    out: dict[int, list[KeyframeTrack]] = {}
    for track in tracks or []:
        if not track.keys:
            raise MediaEditError(
                f"keyframe track {track.clip_id!r} "
                f"({track.effect}.{track.param}) has no keyframes")
        idx: int | None = None
        for i, c in enumerate(clips):
            if c.id and c.id == track.clip_id:
                idx = i
                break
        if idx is None and str(track.clip_id).isdigit():
            i = int(track.clip_id)
            if 0 <= i < len(clips):
                idx = i
        if idx is None:
            for i, c in enumerate(clips):
                if c.path == track.clip_id:
                    idx = i
                    break
        if idx is None:
            raise MediaEditError(
                f"keyframe track targets clip {track.clip_id!r}, which "
                f"is not among {[c.id or c.path for c in clips]}")
        out.setdefault(idx, []).append(track)
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
                      keyframes: list[KeyframeTrack] | None = None,
                      host: Any = None,
                      suffix: str = "edit",
                      workdir: str | os.PathLike[str] | None = None,
                      progress_cb: Any = None) -> dict[str, Any]:
    """Assemble ``clips`` per ``segments`` (or a ``cut_points`` grid).

    ``segments`` = explicit ``[(clip_idx, play_duration), …]``.
    ``cut_points`` = time grid → round-robin plan (beats, silence gaps,
    chapter marks — any grid). ``transitions``: one kind/Transition for
    every boundary, or a per-boundary list.

    ``keyframes`` (Wave 1.2c): expression-family tracks (zoom) and
    opacity tracks are applied per segment, re-based to each
    segment's window of the clip's play time; opacity renders as
    alpha over a black composite at the fold. Position tracks belong
    to the overlay composite — passing them here raises, directing
    the caller to :func:`render_timeline`.

    Wave 5: ``sampled``-family tracks (any other registry-declared
    param — blur radius, grain strength, …) are discretized on the
    frame grid into constant-value windowed sub-effects of the same
    effect; a same-named static effect on the clip becomes the
    sampled base (two or more is an ambiguity error). Effects may
    carry a ``mask`` MaskRef param — resolved once per clip against
    its window-start frame and applied through the mask composite.
    Every application is recorded in the result's
    ``effects_applied``.

    ``host`` (Wave 1.2d): a :class:`~.host.RenderHost` to take the
    encode plan from, instead of re-probing the profile.

    One ffmpeg invocation. Returns the run report.
    """
    if not clips:
        raise MediaEditError("assemble_segments needs at least one clip")
    perf = _perf_for(profile)
    host_plan: dict[str, Any] | None = None
    if host is not None:
        host_plan = host.plan(width=width, height=height, fps=fps,
                              profile=profile)
        perf = {"profile": host_plan["profile"],
                "preset": host_plan["preset"], "crf": host_plan["crf"],
                "threads": host_plan["threads"]}
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset
    kf_by_clip: dict[int, list[KeyframeTrack]] = {}
    kf_applied: list[dict[str, Any]] = []
    kf_families: dict[int, str] = {}
    if keyframes:
        for track in keyframes:
            try:
                family = classify_track(track)
            except ValueError as exc:
                raise MediaEditError(str(exc)) from exc
            if family == "position":
                raise MediaEditError(
                    f"keyframe track {track.effect}.{track.param} is a "
                    f"position animation — those are consumed by the "
                    f"overlay composite in render_timeline, not by "
                    f"assemble_segments")
            kf_families[id(track)] = family
        kf_by_clip = _match_keyframes(clips, keyframes)
    has_opacity = any(f == "opacity" for f in kf_families.values())
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

    # Wave 5: resolve effect masks once per clip (anchored to the
    # clip's window-start frame), then work out the sampled-keyframe
    # bookkeeping against the PREPARED effects: a sampled track's
    # static base is the clip's one same-named effect (which the
    # track then replaces in the chain); two or more is ambiguous.
    masks = _MaskStreams()
    play_frames = [0] * len(clips)
    for ci_, play_d in plan:
        play_frames[ci_] += int(play_d * fps) + 1
    prepared_fx: list[list[Effect]] = [
        _prepare_effects(c.effects, source_path=c.path,
                         source_start=windows[i][0], width=width,
                         height=height, tmp=tmp, masks=masks,
                         tag_prefix=f"c{i}_", fps=fps,
                         speed=float(c.speed),
                         play_frames=play_frames[i])
        for i, c in enumerate(clips)]
    fx_applied: list[dict[str, Any]] = []
    sampled_base: dict[tuple[int, str], dict[str, Any]] = {}
    consumed_static: set[int] = set()   # id() of consumed Effect objs
    for ci, tracks in kf_by_clip.items():
        for kt in tracks:
            if kf_families[id(kt)] != "sampled":
                continue
            statics = [fx for fx in prepared_fx[ci]
                       if fx.name == kt.effect]
            if len(statics) > 1:
                raise MediaEditError(
                    f"keyframe track {kt.effect}.{kt.param}: clip "
                    f"{ci} carries {len(statics)} static {kt.effect} "
                    f"effects — ambiguous base for the sampled curve")
            if statics:
                sampled_base[(ci, kt.effect)] = dict(
                    statics[0].params or {})
                consumed_static.add(id(statics[0]))

    parts: list[str] = []
    masks.preamble(parts, first_index=len(clips), width=width,
                   height=height)
    v_labels: list[str] = []
    v_durs: list[float] = []
    a_labels: list[str] = []
    cursors = [w0 for w0, _ in windows]  # continuous walk per clip
    playheads = [0.0] * len(clips)  # per-clip play time (keyframe axis)
    total_dur = 0.0
    pending_head_fade: list[str] | None = None  # blink head fade from the
    # previous boundary — applied to this segment after its effects, so
    # flash/dip blink on both sides of the cut (tail of A + head of B).
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
        ctx["mask_labels"] = masks.labels
        rec_mark = len(fx_applied)
        for k, fx in enumerate(prepared_fx[ci]):
            if id(fx) in consumed_static:
                continue  # a sampled track owns this effect now
            vcur = _chain_effect(parts, vcur, fx, ctx, f"{tag}e{k}",
                                 rec=fx_applied)
        for entry in fx_applied[rec_mark:]:
            entry["clip"] = clip.id or ci
            entry["segment"] = n
        # keyframed animation for this segment's window of the clip's
        # play time (Wave 1.2c): the track is re-based to segment-local
        # time. Zoom becomes one expression filter; opacity is sampled
        # into constant-alpha windows over an alpha-carrying stream.
        # Sampled tracks on one effect form a group (the joint
        # binding): a group is chained once, at its first track's
        # position, with every component in each window's sub-effect.
        clip_tracks = kf_by_clip.get(ci, [])
        sampled_groups = _group_sampled_tracks(
            [(kt, shifted(kt, playheads[ci], play_d))
             for kt in clip_tracks
             if kf_families[id(kt)] == "sampled"], fps=fps)
        group_of = {id(kt): g for g in sampled_groups
                    for kt in g["tracks"]}
        done_groups: set[int] = set()
        for j, kt in enumerate(clip_tracks):
            window = shifted(kt, playheads[ci], play_d)
            if kf_families[id(kt)] == "expression":
                filt = keyframe_filter(window, width=width,
                                       height=height)
                parts.append(f"[{vcur}]{filt}[{tag}kf{j}]")
                vcur = f"{tag}kf{j}"
            elif kf_families[id(kt)] == "opacity":
                parts.append(f"[{vcur}]format=yuva420p[{tag}alp{j}]")
                vcur = f"{tag}alp{j}"
                for w_i, (a, b, v) in enumerate(discretize(window)):
                    sub = Effect(name="opacity",
                                 params={"alpha": round(v, 4)}, at=(a, b))
                    vcur = _chain_effect(parts, vcur, sub, ctx,
                                         f"{tag}op{j}_{w_i}")
            else:  # sampled (Wave 5): frame-grid windows of the same
                # effect, built by the same chain builder, over the
                # consumed static base (or the builder defaults).
                g = group_of[id(kt)]
                if id(g) not in done_groups:
                    done_groups.add(id(g))
                    base = sampled_base.get((ci, kt.effect), {})
                    for w_i, (a, b, overlay) in enumerate(
                            g["windows"]):
                        sub_params = dict(base)
                        sub_params.update(overlay)
                        sub = Effect(name=kt.effect,
                                     params=sub_params, at=(a, b))
                        vcur = _chain_effect(parts, vcur, sub, ctx,
                                             f"{tag}sm{j}_{w_i}")
                    for t in g["tracks"]:
                        entry = _fx_record(
                            Effect(name=t.effect, params=dict(base)))
                        entry.update({
                            "kind": "keyframe-sampled",
                            "source": f"keyframe:{t.param}",
                            "clip": clip.id or ci, "segment": n,
                            "windows": len(g["windows"])})
                        fx_applied.append(entry)
            kf_applied.append({"clip_id": kt.clip_id,
                               "effect": kt.effect, "param": kt.param,
                               "family": kf_families[id(kt)],
                               "segment": n})
        playheads[ci] += play_d
        # blink-kind head fade carried from the previous boundary
        if pending_head_fade:
            parts.append(f"[{vcur}]{','.join(pending_head_fade)}[{tag}hd]")
            vcur = f"{tag}hd"
            pending_head_fade = None
        # blink-kind transitions bake edge fades into the segment
        if n < len(plan) - 1:
            tr = tlist[n]
            if tr.kind in ("flash", "dip"):
                _head, tail_f = edge_fade_filters(play_d, tr.kind,
                                                  tr.params)
                # tail fade on this segment; head fade on the next
                # (clamped by the next segment's own length)
                if tail_f:
                    parts.append(f"[{vcur}]{','.join(tail_f)}[{tag}tr]")
                    vcur = f"{tag}tr"
                head_next, _tail = edge_fade_filters(plan[n + 1][1],
                                                     tr.kind, tr.params)
                pending_head_fade = head_next or None
            elif tr.kind == "glitch_cut":
                gp = glitch_burst_params(tr.params)
                win = gp.pop("window")
                gfx = Effect(name="rgb_split", params=gp,
                             at=(max(play_d - win, 0.0), play_d))
                vcur = _chain_effect(parts, vcur, gfx, ctx, f"{tag}gl")
        # settb=AVTB: xfade requires identical timebases on both inputs.
        # An opacity-carrying assembly keeps its alpha to the fold,
        # where it composites over black (fade semantics).
        seg_fmt = "yuva420p" if has_opacity else "yuv420p"
        parts.append(f"[{vcur}]format={seg_fmt},settb=AVTB[{tag}v]")
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
    if has_opacity:
        # fade semantics: the alpha the opacity tracks produced reads
        # against black. One in-graph color source + overlay, then
        # back to plain yuv420p for the encoder.
        parts.append(f"color=c=black:s={width}x{height}:r={fps}"
                     f":d={vcat_dur:.4f}[blk]")
        parts.append(f"[blk]{vfinal}overlay=0:0[compv]")
        parts.append(f"[compv]format=yuv420p[vcomp]")
        vfinal = "[vcomp]"
    if keep_audio:
        # audio folds on the same boundaries as video: acrossfade at
        # overlap boundaries (same overlap seconds), concat elsewhere
        # — picture and sound stay the same length.
        afinal = _fold_audio_boundaries(parts, a_labels, v_durs, tlist)
        maps = ["-map", vfinal, "-map", afinal]
    else:
        maps = ["-map", vfinal]

    out_p = Path(out) if out else Path(
        f"{Path(clips[0].path).stem}.{suffix}.mp4")
    cmd = []
    for c in clips:
        cmd += ["-i", c.path]
    cmd += masks.input_args(fps=fps, duration=total_dur)
    cmd += ["-filter_complex", ";".join(parts), *maps,
            "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    cmd.append(str(out_p))
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=total_dur or None)
    result = {"input_clips": [c.path for c in clips], "output": str(out_p),
              "bytes": out_p.stat().st_size, "seconds": run["seconds"],
              "segments": len(plan), "duration": round(vcat_dur, 3),
              "fps": fps, "width": width, "height": height,
              "profile": perf["profile"],
              "backend": "edit_engine.ffmpeg",
              "rungs": [{"stage": "assemble",
                         "rung": "ffmpeg/libx264 (software) — "
                                 "edit_engine.render.assemble_segments",
                         "note": f"profile {perf['profile']}, "
                                 f"preset {preset}, crf {crf}"}],
              "keyframes_applied": kf_applied,
              "effects_applied": fx_applied}
    if host_plan is not None:
        result["host"] = host_plan
    return result


def _fold_audio_boundaries(parts: list[str], labels: list[str],
                           durs: list[float],
                           transitions: list[Transition]) -> str:
    """Fold segment audio on the video's boundaries → final label.

    Overlap boundaries fold pairwise via acrossfade with the *same*
    overlap seconds the video fold computes (Fraction-exact), so a
    keep-audio assembly's sound and picture have identical duration;
    other boundaries concatenate.
    """
    if not labels:
        raise MediaEditError("no audio segments to fold")
    if len(labels) == 1:
        return labels[0]
    pieces: list[str] = []
    cur_lbl, cur_dur = labels[0], _exact(durs[0])
    for i, tr in enumerate(transitions):
        nxt_lbl, nxt_dur = labels[i + 1], _exact(durs[i + 1])
        if is_overlap(tr.kind):
            _name, td_f = xfade_args(tr.kind, tr.params, tr.duration)
            td = min(_exact(td_f), cur_dur, nxt_dur)
            if td < Fraction(1, 20):
                raise MediaEditError(
                    f"acrossfade needs ≥0.05 s overlap; segments "
                    f"{i}/{i + 1} are {float(cur_dur):.2f}s / "
                    f"{float(nxt_dur):.2f}s")
            parts.append(
                f"{cur_lbl}{nxt_lbl}acrossfade=d={float(td):.4f}"
                f"[axf{i}]")
            cur_lbl, cur_dur = f"[axf{i}]", cur_dur + nxt_dur - td
        else:
            pieces.append(cur_lbl)
            cur_lbl, cur_dur = nxt_lbl, nxt_dur
    pieces.append(cur_lbl)
    if len(pieces) == 1:
        return pieces[0]
    cat = "".join(pieces)
    parts.append(f"{cat}concat=n={len(pieces)}:v=0:a=1[afold]")
    return "[afold]"


def _exact(seconds: float) -> Fraction:
    """A float duration as the rational it names (Wave 1.2a).

    Floats enter as decimal intentions (0.1, 2.5, frame multiples);
    ``limit_denominator`` recovers that intention so fold arithmetic
    is exact instead of accumulating binary error.
    """
    return Fraction(seconds).limit_denominator(1_000_000)


def _fold_boundaries(parts: list[str], labels: list[str],
                     durs: list[float],
                     transitions: list[Transition]) -> tuple[str, float]:
    """Fold segment boundaries → (final_label, final_duration).

    Overlap kinds (fade/dissolve/whip_pan) fold pairwise via xfade;
    everything else concatenates. Returns a single label.

    Durations fold as exact Fractions (Wave 1.2a): an xfade offset
    deep into a long assembly is the exact sum of the durations
    before it, not a float that has drifted by accumulated rounding.
    """
    if not labels:
        raise MediaEditError("nothing to fold")
    if len(labels) == 1:
        return labels[0], durs[0]
    # group into runs: maximal runs of xfade boundaries fold together
    pieces: list[str] = []   # labels to concat at the end
    piece_durs: list[Fraction] = []
    cur_lbl, cur_dur = labels[0], _exact(durs[0])
    for i, tr in enumerate(transitions):
        nxt_lbl, nxt_dur = labels[i + 1], _exact(durs[i + 1])
        if is_overlap(tr.kind):
            name, td_f = xfade_args(tr.kind, tr.params, tr.duration)
            td = min(_exact(td_f), cur_dur, nxt_dur)
            if td < Fraction(1, 20):
                raise MediaEditError(
                    f"xfade needs ≥0.05 s overlap; segments {i}/{i + 1} "
                    f"are {float(cur_dur):.2f}s / {float(nxt_dur):.2f}s")
            parts.append(
                f"{cur_lbl}{nxt_lbl}xfade=transition={name}:"
                f"duration={float(td):.4f}:offset={float(cur_dur - td):.4f}"
                f"[xf{i}]")
            cur_lbl, cur_dur = f"[xf{i}]", cur_dur + nxt_dur - td
        else:
            pieces.append(cur_lbl)
            piece_durs.append(cur_dur)
            cur_lbl, cur_dur = nxt_lbl, nxt_dur
    pieces.append(cur_lbl)
    piece_durs.append(cur_dur)
    if len(pieces) == 1:
        return pieces[0], float(piece_durs[0])
    cat = "".join(pieces)
    parts.append(f"{cat}concat=n={len(pieces)}:v=1:a=0[vfold]")
    return "[vfold]", float(sum(piece_durs))


# ---------------------------------------------------------------------------
# timeline ramps — velocity over the assembled program (Wave 1.2)
# ---------------------------------------------------------------------------

def _apply_timeline_ramps(src: str | os.PathLike[str], *,
                          ramps: list[tuple[float, float, float]],
                          out: str | os.PathLike[str],
                          crf: int | None = None,
                          preset: str | None = None,
                          profile: str | None = None,
                          progress_cb: Any = None) -> dict[str, Any]:
    """Retim the assembled program: piecewise speed over [start, end].

    The session model's speed ops live on the *assembled* timeline
    (they span clip boundaries), so they cannot be clip effects. One
    ffmpeg run: video through the shared velocity-piece math
    (:func:`_velocity_pieces`, the same pieces the in-chain builder
    uses), audio through mirrored atempo pieces — picture and sound
    shrink/stretch together.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    if not ramps:
        raise MediaEditError("timeline ramps need at least one ramp")
    perf = _perf_for(profile)
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset
    info = video_probe(p)
    dur = float(info.get("duration") or 0.0)
    if dur <= 0:
        raise MediaEditError(f"could not determine duration of {src}")
    has_a = any(s.get("type") == "audio" for s in info.get("streams", []))
    norm = [(max(0.0, float(a)), min(dur, float(b)), float(s))
            for a, b, s in ramps if float(b) > float(a)]
    if not norm:
        raise MediaEditError("timeline ramps are all outside the program")
    pieces = _velocity_pieces(dur, norm)
    parts: list[str] = []
    fx = Effect(name="velocity_ramp", params={"ramps": norm})
    ctx = _ctx(int(info.get("width") or WIDTH),
               int(info.get("height") or HEIGHT), 30, dur, 0)
    final_v = _chain_velocity(parts, "0:v", fx, ctx, "ramp")
    maps = ["-map", f"[{final_v}]"]
    if has_a:
        if len(pieces) == 1 and pieces[0][2] == 1.0:
            parts.append("[0:a]anull[rampa]")
        else:
            outs = "".join(f"[ras{i}]" for i in range(len(pieces)))
            parts.append(f"[0:a]asplit={len(pieces)}{outs}")
            cat = ""
            for i, ((x0, x1, s), _lbl) in enumerate(
                    zip(pieces, [f"ras{i}" for i in range(len(pieces))])):
                chain = (f"[{_lbl}]atrim=start={x0:.4f}:end={x1:.4f},"
                         f"asetpts=PTS-STARTPTS")
                if abs(s - 1.0) > 1e-6:
                    chain += f",{_atempo_chain(s)}"
                chain += f"[raw{i}]"
                parts.append(chain)
                cat += f"[raw{i}]"
            parts.append(f"{cat}concat=n={len(pieces)}:v=0:a=1[rampa]")
        maps += ["-map", "[rampa]"]
    out_p = Path(out)
    cmd = ["-i", str(p), "-filter_complex", ";".join(parts), *maps,
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p"]
    if has_a:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    cmd.append(str(out_p))
    new_dur = sum((x1 - x0) / s for x0, x1, s in pieces)
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=dur)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "duration": round(new_dur, 3), "ramps": len(norm)}


# ---------------------------------------------------------------------------
# overlay tracks — per-track graph + the composite (Wave 1.2b)
# ---------------------------------------------------------------------------

_BLEND_XFADE = {"screen": "screen", "multiply": "multiply",
                "overlay": "overlay", "add": "addition"}


def _render_overlay_track(track: OverlayTrack, *,
                          duration: float, fps: int, width: int,
                          height: int, seed: int = 0,
                          keyframes: list[KeyframeTrack] | None = None,
                          tmp: Path, index: int,
                          rec: list[dict[str, Any]] | None = None) -> Path:
    """One overlay track → a full-duration alpha-carrying file.

    One ffmpeg graph: a transparent canvas, each clip placed at its
    ``at`` with its effects, expression keyframes (zoom) and opacity
    keyframes applied in-graph, and position keyframes driving the
    overlay() placement expressions. The intermediate is ffv1/mkv —
    lossless with an alpha channel, an honest intermediate format.
    Clip audio is not part of a picture track (audio rides the
    document's audio tracks / mix).

    Wave 5: sampled keyframes and masked effects ride the same
    preparation as the base assembly (see :func:`assemble_segments`);
    ``rec`` collects the per-effect record.
    """
    pw = int(track.width or width)
    ph = int(track.height or height)
    pos_tracks = [t for t in (keyframes or [])
                  if classify_track(t) == "position"]
    op_tracks = [t for t in (keyframes or [])
                 if classify_track(t) == "opacity"]
    ex_tracks = [t for t in (keyframes or [])
                 if classify_track(t) == "expression"]
    sm_tracks = [t for t in (keyframes or [])
                 if classify_track(t) == "sampled"]
    pos_by_clip = _match_keyframes(track.clips, pos_tracks)
    op_by_clip = _match_keyframes(track.clips, op_tracks)
    ex_by_clip = _match_keyframes(track.clips, ex_tracks)
    sm_by_clip = _match_keyframes(track.clips, sm_tracks)

    clips = _materialize_stills(list(track.clips), fps=fps, width=pw,
                                height=ph, crf=21, preset="veryfast",
                                profile="pc", tmp=tmp)
    masks = _MaskStreams()
    ov_windows = [_clip_window(c) for c in clips]
    prepared_fx: list[list[Effect]] = [
        _prepare_effects(c.effects, source_path=c.path,
                         source_start=ov_windows[i][0], width=pw,
                         height=ph, tmp=tmp, masks=masks,
                         tag_prefix=f"ov{index}c{i}_", fps=fps,
                         speed=float(c.speed),
                         play_frames=int(
                             (ov_windows[i][1] - ov_windows[i][0])
                             / max(float(c.speed), 1e-6) * fps) + 1)
        for i, c in enumerate(clips)]
    sampled_base: dict[tuple[int, str], dict[str, Any]] = {}
    consumed_static: set[int] = set()
    for ci, tracks in sm_by_clip.items():
        for kt in tracks:
            statics = [fx for fx in prepared_fx[ci]
                       if fx.name == kt.effect]
            if len(statics) > 1:
                raise MediaEditError(
                    f"keyframe track {kt.effect}.{kt.param}: overlay "
                    f"clip {ci} carries {len(statics)} static "
                    f"{kt.effect} effects — ambiguous base for the "
                    f"sampled curve")
            if statics:
                sampled_base[(ci, kt.effect)] = dict(
                    statics[0].params or {})
                consumed_static.add(id(statics[0]))
    order = sorted(range(len(clips)), key=lambda i: clips[i].at)
    cmd: list[str] = []
    cmd += ["-f", "lavfi", "-i",
            f"color=c=black@0.0:s={width}x{height}:r={fps}"]
    for clip in clips:
        if _is_still(clip.path):
            span = ((clip.end or 0.0) - clip.start) if clip.end \
                else (clip.duration or 3.0)
            cmd += ["-loop", "1", "-framerate", str(fps),
                    "-t", f"{max(span, 0.1):.4f}"]
        cmd += ["-i", clip.path]
    parts: list[str] = ["[0:v]format=yuva420p[ovbase]"]
    masks.preamble(parts, first_index=1 + len(clips), width=pw,
                   height=ph)
    cur = "ovbase"
    for n, ci in enumerate(order):
        clip = clips[ci]
        if _is_still(clip.path):
            s0, s1 = 0.0, (clip.duration or 3.0)
        else:
            s0, s1 = _clip_window(clip)
        speed = max(float(clip.speed), 1e-6)
        play_d = (s1 - s0) / speed
        tag = f"ov{n}"
        parts.append(f"[{ci + 1}:v]trim=start={s0:.4f}:end={s1:.4f},"
                     f"setpts=PTS-STARTPTS[{tag}raw]")
        vcur = f"{tag}raw"
        if abs(speed - 1.0) > 1e-6:
            parts.append(f"[{vcur}]setpts=PTS/{speed:.4f},settb=AVTB"
                         f"[{tag}spd]")
            vcur = f"{tag}spd"
        parts.append(
            f"[{vcur}]scale={pw}:{ph}:"
            f"force_original_aspect_ratio=increase,crop={pw}:{ph},"
            f"setsar=1,fps={fps}[{tag}norm]")
        vcur = f"{tag}norm"
        ctx = _ctx(pw, ph, fps, play_d, seed + ci * 131,
                   beats=[0.0, play_d])
        ctx["mask_labels"] = masks.labels
        rec_mark = len(rec) if rec is not None else 0
        for k, fx in enumerate(prepared_fx[ci]):
            if id(fx) in consumed_static:
                continue  # a sampled track owns this effect now
            vcur = _chain_effect(parts, vcur, fx, ctx, f"{tag}e{k}",
                                 rec=rec)
        for j, kt in enumerate(ex_by_clip.get(ci, [])):
            window = shifted(kt, 0.0, play_d)
            filt = keyframe_filter(window, width=pw, height=ph)
            parts.append(f"[{vcur}]{filt}[{tag}kf{j}]")
            vcur = f"{tag}kf{j}"
        for j, kt in enumerate(op_by_clip.get(ci, [])):
            parts.append(f"[{vcur}]format=yuva420p[{tag}alp{j}]")
            vcur = f"{tag}alp{j}"
            window = shifted(kt, 0.0, play_d)
            for w_i, (a, b, v) in enumerate(discretize(window)):
                sub = Effect(name="opacity",
                             params={"alpha": round(v, 4)}, at=(a, b))
                vcur = _chain_effect(parts, vcur, sub, ctx,
                                     f"{tag}op{j}_{w_i}")
        sm_tracks = sm_by_clip.get(ci, [])
        sm_groups = _group_sampled_tracks(
            [(kt, shifted(kt, 0.0, play_d)) for kt in sm_tracks],
            fps=fps)
        sm_group_of = {id(kt): g for g in sm_groups
                       for kt in g["tracks"]}
        sm_done: set[int] = set()
        for j, kt in enumerate(sm_tracks):
            g = sm_group_of[id(kt)]
            if id(g) in sm_done:
                continue  # chained with its group, at track one
            sm_done.add(id(g))
            base = sampled_base.get((ci, kt.effect), {})
            for w_i, (a, b, overlay) in enumerate(g["windows"]):
                sub_params = dict(base)
                sub_params.update(overlay)
                sub = Effect(name=kt.effect, params=sub_params,
                             at=(a, b))
                vcur = _chain_effect(parts, vcur, sub, ctx,
                                     f"{tag}sm{j}_{w_i}")
            if rec is not None:
                for t in g["tracks"]:
                    entry = _fx_record(
                        Effect(name=t.effect, params=dict(base)))
                    entry.update({
                        "kind": "keyframe-sampled",
                        "source": f"keyframe:{t.param}",
                        "windows": len(g["windows"])})
                    rec.append(entry)
        if rec is not None:
            for entry in rec[rec_mark:]:
                entry["clip"] = clip.id or ci
                entry["track"] = track.label or index
        parts.append(f"[{vcur}]format=yuva420p,"
                     f"setpts=PTS-STARTPTS+{clip.at:.4f}/TB[{tag}v]")
        # placement: position keyframes (clip-time → canvas-time by
        # substituting the expression's t), else the track's x/y.
        xs: Any = track.x if track.x is not None else 0
        ys: Any = track.y if track.y is not None else 0
        for kt in pos_by_clip.get(ci, []):
            expr = ffmpeg_expr(kt)
            expr = re.sub(r"\bt\b", f"(t-{clip.at:.3f})", expr)
            if kt.param == "x":
                xs = f"'{expr}'"
            else:
                ys = f"'{expr}'"
        parts.append(
            f"[{cur}][{tag}v]overlay=x={xs}:y={ys}:"
            f"enable='between(t,{clip.at:.4f},{clip.at + play_d:.4f})'"
            f"[{tag}c]")
        cur = f"{tag}c"
    out_p = tmp / f"overlay{index}.mkv"
    cmd += masks.input_args(fps=fps, duration=duration)
    cmd += ["-filter_complex", ";".join(parts), "-map", f"[{cur}]",
            "-t", f"{duration:.4f}", "-c:v", "ffv1", "-pix_fmt",
            "yuva420p", str(out_p)]
    run_ffmpeg(cmd, duration=duration)
    return out_p


def _composite_tracks(base: str | os.PathLike[str],
                      layers: list[tuple[OverlayTrack, Path]], *,
                      out: str | os.PathLike[str],
                      crf: int | None = None,
                      preset: str | None = None,
                      profile: str | None = None,
                      progress_cb: Any = None) -> dict[str, Any]:
    """Base + overlay track files → one composited mp4 (one graph).

    ``normal`` layers overlay with their alpha (track opacity folds
    into the alpha first); blend modes go through ffmpeg's blend
    with the track opacity as ``all_opacity``. Placement was already
    resolved inside each track's own graph.
    """
    perf = _perf_for(profile)
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset
    cmd: list[str] = ["-i", str(base)]
    for _t, p in layers:
        cmd += ["-i", str(p)]
    parts: list[str] = []
    cur = "0:v"
    applied: list[dict[str, Any]] = []
    for i, (track, _p) in enumerate(layers):
        src = f"{i + 1}:v"
        tag = f"cp{i}"
        if track.blend == "normal":
            parts.append(f"[{cur}]format=yuva420p[{tag}m]")
            prep = f"[{src}]format=yuva420p"
            if track.opacity < 1.0:
                prep += f",colorchannelmixer=aa={track.opacity:.4f}"
            prep += f"[{tag}o]"
            parts.append(prep)
            parts.append(f"[{tag}m][{tag}o]overlay=0:0[{tag}c]")
            cur = f"{tag}c"
        else:
            parts.append(f"[{cur}]format=yuva420p[{tag}m]")
            parts.append(f"[{src}]format=yuva420p[{tag}o]")
            parts.append(
                f"[{tag}m][{tag}o]blend="
                f"all_mode={_BLEND_XFADE[track.blend]}:"
                f"all_opacity={track.opacity:.4f}[{tag}c]")
            cur = f"{tag}c"
        applied.append({"label": track.label, "blend": track.blend,
                        "opacity": track.opacity,
                        "clips": len(track.clips)})
    parts.append(f"[{cur}]format=yuv420p[compout]")
    info = video_probe(str(base))
    has_a = any(s.get("type") == "audio" for s in info.get("streams", []))
    cmd += ["-filter_complex", ";".join(parts), "-map", "[compout]"]
    if has_a:
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", str(out)]
    run = run_ffmpeg(cmd, progress_cb=progress_cb)
    return {"output": str(out), "bytes": Path(out).stat().st_size,
            "seconds": run["seconds"], "tracks": applied}


# ---------------------------------------------------------------------------
# render_timeline — Timeline + text + audio → finished mp4
# ---------------------------------------------------------------------------

def _kf_targets(track: KeyframeTrack, clips: list[Clip]) -> bool:
    """Does ``track`` name one of ``clips`` (id → index → path)?"""
    for c in clips:
        if c.id and c.id == track.clip_id:
            return True
    if str(track.clip_id).isdigit() and int(track.clip_id) < len(clips):
        return True
    return any(c.path == track.clip_id for c in clips)


def render_timeline(timeline: Timeline, *,
                    segments: list[tuple[int, float]] | None = None,
                    cut_points: list[float] | None = None,
                    transitions: Any = "cut",
                    text_layers: list[TextLayer] | None = None,
                    audio_mix: AudioMix | None = None,
                    overlays: list[OverlayTrack] | None = None,
                    keyframes: list[KeyframeTrack] | None = None,
                    out: str | os.PathLike[str] | None = None,
                    crf: int | None = None,
                    quality: str | None = None,
                    preset: str | None = None,
                    profile: str | None = None,
                    keep_clip_audio: bool = False,
                    host: Any = None,
                    suffix: str = "timeline",
                    workdir: str | os.PathLike[str] | None = None,
                    progress_cb: Any = None) -> dict[str, Any]:
    """Render a :class:`Timeline` → finished mp4.

    1. base video track → assemble_segments (effects — static,
       mask-anchored and keyframed — transitions, expression +
       opacity + sampled keyframes)
    2. overlay tracks → per-track alpha renders → composite
       (blend modes, track opacity, position keyframes)
    3. audio_mix → mix_layers (or clip audio when keep_clip_audio)
    4. text_layers → burn_text_layers
    Returns the run report.

    ``overlays``/``keyframes`` default to the Timeline's own fields
    (Wave 1.2b/c — the renderer composites the canonical model, it
    does not detour through a second representation). Keyframe
    identity resolves base track first, then overlay tracks in
    order; a track aimed at nothing raises. ``host`` (1.2d) supplies
    the encode plan for the base assembly.

    ``quality`` picks a CRF preset (archival/high/balanced/social/draft);
    explicit ``crf=`` always wins. The returned dict carries a
    ``render_report`` (:class:`RenderReport`) — call ``.self_review()``
    on it for ffprobe assertions before presenting the file — plus
    the Wave-1.2 honesty record: ``backend``, per-stage ``rungs``,
    ``keyframes_applied`` and the composited ``tracks`` — plus the
    Wave-5 ``effects_applied`` record (every effect application with
    its producing rung, its source and its mask rung when anchored).
    """
    if not timeline.video.clips:
        raise MediaEditError("Timeline needs at least one video clip")
    crf = crf_for(quality, crf)
    tmp = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="tl-"))
    tmp.mkdir(parents=True, exist_ok=True)

    overlays_l = [t if isinstance(t, OverlayTrack)
                  else OverlayTrack.from_dict(dict(t))
                  for t in (getattr(timeline, "overlays", None)
                            if overlays is None else overlays) or []]
    kf_all = [t if isinstance(t, KeyframeTrack)
              else KeyframeTrack.from_dict(dict(t))
              for t in (getattr(timeline, "keyframes", None)
                        if keyframes is None else keyframes) or []]
    fam: dict[int, str] = {}
    for t in kf_all:
        try:
            fam[id(t)] = classify_track(t)
        except ValueError as exc:
            raise MediaEditError(str(exc)) from exc
    base_clips = list(timeline.video.clips)
    assemble_kf = [t for t in kf_all
                   if fam[id(t)] in ("expression", "opacity", "sampled")
                   and _kf_targets(t, base_clips)]
    consumed = {id(t) for t in assemble_kf}
    overlay_kf: dict[int, list[KeyframeTrack]] = {}
    for oi, ot in enumerate(overlays_l):
        rel = [t for t in kf_all if id(t) not in consumed
               and _kf_targets(t, ot.clips)]
        if rel:
            overlay_kf[oi] = rel
            consumed.update(id(t) for t in rel)
    for t in kf_all:
        if id(t) in consumed:
            continue
        if fam[id(t)] == "position" and _kf_targets(t, base_clips):
            raise MediaEditError(
                f"position keyframes target base-track clip "
                f"{t.clip_id!r} — position animates overlay placement; "
                f"put that clip on an overlay track")
        raise MediaEditError(
            f"keyframe track targets clip {t.clip_id!r}, which is not "
            f"in this timeline")

    video = assemble_segments(
        base_clips, segments=segments, cut_points=cut_points,
        transitions=transitions,
        out=tmp / "video.mp4", fps=timeline.fps,
        width=timeline.width, height=timeline.height, seed=timeline.seed,
        crf=crf, preset=preset, profile=profile,
        keep_audio=keep_clip_audio, keyframes=assemble_kf, host=host,
        workdir=tmp, progress_cb=progress_cb)
    final = Path(video["output"])
    video_dur = video["duration"]
    rungs: list[dict[str, Any]] = list(video.get("rungs", []))
    kf_applied: list[dict[str, Any]] = list(
        video.get("keyframes_applied", []))
    fx_applied: list[dict[str, Any]] = list(
        video.get("effects_applied", []))
    track_report: list[dict[str, Any]] = []
    comp_out: dict[str, Any] | None = None

    live_layers: list[tuple[OverlayTrack, Path]] = []
    for oi, ot in enumerate(overlays_l):
        if ot.muted or not ot.clips:
            track_report.append({"label": ot.label, "blend": ot.blend,
                                 "rendered": False,
                                 "reason": "muted" if ot.muted
                                 else "no clips"})
            continue
        rendered = _render_overlay_track(
            ot, duration=video_dur, fps=timeline.fps,
            width=timeline.width, height=timeline.height,
            seed=timeline.seed, keyframes=overlay_kf.get(oi),
            tmp=tmp, index=oi, rec=fx_applied)
        live_layers.append((ot, rendered))
        rungs.append({
            "stage": f"overlay:{ot.label or oi}",
            "rung": "ffmpeg/ffv1 (alpha intermediate) — "
                    "edit_engine.render._render_overlay_track",
            "note": f"{ot.blend} track over {len(ot.clips)} clip(s)"})
        for t in overlay_kf.get(oi, []):
            kf_applied.append({"clip_id": t.clip_id, "effect": t.effect,
                               "param": t.param, "family": fam[id(t)],
                               "track": ot.label or oi})
    if live_layers:
        comp_out = _composite_tracks(
            final, live_layers, out=tmp / "composited.mp4",
            crf=crf, preset=preset, profile=profile,
            progress_cb=progress_cb)
        final = Path(comp_out["output"])
        track_report.extend(comp_out["tracks"])
        rungs.append({
            "stage": "composite",
            "rung": "ffmpeg/libx264 (software) — "
                    "edit_engine.render._composite_tracks",
            "note": f"{len(live_layers)} overlay track(s)"})

    audio_out: dict[str, Any] | None = None
    mix = (audio_mix if audio_mix is not None else timeline.audio_mix)
    if mix is not None and mix.layers:
        audio_out = mix_layers(mix, video_dur, tmp / "mix.m4a")
        muxed = tmp / "muxed.mp4"
        run_ffmpeg(["-i", str(final), "-i", audio_out["output"],
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(muxed)], duration=video_dur)
        final = muxed
        rungs.append({"stage": "audio",
                      "rung": "ffmpeg — edit_engine.audio.mix_layers",
                      "note": f"{len(mix.layers)} layer(s)"})
    elif keep_clip_audio:
        pass  # already in the video stream

    layers = [l if isinstance(l, TextLayer) else TextLayer.from_dict(dict(l))
              for l in (text_layers if text_layers is not None
                        else timeline.text_layers)]
    text_out: dict[str, Any] | None = None
    caption_timing: dict[str, int] = {}
    if layers:
        text_out = burn_text_layers(final, layers, width=timeline.width,
                                    height=timeline.height,
                                    out=tmp / "captioned.mp4",
                                    workdir=tmp)
        final = Path(text_out["output"])
        for l in layers:
            src = l.timing_source or "authored"
            caption_timing[src] = caption_timing.get(src, 0) + 1
        rungs.append({
            "stage": "text",
            "rung": "ffmpeg drawtext — edit_engine.text.burn_text_layers",
            "note": "timing sources: " + ", ".join(
                f"{k}×{v}" for k, v in sorted(caption_timing.items()))})

    if out:
        out_p = Path(out)
    else:
        out_p = Path(timeline.video.clips[0].path).parent / (
            f"{Path(timeline.video.clips[0].path).stem}.{suffix}.mp4")
    if final.resolve() != out_p.resolve():
        run_ffmpeg(["-i", str(final), "-c", "copy", str(out_p)],
                   duration=video_dur)
    if fx_applied:
        masked_n = sum(1 for e in fx_applied if "mask" in e)
        rungs.append({
            "stage": "effects",
            "rung": "edit_engine.effects (per-effect property "
                    "builders, Wave 5 registry)",
            "note": f"{len(fx_applied)} effect application(s)"
                    + (f", {masked_n} mask-anchored" if masked_n
                       else "")
                    + " — the per-effect record is in "
                      "effects_applied"})
    report = RenderReport(
        output=str(out_p), bytes=out_p.stat().st_size,
        duration=video_dur, width=timeline.width, height=timeline.height,
        fps=float(timeline.fps or 0), crf=crf, quality=quality or "",
        filter_graph_path=str(video.get("filter_graph", "")))
    try:
        import hashlib
        h = hashlib.sha256()
        with open(out_p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        report.sha256 = h.hexdigest()
    except Exception as e:  # hash is best-effort
        record_failure("render.render_timeline:sha256", e)
        pass
    result = {"output": str(out_p), "bytes": out_p.stat().st_size,
              "duration": video_dur, "profile": video["profile"],
              "video": video, "audio": audio_out, "text": text_out,
              "workdir": str(tmp), "render_report": report,
              "quality": quality or "", "crf": crf,
              "backend": "edit_engine.ffmpeg", "rungs": rungs,
              "keyframes_applied": kf_applied,
              "effects_applied": fx_applied, "tracks": track_report,
              "caption_timing": caption_timing}
    if video.get("host") is not None:
        result["host"] = video["host"]
    return result


# ---------------------------------------------------------------------------
# render_document — the canonical Document, composited (Wave 1.2)
# ---------------------------------------------------------------------------

#: Wave-1 session (D1) effect kinds the video session façade stamps
#: onto clips. They are directives, not canonical effects: the
#: interpreter below gives each a real engine rung or fails honestly.
_SESSION_DIRECTIVES = frozenset({
    "v_transition", "v_title", "v_lower_third", "v_speed", "v_trim",
    "v_duck", "v_chapter", "v_export",
})


def _session_transition(params: dict[str, Any]) -> Transition:
    """A session transition op → an engine Transition.

    Session kinds come from media_edit's TRANSITIONS vocabulary;
    the engine's kinds are cut/fade/dissolve/flash/dip/whip_pan/
    glitch_cut. Direct names map directly; slide/smooth kinds ride
    the engine's whip_pan xfade (named in the note the caller
    records); fadeblack/fadewhite map to the dip/flash blink kinds.
    Wipe and circle kinds have no engine rung — they raise, here,
    rather than being silently approximated.
    """
    kind = str(params.get("type") or params.get("kind")
               or "fade").strip().lower()
    dur = float(params.get("duration", 0.5))
    if kind in ("fade", "dissolve"):
        return Transition(kind=kind, duration=dur)
    if kind in ("cut", "hardcut", "hard"):
        return Transition(kind="cut")
    if kind == "fadeblack":
        return Transition(kind="dip", duration=dur)
    if kind == "fadewhite":
        return Transition(kind="flash", duration=dur)
    for prefix in ("slide", "smooth"):
        if kind.startswith(prefix):
            direction = kind[len(prefix):]
            if direction in ("left", "right", "up", "down"):
                return Transition(kind="whip_pan", duration=dur,
                                  params={"direction": direction})
    raise MediaEditError(
        f"session transition kind {kind!r} has no engine rung "
        f"(engine kinds: cut/fade/dissolve/flash/dip/whip_pan/"
        f"glitch_cut; slide/smooth map to whip_pan)")


def _title_card_png(out: Path, params: dict[str, Any], *,
                    width: int, height: int) -> Path:
    """A title card PNG at the document's canvas size.

    The session compiler's recipe (media_edit.studio._render_title_card:
    solid bg, optional accent bar, centered shadowed text, optional
    subtitle) rendered at the document's own dimensions instead of
    the compiler's fixed 1920×1080.
    """
    from PIL import Image, ImageDraw

    from ...media_edit.studio import render_text_layer

    bg = str(params.get("bg") or params.get("background")
             or "#14161c")
    canvas = Image.new("RGB", (width, height), bg)
    accent = params.get("accent")
    if accent:
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, height // 2 - 140, width, height // 2 - 128],
                    fill=accent)
    text = str(params.get("text", ""))
    sub = str(params.get("subtitle", "") or "")
    if text:
        layer = render_text_layer(
            text, font=params.get("font"),
            size=int(params.get("font_size",
                                params.get("fontsize", 96))),
            color=str(params.get("color", "white")), align="center",
            box_width=width,
            letter_spacing=int(params.get("letter_spacing", 2)),
            shadow=True)
        x = (width - layer.width) // 2
        y = (height - layer.height) // 2 - (40 if sub else 0)
        canvas.paste(layer, (x, y), layer)
    if sub:
        layer = render_text_layer(
            sub, font=params.get("font"),
            size=int(params.get("subtitle_size", 44)),
            color=str(params.get("subtitle_color", "#9aa0aa")),
            align="center", box_width=width)
        x = (width - layer.width) // 2
        y = height // 2 + 60
        canvas.paste(layer, (x, y), layer)
    canvas.save(out, "PNG")
    return out


def _doc_play_duration(clip: Any) -> float:
    """A document clip's play duration (probe when not authored)."""
    if clip.duration:
        return float(clip.duration) / max(float(clip.speed), 1e-6)
    info = video_probe(clip.path)
    end = clip.end if clip.end is not None else float(
        info.get("duration") or 0.0)
    return max(0.0, (float(end) - float(clip.start))) / max(
        float(clip.speed), 1e-6)


def _session_directives(doc: Any, tmp: Path, *, width: int,
                        height: int) -> tuple[Any, list[Any] | None,
                                              dict[str, Any],
                                              list[dict[str, Any]]]:
    """Interpret the session (``v_*``) directives on the base track.

    Returns (cleaned document, boundary transitions | None,
    post-passes, notes). The cleaned document carries only canonical
    effects; titles become still clips, lower-thirds become text
    layers, and speed/trim/duck/chapter become named post-passes the
    caller executes in order. A directive with no rung raises —
    silently dropping an authored edit is the failure this wave
    exists to kill.

    Directive times are authored on the sequential (cut) timeline;
    an overlap transition compresses everything after its boundary
    by its own duration, the same reading the session compiler gave
    them.
    """
    from .document import Document

    doc2 = Document.from_dict(doc.to_dict())
    notes: list[dict[str, Any]] = []
    post: dict[str, Any] = {"ramps": [], "trims": [], "duck": None,
                            "chapters": []}
    base = doc2.picture_tracks()[0] if doc2.picture_tracks() else None
    transitions: list[Any] | None = None
    if base is not None:
        spans: list[float] = []
        acc = 0.0
        for c in base.clips:
            spans.append(acc)
            acc += _doc_play_duration(c)
        new_clips: list[Any] = []
        declared: list[Transition] = []
        title_n = 0
        shift = 0.0            # title durations inserted so far

        for ci, clip in enumerate(base.clips):
            kept: list[Effect] = []
            span_here = spans[ci] + shift
            for fx in clip.effects:
                name = fx.name
                params = fx.params or {}
                if name not in _SESSION_DIRECTIVES:
                    kept.append(fx)
                    continue
                if name == "v_title":
                    text = str(params.get("text", "")).strip()
                    if not text:
                        raise MediaEditError(
                            "v_title directive carries no text")
                    dur = float(params.get("duration", 2.0))
                    png = _title_card_png(
                        tmp / f"title{title_n}.png", params,
                        width=width, height=height)
                    title_n += 1
                    new_clips.append(Clip(
                        path=str(png), duration=dur,
                        id=f"title{title_n}"))
                    shift += dur
                    span_here += dur
                    notes.append({"directive": "v_title",
                                  "detail": f"title card {dur:.1f}s "
                                            f"before clip {ci}"})
                elif name == "v_transition":
                    tr = _session_transition(params)
                    declared.append(tr)
                    notes.append({
                        "directive": "v_transition",
                        "detail": f"{params.get('type') or params.get('kind')} "
                                  f"→ engine {tr.kind} "
                                  f"{tr.duration:.2f}s (boundary "
                                  f"{len(declared) - 1})"})
                elif name == "v_lower_third":
                    text = str(params.get("text", "")).strip()
                    if not text:
                        raise MediaEditError(
                            "v_lower_third directive carries no text")
                    if params.get("start") is not None:
                        at = float(params["start"])
                    else:
                        at = span_here + float(
                            fx.at[0] if fx.at else 0.0)
                    dur = float(params.get("duration", 3.5))
                    doc2.text_layers.append(TextLayer(
                        text=text, start=round(at, 3),
                        end=round(at + dur, 3),
                        position="bottom-left", size=40,
                        color="#ffffff", box=True,
                        box_color="#00000088"))
                    notes.append({"directive": "v_lower_third",
                                  "detail": f"{text[:24]!r} at {at:.1f}s"})
                elif name == "v_speed":
                    if ("factor" in params or "start" in params
                            or "end" in params):
                        # session shape: timeline-absolute window
                        speed = float(params.get(
                            "factor", params.get("speed", 2.0)))
                        t0 = float(params.get("start") or 0.0)
                        t1 = (float(params["end"])
                              if params.get("end") is not None else None)
                    else:
                        # hand-authored shape: clip-relative window
                        speed = float(params.get("speed", 2.0))
                        a, b = fx.at if fx.at else (0.0, None)
                        t0 = span_here + float(a)
                        t1 = (span_here + float(b)
                              if b is not None else None)
                    post["ramps"].append((t0, t1, speed))
                    notes.append({"directive": "v_speed",
                                  "detail": f"{speed}× over "
                                            f"[{t0:.1f}, "
                                            f"{t1 if t1 is not None else 'end'}]"})
                elif name == "v_trim":
                    post["trims"].append({
                        "start": float(params.get("start", 0.0)),
                        "end": params.get("end")})
                    notes.append({"directive": "v_trim",
                                  "detail": "start "
                                            f"{float(params.get('start', 0.0)):.1f}s"})
                elif name == "v_duck":
                    if post["duck"] is None:
                        post["duck"] = {
                            "amount_db": float(params.get(
                                "amount_db", 12.0)),
                            "bgm": params.get("bgm"),
                            "bgm_gain": float(params.get(
                                "bgm_gain", 1.0))}
                        notes.append({"directive": "v_duck",
                                      "detail": "music ducked "
                                                f"{float(params.get('amount_db', 12.0)):.0f} dB "
                                                "under program audio"})
                elif name == "v_chapter":
                    title = str(params.get("name")
                                or params.get("title") or "")
                    if params.get("at") is not None:
                        at = float(params["at"])
                    else:
                        at = span_here + float(
                            fx.at[0] if fx.at else 0.0)
                    post["chapters"].append({
                        "title": title, "at": round(at, 3),
                        "end": params.get("end")})
                    notes.append({"directive": "v_chapter",
                                  "detail": f"{title!r} at {at:.1f}s"})
                elif name == "v_export":
                    notes.append({
                        "directive": "v_export",
                        "detail": f"preset {params.get('preset')!r} — "
                                  f"honored by the session façade as "
                                  f"output hints, not an engine edit"})
            clip.effects = tuple(kept)
            new_clips.append(clip)
        base.clips = new_clips
        n_bounds = len(new_clips) - 1
        if declared and n_bounds > 0:
            if len(declared) == 1:
                boundaries: list[Any] = [declared[0]] * n_bounds
            elif len(declared) <= n_bounds:
                boundaries = list(declared) + ["cut"] * (
                    n_bounds - len(declared))
            else:
                raise MediaEditError(
                    f"{len(declared)} transitions declared for "
                    f"{n_bounds} clip boundaries")
            if any(b != "cut" for b in boundaries):
                transitions = boundaries
    return doc2, transitions, post, notes


def _look_lock_effects(doc: Any) -> tuple[list[Effect],
                                          dict[str, Any] | None]:
    """A Document's ``look_lock`` as engine effects (plan §1.1c(d)).

    The look is timeline state: the renderable keys become
    ``color_grade``/``grain`` effects appended to every base-track
    clip — applied in the single render graph through the canonical
    grade chain, never a separate re-encode pass. ``grade`` names a
    canonical grade; ``cdl`` carries an ASC-CDL decision (baked by
    studio_color); ``grain`` is a strength. The remaining LookLock
    keys (``name``/``palette``/``lens``) steer generation and QC,
    not the compositor — they are reported as carried, never
    silently treated as applied.
    """
    lock = getattr(doc, "look_lock", None)
    if not lock:
        return [], None
    fx: list[Effect] = []
    applied: list[str] = []
    cdl = lock.get("cdl")
    grade = lock.get("grade")
    if cdl:
        fx.append(Effect(name="color_grade", params={
            "cdl": dict(cdl), "_source": "look_lock"}))
        applied.append("color_grade(cdl)")
    elif grade:
        fx.append(Effect(name="color_grade", params={
            "grade": str(grade), "_source": "look_lock"}))
        applied.append(f"color_grade({grade})")
    grain = lock.get("grain")
    if grain:
        fx.append(Effect(name="grain", params={
            "strength": float(grain), "_source": "look_lock"}))
        applied.append(f"grain({float(grain):g})")
    carried = sorted(k for k in lock
                     if k not in ("grade", "cdl", "grain"))
    return fx, {"name": str(lock.get("name", "")),
                "applied": applied, "carried": carried}


def render_document(doc: Any, *,
                    out: str | os.PathLike[str] | None = None,
                    host: Any = None,
                    workdir: str | os.PathLike[str] | None = None,
                    progress_cb: Any = None,
                    suffix: str = "document") -> dict[str, Any]:
    """Render the canonical :class:`~.document.Document` end-to-end.

    The Wave-1.2 convergence point: the document → EditSpec →
    Timeline → :func:`render_timeline` path composites multi-track
    picture, keyframes, text and audio in the engine itself (the
    compile-via-media_edit detour is retired for video renders).

    Session (``v_*``) directives stamped by the EditStudio façade are
    interpreted onto real rungs (titles, transitions, lower-thirds,
    timeline speed ramps, trim, sidechain duck, chapters) — a
    directive without a rung raises instead of being dropped.

    Post-pass order: render → trims → speed ramps → duck → chapters.
    The result is render_timeline's report enriched with
    ``renderer``, the ``document`` derivative manifest,
    ``session_directives`` and the post-pass records in ``rungs``.
    """
    from .document import Document

    if not isinstance(doc, Document):
        doc = Document.from_dict(doc)
    if not doc.picture_tracks():
        raise MediaEditError("Document has no picture track to render")
    tmp = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="doc-"))
    tmp.mkdir(parents=True, exist_ok=True)
    spec = doc.output
    doc2, transitions, post, notes = _session_directives(
        doc, tmp, width=int(spec.get("width", 1080)),
        height=int(spec.get("height", 1920)))
    # the Document's look_lock is grade state on the base track
    # (Wave 5): its effects ride the clips into the single render
    # graph, after each clip's own effects. Overlay tracks carry
    # their own effects — the look grades the program track.
    look_fx, look_note = _look_lock_effects(doc2)
    if look_fx and doc2.picture_tracks():
        for c in doc2.picture_tracks()[0].clips:
            c.effects = tuple(c.effects) + tuple(
                Effect(name=f.name, params=dict(f.params), at=f.at)
                for f in look_fx)
    tl = doc2.to_edit_spec().to_timeline()
    # a document's clips carry their own audio (the session model's
    # reading — the old compiler kept it too); an authored audio mix
    # still replaces it in render_timeline's mux step.
    rt_kwargs: dict[str, Any] = {"keep_clip_audio": True}
    if post["duck"] is not None:
        # ducking mixes music under the program's own audio, so the
        # document's audio mix must NOT be baked in first — the duck
        # pass below does the mix, sidechained on the program audio.
        rt_kwargs["audio_mix"] = AudioMix(layers=[])
    res = render_timeline(
        tl, transitions=transitions if transitions is not None else "cut",
        out=tmp / "program.mp4", host=host, workdir=tmp,
        crf=(int(spec["crf"]) if spec.get("crf") else None),
        profile=str(spec.get("profile") or "") or None,
        progress_cb=progress_cb, **rt_kwargs)
    cur = Path(res["output"])
    cur_dur = float(res["duration"])
    rungs: list[dict[str, Any]] = list(res.get("rungs", []))
    crf = int(spec.get("crf") or 20)

    for i, tr in enumerate(post["trims"]):
        start = float(tr["start"])
        end = tr["end"]
        dur = (float(end) - start) if end is not None else None
        nxt = tmp / f"trim{i}.mp4"
        cmd = ["-ss", f"{start:.3f}", "-i", str(cur)]
        if dur is not None:
            cmd += ["-t", f"{dur:.3f}"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf",
                str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "160k", str(nxt)]
        run_ffmpeg(cmd, duration=cur_dur)
        cur = nxt
        cur_dur = float(video_probe(cur).get("duration") or cur_dur)
        rungs.append({"stage": "session:v_trim",
                      "rung": "ffmpeg trim — edit_engine.render."
                              "render_document",
                      "note": f"start {start:.1f}s"})
    if post["ramps"]:
        nxt = tmp / "ramped.mp4"
        resolved = [(max(0.0, min(a, cur_dur)),
                     min(b if b is not None else cur_dur, cur_dur), s)
                    for a, b, s in post["ramps"]]
        ramped = _apply_timeline_ramps(
            cur, ramps=resolved, out=nxt, crf=crf,
            progress_cb=progress_cb)
        cur = nxt
        cur_dur = float(ramped["duration"])
        rungs.append({"stage": "session:v_speed",
                      "rung": "ffmpeg setpts/atempo pieces — "
                              "edit_engine.render._apply_timeline_ramps",
                      "note": f"{len(post['ramps'])} ramp(s), video and "
                              f"audio retimed together"})
    if post["duck"] is not None:
        info = video_probe(cur)
        has_a = any(s.get("type") == "audio"
                    for s in info.get("streams", []))
        if not has_a:
            raise MediaEditError(
                "v_duck declared but the rendered program has no "
                "audio to duck under")
        music = post["duck"].get("bgm")
        if not music:
            for t in doc2.tracks:
                if t.kind == "audio" and t.clips:
                    music = t.clips[0].path
                    break
        if not music:
            raise MediaEditError(
                "v_duck declared but no music source: the directive "
                "carries no bgm and the document has no audio track")
        prog_audio = tmp / "program_audio.m4a"
        run_ffmpeg(["-i", str(cur), "-vn", "-c:a", "aac", "-b:a",
                    "192k", str(prog_audio)], duration=cur_dur)
        amount = float(post["duck"]["amount_db"])
        level = min(1.0, max(0.05, 10 ** (-amount / 20.0)))
        mix = AudioMix(
            layers=[AudioLayer(path=str(prog_audio)),
                    AudioLayer(path=music,
                               volume=float(post["duck"].get(
                                   "bgm_gain", 1.0)))],
            ducking="sidechain", duck_key=0, duck_level=level)
        duck_mix = mix_layers(mix, cur_dur, tmp / "duck_mix.m4a")
        mix_src = duck_mix["output"]
        nxt = tmp / "ducked.mp4"
        run_ffmpeg(["-i", str(cur), "-i", mix_src, "-map", "0:v",
                    "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", str(nxt)],
                   duration=cur_dur)
        cur = nxt
        rungs.append({"stage": "session:v_duck",
                      "rung": "ffmpeg sidechaincompress — "
                              "edit_engine.audio.mix_layers",
                      "note": f"music at {level:.2f} under program "
                              f"audio"})
    if post["chapters"]:
        chapters = sorted(post["chapters"], key=lambda c: c["at"])
        lines = [";FFMETADATA1"]
        for i, ch in enumerate(chapters):
            start_ms = int(ch["at"] * 1000)
            end_at = ch.get("end")
            if end_at is None:
                end_at = (chapters[i + 1]["at"] if i + 1 < len(chapters)
                          else cur_dur)
            title = str(ch["title"]).replace("\n", " ")
            lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                      f"START={start_ms}",
                      f"END={max(start_ms + 1, int(end_at * 1000))}",
                      f"title={title}"]
        meta = tmp / "chapters.txt"
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
        nxt = tmp / "chaptered.mp4"
        run_ffmpeg(["-i", str(cur), "-i", str(meta), "-map", "0",
                    "-map_chapters", "1", "-c", "copy", str(nxt)],
                   duration=cur_dur)
        cur = nxt
        rungs.append({"stage": "session:v_chapter",
                      "rung": "ffmpeg ffmetadata remux — "
                              "edit_engine.render.render_document",
                      "note": f"{len(chapters)} chapter(s)"})

    if out:
        out_p = Path(out)
    else:
        first = doc2.picture_tracks()[0].clips[0]
        out_p = Path(first.path).parent / (
            f"{Path(first.path).stem}.{suffix}.mp4")
    if cur.resolve() != out_p.resolve():
        run_ffmpeg(["-i", str(cur), "-c", "copy", str(out_p)],
                   duration=cur_dur)
    if look_note is not None:
        rungs.append({
            "stage": "look_lock",
            "rung": "edit_engine.effects.color_grade_chain / grain "
                    "(Document.look_lock as base-track effects)",
            "note": "applied: "
                    + (", ".join(look_note["applied"]) or "none")
                    + "; carried (generation/QC keys, not compositor "
                      "state): "
                    + (", ".join(look_note["carried"]) or "none")})
    res.update({
        "output": str(out_p), "bytes": out_p.stat().st_size,
        "duration": round(cur_dur, 3),
        "renderer": "edit_engine.render.render_document",
        "rungs": rungs,
        "session_directives": notes,
        "document": {
            "tracks": [t.kind for t in doc.tracks],
            "clips": sum(len(t.clips) for t in doc.tracks),
            "keyframes": len(doc.keyframes),
            "text_layers": len(doc.text_layers),
            "look_lock": look_note,
            "output": dict(spec),
        },
    })
    return res
