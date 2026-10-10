"""Long-form assembly — the "multi-clip cinema" foundation.

A film is a list of scenes (data):

    {"prompt": "neon alley in rain, slow dolly",
     "mode": "t2v", "duration_s": 5.0, "seed": 42}

The chainer generates each clip with the neural backend (LTX fast pass,
Wan finals), chains them with real transitions, grades the whole cut,
and exports the delivery format. Scene prompts get a consistent style
suffix so multi-clip output feels like one film, not five demos.

When the neural backend isn't available the chainer degrades honestly:
image-mode scenes render through the Ken Burns engine (still → motion),
text scenes through typography — the film still gets made, and the
report says exactly which scenes are neural and which are motion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .capabilities import VideogenError, neural_capability
from ..motion_studio._core import (
    new_render_path,
    profile_defaults,
    record_ledger,
)
from ..motion_studio.grading import export, grade
from ..motion_studio.kenburns import kenburns
from ..motion_studio.montage import Segment, assemble
from ..motion_studio.typography import render_quote_card
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["chain_scenes", "ChainReport", "STYLE_SUFFIXES"]

#: style suffix appended to every scene prompt for visual consistency
STYLE_SUFFIXES: dict[str, str] = {
    "cinematic": "cinematic film still, dramatic lighting, 35mm, shallow depth of field",
    "anime": "anime film still, vibrant, detailed background art",
    "noir": "film noir, high contrast black and white, moody",
    "documentary": "documentary footage, natural light, handheld realism",
    "music_video": "music video still, bold color, stylized lighting",
    "none": "",
}


@dataclass
class SceneResult:
    index: int
    path: str
    engine: str          # "ltx" | "wan" | "kenburns" | "typography"
    prompt: str


@dataclass
class ChainReport:
    final_path: str
    scenes: list[SceneResult] = field(default_factory=list)
    neural_scenes: int = 0
    motion_scenes: int = 0
    grade_preset: str = ""
    format: str = ""
    note: str = ""
    consistency: dict = field(default_factory=dict)  # boundary scores

    def summary(self) -> str:
        from ..style import theme as _theme
        th = _theme()
        lines = [th.banner("Chained Film",
                           f"{len(self.scenes)} scenes · "
                           f"{self.neural_scenes} neural / "
                           f"{self.motion_scenes} motion-graphics")]
        for s in self.scenes:
            lines.append(f"  [{s.index + 1}] [{s.engine}] "
                         f"{s.prompt[:64]}")
        bnds = (self.consistency or {}).get("boundaries") or []
        if bnds:
            lines.append(th.section("Boundary consistency"))
            for b in bnds:
                if "score_after" in b:
                    lines.append(
                        f"  {b['a'] + 1}→{b['b'] + 1}: "
                        f"{th.bar(b['score_before'])} → "
                        f"{th.bar(b['score_after'])}")
        if self.note:
            lines.append(th.status_line(None, "note", self.note))
        lines.append(th.status_line(True, "film ready", self.final_path))
        return "\n".join(lines)


def _backend(name: str):
    from .ltx_backend import LTXBackend
    from .wan_backend import WanBackend
    if name == "wan":
        return WanBackend()
    return LTXBackend()


def chain_scenes(scenes: Sequence[dict[str, Any] | str],
                 out: str | os.PathLike | None = None, *,
                 backend: str = "auto",
                 style: str = "cinematic",
                 transition: str = "crossfade",
                 grade_preset: str = "cinematic",
                 format: str = "16:9",
                 seed: int = 100,
                 size: tuple[int, int] | None = None,
                 fps: float | None = None,
                 consistency: bool = True,
                 carry_seed: bool = True,
                 carry_reference: bool = True) -> ChainReport:
    """Assemble a multi-scene film. Returns a :class:`ChainReport`.

    Each scene: ``{"prompt", "mode": "t2v"|"i2v"|"image"|"text",
    "image"|"text", "duration_s", "seed"}`` — or a bare prompt string
    (t2v, 5s). ``backend``: auto | ltx | wan | motion (force motion studio).
    ``consistency``: grade each clip's opening frames toward the
    previous clip's closing frame so cuts don't pop (color + structure
    matched on the boundary frames; skipped honestly without ffmpeg).
    ``carry_seed``: derive each scene's seed from the master seed +
    scene index via SHA-256 (reproducible, prompt-order stable).
    ``carry_reference``: feed scene N's last frame as scene N+1's
    start image (i2v) so neural scenes don't drift apart.
    """
    if not scenes:
        raise VideogenError("no scenes — nothing to chain")
    import hashlib

    def _scene_seed(i: int, prompt: str, override: Any) -> int:
        if override is not None:
            return int(override)
        if carry_seed:
            h = hashlib.sha256(
                f"{seed}:{i}:{prompt}".encode()).hexdigest()
            return int(h[:8], 16)
        return seed + i * 17

    suffix = STYLE_SUFFIXES.get(style, STYLE_SUFFIXES["cinematic"])
    cap = neural_capability(prefer="auto" if backend == "auto" else backend)
    use_neural = cap.available and backend != "motion"
    engine_name = cap.backend if use_neural else "motion"
    note = ""
    if backend != "motion" and not use_neural:
        note = (f"neural unavailable ({cap.reason}); all scenes rendered "
                f"with the motion studio")
    gen = _backend(cap.backend) if use_neural else None
    if gen is not None:
        gen.require()

    results: list[SceneResult] = []
    neural_n = motion_n = 0
    clips: list[str] = []
    for i, raw in enumerate(scenes):
        spec = {"prompt": raw} if isinstance(raw, str) else dict(raw)
        prompt = str(spec.get("prompt", "")).strip()
        mode = str(spec.get("mode", "t2v")).lower()
        dur = float(spec.get("duration_s", 5.0))
        sseed = _scene_seed(i, prompt, spec.get("seed"))
        full_prompt = f"{prompt}, {suffix}" if suffix and mode in ("t2v", "i2v") else prompt

        if use_neural and mode in ("t2v", "i2v"):
            ref_image = spec.get("image")
            ref_mode = mode
            # reference carryover: scene N+1 starts from scene N's last
            # frame so chained neural scenes don't drift apart.
            if (carry_reference and mode == "t2v" and not ref_image
                    and results and i > 0):
                try:
                    import tempfile
                    from ...media_edit.videos import extract_frames
                    from ..motion_studio._core import probe_duration as _pd
                    prev_dur = _pd(results[-1].path)
                    tmpd = tempfile.mkdtemp(prefix="chain-ref-")
                    fr = extract_frames(
                        results[-1].path, out_dir=tmpd,
                        timestamps=[max(0.0, prev_dur - 0.1)])
                    frames = fr.get("frames") or []
                    if frames:
                        ref_image = frames[-1]
                        ref_mode = "i2v"
                        note = (note + "; " if note else "") + \
                            f"scene {i}: reference carried from scene {i - 1}"
                except Exception as exc:  # noqa: BLE001 — chain goes on
                    _log.info("chain: reference carryover skipped: %s", exc)
            path = gen.generate(full_prompt, mode=ref_mode,
                                image=ref_image,
                                duration_s=dur, seed=sseed)
            results.append(SceneResult(i, path, gen.name, prompt))
            neural_n += 1
        elif mode == "image" and spec.get("image"):
            path = kenburns(spec["image"], duration=dur, move="auto",
                            seed=sseed, size=size, fps=fps)
            results.append(SceneResult(i, path, "kenburns", prompt or str(spec["image"])))
            motion_n += 1
        elif mode == "text":
            path = render_quote_card(spec.get("text", prompt) or "…",
                                     duration=dur, seed=sseed,
                                     size=size, fps=fps)
            results.append(SceneResult(i, path, "typography", prompt))
            motion_n += 1
        elif use_neural and mode == "extend" and results:
            # LTX extension: seed from the previous clip's last frame
            import tempfile
            from ...media_edit.videos import extract_frames
            from ..motion_studio._core import probe_duration as _pd
            prev_dur = _pd(results[-1].path)
            tmpd = tempfile.mkdtemp(prefix="chain-extend-")
            fr = extract_frames(results[-1].path, out_dir=tmpd,
                                timestamps=[max(0.0, prev_dur - 0.1)])
            frames = fr.get("frames") or []
            if not frames:
                raise VideogenError(
                    f"scene {i}: could not grab the last frame of scene {i - 1}")
            path = gen.generate(full_prompt, mode="extend", image=frames[-1],
                                duration_s=dur, seed=sseed)
            results.append(SceneResult(i, path, gen.name, prompt))
            neural_n += 1
        else:
            raise VideogenError(
                f"scene {i}: mode {mode!r} needs a neural backend "
                f"({cap.reason}) or switch mode to image/text")
        clips.append(results[-1].path)

    timeline = [Segment(kind="video", src=c, duration=5.0,
                        transition=transition, transition_duration=0.7)
                for c in clips]
    # let each clip play its real length instead of re-trimming
    for seg, res in zip(timeline, results):
        from ..motion_studio._core import probe_duration
        seg.duration = max(1.0, probe_duration(res.path))

    # temporal-consistency pass: color + structure match on boundary
    # frames so chained cuts don't pop. Never fatal.
    consistency_report: dict = {}
    if consistency and len(clips) > 1:
        try:
            from .consistency import consistency_pass
            new_clips, consistency_report = consistency_pass(clips)
            for seg, new_path in zip(timeline, new_clips):
                seg.src = new_path
            clips = new_clips
        except Exception as exc:  # noqa: BLE001 - the chain goes on
            note = (note + "; " if note else "") + \
                f"consistency pass skipped ({exc})"
            _log.info("chain: consistency pass skipped: %s", exc)

    final = assemble(timeline, size=size, fps=fps)
    if grade_preset:
        final = grade(final, preset=grade_preset)
    if format:
        final = export(final, format=format)
    if out:
        import shutil
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(final, dest)
        final = str(dest)
    if gen is not None:
        gen.unload()

    report = ChainReport(final_path=final, scenes=results,
                         neural_scenes=neural_n, motion_scenes=motion_n,
                         grade_preset=grade_preset, format=format, note=note,
                         consistency=consistency_report)
    record_ledger({"kind": "videogen.chain", "path": final,
                   "scenes": len(results), "neural": neural_n,
                   "motion": motion_n, "backend": engine_name})
    return report
