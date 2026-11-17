"""Variants-matrix QC for the motion studio — see before you ship.

The code-video rung already writes a 5-frame contact sheet beside
every MP4 (:mod:`.codegen`) and :mod:`nomorals.media.edit_engine.qc`
already measures a finished render (five-pass frame sampling,
exposure / black-frame / loudness / caption / identity axes, the
:class:`~nomorals.media.edit_engine.quality_gate.QualityGate`).
What neither does is compare: render the SAME job across rungs and
parameters, sheet every variant, and name a winner. That is this
module — the Eigen motion-studio QC discipline as a first-class
capability Devon can call on any render job:

* :func:`plan_variants` — the matrix plan for a job, pure (no render);
* :func:`render_matrix` — render every variant, QC each one through
  the real edit-engine measurer, compose one matrix sheet, write the
  QC report naming the winning variant and why the rest lost;
* :func:`qc_existing` — the same QC (measure + gate + contact sheet)
  for a render that already exists on disk.

The winner is picked on fidelity first, visual axes second. A scene
job's code variants render the scene itself; the motion variant only
renders the job's title as kinetic typography — a stand-in, labelled
as such. Picking the prettier stand-in over the actual diagram would
be the wrong artifact, so scene-faithful variants rank ahead of
stand-ins and the visual axes (black fraction, exposure, aesthetic)
decide within each fidelity class. For a prompt-only job every
variant renders the prompt, fidelity ties, and the visual axes
decide outright.

One honest seam, stated here, not hidden: the edit-engine gate's
``ship`` verdict needs a document — a loudness spec, caption word
timings, identity locks. A silent videogen clip carries none of
those, so its loudness/caption axes read ``unknown`` and the gate
never ships it. The matrix therefore carries its own verdict over
the axes a videogen render CAN be judged on — exposure, black
frames, aesthetic, duration — and keeps the full gate dict beside
it in every report, verbatim, so nothing is papered over.
"""

from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path
from typing import Any

from .capabilities import VideogenError, neural_capability

__all__ = [
    "plan_variants",
    "render_matrix",
    "qc_existing",
    "matrix_message",
    "QC_DURATION_TOLERANCE_S",
    "QC_VISUAL_AXES",
]

#: a render whose measured duration strays this far from the requested
#: one fails the matrix on duration (same tolerance as edit_engine.qc).
QC_DURATION_TOLERANCE_S = 0.5

#: the axes a silent, caption-free videogen render can be judged on.
QC_VISUAL_AXES = ("exposure_clipping", "black_frames", "aesthetic")

#: ranking noise floor: QC samples 15 frames per variant through
#: ffmpeg grabs, so score differences below these steps are sampling
#: noise, not a quality signal — variants within one step tie and
#: plan order (canonical variant first) decides.
QC_RANK_BLACK_STEP = 0.005
QC_RANK_SCORE_STEP = 0.01

#: the matrix plan: (name, backend, size, note).
_MATRIX_PLAN = (
    ("code", "code", None, "full-size deterministic scene render"),
    ("code-480p", "code", (854, 480), "same scene, 480p"),
    ("motion", "motion", None, "motion-studio stand-in for the job"),
    ("diffusion", "diffusion", None, "neural diffusion when the "
     "hardware allows"),
)


def _default_out_dir() -> Path:
    base = Path.home() / ".nomorals" / "videos"
    return base / f"qc-{int(time.time() * 1000)}"


def plan_variants(scene: dict[str, Any] | None = None,
                  prompt: str = "",
                  *, include_diffusion: bool = True
                  ) -> list[dict[str, Any]]:
    """The matrix plan for a job — pure, renders nothing.

    A scene job gets the code rungs (full + 480p), the motion stand-in
    and diffusion; a prompt-only job gets motion + diffusion (there is
    no scene for the code backend to render). Raises
    :class:`VideogenError` for an empty job.
    """
    if scene is not None and not isinstance(scene, dict):
        raise VideogenError(
            f"a scene spec must be a dict, got {type(scene).__name__}")
    if scene is None and not (prompt or "").strip():
        raise VideogenError(
            "QC matrix needs a scene spec or a prompt — nothing to render")
    variants: list[dict[str, Any]] = []
    for name, backend, size, note in _MATRIX_PLAN:
        if backend == "code" and scene is None:
            continue
        if backend == "diffusion" and not include_diffusion:
            continue
        # fidelity: a code variant renders the scene itself; motion and
        # diffusion render the prompt (the motion rung is a stand-in
        # for the job's title when a scene is present).
        fidelity = "scene" if backend == "code" else "prompt"
        variants.append({"name": name, "backend": backend,
                         "size": size, "note": note,
                         "fidelity": fidelity})
    return variants


def _visual_verdict(measure: dict[str, Any], gate: dict[str, Any],
                    expected_duration: float | None
                    ) -> tuple[str, list[str]]:
    """(verdict, reasons) over the visual axes + duration.

    ``pass`` — every applicable axis passed and the duration is within
    tolerance. ``fail`` — an applicable axis failed. ``unmeasured`` —
    an applicable axis could not be measured (named, never guessed).
    The gate's loudness / caption / identity axes are NOT applicable
    here (see the module docstring) and never block the verdict.
    """
    axes = (gate or {}).get("axes", {}) or {}
    failed: list[str] = []
    unmeasured: list[str] = []
    for axis in QC_VISUAL_AXES:
        status = axes.get(axis, {}).get("status")
        if status == "fail":
            failed.append(axis)
        elif status != "pass":
            unmeasured.append(axis)
    if expected_duration is not None:
        dur = (measure or {}).get("duration", {}) or {}
        if dur.get("within_tolerance") is False:
            failed.append("duration")
    if failed:
        return "fail", [f"{a} failed" for a in failed]
    if unmeasured:
        return "unmeasured", [f"{a} unmeasured" for a in unmeasured]
    return "pass", []


def _rank_key(variant: dict[str, Any],
              has_scene: bool) -> tuple[int, float, float, float]:
    """Lower wins: fidelity class, then black fraction, then clipped
    exposure, then aesthetic shortfall. Unmeasured axes count as
    worst. For a scene job, scene-faithful variants (code) rank ahead
    of prompt variants (motion stand-in, diffusion); for a
    prompt-only job every variant renders the prompt, so fidelity
    ties and the visual axes decide."""
    fidelity = 0 if (has_scene and variant.get("fidelity") == "scene") else 1
    measure = variant.get("measure") or {}
    gate = variant.get("gate") or {}
    axes = gate.get("axes", {}) or {}
    meas = measure.get("measurements", {}) or {}
    black = meas.get("black_frame_fraction")
    black = float(black) if black is not None else 1.0
    exposure = axes.get("exposure_clipping", {}).get("score")
    exposure = float(exposure) if exposure is not None else 0.0
    aesthetic = axes.get("aesthetic", {}).get("score")
    aesthetic = float(aesthetic) if aesthetic is not None else 0.0
    return (fidelity,
            math.floor(black / QC_RANK_BLACK_STEP),
            math.floor((1.0 - exposure) / QC_RANK_SCORE_STEP),
            math.floor((1.0 - aesthetic) / QC_RANK_SCORE_STEP))


def _qc_one(path: Path, *, out_dir: Path, stem: str,
            expected_duration: float | None) -> dict[str, Any]:
    """Measure + gate + sheet one finished render (the real measurer)."""
    from ..edit_engine.qc import qc_render

    workdir = out_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    sheet_out = out_dir / f"{stem}.qc-sheet.png"
    return qc_render(None, path, sheet_out=sheet_out,
                     expected_duration=expected_duration,
                     workdir=workdir)


def qc_existing(path: str | Path, *,
                out_dir: str | Path | None = None,
                expected_duration: float | None = None
                ) -> dict[str, Any]:
    """QC a render that already exists: measure + gate + contact sheet.

    Returns ``{path, verdict, reasons, ship, gate, measure,
    contact_sheet, report_path}`` — ``ship`` is the edit-engine
    gate's own verdict (False for a document-less render, honestly),
    ``verdict`` is the visual-axes verdict (see the module
    docstring). A JSON report is written beside the sheet.
    """
    p = Path(path)
    if not p.is_file():
        raise VideogenError(f"no such render to QC: {path}")
    out = Path(out_dir) if out_dir else p.parent
    out.mkdir(parents=True, exist_ok=True)
    res = _qc_one(p, out_dir=out, stem=p.stem,
                  expected_duration=expected_duration)
    verdict, reasons = _visual_verdict(
        res["measure"], res["gate"], expected_duration)
    sheet = res["contact_sheet"]
    sheet_path = (sheet.get("path") if isinstance(sheet, dict) else None)
    report = {"kind": "render_qc", "path": str(p),
              "verdict": verdict, "reasons": reasons,
              "ship": bool(res["gate"].get("ship")),
              "gate": res["gate"], "measure": res["measure"],
              "contact_sheet": sheet_path}
    report_path = out / f"{p.stem}.qc.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _render_variant(variant: dict[str, Any], *,
                    scene: dict[str, Any] | None, prompt: str,
                    duration_s: float, fps: float, seed: int,
                    out_dir: Path) -> dict[str, Any]:
    """Render one variant through its real rung — fail-soft.

    Returns the variant record with ``status`` rendered / failed /
    unavailable and, on success, ``path`` + ``render_hash`` (code
    variants only). A failed variant never kills the matrix.
    """
    name = variant["name"]
    backend = variant["backend"]
    size = variant.get("size")
    out_path = out_dir / f"{name}.mp4"
    rec: dict[str, Any] = dict(variant)
    try:
        if backend == "code":
            from .codegen import render_code_scene

            rendered = render_code_scene(
                scene, out_path, duration=duration_s, size=size, fps=fps)
            rec["status"] = "rendered"
            rec["path"] = rendered["path"]
            rec["render_hash"] = rendered["render_hash"]
            rec["frames"] = rendered["frames"]
            return rec
        if backend == "motion":
            from .pipeline import _motion_fallback

            title = prompt or (scene or {}).get("title", "") or "untitled"
            path, note = _motion_fallback(
                title, duration_s=duration_s, size=size, fps=fps)
            rec["status"] = "rendered"
            rec["path"] = path
            rec["note"] = note
            return rec
        # diffusion — only when the hardware actually allows it.
        cap = neural_capability(prefer="auto")
        if not cap.available:
            rec["status"] = "unavailable"
            rec["reason"] = cap.reason
            return rec
        from .pipeline import _neural_generate

        path = _neural_generate(prompt, cap.backend, duration_s=duration_s,
                                seed=seed, out=out_path)
        rec["status"] = "rendered"
        rec["path"] = str(path)
        rec["model"] = cap.backend
        return rec
    except Exception as exc:  # noqa: BLE001 - variant failure is data
        rec["status"] = "failed"
        rec["reason"] = str(exc)
        return rec


def _compose_matrix_sheet(entries: list[dict[str, Any]], out_path: Path,
                          title: str) -> Path:
    """One PNG: every variant's contact sheet stacked with verdict bars.

    ``entries`` is [{label, verdict_line, sheet}]; a variant with no
    sheet gets a dark placeholder block naming the reason. Pure PIL —
    no render needed.
    """
    from PIL import Image, ImageDraw

    width = 1280
    bar_h = 40
    gap = 12
    blocks: list[tuple[str, str, Any]] = []
    total_h = 28 + gap
    for entry in entries:
        img = None
        sheet = entry.get("sheet")
        if sheet:
            try:
                with Image.open(sheet) as im:
                    img = im.convert("RGB").copy()
            except Exception:  # noqa: BLE001 - placeholder instead
                img = None
        if img is not None:
            scale = width / img.width
            img = img.resize((width, max(40, int(img.height * scale))))
        blocks.append((entry["label"], entry["verdict_line"], img))
        total_h += bar_h + (img.height if img is not None else 180) + gap
    canvas = Image.new("RGB", (width, total_h), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 6), title, fill=(240, 240, 240))
    y = 28 + gap
    for label, verdict_line, img in blocks:
        draw.rectangle([0, y, width, y + bar_h], fill=(38, 38, 48))
        draw.text((12, y + 12), label, fill=(240, 240, 240))
        # verdict on the same bar, right-aligned by rough width.
        vtext = verdict_line[:80]
        draw.text((width - 12 - len(vtext) * 7, y + 12), vtext,
                  fill=(170, 220, 170) if verdict_line.startswith("PASS")
                  else (255, 190, 120))
        y += bar_h
        if img is not None:
            canvas.paste(img, (0, y))
            y += img.height
        else:
            draw.rectangle([0, y, width, y + 180], fill=(8, 8, 10))
            draw.text((12, y + 80), "no contact sheet — variant did not "
                      "render", fill=(160, 160, 160))
            y += 180
        y += gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    return out_path


def render_matrix(scene: dict[str, Any] | None = None,
                  prompt: str = "",
                  *, duration_s: float = 5.0,
                  out_dir: str | Path | None = None,
                  seed: int = 0,
                  variants: list[dict[str, Any]] | None = None,
                  size: tuple[int, int] | None = None,
                  fps: float | None = None) -> dict[str, Any]:
    """Render the variants matrix for a job and QC every variant.

    ``scene`` is an explainer design dict (or HTML scene) — the code
    rungs render it; ``prompt`` is the free-text job for the motion
    and diffusion rungs. Each variant renders through its real rung,
    is measured + gated + sheeted by the real edit-engine QC, and
    gets a visual-axes verdict. Returns the report dict (also written
    as ``matrix.json`` in the out dir): ``{kind, job, variants,
    winner, winner_reason, matrix_sheet, report_path}``. The winner
    is the cleanest passing variant; when none passes, ``winner`` is
    None and the report names why each variant lost.
    """
    planned = list(variants) if variants is not None else plan_variants(
        scene=scene, prompt=prompt)
    if duration_s <= 0:
        raise VideogenError(f"QC matrix needs duration > 0, got {duration_s}")
    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    job_label = prompt or (scene or {}).get("title", "") or "untitled"
    fps_value = float(fps) if fps else None

    done: list[dict[str, Any]] = []
    for variant in planned:
        rec = _render_variant(variant, scene=scene, prompt=prompt,
                              duration_s=duration_s,
                              fps=fps_value or 24.0, seed=seed,
                              out_dir=out)
        if rec["status"] != "rendered":
            rec["verdict"] = rec["status"]  # failed | unavailable
            done.append(rec)
            continue
        # QC the finished file through the real measurer.
        try:
            res = _qc_one(Path(rec["path"]), out_dir=out,
                          stem=rec["name"],
                          expected_duration=duration_s)
        except Exception as exc:  # noqa: BLE001 - QC failure is data
            rec["verdict"] = "unmeasured"
            rec["reason"] = f"QC failed: {exc}"
            done.append(rec)
            continue
        rec["measure"] = res["measure"]
        rec["gate"] = res["gate"]
        rec["ship"] = bool(res["gate"].get("ship"))
        sheet = res["contact_sheet"]
        rec["contact_sheet"] = (sheet.get("path")
                                if isinstance(sheet, dict) else None)
        verdict, reasons = _visual_verdict(
            res["measure"], res["gate"], duration_s)
        rec["verdict"] = verdict
        rec["reasons"] = reasons
        done.append(rec)

    passing = [v for v in done if v.get("verdict") == "pass"]
    winner = None
    winner_reason = ""
    has_scene = scene is not None
    if passing:
        # cleanest visual axes win within each fidelity class; plan
        # order breaks exact ties, so the full-size code rung wins a
        # tie over its 480p sibling.
        ranked = sorted(enumerate(passing),
                        key=lambda pair: (_rank_key(pair[1], has_scene),
                                          pair[0]))
        winner = ranked[0][1]["name"]
        win = ranked[0][1]
        detail = []
        if has_scene and win.get("fidelity") == "scene":
            detail.append("renders the scene itself — the motion "
                          "variant is only a stand-in for the title")
        if win.get("render_hash"):
            detail.append("deterministic — the same scene re-renders "
                          "byte-identical (render hash "
                          f"{win['render_hash'][:12]}…)")
        meas = (win.get("measure") or {}).get("measurements", {}) or {}
        black = meas.get("black_frame_fraction")
        exposure = (win.get("gate") or {}).get("axes", {}).get(
            "exposure_clipping", {}).get("score")
        detail.append(
            "cleanest visual axes of the "
            f"{len(passing)} passing variant(s)"
            + (f" (black {float(black):.1%}" if black is not None else "")
            + (f", exposure {float(exposure):.2f})"
               if exposure is not None else ")"))
        winner_reason = "; ".join(detail)
    else:
        blocks = []
        for v in done:
            why = (v.get("reason") or "; ".join(v.get("reasons") or [])
                   or v.get("verdict", "?"))
            blocks.append(f"{v['name']}: {why}")
        winner_reason = "no variant passed — " + "; ".join(blocks)

    entries = []
    for v in done:
        label = f"{v['name']} — {v.get('backend', '?')}"
        if v.get("verdict") == "pass":
            verdict_line = "PASS"
        elif v.get("verdict") == "fail":
            verdict_line = "FAIL " + "; ".join(v.get("reasons") or [])
        elif v.get("verdict") == "unmeasured":
            verdict_line = "UNMEASURED " + (v.get("reason") or "")
        else:
            verdict_line = v.get("verdict", "?").upper() + " " + (
                v.get("reason") or "")
        entries.append({"label": label, "verdict_line": verdict_line,
                        "sheet": v.get("contact_sheet")})
    sheet_path = _compose_matrix_sheet(
        entries, out / "matrix-sheet.png",
        f"QC matrix — {job_label} ({len(done)} variants)")
    report: dict[str, Any] = {
        "kind": "variants_matrix",
        "job": {"prompt": prompt,
                "scene_kind": ((scene or {}).get("engine", "")
                               if isinstance(scene, dict) else ""),
                "duration_s": float(duration_s), "seed": seed},
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
        "out_dir": str(out),
        "variants": done,
        "winner": winner,
        "winner_reason": winner_reason,
        "matrix_sheet": str(sheet_path),
    }
    report_path = out / "matrix.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def matrix_message(report: dict[str, Any]) -> str:
    """The chat/CLI rendering of a matrix report — one block."""
    lines = [f"🎬 QC matrix — {report['job']['prompt'] or 'scene'} "
             f"({len(report['variants'])} variants)"]
    if report.get("winner"):
        lines.append(f"  winner: {report['winner']} — "
                     f"{report['winner_reason']}")
    else:
        lines.append(f"  winner: none — {report['winner_reason']}")
    for v in report["variants"]:
        verdict = v.get("verdict", "?").upper()
        extra = ""
        if v.get("status") == "rendered":
            meas = (v.get("measure") or {}).get("measurements", {}) or {}
            black = meas.get("black_frame_fraction")
            exposure = (v.get("gate") or {}).get("axes", {}).get(
                "exposure_clipping", {}).get("score")
            bits = []
            if black is not None:
                bits.append(f"black {float(black):.1%}")
            if exposure is not None:
                bits.append(f"exposure {float(exposure):.2f}")
            if bits:
                extra = " (" + ", ".join(bits) + ")"
            extra += f" → {v.get('path', '')}"
        elif v.get("reason"):
            extra = f" — {v['reason']}"
        lines.append(f"  {v['name']}: {verdict}{extra}")
    lines.append(f"  matrix sheet: {report['matrix_sheet']}")
    lines.append(f"  report: {report['report_path']}")
    return "\n".join(lines)
