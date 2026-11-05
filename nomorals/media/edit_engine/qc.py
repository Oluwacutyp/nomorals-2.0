"""QC measurers + the five-pass contact sheet (plan §1.5c tail).

The :class:`~.quality_gate.QualityGate` has always known how to
*judge* — what it never had was anything that *measures*.  Every gate
test hand-fed it dicts; no code computed a gate input from a finished
render.  This module is the missing measurer set:

* **frame sampling** — frames are grabbed across the render's whole
  timeline in five passes (the contact-sheet structure) and measured
  with :mod:`nomorals.media.studio_color` scopes data
  (:func:`measure_frame`): luma range, clipping, saturation;
* **exposure** ← the sampled frames' clipped-black + clipped-white
  fractions;
* **black frames** ← a denser sampling pass: a frame whose luma max
  sits at the black floor counts black; samples that land past the
  end of a too-short render count black too (the picture is absent
  there) and are named as missing in the evidence;
* **loudness** ← :func:`.audio.loudness_measure` (ffmpeg loudnorm
  first pass) mapped onto the gate's ``integrated_lufs`` /
  ``true_peak_dbtp`` keys; digital silence reads at the BS.1770
  absolute-gate floor (−70 LUFS) and says so — never a fabricated
  mid-scale number;
* **caption drift** ← the document's caption word timings vs word
  timings *measured from the render's own audio* through the §2.4
  :class:`~.text.Transcriber` protocol (only a ``measured`` timing
  source counts — an estimate of the render is not a measurement of
  it);
* **identity drift** ← :func:`videogen.consistency.check_identity`
  similarities when per-shot face embeddings are supplied; with locks
  but no embeddings the axis stays *unmeasured with the reason named*
  (there is no face-embedding rung over rendered frames — the
  insightface pack lives in the faceswap pipeline), and per-shot
  palette distances are recorded as evidence instead of a score;
* **aesthetic** ← measured image features (exposure / saturation
  from the same sampled frames) for the gate's critique path.

Every measurer returns measured data with its rung named in
``methods``; every failure is a named entry in ``unmeasured`` — the
gate reads a missing key as ``unknown`` and refuses to ship, which
is the standing discipline.  Nothing here invents a number.

The **contact sheet** is the render artifact the plan names: the
five sampling passes as five rows of frames, each frame judged
(black / clipped flags, red border on defect evidence), the gate's
per-axis verdicts in the header, attached to
:class:`~.render.RenderReport` as ``contact_sheet``.
"""

from __future__ import annotations

import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

from ...media_edit.videos import MediaEditError, run_ffmpeg, video_probe
from ..studio_color import measure_frame
from .audio import loudness_measure
from .failures import record_failure
from .quality_gate import QualityGate

__all__ = ["CONTACT_SHEET_PASSES", "MEASURER_RUNGS", "probe_render",
           "measure_render", "build_contact_sheet", "qc_render"]

#: the frame-QC sheet's structure: five passes over the timeline.
CONTACT_SHEET_PASSES = 5

_FRAMES_PER_PASS = 3
_SAMPLE_WIDTH = 320          # QC / sheet frame grabs
_BLACK_WIDTH = 160           # dense black-scan grabs (luma only)
_BLACK_LUMA_MAX = 3.0 / 255.0   # luma max at/below this = black frame
_CLIP_FAIL_PCT = 20.0        # a frame ≥20% clipped is defect evidence
_SILENCE_FLOOR_LUFS = -70.0  # BS.1770 absolute gate — digital silence
_SILENCE_PEAK_DBTP = -99.0
_DURATION_TOLERANCE_S = 0.5

#: axis → the rung that measures it (comparable data, §1.1e shape).
MEASURER_RUNGS = {
    "exposure_clipping":
        "studio_color.measure_frame (luma clipping) over the "
        "five-pass frame samples",
    "loudness":
        "edit_engine.audio.loudness_measure (ffmpeg loudnorm "
        "first pass, print_format=json)",
    "black_frames":
        "dense frame sampling + studio_color.measure_frame luma "
        "floor (missing tail frames count black, named)",
    "caption_sync_drift":
        "document caption words vs §2.4 Transcriber word timings "
        "measured from the render's audio (median offset)",
    "identity_drift":
        "videogen.consistency.check_identity over supplied per-shot "
        "face embeddings (palette evidence when embeddings absent)",
    "aesthetic":
        "studio_color.measure_frame → critique image features "
        "(exposure, saturation) over the five-pass samples",
}


# ── probe + frame grabs ───────────────────────────────────────────────────────

def probe_render(path: str | Path) -> dict[str, Any]:
    """ffprobe facts for a finished render. Raises MediaEditError."""
    p = Path(path)
    if not p.is_file():
        raise MediaEditError(f"no such render: {path}")
    probe = video_probe(str(p))
    streams = probe.get("streams") or []
    vids = [s for s in streams if s.get("type") == "video"]
    auds = [s for s in streams if s.get("type") == "audio"]
    return {"duration_s": float(probe.get("duration") or 0.0),
            "has_video": bool(vids), "has_audio": bool(auds),
            "width": int(probe.get("width") or 0),
            "height": int(probe.get("height") or 0),
            "fps": float(probe.get("fps") or 0.0),
            "bytes": int(probe.get("bytes") or 0)}


def _grab(path: Path, t: float, width: int, workdir: Path,
          tag: str) -> tuple[Any, str | None]:
    """One frame at ``t`` as a PIL image, or (None, named reason)."""
    out = workdir / f"{tag}.png"
    try:
        run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(path),
                    "-frames:v", "1", "-vf", f"scale={width}:-2",
                    str(out)], timeout=120)
    except MediaEditError as exc:
        return None, f"frame grab at {t:.2f}s failed: {exc}"
    if not out.is_file():
        return None, f"no frame at {t:.2f}s — past the end of the render"
    try:
        from PIL import Image
        with Image.open(out) as im:
            return im.convert("RGB").copy(), None
    except Exception as exc:  # undecodable frame file
        record_failure("qc._grab:decode", exc)
        return None, f"frame at {t:.2f}s undecodable: {exc}"


def _pass_spans(duration: float) -> list[tuple[int, float, float]]:
    """The five passes: (index, start, end) bands over the timeline."""
    return [(i, duration * i / CONTACT_SHEET_PASSES,
             duration * (i + 1) / CONTACT_SHEET_PASSES)
            for i in range(CONTACT_SHEET_PASSES)]


def _sample_times(duration: float,
                  frames_per_pass: int) -> list[tuple[int, float]]:
    """(pass_index, t) for the QC/sheet samples — band midpoints'
    neighbours, always strictly inside the render."""
    out: list[tuple[int, float]] = []
    for idx, a, b in _pass_spans(duration):
        for k in range(frames_per_pass):
            frac = (k + 1) / (frames_per_pass + 1)
            t = a + frac * (b - a)
            out.append((idx, min(t, max(0.0, duration - 0.05))))
    return out


def _frame_stats(image: Any) -> dict[str, Any]:
    """studio_color measurements for one sampled frame, QC-shaped."""
    m = measure_frame(image)
    clipped = float(m["clipped_black_pct"]) + float(m["clipped_white_pct"])
    return {"mean_luma": float(m["luma"]["mean"]),
            "max_luma": float(m["luma"]["max"]),
            "clipped_pct": clipped,
            "clipped_black_pct": float(m["clipped_black_pct"]),
            "clipped_white_pct": float(m["clipped_white_pct"]),
            "mean_saturation": float(m["mean_saturation"]),
            "black": float(m["luma"]["max"]) <= _BLACK_LUMA_MAX}


def _sample_frames(path: Path, duration: float, workdir: Path, *,
                   frames_per_pass: int = _FRAMES_PER_PASS
                   ) -> list[dict[str, Any]]:
    """The five-pass QC samples: [{pass, t, image, stats, reason}]."""
    samples: list[dict[str, Any]] = []
    for i, (pidx, t) in enumerate(_sample_times(duration, frames_per_pass)):
        image, reason = _grab(path, t, _SAMPLE_WIDTH, workdir, f"qc-{i:02d}")
        stats = None
        if image is not None:
            try:
                stats = _frame_stats(image)
            except Exception as exc:  # scopes failure = unmeasured frame
                record_failure("qc._sample_frames:measure_frame", exc)
                reason = f"frame at {t:.2f}s unmeasurable: {exc}"
        samples.append({"pass": pidx, "t": round(t, 3), "image": image,
                        "stats": stats, "reason": reason})
    return samples


def _black_scan(path: Path, span: float, probed: float,
                workdir: Path) -> dict[str, Any]:
    """Dense black-frame scan over ``span`` seconds.

    Samples every ~0.5s (capped at 240).  A sample landing past the
    probed duration is a *missing* frame — counted black, named in
    the result, never silently dropped from the fraction.
    """
    n = int(min(240, max(4, round(span / 0.5)))) if span > 0 else 0
    black = missing = usable = 0
    frozen_spans: list[dict[str, float]] = []
    prev_key: tuple[float, float] | None = None
    run_start = 0.0
    run_len = 0
    step = span / n if n else 0.0
    for i in range(n):
        t = (i + 0.5) * step
        if t > probed + 1e-6:
            missing += 1
            black += 1
            prev_key = None
            run_len = 0
            continue
        image, _reason = _grab(path, t, _BLACK_WIDTH, workdir, f"blk-{i:03d}")
        if image is None:
            missing += 1
            black += 1
            prev_key = None
            run_len = 0
            continue
        usable += 1
        try:
            st = _frame_stats(image)
        except Exception as exc:
            record_failure("qc._black_scan:measure_frame", exc)
            prev_key = None
            run_len = 0
            continue
        if st["black"]:
            black += 1
        # frozen = consecutive near-identical non-black frames
        key = (round(st["mean_luma"], 4), round(st["max_luma"], 4))
        if prev_key is not None and key == prev_key and not st["black"]:
            run_len += 1
        else:
            if run_len >= 2:
                frozen_spans.append(
                    {"start_s": round(run_start, 2),
                     "end_s": round(run_start + run_len * step, 2)})
            prev_key = key
            run_start = t
            run_len = 1
    if run_len >= 2:
        frozen_spans.append({"start_s": round(run_start, 2),
                             "end_s": round(run_start + run_len * step, 2)})
    return {"samples": n, "black": black, "missing": missing,
            "usable": usable,
            "fraction": (black / n) if n else None,
            "frozen_spans": frozen_spans,
            "step_s": round(step, 3)}


# ── per-axis measurers ────────────────────────────────────────────────────────
# Each returns (gate_value | None, evidence).  None = unmeasured; the
# caller records the named reason from the evidence's "reason" key.

def _measure_exposure(samples: list[dict[str, Any]]):
    usable = [s for s in samples if s["stats"] is not None]
    if not usable:
        return None, {"reason": "no sampled frame could be measured"}
    clips = [s["stats"]["clipped_pct"] for s in usable]
    score = 1.0 - (sum(clips) / len(clips)) / 100.0
    worst = max(usable, key=lambda s: s["stats"]["clipped_pct"])
    return round(max(0.0, min(1.0, score)), 4), {
        "frames": len(usable),
        "mean_clipped_pct": round(sum(clips) / len(clips), 2),
        "worst_frame": {"t": worst["t"],
                        "clipped_pct": round(worst["stats"]["clipped_pct"], 2)}}


def _measure_aesthetic(samples: list[dict[str, Any]]):
    usable = [s for s in samples if s["stats"] is not None]
    if not usable:
        return None, {"reason": "no sampled frame could be measured"}
    clip = sum(s["stats"]["clipped_pct"] for s in usable) / len(usable)
    sat = sum(s["stats"]["mean_saturation"] for s in usable) / len(usable)
    feats = {"exposure": round(max(0.0, min(1.0, 1.0 - clip / 100.0)), 4),
             "saturation": round(max(0.0, min(1.0, sat)), 4)}
    return feats, {"frames": len(usable), "image_features": feats}


def _measure_loudness(path: Path, probe: dict[str, Any]):
    if not probe["has_audio"]:
        return None, {"reason": "render has no audio stream — "
                                "loudness not measurable"}
    try:
        m = loudness_measure(path)
    except MediaEditError as exc:
        return None, {"reason": f"loudnorm measurement failed: {exc}"}
    notes: list[str] = []
    integrated = float(m["input_i"])
    if not math.isfinite(integrated):
        integrated = _SILENCE_FLOOR_LUFS
        notes.append("integrated loudness below the −70 LUFS "
                     "absolute-gate floor (digital silence) — "
                     "reported at the floor")
    peak = float(m["input_tp"])
    if not math.isfinite(peak):
        peak = _SILENCE_PEAK_DBTP
        notes.append("true peak below the measurement floor "
                     "(digital silence)")
    return ({"integrated_lufs": round(integrated, 2),
             "true_peak_dbtp": round(peak, 2)},
            {"loudnorm": {k: (round(v, 2) if isinstance(v, float)
                              and math.isfinite(v) else v)
                          for k, v in m.items()},
             "notes": notes})


def _doc_caption_words(document: Any) -> list[dict[str, Any]]:
    """The document's caption word timings: text layers first, then
    the transcript word list.  [] = the doc carries none."""
    words: list[dict[str, Any]] = []
    for layer in getattr(document, "text_layers", None) or []:
        for w in getattr(layer, "words", None) or []:
            if isinstance(w, dict):
                words.append(w)
            else:
                words.append({"start": getattr(w, "start", 0.0),
                              "end": getattr(w, "end", 0.0),
                              "text": getattr(w, "text", "")})
    if words:
        return sorted(words, key=lambda w: float(w.get("start", 0.0)))
    for w in getattr(document, "transcript", None) or []:
        if isinstance(w, dict):
            words.append(w)
    return sorted(words, key=lambda w: float(w.get("start", 0.0)))


def _norm_word(text: Any) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _measure_caption_drift(document: Any, path: Path, transcriber: Any):
    doc_words = [w for w in _doc_caption_words(document)
                 if _norm_word(w.get("text"))]
    if not doc_words:
        return None, {"reason": "document carries no caption word "
                                "timings — drift not measurable"}
    from .text import captions_from_audio
    try:
        got = captions_from_audio(path, transcriber=transcriber)
    except MediaEditError as exc:
        return None, {"reason": f"no word timings measurable from "
                                f"the render's audio: {exc}"}
    if got.get("timing_source") != "measured":
        return None, {"reason": "transcriber produced no measured "
                                "word timings for this render "
                                "(estimate is not a measurement)"}
    measured = [w for w in got.get("words") or [] if _norm_word(w.get("text"))]
    offsets: list[float] = []
    j = 0
    for dw in doc_words:
        want = _norm_word(dw.get("text"))
        k = j
        while k < len(measured) and _norm_word(measured[k].get("text")) != want:
            k += 1
        if k < len(measured):
            offsets.append(float(measured[k]["start"])
                           - float(dw.get("start", 0.0)))
            j = k + 1
    if len(offsets) < 3:
        return None, {"reason": f"only {len(offsets)} caption words "
                                f"matched the measured transcript "
                                f"(< 3) — drift not measurable",
                      "matched": len(offsets),
                      "doc_words": len(doc_words),
                      "measured_words": len(measured)}
    drift = statistics.median(offsets)
    abs_off = sorted(abs(o) for o in offsets)
    return round(drift, 3), {
        "matched": len(offsets), "doc_words": len(doc_words),
        "measured_words": len(measured),
        "median_offset_s": round(drift, 3),
        "p90_abs_offset_s": round(abs_off[int(0.9 * (len(abs_off) - 1))], 3),
        "transcriber": got.get("transcriber")}


def _shot_windows(document: Any, duration: float) -> list[tuple[float, float]]:
    """Base-track clip windows on the render timeline."""
    track = None
    try:
        track = document.track("video")
    except Exception as exc:
        record_failure("qc._shot_windows:track", exc)
    clips = list(getattr(track, "clips", None) or [])
    wins: list[tuple[float, float]] = []
    for c in clips:
        start = float(getattr(c, "start", 0.0) or 0.0)
        end = getattr(c, "end", None)
        end = float(end) if end else start + float(
            getattr(c, "duration", 0.0) or 0.0)
        if end > start:
            wins.append((start, end))
    if not wins and duration > 0:
        wins = [(0.0, duration)]
    return wins


def _palette_of(image: Any) -> list[list[float]]:
    """One-swatch palette (mean RGB, 0..255) of a frame — evidence
    for check_identity's palette leg, never a score by itself."""
    import numpy as np
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    mean = arr.mean(axis=(0, 1))
    return [[float(mean[0]), float(mean[1]), float(mean[2])]]


def _measure_identity(document: Any, samples: list[dict[str, Any]],
                      duration: float, shot_embeddings: Any):
    from ..videogen.consistency import IdentityLock, check_identity
    raw_locks = getattr(document, "identity_locks", None) or []
    if not raw_locks:
        return None, {"applicable": False,
                      "note": "document carries no identity locks — "
                              "the gate passes this axis by rule"}
    locks = [lk if isinstance(lk, IdentityLock)
             else IdentityLock.from_dict(dict(lk)) for lk in raw_locks]
    wins = _shot_windows(document, duration)
    if shot_embeddings is not None:
        scores: list[float] = []
        detail: list[dict[str, Any]] = []
        for i, (a, b) in enumerate(wins):
            emb = (shot_embeddings[i]
                   if i < len(shot_embeddings) else None)
            if emb is None:
                return None, {"reason": f"shot {i} has no supplied "
                                        f"face embedding"}
            sims = []
            for lk in locks:
                res = check_identity(lk, {"face_embedding": list(emb)})
                if res["similarity"] is not None:
                    sims.append(res["similarity"])
            if not sims:
                return None, {"reason": "identity locks carry no face "
                                        "embeddings to compare against"}
            scores.append(round(max(sims), 4))
            detail.append({"shot": i, "window": [round(a, 2), round(b, 2)],
                           "similarity": scores[-1]})
        return scores, {"shots": detail, "locks": len(locks)}
    # No embeddings: the axis stays unmeasured — named, with palette
    # evidence measured from the frames instead of a score.
    evidence: dict[str, Any] = {
        "reason": "no face-embedding rung over rendered frames "
                  "(insightface lives in the faceswap pipeline, not "
                  "QC) and no per-shot embeddings supplied — identity "
                  "similarity unmeasured; palette evidence recorded",
        "palette": []}
    usable = [s for s in samples if s["image"] is not None]
    for i, (a, b) in enumerate(wins):
        mid = (a + b) / 2.0
        near = min(usable, key=lambda s: abs(s["t"] - mid), default=None)
        if near is None:
            continue
        pal = _palette_of(near["image"])
        row: dict[str, Any] = {"shot": i, "window": [round(a, 2), round(b, 2)],
                               "palette": [[round(v, 1) for v in px]
                                           for px in pal],
                               "palette_dist": {}}
        for lk in locks:
            if lk.palette:
                res = check_identity(lk, {"palette": pal})
                if res["palette_dist"] is not None:
                    row["palette_dist"][lk.character_id] = round(
                        float(res["palette_dist"]), 2)
        evidence["palette"].append(row)
    return None, evidence


# ── the measurement bundle ────────────────────────────────────────────────────

def _measure(document: Any, path: str | Path, *,
             expected_duration: float | None = None,
             transcriber: Any = None,
             shot_embeddings: Any = None,
             workdir: str | Path | None = None,
             frames_per_pass: int = _FRAMES_PER_PASS
             ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """(bundle, samples, probe) — the bundle is JSON-shaped; the
    samples carry PIL images for the contact sheet."""
    p = Path(path)
    wd = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="qc-"))
    wd.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {"path": str(p), "ok": False,
                              "measurements": {}, "methods": {},
                              "unmeasured": {}, "evidence": {},
                              "probe": {}, "duration": {}}
    try:
        probe = probe_render(p)
    except MediaEditError as exc:
        for axis in MEASURER_RUNGS:
            bundle["unmeasured"][axis] = f"render unprobable: {exc}"
        return bundle, [], {}
    bundle["probe"] = probe
    measured_dur = probe["duration_s"]
    span = float(expected_duration) if expected_duration else measured_dur
    delta = (round(measured_dur - float(expected_duration), 3)
             if expected_duration else None)
    bundle["duration"] = {
        "measured_s": round(measured_dur, 3),
        "expected_s": (round(float(expected_duration), 3)
                       if expected_duration else None),
        "delta_s": delta,
        "within_tolerance": (abs(delta) <= _DURATION_TOLERANCE_S
                             if delta is not None else None)}
    bundle["ok"] = True

    samples: list[dict[str, Any]] = []
    if probe["has_video"] and span > 0:
        samples = _sample_frames(p, span, wd,
                                 frames_per_pass=frames_per_pass)
    meas: dict[str, Any] = bundle["measurements"]
    meth: dict[str, str] = bundle["methods"]
    unmeas: dict[str, str] = bundle["unmeasured"]
    evid: dict[str, Any] = bundle["evidence"]

    def _take(axis: str, key: str, value: Any, evidence: dict[str, Any]):
        evid[axis] = evidence
        if value is None:
            if evidence.get("applicable") is False:
                meth[axis] = ("not applicable — "
                              + str(evidence.get("note", "")))
            else:
                unmeas[axis] = str(evidence.get(
                    "reason", "measurement failed without a reason"))
        else:
            meas[key] = value
            meth[axis] = MEASURER_RUNGS[axis]

    if not probe["has_video"]:
        for axis in ("exposure_clipping", "black_frames", "aesthetic"):
            evid[axis] = {"reason": "render has no video stream"}
            unmeas[axis] = "render has no video stream"
    else:
        value, ev = _measure_exposure(samples)
        _take("exposure_clipping", "exposure", value, ev)
        scan = _black_scan(p, span, measured_dur, wd)
        evid["black_frames"] = scan
        if scan["fraction"] is None:
            unmeas["black_frames"] = "black scan sampled no frames"
        else:
            meas["black_frame_fraction"] = round(scan["fraction"], 4)
            meth["black_frames"] = MEASURER_RUNGS["black_frames"]
        value, ev = _measure_aesthetic(samples)
        _take("aesthetic", "image_features", value, ev)

    value, ev = _measure_loudness(p, probe)
    _take("loudness", "loudness", value, ev)
    value, ev = _measure_caption_drift(document, p, transcriber)
    _take("caption_sync_drift", "caption_drift_s", value, ev)
    value, ev = _measure_identity(document, samples, span, shot_embeddings)
    _take("identity_drift", "identity_scores", value, ev)
    return bundle, samples, probe


def measure_render(document: Any, path: str | Path, *,
                   expected_duration: float | None = None,
                   transcriber: Any = None,
                   shot_embeddings: Any = None,
                   workdir: str | Path | None = None) -> dict[str, Any]:
    """Measure a finished render into QualityGate inputs.

    Returns the bundle ``{path, ok, probe, duration, measurements,
    methods, unmeasured, evidence}`` — ``measurements`` is exactly
    the dict :meth:`QualityGate.check` consumes; axes that could not
    be measured are absent from it and named in ``unmeasured``.
    Never raises for a missing/broken render: the probe failure is
    recorded and every axis reads unmeasured.
    """
    bundle, _samples, _probe = _measure(
        document, path, expected_duration=expected_duration,
        transcriber=transcriber, shot_embeddings=shot_embeddings,
        workdir=workdir)
    return bundle


# ── the five-pass contact sheet ───────────────────────────────────────────────

_THUMB_W = 320
_STRIP_H = 18        # per-cell timestamp/flag strip
_LABEL_H = 20        # per-pass label strip
_HEADER_LINE_H = 16


def _sheet_failing_axes(sample: dict[str, Any]) -> list[str]:
    stats = sample["stats"]
    if stats is None:
        return []
    axes: list[str] = []
    if stats["black"]:
        axes.append("black_frames")
    if stats["clipped_pct"] >= _CLIP_FAIL_PCT:
        axes.append("exposure_clipping")
    return axes


def _build_sheet_image(samples: list[dict[str, Any]], gate: Any,
                       title: str, out: Path) -> list[dict[str, Any]]:
    """Draw the sheet PNG; returns the per-frame cell rects."""
    from PIL import Image, ImageDraw
    usable = [s for s in samples if s["image"] is not None]
    thumb_h = 180
    if usable:
        w0, h0 = usable[0]["image"].size
        thumb_h = max(40, round(_THUMB_W * h0 / max(1, w0)))
    axis_lines: list[str] = []
    if gate is not None:
        for axis, res in (gate.get("axes") or {}).items():
            score = res.get("score")
            axis_lines.append(
                f"{axis}: {res.get('status', '?')}"
                + (f" ({score})" if score is not None else ""))
    header_h = 28 + _HEADER_LINE_H * max(1, len(axis_lines))
    cols = max(1, max((sum(1 for s in samples if s["pass"] == i)
                       for i in range(CONTACT_SHEET_PASSES)), default=1))
    width = cols * _THUMB_W + (cols + 1) * 8
    row_h = thumb_h + _STRIP_H
    height = (header_h + CONTACT_SHEET_PASSES * (row_h + _LABEL_H)
              + 8)
    img = Image.new("RGB", (width, height), (24, 24, 24))
    dr = ImageDraw.Draw(img)
    dr.text((8, 6), title, fill=(240, 240, 240))
    for i, line in enumerate(axis_lines or ["gate: not run"]):
        dr.text((8, 26 + i * _HEADER_LINE_H), line, fill=(180, 200, 180))
    cells: list[dict[str, Any]] = []
    spans = {}
    for s in samples:
        spans.setdefault(s["pass"], []).append(s["t"])
    y = header_h
    for pidx in range(CONTACT_SHEET_PASSES):
        row = [s for s in samples if s["pass"] == pidx]
        ts = spans.get(pidx) or [0.0]
        dr.text((8, y + 3),
                f"PASS {pidx + 1} / {CONTACT_SHEET_PASSES} — "
                f"{min(ts):.2f}s .. {max(ts):.2f}s",
                fill=(200, 200, 220))
        y += _LABEL_H
        for ci, s in enumerate(row):
            x = 8 + ci * (_THUMB_W + 8)
            failing = _sheet_failing_axes(s)
            if s["image"] is not None:
                thumb = s["image"].resize((_THUMB_W, thumb_h))
                img.paste(thumb, (x, y))
            else:
                dr.rectangle([x, y, x + _THUMB_W, y + thumb_h],
                             fill=(8, 8, 8))
                dr.text((x + 8, y + thumb_h // 2), "no frame",
                        fill=(160, 160, 160))
            if failing:
                for off in range(3):
                    dr.rectangle([x - 1 + off, y - 1 + off,
                                  x + _THUMB_W - off, y + thumb_h - off],
                                 outline=(230, 40, 40))
            stats = s["stats"] or {}
            label = f"t={s['t']:.2f}s"
            if stats:
                label += (f"  luma {stats['mean_luma']:.2f}"
                          f"  clip {stats['clipped_pct']:.0f}%")
            if failing:
                label += "  FAIL " + ",".join(failing)
            dr.text((x + 2, y + thumb_h + 3), label,
                    fill=(255, 120, 120) if failing else (170, 170, 170))
            cells.append({"pass": pidx, "t": s["t"],
                          "cell": [x, y, _THUMB_W, thumb_h],
                          "failing_axes": failing})
        y += row_h
    img.save(str(out))
    return cells


def _sheet_record(out: Path, samples: list[dict[str, Any]], gate: Any,
                  cells: list[dict[str, Any]],
                  duration: float) -> dict[str, Any]:
    by_cell = {(c["pass"], c["t"]): c for c in cells}
    passes: list[dict[str, Any]] = []
    failing_frames: list[dict[str, Any]] = []
    for pidx, a, b in _pass_spans(duration):
        frames = []
        for s in [s for s in samples if s["pass"] == pidx]:
            axes = _sheet_failing_axes(s)
            stats = s["stats"] or {}
            frames.append({
                "t": s["t"], "grabbed": s["stats"] is not None,
                "black": bool(stats.get("black")),
                "clipped_pct": round(float(stats.get("clipped_pct", 0.0)), 2),
                "mean_luma": (round(float(stats["mean_luma"]), 4)
                              if stats else None),
                "failing_axes": axes,
                "cell": by_cell.get((pidx, s["t"]), {}).get("cell")})
            if axes:
                failing_frames.append({"t": s["t"], "axes": axes})
        passes.append({"index": pidx, "span_s": [round(a, 3), round(b, 3)],
                       "frames": frames})
    axes_verdicts = {}
    if gate is not None:
        for axis, res in (gate.get("axes") or {}).items():
            axes_verdicts[axis] = {"status": res.get("status"),
                                   "score": res.get("score"),
                                   "detail": res.get("detail")}
    return {"kind": "five_pass_frame_qc", "ok": True, "path": str(out),
            "passes": passes, "axes": axes_verdicts,
            "failing_frames": failing_frames}


def build_contact_sheet(path: str | Path, out: str | Path, *,
                        gate: dict[str, Any] | None = None,
                        duration: float | None = None,
                        samples: list[dict[str, Any]] | None = None,
                        workdir: str | Path | None = None
                        ) -> dict[str, Any]:
    """Build the five-pass frame-QC contact sheet for a render.

    ``samples`` may carry pre-grabbed QC samples (qc_render passes
    its own so frames are grabbed once); otherwise frames are
    grabbed here.  Returns the sheet record (path, five passes with
    per-frame verdict data + cell rects, per-axis verdicts, failing
    frames).  Raises MediaEditError when the render can't be probed.
    """
    p = Path(path)
    out_p = Path(out)
    if samples is None:
        probe = probe_render(p)
        dur = float(duration) if duration else probe["duration_s"]
        wd = Path(workdir) if workdir else Path(
            tempfile.mkdtemp(prefix="qc-sheet-"))
        wd.mkdir(parents=True, exist_ok=True)
        samples = _sample_frames(p, dur, wd)
    else:
        dur = float(duration) if duration else max(
            (s["t"] for s in samples), default=0.0)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cells = _build_sheet_image(samples, gate, f"Frame QC — {p.name}",
                               out_p)
    return _sheet_record(out_p, samples, gate, cells, dur)


# ── the front door ────────────────────────────────────────────────────────────

def qc_render(document: Any, path: str | Path, *,
              report: Any = None,
              sheet_out: str | Path | None = None,
              expected_duration: float | None = None,
              transcriber: Any = None,
              shot_embeddings: Any = None,
              workdir: str | Path | None = None) -> dict[str, Any]:
    """Measure a finished render, run the gate, build the sheet.

    The §1.5c pipeline end to end: measurers read the actual file,
    :class:`QualityGate` judges only measured inputs, and the
    five-pass contact sheet is attached to ``report`` (a
    :class:`~.render.RenderReport`) when one is supplied.  Returns
    ``{"measure", "gate", "contact_sheet", "ship"}``.  A broken or
    missing render yields a bundle of named unmeasured axes, a
    no-ship gate and a sheet record with the failure named — never
    an exception, never a fabricated pass.
    """
    bundle, samples, probe = _measure(
        document, path, expected_duration=expected_duration,
        transcriber=transcriber, shot_embeddings=shot_embeddings,
        workdir=workdir)
    gate = QualityGate().check(document, bundle["measurements"])
    p = Path(path)
    out_p = (Path(sheet_out) if sheet_out
             else p.with_name(f"{p.stem}.qc-sheet.png"))
    span = (float(expected_duration) if expected_duration
            else float(probe.get("duration_s") or 0.0))
    if bundle["ok"] and probe.get("has_video"):
        sheet = build_contact_sheet(p, out_p, gate=gate, duration=span,
                                    samples=samples)
    else:
        reason = ("render could not be probed — no frames to sheet"
                  if not bundle["ok"] else
                  "render has no video stream — no frames to sheet")
        sheet = {"kind": "five_pass_frame_qc", "ok": False,
                 "path": None, "reason": reason, "passes": [],
                 "axes": {axis: {"status": res.get("status"),
                                 "score": res.get("score"),
                                 "detail": res.get("detail")}
                          for axis, res in gate["axes"].items()},
                 "failing_frames": []}
    if report is not None:
        report.contact_sheet = sheet
    return {"measure": bundle, "gate": gate, "contact_sheet": sheet,
            "ship": gate["ship"]}
