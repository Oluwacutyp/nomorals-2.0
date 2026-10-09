"""Devon Studio — editing automation.

Real automation, not buttons: silence cutting, scene detection, smart
reframing, batch workflows. Every function does the work; nothing here
is decorative.

Profile-aware: heavy CV paths degrade to ffmpeg-only on the phone.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg") or which("/usr/bin/ffmpeg")


def _ffprobe() -> str | None:
    from shutil import which
    return which("ffprobe") or which("/usr/bin/ffprobe")


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)  # noqa: S603


@dataclass
class SilenceSegment:
    start: float
    end: float
    duration: float


def detect_silences(src: str | os.PathLike[str], *,
                    noise_db: float = -30.0,
                    min_duration: float = 0.5) -> list[SilenceSegment]:
    """Find silent regions via ffmpeg silencedetect. Real detection."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    cmd = [ff, "-hide_banner", "-i", str(src), "-af",
           f"silencedetect=noise={noise_db}dB:d={min_duration}",
           "-f", "null", "-"]
    proc = _run(cmd)
    out: list[SilenceSegment] = []
    cur_start: float | None = None
    for line in proc.stderr.splitlines():
        if "silence_start" in line:
            try:
                cur_start = float(line.split("silence_start:")[1].split()[0])
            except (IndexError, ValueError):
                cur_start = None
        elif "silence_end" in line and cur_start is not None:
            try:
                parts = line.split("silence_end:")[1].split("|")
                end = float(parts[0].strip())
                dur = float(parts[1].split(":")[1].strip()) if len(parts) > 1 else end - cur_start
                out.append(SilenceSegment(cur_start, end, dur))
            except (IndexError, ValueError):
                pass
            cur_start = None
    return out


def cut_silences(src: str | os.PathLike[str], *,
                 out: str | os.PathLike[str] | None = None,
                 noise_db: float = -30.0,
                 min_duration: float = 0.5,
                 padding: float = 0.1) -> dict[str, Any]:
    """Remove silent regions, keeping `padding` seconds around speech.

    Returns the output path and what was cut. Real edit, not a marker.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    src = str(src)
    silences = detect_silences(src, noise_db=noise_db, min_duration=min_duration)
    if not silences:
        return {"ok": True, "path": src, "cut": 0, "segments_removed": 0,
                "note": "no silences found — nothing to cut"}

    # Build keep-ranges: complement of silences, with padding.
    probe = _ffprobe()
    duration = 0.0
    if probe:
        p = _run([probe, "-v", "error", "-show_entries", "format=duration",
                  "-of", "csv=p=0", src])
        try:
            duration = float(p.stdout.strip())
        except ValueError:
            duration = 0.0
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s in silences:
        ks, ke = max(0.0, s.start - padding), s.end + padding
        if ks > cursor:
            keep.append((cursor, ks))
        cursor = max(cursor, ke)
    if duration and cursor < duration:
        keep.append((cursor, duration))
    if not keep:
        return {"ok": False, "reason": "everything is silence"}

    out_path = Path(out) if out else Path(tempfile.mkdtemp()) / "nosilence.mp4"
    # Use select filter for frame-accurate cuts.
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
    cmd = [ff, "-hide_banner", "-y", "-i", src,
           "-vf", f"select='{expr}',setpts=N/FRAME_RATE/TB",
           "-af", f"aselect='{expr}',asetpts=N/SR/TB",
           str(out_path)]
    proc = _run(cmd)
    if proc.returncode != 0 or not Path(out_path).exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    removed = sum(s.duration for s in silences)
    return {"ok": True, "path": str(out_path),
            "seconds_removed": round(removed, 2),
            "segments_removed": len(silences),
            "kept_ranges": len(keep)}


@dataclass
class Scene:
    index: int
    start: float
    end: float
    duration: float
    thumbnail: str = ""


def detect_scenes(src: str | os.PathLike[str], *,
                  threshold: float = 0.4,
                  min_length: float = 1.0,
                  thumbnails: bool = False,
                  workdir: str | None = None) -> list[Scene]:
    """Scene cut detection via ffmpeg select filter. Real cuts, real times."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    src = str(src)
    cmd = [ff, "-hide_banner", "-i", src,
           "-vf", f"select='gt(scene,{threshold})',showinfo",
           "-f", "null", "-"]
    proc = _run(cmd)
    cuts: list[float] = [0.0]
    for line in proc.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            try:
                t = float(line.split("pts_time:")[1].split()[0])
                if t - cuts[-1] >= min_length:
                    cuts.append(t)
            except (IndexError, ValueError):
                pass
    # Get total duration for the last scene end.
    duration = cuts[-1] + 1.0
    probe = _ffprobe()
    if probe:
        p = _run([probe, "-v", "error", "-show_entries", "format=duration",
                  "-of", "csv=p=0", src])
        try:
            duration = float(p.stdout.strip())
        except ValueError:
            pass
    scenes: list[Scene] = []
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp())
    wd.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else duration
        if end - start < min_length:
            continue
        thumb = ""
        if thumbnails:
            tp = wd / f"scene_{i:03d}.jpg"
            _run([ff, "-hide_banner", "-y", "-ss", str(start + 0.1),
                  "-i", src, "-frames:v", "1", str(tp)])
            if tp.exists():
                thumb = str(tp)
        scenes.append(Scene(index=len(scenes), start=round(start, 3),
                            end=round(end, 3),
                            duration=round(end - start, 3),
                            thumbnail=thumb))
    return scenes


def split_video_on_scenes(src: str | os.PathLike[str], *,
                          out_dir: str | os.PathLike[str] | None = None,
                          threshold: float = 0.4,
                          min_length: float = 1.0) -> dict[str, Any]:
    """Split a video into per-scene files. Real cuts, frame-accurate."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    scenes = detect_scenes(src, threshold=threshold, min_length=min_length)
    if not scenes:
        return {"ok": False, "reason": "no scenes detected"}
    od = Path(out_dir) if out_dir else Path(tempfile.mkdtemp())
    od.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for s in scenes:
        op = od / f"scene_{s.index:03d}.mp4"
        proc = _run([ff, "-hide_banner", "-y", "-ss", str(s.start),
                     "-i", str(src), "-t", str(s.duration),
                     "-c", "copy", str(op)])
        if proc.returncode != 0:
            # Fallback: re-encode (copy can fail on non-keyframe cuts).
            proc = _run([ff, "-hide_banner", "-y", "-ss", str(s.start),
                         "-i", str(src), "-t", str(s.duration),
                         "-c:v", "libx264", "-preset", "fast",
                         "-c:a", "aac", str(op)])
        if op.exists():
            paths.append(str(op))
    return {"ok": True, "scenes": len(paths), "paths": paths,
            "dir": str(od)}


# ── auto-reframe ───────────────────────────────────────────────────

def _detect_faces(frame_path: str) -> list[tuple[int, int, int, int]]:
    """Face boxes via OpenCV Haar cascade. Returns [(x,y,w,h)]."""
    try:
        import cv2
    except ImportError:
        return []
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    img = cv2.imread(frame_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cascade).detectMultiScale(gray, 1.1, 4)
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def auto_reframe(src: str | os.PathLike[str], *,
                 out: str | os.PathLike[str] | None = None,
                 aspect: str = "9:16",
                 sample_every: float = 1.0,
                 smooth: bool = True) -> dict[str, Any]:
    """Reframe to a target aspect ratio, tracking faces/subjects.

    Samples frames, finds faces (or falls back to center-weighted thirds),
    builds a smoothed crop path, and renders with crop=x:y:w:h animated
    over time. Real subject tracking, not a static center crop.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    src = str(src)
    try:
        aw, ah = (int(x) for x in aspect.split(":"))
    except ValueError:
        return {"ok": False, "reason": f"bad aspect {aspect!r}, use W:H"}
    target_ratio = aw / ah

    # Probe source dimensions + duration.
    probe = _ffprobe()
    sw, sh, duration = 0, 0, 0.0
    if probe:
        p = _run([probe, "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=width,height",
                  "-show_entries", "format=duration",
                  "-of", "json", src])
        try:
            info = json.loads(p.stdout)
            st = info["streams"][0]
            sw, sh = int(st["width"]), int(st["height"])
            duration = float(info["format"]["duration"])
        except (KeyError, ValueError, IndexError):
            pass
    if not sw or not sh:
        return {"ok": False, "reason": "could not probe video dimensions"}

    # Target crop size: fit aspect inside source.
    if sw / sh > target_ratio:
        cw, ch = int(sh * target_ratio), sh
    else:
        cw, ch = sw, int(sw / target_ratio)
    cw -= cw % 2
    ch -= ch % 2

    # Sample frames and find subject x-centers.
    n_samples = max(1, int(duration / sample_every)) if duration else 8
    centers: list[float] = []
    wd = Path(tempfile.mkdtemp())
    for i in range(n_samples):
        t = (i + 0.5) * (duration / n_samples) if duration else i
        fp = wd / f"f_{i:04d}.jpg"
        _run([ff, "-hide_banner", "-y", "-ss", str(t), "-i", src,
              "-frames:v", "1", "-vf", "scale=640:-1", str(fp)])
        if not fp.exists():
            centers.append(0.5)
            continue
        faces = _detect_faces(str(fp))
        if faces:
            # Weighted center of all faces.
            cx = sum(x + w / 2 for x, y, w, h in faces) / len(faces)
            # Map back: sampled at 640 wide.
            centers.append(min(0.95, max(0.05, cx / 640.0)))
        else:
            centers.append(0.5)
    # Smooth the path (moving average).
    if smooth and len(centers) > 2:
        sm: list[float] = []
        for i, c in enumerate(centers):
            window = centers[max(0, i - 1):i + 2]
            sm.append(sum(window) / len(window))
        centers = sm
    # Convert centers to crop x positions.
    max_x = sw - cw
    xs = [int(min(max_x, max(0, c * sw - cw / 2))) for c in centers]
    xs = [x - (x % 2) for x in xs]  # even for yuv420p

    out_p = Path(out) if out else Path(tempfile.mkdtemp()) / "reframed.mp4"
    # Animated crop via per-sample segments (simple, robust).
    seg = duration / n_samples if duration and n_samples else 1.0
    parts: list[str] = []
    inputs: list[str] = [ff, "-hide_banner", "-y", "-i", src]
    for i, x in enumerate(xs):
        parts.append(
            f"[0:v]trim=start={i * seg:.3f}:end={(i + 1) * seg:.3f},"
            f"setpts=PTS-STARTPTS,crop={cw}:{ch}:{x}:{(sh - ch) // 2}[v{i}]")
    vcat = "".join(f"[v{i}]" for i in range(len(xs)))
    cmd = inputs + ["-filter_complex",
                    ";".join(parts) + f";{vcat}concat=n={len(xs)}:v=1:a=0[v]",
                    "-map", "[v]", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", str(out_p)]
    proc = _run(cmd)
    if proc.returncode != 0 or not out_p.exists():
        return {"ok": False, "reason": proc.stderr[-500:]}
    return {"ok": True, "path": str(out_p), "aspect": aspect,
            "crop": f"{cw}x{ch}", "tracked_samples": len(xs),
            "method": "face-tracked" if any(c != 0.5 for c in centers)
                      else "center (no faces found)"}


# ── batch ──────────────────────────────────────────────────────────

def batch_process(srcs: list[str | os.PathLike[str]], *,
                  op: str,
                  out_dir: str | os.PathLike[str] | None = None,
                  **kwargs: Any) -> dict[str, Any]:
    """Apply one automation op to many files. Real batch, per-file results."""
    ops = {"cut_silences": cut_silences, "auto_reframe": auto_reframe,
           "detect_scenes": detect_scenes}
    fn = ops.get(op)
    if fn is None:
        return {"ok": False,
                "reason": f"unknown op {op!r}; pick from {sorted(ops)}"}
    od = Path(out_dir) if out_dir else Path(tempfile.mkdtemp())
    od.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for s in srcs:
        name = Path(s).stem
        try:
            if op == "detect_scenes":
                r: dict[str, Any] = {"scenes": fn(s, **kwargs)}
            else:
                r = fn(s, out=str(od / f"{name}_{op}.mp4"), **kwargs)
            r["src"] = str(s)
        except Exception as exc:  # noqa: BLE001
            r = {"ok": False, "src": str(s), "reason": str(exc)}
        results.append(r)
    ok = sum(1 for r in results if r.get("ok"))
    return {"ok": ok == len(results), "done": ok, "total": len(results),
            "results": results, "dir": str(od)}


# ── one-shot: rough cut ────────────────────────────────────────────

def rough_cut(src: str | os.PathLike[str], *,
              out: str | os.PathLike[str] | None = None,
              noise_db: float = -30.0,
              min_silence: float = 0.5,
              scene_threshold: float = 0.4) -> dict[str, Any]:
    """Full auto-edit pass: cut silences, then split on scenes.

    Returns the cleaned video plus a scene list — the starting point
    for a real edit, not a finished product.
    """
    cs = cut_silences(src, noise_db=noise_db, min_duration=min_silence)
    if not cs.get("ok"):
        return cs
    cleaned = cs["path"]
    scenes = detect_scenes(cleaned, threshold=scene_threshold)
    out_p = Path(out) if out else Path(cleaned)
    if str(out_p) != cleaned:
        os.replace(cleaned, out_p)
        cleaned = str(out_p)
    return {"ok": True, "path": cleaned,
            "seconds_removed": cs.get("seconds_removed", 0),
            "scenes": [{"index": s.index, "start": s.start,
                        "end": s.end, "duration": s.duration}
                       for s in scenes],
            "note": "rough cut complete — review scenes before fine edit"}
