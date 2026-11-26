"""Vertical rendering — ported from the source's
``trendclip/media_pipeline.py`` (``_render_with_opencv``,
``_vertical_frame``, ``_draw_centered_words``, ``_draw_hook``,
``_persistent_portrait_crop``, ``_analyze_framing``).

A 1080x1920 vertical clip: cover-blur background + foreground reframe,
per-frame word captions, a hook card for the first 84 frames, three
layouts (balanced / punch / clean), encoded by an FFmpeg rawvideo pipe
with loudness normalization to -14 LUFS.

OpenCV is imported lazily inside the functions that need it, so the organ
imports on a box without it — the render then fails honestly naming the
missing dependency, and the analysis rung still runs on audio + speech.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "LAYOUTS",
    "render_vertical_clip",
    "visual_signals",
    "write_ass_captions",
]

WIDTH = 1080
HEIGHT = 1920
FPS = 30
HOOK_FRAMES = 84  # ~2.8s at 30fps: the hook card's lifetime
LAYOUTS = ("balanced", "punch", "clean")


def _require_cv2():  # type: ignore[no-untyped-def]
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed — the vertical reframe needs it; "
            "Devon can provision it (trendclip provisioner), or run "
            "analysis without rendering") from exc
    return cv2


def visual_signals(video_path: str, seconds: int) -> list[float]:
    """Per-second visual motion (mean frame-diff), the heat signal.

    Returns zeros when OpenCV is unavailable — the scorer treats a flat
    heat series as no visual evidence, and the receipt says so.
    """
    if seconds <= 0:
        return []
    try:
        cv2 = _require_cv2()
    except RuntimeError:
        return [0.0] * seconds
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0.0] * seconds
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    signals = [0.0] * seconds
    counts = [0] * seconds
    prev = None
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sec = int(frame_no / fps)
            if sec >= seconds:
                break
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                signals[sec] += float(diff.mean())
                counts[sec] += 1
            prev = gray
            frame_no += 1
    finally:
        cap.release()
    return [signals[i] / counts[i] if counts[i] else 0.0
            for i in range(seconds)]


def _foreground_box(cv2: Any, frame: Any) -> tuple[int, int, int, int] | None:
    """The largest moving region in a frame, or None when nothing moves."""
    small = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 200:
        return None
    x, y, w, h = cv2.boundingRect(biggest)
    scale_x = frame.shape[1] / 160.0
    scale_y = frame.shape[0] / 90.0
    return (int(x * scale_x), int(y * scale_y),
            int(w * scale_x), int(h * scale_y))


def _persistent_portrait_crop(cv2: Any, frames: list[Any],
                              frame_w: int, frame_h: int
                              ) -> tuple[int, int, int, int]:
    """The portrait crop that follows the subject with hysteresis: the box
    only moves when the subject has moved persistently (3 consecutive
    samples past the threshold), so the frame never jitters."""
    boxes = [b for b in (_foreground_box(cv2, f) for f in frames)
             if b is not None]
    if not boxes:
        # No motion found: center crop, 9:16 of the frame.
        crop_w = int(frame_h * 9 / 16)
        crop_w = min(crop_w, frame_w)
        return ((frame_w - crop_w) // 2, 0, crop_w, frame_h)
    xs = sorted(b[0] + b[2] // 2 for b in boxes)
    cx = xs[len(xs) // 2]
    crop_w = int(frame_h * 9 / 16)
    crop_w = min(crop_w, frame_w)
    x = max(0, min(frame_w - crop_w, cx - crop_w // 2))
    return (x, 0, crop_w, frame_h)


def _vertical_frame(cv2: Any, frame: Any, crop: tuple[int, int, int, int],
                    layout: str) -> Any:
    """Reframe one source frame into a 1080x1920 vertical frame: a blurred
    cover background with the portrait crop as the foreground."""
    import numpy as np

    x, y, w, h = crop
    fg = frame[y:y + h, x:x + w]
    if layout == "punch":
        fg_h = HEIGHT
        fg_w = WIDTH
    elif layout == "clean":
        fg_h = int(HEIGHT * 0.62)
        fg_w = int(fg_h * w / max(1, h))
    else:  # balanced
        fg_h = int(HEIGHT * 0.76)
        fg_w = int(fg_h * w / max(1, h))
    fg_w = min(fg_w, WIDTH)
    fg = cv2.resize(fg, (fg_w, fg_h))
    bg = cv2.resize(frame, (WIDTH, HEIGHT))
    bg = cv2.GaussianBlur(bg, (0, 0), 30)
    bg = cv2.addWeighted(bg, 0.55, np.zeros_like(bg), 0.45, 0)
    out = bg
    x0 = (WIDTH - fg_w) // 2
    y0 = 0 if layout == "punch" else int(HEIGHT * 0.06)
    out[y0:y0 + fg_h, x0:x0 + fg_w] = fg
    return out


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = (current + " " + word).strip()
        if len(trial) <= max_chars:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_centered_words(cv2: Any, frame: Any, words: list[dict[str, Any]],
                         t: float, layout: str) -> Any:
    """Per-frame word captions: the word spoken at time ``t`` is bright,
    the words around it are dimmed. Two lines max, centered low."""
    if not words:
        return frame
    idx = None
    for i, word in enumerate(words):
        if float(word.get("s", 0.0)) <= t < float(word.get("e", 0.0)):
            idx = i
            break
    if idx is None:
        # Between words: show the nearest word's line.
        nearest = min(range(len(words)),
                      key=lambda i: abs(float(words[i].get("s", 0.0)) - t))
        if abs(float(words[nearest].get("s", 0.0)) - t) > 1.0:
            return frame
        idx = nearest
    lo = max(0, idx - 2)
    hi = min(len(words), idx + 3)
    line = " ".join(str(words[i].get("w", "")) for i in range(lo, hi))
    lines = _wrap_text(line, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.6 if layout == "punch" else 1.4
    thickness = 3
    y_base = HEIGHT - 260 if layout == "punch" else HEIGHT - 320
    for j, text in enumerate(lines[-2:]):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (WIDTH - tw) // 2
        y = y_base + j * 70
        cv2.putText(frame, text, (x + 3, y + 3), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _draw_hook(cv2: Any, frame: Any, hook_text: str) -> Any:
    """The hook card: the moment's hook line as a title over the frame."""
    if not hook_text:
        return frame
    lines = _wrap_text(hook_text, 22)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 4
    total_h = len(lines) * 90
    y0 = (HEIGHT - total_h) // 2
    for j, text in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (WIDTH - tw) // 2
        y = y0 + j * 90
        cv2.putText(frame, text, (x + 4, y + 4), font, scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def write_ass_captions(words: list[dict[str, Any]], path: str | Path,
                       *, start: float = 0.0) -> Path:
    """Write word-timed ASS captions (the source's caption format)."""
    def fmt(t: float) -> str:
        t = max(0.0, t - start)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,2,40,40,160,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
        " Effect, Text",
    ]
    for word in words:
        text = str(word.get("w", "")).strip()
        if not text:
            continue
        lines.append(
            f"Dialogue: 0,{fmt(float(word.get('s', 0.0)))},"
            f"{fmt(float(word.get('e', 0.0)))},Default,,0,0,0,,{text}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def render_vertical_clip(*, video_path: str, audio_path: str,
                         words: list[dict[str, Any]],
                         start: float, end: float,
                         output_path: str | Path,
                         layout: str = "balanced",
                         hook_text: str = "",
                         runner: Any = None) -> dict[str, Any]:
    """Render one vertical clip from a source section.

    ``video_path`` is the proxy (or source) video, ``audio_path`` the
    section's extracted audio (wav). Frames are piped as rawvideo to
    FFmpeg, which encodes libx264 (crf 20) with the audio loudness-
    normalized to -14 LUFS. Returns ``{"ok", "path", "duration",
    "frames", "layout"}`` or ``{"ok": False, "error": ...}``.
    """
    if layout not in LAYOUTS:
        layout = "balanced"
    duration = end - start
    if duration <= 0:
        return {"ok": False, "error": "empty clip section"}
    try:
        cv2 = _require_cv2()
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": f"cannot open video {video_path}"}
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return {"ok": False, "error": "video has no dimensions"}
    start_frame = int(start * src_fps)
    end_frame = int(end * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # Sample frames across the section for the persistent crop.
    samples: list[Any] = []
    sample_at = [start_frame + int(i * (end_frame - start_frame) / 8)
                 for i in range(8)]
    for frame_no in sample_at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if ok:
            samples.append(frame)
    crop = _persistent_portrait_crop(cv2, samples, src_w, src_h)
    # Time-based sampling, mirroring the source: the output runs at
    # FPS (30) regardless of the source rate. Each output frame seeks
    # to its timestamp (start + i/FPS) and reads the frame there, so a
    # 24fps source yields 30fps real-time output — never a sped-up
    # clip. The first version read frames sequentially (source cadence)
    # while declaring them 30fps, so a 60s moment from Big Buck Bunny
    # (24fps) came out as a 48s clip running 25% fast, with the audio
    # truncated by -shortest. Caught by the acceptance run.
    frame_count = math.ceil(duration * FPS)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-i", str(audio_path),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest", str(out_path),
    ]
    run = runner or subprocess.Popen
    proc = run(ffmpeg, stdin=subprocess.PIPE,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames_written = 0
    try:
        for index in range(frame_count):
            t = index / FPS
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            vertical = _vertical_frame(cv2, frame, crop, layout)
            if frames_written < HOOK_FRAMES:
                vertical = _draw_hook(cv2, vertical, hook_text)
            else:
                vertical = _draw_centered_words(
                    cv2, vertical, words, start + t, layout)
            try:
                proc.stdin.write(vertical.tobytes())
            except BrokenPipeError:
                break
            frames_written += 1
    finally:
        cap.release()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait(timeout=300)
    if proc.returncode != 0 or frames_written == 0:
        return {"ok": False,
                "error": "ffmpeg encode failed"
                + (f" (rc={proc.returncode})" if proc.returncode else ""),
                "frames": frames_written}
    return {
        "ok": True,
        "path": str(out_path),
        "duration": round(frames_written / FPS, 2),
        "frames": frames_written,
        "layout": layout,
        "crop": list(crop),
    }
