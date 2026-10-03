"""ffmpeg-backed video editing engine.

No heavy new dependencies (OpenCV stays out): everything goes through the
ffmpeg/ffprobe CLIs via subprocess. Every op writes a NEW file — originals
are never overwritten. If ffmpeg is missing at runtime, video ops raise a
clear error while image tools keep working.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .images import MediaEditError, _unique_output

_log = get_logger(__name__)

FFMPEG_TIMEOUT = 600.0  # 10 minutes default per spec
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB default cap

_FFMPEG_HINT = (
    "ffmpeg is not installed on this machine. Install it to enable video "
    "editing — Debian/Ubuntu: `sudo apt install ffmpeg`; macOS: "
    "`brew install ffmpeg`; Windows: `winget install ffmpeg`; "
    "or download a static build from https://ffmpeg.org/download.html. "
    "Image tools are unaffected."
)


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaEditError(_FFMPEG_HINT)
    return path


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def has_libass() -> bool:
    """True when this ffmpeg build can burn subtitles."""
    try:
        out = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return " subtitles " in out or " ass " in out


# ---------------------------------------------------------------------------
# time parsing / progress
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?)?$")


def parse_time(value: str | float | int) -> float:
    """'90' -> 90.0, '1:30' -> 90.0, '0:01:30.5' -> 90.5, 30 -> 30.0.

    Colon forms are interpreted by component count: 'M:SS' is
    minutes:seconds, 'H:MM:SS' is hours:minutes:seconds.
    """
    if isinstance(value, (int, float)):
        if value < 0:
            raise MediaEditError(f"negative timestamp {value}")
        return float(value)
    text = str(value).strip().lower()
    for suffix, mult in (("ms", 0.001), ("s", 1.0), ("sec", 1.0),
                         ("secs", 1.0), ("second", 1.0), ("seconds", 1.0),
                         ("m", 60.0), ("min", 60.0), ("mins", 60.0),
                         ("minute", 60.0), ("minutes", 60.0)):
        if text.endswith(suffix) and text[: -len(suffix)].strip():
            num = text[: -len(suffix)].strip()
            try:
                secs = float(num) * mult
            except ValueError:
                break
            if secs < 0:
                raise MediaEditError(f"negative timestamp {value!r}")
            return secs
    m = _TIME_RE.match(text)
    if m:
        parts = [float(p) for p in m.groups() if p is not None]
        if len(parts) == 3:
            hours, minutes, seconds = parts
            if minutes >= 60 or seconds >= 60:
                raise MediaEditError(
                    f"bad timestamp {value!r}: minutes/seconds must be < 60")
        elif len(parts) == 2:
            hours, minutes, seconds = 0.0, parts[0], parts[1]
            if seconds >= 60:
                raise MediaEditError(
                    f"bad timestamp {value!r}: seconds must be < 60")
        else:
            hours, minutes, seconds = 0.0, 0.0, parts[0]
        return hours * 3600 + minutes * 60 + seconds
    raise MediaEditError(
        f"could not parse timestamp {value!r}; use seconds, 'MM:SS', or 'HH:MM:SS'")


def run_ffmpeg(args: list[str], *,
               timeout: float = FFMPEG_TIMEOUT,
               progress_cb: Callable[[float], None] | None = None,
               duration: float | None = None,
               cwd: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Run ffmpeg, streaming machine-readable progress. Never hangs forever:
    ``timeout`` kills the whole process tree."""
    cmd = [ffmpeg_path(), "-hide_banner", "-y", "-nostats",
           "-progress", "pipe:1"] + args
    _log.info("ffmpeg: %s", " ".join(cmd[1:5]) + " ...")
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=cwd, start_new_session=True,
        )
    except OSError as exc:
        raise MediaEditError(f"could not start ffmpeg: {exc}") from exc
    progress_lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms=") and progress_cb and duration:
                try:
                    ms = float(line.split("=", 1)[1])
                    progress_cb(min(1.0, max(0.0, (ms / 1_000_000) / duration)))
                except ValueError:  # noqa: E103 - one malformed progress line; keep reading
                    pass
            elif line == "progress=end" and progress_cb:
                progress_cb(1.0)
            if len(progress_lines) < 40:
                progress_lines.append(line)
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_tree(proc)
        raise MediaEditError(
            f"ffmpeg timed out after {timeout:.0f}s and was killed") from exc
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        tail = (stderr or "")[-2000:]
        raise MediaEditError(f"ffmpeg failed (exit {proc.returncode}): {tail}")
    return {"seconds": round(elapsed, 2)}


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:  # noqa: E103 - process already gone; teardown is best-effort
            pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------

def video_probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Duration, streams, codec, resolution, fps. Falls back to parsing
    ``ffmpeg -i`` stderr when ffprobe is missing."""
    p = Path(path)
    if not p.exists():
        raise MediaEditError(f"no such video: {path}")
    ffmpeg_path()  # fail fast with the clear message
    probe_bin = ffprobe_path()
    if probe_bin:
        try:
            out = subprocess.run(
                [probe_bin, "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(p)],
                capture_output=True, text=True, timeout=60, check=True,
            ).stdout
            data = json.loads(out)
            return _summarize_probe(p, data)
        except (subprocess.CalledProcessError, json.JSONDecodeError,
                subprocess.TimeoutExpired) as exc:
            _log.warning("ffprobe failed, falling back to ffmpeg -i: %s", exc)
    return _probe_via_ffmpeg(p)


def _summarize_probe(p: Path, data: dict[str, Any]) -> dict[str, Any]:
    streams = []
    for s in data.get("streams", []):
        streams.append({
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
            "width": s.get("width"),
            "height": s.get("height"),
            "fps": _fps(s.get("avg_frame_rate") or s.get("r_frame_rate")),
            "duration": _safe_float(s.get("duration")),
        })
    fmt = data.get("format", {})
    video = next((s for s in streams if s["type"] == "video"), {})
    return {
        "kind": "video",
        "path": str(p),
        "bytes": p.stat().st_size,
        "duration": _safe_float(fmt.get("duration")) or video.get("duration"),
        "format": fmt.get("format_name"),
        "streams": streams,
        "width": video.get("width"),
        "height": video.get("height"),
        "video_codec": video.get("codec"),
        "fps": video.get("fps"),
    }


def _fps(value: Any) -> float | None:
    if not value or value in ("0/0",):
        return None
    try:
        if "/" in str(value):
            num, den = str(value).split("/")
            return float(num) / float(den) if float(den) else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _probe_via_ffmpeg(p: Path) -> dict[str, Any]:
    out = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(p)],
        capture_output=True, text=True, timeout=60,
    ).stderr
    duration = None
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    streams = []
    for sm in re.finditer(r"Stream #\d+:(\d+).*?: (Video|Audio|Subtitle): (\w+)",
                          out):
        streams.append({"index": int(sm.group(1)),
                        "type": sm.group(2).lower(), "codec": sm.group(3)})
    res = re.search(r"(\d{2,5})x(\d{2,5})", out)
    fps_m = re.search(r"(\d+(?:\.\d+)?) fps", out)
    return {
        "kind": "video", "path": str(p), "bytes": p.stat().st_size,
        "duration": duration, "format": None, "streams": streams,
        "width": int(res.group(1)) if res else None,
        "height": int(res.group(2)) if res else None,
        "video_codec": next((s["codec"] for s in streams
                             if s["type"] == "video"), None),
        "fps": float(fps_m.group(1)) if fps_m else None,
        "probe_method": "ffmpeg -i fallback",
    }


# ---------------------------------------------------------------------------
# ops — every one writes a NEW file
# ---------------------------------------------------------------------------

def _check_size(p: Path, cap: int = MAX_VIDEO_BYTES) -> None:
    size = p.stat().st_size
    if size > cap:
        raise MediaEditError(
            f"{p.name} is {size / 1e6:.0f}MB, over the {cap / 1e6:.0f}MB cap")


def _out(src: Path, out_dir: Path | None, suffix: str, ext: str) -> Path:
    target = out_dir or (src.parent / "edited")
    return _unique_output(src, Path(target), suffix, ext)


def trim(src: str | os.PathLike[str],
         start: str | float | int = 0,
         end: str | float | int | None = None,
         *,
         out_dir: str | os.PathLike[str] | None = None,
         suffix: str = "trimmed",
         ext: str = ".mp4",
         timeout: float = FFMPEG_TIMEOUT,
         progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Cut [start, end). Stream-copies when the cut allows it (fast path);
    re-encodes only when stream copy fails."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    s = parse_time(start)
    e = parse_time(end) if end is not None else None
    if e is not None and e <= s:
        raise MediaEditError(f"trim end ({e}s) must be after start ({s}s)")
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    clip_len = (e - s) if e else (info.get("duration") or 0) - s
    # Fast path: stream copy.
    args = ["-ss", str(s)]
    if e is not None:
        args += ["-to", str(e)]
    args += ["-i", str(p), "-c", "copy", str(out)]
    try:
        run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
                   duration=clip_len or None)
        mode = "stream-copy"
    except MediaEditError as exc:
        _log.info("stream copy failed, re-encoding: %s", exc)
        if out.exists():
            out.unlink()
        out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
        args = ["-ss", str(s)]
        if e is not None:
            args += ["-to", str(e)]
        args += ["-i", str(p), "-c:v", "libx264", "-preset", "fast",
                 "-crf", "20", "-c:a", "aac", str(out)]
        run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
                   duration=clip_len or None)
        mode = "re-encode"
    return {"input": str(p), "output": str(out), "mode": mode,
            "start": s, "end": e, "bytes": out.stat().st_size}


def speed(src: str | os.PathLike[str],
          factor: float,
          *,
          out_dir: str | os.PathLike[str] | None = None,
          suffix: str = "sped",
          ext: str = ".mp4",
          timeout: float = FFMPEG_TIMEOUT,
          progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Change playback speed. factor > 1 = faster, 0 < factor < 1 = slower.

    Video via setpts, audio via atempo (chained for extreme factors).
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if factor <= 0:
        raise MediaEditError(f"speed factor must be > 0, got {factor}")
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    duration = info.get("duration") or 0
    # atempo supports 0.5–100; chain for out-of-range factors.
    atempo_chain = []
    f = factor
    while f > 100:
        atempo_chain.append("atempo=100")
        f /= 100
    while f < 0.5:
        atempo_chain.append("atempo=0.5")
        f /= 0.5
    atempo_chain.append(f"atempo={f}")
    atempo = ",".join(atempo_chain)
    vf = f"setpts=PTS/{factor}"
    args = ["-i", str(p), "-vf", vf, "-af", atempo,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", str(out)]
    run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
               duration=duration / factor if duration else None)
    return {"input": str(p), "output": str(out), "factor": factor,
            "bytes": out.stat().st_size}


def fade(src: str | os.PathLike[str],
         fade_in: float = 0.0,
         fade_out: float = 0.0,
         *,
         out_dir: str | os.PathLike[str] | None = None,
         suffix: str = "faded",
         ext: str = ".mp4",
         timeout: float = FFMPEG_TIMEOUT,
         progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Fade video+audio in from black / out to black.

    fade_in/fade_out in seconds; 0 disables that end.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if fade_in < 0 or fade_out < 0:
        raise MediaEditError("fade durations must be >= 0")
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    duration = info.get("duration") or 0
    vf_parts = []
    af_parts = []
    if fade_in > 0:
        vf_parts.append(f"fade=t=in:st=0:d={fade_in}")
        af_parts.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        st = max(0, duration - fade_out) if duration else 0
        vf_parts.append(f"fade=t=out:st={st}:d={fade_out}")
        af_parts.append(f"afade=t=out:st={st}:d={fade_out}")
    args = ["-i", str(p)]
    if vf_parts:
        args += ["-vf", ",".join(vf_parts)]
    if af_parts:
        args += ["-af", ",".join(af_parts)]
    args += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", str(out)]
    run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
               duration=duration or None)
    return {"input": str(p), "output": str(out),
            "fade_in": fade_in, "fade_out": fade_out,
            "bytes": out.stat().st_size}


def overlay_text(src: str | os.PathLike[str],
                 text: str,
                 *,
                 position: str = "bottom",
                 fontsize: int = 48,
                 fontcolor: str = "white",
                 start: float | None = None,
                 end: float | None = None,
                 out_dir: str | os.PathLike[str] | None = None,
                 suffix: str = "captioned",
                 ext: str = ".mp4",
                 timeout: float = FFMPEG_TIMEOUT,
                 progress_cb: Callable[[float], None] | None = None
                 ) -> dict[str, Any]:
    """Burn text onto video with ffmpeg drawtext.

    position: top|bottom|center. start/end (seconds) limit visibility.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if not text:
        raise MediaEditError("overlay_text needs text")
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    duration = info.get("duration") or 0
    # Position presets.
    y_map = {"top": "y=40", "bottom": "y=h-th-40", "center": "y=(h-th)/2"}
    y = y_map.get(position, y_map["bottom"])
    # Escape for drawtext.
    safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    dt = f"drawtext=text='{safe}':fontsize={fontsize}:fontcolor={fontcolor}"
    dt += ":x=(w-text_w)/2:" + y
    if start is not None or end is not None:
        s = start or 0
        e = end if end is not None else 999999
        dt += f":enable='between(t,{s},{e})'"
    args = ["-i", str(p), "-vf", dt,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy", str(out)]
    run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
               duration=duration or None)
    return {"input": str(p), "output": str(out), "text": text,
            "bytes": out.stat().st_size}


# ---------------------------------------------------------------------------
# pro ops — transitions, one-click effects, speed ramping, audio mixing
# ---------------------------------------------------------------------------

def _has_audio(info: dict[str, Any]) -> bool:
    return any(s.get("type") == "audio" for s in info.get("streams", []))


def _norm_video_chain(info: dict[str, Any], ref: dict[str, Any]) -> str:
    """Normalize one video stream to the reference clip's geometry.

    xfade/concat filters demand identical resolution, pixel format and
    framerate on every input; scaling to clip A keeps the transition
    robust when the clips don't match.
    """
    w, h, fps = ref.get("width"), ref.get("height"), ref.get("fps")
    parts = ["format=yuv420p"]
    if w and h:
        parts.append(f"scale={int(w)}:{int(h)}:flags=lanczos")
    parts.append("setsar=1")
    if fps:
        parts.append(f"fps={fps}")
    return ",".join(parts)


# Validated against `ffmpeg -h filter=xfade` (ffmpeg 8.1.2); "custom" is
# excluded since it needs an extra transition-source input.
_XFADE_TRANSITIONS = (
    "fade", "fadeblack", "fadewhite", "wipeleft", "wiperight", "wipeup",
    "wipedown", "slideleft", "slideright", "slideup", "slidedown",
    "smoothleft", "smoothright", "smoothup", "smoothdown", "circlecrop",
    "rectcrop", "circleopen", "circleclose", "vertopen", "vertclose",
    "horzopen", "horzclose", "dissolve", "pixelize", "radial", "distance",
    "diagtl", "diagtr", "diagbl", "diagbr", "hlslice", "hrslice",
    "vuslice", "vdslice", "hblur", "fadegrays", "fadefast", "fadeslow",
    "squeezeh", "squeezev", "zoomin", "wipetl", "wipetr", "wipebl",
    "wipebr", "hlwind", "hrwind", "vuwind", "vdwind", "coverleft",
    "coverright", "coverup", "coverdown", "revealleft", "revealright",
    "revealup", "revealdow",
)


def transition(a: str | os.PathLike[str],
               b: str | os.PathLike[str], *,
               kind: str = "xfade",
               duration: float = 1.0,
               transition: str = "fade",
               out_dir: str | os.PathLike[str] | None = None,
               suffix: str = "transitioned", ext: str = ".mp4",
               timeout: float = FFMPEG_TIMEOUT,
               progress_cb: Callable[[float], None] | None = None
               ) -> dict[str, Any]:
    """Join two clips with a real transition instead of a hard cut.

    kind="xfade": crossfade via the xfade filter (video) + acrossfade
        (audio). ``transition`` picks the xfade transition type
        (fade, slideleft, dissolve, ...).
    kind="fadeblack": A fades out to black, B fades in from black
        (classic dip-to-black); output is durA + durB.
    """
    pa, pb = Path(a), Path(b)
    for p in (pa, pb):
        if not p.exists():
            raise MediaEditError(f"no such video: {p}")
        _check_size(p)
    if kind not in ("xfade", "fadeblack"):
        raise MediaEditError(
            f"unknown transition kind {kind!r}; valid: xfade, fadeblack")
    if kind == "xfade" and transition not in _XFADE_TRANSITIONS:
        raise MediaEditError(
            f"unknown xfade transition {transition!r}; "
            f"valid: {', '.join(_XFADE_TRANSITIONS)}")
    if not duration > 0:
        raise MediaEditError(f"transition duration must be > 0, got {duration}")
    info_a, info_b = video_probe(pa), video_probe(pb)
    dur_a, dur_b = info_a.get("duration"), info_b.get("duration")
    if not dur_a or not dur_b:
        raise MediaEditError("cannot transition: unknown clip duration")
    if duration >= dur_a or duration >= dur_b:
        raise MediaEditError(
            f"transition duration ({duration}s) must be shorter than both "
            f"clips ({dur_a:.1f}s, {dur_b:.1f}s)")
    out = _out(pa, Path(out_dir) if out_dir else None, suffix, ext)
    norm_a = _norm_video_chain(info_a, info_a)
    norm_b = _norm_video_chain(info_b, info_a)
    a_audio, b_audio = _has_audio(info_a), _has_audio(info_b)
    if kind == "xfade":
        offset = dur_a - duration
        out_dur = dur_a + dur_b - duration
        fc = (f"[0:v]{norm_a}[va];[1:v]{norm_b}[vb];"
              f"[va][vb]xfade=transition={transition}:duration={duration}"
              f":offset={offset}[vout];")
        audio_map: list[str] = []
        if a_audio and b_audio:
            fc += f"[0:a][1:a]acrossfade=d={duration}[aout]"
            audio_map = ["-map", "[aout]"]
        elif a_audio or b_audio:
            idx = 0 if a_audio else 1
            fc += f"[{idx}:a]apad[aout]"
            audio_map = ["-map", "[aout]", "-t", str(out_dur)]
        else:
            audio_map = ["-an"]
    else:  # fadeblack
        out_dur = dur_a + dur_b
        fc = (f"[0:v]{norm_a},fade=t=out:st={dur_a - duration}"
              f":d={duration}[va];"
              f"[1:v]{norm_b},fade=t=in:st=0:d={duration}[vb];"
              f"[va][vb]concat=n=2:v=1:a=0[vout];")
        audio_map = []
        if a_audio and b_audio:
            fc += "[0:a][1:a]concat=n=2:v=0:a=1[aout]"
            audio_map = ["-map", "[aout]"]
        elif a_audio or b_audio:
            idx = 0 if a_audio else 1
            fc += f"[{idx}:a]apad[aout]"
            audio_map = ["-map", "[aout]", "-t", str(out_dur)]
        else:
            audio_map = ["-an"]
    args = (["-i", str(pa), "-i", str(pb), "-filter_complex", fc,
             "-map", "[vout]"] + audio_map +
            ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", str(out)])
    run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
               duration=out_dur or None)
    result: dict[str, Any] = {
        "input": str(pa), "inputs": [str(pa), str(pb)],
        "output": str(out), "kind": kind, "duration": out_dur,
        "bytes": out.stat().st_size,
    }
    if kind == "xfade":
        result["transition"] = transition
    return result


# One-click looks: each preset is a plain ffmpeg -vf string.
_EFFECTS = {
    "grayscale": "hue=s=0",
    "sepia": ("colorchannelmixer=.393:.769:.189:0:"
              ".349:.686:.168:0:.272:.534:.131"),
    "vignette": "vignette=PI/4",
    "sharpen": "unsharp=5:5:1.0:5:5:0.0",
    "denoise": "hqdn3d=4:4:6:6",
    "vintage": "curves=vintage,colorbalance=rs=.1:gs=-.05:bs=-.1",
    "invert": "negate",
}


def effect(src: str | os.PathLike[str],
           preset: str, *,
           out_dir: str | os.PathLike[str] | None = None,
           suffix: str | None = None,
           ext: str = ".mp4",
           timeout: float = FFMPEG_TIMEOUT,
           progress_cb: Callable[[float], None] | None = None
           ) -> dict[str, Any]:
    """Apply a one-click look: grayscale, sepia, vignette, sharpen,
    denoise, vintage, invert. Audio is untouched."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    vf = _EFFECTS.get(preset)
    if vf is None:
        raise MediaEditError(
            f"unknown effect {preset!r}; "
            f"valid presets: {', '.join(sorted(_EFFECTS))}")
    out = _out(p, Path(out_dir) if out_dir else None,
               suffix or f"fx-{preset}", ext)
    info = video_probe(p)
    run_ffmpeg(["-i", str(p), "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "copy", str(out)],
               timeout=timeout, progress_cb=progress_cb,
               duration=info.get("duration"))
    return {"input": str(p), "output": str(out), "preset": preset,
            "filter": vf, "bytes": out.stat().st_size}


def speed_ramp(src: str | os.PathLike[str],
               segments: list[tuple[str | float, str | float, float]], *,
               out_dir: str | os.PathLike[str] | None = None,
               suffix: str = "ramped", ext: str = ".mp4",
               timeout: float = FFMPEG_TIMEOUT,
               progress_cb: Callable[[float], None] | None = None
               ) -> dict[str, Any]:
    """Variable speed: segments=[(start, end, factor), ...] must tile the
    whole timeline with no gaps and no overlaps — anything else is a
    MediaEditError, never a guess.

    Implemented honestly: trim each segment, run the existing speed() on
    it, then concat the parts.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if not segments:
        raise MediaEditError("speed_ramp needs at least one segment")
    duration = (video_probe(p).get("duration"))
    if not duration:
        raise MediaEditError("cannot speed_ramp: unknown video duration")
    parsed: list[tuple[float, float, float]] = []
    for i, seg in enumerate(segments):
        if len(seg) != 3:
            raise MediaEditError(
                f"segment {i} must be (start, end, factor), got {seg!r}")
        s, e, factor = parse_time(seg[0]), parse_time(seg[1]), float(seg[2])
        if not factor > 0:
            raise MediaEditError(
                f"segment {i}: factor must be > 0, got {seg[2]!r}")
        if e <= s:
            raise MediaEditError(
                f"segment {i}: end ({e}s) must be after start ({s}s)")
        if s < 0 or e > duration:
            raise MediaEditError(
                f"segment {i}: [{s}s, {e}s] is outside the video "
                f"(duration {duration:.1f}s)")
        parsed.append((s, e, factor))
    order = sorted(range(len(parsed)), key=lambda i: parsed[i][0])
    if order != list(range(len(parsed))):
        raise MediaEditError("segments must be sorted by start time")
    if abs(parsed[0][0]) > 1e-3:
        raise MediaEditError(
            f"gap at the start: first segment begins at {parsed[0][0]}s, "
            "segments must tile the whole timeline from 0")
    for i in range(len(parsed) - 1):
        prev_e, cur_s = parsed[i][1], parsed[i + 1][0]
        if cur_s < prev_e - 1e-3:
            raise MediaEditError(
                f"segments {i} and {i + 1} overlap "
                f"([{parsed[i][0]}s, {prev_e}s] vs [{cur_s}s, ...])")
        if cur_s - prev_e > 1e-3:
            raise MediaEditError(
                f"gap between segments {i} and {i + 1} "
                f"({prev_e}s -> {cur_s}s): segments must tile the timeline")
    if duration - parsed[-1][1] > 1e-3:
        raise MediaEditError(
            f"gap at the end: last segment ends at {parsed[-1][1]}s, "
            f"video is {duration:.1f}s")
    tmpdir = tempfile.mkdtemp(prefix="speedramp-")
    try:
        parts = []
        for i, (s, e, factor) in enumerate(parsed):
            seg = trim(src, start=s, end=e, out_dir=tmpdir,
                       suffix=f"seg{i:02d}")
            sped = speed(seg["output"], factor, out_dir=tmpdir,
                         suffix=f"seg{i:02d}r")
            parts.append(sped["output"])
        final = concat(parts, out_dir=out_dir, suffix=suffix, ext=ext,
                       timeout=timeout, progress_cb=progress_cb)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    out = Path(final["output"])
    return {"input": str(p), "output": str(out),
            "segments": [{"start": s, "end": e, "factor": f}
                         for s, e, f in parsed],
            "parts": parts, "bytes": out.stat().st_size}


def ducking(src: str | os.PathLike[str],
            music: str | os.PathLike[str], *,
            voice_db: float = 0.0,
            music_db: float = -14.0,
            out_dir: str | os.PathLike[str] | None = None,
            suffix: str = "ducked", ext: str = ".mp4",
            timeout: float = FFMPEG_TIMEOUT,
            progress_cb: Callable[[float], None] | None = None
            ) -> dict[str, Any]:
    """Audio ducking: lay ``music`` under the video's audio and duck it
    whenever the video's voice/audio is present.

    sidechaincompress is keyed on the video's own audio stream against
    the music input; the music is first lowered to ``music_db`` dB, the
    voice trimmed to ``voice_db`` dB. Music is padded so a short track
    never cuts the video short.
    """
    p = Path(src)
    m = Path(music)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    if not m.exists():
        raise MediaEditError(f"no such music file: {music}")
    _check_size(p)
    _check_size(m)
    for name, db in (("voice_db", voice_db), ("music_db", music_db)):
        try:
            float(db)
        except (TypeError, ValueError):
            raise MediaEditError(f"{name} must be a number, got {db!r}")
    info = video_probe(p)
    if not _has_audio(info):
        raise MediaEditError(
            "ducking needs an audio stream in the video to key on")
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    duration = info.get("duration")
    fc = (f"[0:a]asplit[a_voice][a_key];"
          f"[a_voice]volume={voice_db}dB[v];"
          f"[1:a]volume={music_db}dB,apad[m];"
          f"[m][a_key]sidechaincompress=threshold=0.02:ratio=8"
          f":attack=20:release=400[d];"
          f"[v][d]amix=inputs=2:duration=first:dropout_transition=0[aout]")
    run_ffmpeg(["-i", str(p), "-i", str(m), "-filter_complex", fc,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", str(out)],
               timeout=timeout, progress_cb=progress_cb,
               duration=duration or None)
    return {"input": str(p), "output": str(out), "music": str(m),
            "voice_db": voice_db, "music_db": music_db,
            "bytes": out.stat().st_size}


def mix_audio(src: str | os.PathLike[str],
              audio: str | os.PathLike[str], *,
              volume: float = 1.0,
              replace: bool = False,
              out_dir: str | os.PathLike[str] | None = None,
              suffix: str = "mixed", ext: str = ".mp4",
              timeout: float = FFMPEG_TIMEOUT,
              progress_cb: Callable[[float], None] | None = None
              ) -> dict[str, Any]:
    """Lay an audio track under the video.

    replace=False mixes it with the existing audio (amix); replace=True
    swaps the video's audio for the new track. The track is
    padded/trimmed to the video's length so the video is never cut.
    """
    p = Path(src)
    a = Path(audio)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    if not a.exists():
        raise MediaEditError(f"no such audio file: {audio}")
    _check_size(p)
    _check_size(a)
    if not volume >= 0:
        raise MediaEditError(f"volume must be >= 0, got {volume}")
    info = video_probe(p)
    duration = info.get("duration")
    if not duration:
        raise MediaEditError("cannot mix_audio: unknown video duration")
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    src_audio = _has_audio(info)
    if replace or not src_audio:
        fc = (f"[1:a]volume={volume},apad,atrim=0:{duration}[aout]")
    else:
        fc = (f"[1:a]volume={volume},apad[b];"
              f"[0:a][b]amix=inputs=2:duration=first"
              f":dropout_transition=0[aout]")
    run_ffmpeg(["-i", str(p), "-i", str(a), "-filter_complex", fc,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", str(out)],
               timeout=timeout, progress_cb=progress_cb,
               duration=duration or None)
    return {"input": str(p), "output": str(out), "audio": str(a),
            "volume": volume, "replace": replace,
            "bytes": out.stat().st_size}


def concat(sources: list[str | os.PathLike[str]], *,
           out_dir: str | os.PathLike[str] | None = None,
           suffix: str = "joined", ext: str = ".mp4",
           timeout: float = FFMPEG_TIMEOUT,
           progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Join same-codec files with the concat demuxer (no re-encode)."""
    paths = [Path(s) for s in sources]
    if len(paths) < 2:
        raise MediaEditError("concat needs at least two sources")
    for p in paths:
        if not p.exists():
            raise MediaEditError(f"no such video: {p}")
        _check_size(p)
    out = _out(paths[0], Path(out_dir) if out_dir else None, suffix, ext)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in paths:
            fh.write(f"file '{p.resolve()}'\n")
        list_file = fh.name
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file,
                    "-c", "copy", str(out)],
                   timeout=timeout, progress_cb=progress_cb)
    finally:
        os.unlink(list_file)
    return {"inputs": [str(p) for p in paths], "output": str(out),
            "mode": "concat-demuxer", "bytes": out.stat().st_size}


_CONTAINER_DEFAULTS = {
    ".webm": ("libvpx-vp9", "libopus"),
    ".mp4": ("libx264", "aac"),
    ".mov": ("libx264", "aac"),
    ".m4v": ("libx264", "aac"),
    ".mkv": ("libx264", "aac"),
}


def transcode(src: str | os.PathLike[str], *,
              out_dir: str | os.PathLike[str] | None = None,
              suffix: str = "converted", ext: str = ".mp4",
              width: int | None = None, height: int | None = None,
              video_codec: str | None = None, crf: int = 23,
              preset: str = "medium", audio_codec: str | None = None,
              timeout: float = FFMPEG_TIMEOUT,
              progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Resize + codec/CRF/preset control + container conversion.

    Codecs default sensibly per container (VP9/Opus for WebM, H.264/AAC for
    MP4/MOV/MKV) unless explicitly overridden.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if width is not None and width <= 0:
        raise MediaEditError("width must be positive")
    if height is not None and height <= 0:
        raise MediaEditError("height must be positive")
    ext = ext.lower()
    default_v, default_a = _CONTAINER_DEFAULTS.get(ext, ("libx264", "aac"))
    video_codec = video_codec or default_v
    audio_codec = audio_codec or default_a
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    args = ["-i", str(p)]
    vf = []
    if width or height:
        vf.append(f"scale={width or -2}:{height or -2}")
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", video_codec]
    effective_crf: int | None = None
    if video_codec in ("libx264", "libx265"):
        effective_crf = crf
        args += ["-preset", preset, "-crf", str(crf)]
    elif video_codec in ("libvpx-vp9", "libvpx"):
        # VP9's CRF scale differs from x264's — remap honestly and report it.
        effective_crf = min(63, crf + 7)
        args += ["-b:v", "0", "-crf", str(effective_crf), "-cpu-used", "4"]
    args += ["-c:a", audio_codec, str(out)]
    run = run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
                     duration=info.get("duration"))
    result: dict[str, Any] = {
        "input": str(p), "output": str(out), "bytes": out.stat().st_size,
        "seconds": run["seconds"], "width": width, "height": height,
        "video_codec": video_codec, "audio_codec": audio_codec,
    }
    if effective_crf is not None:
        result["crf"] = effective_crf
    return result


def extract_frames(src: str | os.PathLike[str], *,
                   out_dir: str | os.PathLike[str] | None = None,
                   timestamps: list[str | float] | None = None,
                   interval: str | float | None = None,
                   count: int | None = None,
                   width: int = 640,
                   timeout: float = FFMPEG_TIMEOUT) -> dict[str, Any]:
    """Pull thumbnails: at explicit timestamps, every ``interval``, or
    ``count`` evenly spaced frames."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    if sum(x is not None for x in (timestamps, interval, count)) != 1:
        raise MediaEditError(
            "extract_frames needs exactly one of: timestamps, interval, count")
    target = (Path(out_dir) if out_dir else p.parent / "edited") / f"{p.stem}-frames"
    target.mkdir(parents=True, exist_ok=True)
    info = video_probe(p)
    duration = info.get("duration") or 0
    if timestamps is not None:
        ts = [parse_time(t) for t in timestamps]
        files = []
        for i, t in enumerate(ts):
            out = target / f"frame-{i:03d}-t{int(t)}s.jpg"
            run_ffmpeg(["-ss", str(t), "-i", str(p), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", str(out)],
                       timeout=min(timeout, 120))
            files.append(str(out))
    else:
        if count is not None:
            if count <= 0:
                raise MediaEditError("count must be positive")
            if duration <= 0:
                raise MediaEditError("cannot space frames: unknown duration")
            fps = count / duration
        else:
            gap = parse_time(interval)  # type: ignore[arg-type]
            if gap <= 0:
                raise MediaEditError("interval must be positive")
            fps = 1.0 / gap
        run_ffmpeg(["-i", str(p), "-vf",
                    f"fps={fps:.4f},scale={width}:-2",
                    str(target / "frame-%03d.jpg")],
                   timeout=timeout, duration=duration)
        files = sorted(str(f) for f in target.glob("frame-*.jpg"))
    return {"input": str(p), "frames_dir": str(target), "frames": files,
            "count": len(files)}


def extract_audio(src: str | os.PathLike[str], *,
                  out_dir: str | os.PathLike[str] | None = None,
                  suffix: str = "audio", ext: str = ".mp3",
                  timeout: float = FFMPEG_TIMEOUT,
                  progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Pull the full audio track out of a video."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    codec = {"mp3": "libmp3lame", "aac": "aac", "wav": "pcm_s16le",
             "ogg": "libvorbis"}.get(ext.lstrip("."), "libmp3lame")
    run = run_ffmpeg(["-i", str(p), "-vn", "-c:a", codec, str(out)],
                     timeout=timeout, progress_cb=progress_cb,
                     duration=info.get("duration"))
    return {"input": str(p), "output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"]}


def make_gif(src: str | os.PathLike[str], *,
             out_dir: str | os.PathLike[str] | None = None,
             suffix: str = "clip", start: str | float | int = 0,
             duration: str | float | int = 3, fps: int = 10, width: int = 480,
             timeout: float = FFMPEG_TIMEOUT) -> dict[str, Any]:
    """Animated GIF from a clip (fps + width capped for sanity)."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    _check_size(p)
    s = parse_time(start)
    d = parse_time(duration)
    if not 0 < d <= 30:
        raise MediaEditError("gif duration must be within (0, 30] seconds")
    fps = min(max(1, fps), 20)
    width = min(max(64, width), 800)
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ".gif")
    vf = (f"fps={fps},scale={width}:-2:flags=lanczos,"
          "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
    run = run_ffmpeg(["-ss", str(s), "-t", str(d), "-i", str(p),
                      "-vf", vf, str(out)],
                     timeout=timeout, duration=d)
    # fps/width are the CLAMPED values actually used (20 max, 64..800),
    # so the result never pretends a 60fps 4k gif was produced.
    return {"input": str(p), "output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"], "fps": fps, "width": width}


def _escape_filter_path(p: Path) -> str:
    """Escape a file path for use inside an ffmpeg filter argument."""
    text = str(p.resolve())
    for ch in ("\\", "'", ":", ",", "[", "]", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


def burn_subtitles(src: str | os.PathLike[str],
                   subtitles: str | os.PathLike[str], *,
                   out_dir: str | os.PathLike[str] | None = None,
                   suffix: str = "subtitled", ext: str = ".mp4",
                   timeout: float = FFMPEG_TIMEOUT,
                   progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Hard-burn an .srt/.ass file. Needs libass in the ffmpeg build."""
    p = Path(src)
    sub = Path(subtitles)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    if not sub.exists():
        raise MediaEditError(f"no such subtitle file: {subtitles}")
    if not has_libass():
        raise MediaEditError(
            "this ffmpeg build has no libass — cannot burn subtitles. "
            "Reinstall ffmpeg with libass, or keep subtitles as a sidecar file.")
    _check_size(p)
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    info = video_probe(p)
    run = run_ffmpeg(["-i", str(p), "-vf",
                      f"subtitles={_escape_filter_path(sub)}",
                      "-c:a", "copy", str(out)],
                     timeout=timeout, progress_cb=progress_cb,
                     duration=info.get("duration"))
    return {"input": str(p), "output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"]}


def media_probe_any(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Probe an image OR a video by extension."""
    from .images import image_probe
    ext = Path(path).suffix.lower()
    if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"):
        return video_probe(path)
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
               ".gif", ".avif"):
        return image_probe(path)
    # Unknown extension: try image probe first, then video.
    try:
        return image_probe(path)
    except MediaEditError:
        return video_probe(path)


# ---------------------------------------------------------------------------
# Frame-accurate thumbnails / preview grids (multi-backend)
#
# Imported lazily so the ffmpeg paths in this module never require OpenCV.
# Backend "auto" picks the best available: OpenCV for exact decoded frames,
# ffmpeg as fallback (keyframe-approximate seeks / tile filter).
# ---------------------------------------------------------------------------

def frame_accurate_thumbnail(src: str | os.PathLike[str], *,
                             timestamp: str | float | int = 0,
                             width: int = 640,
                             out_dir: str | os.PathLike[str] | None = None,
                             suffix: str = "thumb",
                             ext: str = ".jpg",
                             backend: str = "auto") -> dict[str, Any]:
    """Save the frame at ``timestamp`` as an image.

    Backend: OpenCV primary — decodes by frame index, so the thumbnail is
    the EXACT frame (subtitle timing, defect inspection, cut-point
    matching). ffmpeg fallback seeks with ``-ss`` (nearest keyframe:
    fast but approximate); the result flags ``"exact": False`` then.
    """
    from . import cv_video
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    chosen = cv_video._resolve_backend(backend, ("opencv", "ffmpeg"))
    if chosen == "ffmpeg":
        t = parse_time(timestamp)
        out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
        run_ffmpeg(["-ss", str(t), "-i", str(p), "-frames:v", "1",
                    "-vf", f"scale={width}:-2", str(out)],
                   timeout=min(FFMPEG_TIMEOUT, 120))
        return {"input": str(p), "output": str(out),
                "requested_t": timestamp, "actual_t": round(t, 3),
                "bytes": out.stat().st_size, "backend": "ffmpeg",
                "exact": False}
    cv2 = cv_video._cv2()
    frame, actual = cv_video.grab_frame_at(src, timestamp)
    frame = cv_video._resize_keep_aspect(cv2, frame, width)
    out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
    if not cv2.imwrite(str(out), frame):
        raise MediaEditError(f"could not write thumbnail {out}")
    return {"input": str(p), "output": str(out),
            "requested_t": timestamp, "actual_t": round(actual, 3),
            "bytes": out.stat().st_size, "backend": "opencv", "exact": True}


def make_preview_grid(src: str | os.PathLike[str], *,
                      cols: int = 4, rows: int = 3,
                      cell_width: int = 320,
                      out_dir: str | os.PathLike[str] | None = None,
                      suffix: str = "preview",
                      ext: str = ".jpg",
                      backend: str = "auto") -> dict[str, Any]:
    """Build a contact-sheet preview: ``cols`` x ``rows`` evenly spaced
    frames tiled into one image.

    Backend: OpenCV primary — exact decoded frames, each cell labeled
    with its timestamp. ffmpeg fallback — a single-pass
    ``fps`` + ``scale`` + ``tile`` filter graph (fast, no labels);
    the result flags ``"labeled": False`` then.
    """
    if cols < 1 or rows < 1:
        raise MediaEditError("cols and rows must be >= 1")
    if cell_width < 32:
        raise MediaEditError("cell_width must be >= 32")
    from . import cv_video
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    chosen = cv_video._resolve_backend(backend, ("opencv", "ffmpeg"))
    if chosen == "ffmpeg":
        n = cols * rows
        info = video_probe(p)
        duration = info.get("duration") or 0
        fps = (n / duration) if duration > 0 else 1.0
        out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
        run_ffmpeg(["-i", str(p), "-vf",
                    f"fps={fps:.4f},scale={cell_width}:-2,"
                    f"tile={cols}x{rows}",
                    "-frames:v", "1", str(out)],
                   timeout=FFMPEG_TIMEOUT, duration=duration or None)
        return {"input": str(p), "output": str(out), "cols": cols,
                "rows": rows, "cells": n, "bytes": out.stat().st_size,
                "backend": "ffmpeg", "labeled": False}
    cv2, np = cv_video._cv2(), cv_video._np()
    n = cols * rows
    res = cv_video.extract_frames(
        src, count=n, width=cell_width, fmt="jpg", backend="opencv",
        out_dir=str(Path(tempfile.mkdtemp(prefix="preview-grid-"))))
    try:
        cells = []
        for f in res["frames"]:
            img = cv2.imread(f)
            if img is None:
                raise MediaEditError(f"could not read extracted frame {f}")
            cells.append(img)
        if not cells:
            raise MediaEditError(f"no frames extracted from {src}")
        ch, cw = cells[0].shape[:2]
        # timestamp labels come from the frame filenames (t<secs>s)
        labels = []
        for f in res["frames"]:
            m = re.search(r"t(\d+\.\d+)s", Path(f).name)
            labels.append(f"{float(m.group(1)):.1f}s" if m else "")
        sheet = np.zeros((ch * rows, cw * cols, 3), dtype=np.uint8)
        for i, (img, label) in enumerate(zip(cells, labels)):
            r, c = divmod(i, cols)
            sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = img
            if label:
                cv2.putText(sheet, label, (c * cw + 8, r * ch + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2, cv2.LINE_AA)
        out = _out(p, Path(out_dir) if out_dir else None, suffix, ext)
        if not cv2.imwrite(str(out), sheet):
            raise MediaEditError(f"could not write preview grid {out}")
        return {"input": str(p), "output": str(out), "cols": cols,
                "rows": rows, "cells": len(cells),
                "bytes": out.stat().st_size, "backend": "opencv",
                "labeled": True}
    finally:
        shutil.rmtree(res["frames_dir"], ignore_errors=True)
