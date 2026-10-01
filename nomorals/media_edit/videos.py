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
    "ffmpeg is not installed on this machine. "
    "Install it (e.g. `apt install ffmpeg`) to enable video editing; "
    "image tools are unaffected."
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

_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d{1,2}):)?(\d{1,2}(?:\.\d+)?)$")


def parse_time(value: str | float | int) -> float:
    """'90' -> 90.0, '1:30' -> 90.0, '0:01:30.5' -> 90.5, 30 -> 30.0."""
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
        hours = float(m.group(1) or 0)
        minutes = float(m.group(2) or 0)
        seconds = float(m.group(3))
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
                except ValueError:
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
        except OSError:
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
    if video_codec in ("libx264", "libx265"):
        args += ["-preset", preset, "-crf", str(crf)]
    elif video_codec in ("libvpx-vp9", "libvpx"):
        args += ["-b:v", "0", "-crf", str(min(63, crf + 7)), "-cpu-used", "4"]
    args += ["-c:a", audio_codec, str(out)]
    run = run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
                     duration=info.get("duration"))
    return {"input": str(p), "output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"], "width": width, "height": height}


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
    return {"input": str(p), "output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"]}


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
    run = run_ffmpeg(["-i", str(p), "-vf", f"subtitles={sub.resolve()}",
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
