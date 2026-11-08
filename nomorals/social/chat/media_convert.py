"""Media conversion for the expression layer (DESIGN §21):
sticker→MP4, GIF→MP4, MP4→GIF, MP4→MP3, MP3→video, and the
maker's platform-real outputs (image→sticker, video→animated
sticker, text→sticker).

This organ RIDES the house machinery — ``media_edit.videos``
owns ffmpeg discovery and invocation — and extends the
existing SVG text sticker maker (``community.media_tools``)
rather than replacing it: its deterministic palette and
wrapping are now rendered to a REAL raster sticker (512×512
WebP, the size every platform's sticker pipeline expects)
with PIL, falling back to the SVG artifact honestly when no
rasterizer/font is available.

The engine ladder, per verb, probed at runtime — never
assumed:

* ffmpeg (via ``shutil.which`` inside media_edit) is the
  primary rung for every conversion.
* Animated WebP has a second rung: some ffmpeg builds (the
  8.1.2 in the build sandbox among them) cannot decode
  animated WebP AT ALL, and looping flags either fail
  outright (WebM: "Option loop not found") or hang forever
  (animated WebP under ``-loop``).  So an animated WebP is
  decoded frame-by-frame with PIL — which reads it fine —
  and the PNG sequence is encoded by ffmpeg.  Where neither
  rung exists, the verb says exactly what is missing.
* A video with no audio track fails BEFORE ffmpeg runs,
  with the plain reason ("no audio track in that video"),
  probed via ffprobe when available.

Every conversion runs under a timeout (media_edit kills the
process tree at the limit; chat conversions cap at
``CONVERT_TIMEOUT_S``), refuses inputs above
``MAX_INPUT_BYTES``, cleans its temp files, and deletes a
partial output on failure.  Every verb fails loudly
(ConversionError with the ffmpeg tail) and never fabricates
an output file; a missing ffmpeg binary surfaces the shared
honest message from media_edit, not a silent skip.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from ...core.logging_setup import get_logger
from ...media_edit.videos import MediaEditError, run_ffmpeg
from .failures import record_failure

_log = get_logger("nomorals.social.media_convert")

__all__ = [
    "ConversionError", "sticker_to_mp4", "gif_to_mp4",
    "mp4_to_gif", "mp4_to_mp3", "mp3_to_video",
    "image_to_sticker", "video_to_sticker", "make_text_sticker",
    "CONVERSIONS", "convert_media",
    "CONVERT_TIMEOUT_S", "MAX_INPUT_BYTES",
]


class ConversionError(RuntimeError):
    """A conversion that could not honestly complete."""


Runner = Callable[..., "subprocess.CompletedProcess[str]"]

#: Chat conversions are short clips and stickers, not studio
#: work: a two-minute ceiling per conversion, and inputs are
#: refused above 200 MB (the studio cap in media_edit stays
#: 500 MB for the editing organ's own verbs).
CONVERT_TIMEOUT_S = 120.0
MAX_INPUT_BYTES = 200 * 1024 * 1024

#: Animated-WebP rung bounds: a sticker is seconds long.  An
#: "animation" past these bounds is a video wearing a sticker's
#: name, and the verb refuses it plainly rather than chewing
#: the pipeline.
_MAX_ANIM_FRAMES = 900
_MAX_ANIM_TOTAL_MS = 60_000


def _default_runner(args: list[str]) -> Any:
    return run_ffmpeg(args, timeout=CONVERT_TIMEOUT_S)


def _require_input(src: Path | str) -> Path:
    """The input gate every verb passes: the file exists, is
    non-empty, and is under the conversion size cap.  Refusals
    name the file and the reason — never an ffmpeg dump for a
    file that was never convertible."""
    p = Path(src)
    if not p.exists():
        raise ConversionError(f"no such input file: {p.name}")
    size = p.stat().st_size
    if size == 0:
        raise ConversionError(f"input file is empty: {p.name}")
    if size > MAX_INPUT_BYTES:
        raise ConversionError(
            f"{p.name} is {size // (1024 * 1024)} MB — chat "
            f"conversions cap inputs at "
            f"{MAX_INPUT_BYTES // (1024 * 1024)} MB")
    return p


def _out(src: Path | str, suffix: str, *,
         out_dir: Path | str = "") -> Path:
    src = Path(src)
    dest_dir = Path(out_dir) if out_dir else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{src.stem}{suffix}"


def _discard(dest: Path) -> None:
    """Remove a failed conversion's partial output: ffmpeg
    opens the destination with -y before it can fail, so a
    zero-byte or truncated file would otherwise linger as a
    fake artifact."""
    try:
        if dest.exists():
            dest.unlink()
    except OSError as exc:
        record_failure("media_convert._discard", exc)


def _run(runner: Runner | None, args: list[str], src: Path,
         dest: Path) -> Path:
    run = runner or _default_runner
    try:
        run(args)
    except MediaEditError as exc:
        _discard(dest)
        raise ConversionError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized below
        _discard(dest)
        raise ConversionError(f"conversion failed: {exc}") from exc
    if not dest.exists() or dest.stat().st_size == 0:
        _discard(dest)
        raise ConversionError(
            f"ffmpeg produced no output for {src.name}")
    return dest


_MP4_PAD = ("scale=512:512:force_original_aspect_ratio=decrease,"
            "pad=512:512:(ow-iw)/2:(oh-ih)/2:color=black,"
            "fps=30,format=yuv420p")


def _sniff_format(src: Path) -> str:
    """What the bytes say the file is: "webm" (EBML), "webp"
    (RIFF/WEBP), or "" when the magic is neither — the caller
    falls back to the file name.  Extensions lie (a Telegram
    .tgs download arrives named .webp); magic bytes don't."""
    try:
        head = src.open("rb").read(16)
    except OSError:
        return ""
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return ""


def _webp_is_animated(src: Path) -> bool:
    """Whether a WebP carries more than one frame.  PIL is the
    authority when present; ffprobe's frame count is the
    fallback; when neither can tell, the static path is tried
    and ffmpeg's own verdict decides (it decodes a static
    WebP fine and fails an animated one honestly)."""
    try:
        from PIL import Image

        with Image.open(src) as img:
            return bool(getattr(img, "is_animated", False)) \
                and int(getattr(img, "n_frames", 1) or 1) > 1
    except Exception:  # noqa: BLE001 - fall through to ffprobe
        pass
    try:
        from ...media_edit.videos import ffprobe_path

        probe_bin = ffprobe_path()
        if probe_bin:
            out = subprocess.run(
                [probe_bin, "-v", "quiet", "-count_frames",
                 "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", str(src)],
                capture_output=True, text=True, timeout=30)
            frames = int((out.stdout or "0").strip() or 0)
            return frames > 1
    except Exception:  # noqa: BLE001 - undecidable: static path
        pass
    return False


def _animated_webp_to_mp4(src: Path, dest: Path, *,
                          runner: Runner | None = None) -> Path:
    """The animated-WebP rung: PIL decodes the frames (it
    reads animated WebP where some ffmpeg builds cannot),
    a temp PNG sequence carries them to ffmpeg, and the temp
    dir is always cleaned.  Frame timing is approximated by
    the mean frame duration — sticker animations run at a
    uniform rate in practice; the docstring says so rather
    than pretending per-frame timing survived."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ConversionError(
            "animated WebP stickers need Pillow on this "
            "machine — this ffmpeg build cannot decode "
            "animated WebP, and Pillow is the rung that can "
            "(pip install Pillow)") from exc
    tmp = Path(tempfile.mkdtemp(prefix="webp_frames_"))
    try:
        try:
            img = Image.open(src)
            total = int(getattr(img, "n_frames", 1) or 1)
            if total > _MAX_ANIM_FRAMES:
                raise ConversionError(
                    f"that animation has {total} frames — "
                    f"sticker→MP4 covers up to "
                    f"{_MAX_ANIM_FRAMES}; it's a video, not a "
                    f"sticker")
            durations: list[int] = []
            canvas_size = img.size
            for index in range(total):
                img.seek(index)
                durations.append(
                    int(img.info.get("duration") or 100))
                frame = img.convert("RGBA")
                if frame.size != canvas_size:
                    canvas = Image.new("RGBA", canvas_size,
                                       (0, 0, 0, 0))
                    canvas.paste(frame, (0, 0), frame)
                    frame = canvas
                frame.save(tmp / f"frame_{index:04d}.png")
            img.close()
        except ConversionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConversionError(
                f"cannot decode animated WebP {src.name}: "
                f"{exc}") from exc
        total_ms = sum(durations)
        if total_ms > _MAX_ANIM_TOTAL_MS:
            raise ConversionError(
                f"that animation runs {total_ms // 1000}s — "
                f"sticker→MP4 covers up to "
                f"{_MAX_ANIM_TOTAL_MS // 1000}s")
        mean_ms = (total_ms / len(durations)) if durations else 100
        fps = max(1, min(50, round(1000 / max(mean_ms, 1))))
        args = ["-y", "-framerate", str(fps),
                "-i", str(tmp / "frame_%04d.png"),
                "-vf", _MP4_PAD, "-an", "-c:v", "libx264",
                "-movflags", "+faststart", str(dest)]
        return _run(runner, args, src, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sticker_to_mp4(src: Path | str, *, out_dir: Path | str = "",
                   runner: Runner | None = None) -> Path:
    """A sticker → a shareable MP4 (H.264, even dimensions).

    Three shapes, three honest routes: a WebM sticker is
    transcoded directly (the ``-loop`` flag is an image-only
    option — the WebM demuxer rejects it outright); a static
    WebP is looped for 3s (a one-frame 'video' most players
    refuse to show); an animated WebP goes through the PIL
    frame rung (see the module docstring — some ffmpeg
    builds cannot decode animated WebP, and ``-loop`` on one
    hangs forever)."""
    src = _require_input(src)
    dest = _out(src, ".mp4", out_dir=out_dir)
    fmt = _sniff_format(src) or src.suffix.lower().lstrip(".")
    if fmt == "webm":
        args = ["-y", "-i", str(src), "-vf", _MP4_PAD,
                "-an", "-c:v", "libx264",
                "-movflags", "+faststart", str(dest)]
        return _run(runner, args, src, dest)
    if fmt == "webp" and _webp_is_animated(src):
        return _animated_webp_to_mp4(src, dest, runner=runner)
    args = ["-y", "-loop", "1", "-i", str(src), "-t", "3",
            "-vf", _MP4_PAD, "-an", "-c:v", "libx264",
            "-movflags", "+faststart", str(dest)]
    return _run(runner, args, src, dest)


def gif_to_mp4(src: Path | str, *, out_dir: Path | str = "",
               runner: Runner | None = None) -> Path:
    """GIF → MP4 (the conversion every chat app does under the
    hood; Telegram's animation path wants exactly this)."""
    src = _require_input(src)
    dest = _out(src, ".mp4", out_dir=out_dir)
    args = ["-y", "-i", str(src), "-vf", _MP4_PAD,
            "-an", "-c:v", "libx264", "-movflags", "+faststart",
            str(dest)]
    return _run(runner, args, src, dest)


def mp4_to_gif(src: Path | str, *, out_dir: Path | str = "",
               runner: Runner | None = None) -> Path:
    """MP4 → GIF at a sane width/rate for chat (480w, 12fps)."""
    src = _require_input(src)
    dest = _out(src, ".gif", out_dir=out_dir)
    args = ["-y", "-i", str(src),
            "-vf", "fps=12,scale=480:-2:flags=lanczos",
            str(dest)]
    return _run(runner, args, src, dest)


def _require_audio_track(src: Path) -> None:
    """Refuse a video with no audio track, in plain words,
    BEFORE ffmpeg runs — its own verdict for this is an
    exit-code dump ("Output file does not contain any
    stream"), which is a failure but not a reason.  The
    probe rides media_edit's ffprobe machinery; when the
    probe itself is unavailable the check steps aside and
    ffmpeg's verdict stands (still honest, just uglier)."""
    try:
        from ...media_edit.videos import video_probe

        probe = video_probe(src)
    except Exception:  # noqa: BLE001 - probe unavailable
        return
    streams = probe.get("streams") or []
    if streams and not any(
            (s.get("type") or s.get("codec_type")) == "audio"
            for s in streams):
        raise ConversionError(
            "no audio track in that video — there is nothing "
            "to extract as MP3")


def mp4_to_mp3(src: Path | str, *, out_dir: Path | str = "",
               runner: Runner | None = None) -> Path:
    """Video → its audio track as MP3 (192k).  A video with
    NO audio track fails with that exact reason — never a
    silent empty file, never a raw ffmpeg dump."""
    src = _require_input(src)
    _require_audio_track(src)
    dest = _out(src, ".mp3", out_dir=out_dir)
    args = ["-y", "-i", str(src), "-vn", "-codec:a", "libmp3lame",
            "-b:a", "192k", str(dest)]
    return _run(runner, args, src, dest)


def mp3_to_video(src: Path | str, *, out_dir: Path | str = "",
                 runner: Runner | None = None) -> Path:
    """Audio → a video of its own waveform (the standard
    audio-as-video shape every platform plays inline), the
    waveform running the audio's full duration."""
    src = _require_input(src)
    dest = _out(src, ".mp4", out_dir=out_dir)
    args = ["-y", "-i", str(src),
            "-filter_complex",
            "[0:a]showwaves=s=854x480:mode=line:colors=white[v]",
            "-map", "[v]", "-map", "0:a", "-c:v", "libx264",
            "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
            str(dest)]
    return _run(runner, args, src, dest)


def video_to_sticker(src: Path | str, *,
                     out_dir: Path | str = "",
                     runner: Runner | None = None) -> Path:
    """Video/GIF → an animated sticker: ≤3s of it, 512 wide,
    WebM/VP9 — Telegram's animated-sticker discipline."""
    src = _require_input(src)
    dest = _out(src, ".webm", out_dir=out_dir)
    args = ["-y", "-i", str(src), "-t", "3",
            "-vf", "fps=30,scale=512:-2:flags=lanczos",
            "-an", "-c:v", "libvpx-vp9", "-b:v", "400k",
            "-pix_fmt", "yuva420p", str(dest)]
    return _run(runner, args, src, dest)


def image_to_sticker(src: Path | str, *,
                     out_dir: Path | str = "") -> Path:
    """Any still image → a platform sticker: fitted inside
    512×512 on transparency, saved as WebP (Telegram's static
    sticker format; WhatsApp's maker takes the same file)."""
    src = _require_input(src)
    dest = _out(src, ".webp", out_dir=out_dir)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ConversionError(
            "Pillow is not installed — image→sticker needs it "
            "(pip install Pillow)") from exc
    try:
        img = Image.open(src).convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        raise ConversionError(
            f"cannot read image {src.name}: {exc}") from exc
    img.thumbnail((512, 512), Image.LANCZOS)
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    canvas.paste(img, ((512 - img.width) // 2,
                       (512 - img.height) // 2), img)
    canvas.save(dest, "WEBP", lossless=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise ConversionError("sticker write produced no file")
    return dest


def make_text_sticker(text: str, *, out_dir: Path | str = "",
                      maker: Callable[..., Any] | None = None
                      ) -> dict[str, Any]:
    """The maker, completed: the house SVG sticker spec
    (palette, wrapping, determinism — community.media_tools)
    rendered to a REAL 512×512 WebP where PIL + a font allow
    it.  The SVG remains the artifact of record when
    rasterization is impossible — reported, never hidden."""
    from ...community import media_tools

    make = maker or media_tools.make_sticker
    spec = make(str(text or ""))
    dest = _out(Path(str(spec.id or "sticker")), ".webp",
                out_dir=out_dir)
    result: dict[str, Any] = {"spec": spec, "raster": "",
                              "note": ""}
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bg = _hex(spec.bg)
        draw.rounded_rectangle([16, 16, 496, 496], radius=48,
                               fill=bg + (255,))
        font = _load_font(64)
        fg = _hex(spec.fg)
        wrap = getattr(media_tools, "_wrap_words", None)
        if callable(wrap):
            lines = [str(line) for line in wrap(str(spec.text))]
        else:
            lines = [str(spec.text)]
        if spec.emoji:
            lines = [str(spec.emoji)] + lines
        line_h = 84
        y = 256 - (len(lines) * line_h) // 2 + 8
        for line in lines:
            box = draw.textbbox((0, 0), line, font=font)
            x = (512 - (box[2] - box[0])) // 2
            draw.text((x, y), line, font=font, fill=fg + (255,))
            y += line_h
        img.save(dest, "WEBP", lossless=True)
        result["raster"] = str(dest)
    except Exception as exc:  # noqa: BLE001 - the SVG is the fallback
        result["note"] = (f"raster sticker unavailable ({exc}); "
                          "the SVG sticker is the artifact")
    return result


def _hex(value: str) -> tuple[int, int, int]:
    value = str(value or "#222222").lstrip("#")
    if len(value) != 6:
        return (34, 34, 34)
    return (int(value[0:2], 16), int(value[2:4], 16),
            int(value[4:6], 16))


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    fc_list = Path("/usr/share/fonts")
    for ttf in sorted(fc_list.rglob("*.ttf")):
        if "Bold" in ttf.name and "Emoji" not in ttf.name:
            try:
                return ImageFont.truetype(str(ttf), size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


#: The conversion verb table: name → (function, blurb).  The
#: runtime's /convert help renders from this, so the help can
#: never drift from the code.
CONVERSIONS: dict[str, tuple[Callable[..., Path], str]] = {
    "sticker-to-mp4": (sticker_to_mp4, "a sticker becomes a short MP4"),
    "gif-to-mp4": (gif_to_mp4, "a GIF becomes an MP4"),
    "mp4-to-gif": (mp4_to_gif, "a video becomes a GIF"),
    "mp4-to-mp3": (mp4_to_mp3, "a video's audio becomes an MP3"),
    "mp3-to-video": (mp3_to_video, "an MP3 becomes a waveform video"),
    "image-to-sticker": (image_to_sticker, "an image becomes a sticker"),
    "video-to-sticker": (video_to_sticker,
                          "a short video becomes an animated sticker"),
}


def convert_media(name: str, src: Path | str, *,
                  out_dir: Path | str = "",
                  runner: Runner | None = None) -> Path:
    """Dispatch one named conversion; unknown names and house
    errors surface as ConversionError with the real reason."""
    entry = CONVERSIONS.get(str(name or "").strip().lower())
    if entry is None:
        raise ConversionError(
            f"unknown conversion {name!r} — I know: "
            + ", ".join(sorted(CONVERSIONS)))
    fn, _blurb = entry
    try:
        if fn in (image_to_sticker,):
            return fn(src, out_dir=out_dir)
        return fn(src, out_dir=out_dir, runner=runner)
    except MediaEditError as exc:
        raise ConversionError(str(exc)) from exc
