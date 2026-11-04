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

Every verb fails loudly (ConversionError with the ffmpeg tail)
and never fabricates an output file; a missing ffmpeg binary
surfaces the shared honest message from media_edit, not a
silent skip.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from ...core.logging_setup import get_logger
from ...media_edit.videos import MediaEditError, run_ffmpeg

_log = get_logger("nomorals.social.media_convert")

__all__ = [
    "ConversionError", "sticker_to_mp4", "gif_to_mp4",
    "mp4_to_gif", "mp4_to_mp3", "mp3_to_video",
    "image_to_sticker", "video_to_sticker", "make_text_sticker",
    "CONVERSIONS", "convert_media",
]


class ConversionError(RuntimeError):
    """A conversion that could not honestly complete."""


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _out(src: Path | str, suffix: str, *,
         out_dir: Path | str = "") -> Path:
    src = Path(src)
    dest_dir = Path(out_dir) if out_dir else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{src.stem}{suffix}"


def _run(runner: Runner | None, args: list[str], src: Path,
         dest: Path) -> Path:
    run = runner or run_ffmpeg
    try:
        run(args)
    except MediaEditError as exc:
        raise ConversionError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized below
        raise ConversionError(f"conversion failed: {exc}") from exc
    if not dest.exists() or dest.stat().st_size == 0:
        raise ConversionError(
            f"ffmpeg produced no output for {src.name}")
    return dest


_MP4_PAD = ("scale=512:512:force_original_aspect_ratio=decrease,"
            "pad=512:512:(ow-iw)/2:(oh-ih)/2:color=black,"
            "fps=30,format=yuv420p")


def sticker_to_mp4(src: Path | str, *, out_dir: Path | str = "",
                   runner: Runner | None = None) -> Path:
    """A static or animated WebP sticker → a shareable MP4
    (3s loop for a still — a one-frame 'video' most players
    refuse to show)."""
    src = Path(src)
    dest = _out(src, ".mp4", out_dir=out_dir)
    args = ["-y", "-loop", "1", "-i", str(src), "-t", "3",
            "-vf", _MP4_PAD, "-an", "-c:v", "libx264",
            "-movflags", "+faststart", str(dest)]
    return _run(runner, args, src, dest)


def gif_to_mp4(src: Path | str, *, out_dir: Path | str = "",
               runner: Runner | None = None) -> Path:
    """GIF → MP4 (the conversion every chat app does under the
    hood; Telegram's animation path wants exactly this)."""
    src = Path(src)
    dest = _out(src, ".mp4", out_dir=out_dir)
    args = ["-y", "-i", str(src), "-vf", _MP4_PAD,
            "-an", "-c:v", "libx264", "-movflags", "+faststart",
            str(dest)]
    return _run(runner, args, src, dest)


def mp4_to_gif(src: Path | str, *, out_dir: Path | str = "",
               runner: Runner | None = None) -> Path:
    """MP4 → GIF at a sane width/rate for chat (480w, 12fps)."""
    src = Path(src)
    dest = _out(src, ".gif", out_dir=out_dir)
    args = ["-y", "-i", str(src),
            "-vf", "fps=12,scale=480:-2:flags=lanczos",
            str(dest)]
    return _run(runner, args, src, dest)


def mp4_to_mp3(src: Path | str, *, out_dir: Path | str = "",
               runner: Runner | None = None) -> Path:
    """Video → its audio track as MP3 (192k)."""
    src = Path(src)
    dest = _out(src, ".mp3", out_dir=out_dir)
    args = ["-y", "-i", str(src), "-vn", "-codec:a", "libmp3lame",
            "-b:a", "192k", str(dest)]
    return _run(runner, args, src, dest)


def mp3_to_video(src: Path | str, *, out_dir: Path | str = "",
                 runner: Runner | None = None) -> Path:
    """Audio → a video of its own waveform (the standard
    audio-as-video shape every platform plays inline)."""
    src = Path(src)
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
    src = Path(src)
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
    src = Path(src)
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
