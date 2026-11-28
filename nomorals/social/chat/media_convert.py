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

import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
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
    "SoundSticker", "build_sound_sha256_exif",
    "add_sound_sha256_to_webp",
    "SOUND_STICKER_VIDEO_SECONDS", "SOUND_STICKER_PHOTO_SECONDS",
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

#: Sound stickers (WhatsApp, rolling out on beta clients late
#: 2026): the audio rides as a second upload nested on the
#: StickerMessage, paired to the WebP by the audio's SHA-256 in
#: the EXIF.  The limits are the test builds': 5s for a
#: video-origin sticker, 10s for a photo-origin one; the caps
#: are the fork's (5 MiB sticker, 16 MiB audio).
SOUND_STICKER_VIDEO_SECONDS = 5
SOUND_STICKER_PHOTO_SECONDS = 10
SOUND_STICKER_MAX_STICKER_BYTES = 5 * 1024 * 1024
SOUND_STICKER_MAX_AUDIO_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SoundSticker:
    """A sound sticker: the WebP (its EXIF carries sound-sha256)
    plus the OGG Opus audio that rides with it, its length in
    seconds, and the audio's SHA-256 hex."""
    sticker: Path
    audio: Path
    audio_seconds: float
    audio_sha256: str


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def build_sound_sha256_exif(sound_sha256_hex: str) -> bytes:
    """The sound-sticker EXIF payload: a little-endian TIFF
    header with a single IFD entry (tag 0x5741 'AW', type
    undefined) whose value is ``{"sound-sha256": "<hex>"}``.
    Layout pinned from @aunucore/baileys 2.7.0's
    buildSoundSha256Exif (lib/Utils/messages.js): 26-byte header
    (II, 42, IFD0 at 8, one entry, value offset 26, no next
    IFD) then the JSON."""
    hex_digest = str(sound_sha256_hex or "").lower()
    if not _HEX_SHA256.match(hex_digest):
        raise ConversionError(
            "sound-sha256 must be a hex SHA-256 digest")
    payload = json.dumps(
        {"sound-sha256": hex_digest}).encode("utf-8")
    exif = bytearray(26 + len(payload))
    exif[0:2] = b"II"
    struct.pack_into("<H", exif, 2, 0x2A)
    struct.pack_into("<I", exif, 4, 8)
    struct.pack_into("<H", exif, 8, 1)
    struct.pack_into("<H", exif, 10, 0x5741)
    struct.pack_into("<H", exif, 12, 7)
    struct.pack_into("<I", exif, 14, len(payload))
    struct.pack_into("<I", exif, 18, 26)
    struct.pack_into("<I", exif, 22, 0)
    exif[26:] = payload
    return bytes(exif)


def _webp_canvas_size(data: bytes) -> tuple[int, int] | None:
    """Canvas size of a WebP whose first chunk is 'VP8 ' (lossy)
    or 'VP8L' (lossless) — needed to synthesise a VP8X header
    when the file has none (an EXIF chunk is only valid in the
    extended container).  Layout pinned from the fork's
    webpCanvasSize."""
    if len(data) < 25:
        return None
    fourcc = data[12:16]
    if fourcc == b"VP8L":
        if data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if fourcc == b"VP8 ":
        if len(data) < 30 or data[23] != 0x9D \
                or data[24] != 0x01 or data[25] != 0x2A:
            return None
        width = struct.unpack_from("<H", data, 26)[0] & 0x3FFF
        height = struct.unpack_from("<H", data, 28)[0] & 0x3FFF
        return (width, height)
    return None


#: The VP8X EXIF-profile bit (RFC 9649 §2.7: Rsv|I|L|E|X|A|R —
#: E is bit 3).  The fork this is pinned from writes 0x20 with
#: an "EXIF flag" comment, but 0x20 is the ICC-profile bit;
#: Devon writes the spec bit.
_VP8X_EXIF_FLAG = 0x08


def add_sound_sha256_to_webp(webp: Path | str,
                             sound_sha256_hex: str) -> Path:
    """Write the "sound-sha256" EXIF into a WebP sticker, in
    place.  Replaces any existing EXIF chunk, sets the VP8X
    EXIF flag (a VP8X header is synthesised for plain VP8/VP8L
    files), and rebuilds the RIFF size.  ConversionError for a
    non-WebP input, a bad digest, or a WebP whose canvas size
    cannot be determined when a VP8X header is needed.  Chunk
    mechanics pinned from the fork's addSoundSha256ToStickerWebP
    (the flag value corrected per RFC 9649, above)."""
    p = _require_input(webp)
    raw = p.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" \
            or raw[8:12] != b"WEBP":
        raise ConversionError(
            "sound sticker EXIF can only be written into a "
            "WebP image")
    exif = build_sound_sha256_exif(sound_sha256_hex)
    chunks: list[tuple[bytes, bytearray]] = []
    offset = 12
    while offset + 8 <= len(raw):
        fourcc = raw[offset:offset + 4]
        size = struct.unpack_from("<I", raw, offset + 4)[0]
        if offset + 8 + size > len(raw):
            raise ConversionError(
                "malformed WebP: chunk overruns the file")
        chunks.append((fourcc, bytearray(
            raw[offset + 8:offset + 8 + size])))
        offset += 8 + size + (size % 2)
    kept = [(c, d) for c, d in chunks if c != b"EXIF"]
    vp8x = next((d for c, d in kept if c == b"VP8X"), None)
    if vp8x is not None:
        if len(vp8x) < 10:
            raise ConversionError(
                "malformed WebP: VP8X chunk too short")
        vp8x[0] |= _VP8X_EXIF_FLAG
    else:
        size = _webp_canvas_size(raw)
        if size is None or size[0] < 1 or size[1] < 1:
            raise ConversionError(
                "cannot determine the WebP canvas size needed "
                "to add an EXIF chunk")
        data = bytearray(10)
        data[0] = _VP8X_EXIF_FLAG
        data[4:7] = (size[0] - 1).to_bytes(3, "little")
        data[7:10] = (size[1] - 1).to_bytes(3, "little")
        kept.insert(0, (b"VP8X", data))
    kept.append((b"EXIF", bytearray(exif)))
    body = bytearray(b"WEBP")
    for fourcc, data in kept:
        body += fourcc + struct.pack("<I", len(data)) + data
        if len(data) % 2:
            body += b"\x00"
    out = b"RIFF" + struct.pack("<I", len(body)) + bytes(body)
    p.write_bytes(out)
    return p


def _probe_media_seconds(path: Path) -> float:
    """Length of an audio file in seconds, via the media-edit
    ffprobe machinery; 0.0 when the probe cannot say (the wire
    then omits the seconds — never an invented length)."""
    try:
        from ...media_edit.videos import video_probe

        probe = video_probe(path)
    except Exception:  # noqa: BLE001 - probe unavailable
        return 0.0
    try:
        return float(probe.get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _extract_sound_audio(src: Path | str, *, out_dir: Path | str = "",
                         runner: Runner | None = None,
                         max_seconds: int,
                         from_video: bool = False
                         ) -> tuple[Path, float]:
    """Audio → OGG Opus, trimmed to ``max_seconds``.  From a
    video the track must exist first (_require_audio_track
    refuses a silent video in plain words)."""
    src = _require_input(src)
    if from_video:
        _require_audio_track(src)
    # The "-sound" stem keeps the output from ever colliding
    # with its own input (an .ogg source in the same out_dir
    # would otherwise be ffmpeg's input AND output).
    dest_dir = Path(out_dir) if out_dir else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}-sound.ogg"
    args = ["-y", "-i", str(src), "-vn", "-c:a", "libopus",
            "-b:a", "64k", "-t", str(max_seconds), str(dest)]
    _run(runner, args, src, dest)
    seconds = _probe_media_seconds(dest)
    if seconds > max_seconds:
        seconds = float(max_seconds)
    return dest, seconds


def _finish_sound_sticker(sticker: Path, audio: Any, src: Path, *,
                          out_dir: Path | str = "",
                          runner: Runner | None = None,
                          max_seconds: int) -> SoundSticker:
    """The shared sound tail: extract + trim the audio, cap the
    sizes, hash the audio, write its hash into the WebP EXIF.
    A failed audio step discards the sticker too — no half-made
    sound sticker is left behind."""
    try:
        if audio == "original":
            audio_path, seconds = _extract_sound_audio(
                src, out_dir=out_dir, runner=runner,
                max_seconds=max_seconds, from_video=True)
        else:
            audio_path, seconds = _extract_sound_audio(
                audio, out_dir=out_dir, runner=runner,
                max_seconds=max_seconds)
    except ConversionError:
        _discard(sticker)
        raise
    if sticker.stat().st_size > SOUND_STICKER_MAX_STICKER_BYTES \
            or audio_path.stat().st_size \
            > SOUND_STICKER_MAX_AUDIO_BYTES:
        _discard(sticker)
        _discard(audio_path)
        raise ConversionError(
            "sound sticker too big — the sticker must stay "
            "under 5 MiB and its audio under 16 MiB")
    audio_sha256 = hashlib.sha256(
        audio_path.read_bytes()).hexdigest()
    add_sound_sha256_to_webp(sticker, audio_sha256)
    return SoundSticker(sticker=sticker, audio=audio_path,
                        audio_seconds=seconds,
                        audio_sha256=audio_sha256)


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
                     runner: Runner | None = None,
                     format: str = "webm",
                     audio: Any = None) -> Path | SoundSticker:
    """Video/GIF → an animated sticker: ≤3s of it, 512 wide.

    ``format`` is the TARGET platform's animated-sticker format:
    "webm" (VP9 — Telegram's animated-sticker discipline, the
    default, so every existing caller is unchanged) or "webp"
    (animated WebP — WhatsApp's animated-sticker format; a .webm
    cannot be a WhatsApp sticker at all, and a .webm sent to the
    WhatsApp bridge lands as a document).  Any other value is a
    ConversionError — fail closed, never silently produce the
    wrong form.

    ``audio`` makes a SOUND sticker (WebP only): "original" keeps
    the video's own audio track, or pass an audio file.  The
    video window grows to the sound limit (5s), the audio is
    trimmed to it and transcoded to OGG Opus, its SHA-256 is
    written into the WebP EXIF, and the result is a SoundSticker
    (sticker + audio + seconds + hash).  No audio means exactly
    today's behaviour — a Path, the 3s window, no audio anywhere.
    """
    if format not in ("webm", "webp"):
        raise ConversionError(
            f"unknown sticker format {format!r} — 'webm' or 'webp'")
    if audio is not None and format != "webp":
        raise ConversionError(
            "a sound sticker needs the WebP format — a .webm "
            "cannot carry sound on WhatsApp")
    src = _require_input(src)
    window = "5" if audio is not None else "3"
    if format == "webp":
        dest = _out(src, ".webp", out_dir=out_dir)
        args = ["-y", "-i", str(src), "-t", window,
                "-vf", "fps=30,scale=512:-2:flags=lanczos",
                "-an", "-c:v", "libwebp", "-loop", "0",
                "-pix_fmt", "yuva420p", str(dest)]
    else:
        dest = _out(src, ".webm", out_dir=out_dir)
        args = ["-y", "-i", str(src), "-t", window,
                "-vf", "fps=30,scale=512:-2:flags=lanczos",
                "-an", "-c:v", "libvpx-vp9", "-b:v", "400k",
                "-pix_fmt", "yuva420p", str(dest)]
    sticker = _run(runner, args, src, dest)
    if audio is None:
        return sticker
    return _finish_sound_sticker(
        sticker, audio, src, out_dir=out_dir, runner=runner,
        max_seconds=SOUND_STICKER_VIDEO_SECONDS)


def image_to_sticker(src: Path | str, *,
                     out_dir: Path | str = "",
                     audio: Any = None) -> Path | SoundSticker:
    """Any still image → a platform sticker: fitted inside
    512×512 on transparency, saved as WebP (Telegram's static
    sticker format; WhatsApp's maker takes the same file).

    ``audio`` makes a SOUND sticker: pass an audio file (a photo
    has no track of its own, so "original" is a ConversionError).
    The audio is trimmed to the photo-origin limit (10s) and
    transcoded to OGG Opus, its SHA-256 is written into the WebP
    EXIF, and the result is a SoundSticker.  No audio means
    exactly today's behaviour — a Path."""
    if audio == "original":
        raise ConversionError(
            "a photo has no audio track of its own — send a "
            "voice note or an audio file with it")
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
    if audio is None:
        return dest
    return _finish_sound_sticker(
        dest, audio, src, out_dir=out_dir,
        max_seconds=SOUND_STICKER_PHOTO_SECONDS)


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
    """The conversion-card face through the ONE door: the
    typography service's ``poster`` context policy over the
    stocked registry.  The old private host-path ladder
    (three hardcoded files plus a filesystem walk) is gone;
    should the registry ever report a total miss, the miss
    is logged by name and Pillow's default carries the card
    (this helper has never raised)."""
    from PIL import ImageFont

    try:
        from ...media.typography import get_service, pil_font_for
        return pil_font_for(get_service(), "poster", size,
                            weight=700)
    except Exception as exc:  # noqa: BLE001 - logged miss
        _log.warning(
            "media_convert: stocked library miss (%s); "
            "rendering in Pillow's default face", exc)
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
