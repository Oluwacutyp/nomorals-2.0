"""Real-media probes for mock screens: image loading and
audio measurement.

The organ's first phases drew a placeholder scene for a
photo message and a synthetic waveform for a voice note.
This module is the hand-off that replaces both with the
real thing: anything Devon produces — a generated image,
any image bytes, her TTS/voice audio — composes into a
screen as first-class content.

Two contracts, both honest by construction:

- :func:`load_image` returns the real pixels for a
  hand-off (bytes / a path / a PIL image / a MediaRef,
  duck-typed on ``.path`` the way the service already
  hands refs around).  A source that cannot be opened
  raises :class:`MediaError` — the caller shows the
  reason, never a placeholder dressed as the photo.

- :func:`probe_audio` MEASURES a voice note: the duration
  is read from the decoded audio and the waveform bars are
  the audio's own per-bucket peak amplitudes, normalised
  to 0..1 (silence probes as all-zero bars).  WAV decodes
  with the stdlib ``wave`` machinery her audio stack is
  built on; ogg/opus, mp3 and m4a decode through ffmpeg
  to mono PCM — the repo's established decode idiom
  (``media/cover_video.audio_envelope``).  Audio that
  cannot be decoded raises :class:`MediaError`; no
  waveform is ever invented.

The probe is also the recording seam: the animation
phase consumes the same :class:`~.schema.Screen` (whose
messages carry the media reference) plus these probes —
no schema break when screens start moving.
"""

from __future__ import annotations

import array
import io
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

#: Bar count of a voice bubble's waveform (the renderer's
#: long-standing count; probes are taken at this width).
VOICE_BARS = 26

#: Sample rate non-WAV audio is decoded to (mono) for
#: probing — voice-band, the cover-video idiom's rate.
_DECODE_RATE = 16000


class MediaError(Exception):
    """A media hand-off that cannot be honoured; the
    message IS the user-fit reason."""


@dataclass
class AudioProbe:
    """A measured voice note: true duration in seconds
    and ``bars`` peak amplitudes in 0..1."""

    seconds: float
    bars: tuple


def format_duration(seconds: float) -> str:
    """``m:ss`` the way the apps print a voice note's
    length (rounded to the nearest second)."""
    total = max(0, int(round(float(seconds or 0.0))))
    return str(total // 60) + ":" + format(total % 60, "02d")


def _source_path(source) -> str:
    """The on-disk path of a hand-off, or "" when the
    hand-off is bytes / pixels rather than a file."""
    if isinstance(source, (str, Path)):
        return str(source)
    path = getattr(source, "path", "")
    return str(path) if path else ""


def load_image(source) -> Image.Image:
    """Open a real image hand-off as a PIL image (a copy
    the renderer may crop freely).  Raises MediaError
    with the reason when the source is not a readable
    image."""
    if isinstance(source, Image.Image):
        return source.copy()
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(source)))
        else:
            path = _source_path(source)
            if not path:
                raise MediaError(
                    "that image hand-off carries no bytes "
                    "and no file path")
            if not Path(path).exists():
                raise MediaError(
                    "the image file is not there: " + path)
            img = Image.open(path)
        img.load()
        return img.copy()
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - reason surfaces
        raise MediaError(
            "that file is not a readable image: "
            + str(exc)[:120]) from exc


def _peaks(samples_peak_iter, total: int, bars: int) -> tuple:
    """Bucket raw per-frame peaks into ``bars`` peak
    amplitudes, normalised by the global peak (an all-zero
    input stays all-zero — silence draws flat)."""
    if total <= 0:
        raise MediaError("the audio decoded to nothing")
    bucket = max(1, -(-total // max(1, bars)))
    raw = [0.0] * bars
    for idx, peak in samples_peak_iter:
        slot = min(bars - 1, idx // bucket)
        if peak > raw[slot]:
            raw[slot] = peak
    top = max(raw) if raw else 0.0
    if top <= 0.0:
        return tuple(0.0 for _ in raw)
    return tuple(round(min(1.0, v / top), 4) for v in raw)


def _wav_peaks(path: str, bars: int):
    """(seconds, bars) for a WAV via the stdlib wave
    machinery: per-frame channel peaks, chunked so a long
    note never loads whole."""
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = max(1, wf.getnchannels())
            width = wf.getsampwidth()
            if frames <= 0 or rate <= 0:
                raise MediaError(
                    "the audio decoded to nothing")

            def _iter():
                done = 0
                while done < frames:
                    chunk = wf.readframes(
                        min(8192, frames - done))
                    if not chunk:
                        break
                    count = len(chunk) // (width * channels)
                    for f in range(count):
                        base = f * channels
                        peak = 0.0
                        for c in range(channels):
                            off = (base + c) * width
                            raw = chunk[off:off + width]
                            if width == 1:
                                val = abs(raw[0] - 128) \
                                    / 128.0
                            elif width == 2:
                                val = abs(int.from_bytes(
                                    raw, "little",
                                    signed=True)) / 32768.0
                            elif width == 3:
                                val = abs(int.from_bytes(
                                    raw, "little",
                                    signed=True)) / 8388608.0
                            else:
                                val = abs(int.from_bytes(
                                    raw[:4], "little",
                                    signed=True)) \
                                    / 2147483648.0
                            if val > peak:
                                peak = val
                        yield done + f, peak
                    done += count

            return frames / float(rate), _peaks(
                _iter(), frames, bars)
    except (wave.Error, EOFError, OSError) as exc:
        raise MediaError(
            "that WAV could not be decoded: "
            + str(exc)[:120]) from exc


def _ffmpeg_peaks(path: str, bars: int):
    """(seconds, bars) for ogg/mp3/m4a/…: ffmpeg decodes
    to mono s16le PCM (the cover-video idiom) and the
    peaks are measured off the decoded samples."""
    ffmpeg = shutil.which("ffmpeg") or ""
    if not ffmpeg:
        raise MediaError(
            "that audio is not WAV and ffmpeg is not "
            "available on this host to decode it")
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path),
             "-ac", "1", "-ar", str(_DECODE_RATE),
             "-f", "s16le", "-"],
            capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaError(
            "the audio decode failed: "
            + str(exc)[:120]) from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise MediaError(
            "that audio could not be decoded"
            + (": " + detail[:120] if detail else ""))
    samples = array.array("h")
    samples.frombytes(proc.stdout[:len(proc.stdout) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    total = len(samples)
    if total <= 0:
        raise MediaError("the audio decoded to nothing")
    peaks = _peaks(
        ((i, abs(s) / 32768.0) for i, s in enumerate(samples)),
        total, bars)
    return total / float(_DECODE_RATE), peaks


def probe_audio(source, *, bars: int = VOICE_BARS) -> AudioProbe:
    """Measure a real voice note: true duration + the
    waveform bars derived from its own samples.  Raises
    MediaError when the audio cannot be decoded — the
    caller shows the reason; no bars are ever invented.

    Accepts a path, audio bytes, or a MediaRef
    (duck-typed on ``.path``)."""
    data = bytes(source) if isinstance(
        source, (bytes, bytearray)) else None
    path = "" if data is not None else _source_path(source)
    if data is None and not path:
        raise MediaError(
            "that voice hand-off carries no bytes and "
            "no file path")
    if data is not None:
        if data[:4] == b"RIFF":
            with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False) as fh:
                fh.write(data)
                tmp = fh.name
            try:
                seconds, peaks = _wav_peaks(tmp, bars)
            finally:
                Path(tmp).unlink(missing_ok=True)
            return AudioProbe(seconds=seconds, bars=peaks)
        with tempfile.NamedTemporaryFile(
                suffix=".audio", delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        try:
            seconds, peaks = _ffmpeg_peaks(tmp, bars)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return AudioProbe(seconds=seconds, bars=peaks)
    if not Path(path).exists():
        raise MediaError("the audio file is not there: " + path)
    if Path(path).suffix.lower() in (".wav", ".wave"):
        seconds, peaks = _wav_peaks(path, bars)
    else:
        seconds, peaks = _ffmpeg_peaks(path, bars)
    return AudioProbe(seconds=seconds, bars=peaks)
