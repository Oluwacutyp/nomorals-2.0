"""Audio mixing for the content-edit engine.

The short-form recipe: trending music bed + voiceover, the bed ducked
under the voice, everything loudness-normalised to streaming targets.

- :func:`mix_audio` — music + voiceover → final mix. Ducking is either a
  real sidechain compressor (``sidechaincompress``, smooth and musical)
  or deterministic volume automation from voice-activity segments.
  Finishes with two-pass ``loudnorm`` (accurate) with single-pass
  fallback.
- :func:`normalize_loudness` — standalone two-pass loudnorm.
- :func:`voice_segments` — voice-activity intervals of a voiceover file
  via ``silencedetect`` (drives volume-automation ducking and can feed
  caption timing).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ...media_edit.videos import MediaEditError, run_ffmpeg, video_probe
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "mix_audio",
    "normalize_loudness",
    "voice_segments",
    "loudness_measure",
]

#: streaming loudness target for Shorts/Reels/TikTok
TARGET_LUFS = -14.0
_TRUE_PEAK_DB = -1.0
_LRA = 11.0


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------

def _loudnorm_filter(target: float = TARGET_LUFS,
                      measured: dict[str, float] | None = None) -> str:
    base = f"loudnorm=I={target}:TP={_TRUE_PEAK_DB}:LRA={_LRA}"
    if measured:
        base += (f":measured_I={measured['input_i']}"
                 f":measured_TP={measured['input_tp']}"
                 f":measured_LRA={measured['input_lra']}"
                 f":measured_thresh={measured['input_thresh']}"
                 f":offset={measured['target_offset']}"
                 f":linear=true")
    return base


def loudness_measure(path: str | os.PathLike[str]) -> dict[str, float]:
    """First loudnorm pass → measured values (input_i/tp/lra/thresh,
    target_offset). Raises MediaEditError when the JSON can't be parsed."""
    import subprocess

    from ...media_edit.videos import ffmpeg_path
    cmd = [ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(path),
           "-af", _loudnorm_filter(TARGET_LUFS) + ":print_format=json",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditError(f"loudness measurement failed: {exc}") from exc
    m = re.search(r"\{[^}]*\"input_i\"[^}]*\}", proc.stderr, re.S)
    if not m:
        raise MediaEditError("loudnorm did not print measurement JSON")
    try:
        data = json.loads(m.group(0))
        return {k: float(data[k]) for k in
                ("input_i", "input_tp", "input_lra",
                 "input_thresh", "target_offset")}
    except (ValueError, KeyError) as exc:
        raise MediaEditError(
            f"could not parse loudnorm JSON: {exc}") from exc


def normalize_loudness(src: str | os.PathLike[str], *,
                       out: str | os.PathLike[str] | None = None,
                       target_lufs: float = TARGET_LUFS,
                       suffix: str = "loud") -> dict[str, Any]:
    """Two-pass loudnorm of ``src`` → AAC. Accurate; falls back to
    single-pass when measurement parsing fails (never silently wrong)."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such audio: {src}")
    out_p = Path(out) if out else p.with_name(f"{p.stem}.{suffix}.m4a")
    try:
        measured = loudness_measure(p)
        filt2 = _loudnorm_filter(target_lufs, measured)
        two_pass = True
    except MediaEditError as exc:
        _log.warning("loudness measure failed (%s) — single-pass", exc)
        filt2, measured, two_pass = _loudnorm_filter(target_lufs), {}, False
    run = run_ffmpeg(["-i", str(p), "-af", filt2,
                      "-ar", "48000", "-ac", "2",
                      "-c:a", "aac", "-b:a", "192k", str(out_p)],
                     timeout=600.0)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "target_lufs": target_lufs, "two_pass": two_pass,
            "measured": measured}


# ---------------------------------------------------------------------------
# voice activity (drives volume-automation ducking)
# ---------------------------------------------------------------------------

def voice_segments(vo_path: str | os.PathLike[str], *,
                   noise_db: float = -35.0,
                   min_silence: float = 0.25,
                   pad: float = 0.12) -> list[tuple[float, float]]:
    """Voice-activity intervals of a voiceover file.

    Runs ``silencedetect`` and inverts: returns [(start, end)] of
    non-silent regions, each padded by ``pad`` seconds. Empty list when
    the whole file is silent (caller decides what that means).
    """
    import subprocess

    from ...media_edit.videos import ffmpeg_path
    p = Path(vo_path)
    if not p.exists():
        raise MediaEditError(f"no such voiceover: {vo_path}")
    info = video_probe(p)
    dur = info.get("duration") or 0.0
    cmd = [ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(p),
           "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditError(f"silencedetect failed: {exc}") from exc
    silences: list[tuple[float, float]] = []
    starts = re.findall(r"silence_start:\s*([0-9.]+)", proc.stderr)
    ends = re.findall(r"silence_end:\s*([0-9.]+)", proc.stderr)
    for s, e in zip(starts, ends):
        silences.append((float(s), float(e)))
    # invert: speech = complement of silence over [0, dur]
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(silences):
        if s > cursor:
            speech.append((max(0.0, cursor - pad), s + pad))
        cursor = max(cursor, e)
    if cursor < dur:
        speech.append((max(0.0, cursor - pad), dur))
    # merge overlaps from padding
    merged: list[tuple[float, float]] = []
    for s, e in speech:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(round(s, 3), round(e, 3)) for s, e in merged if e - s > 0.05]


def _volume_duck_filter(segments: list[tuple[float, float]],
                        duck_level: float) -> str:
    """Deterministic ducking: music at ``duck_level`` during speech."""
    if not segments:
        return "volume=1.0"
    conds = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in segments)
    # clamp: volume= accepts expressions; eval=frame for sample accuracy
    return (f"volume='if({conds},{duck_level:.3f},1)':eval=frame")


# ---------------------------------------------------------------------------
# mix
# ---------------------------------------------------------------------------

def mix_audio(music: str | os.PathLike[str],
              voiceover: str | os.PathLike[str] | None = None, *,
              out: str | os.PathLike[str] | None = None,
              ducking: str = "sidechain",
              duck_level: float = 0.30,
              target_lufs: float = TARGET_LUFS,
              duration: str | float = "auto",
              loop_music: bool = False,
              vo_segments: list[tuple[float, float]] | None = None,
              suffix: str = "mix") -> dict[str, Any]:
    """Mix a trending-music bed under a voiceover → loudness-normalised AAC.

    - ``music``: user-supplied trending track (the "trending-audio slot").
    - ``voiceover``: optional VO/narration; the bed ducks under it.
    - ``ducking``: ``"sidechain"`` (smooth compressor — default),
      ``"volume"`` (deterministic automation from voice activity),
      ``"none"``.
    - ``duration``: ``"auto"`` (voiceover length when present, else music),
      ``"longest"``, or seconds as float.
    - ``loop_music``: loop a short trending clip to cover ``duration``.

    Returns {"output", "seconds", "target_lufs", "ducking", ...}.
    Two-pass loudnorm; single-pass fallback on measurement failure.
    """
    m_path = Path(music)
    if not m_path.exists():
        raise MediaEditError(f"no such music file: {music}")
    v_path = Path(voiceover) if voiceover else None
    if v_path is not None and not v_path.exists():
        raise MediaEditError(f"no such voiceover: {voiceover}")
    if ducking not in ("sidechain", "volume", "none"):
        raise MediaEditError(
            f"unknown ducking {ducking!r}; use sidechain/volume/none")
    if not 0.0 < duck_level <= 1.0:
        raise MediaEditError("duck_level must be in (0, 1]")

    out_p = Path(out) if out else m_path.with_name(f"{m_path.stem}.{suffix}.m4a")
    pre = "aresample=48000,aformat=channel_layouts=stereo"

    # --- build the filter graph -------------------------------------
    if v_path is None:
        chain = f"[0:a]{pre}[mix]"
        n_inputs = 1
        inputs = ["-i", str(m_path)]
    else:
        music_chain = f"[0:a]{pre}"
        if loop_music:
            music_chain += ",aloop=loop=-1:size=2147483647"
        music_chain += "[m]"
        vo_chain = f"[1:a]{pre}[v]"
        if ducking == "sidechain":
            # [music][voice]sidechaincompress → music ducked by the voice.
            # The VO feeds TWO consumers (sidechain key + final mix), so
            # it must be asplit first — a link label can only be used once.
            vo_chain = f"[1:a]{pre},asplit=2[vsc][v]"
            ducked = ("[m][vsc]sidechaincompress=threshold=0.06:ratio=10"
                      ":attack=12:release=400:makeup=1.4[ducked]")
            mix = "[ducked][v]amix=inputs=2:normalize=0,alimiter=limit=0.95[mix]"
        elif ducking == "volume":
            segs = (vo_segments if vo_segments is not None
                    else voice_segments(v_path))
            if not segs:
                raise MediaEditError(
                    f"no voice activity found in {v_path} — nothing to duck "
                    "under; use ducking='none' or a different voiceover")
            duck_f = _volume_duck_filter(segs, duck_level)
            ducked = f"[m]{duck_f}[ducked]"
            mix = "[ducked][v]amix=inputs=2:normalize=0,alimiter=limit=0.95[mix]"
        else:
            ducked = ""
            mix = "[m][v]amix=inputs=2:normalize=0,alimiter=limit=0.95[mix]"
        chain = ";".join(c for c in (music_chain, vo_chain, ducked, mix) if c)
        n_inputs = 2
        inputs = ["-i", str(m_path), "-i", str(v_path)]

    # --- duration ----------------------------------------------------
    extra: list[str] = []
    if isinstance(duration, (int, float)):
        extra += ["-t", f"{float(duration):.3f}"]
    elif duration == "auto" and v_path is not None:
        vinfo = video_probe(v_path)
        vdur = vinfo.get("duration")
        if vdur:
            extra += ["-t", f"{vdur:.3f}"]
    # else: longest input wins (ffmpeg default)

    # --- two-pass loudnorm -------------------------------------------
    with tempfile.TemporaryDirectory(prefix="mix-") as tmpd:
        stage1 = str(Path(tmpd) / "stage1.m4a")
        run_ffmpeg(inputs + ["-filter_complex", chain,
                             "-map", "[mix]",
                             "-c:a", "aac", "-b:a", "192k",
                             *extra, stage1],
                   timeout=600.0)
        try:
            measured = loudness_measure(stage1)
            norm = _loudnorm_filter(target_lufs, measured)
            two_pass = True
        except MediaEditError as exc:
            _log.warning("loudness measure failed (%s) — single-pass", exc)
            norm = _loudnorm_filter(target_lufs)
            measured, two_pass = {}, False
        run = run_ffmpeg(["-i", stage1, "-af", norm,
                          "-c:a", "aac", "-b:a", "192k", str(out_p)],
                         timeout=600.0)
    return {"input_music": str(m_path),
            "input_voiceover": str(v_path) if v_path else None,
            "output": str(out_p), "bytes": out_p.stat().st_size,
            "seconds": run["seconds"], "target_lufs": target_lufs,
            "ducking": ducking if v_path else "none",
            "two_pass": two_pass, "measured": measured,
            "n_inputs": n_inputs}
